# Copyright (c) The Mega-UP Authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""BSHD, SBHD and packed-THD host APIs for the two Mega-UP kernels."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor

from .runtime import HIDDEN_MULTIPLE, MegaUPRuntime, unsupported_shape_reason


def _source_major_to_bshd(
    source_major: Tensor,
    *,
    batch: int,
    sequence: int,
    shard_heads: int,
    head_dim: int,
) -> Tensor:
    """Convert ``[source,B*S,H/UP*D]`` to ``[B,UP*S,H/UP,D]``."""

    if source_major.ndim != 3:
        raise ValueError("source_major must be a 3-D tensor")
    up = int(source_major.shape[0])
    expected = (up, batch * sequence, shard_heads * head_dim)
    if tuple(source_major.shape) != expected:
        raise ValueError(
            f"source_major must have shape {expected}, got "
            f"{tuple(source_major.shape)}"
        )
    return (
        source_major.view(
            up,
            batch,
            sequence,
            shard_heads,
            head_dim,
        )
        .permute(1, 0, 2, 3, 4)
        .reshape(batch, up * sequence, shard_heads, head_dim)
        .contiguous()
    )


def _bshd_to_destination_major(
    head_parallel: Tensor,
    *,
    up: int,
    sequence: int,
) -> Tensor:
    """Convert ``[B,UP*S,H/UP,D]`` to ``[destination,B*S,H/UP*D]``."""

    if head_parallel.ndim != 4:
        raise ValueError("head_parallel must be a 4-D tensor")
    batch, global_sequence, shard_heads, head_dim = map(
        int, head_parallel.shape
    )
    if global_sequence != up * sequence:
        raise ValueError(
            f"sequence dimension must be UP*S={up * sequence}, got "
            f"{global_sequence}"
        )
    return (
        head_parallel.view(
            batch,
            up,
            sequence,
            shard_heads,
            head_dim,
        )
        .permute(1, 0, 2, 3, 4)
        .reshape(
            up,
            batch * sequence,
            shard_heads * head_dim,
        )
        .contiguous()
    )


# ---------------------------------------------------------------------------
# Packed THD layout
# ---------------------------------------------------------------------------


def _normalize_cu_seqlens(
    cu_seqlens: Tensor | Sequence[int],
) -> tuple[int, ...]:
    """Validate rank-local packed boundaries and return them as host ints."""

    if isinstance(cu_seqlens, Tensor):
        if cu_seqlens.ndim != 1:
            raise ValueError("cu_seqlens must be a 1-D tensor")
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError("cu_seqlens must be an int32 or int64 tensor")
        values = tuple(int(value) for value in cu_seqlens.tolist())
    else:
        values = tuple(int(value) for value in cu_seqlens)
    if len(values) < 2:
        raise ValueError("cu_seqlens needs at least one sample, e.g. [0, T]")
    if values[0] != 0:
        raise ValueError(f"cu_seqlens[0] must be 0, got {values[0]}")
    for previous, current in zip(values, values[1:]):
        if current < previous:
            raise ValueError("cu_seqlens must be non-decreasing")
    if values[-1] <= 0:
        raise ValueError("cu_seqlens[-1], the local token count, must be positive")
    return values


@dataclass(frozen=True)
class THDLayout:
    """Optional per-sample regroup for the split-sample packing scheme.

    The default THD contract needs no layout: each rank holds a contiguous
    slab of the global packed stream and the head-parallel output is already
    in global order. Use a layout only when every sample is instead split
    evenly across the UP group, so that all ranks share the same rank-local
    ``cu_seqlens``. Arena order is source-major: row ``r*T + j`` is local row
    ``j`` of source rank ``r``. The regrouped order keeps each sample
    contiguous with its UP chunks in rank order, so the public boundaries are
    ``cu_seqlens_out = UP * cu_seqlens``.
    """

    up: int
    tokens: int
    cu_seqlens: tuple[int, ...]
    to_thd_index: Tensor
    to_source_major_index: Tensor
    fingerprint: str

    @property
    def num_samples(self) -> int:
        return len(self.cu_seqlens) - 1

    @property
    def cu_seqlens_out(self) -> tuple[int, ...]:
        return tuple(self.up * value for value in self.cu_seqlens)

    @property
    def device(self) -> torch.device:
        return self.to_thd_index.device


def _build_thd_layout(
    cu_seqlens: Tensor | Sequence[int],
    *,
    up: int,
    device: torch.device | str | None,
) -> THDLayout:
    if up <= 0:
        raise ValueError(f"up must be positive, got {up}")
    cu = _normalize_cu_seqlens(cu_seqlens)
    tokens = cu[-1]
    num_samples = len(cu) - 1
    cu_t = torch.tensor(cu, dtype=torch.int64, device=device)
    lengths = cu_t[1:] - cu_t[:-1]
    sample = torch.repeat_interleave(
        torch.arange(num_samples, dtype=torch.int64, device=device),
        lengths,
        output_size=tokens,
    )
    local_row = torch.arange(tokens, dtype=torch.int64, device=device)
    start = cu_t[sample]
    # THD row of (source 0, local row j); source r adds r * len(sample).
    thd_base = up * start + (local_row - start)
    source_stride = lengths[sample]
    to_source_major = (
        thd_base.unsqueeze(0)
        + torch.arange(up, dtype=torch.int64, device=device).unsqueeze(1)
        * source_stride.unsqueeze(0)
    ).reshape(-1)
    to_thd = torch.empty_like(to_source_major)
    to_thd.scatter_(
        0,
        to_source_major,
        torch.arange(up * tokens, dtype=torch.int64, device=device),
    )
    fingerprint = hashlib.sha1(
        struct.pack(f"<{len(cu)}q", *cu)
    ).hexdigest()
    return THDLayout(
        up=up,
        tokens=tokens,
        cu_seqlens=cu,
        to_thd_index=to_thd,
        to_source_major_index=to_source_major,
        fingerprint=fingerprint,
    )


def prepare_thd_layout(
    cu_seqlens: Tensor | Sequence[int],
    runtime: MegaUPRuntime,
) -> THDLayout:
    """Build the split-sample regroup permutations for ``runtime``.

    ``cu_seqlens`` are rank-local boundaries in tokens and must be identical
    on every UP rank. Build the layout once per step and reuse it for every
    layer; a CUDA ``cu_seqlens`` is synchronized to the host here. Not needed
    for the default contiguous-slab THD contract.
    """

    return _build_thd_layout(
        cu_seqlens,
        up=runtime.up,
        device=runtime.device,
    )


def _source_major_to_thd(
    source_major: Tensor,
    layout: THDLayout,
    *,
    shard_heads: int,
    head_dim: int,
) -> Tensor:
    """Convert ``[source,T,H/UP*D]`` to packed ``[UP*T,H/UP,D]``."""

    expected = (layout.up, layout.tokens, shard_heads * head_dim)
    if tuple(source_major.shape) != expected:
        raise ValueError(
            f"source_major must have shape {expected}, got "
            f"{tuple(source_major.shape)}"
        )
    total = layout.up * layout.tokens
    return (
        source_major.reshape(total, shard_heads * head_dim)
        .index_select(0, layout.to_thd_index)
        .view(total, shard_heads, head_dim)
    )


def _thd_to_destination_major(
    head_parallel: Tensor,
    layout: THDLayout,
) -> Tensor:
    """Convert packed ``[UP*T,H/UP,D]`` to ``[destination,T,H/UP*D]``."""

    if head_parallel.ndim != 3:
        raise ValueError("head_parallel must be a 3-D THD tensor")
    total = layout.up * layout.tokens
    if int(head_parallel.shape[0]) != total:
        raise ValueError(
            f"head_parallel must have UP*T={total} rows, got "
            f"{int(head_parallel.shape[0])}"
        )
    shard_hidden = int(head_parallel.shape[1]) * int(head_parallel.shape[2])
    return (
        head_parallel.reshape(total, shard_hidden)
        .index_select(0, layout.to_source_major_index)
        .view(layout.up, layout.tokens, shard_hidden)
    )


# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------


def _check_runtime_device(runtime: MegaUPRuntime) -> None:
    capability = torch.cuda.get_device_capability(runtime.device)
    if capability[0] not in (10, 11):
        raise ValueError(
            "Mega-UP kernels do not support this GPU (compute capability "
            f"{capability})"
        )


def _check_weight(weight: Tensor, runtime: MegaUPRuntime) -> tuple[int, int]:
    """Validate an ``[out_features, in_features]`` weight; return ``(N, K)``."""

    if not isinstance(weight, Tensor) or weight.ndim != 2 or min(weight.shape) <= 0:
        raise ValueError(
            "weight must be a non-empty [out_features, in_features] tensor, "
            f"got {_tensor_signature(weight)}"
        )
    if (
        weight.dtype != torch.bfloat16
        or weight.device != runtime.device
        or not weight.is_contiguous()
    ):
        raise ValueError(
            "weight must be contiguous BF16 on the runtime device"
        )
    return int(weight.shape[0]), int(weight.shape[1])


def _check_exchanged_width(width: int, runtime: MegaUPRuntime, name: str) -> None:
    reason = unsupported_shape_reason(width, runtime.up)
    if reason is not None:
        raise ValueError(f"{name}: {reason}")


def _check_peer_weight(
    weight: Tensor, runtime: MegaUPRuntime, head_dim: int
) -> tuple[int, int]:
    """Validate a peer-write weight; return ``(N, K)``.

    The output width ``N`` is exchanged: each rank receives ``N/UP`` columns,
    which must hold whole ``head_dim``-wide heads.
    """

    out_features, in_features = _check_weight(weight, runtime)
    _check_exchanged_width(out_features, runtime, "peer-write out_features")
    if out_features % (runtime.up * head_dim):
        raise ValueError(
            f"out_features={out_features} must be a multiple of "
            f"UP*head_dim={runtime.up * head_dim}"
        )
    # TMA loads need 16-byte (8 BF16) row strides.
    if in_features % 8:
        raise ValueError(f"in_features={in_features} must be a multiple of 8")
    return out_features, in_features


def _check_pull_weight(
    weight: Tensor, runtime: MegaUPRuntime, head_dim: int
) -> tuple[int, int]:
    """Validate a pull weight; return ``(N, K)``.

    The input width ``K`` is exchanged. The output width ``N`` is tiled with
    256-wide tiles and returned as ``N/head_dim`` heads.
    """

    out_features, in_features = _check_weight(weight, runtime)
    _check_exchanged_width(in_features, runtime, "pull in_features")
    if out_features % HIDDEN_MULTIPLE or out_features % head_dim:
        raise ValueError(
            f"out_features={out_features} must be a multiple of "
            f"{HIDDEN_MULTIPLE} and of head_dim={head_dim}"
        )
    return out_features, in_features


def _check_bias(
    bias: Tensor | None, runtime: MegaUPRuntime, out_features: int
) -> None:
    if bias is None:
        return
    if not isinstance(bias, Tensor) or tuple(bias.shape) != (out_features,):
        raise ValueError(
            f"bias must have shape ({out_features},), got "
            f"{_tensor_signature(bias)}"
        )
    if (
        bias.dtype != torch.bfloat16
        or bias.device != runtime.device
        or not bias.is_contiguous()
    ):
        raise ValueError("bias must be contiguous BF16 on the runtime device")
    if bias.data_ptr() % 16:
        raise ValueError("bias must be 16-byte aligned")


def _optional_signature(tensor: Tensor | None) -> tuple[object, ...] | None:
    return None if tensor is None else _tensor_signature(tensor)


def _check_layout(layout: THDLayout, runtime: MegaUPRuntime) -> None:
    if not isinstance(layout, THDLayout):
        raise TypeError("layout must be a THDLayout from prepare_thd_layout")
    if layout.up != runtime.up:
        raise ValueError(
            f"THD layout was built for UP={layout.up}, runtime has UP={runtime.up}"
        )
    if layout.device != runtime.device:
        raise ValueError(
            f"THD layout lives on {layout.device}, runtime on {runtime.device}"
        )


def _check_activation(tensor: Tensor, runtime: MegaUPRuntime, name: str) -> None:
    if (
        tensor.dtype != torch.bfloat16
        or tensor.device != runtime.device
        or not tensor.is_contiguous()
    ):
        raise ValueError(f"{name} must be contiguous BF16 on the runtime device")


def _check_out(
    out: Tensor | None,
    expected: tuple[int, ...],
    runtime: MegaUPRuntime,
) -> None:
    if out is not None and (
        tuple(out.shape) != expected
        or out.dtype != torch.bfloat16
        or out.device != runtime.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "out must be contiguous BF16 on the runtime device with "
            f"shape {expected}, got {_tensor_signature(out)}"
        )


def _weight_widths(weight: object) -> dict[str, int]:
    if not isinstance(weight, Tensor) or weight.ndim != 2:
        raise ValueError(
            "weight must be a 2-D [out_features, in_features] tensor, got "
            f"{_tensor_signature(weight)}"
        )
    return {"out_features": int(weight.shape[0]), "in_features": int(weight.shape[1])}


def _check_logical_shape(shape: tuple[int, ...]) -> None:
    if min(shape) <= 0:
        raise ValueError(f"all dimensions must be positive, got {shape}")


def _check_input_heads(heads: int, head_dim: int, in_features: int) -> None:
    if heads * head_dim != in_features:
        raise ValueError(
            f"heads*head_dim={heads}*{head_dim} must equal the weight's "
            f"in_features={in_features}"
        )


def _check_pull_heads(
    heads: int, head_dim: int, runtime: MegaUPRuntime, in_features: int
) -> None:
    if heads % runtime.up:
        raise ValueError(f"heads={heads} must be divisible by UP={runtime.up}")
    _check_input_heads(heads, head_dim, in_features)


def _peer_shape(
    x: Tensor,
    runtime: MegaUPRuntime,
    *,
    in_features: int,
    out_features: int,
) -> tuple[int, int, int, int]:
    if x.ndim != 4:
        raise ValueError(
            f"peer-write input must be [B,S,H,D], got {tuple(x.shape)}"
        )
    batch, sequence, heads, head_dim = map(int, x.shape)
    if min(batch, sequence, heads, head_dim) <= 0:
        raise ValueError(f"all BSHD dimensions must be positive, got {x.shape}")
    _check_input_heads(heads, head_dim, in_features)
    runtime.check_capacity(batch * sequence, out_features)
    _check_activation(x, runtime, "input")
    return batch, sequence, heads, head_dim


def _peer_shape_thd(
    x: Tensor,
    runtime: MegaUPRuntime,
    layout: THDLayout | None,
    *,
    in_features: int,
    out_features: int,
) -> tuple[int, int, int]:
    if x.ndim != 3:
        raise ValueError(
            f"peer-write THD input must be [T,H,D], got {tuple(x.shape)}"
        )
    tokens, heads, head_dim = map(int, x.shape)
    if min(tokens, heads, head_dim) <= 0:
        raise ValueError(f"all THD dimensions must be positive, got {x.shape}")
    if layout is not None:
        _check_layout(layout, runtime)
        if tokens != layout.tokens:
            raise ValueError(
                f"input has {tokens} rows but the THD layout describes "
                f"{layout.tokens} local tokens"
            )
    _check_input_heads(heads, head_dim, in_features)
    runtime.check_capacity(tokens, out_features)
    _check_activation(x, runtime, "input")
    return tokens, heads, head_dim


def _pull_shape(
    head_x: Tensor,
    runtime: MegaUPRuntime,
    *,
    in_features: int,
) -> tuple[int, int, int, int]:
    if head_x.ndim != 4:
        raise ValueError(
            "pull input must be [B,UP*S,H/UP,D], got "
            f"{tuple(head_x.shape)}"
        )
    batch, global_sequence, shard_heads, head_dim = map(int, head_x.shape)
    if min(batch, global_sequence, shard_heads, head_dim) <= 0:
        raise ValueError(
            f"all head-parallel dimensions must be positive, got {head_x.shape}"
        )
    if global_sequence % runtime.up:
        raise ValueError(
            f"sequence dimension {global_sequence} must be divisible by "
            f"UP={runtime.up}"
        )
    sequence = global_sequence // runtime.up
    heads = shard_heads * runtime.up
    _check_input_heads(heads, head_dim, in_features)
    runtime.check_capacity(batch * sequence, in_features)
    _check_activation(head_x, runtime, "head-parallel input")
    return batch, sequence, heads, head_dim


def _pull_shape_thd(
    head_x: Tensor,
    runtime: MegaUPRuntime,
    layout: THDLayout | None,
    *,
    in_features: int,
) -> tuple[int, int, int]:
    """Return ``(tokens, heads, head_dim)`` for packed ``[UP*T,H/UP,D]``."""

    if head_x.ndim != 3:
        raise ValueError(
            "pull THD input must be [UP*T,H/UP,D], got "
            f"{tuple(head_x.shape)}"
        )
    total, shard_heads, head_dim = map(int, head_x.shape)
    if min(total, shard_heads, head_dim) <= 0:
        raise ValueError(
            f"all head-parallel dimensions must be positive, got {head_x.shape}"
        )
    if total % runtime.up:
        raise ValueError(
            f"pull THD input rows {total} must be divisible by UP={runtime.up}"
        )
    tokens = total // runtime.up
    if layout is not None:
        _check_layout(layout, runtime)
        if tokens != layout.tokens:
            raise ValueError(
                f"pull THD input must have UP*T={runtime.up * layout.tokens} "
                f"rows, got {total}"
            )
    heads = shard_heads * runtime.up
    _check_input_heads(heads, head_dim, in_features)
    runtime.check_capacity(tokens, in_features)
    _check_activation(head_x, runtime, "head-parallel input")
    return tokens, heads, head_dim


def _peer_shape_sbhd(
    x: Tensor,
    runtime: MegaUPRuntime,
    *,
    in_features: int,
    out_features: int,
) -> tuple[int, int, int, int]:
    """Return ``(sequence, batch, heads, head_dim)`` for ``[S,B,H,D]``."""

    if x.ndim != 4:
        raise ValueError(
            f"peer-write SBHD input must be [S,B,H,D], got {tuple(x.shape)}"
        )
    sequence, batch, heads, head_dim = map(int, x.shape)
    if min(sequence, batch, heads, head_dim) <= 0:
        raise ValueError(f"all SBHD dimensions must be positive, got {x.shape}")
    _check_input_heads(heads, head_dim, in_features)
    runtime.check_capacity(sequence * batch, out_features)
    _check_activation(x, runtime, "input")
    return sequence, batch, heads, head_dim


def _pull_shape_sbhd(
    head_x: Tensor,
    runtime: MegaUPRuntime,
    *,
    in_features: int,
) -> tuple[int, int, int, int]:
    """Return ``(sequence, batch, heads, head_dim)`` for ``[UP*S,B,H/UP,D]``."""

    if head_x.ndim != 4:
        raise ValueError(
            "pull SBHD input must be [UP*S,B,H/UP,D], got "
            f"{tuple(head_x.shape)}"
        )
    global_sequence, batch, shard_heads, head_dim = map(int, head_x.shape)
    if min(global_sequence, batch, shard_heads, head_dim) <= 0:
        raise ValueError(
            f"all head-parallel dimensions must be positive, got {head_x.shape}"
        )
    if global_sequence % runtime.up:
        raise ValueError(
            f"sequence dimension {global_sequence} must be divisible by "
            f"UP={runtime.up}"
        )
    sequence = global_sequence // runtime.up
    heads = shard_heads * runtime.up
    _check_input_heads(heads, head_dim, in_features)
    runtime.check_capacity(sequence * batch, in_features)
    _check_activation(head_x, runtime, "head-parallel input")
    return sequence, batch, heads, head_dim


def _layout_fingerprint(layout: THDLayout | None) -> str | None:
    return layout.fingerprint if isinstance(layout, THDLayout) else None


def _tensor_signature(tensor: object) -> tuple[object, ...]:
    if not isinstance(tensor, Tensor):
        return ("not-a-tensor", type(tensor).__name__)
    return (
        tuple(int(value) for value in tensor.shape),
        str(tensor.dtype),
        tensor.device.type,
        bool(tensor.is_contiguous()),
    )


def _collective_validate(
    runtime: MegaUPRuntime,
    operation: str,
    metadata: object,
    local_error: str | None,
) -> None:
    """Make local contract failures rank-uniform before any kernel/barrier."""

    record = (operation, metadata, local_error)
    records: list[object | None] = [None] * runtime.up
    device_context = (
        torch.cuda.device(runtime.device)
        if runtime.device.type == "cuda"
        else nullcontext()
    )
    with device_context:
        dist.all_gather_object(records, record, group=runtime.group)
    errors = tuple(
        (rank, candidate[2])
        for rank, candidate in enumerate(records)
        if isinstance(candidate, tuple) and candidate[2] is not None
    )
    if errors:
        raise ValueError(f"Mega-UP collective validation failed: {errors}")
    expected = record[:2]
    if any(
        not isinstance(candidate, tuple) or candidate[:2] != expected
        for candidate in records
    ):
        raise ValueError(
            "all UP ranks must enter the same operation with identical "
            "metadata and runtime configuration"
        )


class _ProjectionWidths:
    """``in_features`` and ``out_features`` of a prepared executable's weight."""

    weight: Tensor

    @property
    def in_features(self) -> int:
        return int(self.weight.shape[1])

    @property
    def out_features(self) -> int:
        return int(self.weight.shape[0])


# ---------------------------------------------------------------------------
# Peer-write GEMM
# ---------------------------------------------------------------------------


def _bind_peer_write(
    weight: Tensor,
    runtime: MegaUPRuntime,
    *,
    bias: Tensor | None,
    rows: int,
    operation: str,
    metadata: object,
    validate_group: bool,
) -> object:
    """Bind the kernel to this rank's slot in every peer arena, with
    collective error checks."""

    launch = None
    setup_error: str | None = None
    try:
        from .kernels.peer_write_gemm import make_peer_write_launch

        out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
        launch = make_peer_write_launch(
            weight.unsqueeze(0),
            None,
            tuple(
                peer.unsqueeze(0)
                for peer in runtime.remote_write_views(rows, out_features)
            ),
            a_shape=(1, rows, in_features),
            bias=bias,
            local_rank=runtime.local_rank,
        )
    except Exception as error:
        setup_error = f"{type(error).__name__}: {error}"
    if validate_group:
        _collective_validate(runtime, f"{operation}_setup", metadata, setup_error)
    if setup_error is not None:
        raise RuntimeError(setup_error)
    assert launch is not None
    return launch


def _run_prepared_peer_write(
    runtime: MegaUPRuntime,
    launch: object,
    x: Tensor,
    rows: int,
    materialize: Callable[[], object],
) -> None:
    with torch.no_grad(), runtime.lease():
        launch(x.view(1, rows, -1))
        materialize()


@dataclass
class PreparedPeerWriteGemm(_ProjectionWidths):
    """Reusable peer-write executable for a fixed ``[B,S,H,D]`` input shape."""

    runtime: MegaUPRuntime
    weight: Tensor
    batch: int
    sequence: int
    heads: int
    head_dim: int
    launch: object
    bias: Tensor | None = None

    @property
    def output_shape(self) -> tuple[int, int, int, int]:
        return (
            self.batch,
            self.runtime.up * self.sequence,
            self.out_features // (self.runtime.up * self.head_dim),
            self.head_dim,
        )

    def __call__(
        self,
        x: Tensor,
        *,
        out: Tensor | None = None,
        validate_group: bool = True,
    ) -> Tensor:
        expected = (self.batch, self.sequence, self.heads, self.head_dim)
        shape: tuple[int, int, int, int] | None = None
        local_error: str | None = None
        try:
            shape = _peer_shape(
                x,
                self.runtime,
                in_features=self.in_features,
                out_features=self.out_features,
            )
            if shape != expected:
                raise ValueError(
                    f"prepared peer-write GEMM expects BSHD {expected}, got {shape}"
                )
            _check_out(out, self.output_shape, self.runtime)
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
        if validate_group:
            _collective_validate(
                self.runtime,
                "peer_write_gemm_bshd",
                (
                    shape,
                    _tensor_signature(self.weight),
                    _optional_signature(self.bias),
                    _optional_signature(out),
                    self.runtime.hidden,
                    self.runtime.max_rows,
                    self.runtime.up,
                ),
                local_error,
            )
        if local_error is not None:
            raise ValueError(local_error)
        if out is None:
            out = torch.empty(
                self.output_shape, dtype=torch.bfloat16, device=self.runtime.device
            )
        _run_prepared_peer_write(
            self.runtime,
            self.launch,
            x,
            self.batch * self.sequence,
            lambda: self.runtime.copy_peer_written(
                batch=self.batch,
                sequence=self.sequence,
                shard_heads=self.output_shape[2],
                head_dim=self.head_dim,
                out=out,
            ),
        )
        return out


@dataclass
class PreparedPeerWriteGemmSBHD(_ProjectionWidths):
    """Reusable peer-write executable for a fixed sequence-first ``[S,B,H,D]``
    input shape."""

    runtime: MegaUPRuntime
    weight: Tensor
    sequence: int
    batch: int
    heads: int
    head_dim: int
    launch: object
    bias: Tensor | None = None

    @property
    def output_shape(self) -> tuple[int, int, int, int]:
        return (
            self.runtime.up * self.sequence,
            self.batch,
            self.out_features // (self.runtime.up * self.head_dim),
            self.head_dim,
        )

    def __call__(
        self,
        x: Tensor,
        *,
        out: Tensor | None = None,
        validate_group: bool = True,
    ) -> Tensor:
        expected = (self.sequence, self.batch, self.heads, self.head_dim)
        shape: tuple[int, int, int, int] | None = None
        local_error: str | None = None
        try:
            shape = _peer_shape_sbhd(
                x,
                self.runtime,
                in_features=self.in_features,
                out_features=self.out_features,
            )
            if shape != expected:
                raise ValueError(
                    f"prepared SBHD peer-write GEMM expects (S,B,H,D) {expected}, "
                    f"got {shape}"
                )
            _check_out(out, self.output_shape, self.runtime)
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
        if validate_group:
            _collective_validate(
                self.runtime,
                "peer_write_gemm_sbhd",
                (
                    shape,
                    _tensor_signature(self.weight),
                    _optional_signature(self.bias),
                    _optional_signature(out),
                    self.runtime.hidden,
                    self.runtime.max_rows,
                    self.runtime.up,
                ),
                local_error,
            )
        if local_error is not None:
            raise ValueError(local_error)
        if out is None:
            out = torch.empty(
                self.output_shape, dtype=torch.bfloat16, device=self.runtime.device
            )
        up = self.runtime.up
        rows = self.sequence * self.batch
        shard_heads = self.output_shape[2]
        # Rows are sequence-major, so the receive slots already hold
        # [UP*S,B,H/UP,D] and the materialization is one contiguous copy.
        _run_prepared_peer_write(
            self.runtime,
            self.launch,
            x,
            rows,
            lambda: self.runtime.copy_peer_written(
                batch=1,
                sequence=rows,
                shard_heads=shard_heads,
                head_dim=self.head_dim,
                out=out.view(1, up * rows, shard_heads, self.head_dim),
            ),
        )
        return out


@dataclass
class PreparedPeerWriteGemmTHD(_ProjectionWidths):
    """Reusable packed-THD peer-write executable for a fixed token count.

    ``layout`` is ``None`` for the contiguous-slab contract (output rows in
    source-rank order) or a ``THDLayout`` that regroups every sample to be
    contiguous.
    """

    runtime: MegaUPRuntime
    weight: Tensor
    tokens: int
    heads: int
    head_dim: int
    launch: object
    layout: THDLayout | None = None
    bias: Tensor | None = None

    @property
    def output_shape(self) -> tuple[int, int, int]:
        return (
            self.runtime.up * self.tokens,
            self.out_features // (self.runtime.up * self.head_dim),
            self.head_dim,
        )

    def __call__(
        self,
        x: Tensor,
        *,
        out: Tensor | None = None,
        validate_group: bool = True,
    ) -> Tensor:
        expected = (self.tokens, self.heads, self.head_dim)
        shape: tuple[int, int, int] | None = None
        local_error: str | None = None
        try:
            shape = _peer_shape_thd(
                x,
                self.runtime,
                self.layout,
                in_features=self.in_features,
                out_features=self.out_features,
            )
            if shape != expected:
                raise ValueError(
                    f"prepared THD peer-write GEMM expects (T,H,D) {expected}, "
                    f"got {shape}"
                )
            _check_out(out, self.output_shape, self.runtime)
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
        if validate_group:
            _collective_validate(
                self.runtime,
                "peer_write_gemm_thd",
                (
                    shape,
                    _layout_fingerprint(self.layout),
                    _tensor_signature(self.weight),
                    _optional_signature(self.bias),
                    _optional_signature(out),
                    self.runtime.hidden,
                    self.runtime.max_rows,
                    self.runtime.up,
                ),
                local_error,
            )
        if local_error is not None:
            raise ValueError(local_error)
        if out is None:
            out = torch.empty(
                self.output_shape, dtype=torch.bfloat16, device=self.runtime.device
            )
        up = self.runtime.up
        shard_heads = self.output_shape[1]
        layout = self.layout

        def materialize() -> None:
            if layout is None:
                # Source-major order is the global packed order: one contiguous copy.
                self.runtime.copy_peer_written(
                    batch=1,
                    sequence=self.tokens,
                    shard_heads=shard_heads,
                    head_dim=self.head_dim,
                    out=out.view(1, up * self.tokens, shard_heads, self.head_dim),
                )
            else:
                self.runtime.copy_peer_written_indexed(
                    rows=self.tokens,
                    index=layout.to_thd_index,
                    out=out,
                )

        _run_prepared_peer_write(self.runtime, self.launch, x, self.tokens, materialize)
        return out


def prepare_peer_write_gemm_bshd(
    weight: Tensor,
    runtime: MegaUPRuntime,
    *,
    batch: int,
    sequence: int,
    heads: int,
    head_dim: int,
    bias: Tensor | None = None,
    validate_group: bool = True,
) -> PreparedPeerWriteGemm:
    """Prepare a reusable ``linear(weight, bias) + all-to-all`` replacement:
    ``[B,S,H,D]`` to ``[B,UP*S,N/(UP*D),D]``.

    ``weight`` is ``[N, K]`` with ``K = heads * head_dim``; the output keeps
    ``head_dim`` and splits its ``N/head_dim`` heads across the ranks. The
    weight, bias, arena slots and tile schedule are validated and bound once;
    a call checks the input, launches the kernel and copies the received rows
    into the output.
    """

    logical_shape = (batch, sequence, heads, head_dim)
    local_error: str | None = None
    try:
        _check_runtime_device(runtime)
        _check_logical_shape(logical_shape)
        out_features, in_features = _check_peer_weight(weight, runtime, head_dim)
        _check_bias(bias, runtime, out_features)
        _check_input_heads(heads, head_dim, in_features)
        runtime.check_capacity(batch * sequence, out_features)
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    metadata = (logical_shape, _tensor_signature(weight), _optional_signature(bias))
    if validate_group:
        _collective_validate(
            runtime,
            "prepare_peer_write_gemm_bshd",
            (*metadata, runtime.hidden, runtime.max_rows, runtime.up),
            local_error,
        )
    if local_error is not None:
        raise ValueError(local_error)
    launch = _bind_peer_write(
        weight,
        runtime,
        bias=bias,
        rows=batch * sequence,
        operation="prepare_peer_write_gemm_bshd",
        metadata=metadata,
        validate_group=validate_group,
    )
    return PreparedPeerWriteGemm(
        runtime=runtime,
        weight=weight,
        batch=batch,
        sequence=sequence,
        heads=heads,
        head_dim=head_dim,
        launch=launch,
        bias=bias,
    )


def prepare_peer_write_gemm_sbhd(
    weight: Tensor,
    runtime: MegaUPRuntime,
    *,
    sequence: int,
    batch: int,
    heads: int,
    head_dim: int,
    bias: Tensor | None = None,
    validate_group: bool = True,
) -> PreparedPeerWriteGemmSBHD:
    """Prepare a reusable sequence-first ``linear(weight, bias) + all-to-all``
    replacement: ``[S,B,H,D]`` to ``[UP*S,B,N/(UP*D),D]`` for an ``[N, K]``
    weight with ``K = heads * head_dim``."""

    logical_shape = (sequence, batch, heads, head_dim)
    local_error: str | None = None
    try:
        _check_runtime_device(runtime)
        _check_logical_shape(logical_shape)
        out_features, in_features = _check_peer_weight(weight, runtime, head_dim)
        _check_bias(bias, runtime, out_features)
        _check_input_heads(heads, head_dim, in_features)
        runtime.check_capacity(sequence * batch, out_features)
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    metadata = (logical_shape, _tensor_signature(weight), _optional_signature(bias))
    if validate_group:
        _collective_validate(
            runtime,
            "prepare_peer_write_gemm_sbhd",
            (*metadata, runtime.hidden, runtime.max_rows, runtime.up),
            local_error,
        )
    if local_error is not None:
        raise ValueError(local_error)
    launch = _bind_peer_write(
        weight,
        runtime,
        bias=bias,
        rows=sequence * batch,
        operation="prepare_peer_write_gemm_sbhd",
        metadata=metadata,
        validate_group=validate_group,
    )
    return PreparedPeerWriteGemmSBHD(
        runtime=runtime,
        weight=weight,
        sequence=sequence,
        batch=batch,
        heads=heads,
        head_dim=head_dim,
        launch=launch,
        bias=bias,
    )


def prepare_peer_write_gemm_thd(
    weight: Tensor,
    runtime: MegaUPRuntime,
    *,
    heads: int,
    head_dim: int,
    tokens: int | None = None,
    layout: THDLayout | None = None,
    bias: Tensor | None = None,
    validate_group: bool = True,
) -> PreparedPeerWriteGemmTHD:
    """Prepare a reusable packed-THD ``linear(weight, bias) + all-to-all``
    replacement: ``[T,H,D]`` to ``[UP*T,N/(UP*D),D]`` for an ``[N, K]``
    weight with ``K = heads * head_dim``.

    ``tokens`` may be omitted when ``layout`` is given. The output order is
    the one described in :func:`peer_write_gemm_thd`.
    """

    local_error: str | None = None
    fingerprint = _layout_fingerprint(layout)
    logical_shape: tuple[int, int, int] | None = None
    try:
        _check_runtime_device(runtime)
        _check_logical_shape((heads, head_dim))
        out_features, in_features = _check_peer_weight(weight, runtime, head_dim)
        _check_bias(bias, runtime, out_features)
        if layout is not None:
            _check_layout(layout, runtime)
            if tokens is None:
                tokens = layout.tokens
            elif tokens != layout.tokens:
                raise ValueError(
                    f"tokens={tokens} disagrees with layout.tokens={layout.tokens}"
                )
        if tokens is None:
            raise ValueError("tokens is required when no THD layout is given")
        logical_shape = (tokens, heads, head_dim)
        _check_logical_shape(logical_shape)
        _check_input_heads(heads, head_dim, in_features)
        runtime.check_capacity(tokens, out_features)
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    metadata = (
        logical_shape,
        fingerprint,
        _tensor_signature(weight),
        _optional_signature(bias),
    )
    if validate_group:
        _collective_validate(
            runtime,
            "prepare_peer_write_gemm_thd",
            (*metadata, runtime.hidden, runtime.max_rows, runtime.up),
            local_error,
        )
    if local_error is not None:
        raise ValueError(local_error)
    assert tokens is not None
    launch = _bind_peer_write(
        weight,
        runtime,
        bias=bias,
        rows=tokens,
        operation="prepare_peer_write_gemm_thd",
        metadata=metadata,
        validate_group=validate_group,
    )
    return PreparedPeerWriteGemmTHD(
        runtime=runtime,
        weight=weight,
        tokens=tokens,
        heads=heads,
        head_dim=head_dim,
        launch=launch,
        layout=layout,
        bias=bias,
    )


def peer_write_gemm_bshd(
    x: Tensor,
    weight: Tensor,
    runtime: MegaUPRuntime,
    *,
    bias: Tensor | None = None,
    out: Tensor | None = None,
    validate_group: bool = True,
) -> Tensor:
    """Fuse ``linear([B,S,H,D], weight, bias)`` with the Ulysses head all-to-all.

    ``weight`` and the optional ``bias`` follow ``torch.nn.functional.linear``
    layout: ``[N, K]`` and ``[N]``, with ``K = H*D``. The returned tensor has
    shape ``[B,UP*S,N/(UP*D),D]`` and owns ordinary CUDA storage. Repeated
    calls with one shape should use :func:`prepare_peer_write_gemm_bshd`,
    which validates and binds once.
    """

    shape: tuple[int, int, int, int] | None = None
    local_error: str | None = None
    try:
        shape = _peer_shape(x, runtime, **_weight_widths(weight))
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    if validate_group:
        _collective_validate(
            runtime,
            "peer_write_gemm_bshd_input",
            (
                shape,
                _tensor_signature(weight),
                _optional_signature(bias),
                _optional_signature(out),
                runtime.hidden,
                runtime.max_rows,
                runtime.up,
            ),
            local_error,
        )
    if local_error is not None:
        raise ValueError(local_error)
    assert shape is not None
    batch, sequence, heads, head_dim = shape
    prepared = prepare_peer_write_gemm_bshd(
        weight,
        runtime,
        batch=batch,
        sequence=sequence,
        heads=heads,
        head_dim=head_dim,
        bias=bias,
        validate_group=validate_group,
    )
    return prepared(x, out=out, validate_group=validate_group)


def peer_write_gemm_thd(
    x: Tensor,
    weight: Tensor,
    runtime: MegaUPRuntime,
    *,
    bias: Tensor | None = None,
    layout: THDLayout | None = None,
    out: Tensor | None = None,
    validate_group: bool = True,
) -> Tensor:
    """Fuse ``linear([T,H,D], weight, bias)`` with the Ulysses head all-to-all
    for packed data.

    ``x`` holds this rank's ``T`` rows of the packed token stream and
    ``weight`` is ``[N, H*D]``. The returned tensor has shape
    ``[UP*T,N/(UP*D),D]``. Without ``layout`` the rows are in source-rank
    order: when every rank holds a contiguous slab of the global stream this
    is the global packed order, so global ``cu_seqlens`` apply unchanged and
    the materialization is one contiguous copy. With a ``layout`` (samples
    split evenly across ranks), every sample is regrouped to be contiguous
    with boundaries ``layout.cu_seqlens_out``. Repeated calls with one shape
    should use :func:`prepare_peer_write_gemm_thd`.
    """

    shape: tuple[int, int, int] | None = None
    local_error: str | None = None
    try:
        shape = _peer_shape_thd(x, runtime, layout, **_weight_widths(weight))
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    if validate_group:
        _collective_validate(
            runtime,
            "peer_write_gemm_thd_input",
            (
                shape,
                _layout_fingerprint(layout),
                _tensor_signature(weight),
                _optional_signature(bias),
                _optional_signature(out),
                runtime.hidden,
                runtime.max_rows,
                runtime.up,
            ),
            local_error,
        )
    if local_error is not None:
        raise ValueError(local_error)
    assert shape is not None
    tokens, heads, head_dim = shape
    prepared = prepare_peer_write_gemm_thd(
        weight,
        runtime,
        heads=heads,
        head_dim=head_dim,
        tokens=tokens,
        layout=layout,
        bias=bias,
        validate_group=validate_group,
    )
    return prepared(x, out=out, validate_group=validate_group)


def peer_write_gemm_sbhd(
    x: Tensor,
    weight: Tensor,
    runtime: MegaUPRuntime,
    *,
    bias: Tensor | None = None,
    out: Tensor | None = None,
    validate_group: bool = True,
) -> Tensor:
    """Fuse ``linear([S,B,H,D], weight, bias)`` with the Ulysses head all-to-all
    for sequence-first data.

    ``x`` holds this rank's ``S`` sequence positions and ``weight`` is
    ``[N, H*D]``. The returned tensor has shape ``[UP*S,B,N/(UP*D),D]``: the
    ``UP`` sequence chunks in rank order. Rows are sequence-major, so the
    receive slots already have this layout and the materialization is one
    contiguous copy for every ``B``. Repeated calls with one shape should use
    :func:`prepare_peer_write_gemm_sbhd`.
    """

    shape: tuple[int, int, int, int] | None = None
    local_error: str | None = None
    try:
        shape = _peer_shape_sbhd(x, runtime, **_weight_widths(weight))
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    if validate_group:
        _collective_validate(
            runtime,
            "peer_write_gemm_sbhd_input",
            (
                shape,
                _tensor_signature(weight),
                _optional_signature(bias),
                _optional_signature(out),
                runtime.hidden,
                runtime.max_rows,
                runtime.up,
            ),
            local_error,
        )
    if local_error is not None:
        raise ValueError(local_error)
    assert shape is not None
    sequence, batch, heads, head_dim = shape
    prepared = prepare_peer_write_gemm_sbhd(
        weight,
        runtime,
        sequence=sequence,
        batch=batch,
        heads=heads,
        head_dim=head_dim,
        bias=bias,
        validate_group=validate_group,
    )
    return prepared(x, out=out, validate_group=validate_group)


# ---------------------------------------------------------------------------
# Head-shard pull GEMM
# ---------------------------------------------------------------------------


def _compile_pull(
    weight: Tensor,
    runtime: MegaUPRuntime,
    *,
    bias: Tensor | None,
    rows: int,
    workspace_epochs: int,
    operation: str,
    metadata: object,
    validate_group: bool,
) -> tuple[Tensor, Tensor, object]:
    """Allocate pull buffers and compile the kernel with collective error checks."""

    materialized_input = None
    output_2d = None
    compiled = None
    setup_error: str | None = None
    try:
        materialized_input = torch.empty(
            (rows, int(weight.shape[1])),
            dtype=torch.bfloat16,
            device=runtime.device,
        )
        output_2d = torch.empty(
            (rows, int(weight.shape[0])),
            dtype=torch.bfloat16,
            device=runtime.device,
        )
        from .kernels.headshard_pull_gemm import (
            compile_headshard_pull_gemm,
        )

        compiled = compile_headshard_pull_gemm(
            runtime.peer_address_table,
            weight,
            output_2d,
            materialized_input,
            logical_m=rows,
            rank=runtime.local_rank,
            up=runtime.up,
            bias=bias,
            workspace_epochs=workspace_epochs,
            peer_rows=rows,
            enable_clc=True,
            enable_bulk_pull=True,
        )
    except Exception as error:
        setup_error = f"{type(error).__name__}: {error}"
    if validate_group:
        _collective_validate(
            runtime,
            f"{operation}_setup",
            metadata,
            setup_error,
        )
    if setup_error is not None:
        raise RuntimeError(setup_error)
    assert materialized_input is not None
    assert output_2d is not None
    assert compiled is not None
    return materialized_input, output_2d, compiled


def _run_prepared_pull(
    runtime: MegaUPRuntime,
    compiled: object,
    stage: Callable[[], object],
) -> None:
    with torch.no_grad(), runtime.lease():
        stream = torch.cuda.current_stream(runtime.device)
        if compiled.epoch >= compiled.workspace_epochs:
            compiled.reset_workspace(stream)
        stage()
        compiled(stream=stream)
        runtime.finish_pull()


@dataclass
class PreparedHeadShardPullGemm(_ProjectionWidths):
    """Reusable pull-GEMM executable with stable input/output buffers."""

    runtime: MegaUPRuntime
    weight: Tensor
    batch: int
    sequence: int
    heads: int
    head_dim: int
    materialized_input: Tensor
    output_2d: Tensor
    compiled: object
    bias: Tensor | None = None

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        return (
            self.batch,
            self.runtime.up * self.sequence,
            self.heads // self.runtime.up,
            self.head_dim,
        )

    @property
    def output(self) -> Tensor:
        return self.output_2d.view(
            self.batch,
            self.sequence,
            self.out_features // self.head_dim,
            self.head_dim,
        )

    def __call__(
        self,
        head_x: Tensor,
        *,
        validate_group: bool = True,
    ) -> Tensor:
        expected = (
            self.batch,
            self.sequence,
            self.heads,
            self.head_dim,
        )
        shape: tuple[int, int, int, int] | None = None
        local_error: str | None = None
        try:
            shape = _pull_shape(head_x, self.runtime, in_features=self.in_features)
            if shape != expected:
                raise ValueError(
                    "prepared pull-GEMM expects logical BSHD "
                    f"{expected}, got {shape}"
                )
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
        if validate_group:
            _collective_validate(
                self.runtime,
                "headshard_pull_gemm_bshd",
                (
                    shape,
                    _tensor_signature(self.weight),
                    _optional_signature(self.bias),
                    int(self.compiled.epoch),
                    int(self.compiled.workspace_epochs),
                    self.runtime.hidden,
                    self.runtime.max_rows,
                    self.runtime.up,
                ),
                local_error,
            )
        if local_error is not None:
            raise ValueError(local_error)
        _run_prepared_pull(
            self.runtime,
            self.compiled,
            lambda: self.runtime.stage_head_parallel(head_x),
        )
        return self.output


@dataclass
class PreparedHeadShardPullGemmSBHD(_ProjectionWidths):
    """Reusable sequence-first pull-GEMM executable with stable buffers."""

    runtime: MegaUPRuntime
    weight: Tensor
    sequence: int
    batch: int
    heads: int
    head_dim: int
    materialized_input: Tensor
    output_2d: Tensor
    compiled: object
    bias: Tensor | None = None

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        return (
            self.runtime.up * self.sequence,
            self.batch,
            self.heads // self.runtime.up,
            self.head_dim,
        )

    @property
    def output(self) -> Tensor:
        return self.output_2d.view(
            self.sequence,
            self.batch,
            self.out_features // self.head_dim,
            self.head_dim,
        )

    def __call__(
        self,
        head_x: Tensor,
        *,
        validate_group: bool = True,
    ) -> Tensor:
        expected = (self.sequence, self.batch, self.heads, self.head_dim)
        shape: tuple[int, int, int, int] | None = None
        local_error: str | None = None
        try:
            shape = _pull_shape_sbhd(
                head_x, self.runtime, in_features=self.in_features
            )
            if shape != expected:
                raise ValueError(
                    "prepared SBHD pull-GEMM expects logical (S,B,H,D) "
                    f"{expected}, got {shape}"
                )
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
        if validate_group:
            _collective_validate(
                self.runtime,
                "headshard_pull_gemm_sbhd",
                (
                    shape,
                    _tensor_signature(self.weight),
                    _optional_signature(self.bias),
                    int(self.compiled.epoch),
                    int(self.compiled.workspace_epochs),
                    self.runtime.hidden,
                    self.runtime.max_rows,
                    self.runtime.up,
                ),
                local_error,
            )
        if local_error is not None:
            raise ValueError(local_error)
        up = self.runtime.up
        rows = self.sequence * self.batch
        # Rank chunks follow each other along S and rows are sequence-major,
        # so the destination-major staging is the contiguous B=1 copy.
        _run_prepared_pull(
            self.runtime,
            self.compiled,
            lambda: self.runtime.stage_head_parallel(
                head_x.view(1, up * rows, self.heads // up, self.head_dim)
            ),
        )
        return self.output


@dataclass
class PreparedHeadShardPullGemmTHD(_ProjectionWidths):
    """Reusable packed-THD pull-GEMM executable for a fixed token count.

    ``layout`` is ``None`` for the contiguous-slab contract (input rows in
    source-rank order) or a ``THDLayout`` for split-sample regrouped input.
    """

    runtime: MegaUPRuntime
    weight: Tensor
    tokens: int
    heads: int
    head_dim: int
    materialized_input: Tensor
    output_2d: Tensor
    compiled: object
    layout: THDLayout | None = None
    bias: Tensor | None = None

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (
            self.runtime.up * self.tokens,
            self.heads // self.runtime.up,
            self.head_dim,
        )

    @property
    def output(self) -> Tensor:
        return self.output_2d.view(
            self.tokens, self.out_features // self.head_dim, self.head_dim
        )

    def __call__(
        self,
        head_x: Tensor,
        *,
        validate_group: bool = True,
    ) -> Tensor:
        expected = (self.tokens, self.heads, self.head_dim)
        shape: tuple[int, int, int] | None = None
        local_error: str | None = None
        try:
            shape = _pull_shape_thd(
                head_x, self.runtime, self.layout, in_features=self.in_features
            )
            if shape != expected:
                raise ValueError(
                    "prepared THD pull-GEMM expects logical (T,H,D) "
                    f"{expected}, got {shape}"
                )
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
        if validate_group:
            _collective_validate(
                self.runtime,
                "headshard_pull_gemm_thd",
                (
                    shape,
                    _layout_fingerprint(self.layout),
                    _tensor_signature(self.weight),
                    _optional_signature(self.bias),
                    int(self.compiled.epoch),
                    int(self.compiled.workspace_epochs),
                    self.runtime.hidden,
                    self.runtime.max_rows,
                    self.runtime.up,
                ),
                local_error,
            )
        if local_error is not None:
            raise ValueError(local_error)
        up = self.runtime.up
        layout = self.layout

        def stage() -> None:
            if layout is None:
                self.runtime.stage_head_parallel(
                    head_x.view(1, up * self.tokens, self.heads // up, self.head_dim)
                )
            else:
                self.runtime.stage_rows_indexed(
                    rows=self.tokens,
                    source=head_x.view(up * self.tokens, self.in_features // up),
                    index=layout.to_source_major_index,
                )

        _run_prepared_pull(self.runtime, self.compiled, stage)
        return self.output


def prepare_headshard_pull_gemm_bshd(
    weight: Tensor,
    runtime: MegaUPRuntime,
    *,
    batch: int,
    sequence: int,
    heads: int,
    head_dim: int,
    bias: Tensor | None = None,
    workspace_epochs: int = 4096,
    validate_group: bool = True,
) -> PreparedHeadShardPullGemm:
    """Prepare a reusable ``all-to-all + linear(weight, bias)`` replacement:
    ``[B,UP*S,H/UP,D]`` to ``[B,S,N/D,D]``.

    ``weight`` is ``[N, K]`` with ``K = heads * head_dim``. The returned
    output aliases a reusable buffer and is overwritten by the next call.
    Call ``clone()`` when the result must outlive a subsequent call.
    """

    logical_shape = (batch, sequence, heads, head_dim)
    local_error: str | None = None
    try:
        _check_runtime_device(runtime)
        _check_logical_shape(logical_shape)
        out_features, in_features = _check_pull_weight(weight, runtime, head_dim)
        _check_bias(bias, runtime, out_features)
        _check_pull_heads(heads, head_dim, runtime, in_features)
        runtime.check_capacity(batch * sequence, in_features)
        if workspace_epochs <= 0:
            raise ValueError(
                "workspace_epochs must be positive, "
                f"got {workspace_epochs}"
            )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    if validate_group:
        _collective_validate(
            runtime,
            "prepare_headshard_pull_gemm_bshd",
            (
                logical_shape,
                _tensor_signature(weight),
                _optional_signature(bias),
                repr(workspace_epochs),
                runtime.hidden,
                runtime.max_rows,
                runtime.up,
            ),
            local_error,
        )
    if local_error is not None:
        raise ValueError(local_error)
    materialized_input, output_2d, compiled = _compile_pull(
        weight,
        runtime,
        bias=bias,
        rows=batch * sequence,
        workspace_epochs=workspace_epochs,
        operation="prepare_headshard_pull_gemm_bshd",
        metadata=(
            logical_shape,
            _tensor_signature(weight),
            _optional_signature(bias),
            repr(workspace_epochs),
        ),
        validate_group=validate_group,
    )
    return PreparedHeadShardPullGemm(
        runtime=runtime,
        weight=weight,
        batch=batch,
        sequence=sequence,
        heads=heads,
        head_dim=head_dim,
        materialized_input=materialized_input,
        output_2d=output_2d,
        compiled=compiled,
        bias=bias,
    )


def prepare_headshard_pull_gemm_sbhd(
    weight: Tensor,
    runtime: MegaUPRuntime,
    *,
    sequence: int,
    batch: int,
    heads: int,
    head_dim: int,
    bias: Tensor | None = None,
    workspace_epochs: int = 4096,
    validate_group: bool = True,
) -> PreparedHeadShardPullGemmSBHD:
    """Prepare a reusable sequence-first ``all-to-all + linear(weight, bias)``
    replacement: ``[UP*S,B,H/UP,D]`` to ``[S,B,N/D,D]`` for an ``[N, K]``
    weight with ``K = heads * head_dim``.

    The returned output aliases a reusable buffer and is overwritten by the
    next call. Call ``clone()`` when the result must outlive a subsequent call.
    """

    logical_shape = (sequence, batch, heads, head_dim)
    local_error: str | None = None
    try:
        _check_runtime_device(runtime)
        _check_logical_shape(logical_shape)
        out_features, in_features = _check_pull_weight(weight, runtime, head_dim)
        _check_bias(bias, runtime, out_features)
        _check_pull_heads(heads, head_dim, runtime, in_features)
        runtime.check_capacity(sequence * batch, in_features)
        if workspace_epochs <= 0:
            raise ValueError(
                f"workspace_epochs must be positive, got {workspace_epochs}"
            )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    metadata = (
        logical_shape,
        _tensor_signature(weight),
        _optional_signature(bias),
        repr(workspace_epochs),
    )
    if validate_group:
        _collective_validate(
            runtime,
            "prepare_headshard_pull_gemm_sbhd",
            (*metadata, runtime.hidden, runtime.max_rows, runtime.up),
            local_error,
        )
    if local_error is not None:
        raise ValueError(local_error)
    materialized_input, output_2d, compiled = _compile_pull(
        weight,
        runtime,
        bias=bias,
        rows=sequence * batch,
        workspace_epochs=workspace_epochs,
        operation="prepare_headshard_pull_gemm_sbhd",
        metadata=metadata,
        validate_group=validate_group,
    )
    return PreparedHeadShardPullGemmSBHD(
        runtime=runtime,
        weight=weight,
        sequence=sequence,
        batch=batch,
        heads=heads,
        head_dim=head_dim,
        materialized_input=materialized_input,
        output_2d=output_2d,
        compiled=compiled,
        bias=bias,
    )


def prepare_headshard_pull_gemm_thd(
    weight: Tensor,
    runtime: MegaUPRuntime,
    *,
    heads: int,
    head_dim: int,
    tokens: int | None = None,
    layout: THDLayout | None = None,
    bias: Tensor | None = None,
    workspace_epochs: int = 4096,
    validate_group: bool = True,
) -> PreparedHeadShardPullGemmTHD:
    """Prepare a reusable packed-THD ``all-to-all + linear(weight, bias)``
    replacement.

    The executable accepts ``[UP*T,H/UP,D]`` inputs in the order produced by
    :func:`peer_write_gemm_thd` with the same ``layout`` (``None`` for the
    contiguous-slab contract) and returns ``[T,N/D,D]`` for an ``[N, K]``
    weight with ``K = heads * head_dim``. ``tokens`` may be omitted when
    ``layout`` is given. The output aliases a reusable buffer.
    """

    local_error: str | None = None
    fingerprint = _layout_fingerprint(layout)
    logical_shape: tuple[int, int, int] | None = None
    try:
        _check_runtime_device(runtime)
        _check_logical_shape((heads, head_dim))
        out_features, in_features = _check_pull_weight(weight, runtime, head_dim)
        _check_bias(bias, runtime, out_features)
        if layout is not None:
            _check_layout(layout, runtime)
            if tokens is None:
                tokens = layout.tokens
            elif tokens != layout.tokens:
                raise ValueError(
                    f"tokens={tokens} disagrees with layout.tokens={layout.tokens}"
                )
        if tokens is None:
            raise ValueError("tokens is required when no THD layout is given")
        logical_shape = (tokens, heads, head_dim)
        _check_logical_shape(logical_shape)
        _check_pull_heads(heads, head_dim, runtime, in_features)
        runtime.check_capacity(tokens, in_features)
        if workspace_epochs <= 0:
            raise ValueError(
                "workspace_epochs must be positive, "
                f"got {workspace_epochs}"
            )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    if validate_group:
        _collective_validate(
            runtime,
            "prepare_headshard_pull_gemm_thd",
            (
                logical_shape,
                fingerprint,
                _tensor_signature(weight),
                _optional_signature(bias),
                repr(workspace_epochs),
                runtime.hidden,
                runtime.max_rows,
                runtime.up,
            ),
            local_error,
        )
    if local_error is not None:
        raise ValueError(local_error)
    assert tokens is not None
    materialized_input, output_2d, compiled = _compile_pull(
        weight,
        runtime,
        bias=bias,
        rows=tokens,
        workspace_epochs=workspace_epochs,
        operation="prepare_headshard_pull_gemm_thd",
        metadata=(
            logical_shape,
            fingerprint,
            _tensor_signature(weight),
            _optional_signature(bias),
            repr(workspace_epochs),
        ),
        validate_group=validate_group,
    )
    return PreparedHeadShardPullGemmTHD(
        runtime=runtime,
        weight=weight,
        tokens=tokens,
        heads=heads,
        head_dim=head_dim,
        materialized_input=materialized_input,
        output_2d=output_2d,
        compiled=compiled,
        layout=layout,
        bias=bias,
    )


def headshard_pull_gemm_bshd(
    head_x: Tensor,
    weight: Tensor,
    runtime: MegaUPRuntime,
    *,
    bias: Tensor | None = None,
    validate_group: bool = True,
) -> Tensor:
    """Pull remote head shards and compute ``linear(x, weight, bias)`` in one
    kernel: ``[B,UP*S,H/UP,D]`` to ``[B,S,N/D,D]`` for an ``[N, H*D]``
    weight."""

    shape: tuple[int, int, int, int] | None = None
    local_error: str | None = None
    try:
        shape = _pull_shape(
            head_x, runtime, in_features=_weight_widths(weight)["in_features"]
        )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    if validate_group:
        _collective_validate(
            runtime,
            "headshard_pull_gemm_bshd_input",
            (
                shape,
                _tensor_signature(weight),
                _optional_signature(bias),
                runtime.hidden,
                runtime.max_rows,
                runtime.up,
            ),
            local_error,
        )
    if local_error is not None:
        raise ValueError(local_error)
    assert shape is not None
    batch, sequence, heads, head_dim = shape
    prepared = prepare_headshard_pull_gemm_bshd(
        weight,
        runtime,
        batch=batch,
        sequence=sequence,
        heads=heads,
        head_dim=head_dim,
        bias=bias,
        workspace_epochs=1,
        validate_group=validate_group,
    )
    return prepared(head_x, validate_group=validate_group)


def headshard_pull_gemm_sbhd(
    head_x: Tensor,
    weight: Tensor,
    runtime: MegaUPRuntime,
    *,
    bias: Tensor | None = None,
    validate_group: bool = True,
) -> Tensor:
    """One-shot sequence-first pull-GEMM: ``[UP*S,B,H/UP,D]`` to
    ``[S,B,N/D,D]`` for an ``[N, H*D]`` weight."""

    shape: tuple[int, int, int, int] | None = None
    local_error: str | None = None
    try:
        shape = _pull_shape_sbhd(
            head_x, runtime, in_features=_weight_widths(weight)["in_features"]
        )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    if validate_group:
        _collective_validate(
            runtime,
            "headshard_pull_gemm_sbhd_input",
            (
                shape,
                _tensor_signature(weight),
                _optional_signature(bias),
                runtime.hidden,
                runtime.max_rows,
                runtime.up,
            ),
            local_error,
        )
    if local_error is not None:
        raise ValueError(local_error)
    assert shape is not None
    sequence, batch, heads, head_dim = shape
    prepared = prepare_headshard_pull_gemm_sbhd(
        weight,
        runtime,
        sequence=sequence,
        batch=batch,
        heads=heads,
        head_dim=head_dim,
        bias=bias,
        workspace_epochs=1,
        validate_group=validate_group,
    )
    return prepared(head_x, validate_group=validate_group)


def headshard_pull_gemm_thd(
    head_x: Tensor,
    weight: Tensor,
    runtime: MegaUPRuntime,
    *,
    bias: Tensor | None = None,
    layout: THDLayout | None = None,
    validate_group: bool = True,
) -> Tensor:
    """One-shot packed-THD pull-GEMM: ``[UP*T,H/UP,D]`` to ``[T,N/D,D]`` for
    an ``[N, H*D]`` weight."""

    shape: tuple[int, int, int] | None = None
    local_error: str | None = None
    try:
        shape = _pull_shape_thd(
            head_x,
            runtime,
            layout,
            in_features=_weight_widths(weight)["in_features"],
        )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    if validate_group:
        _collective_validate(
            runtime,
            "headshard_pull_gemm_thd_input",
            (
                shape,
                _layout_fingerprint(layout),
                _tensor_signature(weight),
                _optional_signature(bias),
                runtime.hidden,
                runtime.max_rows,
                runtime.up,
            ),
            local_error,
        )
    if local_error is not None:
        raise ValueError(local_error)
    assert shape is not None
    tokens, heads, head_dim = shape
    prepared = prepare_headshard_pull_gemm_thd(
        weight,
        runtime,
        heads=heads,
        head_dim=head_dim,
        tokens=tokens,
        layout=layout,
        bias=bias,
        workspace_epochs=1,
        validate_group=validate_group,
    )
    return prepared(head_x, validate_group=validate_group)


__all__ = [
    "PreparedHeadShardPullGemm",
    "PreparedHeadShardPullGemmSBHD",
    "PreparedHeadShardPullGemmTHD",
    "PreparedPeerWriteGemm",
    "PreparedPeerWriteGemmSBHD",
    "PreparedPeerWriteGemmTHD",
    "THDLayout",
    "headshard_pull_gemm_bshd",
    "headshard_pull_gemm_sbhd",
    "headshard_pull_gemm_thd",
    "peer_write_gemm_bshd",
    "peer_write_gemm_sbhd",
    "peer_write_gemm_thd",
    "prepare_headshard_pull_gemm_bshd",
    "prepare_headshard_pull_gemm_sbhd",
    "prepare_headshard_pull_gemm_thd",
    "prepare_peer_write_gemm_bshd",
    "prepare_peer_write_gemm_sbhd",
    "prepare_peer_write_gemm_thd",
    "prepare_thd_layout",
]
