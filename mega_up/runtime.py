# Copyright (c) The Mega-UP Authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Symmetric-memory runtime shared by the two Mega-UP kernels."""

from __future__ import annotations

import operator
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Iterator, Sequence

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch import Tensor

SUPPORTED_UP_DEGREES = (2, 4, 8)
# Default symmetric-arena capacity in rows (B*S) per rank. Callers should pass
# ``max_rows`` explicitly to match their largest local token count.
DEFAULT_MAX_ROWS = 131_072
# The GEMM N tiles are 256 wide: the peer-write tiles its exchanged output
# width with them, the pull its output width.
HIDDEN_MULTIPLE = 256
# One warp-wide pull copy with 64-bit lanes moves 128 BF16 columns.
SHARD_MULTIPLE = 128


def unsupported_shape_reason(width: int, up: int) -> str | None:
    """Return why ``width`` columns cannot be exchanged across ``up`` ranks,
    else ``None``.

    The exchanged width is the output width of a peer-write projection and
    the input width of a pull projection.
    """

    if up not in SUPPORTED_UP_DEGREES:
        return f"UP must be one of {SUPPORTED_UP_DEGREES}, got {up}"
    if width <= 0 or width % up:
        return f"width={width} must be positive and divisible by UP={up}"
    if width % HIDDEN_MULTIPLE:
        return f"width={width} must be a multiple of {HIDDEN_MULTIPLE}"
    if (width // up) % SHARD_MULTIPLE:
        return (
            f"the head shard width/UP={width // up} must be a multiple of "
            f"{SHARD_MULTIPLE}"
        )
    return None


def supports_shape(width: int, up: int) -> bool:
    """Whether ``width`` columns can be exchanged across ``up`` ranks."""

    return unsupported_shape_reason(width, up) is None


def _contiguous_stride(shape: Sequence[int]) -> tuple[int, ...]:
    stride = [1] * len(shape)
    for dim in range(len(shape) - 2, -1, -1):
        stride[dim] = stride[dim + 1] * int(shape[dim + 1])
    return tuple(stride)


def _group_key(group: dist.ProcessGroup) -> str:
    name = getattr(group, "group_name", None)
    return str(name) if name is not None else f"process-group-{id(group)}"


def _allocate_symmetric(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    group: dist.ProcessGroup,
) -> Tensor:
    if hasattr(symm_mem, "empty"):
        try:
            return symm_mem.empty(shape, dtype=dtype, device=device)
        except TypeError:
            pass
    allocator = getattr(symm_mem, "_SymmetricMemory")
    return allocator.empty_strided_p2p(
        size=shape,
        stride=_contiguous_stride(shape),
        dtype=dtype,
        device=device,
        group_name=group.group_name,
    )


def _rendezvous(tensor: Tensor, group: dist.ProcessGroup):
    if hasattr(symm_mem, "rendezvous"):
        for group_arg in (group, group.group_name):
            try:
                return symm_mem.rendezvous(tensor, group_arg)
            except TypeError:
                continue
    allocator = getattr(symm_mem, "_SymmetricMemory")
    handle = allocator.rendezvous(tensor)
    if handle is None:
        raise RuntimeError("Mega-UP arena was not allocated symmetrically")
    return handle


def _barrier(handle) -> None:
    try:
        handle.barrier(channel=0)
    except TypeError:
        handle.barrier()


@dataclass
class MegaUPRuntime:
    """One reusable BF16 symmetric arena for an explicit UP process group.

    The arena holds ``max_rows * hidden`` elements per rank. An operation
    exchanging ``rows`` rows of ``width`` elements fits when
    ``rows * width <= max_rows * hidden``; ``width`` defaults to ``hidden``.
    """

    group: dist.ProcessGroup
    arena: Tensor
    handle: object
    peer_buffers: tuple[Tensor, ...]
    peer_address_table: Tensor
    max_rows: int
    hidden: int
    local_rank: int
    up: int
    _leased: bool = field(default=False, init=False, repr=False)
    _lease_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _completion_event: torch.cuda.Event | None = field(
        default=None, init=False, repr=False
    )

    @property
    def shard_hidden(self) -> int:
        return self.hidden // self.up

    @property
    def device(self) -> torch.device:
        return self.arena.device

    @property
    def arena_nbytes(self) -> int:
        return self.arena.numel() * self.arena.element_size()

    @contextmanager
    def lease(self) -> Iterator["MegaUPRuntime"]:
        stream = torch.cuda.current_stream(self.device)
        with self._lease_lock:
            if self._leased:
                raise RuntimeError(
                    "Mega-UP arena was re-entered before the previous operation "
                    "completed"
                )
            self._leased = True
            completion_event = self._completion_event
        if completion_event is not None:
            stream.wait_event(completion_event)
        try:
            yield self
        finally:
            completion_event = torch.cuda.Event()
            completion_event.record(stream)
            with self._lease_lock:
                self._completion_event = completion_event
                self._leased = False

    def barrier(self) -> None:
        _barrier(self.handle)

    def check_capacity(self, rows: int, width: int) -> None:
        """Check that ``rows`` rows of ``width`` elements fit the arena."""

        if rows <= 0 or width <= 0 or width % self.up:
            raise ValueError(
                "rows and width must be positive and width divisible by "
                f"UP={self.up}, got rows={rows}, width={width}"
            )
        if rows * width > self.max_rows * self.hidden:
            raise ValueError(
                f"{rows} rows of {width} elements exceed the arena of "
                f"max_rows*hidden={self.max_rows}*{self.hidden} elements per rank"
            )

    def slots(self, rows: int, width: int | None = None) -> Tensor:
        """``[UP, rows, width/UP]`` view of the densely packed arena slots.

        Slot ``p`` starts at element ``p * rows * width/UP``. Peer writes
        receive into it source-major and pulls stage into it
        destination-major, so receiving and staging are contiguous copies.
        """

        width = self.hidden if width is None else int(width)
        self.check_capacity(rows, width)
        shard = width // self.up
        return (
            self.arena.view(-1)
            .narrow(0, 0, self.up * rows * shard)
            .view(self.up, rows, shard)
        )

    def remote_write_views(
        self, rows: int, width: int | None = None
    ) -> tuple[Tensor, ...]:
        """Return this source rank's slot in every peer arena.

        Peer-write receives are source-major: ``[source_rank, B*S, width/UP]``
        in the densely packed slots of :meth:`slots`.
        """

        width = self.hidden if width is None else int(width)
        self.check_capacity(rows, width)
        source_numel = rows * (width // self.up)
        source_offset = self.local_rank * source_numel
        return tuple(
            peer.view(-1)
            .narrow(0, source_offset, source_numel)
            .view(rows, width // self.up)
            for peer in self.peer_buffers
        )

    def copy_peer_written(
        self,
        *,
        batch: int,
        sequence: int,
        shard_heads: int,
        head_dim: int,
        out: Tensor | None = None,
    ) -> Tensor:
        """Materialize compact peer writes as ``[B,UP*S,H/UP,D]``."""

        rows = int(batch) * int(sequence)
        width = self.up * int(shard_heads) * int(head_dim)
        self.check_capacity(rows, width)
        expected_shape = (
            batch,
            self.up * sequence,
            shard_heads,
            head_dim,
        )
        if out is None:
            out = torch.empty(
                expected_shape,
                dtype=self.arena.dtype,
                device=self.arena.device,
            )
        elif (
            tuple(out.shape) != expected_shape
            or out.dtype != self.arena.dtype
            or out.device != self.arena.device
            or not out.is_contiguous()
        ):
            raise ValueError(
                "out must be a contiguous BF16 tensor on the runtime device "
                f"with shape {expected_shape}"
            )

        self.barrier()
        source_major = self.slots(rows, width).view(
            self.up,
            batch,
            sequence,
            shard_heads,
            head_dim,
        )
        out.view(
            batch,
            self.up,
            sequence,
            shard_heads,
            head_dim,
        ).copy_(
            source_major.permute(1, 0, 2, 3, 4)
        )
        self.barrier()
        return out

    def _check_row_index(self, index: Tensor, rows: int) -> None:
        total = self.up * rows
        if (
            not isinstance(index, Tensor)
            or index.ndim != 1
            or index.numel() != total
            or index.dtype != torch.int64
            or index.device != self.device
        ):
            raise ValueError(
                f"index must be a 1-D int64 tensor with UP*rows={total} "
                "entries on the runtime device"
            )

    def copy_peer_written_indexed(
        self,
        *,
        rows: int,
        index: Tensor,
        out: Tensor,
    ) -> Tensor:
        """Materialize compact peer writes in an arbitrary row order.

        Source-major arena row ``r*rows + j`` holds local row ``j`` written
        by source rank ``r``. Output row ``i`` receives arena row
        ``index[i]``; packed THD uses ``THDLayout.to_thd_index``.
        """

        self._check_row_index(index, rows)
        total = self.up * rows
        if (
            out.dtype != self.arena.dtype
            or out.device != self.arena.device
            or not out.is_contiguous()
            or out.numel() == 0
            or out.numel() % total
        ):
            raise ValueError(
                "out must be a contiguous BF16 tensor on the runtime device "
                f"with a multiple of UP*rows={total} elements"
            )
        shard = out.numel() // total
        self.check_capacity(rows, self.up * shard)

        self.barrier()
        source_major = self.slots(rows, self.up * shard).view(total, shard)
        torch.index_select(
            source_major,
            0,
            index,
            out=out.view(total, shard),
        )
        self.barrier()
        return out

    def stage_rows_indexed(
        self,
        *,
        rows: int,
        source: Tensor,
        index: Tensor,
    ) -> None:
        """Stage ``UP*rows`` shard rows for remote pull in destination order.

        Slot row ``(destination d, j)`` receives ``source[index[d*rows + j]]``;
        packed THD uses ``THDLayout.to_source_major_index``. The slots are
        densely packed, so this is a single row gather.
        """

        self._check_row_index(index, rows)
        total = self.up * rows
        if (
            source.ndim != 2
            or int(source.shape[0]) != total
            or source.dtype != self.arena.dtype
            or source.device != self.arena.device
            or not source.is_contiguous()
        ):
            raise ValueError(
                "source must be a contiguous 2-D BF16 tensor on the runtime "
                f"device with UP*rows={total} rows"
            )
        shard = int(source.shape[1])
        torch.index_select(
            source,
            0,
            index,
            out=self.slots(rows, self.up * shard).view(total, shard),
        )
        self.barrier()

    def stage_head_parallel(self, tensor: Tensor) -> tuple[int, int, int, int]:
        """Stage ``[B,UP*S,H/UP,D]`` for remote pull.

        Each local arena holds ``[destination_rank, B*S, H/UP*D]`` in the
        densely packed slots of :meth:`slots`; peers pull their slot with row
        stride ``B*S``. For ``B=1`` the staging is one contiguous copy.
        """

        if tensor.ndim != 4:
            raise ValueError(
                "head-parallel input must be [B,UP*S,H/UP,D], got "
                f"{tuple(tensor.shape)}"
            )
        batch, global_sequence, shard_heads, head_dim = map(
            int, tensor.shape
        )
        if global_sequence % self.up:
            raise ValueError(
                f"sequence dimension {global_sequence} is not divisible by "
                f"UP={self.up}"
            )
        sequence = global_sequence // self.up
        rows = batch * sequence
        width = self.up * shard_heads * head_dim
        self.slots(rows, width).view(
            self.up,
            batch,
            sequence,
            shard_heads,
            head_dim,
        ).copy_(
            tensor.view(
                batch,
                self.up,
                sequence,
                shard_heads,
                head_dim,
            )
            .permute(1, 0, 2, 3, 4)
        )
        self.barrier()
        return batch, sequence, shard_heads, head_dim

    def finish_pull(self) -> None:
        """Prevent arena reuse until every peer pull has completed."""

        self.barrier()


# One runtime per process group and ``hidden``.
_RUNTIMES: dict[tuple[str, int], MegaUPRuntime] = {}


def _raise_collective_phase_errors(
    group: dist.ProcessGroup,
    *,
    up: int,
    device: torch.device,
    phase: str,
    local_error: str | None,
) -> None:
    records: list[object | None] = [None] * up
    with torch.cuda.device(device):
        dist.all_gather_object(
            records,
            (phase, local_error),
            group=group,
        )
    errors = tuple(
        (rank, candidate[1])
        for rank, candidate in enumerate(records)
        if isinstance(candidate, tuple) and candidate[1] is not None
    )
    if errors:
        raise RuntimeError(f"Mega-UP {phase} failed: {errors}")


def initialize_runtime(
    group: dist.ProcessGroup,
    *,
    hidden: int,
    max_rows: int = DEFAULT_MAX_ROWS,
    device: torch.device | str | None = None,
) -> MegaUPRuntime:
    """Collectively allocate the runtime for ``group``.

    The arena holds ``max_rows * hidden`` elements per rank; ``hidden`` is
    an exchanged width (peer-write ``out_features`` or pull ``in_features``).
    All ranks in the group must call this function with identical arguments.

    A group can hold several runtimes, one per ``hidden``, each with its own
    arena: projections of different exchanged widths either share a runtime
    sized for the widest of them or use one runtime each. A call with the
    ``hidden`` of an existing runtime returns that runtime and must repeat
    its ``max_rows`` and device.
    """

    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    up = dist.get_world_size(group)
    local_rank = dist.get_rank(group)
    collective_device = torch.device(
        "cuda", torch.cuda.current_device()
    )
    group_key = _group_key(group)
    normalized_device: torch.device | None = None
    local_error: str | None = None
    try:
        hidden = operator.index(hidden)
        reason = unsupported_shape_reason(hidden, up)
        if reason is not None:
            raise ValueError(f"hidden: {reason}")
        if max_rows <= 0:
            raise ValueError(
                f"max_rows must be positive, got {max_rows}"
            )
        if device is None:
            normalized_device = torch.device(
                "cuda", torch.cuda.current_device()
            )
        else:
            normalized_device = torch.device(device)
        if normalized_device.type != "cuda":
            raise ValueError(
                f"Mega-UP requires a CUDA device, got {normalized_device}"
            )
        requested_index = normalized_device.index
        if requested_index is None:
            requested_index = collective_device.index
        if (
            requested_index is None
            or requested_index < 0
            or requested_index >= torch.cuda.device_count()
        ):
            raise ValueError(
                f"invalid CUDA device ordinal {requested_index}"
            )
        normalized_device = torch.device("cuda", requested_index)
        if normalized_device != collective_device:
            raise ValueError(
                "runtime device must be the process's current CUDA device: "
                f"requested {normalized_device}, current {collective_device}"
            )
    except (RuntimeError, TypeError, ValueError) as error:
        local_error = f"{type(error).__name__}: {error}"
    existing = (
        _RUNTIMES.get((group_key, hidden)) if local_error is None else None
    )
    compatible = (
        local_error is None
        and (
            existing is None
            or (
                existing.max_rows == max_rows
                and existing.up == up
                and existing.device == normalized_device
            )
        )
    )
    record = (
        repr(hidden),
        repr(max_rows),
        up,
        None if normalized_device is None else normalized_device.type,
        existing is not None,
        compatible,
        local_error,
    )
    records: list[object | None] = [None] * up
    with torch.cuda.device(collective_device):
        dist.all_gather_object(records, record, group=group)
    errors = tuple(
        (rank, candidate[6])
        for rank, candidate in enumerate(records)
        if candidate is not None and candidate[6] is not None
    )
    if errors:
        raise ValueError(
            f"Mega-UP runtime collective validation failed: {errors}"
        )
    desired = record[:4]
    if any(candidate is None or candidate[:4] != desired for candidate in records):
        raise ValueError(
            "all UP ranks must initialize Mega-UP with identical hidden, "
            "max_rows, degree, and device type"
        )
    cache_states = tuple(
        (bool(candidate[4]), bool(candidate[5]))
        for candidate in records
        if candidate is not None
    )
    if len(set(state[0] for state in cache_states)) != 1:
        raise RuntimeError("Mega-UP runtime cache state differs across ranks")
    if any(not state[1] for state in cache_states):
        raise RuntimeError(
            f"the group already has a Mega-UP runtime for hidden={hidden} "
            "with a different max_rows or device"
        )
    if existing is not None:
        return existing

    assert normalized_device is not None
    device = normalized_device
    shard_hidden = hidden // up
    shape = (up, max_rows, shard_hidden)
    arena = None
    allocation_error: str | None = None
    try:
        arena = _allocate_symmetric(
            shape,
            dtype=torch.bfloat16,
            device=device,
            group=group,
        )
        arena.zero_()
        torch.cuda.synchronize(device)
    except Exception as error:
        allocation_error = f"{type(error).__name__}: {error}"
    _raise_collective_phase_errors(
        group,
        up=up,
        device=collective_device,
        phase="symmetric allocation",
        local_error=allocation_error,
    )
    assert arena is not None

    handle = None
    rendezvous_error: str | None = None
    try:
        handle = _rendezvous(arena, group)
    except Exception as error:
        rendezvous_error = f"{type(error).__name__}: {error}"
    _raise_collective_phase_errors(
        group,
        up=up,
        device=collective_device,
        phase="symmetric rendezvous",
        local_error=rendezvous_error,
    )
    assert handle is not None
    _barrier(handle)

    peer_buffers = None
    peer_address_table = None
    binding_error: str | None = None
    try:
        peer_buffers = tuple(
            handle.get_buffer(rank, shape, torch.bfloat16)
            for rank in range(up)
        )
        peer_address_table = torch.tensor(
            [peer.data_ptr() for peer in peer_buffers],
            dtype=torch.uint64,
            device=device,
        ).contiguous()
    except Exception as error:
        binding_error = f"{type(error).__name__}: {error}"
    _raise_collective_phase_errors(
        group,
        up=up,
        device=collective_device,
        phase="peer buffer binding",
        local_error=binding_error,
    )
    assert peer_buffers is not None
    assert peer_address_table is not None
    runtime = MegaUPRuntime(
        group=group,
        arena=arena,
        handle=handle,
        peer_buffers=peer_buffers,
        peer_address_table=peer_address_table,
        max_rows=max_rows,
        hidden=hidden,
        local_rank=local_rank,
        up=up,
    )
    _RUNTIMES[(group_key, hidden)] = runtime
    return runtime


__all__ = [
    "DEFAULT_MAX_ROWS",
    "HIDDEN_MULTIPLE",
    "SHARD_MULTIPLE",
    "SUPPORTED_UP_DEGREES",
    "MegaUPRuntime",
    "initialize_runtime",
    "supports_shape",
    "unsupported_shape_reason",
]
