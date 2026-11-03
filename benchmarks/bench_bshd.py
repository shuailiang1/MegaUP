#!/usr/bin/env python3
"""Benchmark Mega-UP BSHD kernels against explicit PyTorch pipelines."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.distributed as dist

from mega_up import (
    SUPPORTED_UP_DEGREES,
    THDLayout,
    __version__,
    initialize_runtime,
    prepare_headshard_pull_gemm_bshd,
    prepare_headshard_pull_gemm_thd,
    prepare_peer_write_gemm_bshd,
    prepare_peer_write_gemm_thd,
    prepare_thd_layout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--up", type=int, required=True)
    parser.add_argument(
        "--layout",
        choices=("bshd", "thd", "thd-split"),
        default="bshd",
        help=(
            "bshd: [B,S,H,D] per rank; thd: packed [T,H,D] per rank where "
            "T=--sequence is a contiguous slab of the global stream (output in "
            "global order, no regroup); thd-split: every sample is split "
            "evenly across ranks and regrouped through a THDLayout"
        ),
    )
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--thd-samples",
        type=int,
        default=8,
        help="thd-split only: number of packed samples per rank (random lengths)",
    )
    parser.add_argument(
        "--thd-min-len",
        type=int,
        default=64,
        help="thd-split only: minimum rank-local chunk length of a random sample",
    )
    parser.add_argument(
        "--thd-seqlens",
        type=str,
        default=None,
        help=(
            "thd-split only: explicit comma-separated rank-local sample "
            "lengths; overrides --sequence/--thd-samples"
        ),
    )
    parser.add_argument(
        "--op",
        choices=("peer-write", "pull-gemm", "both"),
        default="both",
    )
    parser.add_argument(
        "--bias",
        action="store_true",
        help="give both projections a bias (baselines use addmm)",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def source_revision() -> str | None:
    override = os.environ.get("MEGA_UP_SOURCE_REVISION")
    if override:
        return override
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def thd_sample_lengths(args: argparse.Namespace) -> tuple[int, ...]:
    """Rank-local packed sample lengths; identical on every rank."""

    if args.thd_seqlens is not None:
        lengths = tuple(int(part) for part in args.thd_seqlens.split(","))
        if not lengths or min(lengths) <= 0:
            raise ValueError("--thd-seqlens must be positive lengths")
        return lengths
    samples, minimum, tokens = args.thd_samples, args.thd_min_len, args.sequence
    if samples <= 0 or minimum <= 0:
        raise ValueError("--thd-samples and --thd-min-len must be positive")
    if tokens < samples * minimum:
        raise ValueError(
            f"--sequence={tokens} cannot hold {samples} samples of at least "
            f"{minimum} tokens"
        )
    generator = torch.Generator().manual_seed(args.seed)
    weights = torch.rand(samples, generator=generator, dtype=torch.float64)
    spare = tokens - samples * minimum
    extra = torch.floor(weights / weights.sum() * spare).to(torch.int64)
    remainder = spare - int(extra.sum().item())
    extra[:remainder] += 1
    return tuple(minimum + int(value) for value in extra.tolist())


def cu_seqlens_from_lengths(lengths: tuple[int, ...]) -> tuple[int, ...]:
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    return tuple(boundaries)


def reference_regroup_thd(
    source_major: torch.Tensor,
    cu_seqlens: tuple[int, ...],
) -> torch.Tensor:
    """Slice-by-slice ``[source,T,C]`` to packed ``[UP*T,C]`` reference."""

    up = source_major.shape[0]
    pieces = []
    for sample in range(len(cu_seqlens) - 1):
        start, stop = cu_seqlens[sample], cu_seqlens[sample + 1]
        for source in range(up):
            pieces.append(source_major[source, start:stop])
    return torch.cat(pieces, dim=0)


def reference_split_thd(
    head_parallel_2d: torch.Tensor,
    cu_seqlens: tuple[int, ...],
    *,
    up: int,
) -> torch.Tensor:
    """Slice-by-slice packed ``[UP*T,C]`` to ``[destination,T,C]`` reference."""

    tokens = cu_seqlens[-1]
    send = torch.empty(
        (up, tokens, head_parallel_2d.shape[1]),
        dtype=head_parallel_2d.dtype,
        device=head_parallel_2d.device,
    )
    for sample in range(len(cu_seqlens) - 1):
        start, stop = cu_seqlens[sample], cu_seqlens[sample + 1]
        length = stop - start
        for destination in range(up):
            begin = up * start + destination * length
            send[destination, start:stop] = head_parallel_2d[begin : begin + length]
    return send


def linear_into(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor,
) -> torch.Tensor:
    """``out = x @ weight.T (+ bias)`` into a preallocated buffer."""

    if bias is None:
        return torch.mm(x, weight.t(), out=out)
    return torch.addmm(bias, x, weight.t(), out=out)


class PeerWriteBaseline:
    """``GEMM + permute + all_to_all`` with reusable buffers."""

    def __init__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        up: int,
        group: dist.ProcessGroup,
        bias: torch.Tensor | None = None,
    ) -> None:
        self.x = x
        self.weight = weight
        self.bias = bias
        self.up = up
        self.group = group
        batch, sequence, heads, head_dim = map(int, x.shape)
        self.batch = batch
        self.sequence = sequence
        self.heads = heads
        self.head_dim = head_dim
        self.shard_heads = heads // up
        self.dense = torch.empty(
            (batch, sequence, heads, head_dim),
            dtype=x.dtype,
            device=x.device,
        )
        exchange_shape = (
            up,
            batch,
            sequence,
            self.shard_heads,
            head_dim,
        )
        self.send = torch.empty(
            exchange_shape, dtype=x.dtype, device=x.device
        )
        self.received = torch.empty_like(self.send)
        self.output = torch.empty(
            (batch, up * sequence, self.shard_heads, head_dim),
            dtype=x.dtype,
            device=x.device,
        )

    def __call__(self) -> torch.Tensor:
        hidden = self.heads * self.head_dim
        linear_into(
            self.x.view(-1, hidden),
            self.weight,
            self.bias,
            self.dense.view(-1, hidden),
        )
        self.send.copy_(
            self.dense.view(
                self.batch,
                self.sequence,
                self.up,
                self.shard_heads,
                self.head_dim,
            ).permute(2, 0, 1, 3, 4)
        )
        dist.all_to_all_single(
            self.received,
            self.send,
            group=self.group,
        )
        self.output.view(
            self.batch,
            self.up,
            self.sequence,
            self.shard_heads,
            self.head_dim,
        ).copy_(
            self.received.permute(1, 0, 2, 3, 4)
        )
        return self.output


class PullGemmBaseline:
    """``all_to_all + permute + GEMM`` with reusable buffers."""

    def __init__(
        self,
        head_x: torch.Tensor,
        weight: torch.Tensor,
        *,
        up: int,
        group: dist.ProcessGroup,
        bias: torch.Tensor | None = None,
    ) -> None:
        self.head_x = head_x
        self.weight = weight
        self.bias = bias
        self.up = up
        self.group = group
        batch, global_sequence, shard_heads, head_dim = map(
            int, head_x.shape
        )
        self.batch = batch
        self.sequence = global_sequence // up
        self.shard_heads = shard_heads
        self.head_dim = head_dim
        self.heads = shard_heads * up
        exchange_shape = (
            up,
            batch,
            self.sequence,
            shard_heads,
            head_dim,
        )
        self.send = torch.empty(
            exchange_shape, dtype=head_x.dtype, device=head_x.device
        )
        self.received = torch.empty_like(self.send)
        self.materialized = torch.empty(
            (batch, self.sequence, self.heads, head_dim),
            dtype=head_x.dtype,
            device=head_x.device,
        )
        self.output = torch.empty_like(self.materialized)

    def __call__(self) -> torch.Tensor:
        self.send.copy_(
            self.head_x.view(
                self.batch,
                self.up,
                self.sequence,
                self.shard_heads,
                self.head_dim,
            ).permute(1, 0, 2, 3, 4)
        )
        dist.all_to_all_single(
            self.received,
            self.send,
            group=self.group,
        )
        self.materialized.view(
            self.batch,
            self.sequence,
            self.up,
            self.shard_heads,
            self.head_dim,
        ).copy_(
            self.received.permute(1, 2, 0, 3, 4)
        )
        hidden = self.heads * self.head_dim
        linear_into(
            self.materialized.view(-1, hidden),
            self.weight,
            self.bias,
            self.output.view(-1, hidden),
        )
        return self.output


class PeerWriteBaselineTHD:
    """``GEMM + permute + all_to_all + index_select regroup`` for packed THD."""

    def __init__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        layout: THDLayout,
        group: dist.ProcessGroup,
        bias: torch.Tensor | None = None,
    ) -> None:
        self.x = x
        self.weight = weight
        self.bias = bias
        self.layout = layout
        self.group = group
        tokens, heads, head_dim = map(int, x.shape)
        up = layout.up
        self.tokens = tokens
        self.hidden = heads * head_dim
        self.shard_hidden = self.hidden // up
        self.dense = torch.empty(
            (tokens, self.hidden), dtype=x.dtype, device=x.device
        )
        self.send = torch.empty(
            (up, tokens, self.shard_hidden), dtype=x.dtype, device=x.device
        )
        self.received = torch.empty_like(self.send)
        self.output = torch.empty(
            (up * tokens, heads // up, head_dim),
            dtype=x.dtype,
            device=x.device,
        )

    def __call__(self) -> torch.Tensor:
        linear_into(
            self.x.view(self.tokens, self.hidden),
            self.weight,
            self.bias,
            self.dense,
        )
        self.send.copy_(
            self.dense.view(self.tokens, self.layout.up, self.shard_hidden)
            .permute(1, 0, 2)
        )
        dist.all_to_all_single(
            self.received,
            self.send,
            group=self.group,
        )
        torch.index_select(
            self.received.view(-1, self.shard_hidden),
            0,
            self.layout.to_thd_index,
            out=self.output.view(-1, self.shard_hidden),
        )
        return self.output


class PullGemmBaselineTHD:
    """``index_select split + all_to_all + permute + GEMM`` for packed THD."""

    def __init__(
        self,
        head_x: torch.Tensor,
        weight: torch.Tensor,
        *,
        layout: THDLayout,
        group: dist.ProcessGroup,
        bias: torch.Tensor | None = None,
    ) -> None:
        self.head_x = head_x
        self.weight = weight
        self.bias = bias
        self.layout = layout
        self.group = group
        total, shard_heads, head_dim = map(int, head_x.shape)
        up = layout.up
        self.tokens = total // up
        self.heads = shard_heads * up
        self.head_dim = head_dim
        self.shard_hidden = shard_heads * head_dim
        self.hidden = self.heads * head_dim
        self.send = torch.empty(
            (up, self.tokens, self.shard_hidden),
            dtype=head_x.dtype,
            device=head_x.device,
        )
        self.received = torch.empty_like(self.send)
        self.materialized = torch.empty(
            (self.tokens, self.hidden), dtype=head_x.dtype, device=head_x.device
        )
        self.output = torch.empty_like(self.materialized)

    def __call__(self) -> torch.Tensor:
        torch.index_select(
            self.head_x.view(-1, self.shard_hidden),
            0,
            self.layout.to_source_major_index,
            out=self.send.view(-1, self.shard_hidden),
        )
        dist.all_to_all_single(
            self.received,
            self.send,
            group=self.group,
        )
        self.materialized.view(self.tokens, self.layout.up, self.shard_hidden).copy_(
            self.received.permute(1, 0, 2)
        )
        linear_into(self.materialized, self.weight, self.bias, self.output)
        return self.output.view(self.tokens, self.heads, self.head_dim)


def reference_peer_write_thd(
    baseline: PeerWriteBaselineTHD,
    cu_seqlens: tuple[int, ...],
) -> torch.Tensor:
    """Independent regroup of the baseline's received arena (slice loop)."""

    baseline()
    return reference_regroup_thd(baseline.received, cu_seqlens).view_as(
        baseline.output
    )


def reference_pull_gemm_thd(
    baseline: PullGemmBaselineTHD,
    cu_seqlens: tuple[int, ...],
) -> torch.Tensor:
    """Independent split (slice loop) + all_to_all + permute + GEMM."""

    send = reference_split_thd(
        baseline.head_x.view(-1, baseline.shard_hidden),
        cu_seqlens,
        up=baseline.layout.up,
    )
    received = torch.empty_like(send)
    dist.all_to_all_single(received, send, group=baseline.group)
    materialized = (
        received.permute(1, 0, 2)
        .reshape(baseline.tokens, baseline.hidden)
        .contiguous()
    )
    return torch.nn.functional.linear(materialized, baseline.weight, baseline.bias).view(
        baseline.tokens, baseline.heads, baseline.head_dim
    )


class MatmulReference:
    """Allocation-free dense GEMM (``torch.mm`` / ``torch.addmm`` with bias):
    the lower bound for one projection."""

    def __init__(
        self,
        x_2d: torch.Tensor,
        weight: torch.Tensor,
        output_2d: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        self.x_2d = x_2d
        self.weight = weight
        self.bias = bias
        self.output_2d = output_2d

    def __call__(self) -> torch.Tensor:
        return linear_into(self.x_2d, self.weight, self.bias, self.output_2d)


def correctness(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    group: dist.ProcessGroup,
) -> dict[str, float | bool]:
    difference = actual.float() - expected.float()
    max_abs = difference.abs().max()
    rel_l2 = difference.square().sum().sqrt() / (
        expected.float().square().sum().sqrt().clamp_min(1e-12)
    )
    passed = torch.tensor(
        int(torch.allclose(actual, expected, rtol=3e-2, atol=3e-2)),
        dtype=torch.int32,
        device=actual.device,
    )
    dist.all_reduce(passed, op=dist.ReduceOp.MIN, group=group)
    dist.all_reduce(max_abs, op=dist.ReduceOp.MAX, group=group)
    dist.all_reduce(rel_l2, op=dist.ReduceOp.MAX, group=group)
    return {
        "passed": bool(passed.item()),
        "max_abs": float(max_abs.item()),
        "max_rank_rel_l2": float(rel_l2.item()),
    }


def benchmark_functions(
    functions: dict[str, Callable[[], object]],
    *,
    warmup: int,
    iterations: int,
    group: dist.ProcessGroup,
) -> dict[str, dict[str, float]]:
    """Time several paths with cyclic launch order and rank-max samples."""

    names = tuple(functions)
    if not names:
        raise ValueError("at least one benchmark function is required")
    for iteration in range(warmup):
        offset = iteration % len(names)
        order = names[offset:] + names[:offset]
        for name in order:
            functions[name]()
    torch.cuda.synchronize()
    dist.barrier(group=group)

    samples: dict[str, list[float]] = {name: [] for name in names}
    for iteration in range(iterations):
        offset = iteration % len(names)
        order = names[offset:] + names[:offset]
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            functions[name]()
            end.record()
            end.synchronize()
            elapsed = torch.tensor(
                start.elapsed_time(end),
                dtype=torch.float64,
                device=torch.device(
                    "cuda", torch.cuda.current_device()
                ),
            )
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=group)
            samples[name].append(float(elapsed.item()))

    def summarize(values: list[float]) -> dict[str, float]:
        return {
            "median_ms": statistics.median(values),
            "p90_ms": percentile(values, 0.90),
            "min_ms": min(values),
            "max_ms": max(values),
        }

    return {name: summarize(samples[name]) for name in names}


def result_set(
    summaries: dict[str, dict[str, float]],
) -> dict[str, object]:
    fused = summaries["fused"]
    baseline = summaries["baseline"]
    kernel_only = summaries["kernel_only"]
    matmul = summaries["matmul"]
    return {
        "fused": fused,
        "baseline": baseline,
        "kernel_only": kernel_only,
        "matmul": matmul,
        "speedup": baseline["median_ms"] / fused["median_ms"],
        "fused_over_matmul": fused["median_ms"] / matmul["median_ms"],
        "kernel_only_over_matmul": (
            kernel_only["median_ms"] / matmul["median_ms"]
        ),
    }


def main() -> None:
    args = parse_args()
    if args.up not in SUPPORTED_UP_DEGREES:
        raise ValueError(
            f"--up must be one of {SUPPORTED_UP_DEGREES}, got {args.up}"
        )
    if min(
        args.batch,
        args.sequence,
        args.heads,
        args.head_dim,
        args.iters,
    ) <= 0:
        raise ValueError("BSHD dimensions and iters must be positive")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.heads % args.up:
        raise ValueError("heads must be divisible by UP")
    if int(os.environ["WORLD_SIZE"]) != args.up:
        raise ValueError("WORLD_SIZE must equal --up")
    thd = args.layout in ("thd", "thd-split")
    split = args.layout == "thd-split"
    lengths: tuple[int, ...] = ()
    cu_seqlens: tuple[int, ...] = ()
    if thd and args.batch != 1:
        raise ValueError(
            "THD layouts pack tokens along T; pass --batch 1 and set the "
            "rank-local token count with --sequence (or --thd-seqlens)"
        )
    if split:
        lengths = thd_sample_lengths(args)
        cu_seqlens = cu_seqlens_from_lengths(lengths)
        args.sequence = cu_seqlens[-1]
    rows = args.batch * args.sequence

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    hidden = args.heads * args.head_dim
    shard_heads = args.heads // args.up
    runtime = initialize_runtime(
        dist.group.WORLD,
        hidden=hidden,
        max_rows=rows,
        device=device,
    )
    layout = prepare_thd_layout(cu_seqlens, runtime) if split else None

    peer_enabled = args.op in ("peer-write", "both")
    pull_enabled = args.op in ("pull-gemm", "both")
    peer_weight = x = peer_output = peer_baseline = peer_matmul = None
    peer_bias = pull_bias = None
    peer_kernel_only = None
    pull_weight = head_x = pull_baseline = pull_matmul = prepared_pull = None

    if peer_enabled:
        torch.manual_seed(args.seed)
        peer_weight = torch.randn(
            (hidden, hidden),
            dtype=torch.bfloat16,
            device=device,
        )
        if args.bias:
            peer_bias = torch.randn(hidden, dtype=torch.bfloat16, device=device)
        torch.manual_seed(args.seed + 100 + rank)
        if thd:
            x = torch.randn(
                (rows, args.heads, args.head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            if split:
                assert layout is not None
                peer_baseline = PeerWriteBaselineTHD(
                    x,
                    peer_weight,
                    layout=layout,
                    group=dist.group.WORLD,
                    bias=peer_bias,
                )
            else:
                # Contiguous-slab THD is the BSHD B=1 pipeline on 3-D views.
                peer_baseline = PeerWriteBaseline(
                    x.view(1, rows, args.heads, args.head_dim),
                    peer_weight,
                    up=args.up,
                    group=dist.group.WORLD,
                    bias=peer_bias,
                )
            peer_output = torch.empty(
                (args.up * rows, shard_heads, args.head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
        else:
            x = torch.randn(
                (args.batch, args.sequence, args.heads, args.head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            peer_baseline = PeerWriteBaseline(
                x,
                peer_weight,
                up=args.up,
                group=dist.group.WORLD,
                bias=peer_bias,
            )
            peer_output = torch.empty(
                (args.batch, args.up * args.sequence, shard_heads, args.head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
        peer_matmul = MatmulReference(
            x.view(-1, hidden),
            peer_weight,
            peer_baseline.dense.view(-1, hidden),
            peer_bias,
        )

    if pull_enabled:
        torch.manual_seed(args.seed + 1)
        pull_weight = torch.randn(
            (hidden, hidden),
            dtype=torch.bfloat16,
            device=device,
        )
        if args.bias:
            pull_bias = torch.randn(hidden, dtype=torch.bfloat16, device=device)
        torch.manual_seed(args.seed + 200 + rank)
        if thd:
            head_x = torch.randn(
                (args.up * rows, shard_heads, args.head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            if split:
                assert layout is not None
                pull_baseline = PullGemmBaselineTHD(
                    head_x,
                    pull_weight,
                    layout=layout,
                    group=dist.group.WORLD,
                    bias=pull_bias,
                )
            else:
                pull_baseline = PullGemmBaseline(
                    head_x.view(1, args.up * rows, shard_heads, args.head_dim),
                    pull_weight,
                    up=args.up,
                    group=dist.group.WORLD,
                    bias=pull_bias,
                )
        else:
            head_x = torch.randn(
                (args.batch, args.up * args.sequence, shard_heads, args.head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            pull_baseline = PullGemmBaseline(
                head_x,
                pull_weight,
                up=args.up,
                group=dist.group.WORLD,
                bias=pull_bias,
            )
        pull_matmul = MatmulReference(
            pull_baseline.materialized.view(-1, hidden),
            pull_weight,
            pull_baseline.output.view(-1, hidden),
            pull_bias,
        )

    compile_started = time.perf_counter()
    if pull_enabled:
        assert pull_weight is not None
        workspace_epochs = 2 * (args.warmup + args.iters) + 4
        if thd:
            prepared_pull = prepare_headshard_pull_gemm_thd(
                pull_weight,
                runtime,
                heads=args.heads,
                head_dim=args.head_dim,
                tokens=rows,
                layout=layout,
                bias=pull_bias,
                workspace_epochs=workspace_epochs,
            )
        else:
            prepared_pull = prepare_headshard_pull_gemm_bshd(
                pull_weight,
                runtime,
                batch=args.batch,
                sequence=args.sequence,
                heads=args.heads,
                head_dim=args.head_dim,
                bias=pull_bias,
                workspace_epochs=workspace_epochs,
            )
    if peer_enabled:
        assert x is not None
        assert peer_weight is not None
        assert peer_output is not None
        from mega_up.kernels.peer_write_gemm import prepare_peer_write

        peer_kernel_only = prepare_peer_write(
            x.view(1, rows, hidden),
            peer_weight.unsqueeze(0),
            None,
            tuple(
                peer.unsqueeze(0)
                for peer in runtime.remote_write_views(rows)
            ),
            bias=peer_bias,
            local_rank=runtime.local_rank,
        )
        if thd:
            prepared_peer = prepare_peer_write_gemm_thd(
                peer_weight,
                runtime,
                heads=args.heads,
                head_dim=args.head_dim,
                tokens=rows,
                layout=layout,
                bias=peer_bias,
            )
        else:
            prepared_peer = prepare_peer_write_gemm_bshd(
                peer_weight,
                runtime,
                batch=args.batch,
                sequence=args.sequence,
                heads=args.heads,
                head_dim=args.head_dim,
                bias=peer_bias,
            )
        fused_peer = prepared_peer(x, out=peer_output, validate_group=True)
    else:
        fused_peer = None
    if pull_enabled:
        assert prepared_pull is not None
        assert head_x is not None
        fused_pull = prepared_pull(head_x, validate_group=True)
    else:
        fused_pull = None
    torch.cuda.synchronize()
    compile_elapsed = torch.tensor(
        (time.perf_counter() - compile_started) * 1000,
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(compile_elapsed, op=dist.ReduceOp.MAX)
    compile_and_first_calls_ms = float(compile_elapsed.item())

    # The THD gate compares against slice-loop references that do not use the
    # layout's index tensors, so a wrong permutation cannot pass by symmetry.
    checks: dict[str, object] = {}
    if fused_peer is not None:
        assert peer_baseline is not None
        if split:
            expected_peer = reference_peer_write_thd(peer_baseline, cu_seqlens)
        else:
            expected_peer = peer_baseline().view_as(fused_peer)
        checks["peer_write"] = correctness(
            fused_peer,
            expected_peer,
            group=dist.group.WORLD,
        )
    if fused_pull is not None:
        assert pull_baseline is not None
        if split:
            expected_pull = reference_pull_gemm_thd(pull_baseline, cu_seqlens)
        else:
            expected_pull = pull_baseline().view_as(fused_pull)
        checks["pull_gemm"] = correctness(
            fused_pull,
            expected_pull,
            group=dist.group.WORLD,
        )
    if not all(bool(value["passed"]) for value in checks.values()):
        raise AssertionError(f"correctness gate failed: {checks}")

    torch.cuda.reset_peak_memory_stats(device)
    timings: dict[str, object] = {}
    if fused_peer is not None:
        assert x is not None
        assert peer_weight is not None
        assert peer_output is not None
        assert peer_baseline is not None
        assert peer_kernel_only is not None
        assert peer_matmul is not None
        summaries = benchmark_functions(
            {
                "fused": lambda: prepared_peer(
                    x,
                    out=peer_output,
                    validate_group=False,
                ),
                "kernel_only": peer_kernel_only,
                "baseline": peer_baseline,
                "matmul": peer_matmul,
            },
            warmup=args.warmup,
            iterations=args.iters,
            group=dist.group.WORLD,
        )
        timings["peer_write"] = result_set(summaries)
        runtime.barrier()
    if fused_pull is not None:
        assert prepared_pull is not None
        assert head_x is not None
        assert pull_baseline is not None
        assert pull_matmul is not None
        if split:
            assert layout is not None
            runtime.stage_rows_indexed(
                rows=rows,
                source=head_x.view(-1, runtime.shard_hidden),
                index=layout.to_source_major_index,
            )
        elif thd:
            runtime.stage_head_parallel(
                head_x.view(1, args.up * rows, shard_heads, args.head_dim)
            )
        else:
            runtime.stage_head_parallel(head_x)
        summaries = benchmark_functions(
            {
                "fused": lambda: prepared_pull(
                    head_x,
                    validate_group=False,
                ),
                "kernel_only": prepared_pull.compiled,
                "baseline": pull_baseline,
                "matmul": pull_matmul,
            },
            warmup=args.warmup,
            iterations=args.iters,
            group=dist.group.WORLD,
        )
        timings["pull_gemm"] = result_set(summaries)
        runtime.finish_pull()

    peak = torch.tensor(
        [
            torch.cuda.max_memory_allocated(device),
            torch.cuda.max_memory_reserved(device),
        ],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    gpu_names: list[object | None] = [None] * args.up
    dist.all_gather_object(gpu_names, torch.cuda.get_device_name(device))

    if rank == 0:
        kernel_revisions: dict[str, str] = {}
        if peer_enabled:
            from mega_up.kernels.peer_write_gemm import (
                IMPLEMENTATION_REVISION as peer_revision,
            )

            kernel_revisions["peer_write"] = peer_revision
        if pull_enabled:
            from mega_up.kernels.headshard_pull_gemm import (
                IMPLEMENTATION_REVISION as pull_revision,
            )

            kernel_revisions["headshard_pull"] = pull_revision
        shape: dict[str, object] = {
            "layout": args.layout,
            "batch": args.batch,
            "sequence_per_rank": args.sequence,
            "tokens_per_rank": rows,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "hidden": hidden,
            "up": args.up,
        }
        if split:
            shape["thd"] = {
                "num_samples": len(lengths),
                "local_seqlens": list(lengths),
                "min_len": min(lengths),
                "max_len": max(lengths),
                "cu_seqlens_out_last": args.up * rows,
            }
        matmul_semantics = (
            "allocation-free torch.addmm with reusable output"
            if args.bias
            else "allocation-free torch.mm with reusable output"
        )
        if split:
            baseline_semantics = {
                "peer_write": (
                    "GEMM + permute + all_to_all + index_select regroup to "
                    "split-sample packed order"
                ),
                "pull_gemm": (
                    "index_select split from split-sample packed order + "
                    "all_to_all + permute + GEMM"
                ),
                "matmul": matmul_semantics,
            }
        elif thd:
            baseline_semantics = {
                "peer_write": (
                    "GEMM + permute + all_to_all + contiguous output copy "
                    "(source-rank order is the global packed order)"
                ),
                "pull_gemm": "contiguous split + all_to_all + permute + GEMM",
                "matmul": matmul_semantics,
            }
        else:
            baseline_semantics = {
                "peer_write": "GEMM + permute + all_to_all + output permute",
                "pull_gemm": "permute + all_to_all + permute + GEMM",
                "matmul": matmul_semantics,
            }
        result = {
            "benchmark": "mega_up_" + args.layout.replace("-", "_"),
            "configuration": {
                "operation": args.op,
                "bias": args.bias,
                "warmup": args.warmup,
                "iterations": args.iters,
                "seed": args.seed,
            },
            "shape": shape,
            "timing_semantics": (
                "CUDA-event elapsed time; each sample is the maximum across "
                "UP ranks; path order rotates every sample; fused is the "
                "prepared public API; kernel_only excludes layout staging, "
                "symmetric barriers, and output copies; matmul is an "
                "allocation-free dense GEMM with the same M,K,N (and bias)"
            ),
            "baseline": baseline_semantics,
            "correctness": checks,
            "compile_and_first_calls_ms": compile_and_first_calls_ms,
            "revisions": {
                "source": source_revision(),
                "kernels": kernel_revisions,
            },
            "timings": timings,
            "arena_bytes_per_rank": runtime.arena_nbytes,
            "peak_memory_bytes": {
                "allocated": int(peak[0].item()),
                "reserved": int(peak[1].item()),
            },
            "environment": {
                "mega_up": __version__,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cutlass_dsl": package_version("nvidia-cutlass-dsl"),
                "gpu_models": sorted({str(name) for name in gpu_names}),
            },
        }
        encoded = json.dumps(result, indent=2, sort_keys=True)
        print("[BENCH_RESULT] " + json.dumps(result, sort_keys=True), flush=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n", encoding="utf-8")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
