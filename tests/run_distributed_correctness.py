#!/usr/bin/env python3
"""Direct SM100 correctness runner; invoke with ``torchrun``."""

from __future__ import annotations

import argparse
import importlib
import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from mega_up import (
    SUPPORTED_UP_DEGREES,
    initialize_runtime,
    peer_write_gemm_bshd,
    peer_write_gemm_sbhd,
    peer_write_gemm_thd,
    prepare_headshard_pull_gemm_bshd,
    prepare_headshard_pull_gemm_sbhd,
    prepare_headshard_pull_gemm_thd,
    prepare_peer_write_gemm_bshd,
    prepare_peer_write_gemm_sbhd,
    prepare_peer_write_gemm_thd,
    prepare_thd_layout,
)

DEFAULT_BSHD_CASES = ((1, 64), (2, 17))
# Sequence-first (S, B) cases.
DEFAULT_SBHD_CASES = ((64, 2), (17, 3))
# Rank-local packed chunk lengths; unequal, odd, and length-1 samples.
DEFAULT_THD_CASES = ((33, 31), (17, 1, 16))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--up", type=int, required=True)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--bias",
        action="store_true",
        help="give both projections a bias (reference: F.linear with bias)",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="BSHD batch,sequence pair such as 2,32; repeatable",
    )
    parser.add_argument(
        "--sbhd-case",
        action="append",
        default=[],
        help="SBHD sequence,batch pair such as 32,2; repeatable",
    )
    parser.add_argument(
        "--thd-case",
        action="append",
        default=[],
        help=(
            "packed THD case as comma-separated rank-local sample lengths "
            "such as 40,24,1; repeatable"
        ),
    )
    return parser.parse_args()


def parse_cases(values: list[str]) -> tuple[tuple[int, int], ...]:
    cases = []
    for value in values:
        try:
            batch, sequence = (int(part) for part in value.split(",", 1))
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid --case {value!r}; expected B,S") from error
        if batch <= 0 or sequence <= 0:
            raise ValueError(f"B and S must be positive, got {(batch, sequence)}")
        cases.append((batch, sequence))
    return tuple(cases)


def parse_thd_cases(values: list[str]) -> tuple[tuple[int, ...], ...]:
    cases = []
    for value in values:
        try:
            lengths = tuple(int(part) for part in value.split(","))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid --thd-case {value!r}; expected comma-separated lengths"
            ) from error
        if not lengths or min(lengths) < 0 or sum(lengths) <= 0:
            raise ValueError(
                f"THD lengths must be non-negative with a positive sum, got {lengths}"
            )
        cases.append(lengths)
    return tuple(cases)


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


def baseline_peer_write_thd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    cu_seqlens: tuple[int, ...] | None,
    *,
    up: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """``linear + all_to_all_single``; ``cu_seqlens=None`` keeps source order."""

    tokens, heads, head_dim = map(int, x.shape)
    shard_heads = heads // up
    dense = F.linear(x.view(tokens, heads * head_dim), weight, bias)
    send = (
        dense.view(tokens, up, shard_heads * head_dim)
        .permute(1, 0, 2)
        .contiguous()
    )
    received = torch.empty_like(send)
    dist.all_to_all_single(received, send, group=group)
    if cu_seqlens is None:
        return received.reshape(up * tokens, shard_heads, head_dim)
    return reference_regroup_thd(received, cu_seqlens).view(
        up * tokens, shard_heads, head_dim
    )


def baseline_pull_gemm_thd(
    head_x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    cu_seqlens: tuple[int, ...] | None,
    *,
    up: int,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor]:
    total, shard_heads, head_dim = map(int, head_x.shape)
    tokens = total // up
    heads = shard_heads * up
    if cu_seqlens is None:
        send = head_x.reshape(up, tokens, shard_heads * head_dim).contiguous()
    else:
        send = reference_split_thd(
            head_x.view(total, shard_heads * head_dim),
            cu_seqlens,
            up=up,
        )
    received = torch.empty_like(send)
    dist.all_to_all_single(received, send, group=group)
    materialized = (
        received.permute(1, 0, 2)
        .reshape(tokens, heads * head_dim)
        .contiguous()
    )
    output = F.linear(materialized, weight, bias).view(tokens, heads, head_dim)
    return materialized, output


def baseline_peer_write(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    up: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    batch, sequence, heads, head_dim = map(int, x.shape)
    shard_heads = heads // up
    dense = F.linear(
        x.view(batch, sequence, heads * head_dim),
        weight,
        bias,
    ).view(batch, sequence, heads, head_dim)
    send = (
        dense.view(batch, sequence, up, shard_heads, head_dim)
        .permute(2, 0, 1, 3, 4)
        .contiguous()
    )
    received = torch.empty_like(send)
    dist.all_to_all_single(received, send, group=group)
    return (
        received.permute(1, 0, 2, 3, 4)
        .reshape(batch, up * sequence, shard_heads, head_dim)
        .contiguous()
    )


def baseline_peer_write_sbhd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    up: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """``linear`` + all-to-all along the sequence dimension of ``[S,B,H,D]``."""

    sequence, batch, heads, head_dim = map(int, x.shape)
    shard_heads = heads // up
    dense = F.linear(x.view(sequence, batch, heads * head_dim), weight, bias)
    send = (
        dense.view(sequence, batch, up, shard_heads, head_dim)
        .permute(2, 0, 1, 3, 4)
        .contiguous()
    )
    received = torch.empty_like(send)
    dist.all_to_all_single(received, send, group=group)
    return received.reshape(up * sequence, batch, shard_heads, head_dim)


def baseline_pull_gemm_sbhd(
    head_x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    up: int,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-to-all of ``[UP*S,B,H/UP,D]`` back to ``[S,B,H*D]``, then ``linear``."""

    global_sequence, batch, shard_heads, head_dim = map(int, head_x.shape)
    sequence = global_sequence // up
    send = head_x.reshape(up, sequence, batch, shard_heads, head_dim).contiguous()
    received = torch.empty_like(send)
    dist.all_to_all_single(received, send, group=group)
    materialized = (
        received.permute(1, 2, 0, 3, 4)
        .reshape(sequence, batch, up * shard_heads * head_dim)
        .contiguous()
    )
    output = F.linear(materialized, weight, bias).view(
        sequence, batch, up * shard_heads, head_dim
    )
    return materialized, output


def baseline_pull_gemm(
    head_x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    up: int,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, global_sequence, shard_heads, head_dim = map(int, head_x.shape)
    sequence = global_sequence // up
    heads = shard_heads * up
    send = (
        head_x.view(batch, up, sequence, shard_heads, head_dim)
        .permute(1, 0, 2, 3, 4)
        .contiguous()
    )
    received = torch.empty_like(send)
    dist.all_to_all_single(received, send, group=group)
    materialized = (
        received.permute(1, 2, 0, 3, 4)
        .reshape(batch, sequence, heads, head_dim)
        .contiguous()
    )
    output = F.linear(
        materialized.view(batch, sequence, heads * head_dim),
        weight,
        bias,
    ).view(batch, sequence, heads, head_dim)
    return materialized, output


def error_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    difference = actual.float() - expected.float()
    max_abs = difference.abs().max()
    denominator = expected.float().square().sum().sqrt().clamp_min(1e-12)
    rel_l2 = difference.square().sum().sqrt() / denominator
    return max_abs, rel_l2


def global_check(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    name: str,
    rtol: float,
    atol: float,
) -> dict[str, float]:
    max_abs, rel_l2 = error_metrics(actual, expected)
    flags = torch.tensor(
        [
            int(torch.allclose(actual, expected, rtol=rtol, atol=atol)),
            int(torch.equal(actual, expected)),
        ],
        dtype=torch.int32,
        device=actual.device,
    )
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    dist.all_reduce(max_abs, op=dist.ReduceOp.MAX)
    dist.all_reduce(rel_l2, op=dist.ReduceOp.MAX)
    passed = flags[0]
    metrics = {
        "bitwise": bool(flags[1].item()),
        "max_abs": float(max_abs.item()),
        "max_rank_rel_l2": float(rel_l2.item()),
    }
    if not bool(passed.item()):
        raise AssertionError(
            f"{name} failed with {metrics}, rtol={rtol}, atol={atol}"
        )
    return metrics


def check_prepared_peer_write(
    prepared,
    x: torch.Tensor,
    expected: torch.Tensor,
    baseline,
    *,
    name: str,
) -> dict[str, float]:
    """Run a prepared peer-write on ``x``, then on a second input that the
    launch has to bind to new storage."""

    global_check(prepared(x), expected, name=f"{name}, first input", rtol=3e-2, atol=3e-2)
    second = torch.randn_like(x)
    return global_check(
        prepared(second, validate_group=False),
        baseline(second),
        name=f"{name}, second input",
        rtol=3e-2,
        atol=3e-2,
    )


def expect_collective_rejection(call, *, name: str, device: torch.device) -> None:
    rejected = torch.zeros((), dtype=torch.int32, device=device)
    try:
        call()
    except ValueError:
        rejected.fill_(1)
    dist.all_reduce(rejected, op=dist.ReduceOp.MIN)
    if not bool(rejected.item()):
        raise AssertionError(f"{name} was not rejected on every rank")


def main() -> None:
    args = parse_args()
    cases = parse_cases(args.case)
    sbhd_cases = parse_cases(args.sbhd_case)
    thd_cases = parse_thd_cases(args.thd_case)
    if not cases and not sbhd_cases and not thd_cases:
        cases = DEFAULT_BSHD_CASES
        sbhd_cases = DEFAULT_SBHD_CASES
        thd_cases = DEFAULT_THD_CASES
    if args.up not in SUPPORTED_UP_DEGREES:
        raise ValueError(
            f"--up must be one of {SUPPORTED_UP_DEGREES}, got {args.up}"
        )
    if args.heads % args.up:
        raise ValueError("heads must be divisible by UP")
    if int(os.environ["WORLD_SIZE"]) != args.up:
        raise ValueError("WORLD_SIZE must equal --up for this runner")

    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    capability = torch.cuda.get_device_capability(device)
    if capability[0] not in (10, 11):
        raise RuntimeError(f"SM100/SM110 required, got {capability}")
    converter_module = importlib.import_module(
        "cutlass.cute._tvm_ffi_args_spec_converter"
    )
    converter_before = converter_module._convert_single_arg

    hidden = args.heads * args.head_dim
    max_rows = max(
        [batch * sequence for batch, sequence in cases]
        + [sequence * batch for sequence, batch in sbhd_cases]
        + [sum(lengths) for lengths in thd_cases]
    )
    invalid_device = (
        torch.device("cuda", torch.cuda.device_count())
        if dist.get_rank() == 0
        else device
    )
    expect_collective_rejection(
        lambda: initialize_runtime(
            dist.group.WORLD,
            hidden=hidden,
            max_rows=max_rows,
            device=invalid_device,
        ),
        name="rank-local invalid CUDA ordinal",
        device=device,
    )
    expect_collective_rejection(
        lambda: initialize_runtime(
            dist.group.WORLD,
            hidden=hidden + (args.head_dim if dist.get_rank() == 0 else 0),
            max_rows=max_rows,
            device=device,
        ),
        name="rank-local runtime configuration mismatch",
        device=device,
    )
    runtime = initialize_runtime(
        dist.group.WORLD,
        hidden=hidden,
        max_rows=max_rows,
        device=device,
    )

    torch.manual_seed(1234)
    peer_weight = torch.randn(
        (hidden, hidden),
        dtype=torch.bfloat16,
        device=device,
    )
    torch.manual_seed(5678)
    pull_weight = torch.randn_like(peer_weight)
    peer_bias = pull_bias = None
    if args.bias:
        torch.manual_seed(9012)
        peer_bias = torch.randn(hidden, dtype=torch.bfloat16, device=device)
        pull_bias = torch.randn_like(peer_bias)
    results: list[dict[str, object]] = []

    for case_index, (batch, sequence) in enumerate(cases):
        torch.manual_seed(10_000 + 100 * case_index + dist.get_rank())
        x = torch.randn(
            (batch, sequence, args.heads, args.head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        actual_peer = peer_write_gemm_bshd(x, peer_weight, runtime, bias=peer_bias)
        converter_restored = torch.tensor(
            int(converter_module._convert_single_arg is converter_before),
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(converter_restored, op=dist.ReduceOp.MIN)
        if not bool(converter_restored.item()):
            raise AssertionError(
                "peer-write import/JIT leaked its CUTLASS converter patch"
            )
        expected_peer = baseline_peer_write(
            x,
            peer_weight,
            peer_bias,
            up=args.up,
            group=dist.group.WORLD,
        )
        peer_metrics = global_check(
            actual_peer,
            expected_peer,
            name=f"peer-write B={batch},S={sequence}",
            rtol=3e-2,
            atol=3e-2,
        )
        prepared_peer_metrics = check_prepared_peer_write(
            prepare_peer_write_gemm_bshd(
                peer_weight,
                runtime,
                batch=batch,
                sequence=sequence,
                heads=args.heads,
                head_dim=args.head_dim,
                bias=peer_bias,
            ),
            x,
            expected_peer,
            lambda second: baseline_peer_write(
                second, peer_weight, peer_bias, up=args.up, group=dist.group.WORLD
            ),
            name=f"prepared peer-write B={batch},S={sequence}",
        )

        torch.manual_seed(20_000 + 100 * case_index + dist.get_rank())
        head_x = torch.randn(
            (
                batch,
                args.up * sequence,
                args.heads // args.up,
                args.head_dim,
            ),
            dtype=torch.bfloat16,
            device=device,
        )
        prepared_pull = prepare_headshard_pull_gemm_bshd(
            pull_weight,
            runtime,
            batch=batch,
            sequence=sequence,
            heads=args.heads,
            head_dim=args.head_dim,
            bias=pull_bias,
            workspace_epochs=2,
        )
        # First use is on a different stream; subsequent calls cross streams
        # and the third launch forces one safe recycle when epochs=2.
        alternate_stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(alternate_stream):
            actual_pull = prepared_pull(head_x, validate_group=True)
        actual_pull = prepared_pull(head_x, validate_group=True)
        with torch.cuda.stream(alternate_stream):
            actual_pull = prepared_pull(head_x, validate_group=True)
        torch.cuda.current_stream(device).wait_stream(alternate_stream)
        expected_materialized, expected_pull = baseline_pull_gemm(
            head_x,
            pull_weight,
            pull_bias,
            up=args.up,
            group=dist.group.WORLD,
        )
        materialized_metrics = global_check(
            prepared_pull.materialized_input.view_as(expected_materialized),
            expected_materialized,
            name=f"pull materialization B={batch},S={sequence}",
            rtol=0.0,
            atol=0.0,
        )
        pull_metrics = global_check(
            actual_pull,
            expected_pull,
            name=f"pull-GEMM B={batch},S={sequence}",
            rtol=3e-2,
            atol=3e-2,
        )
        results.append(
            {
                "batch": batch,
                "sequence": sequence,
                "peer_write": peer_metrics,
                "peer_write_prepared": prepared_peer_metrics,
                "pull_materialization": materialized_metrics,
                "pull_gemm": pull_metrics,
            }
        )

    sbhd_results: list[dict[str, object]] = []
    for case_index, (sequence, batch) in enumerate(sbhd_cases):
        torch.manual_seed(50_000 + 100 * case_index + dist.get_rank())
        x = torch.randn(
            (sequence, batch, args.heads, args.head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        actual_peer = peer_write_gemm_sbhd(x, peer_weight, runtime, bias=peer_bias)
        expected_peer = baseline_peer_write_sbhd(
            x,
            peer_weight,
            peer_bias,
            up=args.up,
            group=dist.group.WORLD,
        )
        peer_metrics = global_check(
            actual_peer,
            expected_peer,
            name=f"peer-write SBHD S={sequence},B={batch}",
            rtol=3e-2,
            atol=3e-2,
        )
        prepared_peer_metrics = check_prepared_peer_write(
            prepare_peer_write_gemm_sbhd(
                peer_weight,
                runtime,
                sequence=sequence,
                batch=batch,
                heads=args.heads,
                head_dim=args.head_dim,
                bias=peer_bias,
            ),
            x,
            expected_peer,
            lambda second: baseline_peer_write_sbhd(
                second, peer_weight, peer_bias, up=args.up, group=dist.group.WORLD
            ),
            name=f"prepared peer-write SBHD S={sequence},B={batch}",
        )

        torch.manual_seed(60_000 + 100 * case_index + dist.get_rank())
        head_x = torch.randn(
            (args.up * sequence, batch, args.heads // args.up, args.head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        prepared_pull = prepare_headshard_pull_gemm_sbhd(
            pull_weight,
            runtime,
            sequence=sequence,
            batch=batch,
            heads=args.heads,
            head_dim=args.head_dim,
            bias=pull_bias,
            workspace_epochs=2,
        )
        actual_pull = prepared_pull(head_x, validate_group=True)
        expected_materialized, expected_pull = baseline_pull_gemm_sbhd(
            head_x,
            pull_weight,
            pull_bias,
            up=args.up,
            group=dist.group.WORLD,
        )
        materialized_metrics = global_check(
            prepared_pull.materialized_input.view_as(expected_materialized),
            expected_materialized,
            name=f"pull materialization SBHD S={sequence},B={batch}",
            rtol=0.0,
            atol=0.0,
        )
        pull_metrics = global_check(
            actual_pull,
            expected_pull,
            name=f"pull-GEMM SBHD S={sequence},B={batch}",
            rtol=3e-2,
            atol=3e-2,
        )
        sbhd_results.append(
            {
                "sequence": sequence,
                "batch": batch,
                "peer_write": peer_metrics,
                "peer_write_prepared": prepared_peer_metrics,
                "pull_materialization": materialized_metrics,
                "pull_gemm": pull_metrics,
            }
        )

    thd_results: list[dict[str, object]] = []
    for case_index, lengths in enumerate(thd_cases):
        cu_seqlens = cu_seqlens_from_lengths(lengths)
        tokens = cu_seqlens[-1]
        # Pass cu_seqlens as a CUDA int32 tensor on even cases and as host
        # ints on odd cases; both must produce the same layout.
        layout = prepare_thd_layout(
            torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
            if case_index % 2 == 0
            else cu_seqlens,
            runtime,
        )
        if layout.cu_seqlens_out != tuple(args.up * value for value in cu_seqlens):
            raise AssertionError("THD output boundaries must be UP * cu_seqlens")

        # Variant "slab": default contract, rows stay in source-rank order.
        # Variant "split": optional per-sample regroup through the layout.
        for variant, active_layout, reference_cu in (
            ("slab", None, None),
            ("split", layout, cu_seqlens),
        ):
            torch.manual_seed(30_000 + 100 * case_index + dist.get_rank())
            x = torch.randn(
                (tokens, args.heads, args.head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            actual_peer = peer_write_gemm_thd(
                x, peer_weight, runtime, bias=peer_bias, layout=active_layout
            )
            expected_peer = baseline_peer_write_thd(
                x,
                peer_weight,
                peer_bias,
                reference_cu,
                up=args.up,
                group=dist.group.WORLD,
            )
            peer_metrics = global_check(
                actual_peer,
                expected_peer,
                name=f"peer-write THD/{variant} lengths={lengths}",
                rtol=3e-2,
                atol=3e-2,
            )
            prepared_peer_metrics = check_prepared_peer_write(
                prepare_peer_write_gemm_thd(
                    peer_weight,
                    runtime,
                    heads=args.heads,
                    head_dim=args.head_dim,
                    tokens=tokens if active_layout is None else None,
                    layout=active_layout,
                    bias=peer_bias,
                ),
                x,
                expected_peer,
                lambda second: baseline_peer_write_thd(
                    second,
                    peer_weight,
                    peer_bias,
                    reference_cu,
                    up=args.up,
                    group=dist.group.WORLD,
                ),
                name=f"prepared peer-write THD/{variant} lengths={lengths}",
            )

            torch.manual_seed(40_000 + 100 * case_index + dist.get_rank())
            head_x = torch.randn(
                (args.up * tokens, args.heads // args.up, args.head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            prepared_pull = prepare_headshard_pull_gemm_thd(
                pull_weight,
                runtime,
                heads=args.heads,
                head_dim=args.head_dim,
                tokens=tokens if active_layout is None else None,
                layout=active_layout,
                bias=pull_bias,
                workspace_epochs=2,
            )
            alternate_stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(alternate_stream):
                actual_pull = prepared_pull(head_x, validate_group=True)
            actual_pull = prepared_pull(head_x, validate_group=True)
            with torch.cuda.stream(alternate_stream):
                actual_pull = prepared_pull(head_x, validate_group=True)
            torch.cuda.current_stream(device).wait_stream(alternate_stream)
            expected_materialized, expected_pull = baseline_pull_gemm_thd(
                head_x,
                pull_weight,
                pull_bias,
                reference_cu,
                up=args.up,
                group=dist.group.WORLD,
            )
            materialized_metrics = global_check(
                prepared_pull.materialized_input.view_as(expected_materialized),
                expected_materialized,
                name=f"pull materialization THD/{variant} lengths={lengths}",
                rtol=0.0,
                atol=0.0,
            )
            pull_metrics = global_check(
                actual_pull,
                expected_pull,
                name=f"pull-GEMM THD/{variant} lengths={lengths}",
                rtol=3e-2,
                atol=3e-2,
            )
            thd_results.append(
                {
                    "variant": variant,
                    "lengths": list(lengths),
                    "tokens": tokens,
                    "peer_write": peer_metrics,
                    "peer_write_prepared": prepared_peer_metrics,
                    "pull_materialization": materialized_metrics,
                    "pull_gemm": pull_metrics,
                }
            )

    shared_lengths = thd_cases[0] if thd_cases else ()
    shared_tokens = sum(shared_lengths)
    if shared_tokens >= 2:
        # A rank-local cu_seqlens mismatch with the same token count must be
        # rejected on every rank through the layout fingerprint instead of
        # silently mis-routing rows.
        if len(shared_lengths) >= 2:
            rank0_lengths: tuple[int, ...] = (shared_tokens,)
        else:
            rank0_lengths = (shared_tokens - 1, 1)
        local_lengths = rank0_lengths if dist.get_rank() == 0 else shared_lengths
        mismatched_layout = prepare_thd_layout(
            cu_seqlens_from_lengths(local_lengths),
            runtime,
        )
        mismatched_x = torch.randn(
            (mismatched_layout.tokens, args.heads, args.head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        expect_collective_rejection(
            lambda: peer_write_gemm_thd(
                mismatched_x,
                peer_weight,
                runtime,
                layout=mismatched_layout,
            ),
            name="rank-local THD cu_seqlens mismatch",
            device=device,
        )

    valid_x = torch.randn(
        (1, 1, args.heads, args.head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    rank_local_x = (
        valid_x[:, :, :-1, :] if dist.get_rank() == 0 else valid_x
    )
    expect_collective_rejection(
        lambda: peer_write_gemm_bshd(
            rank_local_x,
            peer_weight,
            runtime,
        ),
        name="rank-local peer-write input error",
        device=device,
    )
    prepared_single_row = prepare_peer_write_gemm_bshd(
        peer_weight,
        runtime,
        batch=1,
        sequence=1,
        heads=args.heads,
        head_dim=args.head_dim,
    )
    two_rows_x = torch.randn(
        (1, 2, args.heads, args.head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    expect_collective_rejection(
        lambda: prepared_single_row(
            two_rows_x if dist.get_rank() == 0 else valid_x
        ),
        name="rank-local prepared peer-write shape error",
        device=device,
    )
    if peer_bias is not None:
        rank_local_bias = peer_bias[:-1] if dist.get_rank() == 0 else peer_bias
        expect_collective_rejection(
            lambda: peer_write_gemm_bshd(
                valid_x,
                peer_weight,
                runtime,
                bias=rank_local_bias,
            ),
            name="rank-local bias error",
            device=device,
        )

    if dist.get_rank() == 0:
        print(
            "[CORRECTNESS_RESULT] "
            + json.dumps(
                {
                    "status": "passed",
                    "up": args.up,
                    "heads": args.heads,
                    "head_dim": args.head_dim,
                    "bias": args.bias,
                    "cases": results,
                    "sbhd_cases": sbhd_results,
                    "thd_cases": thd_results,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
