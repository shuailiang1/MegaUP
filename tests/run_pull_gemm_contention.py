#!/usr/bin/env python3
# Copyright (c) The Mega-UP Authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Single-GPU race regression test for the head-shard pull GEMM epilogue.

The pull kernel only needs raw peer addresses, so a UP group is emulated on
one GPU with ``up`` local ``[up, peer_rows, hidden/up]`` buffers. That runs
the exact kernel under test without NCCL or symmetric memory. ``hidden`` is
the pulled width K; ``--out-features`` sets N for a rectangular ``[N, K]``
weight.

Every call is checked three ways:

* the materialized input ``dy`` must be bit-identical to a gather reference;
* ``out`` must be bit-identical to the first call (the kernel is
  deterministic, so any difference is a race);
* the first ``out`` must be within BF16 rounding of an FP32 GEMM reference.

``--contention`` launches SMEM-free elementwise kernels (optionally mixed with
dense GEMMs) on a side stream so that their warps co-reside with the pull
GEMM CTAs and steal issue slots from the epilogue warps. That is the regime
in which an epilogue whose staging-SMEM reuse relies on the MMA being slower
than the epilogue corrupts the last 16 columns of a tile; the check reports
the corrupted column range so the failure signature is recognizable.

Examples::

    python tests/run_pull_gemm_contention.py --hidden 2048 --iters 200 --contention
    python tests/run_pull_gemm_contention.py --hidden 4096 --iters 100 --perf
    python tests/run_pull_gemm_contention.py --hidden 4096 --out-features 2048 --perf
    python tests/run_pull_gemm_contention.py --kernel-file /path/to/old.py \\
        --hidden 2048 --iters 200 --contention
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from types import ModuleType

import torch

SM_MARGIN = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument(
        "--out-features",
        type=int,
        default=0,
        help="output width N of the [N, hidden] weight (default: hidden)",
    )
    parser.add_argument("--up", type=int, default=8)
    parser.add_argument(
        "--rank", type=int, default=None, help="emulated rank (default: UP - 1)"
    )
    parser.add_argument("--rows", type=int, default=32270)
    parser.add_argument("--peer-rows", type=int, default=0)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--c-stages", type=int, default=1)
    parser.add_argument("--bias", action="store_true", help="add a fused bias")
    parser.add_argument("--contention", action="store_true")
    parser.add_argument(
        "--stress-kind",
        default="elementwise",
        choices=("elementwise", "matmul", "mixed"),
    )
    parser.add_argument("--stress-numel", type=int, default=1 << 26)
    parser.add_argument("--stress-chunk", type=int, default=40)
    parser.add_argument("--perf", action="store_true")
    parser.add_argument("--perf-iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=str, default="")
    parser.add_argument(
        "--kernel-file",
        type=str,
        default="",
        help=(
            "load headshard_pull_gemm from this file instead of the installed "
            "mega_up package (for A/B comparisons against another revision)"
        ),
    )
    return parser.parse_args()


def load_kernel_module(path: str) -> ModuleType:
    if not path:
        from mega_up.kernels import headshard_pull_gemm

        return headshard_pull_gemm
    spec = importlib.util.spec_from_file_location(
        "headshard_pull_gemm_under_test", path
    )
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stress:
    """Side-stream kernels that co-reside with the pull-GEMM CTAs."""

    def __init__(
        self,
        device: torch.device,
        kind: str,
        numel: int,
        chunk: int,
    ) -> None:
        self.stream = torch.cuda.Stream(device=device)
        self.kind = kind
        self.chunk = chunk
        with torch.cuda.stream(self.stream):
            self.x = torch.randn(numel, device=device, dtype=torch.bfloat16)
            self.y = torch.empty_like(self.x)
            self.a = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
            self.b = torch.randn_like(self.a)
        self.stream.synchronize()

    def burst(self) -> None:
        with torch.cuda.stream(self.stream):
            for index in range(self.chunk):
                if self.kind in ("elementwise", "mixed"):
                    self.y = torch.nn.functional.gelu(self.x)
                    self.x.add_(self.y, alpha=1e-3)
                if self.kind in ("matmul", "mixed") and index % 4 == 0:
                    self.a @ self.b

    def sync(self) -> None:
        self.stream.synchronize()


def analyze_mismatch(
    out: torch.Tensor,
    ref: torch.Tensor,
    tile_n: int,
) -> dict[str, object]:
    diff = out != ref
    rows = torch.nonzero(diff.any(dim=1)).flatten().tolist()
    cols = torch.nonzero(diff.any(dim=0)).flatten().tolist()
    cols_in_tile = sorted({col % tile_n for col in cols})
    rel = (
        (out.float() - ref.float()).abs() / ref.float().abs().clamp_min(1e-3)
    )[diff]
    return {
        "bad_elems": int(diff.sum()),
        "bad_rows": len(rows),
        "row_range": [rows[0], rows[-1]] if rows else None,
        "n_tiles": sorted({col // tile_n for col in cols}),
        "cols_in_tile": (
            [cols_in_tile[0], cols_in_tile[-1]] if cols_in_tile else None
        ),
        "rel_err_median": float(rel.median()) if rel.numel() else 0.0,
    }


def main() -> None:
    args = parse_args()
    if args.hidden % args.up or args.up <= 0 or args.rows <= 0:
        raise ValueError("hidden must be divisible by up; up and rows positive")
    if args.rank is None:
        args.rank = args.up - 1
    if not 0 <= args.rank < args.up:
        raise ValueError("rank must be in [0, up)")
    device = torch.device("cuda", torch.cuda.current_device())
    capability = torch.cuda.get_device_capability(device)
    if capability[0] not in (10, 11):
        raise RuntimeError(f"unsupported GPU (compute capability {capability})")
    torch.manual_seed(args.seed)
    module = load_kernel_module(args.kernel_file)
    print(f"kernel={module.__file__}")
    print(f"revision={module.IMPLEMENTATION_REVISION}")

    up, hidden, rows = args.up, args.hidden, args.rows
    out_features = args.out_features or hidden
    shard_k = hidden // up
    peer_rows = args.peer_rows or rows
    if peer_rows < rows:
        raise ValueError("peer-rows must be >= rows")

    peers = [
        (torch.randn(up, peer_rows, shard_k, device=device) * 0.5).to(
            torch.bfloat16
        )
        for _ in range(up)
    ]
    peer_table = torch.tensor(
        [peer.data_ptr() for peer in peers], dtype=torch.uint64, device=device
    )
    weight = (
        torch.randn(out_features, hidden, device=device) * (1.0 / hidden**0.5)
    ).to(torch.bfloat16)
    bias = (
        torch.randn(out_features, device=device).to(torch.bfloat16)
        if args.bias
        else None
    )
    x_ref = torch.cat(
        [peers[source][args.rank, :rows, :] for source in range(up)], dim=1
    ).contiguous()
    # The kernel computes x @ W^T (+ b) (torch.nn.functional.linear layout).
    out_ref32 = x_ref.float() @ weight.float().t()
    if bias is not None:
        out_ref32 += bias.float()

    out = torch.empty(rows, out_features, device=device, dtype=torch.bfloat16)
    dy = torch.empty(rows, hidden, device=device, dtype=torch.bfloat16)
    total_sms = torch.cuda.get_device_properties(device).multi_processor_count
    max_active_ctas = max(2, ((total_sms - SM_MARGIN) // 2) * 2)
    epochs = 512

    def make():
        # hidden= is the pulled width K; older --kernel-file revisions require
        # it (and a square weight), the current one only checks it.
        return module.compile_headshard_pull_gemm(
            peer_table,
            weight,
            out,
            dy,
            logical_m=rows,
            rank=args.rank,
            up=up,
            hidden=hidden,
            bias=bias,
            workspace_epochs=epochs,
            peer_rows=peer_rows,
            max_active_ctas=max_active_ctas,
            enable_clc=True,
            enable_bulk_pull=True,
            num_c_stages=args.c_stages,
        )

    started = time.perf_counter()
    compiled = make()
    compiled()
    torch.cuda.synchronize()
    print(
        f"compiled in {time.perf_counter() - started:.1f}s "
        f"c_stages={compiled.num_c_stages} ab_stages={compiled.num_ab_stages} "
        f"ctas={max_active_ctas} rows={rows} hidden={hidden} "
        f"out_features={out_features} up={up} "
        f"bulk={compiled.enable_bulk_pull} rows_per_ticket={compiled.pull_rows_per_ticket} "
        f"col_chunks={compiled.pull_col_chunks} bias={bias is not None}"
    )

    def run_once() -> None:
        nonlocal compiled
        if compiled.epoch >= epochs:
            compiled = make()
        compiled()

    result: dict[str, object] = {
        "revision": module.IMPLEMENTATION_REVISION,
        "hidden": hidden,
        "out_features": out_features,
        "rows": rows,
        "up": up,
        "c_stages": args.c_stages,
        "bias": args.bias,
        "contention": args.contention,
        "stress_kind": args.stress_kind if args.contention else None,
        "iters": args.iters,
    }

    def finish(passed: bool) -> None:
        result["status"] = "passed" if passed else "failed"
        print("[CONTENTION_RESULT] " + json.dumps(result, sort_keys=True))
        if args.json:
            with open(args.json, "w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=1, sort_keys=True)
        sys.exit(0 if passed else 1)

    dy.fill_(float("nan"))
    out.fill_(float("nan"))
    run_once()
    torch.cuda.synchronize()
    if not torch.equal(dy, x_ref):
        raise AssertionError("materialized input mismatch on the first call")
    out0 = out.clone()
    first_rel_all = (out0.float() - out_ref32).abs() / out_ref32.abs().clamp_min(1.0)
    first_rel = first_rel_all.max().item()
    print(f"first call: max rel err vs fp32 reference = {first_rel:.3e}")
    if first_rel >= 2e-2:
        # Corruption that is deterministic enough to hit the very first call
        # (e.g. an epilogue that is always slower than the MMA). Report the
        # signature against the BF16-rounded reference and fail.
        out_ref = out_ref32.to(torch.bfloat16)
        bad = first_rel_all >= 2e-2
        info = analyze_mismatch(torch.where(bad, out0, out_ref), out_ref, compiled.tile_n)
        print(f"  FIRST-CALL CORRUPTION: {json.dumps(info)}")
        result["first_call_corruption"] = info
        result["race_events"] = 1
        result["dy_mismatches"] = 0
        finish(False)

    stress = None
    if args.contention:
        stress = Stress(device, args.stress_kind, args.stress_numel, args.stress_chunk)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stress.stream)
        stress.burst()
        end.record(stress.stream)
        stress.sync()
        print(
            f"stress burst ({args.stress_kind} x{args.stress_chunk}) takes "
            f"{start.elapsed_time(end):.1f} ms"
        )

    events: list[dict[str, object]] = []
    dy_bad = 0
    for iteration in range(args.iters):
        dy.fill_(0)
        out.fill_(0)
        torch.cuda.synchronize()
        if stress is not None:
            stress.burst()
        run_once()
        if stress is not None:
            stress.burst()
        torch.cuda.synchronize()
        if not torch.equal(dy, x_ref):
            dy_bad += 1
        if not torch.equal(out, out0):
            info = analyze_mismatch(out, out0, compiled.tile_n)
            info["iter"] = iteration
            events.append(info)
            print(f"  RACE iter={iteration}: {json.dumps(info)}")
    if stress is not None:
        stress.sync()

    result["race_events"] = len(events)
    result["dy_mismatches"] = dy_bad
    result["events"] = events

    if args.perf:
        for _ in range(10):
            run_once()
        torch.cuda.synchronize()
        times: list[float] = []
        for _ in range(args.perf_iters):
            if compiled.epoch >= epochs:
                compiled = make()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            compiled()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000.0)
        median = statistics.median(times)
        print(
            f"[PERF] hidden={hidden} out_features={out_features} rows={rows}: "
            f"median={median:.1f}us min={min(times):.1f}us"
        )
        result["perf_us"] = {"median": median, "min": min(times)}
        if stress is not None:
            contended: list[float] = []
            for _ in range(args.perf_iters):
                if compiled.epoch >= epochs:
                    compiled = make()
                stress.burst()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                compiled()
                end.record()
                end.synchronize()
                stress.sync()
                contended.append(start.elapsed_time(end) * 1000.0)
            cmedian = statistics.median(contended)
            print(
                f"[PERF-CONTENDED] hidden={hidden} out_features={out_features} "
                f"rows={rows}: median={cmedian:.1f}us min={min(contended):.1f}us"
            )
            result["perf_contended_us"] = {"median": cmedian, "min": min(contended)}

    finish(not events and dy_bad == 0)


if __name__ == "__main__":
    main()
