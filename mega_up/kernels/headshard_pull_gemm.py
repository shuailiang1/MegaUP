#!/usr/bin/env python3
# Copyright (c) The Mega-UP Authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Warp-specialized single-kernel head-shard pull + BF16 GEMM.

The kernel computes ``out = dy @ weight.T (+ bias)`` where ``dy``, the dense
``[M, K]`` input, is assembled inside the same launch from the head
shards that every peer staged in its symmetric buffer:

* ``logical_m`` is a runtime value (``dy``/``out`` are shape-dynamic in M and
  the workspaces are dynamic-length), so one compiled kernel serves every
  sequence length; a module-level cache keyed by the static configuration
  avoids recompilation.
* ``peer_rows`` (defaults to ``logical_m``) is the row stride between the
  per-destination slots of the symmetric source buffers.
* The pull copies use the widest per-lane vector (256, 128 or 64 bit) that
  tiles the shard, bound their register footprint with row groups and column
  chunks, and publish one aggregated fence + release per worker and M tile;
  the TMA producer gates each M tile with a single acquire spin.

The kernel keeps communication and GEMM in one GPU launch and assigns fixed
roles to eight warps in every CTA:

    warp 0..3   pipelined TMEM epilogue -> BF16 ``out`` HBM
    warp 4      tcgen05/UMMA issue, double-buffered accumulator in TMEM
    warp 5      one combined TMA producer for A=``dy`` and B=``weight``
    warp 6..7   peer pull -> local ``dy`` HBM -> release ready counter

CTAs are persistent: each CTA processes multiple ``(tile_m, tile_n)`` output
tiles. Pull warps obtain work from a GPU-wide atomic ticket queue. Each
invocation uses a separate queue and ready-counter row selected by ``epoch``;
therefore no reset kernel is needed and every call still launches exactly one
GPU kernel.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Type, Union

import torch
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cuda.bindings.driver import CUstream
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm, nvvm

IMPLEMENTATION_REVISION = "headshard-pull-v7-deep-narrow-tickets"

WARP_SIZE = 32
EPILOGUE_WARPS = (0, 1, 2, 3)
MMA_WARP = 4
TMA_WARP = 5
PULL_WARP = 6
AUX_WARP = 7
THREADS_PER_CTA = 8 * WARP_SIZE

DEFAULT_TILE_M = 128
DEFAULT_TILE_N = 256
DEFAULT_TILE_K = 64
DEFAULT_PULL_CHUNK_M = 16
DEFAULT_AB_STAGES = 0  # 0 = auto from SMEM capacity
DEFAULT_ACC_STAGES = 2
DEFAULT_C_STAGES = 1
DEFAULT_SMEM_RESERVE_BYTES = 1024
# Let auto mode use a deeper A/B pipeline when SMEM capacity allows it.
DEFAULT_AUTO_AB_STAGES_CAP = 4
DEFAULT_WORKSPACE_EPOCHS = 4096
# CLC (Cluster Launch Control) dynamic-persistent scheduler ring depth. The TMA
# warp produces the fetched tile sequence into this CTA-local ring; the MMA and
# epilogue warps consume it. Depth must cover the TMA-lead-over-epilogue skew
# (~num_acc_stage + 1) so producing the next tile never stalls on the epilogue.
DEFAULT_CLC_SCHED_STAGES = 4

# Widest pull: 256-bit, 16 BF16 elements per lane and copy atom. The pull uses
# the widest per-lane vector for which one warp-wide copy tiles ``shard_k``:
# 256-bit lanes need shard_k % 512 == 0, 128-bit lanes % 256 (e.g. hidden=2048
# at UP=8 -> 256) and 64-bit lanes % 128 (e.g. hidden=5120 at UP=8 -> 640).
# The copy width only affects bandwidth, never the copied values.
PULL_ELEMENTS_PER_LANE = 16
PULL_MIN_ELEMENTS_PER_LANE = 4
PULL_DIVBY = 16
# Widest head shard that still uses 8-row pull tickets: the 8-row path
# prefetches two 4-row groups into registers (shard_k / 8 registers per lane),
# which fits at 1280 and spills at 2560.
PULL_PREFETCH_MAX_SHARD_K = 1280
# Widest head shard that uses 16-row pull tickets (see
# _vector_pull_rows_per_ticket).
PULL_DEEP_MAX_SHARD_K = 256
# Most BF16 elements one pull warp holds in registers between its loads and
# stores (4 rows x 3072 columns, 192 registers per lane). Wider row groups
# spill (at 4 x 4096 the kernel needs a stack frame), so they are copied in
# column chunks.
PULL_MAX_ELEMENTS_IN_FLIGHT = 4 * 3072

# Each stage gathers one complete K-element row from all peer shards
# into contiguous SMEM, then emits one SMEM-to-output bulk store of the row.
# Three row stages expose 3 * UP peer requests per CTA. The bulk ring is only
# used for the row lengths it was tuned on; other sizes pull with per-lane
# vectors, which measured faster for them.
DEFAULT_ENABLE_BULK_PULL = True
BULK_PULL_WORKERS = 1
BULK_PULL_STAGES = 3
BULK_PULL_TOTAL_STAGES = BULK_PULL_WORKERS * BULK_PULL_STAGES
BULK_PULL_TUNED_ROW_ELEMENTS = (4096,)


def bulk_pull_storage_reserve_bytes(row_elements: int) -> int:
    return 2 * row_elements * BULK_PULL_TOTAL_STAGES + 256


# Eight dynamic ticket cursors reduce atomic contention. Under CLC, a resident
# worker drains its primary shard first, then probes the other shards so work
# owned by a canceled (never-started) cluster cannot be stranded.
PULL_CURSOR_SHARDS = 8
# Active workers publish their cursor ownership before pulling. Healthy epochs
# skip rescue entirely; incomplete bitmaps use one extra word for CAS election.
PULL_CURSOR_BITMAP_WORDS = (PULL_CURSOR_SHARDS + 31) // 32
PULL_CURSOR_OWNER_BITMAP_OFFSET = PULL_CURSOR_SHARDS
PULL_CURSOR_RESCUE_OWNER_OFFSET = (
    PULL_CURSOR_OWNER_BITMAP_OFFSET + PULL_CURSOR_BITMAP_WORDS
)
PULL_CURSOR_STRIDE = PULL_CURSOR_RESCUE_OWNER_OFFSET + 1
# Limit active asynchronous pull workers so peer requests remain in flight
# without making every resident CTA add atomic pressure. Grouping by all cursor
# shards keeps the queues symmetric.
PULL_WORKER_GROUP_PERIOD = 16
PULL_ACTIVE_WORKER_GROUPS = 12


def _select_pull_elements_per_lane(shard_k: int) -> int:
    """Widest per-lane vector such that one warp-wide copy tiles ``shard_k``."""
    elements = PULL_ELEMENTS_PER_LANE
    while elements > PULL_MIN_ELEMENTS_PER_LANE and shard_k % (WARP_SIZE * elements):
        elements //= 2
    return elements


# One compiled kernel per static configuration; dynamic M never recompiles.
_KERNEL_CACHE: dict = {}
_KERNEL_COMPILE_COUNT = 0


def get_kernel_compile_count() -> int:
    """Number of actual ``cute.compile`` invocations since import."""
    return _KERNEL_COMPILE_COUNT


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _effective_pull_rows_per_ticket(pull_chunk_m: int, up: int) -> int:
    pull_rows_per_ticket = pull_chunk_m
    if pull_chunk_m >= 8:
        pull_rows_per_ticket = pull_chunk_m // 2
    return pull_rows_per_ticket


def _vector_pull_rows_per_ticket(pull_rows_per_ticket: int, shard_k: int) -> int:
    """Ticket size of the per-lane vector pull (not the bulk SMEM ring).

    A ticket's rows of one peer shard are loaded into registers before any is
    stored, so the ticket size sets the remote loads in flight per lane.
    Beyond ``PULL_PREFETCH_MAX_SHARD_K`` 8 rows spill (at shard_k = 2560 ptxas
    emits a 2.2 KB stack frame and the pull GEMM runs at 1.7x the dense GEMM
    instead of 1.2x), so wide shards pull 4 rows per ticket. Up to
    ``PULL_DEEP_MAX_SHARD_K`` 8 rows keep too few bytes in flight to cover
    the NVLink load latency (16 rows raised the hidden 2048 / U = 8 pull
    throughput by about 40%), so the narrowest shards pull 16 rows.
    """
    if pull_rows_per_ticket >= 8 and shard_k > PULL_PREFETCH_MAX_SHARD_K:
        return 4
    if pull_rows_per_ticket == 8 and shard_k <= PULL_DEEP_MAX_SHARD_K:
        return 16
    return pull_rows_per_ticket


def _pull_column_chunks(shard_k: int, rows_in_flight: int) -> int:
    """Smallest number of equal column chunks that bounds the pull registers.

    A chunk must keep ``rows_in_flight * chunk`` within
    ``PULL_MAX_ELEMENTS_IN_FLIGHT`` and stay a multiple of one 64-bit
    warp-wide copy (128 columns).
    """
    min_chunk = WARP_SIZE * PULL_MIN_ELEMENTS_PER_LANE
    if shard_k <= 0 or shard_k % min_chunk:
        raise ValueError(f"shard_k={shard_k} must be a positive multiple of {min_chunk}")
    chunks = _ceil_div(rows_in_flight * shard_k, PULL_MAX_ELEMENTS_IN_FLIGHT)
    while shard_k % chunks or (shard_k // chunks) % min_chunk:
        chunks += 1
    return chunks


# ---------------------------------------------------------------------------
# CLC (Cluster Launch Control) low-level device helpers.
#
# These are inlined here (rather than imported) so the kernel file stays
# self-contained. ``try_cancel`` atomically cancels a not-yet-started cluster
# of the same grid and asynchronously writes its ctaid to shared memory; this
# is what lets an already-running persistent cluster elastically absorb the
# work of pending clusters (and thereby coexist with a concurrent
# communication kernel instead of statically pinning SMs).
# ---------------------------------------------------------------------------
@dsl_user_op
def _clc_set_block_rank(
    smem_ptr: cute.Pointer, peer_cta_rank_in_cluster, *, loc=None, ip=None
) -> cutlass.Int32:
    """Map ``smem_ptr`` to the same offset in another CTA of the cluster."""
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, cutlass.Int32(peer_cta_rank_in_cluster).ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
        )
    )


@dsl_user_op
def _clc_store_shared_remote_i32(
    val,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster,
    *,
    loc=None,
    ip=None,
) -> None:
    """Async-store one i32 into a peer CTA's smem and complete its mbarrier tx."""
    remote_smem = _clc_set_block_rank(
        smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    remote_mbar = _clc_set_block_rank(
        mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    llvm.inline_asm(
        None,
        [remote_smem, cutlass.Int32(val).ir_value(loc=loc, ip=ip), remote_mbar],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.s32 [$0], $1, [$2];",
        "r,r,r",
        has_side_effects=True,
        is_align_stack=False,
    )


@dsl_user_op
def _clc_try_cancel(
    mbar_ptr: cute.Pointer, clc_response_ptr: cute.Pointer, *, loc=None, ip=None
) -> None:
    """Issue ``clusterlaunchcontrol.try_cancel`` into ``clc_response_ptr``."""
    nvvm.clusterlaunchcontrol_try_cancel(
        clc_response_ptr.llvm_ptr, mbar_ptr.llvm_ptr, loc=loc, ip=ip
    )


@dsl_user_op
def _cp_async_bulk_g2s(
    smem_ptr: cute.Pointer,
    gmem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    num_bytes,
    *,
    loc=None,
    ip=None,
) -> None:
    """Issue one 1-D peer/global -> shared bulk copy."""
    smem_addr = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    mbar_addr = mbar_ptr.toint(loc=loc, ip=ip).ir_value()
    cache_hint = cutlass.Int64(0x12F0000000000000)  # EVICT_FIRST
    llvm.inline_asm(
        None,
        [
            smem_addr,
            gmem_ptr.llvm_ptr,
            cutlass.Int32(num_bytes).ir_value(loc=loc, ip=ip),
            mbar_addr,
            cache_hint.ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        ".L2::cache_hint [$0], [$1], $2, [$3], $4;",
        "r,l,r,r,l",
        has_side_effects=True,
        is_align_stack=False,
    )


@dsl_user_op
def _cp_async_bulk_s2g(
    gmem_ptr: cute.Pointer,
    smem_ptr: cute.Pointer,
    num_bytes,
    *,
    loc=None,
    ip=None,
) -> None:
    """Issue one 1-D shared -> local/global bulk-group store."""
    smem_addr = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [
            gmem_ptr.llvm_ptr,
            smem_addr,
            cutlass.Int32(num_bytes).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
    )


class HeadShardPullGemmKernel:
    """Warp-specialized BF16 GEMM with head-shard materialization.

    Steady-state warp roles (8 warps / CTA):

      * warpgroup 0, warps 0..3: pipelined TMEM epilogue
      * warpgroup 1, warp 4:     tcgen05 / UMMA issue
      * warpgroup 1, warp 5:     one combined A+B TMA producer
      * warpgroup 1, warp 6..7:  head-shard pull and dy materialization

    Each CTA is persistent and processes multiple output tiles. ``logical_m``
    and ``peer_rows`` are runtime values; only tile/stage configuration is
    compile-time static.
    """

    def __init__(
        self,
        *,
        tile_m: int,
        tile_n: int,
        tile_k: int = DEFAULT_TILE_K,
        num_ab_stages: int = DEFAULT_AB_STAGES,
        num_acc_stages: int = DEFAULT_ACC_STAGES,
        num_c_stages: int = DEFAULT_C_STAGES,
        max_active_ctas: int,
        use_tma_store: bool,
        enable_setmaxnreg: bool,
        tma_python_k_loop: bool,
        pull_elements_per_lane: int = PULL_ELEMENTS_PER_LANE,
        pull_col_chunks: int = 1,
        pull_rows_in_flight: int = 8,
        enable_bulk_pull: bool = False,
        bulk_row_elements: int = 0,
        enable_clc: bool = False,
        clc_sched_stages: int = DEFAULT_CLC_SCHED_STAGES,
    ) -> None:
        self.tile_m = tile_m
        # Column chunks a group of pulled rows is split into (vector pull).
        self.pull_col_chunks = int(pull_col_chunks)
        self.pull_rows_in_flight = int(pull_rows_in_flight)
        # Row length (= K) gathered by one bulk-pull stage.
        self.bulk_row_elements = int(bulk_row_elements)
        self.tile_n = tile_n
        self.tile_k = tile_k
        self.requested_num_ab_stage = num_ab_stages
        self.num_acc_stage = num_acc_stages
        self.requested_num_c_stage = num_c_stages
        self.max_active_ctas = max_active_ctas
        self.tma_python_k_loop = tma_python_k_loop
        self.pull_elements_per_lane = pull_elements_per_lane
        self.pull_num_bits = pull_elements_per_lane * 16  # BF16 bits per lane
        self.enable_bulk_pull = bool(enable_bulk_pull)
        # CLC dynamic-persistent scheduling. Compile-time constant: when False
        # the kernel is byte-for-byte the original static-persistent path, so
        # the non-CLC build has exactly the previous performance. When True the
        # grid is launched full-size and the TMA warp drives a try_cancel loop.
        self.enable_clc = bool(enable_clc)
        self.clc_sched_stages = int(clc_sched_stages) if self.enable_clc else 0

        self.acc_dtype: Type[cutlass.Numeric] = cutlass.Float32
        self.a_dtype: Type[cutlass.Numeric] = cutlass.BFloat16
        self.b_dtype: Type[cutlass.Numeric] = cutlass.BFloat16
        self.c_dtype: Type[cutlass.Numeric] = cutlass.BFloat16

        self.use_2cta_instrs = False
        self.cta_group = tcgen05.CtaGroup.ONE
        # Two 1-CTA UMMA blocks share each M tile and multicast A.
        self.cluster_shape_mn = (1, 2)
        self.arch = "sm_100"
        self.occupancy = 1

        self.pull_warp_id = PULL_WARP
        self.tma_warp_id = TMA_WARP
        self.mma_warp_id = MMA_WARP
        self.epilogue_warp_id = EPILOGUE_WARPS
        self.threads_per_cta = THREADS_PER_CTA

        self.enable_setmaxnreg = enable_setmaxnreg
        self.request_tma_store = use_tma_store
        self.has_tma_store_helper = bool(
            hasattr(utils, "gemm")
            and hasattr(utils.gemm, "sm100")
            and hasattr(utils.gemm.sm100, "epilogue_tma_store")
        )
        self.use_tma_store = bool(use_tma_store and self.has_tma_store_helper)

        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3

    def _create_tiled_mma(self) -> cute.TiledMma:
        return utils.sm100.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            mma_tiler_mn=(self.tile_m, self.tile_n),
            cta_group=self.cta_group,
        )

    def _setup_attributes(self) -> cute.TiledMma:
        tiled_mma = self._create_tiled_mma()
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        if self.tile_k % mma_inst_shape_k != 0:
            raise ValueError(
                f"tile_k={self.tile_k} must be divisible by MMA K={mma_inst_shape_k}"
            )

        self.mma_tiler = (self.tile_m, self.tile_n, self.tile_k)
        self.cta_tile_shape_mnk = (
            self.tile_m // cute.size(tiled_mma.thr_id.shape),
            self.tile_n,
            self.tile_k,
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        if cutlass.const_expr(self.use_tma_store):
            self.epi_tile = (self.tile_m, 16)
            c_smem_layout_one = utils.sm100.make_smem_layout_epi(
                self.c_dtype, self.c_layout, self.epi_tile, 1
            )
        else:
            self.epi_tile = self.cta_tile_shape_mnk[:2]
            c_smem_layout_one = None

        a_smem_one = utils.sm100.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, 1
        )
        b_smem_one = utils.sm100.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, 1
        )
        ab_bytes_per_stage = cute.size_in_bytes(
            self.a_dtype, a_smem_one
        ) + cute.size_in_bytes(self.b_dtype, b_smem_one)

        smem_capacity = utils.get_smem_capacity_in_bytes()
        reserved_bytes = DEFAULT_SMEM_RESERVE_BYTES
        # Reserve the CLC scheduler ring / cross-CTA handoff / try_cancel-state
        # SMEM up front so the auto A/B stage count accounts for it. Zero when
        # CLC is disabled, keeping the non-CLC A/B pipeline depth identical.
        self.clc_smem_bytes = 0
        if self.enable_clc:
            s = self.clc_sched_stages
            # ring mbars(16*S) + sched_seq(4*S) + clc_resp_mbar(8) +
            # clc_resp 128b(16) + handoff_mbar(8*S) + handoff_seq(4*S), plus
            # generous alignment slack.
            self.clc_smem_bytes = 32 * s + 24 + 128
        reserved_bytes += self.clc_smem_bytes
        if self.enable_bulk_pull:
            reserved_bytes += bulk_pull_storage_reserve_bytes(self.bulk_row_elements)
        c_bytes_per_stage = 0
        c_bytes = 0
        self.num_c_stage = 0
        if cutlass.const_expr(self.use_tma_store):
            self.num_c_stage = self.requested_num_c_stage
            c_bytes_per_stage = cute.size_in_bytes(self.c_dtype, c_smem_layout_one)
            c_bytes = c_bytes_per_stage * self.num_c_stage

        auto_ab_stages = (
            smem_capacity - reserved_bytes - c_bytes
        ) // ab_bytes_per_stage
        # Keep auto stages conservative: a deeper AB pipeline on this kernel
        # increases SMEM pressure and often worsens No Eligible/Long Scoreboard.
        auto_ab_stages = max(2, min(DEFAULT_AUTO_AB_STAGES_CAP, auto_ab_stages))
        if self.requested_num_ab_stage > 0:
            self.num_ab_stage = min(self.requested_num_ab_stage, auto_ab_stages)
        else:
            self.num_ab_stage = auto_ab_stages

        self.a_smem_layout_staged = utils.sm100.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = utils.sm100.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
        )
        self.c_smem_layout_staged = None
        if cutlass.const_expr(self.use_tma_store):
            self.c_smem_layout_staged = utils.sm100.make_smem_layout_epi(
                self.c_dtype,
                self.c_layout,
                self.epi_tile,
                self.num_c_stage,
            )

        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )
        self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(
            tCtAcc_fake, arch=self.arch
        )
        return tiled_mma

    @cute.jit
    def __call__(
        self,
        peer_address_table: cute.Tensor,
        pull_cursors: cute.Tensor,
        ready_counts: cute.Tensor,
        weight_nkl: cute.Tensor,
        out_mnl: cute.Tensor,
        dy_mkl: cute.Tensor,
        bias_n: Optional[cute.Tensor],
        rank: cutlass.Constexpr[int],
        up: cutlass.Constexpr[int],
        pull_chunk_m: cutlass.Constexpr[int],
        pull_rows_per_ticket: cutlass.Constexpr[int],
        logical_m: cutlass.Int32,
        peer_rows: cutlass.Int32,
        epoch: cutlass.Int32,
        stream: CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        self.a_major_mode = utils.LayoutEnum.from_tensor(dy_mkl).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(weight_nkl).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(out_mnl)
        tiled_mma = self._setup_attributes()

        a_op = utils.sm100.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            dy_mkl,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        b_op = utils.sm100.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            weight_nkl,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        self.num_tma_load_bytes = (
            cute.size_in_bytes(self.a_dtype, a_smem_layout)
            + cute.size_in_bytes(self.b_dtype, b_smem_layout)
        ) * atom_thr_size

        tma_atom_c = None
        tma_tensor_c = out_mnl
        if cutlass.const_expr(self.use_tma_store):
            epi_smem_layout = cute.select(self.c_smem_layout_staged, mode=[0, 1])
            tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                out_mnl,
                epi_smem_layout,
                self.epi_tile,
            )

        n = out_mnl.shape[1]
        num_m_tiles = (logical_m + self.tile_m - 1) // self.tile_m
        num_n_tiles = n // self.tile_n
        cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        num_m_tile_clusters = (
            num_m_tiles + self.cluster_shape_mn[0] - 1
        ) // self.cluster_shape_mn[0]
        # An odd N-tile count leaves the second CTA of the last cluster an
        # out-of-bounds tile: TMA zero-fills its loads and drops its stores.
        num_n_tile_clusters = (
            num_n_tiles + self.cluster_shape_mn[1] - 1
        ) // self.cluster_shape_mn[1]
        total_tile_clusters = num_m_tile_clusters * num_n_tile_clusters
        max_active_clusters = max(1, self.max_active_ctas // cluster_size)
        if cutlass.const_expr(self.enable_clc):
            # CLC dynamic-persistent: launch one cluster per output tile (the
            # full problem grid). The hardware runs as many clusters as fit;
            # each running cluster processes its own tile then uses try_cancel
            # to absorb not-yet-started clusters' tiles, so occupancy is elastic
            # and coexists with a concurrent communication kernel.
            grid_z = total_tile_clusters
        else:
            # M is dynamic, so the active cluster count is clamped at runtime.
            # Persistent scheduling handles remainder or fully idle clusters
            # safely.
            grid_z = cutlass.min(
                cutlass.Int32(max_active_clusters), total_tile_clusters
            )

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            dy_mkl,
            bias_n,
            peer_address_table,
            pull_cursors,
            ready_counts,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            rank,
            up,
            pull_chunk_m,
            pull_rows_per_ticket,
            logical_m,
            peer_rows,
            epoch,
            epilogue_op,
        ).launch(
            grid=(self.cluster_shape_mn[0], self.cluster_shape_mn[1], grid_z),
            block=(self.threads_per_cta, 1, 1),
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: cute.Tensor,
        dy_mkl: cute.Tensor,
        bias_n: Optional[cute.Tensor],
        peer_address_table: cute.Tensor,
        pull_cursors: cute.Tensor,
        ready_counts: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        rank: cutlass.Constexpr[int],
        up: cutlass.Constexpr[int],
        pull_chunk_m: cutlass.Constexpr[int],
        pull_rows_per_ticket: cutlass.Constexpr[int],
        logical_m: cutlass.Int32,
        peer_rows: cutlass.Int32,
        epoch: cutlass.Int32,
        epilogue_op: cutlass.Constexpr,
    ):
        tidx, thread_y, thread_z = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = tidx % WARP_SIZE
        block_x, block_y, block_z = cute.arch.block_idx()
        grid_x, grid_y, grid_z = cute.arch.grid_dim()
        cluster_linear = block_x
        cluster_stride = grid_x
        m_tile_cluster_span = cutlass.Int32(1)
        m_tile_cluster_offset = cutlass.Int32(0)
        n_tile_cluster_span = cutlass.Int32(1)
        n_tile_cluster_offset = cutlass.Int32(0)
        if cutlass.const_expr(self.cluster_shape_mn[0] * self.cluster_shape_mn[1] > 1):
            cluster_idx_x, cluster_idx_y, cluster_idx_z = cute.arch.cluster_idx()
            cluster_dim_x, cluster_dim_y, cluster_dim_z = cute.arch.cluster_dim()
            cta_rank_in_cluster = cute.arch.make_warp_uniform(
                cute.arch.block_idx_in_cluster()
            )
            block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
                cta_rank_in_cluster
            )
            cluster_linear = cluster_idx_z
            cluster_stride = cluster_dim_z
            m_tile_cluster_span = cutlass.Int32(self.cluster_shape_mn[0])
            m_tile_cluster_offset = block_in_cluster_coord_vmnk[1]
            n_tile_cluster_span = cutlass.Int32(self.cluster_shape_mn[1])
            n_tile_cluster_offset = block_in_cluster_coord_vmnk[2]

        k_total = mA_mkl.shape[1]
        n = mC_mnl.shape[1]
        shard_k = k_total // up
        num_m_tiles = (logical_m + self.tile_m - 1) // self.tile_m
        num_n_tiles = n // self.tile_n
        num_n_tile_clusters = (
            num_n_tiles + n_tile_cluster_span - 1
        ) // n_tile_cluster_span
        total_tile_clusters = num_m_tiles * num_n_tile_clusters
        tickets_per_m_tile = self.tile_m // pull_rows_per_ticket
        num_pull_tickets = (
            logical_m + pull_rows_per_ticket - 1
        ) // pull_rows_per_ticket
        full_ready_target = cutlass.Int32(tickets_per_m_tile)

        if cutlass.const_expr(self.enable_setmaxnreg):
            if warp_idx < self.mma_warp_id:
                cute.arch.warpgroup_reg_alloc(240)
            else:
                cute.arch.warpgroup_reg_dealloc(120)

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            if cutlass.const_expr(self.use_tma_store):
                cpasync.prefetch_descriptor(tma_atom_c)

        # CLC scheduler SMEM is folded into SharedStorage (only when enabled so
        # the non-CLC build keeps a byte-for-byte identical layout and A/B stage
        # count). A separately-allocated struct cannot be flattened by the DSL
        # across the warp-role dynamic ``if`` regions, so it must live here.
        if cutlass.const_expr(self.enable_clc):
            S = self.clc_sched_stages

            @cute.struct
            class SharedStorage:
                ab_full_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.num_ab_stage * 2
                ]
                acc_full_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.num_acc_stage * 2
                ]
                # MMA publishes tile metadata through the matching accumulator
                # stage so epilogue warps do not consume the CLC ring separately.
                acc_work_seq: cute.struct.MemRange[cutlass.Int32, self.num_acc_stage]
                tmem_dealloc_mbar_ptr: cutlass.Int64
                tmem_holding_buf: cutlass.Int32
                # PipelineAsync full/empty mbarriers for the CTA-local ring.
                clc_ring_mbar: cute.struct.MemRange[cutlass.Int64, S * 2]
                # Fetched work_linear per ring stage (TMA writes, MMA/epi read).
                clc_sched_seq: cute.struct.MemRange[cutlass.Int32, S]
                # try_cancel async-response completion mbarrier (CTA0 only).
                clc_resp_mbar: cute.struct.MemRange[cutlass.Int64, 1]
                # 128-bit try_cancel response buffer (b128 store needs 16B align).
                clc_resp: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 4], 16]
                # Cross-CTA handoff of work_linear (CTA0 -> other cluster CTAs).
                clc_handoff_mbar: cute.struct.MemRange[cutlass.Int64, S]
                clc_handoff_seq: cute.struct.MemRange[cutlass.Int32, S]

        else:

            @cute.struct
            class SharedStorage:
                ab_full_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.num_ab_stage * 2
                ]
                acc_full_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.num_acc_stage * 2
                ]
                tmem_dealloc_mbar_ptr: cutlass.Int64
                tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        acc_work_seq_ptr = None
        if cutlass.const_expr(self.enable_clc):
            acc_work_seq_ptr = storage.acc_work_seq.data_ptr()

        # The bulk puller uses one warp with three independent full-row stages.
        # Pointers are extracted before dynamic warp-role branches so
        # the DSL never has to flatten the user-defined storage object there.
        bulk_mbar_ptr = None
        bulk_buf_ptr = None
        if cutlass.const_expr(self.enable_bulk_pull):

            @cute.struct
            class BulkPullStorage:
                mbar: cute.struct.MemRange[cutlass.Int64, BULK_PULL_TOTAL_STAGES]
                buf: cute.struct.Align[
                    cute.struct.MemRange[
                        cutlass.BFloat16,
                        self.bulk_row_elements * BULK_PULL_TOTAL_STAGES,
                    ],
                    128,
                ]

            bulk_storage = smem.allocate(BulkPullStorage)
            bulk_mbar_ptr = bulk_storage.mbar.data_ptr()
            bulk_buf_ptr = bulk_storage.buf.data_ptr()

        ab_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producers = self.cluster_shape_mn[0] + self.cluster_shape_mn[1] - 1
        ab_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producers
        )
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_producer_group,
            consumer_group=ab_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        acc_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        acc_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len(self.epilogue_warp_id)
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_producer_group,
            consumer_group=acc_consumer_group,
            cta_layout_vmnk=None,
            defer_sync=True,
        )

        # CTA-local CLC scheduler ring: the TMA warp (producer) publishes each
        # fetched work_linear; the MMA warp and the epilogue warps (consumers)
        # read it. This decouples the three tile loops under dynamic scheduling
        # exactly as the AB/ACC pipelines do under static scheduling.
        sched_ring = None
        if cutlass.const_expr(self.enable_clc):
            sched_ring = pipeline.PipelineAsync.create(
                barrier_storage=storage.clc_ring_mbar.data_ptr(),
                num_stages=self.clc_sched_stages,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                defer_sync=True,
            )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=WARP_SIZE * (1 + len(self.epilogue_warp_id)),
        )
        tmem_dealloc_barrier = None
        if cutlass.const_expr(not self.use_tma_store):
            tmem_dealloc_barrier = pipeline.NamedBarrier(
                barrier_id=self.tmem_dealloc_sync_bar_id,
                num_threads=WARP_SIZE * len(self.epilogue_warp_id),
            )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        # Initialize the raw CLC mbarriers (try_cancel response + cross-CTA
        # handoff). The cluster init barrier below makes them visible to every
        # warp/CTA. Each is completed by a single tx-carrying arrival. The
        # struct pointers are extracted OUTSIDE the dynamic ``if`` so the DSL
        # does not try to capture the whole SharedStorage across that region.
        if cutlass.const_expr(self.enable_clc):
            clc_resp_mbar_init = storage.clc_resp_mbar.data_ptr()
            clc_handoff_mbar_init = storage.clc_handoff_mbar.data_ptr()
            if warp_idx == self.tma_warp_id:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(clc_resp_mbar_init, 1)
                    for j in cutlass.range_constexpr(self.clc_sched_stages):
                        cute.arch.mbarrier_init(clc_handoff_mbar_init + j, 1)
                cute.arch.mbarrier_init_fence()

        if cutlass.const_expr(self.enable_bulk_pull):
            if warp_idx == self.pull_warp_id:
                bulk_worker_idx = cutlass.Int32(0)
                with cute.arch.elect_one():
                    for j in cutlass.range_constexpr(BULK_PULL_STAGES):
                        cute.arch.mbarrier_init(
                            bulk_mbar_ptr + bulk_worker_idx * BULK_PULL_STAGES + j,
                            1,
                        )
                cute.arch.mbarrier_init_fence()

        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        sC = None
        if cutlass.const_expr(self.use_tma_store):
            sC = smem.allocate_tensor(
                element_type=self.c_dtype,
                layout=c_smem_layout_staged.outer,
                byte_alignment=128,
                swizzle=c_smem_layout_staged.inner,
            )

        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.mma_tiler, (None, 0, None)),
            (None, None, None),
        )
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.mma_tiler, (0, None, None)),
            (None, None, None),
        )
        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.mma_tiler, (None, None, 0)),
            (None, None, None),
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        thr_mma = tiled_mma.get_slice(0)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        a_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
        )
        b_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
        )

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        # Vectorized pull: 256-, 128- or 64-bit transactions (16, 8 or 4
        # BF16/lane) depending on shard_k; lane starting addresses satisfy the
        # atom alignment in every case.
        # CLC scheduler views (SMEM tensors + the scheduler-CTA flag). Only the
        # rank-0 CTA of each cluster issues try_cancel; it broadcasts the
        # fetched work_linear to the peer CTA(s) via the handoff mbarriers.
        if cutlass.const_expr(self.enable_clc):
            acc_work_seq = cute.make_tensor(
                acc_work_seq_ptr,
                cute.make_layout(self.num_acc_stage),
            )
            clc_seq = cute.make_tensor(
                storage.clc_sched_seq.data_ptr(),
                cute.make_layout(self.clc_sched_stages),
            )
            clc_handoff = cute.make_tensor(
                storage.clc_handoff_seq.data_ptr(),
                cute.make_layout(self.clc_sched_stages),
            )
            clc_handoff_seq_ptr = storage.clc_handoff_seq.data_ptr()
            clc_handoff_mbar = storage.clc_handoff_mbar.data_ptr()
            clc_resp_mbar = storage.clc_resp_mbar.data_ptr()
            clc_resp_ptr = storage.clc_resp.data_ptr()
            clc_is_sched_cta = cta_rank_in_cluster == 0
            # Rank 0 broadcasts every canceled cluster ID to all M peers.

        pull_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.BFloat16,
            num_bits_per_copy=self.pull_num_bits,
        )
        pull_tiled_copy = cute.make_tiled_copy_tv(
            pull_copy_atom,
            cute.make_layout(WARP_SIZE),
            cute.make_layout(self.pull_elements_per_lane),
        )
        pull_thr_copy = pull_tiled_copy.get_slice(lane_idx)

        pull_worker_linear = (
            block_z * cutlass.Int32(self.cluster_shape_mn[0] * self.cluster_shape_mn[1])
            + cta_rank_in_cluster
        )
        cursor_shard = pull_worker_linear % cutlass.Int32(PULL_CURSOR_SHARDS)
        cursor_base_ptr = pull_cursors.iterator + epoch * cutlass.Int32(
            PULL_CURSOR_STRIDE
        )

        if cutlass.const_expr(self.enable_bulk_pull):
            if warp_idx == self.pull_warp_id:
                pull_worker_group = pull_worker_linear // cutlass.Int32(
                    PULL_CURSOR_SHARDS
                )
                pull_worker_active = pull_worker_group % cutlass.Int32(
                    PULL_WORKER_GROUP_PERIOD
                ) < cutlass.Int32(PULL_ACTIVE_WORKER_GROUPS)
                if pull_worker_active:
                    if cutlass.const_expr(self.enable_clc):
                        if lane_idx == 0:
                            bitmap_word = cursor_shard // cutlass.Int32(32)
                            bitmap_bit = cutlass.Int32(1) << (
                                cursor_shard % cutlass.Int32(32)
                            )
                            cute.arch.atomic_or(
                                ptr=(
                                    cursor_base_ptr
                                    + PULL_CURSOR_OWNER_BITMAP_OFFSET
                                    + bitmap_word
                                ).llvm_ptr,
                                val=bitmap_bit,
                                sem="relaxed",
                                scope="gpu",
                            )
                    bulk_worker_idx = cutlass.Int32(0)
                    self._bulk_pull_worker(
                        lane_idx,
                        peer_address_table,
                        cursor_base_ptr,
                        cursor_shard,
                        ready_counts,
                        dy_mkl,
                        pull_tiled_copy,
                        pull_thr_copy,
                        bulk_mbar_ptr + bulk_worker_idx * BULK_PULL_STAGES,
                        bulk_buf_ptr
                        + bulk_worker_idx * BULK_PULL_STAGES * self.bulk_row_elements,
                        rank,
                        up,
                        pull_rows_per_ticket,
                        logical_m,
                        peer_rows,
                        epoch,
                        num_m_tiles,
                        num_pull_tickets,
                        tickets_per_m_tile,
                    )

        if cutlass.const_expr(not self.enable_bulk_pull):
            cursor_ptr = pull_cursors.iterator + epoch * cutlass.Int32(
                PULL_CURSOR_STRIDE
            )
            if warp_idx >= self.pull_warp_id:
                # Rows (and columns) loaded into registers before they are
                # stored; see _vector_pull_rows_per_ticket/_pull_column_chunks.
                pull_rows_in_flight = self.pull_rows_in_flight
                pull_chunk_k = shard_k // self.pull_col_chunks
                rank_base_row_i64 = cutlass.Int64(rank) * cutlass.Int64(peer_rows)
                chunk_id = cutlass.Int32(num_pull_tickets)
                if lane_idx == 0:
                    chunk_id = cute.arch.atomic_add(
                        ptr=cursor_ptr.llvm_ptr,
                        val=cutlass.Int32(1),
                        sem="relaxed",
                        scope="gpu",
                    )
                chunk_id = cute.arch.shuffle_sync(chunk_id, offset=0)
                pending_ready_m_tile = cutlass.Int32(-1)
                pending_ready_count = cutlass.Int32(0)

                while chunk_id < num_pull_tickets:
                    chunk_first_token = chunk_id * pull_rows_per_ticket
                    chunk_has_full_tokens = (
                        chunk_first_token + cutlass.Int32(pull_rows_per_ticket)
                        <= logical_m
                    )
                    for peer in cutlass.range_constexpr(up):
                        peer_address = peer_address_table[peer]
                        peer_ptr = cute.make_ptr(
                            cutlass.BFloat16,
                            peer_address,
                            cute.AddressSpace.gmem,
                            assumed_align=32,
                        )
                        if cutlass.const_expr(pull_rows_per_ticket % 4 == 0):
                            if chunk_has_full_tokens:
                                for group_idx in cutlass.range_constexpr(
                                    pull_rows_per_ticket // pull_rows_in_flight
                                ):
                                    for col_chunk in cutlass.range_constexpr(self.pull_col_chunks):
                                        self._pull_rows(
                                            pull_tiled_copy,
                                            pull_thr_copy,
                                            peer_ptr,
                                            dy_mkl.iterator,
                                            rank_base_row_i64,
                                            chunk_first_token + group_idx * pull_rows_in_flight,
                                            peer,
                                            k_total,
                                            shard_k,
                                            pull_rows_in_flight,
                                            col_chunk * pull_chunk_k,
                                            pull_chunk_k,
                                        )
                            else:
                                self._pull_rows_predicated(
                                    pull_tiled_copy,
                                    pull_thr_copy,
                                    peer_ptr,
                                    dy_mkl.iterator,
                                    rank_base_row_i64,
                                    chunk_first_token,
                                    logical_m,
                                    peer,
                                    k_total,
                                    shard_k,
                                    pull_rows_per_ticket,
                                )
                        else:
                            self._pull_rows_predicated(
                                pull_tiled_copy,
                                pull_thr_copy,
                                peer_ptr,
                                dy_mkl.iterator,
                                rank_base_row_i64,
                                chunk_first_token,
                                logical_m,
                                peer,
                                k_total,
                                shard_k,
                                pull_rows_per_ticket,
                            )

                    # Aggregate the ready release per worker warp and per M tile so
                    # that consecutive tickets of one tile publish a single
                    # Publish one aggregated fence + release atomic per M tile.
                    pull_m_tile = chunk_id // tickets_per_m_tile
                    if pending_ready_count == cutlass.Int32(0):
                        pending_ready_m_tile = pull_m_tile
                    pending_ready_count = pending_ready_count + cutlass.Int32(1)

                    next_chunk_id = cutlass.Int32(num_pull_tickets)
                    if lane_idx == 0:
                        next_chunk_id = cute.arch.atomic_add(
                            ptr=cursor_ptr.llvm_ptr,
                            val=cutlass.Int32(1),
                            sem="relaxed",
                            scope="gpu",
                        )
                    next_chunk_id = cute.arch.shuffle_sync(next_chunk_id, offset=0)

                    should_publish_ready = next_chunk_id >= num_pull_tickets
                    if next_chunk_id < num_pull_tickets:
                        next_m_tile = next_chunk_id // tickets_per_m_tile
                        if next_m_tile != pending_ready_m_tile:
                            should_publish_ready = cutlass.Boolean(1)

                    if should_publish_ready:
                        # Publish this worker warp's dy writes for the current M tile.
                        cute.arch.fence_proxy("async.global")
                        cute.arch.fence_acq_rel_gpu()
                        cute.arch.sync_warp()
                        if lane_idx == 0:
                            ready_linear_idx = (
                                epoch * num_m_tiles + pending_ready_m_tile
                            )
                            ready_ptr = ready_counts.iterator + ready_linear_idx
                            cute.arch.atomic_add(
                                ptr=ready_ptr.llvm_ptr,
                                val=pending_ready_count,
                                sem="release",
                                scope="gpu",
                            )
                        cute.arch.sync_warp()
                        pending_ready_count = cutlass.Int32(0)

                    chunk_id = next_chunk_id

        # One TMA warp issues both A and B into one transaction barrier per
        # pipeline stage.
        if warp_idx == self.tma_warp_id:
            if cutlass.const_expr(self.enable_clc):
                sched_prod_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.clc_sched_stages
                )
                clc_query_count = cutlass.Int32(0)
                clc_handoff_count = cutlass.Int32(0)
            last_ready_m_tile = cutlass.Int32(-1)
            work_linear = cluster_linear
            while work_linear < total_tile_clusters:
                if cutlass.const_expr(self.enable_clc):
                    # Publish this tile into the CTA-local ring for the MMA and
                    # epilogue warps before issuing its loads.
                    sched_ring.producer_acquire(sched_prod_state)
                    with cute.arch.elect_one():
                        clc_seq[sched_prod_state.index] = work_linear
                        sched_ring.producer_commit(sched_prod_state)
                    sched_prod_state.advance()
                    # Start the next CLC cancellation query before waiting for
                    # pull readiness and issuing this tile's TMA loads. The
                    # asynchronous response then overlaps the full producer
                    # body instead of extending the tile-to-tile critical path.
                    if clc_is_sched_cta:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(clc_resp_mbar, 16)
                            _clc_try_cancel(clc_resp_mbar, clc_resp_ptr)
                        cute.arch.sync_warp()

                m_tile_idx = work_linear // num_n_tile_clusters
                n_tile_cluster_idx = work_linear - m_tile_idx * num_n_tile_clusters
                n_tile_idx = (
                    n_tile_cluster_idx * n_tile_cluster_span + n_tile_cluster_offset
                )

                # Readiness is monotonic for the invocation. CLC commonly
                # hands a persistent CTA several adjacent N tiles of the same
                # M tile, so avoid repeating the acquire load after the first
                # successful gate for that M tile.
                if m_tile_idx != last_ready_m_tile:
                    ready_linear_idx = epoch * num_m_tiles + m_tile_idx
                    ready_ptr = ready_counts.iterator + ready_linear_idx
                    tile_first_token = m_tile_idx * self.tile_m
                    remaining_tokens = logical_m - tile_first_token
                    tile_ready_target = full_ready_target
                    if remaining_tokens < cutlass.Int32(self.tile_m):
                        tile_ready_target = (
                            remaining_tokens + cutlass.Int32(pull_rows_per_ticket - 1)
                        ) // cutlass.Int32(pull_rows_per_ticket)
                    observed = cutlass.Int32(0)
                    if lane_idx == 0:
                        observed = cute.arch.load(
                            ready_ptr.llvm_ptr,
                            cutlass.Int32,
                            sem="acquire",
                            scope="gpu",
                        )
                    observed = cute.arch.shuffle_sync(observed, offset=0)
                    while observed < tile_ready_target:
                        if lane_idx == 0:
                            observed = cute.arch.load(
                                ready_ptr.llvm_ptr,
                                cutlass.Int32,
                                sem="acquire",
                                scope="gpu",
                            )
                        observed = cute.arch.shuffle_sync(observed, offset=0)
                    if lane_idx == 0:
                        cute.arch.fence_proxy("async.global")
                    cute.arch.sync_warp()
                    last_ready_m_tile = m_tile_idx

                tAgA_slice = tAgA[(None, m_tile_idx, None, 0)]
                tBgB_slice = tBgB[(None, n_tile_idx, None, 0)]
                ab_producer.reset()
                peek_ab_empty = ab_producer.try_acquire()
                if cutlass.const_expr(self.tma_python_k_loop):
                    for k_iter in range(k_tile_cnt):
                        handle = ab_producer.acquire_and_advance(peek_ab_empty)
                        cute.copy(
                            tma_atom_a,
                            tAgA_slice[(None, handle.count)],
                            tAsA[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            mcast_mask=a_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB_slice[(None, handle.count)],
                            tBsB[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                        )
                        peek_ab_empty = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_empty = ab_producer.try_acquire()
                else:
                    for k_iter in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        handle = ab_producer.acquire_and_advance(peek_ab_empty)
                        cute.copy(
                            tma_atom_a,
                            tAgA_slice[(None, handle.count)],
                            tAsA[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            mcast_mask=a_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB_slice[(None, handle.count)],
                            tBsB[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                        )
                        peek_ab_empty = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_empty = ab_producer.try_acquire()

                if cutlass.const_expr(self.enable_clc):
                    # Fetch the next tile via CLC. Only the rank-0 CTA issues
                    # try_cancel; it hands the fetched work_linear to the peer
                    # CTA so the whole cluster processes the same M tile.
                    if clc_is_sched_cta:
                        # Complete the query issued before this tile's pull
                        # readiness wait and TMA producer loop.
                        cute.arch.mbarrier_wait(clc_resp_mbar, clc_query_count % 2)
                        clc_query_count += 1
                        _, _, clc_bidz, clc_valid = cute.arch.clc_response(clc_resp_ptr)
                        cute.arch.fence_view_async_shared()
                        # bidz is the canceled cluster's z index == work_linear
                        # (cluster z-extent is 1). An invalid response means no
                        # pending tile is left; use the sentinel to stop.
                        next_work = total_tile_clusters
                        if clc_valid != 0:
                            next_work = clc_bidz
                        clc_slot = clc_handoff_count % self.clc_sched_stages
                        with cute.arch.elect_one():
                            for peer_rank in cutlass.range_constexpr(
                                1,
                                self.cluster_shape_mn[0] * self.cluster_shape_mn[1],
                            ):
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    clc_handoff_mbar + clc_slot, 4, peer_rank
                                )
                                _clc_store_shared_remote_i32(
                                    next_work,
                                    clc_handoff_seq_ptr + clc_slot,
                                    clc_handoff_mbar + clc_slot,
                                    peer_rank,
                                )
                        clc_handoff_count += 1
                        work_linear = next_work
                    else:
                        clc_slot = clc_handoff_count % self.clc_sched_stages
                        clc_phase = (clc_handoff_count // self.clc_sched_stages) % 2
                        cute.arch.mbarrier_wait(clc_handoff_mbar + clc_slot, clc_phase)
                        cute.arch.fence_view_async_shared()
                        clc_handoff_count += 1
                        work_linear = clc_handoff[clc_slot]
                else:
                    work_linear += cluster_stride
            if cutlass.const_expr(self.enable_clc):
                # Terminal sentinel so the ring consumers exit their loops.
                sched_ring.producer_acquire(sched_prod_state)
                with cute.arch.elect_one():
                    clc_seq[sched_prod_state.index] = total_tile_clusters
                    sched_ring.producer_commit(sched_prod_state)
                sched_prod_state.advance()
                sched_ring.producer_tail(sched_prod_state)
            ab_producer.tail()

        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            acc_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            if cutlass.const_expr(self.enable_clc):
                sched_cons_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.clc_sched_stages
                )

            work_linear = cluster_linear
            if cutlass.const_expr(self.enable_clc):
                # Prime with the first scheduled tile; subsequent tiles (and the
                # terminating sentinel) are read at the bottom of the loop so the
                # `while` condition alone ends the loop (no `break` in a kernel).
                sched_ring.consumer_wait(sched_cons_state)
                work_linear = clc_seq[sched_cons_state.index]
                with cute.arch.elect_one():
                    sched_ring.consumer_release(sched_cons_state)
                sched_cons_state.advance()
            while work_linear < total_tile_clusters:
                tCtAcc = tCtAcc_base[(None, None, None, acc_state.index)]
                ab_consumer.reset()
                peek_ab_full = ab_consumer.try_wait()
                acc_pipeline.producer_acquire(acc_state)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                for k_iter in range(k_tile_cnt):
                    handle = ab_consumer.wait_and_advance(peek_ab_full)
                    num_kblocks = cute.size(tCrA, mode=[2])
                    for kblk_idx in cutlass.range(num_kblocks, unroll_full=True):
                        kblk_crd = (None, None, kblk_idx, handle.index)
                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            tCrA[kblk_crd],
                            tCrB[kblk_crd],
                            tCtAcc,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    handle.release()
                    peek_ab_full = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek_ab_full = ab_consumer.try_wait()

                if cutlass.const_expr(self.enable_clc):
                    with cute.arch.elect_one():
                        acc_work_seq[acc_state.index] = work_linear
                acc_pipeline.producer_commit(acc_state)
                acc_state.advance()
                if cutlass.const_expr(self.enable_clc):
                    sched_ring.consumer_wait(sched_cons_state)
                    work_linear = clc_seq[sched_cons_state.index]
                    with cute.arch.elect_one():
                        sched_ring.consumer_release(sched_cons_state)
                    sched_cons_state.advance()
                else:
                    work_linear += cluster_stride

            if cutlass.const_expr(self.enable_clc):
                acc_pipeline.producer_acquire(acc_state)
                with cute.arch.elect_one():
                    acc_work_seq[acc_state.index] = total_tile_clusters
                acc_pipeline.producer_commit(acc_state)
                acc_state.advance()
            acc_pipeline.producer_tail(acc_state)

        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            c_pipeline = None
            if cutlass.const_expr(self.use_tma_store):
                c_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, WARP_SIZE * len(self.epilogue_warp_id)
                )
                c_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.num_c_stage,
                    producer_group=c_producer_group,
                )
            work_linear = cluster_linear
            if cutlass.const_expr(self.enable_clc):
                acc_pipeline.consumer_wait(acc_consumer_state)
                work_linear = acc_work_seq[acc_consumer_state.index]
            while work_linear < total_tile_clusters:
                if cutlass.const_expr(not self.enable_clc):
                    acc_pipeline.consumer_wait(acc_consumer_state)
                m_tile_idx = work_linear // num_n_tile_clusters
                n_tile_cluster_idx = work_linear - m_tile_idx * num_n_tile_clusters
                n_tile_idx = (
                    n_tile_cluster_idx * n_tile_cluster_span + n_tile_cluster_offset
                )
                mma_tile_coord_mnl = (m_tile_idx, n_tile_idx, 0)
                # The out-of-bounds tile of an odd N-tile count reads the last
                # valid bias slice; its stores are dropped by TMA anyway.
                bias_tile_n = cutlass.min(n_tile_idx, num_n_tiles - 1)

                if cutlass.const_expr(self.use_tma_store):
                    if cutlass.const_expr(self.num_c_stage == 1):
                        acc_consumer_state = self.epilogue_tma_store_single_cstage(
                            tidx,
                            warp_idx,
                            tma_atom_c,
                            tCtAcc_base,
                            sC,
                            tCgC,
                            bias_n,
                            bias_tile_n,
                            epi_tile,
                            epilogue_op,
                            mma_tile_coord_mnl,
                            acc_consumer_state,
                            acc_pipeline,
                            c_pipeline,
                        )
                    else:
                        acc_consumer_state = self.epilogue_tma_store_skip_tail_barrier(
                            tidx,
                            warp_idx,
                            tma_atom_c,
                            tCtAcc_base,
                            sC,
                            tCgC,
                            bias_n,
                            bias_tile_n,
                            epi_tile,
                            epilogue_op,
                            mma_tile_coord_mnl,
                            acc_consumer_state,
                            acc_pipeline,
                            c_pipeline,
                        )
                else:
                    acc_consumer_state = self.epilogue_direct_store(
                        tidx,
                        acc_pipeline,
                        acc_consumer_state,
                        tiled_mma,
                        tCtAcc_base,
                        tCgC,
                        epi_tile,
                        mma_tile_coord_mnl,
                        epilogue_op,
                    )

                if cutlass.const_expr(self.enable_clc):
                    acc_pipeline.consumer_wait(acc_consumer_state)
                    work_linear = acc_work_seq[acc_consumer_state.index]
                    if work_linear >= total_tile_clusters:
                        with cute.arch.elect_one():
                            acc_pipeline.consumer_release(acc_consumer_state)
                        acc_consumer_state.advance()
                else:
                    work_linear += cluster_stride

            if cutlass.const_expr(self.use_tma_store):
                c_pipeline.producer_tail()
            else:
                tmem_dealloc_barrier.arrive_and_wait()

            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

    @cute.jit
    def _pull_rows(
        self,
        pull_tiled_copy: cute.TiledCopy,
        pull_thr_copy,
        peer_ptr: cute.Pointer,
        dy_ptr: cute.Pointer,
        rank_base_row_i64: cutlass.Int64,
        first_token: cutlass.Int32,
        peer: cutlass.Constexpr[int],
        k_total: cutlass.Constexpr[int],
        shard_k: cutlass.Constexpr[int],
        num_rows: cutlass.Constexpr[int],
        col_offset: cutlass.Constexpr[int],
        num_cols: cutlass.Constexpr[int],
    ) -> None:
        """Copy ``num_rows`` x ``num_cols`` of ``peer``'s shard into ``dy``.

        Every row is loaded into registers before any row is stored, so each
        lane keeps ``num_rows`` peer loads in flight. The caller bounds
        ``num_rows * num_cols`` to what fits in the pull warps' registers.
        """
        first_token_i64 = cutlass.Int64(first_token)
        src_offset_i64 = (rank_base_row_i64 + first_token_i64) * cutlass.Int64(
            shard_k
        ) + cutlass.Int64(col_offset)
        dst_offset_i64 = first_token_i64 * cutlass.Int64(k_total) + cutlass.Int64(
            peer * shard_k + col_offset
        )
        fragments = []
        destinations = []
        for row in cutlass.range_constexpr(num_rows):
            src_offset = cute.assume(
                src_offset_i64 + cutlass.Int64(row * shard_k), divby=PULL_DIVBY
            )
            dst_offset = cute.assume(
                dst_offset_i64 + cutlass.Int64(row * k_total), divby=PULL_DIVBY
            )
            src = pull_thr_copy.partition_S(
                cute.make_tensor(peer_ptr + src_offset, cute.make_layout(num_cols))
            )
            dst = pull_thr_copy.partition_D(
                cute.make_tensor(dy_ptr + dst_offset, cute.make_layout(num_cols))
            )
            fragment = cute.make_rmem_tensor(dst.shape, cutlass.BFloat16)
            cute.copy(pull_tiled_copy, src, fragment)
            fragments.append(fragment)
            destinations.append(dst)
        for row in cutlass.range_constexpr(num_rows):
            cute.copy(pull_tiled_copy, fragments[row], destinations[row])

    @cute.jit
    def _pull_rows_predicated(
        self,
        pull_tiled_copy: cute.TiledCopy,
        pull_thr_copy,
        peer_ptr: cute.Pointer,
        dy_ptr: cute.Pointer,
        rank_base_row_i64: cutlass.Int64,
        first_token: cutlass.Int32,
        logical_m: cutlass.Int32,
        peer: cutlass.Constexpr[int],
        k_total: cutlass.Constexpr[int],
        shard_k: cutlass.Constexpr[int],
        num_rows: cutlass.Constexpr[int],
    ) -> None:
        """Copy ``peer``'s shard of up to ``num_rows`` rows, skipping rows >= M."""
        for row in cutlass.range_constexpr(num_rows):
            token = first_token + row
            if token < logical_m:
                token_i64 = cutlass.Int64(token)
                src_offset = cute.assume(
                    (rank_base_row_i64 + token_i64) * cutlass.Int64(shard_k),
                    divby=PULL_DIVBY,
                )
                dst_offset = cute.assume(
                    token_i64 * cutlass.Int64(k_total)
                    + cutlass.Int64(peer) * cutlass.Int64(shard_k),
                    divby=PULL_DIVBY,
                )
                src = pull_thr_copy.partition_S(
                    cute.make_tensor(peer_ptr + src_offset, cute.make_layout(shard_k))
                )
                dst = pull_thr_copy.partition_D(
                    cute.make_tensor(dy_ptr + dst_offset, cute.make_layout(shard_k))
                )
                cute.copy(pull_tiled_copy, src, dst)

    @cute.jit
    def _bulk_pull_worker(
        self,
        lane_idx: cutlass.Int32,
        peer_address_table: cute.Tensor,
        cursor_base_ptr: cute.Pointer,
        cursor_shard: cutlass.Int32,
        ready_counts: cute.Tensor,
        dy_mkl: cute.Tensor,
        pull_tiled_copy: cute.TiledCopy,
        pull_thr_copy,
        bulk_mbar_ptr: cute.Pointer,
        bulk_buf_ptr: cute.Pointer,
        rank: cutlass.Constexpr[int],
        up: cutlass.Constexpr[int],
        pull_rows_per_ticket: cutlass.Constexpr[int],
        logical_m: cutlass.Int32,
        peer_rows: cutlass.Int32,
        epoch: cutlass.Int32,
        num_m_tiles: cutlass.Int32,
        num_pull_tickets: cutlass.Int32,
        tickets_per_m_tile: cutlass.Int32,
    ) -> None:
        """Full-row bulk pull worker with a predicated tail fallback."""
        k_total = dy_mkl.shape[1]
        shard_k = k_total // up
        bulk_batches = (pull_rows_per_ticket + BULK_PULL_STAGES - 1) // BULK_PULL_STAGES
        shard_bytes = shard_k * 2
        row_bytes = k_total * 2
        rank_base_row_i64 = cutlass.Int64(rank) * cutlass.Int64(peer_rows)
        peer_addresses = cute.make_rmem_tensor(cute.make_layout(up), cutlass.Int64)
        for peer in cutlass.range_constexpr(up):
            peer_addresses[peer] = peer_address_table[peer]

        # Preserve the tuned primary-shard loop exactly. CLC recovery is a
        # separate cold path after this queue is drained.
        cursor_ptr = cursor_base_ptr + cursor_shard
        chunk_id = cutlass.Int32(num_pull_tickets)
        if lane_idx == 0:
            chunk_id = cute.arch.atomic_add(
                ptr=cursor_ptr.llvm_ptr,
                val=cutlass.Int32(1),
                sem="relaxed",
                scope="gpu",
            )
        chunk_id = cute.arch.shuffle_sync(chunk_id, offset=0)
        chunk_id = chunk_id * cutlass.Int32(PULL_CURSOR_SHARDS) + cursor_shard
        pending_ready_m_tile = cutlass.Int32(-1)
        pending_ready_count = cutlass.Int32(0)
        bulk_phases = cute.make_rmem_tensor(
            cute.make_layout(BULK_PULL_STAGES), cutlass.Int32
        )
        bulk_phases.fill(cutlass.Int32(0))

        while chunk_id < num_pull_tickets:
            chunk_first_token = chunk_id * pull_rows_per_ticket
            chunk_has_full_tokens = (
                chunk_first_token + cutlass.Int32(pull_rows_per_ticket) <= logical_m
            )

            if chunk_has_full_tokens:
                # Prime three complete-row stages. Every peer contributes one
                # contiguous shard to the same stage barrier and SMEM row.
                for stage in cutlass.range_constexpr(BULK_PULL_STAGES):
                    stage_mbar = bulk_mbar_ptr + stage
                    stage_buf = bulk_buf_ptr + stage * self.bulk_row_elements
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(stage_mbar, row_bytes)
                    token = chunk_first_token + stage
                    src_offset_i64 = (
                        rank_base_row_i64 + cutlass.Int64(token)
                    ) * cutlass.Int64(shard_k)
                    src_offset = cute.assume(src_offset_i64, divby=PULL_DIVBY)
                    for peer in cutlass.range_constexpr(up):
                        peer_address = peer_addresses[peer]
                        peer_ptr = cute.make_ptr(
                            cutlass.BFloat16,
                            peer_address,
                            cute.AddressSpace.gmem,
                            assumed_align=32,
                        )
                        with cute.arch.elect_one():
                            _cp_async_bulk_g2s(
                                stage_buf + peer * shard_k,
                                peer_ptr + src_offset,
                                stage_mbar,
                                shard_bytes,
                            )

                for batch in cutlass.range_constexpr(bulk_batches):
                    # One contiguous S2G transaction materializes each row.
                    remaining_rows = pull_rows_per_ticket - batch * BULK_PULL_STAGES
                    active_stages = min(BULK_PULL_STAGES, remaining_rows)
                    for stage in cutlass.range_constexpr(active_stages):
                        token = chunk_first_token + batch * BULK_PULL_STAGES + stage
                        stage_mbar = bulk_mbar_ptr + stage
                        stage_buf = bulk_buf_ptr + stage * self.bulk_row_elements
                        stage_phase = bulk_phases[stage]
                        cute.arch.mbarrier_wait(stage_mbar, stage_phase)
                        bulk_phases[stage] = stage_phase ^ cutlass.Int32(1)
                        dst_offset_i64 = cutlass.Int64(token) * cutlass.Int64(k_total)
                        dst_offset = cute.assume(dst_offset_i64, divby=PULL_DIVBY)
                        with cute.arch.elect_one():
                            _cp_async_bulk_s2g(
                                dy_mkl.iterator + dst_offset,
                                stage_buf,
                                row_bytes,
                            )
                            cute.arch.cp_async_bulk_commit_group()

                    if cutlass.const_expr(batch + 1 < bulk_batches):
                        next_remaining_rows = (
                            pull_rows_per_ticket - (batch + 1) * BULK_PULL_STAGES
                        )
                        next_active_stages = min(BULK_PULL_STAGES, next_remaining_rows)
                        with cute.arch.elect_one():
                            cute.arch.cp_async_bulk_wait_group(0, read=True)
                        for stage in cutlass.range_constexpr(next_active_stages):
                            token = (
                                chunk_first_token
                                + (batch + 1) * BULK_PULL_STAGES
                                + stage
                            )
                            src_offset_i64 = (
                                rank_base_row_i64 + cutlass.Int64(token)
                            ) * cutlass.Int64(shard_k)
                            src_offset = cute.assume(src_offset_i64, divby=PULL_DIVBY)
                            stage_mbar = bulk_mbar_ptr + stage
                            stage_buf = bulk_buf_ptr + stage * self.bulk_row_elements
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    stage_mbar, row_bytes
                                )
                            for peer in cutlass.range_constexpr(up):
                                peer_address = peer_addresses[peer]
                                peer_ptr = cute.make_ptr(
                                    cutlass.BFloat16,
                                    peer_address,
                                    cute.AddressSpace.gmem,
                                    assumed_align=32,
                                )
                                with cute.arch.elect_one():
                                    _cp_async_bulk_g2s(
                                        stage_buf + peer * shard_k,
                                        peer_ptr + src_offset,
                                        stage_mbar,
                                        shard_bytes,
                                    )
                    else:
                        with cute.arch.elect_one():
                            cute.arch.cp_async_bulk_wait_group(0)
            else:
                # Dynamic-M tail (at most one ticket): predicated row copies.
                for peer in cutlass.range_constexpr(up):
                    peer_ptr = cute.make_ptr(
                        cutlass.BFloat16,
                        peer_addresses[peer],
                        cute.AddressSpace.gmem,
                        assumed_align=32,
                    )
                    self._pull_rows_predicated(
                        pull_tiled_copy,
                        pull_thr_copy,
                        peer_ptr,
                        dy_mkl.iterator,
                        rank_base_row_i64,
                        chunk_first_token,
                        logical_m,
                        peer,
                        k_total,
                        shard_k,
                        pull_rows_per_ticket,
                    )

            # Publish one aggregated ready release per worker and M tile.
            pull_m_tile = chunk_id // tickets_per_m_tile
            if pending_ready_count == cutlass.Int32(0):
                pending_ready_m_tile = pull_m_tile
            pending_ready_count = pending_ready_count + cutlass.Int32(1)

            next_chunk_id = cutlass.Int32(num_pull_tickets)
            if lane_idx == 0:
                next_chunk_id = cute.arch.atomic_add(
                    ptr=cursor_ptr.llvm_ptr,
                    val=cutlass.Int32(1),
                    sem="relaxed",
                    scope="gpu",
                )
            next_chunk_id = cute.arch.shuffle_sync(next_chunk_id, offset=0)
            next_chunk_id = (
                next_chunk_id * cutlass.Int32(PULL_CURSOR_SHARDS) + cursor_shard
            )

            should_publish_ready = next_chunk_id >= num_pull_tickets
            if next_chunk_id < num_pull_tickets:
                next_m_tile = next_chunk_id // tickets_per_m_tile
                if next_m_tile != pending_ready_m_tile:
                    should_publish_ready = cutlass.Boolean(1)

            if should_publish_ready:
                cute.arch.fence_proxy("async.global")
                cute.arch.fence_acq_rel_gpu()
                cute.arch.sync_warp()
                if lane_idx == 0:
                    ready_linear_idx = epoch * num_m_tiles + pending_ready_m_tile
                    ready_ptr = ready_counts.iterator + ready_linear_idx
                    cute.arch.atomic_add(
                        ptr=ready_ptr.llvm_ptr,
                        val=pending_ready_count,
                        sem="release",
                        scope="gpu",
                    )
                cute.arch.sync_warp()
                pending_ready_count = cutlass.Int32(0)

            chunk_id = next_chunk_id

        if cutlass.const_expr(self.enable_clc):
            owner_bitmap = cute.make_tensor(
                cursor_base_ptr + PULL_CURSOR_OWNER_BITMAP_OFFSET,
                cute.make_layout(PULL_CURSOR_BITMAP_WORDS),
            )
            owners_complete = cutlass.Boolean(1)
            for bitmap_word in cutlass.range_constexpr(PULL_CURSOR_BITMAP_WORDS):
                owner_word = cutlass.Int32(0)
                if lane_idx == 0:
                    owner_word = owner_bitmap[bitmap_word]
                owner_word = cute.arch.shuffle_sync(owner_word, offset=0)
                remaining_bits = PULL_CURSOR_SHARDS - bitmap_word * 32
                expected_word = (
                    -1 if remaining_bits >= 32 else (1 << remaining_bits) - 1
                )
                if owner_word != cutlass.Int32(expected_word):
                    owners_complete = cutlass.Boolean(0)

            if not owners_complete:
                rescue_owner = cutlass.Int32(1)
                if lane_idx == 0:
                    rescue_owner = cute.arch.atomic_cas(
                        (cursor_base_ptr + PULL_CURSOR_RESCUE_OWNER_OFFSET).llvm_ptr,
                        cmp=cutlass.Int32(0),
                        val=cutlass.Int32(1),
                        sem="relaxed",
                        scope="gpu",
                    )
                rescue_owner = cute.arch.shuffle_sync(rescue_owner, offset=0)
                if rescue_owner == cutlass.Int32(0):
                    self._rescue_pull_shards(
                        lane_idx,
                        peer_address_table,
                        cursor_base_ptr,
                        cursor_shard,
                        ready_counts,
                        dy_mkl,
                        pull_tiled_copy,
                        pull_thr_copy,
                        rank,
                        up,
                        pull_rows_per_ticket,
                        logical_m,
                        peer_rows,
                        epoch,
                        num_m_tiles,
                        num_pull_tickets,
                        tickets_per_m_tile,
                    )

    @cute.jit
    def _rescue_pull_shards(
        self,
        lane_idx: cutlass.Int32,
        peer_address_table: cute.Tensor,
        cursor_base_ptr: cute.Pointer,
        primary_cursor_shard: cutlass.Int32,
        ready_counts: cute.Tensor,
        dy_mkl: cute.Tensor,
        pull_tiled_copy: cute.TiledCopy,
        pull_thr_copy,
        rank: cutlass.Constexpr[int],
        up: cutlass.Constexpr[int],
        pull_rows_per_ticket: cutlass.Constexpr[int],
        logical_m: cutlass.Int32,
        peer_rows: cutlass.Int32,
        epoch: cutlass.Int32,
        num_m_tiles: cutlass.Int32,
        num_pull_tickets: cutlass.Int32,
        tickets_per_m_tile: cutlass.Int32,
    ) -> None:
        """Drain cursor shards whose original CLC workers never launched.

        The elected worker uses the simple vector-copy path so the tuned bulk
        loop remains unchanged. This path is cold when every cursor has an
        owner and correctness-critical when CLC cancels one before launch.
        """
        k_total = dy_mkl.shape[1]
        shard_k = k_total // up
        rank_base_row_i64 = cutlass.Int64(rank) * cutlass.Int64(peer_rows)
        cursor_shard = (primary_cursor_shard + cutlass.Int32(1)) % cutlass.Int32(
            PULL_CURSOR_SHARDS
        )
        cursor_shards_remaining = cutlass.Int32(PULL_CURSOR_SHARDS - 1)

        while cursor_shards_remaining > cutlass.Int32(0):
            cursor_ptr = cursor_base_ptr + cursor_shard
            chunk_id = cutlass.Int32(num_pull_tickets)
            if lane_idx == 0:
                chunk_id = cute.arch.atomic_add(
                    ptr=cursor_ptr.llvm_ptr,
                    val=cutlass.Int32(1),
                    sem="relaxed",
                    scope="gpu",
                )
            chunk_id = cute.arch.shuffle_sync(chunk_id, offset=0)
            chunk_id = chunk_id * cutlass.Int32(PULL_CURSOR_SHARDS) + cursor_shard
            pending_ready_m_tile = cutlass.Int32(-1)
            pending_ready_count = cutlass.Int32(0)

            while chunk_id < num_pull_tickets:
                chunk_first_token = chunk_id * pull_rows_per_ticket
                for peer in cutlass.range_constexpr(up):
                    peer_ptr = cute.make_ptr(
                        cutlass.BFloat16,
                        peer_address_table[peer],
                        cute.AddressSpace.gmem,
                        assumed_align=32,
                    )
                    self._pull_rows_predicated(
                        pull_tiled_copy,
                        pull_thr_copy,
                        peer_ptr,
                        dy_mkl.iterator,
                        rank_base_row_i64,
                        chunk_first_token,
                        logical_m,
                        peer,
                        k_total,
                        shard_k,
                        pull_rows_per_ticket,
                    )

                pull_m_tile = chunk_id // tickets_per_m_tile
                if pending_ready_count == cutlass.Int32(0):
                    pending_ready_m_tile = pull_m_tile
                pending_ready_count += cutlass.Int32(1)

                next_chunk_id = cutlass.Int32(num_pull_tickets)
                if lane_idx == 0:
                    next_chunk_id = cute.arch.atomic_add(
                        ptr=cursor_ptr.llvm_ptr,
                        val=cutlass.Int32(1),
                        sem="relaxed",
                        scope="gpu",
                    )
                next_chunk_id = cute.arch.shuffle_sync(next_chunk_id, offset=0)
                next_chunk_id = (
                    next_chunk_id * cutlass.Int32(PULL_CURSOR_SHARDS) + cursor_shard
                )

                should_publish_ready = next_chunk_id >= num_pull_tickets
                if next_chunk_id < num_pull_tickets:
                    next_m_tile = next_chunk_id // tickets_per_m_tile
                    if next_m_tile != pending_ready_m_tile:
                        should_publish_ready = cutlass.Boolean(1)

                if should_publish_ready:
                    cute.arch.fence_proxy("async.global")
                    cute.arch.fence_acq_rel_gpu()
                    cute.arch.sync_warp()
                    if lane_idx == 0:
                        ready_linear_idx = epoch * num_m_tiles + pending_ready_m_tile
                        ready_ptr = ready_counts.iterator + ready_linear_idx
                        cute.arch.atomic_add(
                            ptr=ready_ptr.llvm_ptr,
                            val=pending_ready_count,
                            sem="release",
                            scope="gpu",
                        )
                    cute.arch.sync_warp()
                    pending_ready_count = cutlass.Int32(0)

                chunk_id = next_chunk_id

            cursor_shard = (cursor_shard + cutlass.Int32(1)) % cutlass.Int32(
                PULL_CURSOR_SHARDS
            )
            cursor_shards_remaining -= cutlass.Int32(1)

    @cute.jit
    def _partition_bias(
        self,
        bias_n: cute.Tensor,
        bias_tile_n: cutlass.Int32,
        tiled_copy_t2r: cute.TiledCopy,
        tiled_copy_r2s: cute.TiledCopy,
        epi_tidx: cutlass.Int32,
        epi_tile: cute.Tile,
    ) -> cute.Tensor:
        """The tile's bias broadcast along M, partitioned like the epilogue
        subtiles: element i of subtile s belongs to element i of the retiled
        accumulator fragment of subtile s."""
        tile_m, tile_n = self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1]
        gBias = cute.local_tile(bias_n, (tile_n,), (bias_tile_n,))
        gBias_mn = cute.make_tensor(
            gBias.iterator, cute.make_layout((tile_m, tile_n), stride=(0, 1))
        )
        tDgBias = tiled_copy_t2r.get_slice(epi_tidx).partition_D(
            cute.flat_divide(gBias_mn, epi_tile)
        )
        tDgBias = tiled_copy_r2s.retile(tDgBias)
        return cute.group_modes(tDgBias, 3, cute.rank(tDgBias))

    @cute.jit
    def _add_bias(self, tRS_rAcc: cute.Tensor, tDgBias: cute.Tensor) -> None:
        """``tRS_rAcc += bias`` in FP32, before the single rounding to BF16."""
        tDrBias_in = cute.make_rmem_tensor(tDgBias.layout, tDgBias.element_type)
        cute.autovec_copy(cute.filter_zeros(tDgBias), cute.filter_zeros(tDrBias_in))
        tDrBias = cute.make_rmem_tensor(tDgBias.layout, self.acc_dtype)
        tDrBias.store(tDrBias_in.load().to(self.acc_dtype))
        for i in cutlass.range(cute.size(tDrBias), unroll_full=True):
            tRS_rAcc[i] = tRS_rAcc[i] + tDrBias[i]

    @cute.jit
    def epilogue_tma_store_single_cstage(
        self,
        epi_tidx: cutlass.Int32,
        warp_idx: cutlass.Int32,
        tma_atom_c: cute.CopyAtom,
        tCtAcc_base: cute.Tensor,
        sC: cute.Tensor,
        tCgC_base: cute.Tensor,
        bias_n: Optional[cute.Tensor],
        bias_tile_n: cutlass.Int32,
        epi_tile: cute.Tile,
        epilogue_op: cutlass.Constexpr,
        mma_tile_coord_mnl: Tuple[cutlass.Int32, cutlass.Int32, int],
        acc_consumer_state,
        acc_pipeline: pipeline.PipelineAsync,
        c_pipeline: pipeline.PipelineTmaStore,
    ):
        tCgC = utils.gemm.sm100.transform_partitioned_tensor_layout(tCgC_base)
        tCtAcc = utils.gemm.sm100.transform_partitioned_tensor_layout(tCtAcc_base)
        tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = (
            utils.gemm.sm100.epilogue_tmem_copy_and_partition(
                self,
                epi_tidx,
                tCtAcc,
                tCgC,
                epi_tile,
                self.use_2cta_instrs,
            )
        )
        tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
        tiled_copy_r2s, tRS_rC, tRS_sC = (
            utils.gemm.sm100.epilogue_smem_copy_and_partition(
                self,
                tiled_copy_t2r,
                tTR_rC,
                epi_tidx,
                sC,
            )
        )
        tCgC_epi = cute.flat_divide(tCgC, epi_tile)
        bSG_sC, bSG_gC_partitioned = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            cute.group_modes(sC, 0, 2),
            cute.group_modes(tCgC_epi, 0, 2),
        )
        epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=self.epilog_sync_bar_id,
            num_threads=32 * len(self.epilogue_warp_id),
        )
        bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]
        tTR_tAcc = tTR_tAcc_base[
            (None, None, None, None, None, acc_consumer_state.index)
        ]

        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
        tDgBias = None
        if cutlass.const_expr(bias_n is not None):
            tDgBias = self._partition_bias(
                bias_n, bias_tile_n, tiled_copy_t2r, tiled_copy_r2s, epi_tidx, epi_tile
            )

        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
        for subtile_idx in range(subtile_cnt):
            tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

            tRS_rAcc = tiled_copy_r2s.retile(tTR_rAcc)
            if cutlass.const_expr(bias_n is not None):
                self._add_bias(tRS_rAcc, tDgBias[(None, None, None, subtile_idx)])
            acc_vec = epilogue_op(tRS_rAcc.load().to(self.c_dtype))
            tRS_rC.store(acc_vec)

            # The single C staging buffer is shared by every subtile of every
            # tile this CTA processes. Before overwriting it, the warp that
            # issued the previous TMA store (previous subtile of this tile, or
            # the last subtile of the previous tile) must wait until that
            # store has finished reading SMEM, and the other epilogue warps
            # must observe that wait. Acquiring here (after TMEM->RF) rather
            # than right after the store keeps the accumulator read overlapped
            # with the in-flight store and, crucially, also covers the tile
            # boundary: a post-store acquire that skips the last subtile lets
            # the next tile's first RF->SMEM write race the previous tile's
            # final bulk store whenever the MMA runs ahead of the epilogue
            # (e.g. under SM contention), corrupting the row suffix of the
            # tile's last 16 columns.
            if warp_idx == self.epilogue_warp_id[0]:
                c_pipeline.producer_acquire()
            epilog_sync_barrier.arrive_and_wait()

            cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, 0)])
            cute.arch.fence_proxy("async.shared", space="cta")
            epilog_sync_barrier.arrive_and_wait()

            if warp_idx == self.epilogue_warp_id[0]:
                cute.copy(
                    tma_atom_c,
                    bSG_sC[(None, 0)],
                    bSG_gC[(None, subtile_idx)],
                )
                c_pipeline.producer_commit()

        with cute.arch.elect_one():
            acc_pipeline.consumer_release(acc_consumer_state)
        acc_consumer_state.advance()
        return acc_consumer_state

    @cute.jit
    def epilogue_tma_store_skip_tail_barrier(
        self,
        epi_tidx: cutlass.Int32,
        warp_idx: cutlass.Int32,
        tma_atom_c: cute.CopyAtom,
        tCtAcc_base: cute.Tensor,
        sC: cute.Tensor,
        tCgC_base: cute.Tensor,
        bias_n: Optional[cute.Tensor],
        bias_tile_n: cutlass.Int32,
        epi_tile: cute.Tile,
        epilogue_op: cutlass.Constexpr,
        mma_tile_coord_mnl: Tuple[cutlass.Int32, cutlass.Int32, int],
        acc_consumer_state,
        acc_pipeline: pipeline.PipelineAsync,
        c_pipeline: pipeline.PipelineTmaStore,
    ):
        tCgC = utils.gemm.sm100.transform_partitioned_tensor_layout(tCgC_base)
        tCtAcc = utils.gemm.sm100.transform_partitioned_tensor_layout(tCtAcc_base)
        tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = (
            utils.gemm.sm100.epilogue_tmem_copy_and_partition(
                self,
                epi_tidx,
                tCtAcc,
                tCgC,
                epi_tile,
                self.use_2cta_instrs,
            )
        )
        tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
        tiled_copy_r2s, tRS_rC, tRS_sC = (
            utils.gemm.sm100.epilogue_smem_copy_and_partition(
                self,
                tiled_copy_t2r,
                tTR_rC,
                epi_tidx,
                sC,
            )
        )
        tCgC_epi = cute.flat_divide(tCgC, epi_tile)
        bSG_sC, bSG_gC_partitioned = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            cute.group_modes(sC, 0, 2),
            cute.group_modes(tCgC_epi, 0, 2),
        )
        epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=self.epilog_sync_bar_id,
            num_threads=32 * len(self.epilogue_warp_id),
        )
        bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]
        tTR_tAcc = tTR_tAcc_base[
            (None, None, None, None, None, acc_consumer_state.index)
        ]

        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
        tDgBias = None
        if cutlass.const_expr(bias_n is not None):
            tDgBias = self._partition_bias(
                bias_n, bias_tile_n, tiled_copy_t2r, tiled_copy_r2s, epi_tidx, epi_tile
            )

        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
        for subtile_idx in range(subtile_cnt):
            tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

            tRS_rAcc = tiled_copy_r2s.retile(tTR_rAcc)
            if cutlass.const_expr(bias_n is not None):
                self._add_bias(tRS_rAcc, tDgBias[(None, None, None, subtile_idx)])
            acc_vec = epilogue_op(tRS_rAcc.load().to(self.c_dtype))
            tRS_rC.store(acc_vec)

            c_buffer = cutlass.Int32(0)
            if cutlass.const_expr(self.num_c_stage > 1):
                c_buffer = subtile_idx % self.num_c_stage

            # Same tile-boundary hazard as the single-stage variant: c_buffer
            # restarts at 0 for every tile while the previous tile's store from
            # buffer 0 may still be reading SMEM. Wait (wait_group.read
            # num_c_stage-1) + barrier before every RF->SMEM write instead of
            # after the store so the boundary is covered as well.
            if warp_idx == self.epilogue_warp_id[0]:
                c_pipeline.producer_acquire()
            epilog_sync_barrier.arrive_and_wait()

            cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
            cute.arch.fence_proxy("async.shared", space="cta")
            epilog_sync_barrier.arrive_and_wait()

            if warp_idx == self.epilogue_warp_id[0]:
                cute.copy(
                    tma_atom_c,
                    bSG_sC[(None, c_buffer)],
                    bSG_gC[(None, subtile_idx)],
                )
                c_pipeline.producer_commit()

        with cute.arch.elect_one():
            acc_pipeline.consumer_release(acc_consumer_state)
        acc_consumer_state.advance()
        return acc_consumer_state

    @cute.jit
    def epilogue_direct_store(
        self,
        epi_tidx: cutlass.Int32,
        acc_pipeline: pipeline.PipelineAsync,
        acc_consumer_state,
        tiled_mma: cute.TiledMma,
        tCtAcc_base: cute.Tensor,
        tCgC: cute.Tensor,
        epi_tile: cute.Tile,
        mma_tile_coord_mnl: Tuple[cutlass.Int32, cutlass.Int32, int],
        epilogue_op: cutlass.Constexpr,
    ):
        tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = self.epilog_tmem_copy_and_partition(
            epi_tidx,
            tCtAcc_base,
            tCgC,
            epi_tile,
            False,
        )
        gC_epi = cute.flat_divide(
            tCgC[((None, None), 0, 0, None, None, None)], epi_tile
        )
        thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
        tTR_gC_partitioned = thr_copy_t2r.partition_D(gC_epi)
        tTR_rC = cute.make_rmem_tensor(
            tTR_gC_partitioned[(None, None, None, 0, 0, 0, 0, 0)].shape,
            self.c_dtype,
        )
        simt_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.c_dtype, num_bits_per_copy=128
        )

        tTR_gC = tTR_gC_partitioned[(None, None, None, None, None, *mma_tile_coord_mnl)]
        tTR_tAcc = tTR_tAcc_base[
            (None, None, None, None, None, acc_consumer_state.index)
        ]
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))

        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
        for subtile_idx in cutlass.range(subtile_cnt):
            tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
            acc_vec = epilogue_op(tTR_rAcc.load().to(self.c_dtype))
            tTR_rC.store(acc_vec)
            cute.copy(
                simt_atom,
                tTR_rC,
                tTR_gC[(None, None, None, subtile_idx)],
            )

        with cute.arch.elect_one():
            acc_pipeline.consumer_release(acc_consumer_state)
        acc_consumer_state.advance()
        return acc_consumer_state

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: bool,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        copy_atom_t2r = utils.sm100.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0, None)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r,
            tAcc_epi[(None, None, 0, 0, 0)],
        )
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape,
            self.acc_dtype,
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc


@dataclass
class CompiledHeadShardPullGemm:
    compiled: object
    peer_address_table_torch: torch.Tensor
    pull_cursors_torch: torch.Tensor
    ready_counts_torch: torch.Tensor
    workspace_ready_event: torch.cuda.Event
    weight_t_view: torch.Tensor
    out_3d_view: torch.Tensor
    dy_3d_view: torch.Tensor
    peer_arg: object
    pull_cursor_arg: object
    ready_arg: object
    weight_arg: object
    out_arg: object
    dy_arg: object
    bias_arg: object
    logical_m: int
    m_rows: int
    n: int
    k: int
    rank: int
    up: int
    tile_m: int
    tile_n: int
    pull_chunk_m: int
    num_ab_stages: int
    num_c_stages: int
    max_active_ctas: int
    use_tma_store: bool
    enable_setmaxnreg: bool
    workspace_epochs: int = DEFAULT_WORKSPACE_EPOCHS
    epoch: int = 0
    peer_rows: int = 0
    enable_clc: bool = False
    enable_bulk_pull: bool = False
    pull_rows_per_ticket: int = 0
    pull_col_chunks: int = 1
    bias_torch: torch.Tensor | None = None

    def reset_workspace(
        self,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Recycle per-epoch queues after the prior launch has completed."""

        if stream is None:
            stream = torch.cuda.current_stream(self.out_3d_view.device)
        with torch.cuda.stream(stream):
            self.pull_cursors_torch.zero_()
            self.ready_counts_torch.zero_()
            self.workspace_ready_event.record(stream)
        self.epoch = 0

    def __call__(self, stream: torch.cuda.Stream | None = None) -> None:
        if self.epoch >= self.workspace_epochs:
            raise RuntimeError(
                f"single-kernel workspace exhausted after {self.workspace_epochs} calls"
            )
        if stream is None:
            stream = torch.cuda.current_stream(self.out_3d_view.device)
        stream.wait_event(self.workspace_ready_event)
        self.compiled(
            self.peer_arg,
            self.pull_cursor_arg,
            self.ready_arg,
            self.weight_arg,
            self.out_arg,
            self.dy_arg,
            self.bias_arg,
            cutlass.Int32(self.logical_m),
            cutlass.Int32(self.peer_rows),
            cutlass.Int32(self.epoch),
            stream=CUstream(stream.cuda_stream),
        )
        self.epoch += 1


def _check_bf16_cuda_contiguous(name: str, tensor: torch.Tensor) -> None:
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be CUDA, got {tensor.device}")
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be torch.bfloat16, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def compile_headshard_pull_gemm(
    peer_address_table: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    dy: torch.Tensor,
    *,
    logical_m: int,
    rank: int,
    up: int,
    hidden: int | None = None,
    bias: torch.Tensor | None = None,
    tile_m: int = DEFAULT_TILE_M,
    tile_n: int = DEFAULT_TILE_N,
    tile_k: int = DEFAULT_TILE_K,
    pull_chunk_m: int = DEFAULT_PULL_CHUNK_M,
    num_ab_stages: int = DEFAULT_AB_STAGES,
    num_c_stages: int = DEFAULT_C_STAGES,
    max_active_ctas: int = 0,
    use_tma_store: bool = True,
    enable_setmaxnreg: bool = True,
    workspace_epochs: int = DEFAULT_WORKSPACE_EPOCHS,
    stream: torch.cuda.Stream | None = None,
    peer_rows: int | None = None,
    tma_python_k_loop: bool | None = None,
    enable_bulk_pull: bool = DEFAULT_ENABLE_BULK_PULL,
    enable_clc: bool = True,
    clc_sched_stages: int = DEFAULT_CLC_SCHED_STAGES,
) -> CompiledHeadShardPullGemm:
    """Compile (or fetch from cache) a warp-specialized single-kernel config.

    ``logical_m`` is a runtime value of the compiled kernel: calling this
    function again with a different M but identical static configuration
    reuses the cached compiled kernel and only rebinds tensors, so per-step
    sequence-length changes never trigger recompilation.

    ``peer_rows`` is the token capacity (row count per destination rank) of
    the symmetric source buffers; it defaults to ``logical_m`` and lets
    callers allocate peer buffers once at the maximum M.

    ``out`` and ``dy`` use the real ``logical_m`` rows. Tail M tiles rely on
    TMA out-of-bounds handling and dynamic pull-ready counts. No padding is
    allowed in M, N, or K at the call site.

    ``weight`` follows the standard ``nn.Linear`` layout ``[N, K]`` and the
    kernel computes ``dy @ weight.T`` without materializing a transposed copy:
    ``dy`` is the pulled ``[M, K]`` input and ``out`` the ``[M, N]`` output.
    An optional ``(N,)`` BF16 ``bias`` is added to the FP32 accumulator before
    the single rounding to BF16, as ``torch.nn.functional.linear`` does.

    ``hidden`` is optional and only checked: callers that pass the pulled
    width explicitly must pass K.
    """
    global _KERNEL_COMPILE_COUNT

    if up not in (2, 4, 8):
        raise ValueError(f"up must be one of (2, 4, 8), got {up}")
    if not (0 <= rank < up):
        raise ValueError(f"rank must be in [0, {up}), got {rank}")
    if logical_m <= 0:
        raise ValueError(f"logical_m must be positive, got {logical_m}")
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-D [N, K], got {tuple(weight.shape)}")
    n, k = int(weight.shape[0]), int(weight.shape[1])
    if hidden is not None and hidden != k:
        raise ValueError(
            f"hidden={hidden} must equal the pulled width K={k} of the "
            f"[N, K] weight {tuple(weight.shape)}"
        )
    if k <= 0 or k % up != 0:
        raise ValueError(f"K={k} must be positive and divisible by up={up}")
    if peer_rows is None:
        peer_rows = logical_m
    if peer_rows < logical_m:
        raise ValueError(f"peer_rows={peer_rows} must be >= logical_m={logical_m}")
    shard_k = k // up
    if shard_k % PULL_DIVBY != 0 or k % PULL_DIVBY != 0:
        raise ValueError(
            "the BF16 pull requires K and K/up to be divisible "
            f"by {PULL_DIVBY} elements, got K={k}, shard_k={shard_k}"
        )
    if k % tile_k != 0:
        raise ValueError(f"K={k} must be divisible by tile_k={tile_k}")

    if tile_m not in (64, 128):
        raise ValueError(f"1-CTA tile_m must be 64 or 128, got {tile_m}")
    if tile_n < 32 or tile_n > 256 or tile_n % 32 != 0:
        raise ValueError(f"tile_n must be a multiple of 32 in [32, 256], got {tile_n}")
    if n <= 0 or n % tile_n != 0:
        raise ValueError(
            f"this implementation requires N={n} divisible by tile_n={tile_n}"
        )
    if pull_chunk_m <= 0 or tile_m % pull_chunk_m != 0:
        raise ValueError(f"pull_chunk_m={pull_chunk_m} must divide tile_m={tile_m}")
    pull_rows_per_ticket = _effective_pull_rows_per_ticket(pull_chunk_m, up)
    if pull_rows_per_ticket <= 0 or tile_m % pull_rows_per_ticket != 0:
        raise ValueError(
            "effective pull rows per ticket must divide tile_m, got "
            f"pull_rows_per_ticket={pull_rows_per_ticket}, tile_m={tile_m}"
        )
    if pull_rows_per_ticket % 4 != 0:
        raise ValueError(
            "full-token pull fast path expects effective pull rows per ticket "
            f"to be a multiple of 4, got {pull_rows_per_ticket}"
        )
    bulk_pull_eligible = bool(
        k in BULK_PULL_TUNED_ROW_ELEMENTS
        and (shard_k * 2) % 16 == 0
        and pull_rows_per_ticket >= BULK_PULL_STAGES
    )
    enable_bulk_pull = bool(enable_bulk_pull and bulk_pull_eligible)
    if shard_k % (WARP_SIZE * PULL_MIN_ELEMENTS_PER_LANE) != 0:
        raise ValueError(
            "vectorized BF16 peer pull expects K/up to be divisible by "
            f"{WARP_SIZE * PULL_MIN_ELEMENTS_PER_LANE} elements, got shard_k={shard_k}"
        )
    pull_col_chunks = 1
    pull_rows_in_flight = pull_rows_per_ticket
    if not enable_bulk_pull:
        pull_rows_per_ticket = _vector_pull_rows_per_ticket(pull_rows_per_ticket, shard_k)
        pull_rows_in_flight = pull_rows_per_ticket
        pull_col_chunks = _pull_column_chunks(shard_k, pull_rows_in_flight)
    pull_elements_per_lane = _select_pull_elements_per_lane(shard_k // pull_col_chunks)
    if num_ab_stages != 0 and num_ab_stages < 2:
        raise ValueError("num_ab_stages must be 0(auto) or >= 2")
    if num_c_stages < 1:
        raise ValueError("num_c_stages must be >= 1")
    if workspace_epochs <= 0:
        raise ValueError("workspace_epochs must be positive")
    if enable_clc and clc_sched_stages <= 0:
        raise ValueError(
            f"clc_sched_stages must be positive when CLC is enabled, got {clc_sched_stages}"
        )

    if peer_address_table.device.type != "cuda":
        raise ValueError("peer_address_table must be CUDA")
    if peer_address_table.dtype != torch.uint64:
        raise ValueError(
            f"peer_address_table must be torch.uint64, got {peer_address_table.dtype}"
        )
    if tuple(peer_address_table.shape) != (up,):
        raise ValueError(
            f"peer_address_table.shape must be {(up,)}, got {tuple(peer_address_table.shape)}"
        )
    if not peer_address_table.is_contiguous():
        raise ValueError("peer_address_table must be contiguous")

    _check_bf16_cuda_contiguous("weight", weight)
    _check_bf16_cuda_contiguous("out", out)
    _check_bf16_cuda_contiguous("dy", dy)

    if tuple(out.shape) != (logical_m, n):
        raise ValueError(
            f"out.shape must be {(logical_m, n)}, got {tuple(out.shape)}"
        )
    if tuple(dy.shape) != (logical_m, k):
        raise ValueError(
            f"dy.shape must be {(logical_m, k)}, got {tuple(dy.shape)}"
        )
    if not (peer_address_table.device == weight.device == out.device == dy.device):
        raise ValueError("all tensors must be on the same CUDA device")
    if bias is not None:
        _check_bf16_cuda_contiguous("bias", bias)
        if tuple(bias.shape) != (n,) or bias.device != out.device:
            raise ValueError(
                f"bias must have shape {(n,)} on {out.device}, got "
                f"{tuple(bias.shape)} on {bias.device}"
            )
        if bias.data_ptr() % 16:
            raise ValueError("bias must be 16-byte aligned")

    if tma_python_k_loop is None:
        tma_python_k_loop = up == 4

    num_m_tiles = _ceil_div(logical_m, tile_m)
    if max_active_ctas <= 0:
        max_active_ctas = torch.cuda.get_device_properties(
            out.device
        ).multi_processor_count
    pull_cursors = torch.zeros(
        (workspace_epochs * PULL_CURSOR_STRIDE,),
        dtype=torch.int32,
        device=out.device,
    )
    # Flat dynamic workspace: one Int32 per (epoch, m_tile).
    ready_counts = torch.zeros(
        (workspace_epochs * num_m_tiles,), dtype=torch.int32, device=out.device
    )
    workspace_ready_event = torch.cuda.Event()
    workspace_ready_event.record(torch.cuda.current_stream(out.device))

    # CuTe dense GEMM convention:
    # A = [M,K,L], B = [N,K,L], C = [M,N,L].
    dy_3d_view = dy.unsqueeze(-1)
    weight_t_view = weight.unsqueeze(-1)
    out_3d_view = out.unsqueeze(-1)

    peer_arg = from_dlpack(peer_address_table, assumed_align=8)
    # Both workspaces are dynamic-length so the epoch count is not compiled in.
    pull_cursor_arg = from_dlpack(pull_cursors, assumed_align=4).mark_layout_dynamic(
        leading_dim=0
    )
    ready_arg = from_dlpack(ready_counts, assumed_align=4).mark_layout_dynamic(
        leading_dim=0
    )
    weight_arg = from_dlpack(weight_t_view, assumed_align=16)
    # M is dynamic so a single compiled kernel serves every sequence length.
    out_arg = from_dlpack(out_3d_view, assumed_align=16).mark_compact_shape_dynamic(
        mode=0, stride_order=out_3d_view.dim_order(), divisibility=1
    )
    # 32-byte base alignment is required by the 256-bit pull stores into dy.
    dy_arg = from_dlpack(dy_3d_view, assumed_align=32).mark_compact_shape_dynamic(
        mode=0, stride_order=dy_3d_view.dim_order(), divisibility=1
    )
    bias_arg = None if bias is None else from_dlpack(bias, assumed_align=16)

    if stream is None:
        stream = torch.cuda.current_stream(out.device)

    op = HeadShardPullGemmKernel(
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        num_ab_stages=num_ab_stages,
        num_c_stages=num_c_stages,
        max_active_ctas=max_active_ctas,
        use_tma_store=use_tma_store,
        enable_setmaxnreg=enable_setmaxnreg,
        tma_python_k_loop=tma_python_k_loop,
        pull_elements_per_lane=pull_elements_per_lane,
        pull_col_chunks=pull_col_chunks,
        pull_rows_in_flight=pull_rows_in_flight,
        enable_bulk_pull=enable_bulk_pull,
        bulk_row_elements=k,
        enable_clc=enable_clc,
        clc_sched_stages=clc_sched_stages,
    )
    if not op.use_tma_store and logical_m % tile_m != 0:
        raise ValueError(
            "non-divisible logical_m requires TMA-store tail OOB handling; "
            "enable use_tma_store or use an M divisible by tile_m"
        )
    if not op.use_tma_store and (n // tile_n) % 2 != 0:
        raise ValueError(
            "an odd number of N tiles requires TMA-store OOB handling; "
            f"got N/tile_n={n // tile_n}"
        )
    if not op.use_tma_store and bias is not None:
        raise ValueError("a bias requires the TMA-store epilogue")

    cache_key = (
        IMPLEMENTATION_REVISION,
        rank,
        up,
        n,
        k,
        tile_m,
        tile_n,
        tile_k,
        pull_chunk_m,
        pull_rows_per_ticket,
        pull_elements_per_lane,
        pull_col_chunks,
        num_ab_stages,
        num_c_stages,
        max_active_ctas,
        op.use_tma_store,
        enable_setmaxnreg,
        tma_python_k_loop,
        enable_bulk_pull,
        enable_clc,
        clc_sched_stages if enable_clc else 0,
        bias is not None,
        str(out.device),
    )
    cached = _KERNEL_CACHE.get(cache_key)
    if cached is None:
        compiled = cute.compile(
            op,
            peer_arg,
            pull_cursor_arg,
            ready_arg,
            weight_arg,
            out_arg,
            dy_arg,
            bias_arg,
            rank,
            up,
            pull_chunk_m,
            pull_rows_per_ticket,
            cutlass.Int32(logical_m),
            cutlass.Int32(peer_rows),
            cutlass.Int32(0),
            stream=CUstream(stream.cuda_stream),
            options="--generate-line-info",
        )
        _KERNEL_COMPILE_COUNT += 1
        cached = (compiled, op.num_ab_stage, op.num_c_stage, op.use_tma_store)
        _KERNEL_CACHE[cache_key] = cached
    compiled, num_ab_stage, num_c_stage, use_tma_store_resolved = cached

    return CompiledHeadShardPullGemm(
        compiled=compiled,
        peer_address_table_torch=peer_address_table,
        pull_cursors_torch=pull_cursors,
        ready_counts_torch=ready_counts,
        workspace_ready_event=workspace_ready_event,
        weight_t_view=weight_t_view,
        out_3d_view=out_3d_view,
        dy_3d_view=dy_3d_view,
        peer_arg=peer_arg,
        pull_cursor_arg=pull_cursor_arg,
        ready_arg=ready_arg,
        weight_arg=weight_arg,
        out_arg=out_arg,
        dy_arg=dy_arg,
        bias_arg=bias_arg,
        logical_m=logical_m,
        m_rows=logical_m,
        n=n,
        k=k,
        rank=rank,
        up=up,
        tile_m=tile_m,
        tile_n=tile_n,
        pull_chunk_m=pull_chunk_m,
        num_ab_stages=num_ab_stage,
        num_c_stages=num_c_stage,
        max_active_ctas=max_active_ctas,
        use_tma_store=use_tma_store_resolved,
        enable_setmaxnreg=enable_setmaxnreg,
        workspace_epochs=workspace_epochs,
        peer_rows=peer_rows,
        enable_clc=enable_clc,
        enable_bulk_pull=enable_bulk_pull,
        pull_rows_per_ticket=pull_rows_per_ticket,
        pull_col_chunks=pull_col_chunks,
        bias_torch=bias,
    )
