# Copyright (c) 2025-2026, QuACK team.
# Copyright (c) 2026 The Mega-UP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Derived from QuACK 0.4.0's GEMM, which is based on NVIDIA CUTLASS CuTe
# DSL's persistent dense GEMM example. Modified to prune unused paths and add
# dynamic tile scheduling plus a peer-routed epilogue.

"""Standalone BF16 persistent peer-write GEMM."""

import enum
import math
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, fields
from enum import IntEnum
from functools import cache, partial
from threading import RLock
from typing import Callable, Dict, NamedTuple, Optional, Tuple, Type, Union, get_origin

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as gemm_helpers
import torch
from cutlass import BFloat16, Boolean, Float32, Int32, Int64, const_expr
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.base_dsl.tvm_ffi_builder import spec
from cutlass.base_dsl.typing import Integer
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import NumericMeta, T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import LayoutEnum
from torch import Tensor

IMPLEMENTATION_REVISION = "peer-write-v5-128-byte-peer-rows"

# Tile configuration of the host APIs: 256x256 MMA tiles on 2-CTA clusters.
# Output columns are routed to peers per epilogue subtile, so a head shard
# does not have to be a multiple of the N tile; it only has to be a multiple
# of the subtile width.
PEER_TILE_M = 256
PEER_TILE_N = 256
PEER_CLUSTER_M = 2
PEER_CLUSTER_N = 1
PEER_EPI_TILE_N = 64


# CUTLASS DSL 4.4.2 TVM-FFI and MLIR reconstruction compatibility.
StaticTypes = (cutlass.Constexpr, NumericMeta, int, bool, str, float, type(None))

import cutlass.cute._tvm_ffi_args_spec_converter as _converter_module  # noqa: E402

_converter_patch_lock = RLock()


def _patched_convert_single_arg(delegate, arg, arg_name, arg_type, ctx):
    if arg_type is not None and get_origin(arg_type) is cutlass.Constexpr:
        return spec.ConstNone(arg_name)
    if (
        isinstance(arg, tuple)
        and hasattr(type(arg), "_fields")
        and (arg_type is None or not hasattr(arg_type, "_fields"))
    ):
        return delegate(arg, arg_name, type(arg), ctx)
    return delegate(arg, arg_name, arg_type, ctx)


@contextmanager
def _cutlass_442_converter_compatibility():
    """Install the CUTLASS 4.4.2 workaround only while compiling this kernel."""

    with _converter_patch_lock:
        previous = _converter_module._convert_single_arg
        _converter_module._convert_single_arg = partial(
            _patched_convert_single_arg,
            previous,
        )
        try:
            yield
        finally:
            _converter_module._convert_single_arg = previous


def _partition_fields(obj):
    all_fields = {field.name: getattr(obj, field.name) for field in fields(obj)}
    constexpr = {
        name: value
        for name, value in all_fields.items()
        if isinstance(value, StaticTypes)
    }
    dynamic = {
        name: value
        for name, value in all_fields.items()
        if not isinstance(value, StaticTypes)
    }
    return constexpr, dynamic


def _new_from_mlir_values(self, values):
    constexpr_fields, dynamic_fields = _partition_fields(self)
    for (name, value), n_items in zip(dynamic_fields.items(), self._values_pos):
        dynamic_fields[name] = cutlass.new_from_mlir_values(value, values[:n_items])
        values = values[n_items:]
    return self.__class__(**dynamic_fields, **constexpr_fields)


def _namedtuple_new_from_mlir_values(self, values):
    from cutlass.base_dsl.typing import get_mlir_types

    values = list(values)
    new_fields = []
    for field_value in self:
        if field_value is None or isinstance(field_value, StaticTypes):
            new_fields.append(field_value)
        else:
            n_items = len(get_mlir_types(field_value))
            new_fields.append(
                cutlass.new_from_mlir_values(field_value, values[:n_items])
            )
            values = values[n_items:]
    return self.__class__(*new_fields)


def mlir_namedtuple(cls):
    cls.__new_from_mlir_values__ = _namedtuple_new_from_mlir_values
    return cls


@dataclass
class ParamsBase:
    def __extract_mlir_values__(self):
        _, dynamic_fields = _partition_fields(self)
        values, self._values_pos = [], []
        for obj in dynamic_fields.values():
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    __new_from_mlir_values__ = _new_from_mlir_values


# Host helpers.
torch2cute_dtype_map = {torch.bfloat16: BFloat16}


@cache
def get_max_active_clusters(
    cluster_size: int,
    device_capacity: Tuple[int, int] | None = None,
    device_id: int = 0,
) -> int:
    if device_capacity is None:
        device_capacity = get_device_capacity()
    if device_capacity[0] not in (10, 11):
        raise ValueError(
            f"active-cluster query does not support device capability {device_capacity}"
        )
    return cutlass.utils.HardwareInfo(device_id).get_max_active_clusters(
        cluster_size=cluster_size
    )


def _parse_arch_str(arch_str: str) -> Tuple[int, int]:
    match = re.match(r"^(?:sm_?)?(\d+)(\d)([af]?)$", arch_str.strip(), re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid MEGA_UP_ARCH: {arch_str!r}")
    major, minor, _ = match.groups()
    return int(major), int(minor)


@cache
def _get_device_capacity_cached(device: torch.device | None = None) -> Tuple[int, int]:
    arch_override = os.environ.get("MEGA_UP_ARCH")
    if arch_override is not None:
        return _parse_arch_str(arch_override)
    return torch.cuda.get_device_capability(device)


def get_device_capacity(
    device: torch.device | torch.Tensor | None = None,
) -> Tuple[int, int]:
    if isinstance(device, torch.Tensor):
        device = device.device
    return _get_device_capacity_cached(device)


def make_fake_tensor(
    dtype, shape, divisibility=1, leading_dim=-1
) -> Optional[cute.Tensor]:
    if leading_dim < 0:
        leading_dim = len(shape) + leading_dim
    if dtype is None:
        return None
    stride = tuple(
        cute.sym_int64(divisibility=divisibility) if i != leading_dim else 1
        for i in range(len(shape))
    )
    return cute.runtime.make_fake_tensor(
        dtype, shape, stride=stride, assumed_align=divisibility * dtype.width // 8
    )


def div_for_dtype(dtype):
    return 128 // dtype.width


def perm3d(A, B, D):
    def _perm(t):
        return t.permute(1, 2, 0) if t is not None and t.ndim == 3 else t

    return _perm(A), _perm(B), _perm(D)


def get_major(t, dim0, dim1):
    return dim1 if t.stride(1) == 1 else dim0


def _is_non_overlapping_dense(tensor: Tensor) -> bool:
    expected_stride = 1
    dimensions = sorted(
        (int(stride), int(size))
        for size, stride in zip(tensor.shape, tensor.stride())
        if size > 1
    )
    for stride, size in dimensions:
        if stride != expected_stride:
            return False
        expected_stride *= size
    return True


def _storage_byte_range(tensor: Tensor) -> tuple[int, int]:
    start = tensor.data_ptr()
    return start, start + tensor.numel() * tensor.element_size()


def _tensors_overlap(left: Tensor, right: Tensor) -> bool:
    left_start, left_end = _storage_byte_range(left)
    right_start, right_end = _storage_byte_range(right)
    return left_start < right_end and right_start < left_end


def make_fake_gemm_tensors(
    a_dtype,
    b_dtype,
    d_dtype,
    a_major,
    b_major,
    d_major,
):
    a_leading = 1 if a_major == "k" else 0
    b_leading = 1 if b_major == "k" else 0
    d_leading = 1 if d_major == "n" else 0
    m, n, k, l = cute.sym_int(), cute.sym_int(), cute.sym_int(), cute.sym_int()
    mA = make_fake_tensor(
        a_dtype, (m, k, l), leading_dim=a_leading, divisibility=div_for_dtype(a_dtype)
    )
    mB = make_fake_tensor(
        b_dtype, (n, k, l), leading_dim=b_leading, divisibility=div_for_dtype(b_dtype)
    )
    mD = make_fake_tensor(
        d_dtype,
        (m, n, l),
        leading_dim=d_leading,
        divisibility=div_for_dtype(d_dtype) if d_dtype is not None else 1,
    )
    return mA, mB, mD, m, n, k, l


class RoundingMode(IntEnum):
    RN = 0


# Small DSL primitives needed by the CLC scheduler and epilogue.
class FastDivmod(cute.FastDivmodDivisor):
    @dsl_user_op
    def __init__(
        self,
        divisor: Integer,
        is_power_of_2: bool = None,
        *,
        loc=None,
        ip=None,
    ):
        super().__init__(divisor, is_power_of_2=is_power_of_2, loc=loc, ip=ip)
        self.divisor = divisor

    def __extract_mlir_values__(self):
        return [self._divisor] + cutlass.extract_mlir_values(self.divisor)

    def __new_from_mlir_values__(self, values):
        new_obj = object.__new__(FastDivmod)
        new_obj._divisor = values[0]
        new_obj.divisor = cutlass.new_from_mlir_values(self.divisor, values[1:])
        return new_obj


@dsl_user_op
def set_block_rank(
    smem_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    # CUTLASS DSL 4.4.2 requires Pointer.toint() before mapa.shared.
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
        )
    )


@dsl_user_op
def store_shared_remote_x4(
    val0: Float32 | Int32,
    val1: Float32 | Int32,
    val2: Float32 | Int32,
    val3: Float32 | Int32,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: cute.typing.Int,
    *,
    loc=None,
    ip=None,
) -> None:
    remote_smem_ptr_i32 = set_block_rank(
        smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    remote_mbar_ptr_i32 = set_block_rank(
        mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    assert isinstance(val0, (Float32, Int32))
    dtype = Float32 if isinstance(val0, Float32) else Int32
    suffix = {Float32: "f32", Int32: "s32"}[dtype]
    constraint = {Float32: "f", Int32: "r"}[dtype]
    llvm.inline_asm(
        None,
        [
            remote_smem_ptr_i32,
            remote_mbar_ptr_i32,
            dtype(val0).ir_value(loc=loc, ip=ip),
            dtype(val1).ir_value(loc=loc, ip=ip),
            dtype(val2).ir_value(loc=loc, ip=ip),
            dtype(val3).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        f".reg .v4 .{suffix} abcd;\n\t"
        f"mov.{suffix} abcd.x, $2;\n\t"
        f"mov.{suffix} abcd.y, $3;\n\t"
        f"mov.{suffix} abcd.z, $4;\n\t"
        f"mov.{suffix} abcd.w, $5;\n\t"
        f"st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.{suffix} "
        "[$0], abcd, [$1];\n\t"
        "}\n",
        f"r,r,{constraint},{constraint},{constraint},{constraint}",
        has_side_effects=True,
        is_align_stack=False,
    )


@dsl_user_op
def issue_clc_query_nomulticast(
    mbar_ptr: cute.Pointer,
    clc_response_ptr: cute.Pointer,
    *,
    loc=None,
    ip=None,
) -> None:
    nvvm.clusterlaunchcontrol_try_cancel(
        clc_response_ptr.llvm_ptr, mbar_ptr.llvm_ptr, loc=loc, ip=ip
    )


@dsl_user_op
def cvt_copy(
    tiled_copy: cute.TiledCopy,
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    pred: Optional[cute.Tensor] = None,
    retile: bool = False,
    loc=None,
    ip=None,
    **kwargs,
) -> None:
    assert isinstance(src.iterator, cute.Pointer)
    assert src.memspace == cute.AddressSpace.rmem
    if const_expr(src.element_type != dst.element_type):
        src_cvt = cute.make_rmem_tensor_like(src, dst.element_type)
        src_cvt.store(src.load().to(dst.element_type))
        src = src_cvt
    if const_expr(retile):
        src = tiled_copy.retile(src)
    cute.copy(tiled_copy, src, dst, pred=pred, loc=loc, ip=ip, **kwargs)


@dsl_user_op
def contiguous(src: cute.Tensor, *, loc=None, ip=None) -> cute.Tensor:
    dst = cute.make_rmem_tensor(src.shape, src.element_type, loc=loc, ip=ip)
    cute.autovec_copy(src, dst, loc=loc, ip=ip)
    return dst


@dsl_user_op
def tma_get_copy_fn(
    atom: cute.CopyAtom,
    cta_coord: cute.Coord,
    cta_layout: cute.Layout,
    src_tensor: cute.Tensor,
    dst_tensor: cute.Tensor,
    filter_zeros: bool = False,
    single_stage: bool = False,
    *,
    loc=None,
    ip=None,
    **kwargs,
) -> Callable:
    src_is_smem = const_expr(
        isinstance(src_tensor.iterator, cute.Pointer)
        and src_tensor.memspace == cute.AddressSpace.smem
    )
    smem_tensor, gmem_tensor = (
        (src_tensor, dst_tensor) if src_is_smem else (dst_tensor, src_tensor)
    )
    group_rank_smem = const_expr(
        cute.rank(smem_tensor) - (1 if not single_stage else 0)
    )
    group_rank_gmem = const_expr(
        cute.rank(gmem_tensor) - (1 if not single_stage else 0)
    )
    s, g = cpasync.tma_partition(
        atom,
        cta_coord,
        cta_layout,
        cute.group_modes(smem_tensor, 0, group_rank_smem),
        cute.group_modes(gmem_tensor, 0, group_rank_gmem),
        loc=loc,
        ip=ip,
    )
    if const_expr(filter_zeros):
        s = cute.filter_zeros(s)
        g = cute.filter_zeros(g)
    src, dst = (s, g) if src_is_smem else (g, s)

    @dsl_user_op
    def copy_tma(src_idx, dst_idx, *, loc=None, ip=None, **new_kwargs):
        cute.copy(
            atom,
            src[None, src_idx],
            dst[None, dst_idx],
            **new_kwargs,
            **kwargs,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def copy_tma_single_stage(*, loc=None, ip=None, **new_kwargs):
        cute.copy(atom, src, dst, **new_kwargs, **kwargs, loc=loc, ip=ip)

    return (copy_tma if const_expr(not single_stage) else copy_tma_single_stage), s, g


# CLC-only tile scheduler. PersistenceMode.CLC deliberately remains value 3.
class RasterOrderOption(IntEnum):
    AlongM = 0
    AlongN = 1
    Heuristic = 2


class RasterOrder(IntEnum):
    AlongM = 0
    AlongN = 1


class PersistenceMode(IntEnum):
    CLC = 3


@cute.jit
def get_raster_order_from_option(
    raster_order_option: RasterOrderOption,
    problem_shape_ncluster_mn: cute.Shape,
    group_size: Int32,
) -> RasterOrder:
    raster_order = (
        RasterOrder.AlongM
        if raster_order_option == RasterOrderOption.AlongM
        else RasterOrder.AlongN
    )
    if raster_order_option == RasterOrderOption.Heuristic:
        problem_blocks_m = cute.round_up(problem_shape_ncluster_mn[0], group_size)
        problem_blocks_n = cute.round_up(problem_shape_ncluster_mn[1], group_size)
        raster_order = (
            RasterOrder.AlongM
            if problem_blocks_n > problem_blocks_m
            else RasterOrder.AlongN
        )
    return raster_order


@mlir_namedtuple
class TileSchedulerOptions(NamedTuple):
    max_active_clusters: Int32
    raster_order: cutlass.Constexpr[RasterOrderOption] = RasterOrderOption.Heuristic
    max_swizzle_size: Int32 = Int32(8)
    # Cyclic offset applied to the N cluster coordinate, < number of N clusters.
    n_rotation: Int32 = Int32(0)


@dataclass
class TileSchedulerArguments:
    problem_shape_ntile_mnl: cute.Shape
    raster_order: cutlass.Constexpr[RasterOrderOption]
    group_size: Int32
    cluster_shape_mnk: cutlass.Constexpr[cute.Shape]
    persistence_mode: cutlass.Constexpr[PersistenceMode] = PersistenceMode.CLC
    n_rotation: Int32 = Int32(0)


class TileScheduler:
    @dataclass
    class Params:
        problem_shape_ncluster_mnl: cute.Shape
        raster_order: RasterOrder
        num_clusters_per_problem_fdd: FastDivmod
        num_groups_regular: Int32
        group_size_fdd: FastDivmod
        group_size_tail_fdd: FastDivmod
        num_clusters_in_group_fdd: FastDivmod
        n_rotation: Int32
        cluster_shape_mnk: cutlass.Constexpr[cute.Shape]
        persistence_mode: cutlass.Constexpr[PersistenceMode]

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "TileScheduler.Params":
            assert args.persistence_mode == PersistenceMode.CLC
            problem_shape_ntile_mn = cute.select(
                args.problem_shape_ntile_mnl, mode=[0, 1]
            )
            problem_shape_ncluster_mn = (
                cute.ceil_div(problem_shape_ntile_mn[0], args.cluster_shape_mnk[0]),
                cute.ceil_div(problem_shape_ntile_mn[1], args.cluster_shape_mnk[1]),
            )
            problem_shape_ncluster_mnl = problem_shape_ncluster_mn + (
                args.problem_shape_ntile_mnl[2],
            )
            num_clusters_per_problem = cute.size(problem_shape_ncluster_mn)
            raster_order = get_raster_order_from_option(
                args.raster_order, problem_shape_ncluster_mn, args.group_size
            )
            ncluster_fast = (
                problem_shape_ncluster_mn[0]
                if raster_order == RasterOrder.AlongM
                else problem_shape_ncluster_mn[1]
            )
            ncluster_slow = (
                problem_shape_ncluster_mn[1]
                if raster_order == RasterOrder.AlongM
                else problem_shape_ncluster_mn[0]
            )
            group_size = min(args.group_size, ncluster_fast)
            group_size_tail = ncluster_fast % group_size
            num_groups_regular = ncluster_fast // group_size
            num_clusters_in_group = group_size * ncluster_slow
            return TileScheduler.Params(
                problem_shape_ncluster_mnl,
                raster_order,
                FastDivmod(num_clusters_per_problem),
                num_groups_regular,
                FastDivmod(group_size),
                FastDivmod(group_size_tail if group_size_tail > 0 else 1),
                FastDivmod(num_clusters_in_group),
                args.n_rotation,
                args.cluster_shape_mnk,
                args.persistence_mode,
            )

    def __init__(
        self,
        current_work_idx: Int32,
        num_tiles_executed: Int32,
        current_batch_idx: Int32,
        num_work_idx_before_cur_batch: Int32,
        sched_smem: cute.Tensor,
        scheduler_pipeline: cutlass.pipeline.PipelineAsync,
        pipeline_state: cutlass.pipeline.PipelineState,
        params: Params,
        *,
        loc=None,
        ip=None,
    ):
        self._current_work_idx = current_work_idx
        self.num_tiles_executed = num_tiles_executed
        self._current_batch_idx = current_batch_idx
        self._num_work_idx_before_cur_batch = num_work_idx_before_cur_batch
        self._sched_smem = sched_smem
        self._scheduler_pipeline = scheduler_pipeline
        self._pipeline_state = pipeline_state
        self.params = params
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(
        args: TileSchedulerArguments, *, loc=None, ip=None
    ) -> Params:
        return TileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def _init_clc_mbarrier(sched_smem: cute.Tensor, *, loc=None, ip=None) -> None:
        assert cute.size(sched_smem, mode=[0]) >= 12
        clc_mbar_ptr = sched_smem[None, 0].iterator + 8
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(clc_mbar_ptr, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_warp()

    @staticmethod
    @cute.jit
    def _cluster_idx_to_work_idx_batch(
        params: Params,
        cluster_idx: Tuple[Int32, Int32, Int32],
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32]:
        return Int32(cluster_idx[0]), Int32(cluster_idx[2])

    @staticmethod
    @cute.jit
    def create(
        params: Params,
        sched_smem: cute.Tensor,
        scheduler_pipeline: cutlass.pipeline.PipelineAsync,
        is_scheduler_warp: bool | Boolean = False,
        *,
        loc=None,
        ip=None,
    ) -> "TileScheduler":
        cluster_idx = (
            cute.arch.block_idx()
            if const_expr(cute.size(params.cluster_shape_mnk, loc=loc, ip=ip) == 1)
            else cute.arch.cluster_idx()
        )
        current_work_idx, _ = TileScheduler._cluster_idx_to_work_idx_batch(
            params, cluster_idx, loc=loc, ip=ip
        )
        stages = const_expr(cute.size(sched_smem, mode=[1]))
        if is_scheduler_warp:
            TileScheduler._init_clc_mbarrier(sched_smem, loc=loc, ip=ip)
        return TileScheduler(
            current_work_idx,
            Int32(0),
            Int32(0),
            Int32(0),
            sched_smem,
            scheduler_pipeline,
            pipeline.PipelineState(stages, Int32(0), Int32(0), Int32(0)),
            params,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def get_grid_shape(
        params: Params,
        max_active_clusters: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        _ = max_active_clusters
        return (
            params.cluster_shape_mnk[0]
            * cute.size(params.problem_shape_ncluster_mnl[:2]),
            params.cluster_shape_mnk[1],
            params.cluster_shape_mnk[2] * params.problem_shape_ncluster_mnl[2],
        )

    @cute.jit
    def _swizzle_cta(
        self, cluster_id_in_problem: Int32, *, loc=None, ip=None
    ) -> Tuple[Int32, Int32]:
        params = self.params
        group_id, id_in_group = divmod(
            cluster_id_in_problem, params.num_clusters_in_group_fdd
        )
        cid_fast_in_group, cid_slow = Int32(0), Int32(0)
        if group_id < params.num_groups_regular:
            cid_slow, cid_fast_in_group = divmod(id_in_group, params.group_size_fdd)
        else:
            cid_slow, cid_fast_in_group = divmod(
                id_in_group, params.group_size_tail_fdd
            )
        if group_id % 2 == 1:
            ncluster_slow = (
                params.problem_shape_ncluster_mnl[1]
                if params.raster_order == RasterOrder.AlongM
                else params.problem_shape_ncluster_mnl[0]
            )
            cid_slow = ncluster_slow - 1 - cid_slow
        cid_fast = group_id * params.group_size_fdd.divisor + cid_fast_in_group
        cid_m, cid_n = cid_fast, cid_slow
        if params.raster_order == RasterOrder.AlongN:
            cid_m, cid_n = cid_slow, cid_fast
        return cid_m, cid_n

    @cute.jit
    def _cluster_id_to_cta_id(
        self,
        cid_m: Int32,
        cid_n: Int32,
        *,
        block_zero_only: bool = False,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32]:
        if const_expr(
            block_zero_only
            or cute.size(self.params.cluster_shape_mnk, loc=loc, ip=ip) == 1
        ):
            bidx_in_cluster = (Int32(0), Int32(0))
        else:
            bidx_in_cluster = cute.arch.block_in_cluster_idx()
        pid_m = cid_m * self.params.cluster_shape_mnk[0] + bidx_in_cluster[0]
        pid_n = cid_n * self.params.cluster_shape_mnk[1] + bidx_in_cluster[1]
        return pid_m, pid_n

    @cute.jit
    def _delinearize_work_idx(
        self,
        work_idx: Int32,
        bidz: Optional[Int32] = None,
        is_valid: Optional[Boolean] = None,
        *,
        block_zero_only: bool = False,
        loc=None,
        ip=None,
    ) -> cutlass.utils.WorkTileInfo:
        params = self.params
        if const_expr(is_valid is None):
            is_valid = work_idx < cute.size(params.problem_shape_ncluster_mnl[:2])
        pid_m, pid_n, batch_idx = Int32(0), Int32(0), Int32(0)
        if is_valid:
            cluster_id_in_problem = work_idx
            bidz_ = (
                cute.arch.block_idx()[2]
                if const_expr(cute.size(params.cluster_shape_mnk, loc=loc, ip=ip) == 1)
                else cute.arch.cluster_idx()[2]
            )
            if const_expr(bidz is not None):
                bidz_ = bidz
            cid_m, cid_n = self._swizzle_cta(cluster_id_in_problem, loc=loc, ip=ip)
            # N cluster tiles map to destination peers. A rank-dependent
            # offset makes the ranks write into different peers at any moment
            # instead of all of them into the same few (incast).
            cid_n = cid_n + params.n_rotation
            ncluster_n = params.problem_shape_ncluster_mnl[1]
            if cid_n >= ncluster_n:
                cid_n = cid_n - ncluster_n
            pid_m, pid_n = self._cluster_id_to_cta_id(
                cid_m,
                cid_n,
                block_zero_only=block_zero_only,
                loc=loc,
                ip=ip,
            )
            batch_idx = bidz_
        return cutlass.utils.WorkTileInfo((pid_m, pid_n, None, batch_idx), is_valid)

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        self._scheduler_pipeline.consumer_wait(self._pipeline_state)
        pid_m, pid_n, batch_idx, is_valid_i32 = [
            self._sched_smem[i, self._pipeline_state.index] for i in range(4)
        ]
        if const_expr(cute.size(self.params.cluster_shape_mnk) > 1):
            cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            self._scheduler_pipeline.consumer_release(self._pipeline_state)
        self._pipeline_state.advance()
        return cutlass.utils.WorkTileInfo(
            (pid_m, pid_n, None, batch_idx), Boolean(is_valid_i32)
        )

    def initial_work_tile_info(
        self, *, loc=None, ip=None
    ) -> cutlass.utils.WorkTileInfo:
        return self._delinearize_work_idx(self._current_work_idx, loc=loc, ip=ip)

    @cute.jit
    def _fetch_next_work_idx(
        self, *, loc=None, ip=None
    ) -> Tuple[Int32, Int32, Boolean]:
        params = self.params
        clc_response_ptr = (
            self._sched_smem[None, self._pipeline_state.index].iterator + 4
        )
        mbarrier_addr = self._sched_smem[None, 0].iterator + 8
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(mbarrier_addr, 16, loc=loc, ip=ip)
            issue_clc_query_nomulticast(mbarrier_addr, clc_response_ptr, loc=loc, ip=ip)
        cute.arch.sync_warp()
        cute.arch.mbarrier_wait(
            mbarrier_addr, self._pipeline_state.phase, loc=loc, ip=ip
        )
        bidx, bidy, bidz, valid = cute.arch.clc_response(
            clc_response_ptr, loc=loc, ip=ip
        )
        cute.arch.fence_view_async_shared()
        cluster_idx = (
            bidx // params.cluster_shape_mnk[0],
            bidy // params.cluster_shape_mnk[1],
            bidz // params.cluster_shape_mnk[2],
        )
        work_idx, batch_idx = type(self)._cluster_idx_to_work_idx_batch(
            params, cluster_idx, loc=loc, ip=ip
        )
        return work_idx, batch_idx, Boolean(valid)

    @cute.jit
    def write_work_tile_to_smem(
        self, work_tile_info: cutlass.utils.WorkTileInfo, *, loc=None, ip=None
    ):
        params = self.params
        producer_state = pipeline.PipelineState(
            self._pipeline_state.stages,
            self._pipeline_state.count,
            self._pipeline_state.index,
            self._pipeline_state.phase ^ 1,
        )
        self._scheduler_pipeline.producer_acquire(producer_state)
        sched_data = [
            work_tile_info.tile_idx[0],
            work_tile_info.tile_idx[1],
            work_tile_info.tile_idx[3],
            Int32(work_tile_info.is_valid_tile),
        ]
        lane_idx = cute.arch.lane_idx()
        if lane_idx < cute.size(params.cluster_shape_mnk):
            pipeline_idx = self._pipeline_state.index
            if const_expr(cute.size(params.cluster_shape_mnk) == 1):
                for i in cutlass.range_constexpr(4):
                    self._sched_smem[i, pipeline_idx] = sched_data[i]
                self._scheduler_pipeline.producer_commit(self._pipeline_state)
            else:
                peer_cta_rank_in_cluster = lane_idx
                bidx_in_cluster = peer_cta_rank_in_cluster % params.cluster_shape_mnk[0]
                bidy_in_cluster = (
                    peer_cta_rank_in_cluster // params.cluster_shape_mnk[0]
                ) % params.cluster_shape_mnk[1]
                mbar_ptr = self._scheduler_pipeline.producer_get_barrier(
                    self._pipeline_state
                )
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mbar_ptr, 16, peer_cta_rank_in_cluster
                )
                store_shared_remote_x4(
                    sched_data[0] + bidx_in_cluster,
                    sched_data[1] + bidy_in_cluster,
                    sched_data[2],
                    sched_data[3],
                    smem_ptr=self._sched_smem[None, pipeline_idx].iterator,
                    mbar_ptr=mbar_ptr,
                    peer_cta_rank_in_cluster=peer_cta_rank_in_cluster,
                )

    @cute.jit
    def advance_to_next_work(
        self,
        is_scheduler_warp: bool | Boolean = False,
        *,
        loc=None,
        ip=None,
    ):
        self.num_tiles_executed += Int32(1)
        if is_scheduler_warp:
            self._current_work_idx, batch_idx, is_valid = self._fetch_next_work_idx(
                loc=loc, ip=ip
            )
            work_tile_info = self._delinearize_work_idx(
                self._current_work_idx,
                batch_idx,
                is_valid,
                block_zero_only=True,
                loc=loc,
                ip=ip,
            )
            self.write_work_tile_to_smem(work_tile_info, loc=loc, ip=ip)

    def producer_tail(self):
        producer_state = pipeline.PipelineState(
            self._pipeline_state.stages,
            self._pipeline_state.count,
            self._pipeline_state.index,
            self._pipeline_state.phase ^ 1,
        )
        self._scheduler_pipeline.producer_tail(producer_state)

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [
            self._current_work_idx,
            self.num_tiles_executed,
            self._current_batch_idx,
            self._num_work_idx_before_cur_batch,
            self._sched_smem,
            self._scheduler_pipeline,
            self._pipeline_state,
            self.params,
        ]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [
                self._current_work_idx,
                self.num_tiles_executed,
                self._current_batch_idx,
                self._num_work_idx_before_cur_batch,
                self._sched_smem,
                self._scheduler_pipeline,
                self._pipeline_state,
                self.params,
            ],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*tuple(obj_list), loc=self._loc)


def make_scheduler_args(max_active_clusters, max_swizzle_size, n_rotation=0):
    return TileSchedulerOptions(
        max_active_clusters=Int32(max_active_clusters),
        raster_order=None,
        max_swizzle_size=Int32(max_swizzle_size),
        n_rotation=Int32(n_rotation),
    )


def make_fake_scheduler_args():
    return TileSchedulerOptions(
        max_active_clusters=Int32(1),
        max_swizzle_size=Int32(8),
        n_rotation=Int32(0),
    )


def peer_n_rotation(
    local_rank: int,
    num_peers: int,
    n_total: int,
    tile_n: int,
    cluster_n: int = 1,
) -> int:
    """First N cluster tile visited by ``local_rank``.

    Each rank starts at the first cluster tile of its own column block, so at
    any moment the ranks of a group write into different destination peers.
    The offset only reorders tiles; it never changes a value.
    """

    n_tiles = (n_total + tile_n - 1) // tile_n
    ncluster_n = (n_tiles + cluster_n - 1) // cluster_n
    return ((local_rank * ncluster_n + num_peers - 1) // num_peers) % ncluster_n


# Fixed-shape batch manager.
class BatchManager:
    def __init__(
        self,
        len_m_static: Int32,
        len_k_static: Int32,
        *,
        loc=None,
        ip=None,
    ):
        self._len_m_static = len_m_static
        self._len_k_static = len_k_static
        self._loc = loc
        self._ip = ip

    @staticmethod
    @cute.jit
    def create(
        len_m_static: Int32,
        len_k_static: Int32,
        *,
        loc=None,
        ip=None,
    ) -> "BatchManager":
        return BatchManager(len_m_static, len_k_static, loc=loc, ip=ip)

    def len_m(self, batch_idx: Int32) -> Int32:
        _ = batch_idx
        return self._len_m_static

    def len_k(self, batch_idx: Int32) -> Int32:
        _ = batch_idx
        return self._len_k_static

    def offset_batch_A(self, tensor: cute.Tensor, batch_idx: Int32) -> cute.Tensor:
        return tensor[None, None, batch_idx]

    def offset_batch_B(self, tensor: cute.Tensor, batch_idx: Int32) -> cute.Tensor:
        return tensor[None, None, batch_idx]

    def offset_batch_epi(self, tensor: cute.Tensor, batch_idx: Int32) -> cute.Tensor:
        return tensor[None, None, batch_idx]

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self._len_m_static, self._len_k_static]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [self._len_m_static, self._len_k_static], self._values_pos
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*tuple(obj_list), loc=self._loc)


# Peer epilogue storage accounting and TMA setup.
class EpiSmemBytes(NamedTuple):
    unstaged: int = 0
    d_stage: int = 0
    c_stage: int = 0


def setup_epi_tensor(gemm, tensor, epi_tile=None, stage=None):
    if epi_tile is None:
        epi_tile = gemm.epi_tile
    if stage is None:
        stage = gemm.epi_stage
    smem_layout_staged = gemm_helpers.make_smem_layout_epi(
        tensor.element_type,
        LayoutEnum.from_tensor(tensor),
        epi_tile,
        stage,
    )
    tma_atom, tma_tensor = gemm._make_tma_epi_atoms_and_tensors(
        tensor, smem_layout_staged, epi_tile
    )
    return tma_atom, tma_tensor, smem_layout_staged, epi_tile


# GEMM base and implementation.
class NamedBarrierGemm(enum.IntEnum):
    # Explicit values preserve the source barrier ABI after pruning.
    Epilogue = 1
    EpilogueLoad = 2
    MmaWG0 = 3
    MmaWG1 = 4
    EpiWG0 = 5
    EpiWG1 = 6
    TmemPtr = 7


class GemmBase:
    arch = 100

    @dataclass
    class EpilogueArguments:
        pass

    EpilogueParams = ParamsBase

    @cute.jit
    def epilogue(
        self,
        params,
        extra_sources,
        epi_smem_tensors: Dict[str, cute.Tensor],
        epi_store_pipeline: cutlass.pipeline.PipelineAsync,
        epi_tile: cute.Tile,
        load_acc_subtile: Callable,
        tRS_rD: cute.Tensor,
        tiled_copy_t2r: cute.TiledCopy,
        tiled_copy_r2s: cute.TiledCopy,
        tRS_sD: Optional[cute.Tensor],
        copy_D: Optional[Callable],
        tile_coord_mnkl: cute.Coord,
        batch_manager: BatchManager,
        epilogue_barrier: cutlass.pipeline.NamedBarrier,
        tile_scheduler: TileScheduler,
        tidx: Int32,
        is_tma_warp: Boolean,
    ) -> None:
        has_D = const_expr(copy_D is not None)
        aux_out_ctx = self.epi_setup_aux_out(
            params,
            epi_smem_tensors,
            tiled_copy_r2s,
            tiled_copy_t2r,
            tile_coord_mnkl,
            batch_manager,
            tidx,
        )
        epi_tile_shape = cute.zipped_divide(
            cute.make_layout(self.cta_tile_shape_mnk[:2]), epi_tile
        ).shape[1]
        epi_tile_layout = cute.make_ordered_layout(
            epi_tile_shape,
            order=(0, 1) if const_expr(self.epi_m_major) else (1, 0),
        )
        epi_tile_num = cute.size(epi_tile_shape)
        num_prev_subtiles = tile_scheduler.num_tiles_executed * epi_tile_num
        epi_tensors = self.epi_begin(
            params,
            extra_sources,
            epi_smem_tensors,
            epi_tile,
            tiled_copy_t2r,
            tiled_copy_r2s,
            tile_coord_mnkl,
            batch_manager,
            epilogue_barrier,
            tidx,
            tRS_rD.layout,
        )
        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            epi_coord = epi_tile_layout.get_hier_coord(epi_idx)
            load_acc_subtile(tRS_rD, epi_coord)
            epi_loop_tensors = self.epi_begin_loop(params, epi_tensors, epi_coord)
            tRS_rAuxOut = self.epi_visit_subtile(params, epi_loop_tensors, tRS_rD, None)
            self.epi_end_loop(
                params,
                epi_tensors,
                epi_coord,
                epi_tile,
                tiled_copy_t2r,
                tiled_copy_r2s,
                tile_coord_mnkl,
                batch_manager,
                tidx,
            )
            if const_expr(aux_out_ctx is not None):
                tRS_rAuxOut_out = self.epi_convert_aux_out(
                    tRS_rAuxOut,
                    None,
                    tidx,
                    tile_coord_mnkl,
                    num_prev_subtiles,
                    epi_idx,
                )
            if is_tma_warp:
                epi_store_pipeline.producer_acquire()
            epilogue_barrier.arrive_and_wait()
            epi_buffer = (num_prev_subtiles + epi_idx) % self.epi_stage
            if const_expr(has_D):
                cvt_copy(
                    tiled_copy_r2s,
                    tRS_rD,
                    tRS_sD[None, None, None, epi_buffer],
                )
            if const_expr(aux_out_ctx is not None):
                tiled_copy_aux_r2s, tRS_sAuxOut, copy_aux_out = aux_out_ctx
                cute.copy(
                    tiled_copy_aux_r2s,
                    contiguous(tiled_copy_aux_r2s.retile(tRS_rAuxOut_out)),
                    tRS_sAuxOut[None, None, None, epi_buffer],
                )
            cute.arch.fence_view_async_shared()
            epilogue_barrier.arrive_and_wait()
            if is_tma_warp:
                if const_expr(has_D):
                    copy_D(src_idx=epi_buffer, dst_idx=epi_coord)
                if const_expr(aux_out_ctx is not None):
                    copy_aux_out(src_idx=epi_buffer, dst_idx=epi_coord)
                epi_store_pipeline.producer_commit()
        self.epi_end(
            params,
            epi_tensors,
            epi_tile,
            tiled_copy_t2r,
            tiled_copy_r2s,
            tile_coord_mnkl,
            batch_manager,
            tidx,
        )

    def get_scheduler_arguments(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: Optional[cute.Tensor],
        scheduler_args: TileSchedulerOptions,
    ) -> TileSchedulerArguments:
        num_problems = mD.shape[2] if mD is not None else mB.shape[2]
        return TileSchedulerArguments(
            problem_shape_ntile_mnl=(
                cute.ceil_div(cute.size(mA, mode=[0]), self.cta_tile_shape_mnk[0]),
                cute.ceil_div(cute.size(mB, mode=[0]), self.cta_tile_shape_mnk[1]),
                num_problems,
            ),
            raster_order=scheduler_args.raster_order,
            group_size=scheduler_args.max_swizzle_size,
            cluster_shape_mnk=self.cluster_shape_mnk,
            persistence_mode=PersistenceMode.CLC,
            n_rotation=scheduler_args.n_rotation,
        )


class GemmTmaBase(GemmBase):
    @cute.jit
    def load_tma(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        producer_state: cutlass.pipeline.PipelineState,
        copy_fns: Tuple[Callable, ...],
        k_tile_cnt: Int32,
    ) -> cutlass.pipeline.PipelineState:
        peek_empty_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_empty_status = ab_pipeline.producer_try_acquire(producer_state)
        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            ab_pipeline.producer_acquire(producer_state, peek_empty_status)
            tma_bar_ptr = ab_pipeline.producer_get_barrier(producer_state)
            smem_idx = producer_state.index
            for copy_fn in copy_fns:
                copy_fn(k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)
            ab_pipeline.producer_commit(producer_state)
            producer_state.advance()
            peek_empty_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_empty_status = ab_pipeline.producer_try_acquire(producer_state)
        return producer_state

    def make_tma_epilogue_atoms_and_tensors(self, mD: Optional[cute.Tensor]):
        if const_expr(mD is not None):
            return self._make_tma_epi_atoms_and_tensors(
                mD, self.epi_smem_layout_staged, self.epi_tile
            )
        return None, None

    def epilog_gmem_copy_and_partition(
        self,
        atom: cute.CopyAtom,
        mD_mn: cute.Tensor,
        tile_shape_mn: cute.Tile,
        epi_tile: cute.Tile,
        sD: cute.Tensor,
        tile_coord_mnkl: cute.Coord,
    ):
        gD = cute.local_tile(mD_mn, tile_shape_mn, tile_coord_mnkl[:2])
        tDgD_for_tma_partition = cute.zipped_divide(gD, epi_tile)
        return tma_get_copy_fn(
            atom,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            src_tensor=sD,
            dst_tensor=tDgD_for_tma_partition,
        )

    def make_epi_store_pipeline(self):
        producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_epi_warps * cute.arch.WARP_SIZE,
        )
        return pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stage, producer_group=producer_group
        )

    @staticmethod
    def _make_tma_epi_atoms_and_tensors(
        tensor_d: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: Tuple[int, int],
    ) -> Tuple[cute.CopyAtom, cute.Tensor]:
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        d_cta_v_layout = cute.composition(
            cute.make_identity_layout(tensor_d.shape), epi_tile
        )
        return cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            tensor_d,
            epi_smem_layout,
            d_cta_v_layout,
        )


class PersistentGemm(GemmTmaBase):
    arch = 100
    EpilogueArguments = GemmTmaBase.EpilogueArguments
    EpilogueParams = GemmTmaBase.EpilogueParams

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        a_dtype: Type[cutlass.Numeric],
        mma_tiler_mnk: Union[Tuple[int, int], Tuple[int, int, int]],
        cluster_shape_mnk: Tuple[int, int, int],
        use_clc_persistence: bool = True,
        use_pdl: bool = True,
    ):
        assert acc_dtype is Float32
        assert a_dtype is BFloat16
        assert use_clc_persistence
        assert len(mma_tiler_mnk) in (2, 3)
        self.acc_dtype = acc_dtype
        self.use_2cta_instrs = cluster_shape_mnk[0] % 2 == 0 and mma_tiler_mnk[0] in (
            128,
            256,
        )
        self.cluster_shape_mnk = cluster_shape_mnk
        assert cluster_shape_mnk[2] == 1
        self.mma_tiler = (
            tuple(mma_tiler_mnk) if len(mma_tiler_mnk) == 3 else (*mma_tiler_mnk, 0)
        )
        self.is_persistent = True
        self.use_clc_persistence = True
        self.use_pdl = use_pdl
        self.epi_m_major = True
        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        self.num_ab_load_warps = 1
        self.occupancy = 1
        self.epi_warps_per_accumulator = 4
        self.epilog_warp_id = tuple(range(self.epi_warps_per_accumulator))
        self.mma_warp_id = len(self.epilog_warp_id)
        self.ab_load_warp_id = self.mma_warp_id + 1
        # Preserve the selected source warp assignment.
        self.epi_load_warp_id = self.ab_load_warp_id + 1
        self.scheduler_warp_id = self.epi_load_warp_id + 1
        self.num_epi_warps = len(self.epilog_warp_id)
        self.epilogue_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierGemm.Epilogue),
            num_threads=self.num_epi_warps * cute.arch.WARP_SIZE,
        )
        self.threads_per_cta = cute.arch.WARP_SIZE * (
            self.num_ab_load_warps
            + len(
                (
                    self.mma_warp_id,
                    self.epi_load_warp_id,
                    self.scheduler_warp_id,
                    *self.epilog_warp_id,
                )
            )
        )
        assert self.threads_per_cta % 128 == 0

    def epi_smem_warp_shape_mnk(self):
        warp_m, warp_n = (
            (2, 2)
            if self.cta_tile_shape_mnk[0] == 64 and self.use_2cta_instrs
            else (4, 1)
        )
        return warp_m, warp_n, 1

    def _setup_attributes(self, epilogue_args: EpilogueArguments):
        self.epi_m_major = self.resolve_epi_m_major(epilogue_args)
        mma_inst_shape_n = (
            self.mma_tiler[1] if self.mma_tiler[1] <= 256 else self.mma_tiler[1] // 2
        )
        self.mma_inst_shape_mnk = (
            self.mma_tiler[0],
            mma_inst_shape_n,
            256 // self.a_dtype.width,
        )
        # CUTLASS DSL 4.4.2 signature: explicit OperandMajorMode A/B args.
        self.tiled_mma = gemm_helpers.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_inst_shape_mnk[:2],
        )
        if self.mma_tiler[2] > 0:
            assert self.mma_tiler[2] % self.mma_inst_shape_mnk[2] == 0
            mma_inst_tile_k = self.mma_tiler[2] // self.mma_inst_shape_mnk[2]
        else:
            mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            self.mma_inst_shape_mnk[2] * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(self.tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (self.tiled_mma.thr_id.shape,),
        )
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.epi_tile = gemm_helpers.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.d_layout if self.d_layout is not None else LayoutEnum.ROW_MAJOR,
            self.d_dtype if self.d_dtype is not None else BFloat16,
            layout_c=None,
            elem_ty_c=None,
        )
        # CUTLASS DSL 4.4.2 epilogue tile divisibility fix.
        if const_expr(self.cta_tile_shape_mnk[1] % cute.size(self.epi_tile[1]) != 0):
            warp_n = (
                2 if self.cta_tile_shape_mnk[0] == 64 and self.use_2cta_instrs else 1
            )
            epi_tile_n = math.gcd(
                self.cta_tile_shape_mnk[1], cute.size(self.epi_tile[1])
            )
            epi_tile_n_layout = cute.make_layout(
                (epi_tile_n // warp_n, warp_n),
                stride=(1, self.cta_tile_shape_mnk[1] // warp_n),
            )
            self.epi_tile = (
                self.epi_tile[0],
                cute.coalesce(epi_tile_n_layout),
            )
        # Each row of a subtile reaches a peer as one NVLink write: in the
        # all-to-all, 128-byte rows (64 columns) sustain about 1.6x the peer
        # write bandwidth of 64-byte rows (32 columns).
        assert self.cta_tile_shape_mnk[1] % PEER_EPI_TILE_N == 0
        self.epi_tile = (self.epi_tile[0], cute.make_layout(PEER_EPI_TILE_N))
        (
            self.num_acc_stage,
            self.ab_stage,
            self.epi_stage,
            self.epi_c_stage,
        ) = self._compute_stages(
            self.tiled_mma,
            self.mma_tiler,
            self.cta_tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.b_dtype,
            self.d_dtype,
            self.d_layout,
            epilogue_args,
            cutlass.utils.get_smem_capacity_in_bytes(f"sm_{self.arch}"),
            self.occupancy,
            self.epi_smem_warp_shape_mnk(),
        )
        self.sched_stage = 1
        self.a_prefetch_stage = 0
        self.a_smem_layout_staged = gemm_helpers.make_smem_layout_a(
            self.tiled_mma, self.mma_tiler, self.a_dtype, self.ab_stage
        )
        self.b_smem_layout_staged = gemm_helpers.make_smem_layout_b(
            self.tiled_mma, self.mma_tiler, self.b_dtype, self.ab_stage
        )
        self.epi_smem_layout_staged = None
        if const_expr(self.d_dtype is not None):
            self.epi_smem_layout_staged = gemm_helpers.make_smem_layout_epi(
                self.d_dtype,
                self.d_layout,
                self.epi_tile,
                self.epi_stage,
            )
        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
            self.tiled_mma, self.mma_tiler, self.num_acc_stage
        )
        self.overlap_accum_sf = False
        self.iter_acc_early_release = -1

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: Optional[cute.Tensor],
        epilogue_args: tuple,
        extra_source0: cute.Tensor,
        extra_source1: cute.Tensor,
        extra_source2: cute.Tensor,
        extra_source3: cute.Tensor,
        extra_source4: cute.Tensor,
        extra_source5: cute.Tensor,
        extra_source6: cute.Tensor,
        extra_source7: cute.Tensor,
        scheduler_args: TileSchedulerOptions,
        stream: cuda.CUstream,
    ):
        self.a_dtype = mA.element_type
        self.b_dtype = mB.element_type
        self.d_dtype = mD.element_type if mD is not None else None
        assert self.a_dtype is BFloat16
        assert self.b_dtype is BFloat16
        if const_expr(self.d_dtype is not None):
            assert self.d_dtype is BFloat16
        self.a_layout = LayoutEnum.from_tensor(mA)
        self.b_layout = LayoutEnum.from_tensor(mB)
        self.d_layout = LayoutEnum.from_tensor(mD) if mD is not None else None
        self.a_major_mode = self.a_layout.mma_major_mode()
        self.b_major_mode = self.b_layout.mma_major_mode()
        self._setup_attributes(epilogue_args)

        atom_thr_size = cute.size(self.tiled_mma.thr_id.shape)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        a_op = gemm_helpers.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mnk, self.tiled_mma.thr_id
        )
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            mA,
            a_smem_layout,
            self.mma_tiler,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=None,
        )
        b_op = gemm_helpers.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mnk, self.tiled_mma.thr_id
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            mB,
            b_smem_layout,
            self.mma_tiler,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=None,
        )
        self.num_tma_load_bytes = (
            cute.size_in_bytes(self.a_dtype, a_smem_layout)
            + cute.size_in_bytes(self.b_dtype, b_smem_layout)
        ) * atom_thr_size
        tma_atom_d, tma_tensor_d = self.make_tma_epilogue_atoms_and_tensors(mD)
        epilogue_params = self.epi_to_underlying_arguments(epilogue_args)
        tile_sched_params = TileScheduler.to_underlying_arguments(
            self.get_scheduler_arguments(mA, mB, mD, scheduler_args)
        )
        grid = TileScheduler.get_grid_shape(
            tile_sched_params, scheduler_args.max_active_clusters
        )

        self.buffer_align_bytes = 1024
        epi_smem_size = (
            cute.cosize(self.epi_smem_layout_staged) if mD is not None else 0
        )

        # Source ordering plus zero-sized alignment points preserves active
        # shared-storage offsets while dead features stay physically absent.
        @cute.struct
        class SharedStorage:
            ab_pipeline_array_ptr: cute.struct.MemRange[Int64, self.ab_stage * 2]
            reserved_epi_pipeline_array_ptr: cute.struct.MemRange[Int64, 0]
            acc_pipeline_array_ptr: cute.struct.MemRange[Int64, self.num_acc_stage * 2]
            sched_pipeline_array_ptr: cute.struct.MemRange[Int64, self.sched_stage * 2]
            reserved_prefetch_pipeline_array_ptr: cute.struct.MemRange[Int64, 0]
            sched_data: cute.struct.MemRange[Int32, self.sched_stage * 12]
            tmem_dealloc_mbar_ptr: Int64
            tmem_holding_buf: Int32
            reserved_align16: cute.struct.Align[cute.struct.MemRange[Int32, 0], 16]
            sD: cute.struct.Align[
                cute.struct.MemRange[
                    self.d_dtype if self.d_dtype is not None else Int32,
                    epi_smem_size,
                ],
                self.buffer_align_bytes,
            ]
            reserved_align1024: cute.struct.Align[
                cute.struct.MemRange[Int32, 0], self.buffer_align_bytes
            ]
            epi: self.epi_get_smem_struct(epilogue_params)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype,
                    cute.cosize(self.a_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype,
                    cute.cosize(self.b_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            reserved_tail_align0: cute.struct.Align[
                cute.struct.MemRange[Int32, 0], self.buffer_align_bytes
            ]
            reserved_tail_align1: cute.struct.Align[
                cute.struct.MemRange[Int32, 0], self.buffer_align_bytes
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            self.tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_d,
            tma_tensor_d,
            epilogue_params,
            extra_source0,
            extra_source1,
            extra_source2,
            extra_source3,
            extra_source4,
            extra_source5,
            extra_source6,
            extra_source7,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            self.epi_tile,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_d: Optional[cute.CopyAtom],
        mD_mnl: Optional[cute.Tensor],
        epilogue_params,
        extra_source0: cute.Tensor,
        extra_source1: cute.Tensor,
        extra_source2: cute.Tensor,
        extra_source3: cute.Tensor,
        extra_source4: cute.Tensor,
        extra_source5: cute.Tensor,
        extra_source6: cute.Tensor,
        extra_source7: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        epi_smem_layout: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: TileScheduler.Params,
    ):
        has_D = const_expr(mD_mnl is not None)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == self.ab_load_warp_id:
            for tma_atom in (tma_atom_a, tma_atom_b, tma_atom_d):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        tidx, _, _ = cute.arch.thread_idx()

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        ab_pipeline = self.make_ab_pipeline(
            tiled_mma,
            cluster_layout_vmnk,
            storage.ab_pipeline_array_ptr.data_ptr(),
        )
        acc_pipeline = self.make_acc_pipeline(
            cluster_layout_vmnk,
            storage.acc_pipeline_array_ptr.data_ptr(),
        )
        sched_pipeline = self.make_sched_pipeline(
            self.cluster_shape_mnk,
            storage.sched_pipeline_array_ptr.data_ptr(),
        )
        sched_data = storage.sched_data.get_tensor((12, self.sched_stage))
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierGemm.TmemPtr),
            num_threads=cute.arch.WARP_SIZE
            * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        sD = (
            storage.sD.get_tensor(epi_smem_layout.outer, swizzle=epi_smem_layout.inner)
            if const_expr(has_D)
            else None
        )
        epi_smem_tensors = self.epi_get_smem_tensors(epilogue_params, storage)
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )
        batch_manager = BatchManager.create(
            Int32(cute.size(mA_mkl, mode=[0])),
            Int32(cute.size(mA_mkl, mode=[1])),
        )
        TileSchedulerFactory = partial(
            TileScheduler.create,
            tile_sched_params,
            sched_data,
            sched_pipeline,
        )
        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        # Specialized TMA load warp.
        if warp_idx == self.ab_load_warp_id:
            if const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
            block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
                cta_rank_in_cluster
            )
            a_mcast_mask, b_mcast_mask = None, None
            if const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
                a_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk,
                    block_in_cluster_coord_vmnk,
                    mcast_mode=2,
                )
                b_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk,
                    block_in_cluster_coord_vmnk,
                    mcast_mode=1,
                )
            tile_scheduler = TileSchedulerFactory()
            work_tile = tile_scheduler.initial_work_tile_info()
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.ab_stage
            )
            while work_tile.is_valid_tile:
                tile_coord_mnkl = work_tile.tile_idx
                batch_idx = tile_coord_mnkl[3]
                mma_tile_coord_mnl = (
                    tile_coord_mnkl[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_coord_mnkl[1],
                    batch_idx,
                )
                gA_mk = cute.local_tile(
                    batch_manager.offset_batch_A(mA_mkl, batch_idx),
                    cute.select(self.mma_tiler, [0, 2]),
                    (mma_tile_coord_mnl[0], None),
                )
                gB_nk = cute.local_tile(
                    batch_manager.offset_batch_B(mB_nkl, batch_idx),
                    cute.select(self.mma_tiler, [1, 2]),
                    (mma_tile_coord_mnl[1], None),
                )
                tCgA = thr_mma.partition_A(gA_mk)
                copy_A, _, _ = tma_get_copy_fn(
                    tma_atom_a,
                    cta_coord=block_in_cluster_coord_vmnk[2],
                    cta_layout=cute.make_layout(
                        cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
                    ),
                    src_tensor=tCgA,
                    dst_tensor=sA,
                    mcast_mask=a_mcast_mask,
                )
                tCgB = thr_mma.partition_B(gB_nk)
                copy_B, _, _ = tma_get_copy_fn(
                    tma_atom_b,
                    cta_coord=block_in_cluster_coord_vmnk[1],
                    cta_layout=cute.make_layout(
                        cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
                    ),
                    src_tensor=tCgB,
                    dst_tensor=sB,
                    mcast_mask=b_mcast_mask,
                )
                k_tile_cnt = cute.ceil_div(
                    batch_manager.len_k(batch_idx),
                    self.cta_tile_shape_mnk[2],
                )
                ab_producer_state = self.load_tma(
                    ab_pipeline,
                    ab_producer_state,
                    (copy_A, copy_B),
                    k_tile_cnt,
                )
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
            ab_pipeline.producer_tail(ab_producer_state)

        # Specialized CLC scheduler warp.
        if warp_idx == self.scheduler_warp_id:
            if const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
            is_scheduler_warp = Boolean(True)
            if const_expr(cute.size(cluster_layout_vmnk) > 1):
                is_scheduler_warp = cute.arch.block_idx_in_cluster() == 0
            tile_scheduler = TileSchedulerFactory(is_scheduler_warp=is_scheduler_warp)
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                tile_scheduler.advance_to_next_work(is_scheduler_warp=is_scheduler_warp)
                work_tile = tile_scheduler.get_current_work()
            if is_scheduler_warp:
                tile_scheduler.producer_tail()

        # Specialized MMA warp.
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            tile_scheduler = TileSchedulerFactory()
            work_tile = tile_scheduler.initial_work_tile_info()
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            while work_tile.is_valid_tile:
                batch_idx = work_tile.tile_idx[3]
                k_tile_cnt = cute.ceil_div(
                    batch_manager.len_k(batch_idx), self.mma_tiler[2]
                )
                tCtAcc = tCtAcc_base[None, None, None, acc_producer_state.index]
                (
                    ab_consumer_state,
                    acc_producer_state,
                    tiled_mma,
                ) = self.mma(
                    ab_pipeline,
                    acc_pipeline,
                    ab_consumer_state,
                    acc_producer_state,
                    tiled_mma,
                    tCrA,
                    tCrB,
                    tCtAcc,
                    k_tile_cnt,
                    is_leader_cta,
                )
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
            if const_expr(self.use_pdl):
                cute.arch.griddepcontrol_launch_dependents()
            tmem_alloc_barrier.arrive()
            acc_pipeline.producer_tail(acc_producer_state)

        # Specialized epilogue warps.
        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            is_tma_warp = Boolean(warp_idx == self.epilog_warp_id[0])
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc,
            ) = self.epilog_tmem_copy_and_partition(
                tidx, tCtAcc_base, epi_tile, use_2cta_instrs
            )
            tTR_rD = cute.make_rmem_tensor(tTR_rAcc.shape, self.acc_dtype)
            tiled_copy_r2s, tRS_rD, tRS_sD = self.epilog_smem_store_and_partition(
                tiled_copy_t2r,
                self.d_layout,
                self.d_dtype,
                tTR_rD,
                sD,
                tidx,
            )
            tile_scheduler = TileSchedulerFactory()
            work_tile = tile_scheduler.initial_work_tile_info()
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            epi_store_pipeline = self.make_epi_store_pipeline()
            while work_tile.is_valid_tile:
                tile_coord_mnkl = work_tile.tile_idx
                batch_idx = tile_coord_mnkl[3]
                tTR_tAcc = tTR_tAcc_base[
                    None,
                    None,
                    None,
                    None,
                    None,
                    acc_consumer_state.index,
                ]
                acc_pipeline.consumer_wait(acc_consumer_state)
                copy_D = None
                if const_expr(has_D):
                    copy_D, _, _ = self.epilog_gmem_copy_and_partition(
                        tma_atom_d,
                        batch_manager.offset_batch_epi(mD_mnl, batch_idx),
                        self.cta_tile_shape_mnk[:2],
                        epi_tile,
                        sD,
                        tile_coord_mnkl,
                    )
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                epi_tile_num = cute.size(
                    cute.zipped_divide(
                        cute.make_layout(self.cta_tile_shape_mnk[:2]),
                        epi_tile,
                    ),
                    mode=[1],
                )
                load_acc_subtile = partial(
                    self.epi_load_acc_subtile,
                    tiled_copy_t2r,
                    tiled_copy_r2s,
                    tTR_tAcc,
                    tTR_rAcc,
                    acc_pipeline=acc_pipeline,
                    acc_consumer_state=acc_consumer_state,
                    acc_release_idx=epi_tile_num - 1,
                )
                self.epilogue(
                    epilogue_params,
                    (
                        extra_source0,
                        extra_source1,
                        extra_source2,
                        extra_source3,
                        extra_source4,
                        extra_source5,
                        extra_source6,
                        extra_source7,
                    ),
                    epi_smem_tensors,
                    epi_store_pipeline,
                    epi_tile,
                    load_acc_subtile,
                    tRS_rD,
                    tiled_copy_t2r,
                    tiled_copy_r2s,
                    tRS_sD,
                    copy_D,
                    tile_coord_mnkl,
                    batch_manager,
                    self.epilogue_barrier,
                    tile_scheduler,
                    tidx,
                    is_tma_warp,
                )
                acc_consumer_state.advance()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
            if is_tma_warp:
                epi_store_pipeline.producer_tail()
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(acc_tmem_ptr)

    @cute.jit
    def mma(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        acc_pipeline: cutlass.pipeline.PipelineAsync,
        ab_consumer_state: cutlass.pipeline.PipelineState,
        acc_producer_state: cutlass.pipeline.PipelineState,
        tiled_mma: cute.TiledMma,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        acc: cute.Tensor,
        k_tile_cnt: Int32,
        is_leader_cta: Boolean,
    ):
        peek_ab_full_status = Boolean(True)
        if 0 < k_tile_cnt and is_leader_cta:
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
        if is_leader_cta:
            acc_pipeline.producer_acquire(acc_producer_state)
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        num_k_blocks = cute.size(tCrA, mode=[2])
        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            if is_leader_cta:
                ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
                for k_blk_idx in cutlass.range(num_k_blocks, unroll_full=True):
                    k_blk_coord = (
                        None,
                        None,
                        k_blk_idx,
                        ab_consumer_state.index,
                    )
                    cute.gemm(
                        tiled_mma,
                        acc,
                        tCrA[k_blk_coord],
                        tCrB[k_blk_coord],
                        acc,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                ab_pipeline.consumer_release(ab_consumer_state)
            ab_consumer_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt and is_leader_cta:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
        if is_leader_cta:
            acc_pipeline.producer_commit(acc_producer_state)
        acc_producer_state.advance()
        # Returning tiled_mma is required by CuTe dominance analysis.
        return ab_consumer_state, acc_producer_state, tiled_mma

    @cute.jit
    def epi_load_acc_subtile(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tiled_copy_r2s: cute.TiledCopy,
        tTR_tAcc: cute.Tensor,
        tTR_rAcc: cute.Tensor,
        tRS_rD: cute.Tensor,
        epi_coord,
        acc_pipeline: pipeline.PipelineAsync,
        acc_consumer_state: pipeline.PipelineState,
        acc_release_idx: int,
    ):
        cute.copy(
            tiled_copy_t2r,
            tTR_tAcc[None, None, None, epi_coord],
            tTR_rAcc,
        )
        tRS_rAcc = tiled_copy_r2s.retile(tTR_rAcc)
        tRS_rD.store(tRS_rAcc.load())
        assert epi_coord[0] == 0
        if epi_coord[1] == acc_release_idx:
            cute.arch.fence_view_async_tmem_load()
            with cute.arch.elect_one():
                acc_pipeline.consumer_release(acc_consumer_state)

    def epilog_tmem_copy_and_partition(
        self,
        tidx: Int32,
        tAcc: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[Boolean, bool],
    ):
        copy_atom_t2r = gemm_helpers.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.d_layout if self.d_layout is not None else LayoutEnum.ROW_MAJOR,
            self.d_dtype if self.d_dtype is not None else BFloat16,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0, None)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
        cAcc = cute.make_identity_tensor(
            (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1])
        )
        cAcc_epi = cute.flat_divide(cAcc, epi_tile)
        tTR_cAcc = thr_copy_t2r.partition_D(cAcc_epi)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_cAcc[None, None, None, 0, 0].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_store_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        d_layout: Optional[LayoutEnum],
        dtype: Optional[Type[cutlass.Numeric]],
        tTR_rD: cute.Tensor,
        sD: Optional[cute.Tensor],
        tidx: Int32,
    ):
        copy_atom_r2s = gemm_helpers.get_smem_store_op(
            d_layout if d_layout is not None else LayoutEnum.ROW_MAJOR,
            dtype if dtype is not None else BFloat16,
            self.acc_dtype,
            tiled_copy_t2r,
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sD) if sD is not None else None
        tRS_rD = tiled_copy_r2s.retile(tTR_rD)
        return tiled_copy_r2s, tRS_rD, tRS_sD

    @cute.jit
    def make_ab_pipeline(
        self,
        tiled_mma: cute.TiledMma,
        cluster_layout_vmnk: cute.Layout,
        ab_pipeline_mbar_ptr: cute.Pointer,
    ) -> pipeline.PipelineAsync:
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, mcast_size)
        # Built-in no-gather pipeline.
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=ab_pipeline_mbar_ptr,
            num_stages=self.ab_stage,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

    def make_acc_pipeline(
        self,
        cluster_layout_vmnk: cute.Layout,
        acc_pipeline_mbar_ptr: cute.Pointer,
    ) -> pipeline.PipelineAsync:
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_epi_warps * (2 if self.use_2cta_instrs else 1),
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=acc_pipeline_mbar_ptr,
            num_stages=self.num_acc_stage,
            producer_group=producer_group,
            consumer_group=consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

    def make_sched_pipeline(
        self,
        cluster_layout_mnk: cute.Layout,
        sched_pipeline_mbar_ptr: cute.Pointer,
    ) -> pipeline.PipelineAsync:
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cluster_size = cute.size(cluster_layout_mnk)
        warps_per_cta = self.num_ab_load_warps + len(
            (
                self.mma_warp_id,
                *self.epilog_warp_id,
                self.scheduler_warp_id,
            )
        )
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, warps_per_cta * cluster_size
        )
        return pipeline.PipelineAsync.create(
            barrier_storage=sched_pipeline_mbar_ptr,
            num_stages=self.sched_stage,
            producer_group=producer_group,
            consumer_group=consumer_group,
            consumer_mask=None if const_expr(cluster_size == 1) else 0,
            defer_sync=True,
        )

    @classmethod
    def _compute_stages(
        cls,
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        cta_tile_shape_mnk: Tuple[int, int, int],
        epi_tile: cute.Tile,
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        d_dtype: Optional[Type[cutlass.Numeric]],
        d_layout: Optional[LayoutEnum],
        epilogue_args: EpilogueArguments,
        smem_capacity: int,
        occupancy: int,
        warp_shape_mnk: Tuple[int, int, int] | None = None,
    ) -> Tuple[int, int, int, int]:
        # Exact selected branch of the source stage heuristic.
        num_acc_stage = 1 if mma_tiler_mnk[1] > 256 else 2
        epi_stage = 4 if cute.size(epi_tile[1]) <= 16 else 2
        epi_smem_bytes = cls.epi_smem_bytes(
            epilogue_args,
            cta_tile_shape_mnk,
            epi_tile,
            warp_shape_mnk,
        )
        epi_c_stage = 0
        a_smem_layout_one = gemm_helpers.make_smem_layout_a(
            tiled_mma, mma_tiler_mnk, a_dtype, 1
        )
        b_smem_layout_one = gemm_helpers.make_smem_layout_b(
            tiled_mma, mma_tiler_mnk, b_dtype, 1
        )
        d_smem_layout_one = (
            gemm_helpers.make_smem_layout_epi(d_dtype, d_layout, epi_tile, 1)
            if d_dtype is not None
            else None
        )
        ab_bytes_per_stage = cute.size_in_bytes(
            a_dtype, a_smem_layout_one
        ) + cute.size_in_bytes(b_dtype, b_smem_layout_one)
        mbar_helpers_bytes = 1024
        d_bytes_per_stage = (
            cute.size_in_bytes(d_dtype, d_smem_layout_one) if d_dtype is not None else 0
        )
        epi_bytes_per_stage = d_bytes_per_stage + epi_smem_bytes.d_stage
        epi_bytes = epi_smem_bytes.unstaged + epi_bytes_per_stage * epi_stage
        remaining_bytes = smem_capacity // occupancy - mbar_helpers_bytes - epi_bytes
        ab_stage = remaining_bytes // ab_bytes_per_stage
        if epi_bytes_per_stage > 0:
            epi_stage += (
                remaining_bytes - ab_bytes_per_stage * ab_stage
            ) // epi_bytes_per_stage
        return num_acc_stage, ab_stage, epi_stage, epi_c_stage

    @staticmethod
    def _compute_num_tmem_alloc_cols(
        tiled_mma: cute.TiledMma,
        mma_tiler: Tuple[int, int, int],
        num_acc_stage: int,
    ) -> int:
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage))
        return cutlass.utils.get_num_tmem_alloc_cols(tCtAcc_fake)


# Peer-write-specific epilogue hooks. The MRO is intentionally:
# PeerWriteMixin -> PersistentGemm -> GemmTmaBase -> GemmBase.
class PeerWriteMixin:
    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mPeerOut: cute.Tensor
        peer_extras: tuple
        tiles_per_peer: cutlass.Constexpr[int] = 1
        shard_N: cutlass.Constexpr[int] = 0
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        # Optional (N,) BF16 bias added to the FP32 accumulator before rounding.
        mBias: Optional[cute.Tensor] = None

    @dataclass
    class EpilogueParams(ParamsBase):
        # Required tuple fields stay first to preserve the v1 TVM-FFI ABI.
        peer_extra_atoms: object
        peer_extra_tensors: object
        tma_atom_mPeerOut: object = None
        mPeerOut: object = None
        epi_mPeerOut_smem_layout_staged: object = None
        epi_tile_mPeerOut: object = None
        mBias: object = None

    def resolve_epi_m_major(self, args):
        return True

    @classmethod
    def epi_smem_bytes(cls, args, cta_tile_shape_mnk, epi_tile, warp_shape_mnk=None):
        del cta_tile_shape_mnk, warp_shape_mnk
        return EpiSmemBytes(
            d_stage=cute.size(cute.shape(epi_tile))
            * (args.mPeerOut.element_type.width // 8)
        )

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        self.rounding_mode = args.rounding_mode
        assert self.rounding_mode == RoundingMode.RN
        self.aux_out_dtype = args.mPeerOut.element_type
        self.aux_out_layout = LayoutEnum.from_tensor(args.mPeerOut)
        self.cta_tile_shape_aux_out_mn = self.cta_tile_shape_mnk[:2]
        self.num_peers = 1 + len(args.peer_extras)
        self.tiles_per_peer = args.tiles_per_peer
        self.shard_N = args.shard_N

        atom, tensor, smem_layout, epi_tile = setup_epi_tensor(self, args.mPeerOut)
        extra_atoms, extra_tensors = [], []
        for peer_tensor in args.peer_extras:
            extra_atom, extra_tensor = self._make_tma_epi_atoms_and_tensors(
                peer_tensor, smem_layout, epi_tile
            )
            extra_atoms.append(extra_atom)
            extra_tensors.append(extra_tensor)
        return self.EpilogueParams(
            peer_extra_atoms=tuple(extra_atoms),
            peer_extra_tensors=tuple(extra_tensors),
            tma_atom_mPeerOut=atom,
            mPeerOut=tensor,
            epi_mPeerOut_smem_layout_staged=smem_layout,
            epi_tile_mPeerOut=epi_tile,
            mBias=args.mBias,
        )

    def epi_get_smem_struct(self, params):
        smem_layout = params.epi_mPeerOut_smem_layout_staged

        class EpiSharedStorage:
            s_mPeerOut: cute.struct.Align[
                cute.struct.MemRange[
                    self.aux_out_dtype,
                    cute.cosize(smem_layout),
                ],
                self.buffer_align_bytes,
            ]

        return cute.struct(EpiSharedStorage)

    def epi_get_smem_tensors(self, params, storage):
        smem_layout = params.epi_mPeerOut_smem_layout_staged
        return {
            "mPeerOut": storage.epi.s_mPeerOut.get_tensor(
                smem_layout.outer,
                swizzle=smem_layout.inner,
            )
        }

    @cute.jit
    def epi_begin(
        self,
        params,
        extra_sources,
        epi_smem_tensors,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        batch_manager,
        epilogue_barrier,
        tidx,
        tRS_rD_layout=None,
    ):
        tDgBias = None
        if const_expr(params.mBias is not None):
            # Broadcast the tile's bias slice along M and partition it like
            # the accumulator subtiles, so element i of a subtile's bias
            # fragment belongs to element i of tRS_rD.
            tile_M, tile_N = self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1]
            gBias = cute.local_tile(params.mBias, (tile_N,), (tile_coord_mnkl[1],))
            gBias_mn = cute.make_tensor(
                gBias.iterator, cute.make_layout((tile_M, tile_N), stride=(0, 1))
            )
            tDgBias = tiled_copy_t2r.get_slice(tidx).partition_D(
                cute.flat_divide(gBias_mn, epi_tile)
            )
            tDgBias = tiled_copy_r2s.retile(tDgBias)
            tDgBias = cute.group_modes(tDgBias, 3, cute.rank(tDgBias))
        return {"mPeerOut": None, "bias": tDgBias}

    @cute.jit
    def epi_begin_loop(self, params, epi_tensors, epi_coord):
        tDrBias = None
        if const_expr(epi_tensors["bias"] is not None):
            tDgBias = epi_tensors["bias"][None, None, None, epi_coord]
            tDrBias_in = cute.make_rmem_tensor(tDgBias.layout, params.mBias.element_type)
            cute.autovec_copy(cute.filter_zeros(tDgBias), cute.filter_zeros(tDrBias_in))
            tDrBias = cute.make_rmem_tensor(tDgBias.layout, self.acc_dtype)
            tDrBias.store(tDrBias_in.load().to(self.acc_dtype))
        return {"mPeerOut": None, "bias": tDrBias}

    @cute.jit
    def epi_end_loop(
        self,
        params,
        epi_tensors,
        epi_coord,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        batch_manager,
        tidx,
    ):
        pass

    @cute.jit
    def epi_end(
        self,
        params,
        epi_tensors,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        batch_manager,
        tidx,
    ):
        pass

    def epi_make_aux_out_copy_atom_r2s(self, params, tiled_copy_t2r):
        return gemm_helpers.get_smem_store_op(
            self.aux_out_layout,
            self.aux_out_dtype,
            self.acc_dtype,
            tiled_copy_t2r,
        )

    def epi_make_aux_out_tiled_copy_r2s(self, params, tiled_copy_r2s, tiled_copy_t2r):
        atom = self.epi_make_aux_out_copy_atom_r2s(params, tiled_copy_t2r)
        return cute.make_tiled_copy_S(atom, tiled_copy_r2s)

    def epi_setup_aux_out(
        self,
        params,
        epi_smem_tensors,
        tiled_copy_r2s,
        tiled_copy_t2r,
        tile_coord_mnkl,
        batch_manager,
        tidx,
    ):
        if getattr(params, "mPeerOut", None) is None:
            return None
        sPeerOut = epi_smem_tensors["mPeerOut"]
        tiled_copy_peer_r2s = self.epi_make_aux_out_tiled_copy_r2s(
            params, tiled_copy_r2s, tiled_copy_t2r
        )
        tRS_sPeerOut = tiled_copy_peer_r2s.get_slice(tidx).partition_D(sPeerOut)
        batch_idx = tile_coord_mnkl[3]
        num_peers = const_expr(self.num_peers)
        tiles_per_peer = const_expr(self.tiles_per_peer)
        shard_N = const_expr(self.shard_N)
        tile_N = const_expr(self.cta_tile_shape_mnk[1])
        # When the N tile divides the shard, a whole tile belongs to one peer.
        # Otherwise (a tile wider than the shard, or a shard such as 640 that
        # straddles two 256-wide tiles) every epilogue subtile is routed on its
        # own from its global column; the destination of each peer is then the
        # full shard width of this M tile, indexed by the local subtile.
        aligned = const_expr(shard_N % tile_N == 0)
        tile_m_idx = tile_coord_mnkl[0]
        tile_n_idx = tile_coord_mnkl[1]
        k_idx = tile_coord_mnkl[2]
        l_idx = tile_coord_mnkl[3]
        epi_N = const_expr(cute.size(params.epi_tile_mPeerOut[1]))
        if const_expr(aligned):
            dst_rank = tile_n_idx // tiles_per_peer
        else:
            assert shard_N % epi_N == 0, "peer shard must be a multiple of the epilogue subtile"
            tile_base_col = tile_n_idx * tile_N

        peer_atoms = [
            params.tma_atom_mPeerOut,
            *params.peer_extra_atoms,
        ]
        peer_tensors = [
            params.mPeerOut,
            *params.peer_extra_tensors,
        ]
        copy_fns = []
        for rank in range(num_peers):
            local_tile_n = (
                tile_n_idx - rank * tiles_per_peer if const_expr(aligned) else 0
            )
            local_coord = (
                tile_m_idx,
                local_tile_n,
                k_idx,
                l_idx,
            )
            tile_shape_mn = (
                self.cta_tile_shape_aux_out_mn
                if const_expr(aligned)
                else (self.cta_tile_shape_mnk[0], shard_N)
            )
            copy_fn, _, _ = self.epilog_gmem_copy_and_partition(
                peer_atoms[rank],
                batch_manager.offset_batch_epi(peer_tensors[rank], batch_idx),
                tile_shape_mn,
                params.epi_tile_mPeerOut,
                sPeerOut,
                local_coord,
            )
            copy_fns.append(copy_fn)

        @cute.jit
        def copy_peer_out(src_idx, dst_idx):
            if const_expr(aligned):
                for rank in cutlass.range_constexpr(num_peers):
                    if dst_rank == rank:
                        copy_fns[rank](src_idx=src_idx, dst_idx=dst_idx)
            else:
                global_col = tile_base_col + dst_idx[1] * epi_N
                sub_rank = global_col // shard_N
                local_epi_n = (global_col - sub_rank * shard_N) // epi_N
                local_dst_idx = (dst_idx[0], local_epi_n)
                for rank in cutlass.range_constexpr(num_peers):
                    if sub_rank == rank:
                        copy_fns[rank](
                            src_idx=src_idx,
                            dst_idx=local_dst_idx,
                        )

        return (
            tiled_copy_peer_r2s,
            tRS_sPeerOut,
            copy_peer_out,
        )

    @cute.jit
    def epi_convert_aux_out(
        self,
        tRS_rAuxOut,
        sr_seed,
        tidx,
        tile_coord_mnkl,
        num_prev_subtiles,
        epi_idx,
    ):
        tRS_rPeerOut = cute.make_rmem_tensor_like(tRS_rAuxOut, self.aux_out_dtype)
        tRS_rPeerOut.store(tRS_rAuxOut.load().to(self.aux_out_dtype))
        return tRS_rPeerOut

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        tDrBias = epi_loop_tensors["bias"]
        if const_expr(tDrBias is not None):
            for i in cutlass.range(cute.size(tDrBias), unroll_full=True):
                tRS_rD[i] += tDrBias[i]
        return tRS_rD


class PeerWriteGemm(PeerWriteMixin, PersistentGemm):
    pass


@cache
def _compile_peer_write(
    a_dtype,
    b_dtype,
    d_dtype,
    peer_dtype,
    a_major,
    b_major,
    d_major,
    peer_major,
    tile_shape_mnk,
    cluster_shape_mnk,
    pingpong,
    persistent,
    is_dynamic_persistent,
    device_capacity,
    rounding_mode,
    num_peers,
    shard_N,
    has_bias,
):
    assert a_dtype is BFloat16
    assert b_dtype is BFloat16
    assert d_dtype in (None, BFloat16)
    assert peer_dtype is BFloat16
    assert device_capacity[0] in (10, 11)
    assert pingpong is False
    assert persistent is True
    assert is_dynamic_persistent is True
    assert rounding_mode == RoundingMode.RN
    assert num_peers >= 1
    tile_N = tile_shape_mnk[1]
    mA, mB, mD, m, n, _, l = make_fake_gemm_tensors(
        a_dtype,
        b_dtype,
        d_dtype,
        a_major,
        b_major,
        d_major,
    )
    peer_leading = 1 if peer_major == "n" else 0

    def _fake_peer():
        return make_fake_tensor(
            peer_dtype,
            (m, shard_N, l),
            leading_dim=peer_leading,
            divisibility=div_for_dtype(peer_dtype),
        )

    mPeerOut = _fake_peer()
    peer_extras = tuple(_fake_peer() for _ in range(num_peers - 1))
    tiles_per_peer = shard_N // tile_N if shard_N >= tile_N else 0
    mBias = (
        make_fake_tensor(BFloat16, (n,), leading_dim=0, divisibility=div_for_dtype(BFloat16))
        if has_bias
        else None
    )
    epi_args = PeerWriteGemm.EpilogueArguments(
        mPeerOut=mPeerOut,
        peer_extras=peer_extras,
        tiles_per_peer=tiles_per_peer,
        shard_N=shard_N,
        rounding_mode=rounding_mode,
        mBias=mBias,
    )
    gemm_obj = PeerWriteGemm(
        Float32,
        BFloat16,
        tile_shape_mnk,
        cluster_shape_mnk,
        use_clc_persistence=True,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    with _cutlass_442_converter_compatibility():
        return cute.compile(
            gemm_obj,
            mA,
            mB,
            mD,
            epi_args,
            mA,
            mA,
            mA,
            mA,
            mA,
            mA,
            mA,
            mA,
            make_fake_scheduler_args(),
            stream,
            options="--enable-tvm-ffi",
        )


@dataclass(frozen=True)
class PeerWriteLaunch:
    """A compiled peer-write GEMM bound to ``B``, ``D``, the peer outputs,
    the bias and the tile schedule.

    ``launch(A)`` runs ``A @ B.T (+ bias)`` for a BF16 ``A`` with the shape
    and strides the launch was made for. Only ``A`` is checked per call.
    """

    compiled_fn: Callable
    B_p: Tensor
    D_p: Optional[Tensor]
    epi_args: object
    scheduler_args: object
    a_shape: Tuple[int, ...]
    a_stride: Tuple[int, ...]
    device: torch.device
    output_ranges: Tuple[Tuple[int, int], ...]

    def check(self, A: Tensor) -> None:
        if not isinstance(A, Tensor) or A.dtype != torch.bfloat16:
            raise TypeError("peer_write requires a BF16 A tensor")
        if (
            tuple(A.shape) != self.a_shape
            or A.stride() != self.a_stride
            or A.device != self.device
        ):
            raise ValueError(
                f"A must have shape {self.a_shape} and strides {self.a_stride} "
                f"on {self.device}, got {tuple(A.shape)} and {A.stride()} on "
                f"{A.device}"
            )
        start, end = _storage_byte_range(A)
        if any(start < stop and begin < end for begin, stop in self.output_ranges):
            raise ValueError("dense and peer outputs must not overlap inputs or one another")

    def __call__(self, A: Tensor) -> None:
        self.check(A)
        A_p = A.permute(1, 2, 0) if A.ndim == 3 else A
        self.compiled_fn(
            A_p,
            self.B_p,
            self.D_p,
            self.epi_args,
            A_p,
            A_p,
            A_p,
            A_p,
            A_p,
            A_p,
            A_p,
            A_p,
            self.scheduler_args,
        )


def make_peer_write_launch(
    B: Tensor,
    D: Optional[Tensor],
    PeerOuts,
    *,
    a_shape: Tuple[int, ...],
    a_stride: Optional[Tuple[int, ...]] = None,
    bias: Optional[Tensor] = None,
    local_rank: int = 0,
    tile_M: int = PEER_TILE_M,
    tile_N: int = PEER_TILE_N,
    cluster_M: int = PEER_CLUSTER_M,
    cluster_N: int = PEER_CLUSTER_N,
    tile_K: Optional[int] = None,
    pingpong: bool = False,
    persistent: bool = True,
    is_dynamic_persistent: bool = True,
    max_swizzle_size: int = 8,
    rounding_mode: int = RoundingMode.RN,
) -> PeerWriteLaunch:
    """Compile and bind a reusable BF16 ``A @ B.T (+ bias)`` peer-write launch.

    ``A`` is described by ``a_shape`` and ``a_stride`` (contiguous when
    omitted) and passed on every call of the returned launch. Output column
    block ``p`` (``N / len(PeerOuts)`` columns wide) is stored into
    ``PeerOuts[p]``. ``bias`` is an optional contiguous ``(N,)`` BF16 vector
    added to the FP32 accumulator before the single rounding to BF16, as
    ``torch.nn.functional.linear`` does. ``local_rank`` is this rank's index
    among the peers: it only rotates the order in which output tiles, and
    therefore destination peers, are visited, so that the ranks of a group do
    not all write into the same peer at the same time.
    """
    if pingpong:
        raise ValueError("peer_write is specialized to pingpong=False")
    if not persistent:
        raise ValueError("peer_write is specialized to persistent=True")
    if not is_dynamic_persistent:
        raise ValueError(
            "peer_write is specialized to is_dynamic_persistent=True (CLC)"
        )
    if rounding_mode != RoundingMode.RN:
        raise ValueError("peer_write only supports round-to-nearest")
    if not isinstance(B, Tensor) or B.dtype != torch.bfloat16:
        raise TypeError("peer_write requires a BF16 B tensor")
    if a_stride is None:
        A = torch.empty(a_shape, dtype=torch.bfloat16, device="meta")
    else:
        A = torch.empty_strided(a_shape, a_stride, dtype=torch.bfloat16, device="meta")
    if D is not None and D.dtype != torch.bfloat16:
        raise TypeError("peer_write requires a BF16 dense output tensor")
    if bias is not None and (not isinstance(bias, Tensor) or bias.dtype != torch.bfloat16):
        raise TypeError("peer_write bias must be a BF16 tensor")
    if tile_M <= 0 or tile_N <= 0 or (tile_K is not None and tile_K <= 0):
        raise ValueError(
            f"tile dimensions must be positive, got {(tile_M, tile_N, tile_K)}"
        )
    if cluster_M <= 0 or cluster_N <= 0:
        raise ValueError(
            f"cluster dimensions must be positive, got {(cluster_M, cluster_N)}"
        )
    if max_swizzle_size <= 0:
        raise ValueError(
            f"max_swizzle_size must be positive, got {max_swizzle_size}"
        )

    if isinstance(PeerOuts, Tensor):
        PeerOuts = (PeerOuts,)
    else:
        PeerOuts = tuple(PeerOuts)
    if not PeerOuts:
        raise ValueError("peer_write requires at least one peer output")
    if any(not isinstance(peer, Tensor) for peer in PeerOuts):
        raise TypeError("peer_write peer outputs must be torch tensors")
    if any(peer.dtype != torch.bfloat16 for peer in PeerOuts):
        raise TypeError("peer_write requires BF16 peer outputs")
    all_tensors = (B, *(() if D is None else (D,)), *PeerOuts)
    if any(tensor.device != B.device for tensor in all_tensors):
        raise ValueError("B, D, and all peer outputs must share one device")
    if B.device.type != "cuda":
        raise ValueError(f"peer_write requires CUDA tensors, got {B.device}")

    A_p, B_p, D_p = perm3d(A, B, D)
    PeerOuts_p = tuple(
        peer.permute(1, 2, 0) if peer.ndim == 3 else peer for peer in PeerOuts
    )
    normalized = (A_p, B_p, *(() if D_p is None else (D_p,)), *PeerOuts_p)
    if any(tensor.ndim not in (2, 3) for tensor in normalized):
        raise ValueError("peer_write tensors must all be 2-D or all be 3-D")
    if any(tensor.ndim != A_p.ndim for tensor in normalized):
        raise ValueError("peer_write tensors must use the same batch dimensionality")
    if any(not _is_non_overlapping_dense(tensor) for tensor in normalized):
        raise ValueError(
            "peer_write tensors must use non-overlapping dense storage"
        )
    if any(tensor.stride(0) != 1 and tensor.stride(1) != 1 for tensor in normalized):
        raise ValueError(
            "each peer_write matrix must be contiguous in its M/N or K dimension"
        )
    matrix_batch = int(A_p.shape[2]) if A_p.ndim == 3 else 1
    if A_p.shape[0] <= 0 or A_p.shape[1] <= 0 or matrix_batch <= 0:
        raise ValueError(f"peer_write requires non-empty A, got {tuple(A.shape)}")
    if B_p.shape[0] <= 0 or B_p.shape[1] != A_p.shape[1]:
        raise ValueError(
            "GEMM K dimensions do not match: "
            f"A={tuple(A.shape)}, B={tuple(B.shape)}"
        )
    if B_p.ndim == 3 and B_p.shape[2] != matrix_batch:
        raise ValueError("A and B batch dimensions do not match")
    expected_output_shape = (
        (A_p.shape[0], B_p.shape[0], matrix_batch)
        if A_p.ndim == 3
        else (A_p.shape[0], B_p.shape[0])
    )
    if D_p is not None and tuple(D_p.shape) != expected_output_shape:
        raise ValueError(
            f"dense output shape must be {expected_output_shape}, got "
            f"{tuple(D_p.shape)}"
        )
    for peer_p in PeerOuts_p:
        if peer_p.shape[0] != A_p.shape[0]:
            raise ValueError("peer output M dimension must match A")
        if peer_p.ndim == 3 and peer_p.shape[2] != matrix_batch:
            raise ValueError("peer output batch dimension must match A")

    writable = (*(() if D is None else (D,)), *PeerOuts)
    readable = (B,) if bias is None else (B, bias)
    for index, output in enumerate(writable):
        others = (*readable, *writable[:index], *writable[index + 1 :])
        if any(_tensors_overlap(output, other) for other in others):
            raise ValueError(
                "dense and peer outputs must not overlap inputs or one another"
            )

    a_major = get_major(A_p, "m", "k")
    b_major = get_major(B_p, "n", "k")
    d_major = get_major(D_p, "m", "n") if D_p is not None else None
    peer_major = get_major(PeerOuts_p[0], "m", "n")
    peer_shard_N = PeerOuts_p[0].shape[1]
    for peer_p in PeerOuts_p[1:]:
        if get_major(peer_p, "m", "n") != peer_major:
            raise ValueError("all peer outputs must use the same memory major")
        if peer_p.shape != PeerOuts_p[0].shape:
            raise ValueError("all peer outputs must have the same shape")

    device_capacity = get_device_capacity(B.device)
    if device_capacity[0] not in (10, 11):
        raise ValueError("peer_write does not support this GPU")
    tile_shape_mnk = (
        (tile_M, tile_N, tile_K) if tile_K is not None else (tile_M, tile_N)
    )
    cluster_shape_mnk = (cluster_M, cluster_N, 1)
    num_peers = len(PeerOuts_p)
    n_total = D_p.shape[1] if D_p is not None else B_p.shape[0]
    if n_total != B_p.shape[0]:
        raise ValueError("dense output N dimension must match B")
    if num_peers * peer_shard_N != n_total:
        raise ValueError(
            "peer output shards do not cover the GEMM N dimension: "
            f"{num_peers} * {peer_shard_N} != {n_total}"
        )
    if peer_shard_N % tile_N and peer_shard_N % PEER_EPI_TILE_N:
        raise ValueError(
            f"peer shard N={peer_shard_N} must be a multiple of the tile N "
            f"({tile_N}) or of the epilogue subtile ({PEER_EPI_TILE_N})"
        )
    if not 0 <= local_rank < num_peers:
        raise ValueError(f"local_rank must be in [0, {num_peers}), got {local_rank}")
    if bias is not None:
        if bias.device != B.device or tuple(bias.shape) != (n_total,):
            raise ValueError(
                f"peer_write bias must have shape ({n_total},) on {B.device}, "
                f"got {tuple(bias.shape)} on {bias.device}"
            )
        if not bias.is_contiguous() or bias.data_ptr() % 16:
            raise ValueError("peer_write bias must be contiguous and 16-byte aligned")
        if n_total % tile_N:
            raise ValueError(
                f"a bias requires N={n_total} to be a multiple of the tile N ({tile_N})"
            )

    compiled_fn = _compile_peer_write(
        BFloat16,
        BFloat16,
        BFloat16 if D is not None else None,
        BFloat16,
        a_major,
        b_major,
        d_major,
        peer_major,
        tile_shape_mnk,
        cluster_shape_mnk,
        pingpong,
        persistent,
        is_dynamic_persistent,
        device_capacity,
        rounding_mode,
        num_peers,
        peer_shard_N,
        bias is not None,
    )

    cluster_size = cluster_M * cluster_N
    device_id = (
        B.device.index if B.device.index is not None else torch.cuda.current_device()
    )
    max_active_clusters = get_max_active_clusters(
        cluster_size,
        device_capacity=device_capacity,
        device_id=device_id,
    )

    epi_args = PeerWriteGemm.EpilogueArguments(
        mPeerOut=PeerOuts_p[0],
        peer_extras=PeerOuts_p[1:],
        tiles_per_peer=None,
        shard_N=None,
        rounding_mode=None,
        mBias=bias,
    )
    n_rotation = peer_n_rotation(local_rank, num_peers, n_total, tile_N, cluster_N)
    scheduler_args = make_scheduler_args(
        max_active_clusters, max_swizzle_size, n_rotation
    )
    return PeerWriteLaunch(
        compiled_fn=compiled_fn,
        B_p=B_p,
        D_p=D_p,
        epi_args=epi_args,
        scheduler_args=scheduler_args,
        a_shape=tuple(int(size) for size in A.shape),
        a_stride=tuple(A.stride()),
        device=B.device,
        output_ranges=tuple(_storage_byte_range(output) for output in writable),
    )


def prepare_peer_write(
    A: Tensor,
    B: Tensor,
    D: Optional[Tensor],
    PeerOuts,
    *,
    bias: Optional[Tensor] = None,
    local_rank: int = 0,
    tile_M: int = PEER_TILE_M,
    tile_N: int = PEER_TILE_N,
    cluster_M: int = PEER_CLUSTER_M,
    cluster_N: int = PEER_CLUSTER_N,
    tile_K: Optional[int] = None,
    pingpong: bool = False,
    persistent: bool = True,
    is_dynamic_persistent: bool = True,
    max_swizzle_size: int = 8,
    rounding_mode: int = RoundingMode.RN,
) -> Callable[[], None]:
    """Compile and bind a BF16 ``A @ B.T (+ bias)`` peer-write launch on ``A``.

    See :func:`make_peer_write_launch` for the arguments.
    """
    if not isinstance(A, Tensor):
        raise TypeError("peer_write requires a BF16 A tensor")
    launch = make_peer_write_launch(
        B,
        D,
        PeerOuts,
        a_shape=tuple(A.shape),
        a_stride=tuple(A.stride()),
        bias=bias,
        local_rank=local_rank,
        tile_M=tile_M,
        tile_N=tile_N,
        cluster_M=cluster_M,
        cluster_N=cluster_N,
        tile_K=tile_K,
        pingpong=pingpong,
        persistent=persistent,
        is_dynamic_persistent=is_dynamic_persistent,
        max_swizzle_size=max_swizzle_size,
        rounding_mode=rounding_mode,
    )
    launch.check(A)
    return partial(launch, A)


def peer_write(
    A: Tensor,
    B: Tensor,
    D: Optional[Tensor],
    PeerOuts,
    *,
    bias: Optional[Tensor] = None,
    local_rank: int = 0,
    tile_M: int = PEER_TILE_M,
    tile_N: int = PEER_TILE_N,
    cluster_M: int = PEER_CLUSTER_M,
    cluster_N: int = PEER_CLUSTER_N,
    tile_K: Optional[int] = None,
    pingpong: bool = False,
    persistent: bool = True,
    is_dynamic_persistent: bool = True,
    max_swizzle_size: int = 8,
    rounding_mode: int = RoundingMode.RN,
) -> None:
    """Compile if needed, then launch a BF16 ``A @ B.T (+ bias)`` peer write."""

    prepare_peer_write(
        A,
        B,
        D,
        PeerOuts,
        bias=bias,
        local_rank=local_rank,
        tile_M=tile_M,
        tile_N=tile_N,
        cluster_M=cluster_M,
        cluster_N=cluster_N,
        tile_K=tile_K,
        pingpong=pingpong,
        persistent=persistent,
        is_dynamic_persistent=is_dynamic_persistent,
        max_swizzle_size=max_swizzle_size,
        rounding_mode=rounding_mode,
    )()


__all__ = [
    "IMPLEMENTATION_REVISION",
    "PEER_CLUSTER_M",
    "PEER_CLUSTER_N",
    "PEER_EPI_TILE_N",
    "PEER_TILE_M",
    "PEER_TILE_N",
    "PeerWriteLaunch",
    "make_peer_write_launch",
    "peer_n_rotation",
    "peer_write",
    "prepare_peer_write",
]
