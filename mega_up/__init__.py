# Copyright (c) The Mega-UP Authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Two fused SM100 GEMM/Ulysses communication kernels."""

from .ops import (
    PreparedHeadShardPullGemm,
    PreparedHeadShardPullGemmSBHD,
    PreparedHeadShardPullGemmTHD,
    PreparedPeerWriteGemm,
    PreparedPeerWriteGemmSBHD,
    PreparedPeerWriteGemmTHD,
    THDLayout,
    headshard_pull_gemm_bshd,
    headshard_pull_gemm_sbhd,
    headshard_pull_gemm_thd,
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
from .runtime import (
    DEFAULT_MAX_ROWS,
    SUPPORTED_UP_DEGREES,
    MegaUPRuntime,
    initialize_runtime,
    supports_shape,
    unsupported_shape_reason,
)

__version__ = "0.2.0"

__all__ = [
    "DEFAULT_MAX_ROWS",
    "SUPPORTED_UP_DEGREES",
    "MegaUPRuntime",
    "PreparedHeadShardPullGemm",
    "PreparedHeadShardPullGemmSBHD",
    "PreparedHeadShardPullGemmTHD",
    "PreparedPeerWriteGemm",
    "PreparedPeerWriteGemmSBHD",
    "PreparedPeerWriteGemmTHD",
    "THDLayout",
    "__version__",
    "headshard_pull_gemm_bshd",
    "headshard_pull_gemm_sbhd",
    "headshard_pull_gemm_thd",
    "initialize_runtime",
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
    "supports_shape",
    "unsupported_shape_reason",
]
