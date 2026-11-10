"""Run the distributed correctness runner under torchrun.

Each case needs UP supported GPUs on one node; cases with too few GPUs are
skipped. Select with ``pytest -m gpu``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.gpu

REPO = Path(__file__).resolve().parents[1]

# (heads, UP, bias) of square projections at head_dim 128.
CASES = (
    (16, 8, False),  # hidden 2048: 256-column shards, two peers per N tile
    (40, 8, True),  # hidden 5120: 640-column shards straddle N tiles
    (10, 2, True),  # hidden 1280: 640-column shards straddle N tiles at UP=2
    (32, 4, False),  # hidden 4096: bulk pull
    (18, 2, True),  # hidden 2304: odd number of 256-column N tiles
    (64, 2, False),  # hidden 8192: 4096-column shards pulled in chunks
)

# (runner shape arguments, UP, bias) of rectangular projections. The
# peer-write weight is [peer_out_heads*D, heads*D] and the pull weight
# [pull_out_heads*D, pull_in_heads*D]; D is 128 unless given.
RECTANGULAR_CASES = (
    # peer-write: fused GQA QKV [6144, 4096]; pull: [4096, 4096]
    ("--heads 32 --peer-out-heads 48 --pull-in-heads 32", 8, True),
    # peer-write: K/V [1024, 2048]; pull: [2048, 1024]
    ("--heads 16 --peer-out-heads 8", 2, True),
    # peer-write: [8192, 2048]; pull: [2048, 8192]
    (
        "--heads 16 --peer-out-heads 64 --pull-in-heads 64 --pull-out-heads 16",
        4,
        False,
    ),
    # peer-write: [2560, 1040], K not a multiple of 64; pull: [1280, 2560]
    (
        "--head-dim 80 --heads 13 --peer-out-heads 32 --pull-in-heads 32 "
        "--pull-out-heads 16",
        2,
        False,
    ),
    # peer-write and pull: packed QKV [6144, 2048]; the exchanged widths 6144
    # and 2048 each get their own runtime on the process group
    (
        "--heads 16 --peer-out-heads 48 --pull-in-heads 16 --pull-out-heads 48 "
        "--runtime-per-width",
        4,
        True,
    ),
)


def _supported_device_count() -> int:
    if not torch.cuda.is_available():
        return 0
    if torch.cuda.get_device_capability(0)[0] not in (10, 11):
        return 0
    return torch.cuda.device_count()


def _run(up: int, bias: bool, shape_args: str) -> None:
    if _supported_device_count() < up:
        pytest.skip(f"needs {up} supported GPUs")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={up}",
        "tests/run_distributed_correctness.py",
        "--up",
        str(up),
        *shape_args.split(),
        *(["--bias"] if bias else []),
    ]
    completed = subprocess.run(
        command, cwd=REPO, capture_output=True, text=True, timeout=1800
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output[-6000:]
    assert "[CORRECTNESS_RESULT]" in completed.stdout, output[-6000:]


@pytest.mark.parametrize(("heads", "up", "bias"), CASES)
def test_distributed_correctness(heads: int, up: int, bias: bool) -> None:
    _run(up, bias, f"--heads {heads} --head-dim 128")


@pytest.mark.parametrize(("shape_args", "up", "bias"), RECTANGULAR_CASES)
def test_rectangular_distributed_correctness(
    shape_args: str, up: int, bias: bool
) -> None:
    _run(up, bias, shape_args)
