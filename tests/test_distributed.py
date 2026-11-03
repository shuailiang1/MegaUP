"""Run the distributed correctness runner under torchrun.

Each case needs UP SM100/SM110 GPUs on one node; cases with too few GPUs are
skipped. Select with ``pytest -m sm100``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.sm100

REPO = Path(__file__).resolve().parents[1]

# (heads, UP, bias) at head_dim 128.
CASES = (
    (16, 8, False),  # hidden 2048: 256-column shards, two peers per N tile
    (40, 8, True),  # hidden 5120: 640-column shards straddle N tiles
    (10, 2, True),  # hidden 1280: 640-column shards straddle N tiles at UP=2
    (32, 4, False),  # hidden 4096: bulk pull
    (18, 2, True),  # hidden 2304: odd number of 256-column N tiles
    (64, 2, False),  # hidden 8192: 4096-column shards pulled in chunks
)


def _sm100_device_count() -> int:
    if not torch.cuda.is_available():
        return 0
    if torch.cuda.get_device_capability(0)[0] not in (10, 11):
        return 0
    return torch.cuda.device_count()


@pytest.mark.parametrize(("heads", "up", "bias"), CASES)
def test_distributed_correctness(heads: int, up: int, bias: bool) -> None:
    if _sm100_device_count() < up:
        pytest.skip(f"needs {up} SM100/SM110 GPUs")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={up}",
        "tests/run_distributed_correctness.py",
        "--up",
        str(up),
        "--heads",
        str(heads),
        "--head-dim",
        "128",
        *(["--bias"] if bias else []),
    ]
    completed = subprocess.run(
        command, cwd=REPO, capture_output=True, text=True, timeout=1800
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output[-6000:]
    assert "[CORRECTNESS_RESULT]" in completed.stdout, output[-6000:]
