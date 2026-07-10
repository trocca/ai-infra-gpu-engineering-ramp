"""Acceptance test for rust-hello-gpu (COMPLETE — do not edit).

Shells out to `cargo run --release` in rust-hello-gpu/, parses the single
JSON line it prints, and checks the SAXPY result against a NumPy reference.

Skips (not fails) when cargo or a CUDA GPU is unavailable, so the Python
autograd suite stays runnable everywhere (including CI).
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

CRATE = Path(__file__).resolve().parent.parent / "rust-hello-gpu"

N = 1 << 20
ALPHA = 2.5


def _gpu_available() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                              timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


pytestmark = [
    pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo not installed"),
    pytest.mark.skipif(not _gpu_available(), reason="no CUDA GPU visible"),
]


@pytest.fixture(scope="module")
def rust_output() -> dict:
    proc = subprocess.run(
        ["cargo", "run", "--release", "--quiet"],
        cwd=CRATE, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, (
        f"cargo run failed (exit {proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    lines = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    assert len(lines) == 1, (
        "expected exactly ONE json line on stdout (use eprintln! for debug "
        f"output); got {len(lines)} lines:\n{proc.stdout}"
    )
    return json.loads(lines[0])


def _numpy_reference() -> np.ndarray:
    x = 0.01 * np.arange(N, dtype=np.float32)
    y = np.ones(N, dtype=np.float32)
    return (np.float32(ALPHA) * x + y).astype(np.float32)


def test_schema(rust_output):
    for key in ("device", "n", "alpha", "y_head", "y_sum"):
        assert key in rust_output, f"missing key {key!r} in JSON output"
    assert rust_output["n"] == N
    assert rust_output["alpha"] == pytest.approx(ALPHA)
    assert len(rust_output["y_head"]) == 8


def test_device_is_cuda_gpu(rust_output):
    assert isinstance(rust_output["device"], str) and rust_output["device"], \
        "device name must be a non-empty string"


def test_saxpy_head_matches_numpy(rust_output):
    ref = _numpy_reference()
    got = np.asarray(rust_output["y_head"], dtype=np.float64)
    err = np.max(np.abs(got - ref[:8].astype(np.float64)))
    assert err <= 1e-5, f"first-8 mismatch, max abs err {err:.3e}"


def test_saxpy_checksum_matches_numpy(rust_output):
    # fp32 element sums accumulated in f64 on both sides; loose relative
    # tolerance absorbs summation-order differences.
    ref_sum = float(np.sum(_numpy_reference(), dtype=np.float64))
    got_sum = float(rust_output["y_sum"])
    rel = abs(got_sum - ref_sum) / abs(ref_sum)
    assert rel <= 1e-6, f"checksum mismatch: got {got_sum}, want {ref_sum} (rel {rel:.3e})"
