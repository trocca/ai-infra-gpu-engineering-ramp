"""Day 4 — per-channel symmetric INT8 weight packing (the "W8" in W8A16).

Contract (tests/test_quant.py depends on):

    quantize_int8(w) -> (w_int8, scales)
        w:      (out_features, in_features) fp16/fp32
        w_int8: same shape, torch.int8
        scales: (out_features,) fp32 — PER OUTPUT CHANNEL (per row of W).
        Symmetric: w ~ w_int8 * scales[:, None], no zero-point.

    dequantize_int8(w_int8, scales, dtype) -> dense approx of w

    quantization_error(w, w_int8, scales) -> float   (provided, COMPLETE)
"""

from __future__ import annotations

import torch


def quantize_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """TODO(Day 4):
      1. scales = w.abs().max(dim=1).values / 127.0   (fp32; clamp min 1e-8
         so an all-zero row doesn't divide by zero).
      2. w_int8 = (w / scales[:, None]).round().clamp(-127, 127).to(torch.int8)
         (clamp to -127 not -128: keeps the grid symmetric around 0).

    Key idea — why PER-CHANNEL: one outlier weight in the tensor would set a
    single global scale, crushing every other channel into a handful of int
    levels. Per-row scales isolate outliers; the extra (out_features,) floats
    are noise next to the 2x weight-byte saving.
    """
    raise NotImplementedError("Day 4: implement quantize_int8")


def dequantize_int8(w_int8: torch.Tensor, scales: torch.Tensor,
                    dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """TODO(Day 4): w_int8.to(fp32) * scales[:, None], cast to dtype.
    This is the CPU-side reference — the Triton kernel does the same thing in
    registers, tile by tile, without ever writing the dense fp16 W to HBM."""
    raise NotImplementedError("Day 4: implement dequantize_int8")


def quantization_error(w: torch.Tensor, w_int8: torch.Tensor,
                       scales: torch.Tensor) -> float:
    """COMPLETE. Relative Frobenius error ||W - W_hat|| / ||W||."""
    w32 = w.float()
    w_hat = w_int8.float() * scales.float()[:, None]
    return ((w32 - w_hat).norm() / w32.norm()).item()
