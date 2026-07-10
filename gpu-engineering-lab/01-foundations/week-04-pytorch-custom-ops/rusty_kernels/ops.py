"""autograd.Function wrappers around the Rust extension (Days 1-3).

The compiled extension `rusty_kernels._rusty` (PyO3, see ext/src/lib.rs)
exposes raw launch entry points taking DEVICE ADDRESSES:

    _rusty.passthrough(x_ptr, y_ptr, n, dtype)
    _rusty.softmax_forward(x_ptr, y_ptr, rows, cols, dtype)
    _rusty.layernorm_forward(x_ptr, w_ptr, b_ptr, y_ptr, mean_ptr, rstd_ptr,
                             rows, cols, eps, dtype)
    _rusty.layernorm_backward(dy_ptr, x_ptr, w_ptr, mean_ptr, rstd_ptr,
                              dx_ptr, dw_ptr, db_ptr, rows, cols, dtype)

THE PYTHON SIDE OWNS THE SAFETY CONTRACT (this is the data_ptr half of the
Day-1 binding decision): every tensor handed to _rusty must be CUDA,
contiguous, the right dtype/shape, and kept alive across the call — and
torch's current stream must be synchronized first (`_pre_launch`), because
v0 runs kernels on the extension's own stream. Allocation stays in Python
(torch.empty_*), so PyTorch's caching allocator keeps owning all memory.
"""

from __future__ import annotations

import torch

from rusty_kernels import _rusty

_DTYPE_TAG = {torch.float32: "f32", torch.float64: "f64", torch.float16: "f16"}


def _tag(t: torch.Tensor) -> str:
    try:
        return _DTYPE_TAG[t.dtype]
    except KeyError:
        raise TypeError(f"rusty_kernels: unsupported dtype {t.dtype}") from None


def _pre_launch(*tensors: torch.Tensor) -> None:
    """Enforce the safety contract, then sync torch's stream (v0 design —
    see README 'Architecture'). COMPLETE."""
    for t in tensors:
        assert t.is_cuda, "rusty_kernels: CPU tensor passed to a GPU op"
        assert t.is_contiguous(), "rusty_kernels: non-contiguous tensor"
    torch.cuda.current_stream().synchronize()


class _FusedSoftmax(torch.autograd.Function):
    """Fused softmax over the last dimension.

    TODO(Day 2) forward:
      * flatten leading dims to (rows, cols): x2 = x.reshape(-1, shape[-1])
        .contiguous()
      * allocate y2 = torch.empty_like(x2), _pre_launch(x2, y2)
      * _rusty.softmax_forward(x2.data_ptr(), y2.data_ptr(), rows, cols,
        _tag(x2))
      * save the OUTPUT for backward (softmax backward needs y, not x —
        know why), return y2.reshape(original shape)

    TODO(Day 2) backward (torch-ops fallback until the stretch kernel):
      * dx = (dy - (dy * y).sum(dim=-1, keepdim=True)) * y
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class _FusedLayerNorm(torch.autograd.Function):
    """Fused LayerNorm over the last dimension.

    TODO(Day 3) forward:
      * flatten to (rows, cols); validate weight/bias are (cols,) and same
        dtype as x
      * allocate y (like x) and mean/rstd as torch.empty(rows, dtype=x.dtype,
        device=x.device) — the kernels write stats at input precision
      * _pre_launch(...), call _rusty.layernorm_forward(...)
      * ctx.save_for_backward(x2, weight, mean, rstd); return y reshaped

    TODO(Day 3) backward:
      * allocate dx (like x), dweight/dbias (like weight)
      * _pre_launch(...), call _rusty.layernorm_backward(...)
      * return (dx_reshaped, dweight, dbias, None)   # None matches eps

    float64 gradcheck runs THIS Function — which is why the kernel crate has
    _f64 twins of every kernel.

    fp16 note: the kernel crate ships no layernorm _f16 variant. To pass the
    fp16 forward tests, either add one (escape hatch is fine) or upcast to
    f32 here and cast back — measure and document what the upcast costs.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor,
                bias: torch.Tensor, eps: float) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        raise NotImplementedError


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Fused softmax. TODO(Day 2): support dim=-1 (or dim == x.ndim - 1)
    only — assert that, then apply _FusedSoftmax."""
    raise NotImplementedError


def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
               eps: float = 1e-5) -> torch.Tensor:
    """Fused LayerNorm over the last dim. TODO(Day 3): apply _FusedLayerNorm."""
    raise NotImplementedError


def passthrough(x: torch.Tensor) -> torch.Tensor:
    """Day 1 smoke test: returns a copy of x made by YOUR kernel. COMPLETE
    plumbing — it starts passing as soon as passthrough_f32 has a body."""
    x = x.contiguous()
    y = torch.empty_like(x)
    _pre_launch(x, y)
    _rusty.passthrough(x.data_ptr(), y.data_ptr(), x.numel(), _tag(x))
    return y
