"""rusty_kernels — fused softmax and LayerNorm for PyTorch, in Rust.

Rust host (cudarc) + Rust GPU kernels (Rust-CUDA -> PTX), bound via PyO3.

Public API:
    rusty_kernels.softmax(x, dim=-1)
    rusty_kernels.layer_norm(x, weight, bias, eps=1e-5)

Both accept CUDA tensors and participate in autograd (LayerNorm fully;
softmax per the week's progress — see ops.py).
"""

from .ops import layer_norm, softmax

__all__ = ["softmax", "layer_norm"]
__version__ = "0.1.0"
