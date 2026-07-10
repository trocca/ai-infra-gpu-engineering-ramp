"""Tensor reverse-mode autodiff engine on NumPy (Day 2).

Same graph machinery as the scalar engine, but every node holds an
np.ndarray and every backward is a vector-Jacobian product implemented with
NumPy ops.

THE key difficulty this file exists to teach: broadcasting-aware backward.
If a (3,) bias was broadcast against a (32, 3) activation in the forward
pass, then in the backward pass the (32, 3) upstream gradient must be
SUMMED over the broadcast axes to recover a (3,) gradient. Every binary op
must route its parent gradients through `_unbroadcast`.

Contract relied on by tests/test_tensor_grads.py:
  * Tensor.data -> np.ndarray (float64 by default so gradchecks are tight)
  * Tensor.grad -> np.ndarray of same shape, zeros until backward
  * requires_grad + leaf semantics like PyTorch (grads only accumulate
    where requires_grad=True; intermediate nodes still propagate)
"""

from __future__ import annotations

import numpy as np


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Reduce `grad` down to `shape` by summing over broadcast dimensions.

    TODO(Day 2):
      1. Sum over leading axes that were prepended by broadcasting
         (grad.ndim > len(shape)).
      2. For each remaining axis where shape[i] == 1 but grad.shape[i] > 1,
         sum over that axis with keepdims=True.
      3. Return an array whose .shape == shape exactly.

    This function is the heart of the week. Get it right and every binary
    op's backward becomes three lines.
    """
    raise NotImplementedError


class Tensor:
    def __init__(self, data, requires_grad: bool = False, _children: tuple = (), _op: str = ""):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = tuple(_children)
        self._op = _op

    # -- shape sugar ----------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    # ------------------------------------------------------------------
    # Binary ops (all must be broadcasting-aware via _unbroadcast)
    # ------------------------------------------------------------------
    def __add__(self, other) -> "Tensor":
        """TODO(Day 2): elementwise add. Backward: route the upstream grad
        to each parent through _unbroadcast(upstream, parent.shape)."""
        raise NotImplementedError

    def __mul__(self, other) -> "Tensor":
        """TODO(Day 2): elementwise multiply. Backward: scale upstream by
        the OTHER operand's data, then _unbroadcast."""
        raise NotImplementedError

    def __sub__(self, other) -> "Tensor":
        """TODO(Day 2)."""
        raise NotImplementedError

    def __truediv__(self, other) -> "Tensor":
        """TODO(Day 2): quotient rule, or compose mul and pow."""
        raise NotImplementedError

    def __pow__(self, exponent: float) -> "Tensor":
        """TODO(Day 2): const exponent only."""
        raise NotImplementedError

    def matmul(self, other) -> "Tensor":
        """out = self @ other for 2-D operands (batched matmul is stretch).

        TODO(Day 2):
          dL/dA = dL/dOut @ B.T
          dL/dB = A.T @ dL/dOut
        Derive these on paper before typing them — you will be asked to in
        interviews.
        """
        raise NotImplementedError

    def __matmul__(self, other) -> "Tensor":
        return self.matmul(other)

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------
    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        """TODO(Day 2): backward broadcasts the upstream grad back to
        self.shape (np.broadcast_to after re-inserting reduced axes)."""
        raise NotImplementedError

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        """TODO(Day 2): like sum, scaled by 1/N over the reduced elements."""
        raise NotImplementedError

    def max(self, axis=None, keepdims: bool = False) -> "Tensor":
        """TODO(Day 2, needed for stable softmax): backward routes gradient
        only to the argmax positions (ties: split or pick-first both pass
        the tests, which avoid exact ties)."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Unary / shape ops
    # ------------------------------------------------------------------
    def relu(self) -> "Tensor":
        """TODO(Day 2): elementwise gate."""
        raise NotImplementedError

    def exp(self) -> "Tensor":
        """TODO(Day 2)."""
        raise NotImplementedError

    def log(self) -> "Tensor":
        """TODO(Day 2)."""
        raise NotImplementedError

    def reshape(self, *shape) -> "Tensor":
        """TODO(Day 2): backward is just reshape back — no math."""
        raise NotImplementedError

    def transpose(self) -> "Tensor":
        """2-D transpose. TODO(Day 2): backward transposes the grad."""
        raise NotImplementedError

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def gather_rows(self, index: np.ndarray) -> "Tensor":
        """out[i] = self[i, index[i]] — picks one logit per row, used by
        cross-entropy.

        TODO(Day 3): backward scatters the upstream grad back into the
        picked positions (np.add.at handles duplicates correctly).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------
    def backward(self) -> None:
        """TODO(Day 2): same topological algorithm as the scalar engine.

        Seed: np.ones_like(self.data) — the tests only call backward on
        scalar-shaped losses, but seeding with ones keeps it general.
        """
        raise NotImplementedError

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data)

    # -- reflected ops ---------------------------------------------------
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1.0

    def __rsub__(self, other):
        return (-self) + other

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, op={self._op!r}, requires_grad={self.requires_grad})"
