"""Optimizers (Day 3).

Match the PyTorch documentation update rules EXACTLY so that learning-rate
and beta values transfer between frameworks:
  * SGD w/ momentum: https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html
  * Adam:            https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html
"""

from __future__ import annotations

import numpy as np

from .tensor import Tensor


class Optimizer:
    def __init__(self, params: list[Tensor]):
        self.params = list(params)

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent with (optional) classical momentum.

    TODO(Day 3):
      * No momentum: p.data -= lr * p.grad
      * With momentum mu: keep one velocity buffer per param,
        v = mu * v + grad;  p.data -= lr * v
        (this is PyTorch's convention — note it differs from some textbooks
        that fold lr into v; be able to explain the difference).
    """

    def __init__(self, params: list[Tensor], lr: float = 1e-2, momentum: float = 0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        # TODO(Day 3): allocate velocity buffers (zeros_like each param).
        raise NotImplementedError

    def step(self) -> None:
        raise NotImplementedError


class Adam(Optimizer):
    """Adam with bias correction.

    TODO(Day 3), per param at step t (t starts at 1):
      m = b1*m + (1-b1)*grad
      v = b2*v + (1-b2)*grad^2
      m_hat = m / (1 - b1^t);  v_hat = v / (1 - b2^t)
      p.data -= lr * m_hat / (sqrt(v_hat) + eps)

    Note eps goes OUTSIDE the sqrt in PyTorch's formulation — a classic
    source of tiny mismatches. The tests check several steps of updates
    against torch.optim.Adam, so get the bias correction and eps placement
    right.
    """

    def __init__(self, params: list[Tensor], lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        super().__init__(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        # TODO(Day 3): allocate m and v buffers.
        raise NotImplementedError

    def step(self) -> None:
        raise NotImplementedError
