"""Neural-network layers on top of the Tensor engine (Day 3).

Mirrors a small slice of torch.nn so the test suite can build the same
model in both frameworks and compare gradients end-to-end.
"""

from __future__ import annotations

import numpy as np

from .tensor import Tensor


class Module:
    """Base class: parameter discovery + train/eval flag.

    Implemented for you — subclasses register Tensors as attributes and
    parameters() finds them recursively.
    """

    def parameters(self) -> list[Tensor]:
        params: list[Tensor] = []
        for value in self.__dict__.values():
            if isinstance(value, Tensor) and value.requires_grad:
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Linear(Module):
    """y = x @ W + b, with W shaped (in_features, out_features).

    TODO(Day 3):
      * Initialize W with Kaiming/He init for ReLU nets:
        std = sqrt(2 / in_features), normal draw, requires_grad=True.
      * Initialize b to zeros, requires_grad=True.
      * forward: one matmul + one (broadcast) add. The broadcast add is why
        you sweated over _unbroadcast on Day 2.
    """

    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator | None = None):
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        """TODO(Day 3): one line."""
        raise NotImplementedError


class Sequential(Module):
    """Implemented for you."""

    def __init__(self, *layers: Module):
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def softmax_cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Mean cross-entropy between logits (N, C) and integer targets (N,).

    Must be NUMERICALLY STABLE. TODO(Day 3), using only Tensor ops so the
    gradient flows automatically:
      1. Subtract the row-wise max (logits.max(axis=1, keepdims=True)) —
         this changes nothing mathematically but prevents exp overflow.
      2. log_softmax = shifted - log(sum(exp(shifted), axis=1, keepdims=True))
      3. Pick the target log-probability per row (Tensor.gather_rows).
      4. Return the negative mean.

    Do NOT compute softmax then log — derive why log-sum-exp is the stable
    form and put the derivation in your RESULTS.md.
    """
    raise NotImplementedError
