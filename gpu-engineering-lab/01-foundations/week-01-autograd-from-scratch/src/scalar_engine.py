"""Scalar reverse-mode autodiff engine (Day 1).

A `Value` wraps a single float and remembers how it was produced, forming a
DAG. Calling `backward()` on the final node computes d(output)/d(node) for
every node in the graph via reverse-mode automatic differentiation.

Design contract (the tests in tests/test_scalar.py rely on this):
  * `Value.data`  -> float, the forward value
  * `Value.grad`  -> float, accumulated gradient (0.0 until backward runs)
  * arithmetic between Value and plain int/float must work in both orders
  * gradients ACCUMULATE (+=) so a node used twice gets both contributions
"""

from __future__ import annotations

import math


class Value:
    """A scalar node in a computational graph."""

    __slots__ = ("data", "grad", "_backward", "_prev", "_op")

    def __init__(self, data: float, _children: tuple = (), _op: str = ""):
        self.data = float(data)
        self.grad = 0.0
        # _backward: closure that takes no args and pushes this node's grad
        # into its parents' grads using the local derivative (chain rule).
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # ------------------------------------------------------------------
    # Binary ops
    # ------------------------------------------------------------------
    def __add__(self, other: "Value | float") -> "Value":
        """out = self + other.

        TODO(Day 1):
          1. Coerce `other` to a Value if it is a plain number.
          2. Create the output Value with the right _children and _op.
          3. Define out._backward: what is d(out)/d(self)? d(out)/d(other)?
             Remember to ACCUMULATE into .grad, not overwrite.
        """
        raise NotImplementedError

    def __mul__(self, other: "Value | float") -> "Value":
        """out = self * other.

        TODO(Day 1): local derivatives are each operand's partner's data.
        """
        raise NotImplementedError

    def __pow__(self, exponent: float) -> "Value":
        """out = self ** exponent, exponent is a plain int/float (not a Value).

        TODO(Day 1): power rule. Assert exponent is int/float.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Unary ops
    # ------------------------------------------------------------------
    def relu(self) -> "Value":
        """out = max(0, self).

        TODO(Day 1): the local derivative is a gate: 1 where data > 0 else 0.
        """
        raise NotImplementedError

    def tanh(self) -> "Value":
        """out = tanh(self).

        TODO(Day 1): d/dx tanh(x) = 1 - tanh(x)^2. Use math.tanh.
        """
        raise NotImplementedError

    def exp(self) -> "Value":
        """out = e ** self.

        TODO(Day 1): the derivative of exp is exp — reuse out.data.
        """
        raise NotImplementedError

    def log(self) -> "Value":
        """out = ln(self). Needed for cross-entropy later.

        TODO(Day 1): d/dx ln(x) = 1/x.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------
    def backward(self) -> None:
        """Run reverse-mode autodiff from this node.

        TODO(Day 1):
          1. Build a topological ordering of the graph reachable from self
             (post-order DFS over _prev with a visited set).
          2. Seed self.grad = 1.0.
          3. Call node._backward() for every node in REVERSED topo order.

        Why topological order matters: a node's _backward must only run
        after its grad has received ALL contributions from downstream.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Derived ops + reflected operators — implemented for you in terms of
    # the primitives above. Do not modify; the tests exercise them.
    # ------------------------------------------------------------------
    def __neg__(self) -> "Value":
        return self * -1.0

    def __sub__(self, other: "Value | float") -> "Value":
        return self + (-other if isinstance(other, Value) else -float(other))

    def __rsub__(self, other: float) -> "Value":
        return (-self) + float(other)

    def __truediv__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1.0

    def __rtruediv__(self, other: float) -> "Value":
        return Value(other) * self ** -1.0

    def __radd__(self, other: float) -> "Value":
        return self + other

    def __rmul__(self, other: float) -> "Value":
        return self * other

    def __repr__(self) -> str:
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g}, op={self._op!r})"
