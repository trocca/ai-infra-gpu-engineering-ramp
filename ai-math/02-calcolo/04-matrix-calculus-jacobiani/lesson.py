"""Lesson 04: matrix calculus and Jacobians.

Run with: python lesson.py
The Jacobian of a small vector function three ways, the linear layer
whose Jacobian is W itself, and the shapes of the house loss gradient.
"""

import torch
from torch.autograd.functional import jacobian

torch.manual_seed(0)

print("=" * 60)
print("STEP 1: the Jacobian of f(x, y) = [x^2, x*y] at (2, 3), by hand")
print("=" * 60)


def f(p):
    return torch.stack([p[0] ** 2, p[0] * p[1]])


p0 = torch.tensor([2.0, 3.0])

# By hand: J[i, j] = partial of output i wrt input j.
J_hand = torch.tensor(
    [
        [2 * p0[0], torch.tensor(0.0)],  # row 0: output x^2 feels only x
        [p0[1], p0[0]],  # row 1: output x*y feels both
    ]
)
print(f"J by hand =\n{J_hand}")
print("Shape rule: (outputs x inputs) = (2 x 2). One row per output.")

print()
print("=" * 60)
print("STEP 2: same Jacobian by finite differences, column by column")
print("=" * 60)

# Column j of J = how ALL outputs move when input j is wiggled.
h = 1e-4
J_finite = torch.zeros(2, 2)
for j in range(2):
    moved = p0.clone()
    moved[j] += h
    J_finite[:, j] = (f(moved) - f(p0)) / h
print(f"J by finite differences =\n{J_finite}")
assert torch.allclose(J_finite, J_hand, atol=1e-2), "finite diff != hand formula"
print("OK: wiggle one input, read all outputs: that fills one column.")

print()
print("=" * 60)
print("STEP 3: torch computes it in one call")
print("=" * 60)

J_torch = jacobian(f, p0)
print(f"torch.autograd.functional.jacobian =\n{J_torch}")
assert torch.allclose(J_torch, J_hand, atol=1e-5), "autograd != hand formula"
print("OK: three methods, one matrix.")

print()
print("=" * 60)
print("STEP 4: the Jacobian of a linear layer IS the weight matrix")
print("=" * 60)

W = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3 outputs, 2 inputs


def linear_layer(x):
    return W @ x


x0 = torch.tensor([0.5, -1.0])
J_lin = jacobian(linear_layer, x0)
print(f"W =\n{W}")
print(f"Jacobian of x -> W @ x at any point =\n{J_lin}")
assert torch.allclose(J_lin, W), "Jacobian of a linear map should be W"
print("OK: for linear layers, sensitivity table == weight matrix.")
print("No surprise: output i is a dot product of row i of W with x,")
print("so output i feels input j exactly with strength W[i, j].")

print()
print("=" * 60)
print("STEP 5: shapes of the house loss gradient, start to finish")
print("=" * 60)

X = torch.tensor(
    [
        [0.50, 2.0],
        [0.70, 3.0],
        [0.90, 3.0],
        [1.10, 4.0],
        [1.30, 5.0],
    ]
)  # sqm/100 and rooms
y = torch.tensor([150.0, 200.0, 250.0, 300.0, 350.0])

w = torch.tensor([100.0, 10.0], requires_grad=True)
preds = X @ w  # shape (5): X is (5x2), w is (2)
loss = ((preds - y) ** 2).mean()  # shape (): a scalar
loss.backward()

print(f"X.shape      = {tuple(X.shape)}   (5 houses, 2 features)")
print(f"w.shape      = {tuple(w.shape)}")
print(f"preds.shape  = {tuple(preds.shape)}   (one prediction per house)")
print(f"loss.shape   = {tuple(loss.shape)}   (empty tuple: a scalar)")
print(f"w.grad.shape = {tuple(w.grad.shape)}   (SAME shape as w, always)")
print(f"w.grad       = {[round(g, 2) for g in w.grad.tolist()]}")

# Closed form for the MSE gradient: 2/n * X^T @ (X @ w - y).
grad_formula = 2 / len(y) * X.T @ (X @ w.detach() - y)
assert torch.allclose(w.grad, grad_formula, atol=1e-3), "autograd != closed form"
print(f"closed form 2/n * X.T @ (X@w - y) = {[round(g, 2) for g in grad_formula.tolist()]}")
print("OK. Golden rule: the gradient of the loss wrt a parameter has the")
print("SAME shape as the parameter. If shapes differ, the math is wrong.")
print()
print("Done. Now open exercises.py and complete the TODOs.")
