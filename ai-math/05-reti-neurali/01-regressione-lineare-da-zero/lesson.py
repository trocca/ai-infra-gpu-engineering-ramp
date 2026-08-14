"""Lesson 01: linear regression from scratch.

Run with: python lesson.py
The complete training loop assembled from the previous modules, on the
5 house dataset, checked against the closed form least squares solution.
"""

import os

import torch
import matplotlib

matplotlib.use("Agg")  # save figures to file, no display required
import matplotlib.pyplot as plt

torch.manual_seed(0)

# The recurring dataset, standardized features (module 01, lesson 03).
X_raw = torch.tensor(
    [
        [50.0, 2.0],
        [70.0, 3.0],
        [90.0, 3.0],
        [110.0, 4.0],
        [130.0, 5.0],
    ]
)
y = torch.tensor([150.0, 200.0, 250.0, 300.0, 350.0])
X = (X_raw - X_raw.mean(dim=0)) / X_raw.std(dim=0)

print("=" * 60)
print("STEP 1: one loop turn by hand, single house version")
print("=" * 60)

x1, y1 = X[4, 0].item(), y[4].item()  # fifth house, sqm feature only
w0, b0, lr = 0.0, 0.0, 0.1
pred = w0 * x1 + b0
gw = 2 * (pred - y1) * x1
gb = 2 * (pred - y1)
print(f"forward:  pred = {pred}")
print(f"loss:     {(pred - y1) ** 2:.0f}")
print(f"backward: dloss/dw = {gw:.1f}, dloss/db = {gb:.1f}")
w0, b0 = w0 - lr * gw, b0 - lr * gb
print(f"update:   w = {w0:.1f}, b = {b0:.1f}")
print(f"new pred = {w0 * x1 + b0:.1f}   (was 0, target 350: already much closer)")

print()
print("=" * 60)
print("STEP 2: the full training loop on all 5 houses")
print("=" * 60)

w = torch.zeros(2, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
lr = 0.1
history = []

for epoch in range(2000):
    preds = X @ w + b  # 1. forward (a matmul: module 01)
    loss = ((preds - y) ** 2).mean()  # 2. loss (MSE: modules 02 and 03)
    loss.backward()  # 3. backward (chain rule: module 02)
    with torch.no_grad():  # 4. update (descent: module 04)
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_()  # gradients ACCUMULATE: reset or corrupt the next step
    b.grad.zero_()
    history.append(loss.item())
    if epoch % 400 == 0:
        print(f"epoch {epoch:4d}: loss = {loss.item():12.4f}")

print(f"final loss = {history[-1]:.6f}")

print()
print("=" * 60)
print("STEP 3: predictions vs real prices")
print("=" * 60)

with torch.no_grad():
    final_preds = X @ w + b
print("house (sqm, rooms)    predicted    real")
for i in range(5):
    print(f"{[round(v, 1) for v in X_raw[i].tolist()]}         "
          f"{final_preds[i].item():8.2f}  {y[i].item():8.1f}")
print(f"learned w = {[round(v, 2) for v in w.tolist()]}, b = {round(b.item(), 2)}")

print()
print("=" * 60)
print("STEP 4: the acid test against closed form least squares")
print("=" * 60)

# Augment X with a column of ones so the bias is part of the system.
A = torch.cat([X, torch.ones(5, 1)], dim=1)
exact = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.squeeze()
print(f"lstsq exact solution: w = {[round(v, 2) for v in exact[:2].tolist()]}, "
      f"b = {round(exact[2].item(), 2)}")
assert torch.allclose(w.detach(), exact[:2], atol=0.5), "training missed the lstsq w"
assert torch.allclose(b.detach(), exact[2:], atol=0.5), "training missed the lstsq b"
assert history[-1] < 0.5, "final loss should be near zero on this dataset"
print("OK: the iterative loop landed on the exact solution. For deep")
print("networks no closed form exists, and the loop is all there is.")

print()
print("=" * 60)
print("STEP 5: the training curve")
print("=" * 60)

os.makedirs("figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(history)
ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel("loss (log scale)")
ax.set_title("The first real training curve of this course")
fig.savefig(os.path.join("figures", "training_curve.png"), dpi=120, bbox_inches="tight")
print("saved figures/training_curve.png")
print()
print("Done. Now open exercises.py and complete the TODOs.")
