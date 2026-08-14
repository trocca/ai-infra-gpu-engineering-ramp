"""Lesson 02: partial derivatives and the gradient.

Run with: python lesson.py
Partial derivatives one knob at a time, the gradient as the full list,
a field of arrows pointing uphill, and the gradient of the house loss.
"""

import os

import torch
import matplotlib

matplotlib.use("Agg")  # save figures to file, no display required
import matplotlib.pyplot as plt

torch.manual_seed(0)

print("=" * 60)
print("STEP 1: partial derivatives, one knob at a time")
print("=" * 60)


def f(x, y):
    return x**2 + 3 * y


x0, y0 = 2.0, 1.0
h = 1e-4

# Partial wrt x: freeze y, wiggle x.
df_dx = (f(x0 + h, y0) - f(x0, y0)) / h
# Partial wrt y: freeze x, wiggle y.
df_dy = (f(x0, y0 + h) - f(x0, y0)) / h

print(f"f(x, y) = x^2 + 3y, at point ({x0}, {y0})")
print(f"df/dx by finite difference = {df_dx:.4f}   (exact: 2x = 4)")
print(f"df/dy by finite difference = {df_dy:.4f}   (exact: 3)")

print()
print("=" * 60)
print("STEP 2: the gradient is the full list, autograd gives it at once")
print("=" * 60)

# Both variables in ONE tensor: autograd fills .grad with the whole gradient.
p = torch.tensor([x0, y0], requires_grad=True)
z = p[0] ** 2 + 3 * p[1]
z.backward()
print(f"point        = {p.tolist()}")
print(f"gradient     = {p.grad.tolist()}   (this is nabla f, the list of slopes)")

assert torch.allclose(p.grad, torch.tensor([4.0, 3.0]), atol=1e-5)
assert abs(df_dx - 4.0) < 1e-2 and abs(df_dy - 3.0) < 1e-2
print("OK: finite differences, hand formulas and autograd all agree")

print()
print("=" * 60)
print("STEP 3: the gradient points UPHILL, its opposite points DOWNHILL")
print("=" * 60)

# Take a small step along the gradient and against it, compare f.
step = 0.01
grad = p.grad
up = f(x0 + step * grad[0].item(), y0 + step * grad[1].item())
down = f(x0 - step * grad[0].item(), y0 - step * grad[1].item())
base = f(x0, y0)
print(f"f at the point                  = {base:.4f}")
print(f"f after a step ALONG the grad   = {up:.4f}   (bigger: uphill)")
print(f"f after a step AGAINST the grad = {down:.4f}   (smaller: downhill)")
assert up > base > down
print("OK: minus gradient = direction of descent. Remember this forever.")

print()
print("=" * 60)
print("STEP 4: gradient of the house loss wrt weight AND bias")
print("=" * 60)

sqm100 = torch.tensor([0.50, 0.70, 0.90, 1.10, 1.30])  # sqm / 100
prices = torch.tensor([150.0, 200.0, 250.0, 300.0, 350.0])

# Two knobs now: params[0] is the weight w, params[1] is the bias b.
params = torch.tensor([200.0, 0.0], requires_grad=True)
preds = params[0] * sqm100 + params[1]
loss = ((preds - prices) ** 2).mean()
loss.backward()

print(f"w = {params[0].item():.1f}, b = {params[1].item():.1f}, loss = {loss.item():.1f}")
print(f"gradient [dloss/dw, dloss/db] = {[round(g, 2) for g in params.grad.tolist()]}")
print("Both slopes are negative: both w and b should grow to reduce the error.")
print("One backward call, the whole gradient. With 1 billion knobs it works the same.")

print()
print("=" * 60)
print("STEP 5: the field of arrows, every point shows its uphill")
print("=" * 60)

os.makedirs("figures", exist_ok=True)
# Bowl function g(x, y) = x^2 + y^2: gradient [2x, 2y] points away from center.
grid = torch.linspace(-2, 2, 11)
gx, gy = torch.meshgrid(grid, grid, indexing="xy")
gradx, grady = 2 * gx, 2 * gy

fig, ax = plt.subplots(figsize=(6, 6))
levels = ax.contour(gx.numpy(), gy.numpy(), (gx**2 + gy**2).numpy(), levels=8, alpha=0.4)
ax.quiver(gx.numpy(), gy.numpy(), gradx.numpy(), grady.numpy(), color="tab:blue", alpha=0.8)
ax.set_title("Gradient field of x^2 + y^2: arrows point uphill,\ntraining walks the other way")
ax.set_aspect("equal")
fig.savefig(os.path.join("figures", "campo_gradienti.png"), dpi=120, bbox_inches="tight")
print("saved figures/campo_gradienti.png")
print()
print("Done. Now open exercises.py and complete the TODOs.")
