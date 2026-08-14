"""Lesson 04: the loss landscape.

Run with: python lesson.py
The house loss computed EVERYWHERE on a (w, b) grid, the contour map,
and the GD and momentum trajectories drawn on top. Fog lifted.
"""

import os

import torch
import matplotlib

matplotlib.use("Agg")  # save figures to file, no display required
import matplotlib.pyplot as plt

torch.manual_seed(0)

# One feature keeps the landscape 2D: standardized square meters.
sqm = torch.tensor([50.0, 70.0, 90.0, 110.0, 130.0])
x = (sqm - sqm.mean()) / sqm.std()
y = torch.tensor([150.0, 200.0, 250.0, 300.0, 350.0])


def loss_at(w, b):
    return ((w * x + b - y) ** 2).mean()


print("=" * 60)
print("STEP 1: the loss at a few hand picked map points")
print("=" * 60)

for w0, b0 in [(0.0, 0.0), (50.0, 250.0), (79.0, 250.0), (100.0, 300.0)]:
    print(f"loss(w={w0:6.1f}, b={b0:6.1f}) = {loss_at(w0, b0).item():10.1f}")
print("Somewhere around (79, 250) the valley seems to bottom out.")

print()
print("=" * 60)
print("STEP 2: compute the loss EVERYWHERE on a grid")
print("=" * 60)

ws = torch.linspace(-20, 180, 120)
bs = torch.linspace(130, 370, 120)
L = torch.zeros(len(bs), len(ws))
for i, b0 in enumerate(bs):
    for j, w0 in enumerate(ws):
        L[i, j] = loss_at(w0, b0)

flat = L.argmin()
i_min, j_min = flat // L.shape[1], flat % L.shape[1]
w_grid, b_grid = ws[j_min], bs[i_min]
print(f"grid minimum at w = {w_grid.item():.1f}, b = {b_grid.item():.1f}, "
      f"loss = {L[i_min, j_min].item():.2f}")

# Exact least squares solution via the closed formulas for 1D regression.
w_star = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
b_star = y.mean() - w_star * x.mean()
print(f"exact solution  w* = {w_star.item():.2f}, b* = {b_star.item():.2f}")
assert abs(w_grid - w_star) < 2 and abs(b_grid - b_star) < 2, "grid min far from exact"
print("OK: the deepest cell of the map sits on the exact solution.")

print()
print("=" * 60)
print("STEP 3: run GD and momentum, recording their paths")
print("=" * 60)


def grad_at(w, b):
    err = w * x + b - y
    return 2 * (err * x).mean(), 2 * err.mean()


def descend(lr, beta, steps=40):
    w0, b0 = torch.tensor(-10.0), torch.tensor(150.0)  # start on a high slope
    vw = vb = 0.0
    path = [(w0.item(), b0.item())]
    for _ in range(steps):
        gw, gb = grad_at(w0, b0)
        vw = beta * vw + gw
        vb = beta * vb + gb
        w0 = w0 - lr * vw
        b0 = b0 - lr * vb
        path.append((w0.item(), b0.item()))
    return torch.tensor(path)


path_gd = descend(lr=0.4, beta=0.0)  # beta 0: plain gradient descent
path_mom = descend(lr=0.4, beta=0.5)

for name, path in [("GD", path_gd), ("momentum", path_mom)]:
    end_w, end_b = path[-1]
    print(f"{name:9s} ends at w = {end_w.item():6.2f}, b = {end_b.item():6.2f}, "
          f"loss = {loss_at(end_w, end_b).item():8.2f}")
    assert loss_at(end_w, end_b).item() < 100, f"{name} did not reach the valley floor"
print(f"(exact bottom: w = {w_star.item():.2f}, b = {b_star.item():.2f}, "
      f"loss = {loss_at(w_star, b_star).item():.2f})")

print()
print("=" * 60)
print("STEP 4: draw the whole valley with both trajectories")
print("=" * 60)

os.makedirs("figures", exist_ok=True)
gw_mesh, gb_mesh = torch.meshgrid(ws, bs, indexing="xy")
fig, ax = plt.subplots(figsize=(8, 6))
cs = ax.contour(gw_mesh.numpy(), gb_mesh.numpy(), L.numpy(), levels=25, alpha=0.6)
ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")
ax.plot(path_gd[:, 0].numpy(), path_gd[:, 1].numpy(), "o-", ms=3, label="GD")
ax.plot(path_mom[:, 0].numpy(), path_mom[:, 1].numpy(), "o-", ms=3, label="momentum")
ax.scatter([w_star.item()], [b_star.item()], marker="*", s=200, color="red",
           zorder=5, label="exact minimum")
ax.set_xlabel("w")
ax.set_ylabel("b")
ax.set_title("The house loss valley, fog lifted: contours and descent paths")
ax.legend()
fig.savefig(os.path.join("figures", "loss_landscape.png"), dpi=120, bbox_inches="tight")
print("saved figures/loss_landscape.png")
print()
print("Every training run you will ever launch is this picture, in millions")
print("of dimensions, with the fog permanently on. But the ball still rolls.")
print()
print("Done. Now open exercises.py and complete the TODOs.")
