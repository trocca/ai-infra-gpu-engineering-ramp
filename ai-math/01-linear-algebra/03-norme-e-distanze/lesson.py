"""Lesson 03: norms and distances.

Run with: python lesson.py
Three norms by hand vs torch.linalg.norm, the feature scale trap on the
house dataset, cosine similarity, and a picture of the unit balls.
"""

import os

import torch
import matplotlib

matplotlib.use("Agg")  # save figures to file, no display required
import matplotlib.pyplot as plt

torch.manual_seed(0)

print("=" * 60)
print("STEP 1: three norms of v = [3, 4], by hand and with torch")
print("=" * 60)

v = torch.tensor([3.0, 4.0])

# L1: sum of absolute values.
l1_manual = torch.abs(v).sum()
# L2: square root of the sum of squares.
l2_manual = torch.sqrt((v * v).sum())
# L infinity: max of absolute values.
linf_manual = torch.abs(v).max()

print(f"v = {v.tolist()}")
print(f"L1   by hand: |3| + |4|          = {l1_manual.item():.1f}")
print(f"L2   by hand: sqrt(3*3 + 4*4)    = {l2_manual.item():.1f}")
print(f"Linf by hand: max(|3|, |4|)      = {linf_manual.item():.1f}")

# torch.linalg.norm computes all of them, selected by ord.
assert torch.allclose(l1_manual, torch.linalg.norm(v, ord=1))
assert torch.allclose(l2_manual, torch.linalg.norm(v, ord=2))
assert torch.allclose(linf_manual, torch.linalg.norm(v, ord=float("inf")))
print("OK: manual versions agree with torch.linalg.norm(v, ord=...)")

print()
print("=" * 60)
print("STEP 2: distance = norm of the difference")
print("=" * 60)

u = torch.tensor([1.0, 1.0])
w = torch.tensor([4.0, 5.0])
diff = u - w
dist = torch.linalg.norm(diff)
print(f"u - w = {diff.tolist()}, distance = sqrt(9 + 16) = {dist.item():.1f}")

print()
print("=" * 60)
print("STEP 3: the scale trap on the house dataset")
print("=" * 60)

X = torch.tensor(
    [
        [50.0, 2.0],
        [70.0, 3.0],
        [90.0, 3.0],
        [110.0, 4.0],
        [130.0, 5.0],
    ]
)

# Raw distance between house 1 [70, 3] and house 2 [90, 3]:
# square meters dominate, rooms are irrelevant.
d_raw_12 = torch.linalg.norm(X[1] - X[2])
d_raw_01 = torch.linalg.norm(X[0] - X[1])
print(f"raw distance house1 - house2 = {d_raw_12.item():.2f} (rooms equal, only sqm counts)")
print(f"raw distance house0 - house1 = {d_raw_01.item():.2f} (the +1 room adds almost nothing)")

# Standardize each column: subtract the mean, divide by the std.
# After this, both features live on the same scale.
mean = X.mean(dim=0)
std = X.std(dim=0)
Xn = (X - mean) / std
print(
    f"column means = {[round(m, 2) for m in mean.tolist()]}, "
    f"column stds = {[round(s, 2) for s in std.tolist()]}"
)
print(f"normalized X =\n{Xn}")
d_norm_12 = torch.linalg.norm(Xn[1] - Xn[2])
d_norm_01 = torch.linalg.norm(Xn[0] - Xn[1])
print(f"normalized distance house1 - house2 = {d_norm_12.item():.2f}")
print(f"normalized distance house0 - house1 = {d_norm_01.item():.2f}")
print("Now the room difference matters. Always check your scales.")

print()
print("=" * 60)
print("STEP 4: cosine similarity, by hand and with torch")
print("=" * 60)

a = torch.tensor([1.0, 1.0])
b = torch.tensor([3.0, 3.0])  # same direction as a, 3x longer
c = torch.tensor([-1.0, 1.0])  # perpendicular to a


def cosine_manual(x, y):
    """Cosine similarity: dot product divided by the product of L2 norms."""
    return torch.dot(x, y) / (torch.linalg.norm(x) * torch.linalg.norm(y))


for name, other in [("b (same direction)", b), ("c (perpendicular)", c), ("-a (opposite)", -a)]:
    cos_manual = cosine_manual(a, other)
    cos_torch = torch.nn.functional.cosine_similarity(a, other, dim=0)
    assert torch.allclose(cos_manual, cos_torch, atol=1e-6)
    print(f"cosine(a, {name:20s}) = {cos_manual.item():+.2f}")
print("OK: manual cosine agrees with F.cosine_similarity")
print("Length is ignored: a and b score 1.00 even if b is 3x longer.")

print()
print("=" * 60)
print("STEP 5: unit balls, the shape of 'distance 1' for each norm")
print("=" * 60)

os.makedirs("figures", exist_ok=True)
# Grid of points in [-1.5, 1.5]^2, then draw the curve where norm == 1.
grid = torch.linspace(-1.5, 1.5, 301)
gx, gy = torch.meshgrid(grid, grid, indexing="xy")
points = torch.stack([gx, gy], dim=-1)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
norms = [
    ("L1 (diamond)", torch.abs(points).sum(dim=-1)),
    ("L2 (circle)", torch.linalg.norm(points, dim=-1)),
    ("L infinity (square)", torch.abs(points).max(dim=-1).values),
]
for ax, (title, values) in zip(axes, norms):
    ax.contour(gx.numpy(), gy.numpy(), values.numpy(), levels=[1.0], colors="tab:blue")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
fig.suptitle("All points at distance 1 from the origin, per norm")
fig.savefig(os.path.join("figures", "unit_balls.png"), dpi=120, bbox_inches="tight")
print("saved figures/unit_balls.png")
print()
print("Done. Now open exercises.py and complete the TODOs.")
