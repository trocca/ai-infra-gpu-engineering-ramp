"""Lesson 01: vectors.

Run with: python lesson.py
Mirrors SPIEGAZIONE.md step by step. Every intermediate value is printed
on purpose: follow the numbers, or set breakpoints and inspect the tensors.
"""

import os

import torch
import matplotlib

matplotlib.use("Agg")  # save figures to file, no display required
import matplotlib.pyplot as plt

torch.manual_seed(0)

print("=" * 60)
print("STEP 1: a vector is an ordered list of numbers")
print("=" * 60)

# torch.tensor turns a Python list into a tensor.
# dtype=torch.float32 is the standard type in deep learning.
u = torch.tensor([2.0, 1.0])
v = torch.tensor([1.0, 3.0])

print(f"u        = {u}")
print(f"v        = {v}")
print(f"u.shape  = {u.shape}   (2 elements, so dimension 2)")
print(f"u.dtype  = {u.dtype}")
print(f"u[0]     = {u[0]}   (position 0, like array indexing)")
print(f"u[1]     = {u[1]}")

print()
print("=" * 60)
print("STEP 2: sum and scalar multiplication")
print("=" * 60)

# Sum: element by element, same position with same position.
s = u + v
print(f"u + v = {s}   (expected [3, 4], computed by hand in SPIEGAZIONE)")

# Scalar multiplication: every element times the same number.
d = 2 * u
print(f"2 * u = {d}   (expected [4, 2])")

print()
print("=" * 60)
print("STEP 3: dot product, by hand and with torch")
print("=" * 60)

# Manual version: multiply position by position, then sum.
# This loop IS the definition of the dot product.
dot_manual = 0.0
for i in range(len(u)):
    partial = u[i] * v[i]
    dot_manual += partial
    print(f"  position {i}: {u[i].item():.1f} * {v[i].item():.1f} = {partial.item():.1f}")
print(f"dot_manual = {dot_manual.item():.1f}")

# PyTorch version: one call, same result.
dot_torch = torch.dot(u, v)
print(f"torch.dot(u, v) = {dot_torch.item():.1f}")

# If these two differ, something is deeply wrong. assert makes it explicit.
assert torch.allclose(dot_manual, dot_torch), "manual dot != torch.dot"
print("OK: manual version and torch.dot agree")

print()
print("=" * 60)
print("STEP 4: a house is a vector, a prediction is a dot product")
print("=" * 60)

# The tiny house dataset used across the whole course.
# Each row: [square meters, number of rooms]. Prices in thousands of euros.
houses = torch.tensor(
    [
        [50.0, 2.0],
        [70.0, 3.0],
        [90.0, 3.0],
        [110.0, 4.0],
        [130.0, 5.0],
    ]
)
prices = torch.tensor([150.0, 200.0, 250.0, 300.0, 350.0])

# Hand picked weights: 2 (thousand euros) per square meter equivalent,
# 10 per room, plus a base value of 20. Not trained, just a guess.
w = torch.tensor([2.0, 10.0])  # weights: one per feature
b = 20.0  # bias: base value, added to every prediction

print("house (vector)      prediction = x . w + b      real price   error")
for i in range(len(houses)):
    x = houses[i]  # one house = one vector
    pred = torch.dot(x, w) + b
    err = pred - prices[i]  # the error score, "punteggio di errore"
    print(
        f"{x.tolist()}     {x[0].item():5.0f}*2 + {x[1].item():.0f}*10 + 20 = {pred.item():5.1f}"
        f"      {prices[i].item():5.1f}      {err.item():+6.1f}"
    )
print("The guess is bad. In module 04 gradient descent will find better w and b.")

print()
print("=" * 60)
print("STEP 5: draw u, v and u + v as arrows")
print("=" * 60)

os.makedirs("figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(5, 5))
for vec, color, label in [(u, "tab:blue", "u"), (v, "tab:orange", "v"), (s, "tab:green", "u + v")]:
    ax.annotate(
        "",
        xy=(vec[0].item(), vec[1].item()),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color=color, lw=2),
    )
    ax.text(vec[0].item(), vec[1].item(), f" {label}", color=color, fontsize=12)
ax.set_xlim(-0.5, 5)
ax.set_ylim(-0.5, 5)
ax.grid(True, alpha=0.3)
ax.set_aspect("equal")
ax.set_title("u, v and their sum: walk along u, then along v")
fig.savefig(os.path.join("figures", "vettori.png"), dpi=120, bbox_inches="tight")
print("saved figures/vettori.png")
print()
print("Done. Now open exercises.py and complete the TODOs.")
