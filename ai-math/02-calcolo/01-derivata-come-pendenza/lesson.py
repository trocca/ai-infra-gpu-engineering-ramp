"""Lesson 01: the derivative as a slope.

Run with: python lesson.py
The derivative of x^2 computed three ways (finite difference, exact
formula, autograd), then the house loss as a function of one weight.
"""

import os

import torch
import matplotlib

matplotlib.use("Agg")  # save figures to file, no display required
import matplotlib.pyplot as plt

torch.manual_seed(0)

print("=" * 60)
print("STEP 1: finite difference, the slope by brute force")
print("=" * 60)


def f(x):
    return x**2


x0 = 3.0
print(f"f(x) = x^2, at x = {x0}")
print(f"{'h':>10}   {'(f(x+h)-f(x))/h':>18}")
for h in [1.0, 0.1, 0.01, 0.001]:
    slope = (f(x0 + h) - f(x0)) / h
    print(f"{h:>10}   {slope:>18.4f}")
print("As h shrinks, the slope approaches 6. The exact formula 2x gives 2*3 = 6.")

print()
print("=" * 60)
print("STEP 2: autograd, torch computes the exact slope for us")
print("=" * 60)

# requires_grad=True tells torch: track every operation on this tensor.
x = torch.tensor(x0, requires_grad=True)
y = x**2
print(f"x = {x.item()}, y = x^2 = {y.item()}")

# backward() computes the derivative of y with respect to every tracked
# input, and stores it in the .grad field of each input.
y.backward()
print(f"x.grad = {x.grad.item()}   (the derivative dy/dx at x = 3)")

exact = 2 * x0
finite = (f(x0 + 1e-4) - f(x0)) / 1e-4
assert abs(x.grad.item() - exact) < 1e-6, "autograd != exact formula"
assert abs(finite - exact) < 1e-2, "finite difference too far from exact"
print(f"exact 2x = {exact}, finite difference = {finite:.4f}, autograd = {x.grad.item()}")
print("OK: three methods, one slope.")

print()
print("=" * 60)
print("STEP 3: the house loss as a function of ONE weight")
print("=" * 60)

# House dataset, single feature: square meters (scaled down by 100 so the
# numbers stay small and inspectable). Model: price = w * sqm100.
sqm100 = torch.tensor([0.50, 0.70, 0.90, 1.10, 1.30])  # sqm / 100
prices = torch.tensor([150.0, 200.0, 250.0, 300.0, 350.0])


def loss_fn(w):
    """Mean squared error: the average of squared prediction errors.
    This is the 'punteggio di errore' of the model with weight w."""
    preds = w * sqm100
    return ((preds - prices) ** 2).mean()


for w_try in [100.0, 200.0, 265.0, 300.0]:
    print(f"w = {w_try:6.1f}   loss = {loss_fn(torch.tensor(w_try)).item():10.1f}")
print("The loss has a valley: too small w is bad, too big w is bad.")

print()
print("=" * 60)
print("STEP 4: the slope of the loss tells us WHERE to move w")
print("=" * 60)

w = torch.tensor(200.0, requires_grad=True)
loss = loss_fn(w)
loss.backward()
slope_autograd = w.grad.item()

# Finite difference check on the same point.
h = 0.01
slope_finite = (loss_fn(torch.tensor(200.0 + h)) - loss_fn(torch.tensor(200.0))) / h
print(f"at w = 200: loss = {loss.item():.1f}")
print(f"dloss/dw by autograd           = {slope_autograd:.2f}")
print(f"dloss/dw by finite difference  = {slope_finite.item():.2f}")
assert abs(slope_autograd - slope_finite.item()) < 1.0, "autograd != finite difference"
print("Slope is negative: increasing w DECREASES the loss. So w should grow.")
print("This single sentence is the whole idea of training. Module 04 builds on it.")

print()
print("=" * 60)
print("STEP 5: draw the loss valley and the tangent at w = 200")
print("=" * 60)

os.makedirs("figures", exist_ok=True)
ws = torch.linspace(100, 400, 200)
losses = torch.tensor([loss_fn(wi).item() for wi in ws])

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(ws.numpy(), losses.numpy(), label="loss(w)")
# Tangent line at w = 200: passes through (200, loss) with the autograd slope.
w0, l0 = 200.0, loss.item()
tangent = l0 + slope_autograd * (ws - w0)
ax.plot(ws.numpy(), tangent.numpy(), "--", label=f"tangent, slope = {slope_autograd:.0f}")
ax.scatter([w0], [l0], color="red", zorder=5)
ax.set_xlabel("w")
ax.set_ylabel("loss")
ax.set_ylim(bottom=-2000)
ax.legend()
ax.set_title("The loss valley: the tangent points downhill")
fig.savefig(os.path.join("figures", "loss_parabola.png"), dpi=120, bbox_inches="tight")
print("saved figures/loss_parabola.png")
print()
print("Done. Now open exercises.py and complete the TODOs.")
