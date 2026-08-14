"""Lesson 01: gradient descent by hand.

Run with: python lesson.py
The hand computed steps on (w - 3)^2, then the house model trained with
autograd, then three learning rates compared: ant, right, divergent.
"""

import os

import torch
import matplotlib

matplotlib.use("Agg")  # save figures to file, no display required
import matplotlib.pyplot as plt

torch.manual_seed(0)

print("=" * 60)
print("STEP 1: the hand example, loss(w) = (w - 3)^2 from w = 0")
print("=" * 60)

w = 0.0
lr = 0.25
print(f"{'step':>4} {'gradient':>10} {'new w':>8}   (bottom is at w = 3)")
for step in range(6):
    grad = 2 * (w - 3)  # exact derivative of (w - 3)^2
    w = w - lr * grad  # THE update rule. All of deep learning is here.
    print(f"{step:>4} {grad:>10.4f} {w:>8.4f}")
assert abs(w - 3) < 0.05, "should be very close to the bottom"
print("Each step halves the distance to the bottom. Clean convergence.")

print()
print("=" * 60)
print("STEP 2: same valley, reckless step lr = 1.1")
print("=" * 60)

w = 0.0
for step in range(8):
    grad = 2 * (w - 3)
    w = w - 1.1 * grad
    print(f"step {step}: w = {w:12.4f}   loss = {(w - 3) ** 2:14.4f}")
assert abs(w - 3) > 10, "with lr = 1.1 the loss must diverge"
print("Overshooting the bottom on every step, worse each time: divergence.")

print()
print("=" * 60)
print("STEP 3: train the house model for real, with autograd")
print("=" * 60)

# Standardized features (module 01, lesson 03) make gradient descent behave.
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


def train(lr, steps=1000):
    # 1000 steps: sqm and rooms are highly correlated, so this valley is
    # elongated (lesson 03 shows why that is slow). GD gets there anyway.
    """Full gradient descent loop. Returns the loss history."""
    w = torch.zeros(2, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    history = []
    for _ in range(steps):
        preds = X @ w + b
        loss = ((preds - y) ** 2).mean()
        history.append(loss.item())
        loss.backward()
        with torch.no_grad():  # updates must not be tracked by autograd
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()  # gradients accumulate by default: reset them
        b.grad.zero_()
    return w.detach(), b.detach(), history


w_fit, b_fit, hist = train(lr=0.1)
print(f"loss at start = {hist[0]:10.1f}")
print(f"loss at end   = {hist[-1]:10.4f}")
print(f"learned w = {[round(v, 2) for v in w_fit.tolist()]}, b = {round(b_fit.item(), 2)}")
preds = X @ w_fit + b_fit
for i in range(5):
    print(f"house {i}: predicted {preds[i].item():6.1f}   real {y[i].item():6.1f}")
assert hist[-1] < 1.0, "training should reach near zero loss"
print("The fog ball found the bottom: predictions match prices.")

print()
print("=" * 60)
print("STEP 4: three learning rates, three destinies")
print("=" * 60)

runs = {}
for lr_try in [0.005, 0.1, 1.05]:
    _, _, h = train(lr=lr_try, steps=60)
    runs[lr_try] = h
    label = "ant steps" if lr_try == 0.005 else ("good" if lr_try == 0.1 else "divergent")
    print(f"lr = {lr_try:6.3f}   loss after 60 steps = {h[-1]:14.4f}   ({label})")
assert runs[0.005][-1] > runs[0.1][-1], "tiny lr should still be far from the bottom"
assert runs[1.05][-1] > runs[1.05][0], "huge lr should diverge"

os.makedirs("figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 5))
for lr_try, h in runs.items():
    ax.plot(h, label=f"lr = {lr_try}")
ax.set_yscale("log")  # log scale: divergence and convergence both visible
ax.set_xlabel("step")
ax.set_ylabel("loss (log scale)")
ax.set_title("Same valley, three step sizes")
ax.legend()
fig.savefig(os.path.join("figures", "learning_rates.png"), dpi=120, bbox_inches="tight")
print("saved figures/learning_rates.png")
print()
print("Done. Now open exercises.py and complete the TODOs.")
