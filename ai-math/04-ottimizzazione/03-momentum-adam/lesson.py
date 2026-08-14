"""Lesson 03: momentum and Adam.

Run with: python lesson.py
Momentum and Adam implemented from scratch on a narrow valley, compared
with plain GD, then hand written Adam checked against torch.optim.Adam.
"""

import os

import torch
import matplotlib

matplotlib.use("Agg")  # save figures to file, no display required
import matplotlib.pyplot as plt

torch.manual_seed(0)

# A narrow valley: steep in w2 (factor 200 between the two curvatures),
# almost flat in w1. The classic trap for plain gradient descent.
def valley(p):
    return 0.05 * p[0] ** 2 + 10 * p[1] ** 2


def grad_valley(p):
    return torch.tensor([0.1 * p[0], 20 * p[1]])


START = torch.tensor([-8.0, 1.5])

print("=" * 60)
print("STEP 1: plain GD zigzags in a narrow valley")
print("=" * 60)

# lr near the stability limit of the STEEP direction (2/20 = 0.1): any
# bigger diverges. But it is tiny for the flat direction: w1 crawls.
lr = 0.095


def run_gd(steps=60):
    p = START.clone()
    path = [p.clone()]
    for _ in range(steps):
        p = p - lr * grad_valley(p)
        path.append(p.clone())
    return torch.stack(path)


path_gd = run_gd()
print("first steps of w2 (the steep direction):")
print([round(v, 3) for v in path_gd[:6, 1].tolist()])
print("Sign flips every step: the ball bounces between the walls.")
print("meanwhile w1 (the flat direction) crawls:")
print([round(v, 2) for v in path_gd[:6, 0].tolist()])
print(f"after 60 steps: {[round(v, 4) for v in path_gd[-1].tolist()]}, "
      f"loss = {valley(path_gd[-1]).item():.5f}")
assert abs(path_gd[-1][0]) > 3, "w1 should still be far from 0: GD is stuck crawling"

print()
print("=" * 60)
print("STEP 2: momentum, the ball gains weight")
print("=" * 60)


def run_momentum(steps=60, beta=0.9):
    p = START.clone()
    v = torch.zeros(2)
    path = [p.clone()]
    for _ in range(steps):
        v = beta * v + grad_valley(p)  # accumulate velocity
        p = p - lr * v
        path.append(p.clone())
    return torch.stack(path)


path_mom = run_momentum()
print(f"after 60 steps: {[round(v, 4) for v in path_mom[-1].tolist()]}, "
      f"loss = {valley(path_mom[-1]).item():.6f}")
assert valley(path_mom[-1]) < valley(path_gd[-1]), "momentum should beat plain GD here"
print("Closer to the bottom: sideways bounces cancel, forward speed builds up.")

# Check our momentum against torch.optim.SGD, same formula.
p_torch = START.clone().requires_grad_(True)
opt = torch.optim.SGD([p_torch], lr=lr, momentum=0.9)
for _ in range(60):
    opt.zero_grad()
    valley(p_torch).backward()
    opt.step()
assert torch.allclose(p_torch.detach(), path_mom[-1], atol=1e-4)
print("OK: hand momentum == torch.optim.SGD(momentum=0.9), step by step.")

print()
print("=" * 60)
print("STEP 3: Adam from scratch, six lines of arithmetic")
print("=" * 60)


def run_adam(steps=60, lr_adam=0.5, beta1=0.9, beta2=0.999, eps=1e-8):
    p = START.clone()
    m = torch.zeros(2)  # running mean of gradients (direction)
    v = torch.zeros(2)  # running mean of squared gradients (scale)
    path = [p.clone()]
    for t in range(1, steps + 1):
        g = grad_valley(p)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)  # bias correction: m starts at zero
        v_hat = v / (1 - beta2**t)
        p = p - lr_adam * m_hat / (torch.sqrt(v_hat) + eps)
        path.append(p.clone())
    return torch.stack(path)


path_adam = run_adam()
print(f"after 60 steps: {[round(v, 4) for v in path_adam[-1].tolist()]}, "
      f"loss = {valley(path_adam[-1]).item():.6f}")
print("Per parameter step sizes: the steep direction gets tamed automatically.")

print()
print("=" * 60)
print("STEP 4: the acid test, hand Adam vs torch.optim.Adam")
print("=" * 60)

p_torch = START.clone().requires_grad_(True)
opt = torch.optim.Adam([p_torch], lr=0.5, betas=(0.9, 0.999), eps=1e-8)
for _ in range(60):
    opt.zero_grad()
    valley(p_torch).backward()
    opt.step()
print(f"hand Adam : {[round(v, 6) for v in path_adam[-1].tolist()]}")
print(f"torch Adam: {[round(v, 6) for v in p_torch.detach().tolist()]}")
assert torch.allclose(p_torch.detach(), path_adam[-1], atol=1e-4), "Adam mismatch"
print("OK: identical trajectory. torch.optim.Adam is these six lines, tuned.")

print()
print("=" * 60)
print("STEP 5: the three racers on one picture")
print("=" * 60)

os.makedirs("figures", exist_ok=True)
losses = {
    "GD": [valley(p).item() for p in path_gd],
    "momentum": [valley(p).item() for p in path_mom],
    "Adam": [valley(p).item() for p in path_adam],
}
fig, ax = plt.subplots(figsize=(7, 5))
for name, h in losses.items():
    ax.plot(h, label=name)
ax.set_yscale("log")
ax.set_xlabel("step")
ax.set_ylabel("loss (log scale)")
ax.set_title("Narrow valley: plain GD vs momentum vs Adam")
ax.legend()
fig.savefig(os.path.join("figures", "optimizers.png"), dpi=120, bbox_inches="tight")
print("saved figures/optimizers.png")
print()
print("Done. Now open exercises.py and complete the TODOs.")
