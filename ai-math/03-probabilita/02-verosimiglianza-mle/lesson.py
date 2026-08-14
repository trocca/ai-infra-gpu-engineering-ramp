"""Lesson 02: likelihood and maximum likelihood estimation.

Run with: python lesson.py
The coin likelihood explored on a grid, the log trick, and a gaussian
mean estimated by gradient ascent on the log likelihood.
"""

import os

import torch
import matplotlib

matplotlib.use("Agg")  # save figures to file, no display required
import matplotlib.pyplot as plt

torch.manual_seed(0)

print("=" * 60)
print("STEP 1: 10 flips, 7 heads. How plausible is each p?")
print("=" * 60)

flips = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
heads = flips.sum()
n = len(flips)
print(f"data: {flips.int().tolist()}   ({heads:.0f} heads out of {n})")


def likelihood(p):
    """Probability of this exact sequence if the coin parameter were p."""
    return p ** heads * (1 - p) ** (n - heads)


for p_try in [0.3, 0.5, 0.7, 0.9]:
    print(f"L({p_try}) = {likelihood(torch.tensor(p_try)).item():.6f}")
print("p = 0.7 makes the data least miraculous so far.")

print()
print("=" * 60)
print("STEP 2: try ALL p on a grid, find the maximum")
print("=" * 60)

ps = torch.linspace(0.01, 0.99, 99)
L = likelihood(ps)
best = ps[L.argmax()]
print(f"grid maximum at p = {best.item():.2f}")
assert abs(best.item() - 0.7) < 0.011, "MLE should be 7/10"
print(f"formula heads/n  = {heads.item() / n}. They agree: the MLE of a coin")
print("is just the observed frequency.")

print()
print("=" * 60)
print("STEP 3: the log does not move the maximum")
print("=" * 60)

logL = heads * torch.log(ps) + (n - heads) * torch.log(1 - ps)
best_log = ps[logL.argmax()]
print(f"maximum of L(p)     at p = {best.item():.2f}")
print(f"maximum of log L(p) at p = {best_log.item():.2f}")
assert torch.equal(L.argmax(), logL.argmax()), "log moved the argmax?!"
print(f"L(0.7) = {likelihood(torch.tensor(0.7)).item():.6f} but log L(0.7) = "
      f"{(7 * torch.log(torch.tensor(0.7)) + 3 * torch.log(torch.tensor(0.3))).item():.3f}")
print("Tiny product becomes a comfortable sum. Same winner, safer numbers.")

print()
print("=" * 60)
print("STEP 4: gaussian MLE by gradient, a bridge to module 04")
print("=" * 60)

# Noisy price measurements of ONE house, true center unknown to the model.
data = torch.distributions.Normal(250.0, 20.0).sample((200,))
print(f"200 noisy measurements, sample mean = {data.mean().item():.2f}")

# NLL of Normal(mu, sigma=20), dropping constant terms:
# nll(mu) proportional to sum((x - mu)^2). Minimizing it = maximizing likelihood.
mu = torch.tensor(100.0, requires_grad=True)  # start far away on purpose
lr = 0.1
for step in range(200):
    nll = ((data - mu) ** 2).sum() / (2 * 20.0**2)
    nll.backward()
    with torch.no_grad():
        mu -= lr * mu.grad  # walk downhill on the error score
    mu.grad.zero_()
    if step % 50 == 0:
        print(f"step {step:3d}: mu = {mu.item():8.2f}, nll = {nll.item():10.2f}")

print(f"final mu = {mu.item():.2f}, sample mean = {data.mean().item():.2f}")
assert abs(mu.item() - data.mean().item()) < 0.5, "gradient MLE != sample mean"
print("OK: the gaussian MLE of the center IS the sample mean, and gradient")
print("descent found it. Minimizing squared errors = maximizing likelihood.")
print("This is exactly why MSE is the default loss for regression.")

print()
print("=" * 60)
print("STEP 5: draw the likelihood curves")
print("=" * 60)

os.makedirs("figures", exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(ps.numpy(), L.numpy())
axes[0].axvline(0.7, color="red", linestyle="--", label="MLE = 0.7")
axes[0].set_title("Likelihood L(p) of 7 heads in 10 flips")
axes[0].set_xlabel("p")
axes[0].legend()
axes[1].plot(ps.numpy(), logL.numpy())
axes[1].axvline(0.7, color="red", linestyle="--", label="same maximum")
axes[1].set_title("log L(p): different shape, same peak")
axes[1].set_xlabel("p")
axes[1].legend()
fig.savefig(os.path.join("figures", "likelihood.png"), dpi=120, bbox_inches="tight")
print("saved figures/likelihood.png")
print()
print("Done. Now open exercises.py and complete the TODOs.")
