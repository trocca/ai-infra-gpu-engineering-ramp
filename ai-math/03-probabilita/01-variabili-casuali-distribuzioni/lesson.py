"""Lesson 01: random variables and distributions.

Run with: python lesson.py
Dice, biased coins and gaussians simulated with torch.distributions,
empirical means converging to expected values, and two figures.
"""

import os

import torch
import matplotlib

matplotlib.use("Agg")  # save figures to file, no display required
import matplotlib.pyplot as plt

torch.manual_seed(0)  # same random numbers at every run

print("=" * 60)
print("STEP 1: a fair die, expectation by hand vs simulation")
print("=" * 60)

values = torch.arange(1.0, 7.0)
probs = torch.full((6,), 1 / 6)
expected = (values * probs).sum()  # weighted average: the definition of E[X]
print(f"values = {values.tolist()}")
print(f"E[X] by hand = (1+2+3+4+5+6)/6 = {expected.item():.4f}")

n = 100_000
rolls = torch.randint(1, 7, (n,)).float()
print(f"mean of {n} simulated rolls = {rolls.mean().item():.4f}")
assert abs(rolls.mean().item() - 3.5) < 0.02, "empirical mean too far from 3.5"
print("OK: the empirical mean lands on the expected value.")

print()
print("=" * 60)
print("STEP 2: a biased coin, Bernoulli(0.7)")
print("=" * 60)

coin = torch.distributions.Bernoulli(probs=0.7)
flips = coin.sample((n,))
print(f"first 20 flips: {flips[:20].int().tolist()}")
print(f"E[X] by hand   = 0.7,   empirical mean = {flips.mean().item():.4f}")
print(f"var  by hand   = 0.21,  empirical var  = {flips.var().item():.4f}")
assert abs(flips.mean().item() - 0.7) < 0.01
assert abs(flips.var().item() - 0.21) < 0.01
print("OK: mean p and variance p(1-p), as computed by hand.")

print()
print("=" * 60)
print("STEP 3: the gaussian, center mu and width sigma")
print("=" * 60)

mu, sigma = 250.0, 40.0  # think: house prices around 250 with noise
gauss = torch.distributions.Normal(mu, sigma)
samples = gauss.sample((n,))
print(f"Normal(mu={mu}, sigma={sigma})")
print(f"empirical mean = {samples.mean().item():8.2f}   (expected {mu})")
print(f"empirical std  = {samples.std().item():8.2f}   (expected {sigma})")
inside = ((samples > mu - 2 * sigma) & (samples < mu + 2 * sigma)).float().mean()
print(f"fraction within mu +/- 2 sigma = {inside.item():.3f}   (theory: about 0.95)")
assert abs(samples.mean().item() - mu) < 1.0
assert abs(inside.item() - 0.954) < 0.01
print("OK: the 2 sigma rule holds. Useful for spotting outliers at a glance.")

print()
print("=" * 60)
print("STEP 4: the law of large numbers, watched live")
print("=" * 60)

os.makedirs("figures", exist_ok=True)
flips_small = coin.sample((2000,))
running_mean = flips_small.cumsum(0) / torch.arange(1, 2001)
print(f"running mean after     10 flips: {running_mean[9].item():.3f}")
print(f"running mean after    100 flips: {running_mean[99].item():.3f}")
print(f"running mean after  2,000 flips: {running_mean[-1].item():.3f}")
print("Noisy at the start, glued to 0.7 at the end. More data, less noise.")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(running_mean.numpy())
axes[0].axhline(0.7, color="red", linestyle="--", label="E[X] = 0.7")
axes[0].set_title("Law of large numbers: running mean of coin flips")
axes[0].set_xlabel("number of flips")
axes[0].legend()

axes[1].hist(samples.numpy(), bins=60, density=True, alpha=0.7)
xs = torch.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
pdf = torch.exp(gauss.log_prob(xs))  # log_prob gives log density, exp undoes it
axes[1].plot(xs.numpy(), pdf.numpy(), "r", label="theoretical density")
axes[1].set_title(f"Histogram of {n} gaussian samples vs the bell curve")
axes[1].legend()
fig.savefig(os.path.join("figures", "distribuzioni.png"), dpi=120, bbox_inches="tight")
print("saved figures/distribuzioni.png")
print()
print("Done. Now open exercises.py and complete the TODOs.")
