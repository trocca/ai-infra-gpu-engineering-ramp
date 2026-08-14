"""Lesson 02: stochastic gradient descent and minibatches.

Run with: python lesson.py
200 synthetic houses, minibatch gradients compared with the exact one,
then a real SGD training loop with shuffle and epochs.
"""

import os

import torch
import matplotlib

matplotlib.use("Agg")  # save figures to file, no display required
import matplotlib.pyplot as plt

torch.manual_seed(0)

print("=" * 60)
print("STEP 1: the hand example, three points and their gradients")
print("=" * 60)

x3 = torch.tensor([1.0, 2.0, 3.0])
y3 = torch.tensor([2.0, 4.0, 6.0])
w0 = 1.0
per_example = 2 * (w0 * x3 - y3) * x3  # gradient of (w*x - y)^2 per example
print(f"per example gradients = {per_example.tolist()}")
print(f"exact gradient (mean) = {per_example.mean().item():.2f}")
for pair in [(0, 1), (0, 2), (1, 2)]:
    g = per_example[list(pair)].mean()
    print(f"minibatch {pair}: gradient = {g.item():6.2f}")
print("Different numbers, same sign, correct on average. Noise, not bias.")

print()
print("=" * 60)
print("STEP 2: 200 synthetic houses from the same rule as our 5")
print("=" * 60)

# Generate houses like the tiny dataset: price = 2*sqm + 10*rooms + 20 + noise.
n = 200
sqm = torch.rand(n) * 80 + 50  # between 50 and 130
rooms = (sqm / 25).round() + torch.randint(-1, 2, (n,)).float()
prices = 2 * sqm + 10 * rooms + 20 + torch.randn(n) * 15
X_raw = torch.stack([sqm, rooms], dim=1)
X = (X_raw - X_raw.mean(dim=0)) / X_raw.std(dim=0)
y = prices
print(f"X.shape = {tuple(X.shape)}, first house: {[round(v, 1) for v in X_raw[0].tolist()]}"
      f" -> price {y[0].item():.1f}")


def grad_on(w, b, Xb, yb):
    """Gradient of the MSE on a subset of data, via autograd."""
    wt = w.clone().requires_grad_(True)
    bt = b.clone().requires_grad_(True)
    loss = ((Xb @ wt + bt - yb) ** 2).mean()
    loss.backward()
    return wt.grad, bt.grad


w_now = torch.tensor([50.0, 5.0])
b_now = torch.tensor([100.0])
full_gw, full_gb = grad_on(w_now, b_now, X, y)
print(f"exact gradient on all 200 houses: w {[round(v, 1) for v in full_gw.tolist()]}"
      f", b {round(full_gb.item(), 1)}")
# Split the (shuffled) dataset into 10 disjoint batches of 20. Because the
# batches have EQUAL size and cover everything, their gradients average
# EXACTLY to the full gradient. Unbiasedness you can assert on.
perm = torch.randperm(n)
mini_gws = []
print("gradients of 10 disjoint batches of 20 (first 5 shown):")
for i in range(10):
    idx = perm[i * 20 : (i + 1) * 20]
    gw, gb = grad_on(w_now, b_now, X[idx], y[idx])
    mini_gws.append(gw)
    if i < 5:
        print(f"  batch {i}: w {[round(v, 1) for v in gw.tolist()]}, b {round(gb.item(), 1)}")
mean_gw = torch.stack(mini_gws).mean(dim=0)
print(f"mean of the 10 batch gradients: {[round(v, 1) for v in mean_gw.tolist()]}")
assert torch.allclose(mean_gw, full_gw, atol=0.1), "batch mean must equal the full grad"
print("Individually noisy, exactly right on average. Noise, not bias.")

print()
print("=" * 60)
print("STEP 3: the real SGD loop: shuffle, minibatch, epoch")
print("=" * 60)


def train_sgd(batch_size, epochs, lr=0.05, seed=1):
    torch.manual_seed(seed)
    w = torch.zeros(2)
    b = torch.zeros(1)
    history = []  # full loss recorded once per epoch, for a fair comparison
    for _ in range(epochs):
        perm = torch.randperm(n)  # shuffle: new minibatches every epoch
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            gw, gb = grad_on(w, b, X[idx], y[idx])
            w = w - lr * gw
            b = b - lr * gb
        history.append(((X @ w + b - y) ** 2).mean().item())
    return w, b, history


w_sgd, b_sgd, hist_sgd = train_sgd(batch_size=16, epochs=40)
w_full, b_full, hist_full = train_sgd(batch_size=n, epochs=40)  # full batch = plain GD
print(f"SGD  batch 16 : final loss = {hist_sgd[-1]:8.2f}, w = "
      f"{[round(v, 1) for v in w_sgd.tolist()]}, b = {round(b_sgd.item(), 1)}")
print(f"full batch 200: final loss = {hist_full[-1]:8.2f}, w = "
      f"{[round(v, 1) for v in w_full.tolist()]}, b = {round(b_full.item(), 1)}")
print("Similar endpoint. But per epoch, SGD made 13 cheap updates instead of 1.")
loss_at_start = (y**2).mean().item()  # loss with w = 0, b = 0
assert hist_sgd[-1] < loss_at_start / 100, "SGD should crush the starting loss"
print(f"(starting loss was {loss_at_start:.0f}; the floor is the noise in the")
print(f"prices, whose variance is 15^2 = 225. Nobody can predict pure noise.)")

os.makedirs("figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(hist_full, label="full batch GD (1 update per epoch)")
ax.plot(hist_sgd, label="SGD batch 16 (13 updates per epoch)")
ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel("loss on all data (log scale)")
ax.set_title("SGD: noisier path, much faster progress per epoch")
ax.legend()
fig.savefig(os.path.join("figures", "sgd_vs_full.png"), dpi=120, bbox_inches="tight")
print("saved figures/sgd_vs_full.png")
print()
print("Done. Now open exercises.py and complete the TODOs.")
