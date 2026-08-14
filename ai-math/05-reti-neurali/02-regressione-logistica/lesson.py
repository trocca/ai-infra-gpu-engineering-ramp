"""Lesson 02: logistic regression.

Run with: python lesson.py
Sigmoid and binary cross entropy by hand, checked against torch, then
the same training loop as lesson 01 with the new loss: a classifier.
"""

import os

import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")  # save figures to file, no display required
import matplotlib.pyplot as plt

torch.manual_seed(0)

X_raw = torch.tensor(
    [
        [50.0, 2.0],
        [70.0, 3.0],
        [90.0, 3.0],
        [110.0, 4.0],
        [130.0, 5.0],
    ]
)
prices = torch.tensor([150.0, 200.0, 250.0, 300.0, 350.0])
X = (X_raw - X_raw.mean(dim=0)) / X_raw.std(dim=0)
y = (prices >= 250).float()  # expensive = 1, cheap = 0
print(f"labels (expensive from 250 up): {y.tolist()}")

print()
print("=" * 60)
print("STEP 1: sigmoid by hand vs torch.sigmoid")
print("=" * 60)


def sigmoid_manual(z):
    return 1 / (1 + torch.exp(-z))


for z0 in [-2.0, 0.0, 2.0]:
    z = torch.tensor(z0)
    m, t = sigmoid_manual(z), torch.sigmoid(z)
    print(f"z = {z0:5.1f}   by hand = {m.item():.4f}   torch = {t.item():.4f}")
    assert torch.allclose(m, t, atol=1e-6)
print("OK: the squashing S, hand written. 0 maps to 0.5, the undecided point.")

print()
print("=" * 60)
print("STEP 2: binary cross entropy by hand vs F.binary_cross_entropy")
print("=" * 60)


def bce_manual(p, target):
    """Average surprise: -log p when truth is 1, -log(1-p) when truth is 0."""
    return -(target * torch.log(p) + (1 - target) * torch.log(1 - p)).mean()


p_test = torch.tensor([0.88, 0.12])
t_test = torch.tensor([1.0, 1.0])
m = bce_manual(p_test, t_test)
t = F.binary_cross_entropy(p_test, t_test)
print(f"model says [0.88, 0.12], truth [1, 1]")
print(f"per example surprise: -log(0.88) = {-torch.log(p_test[0]).item():.3f}, "
      f"-log(0.12) = {-torch.log(p_test[1]).item():.3f}")
print(f"BCE by hand = {m.item():.4f}   F.binary_cross_entropy = {t.item():.4f}")
assert torch.allclose(m, t, atol=1e-6)
print("OK. Confident and wrong costs 16x confident and right.")

print()
print("=" * 60)
print("STEP 3: the training loop, same skeleton, new loss")
print("=" * 60)

w = torch.zeros(2, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
lr = 0.5
for epoch in range(400):
    z = X @ w + b  # linear score (logit)
    p = sigmoid_manual(z)  # squash to probability
    loss = bce_manual(p, y)  # surprise vs the truth
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_()
    b.grad.zero_()
    if epoch % 100 == 0:
        print(f"epoch {epoch:3d}: loss = {loss.item():.4f}")
print(f"final loss = {loss.item():.4f}")

print()
print("=" * 60)
print("STEP 4: read the classifier's mind, house by house")
print("=" * 60)

with torch.no_grad():
    probs = sigmoid_manual(X @ w + b)
print("house (sqm, rooms)   P(expensive)   truth    verdict")
for i in range(5):
    verdict = "expensive" if probs[i] > 0.5 else "cheap"
    print(f"{[round(v, 1) for v in X_raw[i].tolist()]}        "
          f"{probs[i].item():10.3f}   {int(y[i].item())}        {verdict}")
correct = ((probs > 0.5).float() == y).all()
assert correct, "the tiny dataset is separable: accuracy must be 100%"
print("All five correct. Note the middle house: the model is least sure there,")
print("right on the price boundary. Calibrated doubt is a feature, not a bug.")

print()
print("=" * 60)
print("STEP 5: the S curve with the houses on it")
print("=" * 60)

os.makedirs("figures", exist_ok=True)
with torch.no_grad():
    zs = torch.linspace(-6, 6, 200)
    scores = X @ w + b
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(zs.numpy(), sigmoid_manual(zs).numpy(), label="sigmoid(z)")
ax.scatter(scores.numpy(), y.numpy(), color="red", zorder=5, label="houses (score, label)")
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("z (linear score)")
ax.set_ylabel("P(expensive)")
ax.set_title("The learned scores push each house to its side of the S")
ax.legend()
fig.savefig(os.path.join("figures", "sigmoid.png"), dpi=120, bbox_inches="tight")
print("saved figures/sigmoid.png")
print()
print("Done. Now open exercises.py and complete the TODOs.")
