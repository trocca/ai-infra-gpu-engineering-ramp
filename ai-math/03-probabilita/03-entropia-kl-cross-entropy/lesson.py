"""Lesson 03: entropy, KL divergence and cross entropy.

Run with: python lesson.py
Surprise, entropy, KL and cross entropy by hand, checked against
F.kl_div, F.softmax and F.cross_entropy. This is where the
classification loss comes from.
"""

import os

import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")  # save figures to file, no display required
import matplotlib.pyplot as plt

torch.manual_seed(0)

print("=" * 60)
print("STEP 1: surprise = -log p")
print("=" * 60)

for p in [0.99, 0.5, 0.1, 0.001]:
    print(f"p = {p:6.3f}   surprise = -log p = {-torch.log(torch.tensor(p)).item():6.3f}")
print("Certain events bore us (about 0), rare events shock us (large).")

print()
print("=" * 60)
print("STEP 2: entropy = average surprise")
print("=" * 60)


def entropy(p):
    """H(P) = sum of p * (-log p). Average surprise of the next sample."""
    return -(p * torch.log(p)).sum()


fair = torch.tensor([0.5, 0.5])
loaded = torch.tensor([0.9, 0.1])
print(f"fair coin   [0.5, 0.5]: H = {entropy(fair).item():.3f}   (log 2 = 0.693)")
print(f"loaded coin [0.9, 0.1]: H = {entropy(loaded).item():.3f}")
assert torch.allclose(entropy(fair), torch.log(torch.tensor(2.0)), atol=1e-6)
print("The fair coin is maximally unpredictable: highest entropy.")

print()
print("=" * 60)
print("STEP 3: KL divergence = average EXTRA surprise from a wrong model")
print("=" * 60)


def kl(p, q):
    """KL(P||Q) = sum of p * (log p - log q)."""
    return (p * (torch.log(p) - torch.log(q))).sum()


p_true = torch.tensor([0.9, 0.1])  # the world
q_good = torch.tensor([0.8, 0.2])  # decent model
q_bad = torch.tensor([0.2, 0.8])  # terrible model
print(f"world P = {p_true.tolist()}")
print(f"KL(P || {q_good.tolist()}) = {kl(p_true, q_good).item():.4f}   (small: model close)")
print(f"KL(P || {q_bad.tolist()}) = {kl(p_true, q_bad).item():.4f}   (large: model far)")
print(f"KL(P || P)            = {kl(p_true, p_true).item():.4f}   (zero: perfect model)")

# torch check: F.kl_div expects LOG probabilities of Q as input, P as target.
kl_torch = F.kl_div(q_good.log(), p_true, reduction="sum")
assert torch.allclose(kl(p_true, q_good), kl_torch, atol=1e-6)
print("OK: hand KL matches F.kl_div. Note KL(P||Q) != KL(Q||P): not symmetric.")

print()
print("=" * 60)
print("STEP 4: cross entropy = entropy + KL, and the one hot shortcut")
print("=" * 60)


def cross_entropy_dist(p, q):
    """H(P, Q) = sum of p * (-log q). Total average surprise using model q."""
    return -(p * torch.log(q)).sum()


ce = cross_entropy_dist(p_true, q_good)
print(f"H(P,Q) = {ce.item():.4f}")
print(f"H(P) + KL(P||Q) = {(entropy(p_true) + kl(p_true, q_good)).item():.4f}")
assert torch.allclose(ce, entropy(p_true) + kl(p_true, q_good), atol=1e-6)

# One hot truth: all mass on class 0. Only one term survives: -log q[0].
one_hot = torch.tensor([1.0, 0.0])
q = torch.tensor([0.8, 0.2])
print(f"truth one hot {one_hot.tolist()}, model {q.tolist()}:")
print(f"full formula   = {-(one_hot * torch.log(q)).sum().item():.4f}")
print(f"-log q[truth]  = {-torch.log(q[0]).item():.4f}   (same: only one term is alive)")

print()
print("=" * 60)
print("STEP 5: softmax by hand, then F.cross_entropy end to end")
print("=" * 60)

# A classifier outputs raw scores (logits). softmax turns them into
# probabilities: positive, summing to 1, big score -> big probability.
logits = torch.tensor([2.0, 0.5, -1.0])


def softmax_manual(z):
    """Stable softmax: subtract the max first so exp cannot overflow."""
    shifted = z - z.max()
    e = torch.exp(shifted)
    return e / e.sum()


probs_manual = softmax_manual(logits)
probs_torch = F.softmax(logits, dim=0)
print(f"logits            = {logits.tolist()}")
print(f"softmax by hand   = {[round(p, 4) for p in probs_manual.tolist()]}")
print(f"F.softmax         = {[round(p, 4) for p in probs_torch.tolist()]}")
assert torch.allclose(probs_manual, probs_torch, atol=1e-6)

# Cross entropy loss for true class 0, from the chain: logits -> softmax -> -log.
target = 0
ce_manual = -torch.log(probs_manual[target])
ce_torch = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target]))
print(f"CE by hand (-log softmax[0]) = {ce_manual.item():.4f}")
print(f"F.cross_entropy              = {ce_torch.item():.4f}")
assert torch.allclose(ce_manual, ce_torch, atol=1e-5)
print("OK: F.cross_entropy IS softmax + minus log of the true class.")

print()
print("=" * 60)
print("STEP 6: entropy of a coin as p varies")
print("=" * 60)

os.makedirs("figures", exist_ok=True)
ps = torch.linspace(0.001, 0.999, 200)
H = -(ps * torch.log(ps) + (1 - ps) * torch.log(1 - ps))
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(ps.numpy(), H.numpy())
ax.axvline(0.5, color="red", linestyle="--", label="max at p = 0.5")
ax.set_xlabel("p (probability of heads)")
ax.set_ylabel("entropy (nats)")
ax.set_title("Coin entropy: unpredictability peaks at a fair coin")
ax.legend()
fig.savefig(os.path.join("figures", "entropia_moneta.png"), dpi=120, bbox_inches="tight")
print("saved figures/entropia_moneta.png")
print()
print("Done. Now open exercises.py and complete the TODOs.")
