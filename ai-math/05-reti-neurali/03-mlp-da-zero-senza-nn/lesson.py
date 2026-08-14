"""Lesson 03: an MLP from scratch, no nn module.

Run with: python lesson.py
The hand built XOR network verified on the truth table, then a randomly
initialized MLP trained until it rediscovers the solution. Raw tensors.
"""

import os

import torch
import matplotlib

matplotlib.use("Agg")  # save figures to file, no display required
import matplotlib.pyplot as plt

torch.manual_seed(0)

# The XOR problem: no straight line separates the two classes.
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([0.0, 1.0, 1.0, 0.0])

print("=" * 60)
print("STEP 1: the hand built XOR network from SPIEGAZIONE")
print("=" * 60)

# Weights chosen by reasoning, not training. Hidden neuron 1 computes
# ReLU(x1 + x2), hidden neuron 2 computes ReLU(x1 + x2 - 1).
W1_hand = torch.tensor([[1.0, 1.0], [1.0, 1.0]])  # (2 inputs, 2 hidden)
b1_hand = torch.tensor([0.0, -1.0])
W2_hand = torch.tensor([1.0, -2.0])  # combine: h1 - 2*h2
b2_hand = torch.tensor(0.0)


def forward(x, W1, b1, W2, b2):
    h = torch.relu(x @ W1 + b1)  # hidden layer: new features, then the bend
    return h @ W2 + b2  # output layer: combine the bent features


out_hand = forward(X, W1_hand, b1_hand, W2_hand, b2_hand)
print("x1 x2   h1 h2   out   target")
h_hand = torch.relu(X @ W1_hand + b1_hand)
for i in range(4):
    print(f" {X[i,0]:.0f}  {X[i,1]:.0f}    {h_hand[i,0]:.0f}  {h_hand[i,1]:.0f}"
          f"    {out_hand[i]:.0f}     {y[i]:.0f}")
assert torch.allclose(out_hand, y), "the hand built network must match XOR exactly"
print("OK: exact XOR. The hidden neurons ARE invented features:")
print("h1 = 'at least one on', h2 = 'both on', out = h1 - 2*h2 = 'exactly one'.")

print()
print("=" * 60)
print("STEP 2: can a purely linear model do this? No.")
print("=" * 60)

# Best linear fit on XOR via least squares: it answers 0.5 everywhere.
A = torch.cat([X, torch.ones(4, 1)], dim=1)
sol = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.squeeze()
lin_preds = A @ sol
print(f"best possible linear predictions: {[round(v, 2) for v in lin_preds.tolist()]}")
print("It shrugs: 0.5 on every point. No line separates XOR. The bend is needed.")

print()
print("=" * 60)
print("STEP 3: train an MLP from random weights, 16 hidden neurons")
print("=" * 60)

# XOR's loss landscape has real local minima (dead ReLU corners) where
# plain GD gets stuck. Two standard remedies, both used here:
# extra width (16 neurons: more spare bends than the 2 strictly needed)
# and Adam, the optimizer YOU implemented by hand in module 04 lesson 03.
hidden = 16
W1 = torch.randn(2, hidden, requires_grad=True)
b1 = torch.zeros(hidden, requires_grad=True)
W2 = torch.randn(hidden, requires_grad=True)
b2 = torch.zeros(1, requires_grad=True)
params = [W1, b1, W2, b2]
n_params = sum(p.numel() for p in params)
print(f"parameters to learn: {n_params}   (GPT class models: around 10^12. Same loop.)")

optimizer = torch.optim.Adam(params, lr=0.01)
for epoch in range(3000):
    optimizer.zero_grad()  # same job as grad.zero_() in lesson 01
    out = forward(X, W1, b1, W2, b2).squeeze()
    loss = ((out - y) ** 2).mean()
    loss.backward()
    optimizer.step()  # the six Adam lines from module 04, prepackaged
    if epoch % 500 == 0:
        print(f"epoch {epoch:4d}: loss = {loss.item():.6f}")

with torch.no_grad():
    final = forward(X, W1, b1, W2, b2).squeeze()
print(f"final outputs: {[round(v, 3) for v in final.tolist()]}   (target {y.tolist()})")
predictions = (final > 0.5).float()
assert torch.equal(predictions, y), "the trained MLP must solve XOR"
assert loss.item() < 0.01, "loss should be tiny at the end"
print("OK: gradient descent rediscovered a bent solution on its own.")

print()
print("=" * 60)
print("STEP 4: the decision map, where the network says 0 or 1")
print("=" * 60)

os.makedirs("figures", exist_ok=True)
grid = torch.linspace(-0.5, 1.5, 200)
gx, gy = torch.meshgrid(grid, grid, indexing="xy")
points = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
with torch.no_grad():
    zmap = forward(points, W1, b1, W2, b2).reshape(200, 200)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.contourf(gx.numpy(), gy.numpy(), zmap.numpy(), levels=20, cmap="RdBu_r")
fig.colorbar(im, ax=ax, label="network output")
ax.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=y.numpy(), cmap="RdBu_r",
           edgecolors="black", s=120, zorder=5)
ax.set_title("XOR decision map: a bent boundary, impossible for a line")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
fig.savefig(os.path.join("figures", "xor_map.png"), dpi=120, bbox_inches="tight")
print("saved figures/xor_map.png")
print()
print("Done. Now open exercises.py and complete the TODOs.")
