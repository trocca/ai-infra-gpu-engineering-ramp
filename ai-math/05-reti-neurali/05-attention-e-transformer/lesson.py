"""Lesson 05: attention and transformer math.

Run with: python lesson.py
Scaled dot product attention built by hand on 4 tokens, checked against
F.scaled_dot_product_attention, then the causal mask. The last brick.
"""

import math
import os

import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")  # save figures to file, no display required
import matplotlib.pyplot as plt

torch.manual_seed(0)

print("=" * 60)
print("STEP 1: the 2 token hand example from EXPLANATION.md")
print("=" * 60)

q = torch.tensor([1.0, 0.0])
K2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
V2 = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
scores = K2 @ q / math.sqrt(2)  # dot with each key, scaled by sqrt(d)
weights = F.softmax(scores, dim=0)
out = weights @ V2
print(f"scores after scaling = {[round(v, 3) for v in scores.tolist()]}")
print(f"softmax weights      = {[round(v, 3) for v in weights.tolist()]}")
print(f"output               = {[round(v, 2) for v in out.tolist()]}")
assert torch.allclose(out, torch.tensor([6.7, 3.3]), atol=0.1)
print("Two thirds from value 1, one third from value 2. As computed by hand.")

print()
print("=" * 60)
print("STEP 2: full attention by hand, 4 tokens, dimension 3")
print("=" * 60)

# In a real transformer Q, K, V come from learned projections of the
# token embeddings. Here we take them directly: the mechanism is the point.
n, d = 4, 3
Q = torch.randn(n, d)
K = torch.randn(n, d)
V = torch.randn(n, d)


def attention_manual(Q, K, V, causal=False):
    scores = Q @ K.T / math.sqrt(Q.shape[-1])  # affinity: every query vs every key
    if causal:
        # No peeking at the future: -inf above the diagonal, softmax makes it 0.
        mask = torch.triu(torch.ones(len(Q), len(Q)), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)  # each row: a probability distribution
    return weights @ V, weights  # weighted average of values


out_manual, weights = attention_manual(Q, K, V)
print(f"attention weights (each ROW sums to 1):\n{weights}")
row_sums = weights.sum(dim=-1)
assert torch.allclose(row_sums, torch.ones(n), atol=1e-5)
print(f"row sums = {[round(v, 4) for v in row_sums.tolist()]}")

# The acid test against the torch built in.
out_torch = F.scaled_dot_product_attention(Q, K, V)
assert torch.allclose(out_manual, out_torch, atol=1e-5), "hand attention != torch"
print("OK: hand attention == F.scaled_dot_product_attention. Three matmuls")
print("and a softmax: you have been able to read this since module 01.")

print()
print("=" * 60)
print("STEP 3: the causal mask, no peeking at the future")
print("=" * 60)

out_causal, weights_causal = attention_manual(Q, K, V, causal=True)
print(f"causal attention weights:\n{weights_causal}")
# Upper triangle must be exactly zero: token i ignores tokens > i.
for i in range(n):
    for j in range(i + 1, n):
        assert weights_causal[i, j].item() == 0.0, "future leak!"
print("Zeros above the diagonal: token 0 sees only itself, token 3 sees all.")
out_torch_causal = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
assert torch.allclose(out_causal, out_torch_causal, atol=1e-5)
print("OK: matches F.scaled_dot_product_attention(is_causal=True).")
print("This mask is why GPT generates left to right: each new token may")
print("only consult the past. Training teaches WHAT to consult; the")
print("mechanism you just wrote decides HOW.")

print()
print("=" * 60)
print("STEP 4: who looks at whom, the attention heatmap")
print("=" * 60)

os.makedirs("figures", exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, w, title in [
    (axes[0], weights, "full attention"),
    (axes[1], weights_causal, "causal attention (masked future)"),
]:
    im = ax.imshow(w.numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("attended token (key)")
    ax.set_ylabel("attending token (query)")
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{w[i, j].item():.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax)
fig.savefig(os.path.join("figures", "attention_heatmap.png"), dpi=120, bbox_inches="tight")
print("saved figures/attention_heatmap.png")

print()
print("=" * 60)
print("STEP 5: the end of the path. Count the bricks.")
print("=" * 60)
print("""
A transformer block = attention (this lesson) + MLP (lesson 03),
stacked dozens of times, trained with cross entropy (module 03)
via backpropagation (lesson 04) using Adam (module 04), and inside
it all is matrix multiplication (module 01) steered by gradients
(module 02). There is no other mathematics in an LLM.

You did not just read the formula. You executed every piece of it.
""")
print("Done. The exercises of this last lesson await.")
