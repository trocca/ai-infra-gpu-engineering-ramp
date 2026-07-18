# Week 05 Companion - Transformer Math, PyTorch Architecture, and Tokenizer Support

[<- Companion index](README.md) · [Master Plan](../MASTER-PLAN.md) · [cert plan](../nvidia-cert-track/month-2-ncp-genl/week-5/plan.md) · [build project](../gpu-engineering-lab/02-llm-engineering/week-05-gpt-from-scratch/README.md)

## Prerequisite Checklist

- You can multiply matrix shapes for projections and attention.
- You understand probability distributions, logits, softmax, and cross-entropy at working level.
- You can write a small `torch.nn.Module` and know where parameters live.
- You know BPE/tokenizer basics: text becomes token IDs before it becomes tensors.
- You can explain train vs generate mode for a language model.

## Mini Lesson

A decoder-only transformer is a repeated block that transforms token embeddings into next-token logits. The build week is hard only if the shapes feel mysterious. Make the shapes explicit.

For batch `B`, sequence length `T`, model width `C`, heads `H`, and head size `D = C / H`:

```text
x: B x T x C
q, k, v: B x H x T x D
scores = q @ k^T: B x H x T x T
attention = softmax(scores / sqrt(D)) @ v: B x H x T x D
```

## Math Insight

The `1 / sqrt(D)` scaling keeps dot products from growing with head dimension. If q and k have roughly unit variance, their dot product has variance proportional to `D`. Without scaling, large scores push softmax into saturation and gradients become poor.

KV-cache memory is the serving trap:

```text
KV bytes = 2 * layers * kv_heads * head_dim * sequence_length * batch * bytes_per_value
```

The `2` is K plus V. This one formula explains why inference is often memory-bound.

## Playbook Bridge

Before this week, read the
[Week 5 Ultra-Scale Playbook bridge](../references/hf-ultrascale-playbook.md#week-5---transformer-memory-and-single-gpu-training).
Add a memory ledger to `notes.md`: parameters, gradients, optimizer states, activations,
and KV cache. The core habit is to predict memory pressure before PyTorch tells you.

## Programming Primer

- PyTorch modules register parameters when assigned as `nn.Parameter` or submodules.
- `model.train()` and `model.eval()` change behavior for layers like dropout; use them deliberately.
- Tokenizers are part of the model contract. Train and inference must use the same tokenizer and chat template.

## 25-Minute Gate

1. For `B=2`, `T=128`, `C=768`, `H=12`, compute `D` and the attention score tensor shape.
2. Explain why causal masking uses `-inf` before softmax.
3. Read the week 5 project tests and identify which one protects attention correctness.
