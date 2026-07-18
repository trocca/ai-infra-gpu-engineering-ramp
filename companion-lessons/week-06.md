# Week 06 Companion - LoRA Rank Math, QLoRA, and Evaluation Support

[<- Companion index](README.md) · [Master Plan](../MASTER-PLAN.md) · [cert plan](../nvidia-cert-track/month-2-ncp-genl/week-6/plan.md) · [build project](../gpu-engineering-lab/02-llm-engineering/week-06-lora-from-scratch/README.md)

## Prerequisite Checklist

- You can explain a matrix rank as the number of independent directions it can express.
- You can compute parameter counts for a dense projection and a LoRA adapter.
- You understand freezing parameters and passing only trainable parameters to an optimizer.
- You can distinguish changing model knowledge from changing model behavior/style.
- You know that evaluation includes both automatic metrics and qualitative inspection.

## Mini Lesson

LoRA freezes a dense weight matrix `W` and learns a low-rank update:

```text
W' = W + (alpha / r) * B @ A
```

If `W` is `d_out x d_in`, then:

```text
A: r x d_in
B: d_out x r
LoRA params = r * d_in + d_out * r
```

For a square `d x d` projection, dense params are `d^2`, while LoRA params are about `2*d*r`. With `d=4096` and `r=16`, LoRA uses `131,072` params instead of `16,777,216`.

## Math Insight

Low rank is a constraint, not a trick. The bet is that task adaptation lives in a smaller subspace than full pretraining. Rank `r` controls capacity; `alpha / r` controls update scale. Initialize B to zero so the adapter starts as a no-op and the base model's initial behavior is preserved.

QLoRA adds memory savings by quantizing the frozen base weights while keeping adapters trainable in higher precision.

## Playbook Bridge

Read the
[Week 6 Ultra-Scale Playbook bridge](../references/hf-ultrascale-playbook.md#week-6---fine-tuning-memory-pressure-and-global-batch-math).
Use it to connect LoRA/QLoRA to the larger training-memory story: frozen base weights
do not need adapter optimizer states, gradient accumulation raises global batch without
raising per-step activation memory, and ZeRO/FSDP are the full-training analogs.

## Programming Primer

- In PyTorch, set `requires_grad=False` on frozen base parameters.
- Build optimizer parameter groups from adapter parameters only.
- State dicts should separate base weights, adapter weights, and merged weights.
- Evaluation should compare against HF PEFT so your from-scratch implementation has an industry reference.

## 25-Minute Gate

1. Compute LoRA params for a `4096 x 4096` projection with `r=8`, `16`, and `64`.
2. Explain why LoRA can have zero inference overhead after merge.
3. Write the discriminator: RAG changes what? Fine-tuning changes what?
