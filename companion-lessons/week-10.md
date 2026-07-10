# Week 10 Companion - Tensor/Pipeline Parallelism Math and Admin Command Support

[<- Companion index](README.md) · [Master Plan](../MASTER-PLAN.md) · [cert plan](../nvidia-cert-track/month-3-ncp-aio/week-10/plan.md) · [build project](../gpu-engineering-lab/03-scale-and-serve/week-10-parallelism-internals/README.md)

## Prerequisite Checklist

- You can distinguish data, tensor, pipeline, and expert parallelism.
- You can reason about matrix shapes for column-parallel and row-parallel linear layers.
- You understand all-reduce, all-gather, and reduce-scatter at a high level.
- You can explain Slurm drain/resume and Kubernetes cordon/drain.
- You know what MIG profiles trade off: isolation, memory, and capacity.

## Mini Lesson

Tensor parallelism splits a layer across GPUs. Pipeline parallelism splits layers across stages. Both are ways to fit or speed up models that exceed the comfortable limits of one GPU, but both create communication costs.

Column-parallel linear:

```text
Y = X @ W
W split by output columns
each GPU computes part of Y
optionally all-gather Y
```

Row-parallel linear:

```text
X and W split by input dimension
each GPU computes partial Y
all-reduce partial results
```

## Math Insight

Pipeline parallelism has a bubble: some stages are idle at the beginning and end. A simple efficiency approximation for `p` pipeline stages and `m` microbatches is:

```text
efficiency ~= m / (m + p - 1)
```

More microbatches reduce the bubble, but increase activation bookkeeping and scheduling complexity.

## Programming Primer

- Shape assertions are not optional. Most tensor-parallel bugs are silent shape mistakes until a collective hangs.
- Slurm admin verbs: `sinfo`, `squeue`, `scontrol show`, drain, resume, hold, release, cancel.
- Kubernetes admin verbs: inspect, cordon, drain, label, taint, describe events, read logs.
- MIG: `single` exposes uniform slices as `nvidia.com/gpu`; `mixed` exposes profile-specific resources.

## 25-Minute Gate

1. Draw column-parallel and row-parallel linear layers.
2. Compute pipeline efficiency for `p=4`, `m=4`, then `m=16`.
3. Say when you would prefer MIG over time-slicing.
4. Open the week 10 cert plan and highlight every command you need to drill.
