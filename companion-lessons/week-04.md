# Week 04 Companion - Operations Primitives, Normalization Math, and PyTorch Extension Support

[<- Companion index](README.md) · [Master Plan](../MASTER-PLAN.md) · [cert plan](../nvidia-cert-track/month-1-nca-aiio/week-4/plan.md) · [build project](../gpu-engineering-lab/01-foundations/week-04-pytorch-custom-ops/README.md)

## Prerequisite Checklist

- You can explain Kubernetes pod, node, container, resource request, and operator.
- You can explain Slurm job, partition, node state, and scheduler at a high level.
- You can derive stable softmax with the max trick.
- You can explain LayerNorm vs RMSNorm.
- You know what PyO3, maturin, and DLPack are for.

## Mini Lesson

Week 4 closes the first month by connecting operations vocabulary to code that plugs into PyTorch. The cert side asks: how do GPU workloads get scheduled, monitored, and isolated? The build side asks: how do low-level kernels become a Python package a model can use?

That bridge is the job profile in miniature: explain the system, then make it runnable.

## Math Insight

Softmax turns scores into probabilities:

```text
softmax(x_i) = exp(x_i) / sum_j exp(x_j)
```

For stability, subtract the max:

```text
softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
```

This does not change the result because the same constant is subtracted from every score, but it prevents overflow.

LayerNorm normalizes with mean and variance. RMSNorm skips mean-centering and divides by root mean square:

```text
rms(x) = sqrt(mean(x^2) + eps)
y = weight * x / rms(x)
```

## Programming Primer

- PyTorch extension: keep Python API small, test against `torch.nn` or `torch.nn.functional`, and benchmark eager, compiled, and custom paths.
- DLPack/data pointer exchange is powerful but dangerous: shape, dtype, device, and lifetime must be correct.
- Operations: memorize "GPU Operator installs and manages the driver, container toolkit, device plugin, DCGM exporter, and MIG manager."

## 25-Minute Gate

1. Explain why subtracting `max(x)` does not change softmax.
2. Write the one-line difference between MIG and time-slicing.
3. Identify how the week 4 wheel will be installed and tested.
4. Decide before Monday what gets lighter because this is also exam week.
