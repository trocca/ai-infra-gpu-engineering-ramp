# Week 09 Companion - Distributed Training, All-Reduce Cost, and NCCL Support

[<- Companion index](README.md) · [Master Plan](../MASTER-PLAN.md) · [cert plan](../nvidia-cert-track/month-3-ncp-aio/week-9/plan.md) · [build project](../gpu-engineering-lab/03-scale-and-serve/week-09-distributed-training/README.md)

## Prerequisite Checklist

- You can explain data parallel training: each GPU gets different data, then gradients are synchronized.
- You know what an all-reduce does.
- You can distinguish process, rank, local rank, world size, and process group.
- You can read basic `torch.distributed` code.
- You have the cloud multi-GPU setup plan open before Monday.

## Mini Lesson

DDP works because every worker starts from the same weights, computes gradients on different mini-batches, averages gradients, and applies the same optimizer step. The synchronization primitive is all-reduce.

Ring all-reduce has two phases:

1. reduce-scatter: split and reduce chunks around the ring;
2. all-gather: circulate reduced chunks so every rank has the full result.

## Math Insight

For tensor size `S` bytes and `N` GPUs, ring all-reduce moves roughly this much data per rank:

```text
bytes_per_rank ~= 2 * (N - 1) / N * S
```

Latency also matters because the ring has `2 * (N - 1)` communication steps. This explains why small tensors can be latency-bound while large gradient buckets are bandwidth-bound.

Scaling efficiency:

```text
efficiency = single_gpu_time / (num_gpus * multi_gpu_time)
```

## Programming Primer

- `RANK` is the global process index; `LOCAL_RANK` selects the GPU on the node.
- NCCL is the GPU collective backend; `NCCL_DEBUG=INFO` is your first visibility tool.
- DDP overlaps communication with backward using gradient buckets and hooks.
- FSDP shards parameters/gradients/optimizer state to reduce memory.

## 25-Minute Gate

1. Explain all-reduce to a non-specialist in three sentences.
2. For `N=2` and a 1 GB gradient tensor, estimate ring all-reduce bytes per rank.
3. Define rank, local rank, and world size.
4. Confirm the cloud GPU budget and teardown steps before launching anything.
