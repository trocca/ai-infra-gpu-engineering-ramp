# Week 11 Companion - Kubernetes GPU Serving, Scheduling Invariants, and Observability Support

[<- Companion index](README.md) · [Master Plan](../MASTER-PLAN.md) · [cert plan](../nvidia-cert-track/month-3-ncp-aio/week-11/plan.md) · [build project](../gpu-engineering-lab/03-scale-and-serve/week-11-k8s-gpu-serving/README.md)

## Prerequisite Checklist

- You can deploy and inspect a Kubernetes workload without looking up every command.
- You know how GPUs are advertised to Kubernetes through the NVIDIA device plugin.
- You can explain GPU Operator components.
- You can distinguish Prometheus metrics, Grafana dashboards, and alerting rules.
- You can read a pod event and decide whether the problem is image pull, scheduling, runtime, or app readiness.

## Mini Lesson

Kubernetes GPU serving is an invariants game:

- a node must have working driver/runtime/device-plugin plumbing;
- a pod must request the GPU resource that exists on that node;
- the container image must match CUDA/runtime expectations;
- the service must expose the app;
- metrics must prove the GPU is doing useful work.

If one invariant breaks, the symptom may appear far away from the cause.

## Math Insight

Autoscaling needs a stable signal. For serving, GPU utilization alone can mislead: a saturated queue with low GPU utilization may be CPU/tokenizer bound or blocked by scheduling. Track:

```text
queue_depth
request_rate
TTFT / p95 latency
tokens_per_second
GPU_utilization
GPU_memory_used
```

A useful HPA metric is one that rises before users feel pain and falls when adding replicas helps.

## Playbook Bridge

Read the
[Week 11 Ultra-Scale Playbook bridge](../references/hf-ultrascale-playbook.md#week-11---benchmarking-observability-and-cluster-reality).
Borrow the benchmarking discipline: profiler traces, NCCL/debug logs, restart behavior,
and resource graphs are evidence. A Kubernetes workload is not healthy merely because
the YAML applied.

## Programming Primer

- `kubectl describe pod` for scheduling and lifecycle events.
- `kubectl logs` for app/runtime errors.
- DCGM exporter surfaces GPU metrics to Prometheus.
- Readiness probes should reflect whether the server can accept traffic, not merely whether the process exists.
- Image size and cold start are part of the week 8 Rust-serving story; measure them.

## 25-Minute Gate

1. Write the shortest command sequence to debug a Pending GPU pod.
2. Explain what GPU Operator installs.
3. Pick three metrics that prove an inference service is healthy.
4. Read the week 11 project README and identify the demo scene it should feed.
