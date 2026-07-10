# Week 08 Companion - Serving Math, Queueing, Rust Async, and Safety Support

[<- Companion index](README.md) · [Master Plan](../MASTER-PLAN.md) · [cert plan](../nvidia-cert-track/month-2-ncp-genl/week-8/plan.md) · [build project](../gpu-engineering-lab/02-llm-engineering/week-08-mini-inference-server/README.md)

## Prerequisite Checklist

- You can distinguish TTFT, ITL, throughput, concurrency, and p95 latency.
- You can compute KV-cache memory from week 5.
- You understand batching as a latency/throughput trade-off.
- You can read basic Rust async code with `async`, `await`, and `tokio`.
- You know the difference between an inference engine, a server, and a packaged product like NIM.

## Mini Lesson

Serving is where model math meets operations. A request waits in a queue, runs prefill, then decodes tokens step by step. Continuous batching improves utilization by admitting new requests between decode steps.

Key metrics:

```text
TTFT = time to first token
ITL = inter-token latency
throughput = tokens / second
p95 latency = 95 percent of requests are faster than this
```

## Math Insight

Little's Law is the queueing sanity check:

```text
L = lambda * W
```

`L` is average items in the system, `lambda` is arrival rate, and `W` is average time in system. If arrival rate rises and service time does not fall, queues grow. That is the math behind autoscaling, backpressure, and admission control.

Paged KV cache is memory management: fixed-size blocks reduce fragmentation and make it possible to serve many sequences with changing lengths.

## Programming Primer

- Rust async: avoid blocking the async runtime with long CPU work; isolate model execution and request scheduling.
- `axum` gives routes and extractors; `tokio` drives async tasks; channels are useful for request queues.
- Safety/compliance: safety filters, prompt injection mitigation, audit logs, and data handling are deployment concerns, not separate from serving.

## 25-Minute Gate

1. Define TTFT, ITL, and throughput without notes.
2. Explain why continuous batching improves GPU utilization.
3. Compute KV-cache bytes for a small hypothetical model using the week 5 formula.
4. Identify what week 8 will compare against vLLM.
