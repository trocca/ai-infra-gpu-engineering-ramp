# Week 8 Notes — Deployment, Monitoring, Safety + Review

## Serving stack map
### Engine layer: vLLM / TensorRT-LLM / SGLang — one-liners
-
-
### Triton Inference Server — role, backends, ensembles
-
-
### NIM — what it packages, profiles, API
-
-
### K8s layer: NIM Operator, autoscaling, MIG/time-slicing/MPS, DRA (demo-repo tie-in)
-
-

## Serving topologies
### Single GPU / TP multi-GPU / replicas + LB
-
### Disaggregated prefill-decode (Dynamo), KV-aware routing
-
-
### Multi-LoRA serving
-
### Topology → workload matching notes
-

## Metrics
### TTFT / ITL (TPOT) / e2e / throughput / goodput — definitions
-
-
### Percentiles; why not means; SLO framing
-
### Little's law intuition
-

## GPU & system monitoring
### DCGM + dcgm-exporter; key fields
-
-
### Why "GPU utilization" lies
-
### XID errors; node health remediation
-
### Engine metrics: queue depth, KV utilization, preemptions
-

## Reliability patterns
### Probes for model servers
-
### Canary / shadow / A-B; promotion gates
-
-
### Drift: input vs output; golden-set evals in CI
-

## Safety
### Prompt injection: direct vs indirect (RAG attack)
-
-
### Defense layers
-
### NeMo Guardrails: rail types, Colang, content-safety models
-
-
### Alignment (in-model) vs guardrails (around-model)
-

## Compliance
### PII: curation-time vs inference-time vs logging
-
-
### Licensing: open-weights vs open-source; Llama license vs Apache-2.0
-
-
### Model cards; bias evals; EU AI Act / GDPR awareness
-

## Lab results (FP16 vs AWQ numbers)
| Metric | FP16 | AWQ |
|---|---|---|
| TTFT (mean/P99) | | |
| Output tok/s | | |
| Requests/s | | |
| VRAM used | | |

## Mock exam misses (concept + discriminator, one line each)
-
-
-

## Weak-domain list after Day 3 re-runs
-
