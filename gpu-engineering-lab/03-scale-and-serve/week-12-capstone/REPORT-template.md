# GPU Engineering Lab — Capstone Report

> Template — COMPLETE with section prompts. Replace every *(prompt)* block
> with your content; delete the prompts. Target length: 3–5 pages. Write it
> for a senior NVIDIA engineer with 10 minutes; put numbers in the first
> screen.

## TL;DR

*(3–5 bullets. Each bullet = one claim + one number. Example shape: "Built
continuous batching from scratch; 14× throughput vs sequential serving at
p95 TTFT under 800 ms on one L4." No adjectives without numbers.)*

## What I built

*(One paragraph per phase, past tense, concrete. Phase 1: CUDA/kernels.
Phase 2: training + inference internals. Phase 3: multi-GPU + serving.
End with the capstone pipeline sentence: the one command, what it produces,
where it runs.)*

### System diagram

*(One mermaid diagram: pipeline stages on the left, the K8s serving stack
on the right, arrows for the artifact flow. If you can't draw it simply,
the architecture isn't clear yet.)*

## The numbers

### Headline table

| Claim | Baseline | Mine | Delta | Where measured |
|---|---|---|---|---|
| *(e.g. matmul vs cuBLAS)* | | | | *(week / hardware)* |
| *(e.g. attention kernel vs SDPA)* | | | | |
| *(e.g. serving throughput vs naive)* | | | | |
| *(e.g. ring all-reduce vs NCCL)* | | | | |
| *(e.g. scaling efficiency 1→2 GPU)* | | | | |
| *(end-to-end: quantized artifact TTFT/tok-s on K8s)* | | | | |

*(Every number links to the JSON + script that produced it. Numbers a
reviewer cannot reproduce are marketing, not engineering.)*

### Measurement honesty

*(Hardware for each number — RTX 5090 laptop vs L4 cloud, and why that
matters. Variance: how many runs, what you report — median? p95? What you
did NOT control for.)*

## What didn't work

*(The credibility section — 3+ entries, specific. For each: what I tried,
what I expected, what actually happened, root cause if found, what I did
instead. "My first ring all-reduce deadlocked because..." beats any
success story. Include at least one thing still unresolved.)*

## What I learned

*(Not a topic list — 4–6 non-obvious, earned insights. The test: would this
sentence surprise you-from-12-weeks-ago? Example shape: "TP on PCIe loses
N% because the all-reduce sits in the critical path of EVERY layer — I
knew 'TP needs NVLink' as a slogan; now I know it as a measured cost.")*

## What I'd do with 8×H100

*(The scaling-judgment section. Concretely: which config — TP within the
NVLink domain × DP across? FSDP threshold? What changes qualitatively vs
2×L4-PCIe: NVLink bandwidth, larger microbatches, real gang scheduling
pressure. What experiment would you run FIRST and what would it decide?
Estimate: model size trainable in a weekend on 8×H100 with your stack,
with your arithmetic shown.)*

## What's next

*(3–5 bullets: gaps you know about — multi-node NCCL, 1F1B, speculative
decoding, DRA — and one sentence each on why it's the natural continuation.)*

## Appendix

- Repo map: *(one line per phase folder)*
- Total cloud spend: *(from the cost logs, per week)*
- Reproduce-everything guide: *(link to root README section)*
- Related: NCP-AIO certification *(status)*, demo repo *(link — KAI gang
  scheduling, Kubeflow TrainJob v2, MIG/time-slicing/MPS, NCCL transports,
  DRA ResourceClaims)*
