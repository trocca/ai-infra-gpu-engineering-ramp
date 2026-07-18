# THE PATH — one page to rule both repos

> Full three-track integration (including the `k8s-ai-stack-demo` touchpoints and the
> interview end-game) lives one level up: `../MASTER-PLAN.md`. This page is the
> day-to-day operating manual for the two daily tracks.

You run two primary tracks in parallel for 12 weeks, ~6 h/day, 5 days/week, plus one optional drill lane and one source-reading lane:

- **Study track** (this repo, ~2 h/day) → three certifications
- **Build track** (`../gpu-engineering-lab`, ~4 h/day) → the public portfolio
- **C++/CUDA mirror drill** (`../cpp-cuda-track`, optional/weekend) → the same parallelism ideas implemented once on CPU and once on GPU
- **Source reading** (`../references`, 20-45 min when assigned) → long-form references mapped into the week they support

They are deliberately aligned: what you study in the morning is what you build in the
afternoon. This page is the only schedule you need to look at.

## Before Monday

Open the week's companion lesson from [`../companion-lessons`](../companion-lessons/README.md).
It is the prerequisite gate for the math, programming-language, and systems concepts that
would otherwise ambush the week. Copy any weak spots into the current week's `notes.md`.
If the gate exercise takes more than 45 minutes, schedule repair time before Day 1.
When the week's core concept feels fuzzy, open the matching C++/CUDA module from
[`../cpp-cuda-track`](../cpp-cuda-track/README.md) and do the smallest runnable lab.
From week 5 onward, read the matching row in the
[`../references/hf-ultrascale-playbook.md`](../references/hf-ultrascale-playbook.md)
map and copy one formula, trade-off, or profiling habit into the week's `notes.md`.

## The day (Mon–Thu)

| Slot | Duration | What | Where |
|------|----------|------|-------|
| 1 | 45 min | read this week's `plan.md` topics | cert repo, current week |
| 2 | 45–60 min | cert lab / self-check questions | cert repo |
| 3 | 15 min | Anki flashcards + fill `notes.md` | cert repo |
| — | break | | |
| 4 | 4 h | build the week's project | lab repo, current week |
| optional | 20–45 min | source reading | `../references`, matching week |
| optional | 30–90 min | C++/CUDA mirror drill | `../cpp-cuda-track`, matching module |

## The Friday (the gate)

1. Cert: `self-check.md` closed-book → tick exit criteria → log row in `PROGRESS.md`.
2. Lab: run `bench/` → put numbers + plots in the project README → update the root
   README index row → **push**. A week isn't done until the results are public.
3. Both green → next week. Cert red → Monday slot 1 re-drills it. Lab red → the
   project gets Saturday morning or its scope gets cut honestly (note what was cut).

## The 12 weeks (start Mon 2026-07-13)

| Wk | Dates | Prep | Study (2 h/day) | Build (4 h/day) | Why they pair |
|----|-------|------|-----------------|-----------------|---------------|
| 1 | Jul 13–17 | [prep](../companion-lessons/week-01.md) | NCA: AI/ML basics, GPU vs CPU, CUDA stack | autograd from scratch + Rust ramp | you study the stack while building backprop by hand |
| 2 | Jul 20–24 | [prep](../companion-lessons/week-02.md) | NCA: GPU hardware, NVLink, precision formats | CUDA-in-Rust kernels, memory hierarchy | morning theory = afternoon's coalescing experiments |
| 3 | Jul 27–31 | [prep](../companion-lessons/week-03.md) | NCA: networking, InfiniBand, DC design | SGEMM ladder vs cuBLAS | roofline thinking feeds both |
| 4 | Aug 3–7 | [prep](../companion-lessons/week-04.md) | NCA review + mock + **EXAM Fri Aug 7** | rusty-kernels → PyTorch (light Friday) | operations vocabulary meets custom ops packaging |
| 5 | Aug 10–14 | [prep](../companion-lessons/week-05.md) | GENL: LLM arch, prompting, data prep | GPT from scratch | the exam's architecture domain IS your build |
| 6 | Aug 17–21 | [prep](../companion-lessons/week-06.md) | GENL: fine-tuning, PEFT, evaluation | LoRA from scratch | study LoRA math, then implement it |
| 7 | Aug 24–28 | [prep](../companion-lessons/week-07.md) | GENL: optimization, quantization, TensorRT-LLM | Triton kernels + INT8 quant | quantization math becomes kernel evidence |
| 8 | Aug 31–Sep 4 | [prep](../companion-lessons/week-08.md) | GENL review + mock + **EXAM Fri Sep 4** | ferrum-serve (Rust/Candle engine, light Friday) | deployment/monitoring domains = your server's metrics |
| 9 | Sep 7–11 | [prep](../companion-lessons/week-09.md) | AIO: BCM, Slurm/K8s setup | distributed training internals (cloud 2×GPU) | NCCL transports appear in both, same week |
| 10 | Sep 14–18 | [prep](../companion-lessons/week-10.md) | AIO: Slurm admin, Run:ai, MIG | tensor + pipeline parallelism (cloud) | admin commands and parallelism both test resource boundaries |
| 11 | Sep 21–25 | [prep](../companion-lessons/week-11.md) | AIO: NGC, workloads, troubleshooting | K8s GPU serving: Operator, DCGM, KAI | the cert's lab drills and your deployment are the same commands |
| 12 | Sep 28–Oct 2 | [prep](../companion-lessons/week-12.md) | AIO lab drills + mock + **EXAM Fri Oct 2** | capstone: train→quantize→serve + repo polish | troubleshooting and portfolio defense converge |

## C++/CUDA mirror modules

| Wk | Module |
|----|--------|
| 1 | [01 execution model](../cpp-cuda-track/01-execution-model/README.md) |
| 2 | [02 memory hierarchy](../cpp-cuda-track/02-memory-hierarchy/README.md) + [03 SAXPY](../cpp-cuda-track/03-data-parallel-saxpy/README.md) |
| 3 | [06 matmul tiling](../cpp-cuda-track/06-matmul-tiling/README.md) + [09 roofline](../cpp-cuda-track/09-profiling-roofline/README.md) |
| 4 | [12 PyTorch extension](../cpp-cuda-track/12-capstone-pytorch-extension/README.md) |
| 5 | [04 reduction](../cpp-cuda-track/04-reduction/README.md) |
| 6 | [05 scan and histogram](../cpp-cuda-track/05-scan-histogram/README.md) |
| 7 | [10 advanced GPU](../cpp-cuda-track/10-advanced-gpu/README.md) |
| 8 | [07 async overlap](../cpp-cuda-track/07-async-overlap/README.md) |
| 9 | [11 multi-device](../cpp-cuda-track/11-multi-device/README.md) |
| 10 | [08 atomics and memory model](../cpp-cuda-track/08-sync-atomics-memory-model/README.md) |
| 11 | [09 profiling roofline](../cpp-cuda-track/09-profiling-roofline/README.md) |
| 12 | [12 PyTorch extension](../cpp-cuda-track/12-capstone-pytorch-extension/README.md) |

## HF Ultra-Scale Playbook modules

| Wk | Source-reading focus |
|----|----------------------|
| 5 | [transformer memory and one-GPU training](../references/hf-ultrascale-playbook.md#week-5---transformer-memory-and-single-gpu-training) |
| 6 | [global batch math, DP, and ZeRO](../references/hf-ultrascale-playbook.md#week-6---fine-tuning-memory-pressure-and-global-batch-math) |
| 7 | [kernels, FlashAttention, and mixed precision](../references/hf-ultrascale-playbook.md#week-7---kernels-flashattention-and-mixed-precision) |
| 8 | [serving memory and precision carryover](../references/hf-ultrascale-playbook.md#week-8---serving-memory-and-precision-carryover) |
| 9 | [data parallelism, ZeRO, collectives, and profiling](../references/hf-ultrascale-playbook.md#week-9---data-parallelism-zero-collectives-and-profiling) |
| 10 | [model parallelism and 5D strategy](../references/hf-ultrascale-playbook.md#week-10---model-parallelism-and-5d-strategy) |
| 11 | [benchmarking, observability, and cluster reality](../references/hf-ultrascale-playbook.md#week-11---benchmarking-observability-and-cluster-reality) |
| 12 | [capstone defense and scale-up story](../references/hf-ultrascale-playbook.md#week-12---capstone-defense-and-scale-up-story) |

## Readiness gates (don't negotiate with yourself)

- **Sit an exam** only if the mock (cert repo, `mock-exam.md`, timed, closed-book)
  scored **≥ 80 %**. Below that: reschedule beats failing — but decide on mock day,
  not exam morning.
- **Ship a week** only if acceptance criteria in the project README pass. Cutting a
  stretch goal is fine; cutting the acceptance criteria means the week rolls over.
- **Priority when life happens**: exam weeks (4, 8, 12) the study track wins; all
  other weeks the build track wins — a public repo with real numbers is worth more
  in the NVIDIA conversation than one extra mock point.

## Readiness, final form (the week-12 checklist)

- [ ] 3 certs passed, badges on LinkedIn
- [ ] 12 project READMEs with real measured numbers, none blank
- [ ] root README of the lab repo reads as a 5-minute tour (a reviewer never opens a folder)
- [ ] the capstone runs end-to-end with one command, on video/GIF
- [ ] you can give the "repo tour" talk from `week-12-capstone/README.md` without notes
- [ ] `REPORT.md` answers: what did you build, what does it prove, what would you do with 8×H100
