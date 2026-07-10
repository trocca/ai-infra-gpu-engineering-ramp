# THE PATH — one page to rule both repos

> Full three-track integration (including the `k8s-ai-stack-demo` touchpoints and the
> interview end-game) lives one level up: `../MASTER-PLAN.md`. This page is the
> day-to-day operating manual for the two daily tracks.

You run two tracks in parallel for 12 weeks, ~6 h/day, 5 days/week:

- **Study track** (this repo, ~2 h/day) → three certifications
- **Build track** (`../gpu-engineering-lab`, ~4 h/day) → the public portfolio

They are deliberately aligned: what you study in the morning is what you build in the
afternoon. This page is the only schedule you need to look at.

## Before Monday

Open the week's companion lesson from [`../companion-lessons`](../companion-lessons/README.md).
It is the prerequisite gate for the math, programming-language, and systems concepts that
would otherwise ambush the week. Copy any weak spots into the current week's `notes.md`.
If the gate exercise takes more than 45 minutes, schedule repair time before Day 1.

## The day (Mon–Thu)

| Slot | Duration | What | Where |
|------|----------|------|-------|
| 1 | 45 min | read this week's `plan.md` topics | cert repo, current week |
| 2 | 45–60 min | cert lab / self-check questions | cert repo |
| 3 | 15 min | Anki flashcards + fill `notes.md` | cert repo |
| — | break | | |
| 4 | 4 h | build the week's project | lab repo, current week |

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
