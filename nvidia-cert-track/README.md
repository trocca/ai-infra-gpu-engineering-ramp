# NVIDIA Certification Track — 3 Months, 3 Certs

Study repository for a 12-week run at three NVIDIA certifications, built around ~2 h/day
(5 study days/week) and aimed at an NVIDIA Developer Evangelist / pre-sales profile on the
Kubernetes-AI stack (GPU Operator, KAI Scheduler, DRA, Kubeflow Trainer, NCCL, DDP/FSDP,
vLLM/Dynamo/NIM).

> Exam details below were verified in **July 2026** against
> https://www.nvidia.com/en-us/learn/certification/ — NVIDIA updates exams; re-check each
> official page before booking.

## The timeline

| Month | Weeks | Cert | Format | Price | Exam date |
|-------|-------|------|--------|-------|-----------|
| 1 | 1–4 | **NCA-AIIO** — Associate: AI Infrastructure & Operations | 50 MCQ / 60 min | $125 | end of week 4 |
| 2 | 5–8 | **NCP-GENL** — Professional: Generative AI LLMs | 60–70 MCQ / 120 min | $200 | end of week 8 |
| 3 | 9–12 | **NCP-AIO** — Professional: AI Operations | 30 MCQ **+ 3 hands-on labs** / 120 min | $500 | end of week 12 |

All three are valid 2 years from issuance.

Difficulty ramps deliberately: NCA-AIIO is foundational vocabulary and architecture;
NCP-GENL leans on ML/LLM depth you partly already have; NCP-AIO is the hardest and most
hands-on — its lab component is why month 3 has five labs and a dedicated drill week.

## Repo layout

```
month-1-nca-aiio/     syllabus, week-1..4 (plan / notes / self-check), flashcards.csv, mock-exam.md
month-2-ncp-genl/     syllabus, week-5..8, labs/ (LoRA, quantize+serve, DDP), flashcards.csv, mock-exam.md
month-3-ncp-aio/      syllabus, week-9..12, labs/ (GPU Operator, MIG, Slurm, KAI/Run:ai, troubleshooting),
                      flashcards.csv, mock-exam.md
tools/                daily.md (the daily loop), booking-checklist.md
PROGRESS.md           one row per week: topics, labs, mock score, confidence
../companion-lessons/ prerequisite lessons for each week: math, Rust/Python/PyTorch, CUDA, Triton, distributed, K8s
```

> **Running the build track too?** The single unified schedule for both this repo and
> `../gpu-engineering-lab` (2 h study + 4 h build per day, week-by-week pairing, exam
> gates) is **[STUDY-PATH.md](STUDY-PATH.md)** — start there each Monday.

## How to use this repo daily

1. **Before Monday**: clear the week's companion lesson in
   [`../companion-lessons`](../companion-lessons/README.md), then copy weak spots into
   `notes.md`.
2. **Monday morning**: open the current week's `plan.md`. It maps every day to an exam
   domain and ends with Friday exit criteria.
3. **Each study day** (~2 h): follow `tools/daily.md` — 45 min reading, 45–60 min
   lab/practice, 15 min flashcards + fill in `notes.md`.
4. **Flashcards**: import each month's `flashcards.csv` into Anki at the *start* of the
   month (File → Import, comma-separated, first field = front). Review daily; Anki's
   scheduler does the spacing for you.
5. **Friday**: run the week's `self-check.md` closed-book, then honestly tick (or don't)
   the exit criteria in `plan.md`. Log the week in `PROGRESS.md` with a 1–5 confidence.
6. **Last week of each month**: take `mock-exam.md` under real conditions (timed,
   closed-book, no pausing). Target ≥ 80 % before sitting the real exam; below that,
   spend the remaining days on the domains you missed, not on re-reading everything.
7. **Booking**: follow `tools/booking-checklist.md` at the start of each month so the
   exam date is fixed and non-negotiable.

## Rules of the road

- `notes.md` files are intentionally empty skeletons — writing your own summaries is the
  point; don't skip it.
- Labs state their estimated cloud cost. Total lab budget across all 12 weeks is roughly
  $40–80 on spot/on-demand L4/A10G instances (plus one short A100 rental for the MIG lab).
- Month 2 and 3 materials cross-reference your demo repo (KAI gang scheduling, TrainJob v2,
  MIG/time-slicing/MPS, NCCL transports, DRA) — when a topic overlaps, re-run your own demo
  as the lab; it doubles as evangelist prep.
