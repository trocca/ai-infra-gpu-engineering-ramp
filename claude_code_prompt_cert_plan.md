# Prompt for Claude Code — NVIDIA Certification Study Repo (copy everything below the line)

---

Create a complete study repository at `./nvidia-cert-track/` for my 3-month NVIDIA certification plan. I am preparing for an NVIDIA Developer Evangelist / pre-sales role focused on the Kubernetes-AI stack (GPU Operator, KAI Scheduler, DRA, Kubeflow Trainer, LWS/Grove, NCCL, DDP/FSDP/Megatron, vLLM/Dynamo/NIM), and I'm adding certifications along the way. I already study ~1.5–2.5 hrs/day.

## The 3 certifications and their timing

**Month 1 (weeks 1–4) → NCA-AIIO — NVIDIA-Certified Associate: AI Infrastructure and Operations.**
$125, 1-hour exam, ~50 multiple-choice questions, valid 2 years. Foundational: AI compute concepts, GPU hardware, datacenter/cluster basics, deployment and operations fundamentals. Exam booked for the end of week 4.

**Month 2 (weeks 5–8) → NCP-GENL — NVIDIA-Certified Professional: Generative AI LLMs.**
$200, 120 minutes, 60–70 proctored questions, valid 2 years. Domains and weights: Model Optimization 17%, GPU Acceleration & Optimization 14%, Prompt Engineering 13%, Fine-Tuning 13%, Data Preparation 9%, Model Deployment 9%, Production Monitoring & Reliability 7%, Evaluation 7%, LLM Architecture 6%, Safety/Ethics/Compliance 5%. Assumes competency in transformer architectures, distributed parallelism (DDP/FSDP/Megatron-style), and PEFT/LoRA. Exam booked for the end of week 8.

**Month 3 (weeks 9–12) → NCP-AIO — NVIDIA-Certified Professional: AI Operations.**
$500, 120 minutes, 30 multiple-choice questions PLUS 3 hands-on lab exercises, valid 2 years. Domains and weights: Installation & Deployment 31% (Base Command Manager, Slurm and Kubernetes cluster setup, patching, users, networking), Administration 23% (Slurm, datacenter architecture, NVIDIA Run:ai, Kubernetes, MIG configuration), Workload Management 23% (training/inference deployment, resource allocation, NGC containers), Troubleshooting & Optimization 23% (Docker, fabric manager, BCM, storage, container issues). Exam booked for the end of week 12.

IMPORTANT: these exam details were verified in July 2026 from nvidia.com/en-us/learn/certification/ but NVIDIA updates exams — if you have web access, re-verify each exam's current domains/pricing from the official pages and correct the files; if not, add a "⚠ verify against the official exam page" note at the top of each cert's syllabus file.

## Folder tree to create

```
nvidia-cert-track/
├── README.md                     # overview, the 3-cert timeline, how to use this repo daily
├── PROGRESS.md                   # master tracker: one row per week — topics done, labs done, mock scores, confidence 1-5
├── month-1-nca-aiio/
│   ├── syllabus.md               # exam blurb, domains, official links, booking checklist
│   ├── week-1/ … week-4/         # one folder per week, each containing:
│   │   ├── plan.md               #   topics for the week mapped to exam domains, day-by-day (5 days ≈ 2h each)
│   │   ├── notes.md              #   pre-structured headings per topic, empty bullets for me to fill
│   │   └── self-check.md         #   15–20 self-test questions WITH answers hidden in <details> blocks
│   ├── flashcards.csv            # front,back — ≥80 cards covering the whole exam, Anki-importable
│   └── mock-exam.md              # 50 realistic multiple-choice questions + answer key at the bottom
├── month-2-ncp-genl/
│   ├── syllabus.md
│   ├── week-5/ … week-8/         # same weekly structure; weight the weeks by exam domain %:
│   │   #  wk5 = LLM architecture + prompt engineering + data prep
│   │   #  wk6 = fine-tuning + PEFT/LoRA + evaluation
│   │   #  wk7 = model optimization + GPU acceleration (quantization, TensorRT-LLM, batching)
│   │   #  wk8 = deployment + monitoring + safety, then review + mock
│   ├── labs/
│   │   ├── lab-finetune-lora.md  # step-by-step: LoRA fine-tune a small model (single GPU, PEFT lib)
│   │   ├── lab-quantize-serve.md # quantize + serve with vLLM, measure latency/throughput before/after
│   │   └── lab-distributed-ddp.md# torchrun DDP on 2 GPUs, read the injected env vars + NCCL logs
│   ├── flashcards.csv            # ≥100 cards
│   └── mock-exam.md              # 60 questions weighted by the official domain percentages + answer key
├── month-3-ncp-aio/
│   ├── syllabus.md
│   ├── week-9/ … week-12/        # weight by domain %:
│   │   #  wk9  = installation & deployment (BCM, Slurm setup, K8s setup) — 31% of exam
│   │   #  wk10 = administration (Slurm, Run:ai, K8s, MIG)
│   │   #  wk11 = workload management (NGC, training/inference deployment) + troubleshooting
│   │   #  wk12 = hands-on lab drill week + review + mock
│   ├── labs/
│   │   ├── lab-gpu-operator.md   # install GPU Operator on a cloud K8s cluster, verify CDI + DCGM
│   │   ├── lab-mig-config.md     # configure MIG profiles via MIG Manager, schedule pods onto instances
│   │   ├── lab-slurm-basics.md   # minimal Slurm cluster (2 nodes or containers), sbatch a GPU job
│   │   ├── lab-runai-kai.md      # install KAI scheduler, gang-schedule two competing jobs, fractional GPU
│   │   └── lab-troubleshoot.md   # break-and-fix drills: bad driver, wrong runtime class, NCCL env issues
│   ├── flashcards.csv            # ≥100 cards
│   └── mock-exam.md              # 30 questions + 3 written lab-scenario exercises + answer key
└── tools/
    ├── daily.md                  # the daily loop: 45m read → 45–60m lab/practice → 15m flashcards+notes
    └── booking-checklist.md      # how/where to book each exam (NVIDIA cert portal), ID/proctoring reqs, costs
```

## Content requirements

- Write REAL content, not placeholders, for: every `plan.md`, `syllabus.md`, `self-check.md`, `flashcards.csv`, `mock-exam.md`, and every lab. Only `notes.md` files are intentionally skeletal (headings + empty bullets for my own notes).
- Every weekly `plan.md` must map each day's topics to the official exam domain it serves, and end with a Friday "exit criteria" list (what I must be able to explain/do before moving on).
- Mock exam questions must be plausible exam-style (scenario-based, one correct + three tempting distractors), not trivia. Put all answer keys at the BOTTOM of the file with one-line explanations.
- Labs must be runnable on rented commodity cloud GPUs (1–2 × L4 or similar; no GB200 assumptions). State estimated cost/time per lab. Where a real cluster is unavoidable (Slurm, BCM), offer a minimal containerized or single-node fallback.
- Flashcards CSV format: `front,back`, properly quoted, importable into Anki without editing.
- In month 2 and 3, cross-reference my existing demo repo concepts where relevant (KAI gang scheduling, TrainJob v2, MIG vs time-slicing vs MPS, NCCL transports, DRA ResourceClaims) — these overlap heavily with the NCP-AIO administration/workload domains.
- Keep every file self-contained and readable in a terminal (no HTML, markdown only, tables where useful).
- After creating everything, print the full tree and a 10-line "start here" summary.

Work through this systematically: create the tree first, then fill month 1 completely, then month 2, then month 3, then tools/. Commit after each month's folder is complete with message "month N: <cert> study materials".
