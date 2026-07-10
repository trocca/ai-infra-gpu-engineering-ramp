# Readiness Review - corrections to the master plan

[<- Master Plan](MASTER-PLAN.md) · [Companion lessons](companion-lessons/README.md) · [Study path](nvidia-cert-track/STUDY-PATH.md)

## Verdict

The plan is strong for an NVIDIA Developer Evangelist / pre-sales profile because it does three things at once: it earns external proof, builds public artifacts, and turns those artifacts into a demo story. That is the right shape.

The main weakness is dependency risk. The master plan assumes that the math, systems, Rust, PyTorch, and Kubernetes prerequisites will be available on demand. For a high-agency student with a software/product profile, that is dangerous: you will push through ambiguity, but you can lose whole days to prerequisite debt and still feel busy.

The correction is to make the prerequisite layer explicit. Every week now has a companion lesson that must be cleared before Monday, and every weekly plan links to it.

## Weaknesses and Corrections

| Weakness | Why it matters for this profile | Correction |
|----------|----------------------------------|------------|
| Prerequisites are implicit | You can build fast, but missing math support will surface late in LoRA, Triton, distributed training, and serving performance analysis. | Add a [companion lesson](companion-lessons/README.md) for each week and link it from the master plan and weekly plans. |
| Too many novelty layers at once | Several weeks stack new math + new tooling + new hardware assumptions. Week 2 is Rust + CUDA + profiling; week 7 is Triton + quantization + FlashAttention; week 9 is cloud + DDP + NCCL. | Each companion lesson separates "must know before Monday" from "learn during the week." Stretch goals stay optional. |
| Exam ambition is compressed | Three certifications in 12 weeks is possible only if the mocks are treated as gates, not rituals. | Keep the `>= 80%` mock rule and add weekly prerequisite gates. Red weeks roll over instead of being hidden. |
| NCP-AIO assumes real operations experience | The official AIO professional cert expects operational data center experience with NVIDIA hardware, while this repo simulates much of it through cloud labs and docs. | Month 3 companion lessons emphasize command drills, failure modes, and lab simulations. Treat BCM/Slurm/K8s admin vocabulary as practical material, not reading. |
| Math insights are uneven | Some docs already include strong formulas, but the learning path does not consistently identify the one formula that unlocks the week. | Each companion lesson includes a "math insight" section: roofline, KV-cache memory, LoRA rank math, quantization scale/error, all-reduce cost, pipeline bubble, or queueing. |
| GitHub Pages navigation can fragment | Reviewers and future-you need a single route through cert plan, build project, prerequisite support, and Friday proof. | The master plan now links every week to `plan`, `day 1`, `notes`, `prep`, project, and `self-check`. |
| Evidence may become too implementation-heavy | A reviewer does not want to inspect every folder. They need a story: what changed, how it was measured, and what it proves. | Friday output should include one short "week story" in each build README: concept, measurement, failure, and interview talking point. |
| Toolchain failure is under-budgeted | Rust-CUDA, WSL2 CUDA, Triton, and cloud GPU images can steal the week before the learning starts. | Companion lessons include preflight checks and an explicit fallback rule: document the escape hatch, but preserve the concept and benchmark. |

## Operating Corrections

1. Do the companion lesson before each Monday. If the 20-30 minute gate takes more than 45 minutes, Sunday becomes a prerequisite repair block.
2. On Monday, start by copying the companion checklist into that week's `notes.md`.
3. On Friday, log not only score and benchmark, but also the one math insight that unlocked the week.
4. Keep the strict priority rule: exam weeks give PROVE the win; non-exam weeks give BUILD the win.
5. Cut stretch goals first. Never cut acceptance criteria silently.
6. Before booking or sitting an exam, re-check the current official NVIDIA certification page.

## External Exam Reality Check

Official certification pages should be treated as the source of truth before booking:

- [NCA-AIIO - AI Infrastructure and Operations Associate](https://www.nvidia.com/en-us/learn/certification/ai-infrastructure-operations-associate/)
- [NCP-GENL - Generative AI LLMs Professional](https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-professional/)
- [NCP-AIO - AI Operations Professional](https://www.nvidia.com/en-us/learn/certification/ai-operations-professional/)

As of the July 2026 verification pass, the plan's broad shape still matches the public NVIDIA pages: NCA-AIIO is foundational, NCP-GENL expects transformer/PEFT/distributed-training competence, and NCP-AIO is the hardest operational exam because it includes hands-on lab work.

## What "Ready" Means

You are not ready because you read the week. You are ready when you can:

- explain the week's core equation or systems invariant without notes;
- run the lab or simulation without discovery-mode flailing;
- describe one failure mode and its fix;
- connect the week to the NVIDIA stack in customer language;
- publish the result in a way a reviewer can verify in under five minutes.
