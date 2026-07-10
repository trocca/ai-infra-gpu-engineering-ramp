# Pitch & Demo Script — The Kubernetes-AI Stack

**Purpose:** Your "teach us something" talk + a runnable demo storyline. This is your single highest-leverage asset: one artifact that proves technical depth, communication, and customer framing simultaneously — the three things every NVIDIA Developer Advocate req screens for.

Two deliverables in here:
- **Part A — The 20–30 minute talk** (the abstraction pitch). Structured so you can also give the 3-minute and 10-minute versions.
- **Part B — The live demo storyline** (what you actually run, and what to say while it runs).

---

# PART A — The Talk: "What the developer writes vs. what's automatic"

## The one-sentence thesis
> "Training and serving frontier models used to require a team of distributed-systems PhDs. Today a developer writes `train.py`, a `Dockerfile`, and about thirty lines of YAML — and *everything* underneath, from picking NVLink over InfiniBand to isolating a GPU memory domain across a rack, is automatic. That line — between intent and mechanism — is the product. Let me show you where it is and why it's worth billions."

Say that, then draw the stack. The whole talk is walking the line.

## The 3-minute version (elevator / recruiter screen)
"Picture the stack as 16 layers. At the top, three layers are *yours*: your model code, your container image, and a short YAML file that says 'I want 8 nodes, this image, this command, these GPUs.' Everything below that is installed once by the platform team and runs itself. The operator injects the rendezvous variables so PyTorch's launcher just works. The KAI scheduler gang-schedules all your workers at once so you never deadlock, and packs fractional GPUs so the cluster stays busy. Below that, a Kubernetes feature called Dynamic Resource Allocation lets NVIDIA's driver carve a *secure, per-job NVLink memory domain* across a GB200 rack — something the old model literally couldn't describe. And at the very bottom the GPU Operator turns a bare node into a GPU node with one Helm chart. NVIDIA's insight is that the whole stack is a machine for one thing: turning expensive GPUs into useful tokens at the highest possible utilization. That's the 'AI factory,' and NVIDIA sells the whole factory."

## The 10-minute version — the spine
Walk the four systems top-down (customers experience it top-down):

**1. "Here's all you write" (layers 15, 10, 9).**
Put three code blocks on one slide: a 20-line `train.py` (`init_process_group()`, DDP/FSDP, your loop), a 4-line `Dockerfile` (`FROM nvcr.io/nvidia/pytorch:26.04-py3`, `COPY train.py`), and a ~30-line `TrainJob` (`numNodes: 8`, `image`, `command`, GPUs). "This is the entire developer surface. Nothing below this slide is written by the ML engineer."

**2. "The operator makes distributed 'just work'" (layers 8).**
"Distributed PyTorch needs every process to know a coordinator address, its own rank, and the world size. On a laptop you'd hard-code IPs. On a cluster, the workload operator — Kubeflow Trainer for training, LeaderWorkerSet or Grove for inference — creates a headless Service giving every pod stable DNS, and injects `MASTER_ADDR`, `RANK`, `WORLD_SIZE`. `torchrun` reads those and spawns one process per GPU. The developer wrote none of it."

**3. "The scheduler is where GPU dollars are won" (layers 7–5).**
This is the heart of the talk — spend the most time here.
- *Gang scheduling / the deadlock story* (draw it — see the whiteboard script below).
- *Utilization*: fractional GPU sharing + fair-share reclaim + bin-packing. "A cluster at 30% utilization is a cluster you're 3x overpaying for."
- *Topology*: "Not all GPUs are equally close. Topograph publishes labels for the NVLink/leaf/spine/core hierarchy; KAI places your 72 workers on the *same NVLink domain* so your all-reduce runs at 1.8 TB/s instead of crossing racks."
- *DRA + ComputeDomains* (the wow moment): "On a GB200 NVL72, 72 GPUs across 18 nodes act like one giant GPU over Multi-Node NVLink. But you can't let every tenant touch every GPU's memory. Dynamic Resource Allocation lets NVIDIA's driver spin up a *per-workload* IMEX memory domain — a secure isolation boundary over a shared physical fabric — and tear it down when the job ends. The old device-plugin model could only count GPUs as integers; it could never express this."

**4. "And the node just becomes a GPU node" (layers 4–0).**
"One Helm chart — the GPU Operator — installs the driver, the container toolkit, telemetry, and MIG. CDI injects `/dev/nvidia*` into your container. You never SSH into a node."

**Close:** "So the whole point is that line. Every year NVIDIA pushes more capability *below* it — this year it was multi-node NVLink domains as a Kubernetes primitive. The higher that line, the fewer specialists an enterprise needs, the faster they get to first token, and the more of their $40M cluster actually does work. That's why NVIDIA open-sourced the scheduler, ships the operators, and calls the result an AI factory."

## The whiteboard set-pieces (practice drawing these from memory)

**Set-piece 1 — The partial-gang deadlock.** Draw 8 GPU boxes. Job A (needs 8) grabs 6 and waits on its NCCL all-reduce; Job B (needs 4) grabs the remaining 2 and waits. Neither can proceed, neither releases — GPUs "allocated," zero work. Then: "Gang scheduling makes the PodGroup atomic — schedule all 8 or none. Now Job A waits *pending* while Job B, which fits, runs to completion and frees its GPUs. Forward progress guaranteed." This single drawing signals distributed-systems depth better than any buzzword.

**Set-piece 2 — Disaggregated serving.** Two boxes: Prefill (compute-bound, processes the whole prompt, wants *low* tensor-parallelism) and Decode (memory-bandwidth-bound, one token at a time, wants *high* tensor-parallelism). "Old way: one pool does both, and you tune for a compromise that's wrong for both. Dynamo splits them into independently-scaled pools, routes each request to the GPU already holding its KV cache, and gets 30x throughput on DeepSeek-R1 on a GB200 rack."

**Set-piece 3 — The compatibility line.** `libcuda.so` (ships with the driver, admin-owned, bottom of stack) vs `libcudart.so` (ships in the container, dev-owned, top of stack). "A CUDA 13 container needs a ≥580 driver — but forward-compat lets it run on an older datacenter-pinned driver. This is why the container image is the clean contract between the two halves of the stack."

## Framing the business value (translate every technical point to money)
Keep a running translation table in your head:
- Gang scheduling → *no wasted GPU-hours on deadlocked jobs.*
- Fractional sharing + fair-share → *utilization from 30% → 70%+ = you bought 2x fewer GPUs for the same work.*
- Topology-aware placement → *training finishes faster = time-to-model, and you're not paying for idle GPUs waiting on cross-rack all-reduce.*
- The abstraction itself → *fewer specialist hires, faster onboarding, portability on-prem↔cloud.*
- Open-source (KAI, Dynamo, DRA/CDI upstream) → *no lock-in; standards you can adopt without betting the company.*
- AI Enterprise → *a supported, CVE-patched path to production instead of a science project.*

---

# PART B — The Live Demo Storyline

**Design principle:** the demo *is* the pitch made real. Every step should visibly move the line. Have it pre-recorded as a fallback (live GPU clusters fail at the worst moment), but be ready to run it. Full commands are in `02_hands_on_labs.md`; here's the *narrative*.

### Demo arc (≈12–15 min): "From nothing to a served model, touching only the top of the stack"

**Scene 1 — "The node is dumb" (30 sec).** `kubectl get nodes -o wide`, then show a fresh node with no GPU resources advertised. "Right now Kubernetes doesn't even know there are GPUs here."

**Scene 2 — "One Helm chart" (1 min).** Show the GPU Operator already installed (or install it). `kubectl get pods -n gpu-operator` — point at the driver, toolkit, device-plugin, DCGM DaemonSets. Then `kubectl exec` into a test pod and run `nvidia-smi`. "That's the entire bottom of the stack — automatic. I never touched the node."

**Scene 3 — "All I write is this" (2 min).** Show the three files on screen: `train.py`, `Dockerfile`, `trainjob.yaml`. Read the TrainJob out loud: "8 nodes, this image, this command. That's it." Emphasize what's *absent*: no IPs, no ranks, no NCCL config, no device mounts.

**Scene 4 — "Submit and watch the machine work" (3 min).** `kubectl apply -f trainjob.yaml`. Then narrate the automatic layers as they happen:
- `kubectl get pods` — the operator created N pods + a headless Service.
- `kubectl describe pod` — show the injected `MASTER_ADDR`/`RANK`/`WORLD_SIZE` env.
- `kubectl logs` — show `torchrun` rendezvous and NCCL selecting NVLink (`NCCL_DEBUG=INFO`). "There — NCCL just discovered the topology and chose NVLink over the network. I didn't configure that."

**Scene 5 — "The scheduler earns its keep" (3 min).** The money shot. Submit two competing gang jobs to a small cluster. First show them *without* gang scheduling deadlocking (pods Pending/partial). Then enable KAI gang scheduling and show one job running cleanly to completion while the other waits Pending, then runs. "That's the difference between a cluster at 30% and 70% utilization."

**Scene 6 — "Now serve it" (3 min).** `kubectl apply` an LWS (or vLLM) deployment, then `curl` the OpenAI-compatible endpoint and stream a completion. "Same pattern — I described intent, the platform placed a sharded model across nodes and gave me an endpoint." If you have the GPUs and time, show a Dynamo disaggregated deployment and point at prefill vs decode pods.

**Scene 7 — "The line" (1 min).** Return to the stack diagram. "Everything I touched was these three layers. Everything else was automatic — and every year NVIDIA moves that line up. That's the AI factory." End.

### Demo delivery tips
- **Pre-stage everything.** Images pulled, cluster warm, manifests ready. Never pull a multi-GB NGC image live.
- **Talk to the *why*, not the keystrokes.** Nobody remembers the `kubectl` flags; they remember "I never SSH'd into a node."
- **Have the fallback recording** cued to each scene so a cluster hiccup becomes a smooth cut, not a stall.
- **One clear "wow"** (the gang-scheduling deadlock fix or the disaggregated-serving throughput) beats five mild ones.
- **Leave them with the repo.** Public GitHub link on the last slide — the portfolio *is* the ask for this role.

---

# Objection handling (keep these crisp — full versions in the Q&A file)
1. **"Our GPUs sit at 25%."** → KAI fair-share queues + fractional GPU + bin-packing + preemption; separate training/inference pools; MIG for guaranteed inference QoS.
2. **"How do we share GPUs safely?"** → Three mechanisms, know the trade-off: **MIG** = hardware partition, hard memory+fault isolation (multi-tenant QoS); **time-slicing** = software round-robin, *no* isolation (bursty dev/inference); **MPS** = concurrent contexts, weak isolation. Plus KAI fractional + quotas on top.
3. **"What's the ROI?"** → AI-factory math: tokens/sec/$ and utilization; software subscription vs hardware TCO; DGX Cloud for burst.
4. **"1000-GPU jobs fail — what happens?"** → gang scheduling avoids partial placement; framework checkpoint/restart (NeMo); auto-requeue on preemption; DCGM health-evicts bad nodes; torchrun elasticity tolerates membership changes.
5. **"Observability?"** → DCGM + DCGM-Exporter → Prometheus/Grafana; per-workload GPU metrics; Run:ai dashboards; NVLink/fabric telemetry.
6. **"Vendor lock-in?"** → KAI (Apache 2.0), Dynamo (open), DRA/CDI/LWS are upstream Kubernetes/CNCF standards, NeMo/Nemotron open models. The platform sits on open standards; AI Enterprise is the *supported* option, not the *only* one.
7. **"Networking bottleneck at scale?"** → Spectrum-X / Quantum InfiniBand + GPUDirect RDMA + rail-optimized topology; SHARP in-network reduction; scheduler topology-awareness to minimize cross-rack traffic.

---

# The culture layer (weave into how you present, not just what)
NVIDIA hires for the intersection of depth and storytelling, and rewards a specific style: *the mission is the boss* (frame everything around the customer's outcome, not features), *default-to-broadcast / intellectual honesty* (say plainly what the product can't do — it builds more trust than overselling), and *"speed of light" thinking* (benchmark against the physical limit, not the competitor). If asked "what would you *not* claim about this stack?" — answer honestly (e.g. "fractional sharing has no memory isolation by default; for hard multi-tenancy you want MIG or HAMi"). That honesty is itself the signal they're looking for.
