# Week 12 Self-Check — Troubleshooting deep-dive

Closed book. Target ≥ 16/18 (it's exam week).

**1. Training throughput is poor; `nvidia-smi` shows GPU util oscillating 0–100% in bursts. Most likely bottleneck and first diagnostic?**

<details><summary>Answer</summary>
Input pipeline/storage-bound: GPUs starve between batches. First diagnostic: check dataloader wait vs step time (profiler) and benchmark the data path with fio from the same mount; also check CPU util of dataloader workers. Fixes: stage to local NVMe, more workers/prefetch, larger sequential reads, sharded formats.
</details>

**2. Give a fio command to measure large sequential read throughput without page-cache pollution, and say why `--direct=1` matters.**

<details><summary>Answer</summary>
<code>fio --name=seqread --rw=read --bs=1M --size=10G --numjobs=4 --iodepth=32 --ioengine=libaio --direct=1 --group_reporting</code>. <code>--direct=1</code> uses O_DIRECT to bypass the page cache — otherwise a warm cache makes the storage look far faster than it is.
</details>

**3. Why is NFS often a poor fit for many-node training data, and what are the two standard alternatives?**

<details><summary>Answer</summary>
Single-server NFS serializes metadata and bandwidth through one endpoint — random small reads from hundreds of workers collapse it. Alternatives: parallel filesystems (Lustre, GPFS/Spectrum Scale) that stripe data and metadata across servers, and staging datasets to node-local NVMe scratch (often via sharded formats or a caching layer).
</details>

**4. In `NCCL_DEBUG=INFO` output, what tells you which transport was selected, and what's the fallback ladder?**

<details><summary>Answer</summary>
Lines like "Channel 00 : ... via P2P/CUMEM", "via SHM", "NET/IB", "NET/Socket" (and "Using network IB/Socket"). Ladder intra-node: NVLink/P2P → shared memory (SHM) → network loopback. Inter-node: NET/IB (RDMA) → NET/Socket (TCP). Seeing NET/Socket on an IB cluster = misconfiguration (e.g. NCCL_IB_DISABLE=1 or missing RDMA devices in the container).
</details>

**5. What does setting `NCCL_P2P_DISABLE=1` do, and how would you notice someone set it by accident?**

<details><summary>Answer</summary>
Disables direct GPU-to-GPU (NVLink/PCIe P2P) transport; intra-node traffic falls back to SHM through host memory. Symptom: all_reduce bus bandwidth drops massively (e.g. NVLink-class ~200+ GB/s down to tens), NCCL_DEBUG shows "via SHM" instead of "via P2P" between local GPUs.
</details>

**6. A 2-node NCCL job hangs at init. Name three classic causes and the env var you'd set first.**

<details><summary>Answer</summary>
Set <code>NCCL_DEBUG=INFO</code> first (plus WARN-level at minimum in prod). Classic causes: (1) wrong/asymmetric <code>NCCL_SOCKET_IFNAME</code> — nodes advertise unreachable interfaces (e.g. docker0); (2) firewall blocking the dynamic TCP ports / IB not reachable; (3) mismatched env or NCCL versions across ranks, or wrong MASTER_ADDR so ranks never rendezvous.
</details>

**7. The four dcgmi diag run levels — rough duration and what each is for?**

<details><summary>Answer</summary>
-r 1: seconds — software/config sanity (driver, permissions, basic checks). -r 2: ~2–3 min — adds PCIe/NVLink and brief GPU stress. -r 3: ~30+ min — adds full memory test, extended stress/targeted power & thermals; the "before RMA / burn-in" level. -r 4: longest, extra-extended stress. Command: <code>dcgmi diag -r 3</code>.
</details>

**8. Xid 31 vs Xid 48 — meaning, culprit, action.**

<details><summary>Answer</summary>
Xid 31: GPU memory page fault — almost always an application bug (bad pointer/OOB access); action: fix the app, GPU is fine. Xid 48: double-bit ECC error — hardware memory fault; action: drain the node, run dcgmi diag -r 3, check retired/remapped pages, likely RMA path if recurring.
</details>

**9. Xid 79 appears in dmesg and the GPU vanished from nvidia-smi. What is it and what do you check?**

<details><summary>Answer</summary>
"GPU has fallen off the bus" — the PCIe device dropped: power delivery, PCIe seating/riser, overheating, or dying hardware. Check dmesg for preceding thermal/power events, BMC/IPMI sensor logs, reseat/swap slot, and treat as hardware until proven otherwise. Reboot may temporarily recover it.
</details>

**10. Xid 63/64 relate to what mechanism, and what's the operational difference between them?**

<details><summary>Answer</summary>
Row remapping — the on-die replacement of failing memory rows (A100+, successor to page retirement). Xid 63: a row was remapped and it's pending — reset/reboot the GPU to activate it, then fine. Xid 64: remapping failed / no spare rows — memory can't self-heal; drain and RMA.
</details>

**11. Which Xids point at the app rather than hardware, and why does the distinction matter operationally?**

<details><summary>Answer</summary>
13 (graphics engine exception), 31 (page fault), 43 (app-triggered abort) are typically application-caused. It matters because the wrong call either RMAs healthy GPUs (cost, capacity) or keeps rescheduling jobs onto a genuinely bad GPU (serial job failures). Triage: same app fails everywhere = app; many apps fail on one GPU = hardware.
</details>

**12. `kubectl describe pod` shows `0/1 nodes are available: 1 Insufficient nvidia.com/gpu`, but the node has a free GPU. Three distinct causes?**

<details><summary>Answer</summary>
(1) Device plugin not running/crashed, so node capacity shows 0 (check <code>kubectl describe node | grep nvidia.com/gpu</code>). (2) The GPU is allocated to another pod even if idle — K8s counts requests, not utilization. (3) MIG config mismatch: node exposes <code>nvidia.com/mig-3g.20gb</code> (mixed strategy) while the pod requests <code>nvidia.com/gpu</code>, or time-slicing replicas not applied. (Also: pod missing toleration for a GPU-node taint shows a different message.)
</details>

**13. After a driver package upgrade, host `nvidia-smi` works, but all GPU pods CrashLoop and DCGM exporter is down. GPU Operator angle — what happened, what's the fix?**

<details><summary>Answer</summary>
The operator's driver validation/toolkit stack is now inconsistent with the host driver (or the driver DaemonSet conflicts with the newly host-installed driver). Fix: pick one owner for the driver — if host-managed, set <code>driver.enabled=false</code> and restart the operator pods (validator → toolkit → device-plugin → DCGM come back in order); if operator-managed, remove the host driver and let the driver DaemonSet reconcile. Check <code>kubectl get pods -n gpu-operator</code> validator logs to confirm the failing layer.
</details>

**14. Your `dcgmi diag -r 2` fails on the PCIe test on one node with low H2D bandwidth. Name two hardware-level causes to verify.**

<details><summary>Answer</summary>
PCIe link degradation: card negotiated fewer lanes or a lower generation (x16 → x4, Gen4 → Gen1) — verify with <code>nvidia-smi -q | grep -A4 "GPU Link"</code> or lspci -vv; and IRQ/NUMA misplacement or a bad riser/slot. Reseat, check BIOS PCIe settings, compare against a known-good node.
</details>

**15. What's your exam-day time budget, and the rule for a stuck MCQ / a stuck lab step?**

<details><summary>Answer</summary>
~45 min for ~30 MCQs (90 s each), ~75 min for 3 labs (25 min each), 5 min buffer. Stuck MCQ: flag, pick best guess, move on — never 5 minutes on one question. Stuck lab step: one targeted diagnostic (describe/journalctl/logs), one fix attempt; if still stuck, complete the parts you can — partial working state beats a perfect plan.
</details>

**16. List the verification command you'd run after EACH of these lab tasks: (a) enabling MIG profiles, (b) installing GPU Operator, (c) adding a Slurm QOS, (d) fixing a container runtime.**

<details><summary>Answer</summary>
(a) <code>nvidia-smi mig -lgi</code> (and <code>nvidia-smi -L</code> to see MIG devices). (b) <code>kubectl get pods -n gpu-operator</code> all Running/Completed + a CUDA test pod, or <code>kubectl describe node | grep nvidia.com/gpu</code>. (c) <code>sacctmgr show qos format=name,maxwall,maxtres</code>. (d) <code>docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi</code> (or the K8s pod equivalent). Habit: never claim done without the verify step — the lab grader checks state, not intentions.
</details>

**17. GPU Operator's DCGM exporter shows a rising `DCGM_FI_DEV_XID_ERRORS` on one node. Walk your response.**

<details><summary>Answer</summary>
(1) Identify the Xid: <code>dmesg -T | grep -i xid</code> on the node (which code, which GPU). (2) Classify app vs hardware (Q11). (3) If hardware-suspect: cordon/drain the node (kubectl drain / scontrol drain), run <code>dcgmi diag -r 3</code>. (4) Pass → return to service and monitor; fail or ECC/remap exhaustion → open RMA, keep node drained. Document in the incident log.
</details>

**18. Why practice labs in a plain terminal before this exam?**

<details><summary>Answer</summary>
The hands-on portion runs in a browser-based terminal with none of your local comforts — no aliases, dotfiles, fzf, shell history, or custom editors. If your muscle memory depends on <code>k</code> for kubectl or your vim config, you'll bleed minutes. Train the way you'll fight: bare bash, default editors, typed-out commands.
</details>
