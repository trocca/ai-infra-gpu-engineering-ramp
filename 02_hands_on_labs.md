# Hands-On Lab Guide — Running the Kubernetes-AI Stack on GPUs

**Purpose:** Build muscle memory for every layer so the demo is second nature and you can answer "have you actually done this?" with "yes, here's the repo." Labs go from a single GPU to multi-node distributed training to disaggregated serving.

> **Reality check on hardware.** Some labs need real NVIDIA GPUs; a few need *multiple* GPU nodes. You do not need a GB200 rack — the *concepts* (ComputeDomains, MNNVL) can be studied from manifests, and everything else runs on commodity cloud GPUs. Recommended: rent GPUs by the hour. Options below.

---

## Lab 0 — Environment options (pick your budget)

**A. Single-GPU, cheapest (most labs 1, 2, 6):**
- One cloud GPU VM (e.g. an L4/L40S/A10 or a single H100) from any provider, or a managed GPU node pool of size 1.
- Install a single-node Kubernetes: `k3s` or `minikube --driver=none` on the GPU host, or `kind` with the NVIDIA runtime.

**B. Multi-GPU single node (labs 3, 4, 6, 7 in miniature):**
- One 2–8 GPU node (e.g. 2×L4 or 8×H100). You can simulate "multi-node" distributed training across GPUs on one node, and do real gang-scheduling demos by requesting more GPUs than exist.

**C. Multi-node cloud cluster (full labs 3–7):**
- A managed Kubernetes cluster (EKS/GKE/AKS/OKE) with a GPU node pool of 2–4 nodes. This is what you want for the honest multi-node story. Budget-control: scale the pool to 0 when not in use.

**Cost tips:** use spot/preemptible nodes for labs; scale node pools to zero between sessions; a single L4 is enough for Labs 1, 2, 6 and costs little. Prove multi-node once on a rented cluster, record it, then tear it down.

**Baseline tooling to install once:** `kubectl`, `helm`, `git`, and the cloud CLI for your provider. Verify: `kubectl version`, `helm version`.

---

## Lab 1 — Turn a bare node into a GPU node (GPU Operator, CDI)
**Layers 2, 4.** *Goal: prove the bottom of the stack is one Helm chart, and see how `/dev/nvidia*` gets into a container.*

```bash
# 1. Add the NVIDIA Helm repo and install the GPU Operator
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia && helm repo update
kubectl create ns gpu-operator
helm install --wait gpu-operator nvidia/gpu-operator \
  -n gpu-operator --set driver.enabled=true

# 2. Watch the operator stand up the node stack (driver, toolkit, device-plugin, DCGM, NFD/GFD)
kubectl get pods -n gpu-operator
kubectl get nodes -o json | jq '.items[].status.capacity' | grep nvidia.com/gpu

# 3. Confirm node labels from GPU Feature Discovery
kubectl get nodes -o json | jq '.items[].metadata.labels' | grep nvidia.com

# 4. Run a GPU workload and see the GPU inside the container
kubectl run cuda-smi --rm -it --restart=Never \
  --image=nvcr.io/nvidia/cuda:13.3.0-base-ubuntu24.04 \
  --limits=nvidia.com/gpu=1 -- nvidia-smi
```

**Observe & explain:**
- The device plugin advertised `nvidia.com/gpu` as node capacity — the *legacy counting model*.
- On the node, inspect the CDI spec that describes the device injection: `sudo nvidia-ctk cdi list` (lists `nvidia.com/gpu=…` devices); the spec lives at `/var/run/cdi/nvidia.yaml`. This is what containerd reads to inject `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm` and the driver libs — no `--device` flags, no `nvidia` runtime class needed.

**Interview tie-in:** "Everything I just did is layers 2 and 4 — and it was one Helm install plus a `limits: nvidia.com/gpu: 1`. That's the whole 'automatic' bottom of the stack."

---

## Lab 2 — NGC container + read the NCCL topology
**Layers 10–12.** *Goal: understand why you use the NGC image, and watch NCCL detect the interconnect.*

```bash
# 1. Pull and inspect the NGC PyTorch container (calendar-versioned)
kubectl run torch --rm -it --restart=Never \
  --image=nvcr.io/nvidia/pytorch:26.04-py3 \
  --limits=nvidia.com/gpu=1 -- python -c \
  "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"
```
Note the PyTorch version string ends in `a0` (NVIDIA build) and CUDA is baked in — that's the pre-validated, ABI-matched stack you'd have to assemble yourself with `pip`.

```bash
# 2. On a 2-GPU node, run a tiny NCCL all-reduce with debug logging
# (save as nccl-test.yaml, apply it — key env: NCCL_DEBUG=INFO)
```
```yaml
apiVersion: v1
kind: Pod
metadata: { name: nccl-topo }
spec:
  restartPolicy: Never
  containers:
  - name: t
    image: nvcr.io/nvidia/pytorch:26.04-py3
    command: ["torchrun","--nproc-per-node=2","-m","torch.distributed.run"]
    env: [{ name: NCCL_DEBUG, value: INFO }]
    resources: { limits: { nvidia.com/gpu: 2 } }
```
Simpler: `kubectl exec` in and run a 2-process script that does `dist.init_process_group("nccl")` + `dist.all_reduce(t)`.

**Observe & explain:** in the logs, NCCL prints the channels/rings it built and which transport it chose (`NVLink`, `PCI`, or `NET`/IB). Point at it: "NCCL auto-detected the topology and chose NVLink between these two GPUs. On a multi-node run this line would show `IB` with GPUDirect RDMA." That's layer 12 doing its job with zero config.

---

## Lab 3 — Multi-node distributed training with Kubeflow Trainer (TrainJob v2)
**Layers 8, 9, 13, 14.** *Goal: the operator injects rendezvous env + headless Service so `torchrun` "just works" across nodes.*

```bash
# 1. Install Kubeflow Trainer (needs JobSet as a dependency)
kubectl apply --server-side -k "https://github.com/kubeflow/trainer.git/manifests/overlays/manager?ref=v2.0.0"
# (also install a ClusterTrainingRuntime, e.g. the torch-distributed one shipped with the release)
kubectl get clustertrainingruntime
```

```yaml
# 2. trainjob.yaml — this is the ENTIRE developer surface
apiVersion: trainer.kubeflow.org/v1alpha1
kind: TrainJob
metadata: { name: torch-ddp-demo }
spec:
  runtimeRef:
    name: torch-distributed
    apiGroup: trainer.kubeflow.org
    kind: ClusterTrainingRuntime
  trainer:
    numNodes: 2
    image: nvcr.io/nvidia/pytorch:26.04-py3
    command: ["torchrun","/workspace/train.py"]
    resourcesPerNode:
      limits: { nvidia.com/gpu: 1 }
```

```bash
kubectl apply -f trainjob.yaml
kubectl get pods -l trainer.kubeflow.org/trainjob-name=torch-ddp-demo
# 3. Prove what was injected automatically:
kubectl describe pod <pod> | grep -E "MASTER_ADDR|RANK|WORLD_SIZE"
kubectl get svc            # the headless Service the operator created
kubectl logs <rank0-pod> | grep -Ei "rendezvous|nccl|world_size"
```

**Observe & explain:** you wrote `numNodes`, `image`, `command`. The operator rendered a JobSet, created a headless Service, gave each pod stable DNS, set `MASTER_ADDR` = rank-0's DNS name, and assigned each pod its `RANK`. `torchrun` read those and spawned one process per GPU. **This is the "what's automatic" story made concrete — screenshot it for your talk.**

> If Kubeflow Trainer install is fiddly on your cluster, the same lesson works with a raw PyTorchJob (v1) or a hand-written JobSet + headless Service. The *point* is seeing the injected env vars.

---

## Lab 4 — Gang scheduling with KAI (the deadlock demo)
**Layer 7.** *Goal: reproduce the partial-gang deadlock, then watch KAI fix it. This is your demo's "wow."*

```bash
# 1. Install KAI Scheduler
helm repo add nvidia-k8s https://nvidia.github.io/KAI-Scheduler
helm repo update
helm install kai-scheduler nvidia-k8s/kai-scheduler -n kai-scheduler --create-namespace
# (verify version against github.com/NVIDIA/KAI-Scheduler/releases before quoting)
kubectl get pods -n kai-scheduler
```

**The demo:** On a cluster with, say, 8 total GPUs, submit **two jobs that each need 8 GPUs** as gangs.
- *Without gang scheduling* (default scheduler, or KAI disabled): each job's pods grab whatever GPUs are free and sit **Pending/partial** — classic deadlock, GPUs "allocated," zero progress.
- *With KAI gang scheduling*: create a PodGroup per job (`minMember` = full gang). KAI schedules Job A **all-or-nothing**; Job B stays cleanly **Pending** until A finishes and frees its GPUs, then B runs. No deadlock, no wasted GPU-hours.

```yaml
# Minimal PodGroup (KAI) — the atomic unit
apiVersion: scheduling.run.ai/v2alpha2
kind: PodGroup
metadata: { name: job-a }
spec:
  minMember: 8
  queue: default
```
Pods opt in with `spec.schedulerName: kai-scheduler` and a label tying them to the PodGroup. (Check the current KAI docs for the exact PodGroup API group/version and the podgrouper auto-integration — KAI can auto-create PodGroups for Kubeflow/Ray jobs.)

**Also try:** fractional GPU sharing — annotate two pods with `gpu-fraction: "0.5"` and watch both land on one physical GPU. Note aloud: "no memory isolation by default — for hard multi-tenancy you'd add MIG or HAMi." (That honest caveat is an interview asset.)

**Observe & explain:** narrate the utilization story. "Without gang scheduling, distributed jobs deadlock and your $40M cluster idles. With it, you get guaranteed forward progress and you can safely pack the cluster."

---

## Lab 5 — DRA & ComputeDomains (concept lab)
**Layer 5.** *Goal: understand the modern resource model and the GB200 multi-node NVLink capability — mostly by reading manifests, since you likely don't have a GB200 rack.*

```bash
# 1. If your cluster is v1.34+, inspect the DRA API surface
kubectl api-resources | grep -i resource.k8s.io
kubectl explain resourceclaim
kubectl explain resourceclaimtemplate
kubectl explain deviceclass
kubectl explain resourceslice
```

```yaml
# 2. Read (and be able to explain every field of) a ResourceClaimTemplate
apiVersion: resource.k8s.io/v1
kind: ResourceClaimTemplate
metadata: { name: single-gpu }
spec:
  spec:
    devices:
      requests:
      - name: gpu
        deviceClassName: gpu.nvidia.com   # from the NVIDIA DRA driver
```
A pod references it via `resourceClaims` → `resourceClaimTemplateName: single-gpu`.

**ComputeDomain (GB200) — read-and-explain (install the NVIDIA DRA driver if you have compatible GPUs, else study the manifest):**
```yaml
apiVersion: resource.nvidia.com/v1beta1
kind: ComputeDomain
metadata: { name: my-cd }
spec:
  numNodes: 4
  channel:
    resourceClaimTemplate:
      name: my-cd-rct
```
Pods that share this ComputeDomain (via the generated ResourceClaimTemplate) get their GPUs wired into **one IMEX memory domain** across nodes, isolated from other tenants, and are co-placed on the same NVLink clique via the `nvidia.com/gpu.clique` label. *(Confirm the exact apiVersion/kind against the current nvidia-dra-driver-gpu docs before quoting — versions move.)*

**Explain in one breath:** "DRA replaced counting GPUs as integers with attribute-based claims the stock scheduler allocates. NVIDIA's DRA driver uses it to do something the device plugin never could: carve a *secure, per-workload NVLink memory domain* across a GB200 rack, using IMEX, and tear it down when the job ends. That's ComputeDomains."

---

## Lab 6 — Single-node inference with vLLM
**Layers 13 (serving), 9.** *Goal: stand up an OpenAI-compatible endpoint and understand PagedAttention / continuous batching.*

```yaml
# vllm.yaml — single-GPU serving
apiVersion: apps/v1
kind: Deployment
metadata: { name: vllm }
spec:
  replicas: 1
  selector: { matchLabels: { app: vllm } }
  template:
    metadata: { labels: { app: vllm } }
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args: ["--model","Qwen/Qwen2.5-0.5B-Instruct"]   # small model for a cheap GPU
        ports: [{ containerPort: 8000 }]
        resources: { limits: { nvidia.com/gpu: 1 } }
---
apiVersion: v1
kind: Service
metadata: { name: vllm }
spec:
  selector: { app: vllm }
  ports: [{ port: 8000, targetPort: 8000 }]
```
```bash
kubectl apply -f vllm.yaml
kubectl port-forward svc/vllm 8000:8000 &
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","prompt":"Explain NVLink in one sentence:","max_tokens":64}'
```
**Observe & explain:** watch vLLM logs for KV-cache block allocation and batching. Talking point: "PagedAttention stores the KV cache in fixed 16-token blocks like OS virtual-memory paging, so there's no fragmentation; continuous batching injects new requests every forward pass to keep the GPU busy."

---

## Lab 7 — Multi-node inference (LWS) and disaggregated serving (Dynamo)
**Layers 8, 9.** *Goal: the inference equivalent of Lab 3 — a model group deployed as one unit; then the 2025-26 disaggregated story.*

**7a — LeaderWorkerSet (multi-node vLLM):**
```bash
# Install LWS
kubectl apply --server-side -f https://github.com/kubernetes-sigs/lws/releases/download/v0.9.0/manifests.yaml
kubectl get pods -n lws-system
```
Deploy a `LeaderWorkerSet` with `replicas: 1`, `size: 2` (1 leader + 1 worker), a `leaderTemplate` running the vLLM/Ray head and a `workerTemplate` running a Ray worker (use the vLLM example manifest from `kubernetes-sigs/lws/docs/examples/vllm`). 
**Observe:** `LWS_LEADER_ADDRESS`, `LWS_GROUP_SIZE`, `LWS_WORKER_INDEX` injected; one headless Service + StatefulSet per group; the group upgrades/restarts all-or-nothing. Explain: "This is the inference analog of the training operator — the *group* is the unit, because the model is sharded across nodes."

**7b — Dynamo disaggregated (aspirational / read-along if GPU-limited):**
Study the Dynamo + Grove + KAI deployment. Point out: separate **prefill** and **decode** pods (PodCliques under a PodCliqueSet), a **Smart Router** frontend, KAI gang-scheduling the whole topology onto one NVLink domain. Even without running it, be able to name the four Dynamo components (Smart Router, Planner, KVBM, NIXL) and why disaggregation wins (compute-bound prefill vs bandwidth-bound decode scaled independently). If you have ≥2 GPUs, follow the Dynamo quickstart to run a minimal disaggregated deployment locally.

---

## Capstone — Record the end-to-end demo
Chain Labs 1 → 3 → 4 → 6 into the demo storyline from `01_pitch_and_demo.md`, and **record it** (`gif`/screen capture). This recording is your portfolio artifact and your live-demo fallback. Push all manifests + `train.py` + a README to a **public GitHub repo** — for a Developer Advocate role the public repo is effectively part of the application.

---

## Cheat-sheet: what each lab proves in the interview
| Lab | The sentence it lets you say |
|---|---|
| 1 | "I've stood up the GPU Operator and traced CDI device injection." |
| 2 | "I've read NCCL's topology detection choosing NVLink vs IB." |
| 3 | "I've run multi-node TrainJob v2 and shown the injected MASTER_ADDR/RANK." |
| 4 | "I've reproduced the partial-gang deadlock and fixed it with KAI gang scheduling." |
| 5 | "I can explain DRA ResourceClaims and GB200 ComputeDomains/IMEX from real manifests." |
| 6 | "I've served a model with vLLM and can explain PagedAttention." |
| 7 | "I've deployed multi-node inference on LWS and can explain Dynamo disaggregation." |

> **Version caution:** image tags, chart versions, and CRD `apiVersion`s in this guide move fast. Before any live demo, re-pin versions against the current GitHub releases / docs (KAI, LWS, Kubeflow Trainer, nvidia-dra-driver-gpu, GPU Operator). Treat every version string here as "verify before quoting."
