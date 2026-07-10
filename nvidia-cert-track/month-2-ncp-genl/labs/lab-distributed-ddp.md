# Lab 3 — torchrun DDP on 2 GPUs: Env Vars and NCCL Transports

Run real DistributedDataParallel training on a 2-GPU node, read the environment torchrun injects, and use `NCCL_DEBUG=INFO` to see exactly which transport NCCL picked. This connects the exam's distributed-training questions to the NCCL-transports material in your demo repo.

| | |
|---|---|
| GPU | 2 × on one node: 2×A10G (AWS g5.12xlarge has 4 — use 2), 2×L4 (g6.12xlarge), or a 2×RTX 4090 machine on RunPod/Vast |
| Est. time | 45–75 min |
| Est. cost | 2×L4/A10G instances ≈ $1.50–2.50/hr, 2×4090 community cloud ≈ $0.70–1.30/hr → **$1–3 total** |
| Exam domains served | GPU Acceleration & Optimization (14%) — the exam explicitly assumes DDP/FSDP competency |

> No H100s, no NVLink required. Part of the point is seeing what NCCL does on a commodity box **without** NVLink (spoiler: usually P2P over PCIe, or SHM), and mapping that to the transport hierarchy you demo.

## 0. Prerequisites

```bash
nvidia-smi                                   # expect TWO GPUs listed
nvidia-smi topo -m                           # topology matrix -- KEEP THIS OUTPUT
python3 -m venv ~/ddp && source ~/ddp/bin/activate
pip install "torch>=2.4" --index-url https://download.pytorch.org/whl/cu124
```

Read `nvidia-smi topo -m` now: the GPU0↔GPU1 cell says `NV#` (NVLink), `PIX`/`PXB` (PCIe same switch/bridge), `PHB` (through host bridge), or `SYS` (across sockets). Predict NCCL's transport from this *before* running — then verify.

## 1. The training script

Save as `ddp_train.py`:

```python
import os, torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

def log(msg):  # rank-prefixed print
    print(f"[rank {os.environ.get('RANK','?')}] {msg}", flush=True)

# --- 1. Read what torchrun injected, BEFORE init ---
for var in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
            "MASTER_ADDR", "MASTER_PORT"]:
    log(f"{var} = {os.environ.get(var)}")

# --- 2. Init process group (rendezvous at MASTER_ADDR:MASTER_PORT) ---
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
log(f"init done; device = cuda:{local_rank} ({torch.cuda.get_device_name(local_rank)})")

# --- 3. Toy model + DistributedSampler (each rank gets a disjoint shard) ---
torch.manual_seed(0)
model = torch.nn.Sequential(
    torch.nn.Linear(1024, 4096), torch.nn.ReLU(), torch.nn.Linear(4096, 10)
).cuda(local_rank)
model = DDP(model, device_ids=[local_rank])   # broadcasts rank-0 weights to all

X, y = torch.randn(8192, 1024), torch.randint(0, 10, (8192,))
ds = TensorDataset(X, y)
sampler = DistributedSampler(ds)              # uses RANK/WORLD_SIZE to shard
loader = DataLoader(ds, batch_size=256, sampler=sampler)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(3):
    sampler.set_epoch(epoch)                  # reshuffle shards each epoch
    for xb, yb in loader:
        xb, yb = xb.cuda(local_rank), yb.cuda(local_rank)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()                       # <-- all-reduce of grads happens here, bucketed & overlapped
        opt.step()
    log(f"epoch {epoch} loss {loss.item():.4f}")

# --- 4. Prove the replicas stayed in sync: all-reduce a weight checksum ---
checksum = model.module[0].weight.sum()
gathered = [torch.zeros_like(checksum) for _ in range(dist.get_world_size())]
dist.all_gather(gathered, checksum)
log(f"weight checksums across ranks: {[f'{t.item():.6f}' for t in gathered]}")

dist.destroy_process_group()
```

## 2. Run it with NCCL debug on

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH \
torchrun --standalone --nproc_per_node=2 ddp_train.py 2>&1 | tee ddp_run.log
```

`--standalone` = single-node convenience: torchrun runs the rendezvous itself on localhost.

## 3. Expected output & what to observe

**A. The injected env vars** (top of the log) — fill this table into `week-7/notes.md`:

| Var | rank 0 | rank 1 | Meaning |
|---|---|---|---|
| RANK | 0 | 1 | global rank across the whole job |
| LOCAL_RANK | 0 | 1 | rank within this node → which GPU to bind |
| WORLD_SIZE | 2 | 2 | total processes in the job |
| LOCAL_WORLD_SIZE | 2 | 2 | processes on this node |
| MASTER_ADDR / MASTER_PORT | 127.0.0.1 / 29400-ish | same | rendezvous point for process group init |

On multi-node these are the vars a Kubeflow **TrainJob v2** / PyTorchJob controller sets for you across pods (MASTER_ADDR = rank-0 pod's service) — same contract, more machines. That's the bridge to your demo repo.

**B. NCCL transport selection** — grep the log:

```bash
grep -E "via|NVLS|Connected all" ddp_run.log | sort -u
```

Expect one of, matching your `topo -m` prediction:

```
NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM          # PCIe/NVLink peer-to-peer (most common on 2-GPU nodes)
NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/IPC            # older driver path, same idea
NCCL INFO Channel 00/0 : 0[0] -> 1[1] via SHM/direct/direct  # no P2P possible -> staged through host shared memory
```

Also look for the ring/tree lines (`NCCL INFO Ring 00 : ...`, `Trees ...`) and `Using network Socket` — on a single node that network line is only for bootstrap/fallback; data should flow P2P or SHM.

**C. Sync proof:** the final checksum line must print *identical* values for both ranks — DDP's all-reduce kept the replicas bit-identical.

**D. Negative experiment — force the fallback** (this is the money observation):

```bash
NCCL_DEBUG=INFO NCCL_P2P_DISABLE=1 \
torchrun --standalone --nproc_per_node=2 ddp_train.py 2>&1 | grep -m4 "via"
```

Expect `via SHM` now. Then optionally `NCCL_SHM_DISABLE=1` too and watch it drop to `via NET/Socket` — the slowest rung. Time the 3 epochs in each mode (`time torchrun ...`); even this toy model usually shows Socket clearly slower.

**E. Map it to the hierarchy from your demo repo:**

| This lab (single node) | Multi-node equivalent | Transport class |
|---|---|---|
| P2P/CUMEM over NVLink | — | NVLink/NVSwitch (fastest) |
| P2P/CUMEM over PCIe | — | PCIe P2P |
| SHM | — | host-memory staging |
| NET/Socket (forced) | NET/Socket over Ethernet | TCP fallback |
| — | NET/IB (+ GPUDirect RDMA) | InfiniBand/RoCE, the real cross-node path |

Exam framing: *intra-node* NCCL prefers NVLink/P2P then SHM; *inter-node* it uses NET (IB with RDMA if present, else sockets). This is also why tensor parallelism stays intra-node while pipeline/data parallelism crosses nodes.

## 4. Troubleshooting

- `torch.cuda.device_count()` = 1 → instance/container wasn't granted both GPUs (`CUDA_VISIBLE_DEVICES`? RunPod pod config?).
- Hang at init → stale port: change `--master_port`, or another job left a zombie (`pkill -f torchrun`).
- `NCCL WARN Cuda failure 'peer access is not supported'` → fine; NCCL falls back to SHM — that *is* a result, record it.
- 4090s on consumer boards: P2P may be unavailable (no NVLink, IOMMU quirks) → expect SHM; still a valid lab.

## 5. Cleanup

```bash
rm -f ddp_run.log && deactivate && rm -rf ~/ddp
```

**Terminate the instance** — 2-GPU boxes burn money faster.

## What you should now be able to explain (exam mapping)

- What torchrun launches and every env var it injects (GPU Acceleration 14%)
- Where DDP's all-reduce happens and how it overlaps with backward
- DistributedSampler's role (data sharding = the "data parallel" in DDP)
- How NCCL selects transports and how to verify it from logs — plus the forced-fallback ladder P2P → SHM → Socket
- How the same env-var contract scales to multi-node under Kubeflow TrainJob v2, and why TP wants NVLink while DP/PP tolerate the network (demo-repo crossover)
