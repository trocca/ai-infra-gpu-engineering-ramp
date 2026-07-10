# Week 10 · Day 5 — Datacenter architecture + domain review

[← Master Plan](../../../MASTER-PLAN.md) · [Week 10 overview](plan.md) · [← previous day](day-4.md) · [next day →](../week-11/day-1.md)

Friday: zoom out to the whole datacenter — the mental picture that makes domains 1 and 2
cohere — then the closed-book self-check that closes out Administration (23%). After
today, two exam domains (31% + 23% = 54% of the MCQs) should feel owned.

## Study block (2 h)

### 1. Anatomy of an AI datacenter (0:00–0:45)

**The node** (DGX/HGX H100-class): 8× GPUs on an SXM baseboard, all-to-all via
**NVSwitch** (~900 GB/s NVLink per GPU — the "NVLink domain" where tensor parallelism
lives, per your own week-10 measurements); 8× InfiniBand NICs — **one per GPU, one per
rail**; 2× CPUs; local NVMe scratch; a BMC.

**The pod (SuperPOD)**: built from **Scalable Units** (SUs, ~32 nodes / 256 GPUs each) that
snap together with a shared fabric — capacity planning by unit, not by ad-hoc node count.

**The fabric**: leaf–spine / **fat-tree** topologies, ideally non-blocking (full bisection
bandwidth). **Rail-optimized**: NIC *i* of every node → leaf switch *i* ("rail *i*"), so
NCCL's per-GPU flows stay one hop within a rail; only crossing rails climbs to the spine.
This is *why* NCCL's ring/tree algorithms map well to these clusters — and you've now
measured both sides of that story (week 9 = the collectives, week 10 = what TP/PP send).

**Rail-optimized wiring — GPU i on every node shares leaf i, so NCCL's per-GPU flows stay one hop.**

```
        node A                      node B
  GPU0--NIC0 ------> leaf 0 <------ NIC0--GPU0     rail 0
  GPU1--NIC1 ------> leaf 1 <------ NIC1--GPU1     rail 1
   ...   ...          ...            ...   ...
  GPU7--NIC7 ------> leaf 7 <------ NIC7--GPU7     rail 7
                       |
                     spine    <- only cross-rail traffic climbs here
```

**Power & cooling**: AI racks run ~10 kW (air) up to 100–120+ kW (liquid) — DGX-class
nodes are ~10 kW *each*, so density forces **liquid cooling** (direct-to-chip cold plates)
in new builds. Power+cooling, not money, is the binding constraint on modern buildouts —
a legitimate exam-level fact.

**Storage tiers**: local NVMe (scratch/checkpoint staging) → parallel filesystem
(**Lustre/GPFS a.k.a. Spectrum Scale**, the training-data workhorse) → object store (bulk,
datasets, archives). Rule of thumb: checkpoints hammer write bandwidth, dataloaders hammer
random-read IOPS — different tiers solve different halves.

### 2. The management-plane diagram (0:45–1:15)

Draw this once, from memory, and keep it — it's the one-picture answer to a dozen MCQs:

```
 [externalnet] ── head node(s): BCM/CMDaemon, slurmctld, K8s control plane
      │
 [management net]  ── provisioning, cmsh/CMDaemon agents, Slurm/K8s control traffic
 [BMC/IPMI net]    ── out-of-band power/console (works when the OS is dead)
 [compute fabric]  ── IB/RoCE rails — NCCL collectives, TP/PP/DP traffic
 [storage net]     ── parallel FS I/O (Lustre/GPFS)
```

Placements to be able to recite: **BCM** on the management net; **BMC** out-of-band;
**slurmctld / K8s API** on management; **NCCL** exclusively on the compute fabric;
**dataloaders/checkpoints** on the storage net. Most "why is X slow/broken" exam scenarios
are one plane's traffic leaking onto another, or a plane being down while the others look
healthy (Week 9 Day 4's failure modes, now with the full map).

### 3. Closed-book review (1:15–2:00)

- [self-check.md](self-check.md) **closed-book**. Log misses in `notes.md` with the
  why-I-missed-it line. Target ≥ 15/18.
- Walk the **exit criteria in [plan.md](plan.md)**: drain + QOS + account + `sacct` without
  docs; MIG 2×3g.20gb from memory (manual + label); Run:ai→KAI mapping with the demo
  behavior you can point at; Role+RoleBinding+GPU ResourceQuota in < 5 min; SuperPOD sketch
  on paper. **Exam slot booked?** — it's an exit criterion this week.
- Flashcards pass over **domains 1–2 combined** (weeks 9+10 decks).
- Add this week's row to [PROGRESS.md](../../PROGRESS.md).

### Quick check

1. What is a Scalable Unit and why do SuperPOD designs think in SUs?
2. Explain "rail-optimized" in two sentences, including what traffic stays on a rail.
3. Which storage tier does a 2 TB checkpoint write hit, and which does shuffled dataloading stress?
4. Name the plane: `cmsh power reset`, an all-reduce, a `kubectl apply`, a Lustre read.

<details><summary>Answers</summary>

1. A standardized block (~32 nodes/256 GPUs) with its own leaf switching that snaps into the spine — pods scale by adding SUs, keeping topology, cabling, and performance predictable.
2. NIC *i* of every node connects to leaf switch *i*, forming rail *i*; since GPU *i* talks through NIC *i*, NCCL's per-GPU peer flows (ring neighbors, tree peers) stay one hop inside a rail and only cross-rail traffic transits the spine.
3. Checkpoint write → parallel FS (Lustre/GPFS) sequential-write bandwidth (staged via local NVMe); shuffled dataloading → random-read IOPS, ideally local NVMe cache or the parallel FS's read side.
4. BMC/out-of-band net · compute fabric (IB/RoCE) · management net (K8s API) · storage net.

</details>

## Build block (4 h)

**Local, free — the taxonomy writeup + publish (Friday rule).**
Brief: [week-10-parallelism-internals/README.md](../../../gpu-engineering-lab/03-scale-and-serve/week-10-parallelism-internals/README.md)

- [ ] Write `PARALLELISM.md`: DP vs TP vs PP vs FSDP — what each shards, what/when each communicates, what link speed each demands, when each wins — **with your measured weeks-9/10 numbers as evidence** (scaling efficiency, all-reduce counts, bubble plot, OOM→fits table).
- [ ] Bubble plot + results table land in the README with real numbers; residuals explained.
- [ ] All acceptance boxes checked; cost log finalized (≤ $25); **push/publish**.

Hint: write PARALLELISM.md as the answer to "why does TP stay inside the NVLink domain
while PP/DP cross nodes?" — your PCIe TP-vs-PP numbers *are* the proof, and this morning's
rail/NVLink-domain material is the same argument at datacenter scale. One document, both
repos' story.

## Close the day (15 min)

- Anki: SU, rail-optimized, storage tiers, plane placements + full weeks-9/10 deck review.
- `notes.md`: self-check score + weakest area going into week 11 (troubleshooting weeks feed on weak spots).
- Blockers → week 11 day 1.
- Cloud: nothing rented today. Final console sweep — **zero instances running** over the weekend; both weeks' cost logs closed out.
