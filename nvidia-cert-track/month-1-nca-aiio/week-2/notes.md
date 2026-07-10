# Week 2 Notes — AI Infrastructure part 1: GPUs and GPU systems

## 1. Datacenter GPU generations

| Architecture | Flagship GPU(s) | Year | Signature features | Memory |
|---|---|---|---|---|
| Ampere | | | | |
| Hopper | | | | |
| Blackwell | | | | |

- Grace CPU is…
- Grace Hopper (GH200) / Grace Blackwell (GB200) superchips are…
- NVLink-C2C is…
- Transformer Engine (one line):

## 2. DGX / HGX / MGX / PODs

- DGX (what it is, what's inside a DGX H100):
- HGX (who uses it, what it contains):
- DGX vs HGX — the one-sentence sales answer:
- MGX:
- DGX BasePOD:
- DGX SuperPOD:
- Base Command (management layer):
- SXM vs PCIe form factor:

## 3. NVLink / NVSwitch / multi-GPU scaling

### Interconnects (rank by bandwidth)
- PCIe Gen5 x16:
- NVLink (per-GPU total, H100):
- NVLink 5 (Blackwell):
- NVSwitch — what problem it solves:
- NVLink Switch System / NVL72:

### Scaling concepts
- Why all-reduce makes interconnect bandwidth matter:
- Data parallelism:
- Tensor (model) parallelism:
- Pipeline parallelism:
- Why 8 GPUs ≠ 8× speedup:
- Strong vs weak scaling:

## 4. GPU memory

- HBM — what it is and why it's fast:
- HBM bandwidth examples (H100 / H200):
- GDDR vs HBM vs system DDR:
- Memory-bound vs compute-bound workloads (LLM inference example):
- Rule of thumb: bytes per parameter at FP16 → GPU memory needed for an N-billion-param model:

## 5. Precision formats

| Format | Bits | Range/precision trade-off | Typical use | Introduced (Tensor Core support) |
|---|---|---|---|---|
| FP64 | | | | |
| FP32 | | | | |
| TF32 | | | | |
| FP16 | | | | |
| BF16 | | | | |
| FP8 | | | | |
| INT8 | | | | |
| FP4 | | | | |

- Why lower precision helps (two reasons):
- BF16 vs FP16 — why BF16 is easier for training:
- Mixed-precision training in one line:
- Quantization (inference) in one line:

## 6. Tensor Cores

- What a Tensor Core does:
- Generation history (Volta → Ampere → Hopper → Blackwell), one feature each:
- Structured sparsity (2:4):

## Parking lot
-
