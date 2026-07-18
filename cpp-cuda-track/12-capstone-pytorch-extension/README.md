# 12 — Capstone: one custom PyTorch op, both backends

Everything converges: write a **fused softmax** (or fused
bias+GeLU — your choice) as a PyTorch extension with a CPU (C++/OpenMP) and a
GPU (CUDA) implementation, registered through the dispatcher so
`torch.ops.ramp.fused_softmax(x)` routes by device — exactly how ATen itself is
built. This is the "PyTorch architecture" briefing made concrete.

## Why fused softmax

Softmax over the last dim = max-reduce, subtract+exp (elementwise), sum-reduce,
divide — i.e., modules 03 and 04 composed. Unfused, that's 4 kernels and 4
round trips to memory; fused, one kernel that keeps the row in
registers/shared memory. It's the canonical memory-bound fusion win and (with
the online-softmax trick) the seed of FlashAttention.

## Plan

1. **Scaffold** — `torch.utils.cpp_extension` with `load()` for iteration:
   `csrc/ops.cpp` (bindings + CPU impl), `csrc/softmax.cu` (kernel). Register
   with `TORCH_LIBRARY(ramp, m)` + `TORCH_LIBRARY_IMPL(ramp, CPU/CUDA, m)` —
   the dispatcher does the device routing, not your Python.
2. **CPU impl** — OpenMP over rows; each row: max, exp-sum, divide. Use
   `at::parallel_for` (plays nice with PyTorch's intra-op thread pool —
   spawning your own OpenMP region inside PyTorch is a classic oversubscription
   bug).
3. **CUDA impl** — one block per row (rows ≤ 4K): grid-stride load to
   registers, warp/block max-reduce (module 04's funnel, with `max` instead of
   `+`), exp + block sum-reduce, divide, coalesced store. Handle the >1-block
   row as a stretch goal with online softmax.
4. **Verify** — `torch.testing.assert_close` vs `torch.softmax` on CPU and GPU,
   fp32 and bf16, including a `requires_grad` note: without a backward
   registration, autograd correctly *refuses* — register a
   `torch.autograd.Function` or an `Autograd` dispatch key impl and re-derive
   the softmax backward by hand.
5. **Measure** — benchmark vs `torch.softmax` and vs `torch.compile`'s fused
   version across row lengths. Inductor's generated Triton will likely match or
   beat your kernel; read the generated code (`TORCH_LOGS=output_code`) and
   note what it did that you didn't.

## Definition of done

- [ ] `pytest` file: correctness CPU+CUDA, fp32+bf16, odd shapes, empty tensor
- [ ] backward implemented and gradcheck passes (fp64 CPU)
- [ ] benchmark table: yours vs eager vs compiled, 3 row lengths
- [ ] half-page NOTES.md: where your kernel loses to Inductor and why
- [ ] stretch: online softmax for arbitrarily long rows; bf16 in / fp32 accumulate

## References

- PyTorch docs: "Custom C++ and CUDA Extensions", "The Custom Operators Manual",
  `TORCH_LIBRARY` registration API.
- Milakov & Gimelshein, "Online normalizer calculation for softmax" (2018).
- PyTorch source: `aten/src/ATen/native/cuda/SoftMax.cu` — read after writing
  yours, not before.
