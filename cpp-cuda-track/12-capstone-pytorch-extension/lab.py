# Lab 12 — build the extension, verify both backends, benchmark honestly.
# Lesson: README.md plan steps 1-5 (this file drives steps 1-3 and the
#         measurement half of step 5; steps 4-5 proper are yours).
# Refs:   torch.utils.cpp_extension.load docs; torch.testing.assert_close;
#         "PyTorch Custom Operators Manual" for the torch.ops.* namespace.
# Run:    python lab.py   (CPU-only boxes skip the CUDA sections cleanly)

import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

CSRC = Path(__file__).parent / "csrc"


def build():
    sources = [str(CSRC / "ops.cpp")]
    flags = []
    if torch.cuda.is_available():
        sources.append(str(CSRC / "softmax.cu"))
        flags.append("-DWITH_CUDA")  # gates the CUDA registration in ops.cpp
    else:
        print("no CUDA device: building CPU backend only")
    # JIT build: recompiles only when sources change. First run is slow.
    load(name="ramp_ops", sources=sources, extra_cflags=flags, verbose=False)
    return torch.ops.ramp.fused_softmax


def verify(op, device):
    torch.manual_seed(0)
    for rows, cols in [(1, 1), (3, 7), (128, 1000), (256, 4096)]:
        x = torch.randn(rows, cols, device=device)
        torch.testing.assert_close(op(x), torch.softmax(x, dim=-1))
    # Large-magnitude inputs: catches a missing max-shift immediately.
    x = torch.randn(64, 512, device=device) * 100
    torch.testing.assert_close(op(x), torch.softmax(x, dim=-1))
    print(f"verify [{device}]: OK")


def bench(fn, x, iters=50, warmup=5):
    for _ in range(warmup):
        fn(x)
    if x.is_cuda:
        torch.cuda.synchronize()  # never trust wall time around async launches
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    op = build()
    verify(op, "cpu")
    if torch.cuda.is_available():
        verify(op, "cuda")

    compiled = torch.compile(lambda t: torch.softmax(t, dim=-1))
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        print(f"\n== {device}: ms/call, rows=4096 ==")
        for cols in (256, 1024, 4096):
            x = torch.randn(4096, cols, device=device)
            ours = bench(op, x)
            eager = bench(lambda t: torch.softmax(t, dim=-1), x)
            comp = bench(compiled, x)
            print(f"cols {cols:5d}: ours {ours:7.3f}  eager {eager:7.3f}  "
                  f"torch.compile {comp:7.3f}")
    print("\nnext (plan steps 4-5): backward + read Inductor's kernel via "
          "TORCH_LOGS=output_code")


if __name__ == "__main__":
    main()
