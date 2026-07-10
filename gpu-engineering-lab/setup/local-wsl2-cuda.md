# Local setup — RTX 5090 Laptop (Blackwell, sm_120) on WSL2

All Phase 1–2 work runs locally. The one thing that matters: **Blackwell consumer GPUs
(compute capability 12.0) need a recent toolchain** — CUDA Toolkit ≥ 12.8 and a PyTorch
build with cu128+ wheels. Older toolkits will compile for sm_90 and silently fail or
JIT-fallback.

## 1. Windows side (already done, verify)

```powershell
nvidia-smi          # driver 592.01 — fine; the Windows driver serves WSL2 too.
wsl --status        # Ubuntu, WSL version 2
```

Never install a Linux NVIDIA driver inside WSL — the Windows driver is passed through
as `/usr/lib/wsl/lib/libcuda.so`.

## 2. Inside WSL2 Ubuntu

```bash
# confirm the GPU is visible
nvidia-smi

# CUDA toolkit (WSL-specific repo — installs toolkit WITHOUT a driver)
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-9 build-essential cmake ninja-build
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
nvcc --version      # must report ≥ 12.8

# sanity: compile & run a first kernel for sm_120
cat > /tmp/hello.cu <<'EOF'
#include <cstdio>
__global__ void hello() { printf("hello from block %d thread %d\n", blockIdx.x, threadIdx.x); }
int main() { hello<<<2,4>>>(); cudaDeviceSynchronize(); }
EOF
nvcc -arch=sm_120 -o /tmp/hello /tmp/hello.cu && /tmp/hello
```

## 3. Python environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv ~/venvs/gpulab --python 3.12 && source ~/venvs/gpulab/bin/activate
# PyTorch with cu128+ wheels (sm_120 support landed in the cu128 builds)
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
# expect: (12, 0)
uv pip install numpy matplotlib pytest triton transformers datasets accelerate
```

If `torch.cuda.is_available()` is False inside WSL: `wsl --shutdown` from Windows and
reopen — the GPU passthrough occasionally needs a restart after driver updates.

## 4. Profilers

- **Nsight Systems** (timeline): `sudo apt-get install nsight-systems` or download the
  .deb from developer.nvidia.com. `nsys profile -o out python train.py`, open the
  report in the Windows Nsight Systems GUI (it reads WSL-generated .nsys-rep files).
- **Nsight Compute** (per-kernel): `ncu --set full -o report ./kernel_binary`. On WSL2,
  GPU performance counters require enabling "Developer > Allow access to GPU
  performance counters" in the Windows NVIDIA Control Panel (NVIDIA app → System).

## 5. Laptop-specific benchmarking honesty

This is a laptop GPU: clocks vary with thermals and power profile. For every benchmark:
plug in AC, set Windows power mode to Best Performance, run
`nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv -l 1` alongside, do 10
warmup iterations, report median of ≥ 50 runs. Note "laptop 5090, power-limited" in
results tables — reviewers respect measurement honesty more than big numbers.
