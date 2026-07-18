// Lab 12/csrc — capstone: fused softmax CUDA kernel. One block per row; the
// row is read once from HBM, reduced twice (max, sum) through the module-04
// funnel (warp shuffle -> shared), normalized, written once. Eager PyTorch
// spends 4 kernels + 4 HBM round trips on the same math — that delta is the
// whole point of fusion.
// Lesson: ../README.md — plan step 3; stretch goal (rows > 4096 via online
//         softmax) deliberately not implemented.
// Refs:   Milakov & Gimelshein, "Online normalizer calculation for softmax"
//         (arXiv:1805.02867); module 04's reduce.cu (the funnel, verbatim);
//         aten/src/ATen/native/cuda/SoftMax.cu (read AFTER writing yours).
// Build:  loaded JIT by ../lab.py via torch.utils.cpp_extension.load().

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cfloat>

namespace {

// Functors, not __device__ lambdas: extended lambdas need a special nvcc
// flag; plain functors compile everywhere the same way.
struct MaxOp {
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};
struct AddOp {
    __device__ float operator()(float a, float b) const { return a + b; }
};

// Generic warp fold: works for max and sum with the same shuffle pattern.
template <typename Op>
__device__ float warp_fold(float v, Op op) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v = op(v, __shfl_down_sync(0xffffffff, v, offset));
    return v;
}

// Block-level fold via the 4-level funnel; result broadcast to all threads.
template <typename Op>
__device__ float block_fold(float v, Op op, float identity) {
    __shared__ float partials[32];
    __shared__ float result;
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    v = warp_fold(v, op);
    if (lane == 0) partials[warp] = v;
    __syncthreads();
    if (warp == 0) {
        int n_warps = (blockDim.x + 31) / 32;
        v = lane < n_warps ? partials[lane] : identity;
        v = warp_fold(v, op);
        if (lane == 0) result = v;
    }
    __syncthreads();
    return result;
}

__global__ void fused_softmax_kernel(const float* __restrict__ in,
                                     float* __restrict__ out,
                                     int64_t cols) {
    const float* row = in + int64_t(blockIdx.x) * cols;   // one block per row
    float* orow = out + int64_t(blockIdx.x) * cols;

    // Pass 1: row max (each thread strides the row, then the funnel).
    float m = -FLT_MAX;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) m = fmaxf(m, row[c]);
    m = block_fold(m, MaxOp{}, -FLT_MAX);

    // Pass 2: exp and sum. exp result parked in out[] so the row is only
    // read from HBM once.
    float s = 0.f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
        float e = __expf(row[c] - m);
        orow[c] = e;
        s += e;
    }
    s = block_fold(s, AddOp{}, 0.f);

    // Pass 3: normalize in place.
    float inv = 1.f / s;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) orow[c] *= inv;
}

}  // namespace

at::Tensor fused_softmax_cuda(const at::Tensor& input) {
    TORCH_CHECK(input.dim() == 2, "expected 2D [rows, cols]");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "fp32 only in the scaffold");
    auto x = input.contiguous();
    auto out = at::empty_like(x);
    const int64_t rows = x.size(0), cols = x.size(1);
    if (rows == 0 || cols == 0) return out;

    const int block = 256;
    // Launch on the current stream — never the legacy default stream, or you
    // serialize against everything PyTorch has in flight.
    auto stream = at::cuda::getCurrentCUDAStream();
    fused_softmax_kernel<<<static_cast<unsigned>(rows), block, 0, stream>>>(
        x.const_data_ptr<float>(), out.mutable_data_ptr<float>(), cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
