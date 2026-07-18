// Lab 04/cuda — hierarchical sum: registers -> warp shuffle -> shared -> one
// atomicAdd per block. The template for every reduction-shaped ML kernel.
// Lesson: ../README.md — §Concept (the 4-level funnel), exercises 1–2, 4.
// Refs:   NVIDIA blog "Faster Parallel Reductions on Kepler" (Harris — the
//         canonical shuffle-reduction walkthrough); CUDA C++ Programming
//         Guide §"Warp Shuffle Functions"; cub::DeviceReduce docs.
// Build:  nvcc -O3 -arch=native reduce.cu -o reduce_gpu

#include <cstdio>
#include <numeric>
#include <vector>

#include "../../common/cuda_check.cuh"

constexpr size_t N = 1 << 26;

__inline__ __device__ float warp_sum(float v) {
    // Fold 32 lane values into lane 0, register-to-register.
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

__global__ void reduce_sum(const float* __restrict__ x, float* __restrict__ out,
                           size_t n) {
    // Level 1: private register accumulation over a grid-stride range.
    float v = 0.f;
    size_t stride = size_t(gridDim.x) * blockDim.x;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
        v += x[i];

    // Level 2: warp-level tree via shuffles.
    v = warp_sum(v);

    // Level 3: one partial per warp -> shared memory -> first warp folds them.
    __shared__ float warp_partials[32];  // max 1024 threads = 32 warps per block
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) warp_partials[warp] = v;
    __syncthreads();

    if (warp == 0) {
        int n_warps = (blockDim.x + 31) / 32;
        v = lane < n_warps ? warp_partials[lane] : 0.f;
        v = warp_sum(v);
        // Level 4: one atomic per block.
        if (lane == 0) atomicAdd(out, v);
    }
}

int main() {
    std::vector<float> hx(N);
    for (size_t i = 0; i < N; ++i) hx[i] = 1e-3f * float(i % 97);

    float *dx, *dout;
    CUDA_CHECK(cudaMalloc(&dx, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dout, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int block = 256;
    const int grid = prop.multiProcessorCount * 32;

    float ms = cuda_time_best_ms([&] {
        CUDA_CHECK(cudaMemsetAsync(dout, 0, sizeof(float)));
        reduce_sum<<<grid, block>>>(dx, dout, N);
        CUDA_CHECK_LAST();
    });

    float result;
    CUDA_CHECK(cudaMemcpy(&result, dout, sizeof(float), cudaMemcpyDeviceToHost));
    double ref = std::accumulate(hx.begin(), hx.end(), 0.0);

    const double bytes = double(N) * sizeof(float);
    std::printf("hierarchical reduce: %.3f ms, %.1f GB/s\n", ms, bytes / ms / 1e6);
    std::printf("gpu: %.6f  double ref: %.6f  (float rounding differs by design)\n",
                result, ref);
    std::printf("atomics executed per run: %d (one per block)\n", grid);

    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dout));
    return 0;
}
