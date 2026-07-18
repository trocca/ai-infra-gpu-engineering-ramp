// Lab 02/cuda — experiment 1 (coalescing sweep), experiment 2 (AoS vs SoA),
// experiment 3 (pageable vs pinned H2D bandwidth).
// Lesson: ../README.md — §Experiments 1–3, GPU column of the table.
// Refs:   CUDA C++ Best Practices Guide §"Coalesced Access to Global Memory",
//         §"Pinned Memory"; NVIDIA blog "How to Access Global Memory
//         Efficiently in CUDA C/C++".
// Build:  nvcc -O3 -arch=native stride_sweep.cu -o stride_gpu

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_check.cuh"

constexpr size_t N = 1 << 26;

// Thread i reads x[(i*stride) % n]: stride 1 = coalesced, stride 2 already
// halves sector utilization, stride 32 = one transaction per lane.
__global__ void gather(const float* __restrict__ x, float* __restrict__ y,
                       size_t n, size_t stride) {
    size_t gs = size_t(gridDim.x) * blockDim.x;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += gs)
        y[i] = x[(i * stride) % n];
}

struct Particle { float x, y, z, w; };

__global__ void sum_aos(const Particle* __restrict__ p, float* out, size_t m) {
    size_t gs = size_t(gridDim.x) * blockDim.x;
    float s = 0.f;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < m; i += gs)
        s += p[i].x;  // warp touches every 4th float: 1/4 of each sector used
    atomicAdd(out, s);
}

__global__ void sum_soa(const float* __restrict__ x, float* out, size_t m) {
    size_t gs = size_t(gridDim.x) * blockDim.x;
    float s = 0.f;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < m; i += gs)
        s += x[i];
    atomicAdd(out, s);
}

int main() {
    float *dx, *dy, *dout;
    CUDA_CHECK(cudaMalloc(&dx, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dout, sizeof(float)));
    CUDA_CHECK(cudaMemset(dx, 0, N * sizeof(float)));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int block = 256, grid = prop.multiProcessorCount * 32;
    const double bytes = 2.0 * N * sizeof(float);  // read + write

    std::printf("== coalescing sweep ==\n");
    for (size_t stride : {1, 2, 4, 8, 16, 32}) {
        float ms = cuda_time_best_ms([&] {
            gather<<<grid, block>>>(dx, dy, N, stride);
            CUDA_CHECK_LAST();
        });
        std::printf("stride %2zu: %8.3f ms  %8.1f GB/s (effective)\n", stride, ms,
                    bytes / ms / 1e6);
    }

    std::printf("\n== AoS vs SoA ==\n");
    const size_t M = N / 4;
    Particle* dp;
    CUDA_CHECK(cudaMalloc(&dp, M * sizeof(Particle)));
    CUDA_CHECK(cudaMemset(dp, 0, M * sizeof(Particle)));
    float ms = cuda_time_best_ms([&] {
        CUDA_CHECK(cudaMemsetAsync(dout, 0, sizeof(float)));
        sum_aos<<<grid, block>>>(dp, dout, M);
        CUDA_CHECK_LAST();
    });
    std::printf("AoS: %8.3f ms\n", ms);
    ms = cuda_time_best_ms([&] {
        CUDA_CHECK(cudaMemsetAsync(dout, 0, sizeof(float)));
        sum_soa<<<grid, block>>>(dx, dout, M);
        CUDA_CHECK_LAST();
    });
    std::printf("SoA: %8.3f ms  (expect ~4x faster)\n", ms);

    std::printf("\n== H2D: pageable vs pinned (1 GB) ==\n");
    const size_t B = 1ull << 30;
    float* dbig;
    CUDA_CHECK(cudaMalloc(&dbig, B));
    std::vector<char> pageable(B);
    ms = cuda_time_best_ms([&] {
        CUDA_CHECK(cudaMemcpy(dbig, pageable.data(), B, cudaMemcpyHostToDevice));
    }, 3, 1);
    std::printf("pageable: %8.3f ms  %6.1f GB/s\n", ms, B / ms / 1e6);
    char* pinned;
    CUDA_CHECK(cudaMallocHost(&pinned, B));
    ms = cuda_time_best_ms([&] {
        CUDA_CHECK(cudaMemcpy(dbig, pinned, B, cudaMemcpyHostToDevice));
    }, 3, 1);
    std::printf("pinned:   %8.3f ms  %6.1f GB/s  (this is PyTorch's pin_memory=True)\n",
                ms, B / ms / 1e6);

    CUDA_CHECK(cudaFreeHost(pinned));
    CUDA_CHECK(cudaFree(dbig));
    CUDA_CHECK(cudaFree(dp));
    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
    CUDA_CHECK(cudaFree(dout));
    return 0;
}
