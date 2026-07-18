// Lab 06/cuda — SGEMM: naive vs shared-memory tiled.
// Lesson: ../README.md — §Concept (shared-memory tiling), exercises 1–4.
// Refs:   CUDA C++ Programming Guide §"Shared Memory" (this exact kernel is
//         its worked example); Boehm, "How to Optimize a CUDA Matmul Kernel
//         for cuBLAS-like Performance" (siboehm.com — the follow-on steps);
//         CUTLASS docs for where the road leads.
// Build:  nvcc -O3 -arch=native matmul.cu -o matmul_gpu

#include <cmath>
#include <cstdio>
#include <vector>

#include "../../common/cuda_check.cuh"

constexpr int N = 1024;
constexpr int TILE = 32;  // block = TILE x TILE threads; exercise 2: try 16

__global__ void matmul_naive(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;
    float acc = 0.f;
    for (int k = 0; k < n; ++k) acc += A[row * n + k] * B[k * n + col];
    C[row * n + col] = acc;
}

// Classic tiled SGEMM: each block loads a TILE x TILE tile of A and of B into
// shared memory (coalesced), computes TILE partial products per thread from
// the scratchpad, slides along K. Each global element is loaded n/TILE times
// instead of n times -> arithmetic intensity up by TILE.
__global__ void matmul_tiled(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.f;

    for (int t = 0; t < n / TILE; ++t) {
        As[threadIdx.y][threadIdx.x] = A[row * n + t * TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * n + col];
        __syncthreads();  // tile fully loaded before anyone computes
#pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();  // everyone done computing before tile is overwritten
    }
    C[row * n + col] = acc;
}

int main() {
    static_assert(N % TILE == 0, "keep it simple: N divisible by TILE");
    std::vector<float> hA(N * N), hB(N * N), hC(N * N), hRef(N * N);
    for (int i = 0; i < N * N; ++i) {
        hA[i] = float((i * 7) % 13) * 0.1f;
        hB[i] = float((i * 5) % 11) * 0.1f;
    }

    float *dA, *dB, *dC;
    size_t bytes = size_t(N) * N * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, N / TILE);
    const double flops = 2.0 * N * N * double(N);

    float ms = cuda_time_best_ms([&] {
        matmul_naive<<<grid, block>>>(dA, dB, dC, N);
        CUDA_CHECK_LAST();
    });
    std::printf("naive: %8.3f ms  %8.1f GFLOP/s\n", ms, flops / ms / 1e6);
    CUDA_CHECK(cudaMemcpy(hRef.data(), dC, bytes, cudaMemcpyDeviceToHost));

    ms = cuda_time_best_ms([&] {
        matmul_tiled<<<grid, block>>>(dA, dB, dC, N);
        CUDA_CHECK_LAST();
    });
    std::printf("tiled %2d: %6.3f ms  %8.1f GFLOP/s\n", TILE, ms, flops / ms / 1e6);
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < N * N && ok; ++i)
        if (std::abs(hC[i] - hRef[i]) > 1e-2f * std::abs(hRef[i]) + 1e-3f) ok = false;
    std::printf("verify tiled vs naive: %s\n", ok ? "OK" : "FAIL");
    std::printf("exercise 3: add cublasSgemm here and weep\n");

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}
