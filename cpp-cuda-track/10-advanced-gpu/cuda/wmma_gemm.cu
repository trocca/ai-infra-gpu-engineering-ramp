// Lab 10/cuda — exercise 2: GEMM on tensor cores via the portable wmma:: API.
// FP16 inputs, FP32 accumulate; one warp computes one 16x16 output tile.
// No CPU twin on purpose: this hardware has no real CPU counterpart (closest
// is Intel AMX — see lesson §10b).
// Lesson: ../README.md — §10b, exercise 2; compare GFLOP/s against your
//         module-06 tiled FP32 kernel and against cuBLAS.
// Refs:   CUDA C++ Programming Guide §"Warp Matrix Functions" (wmma API);
//         NVIDIA blog "Programming Tensor Cores in CUDA 9"; CUTLASS docs for
//         the production version of this idea.
// Build:  nvcc -O3 -arch=native wmma_gemm.cu -o wmma_gpu   (needs CC >= 7.0)

#include <cmath>
#include <cstdio>
#include <cuda_fp16.h>
#include <mma.h>
#include <vector>

#include "../../common/cuda_check.cuh"

using namespace nvcuda;

constexpr int N = 1024;          // square GEMM, multiple of 16
constexpr int WM = 16, WN = 16, WK = 16;  // the fragment shape

// Grid: one warp per 16x16 tile of C. Each warp walks K in steps of 16,
// letting the tensor core do the 16x16x16 multiply-accumulate per step.
__global__ void wmma_gemm(const __half* __restrict__ A, const __half* __restrict__ B,
                          float* __restrict__ C, int n) {
    // Which 16x16 tile of C does this warp own?
    int warp_m = (blockIdx.y * blockDim.y + threadIdx.y);            // tile row
    int warp_n = (blockIdx.x * blockDim.x + threadIdx.x) / 32;       // tile col
    if (warp_m * WM >= n || warp_n * WN >= n) return;

    wmma::fragment<wmma::matrix_a, WM, WN, WK, __half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::row_major> fb;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> facc;
    wmma::fill_fragment(facc, 0.0f);

    for (int k = 0; k < n; k += WK) {
        // Fragments load cooperatively across the warp; ld = row stride.
        wmma::load_matrix_sync(fa, A + warp_m * WM * n + k, n);
        wmma::load_matrix_sync(fb, B + k * n + warp_n * WN, n);
        wmma::mma_sync(facc, fa, fb, facc);  // the tensor core instruction
    }
    wmma::store_matrix_sync(C + warp_m * WM * n + warp_n * WN, facc, n,
                            wmma::mem_row_major);
}

int main() {
    std::vector<__half> hA(N * N), hB(N * N);
    std::vector<float> hC(N * N), ref(N * N, 0.f);
    for (int i = 0; i < N * N; ++i) {
        hA[i] = __float2half(float((i * 7) % 5) * 0.25f);
        hB[i] = __float2half(float((i * 3) % 7) * 0.25f);
    }

    __half *dA, *dB;
    float* dC;
    CUDA_CHECK(cudaMalloc(&dA, N * N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dB, N * N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dC, N * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), N * N * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), N * N * sizeof(__half), cudaMemcpyHostToDevice));

    // 4 warps per block in x, 4 tiles per block in y -> block covers 4x4 tiles.
    dim3 block(4 * 32, 4);
    dim3 grid((N / WN) / 4, (N / WM) / 4);
    const double flops = 2.0 * N * N * double(N);

    float ms = cuda_time_best_ms([&] {
        wmma_gemm<<<grid, block>>>(dA, dB, dC, N);
        CUDA_CHECK_LAST();
    });
    std::printf("wmma fp16->fp32: %8.3f ms  %8.1f GFLOP/s\n", ms, flops / ms / 1e6);
    std::printf("(your module-06 fp32 tiled kernel is the number to beat)\n");

    // Verify a 64x64 corner against a CPU reference (full NxN would be slow).
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int i = 0; i < 64 && ok; ++i)
        for (int j = 0; j < 64 && ok; ++j) {
            float acc = 0.f;
            for (int k = 0; k < N; ++k)
                acc += __half2float(hA[i * N + k]) * __half2float(hB[k * N + j]);
            // fp16 inputs: loose tolerance is the honest tolerance
            if (std::fabs(hC[i * N + j] - acc) > 0.05f * std::fabs(acc) + 0.5f) ok = false;
        }
    std::printf("verify 64x64 corner: %s\n", ok ? "OK" : "FAIL");

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}
