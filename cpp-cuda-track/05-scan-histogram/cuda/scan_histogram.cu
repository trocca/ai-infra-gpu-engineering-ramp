// Lab 05/cuda — single-block shared-memory scan (exercise 1) and histogram:
// global atomics vs block-privatized shared-memory atomics (exercise 2).
// Lesson: ../README.md — §Concept (scan GPU / histogram both), exercises 1–3.
// Refs:   GPU Gems 3 ch.39 "Parallel Prefix Sum (Scan) with CUDA"
//         (Hillis–Steele & Blelloch schemes); Merrill & Garland, "Single-pass
//         Parallel Prefix Scan with Decoupled Look-back" (NVIDIA tech report);
//         cub::DeviceScan / cub::DeviceHistogram docs.
// Build:  nvcc -O3 -arch=native scan_histogram.cu -o scanhist_gpu

#include <cstdio>
#include <numeric>
#include <vector>

#include "../../common/cuda_check.cuh"

constexpr int SCAN_N = 1024;
constexpr size_t H = 1 << 28;  // 256M bytes
constexpr int BINS = 256;

// Hillis–Steele inclusive scan, one block, double-buffered shared memory.
// O(n log n) work but O(log n) depth — the didactic version. CUB does better.
__global__ void scan_block(const float* __restrict__ x, float* __restrict__ out, int n) {
    __shared__ float buf[2][SCAN_N];
    int tid = threadIdx.x;
    int cur = 0;
    buf[cur][tid] = tid < n ? x[tid] : 0.f;
    __syncthreads();
    for (int offset = 1; offset < n; offset <<= 1) {
        int nxt = 1 - cur;
        buf[nxt][tid] = tid >= offset ? buf[cur][tid] + buf[cur][tid - offset]
                                      : buf[cur][tid];
        cur = nxt;
        __syncthreads();
    }
    if (tid < n) out[tid] = buf[cur][tid];
}

__global__ void hist_global(const unsigned char* __restrict__ d, size_t n,
                            unsigned int* __restrict__ bins) {
    size_t gs = size_t(gridDim.x) * blockDim.x;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += gs)
        atomicAdd(&bins[d[i]], 1u);  // all blocks hammer 256 global words
}

__global__ void hist_privatized(const unsigned char* __restrict__ d, size_t n,
                                unsigned int* __restrict__ bins) {
    __shared__ unsigned int local[BINS];
    for (int b = threadIdx.x; b < BINS; b += blockDim.x) local[b] = 0;
    __syncthreads();
    size_t gs = size_t(gridDim.x) * blockDim.x;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += gs)
        atomicAdd(&local[d[i]], 1u);  // shared-memory atomics: fast, block-local
    __syncthreads();
    for (int b = threadIdx.x; b < BINS; b += blockDim.x)
        atomicAdd(&bins[b], local[b]);  // one global add per bin per block
}

int main() {
    // --- scan ---
    std::vector<float> hx(SCAN_N), href(SCAN_N), hout(SCAN_N);
    for (int i = 0; i < SCAN_N; ++i) hx[i] = float(i % 7) * 0.25f;
    std::inclusive_scan(hx.begin(), hx.end(), href.begin());
    float *dx, *dout;
    CUDA_CHECK(cudaMalloc(&dx, SCAN_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dout, SCAN_N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), SCAN_N * sizeof(float), cudaMemcpyHostToDevice));
    scan_block<<<1, SCAN_N>>>(dx, dout, SCAN_N);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(hout.data(), dout, SCAN_N * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int i = 0; i < SCAN_N; ++i)
        if (std::abs(hout[i] - href[i]) > 1e-2f) ok = false;
    std::printf("scan verify: %s\n\n", ok ? "OK" : "FAIL");

    // --- histogram ---
    std::vector<unsigned char> uniform(H), constant(H, 42);
    unsigned s = 12345;
    for (size_t i = 0; i < H; ++i) { s = s * 1664525u + 1013904223u; uniform[i] = s >> 24; }

    unsigned char* dd;
    unsigned int* dbins;
    CUDA_CHECK(cudaMalloc(&dd, H));
    CUDA_CHECK(cudaMalloc(&dbins, BINS * sizeof(unsigned int)));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int block = 256, grid = prop.multiProcessorCount * 32;

    for (auto [name, src] : {std::pair{"uniform", uniform.data()},
                             std::pair{"all-same-value", constant.data()}}) {
        CUDA_CHECK(cudaMemcpy(dd, src, H, cudaMemcpyHostToDevice));
        float ms = cuda_time_best_ms([&] {
            CUDA_CHECK(cudaMemsetAsync(dbins, 0, BINS * sizeof(unsigned int)));
            hist_global<<<grid, block>>>(dd, H, dbins);
            CUDA_CHECK_LAST();
        }, 5, 1);
        std::printf("%-15s global atomics: %9.3f ms  %7.1f GB/s\n", name, ms, H / ms / 1e6);
        ms = cuda_time_best_ms([&] {
            CUDA_CHECK(cudaMemsetAsync(dbins, 0, BINS * sizeof(unsigned int)));
            hist_privatized<<<grid, block>>>(dd, H, dbins);
            CUDA_CHECK_LAST();
        }, 5, 1);
        std::printf("%-15s privatized:     %9.3f ms  %7.1f GB/s\n", name, ms, H / ms / 1e6);
    }
    std::printf("(compare the all-same-value rows with the CPU lab's)\n");

    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dout));
    CUDA_CHECK(cudaFree(dd));
    CUDA_CHECK(cudaFree(dbins));
    return 0;
}
