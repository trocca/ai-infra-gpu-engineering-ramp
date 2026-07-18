// Lab 08/cuda — atomic contention across the funnel: one global counter
// hammered by every thread, warp-aggregated, block-privatized, and
// scoped atomics from libcu++.
// Lesson: ../README.md — §Concept (scopes), exercises 1–2; run the racy
//         variants under compute-sanitizer --tool racecheck.
// Refs:   libcu++ cuda::atomic / thread_scope docs; CUDA C++ Programming
//         Guide §"Atomic Functions"; NVIDIA blog "CUDA Pro Tip: Optimized
//         Filtering with Warp-Aggregated Atomics".
// Build:  nvcc -O3 -arch=native race_and_fix.cu -o race_gpu

#include <cstdio>
#include <cuda/atomic>

#include "../../common/cuda_check.cuh"

constexpr long N = 1 << 26;  // total increments per variant

// 1. Every thread does atomicAdd on the same global word. Correct but the
//    worst case: 32 lanes per warp collide on one address, warps serialize.
__global__ void count_global(unsigned long long* c, long n) {
    size_t gs = size_t(gridDim.x) * blockDim.x;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += gs)
        atomicAdd(c, 1ull);
}

// 2. Warp aggregation: lanes vote, one lane adds 32. Same result, 1/32 the
//    atomic traffic. (Modern nvcc often does this rewrite for you — compare!)
__global__ void count_warp_agg(unsigned long long* c, long n) {
    size_t gs = size_t(gridDim.x) * blockDim.x;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += gs) {
        unsigned mask = __activemask();
        int leader = __ffs(mask) - 1;
        if ((int)(threadIdx.x % 32) == leader) atomicAdd(c, (unsigned long long)__popc(mask));
    }
}

// 3. Block privatization: shared-memory counter (block scope = cheap),
//    one global atomic per block. Module 04's funnel, counter-shaped.
__global__ void count_block_priv(unsigned long long* c, long n) {
    __shared__ unsigned long long local;
    if (threadIdx.x == 0) local = 0;
    __syncthreads();
    size_t gs = size_t(gridDim.x) * blockDim.x;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += gs)
        atomicAdd(&local, 1ull);  // shared-memory atomic: block scope in hardware
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(c, local);
}

// 4. libcu++ scoped atomic: the C++ memory model vocabulary, on device.
//    thread_scope_device == classic atomicAdd; narrower scopes are cheaper.
__global__ void count_scoped(unsigned long long* c, long n) {
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> ref(*c);
    size_t gs = size_t(gridDim.x) * blockDim.x;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += gs)
        ref.fetch_add(1, cuda::memory_order_relaxed);
}

int main() {
    unsigned long long* dc;
    CUDA_CHECK(cudaMalloc(&dc, sizeof(unsigned long long)));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int block = 256, grid = prop.multiProcessorCount * 32;

    struct Variant { const char* name; void (*k)(unsigned long long*, long); };
    Variant variants[] = {{"global atomicAdd", count_global},
                          {"warp-aggregated", count_warp_agg},
                          {"block-privatized", count_block_priv},
                          {"cuda::atomic_ref", count_scoped}};

    for (auto& v : variants) {
        float ms = cuda_time_best_ms([&] {
            CUDA_CHECK(cudaMemsetAsync(dc, 0, sizeof(unsigned long long)));
            v.k<<<grid, block>>>(dc, N);
            CUDA_CHECK_LAST();
        }, 5, 1);
        unsigned long long got;
        CUDA_CHECK(cudaMemcpy(&got, dc, sizeof(got), cudaMemcpyDeviceToHost));
        std::printf("%-18s %9.3f ms  count %llu %s\n", v.name, ms, got,
                    got == (unsigned long long)N ? "OK" : "FAIL");
    }
    CUDA_CHECK(cudaFree(dc));
    return 0;
}
