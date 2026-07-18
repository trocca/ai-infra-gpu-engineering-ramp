// Lab 01/cuda — launch a grid, print who ran where, query the machine.
// Lesson: ../README.md — §Concept (GPU side), exercises 1–3.
// Refs:   CUDA C++ Programming Guide §"Programming Model" (grid/block/thread),
//         §"Hardware Implementation" (SIMT, warp scheduling);
//         cudaGetDeviceProperties API docs.
// Build:  nvcc -O3 -arch=native hello_kernel.cu -o hello_gpu

#include <cstdio>

#include "../../common/cuda_check.cuh"

// Undocumented-but-stable: read the id of the SM this thread runs on.
__device__ unsigned my_smid() {
    unsigned smid;
    asm("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

__global__ void hello() {
    // Global thread id: the idiom you will type ten thousand times.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x == 0)  // one line per block, or 10k lines scroll by
        printf("block %d (of %d), first thread, global id %d, on SM %u\n",
               blockIdx.x, gridDim.x, tid, my_smid());
}

__global__ void empty() {}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("%s | CC %d.%d | %d SMs | %d max threads/SM | %d max threads in flight\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                prop.maxThreadsPerMultiProcessor,
                prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor);

    hello<<<8, 256>>>();  // 8 blocks x 256 threads
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());  // printf flushes on sync

    // Exercise 1: launching 10M threads costs ~nothing. Compare with the CPU
    // version spawning 10 000 OS threads.
    const long long n = 10'000'000;
    const int block = 256;
    const int grid = static_cast<int>((n + block - 1) / block);
    float ms = cuda_time_best_ms([&] {
        empty<<<grid, block>>>();
        CUDA_CHECK_LAST();
    });
    std::printf("empty kernel over %lld threads: %.4f ms\n", n, ms);
    return 0;
}
