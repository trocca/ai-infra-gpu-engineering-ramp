#pragma once
// Shared CUDA error-checking + timing helpers. Include from any cuda/ example.
// Every CUDA runtime call in this track goes through CUDA_CHECK — silent CUDA
// errors surface as garbage numbers three modules later, never where they happen.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                \
                         cudaGetErrorName(err_), __FILE__, __LINE__,            \
                         cudaGetErrorString(err_));                             \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// Kernel launches don't return errors; they must be polled.
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

// Time a callable with CUDA events (measures GPU time, not launch time).
template <typename F>
float cuda_time_best_ms(F&& fn, int iters = 10, int warmup = 2) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        fn();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (ms < best) best = ms;
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return best;
}
