// Lab 11/cuda — exercise 1: the GPU-pair bandwidth matrix, with and without
// peer access enabled (NVLink/PCIe direct path vs bounce-through-host).
// Run `nvidia-smi topo -m` first and predict the matrix before running.
// Degrades gracefully to a single-GPU D2D measurement on one-GPU boxes.
// Lesson: ../README.md — §Concept (fabric), exercise 1; feeds the exercise-3
//         comparison against nccl-tests numbers.
// Refs:   CUDA C++ Programming Guide §"Peer-to-Peer Memory Access";
//         cudaDeviceEnablePeerAccess / cudaMemcpyPeerAsync API docs;
//         NCCL docs §"NVLink" and nccl-tests README.
// Build:  nvcc -O3 -arch=native p2p_bandwidth.cu -o p2p_gpu

#include <cstdio>

#include "../../common/cuda_check.cuh"

constexpr size_t BYTES = 1ull << 28;  // 256 MB per transfer

int main() {
    int n;
    CUDA_CHECK(cudaGetDeviceCount(&n));
    std::printf("%d GPU(s) visible\n", n);

    if (n == 1) {
        // Fallback: within-device copy, so the lab still teaches on one GPU.
        float ms;
        void *a, *b;
        CUDA_CHECK(cudaMalloc(&a, BYTES));
        CUDA_CHECK(cudaMalloc(&b, BYTES));
        ms = cuda_time_best_ms([&] {
            CUDA_CHECK(cudaMemcpyAsync(b, a, BYTES, cudaMemcpyDeviceToDevice));
        }, 5, 1);
        // D2D touches HBM twice (read + write): report both views.
        std::printf("single-GPU D2D: %.3f ms  %.1f GB/s copied (%.1f GB/s HBM traffic)\n",
                    ms, BYTES / ms / 1e6, 2.0 * BYTES / ms / 1e6);
        return 0;
    }

    // Allocate one buffer per GPU.
    void* buf[16] = {};
    for (int d = 0; d < n; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaMalloc(&buf[d], BYTES));
    }

    for (int enable_peer = 0; enable_peer <= 1; ++enable_peer) {
        if (enable_peer)
            for (int i = 0; i < n; ++i) {
                CUDA_CHECK(cudaSetDevice(i));
                for (int j = 0; j < n; ++j) {
                    int can;
                    if (i == j) continue;
                    CUDA_CHECK(cudaDeviceCanAccessPeer(&can, i, j));
                    if (can) cudaDeviceEnablePeerAccess(j, 0);  // idempotent-ish; ignore already-enabled
                }
            }
        std::printf("\n== pairwise bandwidth, peer access %s (GB/s) ==\n",
                    enable_peer ? "ENABLED" : "disabled (bounces through host)");
        std::printf("  dst→ ");
        for (int j = 0; j < n; ++j) std::printf("%8d", j);
        std::printf("\n");
        for (int i = 0; i < n; ++i) {
            std::printf("src %2d ", i);
            for (int j = 0; j < n; ++j) {
                if (i == j) { std::printf("%8s", "-"); continue; }
                CUDA_CHECK(cudaSetDevice(i));
                float ms = cuda_time_best_ms([&] {
                    CUDA_CHECK(cudaMemcpyPeerAsync(buf[j], j, buf[i], i, BYTES));
                }, 5, 1);
                std::printf("%8.1f", BYTES / ms / 1e6);
            }
            std::printf("\n");
        }
    }
    std::printf("\ncompare against `nvidia-smi topo -m`: NVLink pairs vs PCIe pairs\n");

    for (int d = 0; d < n; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaFree(buf[d]));
    }
    return 0;
}
