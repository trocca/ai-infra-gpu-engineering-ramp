// Lab 07/cuda — exercises 1 & 4: chunked H2D/kernel/D2H pipeline across k
// streams (pinned memory), and kernel-launch overhead vs CUDA Graphs.
// Lesson: ../README.md — §Concept (streams/events/graphs), exercises 1–2, 4.
// Refs:   NVIDIA blog "How to Overlap Data Transfers in CUDA C/C++" (Harris);
//         CUDA C++ Programming Guide §"Asynchronous Concurrent Execution",
//         §"CUDA Graphs"; NVIDIA blog "Getting Started with CUDA Graphs".
// Build:  nvcc -O3 -arch=native streams_pipeline.cu -o streams_gpu

#include <cstdio>
#include <vector>

#include "../../common/cuda_check.cuh"

constexpr size_t TOTAL = 1ull << 28;  // 256M floats' worth of bytes? keep bytes: 256 MB
constexpr int MAX_STREAMS = 8;

__global__ void square(float* x, size_t n) {
    size_t gs = size_t(gridDim.x) * blockDim.x;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += gs)
        x[i] = x[i] * x[i];
}

__global__ void tiny() {}

int main() {
    const size_t n = TOTAL / sizeof(float);
    float *h, *d;
    CUDA_CHECK(cudaMallocHost(&h, TOTAL));  // pinned: exercise 1 asks you to
    CUDA_CHECK(cudaMalloc(&d, TOTAL));      // swap for malloc and watch overlap die
    for (size_t i = 0; i < n; ++i) h[i] = 1.5f;

    std::printf("== pipelined H2D + kernel + D2H, 256 MB ==\n");
    for (int ns : {1, 2, 4, 8}) {
        cudaStream_t streams[MAX_STREAMS];
        for (int s = 0; s < ns; ++s) CUDA_CHECK(cudaStreamCreate(&streams[s]));
        const size_t chunk = n / ns;

        float ms = cuda_time_best_ms([&] {
            for (int s = 0; s < ns; ++s) {
                size_t off = chunk * s, bytes = chunk * sizeof(float);
                CUDA_CHECK(cudaMemcpyAsync(d + off, h + off, bytes,
                                           cudaMemcpyHostToDevice, streams[s]));
                square<<<256, 256, 0, streams[s]>>>(d + off, chunk);
                CUDA_CHECK(cudaMemcpyAsync(h + off, d + off, bytes,
                                           cudaMemcpyDeviceToHost, streams[s]));
            }
            for (int s = 0; s < ns; ++s) CUDA_CHECK(cudaStreamSynchronize(streams[s]));
        }, 5, 1);
        std::printf("%d stream(s): %9.3f ms\n", ns, ms);
        for (int s = 0; s < ns; ++s) CUDA_CHECK(cudaStreamDestroy(streams[s]));
    }
    std::printf("(profile this under nsys to *see* the overlap — exercise 2)\n\n");

    std::printf("== launch overhead: loop vs CUDA Graph, 10000 tiny kernels ==\n");
    const int L = 10000;
    float ms = cuda_time_best_ms([&] {
        for (int i = 0; i < L; ++i) tiny<<<1, 32>>>();
        CUDA_CHECK(cudaDeviceSynchronize());
    }, 3, 1);
    std::printf("loop:  %9.3f ms  (%.2f us/launch)\n", ms, ms * 1000 / L);

    cudaStream_t cap;
    CUDA_CHECK(cudaStreamCreate(&cap));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CUDA_CHECK(cudaStreamBeginCapture(cap, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < L; ++i) tiny<<<1, 32, 0, cap>>>();
    CUDA_CHECK(cudaStreamEndCapture(cap, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    ms = cuda_time_best_ms([&] {
        CUDA_CHECK(cudaGraphLaunch(exec, cap));
        CUDA_CHECK(cudaStreamSynchronize(cap));
    }, 3, 1);
    std::printf("graph: %9.3f ms  (%.2f us/launch) — why torch.cuda.CUDAGraph exists\n",
                ms, ms * 1000 / L);

    CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(cap));
    CUDA_CHECK(cudaFreeHost(h));
    CUDA_CHECK(cudaFree(d));
    return 0;
}
