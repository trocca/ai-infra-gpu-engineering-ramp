// Lab 03/cuda — SAXPY with the grid-stride idiom, timed with events,
// reported as achieved bandwidth.
// Lesson: ../README.md — §Concept (SIMT, grid-stride), exercises 1–4.
// Refs:   NVIDIA blog "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride
//         Loops"; CUDA C++ Best Practices Guide §"Effective Bandwidth".
// Build:  nvcc -O3 -arch=native saxpy.cu -o saxpy_gpu

#include <cstdio>
#include <vector>

#include "../../common/cuda_check.cuh"

constexpr size_t N = 1 << 26;  // 64M floats
constexpr float A = 2.5f;

// The production idiom: any grid size handles any n, access stays coalesced
// because consecutive threads touch consecutive elements on every pass.
__global__ void saxpy(float a, const float* __restrict__ x,
                      float* __restrict__ y, size_t n) {
    size_t stride = size_t(gridDim.x) * blockDim.x;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
        y[i] = a * x[i] + y[i];
}

int main() {
    std::vector<float> hx(N), hy(N);
    for (size_t i = 0; i < N; ++i) { hx[i] = float(i % 1000); hy[i] = 1.0f; }

    float *dx, *dy;
    CUDA_CHECK(cudaMalloc(&dx, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, hy.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Size the grid to saturate the machine, not to match N.
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int block = 256;
    const int grid = prop.multiProcessorCount * 32;

    float ms = cuda_time_best_ms([&] {
        saxpy<<<grid, block>>>(A, dx, dy, N);
        CUDA_CHECK_LAST();
    });

    const double bytes = 3.0 * N * sizeof(float);
    std::printf("saxpy grid-stride: %.3f ms, %.1f GB/s\n", ms, bytes / ms / 1e6);
    std::printf("exercise 1: look up %s's peak memory bandwidth and compute %% of peak\n",
                prop.name);

    // Verify with one clean run from known state.
    CUDA_CHECK(cudaMemcpy(dy, hy.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    saxpy<<<grid, block>>>(A, dx, dy, N);
    CUDA_CHECK_LAST();
    float probe;
    CUDA_CHECK(cudaMemcpy(&probe, dy + 3, sizeof(float), cudaMemcpyDeviceToHost));
    const float expect = A * hx[3] + hy[3];
    std::printf("verify: y[3]=%g expect %g -> %s\n", probe, expect,
                probe == expect ? "OK" : "FAIL");

    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
    return 0;
}
