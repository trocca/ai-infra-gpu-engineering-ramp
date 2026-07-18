// Lab 03/cpp — SAXPY: serial, auto-vectorized, and OpenMP variants.
// Lesson: ../README.md — §Concept (two-level CPU parallelism), exercises 1–2.
// Refs:   OpenMP 5.x spec §worksharing-loop; GCC "-fopt-info-vec" docs
//         (verify auto-vectorization); BLAS Level-1 saxpy reference.
// Build:  g++ -O3 -std=c++20 -fopenmp -march=native saxpy.cpp -o saxpy_cpu
//         (add -fopt-info-vec to see what the compiler vectorized)

#include <cstdio>
#include <vector>

#include "../../common/bench.hpp"

constexpr size_t N = 1 << 26;  // 64M floats -> 256 MB per array, beats any L3
constexpr float A = 2.5f;

void saxpy_serial(float a, const float* x, float* y, size_t n) {
    for (size_t i = 0; i < n; ++i) y[i] = a * x[i] + y[i];
}

void saxpy_omp(float a, const float* x, float* y, size_t n) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) y[i] = a * x[i] + y[i];
}

int main() {
    std::vector<float> x(N), y0(N), y(N);
    for (size_t i = 0; i < N; ++i) { x[i] = float(i % 1000); y0[i] = 1.0f; }

    // bytes moved per run: read x, read y, write y
    const double bytes = 3.0 * N * sizeof(float);

    y = y0;
    double ms_serial = bench::time_best_ms([&] { saxpy_serial(A, x.data(), y.data(), N); });
    bench::report("serial (auto-vec)", ms_serial, bytes);

    y = y0;
    double ms_omp = bench::time_best_ms([&] { saxpy_omp(A, x.data(), y.data(), N); });
    bench::report("openmp", ms_omp, bytes);

    // Verify: one clean run of each variant from known state, compare a prefix.
    std::vector<float> ya = y0, yb = y0;
    saxpy_serial(A, x.data(), ya.data(), N);
    saxpy_omp(A, x.data(), yb.data(), N);
    std::vector<float> pa(ya.begin(), ya.begin() + 4096), pb(yb.begin(), yb.begin() + 4096);
    std::printf("verify: %s\n", bench::almost_equal(pa, pb) ? "OK" : "FAIL");
    return 0;
}
