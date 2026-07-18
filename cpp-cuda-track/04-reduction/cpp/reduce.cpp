// Lab 04/cpp — sum 64M floats: serial, OpenMP reduction, and pairwise (accuracy).
// Lesson: ../README.md — §Concept (CPU side) and the float-absorption callout;
//         exercise 3.
// Refs:   Higham, "The accuracy of floating point summation" (SIAM 1993);
//         OpenMP spec §reduction clause; cppreference std::reduce.
// Build:  g++ -O3 -std=c++20 -fopenmp -march=native reduce.cpp -o reduce_cpu

#include <cstdio>
#include <numeric>
#include <vector>

#include "../../common/bench.hpp"

constexpr size_t N = 1 << 26;

float sum_serial(const float* x, size_t n) {
    float s = 0.f;
    for (size_t i = 0; i < n; ++i) s += x[i];  // strict order: no auto-vec without -ffast-math
    return s;
}

float sum_omp(const float* x, size_t n) {
    float s = 0.f;
#pragma omp parallel for reduction(+ : s) schedule(static)
    for (size_t i = 0; i < n; ++i) s += x[i];
    return s;
}

// Pairwise (tree) summation: O(log n) rounding error growth vs O(n) for the
// running sum. This is what NumPy does, and the CPU analogue of the GPU's tree.
float sum_pairwise(const float* x, size_t n) {
    if (n <= 1024) {
        float s = 0.f;
        for (size_t i = 0; i < n; ++i) s += x[i];
        return s;
    }
    size_t half = n / 2;
    return sum_pairwise(x, half) + sum_pairwise(x + half, n - half);
}

int main() {
    std::vector<float> x(N);
    for (size_t i = 0; i < N; ++i) x[i] = 1e-3f * float(i % 97);
    const double bytes = double(N) * sizeof(float);

    volatile float sink;
    double ms;

    ms = bench::time_best_ms([&] { sink = sum_serial(x.data(), N); });
    bench::report("serial (strict order)", ms, bytes);
    std::printf("  value: %.6f\n", sum_serial(x.data(), N));

    ms = bench::time_best_ms([&] { sink = sum_omp(x.data(), N); });
    bench::report("openmp reduction", ms, bytes);
    std::printf("  value: %.6f\n", sum_omp(x.data(), N));

    ms = bench::time_best_ms([&] { sink = sum_pairwise(x.data(), N); });
    bench::report("pairwise (accurate)", ms, bytes);
    std::printf("  value: %.6f  <- compare accuracy vs double reference below\n",
                sum_pairwise(x.data(), N));

    double ref = std::accumulate(x.begin(), x.end(), 0.0);
    std::printf("double reference: %.6f\n", ref);
    (void)sink;
    return 0;
}
