// Lab 02/cpp — experiment 1 (stride sweep) and experiment 2 (AoS vs SoA).
// Lesson: ../README.md — §Experiments 1–2, CPU column of the table.
// Refs:   Drepper, "What Every Programmer Should Know About Memory" §3 (caches),
//         §6.2 (prefetching); Agner Fog, "Optimizing software in C++" §9.
// Build:  g++ -O3 -std=c++20 -march=native stride_sweep.cpp -o stride_cpu

#include <cstdio>
#include <vector>

#include "../../common/bench.hpp"

constexpr size_t N = 1 << 26;  // 64M floats, 256 MB

// Same number of reads for every stride; only the pattern changes.
float sum_strided(const float* x, size_t n, size_t stride) {
    float s = 0.f;
    for (size_t start = 0; start < stride; ++start)
        for (size_t i = start; i < n; i += stride) s += x[i];
    return s;
}

struct Particle { float x, y, z, w; };  // AoS: one field wastes 3/4 of each line

int main() {
    std::vector<float> a(N, 1.0f);
    const double bytes = double(N) * sizeof(float);
    volatile float sink;

    std::printf("== stride sweep (N=%zu floats) ==\n", N);
    for (size_t stride : {1, 2, 4, 8, 16, 64, 256, 1024, 4096}) {
        double ms = bench::time_best_ms([&] { sink = sum_strided(a.data(), N, stride); }, 5, 1);
        std::printf("stride %5zu: %9.3f ms  %7.1f GB/s (effective)\n", stride, ms,
                    bytes / ms / 1e6);
    }

    std::printf("\n== AoS vs SoA: sum one field of 4 ==\n");
    const size_t M = N / 4;
    std::vector<Particle> aos(M, {1, 2, 3, 4});
    std::vector<float> soa_x(M, 1.0f);

    double ms = bench::time_best_ms([&] {
        float s = 0.f;
        for (size_t i = 0; i < M; ++i) s += aos[i].x;
        sink = s;
    });
    bench::report("AoS (.x of xyzw)", ms, double(M) * sizeof(Particle));  // lines fetched

    ms = bench::time_best_ms([&] {
        float s = 0.f;
        for (size_t i = 0; i < M; ++i) s += soa_x[i];
        sink = s;
    });
    bench::report("SoA (x array)", ms, double(M) * sizeof(float));
    std::printf("(AoS moves 4x the bytes for the same answer)\n");
    (void)sink;
    return 0;
}
