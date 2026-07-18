// Lab 05/cpp — 3-pass blocked scan (exercise 1) and histogram under
// contention (exercise 2).
// Lesson: ../README.md — §Concept (scan CPU / histogram both), exercises 1–2.
// Refs:   Blelloch, "Prefix Sums and Their Applications" (CMU tech report —
//         the scan bible); cppreference std::inclusive_scan;
//         Intel guide "Avoiding and Identifying False Sharing".
// Build:  g++ -O3 -std=c++20 -fopenmp -march=native scan_histogram.cpp -o scanhist_cpu

#include <atomic>
#include <cstdio>
#include <numeric>
#include <omp.h>
#include <vector>

#include "../../common/bench.hpp"

constexpr size_t N = 1 << 25;  // scan input
constexpr size_t H = 1 << 28;  // histogram input: 256M bytes
constexpr int BINS = 256;

// Pass 1: each chunk scanned locally. Pass 2: serial exclusive scan of chunk
// totals. Pass 3: add chunk offset. Two parallel sweeps + one tiny serial step.
void scan_blocked(const float* x, float* out, size_t n) {
    int T = omp_get_max_threads();
    std::vector<float> chunk_total(T, 0.f);
#pragma omp parallel num_threads(T)
    {
        int t = omp_get_thread_num();
        size_t lo = n * t / T, hi = n * (t + 1) / T;
        float acc = 0.f;
        for (size_t i = lo; i < hi; ++i) { acc += x[i]; out[i] = acc; }
        chunk_total[t] = acc;
#pragma omp barrier
#pragma omp single
        {
            float run = 0.f;
            for (int k = 0; k < T; ++k) { float c = chunk_total[k]; chunk_total[k] = run; run += c; }
        }
        float off = chunk_total[t];
        for (size_t i = lo; i < hi; ++i) out[i] += off;
    }
}

void hist_shared_atomics(const unsigned char* d, size_t n, long* bins) {
    std::vector<std::atomic<long>> a(BINS);
    for (auto& b : a) b.store(0);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) a[d[i]].fetch_add(1, std::memory_order_relaxed);
    for (int b = 0; b < BINS; ++b) bins[b] = a[b].load();
}

void hist_privatized(const unsigned char* d, size_t n, long* bins) {
    int T = omp_get_max_threads();
    std::vector<long> priv(size_t(T) * BINS, 0);
#pragma omp parallel num_threads(T)
    {
        long* mine = priv.data() + size_t(omp_get_thread_num()) * BINS;
#pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) ++mine[d[i]];
    }
    for (int b = 0; b < BINS; ++b) {
        long s = 0;
        for (int t = 0; t < T; ++t) s += priv[size_t(t) * BINS + b];
        bins[b] = s;
    }
}

int main() {
    // --- scan ---
    // Zero-mean input on purpose: with positive data the prefix reaches ~2.5e7
    // and float scans (any order) drift by module-04-style absorption, making
    // "verify" meaningless. Zero-mean keeps every prefix O(1), so float is
    // exact-ish and the check is tight. Try x[i] = (i%7)*0.25f to see the drift.
    std::vector<float> x(N), out(N), ref(N);
    for (size_t i = 0; i < N; ++i) x[i] = float(i % 7) - 3.0f;
    double acc = 0.0;  // reference still accumulates in double
    for (size_t i = 0; i < N; ++i) { acc += x[i]; ref[i] = float(acc); }
    double ms = bench::time_best_ms([&] { scan_blocked(x.data(), out.data(), N); }, 5, 1);
    bench::report("scan 3-pass blocked", ms, 2.0 * N * sizeof(float));
    std::vector<float> p1(out.end() - 512, out.end()), p2(ref.end() - 512, ref.end());
    std::printf("scan verify: %s\n\n", bench::almost_equal(p1, p2, 1e-4f, 1e-2f) ? "OK" : "FAIL");

    // --- histogram: uniform vs degenerate ---
    std::vector<unsigned char> uniform(H), constant(H, 42);
    unsigned s = 12345;
    for (size_t i = 0; i < H; ++i) { s = s * 1664525u + 1013904223u; uniform[i] = s >> 24; }
    long bins_a[BINS], bins_b[BINS];

    for (auto [name, data] : {std::pair{"uniform", uniform.data()},
                              std::pair{"all-same-value", constant.data()}}) {
        ms = bench::time_best_ms([&] { hist_shared_atomics(data, H, bins_a); }, 3, 1);
        std::printf("%-15s shared atomics: %9.3f ms\n", name, ms);
        ms = bench::time_best_ms([&] { hist_privatized(data, H, bins_b); }, 3, 1);
        std::printf("%-15s privatized:     %9.3f ms\n", name, ms);
        bool ok = std::equal(bins_a, bins_a + BINS, bins_b);
        std::printf("%-15s verify: %s\n", name, ok ? "OK" : "FAIL");
    }
    std::printf("(the all-same-value + shared-atomics cell is the lesson)\n");
    return 0;
}
