#pragma once
// Shared timing + validation helpers for the CPU side of the track.
// Header-only; include from any cpp/ example.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace bench {

class Timer {
    std::chrono::steady_clock::time_point start_;
public:
    Timer() : start_(std::chrono::steady_clock::now()) {}
    double ms() const {
        auto d = std::chrono::steady_clock::now() - start_;
        return std::chrono::duration<double, std::milli>(d).count();
    }
};

// Run fn `iters` times after `warmup` untimed runs; report best time.
// Best-of-N (not mean) filters scheduler noise for compute-bound micro-benchmarks.
template <typename F>
double time_best_ms(F&& fn, int iters = 10, int warmup = 2) {
    for (int i = 0; i < warmup; ++i) fn();
    double best = 1e30;
    for (int i = 0; i < iters; ++i) {
        Timer t;
        fn();
        best = std::min(best, t.ms());
    }
    return best;
}

inline bool almost_equal(const std::vector<float>& a, const std::vector<float>& b,
                         float rtol = 1e-4f, float atol = 1e-5f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = std::fabs(a[i] - b[i]);
        if (diff > atol + rtol * std::fabs(b[i])) {
            std::fprintf(stderr, "mismatch at %zu: %g vs %g\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

inline void report(const std::string& name, double ms, double bytes_moved = 0,
                   double flops = 0) {
    std::printf("%-28s %10.3f ms", name.c_str(), ms);
    if (bytes_moved > 0) std::printf("  %8.1f GB/s", bytes_moved / ms / 1e6);
    if (flops > 0)       std::printf("  %8.1f GFLOP/s", flops / ms / 1e6);
    std::printf("\n");
}

} // namespace bench
