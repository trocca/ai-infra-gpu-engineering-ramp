// Lab 08/cpp — the racy counter, its fixes, and what each fix costs:
// plain int (UB), seq_cst atomic, relaxed atomic, sharded counters.
// Lesson: ../README.md — §Concept (C++ memory model), exercises 1–2.
// Refs:   cppreference std::memory_order; Williams, "C++ Concurrency in
//         Action" ch.5 (the memory model chapter); ThreadSanitizer docs
//         (rebuild with -fsanitize=thread -O1 -g to see the race reported).
// Build:  g++ -O3 -std=c++20 -pthread race_and_fix.cpp -o race_cpu

#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

#include "../../common/bench.hpp"

constexpr int THREADS = 8;
constexpr long PER_THREAD = 5'000'000;
constexpr long EXPECT = THREADS * PER_THREAD;

template <typename F>
double run_threads(F&& body) {
    bench::Timer t;
    std::vector<std::thread> pool;
    for (int i = 0; i < THREADS; ++i) pool.emplace_back(body, i);
    for (auto& th : pool) th.join();
    return t.ms();
}

int main() {
    // 1. The race. A data race is undefined behavior — the wrong count is the
    //    *friendly* outcome; the standard permits anything.
    long racy = 0;
    double ms = run_threads([&](int) {
        for (long k = 0; k < PER_THREAD; ++k) ++racy;
    });
    std::printf("racy int:        %9.3f ms  count %ld / %ld  %s\n", ms, racy, EXPECT,
                racy == EXPECT ? "(got lucky - rerun)" : "(lost updates)");

    // 2. seq_cst atomic: correct, one cache line ping-ponging between 8 cores.
    std::atomic<long> seq{0};
    ms = run_threads([&](int) {
        for (long k = 0; k < PER_THREAD; ++k) seq.fetch_add(1, std::memory_order_seq_cst);
    });
    std::printf("atomic seq_cst:  %9.3f ms  count %ld\n", ms, seq.load());

    // 3. relaxed: same coherence traffic, no ordering fences. On x86 the gap
    //    vs seq_cst is small (strong hardware model); on ARM it's visible.
    std::atomic<long> rel{0};
    ms = run_threads([&](int) {
        for (long k = 0; k < PER_THREAD; ++k) rel.fetch_add(1, std::memory_order_relaxed);
    });
    std::printf("atomic relaxed:  %9.3f ms  count %ld\n", ms, rel.load());

    // 4. Sharded: privatize (one padded counter per thread), combine at the
    //    end — the same cure as the histogram in module 05, and the same idea
    //    as the GPU's one-atomic-per-block funnel in module 04.
    struct alignas(64) Padded { long v = 0; };  // 64 = cache line: no false sharing
    std::vector<Padded> shards(THREADS);
    ms = run_threads([&](int i) {
        for (long k = 0; k < PER_THREAD; ++k) ++shards[i].v;
    });
    long total = 0;
    for (auto& s : shards) total += s.v;
    std::printf("sharded+padded:  %9.3f ms  count %ld  <- privatize-then-combine\n",
                ms, total);
    std::printf("\nexercise: remove alignas(64) and measure false sharing kick in\n");
    return 0;
}
