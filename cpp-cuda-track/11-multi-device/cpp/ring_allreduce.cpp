// Lab 11/cpp — exercise 2: ring all-reduce by hand, with threads standing in
// for ranks. Phase 1: reduce-scatter (each rank ends owning the global sum of
// one chunk). Phase 2: all-gather (the owned chunks circulate). Total data
// moved per rank: 2*(R-1)/R of the buffer — the number DDP's cost model
// stands on.
// Lesson: ../README.md — §Concept (collectives + cost model), exercise 2;
//         then swap threads for MPI ranks / GPU peer copies.
// Refs:   Patarasuk & Yuan, "Bandwidth Optimal All-reduce Algorithms" (2009);
//         Baidu's ring-allreduce write-up; NCCL docs §"Collective Operations".
// Build:  g++ -O3 -std=c++20 -pthread ring_allreduce.cpp -o ring_cpu

#include <barrier>
#include <cstdio>
#include <thread>
#include <vector>

#include "../../common/bench.hpp"

constexpr int R = 8;                 // "ranks"
constexpr size_t PER = 1 << 21;      // elements per chunk
constexpr size_t NEL = PER * R;      // full buffer per rank

int main() {
    // rank r starts with buffer full of (r+1); after all-reduce every element
    // of every rank must equal 1+2+...+R.
    std::vector<std::vector<float>> buf(R, std::vector<float>(NEL));
    for (int r = 0; r < R; ++r)
        for (size_t i = 0; i < NEL; ++i) buf[r][i] = float(r + 1);
    const float expect = R * (R + 1) / 2.0f;

    std::barrier sync(R);  // stands in for the network's implicit step barrier

    auto rank_body = [&](int r) {
        // Phase 1 — reduce-scatter: in step s, rank r adds its chunk
        // (r - s) into the neighbor's copy... expressed here shared-memory
        // style: r reads chunk c from the left neighbor and accumulates.
        for (int s = 0; s < R - 1; ++s) {
            int left = (r - 1 + R) % R;
            int c = (r - s - 1 + R) % R;         // chunk index circulating to r
            float* mine = buf[r].data() + size_t(c) * PER;
            const float* theirs = buf[left].data() + size_t(c) * PER;
            sync.arrive_and_wait();              // neighbor finished its previous step
            for (size_t i = 0; i < PER; ++i) mine[i] += theirs[i];
            sync.arrive_and_wait();
        }
        // After R-1 steps rank r owns fully-reduced chunk (r+1) % R — the one
        // that made the whole circle and finished its trip here.

        // Phase 2 — all-gather: the reduced chunks circulate R-1 more steps;
        // at step s rank r picks up reduced chunk (r-s) % R from its left
        // neighbor (which owned it fully one step earlier).
        for (int s = 0; s < R - 1; ++s) {
            int left = (r - 1 + R) % R;
            int c = (r - s + R) % R;             // reduced chunk arriving now
            float* mine = buf[r].data() + size_t(c) * PER;
            const float* theirs = buf[left].data() + size_t(c) * PER;
            sync.arrive_and_wait();
            for (size_t i = 0; i < PER; ++i) mine[i] = theirs[i];
            sync.arrive_and_wait();
        }
    };

    bench::Timer t;
    std::vector<std::thread> ranks;
    for (int r = 0; r < R; ++r) ranks.emplace_back(rank_body, r);
    for (auto& th : ranks) th.join();
    double ms = t.ms();

    bool ok = true;
    for (int r = 0; r < R && ok; ++r)
        for (size_t i = 0; i < NEL; ++i)
            if (buf[r][i] != expect) { ok = false; break; }

    const double bytes_per_rank = 2.0 * (R - 1) / R * NEL * sizeof(float);
    std::printf("ring all-reduce, %d ranks x %zu floats: %.3f ms\n", R, NEL, ms);
    std::printf("data moved per rank: %.1f MB (2*(R-1)/R of buffer — the formula)\n",
                bytes_per_rank / 1e6);
    std::printf("verify (every element == %.0f on every rank): %s\n", expect,
                ok ? "OK" : "FAIL");
    return 0;
}
