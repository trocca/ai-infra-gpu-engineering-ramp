// Lab 07/cpp — exercise 3: overlap "IO" (chunk copy) with compute, serial vs
// double-buffered pipeline with std::async.
// Lesson: ../README.md — §Concept (CPU side), exercise 3.
// Refs:   cppreference std::async / std::future; Williams, "C++ Concurrency
//         in Action" ch.4 (futures) and ch.6 (pipelines/queues).
// Build:  g++ -O3 -std=c++20 -pthread overlap_pipeline.cpp -o overlap_cpu

#include <cstdio>
#include <cstring>
#include <future>
#include <vector>

#include "../../common/bench.hpp"

constexpr size_t TOTAL = 1ull << 28;  // 256 MB
constexpr size_t CHUNK = 1ull << 24;  // 16 MB
constexpr int CHUNKS = int(TOTAL / CHUNK);

// Stage 1, the "transfer": 4 copy rounds simulate a slow source (disk/NIC),
// sized so fetch and process take comparable time — overlap only pays when
// neither stage dwarfs the other (same rule as H2D-vs-kernel on GPU).
void fetch(const char* src, char* dst) {
    for (int round = 0; round < 4; ++round) std::memcpy(dst, src, CHUNK);
}

// Stage 2, the "compute": two passes over the staged chunk.
long process(const char* buf) {
    long acc = 0;
    for (int pass = 0; pass < 2; ++pass)
        for (size_t i = 0; i < CHUNK; ++i) acc += buf[i] * (pass + 1);
    return acc;
}

int main() {
    std::vector<char> src(TOTAL, 3);
    std::vector<char> bufA(CHUNK), bufB(CHUNK);
    volatile long sink;

    // Baselines: each stage alone. Pipelined ideal = max(fetch, process);
    // serial = fetch + process. Print all four and check the arithmetic.
    double ms = bench::time_best_ms([&] {
        for (int k = 0; k < CHUNKS; ++k)
            fetch(src.data() + size_t(k) * CHUNK, bufA.data());
    }, 3, 1);
    std::printf("fetch only:      %9.3f ms\n", ms);
    ms = bench::time_best_ms([&] {
        long acc = 0;
        for (int k = 0; k < CHUNKS; ++k) acc += process(bufA.data());
        sink = acc;
    }, 3, 1);
    std::printf("process only:    %9.3f ms\n", ms);

    // Serial: fetch(k) then process(k), nothing overlaps.
    ms = bench::time_best_ms([&] {
        long acc = 0;
        for (int k = 0; k < CHUNKS; ++k) {
            fetch(src.data() + size_t(k) * CHUNK, bufA.data());
            acc += process(bufA.data());
        }
        sink = acc;
    }, 3, 1);
    std::printf("serial:          %9.3f ms\n", ms);

    // Pipelined: while chunk k is processed, chunk k+1 is fetched into the
    // other buffer. Two stages, two buffers — the CPU version of two streams.
    ms = bench::time_best_ms([&] {
        long acc = 0;
        fetch(src.data(), bufA.data());
        char *cur = bufA.data(), *nxt = bufB.data();
        for (int k = 0; k < CHUNKS; ++k) {
            std::future<void> f;
            if (k + 1 < CHUNKS)
                f = std::async(std::launch::async, fetch,
                               src.data() + size_t(k + 1) * CHUNK, nxt);
            acc += process(cur);
            if (f.valid()) f.get();
            std::swap(cur, nxt);
        }
        sink = acc;
    }, 3, 1);
    std::printf("double-buffered: %9.3f ms  (ideal: max(fetch,process), not sum)\n", ms);
    (void)sink;
    return 0;
}
