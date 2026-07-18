// Lab 01/cpp — spawn N OS threads, observe the cost and the scheduler.
// Lesson: ../README.md — §Concept (CPU side), exercise 1.
// Refs:   cppreference std::thread; man pthread_create (thread creation cost);
//         Butenhof, "Programming with POSIX Threads" ch.1.
// Build:  g++ -O3 -std=c++20 -pthread hello_threads.cpp -o hello_cpu

#include <cstdio>
#include <thread>
#include <vector>

#include "../../common/bench.hpp"

int main(int argc, char** argv) {
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    const int n_threads = argc > 1 ? std::atoi(argv[1]) : hw;

    std::printf("hardware_concurrency: %d, spawning %d threads\n", hw, n_threads);

    bench::Timer t;
    std::vector<std::thread> pool;
    pool.reserve(n_threads);
    std::vector<long> work(n_threads, 0);
    for (int i = 0; i < n_threads; ++i) {
        pool.emplace_back([i, &work] {
            // A little real work so the thread isn't optimized to nothing.
            long acc = 0;
            for (int k = 0; k < 1'000'000; ++k) acc += k ^ i;
            work[i] = acc;
        });
    }
    for (auto& th : pool) th.join();

    std::printf("spawn+work+join: %.3f ms (%.3f us/thread)\n", t.ms(),
                t.ms() * 1000.0 / n_threads);
    std::printf("exercise: rerun with ./hello_cpu 10000 and compare\n");
    return 0;
}
