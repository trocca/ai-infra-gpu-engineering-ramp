// Lab 06/cpp — SGEMM: naive vs cache-blocked, OpenMP across rows.
// Lesson: ../README.md — §Concept (cache blocking), exercises 1–3.
// Refs:   Goto & van de Geijn, "Anatomy of High-Performance Matrix
//         Multiplication" (ACM TOMS 2008); the flame/BLIS tutorial
//         "how-to-optimize-gemm"; Drepper §6.2.1 (matrix blocking example).
// Build:  g++ -O3 -std=c++20 -fopenmp -march=native matmul.cpp -o matmul_cpu

#include <cstdio>
#include <vector>

#include "../../common/bench.hpp"

constexpr int N = 1024;      // C[N,N] = A[N,N] * B[N,N]
constexpr int BLK = 64;      // exercise 2: tune against your L1/L2

void matmul_naive(const float* A, const float* B, float* C, int n) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            float acc = 0.f;
            for (int k = 0; k < n; ++k) acc += A[i * n + k] * B[k * n + j];
            C[i * n + j] = acc;
        }
}

// ikj loop order inside blocks: the k-loop broadcast of A[i][k] lets the
// compiler vectorize the j-updates; B is walked row-wise (stride-1).
void matmul_blocked(const float* A, const float* B, float* C, int n) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < n; ii += BLK)
        for (int jj = 0; jj < n; jj += BLK)
            for (int kk = 0; kk < n; kk += BLK)
                for (int i = ii; i < ii + BLK; ++i)
                    for (int k = kk; k < kk + BLK; ++k) {
                        float a = A[i * n + k];
                        for (int j = jj; j < jj + BLK; ++j)
                            C[i * n + j] += a * B[k * n + j];
                    }
}

int main() {
    std::vector<float> A(N * N), B(N * N), C1(N * N, 0.f), C2(N * N, 0.f);
    for (int i = 0; i < N * N; ++i) {
        A[i] = float((i * 7) % 13) * 0.1f;
        B[i] = float((i * 5) % 11) * 0.1f;
    }
    const double flops = 2.0 * N * N * double(N);

    double ms = bench::time_best_ms([&] { matmul_naive(A.data(), B.data(), C1.data(), N); }, 3, 1);
    bench::report("naive (omp rows)", ms, 0, flops);

    ms = bench::time_best_ms(
        [&] {
            std::fill(C2.begin(), C2.end(), 0.f);  // blocked variant accumulates into C
            matmul_blocked(A.data(), B.data(), C2.data(), N);
        },
        3, 1);
    bench::report("blocked 64 (omp)", ms, 0, flops);

    std::printf("verify: %s\n", bench::almost_equal(C1, C2, 1e-3f, 1e-2f) ? "OK" : "FAIL");
    std::printf("exercise 3: add cblas_sgemm here and weep\n");
    return 0;
}
