// Lab 12/csrc — capstone: fused softmax, CPU implementation + operator
// registration. The dispatcher routes torch.ops.ramp.fused_softmax(x) here
// for CPU tensors and to softmax.cu's impl for CUDA tensors — the same
// mechanism ATen uses for every builtin op (see the "PyTorch architecture"
// layer cake: this file is you standing at layer 3).
// Lesson: ../README.md — plan steps 1–2; backward (step 4) is intentionally
//         left as the exercise.
// Refs:   PyTorch "Custom C++ and CUDA Extensions" tutorial; "PyTorch Custom
//         Operators Manual" (TORCH_LIBRARY / TORCH_LIBRARY_IMPL);
//         at::parallel_for docs (aten/src/ATen/Parallel.h).
// Build:  loaded JIT by ../lab.py via torch.utils.cpp_extension.load().

#include <torch/extension.h>

#include <cmath>

// One pass per row: max, then exp+sum, then normalize. Three sweeps over a
// row that stays L1/L2-resident — already "fused" relative to eager PyTorch,
// which would materialize intermediate tensors in DRAM for each step.
at::Tensor fused_softmax_cpu(const at::Tensor& input) {
    TORCH_CHECK(input.dim() == 2, "expected 2D [rows, cols], got ", input.dim(), "D");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "fp32 only in the scaffold");
    auto x = input.contiguous();
    auto out = at::empty_like(x);
    const int64_t rows = x.size(0), cols = x.size(1);
    const float* in_p = x.const_data_ptr<float>();
    float* out_p = out.mutable_data_ptr<float>();

    // at::parallel_for, not raw OpenMP: it shares PyTorch's intra-op thread
    // pool. A private omp parallel here would oversubscribe cores the moment
    // this op runs inside a real model (classic extension bug).
    at::parallel_for(0, rows, /*grain_size=*/1, [&](int64_t begin, int64_t end) {
        for (int64_t r = begin; r < end; ++r) {
            const float* row = in_p + r * cols;
            float* orow = out_p + r * cols;
            float mx = row[0];
            for (int64_t c = 1; c < cols; ++c) mx = std::max(mx, row[c]);
            float sum = 0.f;
            for (int64_t c = 0; c < cols; ++c) {
                orow[c] = std::exp(row[c] - mx);  // shift by max: no overflow
                sum += orow[c];
            }
            float inv = 1.f / sum;
            for (int64_t c = 0; c < cols; ++c) orow[c] *= inv;
        }
    });
    return out;
}

// Declared in softmax.cu, compiled by nvcc, linked into the same extension.
// Guarded so a CPU-only build (lab.py omits the .cu and the define) links.
#ifdef WITH_CUDA
at::Tensor fused_softmax_cuda(const at::Tensor& input);
#endif

// Schema first (device-agnostic), then one impl per dispatch key. Calling
// torch.ops.ramp.fused_softmax on a CUDA tensor never touches the CPU path —
// routing is the dispatcher's job, not Python's.
TORCH_LIBRARY(ramp, m) {
    m.def("fused_softmax(Tensor input) -> Tensor");
}
TORCH_LIBRARY_IMPL(ramp, CPU, m) {
    m.impl("fused_softmax", fused_softmax_cpu);
}
#ifdef WITH_CUDA
TORCH_LIBRARY_IMPL(ramp, CUDA, m) {
    m.impl("fused_softmax", fused_softmax_cuda);
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}  // ops registered via TORCH_LIBRARY
