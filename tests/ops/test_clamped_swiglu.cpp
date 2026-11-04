// DeepSeek-V4.1's clamped SwiGLU of a pair of dot products (src/ops/common/clamped_swiglu.h, the
// epilogue of the fused routed-expert kernel, --deepseek-fused-swiglu) against the CPU expert
// products' middle (gguf_cpu_clamped_swiglu_bf16), which is bit for bit what
// kernels::clamped_swiglu makes of the same FP32 sums: random pairs over many magnitudes and guard
// factors, the clamp's edges, zeros, and values that round to BF16 on a tie. No device is used.

#include "ops/common/clamped_swiglu.h"
#include "ops/moe_experts/gguf_expert_cpu.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

namespace {

namespace cpu = ninfer::ops;

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::cerr << "FAIL " << what << '\n';
        ++failures;
    }
}

void compare(const std::vector<float>& gate, const std::vector<float>& up, float scale, float limit,
             cpu::GgufCpuBackend backend, const char* label) {
    if (!cpu::gguf_cpu_backend_available(backend)) { return; }
    std::vector<std::uint16_t> expect(gate.size());
    cpu::gguf_cpu_clamped_swiglu_bf16(gate.data(), up.data(), gate.size(), scale, limit,
                                      expect.data(), backend);
    std::size_t wrong = 0;
    for (std::size_t i = 0; i < gate.size(); ++i) {
        const float value = cpu::clamped_swiglu_value(gate[i], up[i], scale, limit);
        if (cpu::gguf_cpu_detail::float_to_bf16(value) != expect[i]) {
            if (++wrong == 1) {
                std::cerr << label << ": i=" << i << " gate=" << gate[i] << " up=" << up[i]
                          << " scale=" << scale << " limit=" << limit << '\n';
            }
        }
    }
    check(wrong == 0, label);
}

} // namespace

int main() {
    std::mt19937 rng(20261104);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> exponent(-8.0f, 8.0f);
    for (const float limit : {10.0f, 7.0f, 1.0f}) {
        for (const float scale : {1.0f, 0.5f, 1.0f / 1024.0f, 3.7f, 1.0e-3f}) {
            std::vector<float> gate, up;
            for (int i = 0; i < 6000; ++i) {
                gate.push_back(normal(rng) * std::pow(2.0f, exponent(rng)));
                up.push_back(normal(rng) * std::pow(2.0f, exponent(rng)));
            }
            // The clamp's edges (before and after the guard factor), zeros, tiny values.
            for (const float edge : {limit, -limit, limit / scale, -limit / scale,
                                     std::nextafter(limit / scale, 1.0e9f),
                                     std::nextafter(limit / scale, -1.0e9f), 0.0f, -0.0f, 1.0e-30f}) {
                gate.push_back(edge);
                up.push_back(edge);
                gate.push_back(edge);
                up.push_back(-edge);
            }
            // BF16 rounding ties: 1 + 2^-8 sits between two BF16 values.
            for (int i = 0; i < 64; ++i) {
                const float tie = (1.0f + (2 * i + 1) * std::ldexp(1.0f, -8)) / scale;
                gate.push_back(tie);
                up.push_back(tie);
            }
            compare(gate, up, scale, limit, cpu::GgufCpuBackend::Scalar, "clamped swiglu, scalar");
            compare(gate, up, scale, limit, cpu::GgufCpuBackend::Avx2, "clamped swiglu, avx2");
        }
    }
    // round_to_bf16 against the CPU's rounding on its own.
    for (int i = 0; i < 20000; ++i) {
        const float v = normal(rng) * std::pow(2.0f, exponent(rng));
        const std::uint32_t word = std::uint32_t(cpu::gguf_cpu_detail::float_to_bf16(v)) << 16;
        float back;
        std::memcpy(&back, &word, sizeof(back));
        check(cpu::round_to_bf16(v) == back, "round_to_bf16 differs from the CPU's BF16 rounding");
    }
    if (failures == 0) { std::cout << "clamped swiglu: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}
