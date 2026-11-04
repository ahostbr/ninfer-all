// The T0833 kernel switches' GPU kernels over IQ1_M expert banks (src/ops/linear/gguf), at the
// DeepSeek-V4.1 geometries' shapes; needs a device (exit 77 without one):
//  1. the IQ1_M MoE matrix kernel (--deepseek-iq1m-matrix) against an FP64 product of the exactly
//     dequantized weights, within the bound the activation's q8_1 rounding allows, and against the
//     8-column vector kernel it replaces (both within that bound of the truth, so within twice of
//     each other); moe_matrix_supported says no unless asked, and yes when asked on a device that
//     has integer tensor cores.
//  2. the 12-bit code table (--deepseek-iq1m-codes12): moe_vector_product with it gives every bit of
//     the plain decoder's, at chunks of 1, 2 and 4 tokens.
//  3. the fused clamped SwiGLU (--deepseek-fused-swiglu): moe_vector_clamped_swiglu equals the bits
//     of moe_vector_product's FP32 rows through the CPU's clamped SwiGLU (the mirror of
//     kernels::clamped_swiglu), with and without the code table.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/moe_experts/gguf_expert_cpu.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace gguf = ninfer::ops::gguf;

int failures = 0;

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void expect(bool ok, const std::string& what) {
    std::cout << (ok ? "OK   " : "FAIL ") << what << '\n';
    failures += ok ? 0 : 1;
}

std::uint16_t half_bits(float value) {
    const __half h = __float2half(value);
    std::uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

// IQ1_M's blocks (56 bytes) with a finite multiplier spread over the scale words' top nibbles.
std::vector<std::uint8_t> random_blocks(int rows, int k, std::mt19937& rng) {
    const std::int64_t count = std::int64_t(rows) * (k / 256);
    std::vector<std::uint8_t> bytes(std::size_t(count) * 56);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(0.002f, 0.02f);
    for (auto& b : bytes) { b = static_cast<std::uint8_t>(byte(rng)); }
    for (std::int64_t i = 0; i < count; ++i) {
        auto* sc = reinterpret_cast<std::uint16_t*>(bytes.data() + std::size_t(i) * 56 + 48);
        const std::uint16_t d = half_bits(scale(rng));
        for (int j = 0; j < 4; ++j) {
            sc[j] = static_cast<std::uint16_t>((sc[j] & 0x0fff) | (((d >> (4 * j)) & 0xf) << 12));
        }
    }
    return bytes;
}

// `distinct` random matrices of rows x k, expert e using matrix e % distinct.
struct Bank {
    int rows, k, experts, distinct;
    std::int64_t row_bytes;
    std::vector<DeviceBuffer> matrices;
    std::vector<std::vector<float>> exact; // [distinct][rows * k]
    DeviceBuffer table;

    Bank(int rows_, int k_, int experts_, int distinct_, std::mt19937& rng)
        : rows(rows_), k(k_), experts(experts_), distinct(distinct_),
          row_bytes(std::int64_t(k_ / 256) * 56), table(std::size_t(experts_) * sizeof(void*)) {
        std::vector<const void*> pointers(std::size_t(experts), nullptr);
        for (int i = 0; i < distinct; ++i) {
            const auto bytes = random_blocks(rows, k, rng);
            matrices.emplace_back(bytes.size() + 256);
            check_cuda(cudaMemset(matrices.back().p, 0, bytes.size() + 256), "zero a matrix");
            matrices.back().copy_from_host(bytes.data(), bytes.size());
            DeviceBuffer values(std::size_t(rows) * k * sizeof(float));
            gguf::dequantize_rows(gguf::GgmlType::IQ1_M, matrices.back().p, row_bytes, k, nullptr,
                                  rows, static_cast<float*>(values.p), k, nullptr);
            check_cuda(cudaDeviceSynchronize(), "dequantize");
            exact.emplace_back(std::size_t(rows) * k);
            values.copy_to_host(exact.back().data(), exact.back().size() * sizeof(float));
        }
        for (int e = 0; e < experts; ++e) { pointers[std::size_t(e)] = matrices[std::size_t(e % distinct)].p; }
        table.copy_from_host(pointers.data(), pointers.size() * sizeof(void*));
    }

    [[nodiscard]] gguf::MoeTable view() const {
        return {static_cast<const void* const*>(table.p), row_bytes, rows, k};
    }
    [[nodiscard]] const float* rows_of(int expert) const {
        return exact[std::size_t(expert % distinct)].data();
    }
};

struct Routes {
    std::vector<std::int32_t> ids;
    DeviceBuffer d_ids, workspace;
    gguf::MoeRouting routing;
    int pairs, experts;

    Routes(std::vector<std::int32_t> ids_, int experts_)
        : ids(std::move(ids_)), d_ids(ids.size() * 4),
          workspace(gguf::moe_routing_bytes(experts_, int(ids.size()))), pairs(int(ids.size())),
          experts(experts_) {
        d_ids.copy_from_host(ids.data(), ids.size() * 4);
        routing = gguf::moe_sort_routes(static_cast<const std::int32_t*>(d_ids.p), pairs, experts,
                                        workspace.p, nullptr);
        check_cuda(cudaDeviceSynchronize(), "route");
    }
};

std::vector<__nv_bfloat16> random_bf16(std::size_t count, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<__nv_bfloat16> out(count);
    for (auto& v : out) { v = __float2bfloat16(normal(rng)); }
    return out;
}

// 1. The matrix kernel.
void test_matrix(std::mt19937& rng) {
    const int experts = 6, rows = 256, k = 1024, pairs = 48;
    Bank bank(rows, k, experts, 3, rng);
    std::vector<std::int32_t> ids(std::size_t(pairs));
    for (auto& id : ids) { id = std::int32_t(rng() % experts); }
    Routes routes(ids, experts);
    const auto x = random_bf16(std::size_t(pairs) * k, rng);
    DeviceBuffer d_x(x.size() * 2);
    d_x.copy_from_host(x.data(), x.size() * 2);
    const auto* xb = static_cast<const __nv_bfloat16*>(d_x.p);

    expect(!gguf::moe_matrix_supported(gguf::GgmlType::IQ1_M, rows, k, false),
           "IQ1_M MoE matrix kernel is not offered unless asked for");
    const bool offered = gguf::moe_matrix_supported(gguf::GgmlType::IQ1_M, rows, k, false, true);
    cudaDeviceProp props{};
    check_cuda(cudaGetDeviceProperties(&props, 0), "device properties");
    expect(offered == (props.major >= 7 && !(props.major == 7 && props.minor == 0)),
           "IQ1_M MoE matrix kernel is offered when asked for on integer tensor cores");
    if (!offered) { return; }

    const int max_active = std::min(pairs, experts);
    DeviceBuffer d_act(gguf::moe_matrix_activation_bytes(k, pairs));
    gguf::quantize_moe_matrix_activation(gguf::GgmlType::IQ1_M, xb, k, routes.routing, pairs, 1,
                                         d_act.p, nullptr);
    DeviceBuffer d_matrix(std::size_t(pairs) * rows * 4);
    gguf::moe_matrix_product(gguf::GgmlType::IQ1_M, bank.view(), routes.routing, max_active, pairs,
                             false, d_act.p, static_cast<float*>(d_matrix.p), nullptr);
    DeviceBuffer d_vact(gguf::vector_activation_bytes(k, pairs));
    gguf::quantize_vector_activation(xb, k, pairs, nullptr, d_vact.p, nullptr);
    DeviceBuffer d_vector(std::size_t(pairs) * rows * 4);
    gguf::MoeOutput out;
    out.f32 = static_cast<float*>(d_vector.p);
    gguf::moe_vector_product(gguf::GgmlType::IQ1_M, bank.view(), routes.routing, max_active, 1, 1,
                             d_vact.p, pairs, out, 8, nullptr);
    check_cuda(cudaDeviceSynchronize(), "products");
    std::vector<float> matrix(std::size_t(pairs) * rows), vector(matrix.size());
    d_matrix.copy_to_host(matrix.data(), matrix.size() * 4);
    d_vector.copy_to_host(vector.data(), vector.size() * 4);

    double worst_truth = 0, worst_pair = 0;
    bool finite = true;
    for (int p = 0; p < pairs; ++p) {
        const float* column = nullptr;
        std::vector<float> xf(std::size_t(k));
        for (int i = 0; i < k; ++i) { xf[std::size_t(i)] = __bfloat162float(x[std::size_t(p) * k + i]); }
        for (int r = 0; r < rows; ++r) {
            column = bank.rows_of(ids[std::size_t(p)]) + std::size_t(r) * k;
            double truth = 0, bound = 0;
            for (int b = 0; b < k; b += 32) {
                double amax = 0, weights = 0;
                for (int i = 0; i < 32; ++i) {
                    amax    = std::max(amax, double(std::fabs(xf[std::size_t(b + i)])));
                    weights += std::fabs(double(column[b + i]));
                    truth += double(column[b + i]) * double(xf[std::size_t(b + i)]);
                }
                bound += weights * amax * (1.0 / 254.0 + 2e-3);
            }
            const double m = matrix[std::size_t(p) * rows + r], v = vector[std::size_t(p) * rows + r];
            finite = finite && std::isfinite(m);
            worst_truth = std::max(worst_truth, std::fabs(m - truth) / std::max(bound, 1e-30));
            worst_pair  = std::max(worst_pair, std::fabs(m - v) / std::max(2.0 * bound, 1e-30));
        }
    }
    expect(finite && worst_truth <= 1.0,
           "IQ1_M MoE matrix kernel within the q8_1 bound of the exact product (worst " +
               std::to_string(worst_truth) + " of the bound)");
    expect(worst_pair <= 1.0, "IQ1_M MoE matrix kernel within twice the bound of the vector kernel (worst " +
                                  std::to_string(worst_pair) + ")");
}

// 2 and 3. The code table and the fused clamped SwiGLU, at the decode widths.
void test_decode_kernels(std::mt19937& rng) {
    const int experts = 8, width = 128, k = 1024, top_k = 3;
    Bank parent(2 * width, k, experts, 4, rng); // [gate; up]
    for (const int tokens : {1, 2, 4}) {
        const int pairs = tokens * top_k;
        std::vector<std::int32_t> ids;
        for (int t = 0; t < tokens; ++t) {
            std::vector<int> pool(std::size_t(experts));
            for (int e = 0; e < experts; ++e) { pool[std::size_t(e)] = e; }
            std::shuffle(pool.begin(), pool.end(), rng);
            for (int j = 0; j < top_k; ++j) { ids.push_back(pool[std::size_t(j)]); }
        }
        Routes routes(ids, experts);
        const auto x = random_bf16(std::size_t(tokens) * k, rng);
        DeviceBuffer d_x(x.size() * 2);
        d_x.copy_from_host(x.data(), x.size() * 2);
        DeviceBuffer d_act(gguf::vector_activation_bytes(k, tokens));
        gguf::quantize_vector_activation(static_cast<const __nv_bfloat16*>(d_x.p), k, tokens, nullptr,
                                         d_act.p, nullptr);
        const int max_active = std::min(pairs, experts);
        const std::string label = " at " + std::to_string(tokens) + " token(s)";

        const auto product = [&](bool codes12) {
            DeviceBuffer d_rows(std::size_t(pairs) * 2 * width * 4);
            check_cuda(cudaMemset(d_rows.p, 0, d_rows.bytes), "zero rows");
            gguf::MoeOutput out;
            out.f32 = static_cast<float*>(d_rows.p);
            gguf::moe_vector_product(gguf::GgmlType::IQ1_M, parent.view(), routes.routing, max_active,
                                     top_k, top_k, d_act.p, tokens, out, tokens, nullptr, codes12);
            check_cuda(cudaDeviceSynchronize(), "vector product");
            std::vector<float> host(std::size_t(pairs) * 2 * width);
            d_rows.copy_to_host(host.data(), host.size() * 4);
            return host;
        };
        const auto plain = product(false);
        const auto coded = product(true);
        expect(std::memcmp(plain.data(), coded.data(), plain.size() * 4) == 0,
               "IQ1_M 12-bit code table gives the plain decoder's bits" + label);

        // The fused kernel against the unfused rows through the CPU's clamped SwiGLU.
        std::uniform_real_distribution<float> factor(0.25f, 1.5f);
        std::vector<float> scale(std::size_t(tokens));
        for (auto& s : scale) { s = factor(rng); }
        DeviceBuffer d_scale(scale.size() * 4);
        d_scale.copy_from_host(scale.data(), scale.size() * 4);
        const float limit = 10.0f;
        for (const bool codes12 : {false, true}) {
            DeviceBuffer d_middle(std::size_t(pairs) * width * 2);
            check_cuda(cudaMemset(d_middle.p, 0, d_middle.bytes), "zero middles");
            gguf::moe_vector_clamped_swiglu(gguf::GgmlType::IQ1_M, parent.view(), routes.routing,
                                            max_active, top_k, d_act.p, tokens,
                                            static_cast<__nv_bfloat16*>(d_middle.p),
                                            static_cast<const float*>(d_scale.p), top_k, limit,
                                            tokens, nullptr, codes12);
            check_cuda(cudaDeviceSynchronize(), "fused swiglu");
            std::vector<std::uint16_t> got(std::size_t(pairs) * width);
            d_middle.copy_to_host(got.data(), got.size() * 2);
            std::vector<std::uint16_t> want(got.size());
            for (int p = 0; p < pairs; ++p) {
                const float* row = plain.data() + std::size_t(p) * 2 * width;
                ops::gguf_cpu_clamped_swiglu_bf16(row, row + width, std::size_t(width),
                                                  scale[std::size_t(p / top_k)], limit,
                                                  want.data() + std::size_t(p) * width,
                                                  ops::GgufCpuBackend::Automatic);
            }
            expect(got == want, std::string("fused clamped SwiGLU equals the unfused rows' ") +
                                    "(codes12 " + (codes12 ? "on" : "off") + ")" + label);
        }
    }
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        std::mt19937 rng(20261104);
        test_matrix(rng);
        test_decode_kernels(rng);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return failures == 0 ? 0 : 1;
}
