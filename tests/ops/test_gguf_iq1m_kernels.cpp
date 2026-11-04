// The IQ1_M MoE matrix kernel (--deepseek-iq1m-matrix, src/ops/linear/gguf) at the DeepSeek-V4.1
// geometries' shapes; needs a device (exit 77 without one): against an FP64 product of the exactly
// dequantized weights, within the bound the activation's q8_1 rounding allows, and against the
// 8-column vector kernel it replaces (both within that bound of the truth, so within twice of each
// other); moe_matrix_supported says no unless asked, and yes when asked on a device that has
// integer tensor cores.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/linear/gguf/ggml_bridge.h"

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

void test_matrix(std::mt19937& rng) {
    const int experts = 6, rows = 256, k = 1024, pairs = 48;
    Bank bank(rows, k, experts, 3, rng);
    std::vector<std::int32_t> ids(static_cast<std::size_t>(pairs));
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
        std::vector<float> xf(static_cast<std::size_t>(k));
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
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return failures == 0 ? 0 : 1;
}
