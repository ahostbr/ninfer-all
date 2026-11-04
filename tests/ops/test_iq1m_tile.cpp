// IQ1_M's integer matrix-kernel tile (src/ops/linear/gguf/iq1m_tile.h, the loader of
// third_party/ggml-quants mmq-load-tiles.cuh) and its 12-bit code table (ggml_bridge_iq1m_codes.inc,
// made by gen_iq1m_codes12.py), on the CPU with no device:
//  - the packing of the GPU's grid words (iq1s_grid_gpu: byte b holds value b in its low nibble and
//    value b + 4 in its high one) from the CPU grid (iq1s_grid), against literal words of the GPU
//    table;
//  - weights4 over every combination of codes 0..2, both delta signs, against 8 c - 7 / 8 c - 9;
//  - every one of the table's 4096 entries against 8 g + 1 / 8 g - 1 straight from the CPU grid
//    (g in -1..1 plus the delta 1/8 or -1/8, times 8), and against weights4 of the GPU word;
//  - load_slice over random blocks: the tile's integer values and factors, dotted with random int8
//    activations, equal ggml's dequantize_row_iq1_m values (decoded here in FP64) dotted with
//    them, exactly: the tile is the same numbers, not an approximation of them. They are also the
//    values and factors of the CPU expert products' decoder (gguf_expert_cpu decode_slice), the
//    mirror of the vector kernel's Decoder<IQ1_M>.

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include "ops/linear/gguf/iq1m_tile.h"
#include "ops/moe_experts/gguf_expert_cpu.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

namespace {

namespace iq = ninfer::ops::gguf::iq1m;

constexpr std::uint32_t kCodes12[] = {
#include "ops/linear/gguf/ggml_bridge_iq1m_codes.inc"
};
static_assert(sizeof(kCodes12) / sizeof(kCodes12[0]) == 8192, "4096 entries of two words");

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::cerr << "FAIL " << what << '\n';
        ++failures;
    }
}

// iq1s_grid_gpu[i] from the CPU grid: byte b is (value b + 1) | (value b + 4 + 1) << 4.
std::uint32_t gpu_word(int index) {
    const std::uint64_t grid = iq1s_grid[index];
    std::uint32_t word       = 0;
    for (int b = 0; b < 4; ++b) {
        const int low  = int(std::int8_t((grid >> (8 * b)) & 0xFF)) + 1;
        const int high = int(std::int8_t((grid >> (8 * (b + 4))) & 0xFF)) + 1;
        word |= std::uint32_t(low | (high << 4)) << (8 * b);
    }
    return word;
}

int byte_of(std::uint32_t word, int b) { return int(std::int8_t((word >> (8 * b)) & 0xFF)); }

double exact_half(unsigned bits) {
    const unsigned exponent = (bits >> 10) & 31, fraction = bits & 1023;
    const double value = exponent ? std::ldexp(1024.0 + fraction, int(exponent) - 25)
                                  : std::ldexp(double(fraction), -24);
    return bits & 0x8000 ? -value : value;
}

// ggml-quants.c's dequantize_row_iq1_m for one block, in FP64.
void reference_block(const std::uint8_t* bytes, double* y) {
    std::uint16_t sc[4];
    std::memcpy(sc, bytes + iq::kScalesOffset, 8);
    const double d = exact_half(iq::super_scale_bits(sc[0], sc[1], sc[2], sc[3]));
    const std::uint8_t* qs = bytes + iq::kQsOffset;
    const std::uint8_t* qh = bytes + iq::kQhOffset;
    for (int ib = 0; ib < QK_K / 32; ++ib) {
        const double dl1 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
        const double dl2 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
        const int idx[4] = {qs[0] | ((qh[0] << 8) & 0x700), qs[1] | ((qh[0] << 4) & 0x700),
                            qs[2] | ((qh[1] << 8) & 0x700), qs[3] | ((qh[1] << 4) & 0x700)};
        const double delta[4] = {qh[0] & 0x08 ? -0.125 : 0.125, qh[0] & 0x80 ? -0.125 : 0.125,
                                 qh[1] & 0x08 ? -0.125 : 0.125, qh[1] & 0x80 ? -0.125 : 0.125};
        for (int l = 0; l < 4; ++l) {
            const auto* grid = reinterpret_cast<const std::int8_t*>(iq1s_grid + idx[l]);
            for (int j = 0; j < 8; ++j) { y[j] = (l < 2 ? dl1 : dl2) * (grid[j] + delta[l]); }
            y += 8;
        }
        qs += 4;
        qh += 2;
    }
}

void test_packing(const std::vector<std::uint32_t>& gpu) {
    // Words of the real iq1s_grid_gpu (ggml-common.h): the first eight, and others.
    const struct {
        int index;
        std::uint32_t word;
    } known[] = {{0, 0x0},       {1, 0x2},       {2, 0x101},      {3, 0x200},
                 {4, 0x202},     {5, 0x10001},   {6, 0x10101},    {7, 0x20000},
                 {100, 0x1011111}, {1000, 0x10101110}, {2046, 0x22222220}, {2047, 0x22222222}};
    for (const auto& k : known) {
        check(gpu[std::size_t(k.index)] == k.word, "gpu grid word packing differs from the header's");
    }
}

void test_weights4() {
    for (std::uint32_t code = 0; code < 81; ++code) {
        std::uint32_t codes = 0;
        int rest            = int(code);
        int c[4];
        for (int b = 0; b < 4; ++b) {
            c[b] = rest % 3;
            rest /= 3;
            codes |= std::uint32_t(c[b]) << (8 * b);
        }
        for (const bool minus : {false, true}) {
            const std::uint32_t w = iq::weights4(codes, minus);
            for (int b = 0; b < 4; ++b) {
                check(byte_of(w, b) == 8 * c[b] - (minus ? 9 : 7), "weights4 is not 8 c - 7 / 8 c - 9");
            }
        }
    }
}

void test_table(const std::vector<std::uint32_t>& gpu) {
    for (int e = 0; e < 4096; ++e) {
        const int index = e & 2047;
        const bool minus = (e >> 11) != 0;
        const auto* grid = reinterpret_cast<const std::int8_t*>(iq1s_grid + index);
        for (int j = 0; j < 8; ++j) {
            // 8 (g + delta), g in -1..1, delta -1/8 (minus) or 1/8.
            const int expect = 8 * int(grid[j]) + (minus ? -1 : 1);
            const int got    = byte_of(kCodes12[2 * e + j / 4], j % 4);
            check(got == expect, "code table entry differs from 8 g +- 1 of the CPU grid");
        }
        check(kCodes12[2 * e] == iq::weights4(gpu[std::size_t(index)] & 0x0F0F0F0Fu, minus) &&
                  kCodes12[2 * e + 1] ==
                      iq::weights4((gpu[std::size_t(index)] >> 4) & 0x0F0F0F0Fu, minus),
              "code table entry differs from weights4 of the GPU grid word");
    }
}

void test_slices(const std::vector<std::uint32_t>& gpu) {
    std::mt19937 rng(20261104);
    std::uniform_int_distribution<int> byte(0, 255), activation(-127, 127), exponent(0x0400, 0x5800);
    for (int trial = 0; trial < 400; ++trial) {
        std::uint8_t block[iq::kBlockBytes];
        for (auto& b : block) { b = std::uint8_t(byte(rng)); }
        // A finite d spread over the four scale words' top nibbles.
        const unsigned d_bits = unsigned(exponent(rng)) | (trial % 2 ? 0x8000u : 0u);
        std::uint16_t sc[4];
        std::memcpy(sc, block + iq::kScalesOffset, 8);
        for (int j = 0; j < 4; ++j) {
            sc[j] = std::uint16_t((sc[j] & 0x0FFF) | (((d_bits >> (4 * j)) & 0xF) << 12));
        }
        std::memcpy(block + iq::kScalesOffset, sc, 8);
        check(iq::super_scale_bits(sc[0], sc[1], sc[2], sc[3]) == d_bits, "super_scale_bits");

        double reference[256];
        reference_block(block, reference);
        const float d = float(exact_half(d_bits));
        for (int kqsx = 0; kqsx < 8; ++kqsx) {
            std::uint32_t w[8];
            float f[2];
            iq::load_slice(gpu.data(), block, kqsx, d, sc[kqsx / 2], w, f);
            int y[32];
            for (int& v : y) { v = activation(rng); }
            double tile = 0, truth = 0;
            for (int h = 0; h < 2; ++h) {
                long long dot = 0;
                for (int v = h * 16; v < h * 16 + 16; ++v) {
                    const int weight = byte_of(w[2 * (v / 8) + (v % 8) / 4], v % 4);
                    check(weight >= -9 && weight <= 9, "tile value past 9 in magnitude");
                    dot += (long long)weight * y[v];
                }
                tile += double(f[h]) * double(dot);
            }
            for (int v = 0; v < 32; ++v) { truth += reference[32 * kqsx + v] * y[v]; }
            check(tile == truth, "tile dot differs from the dequantized values' dot");
            ninfer::ops::gguf_cpu_detail::GgufCpuSlice cpu;
            ninfer::ops::gguf_cpu_detail::decode_slice(ninfer::QType::GGUF_IQ1_M, block, kqsx, cpu);
            bool same = cpu.factor[0] == f[0] && cpu.factor[1] == f[1];
            for (int v = 0; v < 32; ++v) {
                same = same && int(cpu.weights[v]) == byte_of(w[2 * (v / 8) + (v % 8) / 4], v % 4);
            }
            check(same, "tile differs from the CPU expert decoder's weights or factors");
        }
    }
}

} // namespace

int main() {
    std::vector<std::uint32_t> gpu(2048);
    for (int i = 0; i < 2048; ++i) { gpu[std::size_t(i)] = gpu_word(i); }
    test_packing(gpu);
    test_weights4();
    test_table(gpu);
    test_slices(gpu);
    if (failures == 0) { std::cout << "iq1m tile: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}
