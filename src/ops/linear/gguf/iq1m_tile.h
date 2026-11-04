#pragma once

// IQ1_M's tile loader for the integer tensor-core matrix kernel (third_party/ggml-quants
// mmq-load-tiles.cuh, ggml_cuda_mmq_load_tiles_iq1_m), as plain functions that the device code and
// a host test (tests/ops/test_iq1m_tile.cpp) both run.
//
// llama.cpp has no matrix kernel for IQ1_M. Its value is d (2 s + 1) (g + delta): g in -1..1 from
// the shared IQ1 grid, delta = +-1/8 per group of eight values, s a 3-bit scale per 16 values and d
// a binary16 spread over the four scale words' top nibbles. Eight times g + delta is the integer
// 8 c - 7 (delta +1/8) or 8 c - 9 (delta -1/8) for the grid's code c = g + 1 in 0..2, at most 9 in
// magnitude, so the tile holds those as int8 values and one FP32 factor d (2 s + 1) / 8 per 16
// values, both exact. That is the layout IQ2_XS and IQ2_S already use (an int8 tile with a scale
// per 16 values), so their vec-dot serves, and the activation needs no per-32 sums, unlike IQ1_S.
// The vector kernel's Decoder<IQ1_M> (ggml_bridge_vec.cuh) writes the same integers and factors.

#include <cstdint>

#if defined(__CUDACC__)
#define NINFER_IQ1M_FN __host__ __device__ __forceinline__
#else
#define NINFER_IQ1M_FN inline
#endif

namespace ninfer::ops::gguf::iq1m {

// A block's stored size: qs[32], qh[16], scales[8] (four 16-bit words).
inline constexpr int kBlockBytes = 56;
inline constexpr int kQsOffset = 0, kQhOffset = 32, kScalesOffset = 48;

// Four grid codes (0..2, one per byte) as the four int8 values 8 c - 7, or 8 c - 9 with `minus`,
// packed in a word. Byte lanes never carry into each other: 8 c is at most 16.
NINFER_IQ1M_FN std::uint32_t weights4(std::uint32_t codes, bool minus) {
    const std::uint32_t a = (codes & 0x03030303u) << 3;
    const std::uint32_t b = minus ? 0xF7F7F7F7u : 0xF9F9F9F9u; // -9 or -7 in every byte
    return ((a & 0x7F7F7F7Fu) + (b & 0x7F7F7F7Fu)) ^ ((a ^ b) & 0x80808080u);
}

// The binary16 bits of a block's d, from its four scale words.
NINFER_IQ1M_FN std::uint16_t super_scale_bits(std::uint16_t s0, std::uint16_t s1, std::uint16_t s2,
                                              std::uint16_t s3) {
    return static_cast<std::uint16_t>((s0 >> 12) | ((s1 >> 8) & 0x00F0) | ((s2 >> 4) & 0x0F00) |
                                      (s3 & 0xF000));
}

// The load of one thread of the tile loader: the 32 values [32 kqsx, 32 kqsx + 32) of a block as
// eight int8 words w[0..7] (value j of group l at byte j of w[2 l] for j < 4, of w[2 l + 1] for
// j >= 4) and the factors f[0..1] of its two 16-value halves, given d as a float. `grid` is the
// IQ1 grid with eight 4-bit codes per word (ggml's iq1s_grid_gpu: byte b holds value b in its low
// nibble and value b + 4 in its high one), `block` the 56 stored bytes.
template <typename Grid>
NINFER_IQ1M_FN void load_slice(const Grid* grid, const std::uint8_t* block, int kqsx, float d,
                               std::uint16_t scale_word, std::uint32_t* w, float* f) {
    const std::uint8_t* qs = block + kQsOffset + 4 * kqsx;
    const std::uint8_t* qh = block + kQhOffset + 2 * kqsx;
    for (int l = 0; l < 4; ++l) {
        const std::uint32_t qhl = std::uint32_t(qh[l / 2]) >> (4 * (l % 2));
        const std::uint32_t g   = grid[std::uint32_t(qs[l]) | ((qhl & 7u) << 8)];
        const bool minus        = (qhl & 8u) != 0;
        w[2 * l]                = weights4(g & 0x0F0F0F0Fu, minus);
        w[2 * l + 1]            = weights4((g >> 4) & 0x0F0F0F0Fu, minus);
    }
    const unsigned tmp = scale_word >> (6 * (kqsx % 2));
    f[0]               = d * float(2 * int(tmp & 7u) + 1) * 0.125f;
    f[1]               = d * float(2 * int((tmp >> 3) & 7u) + 1) * 0.125f;
}

} // namespace ninfer::ops::gguf::iq1m

#undef NINFER_IQ1M_FN
