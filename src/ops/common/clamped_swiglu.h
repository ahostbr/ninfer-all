#pragma once

// DeepSeek-V4.1's clamped SwiGLU of one pair of dot products, for the fused routed-expert kernel
// (ggml_bridge_vec.cuh moe_kernel, --deepseek-fused-swiglu) and its CPU test: what
// kernels::clamped_swiglu (deepseek_v41/kernels.cu) and gguf_cpu_clamped_swiglu_bf16
// (moe_experts/gguf_expert_cpu.cpp) compute from the same two FP32 sums. Gate and up are first
// multiplied by the input's guard factor and rounded to BF16, the gate is clamped above at `limit`
// and the up clamped to [-limit, limit], then silu(gate) * up is the value the caller rounds to
// BF16 (the middle that is quantized for the down product). Host and device give the same bits.

#include "silu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#include <cuda_bf16.h>
#define NINFER_CLAMPED_HD __host__ __device__ __forceinline__
#else
#define NINFER_CLAMPED_HD inline
#endif

namespace ninfer::ops {

// `value` rounded to the nearest BF16 (ties to even) and back.
NINFER_CLAMPED_HD float round_to_bf16(float value) {
#if defined(__CUDA_ARCH__)
    return __bfloat162float(__float2bfloat16(value));
#else
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7FFFFFFFU) > 0x7F800000U) { bits |= 0x00400000U; } // NaN stays NaN
    else { bits += 0x7FFFU + ((bits >> 16) & 1U); }
    bits &= 0xFFFF0000U;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
#endif
}

NINFER_CLAMPED_HD float min_float(float a, float b) {
#if defined(__CUDA_ARCH__)
    return fminf(a, b);
#else
    return std::fmin(a, b);
#endif
}

NINFER_CLAMPED_HD float max_float(float a, float b) {
#if defined(__CUDA_ARCH__)
    return fmaxf(a, b);
#else
    return std::fmax(a, b);
#endif
}

// silu(min(g, limit)) * min(max(u, -limit), limit) with g = BF16(gate * scale), u = BF16(up * scale).
NINFER_CLAMPED_HD float clamped_swiglu_value(float gate, float up, float scale, float limit) {
    const float g = round_to_bf16(gate * scale);
    const float u = round_to_bf16(up * scale);
    return portable_silu(min_float(g, limit)) * min_float(max_float(u, -limit), limit);
}

} // namespace ninfer::ops

#undef NINFER_CLAMPED_HD
