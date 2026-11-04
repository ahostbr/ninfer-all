#pragma once

// The kernel-level A/B switches of DeepSeek-V4.1 (T0833). Each one is a flag of its own, default off,
// whose startup line says ON or off, and each exists to be measured against its absence on the GPU:
// whatever does not win is stripped afterwards.
//
//   --deepseek-iq1m-matrix  on|off  NINFER_DS_IQ1M_MATRIX   a wide IQ1_M MoE product for prompts: the
//                                   integer tensor-core kernel (IQ1_M's tile loader in ggml-quants)
//                                   instead of the 8-column vector kernel, on the experts of a call
//                                   of more than 8 tokens.
//   --deepseek-iq1m-codes12 on|off  NINFER_DS_IQ1M_CODES12  the decode and verification kernels read
//                                   IQ1_M weights through a 12-bit table of ready int8 words (grid
//                                   code and delta sign in, eight weights out) instead of expanding
//                                   the 11-bit grid word per group.
//   --deepseek-fused-swiglu on|off  NINFER_DS_FUSED_SWIGLU  the routed experts' gate and up rows of a
//                                   decode or verification call in one kernel that also applies the
//                                   clamped SwiGLU, in place of the product, the activation kernel
//                                   and the memset between them.
//   --prefill-cublas                NINFER_DS_PREFILL_CUBLAS wide dense prompt GEMMs (attention and
//                                   shared-expert projections) dequantize to BF16 and run through
//                                   cuBLAS in place of the integer tensor-core kernel (a quality
//                                   trade: weights round to BF16).
//
// An environment value is 1, on or true; 0, off or false (any case); anything else throws, so a
// misspelt bench script fails at startup instead of measuring the wrong arm. It overrides the flag.

#include <cctype>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::models::deepseek_v41 {

struct KernelSwitches {
    bool iq1m_matrix  = false;
    bool iq1m_codes12 = false;
    bool fused_swiglu = false;
    bool dense_cublas = false;
};

// 1/on/true -> true, 0/off/false -> false (ASCII case folded), anything else -> nullopt.
[[nodiscard]] inline std::optional<bool> parse_switch(std::string_view text) {
    std::string lower;
    for (const char c : text) { lower.push_back(char(std::tolower(static_cast<unsigned char>(c)))); }
    if (lower == "1" || lower == "on" || lower == "true") { return true; }
    if (lower == "0" || lower == "off" || lower == "false") { return false; }
    return std::nullopt;
}

// The value `env` (a variable's text, null or empty for unset) gives the switch `name`, or `option`
// when it is unset; throws std::invalid_argument on a value that is neither.
[[nodiscard]] inline bool resolve_switch(const char* env, const char* name, bool option) {
    if (env == nullptr || *env == '\0') { return option; }
    const std::optional<bool> parsed = parse_switch(env);
    if (!parsed) {
        throw std::invalid_argument(std::string(name) + "=" + env +
                                    " is not 1, on, true, 0, off or false");
    }
    return *parsed;
}

[[nodiscard]] inline bool resolve_switch_env(const char* name, bool option) {
    return resolve_switch(std::getenv(name), name, option);
}

// The switches the options and the environment ask for.
[[nodiscard]] inline KernelSwitches resolve_kernel_switches(const KernelSwitches& options) {
    KernelSwitches out;
    out.iq1m_matrix  = resolve_switch_env("NINFER_DS_IQ1M_MATRIX", options.iq1m_matrix);
    out.iq1m_codes12 = resolve_switch_env("NINFER_DS_IQ1M_CODES12", options.iq1m_codes12);
    out.fused_swiglu = resolve_switch_env("NINFER_DS_FUSED_SWIGLU", options.fused_swiglu);
    out.dense_cublas = resolve_switch_env("NINFER_DS_PREFILL_CUBLAS", options.dense_cublas);
    return out;
}

} // namespace ninfer::models::deepseek_v41
