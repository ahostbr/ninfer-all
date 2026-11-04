#pragma once

// The kernel-level A/B switch of DeepSeek-V4.1 (T0833): a flag of its own, default off, whose startup
// line says ON or off, which exists to be measured against its absence on the GPU.
//
//   --deepseek-iq1m-matrix  on|off  NINFER_DS_IQ1M_MATRIX   a wide IQ1_M MoE product for prompts: the
//                                   integer tensor-core kernel (IQ1_M's tile loader in ggml-quants)
//                                   instead of the 8-column vector kernel, on the experts of a call
//                                   of more than 8 tokens.
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
    bool iq1m_matrix = false;
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

// The switch the options and the environment ask for.
[[nodiscard]] inline KernelSwitches resolve_kernel_switches(const KernelSwitches& options) {
    KernelSwitches out;
    out.iq1m_matrix = resolve_switch_env("NINFER_DS_IQ1M_MATRIX", options.iq1m_matrix);
    return out;
}

} // namespace ninfer::models::deepseek_v41
