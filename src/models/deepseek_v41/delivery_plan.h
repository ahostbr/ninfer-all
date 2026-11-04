#pragma once

// The arithmetic of DeepSeek-V4.1's delivery experiments (T0833), plain functions checked on the
// CPU (tests/models/deepseek_v41/test_delivery_plan.cpp): which switches are on, the order a
// prompt pass calls its blocks' experts in (for the copies that run ahead of it), and which prompts
// run as decode-sized rows.

#include "core/strict_value.h"
#include "models/deepseek_v41/prompt_skip.h"
#include "ninfer/engine_options.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace ninfer::models::deepseek_v41 {

// ---- Switches ---------------------------------------------------------------------------------

struct DeliveryFlags {
    bool prefetch_ahead = false;
    bool cache_settle   = false;
    bool short_prompt   = false;
    bool profile_weights = false;
    std::uint32_t settle_ms      = 400;
    std::uint32_t settle_idle_ms = 6000;
    std::uint32_t short_prompt_tokens = 96;
};

// The options, each overridden by its NINFER_DS_* variable when that is set (strictly: a value that
// is not 1|on|true or 0|off|false, or not an unsigned integer, throws std::invalid_argument).
[[nodiscard]] inline DeliveryFlags delivery_flags(const HybridExpertOptions& o) {
    DeliveryFlags f;
    f.prefetch_ahead  = env_switch("NINFER_DS_PREFETCH_AHEAD", o.deepseek_prefetch_ahead);
    f.cache_settle    = env_switch("NINFER_DS_CACHE_SETTLE", o.deepseek_cache_settle);
    f.short_prompt    = env_switch("NINFER_DS_SHORT_PROMPT", o.deepseek_short_prompt);
    f.profile_weights = env_switch("NINFER_DS_PROFILE_WEIGHTS", o.deepseek_profile_weights);
    f.settle_ms       = env_count("NINFER_DS_CACHE_SETTLE_MS", o.deepseek_cache_settle_ms);
    f.settle_idle_ms  = env_count("NINFER_DS_CACHE_SETTLE_IDLE_MS", o.deepseek_cache_settle_idle_ms);
    f.short_prompt_tokens =
        env_count("NINFER_DS_SHORT_PROMPT_TOKENS", o.deepseek_short_prompt_tokens);
    if (f.short_prompt_tokens == 0 || f.short_prompt_tokens > 4096) {
        throw std::invalid_argument("DeepSeek-V4.1: the short prompt limit takes 1..4096 tokens");
    }
    return f;
}

// ---- The order of a pass's expert calls -----------------------------------------------------------

// For each block of a prompt pass over `n` rows in chunks of `chunk` (the first `skip` of them
// skipping the blocks past `skip_layer`, prompt_skip.h), the block whose expert call follows the
// block's last one: -1 for a block that makes none (kv-only, or skipped). The pass's last block
// maps to its first, where the next pass begins; a pass with a single such block maps none. The
// blocks the skip leaves out are the ones past `skip_layer` when every row skips, and
// `skip_layer` itself keeps only its kv work for the skipped rows.
[[nodiscard]] inline std::vector<std::int32_t> prefetch_chain(int layers, int skip_layer, int n,
                                                              int chunk, int skip) {
    std::vector<std::int32_t> next(std::size_t(std::max(layers, 0)), -1);
    std::vector<int> calling;
    std::vector<SkipPiece> pieces;
    for (int l = 0; l < layers; ++l) {
        plan_skip_pieces(l, skip_layer, n, chunk, skip, pieces);
        if (std::any_of(pieces.begin(), pieces.end(), [](const SkipPiece& p) { return !p.kv_only; })) {
            calling.push_back(l);
        }
    }
    if (calling.size() < 2) { return next; }
    for (std::size_t i = 0; i < calling.size(); ++i) {
        next[std::size_t(calling[i])] = calling[(i + 1) % calling.size()];
    }
    return next;
}

// ---- Short prompts as decode-sized rows --------------------------------------------------------

// Rows a short prompt takes a step: a decode call's vector width (ggml_bridge.h kMaxVectorColumns,
// the widest call that takes the narrow expert path and the CPU share).
inline constexpr std::uint32_t kShortPromptRows = 8;

// Prompt sizes worth timing for the crossover between the rows path and the wide path.
inline constexpr std::array<std::uint32_t, 5> kShortPromptCandidates{32, 64, 96, 192, 384};

// Whether a prompt of `new_tokens` computed tokens runs as rows.
[[nodiscard]] inline bool short_prompt(bool on, std::uint32_t limit, std::uint32_t new_tokens) noexcept {
    return on && new_tokens != 0 && new_tokens <= limit;
}

// The tokens of a rows step with `left` still to compute before the step's frontier.
[[nodiscard]] inline std::uint32_t short_prompt_step(std::uint32_t left) noexcept {
    return std::min(left, kShortPromptRows);
}

// Logit rows of a rows step: only the one that ends the prompt needs them.
[[nodiscard]] inline std::uint32_t short_prompt_logits(std::uint32_t prefilled, std::uint32_t step,
                                                       std::uint32_t prompt_n) noexcept {
    return prefilled + step == prompt_n ? 1U : 0U;
}

// Experts a layer's calls route to for `tokens` tokens of top-`top_k` routing over `experts`, if
// the routes were uniform and independent (an upper bound for real, clustered ones).
[[nodiscard]] inline double expected_routed(std::uint32_t experts, std::uint32_t top_k,
                                            std::uint32_t tokens) noexcept {
    if (experts == 0) { return 0; }
    return double(experts) * (1.0 - std::pow(1.0 - double(top_k) / double(experts), double(tokens)));
}

// The fewest tokens whose call the delivery serves as wide (half the layer's experts or more).
[[nodiscard]] inline std::uint32_t wide_call_tokens(std::uint32_t experts, std::uint32_t top_k) noexcept {
    for (std::uint32_t t = 1; t < 1U << 20; ++t) {
        if (2.0 * expected_routed(experts, top_k, t) >= double(experts)) { return t; }
    }
    return 0;
}

} // namespace ninfer::models::deepseek_v41
