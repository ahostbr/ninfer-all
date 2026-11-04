#pragma once

// T0833: the switches of the Engram read path's speed experiments. Each is an A/B flag that
// defaults to off: it changes how the rows are read, never which bytes. The core resolves them
// from the command line and the environment (NINFER_DS_ENGRAM_DEDUPE, NINFER_DS_ENGRAM_OVERLAP) and
// logs each as ON or off at start. Pure functions, so the parsing runs without a device.

#include "ninfer/model_enums.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::models::deepseek_v41 {

struct EngramTuning {
    // Repeated rows of a batch are read once, and image tokens' rows (never used) are not read.
    bool dedupe = false;
    EngramOverlap overlap = EngramOverlap::Off;
};

// "1", "on", "true" and "0", "off", "false" (lower case); anything else is refused, so a typo in
// a bench script cannot pass for the default.
[[nodiscard]] inline bool parse_switch(std::string_view name, std::string_view text) {
    if (text == "1" || text == "on" || text == "true") { return true; }
    if (text == "0" || text == "off" || text == "false") { return false; }
    throw std::invalid_argument(std::string(name) + " accepts 1, on, true, 0, off or false, not '" +
                                std::string(text) + "'");
}

// Off values as parse_switch; "1", "on", "true" and "pass" are Pass; "ahead" is Ahead.
[[nodiscard]] inline EngramOverlap parse_overlap(std::string_view name, std::string_view text) {
    if (text == "ahead") { return EngramOverlap::Ahead; }
    if (text == "pass") { return EngramOverlap::Pass; }
    return parse_switch(name, text) ? EngramOverlap::Pass : EngramOverlap::Off;
}

[[nodiscard]] inline const char* overlap_name(EngramOverlap overlap) noexcept {
    switch (overlap) {
    case EngramOverlap::Off: return "off";
    case EngramOverlap::Pass: return "pass";
    case EngramOverlap::Ahead: return "ahead";
    }
    return "off";
}

// `tuning` with the environment over it: an unset or empty variable leaves the option alone.
// `lookup` returns the variable's value or nullptr (std::getenv in the core).
template <class Lookup>
[[nodiscard]] EngramTuning resolve_engram_tuning(EngramTuning tuning, Lookup&& lookup) {
    const auto text = [&](const char* name) -> const char* {
        const char* value = lookup(name);
        return value != nullptr && *value != '\0' ? value : nullptr;
    };
    if (const char* v = text("NINFER_DS_ENGRAM_DEDUPE")) {
        tuning.dedupe = parse_switch("NINFER_DS_ENGRAM_DEDUPE", v);
    }
    if (const char* v = text("NINFER_DS_ENGRAM_OVERLAP")) {
        tuning.overlap = parse_overlap("NINFER_DS_ENGRAM_OVERLAP", v);
    }
    return tuning;
}

// The row ids of a pass in the order the device reads them, block after block: out[(b * t + r) *
// columns + c] is column c of Engram block b for token r, from the hasher's token-major
// scratch[r * per + b * columns + c] (per = blocks * columns). With `skip_dead`, a token whose
// dead[r] is set (an image's) gets `skip_row` in every column instead: nothing reads its rows.
inline void layout_engram_ids(std::span<const std::uint64_t> scratch, std::size_t t, std::size_t per,
                              std::size_t columns, std::span<const std::uint8_t> dead, bool skip_dead,
                              std::uint64_t skip_row, std::uint64_t* out) {
    const std::size_t blocks = per / columns;
    for (std::size_t b = 0; b < blocks; ++b) {
        for (std::size_t r = 0; r < t; ++r) {
            std::uint64_t* to = out + (b * t + r) * columns;
            if (skip_dead && !dead.empty() && dead[r] != 0) {
                std::fill_n(to, columns, skip_row);
            } else {
                std::memcpy(to, scratch.data() + r * per + b * columns, columns * sizeof(std::uint64_t));
            }
        }
    }
}

// What a call's Engram rows are of: the compressed ids a sequence's history holds and the tokens
// the call runs (FNV-1a, never 0). A prompt chunk's rows read ahead are used by the call whose
// key equals theirs; equal keys mean equal rows, since the rows are a function of nothing else.
[[nodiscard]] inline std::uint64_t engram_batch_key(std::span<const std::int32_t> history,
                                                    std::span<const std::int32_t> tokens) noexcept {
    std::uint64_t h = 14695981039346656037ULL;
    const auto mix  = [&h](std::uint64_t v) {
        h ^= v;
        h *= 1099511628211ULL;
    };
    mix(history.size());
    for (const std::int32_t v : history) { mix(static_cast<std::uint32_t>(v)); }
    mix(tokens.size());
    for (const std::int32_t v : tokens) { mix(static_cast<std::uint32_t>(v)); }
    return h | 1U;
}

} // namespace ninfer::models::deepseek_v41
