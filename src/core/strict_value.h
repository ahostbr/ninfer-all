#pragma once

// Strict readings of option and environment values: nothing is guessed from a typo. A switch is
// exactly 1, on or true (set) or 0, off or false (clear); a count is decimal digits alone that fit
// the type.

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer {

[[nodiscard]] inline std::optional<bool> parse_switch(std::string_view text) noexcept {
    if (text == "1" || text == "on" || text == "true") { return true; }
    if (text == "0" || text == "off" || text == "false") { return false; }
    return std::nullopt;
}

[[nodiscard]] inline std::optional<std::uint32_t> parse_count(std::string_view text) noexcept {
    std::uint32_t value = 0;
    if (text.empty()) { return std::nullopt; }
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) { return std::nullopt; }
    return value;
}

// A switch from the environment: `fallback` when `name` is unset or empty; a value that is not a
// switch is an error naming the variable, so a mistyped A/B run does not pass for the other arm.
[[nodiscard]] inline bool env_switch(const char* name, bool fallback) {
    const char* text = std::getenv(name);
    if (text == nullptr || *text == '\0') { return fallback; }
    if (const auto value = parse_switch(text)) { return *value; }
    throw std::invalid_argument(std::string(name) + " takes 1|on|true or 0|off|false, not '" + text +
                                "'");
}

[[nodiscard]] inline std::uint32_t env_count(const char* name, std::uint32_t fallback) {
    const char* text = std::getenv(name);
    if (text == nullptr || *text == '\0') { return fallback; }
    if (const auto value = parse_count(text)) { return *value; }
    throw std::invalid_argument(std::string(name) + " takes an unsigned integer, not '" + text + "'");
}

} // namespace ninfer
