#pragma once

// The arithmetic of ExpertDelivery::settle (T0833), after Strata's swap rounds right after a prompt
// and while idle (ds4_model.cu:8707-8723, 8782-8801): a prompt's routes become tier counts of a
// fixed weight per layer, and rounds run until one swaps nothing, the budget ends, the caller
// cancels, or a round limit is reached. Plain functions, checked on the CPU
// (tests/test_expert_settle.cpp).

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer {

// Counts as tier counts: `target` pairs over the layer, in proportion (a prompt routes a thousand
// times what a decode call does, and must not bury the conversation's own counts). Zeros stay
// zero; all zero gives all zero.
[[nodiscard]] inline std::vector<float> scale_counts(std::span<const std::uint32_t> counts,
                                                     float target) {
    std::uint64_t sum = 0;
    for (const auto count : counts) { sum += count; }
    std::vector<float> out(counts.size(), 0.0F);
    if (sum == 0 || !(target > 0)) { return out; }
    const double scale = double(target) / double(sum);
    for (std::size_t i = 0; i < counts.size(); ++i) { out[i] = float(double(counts[i]) * scale); }
    return out;
}

// The same for a profile's 64-bit counts (--expert-profile).
[[nodiscard]] inline std::vector<float> scale_counts(std::span<const std::uint64_t> counts,
                                                     float target) {
    long double sum = 0;
    for (const auto count : counts) { sum += (long double)count; }
    std::vector<float> out(counts.size(), 0.0F);
    if (sum == 0 || !(target > 0)) { return out; }
    for (std::size_t i = 0; i < counts.size(); ++i) {
        out[i] = float((long double)counts[i] * (long double)target / sum);
    }
    return out;
}

struct ExpertSettle {
    enum class Stop : std::uint8_t {
        Settled,  // a round swapped nothing
        Budget,   // the time ran out before one did
        Cancelled,
        Rounds,   // the round limit
        Off,      // nothing to run (no tier rounds)
    };
    std::uint32_t rounds = 0;      // rounds run
    std::uint64_t swaps  = 0;      // experts swapped in over them
    double milliseconds  = 0;
    bool fed             = false;  // a prompt's routes were fed to the tier
    Stop stop            = Stop::Settled;
};

[[nodiscard]] inline const char* stop_name(ExpertSettle::Stop stop) noexcept {
    switch (stop) {
    case ExpertSettle::Stop::Settled: return "settled";
    case ExpertSettle::Stop::Budget: return "out of time";
    case ExpertSettle::Stop::Cancelled: return "cancelled";
    case ExpertSettle::Stop::Rounds: return "round limit";
    case ExpertSettle::Stop::Off: return "no rounds";
    }
    return "";
}

// Runs `round()` (it returns the experts it swapped in) until it returns 0, `max_ms` have passed
// by `elapsed_ms()` before a round starts, `cancel()` is true before one starts, or `max_rounds`
// rounds ran. The first round always runs unless the budget is 0 or the caller cancels first; a
// started round is never abandoned (its swaps leave experts out of their slots until it ends).
template <class Round, class Elapsed, class Cancel>
[[nodiscard]] ExpertSettle run_settle(double max_ms, std::uint32_t max_rounds, Round&& round,
                                      Elapsed&& elapsed_ms, Cancel&& cancel) {
    ExpertSettle out;
    for (;;) {
        if (out.rounds >= max_rounds) {
            out.stop = ExpertSettle::Stop::Rounds;
            break;
        }
        if (cancel()) {
            out.stop = ExpertSettle::Stop::Cancelled;
            break;
        }
        if (!(elapsed_ms() < max_ms)) {
            out.stop = ExpertSettle::Stop::Budget;
            break;
        }
        const std::uint64_t swapped = round();
        ++out.rounds;
        out.swaps += swapped;
        if (swapped == 0) {
            out.stop = ExpertSettle::Stop::Settled;
            break;
        }
    }
    out.milliseconds = elapsed_ms();
    return out;
}

} // namespace ninfer
