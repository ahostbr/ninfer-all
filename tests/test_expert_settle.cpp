// CPU checks of the cache settle's arithmetic (src/core/expert_settle.h) and of the tier's on-demand
// round (ExpertTier::end_round), and of the strict switch and count readings (core/strict_value.h).
// No device is used.

#include "core/expert_settle.h"
#include "core/expert_tier.h"
#include "core/strict_value.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

bool near(float a, float b) { return std::fabs(a - b) < 1e-3F; }

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    if (*value) { setenv(name, value, 1); } else { unsetenv(name); }
#endif
}

void scaling() {
    const std::vector<std::uint32_t> counts{0, 30, 10, 0, 60};
    const auto scaled = scale_counts(counts, 160.0F);
    float sum = 0;
    for (const float v : scaled) { sum += v; }
    check(near(sum, 160.0F), "scaled counts total the target");
    check(scaled[0] == 0 && scaled[3] == 0, "zeros stay zero");
    check(near(scaled[4] / scaled[1], 2.0F) && near(scaled[1] / scaled[2], 3.0F), "proportions are kept");
    const std::vector<std::uint32_t> none(5, 0);
    const auto zero = scale_counts(none, 160.0F);
    check(zero.size() == 5 && zero[1] == 0, "all zero stays all zero");
    check(scale_counts(counts, 0.0F)[1] == 0 && scale_counts(counts, -1.0F)[1] == 0,
          "a target of nothing gives nothing");
    // 64-bit counts (a profile file), beyond what 32 bits hold.
    const std::vector<std::uint64_t> big{0, 3ULL << 40, 1ULL << 40};
    const auto b = scale_counts(big, 8.0F);
    check(near(b[1], 6.0F) && near(b[2], 2.0F) && b[0] == 0, "64-bit counts scale without overflow");
}

void settle_loop() {
    // A fake clock: each round takes 10 ms.
    double now = 0;
    int cancel_after = 99;
    std::vector<std::uint64_t> swaps{5, 3, 0, 9};
    std::size_t at = 0;
    const auto round = [&] {
        now += 10;
        return swaps[at++];
    };
    const auto elapsed = [&] { return now; };
    const auto cancel  = [&] { return int(at) >= cancel_after; };
    auto r = run_settle(1000, 6, round, elapsed, cancel);
    check(r.rounds == 3 && r.swaps == 8 && r.stop == ExpertSettle::Stop::Settled,
          "rounds run until one swaps nothing");
    check(near(float(r.milliseconds), 30.0F), "the time spent is reported");

    now = 0; at = 0; swaps = {5, 5, 5, 5, 5, 5, 5};
    r = run_settle(1000, 4, round, elapsed, cancel);
    check(r.rounds == 4 && r.stop == ExpertSettle::Stop::Rounds, "the round limit stops it");

    now = 0; at = 0;
    r = run_settle(25, 6, round, elapsed, cancel);
    check(r.rounds == 3 && r.stop == ExpertSettle::Stop::Budget,
          "a round starts only while time is left: 0, 10, 20 start, 30 does not");

    now = 0; at = 0;
    r = run_settle(0, 6, round, elapsed, cancel);
    check(r.rounds == 0 && r.stop == ExpertSettle::Stop::Budget, "no budget runs no round");

    now = 0; at = 0; cancel_after = 2;
    r = run_settle(1000, 6, round, elapsed, cancel);
    check(r.rounds == 2 && r.stop == ExpertSettle::Stop::Cancelled,
          "cancel is polled before each round, and a started round finishes");
    cancel_after = 0; now = 0; at = 0;
    r = run_settle(1000, 6, round, elapsed, cancel);
    check(r.rounds == 0 && r.stop == ExpertSettle::Stop::Cancelled, "cancelled before the first round");
    check(std::string(stop_name(ExpertSettle::Stop::Settled)) == "settled", "stop names");
}

void tier_round_now() {
    using State = ExpertTier::State;
    // `every` is 4, so end_pass() alone would wait for the fourth pass; end_round() does not.
    ExpertTier tier({6}, {0}, {.every = 4});
    tier.set_free(0, 1);
    for (std::int32_t e = 0; e < 3; ++e) { tier.set(0, e, State::Held); }
    tier.route(0, 4, 10.0F);
    tier.route(0, 5, 6.0F);
    tier.route(0, 0, 1.0F);
    check(tier.end_pass().empty(), "no round after one of four passes");
    const auto swaps = tier.end_round();
    // Expert 4 takes the free slot, expert 5 replaces the coldest held expert (1 or 2, count 0).
    check(swaps.size() == 2, "a forced round swaps what a full one would: " + std::to_string(swaps.size()));
    check(swaps[0].in == 4 && swaps[0].out == -1, "the hottest candidate takes the free slot");
    check(swaps[1].in == 5 && swaps[1].out >= 1 && swaps[1].out <= 2, "the next replaces a cold one");
    check(near(tier.count(0, 4), 7.0F), "the counts decay by 0.7");
    // The passes counted before a forced round do not carry over: a full round needs four more.
    check(tier.end_pass().empty() && tier.end_pass().empty() && tier.end_pass().empty(),
          "the next full round is `every` passes after a forced one");
    ExpertTier off({4}, {0}, {.every = 0});
    off.route(0, 1, 10.0F);
    check(off.end_round().empty(), "rounds off: a forced round does nothing");
}

void strict_values() {
    check(parse_switch("1") == true && parse_switch("on") == true && parse_switch("true") == true,
          "set switches");
    check(parse_switch("0") == false && parse_switch("off") == false && parse_switch("false") == false,
          "clear switches");
    for (const char* bad : {"", "yes", "no", "ON", "True", "2", " 1", "1 ", "enabled"}) {
        check(!parse_switch(bad).has_value(), std::string("a switch rejects '") + bad + "'");
    }
    check(parse_count("0") == 0U && parse_count("400") == 400U && parse_count("4294967295") == 4294967295U,
          "counts");
    for (const char* bad : {"", "-1", "+1", "1.5", "12ms", " 5", "4294967296", "0x10"}) {
        check(!parse_count(bad).has_value(), std::string("a count rejects '") + bad + "'");
    }
    set_env("NINFER_TEST_SWITCH", "");
    check(env_switch("NINFER_TEST_SWITCH", true) == true && env_switch("NINFER_TEST_SWITCH", false) == false,
          "an unset variable keeps the option");
    set_env("NINFER_TEST_SWITCH", "off");
    check(env_switch("NINFER_TEST_SWITCH", true) == false, "the variable overrides the option");
    set_env("NINFER_TEST_SWITCH", "on");
    check(env_switch("NINFER_TEST_SWITCH", false) == true, "the variable overrides the option, on");
    set_env("NINFER_TEST_SWITCH", "maybe");
    bool threw = false;
    try { (void)env_switch("NINFER_TEST_SWITCH", false); }
    catch (const std::invalid_argument& error) {
        threw = std::string(error.what()).find("NINFER_TEST_SWITCH") != std::string::npos;
    }
    check(threw, "a mistyped variable throws, naming itself");
    set_env("NINFER_TEST_COUNT", "250");
    check(env_count("NINFER_TEST_COUNT", 1) == 250U, "a count from the environment");
    set_env("NINFER_TEST_COUNT", "2.5");
    threw = false;
    try { (void)env_count("NINFER_TEST_COUNT", 1); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "a mistyped count throws");
    set_env("NINFER_TEST_SWITCH", "");
    set_env("NINFER_TEST_COUNT", "");
}

} // namespace

int main() {
    scaling();
    settle_loop();
    tier_round_now();
    strict_values();
    if (failures == 0) { std::cout << "EXPERT_SETTLE_PASS\n"; }
    return failures == 0 ? 0 : 1;
}
