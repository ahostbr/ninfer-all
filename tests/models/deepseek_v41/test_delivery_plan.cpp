// CPU checks of the delivery experiments' arithmetic (src/models/deepseek_v41/delivery_plan.h):
// the switches and their environment overrides, the order a pass's expert calls come in under the
// long-prompt skip, and which prompts run as rows. No device is used.

#include "models/deepseek_v41/delivery_plan.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer::models::deepseek_v41;
using ninfer::HybridExpertOptions;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    if (*value) { setenv(name, value, 1); } else { unsetenv(name); }
#endif
}

void clear_env() {
    for (const char* name : {"NINFER_DS_PREFETCH_AHEAD", "NINFER_DS_CACHE_SETTLE", "NINFER_DS_SHORT_PROMPT",
                             "NINFER_DS_PROFILE_WEIGHTS", "NINFER_DS_CACHE_SETTLE_MS",
                             "NINFER_DS_CACHE_SETTLE_IDLE_MS", "NINFER_DS_SHORT_PROMPT_TOKENS"}) {
        set_env(name, "");
    }
}

bool throws(const HybridExpertOptions& o) {
    try { (void)delivery_flags(o); } catch (const std::invalid_argument&) { return true; }
    return false;
}

void flags() {
    clear_env();
    HybridExpertOptions o;
    check(!o.deepseek_delivery_requested(), "nothing requested by default");
    auto f = delivery_flags(o);
    check(!f.prefetch_ahead && !f.cache_settle && !f.short_prompt && !f.profile_weights,
          "every experiment is off by default");
    check(f.settle_ms == 400 && f.settle_idle_ms == 6000 && f.short_prompt_tokens == 96, "default numbers");
    o.deepseek_short_prompt = true;
    check(o.deepseek_delivery_requested() && delivery_flags(o).short_prompt, "an option turns one on");
    set_env("NINFER_DS_SHORT_PROMPT", "off");
    check(!delivery_flags(o).short_prompt, "the variable turns it off over the option");
    set_env("NINFER_DS_SHORT_PROMPT", "true");
    o.deepseek_short_prompt = false;
    check(delivery_flags(o).short_prompt, "the variable turns it on over the option");
    set_env("NINFER_DS_SHORT_PROMPT", "yes");
    check(throws(o), "a mistyped switch throws instead of reading as one arm");
    set_env("NINFER_DS_SHORT_PROMPT", "");
    set_env("NINFER_DS_PREFETCH_AHEAD", "1");
    set_env("NINFER_DS_CACHE_SETTLE", "on");
    set_env("NINFER_DS_PROFILE_WEIGHTS", "true");
    set_env("NINFER_DS_CACHE_SETTLE_MS", "250");
    set_env("NINFER_DS_CACHE_SETTLE_IDLE_MS", "0");
    set_env("NINFER_DS_SHORT_PROMPT_TOKENS", "192");
    f = delivery_flags(o);
    check(f.prefetch_ahead && f.cache_settle && f.profile_weights && f.settle_ms == 250 &&
              f.settle_idle_ms == 0 && f.short_prompt_tokens == 192,
          "every variable is read");
    set_env("NINFER_DS_SHORT_PROMPT_TOKENS", "0");
    check(throws(o), "a short prompt limit of 0 is refused");
    set_env("NINFER_DS_SHORT_PROMPT_TOKENS", "5000");
    check(throws(o), "a short prompt limit past 4096 is refused");
    set_env("NINFER_DS_SHORT_PROMPT_TOKENS", "ninety");
    check(throws(o), "a mistyped number throws");
    clear_env();
}

// The Flash release: 40 blocks, the skip at block 20.
constexpr int kLayers = 40, kSkipLayer = 20;

void chain_plain() {
    const auto next = prefetch_chain(kLayers, -1, 8192, 2048, 0);
    bool ordered = next.size() == kLayers;
    for (int l = 0; l + 1 < kLayers; ++l) { ordered = ordered && next[std::size_t(l)] == l + 1; }
    check(ordered && next[39] == 0, "without a skip every block is followed by the next, the last by the first");
    const auto skipless = prefetch_chain(kLayers, kSkipLayer, 8192, 2048, 0);
    check(skipless == next, "a pass that skips no rows has the plain order");
}

void chain_skipping() {
    // Every row skips: blocks 0-19 call experts, 20 is kv-only, 21-39 do nothing.
    const auto all = prefetch_chain(kLayers, kSkipLayer, 8192, 2048, 8192);
    bool ordered = true;
    for (int l = 0; l < 19; ++l) { ordered = ordered && all[std::size_t(l)] == l + 1; }
    check(ordered, "blocks before the skip block keep their order");
    check(all[19] == 0, "the last calling block of a fully skipped pass is followed by block 0");
    check(all[20] == -1, "the kv-only block names none");
    bool silent = true;
    for (int l = 21; l < kLayers; ++l) { silent = silent && all[std::size_t(l)] == -1; }
    check(silent, "blocks past the skip block make no call");
    // Part of the rows skip: every block still calls (the kept rows), in the plain order.
    const auto mixed = prefetch_chain(kLayers, kSkipLayer, 8192, 2048, 4096);
    check(mixed[19] == 20 && mixed[20] == 21 && mixed[39] == 0,
          "a pass with kept rows runs through block 20 and on");
    // A skip block whose kept rows exist but all earlier rows skipped still calls at block 20.
    const auto one = prefetch_chain(kLayers, kSkipLayer, 2048, 2048, 1024);
    check(one[19] == 20 && one[39] == 0, "one chunk with a cut calls every block's experts for its kept rows");
    // Degenerate: one calling block names none (nothing follows within the pass), and no layers.
    check(prefetch_chain(1, -1, 128, 128, 0) == std::vector<std::int32_t>{-1}, "a single block has no successor");
    check(prefetch_chain(0, -1, 128, 128, 0).empty(), "no blocks, no order");
}

void rows() {
    check(short_prompt(true, 96, 1) && short_prompt(true, 96, 96) && !short_prompt(true, 96, 97),
          "a prompt of up to the limit runs as rows");
    check(!short_prompt(false, 96, 10) && !short_prompt(true, 96, 0), "off, or nothing to compute: no");
    check(short_prompt_step(100) == 8 && short_prompt_step(8) == 8 && short_prompt_step(5) == 5 &&
              short_prompt_step(1) == 1,
          "a rows step takes eight tokens at most");
    // Walk a 21-token prompt: steps of 8, 8, 5; only the last asks for logits.
    std::uint32_t prefilled = 0, steps = 0, logits = 0;
    while (prefilled < 21) {
        const auto n = short_prompt_step(21 - prefilled);
        logits += short_prompt_logits(prefilled, n, 21);
        prefilled += n;
        ++steps;
    }
    check(steps == 3 && logits == 1 && prefilled == 21, "three steps, the logits once, at the end");
    // A frontier inside the prompt takes no logits.
    check(short_prompt_logits(0, 8, 21) == 0 && short_prompt_logits(16, 5, 21) == 1, "logits only at the prompt's end");
    check(kShortPromptRows == 8, "a decode call's width");
}

void crossover() {
    // 192 experts, top-6: eight tokens route to about 43 experts, the delivery's wide boundary
    // (half the experts) is reached at 22 tokens; 384 experts at 45.
    const double eight = expected_routed(192, 6, 8);
    check(eight > 42.5 && eight < 43.5, "eight tokens route about 43 of 192 experts: " + std::to_string(eight));
    check(wide_call_tokens(192, 6) == 22, "a call of 22 tokens is the first wide one on 192 experts");
    check(wide_call_tokens(384, 6) == 45, "and 45 tokens on 384");
    check(expected_routed(192, 6, 96) > 180.0, "96 tokens route nearly every expert: the wide path copies them all");
    check(expected_routed(0, 6, 8) == 0.0, "no experts, no routes");
    check(kShortPromptCandidates.front() == 32 && kShortPromptCandidates.back() == 384, "the candidate sizes");
    // The rows path's per-layer routed experts for a 96-token prompt: twelve steps of eight.
    const double per_layer = 12.0 * expected_routed(192, 6, 8);
    check(per_layer > 500.0, "twelve rows steps route ~500 expert-uses a layer against 192 for one wide pass: "
                             "the rows path wins on copies only through the cache hits and the CPU share");
}

} // namespace

int main() {
    flags();
    chain_plain();
    chain_skipping();
    rows();
    crossover();
    if (failures == 0) { std::cout << "DELIVERY_PLAN_PASS\n"; }
    return failures == 0 ? 0 : 1;
}
