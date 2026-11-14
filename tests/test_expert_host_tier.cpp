// Fork T1310: which experts the --expert-host-mib tier of Flash-Next / Edge0 disk experts keeps.
#include "models/qwen4_exp/expert_host_tier.h"

#include <iostream>
#include <vector>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

using ninfer::models::ExpertRouteCounts;
using ninfer::models::qwen4_exp::ExpertHostTierRequest;
using ninfer::models::qwen4_exp::ExpertHostTierStats;
using ninfer::models::qwen4_exp::select_host_tier;

} // namespace

int main() {
    int failures = 0;
    // Two layers of four experts, 10 bytes each.
    const std::vector<std::vector<std::uint64_t>> sizes = {{10, 10, 10, 10}, {10, 10, 10, 10}};

    {   // No budget: nothing kept.
        ExpertHostTierStats stats;
        const auto kept = select_host_tier(sizes, ExpertHostTierRequest{0, nullptr}, &stats);
        failures += check(stats.experts == 0 && stats.bytes == 0 && stats.of == 8, "no budget keeps none");
        failures += check(!kept[0][0] && !kept[1][3], "no budget keeps none (bits)");
    }
    {   // A profile ranks across layers: the three most routed experts fit 30 bytes.
        const ExpertRouteCounts profile = {{1, 50, 2, 3}, {40, 4, 5, 30}};
        ExpertHostTierStats stats;
        const auto kept = select_host_tier(sizes, ExpertHostTierRequest{30, &profile}, &stats);
        failures += check(kept[0][1] && kept[1][0] && kept[1][3], "the profile's three highest are kept");
        failures += check(stats.experts == 3 && stats.bytes == 30 && stats.ranked, "stats count the kept");
        const double share = (50.0 + 40.0 + 30.0) / 135.0;
        failures += check(stats.profile_share > share - 1e-9 && stats.profile_share < share + 1e-9,
                          "the route share is the kept experts' routes over all");
    }
    {   // An expert that does not fit is passed over for smaller ones after it.
        const std::vector<std::vector<std::uint64_t>> mixed = {{25, 10, 10}};
        const ExpertRouteCounts profile = {{9, 8, 7}};
        ExpertHostTierStats stats;
        const auto kept = select_host_tier(mixed, ExpertHostTierRequest{20, &profile}, &stats);
        failures += check(!kept[0][0] && kept[0][1] && kept[0][2], "a too-big expert is skipped");
        failures += check(stats.bytes == 20, "the budget is never exceeded");
    }
    {   // No profile: an even share, lowest ids of each layer in turn.
        ExpertHostTierStats stats;
        const auto kept = select_host_tier(sizes, ExpertHostTierRequest{30, nullptr}, &stats);
        failures += check(kept[0][0] && kept[1][0] && kept[0][1] && !kept[1][1],
                          "without a profile the layers take turns from id 0");
        failures += check(!stats.ranked && stats.experts == 3, "an unranked tier says so");
    }
    {   // A profile of another shape is not used.
        const ExpertRouteCounts wrong = {{1, 2, 3}};
        ExpertHostTierStats stats;
        const auto kept = select_host_tier(sizes, ExpertHostTierRequest{20, &wrong}, &stats);
        failures += check(!stats.ranked && kept[0][0] && kept[1][0], "a mismatched profile falls back");
    }
    {   // A budget larger than everything keeps everything.
        const ExpertRouteCounts profile = {{1, 1, 1, 1}, {1, 1, 1, 1}};
        ExpertHostTierStats stats;
        (void)select_host_tier(sizes, ExpertHostTierRequest{1000, &profile}, &stats);
        failures += check(stats.experts == 8 && stats.bytes == 80, "a large budget keeps all");
    }
    if (failures == 0) { std::cout << "expert host tier tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
