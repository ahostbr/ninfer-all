#pragma once

// Fork change T1310 (--expert-host-mib for Qwen3.8-Flash-Next and Edge0 disk experts): a capped
// pinned host tier. The experts the routing profile ranks highest are read once at startup into
// page-locked host memory while their bytes fit the budget; a miss on one of them is a copy from
// there, a miss on any other a read from the artifact's files (ExpertStream). This header holds the
// selection, which needs no device, so that it can be tested on its own.

#include "models/expert_profile.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <tuple>
#include <vector>

namespace ninfer::models::qwen4_exp {

struct ExpertHostTierRequest {
    std::uint64_t bytes = 0; // 0: no tier, every miss is a file read
    // [layer][expert] route counts that rank the experts (the stream's layers in order); null: an
    // even share of every layer, lowest ids first.
    const ExpertRouteCounts* profile = nullptr;
};

struct ExpertHostTierStats {
    std::uint64_t bytes   = 0; // pinned
    std::uint32_t experts = 0; // kept, over every layer
    std::uint32_t of      = 0; // every layer's experts
    double profile_share  = 0; // of the profile's routes, the kept experts' (0 without a profile)
    bool ranked           = false;
};

// Which experts the tier keeps: `kept[layer][expert]`. `expert_bytes[layer][expert]` is an expert's
// gate, up and down bytes together. With a profile, every (layer, expert) pair in order of route
// count, highest first (ties: lower layer, then lower id), joins while its bytes fit what is left;
// one that does not fit is passed over for the smaller ones after it. Without one, rounds over the
// layers each take the layer's lowest id not yet kept, while it fits. A profile whose shape is not
// the layers' is not used.
[[nodiscard]] inline std::vector<std::vector<bool>> select_host_tier(
    std::span<const std::vector<std::uint64_t>> expert_bytes, const ExpertHostTierRequest& request,
    ExpertHostTierStats* stats = nullptr) {
    std::vector<std::vector<bool>> kept(expert_bytes.size());
    ExpertHostTierStats out;
    for (std::size_t l = 0; l < expert_bytes.size(); ++l) {
        kept[l].assign(expert_bytes[l].size(), false);
        out.of += static_cast<std::uint32_t>(expert_bytes[l].size());
    }
    const ExpertRouteCounts* profile = request.profile;
    if (profile != nullptr) {
        bool fits = profile->size() == expert_bytes.size();
        for (std::size_t l = 0; fits && l < expert_bytes.size(); ++l) {
            fits = (*profile)[l].size() == expert_bytes[l].size();
        }
        if (!fits) { profile = nullptr; }
    }
    std::uint64_t left = request.bytes;
    const auto take = [&](std::size_t l, std::size_t e) {
        const std::uint64_t bytes = expert_bytes[l][e];
        if (bytes == 0 || bytes > left) { return false; }
        left -= bytes;
        kept[l][e] = true;
        out.bytes += bytes;
        ++out.experts;
        return true;
    };
    if (profile != nullptr) {
        out.ranked = true;
        std::vector<std::tuple<std::uint64_t, std::size_t, std::size_t>> order;
        std::uint64_t routes = 0;
        for (std::size_t l = 0; l < expert_bytes.size(); ++l) {
            for (std::size_t e = 0; e < expert_bytes[l].size(); ++e) {
                order.emplace_back((*profile)[l][e], l, e);
                routes += (*profile)[l][e];
            }
        }
        std::stable_sort(order.begin(), order.end(), [](const auto& a, const auto& b) {
            return std::get<0>(a) > std::get<0>(b);
        });
        std::uint64_t kept_routes = 0;
        for (const auto& [count, l, e] : order) {
            if (left == 0) { break; }
            if (take(l, e)) { kept_routes += count; }
        }
        out.profile_share = routes == 0 ? 0.0 : double(kept_routes) / double(routes);
    } else {
        std::vector<std::size_t> next(expert_bytes.size(), 0);
        bool progressed = true;
        while (progressed && left != 0) {
            progressed = false;
            for (std::size_t l = 0; l < expert_bytes.size(); ++l) {
                while (next[l] < expert_bytes[l].size()) {
                    const std::size_t e = next[l]++;
                    if (take(l, e)) {
                        progressed = true;
                        break;
                    }
                }
            }
        }
    }
    if (stats != nullptr) { *stats = out; }
    return kept;
}

} // namespace ninfer::models::qwen4_exp
