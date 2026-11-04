#pragma once

// Routing profiles of MoE models with host or disk experts: how often each layer's experts were
// routed, which fills a device expert cache at startup and sizes each layer's share of its slots.
// A recorded profile (--expert-profile-out) names its artifact and bank widths, so it cannot rank
// another model's experts. The built-in profiles were recorded on a family's release and serve
// every artifact of the same family and bank geometry: a profile only seeds the cache, whose tier
// then follows the conversation, so it never changes a result.

#include "artifact/framing.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::models {

using ExpertRouteCounts = std::vector<std::vector<std::uint64_t>>;

// Counts address the family's expert layers in order (Flash-Next: the text layers, then the MTP
// block's). An artifact identity and exact bank widths prevent a profile from silently ranking
// another model's experts.
// `spare_rows` (default none): rows past the model's layers that the file may carry and the read
// ignores, as Strata's DeepSeek-V4.1 profiles carry the drafter's layers after the text layers'.
// The checks name what differs: the version, the artifact (both identities), the layer count, or the
// layer whose width differs.
inline constexpr std::size_t kExpertProfileSpareRows = 8;
ExpertRouteCounts read_expert_profile(const std::filesystem::path& path,
    const artifact::ArtifactId& artifact, std::span<const std::size_t> experts,
    std::size_t spare_rows = 0);
// The artifact identity as a profile file spells it: 32 lowercase hex digits.
std::string expert_profile_artifact_id(const artifact::ArtifactId& artifact);
void write_expert_profile(const std::filesystem::path& path,
    const artifact::ArtifactId& artifact, const ExpertRouteCounts& counts);

// The built-in profile of `family` ("qwen3.8-flash-next", ...) for layers of these bank widths,
// or nothing when none was recorded for that geometry.
std::optional<ExpertRouteCounts> builtin_expert_profile(std::string_view family,
                                                        std::span<const std::size_t> experts);

} // namespace ninfer::models
