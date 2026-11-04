#include "models/expert_profile.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>
#include <string>

namespace ninfer::models {
namespace {
std::string identity(const artifact::ArtifactId& artifact) {
    constexpr char hex[] = "0123456789abcdef";
    std::string out;
    for (const auto byte : artifact) {
        const auto value = std::to_integer<unsigned>(byte);
        out += hex[value >> 4]; out += hex[value & 15];
    }
    return out;
}

// A built-in profile: its family, its layers' bank widths and their counts, layer after layer.
struct Builtin {
    std::string_view family;
    std::span<const std::uint16_t> widths;
    std::span<const std::uint32_t> counts;
};

#include "models/expert_profiles/builtin.inc"

} // namespace

std::string expert_profile_artifact_id(const artifact::ArtifactId& artifact) { return identity(artifact); }

ExpertRouteCounts read_expert_profile(const std::filesystem::path& path,
    const artifact::ArtifactId& artifact, std::span<const std::size_t> experts,
    std::size_t spare_rows) {
    std::ifstream file(path);
    if (!file) { throw std::runtime_error("cannot read expert profile " + path.string()); }
    const auto data = nlohmann::json::parse(file);
    const std::string where = "expert profile " + path.string() + ": ";
    if (!data.is_object() || data.value("version", 0) != 1 || !data.contains("counts") ||
        !data["counts"].is_array()) {
        throw std::invalid_argument(where + "not a version 1 profile with a counts array");
    }
    if (data.value("artifact_id", std::string{}) != identity(artifact)) {
        throw std::invalid_argument(where + "recorded for artifact '" +
                                    data.value("artifact_id", std::string{}) + "', this one is '" +
                                    identity(artifact) + "'");
    }
    const std::size_t rows = data["counts"].size();
    if (rows < experts.size() || rows > experts.size() + spare_rows) {
        throw std::invalid_argument(where + std::to_string(rows) + " layers, the model has " +
                                    std::to_string(experts.size()) +
                                    (spare_rows ? " (and up to " + std::to_string(spare_rows) +
                                                      " more rows are ignored)"
                                                : std::string{}));
    }
    ExpertRouteCounts out(experts.size());
    for (std::size_t layer = 0; layer < experts.size(); ++layer) {
        const auto& values = data["counts"][layer];
        if (!values.is_array() || values.size() != experts[layer]) {
            throw std::invalid_argument(where + "layer " + std::to_string(layer) + " has " +
                                        std::to_string(values.is_array() ? values.size() : 0) +
                                        " experts, the model's has " + std::to_string(experts[layer]));
        }
        for (const auto& value : values) {
            if (!value.is_number_unsigned()) {
                throw std::invalid_argument("expert profile counts must be nonnegative integers");
            }
            out[layer].push_back(value.get<std::uint64_t>());
        }
    }
    return out;
}

void write_expert_profile(const std::filesystem::path& path,
    const artifact::ArtifactId& artifact, const ExpertRouteCounts& counts) {
    const nlohmann::json data{{"version", 1}, {"artifact_id", identity(artifact)}, {"counts", counts}};
    std::ofstream file(path, std::ios::trunc);
    file << data.dump() << '\n';
    file.close();
    if (!file) { throw std::runtime_error("cannot write expert profile " + path.string()); }
}

std::optional<ExpertRouteCounts> builtin_expert_profile(std::string_view family,
                                                        std::span<const std::size_t> experts) {
    for (const Builtin& profile : kBuiltinProfiles) {
        if (profile.family != family || profile.widths.size() != experts.size()) { continue; }
        bool same = true;
        for (std::size_t layer = 0; layer < experts.size(); ++layer) {
            same = same && profile.widths[layer] == experts[layer];
        }
        if (!same) { continue; }
        ExpertRouteCounts out(experts.size());
        std::size_t at = 0;
        for (std::size_t layer = 0; layer < experts.size(); ++layer) {
            out[layer].assign(profile.counts.begin() + std::ptrdiff_t(at),
                              profile.counts.begin() + std::ptrdiff_t(at + experts[layer]));
            at += experts[layer];
        }
        return out;
    }
    return std::nullopt;
}

} // namespace ninfer::models
