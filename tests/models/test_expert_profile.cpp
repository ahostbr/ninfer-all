#include "models/expert_profile.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    using namespace ninfer::models;
    const auto directory = std::filesystem::temp_directory_path() /
        ("ninfer-expert-profile-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    struct Cleanup { std::filesystem::path path; ~Cleanup() { std::filesystem::remove_all(path); } } cleanup{directory};
    try {
        const auto path = directory / "profile.json";
        ninfer::artifact::ArtifactId artifact{};
        const std::array<std::size_t, 2> widths{3, 2};
        const ExpertRouteCounts expected{{0, 17, std::numeric_limits<std::uint64_t>::max()}, {3, 4}};
        write_expert_profile(path, artifact, expected);
        if (read_expert_profile(path, artifact, widths) != expected) {
            throw std::runtime_error("expert profile did not preserve exact counts");
        }
        const auto rejects = [&](auto id, auto geometry) {
            try { (void)read_expert_profile(path, id, geometry); }
            catch (const std::exception&) { return; }
            throw std::runtime_error("expert profile accepted incompatible input");
        };
        auto wrong = artifact; wrong[0] = std::byte{1};
        rejects(wrong, widths);
        rejects(artifact, std::array<std::size_t, 1>{3});
        rejects(artifact, std::array<std::size_t, 2>{2, 3});
        for (const auto* counts : {"[[1,-2,3],[4,5]]", "[[1,2.5,3],[4,5]]", "[[1,2,3],[4]]"}) {
            std::ofstream(path) << "{\"version\":1,\"artifact_id\":\"00000000000000000000000000000000\",\"counts\":" << counts << '}';
            rejects(artifact, widths);
        }
        std::ofstream(path) << "{\"version\":1,";
        rejects(artifact, widths);

        // DeepSeek-V4.1 Coder (T0833): 40 text layers of 192 kept experts, read by position. Strata's
        // profile of the pack may carry three more rows for the drafter's layers; those are ignored
        // where the caller allows spare rows, and refused where it does not.
        {
            const std::vector<std::size_t> coder(40, 192);
            ExpertRouteCounts counts(40, std::vector<std::uint64_t>(192));
            for (std::size_t layer = 0; layer < 40; ++layer) {
                for (std::size_t e = 0; e < 192; ++e) { counts[layer][e] = layer * 1000 + e; }
            }
            ninfer::artifact::ArtifactId id{};
            id[0] = std::byte{0xAB}; id[15] = std::byte{0x01};
            write_expert_profile(path, id, counts);
            if (read_expert_profile(path, id, coder) != counts ||
                read_expert_profile(path, id, coder, kExpertProfileSpareRows) != counts) {
                throw std::runtime_error("a 40 x 192 profile was not read by position");
            }
            auto padded = counts;
            for (int extra = 0; extra < 3; ++extra) { padded.emplace_back(192, 7); }
            write_expert_profile(path, id, padded);
            if (read_expert_profile(path, id, coder, kExpertProfileSpareRows) != counts) {
                throw std::runtime_error("spare drafter rows changed the text layers' counts");
            }
            const auto message = [&](auto&&... arguments) -> std::string {
                try { (void)read_expert_profile(path, id, arguments...); }
                catch (const std::exception& error) { return error.what(); }
                return {};
            };
            if (message(coder).find("43 layers, the model has 40") == std::string::npos) {
                throw std::runtime_error("extra rows without spare rows must be refused, saying how many");
            }
            padded.resize(49, std::vector<std::uint64_t>(192, 1));
            write_expert_profile(path, id, padded);
            if (message(coder, kExpertProfileSpareRows).find("49 layers") == std::string::npos) {
                throw std::runtime_error("more than the spare rows must be refused");
            }
            write_expert_profile(path, id, ExpertRouteCounts(39, std::vector<std::uint64_t>(192)));
            if (message(coder, kExpertProfileSpareRows).find("39 layers") == std::string::npos) {
                throw std::runtime_error("a short profile must be refused");
            }
            ExpertRouteCounts wide(40, std::vector<std::uint64_t>(192));
            wide[17].assign(384, 0); // a profile counted on the base model's 384 experts a layer
            write_expert_profile(path, id, wide);
            if (message(coder, kExpertProfileSpareRows).find("layer 17 has 384 experts, the model's has 192") ==
                std::string::npos) {
                throw std::runtime_error("a 384-wide layer must be refused, naming the layer");
            }
            write_expert_profile(path, id, counts);
            ninfer::artifact::ArtifactId other = id;
            other[3] = std::byte{9};
            try { (void)read_expert_profile(path, other, coder); throw std::runtime_error("another artifact accepted"); }
            catch (const std::invalid_argument& error) {
                const std::string text = error.what();
                if (text.find(expert_profile_artifact_id(id)) == std::string::npos ||
                    text.find(expert_profile_artifact_id(other)) == std::string::npos) {
                    throw std::runtime_error("an artifact mismatch must name both identities: " + text);
                }
            }
            if (expert_profile_artifact_id(id) != std::string("ab") + std::string(28, '0') + "01") {
                throw std::runtime_error("the artifact identity is 32 lowercase hex digits");
            }
        }
        std::cout << "EXPERT_PROFILE_CONTRACT_PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
