// Fork T1320 (--expert-cache-free): the figure an expert cache sizes from. CUDA's by default; with
// min, NVML's when it answered and is smaller, never larger. CPU only: read_vram_free() itself
// needs a device and is not run here.
#include "core/vram_free.h"

#include <iostream>
#include <stdexcept>
#include <string>

using namespace ninfer;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

} // namespace

int main() {
    try {
        constexpr std::uint64_t kMiB = 1ULL << 20;
        VramFree free;
        free.cuda = 30994 * kMiB;
        require(free.for_cache(false) == free.cuda && free.for_cache(true) == free.cuda,
                "without NVML both choices are CUDA's figure");
        free.nvml = 24348 * kMiB; // edge0-strata's RTX 5090 under WDDM (docs/DETAILS.md:1449)
        require(free.for_cache(false) == free.cuda, "cuda: CUDA's figure, NVML or not");
        require(free.for_cache(true) == *free.nvml, "min: the smaller, NVML's here");
        free.nvml = 31500 * kMiB;
        require(free.for_cache(true) == free.cuda, "min never takes the larger figure");
        free.nvml = 0;
        require(free.for_cache(true) == 0, "min takes a full card's 0");
        std::cout << "PASS vram free\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
