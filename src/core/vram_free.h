#pragma once

// Fork T1320: free device memory as CUDA and as NVML report it.
//
// Under Windows WDDM cudaMemGetInfo does not count what other processes hold on the card, so it
// can overstate what an allocation may take without the driver paging device memory to system
// RAM. edge0-strata measured what that costs (docs/DETAILS.md:1398-1410, :1449, at 7b60076): an
// RTX 5090 whose CUDA figure said 30,994 MiB free while NVML said 24,348, and decode at 80-115
// tok/s with the card filled to 0 MiB free against 205-230 with 900 MiB or more left. Its remedy,
// opt-in there as here (STRATA_VRAM_FREE=device, include/strata/core/expert_cache.hpp:75-81),
// sizes the expert cache from the smaller of the two figures. That can only make it smaller.

#include <cstdint>
#include <optional>
#include <string>

namespace ninfer {

struct VramFree {
    std::uint64_t cuda = 0;              // cudaMemGetInfo, the device bound to the calling thread
    std::optional<std::uint64_t> nvml;   // NVML's free memory of the whole card, when it answered
    std::string nvml_problem;            // why NVML did not answer, when it did not

    // What a cache sizes from: CUDA's figure, or the smaller of the two when `smaller` asks for it
    // and NVML answered.
    [[nodiscard]] std::uint64_t for_cache(bool smaller) const noexcept {
        return smaller && nvml && *nvml < cuda ? *nvml : cuda;
    }
};

// Both figures for the device bound to the calling thread. NVML is loaded once, at the first call,
// from the driver's library at run time (a host without it still gets CUDA's figure).
[[nodiscard]] VramFree read_vram_free();

} // namespace ninfer
