// Repro for fork change W4: MSVC 19.38 /O2 folds core::wide_add's overflow flag wrongly.
//   cl /std:c++20 /EHsc /O2 /I src docsork\msvc_19_38_wide_add.cpp
// Expected output: "max128+1: low=0 high=0 overflowed=1". cl 19.38.33145 /O2 prints overflowed=0;
// /Od, volatile inputs, and cl 19.44.35222 at /O1 and /O2 print 1.
#include "core/wide_math.h"
#include <cstdio>
using namespace ninfer::core;
int main() {
    const std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();
    bool o = false;
    Uint128 r = wide_add(Uint128{kMax, kMax}, Uint128{1, 0}, o);
    std::printf("max128+1: low=%llu high=%llu overflowed=%d (expected 0 0 1)\n", (unsigned long long)r.low, (unsigned long long)r.high, (int)o);
    volatile std::uint64_t a = kMax, b = 1;
    r = wide_add(Uint128{a, a}, Uint128{b, 0}, o);
    std::printf("volatile inputs: low=%llu high=%llu overflowed=%d\n", (unsigned long long)r.low, (unsigned long long)r.high, (int)o);
    return 0;
}
