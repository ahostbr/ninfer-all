// Fork change W5: CUDA Graphs stay off for host and disk experts where copies queue in one line.
// Fork T1313: --expert-host-graphs keeps them for host experts there, and only there.
#include "runtime/engine/host_wait_graphs.h"

#include <iostream>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    using ninfer::ExpertResidency;
    using ninfer::runtime::host_waits_need_eager;
    int failures = 0;
    failures += check(host_waits_need_eager(ExpertResidency::Host, true),
                      "host experts under serialized copies must run eager");
    failures += check(host_waits_need_eager(ExpertResidency::Disk, true),
                      "disk experts under serialized copies must run eager");
    failures += check(!host_waits_need_eager(ExpertResidency::Device, true),
                      "device experts never wait for the host and keep their graphs");
    failures += check(!host_waits_need_eager(ExpertResidency::Host, false),
                      "a native Linux driver keeps graphs with host experts");
    failures += check(!host_waits_need_eager(ExpertResidency::Disk, false),
                      "a native Linux driver keeps graphs with disk experts");
    // T1313: the opt-in.
    failures += check(!host_waits_need_eager(ExpertResidency::Host, true, true),
                      "--expert-host-graphs keeps graphs with host experts under serialized copies");
    failures += check(host_waits_need_eager(ExpertResidency::Disk, true, true),
                      "--expert-host-graphs leaves disk experts eager");
    failures += check(!host_waits_need_eager(ExpertResidency::Device, true, true),
                      "--expert-host-graphs changes nothing for device experts");
    failures += check(!host_waits_need_eager(ExpertResidency::Host, false, true),
                      "--expert-host-graphs changes nothing on a native Linux driver");
    failures += check(host_waits_need_eager(ExpertResidency::Host, true, false) ==
                          host_waits_need_eager(ExpertResidency::Host, true),
                      "the opt-in defaults off: W5 unchanged");
    static_assert(host_waits_need_eager(ExpertResidency::Host, true), "usable in constant expressions");
    static_assert(!host_waits_need_eager(ExpertResidency::Host, true, true),
                  "usable in constant expressions");
    if (failures == 0) { std::cout << "host wait graphs tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
