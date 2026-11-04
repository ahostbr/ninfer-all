// CPU checks of the DeepSeek-V4.1 kernel A/B switches' parsing (src/models/deepseek_v41/
// kernel_switches.h): the values an environment variable may carry, that a bad one throws and names
// the variable, that an unset or empty one leaves the flag, and that the environment wins over it.
// No device is used.

#include "models/deepseek_v41/kernel_switches.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using namespace ninfer::models::deepseek_v41;

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::cerr << "FAIL " << what << '\n';
        ++failures;
    }
}

template <typename F>
bool throws_naming(F&& f, const std::string& name) {
    try {
        f();
    } catch (const std::invalid_argument& e) { return std::string(e.what()).find(name) != std::string::npos; }
    return false;
}

} // namespace

int main() {
    for (const char* on : {"1", "on", "ON", "On", "true", "TRUE", "True"}) {
        check(parse_switch(on) == std::optional<bool>(true), "an on value did not parse as on");
    }
    for (const char* off : {"0", "off", "OFF", "false", "False", "FALSE"}) {
        check(parse_switch(off) == std::optional<bool>(false), "an off value did not parse as off");
    }
    for (const char* bad : {"", "2", "yes", "no", "y", "onn", " on", "on ", "-1", "tru", "enabled"}) {
        check(!parse_switch(bad).has_value(), "a bad value parsed");
    }

    // Unset and empty leave the option; a value overrides it either way.
    check(resolve_switch(nullptr, "X", false) == false && resolve_switch(nullptr, "X", true) == true,
          "an unset variable changed the flag");
    check(resolve_switch("", "X", false) == false && resolve_switch("", "X", true) == true,
          "an empty variable changed the flag");
    check(resolve_switch("1", "X", false) == true && resolve_switch("off", "X", true) == false,
          "the environment did not override the flag");
    check(throws_naming([] { (void)resolve_switch("maybe", "NINFER_DS_IQ1M_MATRIX", false); },
                        "NINFER_DS_IQ1M_MATRIX"),
          "a bad value did not throw naming its variable");

    // The defaults are all off.
    const KernelSwitches defaults;
    check(!defaults.iq1m_matrix, "a kernel switch is on by default");
    if (failures == 0) { std::cout << "kernel switches: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}
