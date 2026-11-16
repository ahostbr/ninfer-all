#include "core/cpu_spin_pool.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <set>
#include <string>
#include <utility>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif
#ifdef __linux__
#    include <fstream>
#    include <pthread.h>
#    include <sched.h>
#endif

namespace ninfer {
namespace {

#ifdef __linux__
std::vector<int> allowed_cpus() {
    std::vector<int> cpus;
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (CPU_ISSET(cpu, &set)) { cpus.push_back(cpu); }
        }
    }
    if (cpus.empty()) {
        for (unsigned cpu = 0; cpu < std::max(1U, std::thread::hardware_concurrency()); ++cpu) {
            cpus.push_back(int(cpu));
        }
    }
    return cpus;
}

long topology(int cpu, const char* field) {
    std::ifstream in("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/" + field);
    long value = -1;
    if (!(in >> value)) { return -1; }
    return value;
}

// The container's CPU quota in CPUs, or 0 without one.
double cpu_quota() {
    double quota = -1, period = 0;
    if (std::ifstream v2("/sys/fs/cgroup/cpu.max"); v2) {
        std::string limit;
        if (v2 >> limit >> period && limit != "max") { quota = std::strtod(limit.c_str(), nullptr); }
    } else if (std::ifstream q("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
               p("/sys/fs/cgroup/cpu/cpu.cfs_period_us");
               q && p) {
        if (!(q >> quota && p >> period)) { quota = -1; }
    }
    return quota > 0 && period > 0 ? quota / period : 0;
}

void pin_current(int cpu) {
    if (cpu < 0 || cpu >= CPU_SETSIZE) { return; }
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    // A refusal (a mask the container does not allow) leaves the thread where the OS puts it.
    (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}
#elif defined(_WIN32)
void pin_current(int cpu) {
    if (cpu < 0 || cpu >= 64) { return; }
    (void)SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << cpu);
}
#else
void pin_current(int) {}
#endif

} // namespace

unsigned usable_cpus() {
    unsigned count = std::max(1U, std::thread::hardware_concurrency());
#ifdef __linux__
    count = std::min(count, unsigned(allowed_cpus().size()));
    if (const double quota = cpu_quota(); quota > 0) {
        count = std::min(count, std::max(1U, unsigned(quota)));
    }
#endif
    return count;
}

std::vector<int> physical_core_cpus() {
    std::vector<int> out;
#ifdef __linux__
    std::set<std::pair<long, long>> seen;
    for (const int cpu : allowed_cpus()) {
        const long package = topology(cpu, "physical_package_id");
        const long core    = topology(cpu, "core_id");
        if (core < 0) {
            out.push_back(cpu); // no topology: every allowed CPU counts
            continue;
        }
        if (seen.emplace(package, core).second) { out.push_back(cpu); }
    }
#elif defined(_WIN32)
    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
    std::vector<char> buffer(length);
    if (length != 0 &&
        GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &length)) {
        for (DWORD at = 0; at < length;) {
            const auto* entry =
                reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data() + at);
            const GROUP_AFFINITY& group = entry->Processor.GroupMask[0];
            for (int bit = 0; bit < 64; ++bit) {
                if (group.Mask & (KAFFINITY(1) << bit)) {
                    out.push_back(int(group.Group) * 64 + bit);
                    break;
                }
            }
            at += entry->Size;
        }
    }
#endif
    if (out.empty()) {
        for (unsigned cpu = 0; cpu < std::max(1U, std::thread::hardware_concurrency()); ++cpu) {
            out.push_back(int(cpu));
        }
    }
    return out;
}

std::uint32_t pool_cpu_count() {
    const auto cores = std::uint32_t(physical_core_cpus().size());
    return std::max(1U, std::min(cores, usable_cpus() / 2));
}

std::vector<RowTask> row_tasks(std::span<const RowPart> parts, std::uint32_t threads, int quantum) {
    constexpr double kPerThread = 3;
    quantum                     = std::max(1, quantum);
    double total                = 0;
    for (const auto& part : parts) { total += double(std::max(part.rows, 0)) * part.cost; }
    std::vector<RowTask> tasks;
    if (!(total > 0)) { return tasks; }
    const double target = total / (kPerThread * double(std::max(1U, threads)));
    for (std::size_t i = 0; i < parts.size(); ++i) {
        const int rows = parts[i].rows;
        if (rows <= 0) { continue; }
        const long units = (rows + quantum - 1) / quantum;
        const long count =
            std::clamp(std::lround(double(rows) * parts[i].cost / target), 1L, units);
        for (long k = 0; k < count; ++k) {
            const int begin = int(units * k / count) * quantum;
            const int end   = std::min(rows, int(units * (k + 1) / count) * quantum);
            tasks.push_back({std::uint32_t(i), begin, end});
        }
    }
    return tasks;
}

void spin_hint() noexcept {
#if defined(_WIN32)
    YieldProcessor();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#else
    std::this_thread::yield();
#endif
}

int smt_sibling(int cpu) {
    if (cpu < 0) { return -1; }
#ifdef __linux__
    const long package = topology(cpu, "physical_package_id");
    const long core    = topology(cpu, "core_id");
    if (core < 0) { return -1; }
    for (const int other : allowed_cpus()) {
        if (other != cpu && topology(other, "physical_package_id") == package &&
            topology(other, "core_id") == core) {
            return other;
        }
    }
#elif defined(_WIN32)
    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
    std::vector<char> buffer(length);
    if (length != 0 &&
        GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &length)) {
        for (DWORD at = 0; at < length;) {
            const auto* entry =
                reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data() + at);
            const GROUP_AFFINITY& group = entry->Processor.GroupMask[0];
            if (int(group.Group) == cpu / 64 && (group.Mask & (KAFFINITY(1) << (cpu % 64)))) {
                for (int bit = 0; bit < 64; ++bit) {
                    if (bit != cpu % 64 && (group.Mask & (KAFFINITY(1) << bit))) {
                        return int(group.Group) * 64 + bit;
                    }
                }
                return -1;
            }
            at += entry->Size;
        }
    }
#endif
    return -1;
}

std::vector<int> pool_placement(std::uint32_t threads, PoolHostCore host,
                                std::span<const int> cores, int first_sibling) {
    const std::size_t count = std::max(1U, threads);
    // More threads than physical cores: the rest go unpinned rather than onto a busy core.
    const auto core_of = [&](std::size_t i) { return i < cores.size() ? cores[i] : -1; };
    std::vector<int> out(count, -1);
    if (host == PoolHostCore::Last && cores.size() >= 2) {
        out[0] = cores.back();
        for (std::size_t i = 1; i < count; ++i) {
            out[i] = i - 1 < cores.size() - 1 ? cores[i - 1] : -1;
        }
        return out;
    }
    if (host == PoolHostCore::Sibling && first_sibling >= 0 && !cores.empty()) {
        out[0] = first_sibling;
        for (std::size_t i = 1; i < count; ++i) { out[i] = core_of(i - 1); }
        return out;
    }
    for (std::size_t i = 0; i < count; ++i) { out[i] = core_of(i); }
    return out;
}

CpuSpinPool::CpuSpinPool(std::uint32_t threads, bool pin, PoolHostCore host) {
    std::vector<int> cpus(std::max(1U, threads), -1);
    if (pin) {
        const std::vector<int> cores = physical_core_cpus();
        const int sibling =
            host == PoolHostCore::Sibling && !cores.empty() ? smt_sibling(cores.front()) : -1;
        cpus    = pool_placement(threads, host, cores, sibling);
        pinned_ = true;
    }
    caller_cpu_ = cpus[0];
    for (std::size_t i = 1; i < cpus.size(); ++i) {
        const int cpu = cpus[i];
        workers_.emplace_back([this, cpu] { work(cpu); });
    }
}

CpuSpinPool::~CpuSpinPool() {
    {
        std::lock_guard lock(mutex_);
        wake_line_.stop.store(true);
    }
    wake_.notify_all();
    for (auto& worker : workers_) { worker.join(); }
}

void CpuSpinPool::spin(bool on) {
    {
        std::lock_guard lock(mutex_);
        wake_line_.spinning.store(on);
    }
    wake_.notify_all();
}

void CpuSpinPool::pin_caller() {
    if (!pinned_ || caller_ == std::this_thread::get_id()) { return; }
    caller_ = std::this_thread::get_id();
    pin_current(caller_cpu_);
}

void CpuSpinPool::drain() {
    std::uint64_t size = 0;
    std::size_t ran    = 0;
    for (;;) {
        const std::uint64_t ticket = ticket_.fetch_add(1, std::memory_order_acq_rel);
        const std::uint64_t now    = job_.size.load(std::memory_order_acquire);
        const std::size_t i        = std::size_t(ticket & 0xffffffffULL);
        if ((now >> 32) != (ticket >> 32) || i >= (now & 0xffffffffULL)) { break; }
        // A later run starts only once this part's jobs are counted, so every job of a part
        // belongs to one run.
        size = now;
        try {
            job_.call(job_.context, i);
        } catch (...) {
            std::lock_guard lock(failure_mutex_);
            if (!failure_) { failure_ = std::current_exception(); }
        }
        ++ran;
    }
    if (ran != 0) { finish(size, ran); }
}

void CpuSpinPool::finish(std::uint64_t size, std::size_t ran) {
    if (done_.fetch_add(ran, std::memory_order_acq_rel) + ran == (size & 0xffffffffULL)) {
        complete_.finished.store(std::chrono::steady_clock::now().time_since_epoch().count(),
                                 std::memory_order_relaxed);
        complete_.generation.store(size >> 32, std::memory_order_release);
    }
}

void CpuSpinPool::work(int cpu) {
    pin_current(cpu);
    std::uint64_t seen = 0;
    for (;;) {
        if (wake_line_.stop.load()) { return; }
        const std::uint64_t generation = wake_line_.posted.load(std::memory_order_acquire);
        if (generation != seen) {
            seen = generation;
            drain();
            continue;
        }
        if (wake_line_.spinning.load(std::memory_order_relaxed)) {
            spin_hint();
            continue;
        }
        std::unique_lock lock(mutex_);
        wake_.wait(lock, [&] {
            return wake_line_.stop.load() || wake_line_.spinning.load() ||
                   wake_line_.posted.load() != seen;
        });
    }
}

} // namespace ninfer
