// Fork T1313: the expert-miss service (qwen4_exp ExpertMisses) answering replayed calls in place
// (MissOptions::in_place, --expert-host-graphs), where a device's copies queue behind one another
// (Windows WDDM, WSL2).
//
// A replayed decode call waits on the GPU (wait_kernel) for the service. Before T1313 the service's
// answer needed copies of its own (staged experts, admissions, the second stage's missing pairs,
// on its copy stream), while a pass keeps queueing work on the call's stream behind the waiting
// graph: the n-gram rows' upload before the PLE layer (executor stage_rows), the next segment's
// graph and its first-launch upload, the routes' copy to the host (record_routes). Where copies
// queue in one line, such a copy holds back the service's copies, which the wait needs.
//
// Every case here replays one call of the real service through a CUDA graph and queues exactly
// that work behind it: a second graph never launched before, a host-to-device copy and a
// device-to-host copy, all on the call's stream. In place, the call must return within kHang and
// be served: its GPU experts read where they are in host memory, nothing staged or admitted, the
// second stage's missing pairs in mapped host memory. A call whose experts are all cached posts no
// request and returns at once (the early return). An eager call is served the same way. In place,
// each launch is followed by the executor's submit_now() flush (cudaStreamQuery), and every replay
// must return within a second of the device's busy time: its time is printed.
//
// NINFER_HOST_GRAPHS_CONTROL=1 runs only the control: the staged service (no in-place answers)
// under the same replay. On a native Linux driver it returns; where copies serialize, a deadlock
// there is the cause T1313 removes, and the watchdog ends the process with exit code 3 instead of
// hanging. It holds the GPU in a spinning kernel until then: run it on purpose only.

#include "models/qwen4_exp/expert_misses.h"

#include "core/device.h"
#include "core/device_buffer.h"
#include "core/mapped_host_copy.h"
#include "core/pinned_memory.h"
#include "core/qtype.h"
#include "core/tensor.h"

#include "cuda_availability.h"

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::models::qwen4_exp;
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

constexpr int kHidden = 2560, kWidth = 640, kTop = 10, kExperts = 16;
// The device stays busy this long inside the graph before the call posts its request, so every
// piece of work queued behind the graph is queued while the request is still unanswered.
constexpr auto kBusy = 200ms;
// A replay that has not returned by then deadlocked.
constexpr auto kHang = 8s;
// Bytes of the queued copies: an n-gram rows upload is 16 rows a token.
constexpr std::size_t kQueuedBytes = 64 << 10;

int failures = 0;

void expect(bool condition, const std::string& label) {
    if (condition) { return; }
    std::cerr << "expectation failed: " << label << '\n';
    ++failures;
}

__global__ void busy_kernel(unsigned long long nanoseconds) {
    unsigned long long start = 0, now = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    do {
        __nanosleep(100000);
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    } while (now - start < nanoseconds);
}

// Reads the call's missing pairs as its second stage does, wherever they are, into `out`.
__global__ void read_pairs(const std::int32_t* missing, std::int32_t* out, int pairs) {
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) { out[p] = missing[p]; }
}

__global__ void touch(std::int32_t* word) { *word += 1; }

// Ends the process when a step does not return in time: a deadlocked wait never returns.
class Watchdog {
public:
    Watchdog(std::string step, int exit_code)
        : step_(std::move(step)), exit_code_(exit_code), thread_([this] { watch(); }) {}
    ~Watchdog() {
        {
            std::lock_guard lock(mutex_);
            done_ = true;
        }
        done_changed_.notify_all();
        thread_.join();
    }
    Watchdog(const Watchdog&)            = delete;
    Watchdog& operator=(const Watchdog&) = delete;

private:
    void watch() {
        std::unique_lock lock(mutex_);
        if (!done_changed_.wait_for(lock, kHang, [this] { return done_; })) {
            std::cerr << "FAIL: " << step_ << " did not return within " << kHang.count()
                      << " s (serialized copy queue: "
                      << (serialized_copy_queue() ? "yes" : "no") << ")\n";
            std::_Exit(exit_code_);
        }
    }

    std::string step_;
    int exit_code_;
    std::mutex mutex_;
    std::condition_variable done_changed_;
    bool done_ = false;
    std::thread thread_; // last: the members it reads exist before it starts
};

// One layer of 16 Q8_0 experts at Flash-Next's shapes in pinned host memory, every byte zero, one
// matrix per projection that every expert shares. Uncached: the gate table names no cached
// storage. Cached: every expert's gate entry lies in the layer's resident range.
struct Fixture {
    std::array<std::unique_ptr<PinnedHostBuffer>, 3> matrices;
    DeviceBuffer gate_table, storage, ids, input, pairs_read;
    std::unique_ptr<ExpertMisses> misses;

    Fixture(DeviceContext& device, double cpu_share, bool in_place, bool cached) {
        MissLayer layer;
        layer.rank    = 0;
        layer.experts = kExperts;
        const std::array<int, 3> rows = {kWidth, kWidth, kHidden}, ks = {kHidden, kHidden, kWidth};
        for (std::size_t p = 0; p < 3; ++p) {
            auto& projection        = layer.projections[p];
            projection.format       = QType::GGUF_Q8_0;
            projection.row_bytes    = std::int64_t(ks[p] / 32) * 34;
            projection.rows         = rows[p];
            projection.k            = ks[p];
            projection.expert_bytes = projection.row_bytes * rows[p];
            matrices[p] = std::make_unique<PinnedHostBuffer>(std::size_t(projection.expert_bytes));
            std::memset(matrices[p]->data(), 0, matrices[p]->size());
            projection.host.assign(kExperts, matrices[p]->data());
        }
        gate_table = DeviceBuffer(kExperts * sizeof(void*));
        if (cached) {
            storage = DeviceBuffer(kExperts * 256);
            const auto base = reinterpret_cast<std::uintptr_t>(storage.p);
            std::array<const void*, kExperts> entries{};
            for (int e = 0; e < kExperts; ++e) {
                entries[std::size_t(e)] = reinterpret_cast<const void*>(base + std::uintptr_t(e) * 256);
            }
            gate_table.copy_from_host(entries.data(), sizeof(entries));
            layer.resident[0] = {base, base + std::uintptr_t(kExperts) * 256};
        } else {
            gate_table.fill(0);
        }
        layer.gate_table = static_cast<const void* const*>(gate_table.p);
        std::array<std::int32_t, kTop> routed{};
        for (int i = 0; i < kTop; ++i) { routed[std::size_t(i)] = i; }
        ids = DeviceBuffer(sizeof(routed));
        ids.copy_from_host(routed.data(), sizeof(routed));
        input = DeviceBuffer(kHidden * 2);
        input.fill(0);
        pairs_read = DeviceBuffer(kTop * sizeof(std::int32_t));
        pairs_read.fill(0);
        MissOptions options;
        options.cpu_share   = cpu_share;
        options.cpu_threads = 2;
        options.in_place    = in_place;
        std::vector<MissLayer> layers;
        layers.push_back(std::move(layer));
        misses = std::make_unique<ExpertMisses>(device, std::move(layers),
                                                std::vector<std::filesystem::path>{}, options,
                                                nullptr);
    }

    // The call, enqueued on `stream` (captured when the stream captures): the split, the wait for
    // the service, then the second stage's read of the missing pairs.
    ExpertMisses::Call call(cudaStream_t stream) {
        const Tensor routed(ids.p, DType::I32, {kTop, 1});
        const Tensor m(input.p, DType::BF16, {kHidden, 1});
        auto call = misses->begin(0, routed, m, 1, ops::GgufMoeWeights{}, stream);
        misses->await(0, stream);
        read_pairs<<<1, 32, 0, stream>>>(static_cast<const std::int32_t*>(call.missing_ids.data),
                                         static_cast<std::int32_t*>(pairs_read.p), kTop);
        CUDA_CHECK(cudaGetLastError());
        return call;
    }
};

// What the service did for the last call, read once the stream is idle.
void check_served(const Fixture& f, const ExpertMisses::Call& call, const MissStats& before,
                  bool cpu, bool in_place, const std::string& label) {
    const MissStats after = f.misses->stats();
    const auto computed   = after.computed - before.computed;
    const auto staged     = after.staged - before.staged;
    const auto mapped     = after.mapped - before.mapped;
    expect(after.requests == before.requests + 1, label + ": the request was served");
    expect(computed == (cpu ? 5U : 0U), label + ": the CPU computed its share");
    expect(computed + staged + mapped == std::uint64_t(kTop),
           label + ": every missing expert was served");
    if (in_place) {
        expect(staged == 0 && after.admitted == before.admitted,
               label + ": in place, nothing staged or admitted");
        cudaPointerAttributes where{};
        CUDA_CHECK(cudaPointerGetAttributes(&where, call.missing_ids.data));
        expect(where.type == cudaMemoryTypeHost,
               label + ": in place, the missing pairs live in host memory");
    }
    std::array<std::int32_t, kTop> pairs{};
    CUDA_CHECK(cudaMemcpy(pairs.data(), f.pairs_read.p, sizeof(pairs), cudaMemcpyDeviceToHost));
    std::uint64_t on_gpu = 0;
    for (int p = 0; p < kTop; ++p) {
        const std::int32_t id = pairs[std::size_t(p)];
        expect(id == -1 || id == p, label + ": a missing pair keeps its expert");
        on_gpu += id == p;
        if (in_place && id == p) {
            // The second stage reads the expert where it is: its host bytes.
            for (int k = 0; k < 3; ++k) {
                const void* entry = call.missing_banks.gate.experts[std::size_t(k) * 512 + std::size_t(p)];
                expect(entry == f.matrices[std::size_t(k)]->data(),
                       label + ": in place, a GPU expert is read from host memory");
            }
        }
    }
    expect(on_gpu == staged + mapped, label + ": the second stage takes the GPU's share");
    expect(*call.count == std::int32_t(computed), label + ": one CPU product a computed pair");
    f.misses->raise_failure();
}

// Work a pass queues on the call's stream behind its replayed graph: the next segment's graph
// (never launched, so its launch uploads it), the n-gram rows' upload, the routes' copy back.
struct QueuedBehind {
    cudaGraph_t graph    = nullptr;
    cudaGraphExec_t exec = nullptr;
    DeviceBuffer word, device_bytes;
    std::unique_ptr<PinnedHostBuffer> host_up, host_down;

    explicit QueuedBehind(cudaStream_t stream)
        : word(sizeof(std::int32_t)), device_bytes(kQueuedBytes),
          host_up(std::make_unique<PinnedHostBuffer>(kQueuedBytes)),
          host_down(std::make_unique<PinnedHostBuffer>(kQueuedBytes)) {
        word.fill(0);
        std::memset(host_up->data(), 7, kQueuedBytes);
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        touch<<<1, 1, 0, stream>>>(static_cast<std::int32_t*>(word.p));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    }
    ~QueuedBehind() {
        if (exec) { cudaGraphExecDestroy(exec); }
        if (graph) { cudaGraphDestroy(graph); }
    }
    QueuedBehind(const QueuedBehind&)            = delete;
    QueuedBehind& operator=(const QueuedBehind&) = delete;

    void enqueue(cudaStream_t stream) {
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
        CUDA_CHECK(cudaMemcpyAsync(device_bytes.p, host_up->data(), kQueuedBytes,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(host_down->data(), device_bytes.p, kQueuedBytes,
                                   cudaMemcpyDeviceToHost, stream));
    }
    void check(int launches, const std::string& label) const {
        std::int32_t touched = 0;
        CUDA_CHECK(cudaMemcpy(&touched, word.p, sizeof(touched), cudaMemcpyDeviceToHost));
        expect(touched == launches, label + ": the graph queued behind ran");
        const auto* down = static_cast<const unsigned char*>(host_down->data());
        expect(down[0] == 7 && down[kQueuedBytes - 1] == 7,
               label + ": the copies queued behind landed");
    }
};

// The call replayed from a graph `replays` times, the pass's work queued behind each launch.
// `served`: the call posts a request (some expert uncached).
void replay_with_queued_work(Fixture& f, cudaStream_t stream, bool cpu, bool in_place, bool served,
                             int exit_code, const std::string& label) {
    cudaGraph_t graph    = nullptr;
    cudaGraphExec_t exec = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    busy_kernel<<<1, 1, 0, stream>>>(
        static_cast<unsigned long long>(std::chrono::nanoseconds(kBusy).count()));
    CUDA_CHECK(cudaGetLastError());
    const auto call = f.call(stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    QueuedBehind behind(stream);
    constexpr int kReplays = 2;
    for (int replay = 0; replay < kReplays; ++replay) {
        const std::string step = label + ", replay " + std::to_string(replay + 1);
        const MissStats before = f.misses->stats();
        f.misses->keep_alive(); // as a pass does before it launches its graphs
        const auto started = Clock::now();
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
        if (in_place) {
            // As the executor's submit_now() under --expert-host-graphs: hand the replay to the
            // device now, before the host goes on (the service answering in place never enters
            // the driver). The control runs as the executor did before T1313, without it.
            const cudaError_t state = cudaStreamQuery(stream);
            if (state == cudaErrorNotReady) {
                (void)cudaGetLastError();
            } else {
                CUDA_CHECK(state);
            }
        }
        behind.enqueue(stream); // returns while the graph is still busy: all of it waits behind
        {
            Watchdog guard(step + ": a replayed wait with work queued behind it", exit_code);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        const auto waited = Clock::now() - started;
        std::cout << step << ": returned in "
                  << std::chrono::duration<double, std::milli>(waited).count() << " ms\n";
        // The device is busy kBusy first; a service that saw the request late shows here.
        expect(waited < kBusy + 1s, step + ": the request was answered at once");
        if (served) {
            check_served(f, call, before, cpu, in_place, step);
        } else {
            const MissStats after = f.misses->stats();
            expect(after.requests == before.requests, step + ": a cached call posts no request");
            std::array<std::int32_t, kTop> pairs{};
            CUDA_CHECK(cudaMemcpy(pairs.data(), f.pairs_read.p, sizeof(pairs), cudaMemcpyDeviceToHost));
            for (const auto id : pairs) { expect(id == -1, step + ": a cached call has no missing pair"); }
            f.misses->raise_failure();
        }
    }
    behind.check(kReplays, label);
    CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
}

// The same call eager: the host waits (ExpertMisses::await), answered in place.
void eager(Fixture& f, cudaStream_t stream, bool cpu, const std::string& label) {
    const MissStats before = f.misses->stats();
    ExpertMisses::Call call;
    {
        Watchdog guard(label + ": an eager call", 1);
        call = f.call(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    check_served(f, call, before, cpu, true, label);
}

int run() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (ninfer::test::cuda_unavailable(status) || (status == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(status);
    DeviceContext device;
    RankBinding bind(device, 0);
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::cout << "serialized copy queue: " << (serialized_copy_queue() ? "yes" : "no") << '\n';
    const char* control = std::getenv("NINFER_HOST_GRAPHS_CONTROL");
    if (control != nullptr && std::string(control) == "1") {
        Fixture f(device, 0.0, false, false);
        replay_with_queued_work(f, stream, false, false, true, 3, "control: staged service");
        CUDA_CHECK(cudaStreamDestroy(stream));
        std::cout << "control: the staged service returned with work queued behind its wait"
                  << (serialized_copy_queue() ? " (not reproduced on this driver)" : "") << '\n';
        return failures == 0 ? 0 : 1;
    }
    for (const bool cpu : {false, true}) {
        const std::string label = std::string("in place, ") + (cpu ? "CPU share" : "GPU only");
        Fixture f(device, cpu ? 0.5 : 0.0, true, false);
        expect(f.misses->cpu_enabled() == cpu, label + ": the CPU share as configured");
        replay_with_queued_work(f, stream, cpu, true, true, 1, label + ", replayed");
        eager(f, stream, cpu, label + ", eager");
    }
    {
        Fixture f(device, 0.0, true, true);
        replay_with_queued_work(f, stream, false, true, false, 1, "in place, all cached (early return)");
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    if (failures != 0) { return 1; }
    std::cout << "PASS qwen4_exp expert misses host graphs\n";
    return 0;
}

} // namespace

int main() {
    try {
        return run();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
