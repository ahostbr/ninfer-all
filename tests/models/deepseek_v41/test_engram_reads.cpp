// CPU checks of the Engram read path's speed experiments (T0833): the switches' parsing and the
// sizing rule (engram_tuning.h), the n-gram reader's dedupe and skip rows (a repeated row is read
// once and every output byte is unchanged), the background reads of a pass (engram_flight.h: jobs in
// order, a block's slot completes when its read has, a failure is kept for the executor's thread),
// and a two-block pass read through the fetcher against the reader's synchronous bytes. No device is
// used.

#include "models/deepseek_v41/engram_flight.h"
#include "models/deepseek_v41/engram_tuning.h"
#include "models/qwen4_exp/ngram_table.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::models::deepseek_v41;
using ninfer::models::qwen4_exp::kNgramSkipRow;
using ninfer::models::qwen4_exp::NgramReadOptions;
using ninfer::models::qwen4_exp::NgramTableLayout;
using ninfer::models::qwen4_exp::NgramTableReader;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template <class F>
bool throws_invalid(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) { return true; } catch (...) {}
    return false;
}

// ---- engram_tuning.h -----------------------------------------------------------------------------

void test_tuning() {
    for (const char* on : {"1", "on", "true"}) { check(parse_switch("X", on), std::string("on: ") + on); }
    for (const char* off : {"0", "off", "false"}) {
        check(!parse_switch("X", off), std::string("off: ") + off);
    }
    for (const char* bad : {"", "yes", "no", "ON", "True", "2", "-1", "of", "pass"}) {
        check(throws_invalid([&] { (void)parse_switch("X", bad); }),
              std::string("a malformed switch was accepted: '") + bad + "'");
    }
    check(parse_overlap("X", "ahead") == EngramOverlap::Ahead, "ahead");
    check(parse_overlap("X", "pass") == EngramOverlap::Pass, "pass");
    check(parse_overlap("X", "on") == EngramOverlap::Pass, "on is pass");
    check(parse_overlap("X", "1") == EngramOverlap::Pass, "1 is pass");
    check(parse_overlap("X", "off") == EngramOverlap::Off, "overlap off");
    check(parse_overlap("X", "false") == EngramOverlap::Off, "overlap false");
    check(throws_invalid([] { (void)parse_overlap("X", "aheaad"); }), "a malformed overlap");

    std::map<std::string, std::string> environment;
    const auto lookup = [&](const char* name) -> const char* {
        const auto found = environment.find(name);
        return found == environment.end() ? nullptr : found->second.c_str();
    };
    EngramTuning base;
    check(!base.dedupe && base.overlap == EngramOverlap::Off && !base.refill && !base.default_cache,
          "the defaults are all off");
    auto resolved = resolve_engram_tuning(base, lookup);
    check(!resolved.dedupe && resolved.overlap == EngramOverlap::Off && !resolved.refill &&
              !resolved.default_cache,
          "no environment leaves the options alone");
    environment = {{"NINFER_DS_ENGRAM_DEDUPE", "1"},
                   {"NINFER_DS_ENGRAM_OVERLAP", "ahead"},
                   {"NINFER_FILE_READ_REFILL", "true"},
                   {"NINFER_DS_ENGRAM_DEFAULT_CACHE", "on"}};
    resolved = resolve_engram_tuning(base, lookup);
    check(resolved.dedupe && resolved.overlap == EngramOverlap::Ahead && resolved.refill &&
              resolved.default_cache,
          "the environment turns every experiment on");
    EngramTuning all{true, EngramOverlap::Ahead, true, true};
    environment = {{"NINFER_DS_ENGRAM_DEDUPE", "0"},
                   {"NINFER_DS_ENGRAM_OVERLAP", "off"},
                   {"NINFER_FILE_READ_REFILL", "false"},
                   {"NINFER_DS_ENGRAM_DEFAULT_CACHE", "off"}};
    resolved = resolve_engram_tuning(all, lookup);
    check(!resolved.dedupe && resolved.overlap == EngramOverlap::Off && !resolved.refill &&
              !resolved.default_cache,
          "the environment turns every experiment off over the options");
    environment = {{"NINFER_DS_ENGRAM_DEDUPE", ""}};
    resolved = resolve_engram_tuning(all, lookup);
    check(resolved.dedupe, "an empty variable leaves the option alone");
    environment = {{"NINFER_FILE_READ_REFILL", "maybe"}};
    check(throws_invalid([&] { (void)resolve_engram_tuning(base, lookup); }),
          "a malformed variable was accepted");

    constexpr std::uint64_t MiB = 1ULL << 20, GiB = 1ULL << 30;
    check(default_engram_cache_bytes(0) == 16 * MiB, "unknown memory takes the floor");
    check(default_engram_cache_bytes(512 * MiB) == 16 * MiB, "a small host takes the floor");
    check(default_engram_cache_bytes(4 * GiB) == 64 * MiB, "a 64th of 4 GiB");
    check(default_engram_cache_bytes(96 * GiB) == 256 * MiB, "a large host takes the cap");
    check(default_engram_cache_bytes(16 * GiB) == 256 * MiB, "16 GiB reaches the cap");
    check(default_engram_cache_bytes(15 * GiB) < 256 * MiB, "15 GiB is under the cap");

    {
        // The device's block-major layout from the hasher's token-major rows, with an image
        // token's rows skipped only when asked.
        constexpr std::size_t t = 5, columns = 3, blocks = 2, per = blocks * columns;
        std::vector<std::uint64_t> scratch(t * per);
        for (std::size_t r = 0; r < t; ++r) {
            for (std::size_t j = 0; j < per; ++j) { scratch[r * per + j] = 1000 * r + j; }
        }
        const std::vector<std::uint8_t> dead{0, 0, 1, 0, 1};
        constexpr std::uint64_t skip = ~std::uint64_t{0};
        for (const bool skip_dead : {false, true}) {
            std::vector<std::uint64_t> out(t * per, 7);
            layout_engram_ids(scratch, t, per, columns, dead, skip_dead, skip, out.data());
            for (std::size_t b = 0; b < blocks; ++b) {
                for (std::size_t r = 0; r < t; ++r) {
                    for (std::size_t c = 0; c < columns; ++c) {
                        const std::uint64_t want = skip_dead && dead[r] ? skip : 1000 * r + b * columns + c;
                        check(out[(b * t + r) * columns + c] == want, "an id is misplaced in the layout");
                    }
                }
            }
        }
        std::vector<std::uint64_t> out(t * per);
        layout_engram_ids(scratch, t, per, columns, {}, true, skip, out.data());
        check(std::count(out.begin(), out.end(), skip) == 0, "no dead flags skip nothing");
    }
    const std::vector<std::int32_t> history{3, 4}, tokens{5, 6, 7};
    const auto key = engram_batch_key(history, tokens);
    check(key != 0, "a key is never 0");
    check(key == engram_batch_key(history, tokens), "a key is deterministic");
    check(engram_batch_key({}, {}) != 0, "the empty key is not 0");
    check(key != engram_batch_key(history, std::vector<std::int32_t>{5, 6, 8}), "a token changes the key");
    check(key != engram_batch_key(std::vector<std::int32_t>{3, 5}, tokens), "a history id changes the key");
    check(key != engram_batch_key(std::vector<std::int32_t>{3}, std::vector<std::int32_t>{4, 5, 6, 7}),
          "the history/tokens boundary is part of the key");
    check(key != engram_batch_key(history, std::vector<std::int32_t>{5, 6}), "a length changes the key");
}

// ---- The reader's dedupe and skip rows -----------------------------------------------------------

constexpr std::uint32_t kRowBytes = 144;
constexpr std::uint64_t kRows     = 3000;

struct Table {
    std::filesystem::path path;
    std::vector<std::uint8_t> payload;
    Table() : payload(kRows * kRowBytes) {
        std::mt19937 random(8051u);
        for (auto& byte : payload) { byte = static_cast<std::uint8_t>(random()); }
        path = std::filesystem::current_path() /
               ("ninfer_engram_reads_" + std::to_string(std::random_device{}()) + ".bin");
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(payload.data()),
                   static_cast<std::streamsize>(payload.size()));
    }
    ~Table() {
        std::error_code error;
        std::filesystem::remove(path, error);
    }
    [[nodiscard]] NgramTableLayout layout() const {
        return NgramTableLayout{{{path, 0, kRows * kRowBytes}}, kRowBytes, kRows};
    }
    // The bytes a batch must read back as: each row's, zeros for a skipped one.
    [[nodiscard]] std::vector<std::uint8_t> expected(const std::vector<std::uint64_t>& ids) const {
        std::vector<std::uint8_t> out(ids.size() * kRowBytes, 0);
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] != kNgramSkipRow) {
                std::memcpy(out.data() + i * kRowBytes, payload.data() + ids[i] * kRowBytes, kRowBytes);
            }
        }
        return out;
    }
};

// A batch of n ids drawn from `distinct` rows, every `skip_every`-th one a skip row.
std::vector<std::uint64_t> batch(std::size_t n, std::size_t distinct, std::size_t skip_every,
                                 std::uint32_t seed) {
    std::mt19937 random(seed);
    std::vector<std::uint64_t> pool;
    for (std::size_t i = 0; i < distinct; ++i) { pool.push_back((i * 7919 + seed) % kRows); }
    std::vector<std::uint64_t> ids(n);
    for (std::size_t i = 0; i < n; ++i) {
        ids[i] = skip_every != 0 && i % skip_every == skip_every - 1 ? kNgramSkipRow
                                                                      : pool[random() % pool.size()];
    }
    return ids;
}

void test_reader() {
    Table table;
    for (const auto io : {NgramIo::Buffered, NgramIo::Mapped}) {
        for (const bool dedupe : {false, true}) {
            for (const bool refill : {false, true}) {
                for (const std::uint64_t budget : {std::uint64_t{0}, std::uint64_t{256} << 10}) {
                    const std::string label = std::string("io=") + (io == NgramIo::Mapped ? "mmap" : "buffered") +
                                              " dedupe=" + (dedupe ? "1" : "0") +
                                              " refill=" + (refill ? "1" : "0") +
                                              " cache=" + std::to_string(budget);
                    NgramReadOptions options;
                    options.io           = io;
                    options.dedupe       = dedupe;
                    options.refill       = refill;
                    options.budget_bytes = budget;
                    options.depth        = 8;
                    NgramTableReader reader(table.layout(), options);
                    std::uint64_t rows = 0, repeats = 0, skipped = 0;
                    // 1500 rows go to the worker threads, 300 are read inline; some are skips.
                    const std::vector<std::array<std::size_t, 3>> shapes{
                        {1500, 200, 11}, {300, 40, 0}, {64, 64, 5}, {1, 1, 0}, {40, 1, 0}, {0, 1, 0}};
                    for (const auto& [n, distinct, skip_every] : shapes) {
                        const auto ids = batch(n, distinct, skip_every, std::uint32_t(n + 5));
                        std::vector<std::uint8_t> out(ids.size() * kRowBytes, 0xA5);
                        reader.read_rows(ids, out);
                        check(out == table.expected(ids), "bytes differ: " + label);
                        std::set<std::uint64_t> seen;
                        for (const auto id : ids) {
                            if (id == kNgramSkipRow) {
                                ++skipped;
                            } else {
                                ++rows;
                                repeats += !seen.insert(id).second;
                            }
                        }
                    }
                    const auto counters = reader.counters();
                    check(counters.rows == rows, "rows count addressed rows only: " + label);
                    check(reader.skipped_rows() == skipped, "skipped rows counted: " + label);
                    check(reader.deduped_rows() == (dedupe ? repeats : 0),
                          "deduped rows counted: " + label);
                    if (dedupe && budget == 0) {
                        // Every repeat was served from its first copy, so only the distinct rows
                        // of each batch came from the file.
                        check(counters.rows - counters.resident_rows == rows - repeats,
                              "file reads are the distinct rows: " + label);
                    }
                    if (!dedupe && budget == 0) {
                        check(counters.resident_rows == 0, "nothing is served from RAM: " + label);
                    }
                }
            }
        }
    }

    // A skip row is not an out-of-range row; a real one past the table still is.
    NgramReadOptions options;
    options.dedupe = true;
    NgramTableReader reader(table.layout(), options);
    std::vector<std::uint8_t> out(kRowBytes, 0xFF);
    const std::vector<std::uint64_t> only_skip{kNgramSkipRow};
    reader.read_rows(only_skip, out);
    check(std::all_of(out.begin(), out.end(), [](std::uint8_t b) { return b == 0; }),
          "a skip row reads as zeros");
    bool refused = false;
    try {
        const std::vector<std::uint64_t> past{kRows};
        reader.read_rows(past, out);
    } catch (const std::out_of_range&) { refused = true; }
    check(refused, "a row past the table was not refused");
    // The reader still works after a refused batch.
    const std::vector<std::uint64_t> again{7, 7, 9};
    std::vector<std::uint8_t> three(3 * kRowBytes);
    reader.read_rows(again, three);
    check(three == table.expected(again), "a batch after a refusal reads wrongly");
}

// ---- Background reads: the slot and the fetcher --------------------------------------------------

void test_flight() {
    {
        EngramSlot slot;
        check(slot.done(), "a slot nothing was queued on is done");
        slot.wait(); // returns at once
        check(slot.take_error() == nullptr, "no error");
    }
    {
        EngramFetcher fetcher;
        std::mutex mutex;
        std::vector<int> order;
        std::vector<std::unique_ptr<EngramSlot>> slots;
        for (int i = 0; i < 50; ++i) { slots.push_back(std::make_unique<EngramSlot>()); }
        for (int i = 0; i < 50; ++i) {
            fetcher.post(*slots[std::size_t(i)], [&, i] {
                std::lock_guard lock(mutex);
                order.push_back(i);
            });
        }
        for (auto& slot : slots) { slot->wait(); }
        check(order.size() == 50 && std::is_sorted(order.begin(), order.end()),
              "jobs run in the order they were posted");
        for (auto& slot : slots) { check(slot->take_error() == nullptr, "no job failed"); }
    }
    {
        // A waiter blocks until its job has run, and sees what the job wrote.
        EngramFetcher fetcher;
        EngramSlot gate_slot, slot;
        std::promise<void> release;
        auto released = release.get_future().share();
        int written   = 0;
        fetcher.post(gate_slot, [released] { released.wait(); });
        fetcher.post(slot, [&] { written = 42; });
        std::atomic<bool> woke{false};
        std::thread waiter([&] {
            slot.wait();
            woke = true;
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        check(!woke.load() && !slot.done(), "a waiter ran ahead of its job");
        release.set_value();
        waiter.join();
        check(woke.load() && written == 42, "the waiter did not see the job's write");
    }
    {
        // A failing job completes its slot, keeps the error once, and does not stop the thread.
        EngramFetcher fetcher;
        EngramSlot bad, good;
        fetcher.post(bad, [] { throw std::runtime_error("read failed"); });
        fetcher.post(good, [] {});
        bad.wait();
        good.wait();
        const auto error = bad.take_error();
        check(error != nullptr, "the failure was lost");
        bool same = false;
        try {
            std::rethrow_exception(error);
        } catch (const std::runtime_error& e) { same = std::string(e.what()) == "read failed"; }
        check(same, "the failure changed");
        check(bad.take_error() == nullptr, "a failure is taken once");
        check(good.take_error() == nullptr, "the next job was spoiled");
        // A slot is reusable.
        int ran = 0;
        fetcher.post(bad, [&] { ++ran; });
        bad.wait();
        check(ran == 1 && bad.take_error() == nullptr, "a failed slot did not reset");
    }
    {
        // The destructor runs what is queued.
        std::atomic<int> ran{0};
        {
            EngramFetcher fetcher;
            for (int i = 0; i < 20; ++i) { fetcher.post([&] { ++ran; }); }
        }
        check(ran.load() == 20, "the fetcher dropped queued jobs");
    }
}

// A pass's two blocks read through the fetcher, block 0 first, while another thread (the stream's
// host function) waits for each: the bytes are those of a synchronous read.
void test_two_block_pass() {
    Table table;
    NgramReadOptions options;
    options.dedupe = true;
    options.depth  = 8;
    NgramTableReader reader(table.layout(), options);
    constexpr std::size_t t = 700, columns = 6, block_rows = t * columns, blocks = 2;
    const auto ids = batch(blocks * block_rows, 500, 13, 99);
    std::vector<std::uint8_t> rows(ids.size() * kRowBytes, 0x5A);
    EngramFetcher fetcher;
    std::array<EngramSlot, blocks> slots;
    std::atomic<int> started{0};
    std::vector<int> finished;
    std::mutex mutex;
    for (std::size_t b = 0; b < blocks; ++b) {
        fetcher.post(slots[b], [&, b] {
            check(std::size_t(started.fetch_add(1)) == b, "block jobs started out of order");
            reader.read_rows({ids.data() + b * block_rows, block_rows},
                             {rows.data() + b * block_rows * kRowBytes, block_rows * kRowBytes});
            std::lock_guard lock(mutex);
            finished.push_back(int(b));
        });
    }
    std::array<bool, blocks> correct{};
    std::vector<std::thread> consumers;
    for (std::size_t b = 0; b < blocks; ++b) {
        consumers.emplace_back([&, b] {
            slots[b].wait();
            const std::vector<std::uint64_t> mine(ids.begin() + std::ptrdiff_t(b * block_rows),
                                                  ids.begin() + std::ptrdiff_t((b + 1) * block_rows));
            const auto want = table.expected(mine);
            correct[b]      = std::equal(want.begin(), want.end(),
                                         rows.begin() + std::ptrdiff_t(b * block_rows * kRowBytes));
        });
    }
    for (auto& c : consumers) { c.join(); }
    check(correct[0] && correct[1], "a block's rows differ from the synchronous read");
    check(finished == std::vector<int>({0, 1}), "blocks finished out of order");
}

} // namespace

int main() {
    try {
        test_tuning();
        test_reader();
        test_flight();
        test_two_block_pass();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: exception: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0) { std::cout << "engram reads PASS\n"; }
    return failures == 0 ? 0 : 1;
}
