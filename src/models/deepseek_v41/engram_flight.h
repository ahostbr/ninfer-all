#pragma once

// T0833: the host side of overlapped Engram reads (EngramOverlap::Pass and Ahead), free of CUDA so
// it runs under a CPU test. A pass's rows are read on a background thread, one job per Engram
// block, in the order the blocks are needed; the stream waits for a block's slot just before the
// block's layer (a host function the executor enqueues), so the reads hide behind the layers
// before it.

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

namespace ninfer::models::deepseek_v41 {

// One block's reads of one pass: reset() when its job is queued, complete() by the thread that
// ran the job, wait() by whoever needs the rows. wait() never throws: it runs inside a CUDA host
// function, so a failed read is kept for take_error() and rethrown on the executor's thread.
class EngramSlot {
public:
    void reset() {
        std::lock_guard lock(mutex_);
        done_  = false;
        error_ = nullptr;
    }

    void complete(std::exception_ptr error = nullptr) noexcept {
        {
            std::lock_guard lock(mutex_);
            done_  = true;
            error_ = std::move(error);
        }
        ready_.notify_all();
    }

    void wait() noexcept {
        std::unique_lock lock(mutex_);
        ready_.wait(lock, [this] { return done_; });
    }

    [[nodiscard]] bool done() const {
        std::lock_guard lock(mutex_);
        return done_;
    }

    // The failure of a completed job, once: later calls return null.
    [[nodiscard]] std::exception_ptr take_error() {
        std::lock_guard lock(mutex_);
        return done_ ? std::exchange(error_, nullptr) : nullptr;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable ready_;
    bool done_ = true; // a slot nothing was queued on has nothing to wait for
    std::exception_ptr error_;
};

// A thread that runs the jobs posted to it, in order. The destructor runs what is queued first.
class EngramFetcher {
public:
    EngramFetcher() : thread_([this] { run(); }) {}

    ~EngramFetcher() {
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        wake_.notify_all();
        thread_.join();
    }

    EngramFetcher(const EngramFetcher&)            = delete;
    EngramFetcher& operator=(const EngramFetcher&) = delete;

    void post(std::function<void()> job) {
        {
            std::lock_guard lock(mutex_);
            jobs_.push_back(std::move(job));
        }
        wake_.notify_one();
    }

    // Queues `read` for `slot`: the slot completes when it returns, or with what it threw. The
    // slot is reset here, so a pass may wait on it from the moment it is queued.
    void post(EngramSlot& slot, std::function<void()> read) {
        slot.reset();
        try {
            post([&slot, read = std::move(read)] {
                try {
                    read();
                    slot.complete();
                } catch (...) { slot.complete(std::current_exception()); }
            });
        } catch (...) {
            slot.complete(std::current_exception());
            throw;
        }
    }

private:
    void run() {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock lock(mutex_);
                wake_.wait(lock, [this] { return stop_ || !jobs_.empty(); });
                if (jobs_.empty()) { return; }
                job = std::move(jobs_.front());
                jobs_.pop_front();
            }
            job();
        }
    }

    std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<std::function<void()>> jobs_;
    bool stop_ = false;
    std::thread thread_; // last: starts running only once the rest exists
};

} // namespace ninfer::models::deepseek_v41
