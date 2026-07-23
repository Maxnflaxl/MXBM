#pragma once
#include <condition_variable>
#include <mutex>
#include <thread>

#include "miner/stats.h"

namespace mxbm { namespace ui {

// Background reporter mirroring the reference miner's --shortstats/--longstats cadence:
// prints format_speed_line() every short_s seconds and format_stats_block()
// every long_s seconds, both built from stats.snapshot() at the moment they
// fire, via console::info(). Lifecycle mirrors miner::Engine exactly (see
// engine.h's class comment): start() spawns a worker thread, stop() signals
// it and joins, both are no-ops if already in that state, and the
// destructor calls stop() so a Ticker going out of scope always cleans up.
//
// Unlike Engine's mailbox wait (indefinite, woken only by a new job or
// stop()), this worker wakes on a wall-clock deadline (condition_variable::
// wait_until), recomputed independently for each cadence after it fires --
// so short_s and long_s need not evenly divide one another, and a stop()
// during the wait interrupts immediately rather than waiting out the
// remainder of the current period.
class Ticker {
public:
    ~Ticker();

    // Spawns the worker thread reporting from `stats` (which must outlive
    // the Ticker, or at least until stop()/the destructor returns) on the
    // given cadences. No-op if already started.
    void start(const miner::Stats& stats, int short_s, int long_s);

    // Signals the worker to stop and joins it. No-op if not started.
    void stop();

private:
    void worker_main();

    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_requested_ = false;

    std::thread worker_;
    bool started_ = false;

    const miner::Stats* stats_ = nullptr;
    int short_s_ = 15;   // the reference miner's own --shortstats default
    int long_s_ = 60;    // the reference miner's own --longstats default
};

} } // namespace mxbm::ui
