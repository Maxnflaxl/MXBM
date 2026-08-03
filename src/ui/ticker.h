#pragma once
#include <condition_variable>
#include <mutex>
#include <thread>

#include "miner/stats.h"

namespace mxbm { namespace ui {

// Background reporter on the --shortstats/--longstats cadence:
// prints format_speed_line() every short_s seconds and format_stats_block()
// every long_s seconds, both built from stats.snapshot() at the moment they
// fire. The destructor calls stop().
//
// The worker wakes on a wall-clock deadline (condition_variable::wait_until),
// recomputed independently for each cadence after it fires -- so short_s and
// long_s need not evenly divide one another, and a stop() during the wait
// interrupts immediately rather than waiting out the current period.
class Ticker {
public:
    ~Ticker();

    // Spawns the worker thread reporting from `stats`, which must outlive the
    // Ticker (or at least stop()/the destructor). No-op if already started.
    //
    // `timeprint` (--timeprint) prefixes the SHORT line with an "[HH:MM:SS]"
    // stamp -- only that line, since the long block carries a clock in its own
    // header and the transcript log timestamps every line regardless.
    void start(const miner::Stats& stats, int short_s, int long_s,
               int digits = 2, bool timeprint = false, int api_port = 0,
               int silence = 0);

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
    int short_s_ = 15;   // --shortstats default
    int long_s_ = 60;    // --longstats default
    int digits_ = 2;
    bool timeprint_ = false;
    int api_port_ = 0;   // shown on the stats block's identity line; 0 = API off
    int silence_ = 0;    // --silence; at 3 the speed line is not printed at all
};

} } // namespace mxbm::ui
