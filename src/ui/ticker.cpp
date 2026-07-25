#include "ui/ticker.h"

#include <chrono>
#include <cstdio>
#include <ctime>

#include "ui/console.h"
#include "ui/format.h"
#include "version.h"

namespace mxbm { namespace ui {

namespace {

// "HH:MM:SS" local wall-clock time for the stats block's header --
// format_stats_block is pure and never reads the clock itself.
std::string clock_hhmmss() {
    std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm_buf{};
#if defined(_WIN32)
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif
    char buf[16];
    std::snprintf(buf, sizeof buf, "%02d:%02d:%02d", tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec);
    return buf;
}

} // namespace

Ticker::~Ticker() { stop(); }

void Ticker::start(const miner::Stats& stats, int short_s, int long_s,
                   int digits, bool timeprint, int api_port) {
    if (started_) return;
    stats_ = &stats;
    short_s_ = short_s;
    long_s_ = long_s;
    digits_ = digits;
    timeprint_ = timeprint;
    api_port_ = api_port;
    started_ = true;
    stop_requested_ = false;
    worker_ = std::thread([this] { try { worker_main(); } catch (...) {} });
}

void Ticker::stop() {
    if (!started_) return;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_requested_ = true;
    }
    cv_.notify_one();
    if (worker_.joinable()) worker_.join();
    started_ = false;
}

void Ticker::worker_main() {
    using clock = std::chrono::steady_clock;
    // First fire of each line is one interval after start, not at t=0 --
    // matching the reference miner's shortstats/longstats cadence semantics.
    clock::time_point next_short = clock::now() + std::chrono::seconds(short_s_);
    clock::time_point next_long  = clock::now() + std::chrono::seconds(long_s_);

    while (true) {
        bool fire_short = false, fire_long = false;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            clock::time_point deadline = next_short < next_long ? next_short : next_long;
            // Predicate-form wait_until returns the predicate's final value:
            // true means stop won, not the deadline.
            if (cv_.wait_until(lock, deadline, [this] { return stop_requested_; })) return;

            clock::time_point now = clock::now();
            fire_short = now >= next_short;
            fire_long = now >= next_long;
            if (fire_short) next_short = now + std::chrono::seconds(short_s_);
            if (fire_long) next_long = now + std::chrono::seconds(long_s_);
        }
        // Snapshot + print OUTSIDE the ticker's own mutex_: stats_->snapshot()
        // takes Stats's own, separate mutex, and console I/O should never hold
        // up stop().
        if (fire_short) {
            std::string line = format_speed_line(stats_->snapshot(), digits_);
            if (timeprint_) line = "[" + clock_hhmmss() + "] " + line;
            console::info(line);
        }
        if (fire_long) {
            console::stats_block(format_stats_block(
                stats_->snapshot(), mxbm::version(), clock_hhmmss().c_str(), digits_, api_port_));
        }
    }
}

} } // namespace mxbm::ui
