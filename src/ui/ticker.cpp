#include "ui/ticker.h"

#include <chrono>
#include <cstdio>
#include <ctime>

#include "ui/console.h"
#include "ui/format.h"
#include "version.h"

namespace mxbm { namespace ui {

namespace {

// "HH:MM:SS" for the current wall-clock time in the local time zone --
// matches the reference miner's own "Statistics (HH:MM:SS)" header (see the golden in
// tests/test_format.cpp). format_stats_block itself is pure and never reads
// the clock -- this is the one place that does, on the ticker's own thread.
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
    // First fire of each line is short_s_/long_s_ seconds after start (not
    // immediately at t=0) -- matches the reference miner's own shortstats/longstats
    // cadence semantics ("interval for the ... line", not "also on start").
    clock::time_point next_short = clock::now() + std::chrono::seconds(short_s_);
    clock::time_point next_long  = clock::now() + std::chrono::seconds(long_s_);

    while (true) {
        bool fire_short = false, fire_long = false;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            clock::time_point deadline = next_short < next_long ? next_short : next_long;
            // Predicate-form wait_until loops internally until either the
            // deadline passes or stop_requested_ becomes true, returning
            // the predicate's final value -- true means stop won.
            if (cv_.wait_until(lock, deadline, [this] { return stop_requested_; })) return;

            clock::time_point now = clock::now();
            fire_short = now >= next_short;
            fire_long = now >= next_long;
            if (fire_short) next_short = now + std::chrono::seconds(short_s_);
            if (fire_long) next_long = now + std::chrono::seconds(long_s_);
        }
        // Snapshot + print OUTSIDE the ticker's own mutex_ (mirrors
        // Engine::worker_main extracting the mailbox then processing
        // unlocked) -- stats_->snapshot() takes Stats's own, separate
        // mutex, and console I/O should never hold up stop().
        if (fire_short) {
            std::string line = format_speed_line(stats_->snapshot(), digits_);
            // --timeprint stamps the SHORT line only: the long block already
            // carries a clock in its own header, and the transcript log
            // timestamps every line regardless (see console::open_log).
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
