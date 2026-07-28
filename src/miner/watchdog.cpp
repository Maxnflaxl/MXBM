#include "miner/watchdog.h"

#include <atomic>
#include <cctype>
#include <chrono>
#include <thread>

namespace mxbm { namespace miner {

bool parse_watchdog_action(const std::string& text, WatchdogAction& out) {
    std::string t;
    for (char c : text) t += (char)std::tolower((unsigned char)c);
    if (t == "off")    { out = WatchdogAction::Off;    return true; }
    if (t == "exit")   { out = WatchdogAction::Exit;   return true; }
    if (t == "script") { out = WatchdogAction::Script; return true; }
    return false;
}

const char* watchdog_action_name(WatchdogAction a) {
    switch (a) {
    case WatchdogAction::Off:    return "off";
    case WatchdogAction::Exit:   return "exit";
    case WatchdogAction::Script: return "script";
    }
    return "off";
}

Watchdog::Watchdog(std::chrono::seconds stall) : stall_(stall) {
    now_fn = [] { return std::chrono::steady_clock::now(); };
}

Watchdog::~Watchdog() { stop(); }

void Watchdog::poll() {
    if (!counts) return;
    const std::vector<uint64_t> c = counts();
    const auto now = now_fn();
    // Idle is not hung. Without this, a rig waiting for its first job -- or one
    // whose pool has dropped -- would be declared crashed and, under the exit
    // action, restart itself in a loop while the actual fault was the network.
    const bool active = !mining || mining();

    if (devs_.size() != c.size()) devs_.resize(c.size());

    for (size_t i = 0; i < c.size(); ++i) {
        DevState& d = devs_[i];

        if (!d.seen) {
            // First sighting: start the clock here rather than at construction,
            // so a device that takes 30 s to allocate 7.5 GiB is not accused of
            // hanging before it has had a chance to start.
            d.seen = true;
            d.last_count = c[i];
            d.last_progress = now;
            continue;
        }

        if (c[i] != d.last_count) {
            d.last_count = c[i];
            d.last_progress = now;
            if (d.hung) {
                // Recovered on its own. Clearing the flag means a card that
                // stalls again later is reported again, rather than being
                // written off after the first time.
                d.hung = false;
            }
            continue;
        }

        // No progress. The clock only runs while the miner expects work, so a
        // long idle period does not accumulate toward the stall threshold.
        if (!active) {
            d.last_progress = now;
            continue;
        }
        if (!d.hung && now - d.last_progress >= stall_) {
            d.hung = true;
            if (on_hung) on_hung((unsigned)i);
        }
    }
}

void Watchdog::start(std::chrono::seconds interval) {
    if (running_) return;
    running_ = true;
    thread_ = new std::thread([this, interval] {
        while (running_) {
            // Short sleeps in a loop rather than one long one, so stop() does
            // not have to wait out a whole interval to join.
            for (int i = 0; i < interval.count() * 4 && running_; ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
            if (!running_) break;
            poll();
        }
    });
}

void Watchdog::stop() {
    if (!running_) return;
    running_ = false;
    auto* t = static_cast<std::thread*>(thread_);
    if (t) {
        if (t->joinable()) t->join();
        delete t;
        thread_ = nullptr;
    }
}

} } // namespace mxbm::miner
