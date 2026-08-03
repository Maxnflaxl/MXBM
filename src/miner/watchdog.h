#pragma once
// Detects a GPU that has stopped doing work, and acts on it.
//
// A crashed or hung card is the one failure a mining rig cannot notice by
// itself: the process is alive, the pool connection is up, the console keeps
// printing, and the only symptom is that one device's share of the hashrate
// quietly went missing. On a single-GPU rig the speed goes to zero and someone
// eventually looks; on an eight-GPU rig it is a 12 % dip that reads like
// variance for a week.
//
// WHAT IT WATCHES, and why it is a counter rather than a rate: every device
// reports a monotonic count of completed solve() calls (Stats::Device::attempts).
// A rate cannot tell "hung" from "the 60 s window has not filled yet", and a
// card that finds no candidates for a minute is unlucky rather than crashed. A
// counter that does not advance is unambiguous.
//
// WHAT IT DOES NOT DO: it never restarts a device. Recovering a hung GPU means
// tearing down and rebuilding a CUDA context that a wedged kernel still owns,
// which usually cannot be done from inside the same process. Exiting and
// letting a supervisor restart the miner is the recovery that works.
#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace mxbm { namespace miner {

// What to do when a device is found hung.
enum class WatchdogAction {
    Off,      // report it and keep mining on the remaining cards
    Exit,     // exit(42), the conventional "please restart me" code
    Script,   // run an external script, then keep going
};

// Parses "off" | "exit" | "script" (any case). False on anything else.
bool parse_watchdog_action(const std::string& text, WatchdogAction& out);
const char* watchdog_action_name(WatchdogAction a);

class Watchdog {
public:
    // `stall` is how long a device may report no completed solve before it is
    // called hung. It has to clear the slowest legitimate gap between attempts,
    // which is a whole solve (~35 ms) plus the wait for a new job, so the
    // default is deliberately far above both.
    explicit Watchdog(std::chrono::seconds stall = std::chrono::seconds(90));

    // Per-device monotonic attempt counts, newest first call to last. Supplied
    // by the caller rather than read from Stats directly, so this is testable
    // without a miner and a GPU.
    using CountsFn = std::function<std::vector<uint64_t>()>;
    // True while the miner actually expects the devices to be working. A rig
    // with no job, or between pools, is idle rather than hung, and a watchdog
    // that cannot tell those apart is a watchdog that reboots healthy rigs.
    using MiningFn = std::function<bool()>;
    // Called once per device that goes from working to hung. Not called again
    // for the same device unless it recovers first.
    using HungFn = std::function<void(unsigned device)>;
    // True while `device` is deliberately not mining (miner/thermal.h). Such a
    // device stops advancing its attempt counter, which is indistinguishable
    // from a hang from here -- without this, --tstop plus the default
    // --watchdog exit is a restart loop into the same heat.
    using PausedFn = std::function<bool(unsigned device)>;

    CountsFn counts;
    MiningFn mining;
    HungFn   on_hung;
    PausedFn device_paused;

    // Test seam, same pattern as Stats::now_fn: every time read inside goes
    // through it. Defaults to steady_clock::now.
    std::function<std::chrono::steady_clock::time_point()> now_fn;

    // One check. Call it periodically -- run() below does that on a thread, and
    // tests call it directly with a fake clock.
    void poll();

    // Spawns a thread that polls every `interval` until stop(). No-op if
    // already started.
    void start(std::chrono::seconds interval = std::chrono::seconds(15));
    void stop();
    ~Watchdog();

private:
    struct DevState {
        uint64_t last_count = 0;
        std::chrono::steady_clock::time_point last_progress{};
        bool seen = false;      // has this device ever reported a count?
        bool hung = false;      // already reported; do not report again
    };
    std::chrono::seconds stall_;
    std::vector<DevState> devs_;
    bool  running_ = false;
    void* thread_ = nullptr;    // std::thread*, hidden to keep <thread> out of the header
};

} } // namespace mxbm::miner
