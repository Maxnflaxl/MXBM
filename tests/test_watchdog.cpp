// The hung-GPU watchdog: it must fire when a card stops working, and must NOT
// fire for any of the several reasons a healthy card legitimately looks idle.
// The second half is the harder half -- a watchdog whose action is exit(42)
// turns a false positive into a reboot loop on a rig that was mining fine.
//
// No threads and no GPU: poll() is driven directly with a fake clock, the same
// seam Stats uses.
#include "check.h"
#include "miner/watchdog.h"

#include <vector>

using namespace mxbm;
using namespace mxbm::miner;

int main() {
    section("parsing the action");
    {
        WatchdogAction a;
        check(parse_watchdog_action("off", a) && a == WatchdogAction::Off, "off");
        check(parse_watchdog_action("EXIT", a) && a == WatchdogAction::Exit,
              "any case, so --watchdog Exit is not a silent no-op");
        check(parse_watchdog_action("script", a) && a == WatchdogAction::Script, "script");
        check(!parse_watchdog_action("reboot", a), "an unknown action is rejected");
        check(!parse_watchdog_action("", a), "and so is an empty one");
    }

    // A rig of two cards. `counts` is what each has completed; the test moves
    // them and the clock by hand.
    std::vector<uint64_t> counts{0, 0};
    bool mining = true;
    long long t_ms = 0;
    std::vector<unsigned> hung;

    auto make = [&](int stall_s) {
        auto* w = new Watchdog(std::chrono::seconds(stall_s));
        w->counts  = [&counts] { return counts; };
        w->mining  = [&mining] { return mining; };
        w->on_hung = [&hung](unsigned d) { hung.push_back(d); };
        w->now_fn  = [&t_ms] {
            return std::chrono::steady_clock::time_point(std::chrono::milliseconds(t_ms));
        };
        return w;
    };

    section("a device that stops advancing is reported");
    {
        counts = {0, 0}; mining = true; t_ms = 0; hung.clear();
        Watchdog* w = make(90);
        w->poll();                       // first sighting: starts the clocks
        t_ms = 10'000; counts = {10, 10}; w->poll();
        check(hung.empty(), "two working cards are not hung");

        // Card 1 stops. Card 0 keeps going, which is the case that matters:
        // the rig's total only dips, it does not stop.
        t_ms = 60'000;  counts = {20, 10}; w->poll();
        check(hung.empty(), "a gap shorter than the stall window is not yet a fault");
        t_ms = 105'000; counts = {30, 10}; w->poll();
        check(hung.size() == 1 && hung[0] == 1, "the stalled card is named, and only it");

        t_ms = 200'000; counts = {40, 10}; w->poll();
        check(hung.size() == 1, "a card already reported is not reported again every poll");
        delete w;
    }

    section("recovery re-arms the report");
    {
        counts = {0}; mining = true; t_ms = 0; hung.clear();
        Watchdog* w = make(90);
        w->poll();
        t_ms = 100'000; w->poll();
        check(hung.size() == 1, "stalled");
        t_ms = 110'000; counts = {5}; w->poll();     // it came back
        t_ms = 220'000; w->poll();                    // and stalled again
        check(hung.size() == 2,
              "a card that recovers and stalls again is reported again, not written off");
        delete w;
    }

    section("idle is not hung");
    {
        // The false positive that would matter most: no job, so no device is
        // doing anything, for far longer than the stall window. A watchdog that
        // cannot tell this from a crash restarts healthy rigs whenever their
        // pool goes down -- which is exactly when a restart helps least.
        counts = {0, 0}; mining = false; t_ms = 0; hung.clear();
        Watchdog* w = make(90);
        w->poll();
        t_ms = 600'000; w->poll();
        check(hung.empty(), "no job means idle, not crashed, however long it lasts");

        mining = true;
        t_ms = 610'000; w->poll();
        check(hung.empty(), "and the idle time does not count toward the stall window");
        t_ms = 705'000; w->poll();
        // BOTH cards are stalled here, so both are reported -- the rig has two
        // and neither has moved since mining resumed.
        check(hung.size() == 2, "once mining resumes, a real stall is still caught");
        delete w;
    }

    section("a slow start is not a hang");
    {
        // A card can spend tens of seconds allocating 7.5 GiB before its first
        // solve. The clock starts at first sighting, not at construction.
        counts = {0}; mining = true; t_ms = 500'000; hung.clear();
        Watchdog* w = make(90);
        w->poll();
        t_ms = 560'000; w->poll();
        check(hung.empty(), "60 s after the first sighting, still inside the window");
        delete w;
    }

    section("a device appearing later does not shift the others");
    {
        counts = {0}; mining = true; t_ms = 0; hung.clear();
        Watchdog* w = make(90);
        w->poll();
        counts = {0, 0};                 // a second card joins
        t_ms = 50'000; w->poll();
        t_ms = 100'000; w->poll();
        check(hung.size() == 1 && hung[0] == 0,
              "the card that has been stalled since the start is the one reported");
        delete w;
    }

    return summary("watchdog");
}
