// Thermal pause/resume, and the watchdog interaction that makes it safe: a
// paused device must not read as hung, or --tstop plus the default --watchdog
// exit is a restart loop into the same heat.
#include "check.h"
#include "miner/thermal.h"
#include "miner/watchdog.h"

#include <vector>

using namespace mxbm;
using namespace mxbm::miner;

int main() {
    section("sensor names");
    {
        TempSensor s = TempSensor::Junction;
        check(parse_temp_sensor("edge", s) && s == TempSensor::Edge, "edge parses");
        check(parse_temp_sensor("JUNCTION", s) && s == TempSensor::Junction, "case-insensitive");
        check(parse_temp_sensor("memory", s) && s == TempSensor::Memory, "memory parses");
        check(!parse_temp_sensor("hotspot", s), "an unknown sensor name is refused");
        check(std::string(temp_sensor_name(TempSensor::Memory)) == "memory", "names round-trip");
    }

    section("pause and resume");
    {
        unsigned temp = 60;
        bool readable = true;
        Thermal t;
        t.configure({{80, 70}});          // stop at 80, resume at 70
        t.read_temp = [&](unsigned, unsigned& out) { out = temp; return readable; };
        std::vector<unsigned> paused_at, resumed_at;
        t.on_pause  = [&](unsigned, unsigned c) { paused_at.push_back(c); };
        t.on_resume = [&](unsigned, unsigned c) { resumed_at.push_back(c); };

        t.poll();
        check(!t.paused(0), "a cool device is left alone");

        temp = 79; t.poll();
        check(!t.paused(0), "one degree below the stop point is not the stop point");

        temp = 80; t.poll();
        check(t.paused(0), "at the stop temperature the device pauses");
        check(paused_at.size() == 1 && paused_at[0] == 80, "the pause is reported once, with the reading");

        temp = 78; t.poll();
        check(t.paused(0), "it stays paused above the restart temperature");
        t.poll();
        check(paused_at.size() == 1, "and is not reported again while it stays paused");

        temp = 70; t.poll();
        check(!t.paused(0), "at the restart temperature it resumes");
        check(resumed_at.size() == 1 && resumed_at[0] == 70, "the resume is reported with its reading");

        // An unreadable sensor must not read as cold OR as hot.
        temp = 95; readable = false; t.poll();
        check(!t.paused(0), "a sensor that will not answer changes nothing");
        readable = true; t.poll();
        check(t.paused(0), "and the next real reading is acted on");
    }

    section("the disabling halves");
    {
        unsigned temp = 100;
        Thermal t;
        t.configure({{0, 70}});           // stop disabled
        t.read_temp = [&](unsigned, unsigned& out) { out = temp; return true; };
        t.poll();
        check(!t.paused(0), "--tstop 0 never pauses, however hot it gets");

        Thermal t2;
        t2.configure({{80, 0}});          // no restart temperature
        t2.read_temp = [&](unsigned, unsigned& out) { out = temp; return true; };
        t2.poll();
        check(t2.paused(0), "it pauses at the stop point");
        temp = 20; t2.poll();
        check(t2.paused(0), "--tstart 0 leaves it paused however cool it gets");
    }

    section("per-device independence");
    {
        std::vector<unsigned> temps = {60, 90};
        Thermal t;
        t.configure({{80, 70}, {80, 70}});
        t.read_temp = [&](unsigned dev, unsigned& out) { out = temps[dev]; return true; };
        t.poll();
        check(!t.paused(0) && t.paused(1), "only the hot card pauses");
        check(!t.paused(7), "an out-of-range device is never paused");
    }

    section("the watchdog does not call a paused device hung");
    {
        // A device whose counter never advances, over a span far past the stall
        // threshold: hung if it was meant to be mining, fine if it is paused.
        auto now = std::chrono::steady_clock::now();
        std::vector<uint64_t> counts = {5};
        bool device_is_paused = false;
        std::vector<unsigned> hung;

        Watchdog w{std::chrono::seconds(30)};
        w.now_fn = [&] { return now; };
        w.counts = [&] { return counts; };
        w.mining = [] { return true; };
        w.device_paused = [&](unsigned) { return device_is_paused; };
        w.on_hung = [&](unsigned d) { hung.push_back(d); };

        device_is_paused = true;
        w.poll();                                   // first sighting
        now += std::chrono::seconds(120);
        w.poll();
        check(hung.empty(), "a paused device is not reported hung, however long it sits");

        // Unpaused and still not advancing: now it is a real stall. The stall
        // clock starts from the resume, not from the pause.
        device_is_paused = false;
        w.poll();
        check(hung.empty(), "the stall clock restarts when it resumes");
        now += std::chrono::seconds(120);
        w.poll();
        check(hung.size() == 1 && hung[0] == 0, "a device that resumes and then stalls is reported");
    }

    return summary("thermal");
}
