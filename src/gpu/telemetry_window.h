#pragma once
// Sampled telemetry over a measured window, plus the card's own energy counter.
//
// One instrument for every mode that measures: --benchmark's efficiency line,
// --tune's per-point draw and its held-clock check, and --report's hardware block.
// They differ in what they print, never in how the number was taken, which is what
// lets a --tune point and a --benchmark run be quoted side by side.
//
// Power comes from the millijoule counter when the card has one (exact joules over
// the window, no sampling error) and from the trimmed sampler mean otherwise. The
// clock, temperature and utilization rows are always sampled; each is the median of
// its series.
#include <atomic>
#include <thread>
#include <vector>

#include "gpu/nvml.h"

namespace mxbm { namespace gpu {

struct TelemetrySummary {
    bool     have_power  = false;  double   power_w       = 0.0;   // mean over the window
    bool     have_energy = false;  double   joules        = 0.0;   // exact, counter-derived
    bool     have_sm     = false;  unsigned sm_clock_mhz  = 0;     // median
    bool     have_mem    = false;  unsigned mem_clock_mhz = 0;     // median
    bool     have_temp   = false;  unsigned temp_c        = 0;     // median
    bool     have_util   = false;  unsigned util_pct      = 0;     // median
    // Every clocks-event reason seen at any sample, OR'd. A run that touched a
    // thermal or reliability limit says so here even if it spent most of the window
    // at the power cap.
    bool     have_events = false;  unsigned long long events = 0;
    unsigned samples = 0;
};

// Starts sampling on construction and stops on destruction, so an exception or an
// early return cannot leak the thread. `close()` is the ordinary exit: it joins,
// reads the energy counter again, and reduces the series.
//
// The first fifth of every series is discarded. That window is the governor moving
// to the setting just applied -- a clock lock in particular takes a moment to land --
// so including it reports a transition rather than the state under test.
class TelemetryWindow {
public:
    explicit TelemetryWindow(unsigned device, unsigned interval_ms = 500);
    ~TelemetryWindow();

    // `elapsed_s` is the measured window's own duration; with the energy counter it
    // turns joules into mean watts. Idempotent -- a second call returns the same
    // summary without re-reading anything.
    TelemetrySummary close(double elapsed_s);

private:
    unsigned device_;
    std::atomic<bool> done_{false};
    bool closed_ = false;
    TelemetrySummary summary_;
    unsigned long long energy0_ = 0;
    bool have_energy0_ = false;
    std::vector<double>   power_;
    std::vector<unsigned> sm_, mem_, temp_, util_;
    unsigned long long events_ = 0;
    bool have_events_ = false;
    std::thread sampler_;
};

}} // namespace mxbm::gpu
