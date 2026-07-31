#pragma once
// --tune: measure THIS card's power/speed curve and recommend a --pl value.
//
// The whole point is that nothing here is a constant from the reference card. The
// 2026-07-31 session established three ways a power-curve number goes wrong -- measured
// in the wrong loop (pipeline replay vs the miner), on a stale build, or without a
// drift gauge -- so the sweep runs the SAME Engine path mining runs (run_benchmark),
// inside the shipping binary, and re-measures its first point at the end to price the
// session's own drift. The candidate caps come from the band the driver reports, and
// the knee criterion is a stated preference (sol/s per watt of cap), not a measurement:
// it is the one number the user may reasonably want to move (--tune-knee).
//
// The result is stored (tune_store_path) so `--pl auto` can apply it on every later
// launch without re-sweeping. --tune needs root, exactly like --pl: every NVML write
// does. Run it once under sudo; the store lands in the INVOKING user's config dir, not
// root's, so the unprivileged daily launch finds it.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "miner/solver.h"
#include "miner/stats.h"

namespace mxbm { namespace miner {

struct TunePoint {
    unsigned cap_w = 0;        // the limit that was applied
    double   draw_w = 0.0;     // mean measured board draw during the window (0 = unknown)
    double   sol_s = 0.0;
    double   median_ms = 0.0;
};

struct TuneVerdict {
    unsigned knee_w = 0;       // highest cap whose marginal gain still pays
    unsigned eff_w = 0;        // cap with the best sol/s per measured watt
};

// The recommendation, as a pure function of the measured points so a test can pin it
// without a GPU. Points may arrive in any order; failed points (sol_s <= 0) are
// dropped. The knee walk climbs from the lowest cap and stops at the first step whose
// marginal return -- delta sol/s per delta CAP-watt -- falls below `knee_marginal`;
// the curve is concave in practice, so nothing above a failed step can pay either.
inline TuneVerdict tune_verdict(std::vector<TunePoint> pts, double knee_marginal) {
    TuneVerdict v;
    pts.erase(std::remove_if(pts.begin(), pts.end(),
                             [](const TunePoint& p) { return p.sol_s <= 0.0; }),
              pts.end());
    if (pts.empty()) return v;
    std::sort(pts.begin(), pts.end(),
              [](const TunePoint& a, const TunePoint& b) { return a.cap_w < b.cap_w; });
    double best = -1.0;
    for (const TunePoint& p : pts) {
        // Efficiency is priced against the MEASURED draw when we have it: at high caps
        // a core-limited card can draw under its limit, and sol-per-cap-watt would
        // flatter exactly those points. The cap is only the fallback.
        const double w = p.draw_w > 0.0 ? p.draw_w : (double)p.cap_w;
        if (w > 0.0 && p.sol_s / w > best) { best = p.sol_s / w; v.eff_w = p.cap_w; }
    }
    v.knee_w = pts.front().cap_w;
    for (size_t i = 1; i < pts.size(); ++i) {
        const double dw = (double)pts[i].cap_w - (double)pts[i - 1].cap_w;
        if (dw <= 0.0) continue;
        if ((pts[i].sol_s - pts[i - 1].sol_s) / dw < knee_marginal) break;
        v.knee_w = pts[i].cap_w;
    }
    return v;
}

// Six candidate caps across the card's own band, min..default inclusive. The default
// limit, not the max: above the default is overclocking territory, which --tune does
// not walk into uninvited. Narrow bands can collapse points after rounding; the
// caller checks there are at least two left.
inline std::vector<unsigned> tune_default_caps(unsigned min_w, unsigned default_w) {
    std::vector<unsigned> caps;
    if (min_w == 0 || default_w <= min_w) return caps;
    constexpr int kPoints = 6;
    for (int i = 0; i < kPoints; ++i)
        caps.push_back(min_w + (unsigned)std::lround(
                           (double)(default_w - min_w) * i / (kPoints - 1)));
    caps.erase(std::unique(caps.begin(), caps.end()), caps.end());
    return caps;
}

struct TuneConfig {
    int    seconds_per_point = 60;
    double knee_marginal = 0.07;   // sol/s per W below which more watts stop paying
    std::vector<unsigned> caps;    // empty = tune_default_caps() from the driver band
};

// Runs the sweep on `solver` (single-device: device 0's limit, like --pl). Needs NVML
// write permission -- the first act is a no-op write of the limit already on the card,
// so a permission failure exits before anything changed. Restores the original limit
// on every path it controls, including Ctrl+C via `stop`. `device_key` names the store
// entry ("<name>@<pci>"); it must match what --pl auto builds. Returns a process exit
// code; 0 means the table printed and the store was written.
int run_tune(Solver& solver, Stats& stats, const std::string& device_key,
             const TuneConfig& cfg, std::atomic<bool>& stop);

// The stored knee for `device_key`, or 0 when the store or the entry is absent.
// `date_out` receives the measurement date when found.
unsigned tune_stored_knee(const std::string& device_key, std::string& date_out);

// Where results live: $XDG_CONFIG_HOME/mxbm/tune.json, with the sudo indirection
// described above.
std::string tune_store_path();

}} // namespace mxbm::miner
