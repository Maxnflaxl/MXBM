#pragma once
// --tune: measure THIS card's power/speed curve and recommend --pl (and --mclk).
//
// Nothing here is a constant from the reference card: caps come from the driver's
// band, the rung from its supported-clock list, thresholds from this card's own
// curve. The sweep runs the SAME Engine path mining runs (run_benchmark), inside
// the shipping binary, and re-measures its first point at the end as a drift gauge
// -- the three ways a power-curve number goes wrong, each closed by construction.
// The minimum gain that still buys a watt (sol/s per cap-watt) is the one stated
// preference: --tune-min-gain.
//
// Results are stored (tune_store_path) for `--pl auto`. --tune needs root like every
// NVML write; under sudo the store lands in the INVOKING user's config dir so the
// unprivileged daily launch finds it.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
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
    unsigned recommended_w = 0;   // highest cap whose extra watts still pay
    unsigned eff_w = 0;           // cap with the best sol/s per measured watt
};

// The recommendation, as a pure function of the measured points so a test can pin it
// without a GPU. Points may arrive in any order; failed points (sol_s <= 0) are
// dropped. The walk climbs from the lowest cap and stops at the first step whose
// marginal return -- delta sol/s per delta CAP-watt -- falls below `min_gain`;
// the curve is concave in practice, so nothing above a failed step can pay either.
inline TuneVerdict tune_verdict(std::vector<TunePoint> pts, double min_gain) {
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
    v.recommended_w = pts.front().cap_w;
    for (size_t i = 1; i < pts.size(); ++i) {
        const double dw = (double)pts[i].cap_w - (double)pts[i - 1].cap_w;
        if (dw <= 0.0) continue;
        if ((pts[i].sol_s - pts[i - 1].sol_s) / dw < min_gain) break;
        v.recommended_w = pts[i].cap_w;
    }
    return v;
}

// The caps that refine a coarse verdict, used by passes 2 and 4. The coarse sweep can
// only place a verdict to within its own spacing (37 W on the reference band), so a
// refining pass fills the two coarse intervals TOUCHING `around_w` -- the one below
// it and the one above -- with ~10 W steps, skipping caps already measured. The step
// widens past 10 W only when the bracket would otherwise exceed `max_points`:
// refinement bounds the sweep's tail, it does not restart it.
// Returns an empty list when there is nothing to refine (a single point, or a bracket
// narrower than one step) -- the caller then just keeps the coarse verdict.
inline std::vector<unsigned> tune_refine_caps(std::vector<TunePoint> pts, unsigned around_w,
                                              unsigned max_points = 8) {
    std::vector<unsigned> out;
    std::sort(pts.begin(), pts.end(),
              [](const TunePoint& a, const TunePoint& b) { return a.cap_w < b.cap_w; });
    size_t i = 0;
    while (i < pts.size() && pts[i].cap_w != around_w) ++i;
    if (i == pts.size()) return out;
    const unsigned lo = i > 0 ? pts[i - 1].cap_w : around_w;
    const unsigned hi = i + 1 < pts.size() ? pts[i + 1].cap_w : around_w;
    if (hi <= lo) return out;
    unsigned step = 10u;
    auto interior = [&](unsigned s) {
        unsigned n = 0;
        for (unsigned c = lo + s; c < hi; c += s)
            if (c != around_w) ++n;
        return n;
    };
    while (interior(step) > max_points) step += 5u;
    for (unsigned c = lo + step; c < hi; c += step)
        if (c != around_w) out.push_back(c);
    return out;
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

// The rung pass's candidate: the memory clock worth testing under low caps, from
// the DRIVER's own supported list. The heuristic wants the big half-rate step (the
// reference card's 5001 out of 10501/10251/5001/810/405), not a neighbouring bin
// and not the deep-idle rungs: the largest supported clock below 70 % of the
// card's maximum, rejected if it is under 20 % of it (a card with nothing between
// "full" and "idle" has no rung worth a pass). 0 = no candidate.
inline unsigned tune_rung_pick(const std::vector<unsigned>& supported) {
    unsigned top = 0, pick = 0;
    for (unsigned c : supported) top = c > top ? c : top;
    for (unsigned c : supported)
        if (c < top * 7u / 10u && c > pick) pick = c;
    return (top && pick >= top / 5u) ? pick : 0u;
}

// Where the rung stops paying, from paired stock/rung points at the same caps:
// the sign change of (rung - stock) sol/s, linearly interpolated, so "add --mclk
// below ~X W" is a fact about THIS card's curve. Walks ascending; the crossover is
// taken between the highest winning cap and the next paired cap above it. Returns
// 0 when the rung never wins, and the top paired cap when it always does (it pays
// at least that far -- the sweep did not find its ceiling).
inline unsigned tune_rung_below(std::vector<TunePoint> stock, std::vector<TunePoint> rung) {
    auto bycap = [](const TunePoint& a, const TunePoint& b) { return a.cap_w < b.cap_w; };
    std::sort(stock.begin(), stock.end(), bycap);
    std::sort(rung.begin(), rung.end(), bycap);
    std::vector<std::pair<unsigned, double>> d;              // cap -> rung - stock
    for (const TunePoint& r : rung) {
        if (r.sol_s <= 0.0) continue;
        for (const TunePoint& s : stock)
            if (s.cap_w == r.cap_w && s.sol_s > 0.0)
                d.emplace_back(r.cap_w, r.sol_s - s.sol_s);
    }
    if (d.empty() || d.front().second <= 0.0) return 0;
    for (size_t i = 1; i < d.size(); ++i) {
        if (d[i].second > 0.0) continue;
        const double t = d[i - 1].second / (d[i - 1].second - d[i].second);
        return d[i - 1].first
             + (unsigned)std::lround(t * (double)(d[i].first - d[i - 1].first));
    }
    return d.back().first;
}

// Rough wall-clock for a sweep, in minutes, so a caller can say what it is asking
// for before it asks. Counts the points the passes can actually run: the caps, a
// warmup and a drift gauge either side, pass 2's and pass 4's bracket budgets when
// refining, and at most one rung arm per cap. An upper bound -- a bracket narrower
// than its budget simply runs fewer points. `default_caps` is what
// tune_default_caps() would return for this card, used when cfg.caps is empty.
struct TuneConfig;
int tune_estimated_minutes(const TuneConfig& cfg, unsigned default_caps);

struct TuneConfig {
    // The card being tuned, bus order -- the SAME card every NVML write in the
    // sweep targets and the same index the solver was constructed on. The
    // single-card era's implicit device 0 is what let --tune sweep one card
    // while measuring another (MIXED_RIG.md phase 0).
    unsigned device = 0;
    int    seconds_per_point = 60;
    double min_gain = 0.07;   // sol/s per W below which more watts stop paying
    std::vector<unsigned> caps;    // empty = tune_default_caps() from the driver band
    // Refining passes: tune_refine_caps() around the coarse verdict, and again
    // around the best-efficiency point. main() turns this off when the user passed
    // an explicit --tune-caps list -- a chosen grid means exactly those points.
    bool refine = true;
    // Pass 3: rung arms (tune_rung_pick) at the coarse caps at or below the
    // recommendation, locating where a reduced memory clock starts paying on THIS
    // card. Off when the user chose a memory clock themselves (--mclk) -- theirs
    // stands -- or chose an explicit grid.
    bool rung_pass = true;
    // Reconstruct the solver at each measured point, after the cap (and any rung
    // lock) has landed on the card. Construction is where the cap-keyed choices
    // are made -- the low-power geometry preference, the speculative-entry gate --
    // so a point measured on a startup-time solver is a point of a different
    // program. The old solver is destroyed first so the two allocations never
    // coexist (overlapping them would step a small card down a rung). Empty =
    // keep the one solver for the whole sweep.
    std::function<std::unique_ptr<Solver>()> remake;
    // Where the sweep publishes the solver it is currently driving, for the
    // caller's SIGINT hook; null while a swap is in flight.
    std::atomic<Solver*>* live = nullptr;
};

// Runs the sweep on `solver` (single-device: device 0's limit, like --pl). Needs NVML
// write permission -- the first act is a no-op write of the limit already on the card,
// so a permission failure exits before anything changed. Restores the original limit
// on every path it controls, including Ctrl+C via `stop`. `device_key` names the store
// entry ("<name>@<pci>"); it must match what --pl auto builds. With `cfg.remake` set,
// `solver` is replaced at every measured point (see TuneConfig). Returns a process
// exit code; 0 means the table printed and the store was written.
int run_tune(std::unique_ptr<Solver>& solver, Stats& stats, const std::string& device_key,
             const TuneConfig& cfg, std::atomic<bool>& stop);

// One card's stored sweep, as written by run_tune and read back by --pl auto and
// --report. `mxbm_version` is the binary that measured it; a curve from another
// build describes another program, which is why --report re-measures instead of
// republishing one.
struct TuneStore {
    bool        found = false;
    std::string date, mxbm_version;
    unsigned    recommended_w = 0, eff_w = 0;
    bool        eff_on_rung = false;
    double      min_gain = 0.0, drift_pct = 0.0;
    unsigned    rung_mhz = 0, rung_below_w = 0;
    std::vector<TunePoint> points, points_rung;
};

// Reads `device_key`'s entry. False (and `out.found` clear) when the store, the
// entry, or the file's syntax is absent -- never a throw and never a partial verdict
// built from a half-parsed file.
bool tune_load(const std::string& device_key, TuneStore& out);

// The stored --pl recommendation for `device_key`, or 0 when the store or the entry
// is absent. `date_out` receives the measurement date when found.
unsigned tune_stored_pl(const std::string& device_key, std::string& date_out);

// The stored rung verdict: true when a rung pass ran and measured a paying band.
// `mhz_out` is the rung, `below_w_out` the cap under which it paid. What --pl auto
// uses to say "at this cap, this card also measured faster with --mclk <mhz>" --
// a printed recommendation, never an applied one: locking a memory clock the user
// did not ask for is a hardware setting nobody requested.
bool tune_stored_rung(const std::string& device_key, unsigned& mhz_out,
                      unsigned& below_w_out);

// Whether this process can drive the card's power limit, probed as a no-op write of
// the value already on it: privileged, and a change to nothing if it lands. What
// --report asks before promising the user a sweep, and what run_tune's first act is.
bool tune_can_write(unsigned device);

// Where results live: $XDG_CONFIG_HOME/mxbm/tune.json, with the sudo indirection
// described above.
std::string tune_store_path();

}} // namespace mxbm::miner
