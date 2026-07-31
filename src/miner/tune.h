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

// Pass 2's grid: the caps that refine a coarse knee. The coarse sweep can only place
// the knee to within its own spacing (37 W on the reference band), so the second pass
// fills the two coarse intervals TOUCHING the knee -- below it (the last step that
// paid) and above it (the first that failed) -- with ~10 W steps, skipping caps
// already measured. The step widens past 10 W only when the bracket would otherwise
// exceed `max_points`: refinement bounds the sweep's tail, it does not restart it.
// Returns an empty list when there is nothing to refine (a single point, or a bracket
// narrower than one step) -- the caller then just keeps the coarse verdict.
inline std::vector<unsigned> tune_refine_caps(std::vector<TunePoint> pts, unsigned knee_w,
                                              unsigned max_points = 8) {
    std::vector<unsigned> out;
    std::sort(pts.begin(), pts.end(),
              [](const TunePoint& a, const TunePoint& b) { return a.cap_w < b.cap_w; });
    size_t i = 0;
    while (i < pts.size() && pts[i].cap_w != knee_w) ++i;
    if (i == pts.size()) return out;
    const unsigned lo = i > 0 ? pts[i - 1].cap_w : knee_w;
    const unsigned hi = i + 1 < pts.size() ? pts[i + 1].cap_w : knee_w;
    if (hi <= lo) return out;
    unsigned step = 10u;
    auto interior = [&](unsigned s) {
        unsigned n = 0;
        for (unsigned c = lo + s; c < hi; c += s)
            if (c != knee_w) ++n;
        return n;
    };
    while (interior(step) > max_points) step += 5u;
    for (unsigned c = lo + step; c < hi; c += step)
        if (c != knee_w) out.push_back(c);
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

struct TuneConfig {
    int    seconds_per_point = 60;
    double knee_marginal = 0.07;   // sol/s per W below which more watts stop paying
    std::vector<unsigned> caps;    // empty = tune_default_caps() from the driver band
    // Two-pass by default: coarse locate, then tune_refine_caps() around the coarse
    // knee, one combined verdict. main() turns this off when the user passed an
    // explicit --tune-caps list -- a chosen grid means exactly those points.
    bool refine = true;
    // Pass 3: rung arms (tune_rung_pick) at the coarse caps at or below the knee,
    // locating where a reduced memory clock starts paying on THIS card. Off when
    // the user chose a memory clock themselves (--mclk) -- theirs stands -- or
    // chose an explicit grid.
    bool rung_pass = true;
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

// The stored rung verdict: true when a rung pass ran and measured a paying band.
// `mhz_out` is the rung, `below_w_out` the cap under which it paid. What --pl auto
// uses to say "at this cap, this card also measured faster with --mclk <mhz>" --
// a printed recommendation, never an applied one: locking a memory clock the user
// did not ask for is a hardware setting nobody requested.
bool tune_stored_rung(const std::string& device_key, unsigned& mhz_out,
                      unsigned& below_w_out);

// Where results live: $XDG_CONFIG_HOME/mxbm/tune.json, with the sudo indirection
// described above.
std::string tune_store_path();

}} // namespace mxbm::miner
