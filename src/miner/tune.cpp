#include "miner/tune.h"

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <thread>

#ifndef _WIN32
#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include "gpu/nvml.h"
#include "gpu/telemetry_window.h"
#include "miner/benchmark.h"
#include "nlohmann/json.hpp"
#include "miner/user_paths.h"
#include "ui/console.h"
#include "version.h"

namespace mxbm { namespace miner {
namespace {

using nlohmann::json;

// Sleeps in slices so Ctrl+C (via `stop`) is honoured within ~200 ms.
bool settle(int seconds, std::atomic<bool>& stop) {
    for (int i = 0; i < seconds * 5; ++i) {
        if (stop.load(std::memory_order_relaxed)) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    return true;
}

std::string fmt(const char* f, ...) {
    char buf[256];
    va_list ap; va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}

} // namespace

std::string tune_store_path() { return config_dir() + "/mxbm/tune.json"; }

bool tune_load(const std::string& device_key, TuneStore& out) {
    out = TuneStore{};
    std::ifstream f(tune_store_path());
    if (!f) return false;
    try {
        json j; f >> j;
        const auto it = j.find(device_key);
        if (it == j.end()) return false;
        auto series = [](const json& arr, std::vector<TunePoint>& into) {
            if (!arr.is_array()) return;
            for (const json& p : arr)
                into.push_back(TunePoint{p.value("cap_w", 0u), p.value("draw_w", 0.0),
                                         p.value("sol_s", 0.0), p.value("median_ms", 0.0)});
        };
        out.date          = it->value("date", std::string());
        out.mxbm_version  = it->value("mxbm_version", std::string());
        out.recommended_w = it->value("recommended_w", 0u);
        out.eff_w         = it->value("eff_w", 0u);
        out.eff_on_rung   = it->value("eff_on_rung", false);
        out.min_gain      = it->value("min_gain", 0.0);
        out.drift_pct     = it->value("drift_pct", 0.0);
        out.rung_mhz      = it->value("rung_mhz", 0u);
        out.rung_below_w  = it->value("rung_below_w", 0u);
        if (it->contains("points"))      series((*it)["points"], out.points);
        if (it->contains("points_rung")) series((*it)["points_rung"], out.points_rung);
        out.found = true;
        return true;
    } catch (...) { out = TuneStore{}; return false; }
}

unsigned tune_stored_pl(const std::string& device_key, std::string& date_out) {
    TuneStore st;
    if (!tune_load(device_key, st)) return 0;
    date_out = st.date;
    return st.recommended_w;
}

bool tune_stored_rung(const std::string& device_key, unsigned& mhz_out,
                      unsigned& below_w_out) {
    TuneStore st;
    if (!tune_load(device_key, st)) return false;
    mhz_out = st.rung_mhz;
    below_w_out = st.rung_below_w;
    return mhz_out != 0 && below_w_out != 0;
}

int tune_estimated_minutes(const TuneConfig& cfg, unsigned default_caps) {
    const unsigned n = cfg.caps.empty() ? default_caps : (unsigned)cfg.caps.size();
    unsigned pts = n + 2;                       // warmup + the caps + the drift gauge
    if (cfg.refine)    pts += 8 + 4;            // pass 2's budget, then pass 4's
    if (cfg.rung_pass) pts += n;                // at most one rung arm per cap
    // Each point pays its measured seconds plus the governor settle and the solver
    // rebuild.
    const int secs = (int)pts * (cfg.seconds_per_point + 8);
    return (secs + 59) / 60;
}

bool tune_can_write(unsigned device) {
    const gpu::PowerLimit pl = gpu::nvml_power_limit(device);
    return pl.valid
        && gpu::nvml_set_power_limit(device, pl.current_w) == gpu::NvmlWrite::Ok;
}

int run_tune(std::unique_ptr<Solver>& solver, Stats& stats, const std::string& device_key,
             const TuneConfig& cfg, std::atomic<bool>& stop) {
    const gpu::PowerLimit pl0 = gpu::nvml_power_limit(cfg.device);
    if (!pl0.valid) {
        ui::console::error("--tune: this card reports no power limit, so there is "
                           "no knob to sweep");
        return 1;
    }
    // Permission probe FIRST, as a no-op write of the value already on the card:
    // a refusal here proves nothing was changed, and every later write would have
    // failed the same way.
    if (!tune_can_write(cfg.device)) {
#ifdef _WIN32
        ui::console::error("--tune sets power limits, and every NVML write needs "
                           "elevation. Re-run from an Administrator terminal");
#else
        ui::console::error("--tune sets power limits, and every NVML write needs root. "
                           "Re-run under sudo (the result is stored in your own config "
                           "dir, not root's, so daily runs stay unprivileged)");
#endif
        return 1;
    }

    std::vector<unsigned> caps = cfg.caps;
    if (caps.empty())
        caps = tune_default_caps(pl0.min_w, pl0.default_w ? pl0.default_w : pl0.max_w);
    for (unsigned& c : caps) {
        const unsigned was = c;
        if (pl0.min_w && c < pl0.min_w) c = pl0.min_w;
        if (pl0.max_w && c > pl0.max_w) c = pl0.max_w;
        if (c != was)
            ui::console::info(fmt("--tune: %u W is outside the driver's %u-%u W band, "
                                  "clamped to %u", was, pl0.min_w, pl0.max_w, c));
    }
    std::sort(caps.begin(), caps.end(), std::greater<unsigned>());
    caps.erase(std::unique(caps.begin(), caps.end()), caps.end());
    if (caps.size() < 2) {
        ui::console::error("--tune needs at least two distinct power points to place "
                           "a recommendation; this card's band leaves fewer");
        return 1;
    }

    const int per = cfg.seconds_per_point;
    ui::console::info(fmt("Tuning %s: %zu power points x %d s%s, plus a warmup and a "
                          "drift gauge (~%d min). Ctrl+C aborts and restores %u W.",
                          device_key.c_str(), caps.size(), per,
                          cfg.refine ? ", then two refining passes" : "",
                          tune_estimated_minutes(cfg, (unsigned)caps.size()),
                          pl0.current_w));

    // Restore on EVERY exit from here down. The original limit, not the default:
    // a user who had already capped the card outside MXBM gets their value back.
    // The memory-clock lock is reset only if the rung pass set it -- a user's own
    // --mclk (which disables the pass) must not be unlocked out from under them.
    struct Restore {
        unsigned dev; unsigned w; bool mem = false;
        ~Restore() {
            if (mem) gpu::nvml_reset_locked_mem_clock(dev);
            gpu::nvml_set_power_limit(dev, w);
        }
    } restore{cfg.device, pl0.current_w};

    // One measured point: set the cap (and, for rung arms, lock the memory clock),
    // let the governor settle, then run the SAME Engine path mining runs while a
    // sampler thread records draw and memory clock. A rung arm whose sampled clock
    // did not HOLD the rung is REFUSED (cap_w = 0), never a data point -- the 10501
    // null was a refused rung, and a refusal must not masquerade as "does not pay".
    auto measure = [&](unsigned cap, unsigned mclk, TunePoint& out) -> bool {
        if (gpu::nvml_set_power_limit(cfg.device, cap) != gpu::NvmlWrite::Ok) {
            ui::console::error(fmt("--tune: setting %u W failed mid-sweep; aborting", cap));
            return false;
        }
        if (mclk) {
            if (gpu::nvml_set_locked_mem_clock(cfg.device, mclk, mclk) != gpu::NvmlWrite::Ok) {
                ui::console::info(fmt("    %u MHz memory lock refused; skipping rung arms", mclk));
                out = TunePoint{};
                return !stop.load(std::memory_order_relaxed);
            }
            restore.mem = true;
        }
        if (!settle(5, stop)) return false;
        if (cfg.remake) {
            // Rebuild against the state just applied (see TuneConfig::remake), after
            // the settle: a memory-clock write takes a moment to land, and a solver
            // constructed against the pre-transition clock reads the wrong rung.
            // Old solver first so the two allocations never coexist.
            if (cfg.live) cfg.live->store(nullptr, std::memory_order_relaxed);
            solver.reset();
            solver = cfg.remake();
            if (!solver) {
                ui::console::error("--tune: rebuilding the solver for this point "
                                   "failed; aborting");
                return false;
            }
            if (cfg.live) cfg.live->store(solver.get(), std::memory_order_relaxed);
        }
        gpu::TelemetryWindow tw(cfg.device);
        const BenchmarkResult r = run_benchmark(*solver, stats, per, stop);
        const gpu::TelemetrySummary t = tw.close(r.elapsed_s);
        // A rung arm whose memory clock did not HOLD is refused, never recorded --
        // "the lock did not take" and "the rung does not pay" are different findings.
        if (mclk && (!t.have_mem || t.mem_clock_mhz + 5u < mclk || t.mem_clock_mhz > mclk + 5u)) {
            ui::console::info(fmt("    %u W at %u MHz: REFUSED (memory ran at %u MHz)"
                                  " - dropped", cap, mclk, t.mem_clock_mhz));
            out = TunePoint{};
            return !stop.load(std::memory_order_relaxed);
        }
        out = TunePoint{cap, t.have_power ? t.power_w : 0.0, r.sol_per_s, r.median_ms};
        return !stop.load(std::memory_order_relaxed);
    };

    // Every pass is the same loop: announce, measure, print, keep -- differing only
    // in the caps, the memory clock, and where the points land. Returns false when
    // the sweep must stop (Ctrl+C, or a write that failed); a rung arm the card
    // refused is skipped, not recorded, and ends the pass if the lock itself was
    // refused rather than merely not held.
    auto run_pass = [&](const std::vector<unsigned>& pass_caps, unsigned mclk,
                        std::vector<TunePoint>& into) -> bool {
        for (unsigned cap : pass_caps) {
            ui::console::info(mclk
                ? fmt("  measuring %u W at %u MHz (%d s)...", cap, mclk, per)
                : fmt("  measuring %u W (%d s)...", cap, per));
            TunePoint p;
            if (!measure(cap, mclk, p)) {
                ui::console::info("--tune aborted; card restored");
                return false;
            }
            if (p.cap_w == 0) {
                if (!restore.mem) break;   // lock refused outright: no more arms
                continue;                  // this arm's clock did not hold
            }
            ui::console::info(mclk
                ? fmt("    %u W @ %u MHz: %.2f sol/s, %.1f ms/solve, draw %.1f W",
                      cap, mclk, p.sol_s, p.median_ms, p.draw_w)
                : fmt("    %u W: %.2f sol/s, %.1f ms/solve, draw %.1f W",
                      cap, p.sol_s, p.median_ms, p.draw_w));
            into.push_back(p);
        }
        return true;
    };

    ui::console::info(fmt("  warmup at %u W (%d s, discarded)...", caps.front(), per));
    {
        TunePoint scratch;
        if (!measure(caps.front(), 0, scratch)) {
            ui::console::info("--tune aborted; power limit restored");
            return 1;
        }
    }

    std::vector<TunePoint> points;
    if (!run_pass(caps, 0, points)) return 1;

    // Pass 2: the coarse grid can only place the recommendation to within its own
    // spacing, so refine the two intervals touching it with ~10 W steps and fold them
    // into the SAME verdict. Skipped for an explicit --tune-caps list (a chosen grid
    // means exactly those points) and when the bracket is already tighter than a step.
    if (cfg.refine) {
        const TuneVerdict coarse = tune_verdict(points, cfg.min_gain);
        std::vector<unsigned> fine = tune_refine_caps(points, coarse.recommended_w);
        std::sort(fine.begin(), fine.end(), std::greater<unsigned>());
        if (!fine.empty()) {
            ui::console::info(fmt("  pass 2: coarse recommendation %u W - refining its "
                                  "neighbourhood with %zu points x %d s (~%d min)...",
                                  coarse.recommended_w, fine.size(), per,
                                  (int)(fine.size() * (per + 6) / 60) + 1));
            if (!run_pass(fine, 0, points)) return 1;
        }
    }

    // Pass 3: the memory rung. Under a low cap the interface burns watts for
    // bandwidth the slowed core cannot use; a reduced memory clock returns them as
    // core clock (measured 2026-07-31: −8.5 to −14.4 % below ~173 W on the reference
    // card, catastrophic above ~180). The candidate rung comes from THIS card's
    // driver (tune_rung_pick) and the arms run at the coarse caps at or below the
    // recommendation -- the region where the rung could pay -- so the threshold is a
    // measured property of the user's card, not of ours.
    std::vector<TunePoint> rung_pts;
    unsigned rung_mhz = 0;
    if (cfg.rung_pass) {
        rung_mhz = tune_rung_pick(gpu::nvml_supported_mem_clocks(cfg.device));
        if (rung_mhz) {
            const TuneVerdict sofar = tune_verdict(points, cfg.min_gain);
            std::vector<unsigned> rcaps;
            for (unsigned cap : caps)
                if (cap <= sofar.recommended_w) rcaps.push_back(cap);
            if (!rcaps.empty()) {
                ui::console::info(fmt("  pass 3: the %u MHz memory rung at %zu capped "
                                      "points x %d s (~%d min)...",
                                      rung_mhz, rcaps.size(), per,
                                      (int)(rcaps.size() * (per + 6) / 60) + 1));
                if (!run_pass(rcaps, rung_mhz, rung_pts)) return 1;
                if (restore.mem) {
                    gpu::nvml_reset_locked_mem_clock(cfg.device);
                    restore.mem = false;
                }
            }
        }
    }

    // Pass 4: efficiency refinement. The coarse spacing that blurs the recommendation
    // blurs the efficiency optimum the same way -- on the reference card the 37 W grid
    // skipped ~160 W and the verdict missed the true record sitting between two
    // rung arms. Refine the two intervals touching the best-efficiency point ON
    // ITS OWN memory clock (stock or rung) and fold the points into that series,
    // so the final verdict reads a locally dense curve.
    if (cfg.refine) {
        auto eff = [](const TunePoint& p) {
            const double w = p.draw_w > 0.0 ? p.draw_w : (double)p.cap_w;
            return w > 0.0 ? p.sol_s / w : 0.0;
        };
        double best = 0.0; unsigned best_cap = 0; bool on_rung = false;
        for (const TunePoint& p : points)
            if (eff(p) > best) { best = eff(p); best_cap = p.cap_w; }
        for (const TunePoint& p : rung_pts)
            if (eff(p) > best) { best = eff(p); best_cap = p.cap_w; on_rung = true; }
        std::vector<TunePoint>& series = on_rung ? rung_pts : points;
        std::vector<unsigned> ecaps = tune_refine_caps(series, best_cap, 4);
        ecaps.erase(std::remove_if(ecaps.begin(), ecaps.end(), [&](unsigned c) {
                        for (const TunePoint& p : series) if (p.cap_w == c) return true;
                        return false;
                    }), ecaps.end());
        if (!ecaps.empty()) {
            ui::console::info(fmt("  pass 4: best efficiency so far at %u W%s - "
                                  "refining with %zu points x %d s (~%d min)...",
                                  best_cap,
                                  on_rung ? fmt(" on the %u MHz rung", rung_mhz).c_str() : "",
                                  ecaps.size(), per,
                                  (int)(ecaps.size() * (per + 6) / 60) + 1));
            if (!run_pass(ecaps, on_rung ? rung_mhz : 0u, series)) return 1;
            if (restore.mem) {
                gpu::nvml_reset_locked_mem_clock(cfg.device);
                restore.mem = false;
            }
        }
    }

    // The drift gauge: the first point again, at the end -- AFTER every pass (and
    // back on stock memory), so it prices the whole session. Today's numbers carry
    // at least this bar -- the 2026-07-31 session measured ~2 %/arm of thermal
    // drift, which is what made un-gauged sweeps lie.
    ui::console::info(fmt("  drift gauge: %u W again (%d s)...", caps.front(), per));
    TunePoint again;
    double drift_pct = 0.0;
    if (measure(caps.front(), 0, again) && points.front().median_ms > 0.0) {
        drift_pct = (again.median_ms - points.front().median_ms)
                    / points.front().median_ms * 100.0;
    } else if (stop.load(std::memory_order_relaxed)) {
        ui::console::info("--tune aborted; power limit restored");
        return 1;
    }

    // Both passes in one table, in cap order -- the reader wants the curve, not the
    // order the sweep happened to visit it in.
    std::sort(points.begin(), points.end(),
              [](const TunePoint& a, const TunePoint& b) { return a.cap_w > b.cap_w; });
    auto print_curve = [&](const std::vector<TunePoint>& series, const std::string& title) {
        ui::console::info(title);
        ui::console::info("    cap W   draw W    sol/s   ms/solve   sol/s/W");
        for (const TunePoint& p : series) {
            const double w = p.draw_w > 0.0 ? p.draw_w : (double)p.cap_w;
            ui::console::info(fmt("    %5u  %7.1f  %7.2f  %9.1f  %8.4f",
                                  p.cap_w, p.draw_w, p.sol_s, p.median_ms,
                                  w > 0.0 ? p.sol_s / w : 0.0));
        }
    };
    print_curve(points, "Power curve (miner loop, CPU-verified sol/s):");
    if (!rung_pts.empty()) {
        std::sort(rung_pts.begin(), rung_pts.end(),
                  [](const TunePoint& a, const TunePoint& b) { return a.cap_w > b.cap_w; });
        print_curve(rung_pts, fmt("At the %u MHz memory rung (same loop):", rung_mhz));
    }
    ui::console::info(fmt("  drift over the sweep: %+.1f %% at %u W (re-measured last)%s",
                          drift_pct, caps.front(),
                          std::abs(drift_pct) > 1.5
                              ? " - above the 1.5 % gauge, treat the table as +/- that"
                              : ""));

    const TuneVerdict v = tune_verdict(points, cfg.min_gain);
    ui::console::info(fmt("Recommended: --pl %u  (above it, each watt returns less "
                          "than %.2f sol/s)", v.recommended_w, cfg.min_gain));
    // The efficiency line names the rung when a rung point beats every stock one --
    // on the reference card the record itself sits at a capped point ON the rung.
    const unsigned rung_below = rung_pts.size() >= 2
                              ? tune_rung_below(points, rung_pts) : 0u;
    double best_eff = 0.0; unsigned best_eff_w = v.eff_w; bool best_on_rung = false;
    for (const TunePoint& p : points) {
        const double w = p.draw_w > 0.0 ? p.draw_w : (double)p.cap_w;
        if (w > 0.0 && p.sol_s / w > best_eff) { best_eff = p.sol_s / w; best_eff_w = p.cap_w; }
    }
    for (const TunePoint& p : rung_pts) {
        const double w = p.draw_w > 0.0 ? p.draw_w : (double)p.cap_w;
        if (w > 0.0 && p.sol_s / w > best_eff) {
            best_eff = p.sol_s / w; best_eff_w = p.cap_w; best_on_rung = true;
        }
    }
    ui::console::info(fmt("Best efficiency: %u W%s", best_eff_w,
                          best_on_rung ? fmt(" + --mclk %u", rung_mhz).c_str() : ""));
    if (rung_below)
        ui::console::info(fmt("Memory rung: below ~%u W the %u MHz rung measured "
                              "faster on this card - when capping under that, add "
                              "--mclk %u", rung_below, rung_mhz, rung_mhz));
    else if (!rung_pts.empty())
        ui::console::info(fmt("Memory rung: the %u MHz rung never paid at the "
                              "measured caps on this card - keep stock memory", rung_mhz));

    const std::string path = tune_store_path();
    try {
        json j = json::object();
        {
            std::ifstream f(path);
            if (f) { try { f >> j; } catch (...) { j = json::object(); } }
        }
        json pts = json::array();
        for (const TunePoint& p : points)
            pts.push_back({{"cap_w", p.cap_w}, {"draw_w", p.draw_w},
                           {"sol_s", p.sol_s}, {"median_ms", p.median_ms}});
        char date[32];
        const std::time_t now = std::time(nullptr);
        std::strftime(date, sizeof date, "%Y-%m-%d %H:%M", std::localtime(&now));
        // The binary that measured it. A curve taken on another build is a curve of
        // another program, so --report re-measures rather than republish one.
        j[device_key] = {{"recommended_w", v.recommended_w}, {"eff_w", best_eff_w},
                         {"eff_on_rung", best_on_rung},
                         {"min_gain", cfg.min_gain}, {"date", date},
                         {"mxbm_version", mxbm::version()},
                         {"drift_pct", drift_pct}, {"points", pts}};
        if (!rung_pts.empty()) {
            json rp = json::array();
            for (const TunePoint& p : rung_pts)
                rp.push_back({{"cap_w", p.cap_w}, {"draw_w", p.draw_w},
                              {"sol_s", p.sol_s}, {"median_ms", p.median_ms}});
            j[device_key]["rung_mhz"] = rung_mhz;
            j[device_key]["rung_below_w"] = rung_below;
            j[device_key]["points_rung"] = rp;
        }
        const std::filesystem::path dir = std::filesystem::path(path).parent_path();
        std::filesystem::create_directories(dir);
        chown_to_invoking_user(dir.string());
        std::ofstream out(path);
        out << j.dump(2) << "\n";
        out.close();
        chown_to_invoking_user(path);
        ui::console::info("Stored for `--pl auto`: " + path);
    } catch (const std::exception& e) {
        ui::console::error(std::string("--tune: result could not be stored (") + e.what()
                           + "); the table above is still valid - use --pl "
                           + std::to_string(v.recommended_w) + " directly");
    }
    return 0;
}

}} // namespace mxbm::miner
