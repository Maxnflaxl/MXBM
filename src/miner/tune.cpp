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
#include "miner/benchmark.h"
#include "nlohmann/json.hpp"
#include "ui/console.h"

namespace mxbm { namespace miner {
namespace {

using nlohmann::json;

// Under sudo the store must land in the INVOKING user's config dir: --tune runs as
// root (NVML writes demand it) but --pl auto runs unprivileged every day, and a file
// in /root/.config is one the daily launch will never see. Windows has no such
// indirection to undo -- an elevated prompt keeps the same user profile, so
// %APPDATA% is already where the daily launch will look.
std::string config_home() {
#ifdef _WIN32
    if (const char* a = std::getenv("APPDATA"); a && *a) return a;
    return ".";
#else
    if (const char* x = std::getenv("XDG_CONFIG_HOME"); x && *x) return x;
    if (geteuid() == 0) {
        if (const char* su = std::getenv("SUDO_USER"); su && *su)
            if (const passwd* pw = getpwnam(su)) return std::string(pw->pw_dir) + "/.config";
    }
    if (const char* h = std::getenv("HOME"); h && *h) return std::string(h) + "/.config";
    return ".";
#endif
}

// Root-created files in a user directory must end up owned by the user, or the next
// --tune run WITHOUT sudo cannot rewrite its own store. (On Windows elevation does
// not change the owning user; nothing to fix up.)
void chown_to_sudo_user(const std::string& path) {
#ifdef _WIN32
    (void)path;
#else
    if (geteuid() != 0) return;
    const char* u = std::getenv("SUDO_UID");
    const char* g = std::getenv("SUDO_GID");
    if (!u || !g) return;
    (void)chown(path.c_str(), (uid_t)atoi(u), (gid_t)atoi(g));
#endif
}

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

std::string tune_store_path() { return config_home() + "/mxbm/tune.json"; }

unsigned tune_stored_knee(const std::string& device_key, std::string& date_out) {
    std::ifstream f(tune_store_path());
    if (!f) return 0;
    try {
        json j; f >> j;
        const auto it = j.find(device_key);
        if (it == j.end()) return 0;
        date_out = it->value("date", std::string());
        return it->value("knee_w", 0u);
    } catch (...) { return 0; }
}

bool tune_stored_rung(const std::string& device_key, unsigned& mhz_out,
                      unsigned& below_w_out) {
    std::ifstream f(tune_store_path());
    if (!f) return false;
    try {
        json j; f >> j;
        const auto it = j.find(device_key);
        if (it == j.end()) return false;
        mhz_out = it->value("rung_mhz", 0u);
        below_w_out = it->value("rung_below_w", 0u);
        return mhz_out != 0 && below_w_out != 0;
    } catch (...) { return false; }
}

int run_tune(Solver& solver, Stats& stats, const std::string& device_key,
             const TuneConfig& cfg, std::atomic<bool>& stop) {
    const gpu::PowerLimit pl0 = gpu::nvml_power_limit();
    if (!pl0.valid) {
        ui::console::error("--tune: this card reports no power limit, so there is "
                           "no knob to sweep");
        return 1;
    }
    // Permission probe FIRST, as a no-op write of the value already on the card:
    // a refusal here proves nothing was changed, and every later write would have
    // failed the same way.
    if (gpu::nvml_set_power_limit(pl0.current_w) != gpu::NvmlWrite::Ok) {
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
                           "a knee; this card's band leaves fewer");
        return 1;
    }

    const int per = cfg.seconds_per_point;
    ui::console::info(fmt("Tuning %s: %zu power points x %d s%s, plus a warmup and a "
                          "drift gauge (~%d min%s). Ctrl+C aborts and restores %u W.",
                          device_key.c_str(), caps.size(), per,
                          cfg.refine ? ", then knee- and efficiency-refining passes" : "",
                          (int)((caps.size() + 2) * (per + 6) / 60) + 1,
                          cfg.refine ? " + the refinements" : "", pl0.current_w));

    // Restore on EVERY exit from here down. The original limit, not the default:
    // a user who had already capped the card outside MXBM gets their value back.
    // The memory-clock lock is reset only if the rung pass set it -- a user's own
    // --mclk (which disables the pass) must not be unlocked out from under them.
    struct Restore {
        unsigned w; bool mem = false;
        ~Restore() {
            if (mem) gpu::nvml_reset_locked_mem_clock();
            gpu::nvml_set_power_limit(w);
        }
    } restore{pl0.current_w};

    // One measured point: set the cap (and, for rung arms, lock the memory clock),
    // let the governor settle, then run the SAME Engine path mining runs while a
    // sampler thread records draw and memory clock. A rung arm whose sampled clock
    // did not HOLD the rung is REFUSED (cap_w = 0), never a data point -- the 10501
    // null was a refused rung, and a refusal must not masquerade as "does not pay".
    auto measure = [&](unsigned cap, unsigned mclk, TunePoint& out) -> bool {
        if (gpu::nvml_set_power_limit(cap) != gpu::NvmlWrite::Ok) {
            ui::console::error(fmt("--tune: setting %u W failed mid-sweep; aborting", cap));
            return false;
        }
        if (mclk) {
            if (gpu::nvml_set_locked_mem_clock(mclk, mclk) != gpu::NvmlWrite::Ok) {
                ui::console::info(fmt("    %u MHz memory lock refused; skipping rung arms", mclk));
                out = TunePoint{};
                return !stop.load(std::memory_order_relaxed);
            }
            restore.mem = true;
        }
        if (!settle(5, stop)) return false;
        std::atomic<bool> done{false};
        std::vector<double> draws;
        std::vector<unsigned> mems;
        std::thread sampler([&] {
            while (!done.load(std::memory_order_relaxed)) {
                const gpu::Telemetry t = gpu::nvml_sample(0);
                if (t.have_power) draws.push_back(t.power_w);
                if (t.have_mem) mems.push_back(t.mem_clock_mhz);
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        });
        unsigned long long e0 = 0, e1 = 0;
        const bool e_ok = gpu::nvml_total_energy_mj(e0);
        const BenchmarkResult r = run_benchmark(solver, stats, per, stop);
        done.store(true, std::memory_order_relaxed);
        sampler.join();
        if (mclk) {
            std::sort(mems.begin(), mems.end());
            const unsigned med = mems.empty() ? 0u : mems[mems.size() / 2];
            if (med + 5u < mclk || med > mclk + 5u) {
                ui::console::info(fmt("    %u W at %u MHz: REFUSED (memory ran at %u MHz)"
                                      " - dropped", cap, mclk, med));
                out = TunePoint{};
                return !stop.load(std::memory_order_relaxed);
            }
        }
        // Draw, best instrument first: the card's energy counter gives exact joules
        // over the window (no sampling error, and it covers the settle-free window
        // precisely); the 2 Hz sampler mean is the fallback for cards without it.
        double mean = 0.0;
        const size_t skip = draws.size() / 5;
        for (size_t i = skip; i < draws.size(); ++i) mean += draws[i];
        if (draws.size() > skip) mean /= (double)(draws.size() - skip);
        if (e_ok && gpu::nvml_total_energy_mj(e1) && e1 > e0 && r.elapsed_s > 0.0)
            mean = (double)(e1 - e0) / 1000.0 / r.elapsed_s;
        out = TunePoint{cap, mean, r.sol_per_s, r.median_ms};
        return !stop.load(std::memory_order_relaxed);
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
    for (unsigned cap : caps) {
        ui::console::info(fmt("  measuring %u W (%d s)...", cap, per));
        TunePoint p;
        if (!measure(cap, 0, p)) {
            ui::console::info("--tune aborted; power limit restored");
            return 1;
        }
        ui::console::info(fmt("    %u W: %.2f sol/s, %.1f ms/solve, draw %.1f W",
                              cap, p.sol_s, p.median_ms, p.draw_w));
        points.push_back(p);
    }

    // Pass 2: the coarse grid can only place the knee to within its own spacing, so
    // refine the two intervals touching it with ~10 W steps and fold the new points
    // into the SAME verdict. Skipped for an explicit --tune-caps list (a chosen grid
    // means exactly those points) and when the bracket is already tighter than a step.
    if (cfg.refine) {
        const TuneVerdict coarse = tune_verdict(points, cfg.knee_marginal);
        std::vector<unsigned> fine = tune_refine_caps(points, coarse.knee_w);
        std::sort(fine.begin(), fine.end(), std::greater<unsigned>());
        if (!fine.empty()) {
            ui::console::info(fmt("  pass 2: coarse knee at %u W - refining its "
                                  "neighbourhood with %zu points x %d s (~%d min)...",
                                  coarse.knee_w, fine.size(), per,
                                  (int)(fine.size() * (per + 6) / 60) + 1));
            for (unsigned cap : fine) {
                ui::console::info(fmt("  measuring %u W (%d s)...", cap, per));
                TunePoint p;
                if (!measure(cap, 0, p)) {
                    ui::console::info("--tune aborted; power limit restored");
                    return 1;
                }
                ui::console::info(fmt("    %u W: %.2f sol/s, %.1f ms/solve, draw %.1f W",
                                      cap, p.sol_s, p.median_ms, p.draw_w));
                points.push_back(p);
            }
        }
    }

    // Pass 3: the memory rung. Under a low cap the interface burns watts for
    // bandwidth the slowed core cannot use; a reduced memory clock returns them as
    // core clock (measured 2026-07-31: −8.5 to −14.4 % below ~173 W on the reference
    // card, catastrophic above ~180). The candidate rung comes from THIS card's
    // driver (tune_rung_pick) and the arms run at the coarse caps at or below the
    // knee -- the region where the rung could pay -- so the recommendation's
    // threshold is a measured property of the user's card, not of ours.
    std::vector<TunePoint> rung_pts;
    unsigned rung_mhz = 0;
    if (cfg.rung_pass) {
        rung_mhz = tune_rung_pick(gpu::nvml_supported_mem_clocks(0));
        if (rung_mhz) {
            const TuneVerdict sofar = tune_verdict(points, cfg.knee_marginal);
            std::vector<unsigned> rcaps;
            for (unsigned cap : caps)
                if (cap <= sofar.knee_w) rcaps.push_back(cap);
            if (!rcaps.empty()) {
                ui::console::info(fmt("  pass 3: the %u MHz memory rung at %zu capped "
                                      "points x %d s (~%d min)...",
                                      rung_mhz, rcaps.size(), per,
                                      (int)(rcaps.size() * (per + 6) / 60) + 1));
                for (unsigned cap : rcaps) {
                    ui::console::info(fmt("  measuring %u W at %u MHz (%d s)...",
                                          cap, rung_mhz, per));
                    TunePoint p;
                    if (!measure(cap, rung_mhz, p)) {
                        ui::console::info("--tune aborted; card restored");
                        return 1;
                    }
                    if (p.cap_w == 0) {
                        if (!restore.mem) break;   // lock refused outright: no more arms
                        continue;                  // this arm's clock did not hold
                    }
                    ui::console::info(fmt("    %u W @ %u MHz: %.2f sol/s, %.1f ms/solve, "
                                          "draw %.1f W", cap, rung_mhz, p.sol_s,
                                          p.median_ms, p.draw_w));
                    rung_pts.push_back(p);
                }
                if (restore.mem) {
                    gpu::nvml_reset_locked_mem_clock();
                    restore.mem = false;
                }
            }
        }
    }

    // Pass 4: efficiency refinement. The coarse spacing that blurs the knee blurs
    // the efficiency optimum the same way -- on the reference card the 37 W grid
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
            const unsigned mclk = on_rung ? rung_mhz : 0u;
            for (unsigned cap : ecaps) {
                ui::console::info(mclk
                    ? fmt("  measuring %u W at %u MHz (%d s)...", cap, mclk, per)
                    : fmt("  measuring %u W (%d s)...", cap, per));
                TunePoint p;
                if (!measure(cap, mclk, p)) {
                    ui::console::info("--tune aborted; card restored");
                    return 1;
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
                series.push_back(p);
            }
            if (restore.mem) {
                gpu::nvml_reset_locked_mem_clock();
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
    ui::console::info("Power curve (miner loop, CPU-verified sol/s):");
    ui::console::info("    cap W   draw W    sol/s   ms/solve   sol/s/W");
    for (const TunePoint& p : points) {
        const double w = p.draw_w > 0.0 ? p.draw_w : (double)p.cap_w;
        ui::console::info(fmt("    %5u  %7.1f  %7.2f  %9.1f  %8.4f",
                              p.cap_w, p.draw_w, p.sol_s, p.median_ms,
                              w > 0.0 ? p.sol_s / w : 0.0));
    }
    if (!rung_pts.empty()) {
        std::sort(rung_pts.begin(), rung_pts.end(),
                  [](const TunePoint& a, const TunePoint& b) { return a.cap_w > b.cap_w; });
        ui::console::info(fmt("At the %u MHz memory rung (same loop):", rung_mhz));
        for (const TunePoint& p : rung_pts) {
            const double w = p.draw_w > 0.0 ? p.draw_w : (double)p.cap_w;
            ui::console::info(fmt("    %5u  %7.1f  %7.2f  %9.1f  %8.4f",
                                  p.cap_w, p.draw_w, p.sol_s, p.median_ms,
                                  w > 0.0 ? p.sol_s / w : 0.0));
        }
    }
    ui::console::info(fmt("  drift over the sweep: %+.1f %% at %u W (re-measured last)%s",
                          drift_pct, caps.front(),
                          std::abs(drift_pct) > 1.5
                              ? " - above the 1.5 % gauge, treat the table as +/- that"
                              : ""));

    const TuneVerdict v = tune_verdict(points, cfg.knee_marginal);
    ui::console::info(fmt("Recommended: --pl %u  (above it, each watt returns less "
                          "than %.2f sol/s)", v.knee_w, cfg.knee_marginal));
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
        j[device_key] = {{"knee_w", v.knee_w}, {"eff_w", best_eff_w},
                         {"eff_on_rung", best_on_rung},
                         {"knee_marginal", cfg.knee_marginal}, {"date", date},
                         {"drift_pct", drift_pct}, {"passes", cfg.refine ? 2 : 1},
                         {"points", pts}};
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
        chown_to_sudo_user(dir.string());
        std::ofstream out(path);
        out << j.dump(2) << "\n";
        out.close();
        chown_to_sudo_user(path);
        ui::console::info("Stored for `--pl auto`: " + path);
    } catch (const std::exception& e) {
        ui::console::error(std::string("--tune: result could not be stored (") + e.what()
                           + "); the table above is still valid - use --pl "
                           + std::to_string(v.knee_w) + " directly");
    }
    return 0;
}

}} // namespace mxbm::miner
