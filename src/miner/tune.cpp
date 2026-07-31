#include "miner/tune.h"

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <thread>

#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>

#include "gpu/nvml.h"
#include "miner/benchmark.h"
#include "nlohmann/json.hpp"
#include "ui/console.h"

namespace mxbm { namespace miner {
namespace {

using nlohmann::json;

// Under sudo the store must land in the INVOKING user's config dir: --tune runs as
// root (NVML writes demand it) but --pl auto runs unprivileged every day, and a file
// in /root/.config is one the daily launch will never see.
std::string config_home() {
    if (const char* x = std::getenv("XDG_CONFIG_HOME"); x && *x) return x;
    if (geteuid() == 0) {
        if (const char* su = std::getenv("SUDO_USER"); su && *su)
            if (const passwd* pw = getpwnam(su)) return std::string(pw->pw_dir) + "/.config";
    }
    if (const char* h = std::getenv("HOME"); h && *h) return std::string(h) + "/.config";
    return ".";
}

// Root-created files in a user directory must end up owned by the user, or the next
// --tune run WITHOUT sudo cannot rewrite its own store.
void chown_to_sudo_user(const std::string& path) {
    if (geteuid() != 0) return;
    const char* u = std::getenv("SUDO_UID");
    const char* g = std::getenv("SUDO_GID");
    if (!u || !g) return;
    (void)chown(path.c_str(), (uid_t)atoi(u), (gid_t)atoi(g));
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
        ui::console::error("--tune sets power limits, and every NVML write needs root. "
                           "Re-run under sudo (the result is stored in your own config "
                           "dir, not root's, so daily runs stay unprivileged)");
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
    ui::console::info(fmt("Tuning %s: %zu power points x %d s, plus a warmup and a "
                          "drift gauge (~%d min). Ctrl+C aborts and restores %u W.",
                          device_key.c_str(), caps.size(), per,
                          (int)((caps.size() + 2) * (per + 6) / 60) + 1, pl0.current_w));

    // Restore on EVERY exit from here down. The original limit, not the default:
    // a user who had already capped the card outside MXBM gets their value back.
    struct Restore {
        unsigned w; ~Restore() { gpu::nvml_set_power_limit(w); }
    } restore{pl0.current_w};

    // One measured point: set the cap, let the governor settle, then run the SAME
    // Engine path mining runs while a sampler thread averages the actual board draw
    // (the first fifth is dropped -- it straddles the clock transition).
    auto measure = [&](unsigned cap, TunePoint& out) -> bool {
        if (gpu::nvml_set_power_limit(cap) != gpu::NvmlWrite::Ok) {
            ui::console::error(fmt("--tune: setting %u W failed mid-sweep; aborting", cap));
            return false;
        }
        if (!settle(5, stop)) return false;
        std::atomic<bool> done{false};
        std::vector<double> draws;
        std::thread sampler([&] {
            while (!done.load(std::memory_order_relaxed)) {
                const gpu::Telemetry t = gpu::nvml_sample(0);
                if (t.have_power) draws.push_back(t.power_w);
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        });
        const BenchmarkResult r = run_benchmark(solver, stats, per, stop);
        done.store(true, std::memory_order_relaxed);
        sampler.join();
        double mean = 0.0;
        const size_t skip = draws.size() / 5;
        for (size_t i = skip; i < draws.size(); ++i) mean += draws[i];
        if (draws.size() > skip) mean /= (double)(draws.size() - skip);
        out = TunePoint{cap, mean, r.sol_per_s, r.median_ms};
        return !stop.load(std::memory_order_relaxed);
    };

    ui::console::info(fmt("  warmup at %u W (%d s, discarded)...", caps.front(), per));
    {
        TunePoint scratch;
        if (!measure(caps.front(), scratch)) {
            ui::console::info("--tune aborted; power limit restored");
            return 1;
        }
    }

    std::vector<TunePoint> points;
    for (unsigned cap : caps) {
        ui::console::info(fmt("  measuring %u W (%d s)...", cap, per));
        TunePoint p;
        if (!measure(cap, p)) {
            ui::console::info("--tune aborted; power limit restored");
            return 1;
        }
        ui::console::info(fmt("    %u W: %.2f sol/s, %.1f ms/solve, draw %.1f W",
                              cap, p.sol_s, p.median_ms, p.draw_w));
        points.push_back(p);
    }

    // The drift gauge: the first point again, at the end. Today's numbers carry at
    // least this bar -- the 2026-07-31 session measured ~2 %/arm of thermal drift,
    // which is what made un-gauged sweeps lie.
    ui::console::info(fmt("  drift gauge: %u W again (%d s)...", caps.front(), per));
    TunePoint again;
    double drift_pct = 0.0;
    if (measure(caps.front(), again) && points.front().median_ms > 0.0) {
        drift_pct = (again.median_ms - points.front().median_ms)
                    / points.front().median_ms * 100.0;
    } else if (stop.load(std::memory_order_relaxed)) {
        ui::console::info("--tune aborted; power limit restored");
        return 1;
    }

    ui::console::info("Power curve (miner loop, CPU-verified sol/s):");
    ui::console::info("    cap W   draw W    sol/s   ms/solve   sol/s/W");
    for (const TunePoint& p : points) {
        const double w = p.draw_w > 0.0 ? p.draw_w : (double)p.cap_w;
        ui::console::info(fmt("    %5u  %7.1f  %7.2f  %9.1f  %8.4f",
                              p.cap_w, p.draw_w, p.sol_s, p.median_ms,
                              w > 0.0 ? p.sol_s / w : 0.0));
    }
    ui::console::info(fmt("  drift over the sweep: %+.1f %% at %u W (re-measured last)%s",
                          drift_pct, caps.front(),
                          std::abs(drift_pct) > 1.5
                              ? " - above the 1.5 % gauge, treat the table as +/- that"
                              : ""));

    const TuneVerdict v = tune_verdict(points, cfg.knee_marginal);
    ui::console::info(fmt("Recommended: --pl %u  (above it, each watt returns less "
                          "than %.2f sol/s)", v.knee_w, cfg.knee_marginal));
    ui::console::info(fmt("Best efficiency: %u W", v.eff_w));

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
        j[device_key] = {{"knee_w", v.knee_w}, {"eff_w", v.eff_w},
                         {"knee_marginal", cfg.knee_marginal}, {"date", date},
                         {"drift_pct", drift_pct}, {"points", pts}};
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
