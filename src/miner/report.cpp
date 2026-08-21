#include "miner/report.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/utsname.h>
#endif

#include <filesystem>
#include <fstream>

#include "gpu/nvml.h"
#include "gpu/telemetry_window.h"
#include "miner/benchmark.h"
#include "miner/user_paths.h"
#include "ui/console.h"
#include "version.h"

namespace mxbm { namespace miner {
namespace {

std::string fmt(const char* f, ...) {
    char buf[512];
    va_list ap; va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}

// Kernel and release, which is what distinguishes two reports of the same card.
// Windows has no uname; RtlGetVersion is the one call that still reports the real
// build after the manifest-gated APIs started lying about it.
std::string os_string() {
#ifdef _WIN32
    typedef long (__stdcall *RtlGetVersionFn)(void*);
    unsigned long v[40] = {0};
    v[0] = sizeof(v);
    if (HMODULE nt = GetModuleHandleW(L"ntdll.dll"))
        if (auto fn = (RtlGetVersionFn)GetProcAddress(nt, "RtlGetVersion"))
            if (fn(v) == 0)
                return fmt("Windows %lu.%lu build %lu", v[1], v[2], v[3]);
    return "Windows";
#else
    utsname u{};
    if (uname(&u) == 0) return std::string(u.sysname) + " " + u.release;
    return "unknown";
#endif
}

// A number the way the report quotes it: never more digits than the instrument has.
std::string w(double v)  { return fmt("%.1f", v); }
std::string s2(double v) { return fmt("%.2f", v); }

// One curve as a markdown table, collapsed behind a summary. The full point list is
// the most valuable thing in the report and the least readable, so it ships open to
// a reader who wants it and folded for one who does not.
void curve_table(std::ostringstream& o, const std::vector<TunePoint>& pts,
                 const std::string& title, const std::vector<TunePoint>* against) {
    if (pts.empty()) return;
    std::vector<TunePoint> v = pts;
    std::sort(v.begin(), v.end(),
              [](const TunePoint& a, const TunePoint& b) { return a.cap_w > b.cap_w; });
    o << "\n<details>\n<summary>" << title << " (" << v.size() << " points)</summary>\n\n";
    o << "| cap W | draw W | sol/s | ms/solve | sol/s/W |";
    if (against) o << " vs stock |";
    o << "\n|---|---|---|---|---|";
    if (against) o << "---|";
    o << "\n";
    for (const TunePoint& p : v) {
        const double watts = p.draw_w > 0.0 ? p.draw_w : (double)p.cap_w;
        o << "| " << p.cap_w << " | " << w(p.draw_w) << " | " << s2(p.sol_s)
          << " | " << s2(p.median_ms) << " | "
          << fmt("%.4f", watts > 0.0 ? p.sol_s / watts : 0.0) << " |";
        if (against) {
            std::string cell = "—";
            for (const TunePoint& q : *against)
                if (q.cap_w == p.cap_w && q.sol_s > 0.0)
                    cell = fmt("%+.1f %%", (p.sol_s - q.sol_s) / q.sol_s * 100.0);
            o << " " << cell << " |";
        }
        o << "\n";
    }
    o << "</details>\n";
}

} // namespace

int run_report(std::unique_ptr<Solver>& solver, Stats& stats, const ReportConfig& cfg,
               std::atomic<bool>& stop, std::string* collected) {
    if (!solver) {
        ui::console::error("--report needs a working solver backend; none is available");
        return 1;
    }

    // What is already known about this card, and what a sweep would cost. Decided
    // before anything runs so the announcement is the plan, not a guess: the user
    // is being asked for either two minutes or half an hour.
    TuneStore store;
    tune_load(cfg.device_key, store);
    const std::string ver = version();
    bool curve_fresh = store.found && report_curve_fresh(store.mxbm_version, ver);
    const bool can_tune = cfg.have_nvml && tune_can_write(cfg.device);
    const bool will_tune = !curve_fresh && can_tune;

    // Why the curve is or is not being measured, named rather than described: the
    // stored build against the running one, and the sweep's own cost estimate rather
    // than the default sweep's.
    std::string stored_by = store.mxbm_version.empty() ? "an unstamped build"
                                                       : store.mxbm_version;
    ui::console::info(fmt("Reporting on %s (%d s benchmark).", cfg.device_key.c_str(),
                          cfg.seconds));
    if (cfg.device_count > 1)
        ui::console::info(fmt("  This is GPU %u of %u - a report covers one card. Run "
                              "--devices N for the others.",
                              cfg.device_position, cfg.device_count));
    if (curve_fresh) {
        ui::console::info(fmt("  Power curve: reusing this build's, measured %s.",
                              store.date.c_str()));
    } else if (will_tune) {
        const gpu::PowerLimit pl = gpu::nvml_power_limit(cfg.device);
        const unsigned band = (unsigned)tune_default_caps(
            pl.min_w, pl.default_w ? pl.default_w : pl.max_w).size();
        ui::console::info(fmt("  Power curve: sweeping (~%d min) - %s.",
                              tune_estimated_minutes(cfg.tune, band ? band : 6u),
                              store.found
                                  ? fmt("stored curve is %s, this is %s",
                                        stored_by.c_str(), ver.c_str()).c_str()
                                  : "none stored for this card"));
        ui::console::info("  Ctrl+C during the sweep still produces a report.");
    } else if (store.found) {
        ui::console::info(fmt("  Power curve: reusing %s's - re-run as %s to re-measure "
                              "it on %s.", stored_by.c_str(),
#ifdef _WIN32
                              "Administrator", ver.c_str()));
#else
                              "root", ver.c_str()));
#endif
    } else {
        ui::console::info(fmt("  Power curve: skipped - measuring it needs %s.",
#ifdef _WIN32
                              "an Administrator terminal"));
#else
                              "root (sudo mxbm --report)"));
#endif
    }

    // The throughput block, first and always. It is the half that survives an abort,
    // and running it before the sweep means it is taken at the card's own setting
    // rather than at whatever the sweep left behind.
    gpu::TelemetryWindow tw(cfg.device);
    BenchmarkResult r;
    try {
        r = run_benchmark(*solver, stats, cfg.seconds, stop);
    } catch (const std::exception& e) {
        ui::console::error(std::string("--report: the benchmark failed: ") + e.what());
        return 1;
    }
    const gpu::TelemetrySummary t = tw.close(r.elapsed_s);
    ui::console::info(fmt("  %.2f sol/s, %.2f ms/solve median, %llu solves",
                          r.sol_per_s, r.median_ms, (unsigned long long)r.solves));

    if (will_tune && !stop.load(std::memory_order_relaxed)) {
        ui::console::info("Now measuring this card's power curve.");
        TuneConfig tc = cfg.tune;
        tc.device = cfg.device;
        run_tune(solver, stats, cfg.device_key, tc, stop);
        tune_load(cfg.device_key, store);   // whatever the sweep managed to store
        // Both were decided against the curve this run replaced; the sweep has
        // since stamped its own.
        curve_fresh = store.found && report_curve_fresh(store.mxbm_version, ver);
        stored_by = store.mxbm_version.empty() ? "an unstamped build"
                                              : store.mxbm_version;
    }

    // --- the block itself ------------------------------------------------
    std::ostringstream o;
    uint64_t vram_free = 0, vram_total = 0;
    const bool have_vram = cfg.have_nvml && gpu::nvml_memory_info(cfg.device, vram_free, vram_total);
    const std::string nvname = cfg.have_nvml ? gpu::nvml_device_name(cfg.device) : std::string();
    const gpu::PowerLimit pl = cfg.have_nvml ? gpu::nvml_power_limit(cfg.device) : gpu::PowerLimit{};
    unsigned procs = 0;
    const bool have_procs = cfg.have_nvml && gpu::nvml_compute_process_count(cfg.device, procs);

    o << "### MXBM benchmark report\n\n| | |\n|---|---|\n";
    o << "| MXBM | " << ver << " |\n";
    o << "| GPU | " << (nvname.empty() ? cfg.device_name : nvname);
    if (cfg.device_count > 1)
        o << " (GPU " << cfg.device_position << " of " << cfg.device_count
          << " on this rig; the rest are not covered by this report)";
    o << " |\n";
    if (have_vram) o << "| VRAM | " << (vram_total / (1024ULL * 1024ULL)) << " MiB |\n";
    if (cfg.have_nvml) {
        const std::string drv = gpu::nvml_driver_version();
        if (!drv.empty()) o << "| Driver | " << drv << " |\n";
    }
    o << "| OS | " << os_string() << " |\n";
    // "Cuda" is how the device join spells it; CUDA is how everyone else does.
    o << "| Backend | " << (cfg.backend == "Cuda" ? "CUDA" : cfg.backend) << " |\n";
    if (pl.valid)
        o << "| Board power limit | " << pl.current_w << " W (card default "
          << pl.default_w << " W, driver band " << pl.min_w << "–" << pl.max_w << " W) |\n";
    if (t.have_mem) o << "| Memory clock | " << t.mem_clock_mhz << " MHz |\n";
    if (cfg.have_nvml)
        o << "| Display attached to this GPU | "
          << (gpu::nvml_display_active(cfg.device) ? "**yes**" : "no") << " |\n";
    // MXBM is one of the processes it counts, so "just us" is one, not zero. A
    // report that cannot say is worse than one that says it does not know.
    if (have_procs)
        o << "| Other processes on this GPU | "
          << (procs > 1 ? fmt("**%u** - this run measured contention", procs - 1)
                        : std::string("none")) << " |\n";

    o << "\n### Throughput — " << w(r.elapsed_s) << " s, " << r.solves << " solves\n\n";
    o << "| | |\n|---|---|\n";
    o << "| Throughput | **" << s2(r.sol_per_s) << " sol/s** |\n";
    o << "| Solve time | " << s2(r.median_ms) << " ms median (p5 " << s2(r.p5_ms)
      << ", p95 " << s2(r.p95_ms) << ") |\n";
    o << "| Verified solutions/solve | " << s2(r.per_solve_avg) << " |\n";
    if (t.have_power) {
        o << "| Board power | " << w(t.power_w) << " W mean";
        if (t.have_energy) o << " (card energy counter, " << fmt("%.1f", t.joules / 1000.0) << " kJ over the run)";
        o << " |\n";
    }
    if (t.have_sm)   o << "| Core clock | " << t.sm_clock_mhz << " MHz median |\n";
    if (t.have_mem)  o << "| Memory clock | " << t.mem_clock_mhz << " MHz median |\n";
    if (t.have_temp) o << "| Temperature | " << t.temp_c << " °C median |\n";
    if (t.have_power && t.power_w > 0.0 && r.solutions > 0)
        o << "| Efficiency | " << fmt("%.4f", r.sol_per_s / t.power_w) << " sol/s/W · "
          << s2(t.have_energy ? t.joules / (double)r.solutions : t.power_w / r.sol_per_s)
          << " J/solution |\n";
    if (t.have_events) {
        unsigned n = 0;
        const gpu::ClockEventBit* bits = gpu::clock_event_names(n);
        std::string seen;
        for (unsigned i = 0; i < n; ++i)
            if (t.events & bits[i].bit) seen += (seen.empty() ? "" : ", ") + std::string(bits[i].name);
        o << "| Clocks limited by | " << (seen.empty() ? "nothing" : seen) << " |\n";
    }
    if (r.solves < 100)
        o << "\n> Fewer than 100 solves — too short to quote a margin. Re-run with "
             "`--report-seconds 300`.\n";

    // --- the curve -------------------------------------------------------
    if (store.found && !store.points.empty()) {
        o << "\n### Power curve — measured on this card, " << store.date << ", "
          << (store.points.size() + store.points_rung.size()) << " points\n\n";
        // A curve the running build did not measure describes a different program,
        // so it is published with that said rather than left to look current. The
        // reader is the one who has to decide whether it still applies.
        if (!curve_fresh)
            o << "> Curve measured by " << stored_by << "; the throughput above is "
              << ver << ".\n\n";
        o << "| | |\n|---|---|\n";
        if (store.recommended_w)
            o << "| Recommended `--pl` | **" << store.recommended_w
              << " W** — past this, extra watts buy almost no speed |\n";
        if (store.eff_w) {
            o << "| Most efficient `--pl` | **" << store.eff_w << " W**";
            if (store.eff_on_rung && store.rung_mhz)
                o << " with `--mclk " << store.rung_mhz << "`";
            o << " |\n";
        }
        if (store.rung_mhz && store.rung_below_w)
            o << "| Low memory clock | below ~" << store.rung_below_w << " W, `--mclk "
              << store.rung_mhz << "` measured faster on this card |\n";
        else if (!store.points_rung.empty())
            o << "| Low memory clock | never paid at the measured caps — keep stock memory |\n";
        o << "| Drift over the sweep | " << fmt("%+.2f", store.drift_pct)
          << " % (first cap re-measured last"
          << (std::abs(store.drift_pct) > 1.5 ? ", **above the 1.5 % gauge — read the "
                                                "table as ± that**" : ", under the 1.5 % gauge")
          << ") |\n";
        curve_table(o, store.points, "Full curve — stock memory", nullptr);
        curve_table(o, store.points_rung,
                    fmt("Full curve — %u MHz memory clock", store.rung_mhz), &store.points);
    } else if (!can_tune) {
        o << "\n### Power curve\n\nNot measured: setting the power limit needs "
#ifdef _WIN32
             "an Administrator terminal"
#else
             "root"
#endif
             ". Re-running there adds the curve.\n";
    } else {
        o << "\n### Power curve\n\nNot measured (the sweep was interrupted).\n";
    }

    o << "\n### Anything unusual?\n";
    o << "<!-- overclock/undervolt, laptop/eGPU/risers, unusual cooling, another\n"
         "     miner's figure on this card. Leave blank if it was a clean stock run. -->\n";

    const std::string block = o.str();
    ui::console::info("");
    ui::console::info("--------- paste everything below into the issue ---------");
    ui::console::info("");
    // Line by line through the console rather than one fputs to stdout: print_line is
    // what tees to the --log transcript, and a log that recorded the header and not
    // the block would be missing the only part worth pasting.
    for (size_t i = 0; i < block.size();) {
        const size_t nl = block.find('\n', i);
        ui::console::info(block.substr(i, (nl == std::string::npos ? block.size() : nl) - i));
        if (nl == std::string::npos) break;
        i = nl + 1;
    }
    ui::console::info("--------- end ---------");

    if (collected) *collected += block;
    return 0;
}

}} // namespace mxbm::miner
