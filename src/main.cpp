// mxbm: the runnable miner binary. Wires the Phase A + Phase B pieces
// together -- parse argv -> merge a config file -> connect+login to the pool
// -> (if the Beam oracle is available) solve jobs and submit shares -> track
// stats and report them via the console ticker and the /summary API -> block
// forever reading the stratum socket. Ctrl+C exits; see
// stratum::Client::run()'s doc comment for why there's no return path or
// cleanup here (the reference miner-style).
#include <chrono>
#include <atomic>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <string>

#include "api/http_summary.h"
#include "cli/options.h"
#include "config/config.h"
#include "miner/stats.h"
#include "ui/console.h"
#include "ui/ticker.h"
#include "pow/difficulty.h"
#include "sha256/sha256.h"   // vendored: void sha256(const uint8_t* d, size_t n, uint8_t out[32]);
#include "stratum/client.h"
#include "stratum/messages.h"
#include "miner/engine.h"
#include "miner/benchmark.h"
#include "version.h"
#include "gpu/nvml.h"
#ifdef MXBM_HAVE_OPENCL
#include "gpu/gpu_solver.h"
#endif
#ifdef MXBM_HAVE_CUDA
#include "gpu/cuda_solver.h"
#endif
#ifdef MXBM_HAVE_BEAM_ORACLE
#include "miner/solver_ref.h"
#endif

using namespace mxbm;

namespace {

// Strict hex decode: `hex` must be exactly out_len*2 lowercase-hex-digit
// characters. Any other character (including uppercase) or a length
// mismatch fails without touching `out`. Same contract as
// miner::Engine's file-local helper of the same name (each TU keeps its
// own copy); used below to recover the 104-byte solution bytes from a
// stratum::Solution's hex `output` field so its achieved difficulty can be
// computed for the "Found a share" console line and the Stats best-share/
// share-found bookkeeping. Solver-agnostic (plain hex decoding, no
// GPU/oracle dependency) and so, unlike the includes above, unconditional:
// its only call site is the submit_fn lambda below, which is itself
// runtime-guarded on `if (solver)` rather than compile-time-guarded, since
// which solver backend (if any) compiled in is now a build-time question
// separate from whether one was actually selected/available at runtime.
bool from_hex_strict(const std::string& hex, uint8_t* out, size_t out_len) {
    if (hex.size() != out_len * 2) return false;
    for (size_t i = 0; i < out_len; ++i) {
        int hi = -1, lo = -1;
        char ch = hex[i * 2];
        if (ch >= '0' && ch <= '9') hi = ch - '0';
        else if (ch >= 'a' && ch <= 'f') hi = ch - 'a' + 10;
        char cl = hex[i * 2 + 1];
        if (cl >= '0' && cl <= '9') lo = cl - '0';
        else if (cl >= 'a' && cl <= 'f') lo = cl - 'a' + 10;
        if (hi < 0 || lo < 0) return false;
        out[i] = (uint8_t)((hi << 4) | lo);
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    cli::Options opts;
    std::string err;
    if (!cli::parse_args(argc, argv, opts, err)) {
        if (opts.version_requested) {
            // The one flag whose false return carries an empty err (see
            // options.h): print the version string ourselves and exit,
            // before any of the rest of main() (signal setup, console
            // init/banner) ever runs.
            std::fputs(("MXBM " + std::string(mxbm::version()) + "\n").c_str(), stdout);
            return 0;
        }
        if (opts.help_requested) {
            std::fputs(err.c_str(), stdout);
            return 0;
        }
        std::fputs(err.c_str(), stderr);
        return 1;
    }

    // Config-file merge (Task 5): fills in only the fields the CLI itself
    // left unseen (cli::Options::Seen) -- CLI-supplied values always win.
    // Precedence when both --json and --config appear on the same command
    // line: --json wins. opts.use_json_config latches true the instant
    // --json is seen and nothing ever clears it, so checking it first here
    // already gives "--json wins" regardless of argv order -- even though
    // the two flags share opts.config_path, so a --config appearing AFTER
    // --json can still overwrite the path text itself (see config/config.h
    // and the Task 5 report for the full nuance around that shared field).
    //
    // Captured before the merge below can flip opts.seen.pools true, so the
    // dropped-CLI-credentials warning further down can tell "the CLI gave
    // no --pool at all" apart from "the config file supplied the pools".
    bool cli_pools = opts.seen.pools;
    if (opts.use_json_config) {
        if (!config::load_json_config(opts.config_path, opts.json_profile, opts, err)) {
            std::fputs((err + "\n").c_str(), stderr);
            return 1;
        }
    } else if (!opts.config_path.empty()) {
        if (!config::load_flat_config(opts.config_path, opts, err)) {
            std::fputs((err + "\n").c_str(), stderr);
            return 1;
        }
    }

    // The CLI no longer requires --pool by itself (a config file may supply
    // POOLS/POOL instead -- see cli::parse_args's doc comment) -- so the
    // "at least one pool" check moved here, after the config-merge above
    // has had its chance to fill opts.pools in.
    const bool benchmark_mode = !opts.benchmark.empty();
    if (opts.pools.empty() && !benchmark_mode) {
        std::fputs("missing --pool (or config with POOLS)\n", stderr);
        return 1;
    }

    // OpenSSL's internal write() has no MSG_NOSIGNAL and SO_NOSIGPIPE is
    // Darwin-only: on Linux a pool dropping mid-TLS-write would SIGPIPE-kill
    // the process instead of surfacing a write error to the reconnect loop.
    std::signal(SIGPIPE, SIG_IGN);

    ui::console::init(opts.nocolor);
    ui::console::banner();

    // Out-of-scope-for-this-phase flags still get a console acknowledgment
    // rather than silently doing nothing: failover past pool 1, GPU device
    // SELECTION (which specific device -- the GPU solver itself now exists
    // and always uses the runtime's default device), and the watchdog are
    // none of them implemented yet.
    if (opts.pools.size() > 1) {
        ui::console::info("Failover pools configured - failover across pools is not implemented yet; using pool 1");
    }
    if (!opts.devices.empty()) {
        ui::console::info("--devices noted - device selection arrives in Phase D; currently ignored");
    }
    if (opts.watchdog_requested) {
        ui::console::info("--watchdog noted - watchdog monitoring is not implemented yet");
    }

    // A config file's own POOLS/POOL entries carry their own USER/PASS/TLS.
    // When the CLI supplied --user/--pass/--tls but no --pool at all, those
    // CLI credentials never bind to anything and the config's own USER wins
    // silently -- surface that instead of leaving it silent (cheap
    // mitigation; see the Task 5 report for the fuller Phase C redesign
    // this stands in for).
    if ((opts.seen.user || opts.seen.pass || opts.seen.tls) && !cli_pools && !opts.pools.empty()) {
        ui::console::info("Note: command-line --user/--pass/--tls were ignored; using pool credentials from the config file.");
    }

    miner::Stats stats;

    if (!benchmark_mode) {
        ui::console::connecting(opts.pools[0].host, opts.pools[0].port, opts.pools[0].tls);
    }

    stratum::Client client;

    // connect -> if ok: connected(); if not, note it and let client.run()'s
    // own reconnect loop keep retrying below. login() runs unconditionally to
    // store the api_key credential even on initial connection failure, so the
    // reconnect loop has valid credentials for its re-login attempts.
    if (!benchmark_mode) {
        auto connect_t0 = std::chrono::steady_clock::now();
        bool up = client.connect(opts.pools[0].host, opts.pools[0].port, opts.pools[0].tls);
        auto connect_t1 = std::chrono::steady_clock::now();
        stats.record_connect(
            opts.pools[0].host + ":" + std::to_string(opts.pools[0].port),
            std::chrono::duration_cast<std::chrono::milliseconds>(connect_t1 - connect_t0).count());
        if (up) {
            ui::console::connected(opts.pools[0].tls);
        } else {
            ui::console::error("Initial connection failed — will keep retrying");
        }
        client.login(opts.pools[0].user);   // stores api_key for reconnect re-login even if send fails
    }

    // Solver selection (Phase C Task 4): pick a backend for opts.solver
    // ("gpu" | "ref" | "auto" -- validated by cli::parse_args, so no other
    // value can reach here) into a std::unique_ptr<miner::Solver>. Declared
    // BEFORE `engine` below so `engine` (which only ever borrows a
    // `Solver&` into *solver) is torn down first at scope exit -- C++
    // destroys locals in reverse declaration order, so this ordering alone
    // guarantees the reference never dangles.
    //
    // Selection order, matching the brief exactly:
    //   1. "gpu" (explicit) or "auto": gpu::GpuSolver, but ONLY when built
    //      with OpenCL AND a device is actually present
    //      (GpuSolver::available()) -- its constructor throws ClError with
    //      no device, so availability is checked first and never relied on
    //      to fail safely by itself. A construction failure that slips
    //      through anyway (e.g. alloc_pipeline exceeding a memory-starved
    //      device's headroom even though a device is present) is caught
    //      and reported rather than crashing the whole binary.
    //   2. "ref" (explicit) or "auto" falling back because step 1 didn't
    //      claim it: miner::SolverRef, only when built with the Beam oracle.
    //   3. Neither claimed it: `solver` stays null -- monitor-only, exactly
    //      like today's no-oracle build, with a reason printed below.
    std::unique_ptr<miner::Solver> solver;
    // Set only if a GPU construction attempt was actually made and threw
    // (available() said yes, alloc_pipeline said no -- e.g. a memory-
    // starved device) -- distinguishes that case below from "never even
    // attempted", so the fallback message doesn't claim "no GPU is
    // available" right under a console line that just said otherwise.
    bool gpu_attempt_failed = false;
    // Worker/device label shown in the stats table, the /summary API, and the
    // "Found a share" line. Defaults to the GPU (the default backend); set to
    // the CPU reference below only when SolverRef is actually chosen.
    std::string worker_label = "GPU 0";

#ifdef MXBM_HAVE_CUDA
    // CUDA first when both are built: it measures ~1.16x the OpenCL path's rate on the
    // same card (docs/performance.md). --solver opencl forces the portable path.
    if ((opts.solver == "cuda" || opts.solver == "gpu" || opts.solver == "auto")
        && gpu::CudaSolver::available()) {
        try {
            auto cs = std::make_unique<gpu::CudaSolver>();
            char line[160];
            std::snprintf(line, sizeof line, "CUDA solver ready: %s (%.1f GiB, %u SMs)",
                          cs->device().name.c_str(),
                          cs->device().global_mem / 1073741824.0,
                          cs->device().compute_units);
            ui::console::info(line);
            worker_label = cs->device().name;   // the table showed "GPU 0" for both backends
            solver = std::move(cs);
        } catch (const std::exception& e) {
            ui::console::error(std::string("CUDA solver initialization failed: ") + e.what());
            gpu_attempt_failed = true;
        }
    }
#endif
#ifdef MXBM_HAVE_OPENCL
    if (!solver && (opts.solver == "opencl" || opts.solver == "gpu" || opts.solver == "auto")
        && gpu::GpuSolver::available()) {
        try {
            auto gs = std::make_unique<gpu::GpuSolver>();
            char line[160];
            std::snprintf(line, sizeof line, "OpenCL solver ready: %s (%.1f GiB, %u compute units)",
                          gs->device().name.c_str(),
                          gs->device().global_mem / 1073741824.0,
                          gs->device().compute_units);
            ui::console::info(line);
            worker_label = gs->device().name;
            solver = std::move(gs);
        } catch (const std::exception& e) {
            ui::console::error(std::string("OpenCL solver initialization failed: ") + e.what());
            gpu_attempt_failed = true;
        }
    }
#endif
#ifdef MXBM_HAVE_BEAM_ORACLE
    if (!solver && (opts.solver == "ref" || opts.solver == "auto")) {
        solver = std::make_unique<miner::SolverRef>();
        worker_label = "CPU 0 reference";
    }
#endif
    stats.set_device_label(worker_label);

    // Device telemetry, if the platform can supply it. NVML is dlopen'd, ships with the
    // NVIDIA driver rather than the toolkit, and is independent of which solver backend
    // was chosen -- so an OpenCL run on an NVIDIA card gets it too.
    if (solver && gpu::nvml_init()) {
        stats.set_telemetry_source([](miner::Stats::Snapshot& s) {
            const gpu::Telemetry t = gpu::nvml_sample();
            s.has_power = t.have_power;         s.power_w       = t.power_w;
            s.has_sm_clock = t.have_sm;         s.sm_clock_mhz  = t.sm_clock_mhz;
            s.has_mem_clock = t.have_mem;       s.mem_clock_mhz = t.mem_clock_mhz;
            s.has_temp = t.have_temp;           s.temp_c        = t.temp_c;
            s.has_fan = t.have_fan;             s.fan_pct       = t.fan_pct;
        });
    }
    if (!solver && gpu_attempt_failed) {
        ui::console::info("Falling back to monitoring jobs only (no solving)");
    } else if (!solver) {
        if (opts.solver == "gpu") {
            ui::console::error("--solver gpu requested but no usable GPU backend is available "
                               "(no CUDA or OpenCL device, or built without either) - monitoring jobs only (no solving)");
        } else if (opts.solver == "cuda") {
            ui::console::error("--solver cuda requested but no usable CUDA device is available "
                               "(needs Ampere or newer with room for the full 2^25 seed layer, "
                               "or this build has no CUDA support) - monitoring jobs only (no solving)");
        } else if (opts.solver == "opencl") {
            ui::console::error("--solver opencl requested but no OpenCL device is available "
                               "(or built without OpenCL support) - monitoring jobs only (no solving)");
        } else if (opts.solver == "ref") {
            ui::console::error("--solver ref requested but this build has no Beam reference oracle - monitoring jobs only (no solving)");
        } else {
            ui::console::info("No solver backend available (no CUDA or OpenCL device, and no Beam oracle build) - monitoring jobs only (no solving)");
        }
    }

    // --benchmark: solve synthetic jobs and report, then exit. Placed AFTER
    // solver selection and telemetry so a benchmark run reports the same
    // device line, the same ticker cadence and the same stats table as mining
    // -- the whole value of this mode is that its sol/s is comparable to the
    // mining figure, which only holds if it comes from the same code.
    if (benchmark_mode) {
        if (!solver) {
            ui::console::error("--benchmark needs a working solver backend; none is available");
            return 1;
        }
        char line[192];
        if (opts.benchmark_seconds > 0) {
            std::snprintf(line, sizeof line,
                "Benchmarking %s for %ds - no pool, no wallet",
                opts.benchmark.c_str(), opts.benchmark_seconds);
        } else {
            std::snprintf(line, sizeof line,
                "Benchmarking %s - no pool, no wallet (Ctrl+C to stop)",
                opts.benchmark.c_str());
        }
        ui::console::info(line);

        // Ctrl+C ends the run and still prints the summary, rather than
        // killing the process and losing it. The flag is checked between
        // solves; request_abort() additionally cuts short a solve already in
        // flight, so the wait is bounded by a round, not a whole solve.
        static std::atomic<bool>* s_stop = nullptr;
        static miner::Solver* s_solver = nullptr;
        std::atomic<bool> stop{false};
        s_stop = &stop;
        s_solver = solver.get();
        std::signal(SIGINT, [](int) {
            if (s_stop) s_stop->store(true, std::memory_order_relaxed);
            if (s_solver) s_solver->request_abort();
        });

        ui::Ticker ticker;
        ticker.start(stats, opts.shortstats, opts.longstats);
        miner::BenchmarkResult r;
        try {
            r = miner::run_benchmark(*solver, stats, opts.benchmark_seconds, stop);
        } catch (const std::exception& e) {
            // A backend can construct successfully and still fail on first
            // launch -- most commonly out of memory, because available() sees
            // free VRAM at startup that another process has taken by the time
            // a kernel runs. Report it; do not let it reach the runtime and
            // core-dump, which reads like a solver bug rather than contention.
            ticker.stop();
            std::signal(SIGINT, SIG_DFL);
            ui::console::error(std::string("Benchmark failed: ") + e.what());
            ui::console::info("If another process is using the GPU, stop it and retry: "
                              "a full BeamHash III search needs ~7.5 GiB free.");
            return 1;
        }
        ticker.stop();
        std::signal(SIGINT, SIG_DFL);

        // Quote sol/s as the headline (what pools and other miners report) but
        // print solves and solutions-per-solve alongside, because sol/s is the
        // product of the two and a change in either moves it. p5/p95 are there
        // so a run can be judged stable or not without a second run.
        std::snprintf(line, sizeof line, "Benchmark: %llu solves in %.1fs",
                      (unsigned long long)r.solves, r.elapsed_s);
        ui::console::info(line);
        std::snprintf(line, sizeof line,
            "  %.1f sol/s   (%.2f verified solutions/solve, %.2f solves/s)",
            r.sol_per_s, r.per_solve_avg, r.solves_per_s);
        ui::console::info(line);
        std::snprintf(line, sizeof line,
            "  %.1f ms/solve median   (p5 %.1f, p95 %.1f)",
            r.median_ms, r.p5_ms, r.p95_ms);
        ui::console::info(line);
        if (r.solves < 100) {
            ui::console::info("  note: fewer than 100 solves - too few to quote a margin; "
                              "use --benchmark-seconds to run longer");
        }
        return 0;
    }

    // Engine, submit_fn, and on_attempt are all solver-agnostic (Engine
    // takes any miner::Solver&) and so, like from_hex_strict above, are
    // compiled unconditionally and simply skipped at runtime when no
    // backend claimed a solver -- rather than compile-time-guarded, since
    // "was a solver built in" and "was one actually selected/available"
    // are now two separate questions.
    std::unique_ptr<miner::Engine> engine;
    if (solver) {
        engine = std::make_unique<miner::Engine>(client, *solver);
        engine->submit_fn = [&client, &stats, worker_label](const stratum::Solution& s) {
            // Achieved difficulty for the "Found a share" line and the best-share
            // stat: decode the 104-byte solution back out of its hex `output`
            // field, SHA-256 it (same predicate Engine's own clears_difficulty()
            // used to decide this was worth submitting), and convert to display
            // units. A decode failure "can't happen" (output is always Engine's
            // own to_hex() of a fresh 104-byte candidate) but is handled
            // defensively: skip the console line and the best-share update, still
            // record the submit and still send it -- the actual submission must
            // never be gated on cosmetic/stat reporting.
            uint8_t soln[104];
            if (from_hex_strict(s.output, soln, sizeof soln)) {
                uint8_t hash[32];
                sha256(soln, sizeof soln, hash);
                double units = pow::achieved_units(hash);
                ui::console::share_found(worker_label.c_str(), units);
                stats.record_share_found(units);
            }
            stats.record_submit(s.id);
            client.submit(s);
        };
        engine->on_attempt = [&stats](uint32_t candidates) { stats.record_attempt(candidates); };
    }

    client.on_result = [&opts, &stats](const stratum::Result& r) {
        if (r.id == "login") {
            if (r.code == 0) {
                ui::console::authorized(opts.pools[0].user);
                ui::console::start_mining();
            } else {
                ui::console::error("Login failed (" + std::to_string(r.code) + "): " + r.description);
            }
        } else {
            stats.record_result(r.code);
            ui::console::share_result(r.code, r.description, stats.snapshot().last_latency_ms);
        }
    };

    client.on_job = [&](const stratum::Job& j) {
        stats.record_job(j.id, pow::to_display_units(j.difficulty));
        ui::console::job(j.id, j.difficulty, j.height);
        if (engine) engine->on_job(j);
    };

    client.on_disconnect = [&stats]() {
        ui::console::disconnected();
        stats.record_disconnect();
    };

    ui::Ticker ticker;
    ticker.start(stats, opts.shortstats, opts.longstats);

    api::HttpSummary http_api;
    if (opts.apiport) {
        if (!http_api.start(static_cast<uint16_t>(opts.apiport), stats, mxbm::version())) {
            ui::console::error("API server failed to start on port " + std::to_string(opts.apiport));
        }
        // Either way, mining continues below -- the API is a convenience,
        // never a mining precondition.
    }

    if (engine) engine->start();

    client.run();   // blocks forever, reconnecting on drop; Ctrl+C exits
    return 0;
}
