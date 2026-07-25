// mxbm: the runnable miner binary. Parse argv -> merge a config file ->
// connect+login to the pool -> solve jobs and submit shares -> report stats
// via the ticker and the /summary API -> block forever on the stratum socket.
// Ctrl+C exits; stratum::Client::run() explains why there is no cleanup here.
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
#include "miner/devfee.h"
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
// characters. Any other character (including uppercase) or a length mismatch
// fails without touching `out`. Used by submit_fn below to recover a
// solution's 104 bytes so its achieved difficulty can be computed.
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
            // options.h), so the version string is printed here.
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

    // Config-file merge: fills in only the fields the CLI left unseen, so CLI
    // values always win; --json beats --config when both appear. cli_pools is
    // read first because the merge can flip opts.seen.pools true.
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

    // Rules spanning two options settle once every source has spoken (a log
    // path implies logging). Must run before the algo/pool checks below.
    cli::resolve_implied_options(opts);

    const bool benchmark_mode = !opts.benchmark.empty();
    // Checked here rather than in parse_args: a config file may supply ALGO
    // and POOLS, so these fire only when no source supplied them.
    if (!opts.seen.algo && !benchmark_mode) {
        std::fputs("unsupported algo (pass --algo BEAM-III, or set ALGO in your config)\n", stderr);
        return 1;
    }
    if (opts.pools.empty() && !benchmark_mode) {
        std::fputs("missing --pool (or config with POOLS)\n", stderr);
        return 1;
    }

    // OpenSSL's internal write() has no MSG_NOSIGNAL and SO_NOSIGPIPE is
    // Darwin-only: on Linux a pool dropping mid-TLS-write would SIGPIPE-kill
    // the process instead of surfacing a write error to the reconnect loop.
    std::signal(SIGPIPE, SIG_IGN);

    ui::console::init(opts.nocolor);

    // Open the transcript BEFORE the banner, so the log starts with the same
    // first line the screen does rather than joining part-way through. A log
    // that cannot be opened is reported and mining continues regardless.
    if (opts.log_enabled) {
        std::string log_path;
        if (ui::console::open_log(opts.log_path, &log_path)) {
            ui::console::info("Logging to " + log_path);
        } else {
            ui::console::error("Could not open log file" +
                               (opts.log_path.empty() ? std::string() : " " + opts.log_path) +
                               " - continuing without a log");
        }
    }

    ui::console::banner();

    // Flags that are accepted but not yet implemented get a console
    // acknowledgment rather than silently doing nothing.
    if (opts.pools.size() > 1) {
        ui::console::info("Failover pools configured - failover across pools is not implemented yet; using pool 1");
    }
    if (!opts.devices.empty()) {
        ui::console::info("--devices noted - device selection is not implemented yet; currently ignored");
    }
    if (opts.watchdog_requested) {
        ui::console::info("--watchdog noted - watchdog monitoring is not implemented yet");
    }

    // A config file's POOLS entries carry their own USER/PASS/TLS, so CLI
    // --user/--pass/--tls given without any --pool bind to nothing. Say so.
    if ((opts.seen.user || opts.seen.pass || opts.seen.tls) && !cli_pools && !opts.pools.empty()) {
        ui::console::info("Note: command-line --user/--pass/--tls were ignored; using pool credentials from the config file.");
    }

    miner::Stats stats;

    // Hardware first, pool second: a device that failed to initialise is the
    // reason a pool connection would be pointless.
    ui::console::setup_miner();
#ifdef MXBM_HAVE_CUDA
    if (gpu::CudaSolver::available()) ui::console::driver_detected("Cuda", 1);
#endif
#ifdef MXBM_HAVE_OPENCL
    if (gpu::GpuSolver::available()) ui::console::driver_detected("OpenCL", 1);
#endif

    // Solver selection for opts.solver ("cuda" | "opencl" | "gpu" | "ref" |
    // "auto"; validated by cli::parse_args, so no other value reaches here).
    // Declared BEFORE `engine` below so `engine`, which only borrows a
    // `Solver&` into *solver, is torn down first -- locals are destroyed in
    // reverse declaration order, so the reference can never dangle.
    // A GPU backend is tried only when built in and available(); construction
    // can throw anyway on a memory-starved device, so it is guarded too.
    std::unique_ptr<miner::Solver> solver;
    // Set when a construction attempt was actually made and threw, so the
    // fallback below need not claim "no GPU available" when one was found.
    bool gpu_attempt_failed = false;
    // Shown in the stats table, the /summary API and the "Found a share" line.
    std::string worker_label = "GPU 0";
    // Left empty when no backend claims the device: then no block is printed.
    std::string dev_name, dev_driver;
    unsigned long long dev_mem = 0;

#ifdef MXBM_HAVE_CUDA
    // CUDA first when both are built: it measures ~1.16x the OpenCL path's rate on the
    // same card (docs/performance.md). --solver opencl forces the portable path.
    if ((opts.solver == "cuda" || opts.solver == "gpu" || opts.solver == "auto")
        && gpu::CudaSolver::available()) {
        try {
            auto cs = std::make_unique<gpu::CudaSolver>();
            worker_label = cs->device().name;   // the real name, not a bare "GPU 0"
            dev_name = cs->device().name;
            dev_mem = cs->device().global_mem;
            dev_driver = "Cuda";
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
            worker_label = gs->device().name;
            dev_name = gs->device().name;
            dev_mem = gs->device().global_mem;
            dev_driver = "OpenCL";
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

    // NVML before the device block, not after: it supplies the PCI address and
    // the driver version that block and the stats header report. Independent
    // of the chosen backend, so an OpenCL run on an NVIDIA card gets it too.
    const bool have_nvml = solver && gpu::nvml_init();
    if (have_nvml) stats.set_driver_version(gpu::nvml_driver_version());

    if (!dev_name.empty()) {
        // Vendor is claimed only when NVML answered, itself proof of an NVIDIA
        // card; other vendors get no Vendor line rather than a guessed one.
        ui::console::device_block(
            0, dev_name,
            have_nvml ? gpu::nvml_pci_address() : std::string(),
            have_nvml ? "NVIDIA Corporation" : std::string(),
            dev_driver, dev_mem,
            "Selected Algorithm: BeamHash III (" + dev_driver + ")");
    }

    if (have_nvml) {
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

    stratum::Client client;

    // A failed connect is not fatal: client.run()'s reconnect loop keeps
    // retrying below. login() runs unconditionally so the api_key credential
    // is stored even then, giving that loop something to re-login with.
    if (!benchmark_mode) {
        ui::console::connecting_to_pool();
        auto connect_t0 = std::chrono::steady_clock::now();
        bool up = client.connect(opts.pools[0].host, opts.pools[0].port, opts.pools[0].tls);
        auto connect_t1 = std::chrono::steady_clock::now();
        stats.record_connect(
            opts.pools[0].host + ":" + std::to_string(opts.pools[0].port),
            std::chrono::duration_cast<std::chrono::milliseconds>(connect_t1 - connect_t0).count());
        if (up) {
            ui::console::connected_to(opts.pools[0].host, client.peer_ip(),
                                      opts.pools[0].port, opts.pools[0].tls);
            if (opts.pools[0].tls) ui::console::tls_handshake_ok();
        } else {
            ui::console::error("Initial connection failed — will keep retrying");
        }
        client.login(opts.pools[0].user);   // stores api_key for reconnect re-login even if send fails
    }

    // --benchmark: solve synthetic jobs, report, exit. After solver selection
    // and telemetry so it runs the same code as mining, or its sol/s would not
    // be comparable to the mining figure.
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

        // Ctrl+C ends the run and still prints the summary. request_abort()
        // cuts short a solve in flight, bounding the wait by a round.
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
        ticker.start(stats, opts.shortstats, opts.longstats, opts.digits, opts.timeprint, opts.apiport);
        miner::BenchmarkResult r;
        try {
            r = miner::run_benchmark(*solver, stats, opts.benchmark_seconds, stop);
        } catch (const std::exception& e) {
            // A backend can construct and still fail on first launch, most
            // often out of memory: available() saw free VRAM that another
            // process has since taken. Report it rather than core-dumping.
            ticker.stop();
            std::signal(SIGINT, SIG_DFL);
            ui::console::error(std::string("Benchmark failed: ") + e.what());
            ui::console::info("If another process is using the GPU, stop it and retry: "
                              "a full BeamHash III search needs ~7.5 GiB free.");
            return 1;
        }
        ticker.stop();
        std::signal(SIGINT, SIG_DFL);

        // sol/s is the headline pools quote, but it is the product of solves
        // and solutions-per-solve, so print both; p5/p95 show run stability.
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

    // Null when no backend claimed a solver: jobs are monitored, not solved.
    std::unique_ptr<miner::Engine> engine;

    // Which pool's job the solver works on: a pass-through to Engine::on_job
    // with no fee configured, and what DevFee switches each round with one.
    miner::JobRouter router(
        [&engine](const stratum::Job& j, const std::string& prefix, miner::Origin origin) {
            if (engine) engine->on_job(j, prefix, origin);
        });

    // Declared before submit_fn below, which captures it to route Dev-origin
    // solutions, and destroyed after the router so its scheduler thread can
    // still dispatch while shutting down.
    std::unique_ptr<miner::DevFee> devfee;

    if (solver) {
        engine = std::make_unique<miner::Engine>(client, *solver);
        engine->submit_fn = [&client, &stats, &devfee, worker_label](
                const stratum::Solution& s, miner::Origin origin) {
            // Achieved difficulty for the "Found a share" line and the
            // best-share stat, from the same SHA-256 predicate Engine's
            // clears_difficulty() used. A decode failure skips the reporting
            // but still sends the share: submission is never gated on it.
            uint8_t soln[104];
            if (from_hex_strict(s.output, soln, sizeof soln)) {
                uint8_t hash[32];
                sha256(soln, sizeof soln, hash);
                double units = pow::achieved_units(hash);
                // Only the user's own shares get a console line: a dev-fee one
                // would read as theirs. The round lines report those instead.
                if (origin == miner::Origin::Main) {
                    // The current target is the bar this share had to clear:
                    // process_job() never submits a superseded job.
                    ui::console::share_found(worker_label.c_str(), units,
                                             stats.current_job_units(origin));
                }
                stats.record_share_found(units, origin);
            }
            stats.record_submit(s.id, origin);
            // Route by the Origin the JOB carried, never by which pool is
            // active now: a fee round can end mid-solve, and the other pool
            // would reject the solution as an unknown job id (miner/origin.h).
            if (origin == miner::Origin::Dev) {
                if (devfee) devfee->submit(s);
            } else {
                client.submit(s);
            }
        };
        engine->on_attempt = [&stats](uint32_t candidates) { stats.record_attempt(candidates); };
        engine->on_solver_error = [](const std::string& what, uint32_t n) {
            ui::console::error("Solver error (" + std::to_string(n) + " in a row): " + what);
            // The hint only on the first failure: once it is retrying every
            // 30s, repeating the same advice would bury the errors themselves.
            if (n == 1) {
                ui::console::info("Retrying with backoff - mining continues. If another process "
                                  "is using the GPU, stop it: a full BeamHash III search needs "
                                  "~7.5 GiB free.");
            }
        };
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
            stats.record_result(r.code, miner::Origin::Main);
            ui::console::share_result(r.code, r.description, stats.snapshot().last_latency_ms);
        }
    };

    client.on_job = [&](const stratum::Job& j) {
        stats.record_job(j.id, pow::to_display_units(j.difficulty), miner::Origin::Main);
        ui::console::job(j.id, j.difficulty, j.height);
        // Reading the prefix here is what keeps it race-free -- this runs on
        // client.run()'s thread, the same one that rewrites it (engine.h).
        router.offer(j, client.current_nonceprefix(), miner::Origin::Main);
    };

    client.on_disconnect = [&stats]() {
        ui::console::disconnected();
        stats.record_disconnect();
    };

    // Developer fee. Announced at startup either way, so "does this binary
    // charge me?" is answerable from the console. Mechanism in miner/devfee.h;
    // user-facing terms in docs/devfee.md.
    if (miner::devfee_configured() && engine) {
        miner::DevFeePool pool = miner::devfee_pool();
        miner::DevFeeSchedule sched = miner::devfee_schedule();

        // --dev-fee raises the rate, never lowers it. A lower request is
        // refused rather than clamped, so it cannot pass for having worked.
        if (opts.devfee_pct >= 0.0) {
            const double want = opts.devfee_pct / 100.0;
            if (want < sched.rate) {
                char msg[256];
                std::snprintf(msg, sizeof msg,
                    "--dev-fee %.4g%% is below this build's %.4g%% rate; the fee can be raised, "
                    "not lowered (see docs/devfee.md).",
                    opts.devfee_pct, sched.rate * 100.0);
                ui::console::error(msg);
                return 1;
            }
            sched.rate = want;
        }

        // The fee round logs in under the user's own worker name plus the rate,
        // so it is identifiable on the fee pool as theirs, raised rate and all.
        pool.user = miner::devfee_login(pool.user, opts.pools[0].user, sched.rate);

        stats.set_devfee_rate(sched.rate);
        ui::console::devfee_notice(sched.rate, sched.slice(), sched.cycle,
                                   pool.host + ":" + std::to_string(pool.port));
        devfee = std::make_unique<miner::DevFee>(router, stats, pool, sched);
        devfee->on_slice_begin = [](std::chrono::seconds d) { ui::console::devfee_start(d); };
        devfee->on_slice_end   = [](std::chrono::seconds d) { ui::console::devfee_end(d); };
        devfee->on_note        = [](const std::string& m) { ui::console::info(m); };
    } else if (engine) {
        ui::console::info("Dev fee: none - this build mines entirely for you");
        if (opts.devfee_pct > 0.0) {
            // With no developer address compiled in there is nowhere to send
            // the rounds, so say so rather than accepting the flag silently.
            ui::console::error("--dev-fee was given, but this build has no developer address "
                               "compiled in - no fee can be taken. See docs/devfee.md.");
        }
    }

    // The API starts BEFORE the ticker so the stats block reports the port it
    // actually bound rather than the one asked for: a port already in use
    // leaves the server down, and a header claiming "API port 8080" over a dead
    // listener sends you hunting a network fault that is not there.
    // bound_port() is then 0, which the header treats as "no API".
    api::HttpSummary http_api;
    if (opts.apiport) {
        if (!http_api.start(static_cast<uint16_t>(opts.apiport), stats, mxbm::version())) {
            ui::console::error("API server failed to start on port " + std::to_string(opts.apiport));
        }
        // Either way, mining continues below -- the API is a convenience,
        // never a mining precondition.
    }

    ui::Ticker ticker;
    ticker.start(stats, opts.shortstats, opts.longstats, opts.digits, opts.timeprint,
                 http_api.bound_port());

    if (engine) engine->start();
    // After the engine so the fee's first job never arrives before there is a
    // worker to mine it, and after the API/ticker so a fee round is reportable
    // the moment it can happen.
    if (devfee) devfee->start();

    client.run();   // blocks forever, reconnecting on drop; Ctrl+C exits
    return 0;
}
