// mxbm: the runnable miner binary. Parse argv -> merge a config file ->
// connect+login to the pool -> solve jobs and submit shares -> report stats
// via the ticker and the /summary API -> block forever on the stratum socket.
// Ctrl+C exits; stratum::Client::run() explains why there is no cleanup here.
#include <chrono>
#include <atomic>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
#include "miner/tune.h"
#include "miner/watchdog.h"
#include "version.h"
#include "gpu/nvml.h"
#include "gpu/overclock.h"
#include "gpu/rowbucket_geom.h"   // the restart notice asks the geometry question itself
#ifdef MXBM_HAVE_OPENCL
#include "gpu/gpu_solver.h"
#endif
#ifdef MXBM_HAVE_METAL
#include "gpu/metal_solver.h"
#include "gpu/metal_telemetry.h"
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
    // --tune drives the power limit itself, point by point; a fixed --pl under it
    // would be overwritten by the first sweep step and silently lied about. The
    // other OC knobs (--cclk, --coff, ...) are ALLOWED: tuning the curve of an
    // undervolted card is a legitimate ask, and they hold still across the sweep.
    if (opts.tune && !opts.power_limit.empty()) {
        std::fputs("--tune sets the power limit itself; drop --pl (other OC flags may stay)\n", stderr);
        return 1;
    }
    if (opts.tune && benchmark_mode) {
        std::fputs("--tune and --benchmark are both modes; pick one\n", stderr);
        return 1;
    }
    // Checked here rather than in parse_args: a config file may supply ALGO
    // and POOLS, so these fire only when no source supplied them. --list-devices
    // is exempt for the same reason --version is: it answers a question about
    // the machine, not about a mining run, and demanding a wallet address to
    // answer "which cards do I have" would be absurd.
    if (!opts.seen.algo && !benchmark_mode && !opts.list_devices) {
        std::fputs("unsupported algo (pass --algo BEAM-III, or set ALGO in your config)\n", stderr);
        return 1;
    }
    if (opts.pools.empty() && !benchmark_mode && !opts.list_devices && !opts.tune) {
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
#ifdef MXBM_HAVE_METAL
    if (gpu::MetalSolver::available()) ui::console::driver_detected("Metal", 1);
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
    // Which card(s) the user asked for. Enumerated in PCI order so an index
    // means the same physical card here, in --pl's per-GPU list and in NVML --
    // CUDA's own default ordering is "fastest first", which does not agree.
    std::vector<unsigned> selected_devices;
    unsigned device_count = 0;
#ifdef MXBM_HAVE_CUDA
    const std::vector<gpu::CudaSolver::DeviceInfo> cuda_devices = gpu::CudaSolver::enumerate();
    device_count = (unsigned)cuda_devices.size();
#endif
#ifdef MXBM_HAVE_METAL
    // Only when CUDA reported nothing: on a machine with both, CUDA's PCI-ordered list
    // is the one --devices and --pl are indexed against, and a second list underneath it
    // would make an index mean two things.
    const std::vector<gpu::MetalSolver::DeviceInfo> metal_devices =
        device_count == 0 ? gpu::MetalSolver::enumerate()
                          : std::vector<gpu::MetalSolver::DeviceInfo>{};
    if (device_count == 0) device_count = (unsigned)metal_devices.size();
#endif

    if (opts.list_devices) {
#ifdef MXBM_HAVE_CUDA
        if (cuda_devices.empty()) {
            ui::console::info("No CUDA devices detected.");
        } else {
            ui::console::info("Detected devices (indices are in PCI order, and mean the same "
                              "card in --devices and --pl):");
            for (size_t i = 0; i < cuda_devices.size(); ++i) {
                const auto& d = cuda_devices[i];
                char line[256];
                std::snprintf(line, sizeof line, "  %zu: %-34s %5llu MB  PCI %-6s  Cuda%s",
                              i, d.name.c_str(),
                              (unsigned long long)(d.global_mem / (1024ull * 1024ull)),
                              d.pci.c_str(),
                              d.viable ? "" : "  [too small for BeamHash III]");
                ui::console::info(line);
            }
        }
#endif
#ifdef MXBM_HAVE_METAL
        // Listed whenever CUDA had nothing to say. Saying "no devices can be listed" on
        // a machine that announced a Metal GPU two lines earlier is worse than silence.
        if (!metal_devices.empty()) {
            ui::console::info("Detected devices:");
            for (size_t i = 0; i < metal_devices.size(); ++i) {
                const auto& d = metal_devices[i];
                char line[256];
                std::snprintf(line, sizeof line, "  %zu: %-34s %5llu MB  %2u cores  Metal%s",
                              i, d.name.c_str(),
                              (unsigned long long)(d.global_mem / (1024ull * 1024ull)),
                              d.compute_units,
                              d.viable ? "" : "  [too small for BeamHash III]");
                ui::console::info(line);
            }
        }
#endif
        if (device_count == 0)
            ui::console::info("No GPU devices detected by any backend built into this binary.");
        return 0;
    }

    {
        std::string derr;
        if (!cli::resolve_devices(opts.devices, device_count, selected_devices, derr)) {
            ui::console::error("--devices \"" + opts.devices + "\": " + derr
                               + " (try --list-devices)");
            return 1;
        }
    }
    const int device_index = selected_devices.empty() ? 0 : (int)selected_devices.front();
#ifdef MXBM_HAVE_CUDA
    // Say which card was chosen whenever the choice was the user's, or whenever
    // there was more than one to choose from -- on a single-GPU box with no
    // --devices the line is noise.
    if ((opts.seen.devices || cuda_devices.size() > 1)
        && (size_t)device_index < cuda_devices.size()) {
        ui::console::info("Mining on device " + std::to_string(device_index) + ": "
                          + cuda_devices[(size_t)device_index].name
                          + " (PCI " + cuda_devices[(size_t)device_index].pci + ")");
    }
#endif

    // The first selected device's solver. Everything that is single-device by
    // nature -- the benchmark, the device block, the NVML handle -- uses this
    // one; mining adds the rest below.
    std::unique_ptr<miner::Solver> solver;
    // Devices two and up, when --devices named several and the backend can
    // drive them. Each gets its own Engine and its own nonce lane.
    std::vector<std::unique_ptr<miner::Solver>> extra_solvers;
    std::vector<std::string> extra_labels;
    // Set when a construction attempt was actually made and threw, so the
    // fallback below need not claim "no GPU available" when one was found.
    bool gpu_attempt_failed = false;
    // Shown in the stats table, the /summary API and the "Found a share" line.
    std::string worker_label = "GPU 0";
    // Left empty when no backend claims the device: then no block is printed.
    std::string dev_name, dev_driver;
    unsigned long long dev_mem = 0;

#ifdef MXBM_HAVE_CUDA
    // enumerate() returned PCI order, so the user's index is a position in that
    // list; the CUDA index to construct on is the one recorded in the entry.
    const int cuda_index = (device_index >= 0 && (size_t)device_index < cuda_devices.size())
                         ? cuda_devices[(size_t)device_index].index : device_index;
#endif
#ifdef MXBM_HAVE_METAL
    const int metal_index = device_index >= 0 ? device_index : 0;
#endif
#ifdef MXBM_HAVE_OPENCL
    // The OpenCL index is the user's selection used directly: OpenCL enumerates
    // per platform and exposes no PCI address to sort on, so unlike the CUDA
    // path there is no translation to make. On a single-vendor rig the two
    // orderings agree; --list-devices is what to check when they might not.
    const unsigned cl_index = device_index >= 0 ? (unsigned)device_index : 0u;
#endif

    // Which backend WILL be attempted -- decided before construction, mirroring the
    // guards below: the power limit must be applied and observed before any buffer is
    // sized. NVML still opens only when a GPU backend will actually run.
    bool cuda_will_run = false, metal_will_run = false, opencl_will_run = false;
#ifdef MXBM_HAVE_CUDA
    cuda_will_run = (opts.solver == "cuda" || opts.solver == "gpu" || opts.solver == "auto")
                    && gpu::CudaSolver::available(cuda_index);
#endif
#ifdef MXBM_HAVE_METAL
    metal_will_run = !cuda_will_run
                  && (opts.solver == "metal" || opts.solver == "gpu" || opts.solver == "auto")
                  && gpu::MetalSolver::available(metal_index);
#endif
#ifdef MXBM_HAVE_OPENCL
    opencl_will_run = !cuda_will_run && !metal_will_run
                   && (opts.solver == "opencl" || opts.solver == "gpu" || opts.solver == "auto")
                   && gpu::GpuSolver::available(cl_index);
#endif
    const bool gpu_will_run = cuda_will_run || metal_will_run || opencl_will_run;
    const bool have_nvml = gpu_will_run && gpu::nvml_init();

    // Overclock settings BEFORE the solver exists, not after: geometry selection reads
    // the board power limit, so --pl has to land first -- apply, then observe, then
    // size buffers. A card that cannot be set still mines, at stock: a miner that
    // failed to apply OC is degraded, not wrong (docs/overclocking.md).
    {
        gpu::OcRequest ocreq;
        ocreq.pl   = opts.power_limit;
        ocreq.cclk = opts.core_clock;
        ocreq.mclk = opts.mem_clock;
        ocreq.coff = opts.core_offset;
        ocreq.moff = opts.mem_offset;
        ocreq.fan  = opts.fan;

        // --pl auto: the value a --tune run stored for THIS card, resolved into a
        // plain wattage here -- after the card is known, before the numeric
        // validation below would choke on the word.
        if (ocreq.pl == "auto") {
            std::string key;
#ifdef MXBM_HAVE_CUDA
            if ((size_t)device_index < cuda_devices.size())
                key = cuda_devices[(size_t)device_index].name + "@"
                    + (have_nvml ? gpu::nvml_pci_address((unsigned)device_index)
                                 : std::string());
#endif
            std::string date;
            const unsigned w = key.empty() ? 0u : miner::tune_stored_knee(key, date);
            if (w) {
                ocreq.pl = std::to_string(w);
                ui::console::info("--pl auto: " + std::to_string(w)
                                  + " W, tuned for this card on " + date
                                  + " (re-run --tune after driver or cooling changes)");
                // Recommended, never auto-applied: locking a memory clock the user
                // did not ask for is a hardware setting nobody requested.
                unsigned rmhz = 0, rbelow = 0;
                if (opts.mem_clock.empty() && miner::tune_stored_rung(key, rmhz, rbelow)
                    && w < rbelow) {
                    ui::console::info("--pl auto: at this cap the tune also measured "
                                      "the " + std::to_string(rmhz)
                                      + " MHz memory rung faster - consider adding "
                                        "--mclk " + std::to_string(rmhz));
                }
            } else {
                ocreq.pl.clear();
                ui::console::error("--pl auto: no stored tune for this card in "
                                   + miner::tune_store_path()
                                   + " - run `sudo mxbm --tune` once; continuing at "
                                     "the card's current limit");
            }
        }
        const bool any = !(ocreq.pl.empty() && ocreq.cclk.empty() && ocreq.mclk.empty()
                           && ocreq.coff.empty() && ocreq.moff.empty() && ocreq.fan.empty());

        // Malformed input is an error wherever it came from. The CLI rejects it
        // at parse time; this catches the config-file path, which stores
        // strings without a syntax check.
        const struct { const std::string& spec; const char* flag; bool sign; } checks[] = {
            {ocreq.pl, "--pl", false}, {ocreq.cclk, "--cclk", false},
            {ocreq.mclk, "--mclk", false}, {ocreq.coff, "--coff", true},
            {ocreq.moff, "--moff", true}, {ocreq.fan, "--fan", true},
        };
        for (const auto& c : checks) {
            if (c.spec.empty()) continue;
            long v = 0; bool found = false; std::string perr;
            if (!gpu::oc_parse_list(c.spec, 0, v, found, perr, c.sign)) {
                ui::console::error(std::string("Invalid ") + c.flag + " \"" + c.spec + "\": " + perr);
                return 1;
            }
        }

        if (any) {
            gpu::oc_set_restore_enabled(!opts.no_oc_reset);
            if (!have_nvml) {
                // Three different causes, and saying the wrong one sends the user
                // hunting for a driver they already have. NVML is only opened
                // when a GPU backend will run, because changing the power limit
                // of a card this process is not going to use is a side effect
                // nobody asked for.
                ui::console::error(
                    metal_will_run
                    ? std::string("Overclock settings are not supported on Apple Silicon: macOS "
                                  "exposes no power-limit, clock or fan control. Continuing at "
                                  "the system's own settings")
                    : gpu_will_run
                    ? std::string("Overclock settings need NVML (an NVIDIA driver), which is not "
                                  "available here; continuing at the card's current settings")
                    : std::string("Overclock settings were not applied: no GPU solver is in use, "
                                  "so there is no card for them to apply to"));
            } else {
                // the reference miner prints this banner before its own OC block and rig
                // operators grep for it, so the line stays even when every knob
                // then fails -- "it tried and could not" is the useful log, and
                // silence is the one outcome that is not.
                ui::console::info("Applying overclock settings...");
                for (const gpu::OcResult& r : gpu::oc_apply(ocreq)) {
                    if (r.status == gpu::OcStatus::Applied || r.status == gpu::OcStatus::Clamped)
                        ui::console::info(r.message);
                    else
                        ui::console::error(r.message);
                }
                if (opts.no_oc_reset)
                    ui::console::info("--no-oc-reset: applied settings will be left on the card at exit");
            }
        } else if (opts.no_oc_reset) {
            ui::console::info("--no-oc-reset noted, but no overclock setting was given - "
                              "nothing to reset");
        }
    }

    // The limit geometry selection turns on: read back AFTER --pl landed, so a cap
    // MXBM applied and one set outside it (nvidia-smi before launch) look the same.
    // Zero means "no limit known" and selects nothing.
    unsigned startup_pl_w = 0;
    if (have_nvml) {
        const gpu::PowerLimit board_pl = gpu::nvml_power_limit();
        if (board_pl.valid) startup_pl_w = board_pl.current_w;
    }

#ifdef MXBM_HAVE_CUDA
    // CUDA first when both are built: it measures ~1.16x the OpenCL path's rate on the
    // same card (docs/performance.md). --solver opencl forces the portable path.
    if (cuda_will_run) {
        try {
            auto cs = std::make_unique<gpu::CudaSolver>(cuda_index, startup_pl_w);
            worker_label = cs->device().name;   // the real name, not a bare "GPU 0"
            dev_name = cs->device().name;
            dev_mem = cs->device().global_mem;
            dev_driver = "Cuda";
            solver = std::move(cs);
        } catch (const std::exception& e) {
            ui::console::error(std::string("CUDA solver initialization failed: ") + e.what());
            gpu_attempt_failed = true;
        }
        // The remaining selected devices. A card that fails to initialise is
        // reported and skipped rather than taking the rig down with it: on a
        // multi-GPU box the whole point is that the others keep mining.
        for (size_t k = 1; solver && k < selected_devices.size(); ++k) {
            const size_t pos = selected_devices[k];
            const int idx = pos < cuda_devices.size() ? cuda_devices[pos].index : (int)pos;
            try {
                // Each card gets ITS observed limit -- NVML orders by bus id, the same
                // order `pos` indexes -- because a mixed rig can cap the cards apart.
                const gpu::PowerLimit epl = have_nvml ? gpu::nvml_power_limit_at((unsigned)pos)
                                                      : gpu::PowerLimit{};
                auto cs = std::make_unique<gpu::CudaSolver>(idx, epl.valid ? epl.current_w : 0u);
                extra_labels.push_back(cs->device().name);
                extra_solvers.push_back(std::move(cs));
            } catch (const std::exception& e) {
                ui::console::error("Device " + std::to_string(pos) + " could not be initialised ("
                                   + e.what() + ") - continuing without it");
            }
        }
    }
#endif
#ifdef MXBM_HAVE_METAL
    // Metal before OpenCL on Apple Silicon, and this is measured rather than assumed.
    // Apple's OpenCL is a deprecated 1.2-era shim that cannot build the fused
    // row-bucket kernels AT ALL (clCreateKernel -> CL_INVALID_KERNEL), so it is
    // confined to the sort path -- the slowest of the three collision finders. On an
    // M3 Max that is ~505 ms/solve against Metal's ~117 ms, a 4.3x difference.
    // --solver opencl still forces the portable path.
    if (!solver && (opts.solver == "metal" || opts.solver == "gpu" || opts.solver == "auto")
        && gpu::MetalSolver::available(metal_index)) {
        try {
            auto ms = std::make_unique<gpu::MetalSolver>(metal_index);
            worker_label = ms->device().name;
            dev_name = ms->device().name;
            dev_mem = ms->device().global_mem;
            dev_driver = "Metal";
            solver = std::move(ms);
        } catch (const std::exception& e) {
            ui::console::error(std::string("Metal solver initialization failed: ") + e.what());
            gpu_attempt_failed = true;
        }
    }
#endif
#ifdef MXBM_HAVE_OPENCL
    if (!solver && (opts.solver == "opencl" || opts.solver == "gpu" || opts.solver == "auto")
        && gpu::GpuSolver::available(cl_index)) {
        try {
            auto gs = std::make_unique<gpu::GpuSolver>(cl_index);
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

    // NVML came up BEFORE construction -- the power limit had to be applied and
    // observed before geometry was chosen. It also supplies the PCI address and the
    // driver version the device block and stats header report, independent of the
    // chosen backend, so an OpenCL run on an NVIDIA card gets them too.
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

#ifdef MXBM_HAVE_METAL
    // Apple Silicon telemetry, when NVML is not the source. Much thinner than NVML by
    // necessity: utilization is the only figure the public API exposes (see
    // gpu/metal_telemetry.h), so the power/clock/temp/fan columns stay blank rather
    // than being filled with a scraped guess.
    const bool have_metal_telem = !have_nvml && dev_driver == "Metal"
                               && gpu::metal_telemetry_init();
    if (have_metal_telem) {
        stats.set_telemetry_source([](unsigned row, miner::Stats::Device& d) {
            const gpu::Telemetry t = gpu::metal_sample(row);
            d.has_util = t.have_util;  d.util_pct = t.util_pct;
        });
    }
#endif
    if (have_nvml && solver) {
        // Each row samples its OWN card. The index is the position in the
        // PCI-sorted list --devices selected from, and NVML enumerates by bus
        // id too, so the two agree without a translation table.
        std::vector<unsigned> nvml_index;
        for (unsigned d : selected_devices) nvml_index.push_back(d);
        if (nvml_index.empty()) nvml_index.push_back(0);
        // The restart notice: geometry is chosen once at startup (a switch is a
        // multi-GiB realloc), so a cap that later crosses the threshold gets one line,
        // not a re-select -- and only when the selection would actually DIFFER, never
        // under a hand-forced geometry. CUDA only: OpenCL takes no cap hint.
        const bool watch_pl = dev_driver == "Cuda"
                           && !std::getenv("MXBM_BB") && !std::getenv("MXBM_SM");
        auto pl_noticed = std::make_shared<std::atomic<bool>>(false);
        const unsigned long long mem0 = dev_mem;
        const unsigned pl0 = startup_pl_w;
        stats.set_telemetry_source([nvml_index, watch_pl, pl_noticed, mem0, pl0]
                                   (unsigned row, miner::Stats::Device& d) {
            const unsigned idx = row < nvml_index.size() ? nvml_index[row] : row;
            const gpu::Telemetry t = gpu::nvml_sample(idx);
            d.has_power = t.have_power;         d.power_w       = t.power_w;
            d.has_sm_clock = t.have_sm;         d.sm_clock_mhz  = t.sm_clock_mhz;
            d.has_mem_clock = t.have_mem;       d.mem_clock_mhz = t.mem_clock_mhz;
            d.has_temp = t.have_temp;           d.temp_c        = t.temp_c;
            d.has_fan = t.have_fan;             d.fan_pct       = t.fan_pct;
            d.has_util = t.have_util;           d.util_pct      = t.util_pct;
            if (row == 0 && watch_pl && !pl_noticed->load(std::memory_order_relaxed)) {
                const gpu::PowerLimit now = gpu::nvml_power_limit_at(nvml_index[0]);
                if (now.valid && now.current_w != pl0) {
                    const gpu::RbGeometry was = gpu::rb_geometry_for(
                        gpu::kRbCapacity, 0, mem0, /*allow_quad=*/true, pl0);
                    const gpu::RbGeometry would = gpu::rb_geometry_for(
                        gpu::kRbCapacity, 0, mem0, /*allow_quad=*/true, now.current_w);
                    if (was.bb != would.bb) {
                        pl_noticed->store(true, std::memory_order_relaxed);
                        ui::console::info(
                            "Board power limit is now " + std::to_string(now.current_w)
                            + " W (" + (pl0 ? std::to_string(pl0) + " W at startup"
                                            : std::string("not known at startup"))
                            + "). Solver geometry is chosen once, at startup, and stays "
                              "for this run - restart MXBM to re-select for the new limit.");
                    }
                }
            }
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
        } else if (opts.solver == "metal") {
            ui::console::error("--solver metal requested but no usable Metal device is available "
                               "(or this build has no Metal backend) - monitoring jobs only (no solving)");
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
    if (!benchmark_mode && !opts.tune) {
        ui::console::connecting_to_pool();
        auto connect_t0 = std::chrono::steady_clock::now();
        // Failover list, in the order the pools were given. Each carries its own
        // credentials: they are different accounts, and re-logging into pool B
        // with pool A's wallet would mine for the wrong address.
        if (opts.pools.size() > 1) {
            std::vector<stratum::Client::Endpoint> eps;
            for (const auto& p : opts.pools)
                eps.push_back(stratum::Client::Endpoint{p.host, p.port, p.tls, p.user});
            client.set_failover_pools(eps);
            client.on_failover = [](const std::string& from, const std::string& to) {
                ui::console::error("Pool " + from + " is not answering - failing over to " + to);
            };
            ui::console::info("Failover pools: " + std::to_string(opts.pools.size())
                              + " configured; moving on after "
                              + std::to_string(stratum::Client::kRedialsBeforeFailover)
                              + " consecutive failed redials");
        }
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

    // --tune: sweep power caps through the SAME solver mining uses, recommend a
    // --pl value, store it for --pl auto, exit. Placed exactly like --benchmark
    // and for the same reason: a curve measured in any other loop is a curve of
    // a different program (the 2026-07-31 lesson, docs/performance-research.md).
    if (opts.tune) {
        if (!solver) {
            ui::console::error("--tune needs a working solver backend; none is available");
            return 1;
        }
        if (!have_nvml) {
            ui::console::error("--tune needs NVML (an NVIDIA driver) to drive the power "
                               "limit; none is available here");
            return 1;
        }
        miner::TuneConfig tcfg;
        tcfg.seconds_per_point = opts.tune_seconds;
        tcfg.knee_marginal = opts.tune_knee;
        // An explicit cap list is a chosen grid: measure exactly those points, no
        // second pass. The default grid gets the knee-refining pass.
        tcfg.refine = opts.tune_caps.empty();
        // The rung pass runs in default mode only, and never over a user's own
        // --mclk: a chosen memory clock stands, exactly like a chosen grid.
        tcfg.rung_pass = opts.tune_caps.empty() && opts.mem_clock.empty();
        if (!opts.tune_caps.empty()) {
            // Parse-time checked to be digits and commas; split it here.
            size_t pos = 0;
            while (pos <= opts.tune_caps.size()) {
                const size_t comma = opts.tune_caps.find(',', pos);
                tcfg.caps.push_back((unsigned)std::strtoul(
                    opts.tune_caps.substr(pos, comma == std::string::npos
                                                   ? std::string::npos : comma - pos).c_str(),
                    nullptr, 10));
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
        }
        // Same Ctrl+C contract as the benchmark: stop the sweep, cut the solve in
        // flight -- and run_tune itself restores the power limit on that path.
        static std::atomic<bool>* s_tstop = nullptr;
        static miner::Solver* s_tsolver = nullptr;
        std::atomic<bool> stop{false};
        s_tstop = &stop;
        s_tsolver = solver.get();
        std::signal(SIGINT, [](int) {
            if (s_tstop) s_tstop->store(true, std::memory_order_relaxed);
            if (s_tsolver) s_tsolver->request_abort();
        });
        const std::string key = dev_name + "@"
                              + gpu::nvml_pci_address((unsigned)device_index);
        const int rc = miner::run_tune(*solver, stats, key, tcfg, stop);
        // Same epilogue as the benchmark: if some OTHER OC knob (--cclk with --tune
        // is legitimate) still has a restore pending, SIGINT must keep triggering it.
        if (gpu::oc_has_pending_restore()) gpu::oc_install_restore_hooks();
        else                               std::signal(SIGINT, SIG_DFL);
        return rc;
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
        // The card's own energy counter, bracketing the run: exact joules, no
        // sampling. e_ok stays false on non-NVIDIA/pre-Volta and the line below
        // is simply not printed.
        unsigned long long e0 = 0, e1 = 0;
        const bool e_ok = have_nvml && gpu::nvml_total_energy_mj(e0);
        miner::BenchmarkResult r;
        try {
            r = miner::run_benchmark(*solver, stats, opts.benchmark_seconds, stop);
        } catch (const std::exception& e) {
            // A backend can construct and still fail on first launch, most
            // often out of memory: available() saw free VRAM that another
            // process has since taken. Report it rather than core-dumping.
            ticker.stop();
            // Hand SIGINT back to whoever should own it: the OC restore hook while a
            // power limit is still on the card, else the default disposition. Without
            // this the window between here and process exit would drop a Ctrl+C and
            // leave the card capped.
            if (gpu::oc_has_pending_restore()) gpu::oc_install_restore_hooks();
            else                               std::signal(SIGINT, SIG_DFL);
            ui::console::error(std::string("Benchmark failed: ") + e.what());
            ui::console::info("If another process is using the GPU, stop it and retry: "
                              "a full BeamHash III search needs ~5.7 GiB free at the smallest geometry.");
            return 1;
        }
        ticker.stop();
        // Hand SIGINT back to whoever should own it: the OC restore hook while a
        // power limit is still on the card, else the default disposition. Without
        // this the window between here and process exit would drop a Ctrl+C and
        // leave the card capped.
        if (gpu::oc_has_pending_restore()) gpu::oc_install_restore_hooks();
        else                               std::signal(SIGINT, SIG_DFL);

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
        // Efficiency from the energy counter, when the card has one: exact
        // joules over the whole window, so J/solution carries no sampling error
        // -- the number the power-sweep docs integrate 5 Hz samples to estimate.
        if (e_ok && gpu::nvml_total_energy_mj(e1) && e1 > e0
            && r.elapsed_s > 0.0 && r.solutions > 0) {
            const double joules = (double)(e1 - e0) / 1000.0;
            std::snprintf(line, sizeof line,
                "  %.0f J total   (%.1f W mean, %.2f J/solution - from the card's energy counter)",
                joules, joules / r.elapsed_s, joules / (double)r.solutions);
            ui::console::info(line);
        }
        if (r.solves < 100) {
            ui::console::info("  note: fewer than 100 solves - too few to quote a margin; "
                              "use --benchmark-seconds to run longer");
        }
        return 0;
    }

    // One Engine per device, each with its own solver and its own nonce lane.
    // Empty when no backend claimed a solver: jobs are monitored, not solved.
    // engines[0] drives `solver`; the rest drive extra_solvers in order.
    std::vector<std::unique_ptr<miner::Engine>> engines;

    // Which pool's job the solvers work on: a pass-through to Engine::on_job
    // with no fee configured, and what DevFee switches each round with one.
    // EVERY engine gets every job -- they divide the nonce space, not the jobs,
    // so each card searches the same block header from a disjoint set of
    // nonces (miner/engine.h, the lane parameter).
    miner::JobRouter router(
        [&engines](const stratum::Job& j, const std::string& prefix, miner::Origin origin) {
            for (auto& e : engines) e->on_job(j, prefix, origin);
        });

    // Declared before submit_fn below, which captures it to route Dev-origin
    // solutions, and destroyed after the router so its scheduler thread can
    // still dispatch while shutting down.
    std::unique_ptr<miner::DevFee> devfee;

    if (solver) {
        const uint64_t lanes = 1 + extra_solvers.size();
        engines.push_back(std::make_unique<miner::Engine>(client, *solver, 0, lanes));
        for (size_t k = 0; k < extra_solvers.size(); ++k)
            engines.push_back(std::make_unique<miner::Engine>(client, *extra_solvers[k],
                                                              k + 1, lanes));
        if (lanes > 1) {
            std::string names = worker_label;
            for (const auto& l : extra_labels) names += ", " + l;
            ui::console::info("Mining on " + std::to_string(lanes) + " devices: " + names);
        }

        // Labels in the same order as the engines, so row N of the statistics
        // table is engine N is device N.
        {
            std::vector<std::string> labels{worker_label};
            for (const auto& l : extra_labels) labels.push_back(l);
            stats.set_device_labels(labels);
        }

        // Built per engine rather than shared, so each carries its own device
        // index into Stats. Sharing one closure was what made every card report
        // as device 0.
        for (size_t k = 0; k < engines.size(); ++k) {
            const unsigned dev = (unsigned)k;
            const std::string label = (k == 0) ? worker_label : extra_labels[k - 1];
            engines[k]->submit_fn = [&client, &stats, &devfee, label, dev](
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
                    ui::console::share_found(label.c_str(), units,
                                             stats.current_job_units(origin));
                }
                stats.record_share_found(units, origin, dev);
            }
            stats.record_submit(s.id, origin, dev);
            // Route by the Origin the JOB carried, never by which pool is
            // active now: a fee round can end mid-solve, and the other pool
            // would reject the solution as an unknown job id (miner/origin.h).
            if (origin == miner::Origin::Dev) {
                if (devfee) devfee->submit(s);
            } else {
                client.submit(s);
            }
            };
            engines[k]->on_attempt = [&stats, dev](uint32_t candidates) {
                stats.record_attempt(candidates, dev);
            };
            engines[k]->on_solver_error = [label](const std::string& what, uint32_t n) {
            ui::console::error(label + " solver error (" + std::to_string(n)
                               + " in a row): " + what);
            // The hint only on the first failure: once it is retrying every
            // 30s, repeating the same advice would bury the errors themselves.
            if (n == 1) {
                ui::console::info("Retrying with backoff - mining continues. If another process "
                                  "is using the GPU, stop it: a full BeamHash III search needs "
                                  "~5.7 GiB free at the smallest geometry.");
                }
            };
        }
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
    if (miner::devfee_configured() && !engines.empty()) {
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
    } else if (!engines.empty()) {
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

    // Each engine spawns its own worker thread; they share one Stats and one
    // Client, both of which are mutex-guarded.
    for (auto& e : engines) e->start();

    // Watchdog. Started after the engines, so a card that takes half a minute
    // to allocate is not accused of hanging before it has begun -- and it takes
    // its first sighting from wherever the counters are then.
    miner::Watchdog watchdog;
    if (opts.watchdog_requested && !engines.empty()) {
        miner::WatchdogAction action = miner::WatchdogAction::Exit;
        if (!miner::parse_watchdog_action(opts.watchdog_action, action)) {
            ui::console::error("Unknown --watchdog action \"" + opts.watchdog_action
                               + "\" - expected off, exit or script");
            return 1;
        }
        if (action == miner::WatchdogAction::Script && opts.watchdog_script.empty()) {
            ui::console::error("--watchdog script needs --watchdogscript PATH");
            return 1;
        }
        watchdog.counts = [&stats] {
            std::vector<uint64_t> c;
            for (const auto& d : stats.snapshot().devices) c.push_back(d.attempts);
            return c;
        };
        // "Mining is expected" means a job has arrived. Before that, and while
        // the pool is down, every device is legitimately idle.
        watchdog.mining = [&stats] { return !stats.snapshot().last_job_id.empty(); };
        const std::string script = opts.watchdog_script;
        watchdog.on_hung = [action, script](unsigned dev) {
            const std::string who = "GPU " + std::to_string(dev);
            ui::console::error(who + " has stopped completing solves - it looks hung.");
            switch (action) {
            case miner::WatchdogAction::Exit:
                ui::console::error("Watchdog: exiting with code 42 so a supervisor can "
                                   "restart the miner. A wedged GPU context cannot be "
                                   "rebuilt from inside this process.");
                std::exit(42);
            case miner::WatchdogAction::Script:
                ui::console::info("Watchdog: running " + script);
                // The device index is the argument, so one script can serve a
                // rig and know which card it is being called about.
                if (std::system((script + " " + std::to_string(dev)).c_str()) != 0)
                    ui::console::error("Watchdog script exited non-zero");
                break;
            case miner::WatchdogAction::Off:
                ui::console::info("Watchdog: action is 'off' - mining continues on the "
                                  "remaining devices.");
                break;
            }
        };
        watchdog.start();
        ui::console::info("Watchdog: on, action '" + std::string(miner::watchdog_action_name(action))
                          + "'");
    }
    // After the engine so the fee's first job never arrives before there is a
    // worker to mine it, and after the API/ticker so a fee round is reportable
    // the moment it can happen.
    if (devfee) devfee->start();

    client.run();   // blocks forever, reconnecting on drop; Ctrl+C exits
    return 0;
}
