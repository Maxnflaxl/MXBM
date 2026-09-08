// mxbm: the runnable miner binary. Parse argv -> merge a config file ->
// connect+login to the pool -> solve jobs and submit shares -> report stats
// via the ticker and the /summary API -> block forever on the stratum socket.
// Ctrl+C exits; stratum::Client::run() explains why there is no cleanup here.
#include <chrono>
#include <atomic>
#include <csignal>
#include <fstream>
#include <functional>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <memory>
#include <string>

#include "api/http_summary.h"
#include "cli/options.h"
#include "config/config.h"
#include "miner/stats.h"
#include "ui/console.h"
#include "ui/keys.h"
#include "ui/ticker.h"
#include "pow/difficulty.h"
#include "sha256/sha256.h"   // vendored: void sha256(const uint8_t* d, size_t n, uint8_t out[32]);
#include "stratum/client.h"
#include "stratum/messages.h"
#include "miner/engine.h"
#include "miner/devfee.h"
#include "miner/benchmark.h"
#include "miner/report.h"
#include "miner/tune.h"
#include "miner/watchdog.h"
#include "miner/thermal.h"
#include "version.h"
#include "gpu/nvml.h"
#include "gpu/telemetry_window.h"
#include "gpu/overclock.h"
#include "gpu/budget.h"           // the VRAM reserve the pipeline is sized against
#include "gpu/device_join.h"      // which backend drives which physical card
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

// Ctrl+C for the modes that measure instead of mine (--benchmark, --tune,
// --report): set the stop flag and cut the solve in flight, so the wait is bounded
// by a round rather than by a whole solve.
//
// Restoring the handler is the destructor's job. SIGINT goes back to whoever should
// own it -- the overclock restore hook while a setting is still on the card, else
// the default disposition -- and doing it here is what keeps a mode's failure path
// identical to its success path.
std::atomic<bool>*           g_stop = nullptr;
std::atomic<miner::Solver*>* g_live = nullptr;

struct MeasureSignals {
    MeasureSignals(std::atomic<bool>& stop, std::atomic<miner::Solver*>& live) {
        g_stop = &stop;
        g_live = &live;
        std::signal(SIGINT, [](int) {
            if (g_stop) g_stop->store(true, std::memory_order_relaxed);
            if (g_live)
                if (miner::Solver* s = g_live->load(std::memory_order_relaxed))
                    s->request_abort();
        });
    }
    ~MeasureSignals() {
        if (gpu::oc_has_pending_restore()) gpu::oc_install_restore_hooks();
        else                               std::signal(SIGINT, SIG_DFL);
    }
};

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

    if (opts.list_algos || opts.list_coins) {
        const double fee_pct = miner::devfee_schedule().rate * 100.0;
        if (opts.list_algos) {
            std::fputs("Supported algorithms:\n\n"
                       "Parameter   Algorithm      Fee %\n", stdout);
            std::fprintf(stdout, "BEAM-III    BeamHash III   %.1f\n", fee_pct);
        }
        if (opts.list_coins) {
            if (opts.list_algos) std::fputc('\n', stdout);
            std::fputs("Supported coins:\n\n"
                       "Parameter   Coin   Algorithm      Fee %\n", stdout);
            std::fprintf(stdout, "BEAM        Beam   BeamHash III   %.1f\n", fee_pct);
        }
        return 0;
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

    // Must precede every solver construction: the reserve is what the pipeline
    // is sized against.
    if (opts.keepfree_mb >= 0)
        gpu::set_keepfree_bytes((uint64_t)opts.keepfree_mb * 1024ull * 1024ull);

    // Refused rather than quietly read from another sensor: protecting a card by
    // a temperature the operator did not choose is worse than not protecting it.
    // Only the edge sensor is readable here -- NVML's memory-temperature field and
    // the whole T.Limit family answer "Not Supported" on consumer boards
    // (docs/performance.md), and there is no junction reader at all.
    if ((opts.tstop != 0 || opts.tstart != 0) && opts.tmode != "edge") {
        std::fputs(("--tmode " + opts.tmode + " is not available on this card: the driver "
                    "reports no " + opts.tmode + " sensor. Use --tmode edge.\n").c_str(), stderr);
        return 1;
    }

    const bool benchmark_mode = !opts.benchmark.empty();
    // Every mode that measures instead of mining: no pool, one card, exits when
    // done. Grouped so the checks below and the device work further on say it once.
    const bool measure_mode = benchmark_mode || opts.tune || opts.report;
    // --tune drives the power limit itself, point by point; a fixed --pl under it
    // would be overwritten by the first sweep step and silently lied about. The
    // other OC knobs (--cclk, --coff, ...) are ALLOWED: tuning the curve of an
    // undervolted card is a legitimate ask, and they hold still across the sweep.
    if (opts.report && (benchmark_mode || opts.tune)) {
        std::fputs("--report already runs both; drop --benchmark/--tune\n", stderr);
        return 1;
    }
    // The --tune knobs configure --tune. --report sweeps at the defaults on purpose:
    // its output is a figure other people read, and a hand-picked grid produces a
    // verdict that looks like a full sweep's without the refining passes behind it.
    if (opts.report && (opts.seen.tune_caps || opts.seen.tune_seconds)) {
        std::fputs("--tune-caps and --tune-seconds configure --tune; --report always "
                   "sweeps at the defaults. Run --tune directly to choose a grid\n", stderr);
        return 1;
    }
    // --report may sweep, and a sweep sets the limit point by point. A fixed --pl
    // under it would be overwritten by the first step and silently lied about.
    if (opts.report && !opts.power_limit.empty()) {
        std::fputs("--report measures the power curve itself; drop --pl "
                   "(other OC flags may stay)\n", stderr);
        return 1;
    }
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
    if (!opts.seen.algo && !measure_mode && !opts.list_devices) {
        std::fputs("unsupported algo (pass --algo BEAM-III, or set ALGO in your config)\n", stderr);
        return 1;
    }
    if (opts.pools.empty() && !measure_mode && !opts.list_devices) {
        std::fputs("missing --pool (or config with POOLS)\n", stderr);
        return 1;
    }

#ifndef _WIN32
    // OpenSSL's internal write() has no MSG_NOSIGNAL and SO_NOSIGPIPE is
    // Darwin-only: on Linux a pool dropping mid-TLS-write would SIGPIPE-kill
    // the process instead of surfacing a write error to the reconnect loop.
    // (Windows has no SIGPIPE; a dead socket just returns a write error.)
    std::signal(SIGPIPE, SIG_IGN);
#endif

    // Colour only where it renders; also switches the Windows console into the
    // mode that needs.
    ui::console::init(opts.nocolor || !ui::console::enable_terminal_color());
    ui::console::set_verbosity(opts.silence, opts.compactaccept);

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

    // What this BUILD can drive, which is the question a bug report keeps needing and
    // no runtime probe can answer: a machine with no CUDA card and a binary with no
    // CUDA support look identical from the device table alone.
    {
        std::string built, missing;
        // ASCII throughout: console::banner pads its rows by byte count, and it is
        // the one block that has to render identically in any terminal and any log.
        auto add = [](std::string& to, const std::string& what) {
            if (!to.empty()) to += ", ";
            to += what;
        };
#ifdef MXBM_HAVE_CUDA
        add(built, std::string("CUDA ") + MXBM_CUDA_VERSION);
#else
        add(missing, "CUDA");
#endif
#ifdef MXBM_OPENCL_VERSION
        add(built, std::string("OpenCL ") + MXBM_OPENCL_VERSION);
#elif defined(MXBM_HAVE_OPENCL)
        add(built, "OpenCL");   // a find module that reports no version
#else
        add(missing, "OpenCL");
#endif
#ifdef MXBM_HAVE_METAL
        add(built, "Metal");
#else
        add(missing, "Metal");
#endif
        if (built.empty()) built = "none - CPU reference only";
        if (!missing.empty()) built += "   (not built: " + missing + ")";
        ui::console::banner(built, miner::devfee_schedule().rate);
    }

    // Flags that are accepted but not yet implemented get a console
    // acknowledgment rather than silently doing nothing.

    // A config file's POOLS entries carry their own USER/PASS/TLS, so CLI
    // --user/--pass/--tls given without any --pool bind to nothing. Say so.
    if ((opts.seen.user || opts.seen.pass || opts.seen.tls) && !cli_pools && !opts.pools.empty()) {
        ui::console::info("Note: command-line --user/--pass/--tls were ignored; using pool credentials from the config file.");
    }

    // Declared before `stats` so it outlives the telemetry callback below, which
    // reads it: destruction is reverse declaration order.
    miner::Thermal thermal;

    // The console 'p' key. Read wherever thermal.paused() is: the engines'
    // pause predicate, the watchdog, and the statistics table's paused column.
    std::atomic<bool> user_paused{false};

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
#endif
#ifdef MXBM_HAVE_OPENCL
    // Info only: no context is created on anything. The PCI address each entry
    // carries is what lets an OpenCL device be recognised as a card CUDA also sees.
    const std::vector<gpu::DeviceInfo> cl_devices = gpu::Runtime::enumerate();
#endif
#ifdef MXBM_HAVE_METAL
    // Only when CUDA reported nothing: on a machine with both, CUDA's PCI-ordered list
    // is the one --devices and --pl are indexed against, and a second list underneath it
    // would make an index mean two things.
    const std::vector<gpu::MetalSolver::DeviceInfo> metal_devices =
#ifdef MXBM_HAVE_CUDA
        !cuda_devices.empty() ? std::vector<gpu::MetalSolver::DeviceInfo>{} :
#endif
        gpu::MetalSolver::enumerate();
#endif

    // NVML opens before the backend decision because device IDENTITY is decided
    // here: the join cross-checks each position against NVML's own view of the same
    // card. Opening it does nothing on its own -- every WRITE still waits for
    // `have_nvml` below, which waits for a GPU to actually claim a solver.
    const bool nvml_up = gpu::nvml_init();
    std::vector<gpu::NvmlCard> nvml_cards;
    for (unsigned i = 0, n = nvml_up ? gpu::nvml_device_count() : 0u; i < n; ++i)
        nvml_cards.push_back({gpu::nvml_device_name(i), gpu::nvml_pci_address(i)});

    // One row per PHYSICAL card: the backend that will drive it, each backend's own
    // index for it, and why not when the answer is none -- what makes a mixed rig
    // one process instead of one per backend (docs-internal/MIXED_RIG.md).
    std::vector<gpu::JoinedCard> cards;
#if defined(MXBM_HAVE_CUDA) || defined(MXBM_HAVE_OPENCL)
    {
        std::vector<gpu::CudaCard> cuda_view;
        std::vector<gpu::ClCard>   cl_view;
#ifdef MXBM_HAVE_CUDA
        for (const auto& d : cuda_devices)
            cuda_view.push_back({d.name, d.pci, d.index, d.global_mem, d.viable});
#endif
#ifdef MXBM_HAVE_OPENCL
        for (const auto& d : cl_devices)
            cl_view.push_back({d.name, d.vendor, d.pci, d.index, (unsigned long long)d.global_mem});
#endif
        cards = gpu::join_devices(cuda_view, cl_view, nvml_cards, opts.solver);
    }
#endif
#ifdef MXBM_HAVE_METAL
    // Metal outranks the joined table wherever it has devices: Apple's OpenCL is a
    // deprecated shim confined to the sort path, ~4.3x slower (see the construction
    // below), so on a Mac an index has to mean a Metal device.
    if (!metal_devices.empty()) cards.clear();
#endif
    device_count = (unsigned)cards.size();
#ifdef MXBM_HAVE_METAL
    if (device_count == 0) device_count = (unsigned)metal_devices.size();
#endif

    if (opts.list_devices) {
        // The joined table is exactly what will happen: the backend column is the
        // one that gets constructed, and an unused card prints why.
        if (!cards.empty()) {
            ui::console::info("Detected devices (indices are in PCI order, and mean the same "
                              "card in --devices and --pl):");
            for (size_t i = 0; i < cards.size(); ++i) {
                const gpu::JoinedCard& c = cards[i];
                const char* backend = c.backend == gpu::Backend::Cuda   ? "Cuda"
                                    : c.backend == gpu::Backend::OpenCL ? "OpenCL"
                                                                        : "not used";
                char line[256];
                std::snprintf(line, sizeof line, "  %zu: %-34s %5llu MB  PCI %-6s  %s",
                              i, c.name.c_str(),
                              (unsigned long long)(c.global_mem / (1024ull * 1024ull)),
                              c.pci.empty() ? "-" : c.pci.c_str(), backend);
                ui::console::info(line);
                if (!c.reason.empty()) ui::console::info("       " + c.reason);
            }
        }
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
        // Identities come from the joined table where there is one; a Metal-only
        // machine has no PCI or vendor to match, so it selects by index alone.
        std::vector<cli::DeviceRef> refs;
        refs.reserve(cards.size());
        for (const auto& c : cards) refs.push_back({c.pci, c.vendor});
        const bool ok = refs.size() == device_count
            ? cli::resolve_devices(opts.devices, refs, opts.devices_by_pcie, selected_devices, derr)
            : cli::resolve_devices(opts.devices, device_count, selected_devices, derr);
        if (!ok) {
            ui::console::error("--devices \"" + opts.devices + "\": " + derr
                               + " (try --list-devices)");
            return 1;
        }
    }
    const int device_index = selected_devices.empty() ? 0 : (int)selected_devices.front();
    // The primary card: everything that is single-device by nature -- the
    // benchmark, the tune, the device block -- is about this one.
    const gpu::JoinedCard* primary =
        (device_index >= 0 && (size_t)device_index < cards.size())
            ? &cards[(size_t)device_index] : nullptr;
    // Say which card was chosen whenever the choice was the user's, or whenever
    // there was more than one to choose from -- on a single-GPU box with no
    // --devices the line is noise.
    if (primary && (opts.seen.devices || cards.size() > 1)) {
        ui::console::info("Mining on device " + std::to_string(device_index) + ": "
                          + primary->name + " (PCI " + primary->pci + ")");
    }

    // The first selected device's solver. Everything that is single-device by
    // nature -- the benchmark, the device block, the NVML handle -- uses this
    // one; mining adds the rest below.
    std::unique_ptr<miner::Solver> solver;
    // Devices two and up, when --devices named several and the backend can
    // drive them. Each gets its own Engine and its own nonce lane.
    std::vector<std::unique_ptr<miner::Solver>> extra_solvers;
    std::vector<std::string> extra_labels;
    // Table positions of the cards that actually got a solver, in engine order.
    // NOT the selection: a skipped card would shift every telemetry row after it
    // onto the wrong card's sensors.
    std::vector<unsigned> solver_positions;
    // Set when a construction attempt was actually made and threw, so the
    // fallback below need not claim "no GPU available" when one was found.
    bool gpu_attempt_failed = false;
    // Shown in the stats table, the /summary API and the "Found a share" line.
    std::string worker_label = "GPU 0";
    // Left empty when no backend claims the device: then no block is printed.
    std::string dev_name, dev_driver;
    unsigned long long dev_mem = 0;

#ifdef MXBM_HAVE_METAL
    const int metal_index = device_index >= 0 ? device_index : 0;
#endif

    // Which backend WILL drive the primary card -- decided before construction,
    // because the power limit must be applied and observed before any buffer is
    // sized. The join already answered it per card; only Metal is added here,
    // being Apple-only and with no NVIDIA rig to be mixed with.
    bool cuda_will_run = false, metal_will_run = false, opencl_will_run = false;
#ifdef MXBM_HAVE_CUDA
    cuda_will_run = primary && primary->backend == gpu::Backend::Cuda;
#endif
#ifdef MXBM_HAVE_METAL
    metal_will_run = !cuda_will_run && cards.empty()
                  && (opts.solver == "metal" || opts.solver == "gpu" || opts.solver == "auto")
                  && gpu::MetalSolver::available(metal_index);
#endif
#ifdef MXBM_HAVE_OPENCL
    // True also when CUDA is the choice and OpenCL only the fallback: this gates
    // NVML, and a run that falls back must not lose its telemetry with the fall.
    opencl_will_run = !metal_will_run && primary && primary->cl_index >= 0
                   && (primary->backend == gpu::Backend::OpenCL
                       || primary->backend == gpu::Backend::Cuda);
#endif
    const bool gpu_will_run = cuda_will_run || metal_will_run || opencl_will_run;
    const bool have_nvml = gpu_will_run && nvml_up;

    // Sampled HERE, before any solver allocates: after construction the free
    // figure is what is left over, which is not what the sizing saw.
    struct VramSnapshot { bool have = false; uint64_t total = 0, free = 0, reserve = 0, usable = 0;
                          bool display = false; } vram;
    if (have_nvml && gpu::nvml_memory_info((unsigned)device_index, vram.free, vram.total)) {
        vram.have    = true;
        vram.display = gpu::nvml_display_active((unsigned)device_index);
        vram.reserve = gpu::reserve_bytes(vram.display);
        vram.usable  = gpu::usable_vram(vram.free, vram.total, vram.reserve);
    }

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

        // --pl auto: the value a --tune run stored, resolved PER CARD at apply
        // time -- each card has its own store entry, so on a multi-GPU rig each
        // gets its own wattage (or its own "no stored tune" complaint).
        auto resolve_pl_auto = [&](unsigned pos, const std::string& prefix,
                                   gpu::OcRequest r) -> gpu::OcRequest {
            if (r.pl != "auto") return r;
            // The key prefers NVML's own device name so it resolves on any
            // backend build; the CUDA enumeration is the fallback. Both produce
            // the same string on NVIDIA cards, so stores stay compatible.
            std::string key;
            if (have_nvml) {
                const std::string n = gpu::nvml_device_name(pos);
                if (!n.empty())
                    key = n + "@" + gpu::nvml_pci_address(pos);
            }
#ifdef MXBM_HAVE_CUDA
            if (key.empty() && (size_t)pos < cuda_devices.size())
                key = cuda_devices[(size_t)pos].name + "@"
                    + (have_nvml ? gpu::nvml_pci_address(pos) : std::string());
#endif
            std::string date;
            const unsigned w = key.empty() ? 0u : miner::tune_stored_pl(key, date);
            if (w) {
                // At this card's list position: oc_apply reads --pl as a per-GPU
                // list, so a bare number caps GPU 0 and leaves the rest at stock.
                r.pl = gpu::oc_spec_at(pos, (long)w);
                ui::console::info(prefix + "--pl auto: " + std::to_string(w)
                                  + " W, tuned for this card on " + date
                                  + " (re-run --tune after driver or cooling changes)");
                // Recommended, never auto-applied: locking a memory clock the user
                // did not ask for is a hardware setting nobody requested.
                unsigned rmhz = 0, rbelow = 0;
                if (opts.mem_clock.empty() && miner::tune_stored_rung(key, rmhz, rbelow)
                    && w < rbelow) {
                    ui::console::info(prefix + "--pl auto: at this cap the tune also "
                                      "measured the " + std::to_string(rmhz)
                                      + " MHz memory rung faster - consider adding "
                                        "--mclk " + std::to_string(rmhz));
                }
            } else {
                r.pl.clear();
                ui::console::error(prefix + "--pl auto: no stored tune for this card in "
                                   + miner::tune_store_path()
#ifdef _WIN32
                                   + " - run `mxbm --tune` once from an Administrator "
                                     "terminal; continuing at the card's current limit");
#else
                                   + " - run `sudo mxbm --tune` once; continuing at "
                                     "the card's current limit");
#endif
            }
            return r;
        };
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
            if (&c.spec == &ocreq.pl && c.spec == "auto") continue;   // resolved per card above
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
                // Printed before the block, and rig operators grep for it, so
                // the line stays even when every knob then fails -- "it tried
                // and could not" is the useful log; silence is not.
                ui::console::info("Applying overclock settings...");
                // Once per card that will actually run: mining applies to every
                // selected device (each per-GPU list entry lands at its own
                // position), benchmark and tune only to the card they measure.
                std::vector<unsigned> oc_targets;
                if (measure_mode)
                    oc_targets.push_back(device_index >= 0 ? (unsigned)device_index : 0u);
                else
                    oc_targets.assign(selected_devices.begin(), selected_devices.end());
                if (oc_targets.empty()) oc_targets.push_back(0u);
                const bool multi = oc_targets.size() > 1;
                for (unsigned pos : oc_targets) {
                    const std::string prefix =
                        multi ? "GPU " + std::to_string(pos) + ": " : std::string();
                    const gpu::OcRequest r = resolve_pl_auto(pos, prefix, ocreq);
                    for (const gpu::OcResult& res : gpu::oc_apply(pos, r)) {
                        if (res.status == gpu::OcStatus::Applied || res.status == gpu::OcStatus::Clamped)
                            ui::console::info(prefix + res.message);
                        else
                            ui::console::error(prefix + res.message);
                    }
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
        // The PRIMARY card's limit, not device 0's -- with --devices 1 the two
        // can differ, and geometry priced against another card's cap is wrong.
        const gpu::PowerLimit board_pl =
            gpu::nvml_power_limit(device_index >= 0 ? (unsigned)device_index : 0u);
        if (board_pl.valid) startup_pl_w = board_pl.current_w;
    }

#if defined(MXBM_HAVE_CUDA) || defined(MXBM_HAVE_OPENCL)
    // Every selected card through the SAME rule, in table order: the join's
    // backend, the other one as a fallback when construction throws, a named skip
    // when neither can drive it. The first card that yields a solver is the
    // primary. --benchmark and --tune stop after it -- allocating gigabytes on
    // cards they will never measure is cost without product.
    // One card's solver, by the rule above. Factored out because --report drives it
    // once per card in turn, rebuilding between cards so two allocations never
    // coexist -- on a rig of small cards, overlapping them would step one down a rung.
    struct BuiltSolver {
        std::unique_ptr<miner::Solver> s;
        std::string name;
        unsigned long long mem = 0;
        const char* driver = nullptr;
    };
    auto make_solver = [&](unsigned pos) -> BuiltSolver {
        BuiltSolver out;
        if (pos >= cards.size()) return out;
        const gpu::JoinedCard& c = cards[pos];
        const std::string tag = "Device " + std::to_string(pos) + " (" + c.name + ")";
        // Skipped with the reason, never constructed into a first-kernel-launch
        // failure the watchdog reads as a dead GPU and takes the rig down for.
        if (c.backend == gpu::Backend::None) {
            // No reason means no GPU backend was ASKED for (--solver ref, metal):
            // nothing was denied, so there is nothing to report.
            if (!c.reason.empty())
                ui::console::error(tag + ": " + c.reason + ". Continuing without it.");
            return out;
        }
        // Mines, but its identity is not fully pinned -- said out loud because the
        // consequence lands on hardware settings rather than on hashrate.
        if (c.degraded && !c.reason.empty()) ui::console::error(tag + ": " + c.reason);

        // Each card gets ITS observed limit -- NVML orders by bus id, the same
        // order `pos` indexes -- because a mixed rig can cap its cards apart.
        const gpu::PowerLimit cpl = have_nvml ? gpu::nvml_power_limit(pos) : gpu::PowerLimit{};
        const unsigned cpl_w = cpl.valid ? cpl.current_w : 0u;
        // The memory clock, observed the same way for the same reason: a rung MXBM
        // applied (--mclk) and one locked outside it look the same, because a held
        // lock reports its value even on an idle card.
        const gpu::Telemetry ctl = have_nvml ? gpu::nvml_sample(pos) : gpu::Telemetry{};
        const unsigned cmclk_mhz = ctl.have_mem ? ctl.mem_clock_mhz : 0u;

        std::unique_ptr<miner::Solver> s;
        std::string sname, why;
        unsigned long long smem = 0;
        const char* driver = nullptr;
#ifdef MXBM_HAVE_CUDA
        // CUDA first wherever the join chose it: it measures ~1.16x the OpenCL
        // path's rate on the same card (docs/performance.md).
        if (c.backend == gpu::Backend::Cuda) {
            try {
                auto cs = std::make_unique<gpu::CudaSolver>(c.cuda_index, cpl_w, cmclk_mhz);
                sname = cs->device().name;   // the real name, not a bare "GPU 0"
                smem = cs->device().global_mem;
                driver = "Cuda";
                s = std::move(cs);
            } catch (const std::exception& e) { why = e.what(); }
        }
#endif
#ifdef MXBM_HAVE_OPENCL
        // Both the choice for cards CUDA cannot drive and the fallback when a CUDA
        // constructor throws: a compositor holding a gigabyte can leave room for
        // the portable geometry and not the fast one's. --solver cuda opts out.
        if (!s && c.cl_index >= 0 && opts.solver != "cuda") {
            if (!why.empty())
                ui::console::error(tag + ": CUDA initialization failed (" + why
                                   + ") - trying the OpenCL path on the same card");
            try {
                auto gs = std::make_unique<gpu::GpuSolver>((unsigned)c.cl_index);
                sname = gs->device().name;
                smem = gs->device().global_mem;
                driver = "OpenCL";
                why.clear();
                s = std::move(gs);
            } catch (const std::exception& e) { why = e.what(); }
        }
#endif
        if (!s) {
            // Reported and skipped rather than taking the rig down with it: on a
            // multi-GPU box the whole point is that the others keep mining.
            gpu_attempt_failed = true;
            ui::console::error(tag + ": could not be initialised (" + why
                               + ") - continuing without it");
            return out;
        }
        out.s = std::move(s);
        out.name = sname;
        out.mem = smem;
        out.driver = driver;
        return out;
    };

    // The sweep's own per-point rebuild, for the card at `pos`. Rebuilding is what
    // makes a swept point a point of the program that would actually run at that cap:
    // the geometry preference and the speculative-entry gate are chosen at
    // construction. CUDA only -- the OpenCL solver takes no such inputs, and
    // rebuilding it would recompile its kernels at every point for nothing.
    //
    // Bound per card rather than once: --report walks several, and a remake pinned to
    // the first would rebuild on the wrong silicon for every card after it.
    // `driver` is what the RUNNING solver is, not what the join chose: a card whose
    // CUDA constructor threw is being driven by the OpenCL path, and rebuilding it as
    // a CudaSolver would swap the program mid-sweep.
    auto make_remake = [&](unsigned pos,
                           const std::string& driver) -> std::function<std::unique_ptr<miner::Solver>()> {
#ifdef MXBM_HAVE_CUDA
        if (driver == "Cuda" && pos < cards.size() && cards[pos].cuda_index >= 0) {
            const int rcuda = cards[pos].cuda_index;
            return [pos, rcuda]() -> std::unique_ptr<miner::Solver> {
                const gpu::PowerLimit rpl = gpu::nvml_power_limit(pos);
                const gpu::Telemetry rtl = gpu::nvml_sample(pos);
                try {
                    return std::make_unique<gpu::CudaSolver>(
                        rcuda, rpl.valid ? rpl.current_w : 0u,
                        rtl.have_mem ? rtl.mem_clock_mhz : 0u);
                } catch (const std::exception& e) {
                    ui::console::error(std::string("solver reconstruction failed (")
                                       + e.what() + ")");
                    return nullptr;
                }
            };
        }
#else
        (void)pos; (void)driver;
#endif
        return {};
    };

    for (size_t k = 0; k < selected_devices.size(); ++k) {
        // A measuring mode drives one card at a time, so only the first is built
        // here; --report rebuilds per card as it walks the rest.
        if (measure_mode && k > 0) break;
        BuiltSolver b = make_solver(selected_devices[k]);
        if (!b.s) continue;
        if (!solver) {
            worker_label = b.name;
            dev_name = b.name;
            dev_mem = b.mem;
            dev_driver = b.driver;
            solver = std::move(b.s);
        } else {
            extra_labels.push_back(b.name);
            extra_solvers.push_back(std::move(b.s));
        }
        solver_positions.push_back(selected_devices[k]);
    }
#endif
#ifdef MXBM_HAVE_METAL
    // Metal before OpenCL on Apple Silicon, and this is measured rather than assumed.
    // Apple's OpenCL is a deprecated 1.2-era shim that cannot build the fused
    // row-bucket kernels AT ALL (clCreateKernel -> CL_INVALID_KERNEL), so it is
    // confined to the sort path -- the slowest of the three collision finders. On an
    // M3 Max that is ~505 ms/solve against Metal's ~117 ms, a 4.3x difference.
    // --solver opencl still forces the portable path.
    if (!solver && cards.empty()
        && (opts.solver == "metal" || opts.solver == "gpu" || opts.solver == "auto")
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

    // What the pipeline was sized against, on the card that will run it. All
    // four numbers, because tuning a card near the edge needs them and because a
    // refusal to start is otherwise unattributable.
    if (vram.have && !dev_name.empty()) {
        const double mib = 1024.0 * 1024.0;
        char line[256];
        std::snprintf(line, sizeof line,
            "VRAM: %.0f MB total, %.0f MB free, reserving %.0f MB (%s) -> %.0f MB usable",
            vram.total / mib, vram.free / mib, vram.reserve / mib,
            gpu::keepfree_given() ? "--keepfree"
                                  : (vram.display ? "display attached" : "headless"),
            vram.usable / mib);
        ui::console::info(line);
    }

    if (!dev_name.empty()) {
        // The PRIMARY card, not device 0: with --devices 1 the block would
        // otherwise introduce the run under another card's index and PCI address.
        const unsigned primary = solver_positions.empty() ? 0u : solver_positions.front();
        ui::console::device_block(
            (int)primary, dev_name,
            have_nvml ? gpu::nvml_pci_address(primary) : std::string(),
            // Vendor is claimed only when NVML answered, itself proof of an NVIDIA
            // card; other vendors get no Vendor line rather than a guessed one.
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
        // id too, so the two agree without a translation table. It is the list of
        // cards that got a SOLVER, not the selection: engine N is row N is
        // solver_positions[N], and a card skipped mid-list must not slide the
        // rows after it onto their neighbours' sensors.
        std::vector<unsigned> nvml_index = solver_positions;
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
        stats.set_telemetry_source([nvml_index, watch_pl, pl_noticed, mem0, pl0, &thermal,
                                    &user_paused]
                                   (unsigned row, miner::Stats::Device& d) {
            const unsigned idx = row < nvml_index.size() ? nvml_index[row] : row;
            const gpu::Telemetry t = gpu::nvml_sample(idx);
            d.has_power = t.have_power;         d.power_w       = t.power_w;
            d.has_sm_clock = t.have_sm;         d.sm_clock_mhz  = t.sm_clock_mhz;
            d.has_mem_clock = t.have_mem;       d.mem_clock_mhz = t.mem_clock_mhz;
            d.has_temp = t.have_temp;           d.temp_c        = t.temp_c;
            d.has_fan = t.have_fan;             d.fan_pct       = t.fan_pct;
            d.has_util = t.have_util;           d.util_pct      = t.util_pct;
            d.paused = thermal.paused(row) || user_paused.load(std::memory_order_relaxed);
            if (row == 0 && watch_pl && !pl_noticed->load(std::memory_order_relaxed)) {
                const gpu::PowerLimit now = gpu::nvml_power_limit(nvml_index[0]);
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
                               "(needs compute capability 7.5 or newer with room for the full 2^25 seed layer, "
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
    if (!measure_mode) {
        if (opts.substituted_user)
            ui::console::error("No --user given for " + opts.pools[0].host +
                               "; sending \"" + opts.pools[0].user +
                               "\". Correct only for a local daemon -- if this endpoint "
                               "forwards to a real pool, your shares credit nobody.");
        if (!opts.pools[0].tls && cli::is_loopback_host(opts.pools[0].host))
            ui::console::info("TLS off (loopback pool); pass --tls 1 if it terminates TLS");

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
    // The measuring modes share their whole setup -- the same card, the same Ctrl+C
    // contract, and (for the two that sweep) the same TuneConfig, so --report's
    // curve is --tune's curve and not a second implementation of one.
    const unsigned measure_dev = device_index >= 0 ? (unsigned)device_index : 0u;
    // Atomic because a sweep replaces the solver at every point and publishes the
    // live one here for the signal handler to abort.
    static std::atomic<miner::Solver*> s_live{nullptr};
    std::atomic<bool> stop{false};
    miner::TuneConfig tcfg;
    std::string measure_key;
    if (measure_mode) {
        s_live.store(solver.get(), std::memory_order_relaxed);
        tcfg.device = measure_dev;
        tcfg.seconds_per_point = opts.tune_seconds;
        tcfg.min_gain = opts.tune_min_gain;
        // An explicit cap list is a chosen grid: measure exactly those points, no
        // refining passes. The default grid gets them.
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
        tcfg.live = &s_live;
        // The store key, the same derivation --pl auto uses: NVML's name first, the
        // active solver's as fallback, so store and lookup agree on any backend.
        const std::string nname = gpu::nvml_device_name(measure_dev);
        measure_key = (nname.empty() ? dev_name : nname) + "@"
                    + gpu::nvml_pci_address(measure_dev);
    }

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
        MeasureSignals sig(stop, s_live);
        return miner::run_tune(solver, stats, measure_key, tcfg, stop);
    }

    // --report: the benchmark below plus this card's power curve, as one
    // paste-ready block. Placed with the other measuring modes for the same
    // reason -- everything it reports must come from the loop mining drives.
    if (opts.report) {
        // A report describes ONE card. On a rig, which card is a choice the user has
        // to make out loud: a block that silently covered device 0 would be read as a
        // rig figure, and the other cards were idle while it was taken.
        if (cards.size() > 1 && opts.devices.empty()) {
            ui::console::error("--report measures one GPU at a time, and this rig has "
                               + std::to_string(cards.size())
                               + ". Say which, or ask for all of them:");
            size_t widest = 0;
            for (const gpu::JoinedCard& c : cards) widest = std::max(widest, c.name.size());
            for (size_t i = 0; i < cards.size(); ++i) {
                std::string row = "    --devices " + std::to_string(i) + "    " + cards[i].name;
                if (!cards[i].pci.empty())
                    row.append(widest - cards[i].name.size() + 2, ' ').append("(" + cards[i].pci + ")");
                ui::console::info(row);
            }
            ui::console::info("    --devices ALL  every card, one after another (~"
                              + std::to_string(miner::tune_estimated_minutes(tcfg, 6))
                              + " min each)");
            return 1;
        }

        MeasureSignals sig(stop, s_live);
        std::string collected;
        int rc = 0;
        // One card at a time, rebuilt between cards. Sequential on purpose: two
        // cards measured at once would each be reporting the other's contention.
        for (size_t i = 0; i < selected_devices.size(); ++i) {
            const unsigned pos = selected_devices[i];
            if (pos >= cards.size()) continue;
            if (solver_positions.empty() || solver_positions.front() != pos) {
                solver.reset();          // free the previous card before taking the next
                s_live.store(nullptr, std::memory_order_relaxed);
                BuiltSolver b = make_solver(pos);
                if (!b.s) { rc = 1; continue; }
                dev_name   = b.name;
                dev_driver = b.driver;
                solver     = std::move(b.s);
                solver_positions.assign(1, pos);
                s_live.store(solver.get(), std::memory_order_relaxed);
            }
            if (selected_devices.size() > 1)
                ui::console::info("=== GPU " + std::to_string(pos) + " of "
                                  + std::to_string(cards.size()) + ": " + dev_name
                                  + " (" + std::to_string(i + 1) + " of "
                                  + std::to_string(selected_devices.size()) + ") ===");
            const std::string nname = gpu::nvml_device_name(pos);
            miner::ReportConfig rcfg;
            rcfg.device          = pos;
            rcfg.seconds         = opts.report_seconds;
            rcfg.device_key      = (nname.empty() ? dev_name : nname) + "@"
                                 + gpu::nvml_pci_address(pos);
            rcfg.device_name     = dev_name;
            rcfg.backend         = dev_driver.empty() ? std::string("unknown") : dev_driver;
            rcfg.have_nvml       = have_nvml;
            rcfg.device_position = pos;
            rcfg.device_count    = cards.empty() ? 1u : (unsigned)cards.size();
            rcfg.tune            = tcfg;
            rcfg.tune.device     = pos;
            rcfg.tune.remake     = make_remake(pos, dev_driver);
            rc |= miner::run_report(solver, stats, rcfg, stop, &collected);
            if (stop.load(std::memory_order_relaxed)) break;
        }
        // Always saved: a report that only ever existed in a closed terminal cost the
        // user half an hour of sweeping. --report-out only chooses where.
        if (!collected.empty()) {
            std::string err;
            const std::string path = miner::report_resolve_path(
                opts.report_out, dev_name, selected_devices.size() > 1, err);
            if (path.empty() || !miner::report_write(path, collected, err))
                ui::console::error("Could not save the report (" + err
                                   + "); the block above is the same text");
            else
                ui::console::info("Saved to " + path);
        }
        ui::console::info("Report it at https://github.com/maxnflaxl/MXBM/issues/new"
                          "?template=benchmark-report.yml - nothing was uploaded.");
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

        // Ctrl+C ends the run and still prints the summary; the guard's destructor
        // hands SIGINT back to the OC restore hook on every exit from here.
        MeasureSignals sig(stop, s_live);

        ui::Ticker ticker;
        ticker.start(stats, opts.shortstats, opts.longstats, opts.digits, opts.timeprint, opts.apiport);
        // Telemetry over the whole run: the same window --tune and --report use, so
        // the efficiency line here is the same measurement they quote.
        gpu::TelemetryWindow tw(measure_dev);
        miner::BenchmarkResult r;
        // MXBM_ROUND_STATS=1: per-stage GPU medians after the summary. Opt-in so the
        // A/B loop runs the shipping path exactly.
        const bool round_stats = std::getenv("MXBM_ROUND_STATS") != nullptr;
        if (round_stats) solver->stage_timing(true);
        try {
            r = miner::run_benchmark(*solver, stats, opts.benchmark_seconds, stop);
        } catch (const std::exception& e) {
            // A backend can construct and still fail on first launch, most
            // often out of memory: available() saw free VRAM that another
            // process has since taken. Report it rather than core-dumping.
            ticker.stop();
            ui::console::error(std::string("Benchmark failed: ") + e.what());
            ui::console::info("If another process is using the GPU, stop it and retry: "
                              "a full BeamHash III search needs ~5.7 GiB free at the smallest geometry.");
            return 1;
        }
        ticker.stop();
        const gpu::TelemetrySummary t = tw.close(r.elapsed_s);

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
        if (t.have_sm || t.have_temp) {
            std::snprintf(line, sizeof line, "  %u MHz core, %u MHz memory, %u C (medians)",
                          t.sm_clock_mhz, t.mem_clock_mhz, t.temp_c);
            ui::console::info(line);
        }
        // Efficiency from the energy counter, when the card has one: exact
        // joules over the whole window, so J/solution carries no sampling error.
        if (t.have_energy && r.solutions > 0) {
            std::snprintf(line, sizeof line,
                "  %.0f J total   (%.1f W mean, %.2f J/solution - from the card's energy counter)",
                t.joules, t.power_w, t.joules / (double)r.solutions);
            ui::console::info(line);
        }
        if (r.solves < 100) {
            ui::console::info("  note: fewer than 100 solves - too few to quote a margin; "
                              "use --benchmark-seconds to run longer");
        }
        if (round_stats) {
            double gpu_ms = 0.0;
            const auto stages = solver->stage_times();
            for (const auto& s : stages) gpu_ms += s.median_ms;
            for (const auto& s : stages) {
                std::snprintf(line, sizeof line, "  %-9s %7.2f ms  %5.1f %%", s.name.c_str(),
                              s.median_ms, gpu_ms > 0.0 ? 100.0 * s.median_ms / gpu_ms : 0.0);
                ui::console::info(line);
            }
            if (!stages.empty()) {
                std::snprintf(line, sizeof line, "  GPU total %.2f ms of %.2f ms solve wall",
                              gpu_ms, r.median_ms);
                ui::console::info(line);
            }
        }
        ui::console::info("  (`--report` produces this plus the power curve, as a "
                          "paste-ready block)");
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

    client.on_disconnect = [&stats, &router]() {
        ui::console::disconnected();
        stats.record_disconnect();
        // Stops the fee clock for the outage: with the pool gone the user is
        // earning nothing, so nothing is owed until it sends work again.
        router.clear(miner::Origin::Main);
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
        ui::console::devfee_notice(sched.rate, sched.slice(), sched.cycle);
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
        if (!http_api.start(static_cast<uint16_t>(opts.apiport), stats, mxbm::version(),
                            opts.apihost.c_str())) {
            ui::console::error("API server failed to start on " + opts.apihost + ":"
                               + std::to_string(opts.apiport));
        } else {
            ui::console::info("API on " + opts.apihost + ":" + std::to_string(http_api.bound_port())
                              + (opts.apihost == "127.0.0.1"
                                     ? " (this machine only)"
                                     : " (every interface, unauthenticated)"));
        }
        // Either way, mining continues below -- the API is a convenience,
        // never a mining precondition.
    }

    ui::StatsLayout layout;
    if (!opts.statsformat.empty()) {
        std::string ferr;
        // Already validated at parse time; a config file goes through the same
        // check here so a bad key cannot reach the formatter.
        if (!ui::parse_stats_format(opts.statsformat, layout.columns, ferr)) {
            ui::console::error("--statsformat: " + ferr);
            return 1;
        }
    }
    layout.vertical = opts.vstats;
    if (opts.hstats) {
        // An explicit width wins; a bare --hstats asks the terminal, and a
        // terminal that will not say leaves the table unwrapped.
        layout.wrap_width = opts.stats_width > 0 ? opts.stats_width
                                                 : ui::console::terminal_width();
    }
    ui::Ticker ticker;
    ticker.start(stats, opts.shortstats, opts.longstats, opts.digits, opts.timeprint,
                 http_api.bound_port(), opts.silence, layout);

    // Thermal protection, wired BEFORE the engines start so a card that is
    // already too hot never takes a job. `solver_positions` maps an engine to the
    // physical device its temperature comes from.
    if (opts.tstop != 0 || opts.tstart != 0) {
        std::vector<miner::Thermal::Limits> limits(
            engines.size(), miner::Thermal::Limits{(unsigned)opts.tstop, (unsigned)opts.tstart});
        thermal.configure(std::move(limits));
        const std::vector<unsigned> positions = solver_positions;
        thermal.read_temp = [positions](unsigned engine, unsigned& temp) {
            if (engine >= positions.size()) return false;
            const gpu::Telemetry t = gpu::nvml_sample(positions[engine]);
            if (!t.have_temp) return false;
            temp = t.temp_c;
            return true;
        };
        thermal.on_pause = [](unsigned dev, unsigned temp) {
            ui::console::error("GPU " + std::to_string(dev) + ": paused at "
                               + std::to_string(temp) + " C (--tstop). It is not hung; "
                               "mining resumes when it cools.");
        };
        thermal.on_resume = [](unsigned dev, unsigned temp) {
            ui::console::info("GPU " + std::to_string(dev) + ": resuming at "
                              + std::to_string(temp) + " C (--tstart).");
        };
        thermal.start();
        char note[160];
        std::snprintf(note, sizeof note,
                      "Thermal: pausing at %d C, resuming at %s (--tmode %s)",
                      opts.tstop,
                      opts.tstart ? (std::to_string(opts.tstart) + " C").c_str() : "no restart temperature",
                      opts.tmode.c_str());
        ui::console::info(note);
    }

    // Every device holds while EITHER pause is on: thermal protection or the
    // console 'p' key. Thermal::paused() is false for an unconfigured device,
    // so without --tstop this reduces to the key alone.
    for (size_t k = 0; k < engines.size(); ++k) {
        const unsigned idx = (unsigned)k;
        engines[k]->paused = [&thermal, &user_paused, idx] {
            return user_paused.load(std::memory_order_relaxed) || thermal.paused(idx);
        };
    }

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
        // A paused card stops advancing its counter on purpose -- thermally or by
        // the console 'p' key; without this the first pause would exit(42) into a
        // restart loop.
        watchdog.device_paused = [&thermal, &user_paused](unsigned dev) {
            return user_paused.load(std::memory_order_relaxed) || thermal.paused(dev);
        };
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

    // Single-key console commands, on an interactive stdin only -- a piped or
    // service stdin leaves the reader off and the terminal untouched. Every
    // command prints through ui::console, so it lands in the transcript log
    // and survives --silence: a key press is an explicit request.
    ui::KeyReader keys;
    {
        // The ticker sets digits/api_port on its own copy of the layout; this
        // handler renders the same table, so it needs the same two fields.
        ui::StatsLayout klayout = layout;
        klayout.digits = opts.digits;
        klayout.api_port = http_api.bound_port();
        const bool can_pause = !engines.empty();
        const int digits = opts.digits;
        const bool started = keys.start(
            [&stats, &user_paused, klayout, can_pause, digits](ui::Key k) {
            switch (k) {
            case ui::Key::Speed:
                ui::console::info(ui::format_speed_line(stats.snapshot(), digits));
                break;
            case ui::Key::StatsBlock: {
                std::time_t t = std::time(nullptr);
                std::tm tm_buf{};
#ifdef _WIN32
                localtime_s(&tm_buf, &t);
#else
                localtime_r(&t, &tm_buf);
#endif
                char clock[16];
                std::snprintf(clock, sizeof clock, "%02d:%02d:%02d",
                              tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec);
                ui::console::stats_block(ui::format_stats_block(
                    stats.snapshot(), mxbm::version(), clock, klayout));
                break;
            }
            case ui::Key::Connection: {
                const miner::Stats::Snapshot s = stats.snapshot();
                if (s.pool.empty()) {
                    ui::console::info("No pool connection");
                    break;
                }
                std::string line = "Pool " + s.pool + " - uptime "
                    + ui::format_uptime(s.uptime) + ", connected in "
                    + std::to_string(s.connect_ms) + " ms, reconnects "
                    + std::to_string((unsigned long long)s.reconnects);
                if (!s.last_job_id.empty()) {
                    char job[96];
                    std::snprintf(job, sizeof job, ", last job %s (difficulty %.0f)",
                                  s.last_job_id.c_str(), s.last_job_units);
                    line += job;
                }
                if (s.last_latency_ms >= 0)
                    line += ", share latency " + std::to_string(s.last_latency_ms) + " ms";
                ui::console::info(line);
                break;
            }
            case ui::Key::Pause:
                if (!can_pause) {
                    ui::console::info("Nothing to pause - no mining device is active");
                } else if (user_paused.exchange(true)) {
                    ui::console::info("Already paused - press r to resume");
                } else {
                    ui::console::info("Paused - devices idle after the solve in "
                                      "flight; press r to resume");
                }
                break;
            case ui::Key::Resume:
                if (!can_pause) break;
                if (user_paused.exchange(false)) ui::console::info("Resuming mining");
                else                             ui::console::info("Not paused");
                break;
            case ui::Key::Help:
                ui::console::info(ui::key_help_line());
                break;
            case ui::Key::None:
                break;
            }
        });
        if (started) ui::console::info(ui::key_help_line());
    }

    client.run();   // blocks forever, reconnecting on drop; Ctrl+C exits
    return 0;
}
