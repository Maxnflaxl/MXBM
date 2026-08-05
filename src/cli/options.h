#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace mxbm { namespace cli {

// Accepted ranges for the numeric options. The config loaders validate against
// these same constants, so a config file cannot set what the CLI rejects.
constexpr int kApiPortMin = 0, kApiPortMax = 65535;
constexpr int kStatsIntervalMin = 1;          // upper bound is INT_MAX
constexpr int kDigitsMin = 0, kDigitsMax = 6;
constexpr int kBenchmarkSecondsMin = 1;       // upper bound is INT_MAX
constexpr int kSilenceMin = 0, kSilenceMax = 3;
constexpr int kKeepFreeMbMin = 0, kKeepFreeMbMax = 1024 * 1024;   // 1 TiB, i.e. no real ceiling
// Degrees C. 0 disables; the upper bound is above any silicon's shutdown point,
// so the card's own thresholds do the real validating (nvml_temperature_limits).
constexpr int kTempCMin = 0, kTempCMax = 150;
constexpr double kDevFeePctMin = 0.0, kDevFeePctMax = 100.0;

// Host/port split out of a --pool value, plus the user/pass/tls bound to it.
struct PoolEntry { std::string host; uint16_t port = 0; std::string user, pass; bool tls = true; };

// True for "localhost", 127.0.0.0/8 and ::1 (bracketed or not). A loopback pool is
// a local process -- a node's own stratum server, a proxy, a p2pool daemon -- with
// no certificate to present and no wallet address to authenticate as, so TLS
// defaults off there and --user becomes optional.
bool is_loopback_host(const std::string& host);

// What a loopback pool sends when --user is omitted: the credential there is a
// local API key, often not enforced at all.
constexpr const char* kLoopbackDefaultUser = "x";

// Parsed CLI options. MXBM mines exactly one algorithm (BeamHash III), so
// --algo is a confirmation, not a selector.
struct Options {
    std::vector<PoolEntry> pools;    // >=1 required (from CLI or config)
    bool nocolor = false;
    int  apiport = 0;                // 0 = API off
    std::string apihost = "0.0.0.0";   // --apihost, dotted-quad IPv4 only
    int  shortstats = 15, longstats = 60;   // seconds, >=1
    // --devices: which GPU(s) to mine on. "ALL" (the default) or a
    // comma-separated list of indices as printed by --list-devices, which are
    // in PCI order so an index means the same card here, in --pl and in NVML.
    std::string devices;
    // --devicesbypcie: read --devices as PCI addresses instead of indices.
    bool devices_by_pcie = false;
    // --list-devices: print the device table and exit, mining nothing.
    bool list_devices = false;
    bool list_algos = false, list_coins = false;
    // --watchdog [off|exit|script]: what to do when a card stops doing work.
    // A bare --watchdog means "exit": a hung CUDA context usually cannot be
    // rebuilt from inside the process that wedged it, so handing the problem to
    // a supervisor is the only recovery that actually works.
    bool watchdog_requested = false;
    std::string watchdog_action = "exit";
    std::string watchdog_script;     // --watchdogscript PATH, for action=script
    std::string solver = "auto";     // --solver cuda|metal|opencl|gpu|ref|auto (default: prefer gpu, fall back to ref)

    // Console transcript (--log/--logfile). An explicit --logfile turns logging
    // on; an empty log_path means "logs/mxbm_<timestamp>.log".
    bool log_enabled = false;
    std::string log_path;

    // --timeprint: stamp the short-stats console line with "[HH:MM:SS]".
    bool timeprint = false;

    // --silence 0..3: 0 prints everything, 1 drops job lines, 2 also drops share
    // lines (accepts become '*' marks on the speed line), 3 leaves only the
    // statistics block. --compactaccept selects the '*' marks on their own.
    int  silence = 0;
    bool compactaccept = false;

    // --statsformat: comma-separated field names for the statistics block.
    // Empty = the built-in set. There are no presets; ui/format.h owns the list.
    std::string statsformat;
    // --vstats / --hstats [N]: one column per device (vertical), or wrap the
    // horizontal table into groups N characters wide. 0 = detect the terminal.
    bool vstats = false, hstats = false;
    int  stats_width = 0;

    // --keepfree MB: VRAM to leave unallocated. Replaces the built-in reserve
    // outright when given, 0 included. Negative means "not given".
    int  keepfree_mb = -1;

    // --tstop C pauses a device at that temperature, --tstart C resumes it. 0
    // disables either half. --tmode picks the sensor both read.
    int  tstop = 0, tstart = 0;
    std::string tmode = "edge";

    // --digits: decimals on the speed figures, 0..6.
    int digits = 2;

    // --pl: board power limit in watts, as a per-GPU list ("240", "240,*,260",
    // "*" to skip). Empty = leave the card alone. This is the
    // highest-value knob on the reference card -- MXBM runs pinned at the
    // limit in every kernel, so the limit picks the operating point. See
    // docs/performance.md "Power and efficiency" and docs/overclocking.md.
    std::string power_limit;

    // The rest of the overclock surface, same per-GPU list syntax, all empty
    // when not given. --cclk/--mclk LOCK a clock to a value; --coff/--moff
    // shift its voltage/frequency curve and may be negative. The pair is the
    // standard undervolt idiom -- lock the clock, raise the offset, so the
    // locked frequency runs at a lower voltage. docs/overclocking.md.
    std::string core_clock;      // --cclk MHz
    std::string mem_clock;       // --mclk MHz
    std::string core_offset;     // --coff MHz, signed
    std::string mem_offset;      // --moff MHz, signed
    std::string fan;             // --fan percent

    // --no-oc-reset: leave applied settings on the card at exit instead of
    // putting the previous ones back. Defaults to false (restore).
    bool no_oc_reset = false;

    // --dev-fee PCT: the fee the user WANTS to pay, as a percentage. Raise-only
    // -- main() rejects a value below the built-in rate rather than clamping
    // it. Negative means "not given"; the built-in rate applies. docs/devfee.md.
    double devfee_pct = -1.0;

    // Offline benchmark mode. Non-empty => solve synthetic jobs and report
    // sol/s instead of connecting to a pool; --pool/--user are then unused.
    // benchmark_seconds 0 means run until Ctrl+C.
    std::string benchmark;
    int benchmark_seconds = 0;

    // --tune: sweep power caps in the miner loop, recommend --pl/--mclk values
    // (miner/tune.h). A mode like --benchmark: no pool, exits when done, needs
    // root. --pl "auto" (stored in power_limit above) applies the stored knee.
    bool tune = false;
    int  tune_seconds = 60;      // --tune-seconds: measured seconds per power point
    std::string tune_caps;       // --tune-caps: watts list ("100,140,220"); empty = auto
    double tune_knee = 0.07;     // --tune-knee: sol/s per W below which watts stop paying
    std::string config_path, json_profile;  // --config PATH, --profile NAME (with --json)
    bool use_json_config = false;    // --json
    bool version_requested = false, help_requested = false;

    // Which fields the CLI itself populated. The config loaders only fill in
    // fields still false here, so CLI-supplied values always win. Set per flag,
    // not per pool: seen.user is true as soon as any --user appears.
    struct Seen {
        bool pools = false, user = false, pass = false, tls = false, nocolor = false;
        bool apiport = false, shortstats = false, longstats = false, devices = false;
        bool solver = false, devfee = false;
        bool log = false, logfile = false, timeprint = false, digits = false;
        bool silence = false, compactaccept = false, keepfree = false;
        bool tstop = false, tstart = false, tmode = false;
        bool statsformat = false, vstats = false, hstats = false;
        bool watchdog = false, benchmark = false, benchmark_seconds = false;
        bool list_devices = false, apihost = false, devices_by_pcie = false;
        bool power_limit = false, no_oc_reset = false;
        bool core_clock = false, mem_clock = false, core_offset = false;
        bool mem_offset = false, fan = false;
        // Not precedence (there is only one legal value): it records that SOME
        // source supplied an algorithm, for main() to enforce after the merge.
        bool algo = false;
    } seen;
};

// Parses argv[1..argc) into `out`. Returns true on success; `out.pools` may
// still be EMPTY then, since a config file can supply pools and main() enforces
// "at least one pool" after the merge. On failure `err` prints as-is, except:
//   - --help: err is exactly the usage text (no error prefix); help_requested=true.
//   - --version: the ONE case where a false return carries an EMPTY err;
//     nothing else is validated, and main prints the version itself and exits 0.
//
// Repeated --pool/--user/--pass/--tls bind positionally: the Nth --user (resp.
// --pass/--tls) goes with the Nth --pool, and once a list runs out its last
// value is reused for the remaining pools (so a single --user covers them all).
// They need not be interleaved -- all four are collected independently and
// bound by occurrence order once the whole command line has been scanned.
bool parse_args(int argc, char** argv, Options& out, std::string& err);

// One detected device as --devices can name it: its PCI address in the short
// "bus:device" form, and the driver's vendor string.
struct DeviceRef { std::string pci, vendor; };

// Resolves --devices against the devices actually detected.
//
// Accepts the same index lists as the count-based form below, plus two things
// that need to know what the cards are: vendor keywords (NVIDIA, AMD, INTEL,
// APPLE), and, with `by_pcie`, PCI addresses instead of indices. Addresses are
// accepted in any of the forms the tools print -- "1:0", "01:00", "0000:01:00.0"
// -- and matched against `devices[i].pci`.
bool resolve_devices(const std::string& spec, const std::vector<DeviceRef>& devices,
                     bool by_pcie, std::vector<unsigned>& selected, std::string& err);

// Resolves --devices against the number of devices actually detected.
//
// "ALL" (any case) and an empty spec both mean every device; anything else is a
// comma-separated list of indices as printed by --list-devices. Indices are
// validated against `count` HERE rather than at parse time, because the CLI is
// parsed before any driver has been asked what exists -- and "--devices 3" on a
// two-card rig has to be an error rather than a silent fallback to card 0.
//
// Returns false with `err` set on a malformed or out-of-range list. Duplicates
// collapse; order is preserved.
bool resolve_devices(const std::string& spec, unsigned count,
                     std::vector<unsigned>& selected, std::string& err);

// Applies the rules that hold BETWEEN options and so cannot be settled until
// every source -- command line and config file -- has had its say. Call once,
// after any config merge; idempotent.
//
//   - A log path with no explicit --log/LOG anywhere means logging on. Doing it
//     here is what lets a config's LOGFILE combine with a bare --log, and stops
//     a config's `LOG = off` from overriding a --logfile the user typed.
//   - A benchmark algorithm satisfies the --algo requirement, whichever source
//     supplied it.
void resolve_implied_options(Options& opts);

} } // namespace mxbm::cli
