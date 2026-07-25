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
constexpr double kDevFeePctMin = 0.0, kDevFeePctMax = 100.0;

// Host/port split out of a --pool value, plus the user/pass/tls bound to it.
struct PoolEntry { std::string host; uint16_t port = 0; std::string user, pass; bool tls = true; };

// Parsed CLI options -- a the reference miner-shaped flag surface. MXBM mines exactly one
// algorithm (BeamHash III), so --algo is a confirmation, not a selector.
struct Options {
    std::vector<PoolEntry> pools;    // >=1 required (from CLI or config)
    bool nocolor = false;
    int  apiport = 0;                // 0 = API off
    int  shortstats = 15, longstats = 60;   // seconds, >=1
    std::string devices;             // accepted, stored; selection not yet implemented
    bool watchdog_requested = false; // accepted; monitoring not yet implemented
    std::string solver = "auto";     // --solver gpu|ref|auto (default: prefer gpu, fall back to ref)

    // Console transcript (the reference miner's --log/--logfile). An explicit --logfile
    // turns logging on; an empty log_path means "logs/mxbm_<timestamp>.log".
    bool log_enabled = false;
    std::string log_path;

    // --timeprint: stamp the short-stats console line with "[HH:MM:SS]".
    bool timeprint = false;

    // --digits: decimals on the speed figures, 0..6.
    int digits = 2;

    // --pl: board power limit in watts, as the reference miner's per-GPU list ("240",
    // "240,*,260", "*" to skip). Empty = leave the card alone. This is the
    // highest-value knob on the reference card -- MXBM runs pinned at the
    // limit in every kernel, so the limit picks the operating point. See
    // docs/performance.md "Power and efficiency" and docs/overclocking.md.
    std::string power_limit;

    // --no-oc-reset: leave applied settings on the card at exit instead of
    // putting the previous ones back. Defaults to false (restore), matching
    // the reference miner's own default of 0.
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
        bool watchdog = false, benchmark = false, benchmark_seconds = false;
        bool power_limit = false, no_oc_reset = false;
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
