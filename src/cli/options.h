#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace mxbm { namespace cli {

// Accepted ranges for the numeric options, declared here rather than inline in
// the parser because the config loaders (config/config.cpp) validate the same
// options and the two must not drift: a config file that could set a value the
// command line rejects is a bug waiting to happen -- see the INT_MAX note on
// the stats intervals, where exactly that kind of gap once let a config wrap an
// interval negative and busy-loop the ticker.
constexpr int kApiPortMin = 0, kApiPortMax = 65535;
constexpr int kStatsIntervalMin = 1;          // upper bound is INT_MAX
constexpr int kDigitsMin = 0, kDigitsMax = 6;
constexpr int kBenchmarkSecondsMin = 1;       // upper bound is INT_MAX
constexpr double kDevFeePctMin = 0.0, kDevFeePctMax = 100.0;

// One pool entry: the host/port split out of a --pool host:port value, plus
// the user/pass/tls triple bound to it. See parse_args's doc comment for the
// binding rules that apply when --user/--pass/--tls are repeated across
// multiple --pool entries.
struct PoolEntry { std::string host; uint16_t port = 0; std::string user, pass; bool tls = true; };

// Parsed CLI options for the mxbm binary -- a the reference miner-shaped flag surface,
// matching the reference miner's documented command-line reference. MXBM mines exactly
// one algorithm (BeamHash III), so --algo is a
// required, exact-match confirmation rather than a selector. Phase B extends
// Phase A's single pool_host/pool_port/user/pass/tls fields into a `pools`
// vector plus the API/stats/config/version flag surface.
struct Options {
    std::vector<PoolEntry> pools;    // >=1 required (from CLI or config); Phase B code uses pools[0]
    bool nocolor = false;
    int  apiport = 0;                // 0 = API off
    int  shortstats = 15, longstats = 60;   // seconds, >=1
    std::string devices;             // accepted, stored; device SELECTION is a Phase-D notice
    bool watchdog_requested = false; // accepted; watchdog monitoring notice
    std::string solver = "auto";     // --solver gpu|ref|auto (default: prefer gpu, fall back to ref)

    // Console transcript (the reference miner's --log/--logfile). Off by default; an
    // explicit --logfile turns it on, since nobody names a log file expecting
    // no log. An empty log_path with logging on means the default location,
    // "logs/mxbm_<timestamp>.log" -- see ui::console::open_log.
    bool log_enabled = false;
    std::string log_path;

    // --timeprint: stamp the short-stats console line with "[HH:MM:SS]".
    // Off by default, matching the reference miner. Does not affect the transcript, whose
    // lines are always timestamped.
    bool timeprint = false;

    // --digits: decimals on the speed figures, on the short-stats line and in
    // the stats table's Speed/Pool columns alike. 2 by default.
    int digits = 2;

    // --dev-fee PCT: the developer fee the user WANTS to pay, as a
    // percentage. Raise-only -- main() rejects a value below the built-in
    // rate rather than clamping it, so "I lowered the fee" can never be
    // silently untrue. Negative means "not given"; the built-in rate applies.
    // See miner/devfee.h and docs/devfee.md.
    double devfee_pct = -1.0;

    // Offline benchmark mode. Non-empty => solve synthetic jobs and report
    // sol/s instead of connecting to a pool; --pool/--user are then neither
    // required nor used. benchmark_seconds 0 means run until Ctrl+C.
    std::string benchmark;
    int benchmark_seconds = 0;
    std::string config_path, json_profile;  // --config PATH, --profile NAME (with --json)
    bool use_json_config = false;    // --json
    bool version_requested = false, help_requested = false;

    // Which fields the CLI itself populated. Task 5's config-file loaders
    // (load_json_config / load_flat_config) consult this and only fill in
    // fields that are still false here -- CLI-supplied values always win
    // over a config file's. Set true the moment the corresponding flag
    // appears on the command line at all (not per-pool): e.g. `seen.pools`
    // is true as soon as any --pool is seen, `seen.user` as soon as any
    // --user is seen, regardless of how many pools that one flag ends up
    // bound to.
    struct Seen {
        bool pools = false, user = false, pass = false, tls = false, nocolor = false;
        bool apiport = false, shortstats = false, longstats = false, devices = false;
        bool solver = false, devfee = false;
        bool log = false, logfile = false, timeprint = false, digits = false;
        bool watchdog = false, benchmark = false, benchmark_seconds = false;
        // Unlike the rest, `algo` is not about CLI-wins-over-config precedence
        // (there is only one legal value, so nothing can conflict) -- it
        // records that SOME source supplied an algorithm at all, so main() can
        // enforce that after the config merge rather than parse_args having to
        // reject a command line a config file was about to complete.
        bool algo = false;
    } seen;
};

// Parses argv[1..argc) into `out`. Returns true on success, with `out` fully
// populated. Note: `out.pools` may be EMPTY on a true return -- an empty
// --pool list is not itself a parse error, since Task 5's config-file
// loaders (mxbm::config::load_json_config/load_flat_config) can supply
// pools instead; main() enforces "at least one pool, from CLI or config"
// after merging in any config file. On failure `out` must not be relied on,
// and `err` holds a human-readable message ready to print as-is -- with one
// exception:
//   - --help: err is exactly the usage text (no error prefix); help_requested=true.
//   - --version: the ONE case where a false return carries an EMPTY err.
//     version_requested=true and nothing else is validated; main is expected
//     to print the version string itself (not `err`) and exit 0.
//   - any other failure: err is "<specific reason>\n\n<usage text>".
//
// Repeated --pool/--user/--pass/--tls group positionally: the Nth --user
// (resp. --pass/--tls) binds to the Nth --pool. A single --user (or
// --pass/--tls) with multiple --pool entries applies to all of them; more
// generally, once a list runs out its last value is reused for any further
// pools. --pool/--user/--pass/--tls need not be interleaved in argv -- they
// are collected independently and bound by occurrence order once the whole
// command line has been scanned. Values in excess of the pool count are ignored.
bool parse_args(int argc, char** argv, Options& out, std::string& err);

// Applies the rules that hold BETWEEN options rather than within one, and so
// cannot be settled until every source -- command line and config file -- has
// had its say. Call once, after any config merge; idempotent, and a no-op when
// no config file is involved.
//
// The Seen mechanism gives each field its own CLI-wins-over-config precedence,
// which is the right model for a value, but it cannot express "this field
// implies something about that one". Both rules here are of that shape:
//
//   - A log path with no explicit --log/LOG anywhere means logging on. Nobody
//     names a log file expecting no log. Resolving it here rather than in
//     parse_args is what makes `LOGFILE` in a config work with a bare `--log`
//     on the command line, and what stops a config's `LOG = off` from
//     overriding a --logfile the user typed -- the parser cannot see either.
//   - A benchmark algorithm satisfies the --algo requirement, exactly as
//     the reference miner's --benchmark does, whichever source supplied it.
void resolve_implied_options(Options& opts);

} } // namespace mxbm::cli
