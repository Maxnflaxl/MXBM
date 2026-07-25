#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace mxbm { namespace cli {

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
        bool solver = false;
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

} } // namespace mxbm::cli
