#include "cli/options.h"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <utility>
#include <vector>

namespace mxbm { namespace cli {

namespace {

std::string usage_text() {
    return
        "Usage: mxbm --algo BEAM-III --pool host:port --user addr[.worker] [options]\n"
        "\n"
        "Required:\n"
        "  --algo BEAM-III        algorithm to mine (only BEAM-III is supported)\n"
        "  --pool host:port       pool address (repeatable for failover pools)\n"
        "  --user addr[.worker]   wallet address, optionally with a worker suffix\n"
        "                         (repeat to bind one per --pool, or pass once for all)\n"
        "\n"
        "Options:\n"
        "  --pass x               pool password (optional; binds like --user)\n"
        "  --tls [0|1]            enable/disable TLS to the pool (default: on; binds like --user)\n"
        "  --apiport N            enable the /summary API on port N, 0 disables it (default: 0)\n"
        "  --shortstats N         short-stats interval in seconds, >=1 (default: 15)\n"
        "  --longstats N          long-stats interval in seconds, >=1 (default: 60)\n"
        "  --devices LIST         device selector (accepted, stored; no GPU backend yet)\n"
        "  --watchdog             enable the watchdog (accepted; arrives in a later phase)\n"
        "  --json [PATH]          load a JSON config file (default: user_config.json)\n"
        "  --profile NAME         select a profile from the --json config\n"
        "  --config PATH          load a flat KEY = VALUE config file\n"
        "  --nocolor, --nocolour  disable ANSI colors in console output\n"
        "  --version              print the version string and exit\n"
        "  --help                 show this help text\n";
}

// Strict decimal port parse in 1..65535; false on empty, any non-digit, or
// out-of-range input. Never touches `out` on failure.
bool parse_port(const std::string& s, uint16_t& out) {
    if (s.empty()) return false;
    for (char c : s) {
        if (c < '0' || c > '9') return false;
    }
    char* end = nullptr;
    long v = std::strtol(s.c_str(), &end, 10);
    if (end != s.c_str() + s.size() || v < 1 || v > 65535) return false;
    out = static_cast<uint16_t>(v);
    return true;
}

// Strict decimal integer parse in [lo, hi]; false on empty, any non-digit,
// out-of-range, or overflowing input. Never touches `out` on failure. Used
// for --apiport (lo=0) and --shortstats/--longstats (lo=1); unlike
// parse_port these are plain `int`, not a 16-bit port.
bool parse_int(const std::string& s, long lo, long hi, int& out) {
    if (s.empty()) return false;
    for (char c : s) {
        if (c < '0' || c > '9') return false;
    }
    errno = 0;
    char* end = nullptr;
    long v = std::strtol(s.c_str(), &end, 10);
    if (end != s.c_str() + s.size() || errno == ERANGE || v < lo || v > hi) return false;
    out = static_cast<int>(v);
    return true;
}

// The value bound to pool index `i`: the i-th entry of `vals` if it exists,
// otherwise its last entry (so a shorter list's final value carries forward
// to any remaining pools -- one value with N pools is the case where the
// list has exactly one entry, reused for all of them), otherwise `def` when
// the flag never appeared at all.
template <typename T>
T bound_value(const std::vector<T>& vals, size_t i, const T& def) {
    if (vals.empty()) return def;
    return vals[i < vals.size() ? i : vals.size() - 1];
}

} // namespace

bool parse_args(int argc, char** argv, Options& out, std::string& err) {
    out = Options{};
    err.clear();

    std::string algo;
    std::vector<std::string> pool_args, user_args, pass_args;
    std::vector<bool> tls_args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help") {
            out.help_requested = true;
            err = usage_text();
            return false;
        }
        if (arg == "--version") {
            // The one flag whose false return carries no err: main prints
            // the version string itself and exits 0.
            out.version_requested = true;
            return false;
        }
        if (arg == "--algo") {
            if (i + 1 >= argc) { err = "unsupported algo\n\n" + usage_text(); return false; }
            algo = argv[++i];
            continue;
        }
        if (arg == "--pool") {
            if (i + 1 >= argc) { err = "missing --pool\n\n" + usage_text(); return false; }
            pool_args.push_back(argv[++i]);
            continue;
        }
        if (arg == "--user") {
            if (i + 1 >= argc) { err = "missing --user\n\n" + usage_text(); return false; }
            user_args.push_back(argv[++i]);
            continue;
        }
        if (arg == "--pass") {
            if (i + 1 >= argc) { err = "missing value for --pass\n\n" + usage_text(); return false; }
            pass_args.push_back(argv[++i]);
            continue;
        }
        if (arg == "--tls") {
            // Optional value: only consumed when it's actually one of the
            // recognized tls tokens, so a bare "--tls" followed by the next
            // flag doesn't accidentally swallow that flag.
            if (i + 1 < argc) {
                std::string v = argv[i + 1];
                if (v == "0" || v == "off") { tls_args.push_back(false); ++i; continue; }
                if (v == "1" || v == "on")  { tls_args.push_back(true);  ++i; continue; }
            }
            tls_args.push_back(true);   // bare --tls
            continue;
        }
        if (arg == "--nocolor" || arg == "--nocolour") {
            out.nocolor = true;
            out.seen.nocolor = true;
            continue;
        }
        if (arg == "--watchdog") {
            out.watchdog_requested = true;
            continue;
        }
        if (arg == "--apiport") {
            if (i + 1 >= argc) { err = "missing value for --apiport\n\n" + usage_text(); return false; }
            int v;
            if (!parse_int(argv[++i], 0, 65535, v)) { err = "invalid --apiport (must be 0..65535)\n\n" + usage_text(); return false; }
            out.apiport = v;
            out.seen.apiport = true;
            continue;
        }
        if (arg == "--shortstats") {
            if (i + 1 >= argc) { err = "missing value for --shortstats\n\n" + usage_text(); return false; }
            int v;
            if (!parse_int(argv[++i], 1, INT_MAX, v)) { err = "invalid --shortstats (must be >=1)\n\n" + usage_text(); return false; }
            out.shortstats = v;
            out.seen.shortstats = true;
            continue;
        }
        if (arg == "--longstats") {
            if (i + 1 >= argc) { err = "missing value for --longstats\n\n" + usage_text(); return false; }
            int v;
            if (!parse_int(argv[++i], 1, INT_MAX, v)) { err = "invalid --longstats (must be >=1)\n\n" + usage_text(); return false; }
            out.longstats = v;
            out.seen.longstats = true;
            continue;
        }
        if (arg == "--devices") {
            if (i + 1 >= argc) { err = "missing value for --devices\n\n" + usage_text(); return false; }
            out.devices = argv[++i];
            out.seen.devices = true;
            continue;
        }
        if (arg == "--json") {
            // Optional value, same pattern as --tls above: only consumed
            // when the next token doesn't itself look like a flag, so a
            // bare "--json" followed by e.g. "--profile" doesn't swallow
            // it. the reference miner's own default filename when no path is given.
            out.use_json_config = true;
            if (i + 1 < argc) {
                std::string v = argv[i + 1];
                if (!v.empty() && v.rfind("--", 0) != 0) {
                    out.config_path = v;
                    ++i;
                    continue;
                }
            }
            out.config_path = "user_config.json";
            continue;
        }
        if (arg == "--profile") {
            if (i + 1 >= argc) { err = "missing value for --profile\n\n" + usage_text(); return false; }
            out.json_profile = argv[++i];
            continue;
        }
        if (arg == "--config") {
            if (i + 1 >= argc) { err = "missing value for --config\n\n" + usage_text(); return false; }
            out.config_path = argv[++i];
            continue;
        }

        err = "unknown flag: " + arg + "\n\n" + usage_text();
        return false;
    }

    // MXBM mines exactly one algorithm; missing and mismatched both land here.
    if (algo != "BEAM-III") {
        err = "unsupported algo\n\n" + usage_text();
        return false;
    }
    // Unlike a missing/mismatched --algo, an empty --pool list is NOT
    // rejected here: Task 5's config-file loaders (mxbm::config) may supply
    // pools instead when opts.use_json_config/config_path is set. main()
    // enforces "at least one pool, from CLI or config" AFTER the
    // config-merge step, once it's clear no further source can add one.
    out.seen.pools = !pool_args.empty();
    out.seen.user  = !user_args.empty();
    out.seen.pass  = !pass_args.empty();
    out.seen.tls   = !tls_args.empty();

    out.pools.reserve(pool_args.size());
    for (size_t i = 0; i < pool_args.size(); ++i) {
        // Split on the LAST ':' (tolerates a bare IPv6 literal host).
        size_t colon = pool_args[i].rfind(':');
        uint16_t port = 0;
        if (colon == std::string::npos || colon == 0 ||
            !parse_port(pool_args[i].substr(colon + 1), port)) {
            err = "invalid pool port\n\n" + usage_text();
            return false;
        }

        PoolEntry pe;
        pe.host = pool_args[i].substr(0, colon);
        pe.port = port;
        pe.user = bound_value(user_args, i, std::string());
        pe.pass = bound_value(pass_args, i, std::string());
        pe.tls  = bound_value(tls_args, i, true);

        if (pe.user.empty()) {
            err = "missing --user\n\n" + usage_text();
            return false;
        }

        out.pools.push_back(std::move(pe));
    }

    return true;
}

} } // namespace mxbm::cli
