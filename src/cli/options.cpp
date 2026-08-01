#include "cli/options.h"
#include "gpu/overclock.h"

#include <cctype>
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
        "  --devices LIST         which GPU to mine on: ALL (default) or a comma-separated\n"
        "                         list of indices as shown by --list-devices\n"
        "  --list-devices         print the detected GPUs, with their indices, and exit\n"
        "  --watchdog [ACTION]    watch for a GPU that stops working. ACTION is exit\n"
        "                         (default; exits 42 for a supervisor to restart), script\n"
        "                         (runs --watchdogscript), or off (report only)\n"
        "  --watchdogscript PATH  script to run when ACTION is script\n"
        "  --benchmark ALGO       offline benchmark (no pool, no wallet); ALGO is BEAM-III\n"
        "  --benchmark-seconds N  stop the benchmark after N seconds (default: until Ctrl+C)\n"
        "  --tune                 measure this card's power/speed curve and recommend a\n"
#ifdef _WIN32
        "                         --pl value (needs admin, ~25 min, no pool): a coarse pass\n"
#else
        "                         --pl value (needs root, ~25 min, no pool): a coarse pass\n"
#endif
        "                         across the card's band, ~10 W steps around the knee, the\n"
        "                         card's low memory rung at the capped points -- so the\n"
        "                         verdict can also say when to add --mclk -- and a final\n"
        "                         refinement around the best-efficiency point. Stored per\n"
        "                         card; apply the wattage with --pl auto.\n"
        "                         --tune-seconds N   seconds per power point (default: 60)\n"
        "                         --tune-caps LIST   exactly these watt points, single pass\n"
        "                                            (\"100,140,220\"; default: 6 across the\n"
        "                                            band + the refining second pass)\n"
        "                         --tune-knee X      sol/s per extra watt below which more\n"
        "                                            power stops paying (default: 0.07)\n"
        "  --solver cuda|metal|opencl|gpu|ref|auto\n"
        "                         solver backend. gpu = any GPU (CUDA, then Metal, then\n"
        "                         OpenCL), cuda/metal/opencl\n"
        "                         pin one, ref = CPU reference. Default: auto\n"
        "  --dev-fee PCT          raise the developer fee above its built-in rate, as a\n"
        "                         percentage (e.g. 2.5). Can only be raised, never lowered;\n"
        "                         see docs/devfee.md\n"
        "  --json [PATH]          load a JSON config file (default: user_config.json)\n"
        "  --profile NAME         select a profile from the --json config\n"
        "  --config PATH          load a flat KEY = VALUE config file\n"
        "  --nocolor, --nocolour  disable ANSI colors in console output\n"
        "  --log [0|1]            write a timestamped transcript of the console to a file\n"
        "                         (default: off; see --logfile for where it goes)\n"
        "  --logfile PATH         transcript location; implies --log. Default when --log is\n"
        "                         given alone: logs/mxbm_<date>_<time>.log\n"
        "  --timeprint [0|1]      stamp the short-stats line with [HH:MM:SS] (default: off)\n"
        "  --digits N             decimals on the speed figures, 0..6 (default: 2)\n"
        "  --pl W                 board power limit in watts, per GPU (\"240\", \"240,*,260\";\n"
        "                         * skips a GPU), or \"auto\" for the value a --tune run\n"
        "                         stored for this card. Needs root. Restored on exit\n"
        "                         unless --no-oc-reset. See docs/overclocking.md\n"
        "  --cclk MHz             lock the core clock       --coff MHz  shift its V/F curve\n"
        "  --mclk MHz             lock the memory clock     --moff MHz  shift its V/F curve\n"
        "  --fan PCT              fan target, in percent\n"
        "                         All take the same per-GPU list syntax as --pl and need\n"
        "                         root; the two offsets may be negative. Clamped to the\n"
        "                         band the driver reports, and restored on exit.\n"
        "  --no-oc-reset [0|1]    leave --pl applied at exit instead of restoring (default: off)\n"
        "  --version              print the version string and exit\n"
        "  --help                 show this help text\n";
}

// Strict decimal port parse in 1..65535. Never touches `out` on failure.
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

// Strict decimal integer parse in [lo, hi]. Never touches `out` on failure.
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

// The value bound to pool index `i`: the i-th entry of `vals`, else its last
// (a short list's final value carries forward), else `def` if it was empty.
template <typename T>
T bound_value(const std::vector<T>& vals, size_t i, const T& def) {
    if (vals.empty()) return def;
    return vals[i < vals.size() ? i : vals.size() - 1];
}

} // namespace

bool resolve_devices(const std::string& spec, unsigned count,
                     std::vector<unsigned>& selected, std::string& err) {
    selected.clear();
    err.clear();

    std::string norm;
    for (char c : spec) norm += (char)std::tolower((unsigned char)c);
    // Trim, so " ALL " and a config file's stray whitespace behave.
    while (!norm.empty() && (norm.front() == ' ' || norm.front() == '\t')) norm.erase(0, 1);
    while (!norm.empty() && (norm.back()  == ' ' || norm.back()  == '\t')) norm.pop_back();

    if (norm.empty() || norm == "all") {
        for (unsigned i = 0; i < count; ++i) selected.push_back(i);
        return true;
    }
    if (count == 0) { err = "no devices were detected"; return false; }

    size_t pos = 0;
    for (;;) {
        const size_t comma = norm.find(',', pos);
        std::string tok = norm.substr(pos, comma == std::string::npos ? std::string::npos
                                                                     : comma - pos);
        while (!tok.empty() && (tok.front() == ' ' || tok.front() == '\t')) tok.erase(0, 1);
        while (!tok.empty() && (tok.back()  == ' ' || tok.back()  == '\t')) tok.pop_back();
        if (tok.empty()) { err = "empty entry in the device list"; return false; }
        for (char c : tok) {
            if (c < '0' || c > '9') {
                err = "'" + tok + "' is not a device index (or the word ALL)";
                return false;
            }
        }
        const long v = std::strtol(tok.c_str(), nullptr, 10);
        if (v < 0 || (unsigned long)v >= count) {
            err = "device " + tok + " does not exist; "
                + std::to_string(count) + (count == 1 ? " was detected" : " were detected");
            return false;
        }
        // Collapse duplicates rather than mining the same card twice, which
        // would halve its rate and look like a hardware fault.
        bool dup = false;
        for (unsigned s2 : selected) dup = dup || s2 == (unsigned)v;
        if (!dup) selected.push_back((unsigned)v);
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    if (selected.empty()) { err = "the device list selected nothing"; return false; }
    return true;
}

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
            // Optional value: consumed only when it really is one of the
            // recognized tokens, so a bare --tls doesn't swallow the next flag.
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
        if (arg == "--pl") {
            if (i + 1 >= argc) { err = "missing value for --pl\n\n" + usage_text(); return false; }
            // Syntax only here; the WATTS are validated against the band the
            // driver reports for the actual card, which the CLI cannot see.
            const std::string spec = argv[++i];
            // "auto": the value --tune stored for this card. Resolved in main()
            // once the card is known -- the CLI cannot look it up, only pass it on.
            std::string low;
            for (char c : spec) low += (char)std::tolower((unsigned char)c);
            if (low == "auto") {
                out.power_limit = "auto";
                out.seen.power_limit = true;
                continue;
            }
            long v = 0; bool found = false; std::string perr;
            if (!gpu::oc_parse_list(spec, 0, v, found, perr)) {
                err = "invalid --pl (" + perr + ")\n\n" + usage_text();
                return false;
            }
            out.power_limit = spec;
            out.seen.power_limit = true;
            continue;
        }
        if (arg == "--tune") {
            out.tune = true;
            // A mode like --benchmark: it names the work itself, so it satisfies
            // --algo (there is only one algorithm to tune for).
            algo = "BEAM-III";
            continue;
        }
        if (arg == "--tune-seconds") {
            if (i + 1 >= argc) { err = "missing value for --tune-seconds\n\n" + usage_text(); return false; }
            char* end = nullptr;
            long v = std::strtol(argv[++i], &end, 10);
            if (!end || *end != '\0' || v < 15) {
                err = "invalid --tune-seconds (must be an integer >= 15; shorter "
                      "windows measure ramp, not rate)\n\n" + usage_text();
                return false;
            }
            out.tune_seconds = (int)v;
            continue;
        }
        if (arg == "--tune-caps") {
            if (i + 1 >= argc) { err = "missing value for --tune-caps\n\n" + usage_text(); return false; }
            // Syntax only, like --pl: a comma list of plain watt values. The band
            // check belongs to the driver, which the CLI cannot see.
            const std::string spec = argv[++i];
            size_t pos = 0;
            while (pos <= spec.size()) {
                const size_t comma = spec.find(',', pos);
                const std::string tok = spec.substr(pos, comma == std::string::npos
                                                             ? std::string::npos : comma - pos);
                if (tok.empty() || tok.find_first_not_of("0123456789") != std::string::npos) {
                    err = "invalid --tune-caps ('" + tok + "' is not a wattage)\n\n" + usage_text();
                    return false;
                }
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
            out.tune_caps = spec;
            continue;
        }
        if (arg == "--tune-knee") {
            if (i + 1 >= argc) { err = "missing value for --tune-knee\n\n" + usage_text(); return false; }
            char* end = nullptr;
            double v = std::strtod(argv[++i], &end);
            if (!end || *end != '\0' || !(v > 0.0)) {
                err = "invalid --tune-knee (sol/s per watt, must be > 0)\n\n" + usage_text();
                return false;
            }
            out.tune_knee = v;
            continue;
        }
        // The clock and fan knobs. Same list grammar as --pl; the only
        // difference is that the two OFFSETS accept a sign, because a negative
        // offset is an undervolt rather than a typo. As with --pl the VALUES
        // are validated against the band the driver reports for the actual
        // card, which the CLI cannot see -- only the syntax is checked here.
        if (arg == "--cclk" || arg == "--mclk" || arg == "--coff" || arg == "--moff"
            || arg == "--fan") {
            if (i + 1 >= argc) {
                err = "missing value for " + arg + "\n\n" + usage_text();
                return false;
            }
            const bool signed_ok = (arg == "--coff" || arg == "--moff" || arg == "--fan");
            const std::string spec = argv[++i];
            long v = 0; bool found = false; std::string perr;
            if (!gpu::oc_parse_list(spec, 0, v, found, perr, signed_ok)) {
                err = "invalid " + arg + " (" + perr + ")\n\n" + usage_text();
                return false;
            }
            if (arg == "--cclk")      { out.core_clock  = spec; out.seen.core_clock  = true; }
            else if (arg == "--mclk") { out.mem_clock   = spec; out.seen.mem_clock   = true; }
            else if (arg == "--coff") { out.core_offset = spec; out.seen.core_offset = true; }
            else if (arg == "--moff") { out.mem_offset  = spec; out.seen.mem_offset  = true; }
            else                      { out.fan         = spec; out.seen.fan         = true; }
            continue;
        }
        if (arg == "--log" || arg == "--timeprint" || arg == "--no-oc-reset") {
            // Optional value, handled exactly like --tls above.
            bool value = true;
            if (i + 1 < argc) {
                std::string v = argv[i + 1];
                if (v == "0" || v == "off")     { value = false; ++i; }
                else if (v == "1" || v == "on") { value = true;  ++i; }
            }
            if (arg == "--log")              { out.log_enabled = value; out.seen.log = true; }
            else if (arg == "--timeprint")   { out.timeprint = value;   out.seen.timeprint = true; }
            else                             { out.no_oc_reset = value; out.seen.no_oc_reset = true; }
            continue;
        }
        if (arg == "--watchdogscript") {
            if (i + 1 >= argc) {
                err = "missing value for --watchdogscript\n\n" + usage_text();
                return false;
            }
            out.watchdog_script = argv[++i];
            continue;
        }
        if (arg == "--logfile") {
            if (i + 1 >= argc) { err = "missing value for --logfile\n\n" + usage_text(); return false; }
            out.log_path = argv[++i];
            out.seen.logfile = true;
            // The "a log path means logging" rule lives in
            // resolve_implied_options(); a config may supply either half.
            continue;
        }
        if (arg == "--digits") {
            if (i + 1 >= argc) { err = "missing value for --digits\n\n" + usage_text(); return false; }
            int v;
            if (!parse_int(argv[++i], kDigitsMin, kDigitsMax, v)) {
                err = "invalid --digits (must be 0..6)\n\n" + usage_text();
                return false;
            }
            out.digits = v;
            out.seen.digits = true;
            continue;
        }
        if (arg == "--watchdog") {
            // Optional value, consumed only when it really is an action -- so a
            // bare --watchdog followed by another flag does not swallow it.
            out.watchdog_requested = true;
            out.seen.watchdog = true;
            // Recognised locally rather than by calling into miner/: the CLI
            // layer does not depend on the miner, and --solver validates its
            // own domain the same way. main() re-parses it through
            // miner::parse_watchdog_action, which is the single definition.
            if (i + 1 < argc) {
                const std::string v = argv[i + 1];
                if (v == "off" || v == "exit" || v == "script") { out.watchdog_action = v; ++i; }
            }
            continue;
        }
        if (arg == "--apiport") {
            if (i + 1 >= argc) { err = "missing value for --apiport\n\n" + usage_text(); return false; }
            int v;
            if (!parse_int(argv[++i], kApiPortMin, kApiPortMax, v)) { err = "invalid --apiport (must be 0..65535)\n\n" + usage_text(); return false; }
            out.apiport = v;
            out.seen.apiport = true;
            continue;
        }
        if (arg == "--shortstats") {
            if (i + 1 >= argc) { err = "missing value for --shortstats\n\n" + usage_text(); return false; }
            int v;
            if (!parse_int(argv[++i], kStatsIntervalMin, INT_MAX, v)) { err = "invalid --shortstats (must be >=1)\n\n" + usage_text(); return false; }
            out.shortstats = v;
            out.seen.shortstats = true;
            continue;
        }
        if (arg == "--longstats") {
            if (i + 1 >= argc) { err = "missing value for --longstats\n\n" + usage_text(); return false; }
            int v;
            if (!parse_int(argv[++i], kStatsIntervalMin, INT_MAX, v)) { err = "invalid --longstats (must be >=1)\n\n" + usage_text(); return false; }
            out.longstats = v;
            out.seen.longstats = true;
            continue;
        }
        if (arg == "--list-devices") {
            out.list_devices = true;
            out.seen.list_devices = true;
            continue;
        }
        if (arg == "--devices") {
            if (i + 1 >= argc) { err = "missing value for --devices\n\n" + usage_text(); return false; }
            out.devices = argv[++i];
            out.seen.devices = true;
            continue;
        }
        if (arg == "--benchmark") {
            if (i + 1 >= argc) { err = "missing value for --benchmark\n\n" + usage_text(); return false; }
            std::string v = argv[++i];
            // Accept the same spellings --algo does, so the two flags agree.
            std::string up;
            for (char c : v) up += (char)std::toupper((unsigned char)c);
            if (up != "BEAM-III" && up != "BEAMHASH3" && up != "BEAMHASHIII" && up != "BEAM") {
                err = "invalid --benchmark algorithm (only BEAM-III is supported)\n\n" + usage_text();
                return false;
            }
            out.benchmark = "BEAM-III";
            out.seen.benchmark = true;
            // --benchmark names the algorithm itself, so it satisfies --algo.
            // A conflicting --algo is caught below: both write this variable.
            algo = "BEAM-III";
            continue;
        }
        if (arg == "--benchmark-seconds") {
            if (i + 1 >= argc) { err = "missing value for --benchmark-seconds\n\n" + usage_text(); return false; }
            char* end = nullptr;
            long v = std::strtol(argv[++i], &end, 10);
            if (!end || *end != '\0' || v < kBenchmarkSecondsMin) {
                err = "invalid --benchmark-seconds (must be an integer >= 1)\n\n" + usage_text();
                return false;
            }
            out.benchmark_seconds = (int)v;
            out.seen.benchmark_seconds = true;
            continue;
        }
        if (arg == "--dev-fee") {
            // Raise-only: main() rejects a value below the built-in rate rather
            // than clamping it silently. See docs/devfee.md.
            if (i + 1 >= argc) { err = "missing value for --dev-fee\n\n" + usage_text(); return false; }
            const std::string v = argv[++i];
            char* end = nullptr;
            errno = 0;
            const double pct = std::strtod(v.c_str(), &end);
            if (v.empty() || !end || *end != '\0' || errno == ERANGE
                || !(pct == pct) || pct < kDevFeePctMin || pct > kDevFeePctMax) {
                err = "invalid --dev-fee (must be a percentage in 0..100, e.g. 2.5)\n\n" + usage_text();
                return false;
            }
            out.devfee_pct = pct;
            out.seen.devfee = true;
            continue;
        }
        if (arg == "--solver") {
            if (i + 1 >= argc) { err = "missing value for --solver\n\n" + usage_text(); return false; }
            std::string v = argv[++i];
            if (v != "gpu" && v != "cuda" && v != "metal" && v != "opencl" && v != "ref" && v != "auto") {
                err = "invalid --solver (must be cuda, metal, opencl, gpu, ref, or auto)\n\n" + usage_text();
                return false;
            }
            out.solver = v;
            out.seen.solver = true;
            continue;
        }
        if (arg == "--json") {
            // Optional value: consumed only when the next token doesn't itself
            // look like a flag. "user_config.json" is the reference miner's own default.
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

    // A MISMATCHED --algo is rejected right here -- no later source can make
    // "ETHASH" valid. A MISSING one is not, for the reason below.
    if (!algo.empty() && algo != "BEAM-III") {
        err = "unsupported algo\n\n" + usage_text();
        return false;
    }
    out.seen.algo = !algo.empty();
    // A missing algo and an empty --pool list are both left to main(): a config
    // file may still supply either, so those requirements are enforced AFTER
    // the merge, once it is clear no further source can add one.
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

void resolve_implied_options(Options& opts) {
    // A log path, from either source, means logging -- unless seen.log records
    // that something said otherwise explicitly.
    if (!opts.seen.log && !opts.log_path.empty()) opts.log_enabled = true;

    // A benchmark algorithm satisfies main()'s post-merge --algo requirement.
    if (!opts.benchmark.empty()) opts.seen.algo = true;
}

} } // namespace mxbm::cli
