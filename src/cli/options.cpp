#include "cli/options.h"
#include "gpu/overclock.h"
#include "ui/format.h"   // the statistics columns --statsformat may name

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
        "  -a, --algo BEAM-III    algorithm to mine (only BEAM-III is supported)\n"
        "  -p, --pool host:port   pool address (repeatable for failover pools)\n"
        "  -u, --user addr[.worker]\n"
        "                         wallet address, optionally with a worker suffix\n"
        "                         (repeat to bind one per --pool, or pass once for all).\n"
        "                         Optional for a loopback pool, which has no address\n"
        "\n"
        "Options:\n"
        "  -c, --coin BEAM        select by coin instead of by algorithm\n"
        "  --pass x               accepted for compatibility with pool instructions, and\n"
        "                         ignored: BeamHash III stratum logs in with the address\n"
        "                         alone, so there is no password on the wire\n"
        "  --tls [0|1]            enable/disable TLS to the pool (default: on for a remote\n"
        "                         pool, off for a loopback one; binds like --user)\n"
        "  --apiport N            enable the /summary API on port N, 0 disables it (default: 0)\n"
        "  --apihost ADDR         interface the API binds (default: 0.0.0.0, every interface;\n"
        "                         127.0.0.1 restricts it to this machine). The API is\n"
        "                         unauthenticated -- see docs/usage.md\n"
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
        "  --report               benchmark this card and print a paste-ready report for\n"
        "                         the project: hardware, throughput, telemetry, and this\n"
        "                         card's power curve. Measures the curve first (--tune,\n"
        "                         ~25 min, needs root) unless this build already stored\n"
        "                         one. Nothing is uploaded.\n"
        "                         --report-seconds N  benchmark length (default: 120)\n"
        "                         --report-out FILE   also write the block to a file\n"
        "  --tune                 measure this card's power/speed curve and recommend a\n"
#ifdef _WIN32
        "                         --pl value (needs admin, ~25 min, no pool): a coarse pass\n"
#else
        "                         --pl value (needs root, ~25 min, no pool): a coarse pass\n"
#endif
        "                         across the card's band, ~10 W steps around the best\n"
        "                         cap found, the card's low memory clock at the capped\n"
        "                         points -- so the verdict can also say when to add\n"
        "                         --mclk -- and a final refinement around the\n"
        "                         best-efficiency point. Stored per card; apply the\n"
        "                         wattage with --pl auto.\n"
        "                         --tune-seconds N   seconds per power point (default: 60)\n"
        "                         --tune-caps LIST   exactly these watt points, single pass\n"
        "                                            (\"100,140,220\"; default: 6 across the\n"
        "                                            band + the refining second pass)\n"
        "                         --tune-min-gain X  sol/s per extra watt below which more\n"
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
        "  --silence N            console verbosity: 0 everything (default), 1 no job lines,\n"
        "                         2 no job or share lines (accepts become * marks on the\n"
        "                         speed line), 3 the statistics block only. Applies to the\n"
        "                         --log transcript as well\n"
        "  --compactaccept        report accepted shares as * marks on the speed line\n"
        "  --statsformat LIST     which columns the statistics block shows, in order\n"
        "                         (gpuName,speed,poolHr,iter,shares,bestShare,hrPerWatt,\n"
        "                         power,coreClk,memClk,coreT,fanPct,sharesPerMin,wattPerHr,\n"
        "                         util,state). No presets: the default is what you get\n"
        "                         without the flag\n"
        "  --vstats               one column per GPU, fields as rows\n"
        "  --hstats [N]           wrap the table into groups N characters wide; a bare\n"
        "                         --hstats asks the terminal how wide it is\n"
        "  --tstop C              pause a GPU at this temperature; 0 disables (default)\n"
        "  --tstart C             resume it at this temperature; 0 leaves it paused\n"
        "  --tmode MODE           which sensor those read: edge (default), junction or\n"
        "                         memory. A sensor the card does not report is an error,\n"
        "                         not a silent fallback to edge\n"
        "  --keepfree MB          VRAM to leave unallocated, replacing the built-in reserve\n"
        "                         (256 MB with a display attached, 64 MB headless). 0 takes\n"
        "                         everything the driver reports free\n"
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
        "                         --moff is in MHz of TRANSFER RATE, as every other tool\n"
        "                         and guide states it, so it moves the reported memory\n"
        "                         clock by half the number given (--moff 200 -> +100 MHz).\n"
        "  --no-oc-reset [0|1]    leave --pl applied at exit instead of restoring (default: off)\n"
        "  --list-algos           print the supported algorithms and exit\n"
        "  --list-coins           print the supported coins and exit\n"
        "  -v, --version          print the version string and exit\n"
        "  -h, --help             show this help text\n";
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

std::string lower_trim(const std::string& s) {
    std::string out;
    for (char c : s) out += (char)std::tolower((unsigned char)c);
    while (!out.empty() && (out.front() == ' ' || out.front() == '\t')) out.erase(0, 1);
    while (!out.empty() && (out.back()  == ' ' || out.back()  == '\t')) out.pop_back();
    return out;
}

// "0000:01:00.0", "01:00" and "1:0" all normalise to "1:0" -- the short hex form
// NVML prints and the join stores. False on anything that is not two hex fields.
bool normalize_pci(const std::string& s, std::string& out) {
    std::string t = lower_trim(s);
    const size_t dot = t.find('.');
    if (dot != std::string::npos) t.erase(dot);              // drop the function
    size_t c1 = t.find(':');
    if (c1 == std::string::npos) return false;
    if (t.find(':', c1 + 1) != std::string::npos) t.erase(0, c1 + 1);   // drop the domain
    c1 = t.find(':');
    if (c1 == std::string::npos) return false;
    const std::string bus = t.substr(0, c1), dev = t.substr(c1 + 1);
    if (bus.empty() || dev.empty() || bus.size() > 8 || dev.size() > 8) return false;
    for (const std::string* f : {&bus, &dev}) {
        for (char c : *f) if (!std::isxdigit((unsigned char)c)) return false;
    }
    char buf[32];
    std::snprintf(buf, sizeof buf, "%lx:%lx", std::strtoul(bus.c_str(), nullptr, 16),
                  std::strtoul(dev.c_str(), nullptr, 16));
    out = buf;
    return true;
}

bool vendor_keyword(const std::string& tok) {
    return tok == "nvidia" || tok == "amd" || tok == "intel" || tok == "apple";
}

// Driver vendor strings are prose ("Advanced Micro Devices, Inc."), so a keyword
// matches on the spellings that string can take, not on equality.
bool vendor_matches(const std::string& keyword, const std::string& vendor) {
    const std::string v = lower_trim(vendor);
    if (v.empty()) return false;
    if (keyword == "amd") return v.find("amd") != std::string::npos
                              || v.find("advanced micro devices") != std::string::npos;
    return v.find(keyword) != std::string::npos;
}

// Dotted-quad IPv4, four octets in 0..255. No hostnames, no IPv6, no shorthand:
// the listener is AF_INET and binds exactly what is typed.
bool parse_ipv4(const std::string& s) {
    int octets = 0;
    size_t pos = 0;
    while (octets < 4) {
        const size_t dot = s.find('.', pos);
        const std::string tok = s.substr(pos, dot == std::string::npos ? std::string::npos
                                                                       : dot - pos);
        int v = 0;
        if (tok.empty() || tok.size() > 3 || !parse_int(tok, 0, 255, v)) return false;
        ++octets;
        if (dot == std::string::npos) return octets == 4;
        pos = dot + 1;
    }
    return false;
}

// The value bound to pool index `i`: the i-th entry of `vals`, else its last
// (a short list's final value carries forward), else `def` if it was empty.
template <typename T>
T bound_value(const std::vector<T>& vals, size_t i, const T& def) {
    if (vals.empty()) return def;
    return vals[i < vals.size() ? i : vals.size() - 1];
}

} // namespace

bool is_loopback_host(const std::string& h) {
    if (h.empty()) return false;
    if (h == "::1") return true;

    if (lower_trim(h) == "localhost") return true;

    // The whole 127.0.0.0/8 block: several local daemons are often separated by
    // address rather than by port.
    if (h.compare(0, 4, "127.") != 0) return false;
    int dots = 0;
    for (char c : h) {
        if (c == '.') { ++dots; continue; }
        if (c < '0' || c > '9') return false;
    }
    return dots == 3;
}

bool resolve_devices(const std::string& spec, const std::vector<DeviceRef>& devices,
                     bool by_pcie, std::vector<unsigned>& selected, std::string& err) {
    selected.clear();
    err.clear();

    const unsigned count = (unsigned)devices.size();
    const std::string norm = lower_trim(spec);

    if (norm.empty() || norm == "all") {
        for (unsigned i = 0; i < count; ++i) selected.push_back(i);
        return true;
    }
    if (count == 0) { err = "no devices were detected"; return false; }

    // Duplicates collapse rather than mining the same card twice, which would
    // halve its rate and look like a hardware fault.
    auto take = [&selected](unsigned i) {
        for (unsigned s : selected) if (s == i) return;
        selected.push_back(i);
    };

    size_t pos = 0;
    for (;;) {
        const size_t comma = norm.find(',', pos);
        const std::string tok = lower_trim(norm.substr(
            pos, comma == std::string::npos ? std::string::npos : comma - pos));
        if (tok.empty()) { err = "empty entry in the device list"; return false; }

        if (by_pcie) {
            std::string want;
            if (!normalize_pci(tok, want)) {
                err = "'" + tok + "' is not a PCI address (expected 1:0, 01:00 or 0000:01:00.0)";
                return false;
            }
            bool found = false;
            for (unsigned i = 0; i < count; ++i) {
                std::string have;
                if (!normalize_pci(devices[i].pci, have) || have != want) continue;
                take(i);
                found = true;
            }
            if (!found) { err = "no detected card is at PCI " + tok; return false; }
        } else if (vendor_keyword(tok)) {
            bool found = false;
            for (unsigned i = 0; i < count; ++i) {
                if (!vendor_matches(tok, devices[i].vendor)) continue;
                take(i);
                found = true;
            }
            if (!found) { err = "no detected card is made by " + tok; return false; }
        } else {
            for (char c : tok) {
                if (c < '0' || c > '9') {
                    err = "'" + tok + "' is not a device index, a vendor (NVIDIA, AMD, "
                          "INTEL, APPLE) or the word ALL";
                    return false;
                }
            }
            const long v = std::strtol(tok.c_str(), nullptr, 10);
            if (v < 0 || (unsigned long)v >= count) {
                err = "device " + tok + " does not exist; "
                    + std::to_string(count) + (count == 1 ? " was detected" : " were detected");
                return false;
            }
            take((unsigned)v);
        }

        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    if (selected.empty()) { err = "the device list selected nothing"; return false; }
    return true;
}

bool resolve_devices(const std::string& spec, unsigned count,
                     std::vector<unsigned>& selected, std::string& err) {
    // No identities: a vendor or PCI token then has nothing to match and is
    // reported as an unknown token rather than silently selecting nothing.
    return resolve_devices(spec, std::vector<DeviceRef>(count), false, selected, err);
}

bool parse_args(int argc, char** argv, Options& out, std::string& err) {
    out = Options{};
    err.clear();

    std::string algo;
    std::vector<std::string> pool_args, user_args, pass_args;
    std::vector<bool> tls_args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg.size() >= 2 && arg[0] == '-' && arg[1] != '-') {
            if (arg.size() != 2) {
                err = "unknown flag: " + arg + " (short flags cannot be combined)\n\n"
                    + usage_text();
                return false;
            }
            const char* expanded = nullptr;
            switch (arg[1]) {
                case 'h': expanded = "--help";    break;
                case 'v': expanded = "--version"; break;
                case 'a': expanded = "--algo";    break;
                case 'p': expanded = "--pool";    break;
                case 'u': expanded = "--user";    break;
                case 'c': expanded = "--coin";    break;
                default: break;
            }
            if (!expanded) { err = "unknown flag: " + arg + "\n\n" + usage_text(); return false; }
            arg = expanded;
        }

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
        if (arg == "--coin") {
            if (i + 1 >= argc) { err = "missing value for --coin\n\n" + usage_text(); return false; }
            std::string coin;
            for (const char* c = argv[++i]; *c; ++c) coin += (char)std::toupper((unsigned char)*c);
            if (coin != "BEAM") {
                err = "unsupported coin (MXBM mines BEAM only)\n\n" + usage_text();
                return false;
            }
            algo = "BEAM-III";
            continue;
        }
        if (arg == "--list-algos")  { out.list_algos = true; continue; }
        if (arg == "--list-coins")  { out.list_coins = true; continue; }
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
            out.seen.tune_seconds = true;
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
            out.seen.tune_caps = true;
            continue;
        }
        if (arg == "--tune-min-gain") {
            if (i + 1 >= argc) { err = "missing value for --tune-min-gain\n\n" + usage_text(); return false; }
            char* end = nullptr;
            double v = std::strtod(argv[++i], &end);
            if (!end || *end != '\0' || !(v > 0.0)) {
                err = "invalid --tune-min-gain (sol/s per watt, must be > 0)\n\n" + usage_text();
                return false;
            }
            out.tune_min_gain = v;
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
        if (arg == "--silence") {
            if (i + 1 >= argc) { err = "missing value for --silence\n\n" + usage_text(); return false; }
            int v;
            if (!parse_int(argv[++i], kSilenceMin, kSilenceMax, v)) {
                err = "invalid --silence (must be 0, 1, 2 or 3)\n\n" + usage_text();
                return false;
            }
            out.silence = v;
            out.seen.silence = true;
            continue;
        }
        if (arg == "--keepfree") {
            if (i + 1 >= argc) { err = "missing value for --keepfree\n\n" + usage_text(); return false; }
            int v;
            if (!parse_int(argv[++i], kKeepFreeMbMin, kKeepFreeMbMax, v)) {
                err = "invalid --keepfree (megabytes, 0 or more)\n\n" + usage_text();
                return false;
            }
            out.keepfree_mb = v;
            out.seen.keepfree = true;
            continue;
        }
        if (arg == "--tstop" || arg == "--tstart") {
            if (i + 1 >= argc) { err = "missing value for " + arg + "\n\n" + usage_text(); return false; }
            int v;
            if (!parse_int(argv[++i], kTempCMin, kTempCMax, v)) {
                err = "invalid " + arg + " (degrees C, 0 disables)\n\n" + usage_text();
                return false;
            }
            if (arg == "--tstop") { out.tstop = v;  out.seen.tstop = true; }
            else                  { out.tstart = v; out.seen.tstart = true; }
            continue;
        }
        if (arg == "--tmode") {
            if (i + 1 >= argc) { err = "missing value for --tmode\n\n" + usage_text(); return false; }
            const std::string m = lower_trim(argv[++i]);
            if (m != "edge" && m != "junction" && m != "memory") {
                err = "invalid --tmode (edge, junction or memory)\n\n" + usage_text();
                return false;
            }
            out.tmode = m;
            out.seen.tmode = true;
            continue;
        }
        if (arg == "--statsformat") {
            if (i + 1 >= argc) { err = "missing value for --statsformat\n\n" + usage_text(); return false; }
            out.statsformat = argv[++i];
            out.seen.statsformat = true;
            continue;
        }
        if (arg == "--vstats" || arg == "--hstats") {
            const bool vertical = (arg == "--vstats");
            // The width is optional: "--hstats 60" fixes it, a bare "--hstats"
            // asks the terminal. A following flag is not a width.
            if (i + 1 < argc) {
                const std::string next = argv[i + 1];
                if (!next.empty() && next.rfind("-", 0) != 0) {
                    int v;
                    if (!parse_int(next, 20, 1000, v)) {
                        err = "invalid width for " + arg + " (characters, 20..1000)\n\n"
                            + usage_text();
                        return false;
                    }
                    out.stats_width = v;
                    ++i;
                }
            }
            if (vertical) { out.vstats = true; out.seen.vstats = true; }
            else          { out.hstats = true; out.seen.hstats = true; }
            continue;
        }
        if (arg == "--compactaccept") {
            out.compactaccept = true;
            out.seen.compactaccept = true;
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
        if (arg == "--apihost") {
            if (i + 1 >= argc) { err = "missing value for --apihost\n\n" + usage_text(); return false; }
            const std::string host = argv[++i];
            if (!parse_ipv4(host)) {
                err = "invalid --apihost (expected an IPv4 address such as 0.0.0.0 or 127.0.0.1)\n\n"
                    + usage_text();
                return false;
            }
            out.apihost = host;
            out.seen.apihost = true;
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
        if (arg == "--devicesbypcie") {
            out.devices_by_pcie = true;
            out.seen.devices_by_pcie = true;
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
        if (arg == "--report") {
            out.report = true;
            out.seen.report = true;
            // A mode that names its own work, exactly like --benchmark and --tune.
            algo = "BEAM-III";
            continue;
        }
        if (arg == "--report-seconds") {
            if (i + 1 >= argc) { err = "missing value for --report-seconds\n\n" + usage_text(); return false; }
            char* end = nullptr;
            long v = std::strtol(argv[++i], &end, 10);
            if (!end || *end != '\0' || v < kBenchmarkSecondsMin) {
                err = "invalid --report-seconds (must be an integer >= 1)\n\n" + usage_text();
                return false;
            }
            out.report_seconds = (int)v;
            out.seen.report_seconds = true;
            continue;
        }
        if (arg == "--report-out") {
            if (i + 1 >= argc) { err = "missing value for --report-out\n\n" + usage_text(); return false; }
            out.report_out = argv[++i];
            out.seen.report_out = true;
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
            // look like a flag. "user_config.json" is the default path.
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
    if (out.vstats && out.hstats) {
        err = "--vstats and --hstats are two layouts; pick one\n\n" + usage_text();
        return false;
    }
    if (!out.statsformat.empty()) {
        std::vector<std::string> fields;
        std::string ferr;
        if (!ui::parse_stats_format(out.statsformat, fields, ferr)) {
            err = "invalid --statsformat: " + ferr + "\n\n" + usage_text();
            return false;
        }
    }
    // A stop that is not above the restart point can only flap: the device would
    // resume into the same reading that paused it.
    if (out.tstop != 0 && out.tstart != 0 && out.tstart >= out.tstop) {
        err = "--tstart must be below --tstop (it is the temperature to resume at)\n\n"
            + usage_text();
        return false;
    }
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

        // An IPv6 literal arrives bracketed because the split is on the last colon.
        // The brackets are URI syntax; getaddrinfo wants the address alone.
        if (pe.host.size() >= 2 && pe.host.front() == '[' && pe.host.back() == ']')
            pe.host = pe.host.substr(1, pe.host.size() - 2);
        pe.user = bound_value(user_args, i, std::string());
        pe.pass = bound_value(pass_args, i, std::string());

        const bool local = is_loopback_host(pe.host);
        pe.tls = bound_value(tls_args, i, !local);

        if (pe.user.empty()) {
            // A remote pool without an address mines to nobody.
            if (!local) {
                err = "missing --user\n\n" + usage_text();
                return false;
            }
            pe.user = kLoopbackDefaultUser;
            out.substituted_user = true;
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
