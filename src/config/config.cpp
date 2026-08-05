// Config-file loaders. See config.h for the merge contract.
#include "config/config.h"

#include <cctype>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <map>

#include "nlohmann/json.hpp"

namespace mxbm { namespace config {

namespace {

// --- small string/number helpers, all file-local -----------------------

std::string trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return std::string();
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

std::string to_upper(std::string s) {
    for (char& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

// Splits "host:port" on the LAST ':' (tolerates a bare IPv6 literal host),
// port strictly 1..65535. options.cpp keeps its identical copy file-local.
bool split_host_port(const std::string& s, std::string& host, uint16_t& port) {
    size_t colon = s.rfind(':');
    if (colon == std::string::npos || colon == 0) return false;
    const std::string port_str = s.substr(colon + 1);
    if (port_str.empty()) return false;
    for (char c : port_str) {
        if (c < '0' || c > '9') return false;
    }
    char* end = nullptr;
    long v = std::strtol(port_str.c_str(), &end, 10);
    if (end != port_str.c_str() + port_str.size() || v < 1 || v > 65535) return false;
    host = s.substr(0, colon);
    port = static_cast<uint16_t>(v);
    return true;
}

// Strict decimal integer parse in [lo, hi]; twin of options.cpp's parse_int.
bool parse_flat_int(const std::string& s, long lo, long hi, int& out) {
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

// The boolean spellings a hand-edited flat config is likely to use.
bool parse_flat_bool(const std::string& s, bool& out) {
    std::string u = to_upper(s);
    if (u == "1" || u == "TRUE" || u == "ON")  { out = true;  return true; }
    if (u == "0" || u == "FALSE" || u == "OFF") { out = false; return true; }
    return false;
}

// Hand-written configs use bare 0/1 (e.g. "LOG" : 1), so accept both forms.
bool parse_json_bool(const nlohmann::ordered_json& j, bool& out) {
    if (j.is_boolean()) { out = j.get<bool>(); return true; }
    if (j.is_number_integer()) { out = j.get<long long>() != 0; return true; }
    return false;
}

// --- the option tables ---------------------------------------------------
//
// One row per option: the config key, where the value lands in cli::Options,
// which Seen flag guards it, and the range or domain it must satisfy. BOTH
// loaders walk these same tables, so a row cannot drift between the flat and
// the JSON format. The bounds come from cli/options.h, shared with the
// command-line parser so a config cannot set what the CLI rejects.
//
// Only the structural keys need bespoke code below: ALGO (a pure guard --
// Options has nowhere to store it) and POOL/POOLS with their USER/PASS/TLS.

struct IntOpt {
    const char* key;
    int cli::Options::*field;
    bool cli::Options::Seen::*seen;
    long lo, hi;
};

struct BoolOpt {
    const char* key;
    const char* alias;                    // nullptr when the key has no alias
    bool cli::Options::*field;
    bool cli::Options::Seen::*seen;
};

struct StrOpt {
    const char* key;
    std::string cli::Options::*field;
    bool cli::Options::Seen::*seen;
    // Accepted values, nullptr-terminated (case-insensitive), or nullptr for
    // free-form text.
    const char* const* domain;
    // When non-null, the value stored on a domain hit, so spelling variants
    // normalise to one canonical form.
    const char* canonical;
    bool join_array;                      // JSON only: accept an array, join with commas
};

struct DblOpt {
    const char* key;
    double cli::Options::*field;
    bool cli::Options::Seen::*seen;
    double lo, hi;
};

constexpr const char* kTmodeDomain[] = {"edge", "junction", "memory", nullptr};
constexpr const char* kSolverDomain[] = {"cuda", "opencl", "gpu", "ref", "auto", nullptr};
constexpr const char* kBenchmarkDomain[] = {"BEAM-III", "BEAMHASH3", "BEAMHASHIII", "BEAM", nullptr};

constexpr IntOpt kIntOpts[] = {
    {"APIPORT",           &cli::Options::apiport,           &cli::Options::Seen::apiport,           cli::kApiPortMin, cli::kApiPortMax},
    {"SHORTSTATS",        &cli::Options::shortstats,        &cli::Options::Seen::shortstats,        cli::kStatsIntervalMin, INT_MAX},
    {"LONGSTATS",         &cli::Options::longstats,         &cli::Options::Seen::longstats,         cli::kStatsIntervalMin, INT_MAX},
    {"DIGITS",            &cli::Options::digits,            &cli::Options::Seen::digits,            cli::kDigitsMin, cli::kDigitsMax},
    {"SILENCE",           &cli::Options::silence,           &cli::Options::Seen::silence,           cli::kSilenceMin, cli::kSilenceMax},
    {"KEEPFREE",          &cli::Options::keepfree_mb,       &cli::Options::Seen::keepfree,          cli::kKeepFreeMbMin, cli::kKeepFreeMbMax},
    {"TSTOP",             &cli::Options::tstop,             &cli::Options::Seen::tstop,             cli::kTempCMin, cli::kTempCMax},
    {"TSTART",            &cli::Options::tstart,            &cli::Options::Seen::tstart,            cli::kTempCMin, cli::kTempCMax},
    {"BENCHMARK_SECONDS", &cli::Options::benchmark_seconds, &cli::Options::Seen::benchmark_seconds, cli::kBenchmarkSecondsMin, INT_MAX},
};

constexpr BoolOpt kBoolOpts[] = {
    {"NOCOLOR",   "NOCOLOUR", &cli::Options::nocolor,            &cli::Options::Seen::nocolor},
    {"LOG",       nullptr,    &cli::Options::log_enabled,        &cli::Options::Seen::log},
    {"TIMEPRINT", nullptr,    &cli::Options::timeprint,          &cli::Options::Seen::timeprint},
    {"WATCHDOG",  nullptr,    &cli::Options::watchdog_requested, &cli::Options::Seen::watchdog},
    {"NO_OC_RESET", "NOOCRESET", &cli::Options::no_oc_reset,      &cli::Options::Seen::no_oc_reset},
    {"DEVICESBYPCIE", nullptr,   &cli::Options::devices_by_pcie,  &cli::Options::Seen::devices_by_pcie},
    {"COMPACTACCEPT", nullptr,   &cli::Options::compactaccept,    &cli::Options::Seen::compactaccept},
    {"VSTATS",    nullptr,       &cli::Options::vstats,           &cli::Options::Seen::vstats},
    {"HSTATS",    nullptr,       &cli::Options::hstats,           &cli::Options::Seen::hstats},
};

constexpr StrOpt kStrOpts[] = {
    {"APIHOST",   &cli::Options::apihost,   &cli::Options::Seen::apihost,   nullptr,          nullptr,    false},
    {"DEVICES",   &cli::Options::devices,   &cli::Options::Seen::devices,   nullptr,          nullptr,    true},
    {"TMODE",     &cli::Options::tmode,     &cli::Options::Seen::tmode,     kTmodeDomain,     nullptr,    false},
    {"STATSFORMAT", &cli::Options::statsformat, &cli::Options::Seen::statsformat, nullptr,   nullptr,    true},
    {"LOGFILE",   &cli::Options::log_path,  &cli::Options::Seen::logfile,   nullptr,          nullptr,    false},
    {"SOLVER",    &cli::Options::solver,    &cli::Options::Seen::solver,    kSolverDomain,    nullptr,    false},
    {"BENCHMARK", &cli::Options::benchmark, &cli::Options::Seen::benchmark, kBenchmarkDomain, "BEAM-III", false},
    {"PL",        &cli::Options::power_limit, &cli::Options::Seen::power_limit, nullptr,        nullptr,    true},
    {"CCLK",      &cli::Options::core_clock,  &cli::Options::Seen::core_clock,  nullptr,          nullptr,    true},
    {"MCLK",      &cli::Options::mem_clock,   &cli::Options::Seen::mem_clock,   nullptr,          nullptr,    true},
    {"COFF",      &cli::Options::core_offset, &cli::Options::Seen::core_offset, nullptr,          nullptr,    true},
    {"MOFF",      &cli::Options::mem_offset,  &cli::Options::Seen::mem_offset,  nullptr,          nullptr,    true},
    {"FAN",       &cli::Options::fan,         &cli::Options::Seen::fan,         nullptr,          nullptr,    true},
    {"WATCHDOGSCRIPT", &cli::Options::watchdog_script, &cli::Options::Seen::watchdog, nullptr, nullptr, false},
};

constexpr DblOpt kDblOpts[] = {
    // Raise-only is enforced by main(), not here, so a config file gets the
    // same refusal a command line would.
    {"DEVFEE", &cli::Options::devfee_pct, &cli::Options::Seen::devfee, cli::kDevFeePctMin, cli::kDevFeePctMax},
};

// Case-insensitive domain lookup; returns the matching entry, or nullptr.
const char* domain_find(const char* const* domain, const std::string& v) {
    const std::string u = to_upper(v);
    for (const char* const* p = domain; *p; ++p) {
        if (to_upper(*p) == u) return *p;
    }
    return nullptr;
}

// --- JSON profile application -------------------------------------------

// ALGO is validated whenever present, regardless of any Seen flag (a pure
// guard); every other key fills its twin only while its Seen flag is false.
bool apply_json_scalars(const nlohmann::ordered_json& prof, const std::string& profile_name,
                         cli::Options& opts, std::string& err) {
    auto ait = prof.find("ALGO");
    if (ait != prof.end()) {
        if (!ait->is_string() || ait->get<std::string>() != "BEAM-III") {
            err = "unsupported ALGO in profile '" + profile_name + "' (only BEAM-III is supported)";
            return false;
        }
        // A profile's ALGO satisfies main()'s post-merge requirement on its own.
        opts.seen.algo = true;
    }

    auto bad = [&](const char* key) {
        err = "invalid " + std::string(key) + " in profile '" + profile_name + "'";
        return false;
    };

    // The INT_MAX bound is not decoration: a value >= 2^31 wraps when narrowed
    // to int, and a non-positive stats interval busy-loops the ticker.
    for (const IntOpt& o : kIntOpts) {
        if (opts.seen.*(o.seen)) continue;
        auto it = prof.find(o.key);
        if (it == prof.end()) continue;
        if (!it->is_number_integer() || it->get<long long>() < o.lo || it->get<long long>() > o.hi) {
            err = "invalid " + std::string(o.key) + " in profile '" + profile_name +
                  "' (must be " + std::to_string(o.lo) + ".." + std::to_string(o.hi) + ")";
            return false;
        }
        opts.*(o.field) = static_cast<int>(it->get<long long>());
        opts.seen.*(o.seen) = true;
    }

    for (const BoolOpt& o : kBoolOpts) {
        if (opts.seen.*(o.seen)) continue;
        auto it = prof.find(o.key);
        if (it == prof.end() && o.alias) it = prof.find(o.alias);
        if (it == prof.end()) continue;
        bool v;
        if (!parse_json_bool(*it, v)) return bad(o.key);
        opts.*(o.field) = v;
        opts.seen.*(o.seen) = true;
    }

    for (const DblOpt& o : kDblOpts) {
        if (opts.seen.*(o.seen)) continue;
        auto it = prof.find(o.key);
        if (it == prof.end()) continue;
        if (!it->is_number()) return bad(o.key);
        const double v = it->get<double>();
        if (!(v >= o.lo) || !(v <= o.hi)) {   // written to reject NaN too
            err = "invalid " + std::string(o.key) + " in profile '" + profile_name +
                  "' (must be " + std::to_string(o.lo) + ".." + std::to_string(o.hi) + ")";
            return false;
        }
        opts.*(o.field) = v;
        opts.seen.*(o.seen) = true;
    }

    for (const StrOpt& o : kStrOpts) {
        if (opts.seen.*(o.seen)) continue;
        auto it = prof.find(o.key);
        if (it == prof.end()) continue;

        std::string value;
        if (it->is_string()) {
            value = it->get<std::string>();
        } else if (o.join_array && it->is_array()) {
            // DEVICES may be an array; join into --devices' form.
            // Entries may be numbers as well as strings, because the per-GPU
            // lists this serves are not all textual: "PL": [220, "*", 260] is
            // the natural way to write watts, and quoting them to satisfy the
            // parser would be a papercut with nothing behind it.
            for (size_t i = 0; i < it->size(); ++i) {
                const auto& e = (*it)[i];
                if (i) value += ",";
                if (e.is_string())            value += e.get<std::string>();
                else if (e.is_number_integer()) value += std::to_string(e.get<long long>());
                else                          return bad(o.key);
            }
        } else {
            return bad(o.key);
        }

        if (o.domain) {
            const char* hit = domain_find(o.domain, value);
            if (!hit) return bad(o.key);
            value = o.canonical ? o.canonical : hit;
        }
        opts.*(o.field) = value;
        opts.seen.*(o.seen) = true;
    }

    return true;
}

// The profile's POOLS array -> opts.pools, only ever called when
// !opts.seen.pools, which guarantees opts.pools is empty on entry. TLS may also
// be given once at profile level, for entries that omit their own.
bool apply_json_pools(const nlohmann::ordered_json& prof, const std::string& profile_name,
                       cli::Options& opts, std::string& err) {
    auto pit = prof.find("POOLS");
    if (pit == prof.end()) return true;   // no POOLS key: nothing to add, not an error
    if (!pit->is_array()) {
        err = "POOLS is not an array in profile '" + profile_name + "'";
        return false;
    }

    bool profile_tls_set = false, profile_tls_val = true;
    auto tit = prof.find("TLS");
    if (tit != prof.end()) {
        if (!parse_json_bool(*tit, profile_tls_val)) {
            err = "invalid TLS in profile '" + profile_name + "'";
            return false;
        }
        profile_tls_set = true;
    }

    std::vector<cli::PoolEntry> pools;
    pools.reserve(pit->size());
    for (const auto& entry : *pit) {
        if (!entry.is_object()) {
            err = "a POOLS entry is not an object in profile '" + profile_name + "'";
            return false;
        }

        auto poolIt = entry.find("POOL");
        if (poolIt == entry.end() || !poolIt->is_string()) {
            err = "a POOLS entry is missing POOL in profile '" + profile_name + "'";
            return false;
        }
        cli::PoolEntry pe;
        if (!split_host_port(poolIt->get<std::string>(), pe.host, pe.port)) {
            err = "invalid POOL host:port in profile '" + profile_name + "'";
            return false;
        }

        const bool local = cli::is_loopback_host(pe.host);

        auto userIt = entry.find("USER");
        if (userIt != entry.end() && !userIt->is_string()) {
            err = "invalid USER in a POOLS entry in profile '" + profile_name + "'";
            return false;
        }
        if (userIt == entry.end() || userIt->get<std::string>().empty()) {
            if (!local) {
                err = "a POOLS entry is missing USER in profile '" + profile_name + "'";
                return false;
            }
            pe.user = cli::kLoopbackDefaultUser;
        } else {
            pe.user = userIt->get<std::string>();
        }

        auto passIt = entry.find("PASS");
        if (passIt != entry.end()) {
            if (!passIt->is_string()) {
                err = "invalid PASS in a POOLS entry in profile '" + profile_name + "'";
                return false;
            }
            pe.pass = passIt->get<std::string>();
        }

        pe.tls = profile_tls_set ? profile_tls_val : !local;
        auto etls = entry.find("TLS");
        if (etls != entry.end() && !parse_json_bool(*etls, pe.tls)) {
            err = "invalid TLS in a POOLS entry in profile '" + profile_name + "'";
            return false;
        }

        pools.push_back(std::move(pe));
    }

    opts.pools = std::move(pools);
    opts.seen.pools = true;
    return true;
}

} // namespace

bool load_json_config(const std::string& path, const std::string& profile,
                       cli::Options& opts, std::string& err) {
    err.clear();
    std::ifstream f(path);
    if (!f.is_open()) {
        err = "cannot open config file: " + path;
        return false;
    }

    nlohmann::ordered_json root;
    try {
        f >> root;
    } catch (const std::exception&) {
        err = "malformed JSON in config file: " + path;
        return false;
    }

    if (!root.is_object() || root.empty()) {
        err = "no profiles found in config file: " + path;
        return false;
    }

    // ordered_json preserves file order; the default nlohmann::json's object
    // type is a std::map, so "first profile" would stop meaning first-in-file.
    std::string want = profile.empty() ? root.begin().key() : profile;
    auto it = root.find(want);
    if (it == root.end()) {
        err = "profile '" + want + "' not found in config file: " + path;
        return false;
    }
    if (!it->is_object()) {
        err = "profile '" + want + "' is not an object in config file: " + path;
        return false;
    }
    const nlohmann::ordered_json& prof = *it;

    if (!apply_json_scalars(prof, want, opts, err)) return false;
    if (!opts.seen.pools) {
        if (!apply_json_pools(prof, want, opts, err)) return false;
    }
    return true;
}

bool load_flat_config(const std::string& path, cli::Options& opts, std::string& err) {
    err.clear();
    std::ifstream f(path);
    if (!f.is_open()) {
        err = "cannot open config file: " + path;
        return false;
    }

    // Collected first, applied after: POOL needs the USER/PASS/TLS bound to it
    // whatever order the file lists them in. A repeated key keeps its LAST value.
    std::map<std::string, std::string> kv;

    std::string line;
    while (std::getline(f, line)) {
        std::string t = trim(line);
        if (t.empty() || t[0] == '#') continue;
        size_t eq = t.find('=');
        if (eq == std::string::npos) continue;   // no '=' on the line: ignore (forward-compat)
        kv[to_upper(trim(t.substr(0, eq)))] = trim(t.substr(eq + 1));
    }

    auto find = [&kv](const char* key, const char* alias = nullptr) {
        auto it = kv.find(key);
        if (it == kv.end() && alias) it = kv.find(alias);
        return it;
    };
    auto bad = [&](const char* key) {
        err = "invalid " + std::string(key) + " in config file: " + path;
        return false;
    };

    auto algo_it = find("ALGO");
    if (algo_it != kv.end()) {
        if (algo_it->second != "BEAM-III") {
            err = "unsupported ALGO in config file: " + path + " (only BEAM-III is supported)";
            return false;
        }
        opts.seen.algo = true;   // see the JSON loader's note above
    }

    // The same tables the JSON loader walks -- see kIntOpts above.
    for (const IntOpt& o : kIntOpts) {
        auto it = find(o.key);
        if (it == kv.end() || opts.seen.*(o.seen)) continue;
        int v;
        if (!parse_flat_int(it->second, o.lo, o.hi, v)) return bad(o.key);
        opts.*(o.field) = v;
        opts.seen.*(o.seen) = true;
    }

    for (const BoolOpt& o : kBoolOpts) {
        auto it = find(o.key, o.alias);
        if (it == kv.end() || opts.seen.*(o.seen)) continue;
        bool v;
        if (!parse_flat_bool(it->second, v)) return bad(o.key);
        opts.*(o.field) = v;
        opts.seen.*(o.seen) = true;
    }

    for (const DblOpt& o : kDblOpts) {
        auto it = find(o.key);
        if (it == kv.end() || opts.seen.*(o.seen)) continue;
        errno = 0;
        char* end = nullptr;
        const double v = std::strtod(it->second.c_str(), &end);
        if (it->second.empty() || !end || *end != '\0' || errno == ERANGE ||
            !(v >= o.lo) || !(v <= o.hi)) {   // the >=/<= form rejects NaN too
            return bad(o.key);
        }
        opts.*(o.field) = v;
        opts.seen.*(o.seen) = true;
    }

    for (const StrOpt& o : kStrOpts) {
        auto it = find(o.key);
        if (it == kv.end() || opts.seen.*(o.seen)) continue;
        std::string value = it->second;
        if (o.domain) {
            const char* hit = domain_find(o.domain, value);
            if (!hit) return bad(o.key);
            value = o.canonical ? o.canonical : hit;
        }
        opts.*(o.field) = value;
        opts.seen.*(o.seen) = true;
    }

    auto pool_it = find("POOL"), user_it = find("USER"), pass_it = find("PASS"), tls_it = find("TLS");
    const bool has_user = user_it != kv.end(), has_pass = pass_it != kv.end(), has_tls = tls_it != kv.end();
    if (!opts.seen.pools && pool_it != kv.end()) {
        const std::string& pool = pool_it->second;
        cli::PoolEntry pe;
        if (!split_host_port(pool, pe.host, pe.port)) {
            err = "invalid POOL host:port in config file: " + path;
            return false;
        }
        const bool local = cli::is_loopback_host(pe.host);   // same defaults as the CLI

        pe.user = has_user ? user_it->second : std::string();
        if (pe.user.empty()) {
            if (!local) {
                err = "POOL given without USER in config file: " + path;
                return false;
            }
            pe.user = cli::kLoopbackDefaultUser;
        }
        pe.pass = has_pass ? pass_it->second : std::string();
        pe.tls = !local;
        if (has_tls) {
            bool v;
            if (!parse_flat_bool(tls_it->second, v)) {
                err = "invalid TLS in config file: " + path;
                return false;
            }
            pe.tls = v;
        }
        opts.pools.push_back(std::move(pe));
        opts.seen.pools = true;
    }

    return true;
}

} } // namespace mxbm::config
