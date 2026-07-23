// the reference miner-shaped config-file loaders. See config.h for the merge contract
// shared with cli::parse_args (Task 4) and for the real user_config.json
// shape this mirrors.
#include "config/config.h"

#include <cctype>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <fstream>

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
// port strictly 1..65535 digits-only. Intentionally duplicated from
// cli::parse_args's identical rule (options.cpp keeps parse_port/the split
// file-local in an anonymous namespace, so there's no header symbol to
// share without widening options.h/.cpp beyond Task 5's scope -- see the
// Task 5 report) rather than sharing a helper across the two source files.
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

// Strict decimal integer parse in [lo, hi] for the flat-file loader --
// same contract as cli::parse_args's file-local parse_int, duplicated for
// the same reason as split_host_port above.
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

// Accepts the handful of boolean spellings a hand-edited flat config is
// likely to use: 1/0, true/false, on/off, case-insensitive.
bool parse_flat_bool(const std::string& s, bool& out) {
    std::string u = to_upper(s);
    if (u == "1" || u == "TRUE" || u == "ON")  { out = true;  return true; }
    if (u == "0" || u == "FALSE" || u == "OFF") { out = false; return true; }
    return false;
}

// Same idea for a JSON value: the reference miner's own examples use bare 0/1
// (e.g. "LOG" : 1), so a JSON boolean literal and a 0/1 integer are both
// accepted.
bool parse_json_bool(const nlohmann::ordered_json& j, bool& out) {
    if (j.is_boolean()) { out = j.get<bool>(); return true; }
    if (j.is_number_integer()) { out = j.get<long long>() != 0; return true; }
    return false;
}

// --- JSON profile application -------------------------------------------

// ALGO/APIPORT/NOCOLOR/SHORTSTATS/LONGSTATS/DEVICES. ALGO is validated
// whenever present regardless of any Seen flag (Options has nowhere to
// store it -- this is a pure guard, mirroring cli::parse_args's --algo
// check); the rest only fill their cli::Options twin when its Seen flag is
// still false.
bool apply_json_scalars(const nlohmann::ordered_json& prof, const std::string& profile_name,
                         cli::Options& opts, std::string& err) {
    auto ait = prof.find("ALGO");
    if (ait != prof.end()) {
        if (!ait->is_string() || ait->get<std::string>() != "BEAM-III") {
            err = "unsupported ALGO in profile '" + profile_name + "' (only BEAM-III is supported)";
            return false;
        }
    }

    if (!opts.seen.apiport) {
        auto it = prof.find("APIPORT");
        if (it != prof.end()) {
            if (!it->is_number_integer()) {
                err = "invalid APIPORT in profile '" + profile_name + "' (must be an integer)";
                return false;
            }
            long long v = it->get<long long>();
            if (v < 0 || v > 65535) {
                err = "invalid APIPORT in profile '" + profile_name + "' (must be 0..65535)";
                return false;
            }
            opts.apiport = static_cast<int>(v);
            opts.seen.apiport = true;
        }
    }

    if (!opts.seen.nocolor) {
        // NOCOLOUR is the British-spelling alias -- same acceptance as the
        // flat loader's "NOCOLOR" || "NOCOLOUR" key check and the CLI's
        // --nocolor/--nocolour.
        auto it = prof.find("NOCOLOR");
        if (it == prof.end()) it = prof.find("NOCOLOUR");
        if (it != prof.end()) {
            bool v;
            if (!parse_json_bool(*it, v)) {
                err = "invalid NOCOLOR in profile '" + profile_name + "'";
                return false;
            }
            opts.nocolor = v;
            opts.seen.nocolor = true;
        }
    }

    if (!opts.seen.shortstats) {
        auto it = prof.find("SHORTSTATS");
        if (it != prof.end()) {
            // Upper-bounded at INT_MAX (matching the flat loader's
            // parse_flat_int(..., 1, INT_MAX, ...) range): a JSON value
            // >= 2^31 would otherwise wrap when cast to int below, and a
            // negative/zero shortstats interval sends the ticker's wake
            // deadline into the past, busy-looping it.
            if (!it->is_number_integer() || it->get<long long>() < 1 || it->get<long long>() > INT_MAX) {
                err = "invalid SHORTSTATS in profile '" + profile_name + "' (must be 1.." + std::to_string(INT_MAX) + ")";
                return false;
            }
            opts.shortstats = static_cast<int>(it->get<long long>());
            opts.seen.shortstats = true;
        }
    }

    if (!opts.seen.longstats) {
        auto it = prof.find("LONGSTATS");
        if (it != prof.end()) {
            // Same INT_MAX upper bound as SHORTSTATS above, same reason.
            if (!it->is_number_integer() || it->get<long long>() < 1 || it->get<long long>() > INT_MAX) {
                err = "invalid LONGSTATS in profile '" + profile_name + "' (must be 1.." + std::to_string(INT_MAX) + ")";
                return false;
            }
            opts.longstats = static_cast<int>(it->get<long long>());
            opts.seen.longstats = true;
        }
    }

    if (!opts.seen.devices) {
        auto it = prof.find("DEVICES");
        if (it != prof.end()) {
            std::string joined;
            if (it->is_string()) {
                joined = it->get<std::string>();
            } else if (it->is_array()) {
                // the reference miner accepts DEVICES as an array; join into the same
                // comma-separated form --devices takes on the CLI.
                for (size_t i = 0; i < it->size(); ++i) {
                    if (!(*it)[i].is_string()) {
                        err = "invalid DEVICES in profile '" + profile_name + "'";
                        return false;
                    }
                    if (i) joined += ",";
                    joined += (*it)[i].get<std::string>();
                }
            } else {
                err = "invalid DEVICES in profile '" + profile_name + "'";
                return false;
            }
            opts.devices = joined;
            opts.seen.devices = true;
        }
    }

    return true;
}

// The profile's POOLS array -> opts.pools, only ever called when
// !opts.seen.pools. Each entry is {POOL:"host:port", USER, PASS?, TLS?};
// TLS can also be given once at profile level as a scalar-for-all fallback
// when a given entry omits its own TLS key. Replaces opts.pools wholesale
// (it is guaranteed empty on entry: seen.pools false means parse_args never
// populated it, and this is the only place that ever fills it from config).
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

        auto userIt = entry.find("USER");
        if (userIt == entry.end() || !userIt->is_string() || userIt->get<std::string>().empty()) {
            err = "a POOLS entry is missing USER in profile '" + profile_name + "'";
            return false;
        }
        pe.user = userIt->get<std::string>();

        auto passIt = entry.find("PASS");
        if (passIt != entry.end()) {
            if (!passIt->is_string()) {
                err = "invalid PASS in a POOLS entry in profile '" + profile_name + "'";
                return false;
            }
            pe.pass = passIt->get<std::string>();
        }

        pe.tls = profile_tls_set ? profile_tls_val : true;
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

    // ordered_json preserves file order (default nlohmann::json's object
    // type is a std::map and would silently reorder to alphabetical --
    // "first profile" needs to mean first-in-file, so this is deliberate,
    // not incidental).
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

    std::string algo, pool, user, pass, tls_str, apiport_str, nocolor_str, shortstats_str, longstats_str, devices;
    bool has_algo = false, has_pool = false, has_user = false, has_pass = false, has_tls = false;
    bool has_apiport = false, has_nocolor = false, has_shortstats = false, has_longstats = false, has_devices = false;

    std::string line;
    while (std::getline(f, line)) {
        std::string t = trim(line);
        if (t.empty() || t[0] == '#') continue;
        size_t eq = t.find('=');
        if (eq == std::string::npos) continue;   // no '=' on the line: ignore (forward-compat)

        std::string key = to_upper(trim(t.substr(0, eq)));
        std::string val = trim(t.substr(eq + 1));

        if      (key == "ALGO")       { algo = val;       has_algo = true; }
        else if (key == "POOL")       { pool = val;       has_pool = true; }
        else if (key == "USER")       { user = val;       has_user = true; }
        else if (key == "PASS")       { pass = val;       has_pass = true; }
        else if (key == "TLS")        { tls_str = val;    has_tls = true; }
        else if (key == "APIPORT")    { apiport_str = val; has_apiport = true; }
        else if (key == "NOCOLOR" || key == "NOCOLOUR") { nocolor_str = val; has_nocolor = true; }
        else if (key == "SHORTSTATS") { shortstats_str = val; has_shortstats = true; }
        else if (key == "LONGSTATS")  { longstats_str = val;  has_longstats = true; }
        else if (key == "DEVICES")    { devices = val;    has_devices = true; }
        // else: unknown key, ignored (forward-compat)
    }

    if (has_algo && algo != "BEAM-III") {
        err = "unsupported ALGO in config file: " + path + " (only BEAM-III is supported)";
        return false;
    }

    if (!opts.seen.apiport && has_apiport) {
        int v;
        if (!parse_flat_int(apiport_str, 0, 65535, v)) {
            err = "invalid APIPORT in config file: " + path;
            return false;
        }
        opts.apiport = v;
        opts.seen.apiport = true;
    }

    if (!opts.seen.nocolor && has_nocolor) {
        bool v;
        if (!parse_flat_bool(nocolor_str, v)) {
            err = "invalid NOCOLOR in config file: " + path;
            return false;
        }
        opts.nocolor = v;
        opts.seen.nocolor = true;
    }

    if (!opts.seen.shortstats && has_shortstats) {
        int v;
        if (!parse_flat_int(shortstats_str, 1, INT_MAX, v)) {
            err = "invalid SHORTSTATS in config file: " + path;
            return false;
        }
        opts.shortstats = v;
        opts.seen.shortstats = true;
    }

    if (!opts.seen.longstats && has_longstats) {
        int v;
        if (!parse_flat_int(longstats_str, 1, INT_MAX, v)) {
            err = "invalid LONGSTATS in config file: " + path;
            return false;
        }
        opts.longstats = v;
        opts.seen.longstats = true;
    }

    if (!opts.seen.devices && has_devices) {
        opts.devices = devices;
        opts.seen.devices = true;
    }

    if (!opts.seen.pools && has_pool) {
        cli::PoolEntry pe;
        if (!split_host_port(pool, pe.host, pe.port)) {
            err = "invalid POOL host:port in config file: " + path;
            return false;
        }
        pe.user = has_user ? user : std::string();
        if (pe.user.empty()) {
            err = "POOL given without USER in config file: " + path;
            return false;
        }
        pe.pass = has_pass ? pass : std::string();
        pe.tls = true;
        if (has_tls) {
            bool v;
            if (!parse_flat_bool(tls_str, v)) {
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
