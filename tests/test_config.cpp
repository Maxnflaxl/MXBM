#include "check.h"
#include "config/config.h"
#include "cli/options.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

using namespace mxbm;
using namespace mxbm::cli;
using namespace mxbm::config;

namespace {

// MXBM_FIXTURES is injected by CMake as the absolute path to
// tests/fixtures/, so the test binary finds its fixtures regardless of the
// directory ctest runs it from.
std::string fixture(const char* name) {
    return std::string(MXBM_FIXTURES) + "/" + name;
}

// Writes `contents` to a fresh file under the OS temp dir and returns its
// path -- used for the malformed-JSON case, which (unlike the two pinned
// fixtures) isn't a tracked fixture file.
std::string write_temp(const char* basename, const std::string& contents) {
    const char* dir = std::getenv("TMPDIR");
    std::string path = (dir && *dir ? std::string(dir) : std::string("/tmp"));
    if (!path.empty() && path.back() != '/') path += '/';
    path += basename;
    std::ofstream f(path, std::ios::trunc | std::ios::binary);
    f << contents;
    f.close();
    return path;
}

} // namespace

int main() {
    // --- JSON: RIG1 -> 2 pools with bound users, apiport 8080 ---
    {
        Options o; std::string err;
        check(load_json_config(fixture("user_config.json"), "RIG1", o, err), "RIG1 loads ok");
        check(err.empty(), "RIG1 load leaves err empty");
        check(o.pools.size() == 2, "RIG1 has 2 pools");
        check(o.pools[0].host == "pool.example.com" && o.pools[0].port == 1130, "RIG1 pool 1 host:port split");
        check(o.pools[0].user == "addr123.rig1" && o.pools[0].pass == "x", "RIG1 pool 1 user+pass bound");
        check(o.pools[1].host == "backup.example.com" && o.pools[1].port == 3334, "RIG1 pool 2 host:port split");
        check(o.pools[1].user == "addr123.rig1" && o.pools[1].pass.empty(), "RIG1 pool 2 user bound, pass defaults empty");
        check(o.pools[0].tls == true && o.pools[1].tls == true, "RIG1 pools default tls on (no TLS key in fixture)");
        check(o.apiport == 8080 && o.seen.apiport, "RIG1 apiport 8080 + seen.apiport set");
        check(o.seen.pools, "RIG1 sets seen.pools true after filling from config");
    }

    // --- JSON: profile RIG2 selects the other profile ---
    {
        Options o; std::string err;
        check(load_json_config(fixture("user_config.json"), "RIG2", o, err), "RIG2 loads ok");
        check(o.pools.size() == 1, "RIG2 has 1 pool");
        check(o.pools[0].host == "other.example.com" && o.pools[0].port == 5252, "RIG2 pool host:port split");
        check(o.pools[0].user == "addr456", "RIG2 pool user bound");
        check(o.apiport == 0 && !o.seen.apiport, "RIG2 has no APIPORT key -> apiport stays default, seen.apiport false");
    }

    // --- JSON: empty profile arg -> first profile (RIG1, file order) ---
    {
        Options o; std::string err;
        check(load_json_config(fixture("user_config.json"), "", o, err), "empty profile arg loads ok");
        check(o.pools.size() == 2 && o.apiport == 8080, "empty profile arg defaults to RIG1 (first in file)");
    }

    // --- JSON: missing profile -> false, err names it ---
    {
        Options o; std::string err;
        check(!load_json_config(fixture("user_config.json"), "NOSUCHRIG", o, err), "unknown profile rejected");
        check(err.find("NOSUCHRIG") != std::string::npos, "unknown-profile error names the profile");
    }

    // --- JSON: missing file -> false ---
    {
        Options o; std::string err;
        check(!load_json_config(fixture("does_not_exist.json"), "RIG1", o, err), "missing config file rejected");
        check(!err.empty(), "missing config file error message non-empty");
    }

    // --- JSON: malformed JSON -> false ---
    {
        std::string bad = write_temp("mxbm_test_config_malformed.json", "{ this is not valid json ");
        Options o; std::string err;
        check(!load_json_config(bad, "", o, err), "malformed JSON rejected");
        check(!err.empty(), "malformed JSON error message non-empty");
        std::remove(bad.c_str());
    }

    // --- JSON: ALGO mismatch in profile -> false ---
    {
        std::string bad = write_temp("mxbm_test_config_wrongalgo.json",
            R"({"RIG1": {"ALGO": "ETHASH", "POOLS": [{"POOL":"pool.example.com:1130","USER":"addr123"}]}})");
        Options o; std::string err;
        check(!load_json_config(bad, "RIG1", o, err), "wrong ALGO in profile rejected");
        check(!err.empty(), "wrong ALGO error message non-empty");
        std::remove(bad.c_str());
    }

    // --- JSON precedence: seen.pools already true -> loader must not touch
    //     pools, but still fills apiport (the merge rule under test) ---
    {
        Options o; std::string err;
        o.seen.pools = true;
        PoolEntry existing; existing.host = "keep.example.com"; existing.port = 9999; existing.user = "keepme";
        o.pools.push_back(existing);
        check(load_json_config(fixture("user_config.json"), "RIG1", o, err), "RIG1 loads ok with seen.pools preset");
        check(o.pools.size() == 1 && o.pools[0].host == "keep.example.com" && o.pools[0].user == "keepme",
              "preset pools untouched when seen.pools was already true");
        check(o.apiport == 8080 && o.seen.apiport, "apiport still filled even though pools were skipped");
    }

    // --- Flat: pools/user/apiport/nocolor land ---
    {
        Options o; std::string err;
        check(load_flat_config(fixture("flat.cfg"), o, err), "flat.cfg loads ok");
        check(err.empty(), "flat.cfg load leaves err empty");
        check(o.pools.size() == 1, "flat.cfg yields exactly one pool");
        check(o.pools[0].host == "pool.example.com" && o.pools[0].port == 1130, "flat.cfg pool host:port split");
        check(o.pools[0].user == "addr123.rig1", "flat.cfg pool user bound");
        check(o.pools[0].pass.empty() && o.pools[0].tls == true, "flat.cfg pool pass/tls default (no PASS/TLS keys)");
        check(o.apiport == 8080 && o.seen.apiport, "flat.cfg apiport 8080 + seen.apiport set");
        check(o.nocolor == true && o.seen.nocolor, "flat.cfg NOCOLOR = 1 -> nocolor true + seen.nocolor set");
        check(o.seen.pools, "flat.cfg sets seen.pools true after filling from config");
    }

    // --- Flat: missing file -> false ---
    {
        Options o; std::string err;
        check(!load_flat_config(fixture("does_not_exist.cfg"), o, err), "missing flat config file rejected");
        check(!err.empty(), "missing flat config file error message non-empty");
    }

    // --- Flat: unknown keys ignored (forward-compat) ---
    {
        std::string p = write_temp("mxbm_test_config_unknownkey.cfg",
            "ALGO = BEAM-III\nPOOL = pool.example.com:1130\nUSER = addr123\nFUTUREKEY = whatever\n");
        Options o; std::string err;
        check(load_flat_config(p, o, err), "flat config with an unknown key still loads");
        check(o.pools.size() == 1, "unknown key doesn't block the known keys from landing");
        std::remove(p.c_str());
    }

    // --- Flat precedence: seen.pools already true -> loader must not touch
    //     pools, but still fills apiport ---
    {
        Options o; std::string err;
        o.seen.pools = true;
        PoolEntry existing; existing.host = "keep.example.com"; existing.port = 9999; existing.user = "keepme";
        o.pools.push_back(existing);
        check(load_flat_config(fixture("flat.cfg"), o, err), "flat.cfg loads ok with seen.pools preset");
        check(o.pools.size() == 1 && o.pools[0].host == "keep.example.com" && o.pools[0].user == "keepme",
              "preset pools untouched when seen.pools was already true (flat)");
        check(o.apiport == 8080 && o.seen.apiport, "apiport still filled even though pools were skipped (flat)");
    }

    // --- Precedence beyond pools: an already-seen scalar is left alone too ---
    {
        Options o; std::string err;
        o.seen.apiport = true;
        o.apiport = 4444;
        check(load_json_config(fixture("user_config.json"), "RIG1", o, err), "RIG1 loads ok with seen.apiport preset");
        check(o.apiport == 4444, "preset apiport untouched when seen.apiport was already true");
        check(o.pools.size() == 2, "pools still filled since seen.pools was false");
    }

    // --- JSON: SHORTSTATS >= 2^31 rejected (would wrap to negative/zero on
    //     the static_cast<int> below and busy-loop the ticker) ---
    {
        std::string bad = write_temp("mxbm_test_config_shortstats_overflow.json",
            R"({"RIG1": {"SHORTSTATS": 2147483648, "POOLS": [{"POOL":"pool.example.com:1130","USER":"addr123"}]}})");
        Options o; std::string err;
        check(!load_json_config(bad, "RIG1", o, err), "SHORTSTATS >= 2^31 rejected");
        check(!err.empty(), "SHORTSTATS overflow error message non-empty");
        std::remove(bad.c_str());
    }

    // --- JSON: LONGSTATS well past INT_MAX rejected, same reason ---
    {
        std::string bad = write_temp("mxbm_test_config_longstats_overflow.json",
            R"({"RIG1": {"LONGSTATS": 3000000000, "POOLS": [{"POOL":"pool.example.com:1130","USER":"addr123"}]}})");
        Options o; std::string err;
        check(!load_json_config(bad, "RIG1", o, err), "LONGSTATS >= 2^31 rejected");
        check(!err.empty(), "LONGSTATS overflow error message non-empty");
        std::remove(bad.c_str());
    }

    // --- JSON: NOCOLOUR (British spelling) accepted as a NOCOLOR alias ---
    {
        std::string p = write_temp("mxbm_test_config_nocolour.json",
            R"({"RIG1": {"NOCOLOUR": true, "POOLS": [{"POOL":"pool.example.com:1130","USER":"addr123"}]}})");
        Options o; std::string err;
        check(load_json_config(p, "RIG1", o, err), "NOCOLOUR profile loads ok");
        check(o.nocolor == true && o.seen.nocolor, "NOCOLOUR = true -> nocolor true + seen.nocolor set");
        std::remove(p.c_str());
    }

    return summary("config");
}
