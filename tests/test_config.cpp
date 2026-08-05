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

// MXBM_FIXTURES is injected by CMake as the absolute path to tests/fixtures/,
// so the binary finds them whatever directory ctest runs it from.
std::string fixture(const char* name) {
    return std::string(MXBM_FIXTURES) + "/" + name;
}

// Writes `contents` to a fresh file under the OS temp dir and returns its path
// -- for the malformed cases, which are not worth tracking as fixture files.
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

    {
        Options o; std::string err;
        check(load_json_config(fixture("user_config.json"), "RIG2", o, err), "RIG2 loads ok");
        check(o.pools.size() == 1, "RIG2 has 1 pool");
        check(o.pools[0].host == "other.example.com" && o.pools[0].port == 5252, "RIG2 pool host:port split");
        check(o.pools[0].user == "addr456", "RIG2 pool user bound");
        check(o.apiport == 0 && !o.seen.apiport, "RIG2 has no APIPORT key -> apiport stays default, seen.apiport false");
    }

    {
        Options o; std::string err;
        check(load_json_config(fixture("user_config.json"), "", o, err), "empty profile arg loads ok");
        check(o.pools.size() == 2 && o.apiport == 8080, "empty profile arg defaults to RIG1 (first in file)");
    }

    {
        Options o; std::string err;
        check(!load_json_config(fixture("user_config.json"), "NOSUCHRIG", o, err), "unknown profile rejected");
        check(err.find("NOSUCHRIG") != std::string::npos, "unknown-profile error names the profile");
    }

    {
        Options o; std::string err;
        check(!load_json_config(fixture("does_not_exist.json"), "RIG1", o, err), "missing config file rejected");
        check(!err.empty(), "missing config file error message non-empty");
    }

    {
        std::string bad = write_temp("mxbm_test_config_malformed.json", "{ this is not valid json ");
        Options o; std::string err;
        check(!load_json_config(bad, "", o, err), "malformed JSON rejected");
        check(!err.empty(), "malformed JSON error message non-empty");
        std::remove(bad.c_str());
    }

    {
        std::string bad = write_temp("mxbm_test_config_wrongalgo.json",
            R"({"RIG1": {"ALGO": "ETHASH", "POOLS": [{"POOL":"pool.example.com:1130","USER":"addr123"}]}})");
        Options o; std::string err;
        check(!load_json_config(bad, "RIG1", o, err), "wrong ALGO in profile rejected");
        check(!err.empty(), "wrong ALGO error message non-empty");
        std::remove(bad.c_str());
    }

    // --- the merge rule: an already-seen option is left alone, the rest fill ---
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

    // --- the same rules, the other format ---
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

    {
        Options o; std::string err;
        check(!load_flat_config(fixture("does_not_exist.cfg"), o, err), "missing flat config file rejected");
        check(!err.empty(), "missing flat config file error message non-empty");
    }

    {
        std::string p = write_temp("mxbm_test_config_apihost.cfg",
            "ALGO = BEAM-III\nPOOL = pool.example.com:1130\nUSER = addr123\n"
            "APIPORT = 8080\nAPIHOST = 127.0.0.1\n");
        Options o; std::string err;
        check(load_flat_config(p, o, err), "flat config with APIHOST loads");
        check(o.apihost == "127.0.0.1" && o.seen.apihost, "APIHOST lands in apihost + seen.apihost");
        std::remove(p.c_str());
    }

    {
        std::string p = write_temp("mxbm_test_config_apihost_cliwins.cfg",
            "ALGO = BEAM-III\nPOOL = pool.example.com:1130\nUSER = addr123\nAPIHOST = 10.0.0.5\n");
        Options o; std::string err;
        o.apihost = "127.0.0.1"; o.seen.apihost = true;
        check(load_flat_config(p, o, err), "flat config loads with apihost already seen");
        check(o.apihost == "127.0.0.1", "a command-line --apihost is not overwritten by the config");
        std::remove(p.c_str());
    }

    // Unknown keys are ignored rather than rejected, for forward compatibility.
    {
        std::string p = write_temp("mxbm_test_config_unknownkey.cfg",
            "ALGO = BEAM-III\nPOOL = pool.example.com:1130\nUSER = addr123\nFUTUREKEY = whatever\n");
        Options o; std::string err;
        check(load_flat_config(p, o, err), "flat config with an unknown key still loads");
        check(o.pools.size() == 1, "unknown key doesn't block the known keys from landing");
        std::remove(p.c_str());
    }

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

    {
        Options o; std::string err;
        o.seen.apiport = true;
        o.apiport = 4444;
        check(load_json_config(fixture("user_config.json"), "RIG1", o, err), "RIG1 loads ok with seen.apiport preset");
        check(o.apiport == 4444, "preset apiport untouched when seen.apiport was already true");
        check(o.pools.size() == 2, "pools still filled since seen.pools was false");
    }

    // An oversized interval would wrap to negative/zero on the static_cast<int>
    // and busy-loop the ticker.
    {
        std::string bad = write_temp("mxbm_test_config_shortstats_overflow.json",
            R"({"RIG1": {"SHORTSTATS": 2147483648, "POOLS": [{"POOL":"pool.example.com:1130","USER":"addr123"}]}})");
        Options o; std::string err;
        check(!load_json_config(bad, "RIG1", o, err), "SHORTSTATS >= 2^31 rejected");
        check(!err.empty(), "SHORTSTATS overflow error message non-empty");
        std::remove(bad.c_str());
    }

    {
        std::string bad = write_temp("mxbm_test_config_longstats_overflow.json",
            R"({"RIG1": {"LONGSTATS": 3000000000, "POOLS": [{"POOL":"pool.example.com:1130","USER":"addr123"}]}})");
        Options o; std::string err;
        check(!load_json_config(bad, "RIG1", o, err), "LONGSTATS >= 2^31 rejected");
        check(!err.empty(), "LONGSTATS overflow error message non-empty");
        std::remove(bad.c_str());
    }

    {
        std::string p = write_temp("mxbm_test_config_nocolour.json",
            R"({"RIG1": {"NOCOLOUR": true, "POOLS": [{"POOL":"pool.example.com:1130","USER":"addr123"}]}})");
        Options o; std::string err;
        check(load_json_config(p, "RIG1", o, err), "NOCOLOUR profile loads ok");
        check(o.nocolor == true && o.seen.nocolor, "NOCOLOUR = true -> nocolor true + seen.nocolor set");
        std::remove(p.c_str());
    }

    // --- the whole option table, both formats ---
    // Both loaders walk the same table (config.cpp), so what matters is that
    // every row really is reachable from each format.
    {
        std::string p = write_temp("mxbm_test_config_alloptions.cfg",
            "POOL = pool.example.com:1130\n"
            "USER = addr123\n"
            "LOG = on\n"
            "LOGFILE = /tmp/mxbm-flat.log\n"
            "TIMEPRINT = true\n"
            "DIGITS = 4\n"
            "SOLVER = cuda\n"
            "WATCHDOG = 1\n"
            "PL = 220\n"
            "NO_OC_RESET = 1\n"
            "DEVFEE = 2.5\n");
        Options o; std::string err;
        check(load_flat_config(p, o, err), "flat config with the full option set loads");
        check(o.log_enabled && o.log_path == "/tmp/mxbm-flat.log", "flat LOG/LOGFILE apply");
        check(o.timeprint && o.digits == 4, "flat TIMEPRINT/DIGITS apply");
        check(o.solver == "cuda" && o.watchdog_requested, "flat SOLVER/WATCHDOG apply");
        check(o.devfee_pct == 2.5 && o.seen.devfee, "flat DEVFEE applies");
        check(o.power_limit == "220" && o.no_oc_reset, "flat PL/NO_OC_RESET apply");
        std::remove(p.c_str());
    }
    {
        std::string p = write_temp("mxbm_test_config_alloptions.json",
            R"({"RIG1": {"LOG": 1, "LOGFILE": "/tmp/mxbm-json.log", "TIMEPRINT": true,
                         "DIGITS": 3, "SOLVER": "opencl", "WATCHDOG": true, "DEVFEE": 1.5,
                         "PL": [220, "*", 260], "NO_OC_RESET": true,
                         "POOLS": [{"POOL":"pool.example.com:1130","USER":"addr123"}]}})");
        Options o; std::string err;
        check(load_json_config(p, "RIG1", o, err), "JSON profile with the full option set loads");
        check(o.log_enabled && o.log_path == "/tmp/mxbm-json.log", "JSON LOG/LOGFILE apply");
        check(o.timeprint && o.digits == 3, "JSON TIMEPRINT/DIGITS apply");
        check(o.solver == "opencl" && o.watchdog_requested, "JSON SOLVER/WATCHDOG apply");
        check(o.devfee_pct == 1.5, "JSON DEVFEE applies");
        // A per-GPU knob is natural to write as a JSON array; the loader joins
        // it into the same comma list the CLI accepts, so one apply path serves
        // both formats.
        check(o.power_limit == "220,*,260" && o.no_oc_reset, "JSON PL array joins, NO_OC_RESET applies");
        std::remove(p.c_str());
    }

    // --- ranges and domains are enforced identically in both formats ---
    // The point of sharing one table with the CLI: a config file must never be
    // able to set a value the command line would reject.
    {
        std::string p = write_temp("mxbm_test_config_baddigits.cfg", "DIGITS = 40\n");
        Options o; std::string err;
        check(!load_flat_config(p, o, err), "flat DIGITS outside 0..6 is rejected");
        check(err.find("DIGITS") != std::string::npos, "the rejection names the key");
        std::remove(p.c_str());
    }
    {
        std::string p = write_temp("mxbm_test_config_baddigits.json", R"({"R": {"DIGITS": 40}})");
        Options o; std::string err;
        check(!load_json_config(p, "R", o, err), "JSON DIGITS outside 0..6 is rejected too");
        std::remove(p.c_str());
    }
    {
        std::string p = write_temp("mxbm_test_config_badsolver.cfg", "SOLVER = quantum\n");
        Options o; std::string err;
        check(!load_flat_config(p, o, err), "a SOLVER outside the accepted set is rejected");
        std::remove(p.c_str());
    }
    {
        // Hand-typed values get the same case-insensitivity the booleans have.
        std::string p = write_temp("mxbm_test_config_casesolver.cfg", "SOLVER = CUDA\n");
        Options o; std::string err;
        check(load_flat_config(p, o, err), "SOLVER accepts a differently-cased spelling");
        check(o.solver == "cuda", "...and stores the canonical lower-case form");
        std::remove(p.c_str());
    }

    // --- CLI still wins over the config ---
    {
        std::string p = write_temp("mxbm_test_config_precedence.cfg",
            "DIGITS = 5\nLOG = on\nSOLVER = ref\n");
        Options o; std::string err;
        o.digits = 2; o.seen.digits = true;          // as if --digits 2 had been given
        o.log_enabled = false; o.seen.log = true;    // as if --log off
        check(load_flat_config(p, o, err), "config loads over CLI-supplied values");
        check(o.digits == 2 && !o.log_enabled, "a CLI-set option is not overwritten by the config");
        check(o.solver == "ref", "...while an option the CLI did not set still comes from the config");
        std::remove(p.c_str());
    }

    // --- the cross-source rules (cli::resolve_implied_options) ---
    // These cannot live in either the parser or the loader: each sees only half
    // the picture. The interesting case is a path from one source and the
    // switch from the other.
    {
        Options o;                       // LOGFILE from a config, nothing from the CLI
        o.log_path = "/tmp/x.log";
        resolve_implied_options(o);
        check(o.log_enabled, "a log path with no explicit switch means logging on");
    }
    {
        Options o;                       // --log off on the CLI, LOGFILE in the config
        o.log_enabled = false; o.seen.log = true;
        o.log_path = "/tmp/x.log";
        resolve_implied_options(o);
        check(!o.log_enabled, "an explicit off is never overridden by a path");
    }
    {
        Options o;                       // BENCHMARK from a config satisfies --algo
        o.benchmark = "BEAM-III";
        check(!o.seen.algo, "precondition: no algo seen yet");
        resolve_implied_options(o);
        check(o.seen.algo, "a benchmark algorithm stands in for --algo, whatever supplied it");
    }

    // The loopback relaxations hold in a config file too.
    {
        const std::string p = write_temp("mxbm_loopback.conf",
            "ALGO = BEAM-III\nPOOL = 127.0.0.1:3416\n");
        Options o; std::string err;
        check(load_flat_config(p, o, err), "a loopback POOL loads without USER");
        check(o.pools.size() == 1 && !o.pools[0].tls, "TLS defaults off on loopback");
        check(o.pools[0].user == std::string(kLoopbackDefaultUser), "default credential");
        std::remove(p.c_str());
    }
    {
        const std::string p = write_temp("mxbm_loopback_tls.conf",
            "ALGO = BEAM-III\nPOOL = 127.0.0.1:3416\nTLS = 1\n");
        Options o; std::string err;
        check(load_flat_config(p, o, err), "loopback POOL with TLS = 1 loads");
        check(o.pools[0].tls, "an explicit TLS still wins");
        std::remove(p.c_str());
    }
    {
        const std::string p = write_temp("mxbm_remote_nouser.conf",
            "ALGO = BEAM-III\nPOOL = beam.2miners.com:5252\n");
        Options o; std::string err;
        check(!load_flat_config(p, o, err), "a remote POOL without USER is still an error");
        std::remove(p.c_str());
    }
    {
        const std::string p = write_temp("mxbm_loopback.json",
            "{\"RIG1\": {\"ALGO\": \"BEAM-III\","
            " \"POOLS\": [{\"POOL\": \"127.0.0.1:3416\"},"
            "             {\"POOL\": \"remote.example.com:5252\", \"USER\": \"addr\"}]}}");
        Options o; std::string err;
        check(load_json_config(p, "RIG1", o, err), "a JSON profile with a loopback entry loads");
        check(o.pools.size() == 2, "two pools");
        check(!o.pools[0].tls && o.pools[0].user == std::string(kLoopbackDefaultUser),
              "the loopback entry gets both defaults");
        check(o.pools[1].tls && o.pools[1].user == "addr", "the remote entry is unchanged");
        std::remove(p.c_str());
    }

    return summary("config");
}
