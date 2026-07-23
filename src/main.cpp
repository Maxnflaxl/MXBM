// mxbm: the runnable miner binary. Wires the Phase A + Phase B pieces
// together -- parse argv -> merge a config file -> connect+login to the pool
// -> (if the Beam oracle is available) solve jobs and submit shares -> track
// stats and report them via the console ticker and the /summary API -> block
// forever reading the stratum socket. Ctrl+C exits; see
// stratum::Client::run()'s doc comment for why there's no return path or
// cleanup here (the reference miner-style).
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <string>

#include "api/http_summary.h"
#include "cli/options.h"
#include "config/config.h"
#include "miner/stats.h"
#include "ui/console.h"
#include "ui/ticker.h"
#include "pow/difficulty.h"
#include "sha256/sha256.h"   // vendored: void sha256(const uint8_t* d, size_t n, uint8_t out[32]);
#include "stratum/client.h"
#include "stratum/messages.h"
#include "miner/engine.h"
#include "version.h"
#ifdef MXBM_HAVE_BEAM_ORACLE
#include "miner/solver_ref.h"
#endif

using namespace mxbm;

#ifdef MXBM_HAVE_BEAM_ORACLE
namespace {

// Strict hex decode: `hex` must be exactly out_len*2 lowercase-hex-digit
// characters. Any other character (including uppercase) or a length
// mismatch fails without touching `out`. Same contract as
// miner::Engine's file-local helper of the same name (each TU keeps its
// own copy); used below to recover the 104-byte solution bytes from a
// stratum::Solution's hex `output` field so its achieved difficulty can be
// computed for the "Found a share" console line and the Stats best-share/
// share-found bookkeeping. Guarded by MXBM_HAVE_BEAM_ORACLE like its only
// call site below, since without the oracle nothing ever solves/submits a
// share to decode in the first place.
bool from_hex_strict(const std::string& hex, uint8_t* out, size_t out_len) {
    if (hex.size() != out_len * 2) return false;
    for (size_t i = 0; i < out_len; ++i) {
        int hi = -1, lo = -1;
        char ch = hex[i * 2];
        if (ch >= '0' && ch <= '9') hi = ch - '0';
        else if (ch >= 'a' && ch <= 'f') hi = ch - 'a' + 10;
        char cl = hex[i * 2 + 1];
        if (cl >= '0' && cl <= '9') lo = cl - '0';
        else if (cl >= 'a' && cl <= 'f') lo = cl - 'a' + 10;
        if (hi < 0 || lo < 0) return false;
        out[i] = (uint8_t)((hi << 4) | lo);
    }
    return true;
}

} // namespace
#endif

int main(int argc, char** argv) {
    cli::Options opts;
    std::string err;
    if (!cli::parse_args(argc, argv, opts, err)) {
        if (opts.version_requested) {
            // The one flag whose false return carries an empty err (see
            // options.h): print the version string ourselves and exit,
            // before any of the rest of main() (signal setup, console
            // init/banner) ever runs.
            std::fputs(("MXBM " + std::string(mxbm::version()) + "\n").c_str(), stdout);
            return 0;
        }
        if (opts.help_requested) {
            std::fputs(err.c_str(), stdout);
            return 0;
        }
        std::fputs(err.c_str(), stderr);
        return 1;
    }

    // Config-file merge (Task 5): fills in only the fields the CLI itself
    // left unseen (cli::Options::Seen) -- CLI-supplied values always win.
    // Precedence when both --json and --config appear on the same command
    // line: --json wins. opts.use_json_config latches true the instant
    // --json is seen and nothing ever clears it, so checking it first here
    // already gives "--json wins" regardless of argv order -- even though
    // the two flags share opts.config_path, so a --config appearing AFTER
    // --json can still overwrite the path text itself (see config/config.h
    // and the Task 5 report for the full nuance around that shared field).
    //
    // Captured before the merge below can flip opts.seen.pools true, so the
    // dropped-CLI-credentials warning further down can tell "the CLI gave
    // no --pool at all" apart from "the config file supplied the pools".
    bool cli_pools = opts.seen.pools;
    if (opts.use_json_config) {
        if (!config::load_json_config(opts.config_path, opts.json_profile, opts, err)) {
            std::fputs((err + "\n").c_str(), stderr);
            return 1;
        }
    } else if (!opts.config_path.empty()) {
        if (!config::load_flat_config(opts.config_path, opts, err)) {
            std::fputs((err + "\n").c_str(), stderr);
            return 1;
        }
    }

    // The CLI no longer requires --pool by itself (a config file may supply
    // POOLS/POOL instead -- see cli::parse_args's doc comment) -- so the
    // "at least one pool" check moved here, after the config-merge above
    // has had its chance to fill opts.pools in.
    if (opts.pools.empty()) {
        std::fputs("missing --pool (or config with POOLS)\n", stderr);
        return 1;
    }

    // OpenSSL's internal write() has no MSG_NOSIGNAL and SO_NOSIGPIPE is
    // Darwin-only: on Linux a pool dropping mid-TLS-write would SIGPIPE-kill
    // the process instead of surfacing a write error to the reconnect loop.
    std::signal(SIGPIPE, SIG_IGN);

    ui::console::init(opts.nocolor);
    ui::console::banner();

    // Out-of-scope-for-this-phase flags still get a console acknowledgment
    // rather than silently doing nothing: failover past pool 1, GPU device
    // selection, and the watchdog all arrive in a later phase.
    if (opts.pools.size() > 1) {
        ui::console::info("Failover pools configured - failover logic arrives in Phase C; using pool 1");
    }
    if (!opts.devices.empty()) {
        ui::console::info("--devices noted - no GPU backend until M3");
    }
    if (opts.watchdog_requested) {
        ui::console::info("--watchdog arrives in Phase C");
    }

    // A config file's own POOLS/POOL entries carry their own USER/PASS/TLS.
    // When the CLI supplied --user/--pass/--tls but no --pool at all, those
    // CLI credentials never bind to anything and the config's own USER wins
    // silently -- surface that instead of leaving it silent (cheap
    // mitigation; see the Task 5 report for the fuller Phase C redesign
    // this stands in for).
    if ((opts.seen.user || opts.seen.pass || opts.seen.tls) && !cli_pools && !opts.pools.empty()) {
        ui::console::info("Note: command-line --user/--pass/--tls were ignored; using pool credentials from the config file.");
    }

    miner::Stats stats;

    ui::console::connecting(opts.pools[0].host, opts.pools[0].port, opts.pools[0].tls);

    stratum::Client client;

    // connect -> if ok: connected(); if not, note it and let client.run()'s
    // own reconnect loop keep retrying below. login() runs unconditionally to
    // store the api_key credential even on initial connection failure, so the
    // reconnect loop has valid credentials for its re-login attempts.
    auto connect_t0 = std::chrono::steady_clock::now();
    bool up = client.connect(opts.pools[0].host, opts.pools[0].port, opts.pools[0].tls);
    auto connect_t1 = std::chrono::steady_clock::now();
    stats.record_connect(
        opts.pools[0].host + ":" + std::to_string(opts.pools[0].port),
        std::chrono::duration_cast<std::chrono::milliseconds>(connect_t1 - connect_t0).count());
    if (up) {
        ui::console::connected(opts.pools[0].tls);
    } else {
        ui::console::error("Initial connection failed — will keep retrying");
    }
    client.login(opts.pools[0].user);   // stores api_key for reconnect re-login even if send fails

#ifdef MXBM_HAVE_BEAM_ORACLE
    static miner::SolverRef solver;
    miner::Engine engine(client, solver);
    engine.submit_fn = [&client, &stats](const stratum::Solution& s) {
        // Achieved difficulty for the "Found a share" line and the best-share
        // stat: decode the 104-byte solution back out of its hex `output`
        // field, SHA-256 it (same predicate Engine's own clears_difficulty()
        // used to decide this was worth submitting), and convert to display
        // units. A decode failure "can't happen" (output is always Engine's
        // own to_hex() of a fresh 104-byte candidate) but is handled
        // defensively: skip the console line and the best-share update, still
        // record the submit and still send it -- the actual submission must
        // never be gated on cosmetic/stat reporting.
        uint8_t soln[104];
        if (from_hex_strict(s.output, soln, sizeof soln)) {
            uint8_t hash[32];
            sha256(soln, sizeof soln, hash);
            double units = pow::achieved_units(hash);
            ui::console::share_found("CPU 0", units);
            stats.record_share_found(units);
        }
        stats.record_submit(s.id);
        client.submit(s);
    };
    engine.on_attempt = [&stats](uint32_t candidates) { stats.record_attempt(candidates); };
#else
    ui::console::info("Built without the Beam oracle — monitoring jobs only (no solving)");
#endif

    client.on_result = [&opts, &stats](const stratum::Result& r) {
        if (r.id == "login") {
            if (r.code == 0) {
                ui::console::authorized(opts.pools[0].user);
                ui::console::start_mining();
            } else {
                ui::console::error("Login failed (" + std::to_string(r.code) + "): " + r.description);
            }
        } else {
            stats.record_result(r.code);
            ui::console::share_result(r.code, r.description, stats.snapshot().last_latency_ms);
        }
    };

    client.on_job = [&](const stratum::Job& j) {
        stats.record_job(j.id, pow::to_display_units(j.difficulty));
        ui::console::job(j.id, j.difficulty, j.height);
#ifdef MXBM_HAVE_BEAM_ORACLE
        engine.on_job(j);
#endif
    };

    client.on_disconnect = [&stats]() {
        ui::console::disconnected();
        stats.record_disconnect();
    };

    ui::Ticker ticker;
    ticker.start(stats, opts.shortstats, opts.longstats);

    api::HttpSummary http_api;
    if (opts.apiport) {
        if (!http_api.start(static_cast<uint16_t>(opts.apiport), stats, mxbm::version())) {
            ui::console::error("API server failed to start on port " + std::to_string(opts.apiport));
        }
        // Either way, mining continues below -- the API is a convenience,
        // never a mining precondition.
    }

#ifdef MXBM_HAVE_BEAM_ORACLE
    engine.start();
#endif

    client.run();   // blocks forever, reconnecting on drop; Ctrl+C exits
    return 0;
}
