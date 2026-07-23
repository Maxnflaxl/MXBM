// Stats metrics-core test: fake clock for deterministic window/latency math
// (same documented-seam pattern as Client::handle_line / Engine::submit_fn;
// see miner/stats.h). No threads -- record_*()/snapshot() are called
// directly.
#include <chrono>
#include <cstdint>
#include <string>

#include "check.h"
#include "miner/stats.h"

using namespace mxbm;
using namespace mxbm::miner;

namespace {

// Installs a fake now_fn on `s`, driven by an integer-millisecond offset
// from a real steady_clock::now() captured when this helper runs (AFTER
// Stats's own construction -- so it never touches Stats's construction-time
// start_ capture; see the "uptime sanity" checks below, which only assert
// non-negativity for exactly that reason). `fake_ms` is owned by the
// caller and mutated directly to move the fake clock; deliberately integer
// milliseconds rather than fractional seconds so latency/window comparisons
// below are exact -- no double-rounding on the time axis.
void install_fake_clock(Stats& s, long long& fake_ms) {
    auto base = std::chrono::steady_clock::now();
    s.now_fn = [&fake_ms, base]() {
        return base + std::chrono::milliseconds(fake_ms);
    };
}

} // namespace

int main() {
    // -- default-constructed snapshot: all-zero/empty baseline --
    {
        Stats s;
        Stats::Snapshot snap = s.snapshot();
        check(snap.sol15 == 0.0, "default: sol15 == 0");
        check(snap.sol60 == 0.0, "default: sol60 == 0");
        check(snap.sol_session == 0.0, "default: sol_session == 0");
        check(snap.iter60 == 0.0, "default: iter60 == 0");
        check(snap.accepted == 0 && snap.stale == 0 && snap.rejected == 0,
              "default: accepted/stale/rejected all 0");
        check(snap.best_share_units == 0.0, "default: best_share_units == 0");
        check(snap.last_latency_ms == -1, "default: last_latency_ms == -1 (no result yet)");
        check(snap.pool.empty(), "default: pool empty");
        check(snap.connect_ms == 0, "default: connect_ms == 0");
        check(snap.uptime.count() >= 0, "default: uptime non-negative");
        check(snap.last_job_id.empty(), "default: last_job_id empty");
        check(snap.last_job_units == 0.0, "default: last_job_units == 0");
        check(snap.reconnects == 0, "default: reconnects == 0");
    }

    // -- window math: 3 attempts of 2 candidates each at t=0,5,10s --
    {
        Stats s;
        long long fake_ms = 0;
        install_fake_clock(s, fake_ms);

        fake_ms = 0;     s.record_attempt(2);
        fake_ms = 5000;  s.record_attempt(2);
        fake_ms = 10000; s.record_attempt(2);

        // t=12s: all three events are within the last 15s (ages 12,7,2s).
        fake_ms = 12000;
        Stats::Snapshot at12 = s.snapshot();
        check(at12.sol15 == 6.0 / 15.0, "t=12s: sol15 == 6/15.0 (all 3 events in window)");

        // t=30s: the t=0,5,10s events (ages 30,25,20s) have all aged out of
        // the 15s window, but remain within the 60s window.
        fake_ms = 30000;
        Stats::Snapshot at30 = s.snapshot();
        check(at30.sol15 == 0.0, "t=30s: sol15 == 0 (all 3 events aged out of 15s window)");
        check(at30.sol60 == 6.0 / 60.0, "t=30s: sol60 == 6/60.0 (events remain in 60s window)");
        check(at30.iter60 == 3.0 / 60.0,
              "t=30s: iter60 == 3/60.0 (counts attempts, not candidates)");
    }

    // -- latency + accepted: submit at t=1s, accepted result at t=1.012s --
    {
        Stats s;
        long long fake_ms = 0;
        install_fake_clock(s, fake_ms);

        fake_ms = 1000; s.record_submit("job-1");
        fake_ms = 1012; s.record_result(1);   // code 1 == accepted

        Stats::Snapshot snap = s.snapshot();
        check(snap.last_latency_ms == 12, "submit-to-result latency == 12ms");
        check(snap.accepted == 1, "code 1 increments accepted");
        check(snap.stale == 0 && snap.rejected == 0, "code 1 does not touch stale/rejected");
    }

    // -- record_result with no pending submit: no crash, latency untouched --
    {
        Stats s;
        s.record_result(1);   // nothing was ever record_submit()'d
        Stats::Snapshot snap = s.snapshot();
        check(snap.accepted == 1, "counter still increments with no pending submit");
        check(snap.last_latency_ms == -1, "latency stays -1 with no matching submit to pop");
    }

    // -- best share: max across multiple record_share_found calls --
    {
        Stats s;
        s.record_share_found(5.0);
        s.record_share_found(2.0);
        s.record_share_found(9.0);
        s.record_share_found(3.0);
        Stats::Snapshot snap = s.snapshot();
        check(snap.best_share_units == 9.0, "best_share_units tracks the max, not the latest");
    }

    // -- result-code -> counter mapping (per task context): 1=accepted,
    // 3=stale, any other code (including the never-in-practice 0)=rejected --
    {
        Stats s;
        s.record_result(3);   // stale
        s.record_result(2);   // rejected (a real "not accepted, not stale" share-verdict code)
        s.record_result(0);   // rejected -- defensive only: wire code 0 (id "login") never
                               // actually reaches record_result (main filters login results
                               // before forwarding to Stats), but the mapping must still be
                               // well-defined and not UB for it.
        s.record_result(1);   // accepted
        Stats::Snapshot snap = s.snapshot();
        check(snap.stale == 1, "code 3 -> stale");
        check(snap.rejected == 2, "any other nonzero/zero code -> rejected");
        check(snap.accepted == 1, "code 1 -> accepted");
    }

    // -- record_job / record_connect / record_disconnect: plain field wiring --
    {
        Stats s;
        s.record_job("51550", 512.0);
        s.record_connect("de.beam.herominers.com:1130", 11);
        s.record_disconnect();
        s.record_disconnect();
        Stats::Snapshot snap = s.snapshot();
        check(snap.last_job_id == "51550", "record_job sets last_job_id");
        check(snap.last_job_units == 512.0, "record_job sets last_job_units");
        check(snap.pool == "de.beam.herominers.com:1130", "record_connect sets pool");
        check(snap.connect_ms == 11, "record_connect sets connect_ms");
        check(snap.reconnects == 2, "record_disconnect increments reconnects");
    }

    return summary("stats");
}
