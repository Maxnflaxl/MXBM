// Stats metrics-core test: a fake clock (the seam miner/stats.h documents) makes
// the window and latency math deterministic. No threads.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>

#include "check.h"
#include "miner/stats.h"

using namespace mxbm;
using namespace mxbm::miner;

namespace {

// Installs a fake now_fn on `s`, offset in integer milliseconds (kept integer so
// the comparisons stay exact) from a real now() captured when this helper runs
// -- AFTER construction, so Stats's own start_ capture is untouched.
void install_fake_clock(Stats& s, long long& fake_ms) {
    auto base = std::chrono::steady_clock::now();
    s.now_fn = [&fake_ms, base]() {
        return base + std::chrono::milliseconds(fake_ms);
    };
}

} // namespace

int main() {
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

        fake_ms = 12000;
        Stats::Snapshot at12 = s.snapshot();
        check(at12.sol15 == 6.0 / 15.0, "t=12s: sol15 == 6/15.0 (all 3 events in window)");

        fake_ms = 30000;
        Stats::Snapshot at30 = s.snapshot();
        check(at30.sol15 == 0.0, "t=30s: sol15 == 0 (all 3 events aged out of 15s window)");
        check(at30.sol60 == 6.0 / 60.0, "t=30s: sol60 == 6/60.0 (events remain in 60s window)");
        check(at30.iter60 == 3.0 / 60.0,
              "t=30s: iter60 == 3/60.0 (counts attempts, not candidates)");
    }

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

    {
        Stats s;
        s.record_result(1);
        Stats::Snapshot snap = s.snapshot();
        check(snap.accepted == 1, "counter still increments with no pending submit");
        check(snap.last_latency_ms == -1, "latency stays -1 with no matching submit to pop");
    }

    {
        Stats s;
        s.record_share_found(5.0);
        s.record_share_found(2.0);
        s.record_share_found(9.0);
        s.record_share_found(3.0);
        Stats::Snapshot snap = s.snapshot();
        check(snap.best_share_units == 9.0, "best_share_units tracks the max, not the latest");
    }

    // -- result-code -> counter mapping: 1=accepted, 3=stale, anything else=rejected --
    {
        Stats s;
        s.record_result(3);   // stale
        s.record_result(2);   // rejected (a real "not accepted, not stale" share-verdict code)
        s.record_result(0);   // defensive only: main filters login results out before they
                               // reach Stats, but the mapping must still be defined for 0.
        s.record_result(1);   // accepted
        Stats::Snapshot snap = s.snapshot();
        check(snap.stale == 1, "code 3 -> stale");
        check(snap.rejected == 2, "any other nonzero/zero code -> rejected");
        check(snap.accepted == 1, "code 1 -> accepted");
    }

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

    // -- pool_sol_session: pool-credited work rate --
    // Regression: this column rendered a hardcoded 0.0 for its entire life. The
    // numerator is the job's TARGET difficulty at submit time, not the share's
    // achieved difficulty, and only ACCEPTED shares pay.
    //
    // The fake clock is installed after construction, so `elapsed` carries a
    // sub-millisecond real offset; the kilosecond-scale timestamps below make
    // that a <1e-6 relative error, hence a tolerance rather than equality.
    {
        Stats s;
        long long fake_ms = 0;
        install_fake_clock(s, fake_ms);

        s.record_job("j1", 2000.0);
        fake_ms = 1000;
        s.record_submit("j1");        // captures the 2000 target in flight
        s.record_job("j2", 6000.0);   // difficulty changes before the result lands
        s.record_result(1);           // must bank 2000 (submit-time), not 6000

        s.record_submit("j2");
        s.record_result(3);           // stale -- pays nothing
        s.record_submit("j2");
        s.record_result(2);           // rejected -- pays nothing

        fake_ms = 4000000;            // 4000 s
        Stats::Snapshot a = s.snapshot();
        check(std::fabs(a.pool_sol_session - 0.5) < 1e-3,
              "pool_sol_session == accepted submit-time difficulty / elapsed (2000/4000)");

        s.record_submit("j2");
        s.record_result(1);           // banks 6000 -> 8000 total
        fake_ms = 8000000;            // 8000 s
        Stats::Snapshot b = s.snapshot();
        check(std::fabs(b.pool_sol_session - 1.0) < 1e-3,
              "a second accept adds its own job's difficulty (8000/8000)");
    }

    // -- Series: Welford against a two-pass reference --
    // The values are the live-session 60 s samples from docs/performance.md,
    // outlier included.
    {
        const double vals[] = { 56.1, 55.9, 58.9, 39.7, 54.2, 57.0, 56.4, 55.1 };
        const size_t n = sizeof vals / sizeof vals[0];
        Stats::Series ser;
        for (size_t i = 0; i < n; ++i) ser.add(vals[i]);

        double sum = 0.0, lo = vals[0], hi = vals[0];
        for (size_t i = 0; i < n; ++i) {
            sum += vals[i];
            if (vals[i] < lo) lo = vals[i];
            if (vals[i] > hi) hi = vals[i];
        }
        const double mean = sum / (double)n;
        double ss = 0.0;
        for (size_t i = 0; i < n; ++i) ss += (vals[i] - mean) * (vals[i] - mean);
        const double sd = std::sqrt(ss / (double)(n - 1));

        check(ser.n == n, "Series counts every sample");
        check(std::fabs(ser.mean - mean) < 1e-9, "Series mean matches a two-pass mean");
        check(std::fabs(ser.stddev() - sd) < 1e-9,
              "Series stddev matches a two-pass sample stddev (n-1)");
        check(ser.min == lo && ser.max == hi, "Series tracks both extremes");
    }

    // -- Series: the counts where a variance formula falls over --
    {
        Stats::Series e;
        check(e.n == 0 && e.stddev() == 0.0, "an unsampled Series reports no spread");
        e.add(42.0);
        check(e.n == 1 && e.mean == 42.0 && e.min == 42.0 && e.max == 42.0,
              "one sample: it is the mean and both extremes");
        check(e.stddev() == 0.0, "one sample: stddev is 0, not a divide by n-1 == 0");
        e.add(42.0);
        check(e.stddev() == 0.0, "two identical samples have exactly zero spread");
    }

    // -- session series: the sampling rules, through the real snapshot() path --
    {
        Stats s;
        long long fake_ms = 0;
        install_fake_clock(s, fake_ms);

        double cur_power = 300.0;
        s.set_telemetry_source([&cur_power](unsigned, Stats::Device& d) {
            d.has_power = true;
            d.power_w = cur_power;
        });

        s.snapshot();                       // t=0: first fold, banks 300
        fake_ms = 500;  cur_power = 999.0;
        Stats::Snapshot limited = s.snapshot();   // <1 s later: must NOT fold
        check(limited.series.power_w.n == 1 && limited.series.power_w.max == 300.0,
              "a second snapshot within the sample interval does not fold again");

        fake_ms = 1000; cur_power = 310.0; s.snapshot();
        fake_ms = 2000; cur_power = 290.0;
        Stats::Snapshot three = s.snapshot();
        check(three.series.power_w.n == 3, "one fold per sample interval, no more");
        check(three.series.power_w.min == 290.0 && three.series.power_w.max == 310.0,
              "telemetry extremes track the session, not the latest reading");
        check(std::fabs(three.series.power_w.mean - 300.0) < 1e-9,
              "telemetry mean averages the folded samples only (the 999 was skipped)");

        // sol15 reads 0.0 for the first 15 s, and sampling that would pin the
        // session minimum at zero forever.
        check(three.series.sol15.n == 0, "sol15 is not sampled before 15 s of uptime");
        check(three.series.sol60.n == 0, "sol60 is not sampled before 60 s of uptime");

        s.record_attempt(30);               // 30 candidates at t=2 s
        fake_ms = 15000;
        Stats::Snapshot at15 = s.snapshot();
        check(at15.series.sol15.n == 1, "sol15 starts being sampled once its window is full");
        check(std::fabs(at15.series.sol15.mean - 2.0) < 1e-9,
              "the sampled sol15 is the rate itself (30 candidates / 15 s)");
        check(at15.series.sol60.n == 0, "sol60 still waits for its own longer window");

        fake_ms = 60000;
        Stats::Snapshot at60 = s.snapshot();
        check(at60.series.sol60.n == 1 && at60.series.iter60.n == 1,
              "the 60 s window starts sampling at 60 s, attempts rate alongside it");
    }

    return summary("stats");
}
