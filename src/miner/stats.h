#pragma once
#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include "miner/origin.h"

namespace mxbm { namespace miner {

// Central metrics store for the miner: one instance owned by main and
// passed by reference into the wrapper callbacks that main installs on
// Engine/Client (Engine::on_attempt/submit_fn, Client::on_result/on_job/
// on_disconnect -- that wiring lives in main.cpp, not here). Producers
// below run on whichever thread drives the callback that calls them (the
// worker thread for record_attempt/record_submit/record_share_found, and
// Client::run()'s thread for record_result/record_job/record_connect/
// record_disconnect); consumers (console ticker, /summary API) read via
// snapshot() from their own thread(s). Update rate is a handful of
// events/sec at most, so one std::mutex guarding everything is simplest
// and correct -- no atomics needed.
//
// Two ledgers, not one. Every share-shaped producer takes an Origin
// (miner/origin.h) naming which pool the share belongs to, and the
// developer-fee pool's shares are counted SEPARATELY from the user's
// throughout: they never touch accepted_/stale_/rejected_, never bank into
// accepted_units_ (so pool_sol_session stays the rate the user's own pool
// credited them), and never move best_share_units_. A user reading the
// stats table must see their own numbers there, not a blend. The fee's own
// totals -- shares, seconds spent, slices run -- are reported alongside on
// their own row rather than hidden; see docs/devfee.md.
//
// The one deliberately SHARED figure is the hashrate (record_attempt and
// the sol15/sol60/sol_session/iter60 windows it feeds). That measures what
// the GPU is doing, and the GPU does the same work either way; splitting it
// would make the headline speed dip once an hour for no reason the user
// could act on. What the fee actually costs them is time, which is exactly
// what devfee_seconds reports.
class Stats {
public:
    Stats();

    // -- producers ----------------------------------------------------------

    // Called once per Engine::process_job() solve() return (wire this to
    // Engine::on_attempt), with that attempt's candidate count. Feeds the
    // sol15/sol60/sol_session/iter60 windowed-rate fields in Snapshot.
    void record_attempt(uint32_t candidates);

    // Called when a share is submitted (wire this into Engine::submit_fn,
    // alongside the existing console line), to start the accept/reject
    // latency clock. job_id is accepted for documentation/future use;
    // matching against record_result() is oldest-submit-first via a small
    // internal deque (see class comment below), not by id -- Beam results
    // don't echo a submit id beyond the job id, and oldest-first is what
    // the reference miner's own latency numbers reflect at our submit rates.
    // The submitted share's credited value is the CURRENT job's target difficulty
    // FOR THAT ORIGIN (the two pools run independent difficulties and a fee
    // slice does not disturb the user's), captured here so record_result()
    // can bank it when the pool accepts.
    void record_submit(const std::string& job_id, Origin origin = Origin::Main);

    // Called with the achieved difficulty of a newly found share, in the
    // same display units as pow::to_display_units/pow::achieved_units.
    // Tracks the session maximum (Snapshot::best_share_units) -- for Main
    // only: a lucky share found during a fee slice is the developer's, and
    // showing it as the user's session best would be a plain lie.
    //
    // Also appends to a small ring of RECENT SHARES (Snapshot::recent_shares,
    // capped at kRecentShares). Every other share statistic here is an
    // aggregate -- a count, a maximum, a rate -- which cannot answer "what
    // did the last twenty shares actually look like". The ring is what backs
    // the /summary API's per-share list and the dashboard's difficulty
    // scatter; it holds BOTH origins, tagged, so a fee round's shares are
    // distinguishable rather than silently mixed in.
    void record_share_found(double achieved_units, Origin origin = Origin::Main);

    // Called with a Result's wire result code: 1 = accepted, 3 = stale, any
    // other value = rejected. Never UB regardless of code (defensive: code
    // 0 with id "login" is filtered out by main before it would otherwise
    // reach here, but this function does not assume that). Also pops the
    // oldest pending record_submit() entry OF THE SAME ORIGIN, if any, and
    // records the elapsed time as Snapshot::last_latency_ms; if no such
    // submit is pending, the counter still updates but last_latency_ms is
    // left unchanged. Matching within an origin rather than across the
    // whole queue is what keeps the two connections' round trips from
    // stealing each other's timings when a fee slice straddles a submit.
    void record_result(int code, Origin origin = Origin::Main);

    // Called when a new job arrives, with its id and its difficulty in
    // display units. Main fills Snapshot::last_job_id/last_job_units; Dev
    // updates only the target difficulty that a later record_submit(Dev)
    // will bank, since the table shows the user's job, not the fee's.
    void record_job(const std::string& id, double units, Origin origin = Origin::Main);

    // -- developer fee ------------------------------------------------------

    // The configured fee rate as a fraction (0.01 == 1.0%), set once at
    // startup so the stats table and /summary can state it without
    // reaching into miner::DevFee. 0 in a build with no fee configured.
    void set_devfee_rate(double rate);

    // Flipped by DevFee as each slice begins and ends, so the console and
    // the API can show that a fee round is running right now rather than
    // leaving the user to wonder why the pool changed.
    void set_devfee_active(bool active);

    // Called once per completed fee slice with its measured duration.
    // Feeds Snapshot::devfee_seconds/devfee_slices -- the honest cost
    // figure, since the fee is charged in time, not in shares.
    void record_devfee_slice(std::chrono::seconds elapsed);

    // Called once a connection completes, with "host:port" and the
    // TCP+TLS handshake duration in milliseconds.
    void record_connect(const std::string& hostport, long long connect_ms);

    // Called on every reconnect (i.e. every recv-failure-driven redial;
    // wire this to Client::on_disconnect). Increments Snapshot::reconnects.
    void record_disconnect();

    // -- consumer -------------------------------------------------------------

    // One found share, as the consumer sees it. Age rather than a timestamp
    // on purpose: Stats never reads wall-clock time (see now_fn), so it has
    // no absolute instant to report -- but "3.4 seconds before this snapshot"
    // is exact, and a caller with a clock can place it however it likes.
    struct RecentShare {
        double age_s;    // seconds before the snapshot was taken
        double units;    // achieved difficulty, same units as best_share_units
        // The job target this share was found against, captured at the moment
        // it was found; 0 if that pool had not sent a job yet. Stored per
        // share rather than left to the consumer to pair up, because pool
        // vardiff moves the target through a session: dividing an old share by
        // the CURRENT target silently restates it every time the pool steps.
        double target;
        bool dev;        // found during a developer-fee round
    };

    // Running summary of one repeatedly-sampled quantity: how many samples,
    // their mean and standard deviation, and the extremes seen. Constant
    // memory -- no sample is retained -- via Welford's online algorithm, which
    // stays numerically stable where the textbook sum-of-squares does not (that
    // one subtracts two large nearly-equal numbers, and a long session reaches
    // sample counts where the cancellation eats most of the precision).
    //
    // SESSION-SCOPED AND NEVER RESET, which is the whole point: these outlive
    // every consumer's own history. The dashboard's chart ring holds ~30
    // minutes and starts over on reload, so the range it can draw is only ever
    // the range it still happens to be holding -- a thermal spike or a clock
    // dip from an hour ago is simply gone. These are the whole run's.
    struct Series {
        uint64_t n = 0;
        double mean = 0.0;
        double min = 0.0, max = 0.0;   // both meaningless while n == 0
        double m2 = 0.0;               // sum of squared deviations from the mean
        void add(double v);
        // SAMPLE standard deviation (n-1 denominator), matching the +/-1 sigma
        // figures docs/performance.md quotes. 0 with fewer than two samples.
        double stddev() const;
    };

    // Everything summarised over the session. Field names mirror the Snapshot
    // fields each one summarises.
    struct SeriesSet {
        Series sol15, sol60, iter60;
        Series power_w, sm_clock_mhz, mem_clock_mhz, temp_c, fan_pct;
    };

    // How many found shares are retained. At a typical few shares a minute
    // this is roughly the last half-hour -- enough to see the spread and any
    // vardiff step, while keeping the /summary body small enough that polling
    // it every couple of seconds stays cheap.
    static constexpr size_t kRecentShares = 128;

    struct Snapshot {
        double sol15, sol60, sol_session;   // candidates/sec over windows
        // Pool-side rate: the work the POOL has credited, i.e. the summed target
        // difficulty of accepted shares over session time. Uses the job's target
        // rather than each share's achieved difficulty -- achieved is >= target by
        // construction and averages ~2x it, so summing achieved would overstate the
        // rate by about a factor of two. 0 until the first share is accepted.
        double pool_sol_session = 0.0;
        double iter60;                      // attempts/sec over the 60 s window
        uint64_t accepted, stale, rejected;         // the USER's pool only
        double best_share_units;                    // the USER's best share only
        long long last_latency_ms;          // -1 until the first record_result
        std::string pool;                   // "host:port"
        std::string device_label;           // e.g. "NVIDIA GeForce RTX 4070 Ti SUPER" or "CPU 0 reference"
        // Installed GPU driver version ("610.43.03"), empty when unknown. Set
        // once at startup; reported on the stats block's identity line and in
        // the /summary API, because a driver update is the likeliest
        // explanation for a hashrate that moved with no change to the miner.
        std::string driver_version;
        long long connect_ms;               // TCP+TLS handshake duration
        std::chrono::seconds uptime;
        std::string last_job_id;
        double last_job_units;
        uint64_t reconnects;
        // Developer fee, reported rather than folded away. devfee_rate is the
        // configured fraction (0.01 == 1.0%); devfee_seconds/devfee_slices are
        // what has actually been spent so far this session, which is the number
        // that lets a user check the rate for themselves against uptime.
        double devfee_rate = 0.0;
        double devfee_seconds = 0.0;
        uint64_t devfee_slices = 0;
        bool devfee_active = false;                 // a fee slice is running right now
        uint64_t devfee_accepted = 0, devfee_stale = 0, devfee_rejected = 0;
        // Oldest first, at most kRecentShares entries.
        std::vector<RecentShare> recent_shares;
        // Session-long summary of the sampled quantities above -- see Series.
        SeriesSet series;
        // Device telemetry, filled by whatever the platform can supply (NVML on
        // NVIDIA). Each has_* is false when that particular query is unavailable --
        // laptops commonly report power but not fan -- so the table can show the
        // fields that exist rather than all-or-nothing.
        bool has_power = false;  double power_w = 0.0;
        bool has_sm_clock = false;  unsigned sm_clock_mhz = 0;
        bool has_mem_clock = false; unsigned mem_clock_mhz = 0;
        bool has_temp = false;   unsigned temp_c = 0;
        bool has_fan = false;    unsigned fan_pct = 0;
    };
    Snapshot snapshot() const;

    // The target difficulty of `origin`'s current job, in display units, or 0
    // before that pool has sent one. A one-field read, separate from
    // snapshot(), because the share path only wants this one number and
    // snapshot() now copies the whole recent-share ring to hand it over.
    double current_job_units(Origin origin = Origin::Main) const;

    // Sets the miner/worker display name shown in the stats table and the
    // /summary API (e.g. the GPU device name, or "CPU 0 reference"). Set once
    // at startup after the solver backend is chosen; thread-safe. Defaults to
    // "GPU 0" since the GPU solver is the default backend.
    void set_device_label(std::string label);

    // Sets the GPU driver version shown alongside the miner version. Set once
    // at startup, from whatever the platform can report (NVML on NVIDIA);
    // left empty when there is none. Thread-safe, like set_device_label.
    void set_driver_version(std::string v);

    // Installed once at startup by whichever backend can supply telemetry; called
    // while building a snapshot. Left null on platforms with none.
    using TelemetryFn = std::function<void(Snapshot&)>;
    void set_telemetry_source(TelemetryFn fn);

    // Test seam (same documented-seam pattern as Client::handle_line and
    // Engine::submit_fn): defaults to std::chrono::steady_clock::now in the
    // constructor; tests override it with a fake clock so window/latency/
    // uptime math is deterministic. ALL time reads inside Stats -- attempt
    // timestamps, submit timestamps, and the session-start capture used for
    // uptime/sol_session -- go through this, never steady_clock::now()
    // directly. Because start_ is captured at construction time (see class
    // comment on start_ below), a test that wants deterministic uptime/
    // sol_session values must install its fake now_fn before doing
    // anything else it wants timed relative to session start.
    std::function<std::chrono::steady_clock::time_point()> now_fn;

private:
    double accepted_units_ = 0.0;   // summed target difficulty of accepted shares
    TelemetryFn telemetry_;
    struct Event {
        std::chrono::steady_clock::time_point t;
        uint32_t candidates;
    };
    struct PendingSubmit {
        std::string job_id;
        std::chrono::steady_clock::time_point t;
        double units;      // the job's target difficulty when this was submitted
        Origin origin;     // which connection it went to; matched by record_result
    };

    // Drops attempts_ entries older than 15 minutes relative to `now`.
    // Called on every record_attempt() insert. The much shorter 15 s/60 s
    // windows snapshot() itself sums over are computed by filtering
    // attempts_ by age directly (not by relying on the deque already being
    // trimmed to exactly that size), so this prune only bounds memory/CPU
    // for a long-running session -- it does not implement the windowing.
    void prune_attempts(std::chrono::steady_clock::time_point now);

    mutable std::mutex mutex_;

    // Session start, captured via now_fn() in the constructor -- backs
    // Snapshot::uptime and Snapshot::sol_session.
    std::chrono::steady_clock::time_point start_;

    std::deque<Event> attempts_;      // ring of attempt events, pruned to 15 min on insert
    uint64_t total_candidates_ = 0;   // never pruned -- total across the whole session, backs sol_session

    std::deque<PendingSubmit> submits_;   // cap 64 -- drop oldest beyond
    long long last_latency_ms_ = -1;

    double best_share_units_ = 0.0;
    uint64_t accepted_ = 0, stale_ = 0, rejected_ = 0;

    std::string pool_;
    long long connect_ms_ = 0;
    uint64_t reconnects_ = 0;

    std::string last_job_id_;
    double last_job_units_ = 0.0;
    std::string device_label_ = "GPU 0";
    std::string driver_version_;

    // Developer-fee ledger, kept strictly apart from the counters above.
    double devfee_rate_ = 0.0;
    double devfee_seconds_ = 0.0;
    uint64_t devfee_slices_ = 0;
    bool devfee_active_ = false;
    double devfee_last_job_units_ = 0.0;   // the fee pool's own target difficulty
    uint64_t devfee_accepted_ = 0, devfee_stale_ = 0, devfee_rejected_ = 0;

    struct FoundShare {
        std::chrono::steady_clock::time_point t;
        double units;
        double target;   // that origin's job target when this share was found
        bool dev;
    };
    std::deque<FoundShare> found_;   // oldest first, capped at kRecentShares

    // Session-long summaries, folded from inside snapshot() -- see the comment
    // there for the sampling rules. `mutable` because snapshot() is const and
    // stays that way: it is a read for every caller, and the accumulation is
    // bookkeeping about the reads themselves. Both are written only under
    // mutex_, like every other member here.
    mutable SeriesSet series_;
    mutable std::chrono::steady_clock::time_point last_series_sample_;
    mutable bool series_sampled_ = false;   // false until the first fold
};

} } // namespace mxbm::miner
