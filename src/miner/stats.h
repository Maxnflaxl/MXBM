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

// Central metrics store for the miner. Producers run on whichever thread
// drives the callback that calls them; consumers (console ticker, /summary
// API) read via snapshot() from their own threads. At a handful of events
// per second, one std::mutex guarding everything is correct and simplest.
//
// Two ledgers, not one: every share-shaped producer takes an Origin
// (miner/origin.h), and the fee pool's shares are counted SEPARATELY from
// the user's throughout, so the table shows the user's own numbers rather
// than a blend (docs/devfee.md). The hashrate is the deliberate exception:
// it measures what the GPU is doing, and the GPU does the same work either
// way, so splitting it would only make the headline speed dip once an hour
// for no reason the user could act on. What the fee costs them is time,
// which is what devfee_seconds reports.
class Stats {
public:
    Stats();

    // -- producers ----------------------------------------------------------

    // Feeds the sol15/sol60/sol_session/iter60 windowed rates. `device` is the
    // index of the card that did the work, so a rig operator can see WHICH card
    // is lagging rather than only that the total dropped.
    void record_attempt(uint32_t candidates, unsigned device = 0);

    // Starts the accept/reject latency clock. record_result() matches
    // oldest-submit-first within an origin, not by job_id, since Beam results
    // echo no submit id. Captures that origin's current job target -- the two
    // pools run independent difficulties -- for record_result() to bank.
    void record_submit(const std::string& job_id, Origin origin = Origin::Main,
                       unsigned device = 0);

    // Achieved difficulty of a newly found share, in the same display units as
    // pow::to_display_units. Tracks Snapshot::best_share_units for Main only:
    // a share found during a fee slice is the developer's. Also appends to the
    // recent-share ring, which holds BOTH origins, tagged.
    void record_share_found(double achieved_units, Origin origin = Origin::Main,
                            unsigned device = 0);

    // Wire result code: 1 = accepted, 3 = stale, any other value = rejected.
    // Pops the oldest pending submit OF THE SAME ORIGIN and records its round
    // trip as Snapshot::last_latency_ms; with none pending the counter still
    // updates but last_latency_ms is left unchanged. Per-origin matching stops
    // the two connections stealing each other's timings.
    void record_result(int code, Origin origin = Origin::Main);

    // `units` is the job's difficulty in display units. Main fills
    // Snapshot::last_job_id/last_job_units; Dev updates only the target a
    // later record_submit(Dev) banks, since the table shows the user's job.
    void record_job(const std::string& id, double units, Origin origin = Origin::Main);

    // -- developer fee ------------------------------------------------------

    // The fee rate as a fraction (0.01 == 1.0%); 0 when none is configured.
    void set_devfee_rate(double rate);

    void set_devfee_active(bool active);

    // One completed fee slice: the fee is charged in time, not in shares.
    void record_devfee_slice(std::chrono::seconds elapsed);

    // "host:port" and the TCP+TLS handshake duration in milliseconds.
    void record_connect(const std::string& hostport, long long connect_ms);

    void record_disconnect();

    // -- consumer -------------------------------------------------------------

    // One found share, as the consumer sees it. Age rather than a timestamp:
    // Stats never reads wall-clock time (see now_fn).
    struct RecentShare {
        double age_s;    // seconds before the snapshot was taken
        double units;    // achieved difficulty, same units as best_share_units
        // The job target this share was found against, captured when it was
        // found; 0 if that pool had not sent a job yet. Pool vardiff moves the
        // target, so dividing by the CURRENT target would restate old shares.
        double target;
        bool dev;        // found during a developer-fee round
    };

    // Running summary of one repeatedly-sampled quantity, in constant memory
    // via Welford's online algorithm -- numerically stable where the textbook
    // sum-of-squares is not. Session-scoped and never reset, so these outlive
    // every consumer's own history (the dashboard's chart ring holds ~30
    // minutes and starts over on reload).
    struct Series {
        uint64_t n = 0;
        double mean = 0.0;
        double min = 0.0, max = 0.0;   // both meaningless while n == 0
        double m2 = 0.0;               // sum of squared deviations from the mean
        void add(double v);
        // SAMPLE standard deviation (n-1 denominator); 0 with fewer than two.
        double stddev() const;
    };

    // One card's own numbers. The rig total is the sum of these where summing
    // means anything -- speeds and counts add, a temperature does not, which is
    // why the Total row leaves the telemetry columns blank rather than
    // averaging them into a figure no card actually reported.
    struct Device {
        std::string label;
        double sol15 = 0.0, sol60 = 0.0, sol_session = 0.0;
        double iter60 = 0.0;
        // Monotonic count of solve() calls this device has completed. The
        // watchdog compares successive samples of THIS rather than of a rate:
        // a rate that reads zero cannot distinguish "hung" from "the window has
        // not filled yet", and a card that finds no candidates for a minute is
        // unlucky, not crashed. A counter that stops advancing is unambiguous.
        uint64_t attempts = 0;
        uint64_t accepted = 0, stale = 0, rejected = 0;   // the USER's pool only
        double best_share_units = 0.0;
        bool has_power = false;     double power_w = 0.0;
        bool has_sm_clock = false;  unsigned sm_clock_mhz = 0;
        bool has_mem_clock = false; unsigned mem_clock_mhz = 0;
        bool has_temp = false;      unsigned temp_c = 0;
        bool has_fan = false;       unsigned fan_pct = 0;
    };

    struct SeriesSet {
        Series sol15, sol60, iter60;
        Series power_w, sm_clock_mhz, mem_clock_mhz, temp_c, fan_pct;
    };

    // Found shares retained: roughly the last half-hour, and a /summary body
    // still cheap to poll every couple of seconds.
    static constexpr size_t kRecentShares = 128;

    struct Snapshot {
        double sol15, sol60, sol_session;   // candidates/sec over windows
        // Pool-side rate: summed target difficulty of accepted shares over
        // session time. Target, not achieved difficulty: achieved is >= target
        // by construction and averages ~2x it, so it would overstate the rate
        // twofold. 0 until the first share is accepted.
        double pool_sol_session = 0.0;
        double iter60;                      // attempts/sec over the 60 s window
        uint64_t accepted, stale, rejected;         // the USER's pool only
        double best_share_units;                    // the USER's best share only
        long long last_latency_ms;          // -1 until the first record_result
        std::string pool;                   // "host:port"
        std::string device_label;           // e.g. "NVIDIA GeForce RTX 4070 Ti SUPER" or "CPU 0 reference"
        // Installed GPU driver version ("610.43.03"), empty when unknown.
        std::string driver_version;
        long long connect_ms;               // TCP+TLS handshake duration
        std::chrono::seconds uptime;
        std::string last_job_id;
        double last_job_units;
        uint64_t reconnects;
        // devfee_rate is the configured fraction (0.01 == 1.0%); the other two
        // are what has actually been spent, to check the rate against uptime.
        double devfee_rate = 0.0;
        double devfee_seconds = 0.0;
        uint64_t devfee_slices = 0;
        bool devfee_active = false;                 // a fee slice is running right now
        uint64_t devfee_accepted = 0, devfee_stale = 0, devfee_rejected = 0;
        // One entry per mining device, in the order set_device_labels() gave.
        // Always at least one, so a consumer never has to special-case an empty
        // rig: a single-GPU run has exactly one entry and the Total row of the
        // console is suppressed as redundant.
        std::vector<Device> devices;
        // Oldest first, at most kRecentShares entries.
        std::vector<RecentShare> recent_shares;
        SeriesSet series;
        // Device telemetry (NVML on NVIDIA). Each has_* is false when that
        // query is unavailable -- laptops often report power but not fan.
        bool has_power = false;  double power_w = 0.0;
        bool has_sm_clock = false;  unsigned sm_clock_mhz = 0;
        bool has_mem_clock = false; unsigned mem_clock_mhz = 0;
        bool has_temp = false;   unsigned temp_c = 0;
        bool has_fan = false;    unsigned fan_pct = 0;
    };
    Snapshot snapshot() const;

    // The target difficulty of `origin`'s current job, in display units, or 0
    // before that pool has sent one.
    double current_job_units(Origin origin = Origin::Main) const;

    // e.g. the GPU device name, or "CPU 0 reference". Set once at startup.
    // The single-device form; equivalent to set_device_labels({label}).
    void set_device_label(std::string label);
    // One label per mining device, in the order --devices selected them. Sets
    // the number of per-device rows every later snapshot reports.
    void set_device_labels(std::vector<std::string> labels);

    // Set once at startup, empty when the platform reports none. Thread-safe.
    void set_driver_version(std::string v);

    // Called once per device while building a snapshot; null on platforms with
    // no telemetry. The index is the device's own, so the callback knows which
    // card to sample.
    using TelemetryFn = std::function<void(unsigned index, Device&)>;
    void set_telemetry_source(TelemetryFn fn);

    // Test seam: defaults to steady_clock::now in the constructor; tests swap
    // in a fake clock so window/latency/uptime math is deterministic. ALL time
    // reads inside Stats go through this. start_ is captured at construction,
    // so install the fake clock before anything that must be timed from it.
    std::function<std::chrono::steady_clock::time_point()> now_fn;

private:
    double accepted_units_ = 0.0;   // summed target difficulty of accepted shares
    TelemetryFn telemetry_;
    struct Event {
        std::chrono::steady_clock::time_point t;
        uint32_t candidates;
        unsigned device;
    };
    struct PendingSubmit {
        std::string job_id;
        std::chrono::steady_clock::time_point t;
        double units;      // the job's target difficulty when this was submitted
        Origin origin;     // which connection it went to; matched by record_result
        unsigned device;   // which card found it, so the verdict lands on its row
    };

    // Called on every insert. snapshot() computes the 15 s/60 s windows by
    // filtering on age, so this only bounds memory; it is not the windowing.
    void prune_attempts(std::chrono::steady_clock::time_point now);

    mutable std::mutex mutex_;

    // Captured via now_fn() in the constructor; backs uptime and sol_session.
    std::chrono::steady_clock::time_point start_;

    std::deque<Event> attempts_;      // pruned to 15 min on insert
    uint64_t total_candidates_ = 0;   // never pruned -- backs sol_session

    std::deque<PendingSubmit> submits_;   // cap 64 -- drop oldest beyond
    long long last_latency_ms_ = -1;

    double best_share_units_ = 0.0;
    uint64_t accepted_ = 0, stale_ = 0, rejected_ = 0;

    // Per-device ledgers, sized by set_device_labels(). The rig totals above
    // are kept independently rather than summed on read: they must stay right
    // even for a result that arrives with no pending submit to attribute it to.
    struct DevLedger {
        std::string label;
        uint64_t total_candidates = 0;
        uint64_t attempts = 0;
        uint64_t accepted = 0, stale = 0, rejected = 0;
        double best_share_units = 0.0;
    };
    std::vector<DevLedger> devices_;

    std::string pool_;
    long long connect_ms_ = 0;
    uint64_t reconnects_ = 0;

    std::string last_job_id_;
    double last_job_units_ = 0.0;

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

    // Folded inside snapshot() -- see the sampling rules there. `mutable`
    // because snapshot() is const; written only under mutex_.
    mutable SeriesSet series_;
    mutable std::chrono::steady_clock::time_point last_series_sample_;
    mutable bool series_sampled_ = false;   // false until the first fold
};

} } // namespace mxbm::miner
