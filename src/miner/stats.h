#pragma once
#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>

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
    // The submitted share's credited value is the CURRENT job's target difficulty,
    // captured here so record_result() can bank it when the pool accepts.
    void record_submit(const std::string& job_id);

    // Called with the achieved difficulty of a newly found share, in the
    // same display units as pow::to_display_units/pow::achieved_units.
    // Tracks the session maximum (Snapshot::best_share_units).
    void record_share_found(double achieved_units);

    // Called with a Result's wire result code: 1 = accepted, 3 = stale, any
    // other value = rejected. Never UB regardless of code (defensive: code
    // 0 with id "login" is filtered out by main before it would otherwise
    // reach here, but this function does not assume that). Also pops the
    // oldest pending record_submit() entry, if any, and records the
    // elapsed time as Snapshot::last_latency_ms; if no submit is currently
    // pending, the counter still updates but last_latency_ms is left
    // unchanged.
    void record_result(int code);

    // Called when a new job arrives, with its id and its difficulty in
    // display units (Snapshot::last_job_id/last_job_units).
    void record_job(const std::string& id, double units);

    // Called once a connection completes, with "host:port" and the
    // TCP+TLS handshake duration in milliseconds.
    void record_connect(const std::string& hostport, long long connect_ms);

    // Called on every reconnect (i.e. every recv-failure-driven redial;
    // wire this to Client::on_disconnect). Increments Snapshot::reconnects.
    void record_disconnect();

    // -- consumer -------------------------------------------------------------

    struct Snapshot {
        double sol15, sol60, sol_session;   // candidates/sec over windows
        // Pool-side rate: the work the POOL has credited, i.e. the summed target
        // difficulty of accepted shares over session time. Uses the job's target
        // rather than each share's achieved difficulty -- achieved is >= target by
        // construction and averages ~2x it, so summing achieved would overstate the
        // rate by about a factor of two. 0 until the first share is accepted.
        double pool_sol_session = 0.0;
        double iter60;                      // attempts/sec over the 60 s window
        uint64_t accepted, stale, rejected;
        double best_share_units;
        long long last_latency_ms;          // -1 until the first record_result
        std::string pool;                   // "host:port"
        std::string device_label;           // e.g. "NVIDIA GeForce RTX 4070 Ti SUPER" or "CPU 0 reference"
        long long connect_ms;               // TCP+TLS handshake duration
        std::chrono::seconds uptime;
        std::string last_job_id;
        double last_job_units;
        uint64_t reconnects;
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

    // Sets the miner/worker display name shown in the stats table and the
    // /summary API (e.g. the GPU device name, or "CPU 0 reference"). Set once
    // at startup after the solver backend is chosen; thread-safe. Defaults to
    // "GPU 0" since the GPU solver is the default backend.
    void set_device_label(std::string label);

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
};

} } // namespace mxbm::miner
