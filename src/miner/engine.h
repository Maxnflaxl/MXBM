#pragma once
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

#include "miner/solver.h"
#include "stratum/client.h"
#include "stratum/messages.h"

namespace mxbm { namespace miner {

// Job -> solve -> difficulty filter -> submit pipeline.
//
// Engine has a synchronous core and a thin async wrapper:
//
//   process_job() is the synchronous core, fully unit-testable with no
//   threads: hex-decode job.input, build the 8-byte nonce from the given
//   nonceprefix plus a per-attempt counter, call solver.solve() once, and
//   submit every returned candidate that clears job.difficulty. Malformed
//   job.input (not exactly 64 lowercase hex chars) makes it return early --
//   no crash, no submit. process_job() is pure with respect to Client: it
//   never reads client_ (the caller supplies nonceprefix) and never writes
//   the mailbox (latest_job_/has_job_) -- the only mailbox touch is the
//   read-only staleness check right before submitting, described below.
//
//   on_job()/start()/stop() are the async wrapper used for live mining. The
//   thread-safety contract lives entirely in on_job(): it is meant to be
//   wired to Client::on_job, so it runs on Client::run()'s thread -- the
//   SAME thread that writes Client::nonceprefix_ on every Result line
//   (reconnect re-login rewrites it). That makes on_job() the one safe
//   place to read client_.current_nonceprefix(); it snapshots {job,
//   nonceprefix} together into the mailbox (mutex + condition_variable)
//   and wakes the worker thread. The worker thread later takes that
//   (job, nonceprefix) pair out of the mailbox and passes both to
//   process_job() there, so a multi-minute reference solve never blocks
//   the Client's socket read loop, and process_job() itself never touches
//   client_ from the worker thread -- no cross-thread Client access, so
//   nothing left to race. Right before submitting, process_job() re-checks
//   the mailbox and skips the submit if the mailbox currently holds a
//   not-yet-taken entry (has_job_) for a DIFFERENT job id than the one
//   being processed (stale-share avoidance). With no pending mailbox entry
//   (has_job_ false -- the common case once the worker has taken the job,
//   or a direct process_job() call as the unit tests make) nothing is
//   superseded.
//
// With today's CPU reference solver, solve() cannot be interrupted
// mid-flight: stop() sets a flag and joins, but a solve() already in
// progress must still return on its own first, so stop()/the destructor
// can block for as long as that call takes. M3's abortable (GPU) solver
// closes this gap.
class Engine {
public:
    Engine(stratum::Client& client, Solver& solver);
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // Synchronous pipeline. Reads nothing from client_ -- nonceprefix is
    // supplied by the caller, not read live off the Client. Safe to call
    // directly (unit test) or from the worker thread spawned by start().
    void process_job(const stratum::Job& job, const std::string& nonceprefix);

    // Mailbox write: snapshots job together with client_.current_nonceprefix()
    // and records the pair as the latest one to process, then wakes the
    // worker thread. Must run on Client::run()'s thread -- see the class
    // comment for why that's what makes the current_nonceprefix() read
    // here race-free. Cheap, never blocks on solving -- wire this to
    // stratum::Client::on_job for live mining.
    void on_job(const stratum::Job& job);

    // Spawns the worker thread (loop: wait for a job, take the latest
    // (job, nonceprefix) pair, process_job it). No-op if already started.
    void start();

    // Signals the worker thread to stop and joins it (see class comment for
    // why this can block on an in-flight solve()). No-op if not started.
    void stop();

    // Test seam (mirrors Client::handle_line's role): defaults to
    // client_.submit in the constructor; tests override it to capture
    // submissions instead of driving a real Client/transport.
    std::function<void(const stratum::Solution&)> submit_fn;

    // Stats hook: fired (if set -- empty by default, guarded at the call
    // site) after every solve() return, with that attempt's candidate
    // count, BEFORE the difficulty filter runs -- so it observes every
    // attempt regardless of how many (if any) candidates end up clearing
    // the job's difficulty or getting superseded by a newer job. Wired to
    // Stats::record_attempt in main.cpp; unset in the unit tests that don't
    // care about it (same empty-by-default pattern as submit_fn/on_job
    // callbacks elsewhere in this codebase).
    std::function<void(uint32_t)> on_attempt;

private:
    void worker_main();

    // True if the mailbox currently holds a not-yet-taken entry (has_job_)
    // for a job whose id differs from `job_id` -- i.e. `job_id`'s work is
    // stale. False whenever the mailbox has no pending entry (has_job_
    // false), so a direct process_job() call with nothing ever posted to
    // the mailbox is never considered superseded.
    bool superseded_by_newer_job(const std::string& job_id) const;

    stratum::Client& client_;
    Solver& solver_;

    mutable std::mutex mailbox_mutex_;
    std::condition_variable mailbox_cv_;
    stratum::Job latest_job_{};
    std::string latest_nonceprefix_;   // snapshotted alongside latest_job_ in on_job()
    bool has_job_ = false;
    bool stop_requested_ = false;

    std::thread worker_;
    bool started_ = false;

    // Per-attempt nonce counter. Never reset (not even across job changes),
    // so no (prefix, counter) nonce pair is ever attempted twice.
    uint64_t nonce_counter_ = 0;
};

} } // namespace mxbm::miner
