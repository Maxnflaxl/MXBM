#pragma once
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

#include "miner/origin.h"
#include "miner/stats.h"
#include "stratum/client.h"
#include "stratum/messages.h"

namespace mxbm { namespace miner {

// The developer fee: MXBM spends a small, fixed fraction of mining time
// solving for the developer's wallet instead of the user's.
//
// This file is the whole mechanism, and it is deliberately readable. The fee
// is reported rather than buried: announced at startup, announced again as
// each round starts and ends, reported on its own row of the statistics table
// and in /summary, and specified in docs/devfee.md. See that file for the
// user-facing terms, and docs/architecture.md for where this sits in the
// miner.
//
// Shape of the mechanism (the same time-slicing the closed-source miners
// use): a second stratum connection to the
// fee pool is held open alongside the user's, and once the accrued fee debt
// reaches one slice the JobRouter below switches the solver over to the fee
// pool's job for that slice, then switches it straight back. There is only
// one GPU and one Engine; only the job it is fed changes.

// Where fee slices mine to.
struct DevFeePool {
    std::string host;
    uint16_t port = 0;
    bool tls = true;
    std::string user;   // the developer's Beam address
    std::string pass;
};

// How much, and how often.
struct DevFeeSchedule {
    double rate = 0.0;                        // fraction of mining time, 0.01 == 1.0%
    std::chrono::seconds cycle{3600};         // the window `rate` is expressed over

    // Length of one fee round: rate * cycle, e.g. 1% of an hour == 36 s.
    // The fee is charged in whole slices of this length rather than
    // continuously, because switching pools costs a job change either way
    // and doing it once an hour is cheaper than doing it constantly.
    std::chrono::seconds slice() const;
};

// The fee MXBM ships with, and the pool it goes to. Defined in devfee.cpp
// next to the address constant.
DevFeePool devfee_pool();
DevFeeSchedule devfee_schedule();

// False when this build has no developer address compiled in (the constant
// in devfee.cpp is empty). The fee is then genuinely inert: no second
// connection is opened and no slice ever runs, rather than mining to an
// address that does not exist. Checked by main.cpp before any fee wiring.
bool devfee_configured();

// Builds the credential the fee connection logs in with:
//
//     <developer address>.<user's worker name>_<fee percent>
//
// e.g. "<devaddr>.rig1_1" for a user mining as "<useraddr>.rig1" at the
// built-in 1% rate, or "<devaddr>.rig1_2.5" for one who raised it with
// --dev-fee 2.5. Naming the round after the user's own worker means a fee
// round shows up on the fee pool's dashboard as identifiably theirs rather
// than as anonymous hashrate, and carrying the rate means a raised fee is
// visible as such.
//
// `user_credential` is the user's own "address[.worker]" exactly as they
// configured it; only the worker part is used, and it is sanitised to
// [A-Za-z0-9_-] and capped at 32 characters -- both because pools reject
// exotic worker names and because the credential goes onto the wire inside
// a JSON login line, where an unescaped quote or newline would corrupt it.
// A user mining with no worker suffix, or one that sanitises away to
// nothing, gets "mxbm".
std::string devfee_login(const std::string& dev_address,
                         const std::string& user_credential,
                         double rate);

// Accrual clock: turns time spent mining into fee debt, in seconds owed.
//
// Pure and thread-free so the arithmetic can be tested without waiting an
// hour (tests/test_devfee.cpp). Debt accrues against time ACTUALLY spent
// mining -- the caller only feeds it ticks during which the user's pool had
// a job to work on -- so a session that spends twenty minutes waiting for a
// dead pool is not charged for those twenty minutes.
//
// Debt carries across slices instead of being reset: settling subtracts
// only what was really spent, so a round cut short by a disconnect leaves
// the remainder owed rather than silently forgiven or silently doubled.
class FeeAccrual {
public:
    explicit FeeAccrual(DevFeeSchedule schedule);

    void add_mining_time(std::chrono::seconds dt);
    bool slice_due() const;                    // debt has reached a full slice
    std::chrono::seconds slice_len() const;
    void settle(std::chrono::seconds spent);   // floors at zero; never goes negative
    double debt_seconds() const;

private:
    DevFeeSchedule schedule_;
    double debt_s_ = 0.0;
};

// Decides which pool's job the solver is working on.
//
// Holds the latest job from each connection and dispatches whichever one
// belongs to the currently active origin. Pure of sockets and threads (the
// dispatch target is injected), so the switching logic -- which is where
// the bugs in a mechanism like this actually live -- is unit-testable.
//
// Both connections keep feeding it the whole time; only one of them is
// dispatched onward at any moment. That is what makes the switch in either
// direction instant and lossless: switching back to the user's pool
// re-dispatches the job that arrived while the fee round was running,
// rather than idling until that pool happens to send the next one.
class JobRouter {
public:
    // Called with (job, nonceprefix, origin) whenever the solver should
    // change what it is working on. Wired to Engine::on_job in main.cpp.
    using DispatchFn = std::function<void(const stratum::Job&, const std::string&, Origin)>;

    explicit JobRouter(DispatchFn dispatch);

    // Record `origin`'s newest job. Dispatched onward only if that origin
    // is the active one; otherwise it is cached, ready for the switch.
    void offer(const stratum::Job& job, const std::string& nonceprefix, Origin origin);

    // Switch the solver to `origin`, re-dispatching that origin's latest
    // cached job if it has one. No-op if already active.
    void set_active(Origin origin);

    Origin active() const;
    bool has(Origin origin) const;   // this origin has sent at least one job

private:
    struct Slot {
        stratum::Job job{};
        std::string nonceprefix;
        bool present = false;
    };
    Slot& slot(Origin origin);
    const Slot& slot(Origin origin) const;

    DispatchFn dispatch_;
    mutable std::mutex mutex_;   // guards both slots AND active_, and is held
                                 // across dispatch_ so a job arriving mid-switch
                                 // cannot leave the solver on the older one
    Slot main_, dev_;
    Origin active_ = Origin::Main;
};

// The fee's network side and its clock: owns the second stratum connection,
// accrues debt while the user is mining, and drives JobRouter across each
// round.
//
// THREADING / LIFETIME. start() spawns two threads: the fee Client's own
// run() loop, and the scheduler. stratum::Client::run() never returns by
// design (see client.h), so the network thread cannot be joined -- stop()
// detaches it. A DevFee that has been start()ed must therefore live until
// the process exits, which is exactly how main.cpp holds it. Do not
// start() one with a shorter lifetime; the tests exercise JobRouter and
// FeeAccrual directly and never call start() for this reason.
class DevFee {
public:
    DevFee(JobRouter& router, Stats& stats, DevFeePool pool, DevFeeSchedule schedule);
    ~DevFee();

    DevFee(const DevFee&) = delete;
    DevFee& operator=(const DevFee&) = delete;

    // Connects and logs in the fee connection, then starts accruing.
    void start();

    // Signals both threads to finish; see the lifetime note above.
    void stop();

    // Sends a solution on the fee connection. main.cpp's submit_fn routes
    // here for Origin::Dev work -- never by "is a round running now", but
    // by the Origin the solved job itself carried, so a round ending
    // mid-solve still submits to the pool that issued the job.
    void submit(const stratum::Solution& solution);

    // Console hooks, empty by default and guarded at the call site --
    // mxbm_ui links mxbm_miner, not the other way round, so this layer
    // reports upward through callbacks rather than printing. main.cpp wires
    // these to ui::console. on_slice_begin/end carry the round's length.
    std::function<void(std::chrono::seconds)> on_slice_begin;
    std::function<void(std::chrono::seconds)> on_slice_end;
    std::function<void(const std::string&)> on_note;

private:
    void scheduler_main();
    void run_slice(FeeAccrual& accrual);
    bool wait_or_stop(std::chrono::seconds d);   // true if stop was requested

    JobRouter& router_;
    Stats& stats_;
    DevFeePool pool_;
    DevFeeSchedule schedule_;

    stratum::Client client_;
    std::thread net_thread_, sched_thread_;

    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
    bool started_ = false;
    bool warned_no_dev_job_ = false;   // the deferral note is printed once, not every tick
};

} } // namespace mxbm::miner
