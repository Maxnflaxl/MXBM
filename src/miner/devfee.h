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
// solving for the developer's wallet instead of the user's. The mechanism is
// time-slicing -- a second stratum connection to the fee pool is held open
// alongside the user's, and once the debt reaches one slice the JobRouter
// below switches the solver to that pool's job and straight back. There is
// one GPU and one Engine; only the job it is fed changes.
//
// The fee is reported rather than hidden: announced at startup and at each
// round, shown on its own row of the statistics table and in /summary, and
// specified in docs/devfee.md. Charging in time rather than in shares is what
// keeps the user's counters clean -- see the Origin note on JobRouter below.

// Where fee slices mine to.
struct DevFeePool {
    std::string host;
    uint16_t port = 0;
    bool tls = true;
    std::string user;   // the developer's Beam address
    std::string pass;
};

struct DevFeeSchedule {
    double rate = 0.0;                        // fraction of mining time, 0.01 == 1.0%
    std::chrono::seconds cycle{3600};         // the window `rate` is expressed over

    // Length of one fee round: rate * cycle, e.g. 1% of an hour == 36 s. Whole
    // slices rather than continuously: a switch costs a job change either way.
    std::chrono::seconds slice() const;
};

// The fee MXBM ships with, and the pool it goes to.
DevFeePool devfee_pool();
DevFeeSchedule devfee_schedule();

// False when this build has no developer address compiled in; the fee is then
// genuinely inert -- no second connection, no slice. Checked before any wiring.
bool devfee_configured();

// Builds the fee connection's login credential:
//
//     <developer address>.<user's worker name>_<fee percent>
//
// e.g. "<devaddr>.rig1_2.5" for a user mining as "<useraddr>.rig1" who raised
// the fee with --dev-fee 2.5, so the round is identifiably theirs on the fee
// pool's dashboard. Only the worker part of `user_credential` is used,
// sanitised to [A-Za-z0-9_-] and capped at 32 characters since it goes onto
// the wire inside a JSON login line; absent or sanitised away it is "mxbm".
std::string devfee_login(const std::string& dev_address,
                         const std::string& user_credential,
                         double rate);

// Accrual clock: turns time spent mining into fee debt, in seconds owed.
// Pure and thread-free so the arithmetic can be tested without waiting an
// hour (tests/test_devfee.cpp). The caller only feeds it time during which the
// user's pool had a job AND a device was solving, so a dead pool, a pause or a
// solver backoff is not charged. Debt carries across slices: a round cut short
// stays owed. Sub-second throughout -- truncating a round's overshoot would
// leave it unsettled and charge above the stated rate.
class FeeAccrual {
public:
    explicit FeeAccrual(DevFeeSchedule schedule);

    void add_mining_time(std::chrono::duration<double> dt);
    bool slice_due() const;                    // debt has reached a full slice
    std::chrono::seconds slice_len() const;
    void settle(std::chrono::duration<double> spent);   // floors at zero
    double debt_seconds() const;

private:
    DevFeeSchedule schedule_;
    double debt_s_ = 0.0;
};

// Decides which pool's job the solver is working on. Pure of sockets and
// threads (the dispatch target is injected), so the switching logic is
// unit-testable. Both connections keep feeding it the whole time and only one
// is dispatched onward at any moment, which is what makes the switch
// lossless: switching back re-dispatches the job that arrived during the fee
// round, rather than idling until that pool sends the next.
class JobRouter {
public:
    // Wired to Engine::on_job in main.cpp.
    using DispatchFn = std::function<void(const stratum::Job&, const std::string&, Origin)>;

    explicit JobRouter(DispatchFn dispatch);

    // Dispatched onward only if `origin` is the active one; otherwise cached.
    void offer(const stratum::Job& job, const std::string& nonceprefix, Origin origin);

    // Re-dispatches `origin`'s latest cached job. No-op if already active.
    void set_active(Origin origin);

    Origin active() const;
    bool has(Origin origin) const;   // this origin has a job cached right now

    // Drops `origin`'s cached job. Wired to both connections' on_disconnect:
    // a job from a pool that has since dropped is neither mineable nor
    // something to accrue fee debt against.
    void clear(Origin origin);

    // has(), plus the job arrived within `max_age` -- the half-dead socket
    // that never reports a disconnect.
    bool fresh(Origin origin, std::chrono::seconds max_age) const;

private:
    struct Slot {
        stratum::Job job{};
        std::string nonceprefix;
        bool present = false;
        std::chrono::steady_clock::time_point at{};
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
// accrues debt while the user is mining, and drives JobRouter across a round.
//
// THREADING / LIFETIME. start() spawns two threads: the fee Client's run()
// loop and the scheduler. stratum::Client::run() never returns by design
// (see client.h), so the network thread cannot be joined -- stop() detaches
// it, and a started DevFee must live until the process exits. The tests
// exercise JobRouter and FeeAccrual directly and never call start().
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

    // main.cpp's submit_fn routes here by the Origin the solved job carried,
    // never by "is a round running now", so a round ending mid-solve still
    // submits to the pool that issued the job.
    void submit(const stratum::Solution& solution);

    // Empty by default and guarded at the call site: mxbm_ui links mxbm_miner,
    // not the reverse, so this layer reports upward rather than printing. The
    // slice hooks carry the round's length.
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
