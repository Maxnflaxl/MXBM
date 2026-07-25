// The developer fee. See devfee.h for the design and docs/devfee.md for the
// user-facing description of what it costs and when it runs.
#include "miner/devfee.h"

#include <cmath>
#include <cstdio>

#include "pow/difficulty.h"

namespace mxbm { namespace miner {

namespace {

// -- what MXBM charges -------------------------------------------------------
//
// 1.0% of mining time, matching what the closed-source BeamHash III miners
// take, charged as one 36-second round per hour of actual mining. Changing
// kDevFeeRate here changes the rate everywhere: the schedule, the startup
// disclosure, the statistics table and /summary all read it from
// devfee_schedule() rather than restating it.
constexpr double kDevFeeRate = 0.01;
constexpr std::chrono::seconds kDevFeeCycle{3600};

// The developer's Beam address, and the single switch every other component
// reads: devfee_configured() reports whether one is set, and main.cpp wires up
// the fee pool, the scheduler and the stats row only when it is. Keeping that
// one predicate is what lets the console, the table, the dashboard and
// /summary all say the same thing about whether a fee applies.
const char* const kDevAddress =
    "12cafbe121b5f063d2c63152058575479a2826a41fd4176296dae2e8ad3fc9ffc60";

// beam.herominers.com auto-routes to the nearest region. Port 1130 is the
// standard-difficulty endpoint and speaks TLS, which is what the user's own
// pool connection uses too -- the fee connection gets no special transport
// treatment, and nothing here tries to make it hard to see or to block.
const char* const kDevPoolHost = "beam.herominers.com";
constexpr uint16_t kDevPoolPort = 1130;
constexpr bool kDevPoolTls = true;

} // namespace

std::chrono::seconds DevFeeSchedule::slice() const {
    const double s = rate * (double)cycle.count();
    return std::chrono::seconds((long long)std::llround(s));
}

DevFeePool devfee_pool() {
    DevFeePool p;
    p.host = kDevPoolHost;
    p.port = kDevPoolPort;
    p.tls = kDevPoolTls;
    p.user = kDevAddress;
    return p;
}

DevFeeSchedule devfee_schedule() {
    DevFeeSchedule s;
    s.rate = kDevFeeRate;
    s.cycle = kDevFeeCycle;
    return s;
}

bool devfee_configured() {
    // A rate with nowhere to send it, or a destination with no rate, is not
    // a configured fee -- either way nothing should be charged.
    return kDevAddress[0] != '\0' && kDevFeeRate > 0.0
           && devfee_schedule().slice() > std::chrono::seconds(0);
}

std::string devfee_login(const std::string& dev_address,
                         const std::string& user_credential,
                         double rate) {
    // Beam credentials are "address[.worker]" and pools split on the FIRST
    // dot, so everything after it is the worker name -- which is also why a
    // fractional percent below ("2.5") is safe to place inside it.
    std::string worker;
    const size_t dot = user_credential.find('.');
    if (dot != std::string::npos) worker = user_credential.substr(dot + 1);

    std::string clean;
    for (char c : worker) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
                     || (c >= '0' && c <= '9') || c == '_' || c == '-';
        if (ok) clean.push_back(c);
        if (clean.size() >= 32) break;
    }
    if (clean.empty()) clean = "mxbm";

    // %g renders 1.0 as "1" and 2.5 as "2.5" -- the rate as a human would
    // write it, with no trailing zeros to read as extra precision.
    char pct[32];
    std::snprintf(pct, sizeof pct, "%g", rate * 100.0);

    return dev_address + "." + clean + "_" + pct;
}

// -- FeeAccrual ---------------------------------------------------------------

FeeAccrual::FeeAccrual(DevFeeSchedule schedule) : schedule_(schedule) {}

void FeeAccrual::add_mining_time(std::chrono::seconds dt) {
    if (dt.count() <= 0) return;
    debt_s_ += schedule_.rate * (double)dt.count();
}

bool FeeAccrual::slice_due() const {
    const auto slice = schedule_.slice();
    if (slice.count() <= 0) return false;
    return debt_s_ >= (double)slice.count();
}

std::chrono::seconds FeeAccrual::slice_len() const { return schedule_.slice(); }

void FeeAccrual::settle(std::chrono::seconds spent) {
    debt_s_ -= (double)spent.count();
    // A round that overran its slice (a slow final solve) settles the excess
    // too rather than banking negative debt toward the next one.
    if (debt_s_ < 0.0) debt_s_ = 0.0;
}

double FeeAccrual::debt_seconds() const { return debt_s_; }

// -- JobRouter ----------------------------------------------------------------

JobRouter::JobRouter(DispatchFn dispatch) : dispatch_(std::move(dispatch)) {}

JobRouter::Slot& JobRouter::slot(Origin origin) {
    return origin == Origin::Dev ? dev_ : main_;
}

const JobRouter::Slot& JobRouter::slot(Origin origin) const {
    return origin == Origin::Dev ? dev_ : main_;
}

void JobRouter::offer(const stratum::Job& job, const std::string& nonceprefix, Origin origin) {
    std::lock_guard<std::mutex> lock(mutex_);
    Slot& s = slot(origin);
    s.job = job;
    s.nonceprefix = nonceprefix;
    s.present = true;
    // Dispatch under the lock, not after releasing it: a set_active()
    // racing this between the unlock and the dispatch could otherwise
    // deliver its own job first and let this older one land on top, leaving
    // the solver mining the pool that is no longer active. Safe to hold --
    // dispatch_ reaches Engine::on_job, which takes only Engine's own
    // mailbox mutex and never calls back into the router.
    if (origin == active_ && dispatch_) dispatch_(s.job, s.nonceprefix, origin);
}

void JobRouter::set_active(Origin origin) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (origin == active_) return;
    active_ = origin;
    // Re-dispatch this origin's newest cached job so the switch takes effect
    // immediately, instead of leaving the solver on the other pool's job
    // until this one happens to send a fresh one.
    const Slot& s = slot(origin);
    if (s.present && dispatch_) dispatch_(s.job, s.nonceprefix, origin);
}

Origin JobRouter::active() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_;
}

bool JobRouter::has(Origin origin) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return slot(origin).present;
}

// -- DevFee -------------------------------------------------------------------

DevFee::DevFee(JobRouter& router, Stats& stats, DevFeePool pool, DevFeeSchedule schedule)
    : router_(router), stats_(stats), pool_(std::move(pool)), schedule_(schedule) {}

DevFee::~DevFee() { stop(); }

void DevFee::start() {
    if (started_) return;
    started_ = true;

    client_.on_job = [this](const stratum::Job& j) {
        // Runs on client_'s own run() thread -- the same thread that writes
        // its nonceprefix_ -- which is what makes this read race-free, the
        // same contract main.cpp's client.on_job relies on for the user's
        // connection (see engine.h).
        router_.offer(j, client_.current_nonceprefix(), Origin::Dev);
        stats_.record_job(j.id, pow::to_display_units(j.difficulty), Origin::Dev);
    };
    client_.on_result = [this](const stratum::Result& r) {
        if (r.id == "login") return;   // the fee connection's own handshake; not share traffic
        stats_.record_result(r.code, Origin::Dev);
    };

    client_.connect(pool_.host, pool_.port, pool_.tls);
    client_.login(pool_.user);   // stores the credential for run()'s reconnect path too

    // Both wrapped: an exception escaping a thread entry function calls
    // std::terminate() and takes the whole miner down. Nothing on these
    // paths is expected to throw -- the transport reports by return value
    // and the scheduler only touches mutexes -- which is exactly why an
    // unexpected one must not be allowed to be fatal.
    net_thread_ = std::thread([this] {
        try { client_.run(); } catch (...) {}
    });
    sched_thread_ = std::thread([this] {
        try { scheduler_main(); } catch (...) {}
    });
}

void DevFee::stop() {
    if (!started_) return;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    if (sched_thread_.joinable()) sched_thread_.join();
    // client_.run() never returns (client.h), so this one cannot be joined.
    // See the lifetime note in devfee.h: a started DevFee outlives the process.
    if (net_thread_.joinable()) net_thread_.detach();
    started_ = false;
}

void DevFee::submit(const stratum::Solution& solution) {
    client_.submit(solution);
}

bool DevFee::wait_or_stop(std::chrono::seconds d) {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, d, [this] { return stop_; });
}

void DevFee::scheduler_main() {
    FeeAccrual accrual(schedule_);
    constexpr std::chrono::seconds kTick{1};

    while (true) {
        if (wait_or_stop(kTick)) return;

        // Only time the user is actually mining counts toward the fee. With
        // no job from their pool -- still connecting, pool down, between
        // reconnects -- nothing is being earned, so nothing is owed.
        if (!router_.has(Origin::Main)) continue;

        accrual.add_mining_time(kTick);
        if (!accrual.slice_due()) continue;

        run_slice(accrual);
        if (wait_or_stop(std::chrono::seconds(0))) return;
    }
}

void DevFee::run_slice(FeeAccrual& accrual) {
    if (!router_.has(Origin::Dev)) {
        // The fee pool has not sent a job yet (still connecting, or
        // unreachable). Defer: the debt stays owed and the next tick tries
        // again. Burning the user's hashrate on a job we do not have would
        // cost them time and pay the developer nothing.
        if (!warned_no_dev_job_ && on_note) {
            warned_no_dev_job_ = true;
            on_note("dev fee pool has not sent a job yet - fee round deferred");
        }
        return;
    }
    warned_no_dev_job_ = false;

    const auto len = accrual.slice_len();
    stats_.set_devfee_active(true);
    if (on_slice_begin) on_slice_begin(len);

    const auto t0 = std::chrono::steady_clock::now();
    router_.set_active(Origin::Dev);
    wait_or_stop(len);   // a stop request cuts the round short; the restore below still runs
    router_.set_active(Origin::Main);
    const auto spent = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - t0);

    stats_.set_devfee_active(false);
    stats_.record_devfee_slice(spent);
    accrual.settle(spent);
    if (on_slice_end) on_slice_end(spent);
}

} } // namespace mxbm::miner
