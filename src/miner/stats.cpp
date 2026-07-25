#include "miner/stats.h"

namespace mxbm { namespace miner {

namespace {
constexpr std::chrono::minutes kPruneAge{15};
constexpr std::chrono::seconds kWindow15{15};
constexpr std::chrono::seconds kWindow60{60};
constexpr size_t kSubmitCap = 64;
}

void Stats::set_telemetry_source(TelemetryFn fn) {
    std::lock_guard<std::mutex> lock(mutex_);
    telemetry_ = std::move(fn);
}

void Stats::set_device_label(std::string label) {
    std::lock_guard<std::mutex> lock(mutex_);
    device_label_ = std::move(label);
}

Stats::Stats() {
    now_fn = [] { return std::chrono::steady_clock::now(); };
    start_ = now_fn();
}

void Stats::prune_attempts(std::chrono::steady_clock::time_point now) {
    while (!attempts_.empty() && now - attempts_.front().t > kPruneAge)
        attempts_.pop_front();
}

void Stats::record_attempt(uint32_t candidates) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto t = now_fn();
    attempts_.push_back(Event{t, candidates});
    total_candidates_ += candidates;
    prune_attempts(t);
}

void Stats::record_submit(const std::string& job_id, Origin origin) {
    std::lock_guard<std::mutex> lock(mutex_);
    const double units = origin == Origin::Dev ? devfee_last_job_units_ : last_job_units_;
    submits_.push_back(PendingSubmit{job_id, now_fn(), units, origin});
    while (submits_.size() > kSubmitCap) submits_.pop_front();
}

void Stats::record_share_found(double achieved_units, Origin origin) {
    std::lock_guard<std::mutex> lock(mutex_);

    // The ring records EVERY found share, both origins, tagged -- it is a log
    // of what happened, not a scoreboard, so excluding fee shares here would
    // leave unexplained gaps in it. (Note this runs before the Main-only
    // early return below; that return guards the best-share record, not this.)
    // The target is captured HERE, not left for the consumer to pair up later:
    // Stats already knows each origin's current job target, and by the time
    // anyone reads this share the pool may well have stepped it.
    const double target = (origin == Origin::Dev) ? devfee_last_job_units_ : last_job_units_;
    found_.push_back(FoundShare{now_fn(), achieved_units, target, origin == Origin::Dev});
    while (found_.size() > kRecentShares) found_.pop_front();

    // Best-share is the user's session record. A share found during a fee
    // slice belongs to the developer, so it does not compete for that slot.
    if (origin != Origin::Main) return;
    if (achieved_units > best_share_units_) best_share_units_ = achieved_units;
}

void Stats::record_result(int code, Origin origin) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (origin == Origin::Dev) {
        if (code == 1) ++devfee_accepted_;
        else if (code == 3) ++devfee_stale_;
        else ++devfee_rejected_;
    } else {
        if (code == 1) ++accepted_;
        else if (code == 3) ++stale_;
        else ++rejected_;   // any other code, including defensively-handled 0: rejected bucket
    }

    // Match the oldest STILL-PENDING submit of this same origin, rather than
    // simply the oldest overall: both connections can have a submit in flight
    // across a fee-slice boundary, and popping the other one's would both
    // bank the wrong difficulty and report the wrong round trip.
    for (auto it = submits_.begin(); it != submits_.end(); ++it) {
        if (it->origin != origin) continue;
        // Bank the pool-credited work only on ACCEPT: stale and rejected shares pay
        // nothing, and counting them would inflate the pool rate exactly when
        // something is going wrong. Dev-fee shares never bank at all --
        // pool_sol_session is what the USER's pool credited them.
        if (code == 1 && origin == Origin::Main) accepted_units_ += it->units;
        const auto t0 = it->t;
        submits_.erase(it);
        // Likewise the displayed latency is the user's pool round trip; a fee
        // slice must not overwrite it with a different pool's timing.
        if (origin == Origin::Main) {
            last_latency_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(
                now_fn() - t0).count();
        }
        break;
    }
}

void Stats::record_job(const std::string& id, double units, Origin origin) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (origin == Origin::Dev) {
        devfee_last_job_units_ = units;   // banked by a later record_submit(Dev)
        return;                           // the table's job line stays the user's
    }
    last_job_id_ = id;
    last_job_units_ = units;
}

double Stats::current_job_units(Origin origin) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return origin == Origin::Dev ? devfee_last_job_units_ : last_job_units_;
}

void Stats::set_devfee_rate(double rate) {
    std::lock_guard<std::mutex> lock(mutex_);
    devfee_rate_ = rate;
}

void Stats::set_devfee_active(bool active) {
    std::lock_guard<std::mutex> lock(mutex_);
    devfee_active_ = active;
}

void Stats::record_devfee_slice(std::chrono::seconds elapsed) {
    std::lock_guard<std::mutex> lock(mutex_);
    devfee_seconds_ += (double)elapsed.count();
    ++devfee_slices_;
}

void Stats::record_connect(const std::string& hostport, long long connect_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    pool_ = hostport;
    connect_ms_ = connect_ms;
}

void Stats::record_disconnect() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++reconnects_;
}

Stats::Snapshot Stats::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto now = now_fn();

    uint64_t cand15 = 0, cand60 = 0, n60 = 0;
    for (const auto& e : attempts_) {
        auto age = now - e.t;
        if (age <= kWindow60) { cand60 += e.candidates; ++n60; }
        if (age <= kWindow15) { cand15 += e.candidates; }
    }

    Snapshot s;
    s.sol15 = (double)cand15 / 15.0;
    s.sol60 = (double)cand60 / 60.0;
    double elapsed = std::chrono::duration<double>(now - start_).count();
    s.sol_session = elapsed > 0.0 ? (double)total_candidates_ / elapsed : 0.0;
    s.pool_sol_session = elapsed > 0.0 ? accepted_units_ / elapsed : 0.0;
    s.iter60 = (double)n60 / 60.0;
    s.accepted = accepted_;
    s.stale = stale_;
    s.rejected = rejected_;
    s.best_share_units = best_share_units_;
    s.last_latency_ms = last_latency_ms_;
    s.pool = pool_;
    s.device_label = device_label_;
    if (telemetry_) telemetry_(s);
    s.connect_ms = connect_ms_;
    s.uptime = std::chrono::duration_cast<std::chrono::seconds>(now - start_);
    s.last_job_id = last_job_id_;
    s.last_job_units = last_job_units_;
    s.reconnects = reconnects_;
    s.devfee_rate = devfee_rate_;
    s.devfee_seconds = devfee_seconds_;
    s.devfee_slices = devfee_slices_;
    s.devfee_active = devfee_active_;
    s.devfee_accepted = devfee_accepted_;
    s.devfee_stale = devfee_stale_;
    s.devfee_rejected = devfee_rejected_;
    s.recent_shares.reserve(found_.size());
    for (const auto& f : found_) {
        s.recent_shares.push_back(
            RecentShare{std::chrono::duration<double>(now - f.t).count(), f.units, f.target, f.dev});
    }
    return s;
}

} } // namespace mxbm::miner
