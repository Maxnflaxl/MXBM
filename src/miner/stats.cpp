#include "miner/stats.h"

namespace mxbm { namespace miner {

namespace {
constexpr std::chrono::minutes kPruneAge{15};
constexpr std::chrono::seconds kWindow15{15};
constexpr std::chrono::seconds kWindow60{60};
constexpr size_t kSubmitCap = 64;
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

void Stats::record_submit(const std::string& job_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    submits_.push_back(PendingSubmit{job_id, now_fn()});
    while (submits_.size() > kSubmitCap) submits_.pop_front();
}

void Stats::record_share_found(double achieved_units) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (achieved_units > best_share_units_) best_share_units_ = achieved_units;
}

void Stats::record_result(int code) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (code == 1) ++accepted_;
    else if (code == 3) ++stale_;
    else ++rejected_;   // any other code, including defensively-handled 0: rejected bucket

    if (!submits_.empty()) {
        auto t0 = submits_.front().t;
        submits_.pop_front();
        auto now = now_fn();
        last_latency_ms_ =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - t0).count();
    }
}

void Stats::record_job(const std::string& id, double units) {
    std::lock_guard<std::mutex> lock(mutex_);
    last_job_id_ = id;
    last_job_units_ = units;
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
    s.iter60 = (double)n60 / 60.0;
    s.accepted = accepted_;
    s.stale = stale_;
    s.rejected = rejected_;
    s.best_share_units = best_share_units_;
    s.last_latency_ms = last_latency_ms_;
    s.pool = pool_;
    s.device_label = device_label_;
    s.connect_ms = connect_ms_;
    s.uptime = std::chrono::duration_cast<std::chrono::seconds>(now - start_);
    s.last_job_id = last_job_id_;
    s.last_job_units = last_job_units_;
    s.reconnects = reconnects_;
    return s;
}

} } // namespace mxbm::miner
