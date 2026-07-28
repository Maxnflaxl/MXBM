#include "miner/stats.h"

#include <cmath>

namespace mxbm { namespace miner {

namespace {
constexpr std::chrono::minutes kPruneAge{15};
constexpr std::chrono::seconds kWindow15{15};
constexpr std::chrono::seconds kWindow60{60};
constexpr size_t kSubmitCap = 64;
// Shortest gap between two folds into Snapshot::series, so a second dashboard
// polling the API cannot double the sample count and reweight the mean.
constexpr std::chrono::seconds kSeriesInterval{1};
}

// Welford's online update: squared deviations against the running mean.
void Stats::Series::add(double v) {
    if (n == 0) { min = max = v; }
    else { if (v < min) min = v; if (v > max) max = v; }
    ++n;
    double delta = v - mean;
    mean += delta / (double)n;
    m2 += delta * (v - mean);   // deliberately the NEW mean: that is what makes it Welford
}

double Stats::Series::stddev() const {
    return n >= 2 ? std::sqrt(m2 / (double)(n - 1)) : 0.0;
}

void Stats::set_telemetry_source(TelemetryFn fn) {
    std::lock_guard<std::mutex> lock(mutex_);
    telemetry_ = std::move(fn);
}

void Stats::set_device_label(std::string label) {
    set_device_labels({std::move(label)});
}

void Stats::set_device_labels(std::vector<std::string> labels) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (labels.empty()) labels.push_back("GPU 0");
    devices_.clear();
    devices_.reserve(labels.size());
    for (auto& l : labels) {
        DevLedger d;
        d.label = std::move(l);
        devices_.push_back(std::move(d));
    }
}

void Stats::set_driver_version(std::string v) {
    std::lock_guard<std::mutex> lock(mutex_);
    driver_version_ = std::move(v);
}

Stats::Stats() {
    now_fn = [] { return std::chrono::steady_clock::now(); };
    start_ = now_fn();
    // One device until told otherwise, so every consumer can assume at least
    // one row and a miner that never calls set_device_labels() is unchanged.
    devices_.push_back(DevLedger{"GPU 0", 0, 0, 0, 0, 0, 0.0});
}

void Stats::prune_attempts(std::chrono::steady_clock::time_point now) {
    while (!attempts_.empty() && now - attempts_.front().t > kPruneAge)
        attempts_.pop_front();
}

void Stats::record_attempt(uint32_t candidates, unsigned device) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto t = now_fn();
    attempts_.push_back(Event{t, candidates, device});
    total_candidates_ += candidates;
    // An out-of-range index is dropped from the per-device ledger rather than
    // folded into device 0's: a row that silently absorbs another card's work
    // is exactly the confusion these rows exist to remove. The rig total above
    // still counts it, so nothing goes missing from the headline.
    if (device < devices_.size()) {
        devices_[device].total_candidates += candidates;
        ++devices_[device].attempts;
    }
    prune_attempts(t);
}

void Stats::record_submit(const std::string& job_id, Origin origin, unsigned device) {
    std::lock_guard<std::mutex> lock(mutex_);
    const double units = origin == Origin::Dev ? devfee_last_job_units_ : last_job_units_;
    submits_.push_back(PendingSubmit{job_id, now_fn(), units, origin, device});
    while (submits_.size() > kSubmitCap) submits_.pop_front();
}

void Stats::record_share_found(double achieved_units, Origin origin, unsigned device) {
    std::lock_guard<std::mutex> lock(mutex_);

    // The ring records EVERY found share, both origins, tagged -- a log, not a
    // scoreboard, so it runs before the Main-only return below. The target is
    // captured here because the pool may have stepped it by the time anyone
    // reads the share.
    const double target = (origin == Origin::Dev) ? devfee_last_job_units_ : last_job_units_;
    found_.push_back(FoundShare{now_fn(), achieved_units, target, origin == Origin::Dev});
    while (found_.size() > kRecentShares) found_.pop_front();

    // Best-share is the user's record; a fee slice's share is the developer's.
    if (origin != Origin::Main) return;
    if (achieved_units > best_share_units_) best_share_units_ = achieved_units;
    if (device < devices_.size() && achieved_units > devices_[device].best_share_units)
        devices_[device].best_share_units = achieved_units;
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
        else ++rejected_;   // any other code, including a defensive 0
    }

    // The oldest still-pending submit of this same origin, not the oldest
    // overall: both connections can have one in flight across a fee-slice
    // boundary, and popping the other's would bank the wrong difficulty.
    for (auto it = submits_.begin(); it != submits_.end(); ++it) {
        if (it->origin != origin) continue;
        // Only on ACCEPT: stale and rejected shares pay nothing, and dev-fee
        // shares never bank at all.
        if (code == 1 && origin == Origin::Main) accepted_units_ += it->units;
        // The verdict lands on the card that found it. Beam's results echo no
        // submit id, so this is the SAME oldest-pending-first matching the
        // latency uses -- with several cards submitting to one pool the pairing
        // can be wrong under heavy overlap, which is why only the per-device
        // A/S/R breakdown depends on it and the rig totals do not.
        if (origin == Origin::Main && it->device < devices_.size()) {
            DevLedger& d = devices_[it->device];
            if (code == 1) ++d.accepted;
            else if (code == 3) ++d.stale;
            else ++d.rejected;
        }
        const auto t0 = it->t;
        submits_.erase(it);
        // Likewise the displayed latency is the user's pool round trip.
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
    std::vector<uint64_t> dev15(devices_.size(), 0), dev60(devices_.size(), 0),
                          devn60(devices_.size(), 0);
    for (const auto& e : attempts_) {
        auto age = now - e.t;
        const bool in60 = age <= kWindow60, in15 = age <= kWindow15;
        if (in60) { cand60 += e.candidates; ++n60; }
        if (in15) { cand15 += e.candidates; }
        if (e.device < devices_.size()) {
            if (in60) { dev60[e.device] += e.candidates; ++devn60[e.device]; }
            if (in15) { dev15[e.device] += e.candidates; }
        }
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
    s.driver_version = driver_version_;

    s.devices.reserve(devices_.size());
    for (size_t i = 0; i < devices_.size(); ++i) {
        Device d;
        d.label  = devices_[i].label;
        d.sol15  = (double)dev15[i] / 15.0;
        d.sol60  = (double)dev60[i] / 60.0;
        d.sol_session = elapsed > 0.0 ? (double)devices_[i].total_candidates / elapsed : 0.0;
        d.iter60 = (double)devn60[i] / 60.0;
        d.attempts = devices_[i].attempts;
        d.accepted = devices_[i].accepted;
        d.stale    = devices_[i].stale;
        d.rejected = devices_[i].rejected;
        d.best_share_units = devices_[i].best_share_units;
        if (telemetry_) telemetry_((unsigned)i, d);
        s.devices.push_back(std::move(d));
    }
    // The legacy single-device fields mirror device 0, so every consumer that
    // has not learned about the vector yet keeps working and keeps meaning what
    // it meant -- on a one-card rig they are identical by construction.
    if (!s.devices.empty()) {
        const Device& d0 = s.devices.front();
        s.device_label  = d0.label;
        s.has_power     = d0.has_power;     s.power_w       = d0.power_w;
        s.has_sm_clock  = d0.has_sm_clock;  s.sm_clock_mhz  = d0.sm_clock_mhz;
        s.has_mem_clock = d0.has_mem_clock; s.mem_clock_mhz = d0.mem_clock_mhz;
        s.has_temp      = d0.has_temp;      s.temp_c        = d0.temp_c;
        s.has_fan       = d0.has_fan;       s.fan_pct       = d0.fan_pct;
    }
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

    // Fold this observation into the session-long summaries. Sampled on the
    // read path, not from the mining thread: telemetry is only queried while
    // building a snapshot, and a per-second NVML query on the worker would
    // perturb the rate being measured. The cost is that Series::n counts
    // SAMPLES TAKEN, never seconds elapsed -- the ticker yields one per tick
    // and an open dashboard one per poll, up to kSeriesInterval -- so the mean
    // is over observations rather than time-weighted. The two coincide as long
    // as observation times don't correlate with the value.
    if (!series_sampled_ || now - last_series_sample_ >= kSeriesInterval) {
        series_sampled_ = true;
        last_series_sample_ = now;
        // A windowed rate reads far below the truth until its window has filled
        // -- sol15 is 0.0 for the first 15 seconds -- so a window is only
        // sampled once full. Otherwise every speed series would have its
        // minimum pinned at 0.0 and its mean dragged down by startup.
        if (now - start_ >= kWindow15) series_.sol15.add(s.sol15);
        if (now - start_ >= kWindow60) {
            series_.sol60.add(s.sol60);
            series_.iter60.add(s.iter60);
        }
        // Only where the platform supplied a reading: a card reporting power
        // but not fan leaves the fan series empty rather than full of zeroes.
        if (s.has_power)     series_.power_w.add(s.power_w);
        if (s.has_sm_clock)  series_.sm_clock_mhz.add((double)s.sm_clock_mhz);
        if (s.has_mem_clock) series_.mem_clock_mhz.add((double)s.mem_clock_mhz);
        if (s.has_temp)      series_.temp_c.add((double)s.temp_c);
        if (s.has_fan)       series_.fan_pct.add((double)s.fan_pct);
    }
    s.series = series_;
    return s;
}

} } // namespace mxbm::miner
