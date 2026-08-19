#include "gpu/telemetry_window.h"

#include <algorithm>
#include <chrono>

namespace mxbm { namespace gpu {
namespace {

// The median of the series past its first fifth, or 0 with `have` left false.
template <typename T>
bool trimmed_median(std::vector<T>& v, T& out) {
    if (v.empty()) return false;
    const size_t skip = v.size() / 5;
    std::sort(v.begin() + (long)skip, v.end());
    out = v[skip + (v.size() - skip) / 2];
    return true;
}

} // namespace

TelemetryWindow::TelemetryWindow(unsigned device, unsigned interval_ms) : device_(device) {
    have_energy0_ = nvml_total_energy_mj(energy0_, device_);
    sampler_ = std::thread([this, interval_ms] {
        while (!done_.load(std::memory_order_relaxed)) {
            const Telemetry t = nvml_sample(device_);
            if (t.have_power) power_.push_back(t.power_w);
            if (t.have_sm)    sm_.push_back(t.sm_clock_mhz);
            if (t.have_mem)   mem_.push_back(t.mem_clock_mhz);
            if (t.have_temp)  temp_.push_back(t.temp_c);
            if (t.have_util)  util_.push_back(t.util_pct);
            unsigned long long ev = 0;
            if (nvml_clock_event_reasons(device_, ev)) { events_ |= ev; have_events_ = true; }
            std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
        }
    });
}

TelemetryWindow::~TelemetryWindow() {
    done_.store(true, std::memory_order_relaxed);
    if (sampler_.joinable()) sampler_.join();
}

TelemetrySummary TelemetryWindow::close(double elapsed_s) {
    if (closed_) return summary_;
    closed_ = true;
    done_.store(true, std::memory_order_relaxed);
    if (sampler_.joinable()) sampler_.join();

    summary_.samples = (unsigned)power_.size();
    // The sampler mean is the fallback; the counter overwrites it below when the
    // card keeps one. Trimmed like every other series.
    if (!power_.empty()) {
        const size_t skip = power_.size() / 5;
        double sum = 0.0;
        for (size_t i = skip; i < power_.size(); ++i) sum += power_[i];
        summary_.power_w = sum / (double)(power_.size() - skip);
        summary_.have_power = true;
    }
    unsigned long long e1 = 0;
    if (have_energy0_ && nvml_total_energy_mj(e1, device_) && e1 > energy0_) {
        summary_.joules = (double)(e1 - energy0_) / 1000.0;
        summary_.have_energy = true;
        if (elapsed_s > 0.0) {
            summary_.power_w = summary_.joules / elapsed_s;
            summary_.have_power = true;
        }
    }
    summary_.have_sm   = trimmed_median(sm_, summary_.sm_clock_mhz);
    summary_.have_mem  = trimmed_median(mem_, summary_.mem_clock_mhz);
    summary_.have_temp = trimmed_median(temp_, summary_.temp_c);
    summary_.have_util = trimmed_median(util_, summary_.util_pct);
    summary_.have_events = have_events_;
    summary_.events = events_;
    return summary_;
}

}} // namespace mxbm::gpu
