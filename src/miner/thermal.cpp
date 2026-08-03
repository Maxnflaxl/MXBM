#include "miner/thermal.h"

#include <cctype>

namespace mxbm { namespace miner {

bool parse_temp_sensor(const std::string& text, TempSensor& out) {
    std::string t;
    for (char c : text) t += (char)std::tolower((unsigned char)c);
    if (t == "edge")     { out = TempSensor::Edge;     return true; }
    if (t == "junction") { out = TempSensor::Junction; return true; }
    if (t == "memory")   { out = TempSensor::Memory;   return true; }
    return false;
}

const char* temp_sensor_name(TempSensor s) {
    switch (s) {
        case TempSensor::Edge:     return "edge";
        case TempSensor::Junction: return "junction";
        case TempSensor::Memory:   return "memory";
    }
    return "edge";
}

void Thermal::configure(std::vector<Limits> limits) {
    std::lock_guard<std::mutex> lock(mutex_);
    limits_ = std::move(limits);
    paused_.assign(limits_.size(), false);
}

bool Thermal::configured() const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const Limits& l : limits_) if (l.stop_c != 0) return true;
    return false;
}

bool Thermal::paused(unsigned device) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return device < paused_.size() && paused_[device];
}

void Thermal::poll() {
    if (!read_temp) return;

    size_t n;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        n = limits_.size();
    }
    for (size_t i = 0; i < n; ++i) {
        Limits lim;
        bool was_paused;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            lim = limits_[i];
            was_paused = paused_[i];
        }
        if (lim.stop_c == 0 && !was_paused) continue;

        unsigned temp = 0;
        if (!read_temp((unsigned)i, temp)) continue;   // no reading: change nothing

        bool now_paused = was_paused;
        if (!was_paused && lim.stop_c != 0 && temp >= lim.stop_c)      now_paused = true;
        // start_c 0 leaves a paused device paused: that is the documented meaning
        // of "no restart temperature", not an oversight.
        else if (was_paused && lim.start_c != 0 && temp <= lim.start_c) now_paused = false;
        if (now_paused == was_paused) continue;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            paused_[i] = now_paused;
        }
        if (now_paused) { if (on_pause)  on_pause((unsigned)i, temp); }
        else            { if (on_resume) on_resume((unsigned)i, temp); }
    }
}

void Thermal::start(std::chrono::seconds interval) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (started_) return;
        started_ = true;
        stop_requested_ = false;
        interval_ = interval;
    }
    worker_ = std::thread([this] {
        for (;;) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (stop_requested_) return;
            }
            try { poll(); } catch (...) {}
            // Slept in short slices so stop() is not held up by the interval;
            // a condition variable would need the poll to hold mutex_, which it
            // deliberately does not.
            for (int i = 0; i < (int)interval_.count() * 10; ++i) {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (stop_requested_) return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    });
}

void Thermal::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!started_) return;
        stop_requested_ = true;
    }
    if (worker_.joinable()) worker_.join();
    std::lock_guard<std::mutex> lock(mutex_);
    started_ = false;
}

Thermal::~Thermal() { stop(); }

} } // namespace mxbm::miner
