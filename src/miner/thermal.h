#pragma once
// Thermal protection: pauses a device that runs too hot, resumes it when it has
// cooled, and keeps the watchdog from mistaking either for a hung card.
//
// The pause is cooperative -- Engine checks paused() before each solve -- rather
// than a kill: a solve in flight finishes, and no driver state is torn down, so
// resuming costs nothing but the wait.
//
// A paused device stops advancing Stats::Device::attempts, which is exactly what
// miner::Watchdog calls hung. Wiring one without the other turns the first hot
// card into an exit(42) and a supervisor restart into the same heat, so Watchdog
// takes a paused predicate and the two are tested together.
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace mxbm { namespace miner {

// Which sensor --tmode selects. Availability is a property of the card, not of
// this code: consumer NVIDIA boards report Edge only (docs/performance.md), so
// main() refuses a mode the driver cannot answer rather than silently reading
// another sensor.
enum class TempSensor { Edge, Junction, Memory };

bool parse_temp_sensor(const std::string& text, TempSensor& out);
const char* temp_sensor_name(TempSensor s);

class Thermal {
public:
    // Per device, in degrees C. 0 disables that half: stop_c 0 means the device
    // is never paused, start_c 0 means a paused device is never resumed.
    struct Limits { unsigned stop_c = 0, start_c = 0; };

    // Reads `device`'s temperature. False means the sensor did not answer, which
    // is treated as "no reading" -- never as cold, and never as hot.
    using TempFn = std::function<bool(unsigned device, unsigned& temp_c)>;

    void configure(std::vector<Limits> limits);
    bool configured() const;

    TempFn read_temp;
    std::function<void(unsigned device, unsigned temp_c)> on_pause, on_resume;

    // Test seam, as elsewhere in miner/: every clock read goes through it.
    std::function<std::chrono::steady_clock::time_point()> now_fn;

    // True while `device` is held out of mining. Cheap enough to call per solve.
    bool paused(unsigned device) const;

    // One check of every configured device. run() calls it on a thread; tests
    // call it directly with a fake sensor.
    void poll();

    // 2 s, not 5: a card under load can climb 20 C between polls, and every
    // degree of that overshoot is above the limit the operator set.
    void start(std::chrono::seconds interval = std::chrono::seconds(2));
    void stop();
    ~Thermal();

private:
    mutable std::mutex mutex_;
    std::vector<Limits> limits_;
    std::vector<bool> paused_;

    std::thread worker_;
    bool started_ = false, stop_requested_ = false;
    std::chrono::seconds interval_{2};
};

} } // namespace mxbm::miner
