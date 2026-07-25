#pragma once
// Optional NVIDIA telemetry (power, clocks, temperature, fan) via NVML.
//
// dlopen'd at runtime rather than linked: NVML ships with the DRIVER, not the CUDA
// toolkit, so this works on a machine with an NVIDIA driver and no toolkit, and simply
// reports nothing on a machine with neither. Nothing in the build depends on it.
#include <cstdint>

namespace mxbm { namespace gpu {

struct Telemetry {
    bool  have_power = false;  double power_w      = 0.0;
    bool  have_sm    = false;  unsigned sm_clock_mhz  = 0;
    bool  have_mem   = false;  unsigned mem_clock_mhz = 0;
    bool  have_temp  = false;  unsigned temp_c        = 0;
    bool  have_fan   = false;  unsigned fan_pct       = 0;
    bool any() const { return have_power || have_sm || have_mem || have_temp || have_fan; }
};

// Returns false if NVML is unavailable; safe to call once at startup and ignore.
bool nvml_init();
void nvml_shutdown();
// Samples device 0. Fields whose query fails keep have_* == false, so a card that
// reports power but not fan (common on laptops) still shows what it has.
Telemetry nvml_sample();

}} // namespace mxbm::gpu
