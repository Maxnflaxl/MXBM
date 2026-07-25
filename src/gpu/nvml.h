#pragma once
// Optional NVIDIA telemetry (power, clocks, temperature, fan) via NVML.
//
// dlopen'd at runtime rather than linked: NVML ships with the DRIVER, not the CUDA
// toolkit, so this works on a machine with an NVIDIA driver and no toolkit, and simply
// reports nothing on a machine with neither. Nothing in the build depends on it.
#include <cstdint>
#include <string>

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

// The installed NVIDIA driver version ("610.43.03"), or "" when NVML is not
// available. Shown on the statistics block's header line: a driver change is
// the likeliest explanation for a hashrate that moved on its own, so a log
// that records it answers "what changed?" by itself.
std::string nvml_driver_version();

// Device 0's PCI address as "bus:device" ("1:0"), or "" when unavailable --
// the short form the reference miner prints, not NVML's full "00000000:01:00.0". On a
// multi-GPU rig this is what tells two identical cards apart.
std::string nvml_pci_address();

// --- board power limit ---------------------------------------------------
//
// The most valuable knob on this card: MXBM runs pinned at the limit in every
// kernel, so the limit sets the operating point outright. See
// docs/performance.md "Power and efficiency" for the measured curve.

struct PowerLimit {
    bool     valid       = false;   // false => the device does not report one
    unsigned current_w   = 0;
    unsigned default_w   = 0;       // the card's own default, restored on exit
    unsigned min_w       = 0;       // driver-reported constraints, not our guess
    unsigned max_w       = 0;
};

// Reads device 0's limit and the band the DRIVER permits. Clamping against
// these is a fact about the installed card rather than an assumption about
// someone else's silicon, which is what the open question in
// docs/overclocking.md ("Clamps") was waiting on.
PowerLimit nvml_power_limit();

// NVML speaks milliwatts. Rounding rather than truncating matters: a card whose
// limit reads 284999 mW must report 285 W, or a no-op write looks like a change
// and the restore-on-exit bookkeeping puts back a value that was never set.
inline unsigned nvml_mw_to_w(unsigned mw) { return (mw + 500u) / 1000u; }

enum class NvmlWrite { Ok, NoPermission, Unsupported, Failed };

// Sets device 0's limit. Every NVML write needs root (verified: called as uid
// 1000 with the value the card already had, a true no-op, it still returns
// NVML_ERROR_NO_PERMISSION), so NoPermission is the expected outcome for an
// ordinary user and callers must report it as a cause, not a crash.
NvmlWrite nvml_set_power_limit(unsigned watts);

}} // namespace mxbm::gpu
