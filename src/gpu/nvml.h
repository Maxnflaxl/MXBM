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
    // Busy percentage. NVML reports it via nvmlDeviceGetUtilizationRates; on Apple
    // Silicon it is the ONLY thing the public API exposes, which is why it earns a
    // field rather than living in the Metal backend.
    bool  have_util  = false;  unsigned util_pct      = 0;
    bool any() const { return have_power || have_sm || have_mem || have_temp
                           || have_fan || have_util; }
};

// Returns false if NVML is unavailable; safe to call once at startup and ignore.
bool nvml_init();
void nvml_shutdown();

// How many devices NVML sees. 0 when NVML is unavailable.
unsigned nvml_device_count();

// Samples one device. Fields whose query fails keep have_* == false, so a card
// that reports power but not fan (common on laptops) still shows what it has.
// An out-of-range index returns an empty Telemetry rather than another card's:
// a per-device row showing the wrong card's numbers is worse than a blank one.
Telemetry nvml_sample(unsigned index = 0);

// The installed NVIDIA driver version ("610.43.03"), or "" when NVML is not
// available. Shown on the statistics block's header line: a driver change is
// the likeliest explanation for a hashrate that moved on its own, so a log
// that records it answers "what changed?" by itself.
std::string nvml_driver_version();

// A device's PCI address as "bus:device" ("1:0"), or "" when unavailable --
// the short form the reference miner prints, not NVML's full "00000000:01:00.0". On a
// multi-GPU rig this is what tells two identical cards apart.
std::string nvml_pci_address(unsigned index = 0);

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

// --- clocks and fans -----------------------------------------------------
//
// Two different mechanisms, deliberately kept apart because they fail
// differently and are restored differently:
//
//   OFFSETS (--coff/--moff) shift the whole voltage/frequency curve, so the
//   card still manages itself and a too-high offset shows up as instability.
//   Restoring means writing the previous offset back.
//
//   LOCKED CLOCKS (--cclk/--mclk) pin the clock to a range, taking that
//   management away from the driver. Restoring means an explicit reset call,
//   not writing a previous value -- there is no "previous lock" to put back.
//
// The pair is the standard Linux undervolt idiom: lock the core clock and
// raise the V/F offset, so the locked frequency runs at a lower voltage than
// it otherwise would.

struct ClockOffset {
    bool valid = false;         // false => the card does not expose this knob
    int  current_mhz = 0;
    int  min_mhz = 0, max_mhz = 0;   // the driver's permitted band
};

ClockOffset nvml_core_clock_offset();
ClockOffset nvml_mem_clock_offset();
NvmlWrite   nvml_set_core_clock_offset(int mhz);
NvmlWrite   nvml_set_mem_clock_offset(int mhz);

// The card's maximum for a clock domain, 0 when unreported. Used to clamp
// --cclk/--mclk against the hardware rather than against a constant of ours.
unsigned nvml_max_core_clock_mhz();
unsigned nvml_max_mem_clock_mhz();

// Locked clocks take a (min, max) pair; MXBM pins both to the same value,
// which is what "run at exactly this clock" means and what --cclk promises.
NvmlWrite nvml_set_locked_core_clock(unsigned min_mhz, unsigned max_mhz);
NvmlWrite nvml_set_locked_mem_clock(unsigned min_mhz, unsigned max_mhz);
NvmlWrite nvml_reset_locked_core_clock();
NvmlWrite nvml_reset_locked_mem_clock();

struct FanInfo {
    bool     valid = false;
    unsigned count = 0;         // a card can have several; --fan sets all of them
    unsigned pct   = 0;         // fan 0's current speed
    unsigned min_pct = 0, max_pct = 0;   // driver-reported band, 0/0 if unknown
};

FanInfo   nvml_fans();
NvmlWrite nvml_set_fan_speed(unsigned fan, unsigned pct);
// Hands the fan back to the driver's own curve. NOT the same as writing the
// speed we first read: that would leave it pinned at a fixed value forever,
// which on a card that later gets hot is a way to cook it.
NvmlWrite nvml_reset_fan(unsigned fan);

}} // namespace mxbm::gpu
