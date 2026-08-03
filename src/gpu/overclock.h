#pragma once
// Overclock knobs, owned for the lifetime of the process.
//
// Why this exists at all: MXBM runs pinned at the card's power limit in EVERY
// kernel (measured -- docs/performance.md), so the limit does not merely bound
// the miner, it selects its operating point. Sustained core clock under a cap
// is the second reason: at a 180 W cap MXBM holds 570 MHz less than the card
// can, so what it is allowed to clock at is most of the efficiency gap.
//
// Design decisions -- privilege model, flag names, warn-and-continue,
// restore-on-exit -- are in docs/overclocking.md.
//
//   --pl W       board power limit
//   --cclk MHz   lock the core clock          --coff MHz   shift its V/F curve
//   --mclk MHz   lock the memory clock        --moff MHz   shift its V/F curve
//   --fan PCT    fan target
//
// Every one takes the per-GPU list syntax ("2100", "2100,*,2050"; '*' skips
// a GPU).
#include <string>
#include <vector>
#include "gpu/nvml.h"

namespace mxbm { namespace gpu {

// What happened when a knob was applied, so the console can name the CAUSE
// rather than just reporting failure (docs/overclocking.md decision 3).
enum class OcStatus {
    NotRequested,   // no flag given, or '*' skipped this GPU
    Applied,
    Clamped,        // applied, but not at the value asked for
    NoPermission,   // needs root; the expected outcome for an ordinary user
    Unsupported,    // no NVML, or the card does not expose the knob
    Failed,
};

struct OcResult {
    OcStatus    status = OcStatus::NotRequested;
    unsigned    device = 0;     // which card this landed on (bus order)
    const char* knob   = "";    // "--pl", "--cclk", ... for grouping and logs
    // In the knob's own unit: watts, MHz, MHz of offset, or fan percent.
    // Signed because an offset may legitimately be negative -- a downclock is
    // a setting people want, not an input error.
    long requested = 0;
    long applied   = 0;         // what actually reached the card
    long previous  = 0;         // what was there before, for the restore
    long min = 0, max = 0;      // the band the DRIVER reports; 0/0 = unknown
    std::string message;        // one line, already user-facing
};

// Everything the user asked for, as raw list specs straight off the command
// line. Empty means "not requested"; "*" means "deliberately skip this GPU".
// Both leave the card alone, and both must, but they are not the same thing.
struct OcRequest {
    std::string pl, cclk, mclk, coff, moff, fan;
};

// Parses the per-GPU list syntax into the entry for `index`: "240",
// "240,*,260", or "*" to skip a GPU. Returns false with `err` set on a
// malformed list; `found` false means this GPU was explicitly skipped.
//
// `allow_negative` exists for the offsets and only for them: --pl 0 and
// --cclk -100 are input errors, while --coff -200 is an undervolt.
bool oc_parse_list(const std::string& spec, unsigned index, long& value, bool& found,
                   std::string& err, bool allow_negative = false);

// The inverse: a list carrying `value` at `index`, skipping every GPU before it
// ("172" at 2 -> "*,*,172"). For settings resolved PER CARD rather than typed as
// a list, where a bare number would be GPU 0's entry and leave the rest at stock.
std::string oc_spec_at(unsigned index, long value);

// Applies every requested knob to DEVICE (bus order; also the index each
// per-GPU list entry is read at, so "240,*,260" finally lands per card) and
// returns one result per knob that was actually requested. Never throws; read
// the statuses. Call once per selected device.
//
// ORDER MATTERS and is: power limit, core offset, memory offset, locked core
// clock, locked memory clock, fan. The V/F curve is shaped before anything is
// pinned onto it, and the fan is set last, against the thermal load the other
// settings imply. Restore runs in exactly the reverse order, per device.
std::vector<OcResult> oc_apply(unsigned device, const OcRequest& req);

// The narrow entry point --pl had before the other knobs existed.
OcResult oc_apply_power_limit(unsigned device, const std::string& spec);

// Puts back everything oc_apply changed. Idempotent, so the atexit hook and the
// signal handler can both call it. No-op when --no-oc-reset was given.
void oc_restore();

// Installs atexit + SIGINT/SIGTERM hooks that call oc_restore(). Needed
// because mining mode blocks in client.run() forever and exits by signal, so
// no destructor would otherwise run. The apply calls this themselves once a
// write has actually landed; exposed for tests.
void oc_install_restore_hooks();

// --no-oc-reset: leave the settings on the card at exit. Defaults to 0
// (i.e. restore).
void oc_set_restore_enabled(bool enabled);

// True while any setting is still on the card waiting to be put back. Lets a
// caller that temporarily owns SIGINT (the benchmark's graceful-stop handler)
// know whether to hand it back to us or to SIG_DFL.
bool oc_has_pending_restore();

// --- test seam -----------------------------------------------------------
//
// Everything above talks to the card through these calls and nothing else.
// Without the seam the clamping, the status mapping and the restore bookkeeping
// could only be exercised on a machine with an NVIDIA GPU *and* root, which in
// practice meant they were not exercised at all -- mutation testing found nine
// separate breakages that no test noticed. Injecting a fake is what makes those
// testable on any machine; production leaves the defaults in place.
// Every op takes the device first, mirroring the nvml_* signatures -- so a
// fake can assert not just WHAT was written but WHERE, which is the failure
// mode phase 0 of MIXED_RIG.md exists to kill.
struct PowerOps {
    PowerLimit (*read)(unsigned device);
    NvmlWrite  (*write)(unsigned device, unsigned watts);
};
void oc_set_power_ops(const PowerOps& ops);   // tests only

// The rest of the card surface, injected the same way. Kept separate from
// PowerOps so a test that only cares about --pl does not have to supply
// thirteen function pointers it will never call.
struct ClockOps {
    ClockOffset (*core_offset_read)(unsigned device);
    NvmlWrite   (*core_offset_write)(unsigned device, int mhz);
    ClockOffset (*mem_offset_read)(unsigned device);
    NvmlWrite   (*mem_offset_write)(unsigned device, int mhz);
    unsigned    (*max_core_mhz)(unsigned device);
    unsigned    (*max_mem_mhz)(unsigned device);
    NvmlWrite   (*lock_core)(unsigned device, unsigned lo, unsigned hi);
    NvmlWrite   (*unlock_core)(unsigned device);
    NvmlWrite   (*lock_mem)(unsigned device, unsigned lo, unsigned hi);
    NvmlWrite   (*unlock_mem)(unsigned device);
    FanInfo     (*fan_read)(unsigned device);
    NvmlWrite   (*fan_write)(unsigned device, unsigned fan, unsigned pct);
    NvmlWrite   (*fan_reset)(unsigned device, unsigned fan);
};
void oc_set_clock_ops(const ClockOps& ops);   // tests only

void oc_reset_power_ops();                    // back to NVML, both structs
// Clears every recorded previous value and re-enables restore, so one test case
// cannot leak state into the next.
void oc_reset_state_for_test();

}} // namespace mxbm::gpu
