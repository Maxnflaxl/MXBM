#pragma once
// Board power limit, owned for the lifetime of the process.
//
// Why this exists at all: MXBM runs pinned at the card's power limit in EVERY
// kernel (measured -- docs/performance.md "Power and efficiency"), so the limit
// does not merely bound the miner, it selects its operating point. On the
// reference card 220 W gives the same throughput as the reference miner for
// 19 W less, and stock 285 W buys the last 2.3 sol/s for 45 W.
//
// Design decisions -- privilege model, the reference miner-compatible flag names, warn-and-
// continue, restore-on-exit -- are in docs/overclocking.md. Only --pl is
// implemented; the clock/fan knobs from that document are deliberately not, and
// the list-parsing and restore machinery here is shaped to take them.
#include <string>
#include "gpu/nvml.h"

namespace mxbm { namespace gpu {

// What happened when a knob was applied, so the console can name the CAUSE
// rather than just reporting failure (docs/overclocking.md decision 3).
enum class OcStatus {
    NotRequested,   // no flag given
    Applied,
    Clamped,        // applied, but not at the value asked for
    NoPermission,   // needs root; the expected outcome for an ordinary user
    Unsupported,    // no NVML, or the card does not expose the knob
    Failed,
};

struct OcResult {
    OcStatus status = OcStatus::NotRequested;
    unsigned requested_w = 0;
    unsigned applied_w   = 0;   // what actually reached the card
    unsigned min_w = 0, max_w = 0, previous_w = 0;
    std::string message;        // one line, already user-facing
};

// Parses the reference miner's per-GPU list syntax into the entry for `index`: "240",
// "240,*,260", or "*" to skip a GPU. MXBM drives one GPU today, but parsing
// the full syntax now means multi-GPU will not be a breaking change
// (docs/overclocking.md decision 2). Returns false with `err` set on a
// malformed list. `found` false means this GPU was explicitly skipped.
bool oc_parse_list(const std::string& spec, unsigned index, long& value, bool& found,
                   std::string& err);

// Applies `spec` to device 0 and remembers the previous limit. Never throws;
// read the status. Clamps to the band the DRIVER reports rather than to a
// constant of ours -- a limit outside it is reported, not silently accepted.
OcResult oc_apply_power_limit(const std::string& spec);

// Puts back what oc_apply_* changed. Idempotent, so the atexit hook and the
// signal handler can both call it. No-op when --no-oc-reset was given.
void oc_restore();

// Installs atexit + SIGINT/SIGTERM hooks that call oc_restore(). Needed
// because mining mode blocks in client.run() forever and exits by signal, so
// no destructor would otherwise run. oc_apply_power_limit() calls this itself
// once a write has actually landed; exposed for tests.
void oc_install_restore_hooks();

// --no-oc-reset: leave the setting on the card at exit. Matches the reference miner,
// whose --no-oc-reset also defaults to 0 (i.e. restore).
void oc_set_restore_enabled(bool enabled);

// True while a setting is still on the card waiting to be put back. Lets a
// caller that temporarily owns SIGINT (the benchmark's graceful-stop handler)
// know whether to hand it back to us or to SIG_DFL.
bool oc_has_pending_restore();

// --- test seam -----------------------------------------------------------
//
// Everything above talks to the card through these two calls and nothing else.
// Without the seam the clamping, the status mapping and the restore bookkeeping
// could only be exercised on a machine with an NVIDIA GPU *and* root, which in
// practice meant they were not exercised at all -- mutation testing found nine
// separate breakages that no test noticed. Injecting a fake is what makes those
// testable on any machine; production leaves the defaults in place.
struct PowerOps {
    PowerLimit (*read)();
    NvmlWrite  (*write)(unsigned watts);
};
void oc_set_power_ops(const PowerOps& ops);   // tests only
void oc_reset_power_ops();                    // back to NVML
// Clears the recorded previous limit and re-enables restore, so one test case
// cannot leak state into the next.
void oc_reset_state_for_test();

}} // namespace mxbm::gpu
