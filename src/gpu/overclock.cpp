#include "gpu/overclock.h"
#include "gpu/nvml.h"

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace mxbm { namespace gpu {
namespace {

// What to put back, and whether to bother. Plain globals rather than a class:
// the signal handler below has to reach them, and a handler cannot safely walk
// an object graph it does not own.
std::atomic<unsigned> g_prev_pl_w{0};        // 0 => nothing to restore
std::atomic<bool>     g_restore_enabled{true};
std::atomic<bool>     g_atexit_installed{false};

// The only two calls that touch the card. Defaults are NVML; tests swap them.
PowerOps g_ops = { &nvml_power_limit, &nvml_set_power_limit };

// Restore-then-default: the handler re-raises with the default disposition so
// the exit status still reads as "killed by SIGINT" rather than a clean 0.
// oc_restore() is idempotent, so the atexit hook running afterwards is fine.
//
// Calling NVML from a handler is not async-signal-safe, and there is no version
// of this that is: mining mode blocks in client.run() forever, so a flag for
// the main loop to notice would never be read. The alternative -- leaving a
// card capped after Ctrl+C -- is the failure this exists to prevent, and it is
// the same trade the reference miner makes.
void oc_signal_handler(int sig) {
    oc_restore();
    std::signal(sig, SIG_DFL);
    std::raise(sig);
}

void oc_atexit_handler() { oc_restore(); }

} // namespace

bool oc_parse_list(const std::string& spec, unsigned index, long& value, bool& found,
                   std::string& err) {
    found = false;
    err.clear();
    if (spec.empty()) { err = "empty value"; return false; }

    unsigned n = 0;
    size_t pos = 0;
    for (;;) {
        const size_t comma = spec.find(',', pos);
        std::string tok = spec.substr(pos, comma == std::string::npos ? std::string::npos
                                                                     : comma - pos);
        // Trim: "240, 260" is the natural thing to type and the reference miner accepts it.
        while (!tok.empty() && (tok.front() == ' ' || tok.front() == '\t')) tok.erase(0, 1);
        while (!tok.empty() && (tok.back()  == ' ' || tok.back()  == '\t')) tok.pop_back();

        if (tok.empty()) { err = "empty entry in list"; return false; }
        if (tok != "*") {
            for (char c : tok) {
                if (c < '0' || c > '9') {
                    err = "'" + tok + "' is not a number or '*'";
                    return false;
                }
            }
            const long v = std::strtol(tok.c_str(), nullptr, 10);
            if (v <= 0) { err = "'" + tok + "' must be greater than zero"; return false; }
            if (n == index) { value = v; found = true; }
        }
        ++n;
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    // A list SHORTER than the GPU count leaves later GPUs at stock rather than
    // reusing the last entry: --pool's carry-forward rule is a convenience,
    // whereas silently applying one card's power limit to another is a
    // hardware setting nobody asked for.
    return true;
}

OcResult oc_apply_power_limit(const std::string& spec) {
    OcResult r;
    if (spec.empty()) return r;

    long want = 0;
    bool found = false;
    std::string err;
    if (!oc_parse_list(spec, 0, want, found, err)) {
        r.status  = OcStatus::Failed;
        r.message = "--pl " + spec + ": " + err;
        return r;
    }
    if (!found) return r;                         // '*' -- this GPU opted out
    r.requested_w = (unsigned)want;

    const PowerLimit pl = g_ops.read();
    if (!pl.valid) {
        r.status  = OcStatus::Unsupported;
        r.message = "--pl: this device does not report a power limit; continuing at stock";
        return r;
    }
    r.min_w = pl.min_w;
    r.max_w = pl.max_w;
    r.previous_w = pl.current_w;

    // Clamp to the DRIVER's band, not to a constant of ours. Reported either
    // way -- a clamped value that looked applied would misattribute every
    // measurement taken after it.
    unsigned target = r.requested_w;
    bool clamped = false;
    if (pl.min_w && target < pl.min_w) { target = pl.min_w; clamped = true; }
    if (pl.max_w && target > pl.max_w) { target = pl.max_w; clamped = true; }
    r.applied_w = target;

    char buf[256];
    const NvmlWrite w = g_ops.write(target);
    switch (w) {
    case NvmlWrite::Ok:
        // Only now is there something to undo. Recording the PREVIOUS value
        // rather than the default means a user who had already set a limit
        // outside MXBM gets their own value back, not the factory one.
        g_prev_pl_w.store(pl.current_w);
        oc_install_restore_hooks();
        if (clamped) {
            r.status = OcStatus::Clamped;
            std::snprintf(buf, sizeof buf,
                "Power limit: %u W requested, applied %u W (device allows %u-%u W)",
                r.requested_w, target, pl.min_w, pl.max_w);
        } else {
            r.status = OcStatus::Applied;
            std::snprintf(buf, sizeof buf, "Power limit: %u W (was %u W, device allows %u-%u W)",
                          target, pl.current_w, pl.min_w, pl.max_w);
        }
        break;
    case NvmlWrite::NoPermission:
        r.status = OcStatus::NoPermission;
        std::snprintf(buf, sizeof buf,
            "Power limit %u W not applied: insufficient permission - re-run under sudo. "
            "Mining continues at the card's current %u W.", r.requested_w, pl.current_w);
        break;
    case NvmlWrite::Unsupported:
        r.status = OcStatus::Unsupported;
        std::snprintf(buf, sizeof buf,
            "Power limit %u W not applied: this device does not support setting it. "
            "Mining continues at %u W.", r.requested_w, pl.current_w);
        break;
    default:
        r.status = OcStatus::Failed;
        std::snprintf(buf, sizeof buf,
            "Power limit %u W not applied: the driver rejected it. Mining continues at %u W.",
            r.requested_w, pl.current_w);
        break;
    }
    r.message = buf;
    return r;
}

void oc_set_restore_enabled(bool enabled) { g_restore_enabled.store(enabled); }

void oc_set_power_ops(const PowerOps& ops) { g_ops = ops; }
void oc_reset_power_ops() { g_ops = PowerOps{ &nvml_power_limit, &nvml_set_power_limit }; }
void oc_reset_state_for_test() { g_prev_pl_w.store(0); g_restore_enabled.store(true); }

void oc_install_restore_hooks() {
    // atexit once -- it cannot be unregistered, and running the same handler
    // twice would be harmless but pointless. The signal handlers are re-armable
    // on purpose: the benchmark borrows SIGINT for its graceful stop and hands
    // it back here afterwards, so the window between the two is not a hole.
    if (!g_atexit_installed.exchange(true)) std::atexit(oc_atexit_handler);
    std::signal(SIGINT,  oc_signal_handler);
    std::signal(SIGTERM, oc_signal_handler);
}

bool oc_has_pending_restore() {
    return g_restore_enabled.load() && g_prev_pl_w.load() != 0;
}

void oc_restore() {
    if (!g_restore_enabled.load()) return;
    // exchange, not load-then-store: makes the whole thing idempotent even if
    // a signal arrives while atexit is already running it.
    const unsigned prev = g_prev_pl_w.exchange(0);
    if (prev) (void)g_ops.write(prev);
}

}} // namespace mxbm::gpu
