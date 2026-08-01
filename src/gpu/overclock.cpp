#include "gpu/overclock.h"
#include "gpu/nvml.h"

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace mxbm { namespace gpu {
namespace {

// What to put back, and whether to bother -- PER DEVICE. Plain atomics in
// fixed arrays rather than a class or a vector: the signal handler below has
// to reach them, and a handler cannot safely walk an object graph it does not
// own -- a compile-time array it can. 16 is far above any rig this targets;
// an index past it is refused at apply time, never silently aliased.
//
// One flag per knob, because "0" is a legitimate previous value for an offset
// and a legitimate fan percent, so a sentinel value cannot mean "nothing to
// undo" the way g_prev_pl_w's zero can.
constexpr unsigned kMaxOcDevices = 16;
std::atomic<unsigned> g_prev_pl_w[kMaxOcDevices];    // 0 => nothing to restore
std::atomic<bool>     g_coff_applied[kMaxOcDevices];
std::atomic<int>      g_prev_coff[kMaxOcDevices];
std::atomic<bool>     g_moff_applied[kMaxOcDevices];
std::atomic<int>      g_prev_moff[kMaxOcDevices];
std::atomic<bool>     g_core_locked[kMaxOcDevices];
std::atomic<bool>     g_mem_locked[kMaxOcDevices];
std::atomic<bool>     g_fan_applied[kMaxOcDevices];
std::atomic<unsigned> g_fan_count[kMaxOcDevices];
std::atomic<bool>     g_restore_enabled{true};
std::atomic<bool>     g_atexit_installed{false};

// The calls that touch the card. Defaults are NVML; tests swap them.
PowerOps g_ops = { &nvml_power_limit, &nvml_set_power_limit };
ClockOps g_clk = {
    &nvml_core_clock_offset, &nvml_set_core_clock_offset,
    &nvml_mem_clock_offset,  &nvml_set_mem_clock_offset,
    &nvml_max_core_clock_mhz, &nvml_max_mem_clock_mhz,
    &nvml_set_locked_core_clock, &nvml_reset_locked_core_clock,
    &nvml_set_locked_mem_clock,  &nvml_reset_locked_mem_clock,
    &nvml_fans, &nvml_set_fan_speed, &nvml_reset_fan,
};

// Restore-then-default: the handler re-raises with the default disposition so
// the exit status still reads as "killed by SIGINT" rather than a clean 0.
// oc_restore() is idempotent, so the atexit hook running afterwards is fine.
//
// Calling NVML from a handler is not async-signal-safe, and there is no version
// of this that is: mining mode blocks in client.run() forever, so a flag for
// the main loop to notice would never be read. The alternative -- leaving a
// card capped, locked or with its fans pinned after Ctrl+C -- is the failure
// this exists to prevent, and it is the same trade the reference miner makes.
void oc_signal_handler(int sig) {
    oc_restore();
    std::signal(sig, SIG_DFL);
    std::raise(sig);
}

void oc_atexit_handler() { oc_restore(); }

// Every knob reports failure the same way, so one place decides the wording and
// no knob can invent a different name for "you need root".
void describe_failure(OcResult& r, NvmlWrite w, const char* unit) {
    char buf[256];
    switch (w) {
    case NvmlWrite::NoPermission:
        r.status = OcStatus::NoPermission;
        std::snprintf(buf, sizeof buf,
#ifdef _WIN32
            "%s %ld%s not applied: insufficient permission - re-run from an "
            "Administrator terminal. "
#else
            "%s %ld%s not applied: insufficient permission - re-run under sudo. "
#endif
            "Mining continues at the card's current setting.", r.knob, r.requested, unit);
        break;
    case NvmlWrite::Unsupported:
        r.status = OcStatus::Unsupported;
        std::snprintf(buf, sizeof buf,
            "%s %ld%s not applied: this device does not support it. Mining continues.",
            r.knob, r.requested, unit);
        break;
    default:
        r.status = OcStatus::Failed;
        std::snprintf(buf, sizeof buf,
            "%s %ld%s not applied: the driver rejected it. Mining continues.",
            r.knob, r.requested, unit);
        break;
    }
    r.message = buf;
}

// Reads one knob's entry out of a per-GPU list. Returns false when there is
// nothing to do -- either the flag was not given, or '*' skipped this GPU, or
// the list was malformed, in which case `r` already carries the complaint.
bool want(const std::string& spec, unsigned device, const char* knob,
          bool allow_negative, OcResult& r) {
    r.knob = knob;
    r.device = device;
    if (spec.empty()) return false;
    long v = 0;
    bool found = false;
    std::string err;
    if (!oc_parse_list(spec, device, v, found, err, allow_negative)) {
        r.status  = OcStatus::Failed;
        r.message = std::string(knob) + " " + spec + ": " + err;
        return false;
    }
    if (!found) return false;                     // '*' -- this GPU opted out
    r.requested = v;
    return true;
}

// Clamps to the band the DRIVER reports, never to a constant of ours. A band of
// 0/0 means the driver would not say, and then the value goes through untouched
// -- rejecting it on a guess would refuse settings the card actually allows.
bool clamp_to(long& v, long lo, long hi) {
    if (lo == 0 && hi == 0) return false;
    if (v < lo) { v = lo; return true; }
    if (v > hi) { v = hi; return true; }
    return false;
}

} // namespace

bool oc_parse_list(const std::string& spec, unsigned index, long& value, bool& found,
                   std::string& err, bool allow_negative) {
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
            size_t digits_from = 0;
            if (allow_negative && (tok[0] == '-' || tok[0] == '+')) digits_from = 1;
            if (tok.size() == digits_from) { err = "'" + tok + "' has no digits"; return false; }
            for (size_t i = digits_from; i < tok.size(); ++i) {
                if (tok[i] < '0' || tok[i] > '9') {
                    err = "'" + tok + "' is not a number or '*'";
                    return false;
                }
            }
            const long v = std::strtol(tok.c_str(), nullptr, 10);
            // Zero is a real setting for an offset (back to stock) and for a
            // fan, but not for a power limit or a locked clock, which is why
            // the same flag decides both.
            if (!allow_negative && v <= 0) {
                err = "'" + tok + "' must be greater than zero";
                return false;
            }
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

OcResult oc_apply_power_limit(unsigned device, const std::string& spec) {
    OcResult r;
    if (!want(spec, device, "--pl", /*allow_negative=*/false, r)) return r;
    if (device >= kMaxOcDevices) {
        r.status  = OcStatus::Failed;
        r.message = "--pl: device index out of range for restore bookkeeping";
        return r;
    }

    const PowerLimit pl = g_ops.read(device);
    if (!pl.valid) {
        r.status  = OcStatus::Unsupported;
        r.message = "--pl: this device does not report a power limit; continuing at stock";
        return r;
    }
    r.min = pl.min_w;
    r.max = pl.max_w;
    r.previous = pl.current_w;

    long target = r.requested;
    const bool clamped = clamp_to(target, pl.min_w, pl.max_w);
    r.applied = target;

    const NvmlWrite w = g_ops.write(device, (unsigned)target);
    if (w != NvmlWrite::Ok) {
        char buf[256];
        describe_failure(r, w, " W");
        std::snprintf(buf, sizeof buf, "%s Card stays at %u W.", r.message.c_str(),
                      pl.current_w);
        r.message = buf;
        return r;
    }
    // Only now is there something to undo. Recording the PREVIOUS value rather
    // than the default means a user who had already set a limit outside MXBM
    // gets their own value back, not the factory one.
    g_prev_pl_w[device].store(pl.current_w);
    oc_install_restore_hooks();

    char buf[256];
    if (clamped) {
        r.status = OcStatus::Clamped;
        std::snprintf(buf, sizeof buf,
            "Power limit: %ld W requested, applied %ld W (device allows %u-%u W)",
            r.requested, target, pl.min_w, pl.max_w);
    } else {
        r.status = OcStatus::Applied;
        std::snprintf(buf, sizeof buf, "Power limit: %ld W (was %u W, device allows %u-%u W)",
                      target, pl.current_w, pl.min_w, pl.max_w);
    }
    r.message = buf;
    return r;
}

namespace {

// --coff / --moff. The offset shifts the whole V/F curve, so the card keeps
// managing itself and the setting shows up as instability rather than as a
// refusal if it is too aggressive.
OcResult apply_offset(const std::string& spec, unsigned device, const char* knob,
                      ClockOffset (*read)(unsigned), NvmlWrite (*write)(unsigned, int),
                      std::atomic<bool>& applied_flag, std::atomic<int>& prev_slot,
                      const char* domain) {
    OcResult r;
    if (!want(spec, device, knob, /*allow_negative=*/true, r)) return r;

    const ClockOffset o = read(device);
    if (!o.valid) {
        r.status  = OcStatus::Unsupported;
        r.message = std::string(knob) + ": this device does not expose a " + domain
                  + " clock offset; continuing at stock";
        return r;
    }
    r.min = o.min_mhz;
    r.max = o.max_mhz;
    r.previous = o.current_mhz;

    long target = r.requested;
    const bool clamped = clamp_to(target, o.min_mhz, o.max_mhz);
    r.applied = target;

    const NvmlWrite w = write(device, (int)target);
    if (w != NvmlWrite::Ok) { describe_failure(r, w, " MHz"); return r; }

    prev_slot.store(o.current_mhz);
    applied_flag.store(true);
    oc_install_restore_hooks();

    char buf[256];
    if (clamped) {
        r.status = OcStatus::Clamped;
        std::snprintf(buf, sizeof buf,
            "%s clock offset: %+ld MHz requested, applied %+ld MHz (device allows %+d..%+d)",
            domain, r.requested, target, o.min_mhz, o.max_mhz);
    } else {
        r.status = OcStatus::Applied;
        // The RESULTING clock is not printed here because it has not settled
        // yet -- the statistics block reports it once mining starts, which is
        // also what settles the open question in docs/overclocking.md about
        // whether the memory offset is in clock or transfer-rate MHz.
        std::snprintf(buf, sizeof buf,
            "%s clock offset: %+ld MHz (was %+d, device allows %+d..%+d)",
            domain, target, o.current_mhz, o.min_mhz, o.max_mhz);
    }
    r.message = buf;
    return r;
}

// --cclk / --mclk. A lock takes clock management away from the driver, so
// there is no "previous lock" to write back and the restore is a reset call.
OcResult apply_lock(const std::string& spec, unsigned device, const char* knob,
                    unsigned (*max_mhz)(unsigned),
                    NvmlWrite (*lock)(unsigned, unsigned, unsigned),
                    std::atomic<bool>& locked_flag, const char* domain) {
    OcResult r;
    if (!want(spec, device, knob, /*allow_negative=*/false, r)) return r;

    const unsigned hi = max_mhz(device);
    r.max = hi;
    long target = r.requested;
    // Only an upper bound is known. NVML has no "minimum lockable clock", and
    // inventing one would refuse settings the card may well accept.
    const bool clamped = hi && target > (long)hi ? (target = hi, true) : false;
    r.applied = target;

    // Both ends of the range get the same value: "run at exactly this clock" is
    // what --cclk promises, and a range would let the driver pick within it.
    const NvmlWrite w = lock(device, (unsigned)target, (unsigned)target);
    if (w != NvmlWrite::Ok) { describe_failure(r, w, " MHz"); return r; }

    locked_flag.store(true);
    oc_install_restore_hooks();

    char buf[256];
    if (clamped) {
        r.status = OcStatus::Clamped;
        std::snprintf(buf, sizeof buf,
            "%s clock locked: %ld MHz requested, applied %ld MHz (device maximum %u)",
            domain, r.requested, target, hi);
    } else {
        r.status = OcStatus::Applied;
        std::snprintf(buf, sizeof buf, "%s clock locked: %ld MHz (device maximum %u)",
                      domain, target, hi);
    }
    r.message = buf;
    return r;
}

OcResult apply_fan(const std::string& spec, unsigned device) {
    OcResult r;
    if (!want(spec, device, "--fan", /*allow_negative=*/true, r)) return r;
    // Negative is allowed through the parser only so "-0" and a stray sign do
    // not read as a grammar error; an actual negative fan speed is nonsense.
    if (r.requested < 0) {
        r.status  = OcStatus::Failed;
        r.message = "--fan: a fan speed below 0 % is not a setting";
        return r;
    }

    const FanInfo f = g_clk.fan_read(device);
    if (!f.valid) {
        r.status  = OcStatus::Unsupported;
        r.message = "--fan: this device does not report a controllable fan; continuing on auto";
        return r;
    }
    r.min = f.min_pct;
    r.max = f.max_pct;
    r.previous = f.pct;

    long target = r.requested;
    bool clamped = clamp_to(target, f.min_pct, f.max_pct);
    if (target > 100) { target = 100; clamped = true; }
    r.applied = target;

    // Every fan, not just fan 0: a two-fan card with one of them still on the
    // driver's curve is a card whose cooling does not match what was asked for.
    NvmlWrite w = NvmlWrite::Ok;
    unsigned done = 0;
    for (unsigned i = 0; i < f.count; ++i) {
        w = g_clk.fan_write(device, i, (unsigned)target);
        if (w != NvmlWrite::Ok) break;
        ++done;
    }
    if (done == 0) { describe_failure(r, w, " %"); return r; }

    // Record however many actually took, so restore hands back exactly those.
    g_fan_count[device].store(done);
    g_fan_applied[device].store(true);
    oc_install_restore_hooks();

    char buf[256];
    if (w != NvmlWrite::Ok) {
        // Partial success on a multi-fan card: say so rather than claiming the
        // whole setting landed.
        r.status = OcStatus::Clamped;
        std::snprintf(buf, sizeof buf,
            "Fan: %ld %% applied to %u of %u fans; the rest stay on the driver's curve",
            target, done, f.count);
    } else if (clamped) {
        r.status = OcStatus::Clamped;
        std::snprintf(buf, sizeof buf,
            "Fan: %ld %% requested, applied %ld %% to %u fan(s) (device allows %u-%u %%)",
            r.requested, target, done, f.min_pct, f.max_pct);
    } else {
        r.status = OcStatus::Applied;
        std::snprintf(buf, sizeof buf, "Fan: %ld %% on %u fan(s) (was %u %%, device allows %u-%u %%)",
                      target, done, f.pct, f.min_pct, f.max_pct);
    }
    r.message = buf;
    return r;
}

} // namespace

std::vector<OcResult> oc_apply(unsigned device, const OcRequest& req) {
    std::vector<OcResult> out;
    auto keep = [&out](OcResult r) {
        // A knob that was never asked for produces no line at all; one that was
        // asked for and skipped by '*' does not either. Everything else is
        // reportable, including the failures -- especially the failures.
        if (r.status != OcStatus::NotRequested) out.push_back(std::move(r));
    };
    if (device >= kMaxOcDevices) {
        OcResult r;
        r.device  = device;
        r.status  = OcStatus::Failed;
        r.message = "overclock: device index out of range for restore bookkeeping";
        out.push_back(std::move(r));
        return out;
    }

    keep(oc_apply_power_limit(device, req.pl));
    keep(apply_offset(req.coff, device, "--coff", g_clk.core_offset_read, g_clk.core_offset_write,
                      g_coff_applied[device], g_prev_coff[device], "Core"));
    keep(apply_offset(req.moff, device, "--moff", g_clk.mem_offset_read, g_clk.mem_offset_write,
                      g_moff_applied[device], g_prev_moff[device], "Memory"));
    keep(apply_lock(req.cclk, device, "--cclk", g_clk.max_core_mhz, g_clk.lock_core,
                    g_core_locked[device], "Core"));
    keep(apply_lock(req.mclk, device, "--mclk", g_clk.max_mem_mhz, g_clk.lock_mem,
                    g_mem_locked[device], "Memory"));
    keep(apply_fan(req.fan, device));
    return out;
}

void oc_set_restore_enabled(bool enabled) { g_restore_enabled.store(enabled); }

void oc_set_power_ops(const PowerOps& ops) { g_ops = ops; }
void oc_set_clock_ops(const ClockOps& ops) { g_clk = ops; }
void oc_reset_power_ops() {
    g_ops = PowerOps{ &nvml_power_limit, &nvml_set_power_limit };
    g_clk = ClockOps{
        &nvml_core_clock_offset, &nvml_set_core_clock_offset,
        &nvml_mem_clock_offset,  &nvml_set_mem_clock_offset,
        &nvml_max_core_clock_mhz, &nvml_max_mem_clock_mhz,
        &nvml_set_locked_core_clock, &nvml_reset_locked_core_clock,
        &nvml_set_locked_mem_clock,  &nvml_reset_locked_mem_clock,
        &nvml_fans, &nvml_set_fan_speed, &nvml_reset_fan,
    };
}
void oc_reset_state_for_test() {
    for (unsigned d = 0; d < kMaxOcDevices; ++d) {
        g_prev_pl_w[d].store(0);
        g_coff_applied[d].store(false); g_prev_coff[d].store(0);
        g_moff_applied[d].store(false); g_prev_moff[d].store(0);
        g_core_locked[d].store(false);
        g_mem_locked[d].store(false);
        g_fan_applied[d].store(false);  g_fan_count[d].store(0);
    }
    g_restore_enabled.store(true);
}

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
    if (!g_restore_enabled.load()) return false;
    for (unsigned d = 0; d < kMaxOcDevices; ++d) {
        if (g_prev_pl_w[d].load() != 0 || g_coff_applied[d].load() || g_moff_applied[d].load()
            || g_core_locked[d].load() || g_mem_locked[d].load() || g_fan_applied[d].load())
            return true;
    }
    return false;
}

void oc_restore() {
    if (!g_restore_enabled.load()) return;
    // exchange, not load-then-store, on every knob: makes the whole thing
    // idempotent even if a signal arrives while atexit is already running it.
    //
    // Reverse of the apply order, per device. The fan goes back to the
    // driver's curve first so the card is cooling itself normally while the
    // clocks come down, and the power limit is restored last so nothing is
    // unlocked into a higher clock than the old limit would have allowed.
    for (unsigned d = 0; d < kMaxOcDevices; ++d) {
        if (g_fan_applied[d].exchange(false)) {
            const unsigned n = g_fan_count[d].exchange(0);
            for (unsigned i = 0; i < n; ++i) (void)g_clk.fan_reset(d, i);
        }
        if (g_mem_locked[d].exchange(false))  (void)g_clk.unlock_mem(d);
        if (g_core_locked[d].exchange(false)) (void)g_clk.unlock_core(d);
        if (g_moff_applied[d].exchange(false)) (void)g_clk.mem_offset_write(d, g_prev_moff[d].exchange(0));
        if (g_coff_applied[d].exchange(false)) (void)g_clk.core_offset_write(d, g_prev_coff[d].exchange(0));
        const unsigned prev = g_prev_pl_w[d].exchange(0);
        if (prev) (void)g_ops.write(d, prev);
    }
}

}} // namespace mxbm::gpu
