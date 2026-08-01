// Board power limit: the list grammar, the apply/clamp/restore logic against a
// fake card, and a read-only look at what the installed card reports. Nothing
// here writes to real hardware.
//
// Scope was set by mutation testing rather than by guesswork. An earlier version
// covered only the parser and passed while NINE separate breakages of the clamp,
// the status mapping and the restore bookkeeping went unnoticed -- they were
// unreachable because oc_apply_power_limit() talked to NVML directly. The
// PowerOps seam exists for that reason, and every case below corresponds to a
// mutation that now fails.
//
// ONE known gap remains, deliberately: the watts-to-milliwatts conversion on the
// real nvmlDeviceSetPowerManagementLimit call cannot be reached without an
// NVIDIA device, since it lives past the dlsym boundary. It is verified by hand
// under sudo (docs/overclocking.md, "Verified: --pl end to end") and its failure
// mode is loud rather than silent -- 220 instead of 220000 is far below any
// card's minimum, so NVML rejects it and MXBM reports a failed apply.
#include "gpu/overclock.h"
#include "gpu/nvml.h"
#include "check.h"
#include <cstdarg>
#include <cstdio>
#include <string>
#include <vector>
using namespace mxbm;
using namespace mxbm::gpu;

namespace {

// A fake card. Everything the miner does to a power limit goes through the two
// calls in PowerOps, so recording them is enough to check what really happened.
struct FakeCard { bool valid=false; unsigned current=0, min=0, max=0; NvmlWrite result=NvmlWrite::Ok; };
FakeCard g_fake;
std::vector<unsigned> g_writes;
std::vector<unsigned> g_write_devs;   // which device each write (incl. restore) hit

PowerLimit fake_read(unsigned) {
    PowerLimit p;
    p.valid = g_fake.valid; p.current_w = g_fake.current;
    p.default_w = g_fake.current; p.min_w = g_fake.min; p.max_w = g_fake.max;
    return p;
}
NvmlWrite fake_write(unsigned dev, unsigned w) {
    if (g_fake.result == NvmlWrite::Ok) {
        g_writes.push_back(w); g_write_devs.push_back(dev); g_fake.current = w;
    }
    return g_fake.result;
}
void fake_reset(unsigned cur, unsigned lo, unsigned hi, NvmlWrite res) {
    g_fake = FakeCard{true, cur, lo, hi, res};
    g_writes.clear();
    g_write_devs.clear();
    oc_reset_state_for_test();
    const PowerOps ops = { &fake_read, &fake_write };
    oc_set_power_ops(ops);
}

// The clock/fan half of the same fake. Every call appends to one log string,
// so a test asserts on the exact sequence of writes -- which is the only way to
// catch a restore that puts things back in the wrong ORDER, or that writes a
// value back where it should have issued a reset.
std::string g_clk_log;
NvmlWrite   g_clk_fail = NvmlWrite::Ok;
int g_coff_cur = 0, g_moff_cur = 0;
unsigned g_fan_cur = 45;

void logf(const char* fmt, ...) {
    char buf[64];
    va_list ap; va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    g_clk_log += buf;
}

ClockOffset fk_coff_read(unsigned) { ClockOffset o; o.valid = true; o.current_mhz = g_coff_cur;
                             o.min_mhz = -1000; o.max_mhz = 1000; return o; }
ClockOffset fk_moff_read(unsigned) { ClockOffset o; o.valid = true; o.current_mhz = g_moff_cur;
                             o.min_mhz = -2000; o.max_mhz = 6000; return o; }
NvmlWrite fk_coff_write(unsigned, int v) { if (g_clk_fail != NvmlWrite::Ok) return g_clk_fail;
                                 logf("coff=%d;", v); g_coff_cur = v; return NvmlWrite::Ok; }
NvmlWrite fk_moff_write(unsigned, int v) { if (g_clk_fail != NvmlWrite::Ok) return g_clk_fail;
                                 logf("moff=%d;", v); g_moff_cur = v; return NvmlWrite::Ok; }
unsigned fk_max_core(unsigned) { return 3150; }
unsigned fk_max_mem(unsigned)  { return 10501; }
NvmlWrite fk_lock_core(unsigned, unsigned lo, unsigned hi) { if (g_clk_fail != NvmlWrite::Ok) return g_clk_fail;
                                                   logf("lockcore=%u,%u;", lo, hi); return NvmlWrite::Ok; }
NvmlWrite fk_lock_mem(unsigned, unsigned lo, unsigned hi)  { if (g_clk_fail != NvmlWrite::Ok) return g_clk_fail;
                                                   logf("lockmem=%u,%u;", lo, hi); return NvmlWrite::Ok; }
NvmlWrite fk_unlock_core(unsigned) { logf("unlockcore;"); return NvmlWrite::Ok; }
NvmlWrite fk_unlock_mem(unsigned)  { logf("unlockmem;");  return NvmlWrite::Ok; }
FanInfo fk_fan_read(unsigned) { FanInfo f; f.valid = true; f.count = 2; f.pct = g_fan_cur;
                        f.min_pct = 30; f.max_pct = 100; return f; }
NvmlWrite fk_fan_write(unsigned, unsigned i, unsigned p) { if (g_clk_fail != NvmlWrite::Ok) return g_clk_fail;
                                                 logf("fan%u=%u;", i, p); return NvmlWrite::Ok; }
NvmlWrite fk_fan_reset(unsigned, unsigned i) { logf("fanauto%u;", i); return NvmlWrite::Ok; }

void fake_clock_reset() {
    g_clk_log.clear();
    g_coff_cur = 0; g_moff_cur = 0; g_fan_cur = 45;
    oc_reset_state_for_test();
    const ClockOps ops = {
        &fk_coff_read, &fk_coff_write, &fk_moff_read, &fk_moff_write,
        &fk_max_core, &fk_max_mem,
        &fk_lock_core, &fk_unlock_core, &fk_lock_mem, &fk_unlock_mem,
        &fk_fan_read, &fk_fan_write, &fk_fan_reset,
    };
    oc_set_clock_ops(ops);
}

// Parses `spec` for `index` and reports the outcome as one comparable string:
// "240" applied, "-" skipped, "!" rejected. Keeps each case to one line.
std::string parse_as(const std::string& spec, unsigned index) {
    long v = 0; bool found = false; std::string err;
    if (!oc_parse_list(spec, index, v, found, err)) return "!";
    return found ? std::to_string(v) : "-";
}

void expect(const std::string& spec, unsigned index, const char* want) {
    const std::string got = parse_as(spec, index);
    char msg[128];
    std::snprintf(msg, sizeof msg, "--pl \"%s\"[%u] -> %s", spec.c_str(), index, want);
    if (got != want) std::printf("  (got \"%s\")\n", got.c_str());
    check(got == want, msg);
}

} // namespace

int main() {
    section("--pl list grammar");
    expect("240", 0, "240");
    expect("240", 1, "-");            // a short list leaves later GPUs alone,
                                      // rather than carrying the last value
                                      // forward onto hardware nobody named
    expect("*", 0, "-");
    expect("240,*,260", 0, "240");
    expect("240,*,260", 1, "-");
    expect("240,*,260", 2, "260");
    expect("240,*,260", 3, "-");
    expect(" 240 , 260 ", 0, "240");  // spaces are natural to type
    expect(" 240 , 260 ", 1, "260");
    expect("366", 0, "366");          // no ceiling here: the DEVICE's band decides

    section("--pl rejects malformed input");
    expect("", 0, "!");
    expect("abc", 0, "!");
    expect("240,,260", 0, "!");
    expect("-5", 0, "!");
    expect("0", 0, "!");              // zero watts is not a power limit
    expect("240w", 0, "!");
    expect("2.5", 0, "!");
    expect(",240", 0, "!");
    expect("240,", 0, "!");
    // A malformed entry is rejected even when it is not the one being read,
    // so a typo in GPU 3's slot cannot pass unnoticed on a single-GPU rig.
    expect("240,abc", 0, "!");

    // --- apply / clamp / restore, against a fake card -------------------
    //
    // These were unreachable until oc_set_power_ops() existed, and mutation
    // testing confirmed the consequence: nine separate breakages of the clamp,
    // the status mapping and the restore bookkeeping went unnoticed. The fake
    // records every write, so what reached "the card" is checked directly
    // rather than inferred.
    section("apply, clamp and restore against a fake device");
    fake_reset(285, 100, 366, NvmlWrite::Ok);
    {
        const OcResult r = oc_apply_power_limit(0, "220");
        check(r.status == OcStatus::Applied, "in-band value applies");
        check(r.applied == 220, "the requested value is what is applied");
        check(g_writes.size() == 1 && g_writes[0] == 220, "220 W reached the device");
        check(r.previous == 285, "the previous limit is reported");
        check(oc_has_pending_restore(), "a restore is now pending");
        oc_restore();
        check(g_writes.size() == 2 && g_writes[1] == 285,
              "restore puts back the PREVIOUS limit, not the device default");
        check(!oc_has_pending_restore(), "nothing pending after a restore");
        oc_restore();
        check(g_writes.size() == 2, "restore is idempotent -- the second call writes nothing");
    }

    fake_reset(285, 100, 366, NvmlWrite::Ok);
    {
        const OcResult r = oc_apply_power_limit(0, "50");
        check(r.status == OcStatus::Clamped, "below the band reports Clamped, not Applied");
        check(r.applied == 100 && g_writes.size() == 1 && g_writes[0] == 100,
              "clamped UP to the device minimum, and that is what was written");
        check(r.requested == 50, "the request is still reported, so the clamp is visible");
    }
    fake_reset(285, 100, 366, NvmlWrite::Ok);
    {
        const OcResult r = oc_apply_power_limit(0, "500");
        check(r.status == OcStatus::Clamped, "above the band reports Clamped");
        check(r.applied == 366 && g_writes[0] == 366, "clamped DOWN to the device maximum");
    }

    section("a refused write leaves nothing to restore");
    fake_reset(285, 100, 366, NvmlWrite::NoPermission);
    {
        const OcResult r = oc_apply_power_limit(0, "220");
        check(r.status == OcStatus::NoPermission, "no permission is reported as such");
        check(!oc_has_pending_restore(),
              "a write that never landed must not schedule a restore");
        oc_restore();
        check(g_writes.empty(), "restore after a failed apply writes nothing");
        check(r.message.find("sudo") != std::string::npos, "the message names the cause");
    }
    fake_reset(285, 100, 366, NvmlWrite::Unsupported);
    check(oc_apply_power_limit(0, "220").status == OcStatus::Unsupported,
          "an unsupported knob is reported as such");
    fake_reset(285, 100, 366, NvmlWrite::Failed);
    check(oc_apply_power_limit(0, "220").status == OcStatus::Failed, "a rejected write is reported");

    section("--no-oc-reset leaves the setting applied");
    fake_reset(285, 100, 366, NvmlWrite::Ok);
    {
        oc_set_restore_enabled(false);
        oc_apply_power_limit(0, "220");
        check(g_writes.size() == 1, "the apply itself still happened");
        oc_restore();
        check(g_writes.size() == 1, "--no-oc-reset means exit writes nothing back");
        oc_set_restore_enabled(true);
    }

    section("'*' never touches the device");
    fake_reset(285, 100, 366, NvmlWrite::Ok);
    {
        const OcResult r = oc_apply_power_limit(0, "*");
        check(r.status == OcStatus::NotRequested, "'*' is a skip, not an apply");
        check(g_writes.empty(), "'*' issues no write at all");
    }

    section("a device with no reported limit is refused, not guessed at");
    {
        g_writes.clear();
        g_fake = FakeCard{};                       // valid = false
        oc_reset_state_for_test();
        const OcResult r = oc_apply_power_limit(0, "220");
        check(r.status == OcStatus::Unsupported, "no readable limit -> Unsupported");
        check(g_writes.empty(), "and no write is attempted");
    }

    section("milliwatt conversion rounds rather than truncates");
    check(nvml_mw_to_w(285000) == 285, "285000 mW -> 285 W");
    check(nvml_mw_to_w(284999) == 285, "284999 mW -> 285 W (truncation would give 284)");
    check(nvml_mw_to_w(100000) == 100, "100000 mW -> 100 W");
    check(nvml_mw_to_w(220400) == 220, "220400 mW -> 220 W");

    // --- the clock and fan knobs ----------------------------------------
    //
    // Same discipline as above: a fake card records every write, so what
    // reached the device is checked rather than inferred. These knobs restore
    // in two different ways -- an offset is written back, a lock is RESET --
    // and getting that backwards leaves a card pinned after exit, which is the
    // failure the whole restore path exists to prevent.
    section("clock offsets, locks and fans against a fake device");
    fake_clock_reset();
    {
        OcRequest req; req.coff = "150"; req.moff = "-200";
        const std::vector<OcResult> rs = oc_apply(0, req);
        check(rs.size() == 2, "one result per requested knob, none for the rest");
        check(rs[0].status == OcStatus::Applied && rs[0].applied == 150, "core offset applied");
        check(rs[1].status == OcStatus::Applied && rs[1].applied == -200,
              "a NEGATIVE memory offset is a downclock, not a parse error");
        check(g_clk_log == "coff=150;moff=-200;", "both offsets reached the device in order");
        oc_restore();
        check(g_clk_log == "coff=150;moff=-200;moff=0;coff=0;",
              "restore writes the PREVIOUS offsets back, in reverse order");
    }

    fake_clock_reset();
    {
        OcRequest req; req.cclk = "2100"; req.mclk = "10000";
        const std::vector<OcResult> rs = oc_apply(0, req);
        check(rs.size() == 2 && rs[0].applied == 2100 && rs[1].applied == 10000,
              "both clocks locked at the requested values");
        check(g_clk_log == "lockcore=2100,2100;lockmem=10000,10000;",
              "a lock pins BOTH ends of the range -- a range would let the driver choose");
        oc_restore();
        check(g_clk_log == "lockcore=2100,2100;lockmem=10000,10000;unlockmem;unlockcore;",
              "a lock is undone by a RESET, not by writing an old value back");
    }

    fake_clock_reset();
    {
        OcRequest req; req.cclk = "9000";              // above the fake's 3150 max
        const std::vector<OcResult> rs = oc_apply(0, req);
        check(rs[0].status == OcStatus::Clamped && rs[0].applied == 3150,
              "a locked clock is clamped to the device maximum");
        check(rs[0].requested == 9000, "and the request is still reported, so the clamp shows");
    }

    fake_clock_reset();
    {
        OcRequest req; req.coff = "5000";              // above the fake's +1000
        const std::vector<OcResult> rs = oc_apply(0, req);
        check(rs[0].status == OcStatus::Clamped && rs[0].applied == 1000,
              "an offset is clamped to the band the driver reports");
        OcRequest low; low.coff = "-5000";
        fake_clock_reset();
        check(oc_apply(0, low)[0].applied == -1000, "and clamped at the bottom of it too");
    }

    section("--fan drives every fan the card has");
    fake_clock_reset();
    {
        OcRequest req; req.fan = "70";
        const std::vector<OcResult> rs = oc_apply(0, req);
        check(rs[0].status == OcStatus::Applied && rs[0].applied == 70, "fan target applied");
        check(g_clk_log == "fan0=70;fan1=70;",
              "BOTH fans are set -- one left on the driver's curve is not the setting asked for");
        oc_restore();
        check(g_clk_log == "fan0=70;fan1=70;fanauto0;fanauto1;",
              "restore hands the fans back to AUTO, not to the percentage they happened to read");
    }
    fake_clock_reset();
    {
        OcRequest req; req.fan = "10";                 // below the fake's 30 % floor
        check(oc_apply(0, req)[0].applied == 30, "fan speed clamps up to the device minimum");
    }

    section("a knob that fails leaves nothing of ITS OWN to restore");
    fake_clock_reset();
    g_clk_fail = NvmlWrite::NoPermission;
    {
        OcRequest req; req.coff = "150";
        const std::vector<OcResult> rs = oc_apply(0, req);
        check(rs[0].status == OcStatus::NoPermission, "no permission is reported as such");
        check(rs[0].message.find("sudo") != std::string::npos, "the message names the cause");
        check(!oc_has_pending_restore(), "a write that never landed schedules no restore");
        oc_restore();
        check(g_clk_log.empty(), "and restore writes nothing");
    }
    g_clk_fail = NvmlWrite::Ok;

    section("'*' and absence both leave the clocks alone");
    fake_clock_reset();
    {
        OcRequest req; req.coff = "*"; req.cclk = "";
        check(oc_apply(0, req).empty(), "'*' and an unset flag produce no results at all");
        check(g_clk_log.empty(), "and issue no writes");
    }

    section("offsets accept a leading sign, and reject a bare one");
    {
        long v = 0; bool found = false; std::string err;
        check(oc_parse_list("+300", 0, v, found, err, true) && found && v == 300,
              "an explicit + is accepted on an offset");
        check(oc_parse_list("-300", 0, v, found, err, true) && found && v == -300,
              "and a - gives a negative offset");
        check(oc_parse_list("0", 0, v, found, err, true) && found && v == 0,
              "0 is a real offset -- back to stock -- where it is not a real power limit");
        check(!oc_parse_list("-", 0, v, found, err, true), "a bare sign has no digits");
        check(!oc_parse_list("-300", 0, v, found, err, false),
              "a negative is still rejected where negatives are meaningless (--pl, --cclk)");
    }

    oc_reset_power_ops();

    section("device power limit (read-only)");
    if (!nvml_init()) {
        std::printf("  SKIP: NVML unavailable\n");
        return summary("overclock");
    }
    const PowerLimit pl = nvml_power_limit(0);
    if (!pl.valid) {
        std::printf("  SKIP: device reports no power limit\n");
        nvml_shutdown();
        return summary("overclock");
    }
    if (verbose())
        std::printf("  current %u W, default %u W, band %u-%u W\n",
                    pl.current_w, pl.default_w, pl.min_w, pl.max_w);
    check(pl.current_w > 0, "current limit is positive");
    check(pl.min_w > 0 && pl.max_w >= pl.min_w, "driver reports a usable min<=max band");
    check(pl.current_w >= pl.min_w && pl.current_w <= pl.max_w,
          "current limit lies inside the band the driver reports");
    check(pl.default_w == 0 || (pl.default_w >= pl.min_w && pl.default_w <= pl.max_w),
          "default limit, when reported, lies inside the band");
    nvml_shutdown();
    section("per-device apply and restore");
    {
        // The write AND the restore must land on the card that was applied to,
        // and the per-GPU list entry must be read at that card's own position
        // -- the two halves of MIXED_RIG.md phase 0.
        fake_reset(285, 100, 285, NvmlWrite::Ok);
        const OcResult r = oc_apply_power_limit(2, "240,*,250");
        check(r.status == OcStatus::Applied && r.applied == 250,
              "device 2 reads its own list entry, not device 0's");
        check(r.device == 2, "the result names the device it landed on");
        check(g_write_devs.size() == 1 && g_write_devs[0] == 2,
              "the write went to device 2");
        oc_restore();
        check(g_write_devs.size() == 2 && g_write_devs[1] == 2,
              "the restore went back to device 2, nobody else");
        check(g_writes[1] == 285, "and put back device 2's previous value");
    }
    {
        // Two cards applied, both restored -- one visit each.
        fake_reset(285, 100, 285, NvmlWrite::Ok);
        (void)oc_apply_power_limit(0, "200,220");
        (void)oc_apply_power_limit(1, "200,220");
        oc_restore();
        check(g_write_devs.size() == 4 && g_write_devs[2] != g_write_devs[3],
              "restore visits each applied device exactly once");
    }

    return summary("overclock");
}
