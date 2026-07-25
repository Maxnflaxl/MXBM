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

PowerLimit fake_read() {
    PowerLimit p;
    p.valid = g_fake.valid; p.current_w = g_fake.current;
    p.default_w = g_fake.current; p.min_w = g_fake.min; p.max_w = g_fake.max;
    return p;
}
NvmlWrite fake_write(unsigned w) {
    if (g_fake.result == NvmlWrite::Ok) { g_writes.push_back(w); g_fake.current = w; }
    return g_fake.result;
}
void fake_reset(unsigned cur, unsigned lo, unsigned hi, NvmlWrite res) {
    g_fake = FakeCard{true, cur, lo, hi, res};
    g_writes.clear();
    oc_reset_state_for_test();
    const PowerOps ops = { &fake_read, &fake_write };
    oc_set_power_ops(ops);
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
        const OcResult r = oc_apply_power_limit("220");
        check(r.status == OcStatus::Applied, "in-band value applies");
        check(r.applied_w == 220, "the requested value is what is applied");
        check(g_writes.size() == 1 && g_writes[0] == 220, "220 W reached the device");
        check(r.previous_w == 285, "the previous limit is reported");
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
        const OcResult r = oc_apply_power_limit("50");
        check(r.status == OcStatus::Clamped, "below the band reports Clamped, not Applied");
        check(r.applied_w == 100 && g_writes.size() == 1 && g_writes[0] == 100,
              "clamped UP to the device minimum, and that is what was written");
        check(r.requested_w == 50, "the request is still reported, so the clamp is visible");
    }
    fake_reset(285, 100, 366, NvmlWrite::Ok);
    {
        const OcResult r = oc_apply_power_limit("500");
        check(r.status == OcStatus::Clamped, "above the band reports Clamped");
        check(r.applied_w == 366 && g_writes[0] == 366, "clamped DOWN to the device maximum");
    }

    section("a refused write leaves nothing to restore");
    fake_reset(285, 100, 366, NvmlWrite::NoPermission);
    {
        const OcResult r = oc_apply_power_limit("220");
        check(r.status == OcStatus::NoPermission, "no permission is reported as such");
        check(!oc_has_pending_restore(),
              "a write that never landed must not schedule a restore");
        oc_restore();
        check(g_writes.empty(), "restore after a failed apply writes nothing");
        check(r.message.find("sudo") != std::string::npos, "the message names the cause");
    }
    fake_reset(285, 100, 366, NvmlWrite::Unsupported);
    check(oc_apply_power_limit("220").status == OcStatus::Unsupported,
          "an unsupported knob is reported as such");
    fake_reset(285, 100, 366, NvmlWrite::Failed);
    check(oc_apply_power_limit("220").status == OcStatus::Failed, "a rejected write is reported");

    section("--no-oc-reset leaves the setting applied");
    fake_reset(285, 100, 366, NvmlWrite::Ok);
    {
        oc_set_restore_enabled(false);
        oc_apply_power_limit("220");
        check(g_writes.size() == 1, "the apply itself still happened");
        oc_restore();
        check(g_writes.size() == 1, "--no-oc-reset means exit writes nothing back");
        oc_set_restore_enabled(true);
    }

    section("'*' never touches the device");
    fake_reset(285, 100, 366, NvmlWrite::Ok);
    {
        const OcResult r = oc_apply_power_limit("*");
        check(r.status == OcStatus::NotRequested, "'*' is a skip, not an apply");
        check(g_writes.empty(), "'*' issues no write at all");
    }

    section("a device with no reported limit is refused, not guessed at");
    {
        g_writes.clear();
        g_fake = FakeCard{};                       // valid = false
        oc_reset_state_for_test();
        const OcResult r = oc_apply_power_limit("220");
        check(r.status == OcStatus::Unsupported, "no readable limit -> Unsupported");
        check(g_writes.empty(), "and no write is attempted");
    }

    section("milliwatt conversion rounds rather than truncates");
    check(nvml_mw_to_w(285000) == 285, "285000 mW -> 285 W");
    check(nvml_mw_to_w(284999) == 285, "284999 mW -> 285 W (truncation would give 284)");
    check(nvml_mw_to_w(100000) == 100, "100000 mW -> 100 W");
    check(nvml_mw_to_w(220400) == 220, "220400 mW -> 220 W");

    oc_reset_power_ops();

    section("device power limit (read-only)");
    if (!nvml_init()) {
        std::printf("  SKIP: NVML unavailable\n");
        return summary("overclock");
    }
    const PowerLimit pl = nvml_power_limit();
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
    return summary("overclock");
}
