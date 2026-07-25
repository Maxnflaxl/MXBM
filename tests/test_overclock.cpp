// Board power limit: the per-GPU list grammar, and a read-only look at what the
// installed card reports. Applying a limit needs root, so nothing here writes --
// the write paths are exercised by hand under sudo (see docs/overclocking.md).
#include "gpu/overclock.h"
#include "gpu/nvml.h"
#include "check.h"
#include <cstdio>
#include <string>
using namespace mxbm;
using namespace mxbm::gpu;

namespace {

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

    section("restore is idempotent with nothing applied");
    // Nothing was applied, so these must be no-ops rather than writes of 0 W.
    // If this ever starts needing root, that is the bug it exists to catch.
    oc_set_restore_enabled(true);
    oc_restore();
    oc_restore();
    check(true, "oc_restore() with no prior apply does nothing");

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
