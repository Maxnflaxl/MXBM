// Apple Silicon telemetry.
//
// The contract worth pinning is mostly about what is DELIBERATELY absent. Power,
// clocks, temperature and fan have no public API on Apple Silicon, and the design
// decision is that they stay unavailable rather than being scraped out of a private
// framework whose layout changes between OS releases. A later change that starts
// filling those fields from IOReport would be a real decision, not a tidy-up, and
// this test is what makes it a visible one.
#include "gpu/metal_telemetry.h"
#include "check.h"

using namespace mxbm;

int main() {
    section("Metal telemetry: utilization only, by design");

    if (!gpu::metal_telemetry_init()) {
        // Not a failure: a machine without a readable IOAccelerator is a machine
        // where the miner correctly shows blank telemetry columns.
        std::printf("  SKIP: IOAccelerator not readable\n");
        return summary("metal_telemetry");
    }

    const gpu::Telemetry t = gpu::metal_sample(0);

    check(t.have_util, "utilization is available");
    check(t.util_pct <= 100, "utilization is a percentage");
    check(t.any(), "Telemetry::any() is true when only utilization is present");

    // The absent-by-design fields. If one of these ever starts reporting, the person
    // who made that happen should have to change this test on purpose.
    check(!t.have_power, "power is NOT reported (no public API)");
    check(!t.have_sm,    "core clock is NOT reported (no public API)");
    check(!t.have_mem,   "memory clock is NOT reported (no public API)");
    check(!t.have_temp,  "temperature is NOT reported (no public API)");
    check(!t.have_fan,   "fan is NOT reported (no public API)");

    // Apple Silicon has one integrated GPU. A per-device row showing the same card's
    // numbers under a different index is worse than a blank one.
    const gpu::Telemetry t1 = gpu::metal_sample(1);
    check(!t1.any(), "index 1 reports nothing rather than device 0's numbers");
    check(gpu::metal_memory_in_use(1) == 0, "index 1 reports no memory either");

    const uint64_t mem = gpu::metal_memory_in_use(0);
    check(mem > 0, "driver memory in use is readable");

    if (verbose())
        std::printf("     utilization %u%%, driver memory in use %.2f GiB\n",
                    t.util_pct, mem / 1073741824.0);

    return summary("metal_telemetry");
}
