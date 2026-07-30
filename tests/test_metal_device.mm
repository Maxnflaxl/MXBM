// Metal device enumeration and viability.
//
// The viability rule is the load-bearing part: a device that cannot host the full
// 2^25 seed layer must NOT claim the job, because a reduced layer does not mine
// slowly -- per budget_can_find_solutions it mines nothing at all. This test pins
// that MetalSolver asks the shared budget code rather than inventing its own rule.
#include "gpu/metal_solver.h"
#include "gpu/budget.h"
#include "check.h"

using namespace mxbm;

int main() {
    section("Metal device enumeration");

    auto devs = gpu::MetalSolver::enumerate();
    check(!devs.empty(), "at least one Metal device");
    if (devs.empty()) return summary("metal_device");

    const auto& d = devs[0];
    if (verbose())
        std::printf("     %s | %.2f GB working set | %u cores | viable=%d\n",
                    d.name.c_str(), d.global_mem / 1073741824.0,
                    d.compute_units, (int)d.viable);

    check(!d.name.empty(), "device reports a name");
    check(d.global_mem > (1ull << 32), "working set > 4 GB");
    check(d.index == 0, "first device is index 0");

    // Not asserted as non-zero: gpu-core-count comes from IOKit and a future OS could
    // stop publishing it. The contract is "0 or the truth", never a guess.
    check(d.compute_units < 256, "core count is plausible or zero");

    // Viability must agree with the shared budget rule, computed independently here.
    const gpu::Budget b = gpu::compute_budget(d.global_mem, d.global_mem, 0.85,
                                              gpu::kBytesPerElementRowbucket);
    check(d.viable == gpu::budget_can_find_solutions(b.elems_per_round),
          "viable matches budget_can_find_solutions");

    check(gpu::MetalSolver::available(0) == d.viable, "available(0) tracks viability");
    check(!gpu::MetalSolver::available(99), "out-of-range index is not available");

    // An out-of-range device_info is a zeroed struct, not a crash and not a lie.
    const auto bad = gpu::MetalSolver::device_info(99);
    check(bad.name.empty() && !bad.viable, "out-of-range device_info is empty");

    return summary("metal_device");
}
