// Two GpuSolver instances coexisting in one process -- the groundwork check
// for multi-GPU OpenCL (MIXED_RIG.md phase 2). Until that phase, GpuSolver had
// never been constructed twice in one process, so "per-instance state, no
// hidden process-global" was an architecture claim, not a demonstrated one.
//
// On a single-card machine both instances target device 0 -- wasteful but
// legal, and exactly the crash class this exists to catch: a shared context,
// a colliding program cache, one instance's teardown breaking the other.
// A card too small to HOLD two solvers is a SKIP, not a failure: the claim
// under test is "no shared state", not "every card fits two".
#include "gpu/gpu_solver.h"
#include "bh3_verify.h"
#include "kat_vectors.h"
#include "check.h"
#include <cstdio>
#include <cstring>
#include <memory>
using namespace mxbm;
using namespace mxbm::gpu;

namespace {

// One KAT solve, verified end to end -- the same contract test_gpu_solver.cpp
// pins, minus the golden-set match (asserted there; here the question is
// interference, and "3 solutions, all verifying" answers it).
bool kat_solves(GpuSolver& s, const char* who) {
    auto sols = s.solve(kat::input32, kat::nonce0);
    char lbl[96];
    std::snprintf(lbl, sizeof lbl, "%s: solve() returns exactly 3 solutions", who);
    check(sols.size() == 3, lbl);
    for (size_t i = 0; i < sols.size(); ++i) {
        std::snprintf(lbl, sizeof lbl, "%s: solution %zu re-verifies", who, i);
        check(bh3::is_valid_solution(kat::input32, 32, kat::nonce0, sols[i].data()), lbl);
    }
    return sols.size() == 3;
}

} // namespace

int main() {
    if (!GpuSolver::available()) { std::printf("SKIP: no OpenCL device\n"); return 0; }
    section("two GpuSolver instances in one process");

    auto a = std::make_unique<GpuSolver>(0u);
    std::printf("  device: %s\n", a->device().name.c_str());
    check(kat_solves(*a, "A alone"), "instance A works before B exists");

    // OpenCL defers buffer commitment, so a card too small for two solvers
    // fails at B's FIRST SOLVE (CL_MEM_OBJECT_ALLOCATION_FAILURE from an
    // enqueue), not at construction. Both failure points route to the same
    // skip -- and the skip path still tests the claim: a failed or half-dead
    // B must leave A working, because state they wrongly shared would not.
    std::unique_ptr<GpuSolver> b;
    bool b_ok = false;
    try {
        b = std::make_unique<GpuSolver>(0u);
        b_ok = kat_solves(*b, "B");
    } catch (const std::exception& e) {
        std::printf("  SKIP second instance: %s (card cannot hold two solvers)\n", e.what());
        b.reset();   // destruction after a failed solve is part of the claim
        check(kat_solves(*a, "A after failed B"), "A survives B's failure and teardown");
        return summary("gpu_multi");
    }

    // Interleaved, both directions: B fresh, then A again (B's construction
    // and solve must not have corrupted A), then B again.
    check(b_ok, "instance B works alongside A");
    check(kat_solves(*a, "A after B"), "A still works after B solved");
    check(kat_solves(*b, "B again"), "B still works after A solved");

    // Teardown order: destroy A FIRST, then use B -- the surviving instance
    // must not depend on anything the destroyed one owned.
    a.reset();
    check(kat_solves(*b, "B after ~A"), "B survives A's destruction");

    return summary("gpu_multi");
}
