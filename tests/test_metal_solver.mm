// THE MILESTONE for the Metal backend: MetalSolver::solve() drives the full fused
// row-bucket pipeline (entry -> r1..r4 -> terminal -> recover) plus the CPU verify
// gate, and must return exactly the KAT goldens.
//
// kat::input32 / kat::nonce0 is known (tests/vectors/beamhash3-kat.md section 5) to
// have exactly 3 golden 104-byte solutions. This is a far stronger gate than "finds
// something": any error anywhere in the port -- a wrong record layout, a reversed
// lead/gi tie-break, a dropped collision, a mis-ordered recovery -- changes which
// solutions come out, and the goldens catch it.
//
// solve() is called exactly as a real miner would call it, through the miner::Solver
// interface, with no internal helper reached into directly.
#include "gpu/metal_solver.h"
#include "bh3_verify.h"
#include "kat_vectors.h"
#include "check.h"
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace mxbm;
using namespace mxbm::gpu;

static void verify_goldens(const std::vector<std::array<uint8_t, 104>>& sols,
                           const char* pass) {
    char lbl[128];
    std::snprintf(lbl, sizeof lbl, "%s: solve() returns EXACTLY 3 solutions", pass);
    check(sols.size() == 3, lbl);

    for (size_t i = 0; i < sols.size(); ++i) {
        // Re-checked from OUTSIDE MetalSolver: never short-circuit the gate by
        // trusting that solve() already applied it.
        const bool ok = bh3::is_valid_solution(kat::input32, 32, kat::nonce0, sols[i].data());
        std::snprintf(lbl, sizeof lbl, "%s: solution[%zu] independently passes the CPU gate",
                      pass, i);
        check(ok, lbl);
        if (!ok) show_hex("FAILED-gate solution", sols[i].data(), 104);
    }

    // Distinct-match: GPU atomics make survivor and scatter order nondeterministic,
    // so the RETURNED SET must equal the golden set, order-independently.
    bool matched[3] = {false, false, false};
    int unmatched = 0;
    for (size_t i = 0; i < sols.size(); ++i) {
        int hit = -1;
        for (int g = 0; g < 3; ++g) {
            if (matched[g]) continue;
            if (std::memcmp(sols[i].data(), kat::golden[g], 104) == 0) { hit = g; break; }
        }
        if (hit >= 0) matched[hit] = true;
        else { ++unmatched; show_hex("unmatched solution", sols[i].data(), 104); }
    }
    std::snprintf(lbl, sizeof lbl, "%s: every returned solution matches a distinct golden", pass);
    check(unmatched == 0, lbl);
    std::snprintf(lbl, sizeof lbl, "%s: all 3 KAT goldens found, verified, returned", pass);
    check(matched[0] && matched[1] && matched[2], lbl);
    std::printf("  %s: matched goldens [%d,%d,%d]\n", pass, matched[0], matched[1], matched[2]);
}

int main() {
    if (!MetalSolver::available()) { std::printf("SKIP: no Metal device\n"); return 0; }
    section("MetalSolver::solve(): fused pipeline + recovery + CPU gate -> the 3 KAT goldens");

    MetalSolver s;
    const auto& d = s.device();
    std::printf("  device: %s | %.2f GiB working set | %u cores\n",
                d.name.c_str(), d.global_mem / 1073741824.0, d.compute_units);

    std::printf("  solve #1 over the full 2^25 seed layer...\n");
    auto t0 = std::chrono::steady_clock::now();
    auto sols = s.solve(kat::input32, kat::nonce0);
    auto t1 = std::chrono::steady_clock::now();
    const double ms1 = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("  solve #1: %.1f ms\n", ms1);
    verify_goldens(sols, "solve #1");

    // Second solve on the SAME solver instance: buffers are allocated once in the
    // constructor and reused, so a stale counter or an unzeroed bucket array would
    // show up here and nowhere else.
    std::printf("  solve #2 (same instance, reused buffers)...\n");
    t0 = std::chrono::steady_clock::now();
    auto sols2 = s.solve(kat::input32, kat::nonce0);
    t1 = std::chrono::steady_clock::now();
    const double ms2 = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("  solve #2: %.1f ms\n", ms2);
    verify_goldens(sols2, "solve #2");

    return summary("metal_solver");
}
