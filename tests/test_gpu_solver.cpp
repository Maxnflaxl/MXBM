// Task 2 of M3 Phase C -- THE MILESTONE: GpuSolver wraps run_pipeline (the
// 5-round Wagner search) + recover_candidates (C-T1's on-device back-ref
// walk -> 104-byte candidates) + the CPU verify gate (bh3::is_valid_solution,
// which applies distinct + indexAfter -- the checks the GPU rounds
// deliberately skip) behind the miner::Solver interface. This is the
// Phase-C exit criterion: a real GPU solver, driven purely through solve(),
// producing verified, correctly-encoded, protocol-conformant solutions --
// not raw recovered candidates (tests/test_gpu_recover.cpp's job), but
// candidates that have survived the SAME gate a submitted share must pass.
//
// kat::input32/kat::nonce0 is known (tests/vectors/beamhash3-kat.md section
// 5) to have exactly 3 golden 104-byte solutions. GpuSolver::solve() is
// called exactly as a real miner would call it -- no internal helpers
// reached into directly -- over the full seed layer on real device
// hardware; every returned candidate must independently pass
// bh3::is_valid_solution, and the returned SET (order-independent -- GPU
// atomics make survivor/scatter order nondeterministic) must equal the set
// of all 3 known goldens, byte-for-byte.
#include "gpu/gpu_solver.h"
#include "bh3_verify.h"
#include "kat_vectors.h"
#include "check.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <vector>
using namespace mxbm;
using namespace mxbm::gpu;

int main() {
    if (!GpuSolver::available()) { std::printf("SKIP: no OpenCL device\n"); return 0; }
    section("GpuSolver::solve(): full pipeline + recovery + CPU gate -> the 3 KAT goldens, verified");

    GpuSolver s;
    const DeviceInfo& d = s.device();
    std::printf("  device: %s | gmem=%.2fGiB | max_alloc=%.2fGiB | CUs=%u\n",
                d.name.c_str(), d.global_mem / 1073741824.0, d.max_alloc / 1073741824.0, d.compute_units);

    std::printf("  running GpuSolver::solve() over the full seed layer on the KAT input...\n");
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::array<uint8_t, 104>> sols = s.solve(kat::input32, kat::nonce0);
    auto t1 = std::chrono::steady_clock::now();
    double wallSecs = std::chrono::duration<double>(t1 - t0).count();
    std::printf("  wall-clock: %.3fs, %zu solution(s) returned\n", wallSecs, sols.size());

    // THE MILESTONE, part 1: exactly 3 solutions -- the GPU search found all
    // 3 known survivors AND every one of them cleared the CPU gate (no
    // false-positive is_zero round-5 collision that fails distinct/indexAfter
    // is silently returned; none of the 3 real goldens is silently dropped).
    check(sols.size() == 3, "GpuSolver::solve() returns EXACTLY 3 solutions for the KAT input");

    // THE MILESTONE, part 2: every returned solution independently passes
    // the same CPU gate solve() itself already filtered through -- this is
    // solve()'s own contract, re-checked here from the outside (the test
    // never reaches into GpuSolver's internals to short-circuit the gate).
    for (size_t i = 0; i < sols.size(); ++i) {
        bool ok = bh3::is_valid_solution(kat::input32, 32, kat::nonce0, sols[i].data());
        char lbl[80];
        std::snprintf(lbl, sizeof lbl, "solution[%zu] independently passes bh3::is_valid_solution (the CPU gate)", i);
        check(ok, lbl);
        if (!ok) show_hex("FAILED-gate solution", sols[i].data(), 104);
    }

    // THE MILESTONE, part 3: the returned SET equals {golden[0], golden[1],
    // golden[2]} byte-for-byte (multiset/distinct-match -- solve()'s
    // internal ordering is whatever the GPU's atomics produced).
    bool matched[3] = {false, false, false};
    int unmatchedCount = 0;
    for (size_t i = 0; i < sols.size(); ++i) {
        int hit = -1;
        for (int g = 0; g < 3; ++g) {
            if (matched[g]) continue;
            if (std::memcmp(sols[i].data(), kat::golden[g], 104) == 0) { hit = g; break; }
        }
        if (hit >= 0) {
            matched[hit] = true;
            if (mxbm::verbose())
                std::printf("  solution[%zu] == golden[%d]\n", i, hit);
        } else {
            ++unmatchedCount;
            show_hex("unmatched solution", sols[i].data(), 104);
        }
    }
    check(unmatchedCount == 0, "every returned solution matches a distinct golden solution");
    check(matched[0] && matched[1] && matched[2], "all 3 KAT golden solutions were found, verified, and returned");

    std::printf("  matched goldens: [%d,%d,%d]\n", matched[0], matched[1], matched[2]);

    return summary("gpu_solver");
}
