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

// Verify one solve()'s result against the milestone contract: EXACTLY 3
// solutions, each independently passing the CPU gate (bh3::is_valid_solution
// -- re-checked from outside GpuSolver, never short-circuiting the gate), and
// together == {golden[0..2]} byte-for-byte (distinct-match; GPU atomics make
// solve()'s internal ordering nondeterministic). `pass` labels the check
// messages so a failure names which solve() broke.
static void verify_goldens(const std::vector<std::array<uint8_t, 104>>& sols, const char* pass) {
    char lbl[128];
    std::snprintf(lbl, sizeof lbl, "%s: solve() returns EXACTLY 3 solutions", pass);
    check(sols.size() == 3, lbl);
    for (size_t i = 0; i < sols.size(); ++i) {
        bool ok = bh3::is_valid_solution(kat::input32, 32, kat::nonce0, sols[i].data());
        std::snprintf(lbl, sizeof lbl, "%s: solution[%zu] independently passes the CPU gate", pass, i);
        check(ok, lbl);
        if (!ok) show_hex("FAILED-gate solution", sols[i].data(), 104);
    }
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
    if (!GpuSolver::available()) { std::printf("SKIP: no OpenCL device\n"); return 0; }
    section("GpuSolver::solve(): full pipeline + recovery + CPU gate -> the 3 KAT goldens, verified");

    GpuSolver s;
    const DeviceInfo& d = s.device();
    std::printf("  device: %s | gmem=%.2fGiB | max_alloc=%.2fGiB | CUs=%u\n",
                d.name.c_str(), d.global_mem / 1073741824.0, d.max_alloc / 1073741824.0, d.compute_units);

    // Solve #1.
    std::printf("  solve #1 over the full seed layer on the KAT input...\n");
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::array<uint8_t, 104>> sols1 = s.solve(kat::input32, kat::nonce0);
    std::printf("  solve #1: %.3fs, %zu solution(s)\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), sols1.size());
    verify_goldens(sols1, "solve#1");

    // Solve #2 on the SAME GpuSolver -- this is production's continuous-mining
    // mode: the Engine calls solve() back-to-back over incrementing nonces,
    // reusing the persistent PipelineBuffers. The per-run GPU state (counters[3]
    // survivor count, bucket counts) MUST be re-zeroed each call; if it weren't,
    // solve #2 would yield garbage the gate rejects -- the pool stays safe, but
    // the miner would mine NOTHING from the second nonce onward. Same
    // (input, nonce) -> the same 3 verified goldens proves the reuse is clean.
    std::printf("  solve #2 on the SAME solver (persistent-buffer reuse -- the continuous-mining path)...\n");
    auto t1 = std::chrono::steady_clock::now();
    std::vector<std::array<uint8_t, 104>> sols2 = s.solve(kat::input32, kat::nonce0);
    std::printf("  solve #2: %.3fs, %zu solution(s)\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count(), sols2.size());
    verify_goldens(sols2, "solve#2");

    return summary("gpu_solver");
}
