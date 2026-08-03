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
// One solve's end-to-end throughput next to the reference figure -- the
// headline that makes "too slow" obvious from an integration-test run. Logged,
// never asserted (wall-time flakes across GPUs); per-phase drill-down lives in
// test_gpu_rounds.cpp, which drives run_pipeline directly.
static void perf_line(const char* pass, double secs) {
    const double kRefSolPerSec = 53.0, kSolsPerNonce = 1.9;
    const double kRefSolvePerSec = kRefSolPerSec / kSolsPerNonce;   // ~28 solve/s
    double sps = secs > 0.0 ? 1.0 / secs : 0.0;
    std::printf("  [PERF] %s: %.2f solve/s (~%.1f sol/s) | ref ~%.0f sol/s (~%.0f solve/s) | ~%.0fx slower\n",
                pass, sps, sps * kSolsPerNonce, kRefSolPerSec, kRefSolvePerSec,
                sps > 0.0 ? kRefSolvePerSec / sps : 0.0);
}

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
    double secs1 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("  solve #1: %.3fs, %zu solution(s)\n", secs1, sols1.size());
    perf_line("solve#1", secs1);
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
    double secs2 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    std::printf("  solve #2: %.3fs, %zu solution(s)\n", secs2, sols2.size());
    perf_line("solve#2", secs2);
    verify_goldens(sols2, "solve#2");

    // SPECULATIVE ENTRY. Nothing else in the suite reaches it: run_pipeline's own tests
    // pass no SpecEntry, and it takes THREE solves at a fixed nonce stride before a
    // prediction can land. Walk n0-2, n0-1, n0 so the hit falls on the KAT nonce -- the
    // goldens are then the assertion, and a wrong entry set cannot produce them.
    // Asserting the hit HAPPENED matters as much: without it this passes trivially the
    // day speculation stops engaging.
    if (s.speculation_available()) {
        std::printf("  speculative entry: walking a fixed nonce stride onto the KAT nonce...\n");
        uint64_t n0 = 0; std::memcpy(&n0, kat::nonce0, 8);
        uint8_t nonce[8];
        for (int step = 2; step >= 1; --step) {   // n0-2 then n0-1: teaches the stride
            const uint64_t n = n0 - (uint64_t)step;
            std::memcpy(nonce, &n, 8);
            (void)s.solve(kat::input32, nonce);
        }
        check(!s.last_solve_speculated(), "the stride-learning solves are honest misses");
        std::vector<std::array<uint8_t, 104>> sols3 = s.solve(kat::input32, kat::nonce0);
        std::printf("  solve #5 on the KAT nonce: speculated=%s\n",
                    s.last_solve_speculated() ? "yes" : "NO");
        check(s.last_solve_speculated(),
              "solve #5 was SERVED by speculation -- its entry pass came from solve #4's round 4");
        verify_goldens(sols3, "speculated");
    } else {
        std::printf("  speculative entry: unavailable (MXBM_NO_SPEC, or the buffers did "
                    "not fit) -- skipped\n");
    }

    // MXBM_SOLRATE=N: run N solves over DISTINCT nonces and report the mean number of
    // CPU-VERIFIED solutions per solve. This is the factor that converts solve/s into the
    // sol/s figure quoted against other miners, and it CANNOT be measured on the KAT
    // input, which has exactly 3 solutions by construction.
    if (const char* e = std::getenv("MXBM_SOLRATE")) {
        const int n = std::atoi(e) > 0 ? std::atoi(e) : 16;
        uint8_t nonce[8];
        std::memcpy(nonce, kat::nonce0, 8);
        size_t total = 0, verified = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n; ++i) {
            nonce[7] = (uint8_t)(kat::nonce0[7] + i + 1);   // distinct nonce per solve
            auto sols = s.solve(kat::input32, nonce);
            total += sols.size();
            for (const auto& sol : sols)
                if (bh3::is_valid_solution(kat::input32, 32, nonce, sol.data())) ++verified;
        }
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count() / n;
        std::printf("\n  [solrate] %d distinct nonces: %.2f verified solutions/solve, "
                    "%.1f ms/solve  =>  %.1f sol/s\n",
                    n, (double)verified / n, ms, (double)verified / n * 1000.0 / ms);
        check(verified == total, "every solution GpuSolver::solve() returns re-verifies");
    }

    return summary("gpu_solver");
}
