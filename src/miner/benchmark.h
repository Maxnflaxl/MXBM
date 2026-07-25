#pragma once
// Offline solver benchmark: no pool, no wallet, no network.
//
// The point of this living in the miner rather than in tests/bench_rounds is
// comparability. bench_rounds times the solver PIPELINE; this drives the exact
// same Engine::process_job() path that live mining drives, with the same
// Stats::record_attempt() hook, so the sol/s it prints is the same number the
// stats table prints while mining -- not a differently-defined one that happens
// to share a unit. That is what makes it usable for A/B-ing overclock settings
// against a mining baseline (see docs/overclocking.md).
//
// It does NOT settle MXBM vs another miner: reported sol/s is
// implementation-defined (we count CPU-verified solutions, 1.98/solve, not raw
// survivors, 2.29/solve), so a cross-miner comparison still needs accepted pool
// shares. A self-consistent metric measures a delta within one miner; it takes a
// shared metric to measure a gap between two.
#include <atomic>
#include <cstdint>

#include "miner/solver.h"
#include "miner/stats.h"

namespace mxbm { namespace miner {

struct BenchmarkResult {
    uint64_t solves = 0;          // completed solve() calls, each on a distinct nonce
    uint64_t solutions = 0;       // CPU-verified solutions found across those solves
    double elapsed_s = 0.0;
    double sol_per_s = 0.0;       // solutions / elapsed -- the headline, same metric as mining
    double solves_per_s = 0.0;
    double per_solve_avg = 0.0;   // solutions per solve (~1.98 for BeamHash III)
    double median_ms = 0.0;       // median wall time of one solve()
    double p5_ms = 0.0, p95_ms = 0.0;
};

// Solves distinct nonces off a fixed synthetic header until `seconds` elapse
// (0 = until `stop` is set), feeding `stats` exactly as mining does so the
// ticker's speed line and statistics table work unchanged.
//
// `stop` is polled between solves; a solve already in flight is also aborted
// via Solver::request_abort() so Ctrl+C does not wait out a full solve.
BenchmarkResult run_benchmark(Solver& solver, Stats& stats, int seconds,
                              std::atomic<bool>& stop);

}} // namespace mxbm::miner
