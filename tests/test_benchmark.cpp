// Offline-benchmark accounting, driven by a fake solver so it runs with no GPU.
//
// The value of --benchmark is that its sol/s is the SAME number the stats table
// reports while mining, so an overclock A/B against a mining baseline is
// meaningful. That only holds if the arithmetic behind it is right, which is
// what this pins: solutions counted, solutions-per-solve, and the percentiles.
#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>

#include "check.h"
#include "miner/benchmark.h"

using namespace mxbm;

namespace {

// Returns a fixed number of candidates per solve, after a fixed delay, so both
// the rate and the timing percentiles have known expected values.
class FakeSolver : public miner::Solver {
public:
    FakeSolver(int per_solve, int delay_ms) : per_solve_(per_solve), delay_ms_(delay_ms) {}

    std::vector<std::array<uint8_t, 104>> solve(const uint8_t[32], const uint8_t[8]) override {
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms_));
        ++calls_;
        // Content is irrelevant -- run_benchmark counts them and its no-op
        // submit_fn discards them -- but they must not clear the difficulty
        // filter by accident, so leave them zero (an all-zero hash would, but
        // these are hashed first, so this is merely arbitrary).
        return std::vector<std::array<uint8_t, 104>>(per_solve_);
    }
    int calls() const { return calls_; }

private:
    int per_solve_, delay_ms_;
    int calls_ = 0;
};

} // namespace

int main() {
    // -- accounting: 2 solutions per solve, ~10 ms each --
    {
        FakeSolver solver(2, 10);
        miner::Stats stats;
        std::atomic<bool> stop{false};

        miner::BenchmarkResult r = miner::run_benchmark(solver, stats, 1, stop);

        check(r.solves > 0, "the benchmark completed at least one solve");
        check(r.solves == (uint64_t)solver.calls(),
              "every counted solve is exactly one solver.solve() call");
        check(r.solutions == r.solves * 2,
              "solutions == solves x candidates-per-solve");
        check(std::fabs(r.per_solve_avg - 2.0) < 1e-9,
              "solutions-per-solve is reported exactly");

        // sol/s must be solutions over elapsed, not solves over elapsed -- the
        // distinction is the whole 1.98x between the two metrics.
        check(std::fabs(r.sol_per_s - (double)r.solutions / r.elapsed_s) < 1e-6,
              "sol/s == solutions / elapsed");
        check(r.sol_per_s > r.solves_per_s,
              "sol/s exceeds solves/s when a solve yields >1 solution");

        // ~10 ms sleep per solve; allow generous slack for scheduling.
        check(r.median_ms >= 8.0 && r.median_ms < 60.0,
              "median solve time tracks the solver's actual duration");
        check(r.p5_ms <= r.median_ms && r.median_ms <= r.p95_ms,
              "percentiles are ordered p5 <= median <= p95");

        // The benchmark must feed Stats the same way mining does, or the
        // ticker's speed line would read zero during a benchmark run.
        miner::Stats::Snapshot s = stats.snapshot();
        check(s.sol_session > 0.0, "the run fed Stats, so the stats table reports a rate");
    }

    // -- a solve yielding nothing still counts as a solve --
    // Guards against defining the rate as "solves that found something", which
    // would silently inflate sol/s on a card that is finding fewer solutions.
    {
        FakeSolver solver(0, 5);
        miner::Stats stats;
        std::atomic<bool> stop{false};
        miner::BenchmarkResult r = miner::run_benchmark(solver, stats, 1, stop);
        check(r.solves > 0, "zero-solution solves are still counted as solves");
        check(r.solutions == 0 && r.sol_per_s == 0.0,
              "no solutions means a zero rate, not a division artifact");
    }

    // -- stop flag ends the run promptly --
    {
        FakeSolver solver(1, 5);
        miner::Stats stats;
        std::atomic<bool> stop{true};   // already set: must not start a long run
        auto t0 = std::chrono::steady_clock::now();
        miner::BenchmarkResult r = miner::run_benchmark(solver, stats, 600, stop);
        double waited = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        check(waited < 5.0, "a pre-set stop flag returns immediately, not after the duration");
        check(r.solves == 0, "no solves run when stop is already set");
    }

    return summary("benchmark");
}
