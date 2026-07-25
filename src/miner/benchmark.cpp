#include "miner/benchmark.h"

#include <algorithm>
#include <chrono>
#include <vector>

#include "miner/engine.h"
#include "stratum/client.h"
#include "stratum/messages.h"

namespace mxbm { namespace miner {
namespace {

// A fixed 32-byte header, hex-encoded as Engine::process_job expects. Its
// contents do not matter -- BeamHash III hashes it before any of the work the
// benchmark measures -- but it must be EXACTLY 64 lowercase hex chars, since
// process_job's from_hex_strict rejects anything else by returning early, which
// would otherwise show up as an impossibly fast benchmark rather than an error.
// Fixed rather than random so two runs on the same machine are comparable.
const char* const kHeaderHex =
    "4d58424d2062656e63686d61726b20686561646572000000000000000000beef";

// Nonce variation comes from Engine's own per-attempt counter, so successive
// process_job() calls sweep distinct nonces exactly as mining does. The prefix
// is what a pool would assign; any fixed value works offline.
const char* const kNoncePrefix = "0000";

double percentile(std::vector<double>& v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t i = (size_t)(p * (double)(v.size() - 1) + 0.5);
    return v[std::min(i, v.size() - 1)];
}

} // namespace

BenchmarkResult run_benchmark(Solver& solver, Stats& stats, int seconds,
                              std::atomic<bool>& stop) {
    // Never connected. Engine only touches the client for submit_fn (replaced
    // below) and for on_job()'s nonceprefix read, which the benchmark does not
    // call -- it drives the synchronous process_job() core directly.
    stratum::Client client;
    Engine engine(client, solver);

    // Engine's constructor defaults submit_fn to client_.submit(). Replace it
    // before the first solve: a solution clearing the nominal difficulty below
    // would otherwise be written to a socket that was never opened.
    engine.submit_fn = [](const stratum::Solution&, Origin) {};

    std::vector<double> samples;
    uint64_t solutions = 0;
    engine.on_attempt = [&](uint32_t candidates) {
        stats.record_attempt(candidates);   // the identical hook live mining uses
        solutions += candidates;
    };

    stratum::Job job;
    job.id = "benchmark";
    job.input = kHeaderHex;
    // Nominal difficulty. It cannot affect the reported rate -- on_attempt fires
    // before any filtering -- but keeping it realistic means the difficulty
    // filter runs, so the benchmark exercises the same work mining does.
    job.difficulty = 512;

    const auto t0 = std::chrono::steady_clock::now();
    const auto deadline = t0 + std::chrono::seconds(seconds > 0 ? seconds : 0);

    while (!stop.load(std::memory_order_relaxed)) {
        const auto s0 = std::chrono::steady_clock::now();
        if (seconds > 0 && s0 >= deadline) break;

        engine.process_job(job, kNoncePrefix);

        const auto s1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(s1 - s0).count());
    }

    const auto t1 = std::chrono::steady_clock::now();

    BenchmarkResult r;
    r.solves    = samples.size();
    r.solutions = solutions;
    r.elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    if (r.elapsed_s > 0.0) {
        r.sol_per_s    = (double)r.solutions / r.elapsed_s;
        r.solves_per_s = (double)r.solves / r.elapsed_s;
    }
    if (r.solves > 0) r.per_solve_avg = (double)r.solutions / (double)r.solves;
    r.median_ms = percentile(samples, 0.50);
    r.p5_ms     = percentile(samples, 0.05);
    r.p95_ms    = percentile(samples, 0.95);
    return r;
}

}} // namespace mxbm::miner
