// Phase D benchmark harness (manual-run, NOT a ctest gate). Runs the full
// 2^25 GPU Wagner search over the KAT prePow ITERS times, enforces the D0
// correctness gate on every solve, and reports MEDIAN total + per-phase ms.
// Usage: bench_rounds [ITERS]   (default 50; pass 1 when profiling under ncu).
//
// run_pipeline() is invoked with verbose=false so the bench prints only its
// own median summary, not 50x the per-round breakdown (the per-round logging
// is what test_gpu_rounds uses; here it would bury the numbers we care about).
#include "gpu/cl_runtime.h"
#include "gpu/budget.h"
#include "gpu/round_pipeline.h"
#include "kat_vectors.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace mxbm;
using namespace mxbm::gpu;

static double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n & 1) ? v[n/2] : 0.5*(v[n/2 - 1] + v[n/2]);
}

int main(int argc, char** argv) {
    int iters = (argc > 1) ? std::atoi(argv[1]) : 50;
    if (iters < 1) iters = 1;
    if (!Runtime::any_device_available()) { std::printf("SKIP: no OpenCL device\n"); return 0; }
    Runtime rt;
    const DeviceInfo& d = rt.device();
    Budget b = compute_budget(d.global_mem, d.max_alloc);
    PipelineBuffers pb = alloc_pipeline(rt, b);
    std::printf("device: %s | bucket_bits=%u num_buckets=%u slots=%u capacity=%u\n",
                d.name.c_str(), b.bucket_bits, b.num_buckets, b.slots_per_bucket, b.capacity);

    // Warm the OpenCL program cache (first solve pays the compile).
    (void)run_pipeline(rt, pb, b, kat::prePow, nullptr, /*verbose=*/false);

    std::vector<double> tot, mix, scat, mat;
    int failures = 0;
    for (int it = 0; it < iters; ++it) {
        PipelineResult res = run_pipeline(rt, pb, b, kat::prePow, nullptr, /*verbose=*/false);
        uint32_t bd = 0, pd = 0; double m = 0, s = 0, mt = 0;
        for (int r = 0; r < 5; ++r) {
            bd += res.rounds[r].bucket_drops; pd += res.rounds[r].pair_drops;
            m += res.rounds[r].t_mix_ms; s += res.rounds[r].t_scatter_ms; mt += res.rounds[r].t_match_ms;
        }
        tot.push_back(res.t_total_ms); mix.push_back(m); scat.push_back(s); mat.push_back(mt);

        // Correctness gate.
        bool ok = (bd == 0) && (pd == 0) && (res.survivors == 3);
        if (ok) {
            auto cand = recover_candidates(rt, pb, res.survivor_slots);
            bool matched[3] = {false,false,false};
            for (auto& c : cand)
                for (int g = 0; g < 3; ++g)
                    if (!matched[g] && std::memcmp(c.data(), kat::golden[g], 104) == 0) { matched[g]=true; break; }
            ok = matched[0] && matched[1] && matched[2];
        }
        if (!ok) {
            ++failures;
            std::printf("  solve %d FAIL: bucketDrops=%u pairDrops=%u survivors=%u\n", it, bd, pd, res.survivors);
        }
    }
    std::printf("\n==== BENCH (%d solves) ====\n", iters);
    std::printf("median total   : %.1f ms  (%.2f solve/s)\n", median(tot), 1000.0/median(tot));
    std::printf("median match   : %.1f ms\n", median(mat));
    std::printf("median mix     : %.1f ms\n", median(mix));
    std::printf("median scatter : %.1f ms\n", median(scat));
    std::printf("correctness    : %d/%d clean\n", iters - failures, iters);
    std::printf("===========================\n");
    return failures == 0 ? 0 : 1;   // nonzero exit => a gate failed
}
