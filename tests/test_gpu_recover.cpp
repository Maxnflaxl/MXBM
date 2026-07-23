// Task 1 of M3 Phase C: recover each round-5 survivor's 32 leaf indices and
// pack them into a 104-byte candidate solution, then prove -- byte-for-byte,
// against the KAT input's 3 known golden solutions -- that the on-device DFS
// (kernels/opencl/round.cl's `recover` kernel) walks the consolidated
// back-refs in the exact tree order Beam's index-packer expects (the
// ordered-concat invariant Phase B established, now carried end to end).
//
// This reuses run_pipeline() (T5's unmodified 5-round driver, src/gpu/
// round_pipeline.{h,cpp}) exactly as tests/test_gpu_rounds.cpp does to
// produce real is_zero survivors on real device hardware over the full 2^25
// seed layer -- no synthetic/crafted fixture. tests/test_gpu_rounds.cpp
// already proved run_pipeline() finds all 3 goldens as is_zero survivors on
// this machine (M3 Max); this test's job is the NEXT step: recovering each
// survivor's 32-leaf ancestry and packing it, then checking the resulting
// candidate set against tests/kat_vectors.h's kat::golden[0..2] as a
// multiset of std::array<uint8_t,104> (recovery order need not match golden
// order -- survivor_scan's slot order is whatever the GPU produced).
#include "gpu/cl_runtime.h"
#include "gpu/budget.h"
#include "gpu/round_pipeline.h"
#include "generated/mxbm_kernels.h"
#include "bh3_primitives.h"
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
    if (!Runtime::any_device_available()) { std::printf("SKIP: no OpenCL device\n"); return 0; }
    Runtime rt;
    const DeviceInfo& d = rt.device();
    section("recover: on-device leaf recovery -> 104-byte candidates == the 3 KAT goldens");
    std::printf("  device: %s | gmem=%.2fGiB | max_alloc=%.2fGiB | CUs=%u\n",
                d.name.c_str(), d.global_mem / 1073741824.0, d.max_alloc / 1073741824.0, d.compute_units);

    Budget b = compute_budget(d.global_mem, d.max_alloc);
    PipelineBuffers pb = alloc_pipeline(rt, b);
    check(pb.capacity == b.elems_per_round, "pipeline capacity == elems_per_round");

    std::printf("  running run_pipeline() over the full seed layer on the KAT prePow...\n");
    auto t0 = std::chrono::steady_clock::now();
    PipelineResult res = run_pipeline(rt, pb, b, kat::prePow);
    auto t1 = std::chrono::steady_clock::now();
    double wallSecs = std::chrono::duration<double>(t1 - t0).count();
    std::printf("  wall-clock: %.3fs\n", wallSecs);
    std::printf("  survivors=%u survivor_slots.size()=%zu\n", res.survivors, res.survivor_slots.size());

    check(res.survivor_slots.size() == res.survivors, "survivor_slots.size() == the reported survivor count");
    check(res.survivors >= 1, "GPU 5-round search yields is_zero survivor(s) on KAT input (prereq for recovery)");
    // tests/vectors/beamhash3-kat.md documents exactly 3 known golden
    // solutions for this KAT input; test_gpu_rounds.cpp's Phase-B milestone
    // deliberately only asserted >=1 (paranoid re: budget-driven drops), but
    // Phase C's job is the full recovery -- so pin the exact count here.
    check(res.survivors == 3, "GPU 5-round search yields EXACTLY the 3 known KAT survivors");

    std::vector<std::array<uint8_t,104>> cand = recover_candidates(rt, pb, res.survivor_slots);
    check(cand.size() == res.survivors, "recover_candidates returned exactly one candidate per survivor");

    // THE MILESTONE: the recovered candidate SET (order-independent -- GPU
    // atomics make survivor/scatter order nondeterministic) must equal the
    // set of all 3 known golden solutions, byte-for-byte. This is only
    // possible if the recover kernel's pre-order DFS visits the 32 leaves in
    // EXACTLY the order bh3::pack_indices (proven bit-identical in Phase A)
    // expects.
    bool matched[3] = {false, false, false};
    int unmatchedCount = 0;
    for (size_t i = 0; i < cand.size(); ++i) {
        int hit = -1;
        for (int g = 0; g < 3; ++g) {
            if (matched[g]) continue;
            if (std::memcmp(cand[i].data(), kat::golden[g], 104) == 0) { hit = g; break; }
        }
        if (hit >= 0) {
            matched[hit] = true;
            if (mxbm::verbose())
                std::printf("  candidate[%zu] == golden[%d]\n", i, hit);
        } else {
            ++unmatchedCount;
            show_hex("unmatched candidate", cand[i].data(), 104);
        }
    }
    check(unmatchedCount == 0, "every recovered candidate matches a distinct golden solution");
    check(matched[0] && matched[1] && matched[2], "all 3 KAT golden solutions were recovered byte-for-byte");

    std::printf("  recovered %zu candidate(s); matched goldens: [%d,%d,%d]\n",
                cand.size(), matched[0], matched[1], matched[2]);

    return summary("gpu_recover");
}
