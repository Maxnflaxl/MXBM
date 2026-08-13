#include "gpu/rowbucket_geom.h"
#include <cmath>

namespace mxbm { namespace gpu {

uint32_t fb_cap_for(uint32_t mean) {
    const double sd = std::sqrt((double)mean);
    return mean + (uint32_t)(8.0 * sd) + 32u;
}

void rowbucket_bytes(uint32_t capacity, uint32_t bb, size_t& total, size_t& single,
                     bool quad) {
    const uint32_t nb = 1u << bb;
    const uint32_t cap = fb_cap_for(capacity / nb);
    const size_t nslots = (size_t)nb * cap;
    const uint32_t s0 = fb_set_stride(0, quad), s1 = fb_set_stride(1, quad);
    // Under the quad record set 0 (3 u64) is no longer the larger of the two, so the
    // largest single allocation becomes set 1. Taking max() rather than assuming set 0
    // is what keeps the OpenCL single-allocation ceiling honest for both formats.
    single = nslots * (s0 > s1 ? s0 : s1) * 8;
    // Back-refs: rows 1-4 are gi-indexed (capacity each); row 5 is written only at
    // terminal-survivor indices, so it is sized by the 1024 survivor cap, not by
    // capacity. Must match the allocations in cuda_solver.cu / round_pipeline.cpp.
    const size_t backrefs = ((size_t)4*capacity + 1024) * 4 * 2;
    total = nslots * (s0 + s1) * 8                                // both record sets
          + 2*(size_t)nb*4 + backrefs + 64;                       // +counts/refs/counters
}

// Ordered by MEASURED time, fastest first -- NOT by footprint, which disagrees:
// quad (16,1) at 5.28 GiB / 38.39 ms is both smaller and faster than packed (14,3) at
// 6.50 / 40.02, so a footprint-sorted ladder would hand a card the slower rung. packed
// (14,3) stays in the list after it because its single allocation is smaller (2.76 GiB
// against 2.90), so a max_alloc-bound backend can still want it.
const RbRung* rb_rungs(int& n) {
    static const RbRung kRungs[] = {
        { 16u, 1u, false },   // 7.46 GiB  33.67 ms
        { 15u, 2u, false },   // 6.88      36.01
        { 16u, 1u, true  },   // 5.28      38.39
        { 14u, 3u, false },   // 6.50      40.02   -- kept for max_alloc-bound backends
        { 15u, 2u, true  },   // 4.91      40.65
        { 14u, 3u, true  },   // 4.66      44.75
    };
    n = (int)(sizeof kRungs / sizeof kRungs[0]);
    return kRungs;
}

size_t rowbucket_single_split(uint32_t capacity, uint32_t bb, bool quad) {
    size_t total = 0, single = 0;
    rowbucket_bytes(capacity, bb, total, single, quad);
    const size_t half = single / 2;                     // nb is even on every rung
    const size_t backrefs = ((size_t)4 * capacity + 1024) * 4;  // left/right rows never split
    return half > backrefs ? half : backrefs;
}

RbGeometry rb_geometry_for(uint32_t capacity, uint64_t max_alloc, uint64_t global_mem,
                           bool allow_quad, unsigned power_limit_w, bool allow_split,
                           uint64_t slack_bytes) {
    // The low-power exception to the ladder's order. (17,0) is NOT a rung: it loses
    // 10.9 % at stock, so it can only be reached by policy under a cap -- and the
    // policy is currently DISARMED (kRbLowPowerW == 0: the measured crossover failed
    // reproduction; the constant's comment tells the story). Same fit rules as the
    // ladder; if it does not fit, the ladder answers exactly as without the hint.
    // The figure max_alloc binds on: the whole larger set, or -- when the backend
    // can split each set into two bucket-halves -- whatever rowbucket_single_split
    // says is the largest piece left.
    auto binding_single = [&](uint32_t bb, bool quad, size_t single) {
        if (!allow_split) return single;
        const size_t s = rowbucket_single_split(capacity, bb, quad);
        return s < single ? s : single;
    };
    if (kRbLowPowerW != 0 && power_limit_w != 0 && power_limit_w < kRbLowPowerW) {
        size_t total = 0, single = 0;
        rowbucket_bytes(capacity, 17u, total, single, /*quad=*/false);
        if ((!max_alloc || binding_single(17u, false, single) <= (size_t)max_alloc)
            && (!global_mem || total + (size_t)slack_bytes <= (size_t)global_mem))
            return { 17u, 0u, false, true };
    }
    int n = 0;
    const RbRung* rungs = rb_rungs(n);
    for (int i = 0; i < n; ++i) {
        const RbRung& r = rungs[i];
        if (r.quad && !allow_quad) continue;
        size_t total = 0, single = 0;
        rowbucket_bytes(capacity, r.bb, total, single, r.quad);
        if (max_alloc && binding_single(r.bb, r.quad, single) > (size_t)max_alloc) continue;
        if (global_mem && total + (size_t)slack_bytes > (size_t)global_mem) continue;
        return { r.bb, r.sm, r.quad, true };
    }
    return { 14u, 3u, false, false };   // nothing fits; callers fall back to the sort path
}

}} // namespace mxbm::gpu
