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
    total = nslots * (s0 + s1) * 8                                // both record sets
          + 2*(size_t)nb*4 + (size_t)5*capacity*4*2 + 64;         // +counts/left/right/counters
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

RbGeometry rb_geometry_for(uint32_t capacity, uint64_t max_alloc, uint64_t global_mem,
                           bool allow_quad, unsigned power_limit_w) {
    // The low-power exception to the ladder's order. (17,0) is NOT a rung: it loses
    // 10.9 % at stock, so it can only be reached by policy under a cap -- and the
    // policy is currently DISARMED (kRbLowPowerW == 0: the measured crossover failed
    // reproduction; the constant's comment tells the story). Same fit rules as the
    // ladder; if it does not fit, the ladder answers exactly as without the hint.
    if (kRbLowPowerW != 0 && power_limit_w != 0 && power_limit_w < kRbLowPowerW) {
        size_t total = 0, single = 0;
        rowbucket_bytes(capacity, 17u, total, single, /*quad=*/false);
        if ((!max_alloc || single <= (size_t)max_alloc)
            && (!global_mem || total + (size_t)(1ull << 30) <= (size_t)global_mem))
            return { 17u, 0u, false, true };
    }
    int n = 0;
    const RbRung* rungs = rb_rungs(n);
    for (int i = 0; i < n; ++i) {
        const RbRung& r = rungs[i];
        if (r.quad && !allow_quad) continue;
        size_t total = 0, single = 0;
        rowbucket_bytes(capacity, r.bb, total, single, r.quad);
        if (max_alloc && single > (size_t)max_alloc) continue;
        if (global_mem && total + (size_t)(1ull << 30) > (size_t)global_mem) continue;
        return { r.bb, r.sm, r.quad, true };
    }
    return { 14u, 3u, false, false };   // nothing fits; callers fall back to the sort path
}

}} // namespace mxbm::gpu
