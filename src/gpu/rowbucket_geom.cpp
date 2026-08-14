#include "gpu/rowbucket_geom.h"
#include <cmath>

namespace mxbm { namespace gpu {

uint32_t fb_cap_for(uint32_t mean, double sigma) {
    const double sd = std::sqrt((double)mean);
    return mean + (uint32_t)(sigma * sd) + 32u;
}

void rowbucket_bytes(uint32_t capacity, uint32_t bb, size_t& total, size_t& single,
                     bool quad, bool impb, bool arena, bool octo) {
    const uint32_t nb = 1u << bb;
    const uint32_t cap = fb_cap_for(capacity / nb, arena ? kRbDenseSigma : kRbCapSigma);
    // The pool sits behind the bucket records in the SAME buffer, so one slot index
    // addresses both regions and no store in the kernels has to know which it hit.
    const size_t nslots = (size_t)nb * cap + (arena ? kRbArenaSlots : 0u);
    const uint32_t s0 = fb_set_stride(0, quad, impb, octo),
                   s1 = fb_set_stride(1, quad, impb, octo);
    // Under the quad record set 0 (3 u64) is no longer the larger of the two, so the
    // largest single allocation becomes set 1. Taking max() rather than assuming set 0
    // is what keeps the OpenCL single-allocation ceiling honest for both formats.
    single = nslots * (s0 > s1 ? s0 : s1) * 8;
    // Back-refs: rows 1-4 are gi-indexed (capacity each); row 5 is written only at
    // terminal-survivor indices, so it is sized by the 1024 survivor cap, not by
    // capacity. Must match the allocations in cuda_solver.cu / round_pipeline.cpp.
    const size_t backrefs = ((size_t)4*capacity + 1024) * 4 * 2;
    // Per record set the arena also carries a chain head per bucket and a next/tag pair
    // per pool slot -- ~1.5 MB at (16,1), against the ~1.5 GiB the dense cap gives back.
    const size_t arena_meta = arena ? 2 * ((size_t)nb*4 + 2*(size_t)kRbArenaSlots*4 + 4) : 0;
    total = nslots * (s0 + s1) * 8                                // both record sets
          + 2*(size_t)nb*4 + backrefs + arena_meta + 64;          // +counts/refs/counters
}

// Ordered by MEASURED time, fastest first -- NOT by footprint, which disagrees:
// quad (16,1) at 5.02 GiB / 38.39 ms is both smaller and faster than packed (14,3) at
// 6.24 / 40.02, so a footprint-sorted ladder would hand a card the slower rung. packed
// (14,3) stays in the list after it because its single allocation is smaller (2.76 GiB
// against 2.90), so a max_alloc-bound backend can still want it.
//
// An arena row is its neighbour's geometry at a dense cap: same kernels, same record,
// ~22 % fewer record slots, plus the pool's fixed mechanism cost. That cost is measured
// on (16,1) and carried across the other rows, so their times are placings rather than
// figures. Footprints are with the implicit-bits pack where the rung admits it.
const RbRung* rb_rungs(int& n) {
    //     bb   sm   quad   arena  octo        footprint / time
    static const RbRung kRungs[] = {
        { 16u, 1u, false, false, false },   // 6.84 GiB  33.67 ms  (7.20 without impl. bits)
        { 16u, 1u, false, true,  false },   // 5.77      +2 % on the row above
        { 15u, 2u, false, false, false },   // 6.62      36.01
        { 15u, 2u, false, true,  false },   // 5.82
        { 16u, 1u, true,  false, false },   // 5.02      38.39
        { 16u, 1u, true,  true,  false },   // 4.29
        { 14u, 3u, false, false, false },   // 6.24      40.02  -- for max_alloc-bound backends
        { 15u, 2u, true,  false, false },   // 4.65      40.65
        { 15u, 2u, true,  true,  false },   // 4.13
        { 14u, 3u, true,  false, false },   // 4.40      44.75
        { 14u, 3u, true,  true,  false },   // 4.03
        // The octo rows, below everything: round 4 rebuilds its work state from eight
        // leaves, which is the deepest re-derivation on the ladder. Only ever reached
        // when no row above fits, and only paired with quad and dense caps -- a card
        // that can host the packed record has no use for them.
        { 16u, 1u, true,  true,  true  },   // 3.10      56.0
        { 15u, 2u, true,  true,  true  },   // 3.00      58.0
        { 14u, 3u, true,  true,  true  },   // 2.94      62.1  -- the floor
    };
    n = (int)(sizeof kRungs / sizeof kRungs[0]);
    return kRungs;
}

size_t rowbucket_single_split(uint32_t capacity, uint32_t bb, bool quad, bool impb,
                              bool arena, bool octo) {
    size_t total = 0, single = 0;
    rowbucket_bytes(capacity, bb, total, single, quad, impb, arena, octo);
    const size_t half = single / 2;                     // nb is even on every rung
    const size_t backrefs = ((size_t)4 * capacity + 1024) * 4;  // left/right rows never split
    return half > backrefs ? half : backrefs;
}

RbGeometry rb_geometry_for(uint32_t capacity, uint64_t max_alloc, uint64_t global_mem,
                           bool allow_quad, unsigned power_limit_w, bool allow_split,
                           uint64_t slack_bytes, bool allow_impb, bool allow_arena,
                           bool allow_octo) {
    // The implicit-bits record is not a rung, it is a NARROWER SET-0 STRIDE on the rungs
    // that admit it -- so it is folded into each rung's footprint here rather than
    // appearing in the ladder. A backend without it passes allow_impb false and every
    // figure below is what it always was.
    auto impb_at = [&](uint32_t bb, uint32_t sm, bool quad) {
        return allow_impb && rb_impb_ok(bb, sm, quad);
    };
    // The low-power exception to the ladder's order. (17,0) is NOT a rung: it loses
    // 10.9 % at stock, so it can only be reached by policy under a cap -- and the
    // policy is currently DISARMED (kRbLowPowerW == 0: the measured crossover failed
    // reproduction; the constant's comment tells the story). Same fit rules as the
    // ladder; if it does not fit, the ladder answers exactly as without the hint.
    // The figure max_alloc binds on: the whole larger set, or -- when the backend
    // can split each set into two bucket-halves -- whatever rowbucket_single_split
    // says is the largest piece left.
    auto binding_single = [&](uint32_t bb, bool quad, bool impb, bool arena, bool octo,
                              size_t single) {
        if (!allow_split) return single;
        const size_t s = rowbucket_single_split(capacity, bb, quad, impb, arena, octo);
        return s < single ? s : single;
    };
    if (kRbLowPowerW != 0 && power_limit_w != 0 && power_limit_w < kRbLowPowerW) {
        size_t total = 0, single = 0;
        const bool im = impb_at(17u, 0u, false);
        rowbucket_bytes(capacity, 17u, total, single, /*quad=*/false, im);
        if ((!max_alloc || binding_single(17u, false, im, false, false, single) <= (size_t)max_alloc)
            && (!global_mem || total + (size_t)slack_bytes <= (size_t)global_mem))
            return { 17u, 0u, false, true, false, false };
    }
    int n = 0;
    const RbRung* rungs = rb_rungs(n);
    for (int i = 0; i < n; ++i) {
        const RbRung& r = rungs[i];
        if (r.quad && !allow_quad) continue;
        if (r.arena && !allow_arena) continue;
        if (r.octo && !allow_octo) continue;
        size_t total = 0, single = 0;
        const bool im = impb_at(r.bb, r.sm, r.quad);
        rowbucket_bytes(capacity, r.bb, total, single, r.quad, im, r.arena, r.octo);
        if (max_alloc && binding_single(r.bb, r.quad, im, r.arena, r.octo, single)
                         > (size_t)max_alloc) continue;
        if (global_mem && total + (size_t)slack_bytes > (size_t)global_mem) continue;
        return { r.bb, r.sm, r.quad, true, r.arena, r.octo };
    }
    return { 14u, 3u, false, false, false, false };  // nothing fits; callers fall back to sort
}

}} // namespace mxbm::gpu
