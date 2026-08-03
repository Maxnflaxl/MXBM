#pragma once
#include <cstddef>
#include <cstdint>

// Row-bucket geometry and footprint arithmetic, shared by both backends so they cannot
// answer the same question differently. Here rather than in round_pipeline.h because
// that header pulls in OpenCL, which the CUDA translation unit must not see.

namespace mxbm { namespace gpu {

// Per-element record width of each ping-pong set, in u64. The two sets do NOT need the
// same width: round r writes set (r&1) and the entry writes set 0, so set 0 holds the
// round-2 and round-4 outputs (max 9) and set 1 the round-1 and round-3 outputs (max 8).
// Sizing them separately frees ~0.7 GiB. Derived from the per-round stride table rather
// than written out, so it cannot drift from the kernels.
constexpr uint32_t kFbStride[6] = { 0u, 1u, 2u, 9u, 8u, 2u };

// Round r's output stride. Under the QUAD RECORD (MXBM_R3_QUAD, CUDA only) round 2 emits
// key + four leaves + gi in 3 u64 instead of the 72 B packed record, and round 3 rebuilds
// the work words from those leaves. Set 0's width is then round 2's record and nothing
// else -- round 4 writes 2 -- so 9 u64 becomes 3, which is where the ~2.1 GiB goes.
constexpr uint32_t fb_round_stride(int r, bool quad) {
    return (quad && r == 2) ? 3u : kFbStride[r + 1];
}
constexpr uint32_t fb_set_stride(int set, bool quad = false) {
    uint32_t m = (set == 0) ? kFbStride[1] : 0u;          // entry writes set 0
    for (int r = 1; r <= 4; ++r)
        if ((r & 1) == set && fb_round_stride(r, quad) > m) m = fb_round_stride(r, quad);
    return m;
}

// Bucket capacity. Child keys come out of apply_mix and are effectively uniform, so
// occupancy is ~Poisson(mean) with mean = capacity/nb and **sigma = sqrt(mean)**.
//
// MEASURED (MXBM_OCC): OpenCL 4.07..4.35 sigma at nb=32768; CUDA 4.6..5.35 at nb=65536
// over 201 solves x 5 scatter stages, i.e. 85-91 % of what is reserved.
//
// **The 8 is load-bearing, not slack.** Reading those samples as "5 sigma is enough"
// prices the tail per bucket; a miner draws 2.45e6 solves x 5 stages x 65536 buckets =
// 8.0e11 bucket occupancies per DAY, and that is what has to survive:
//
//     6 sigma  cap 697   8.1e-01 overflows/day     <- ~one dropped element per day
//     7 sigma  cap 720   8.1e-04 overflows/day     MTBF 3.4 years
//     8 sigma  cap 743   4.0e-07 overflows/day     MTBF 6900 years
//
// A dropped child can be a valid solution's ancestor, and bucketDrops != 0 fails the
// gate every run is held to. So the geometry ladder below -- not a tighter cap -- is
// the lever for fitting a smaller card.
uint32_t fb_cap_for(uint32_t mean);

// Bytes the row-bucket path needs at a given geometry: {total, largest single
// allocation}. The single figure is what OpenCL's CL_DEVICE_MAX_MEM_ALLOC_SIZE caps,
// and is what binds below 12 GB.
void rowbucket_bytes(uint32_t capacity, uint32_t bb, size_t& total, size_t& single,
                     bool quad = false);

// The shipping element capacity: the 2^25 seed layer plus the 1/32 growth slack every
// round is budgeted against. Each backend keeps its own kCapacity for kernel arithmetic
// and static_asserts it equals this one; exposed so main.cpp's restart notice can ask
// the geometry question below with the same number the solvers used.
constexpr uint32_t kRbCapacity = (1u << 25) + (1u << 25) / 32;    // 34,603,008

// The power band that would earn the (17,0) rung. ZERO: no band currently does.
// The eco sweep measured a ~190 W crossover and the selection shipped wired to it
// (2026-07-31) -- then the prize failed same-day reproduction in both the miner loop
// and the sweep's own binary (eco-sweep addenda, docs/performance-research.md), so
// the policy is disarmed. Everything around it stays live -- the observed-limit hint
// through the CUDA ctor, the restart notice, MXBM_BB -- and re-arming is setting this
// constant to a band that reproduces. Selection, when armed, happens ONCE, from the
// limit observed after --pl landed: a geometry switch is a multi-GiB realloc, so a
// cap changed mid-run gets a restart notice, never a re-select.
constexpr unsigned kRbLowPowerW = 0;

// The row-bucket geometry decision, with the device reduced to the two numbers it
// actually turns on. Pure, so docs/HW_REQUIREMENTS.md's "which cards get the fast path"
// table can be pinned by a test instead of worked out by hand.
//
// The staged group is mean_bucket / 2^sm and wants to be ~256, which pins bb + sm = 17;
// within that the smallest sm wins, being a direct multiplier on the redundant bucket
// rescan. Coarser buckets pay the per-bucket capacity slack fewer times, so they need
// LESS memory -- which is the whole ladder. Measured, 200 nonces, KAT green and drops
// zero at every step:
//
//            CUDA                          OpenCL
//   (16,1)   7.46 GiB   35.1 ms            3.27 GiB single   40.4 ms
//   (15,2)   6.88 GiB   37.8 ms            2.96 GiB single   42.6 ms
//   (14,3)   6.50 GiB   42.8 ms            2.76 GiB single   47.5 ms
//
// A card that cannot host (16,1) drops one step -- 8-14 % slower -- rather than falling
// back to the sort path, which measures 215 ms.
//
// THE QUAD RECORD IS A SECOND AXIS, not a finer geometry, and the ladder is ordered by
// MEASURED time rather than by footprint because the two disagree. On the reference card:
//
//   packed (16,1)  7.46 GiB  33.67 ms      quad (16,1)  5.28 GiB  38.39 ms
//   packed (15,2)  6.88 GiB  36.01 ms      quad (15,2)  4.91 GiB  40.65 ms
//   packed (14,3)  6.50 GiB  40.02 ms      quad (14,3)  4.66 GiB  44.75 ms
//
// So quad (16,1) is BOTH smaller and faster than packed (14,3): a coarser geometry pays
// in scatter locality what the quad record pays in re-derivation arithmetic, and the
// arithmetic is cheaper. Sorting the rungs by footprint would pick the wrong one.
//
// The quad record exists on BOTH backends (kernels/cuda/fused_round.cuh;
// round_fused_rd2q/rd3 in kernels/opencl/lds.cl since 2026-07-28). `allow_quad`
// stays a parameter -- false by default -- so the packed-only ladder remains
// individually testable and each caller states its capability explicitly. It buys
// no speed and no watts at any power cap (measured; see docs/performance.md), so
// it is purely what lets a smaller card run at all.
//
//   max_alloc : largest single allocation the backend permits. OpenCL is bound by
//               CL_DEVICE_MAX_MEM_ALLOC_SIZE (VRAM/4 on NVIDIA); CUDA has no such
//               limit, and passes 0 to say so. That asymmetry is the reach difference.
//   bb/sm     : bucket bits and sub-mask bits, always summing to 17
//   quad      : use the 24 B quad record for round 2 -> round 3
//   viable    : false when no rung will fit
//   allow_split : the backend can allocate each record set as two bucket-halves
//               (PipelineBuffers::fb_elem_hi), so a rung binds max_alloc on
//               rowbucket_single_split() -- half the larger set, or the never-split
//               back-ref row when that is bigger -- instead of on the whole set.
//               The OpenCL backend passes true; CUDA never needs it (max_alloc 0).
//   power_limit_w : the board power limit observed at startup, in watts; 0 means
//               unknown or uncapped and changes nothing. Below kRbLowPowerW the
//               (17,0) rung is preferred when its 8.35 GiB fits -- currently
//               DISARMED (kRbLowPowerW == 0; see its comment for why), so every
//               value is inert. Only the CUDA backend passes a real value.
struct RbGeometry { uint32_t bb, sm; bool quad; bool viable; };
//   slack_bytes : held back from `global_mem` on top of the footprint. The 1 GiB
//               default is for callers passing TOTAL VRAM, which says nothing
//               about what is free; a caller that already sized against the
//               driver's free figure passes 0, since the reserve is applied
//               there (gpu/budget.h).
RbGeometry rb_geometry_for(uint32_t capacity, uint64_t max_alloc, uint64_t global_mem,
                           bool allow_quad = false, unsigned power_limit_w = 0,
                           bool allow_split = false,
                           uint64_t slack_bytes = (uint64_t)1 << 30);

// Largest single allocation when each record set may split into two bucket-halves:
// half the larger set, or the never-split back-ref rows if bigger. The arithmetic
// behind allow_split, exposed so viability and the tests ask the same question.
size_t rowbucket_single_split(uint32_t capacity, uint32_t bb, bool quad);

// The ladder itself, in the order rb_geometry_for walks it. Exposed because the CUDA
// backend has to keep stepping when the ALLOCATOR refuses a rung the arithmetic said
// would fit -- a desktop compositor holding a gigabyte is not visible in totalGlobalMem.
// Walking this list rather than decrementing `bb` is what makes that retry cross from
// the packed rungs onto the quad ones instead of stopping at the bottom of the packed
// half. `n` receives the count.
struct RbRung { uint32_t bb, sm; bool quad; };
const RbRung* rb_rungs(int& n);

}} // namespace mxbm::gpu
