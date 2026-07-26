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
constexpr uint32_t fb_set_stride(int set) {
    uint32_t m = (set == 0) ? kFbStride[1] : 0u;          // entry writes set 0
    for (int r = 1; r <= 4; ++r)
        if ((r & 1) == set && kFbStride[r + 1] > m) m = kFbStride[r + 1];
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
void rowbucket_bytes(uint32_t capacity, uint32_t bb, size_t& total, size_t& single);

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
//   max_alloc : largest single allocation the backend permits. OpenCL is bound by
//               CL_DEVICE_MAX_MEM_ALLOC_SIZE (VRAM/4 on NVIDIA); CUDA has no such
//               limit, and passes 0 to say so. That asymmetry is the reach difference.
//   bb/sm     : bucket bits and sub-mask bits, always summing to 17
//   viable    : false when even the coarsest geometry will not fit
struct RbGeometry { uint32_t bb, sm; bool viable; };
RbGeometry rb_geometry_for(uint32_t capacity, uint64_t max_alloc, uint64_t global_mem);

}} // namespace mxbm::gpu
