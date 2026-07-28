#pragma once
// MXBM record packing, shared by the CUDA round kernels. These mirror the helpers in
// kernels/opencl/lds.cl exactly -- the layouts are load-bearing and documented in
// docs/performance.md ("stored record per round"). The BeamHash III primitives are NOT
// duplicated here: src/beamhash/bh3_primitives.h is MXBM_HD-annotated and compiles as
// device code, so the CUDA kernels call the same source as the CPU verifier.
#include "beamhash/bh3_primitives.h"
#include <cstdint>

namespace mxbm { namespace cuda {

__constant__ constexpr uint32_t kIdxMask = 0x1FFFFFFu;   // seed indices are 25-bit

// r1 -> r2: 16 B pair record. w0 = key | i0<<24, w1 = i1 | gi<<25.
__device__ __forceinline__ uint32_t pair_left (uint64_t w0) { return (uint32_t)(w0 >> 24) & kIdxMask; }
__device__ __forceinline__ uint32_t pair_right(uint64_t w1) { return (uint32_t)(w1)       & kIdxMask; }
__device__ __forceinline__ uint32_t pair_gi   (uint64_t w1) { return (uint32_t)(w1 >> 25); }
__device__ __forceinline__ uint64_t pair_w0(uint32_t key, uint32_t i0) { return (uint64_t)key | ((uint64_t)i0 << 24); }
__device__ __forceinline__ uint64_t pair_w1(uint32_t i1, uint32_t gi)  { return (uint64_t)i1  | ((uint64_t)gi << 25); }

// r2 -> r3: 72 B = 7 work words + 4 leaves and gi bit-packed into 2 u64. `lead` is not
// stored: it IS leaf 0, and storing it again cost a whole u64 (see "the redundant lead
// field" in docs/performance.md).
//   p0 = l0 | l1<<25 | (l2 low 14)<<50      p1 = l2>>14 | l3<<11 | gi<<36
__device__ __forceinline__ uint64_t r3_p0(uint32_t l0, uint32_t l1, uint32_t l2) {
    return (uint64_t)l0 | ((uint64_t)l1 << 25) | (((uint64_t)l2 & 0x3FFFull) << 50);
}
__device__ __forceinline__ uint64_t r3_p1(uint32_t l2, uint32_t l3, uint32_t gi) {
    return ((uint64_t)l2 >> 14) | ((uint64_t)l3 << 11) | ((uint64_t)gi << 36);
}
__device__ __forceinline__ uint32_t r3_l0(uint64_t p0) { return (uint32_t)(p0 & kIdxMask); }
__device__ __forceinline__ uint32_t r3_l1(uint64_t p0) { return (uint32_t)((p0 >> 25) & kIdxMask); }
__device__ __forceinline__ uint32_t r3_l2(uint64_t p0, uint64_t p1) {
    return (uint32_t)(((p0 >> 50) & 0x3FFFull) | ((p1 & 0x7FFull) << 14));
}
__device__ __forceinline__ uint32_t r3_l3(uint64_t p1) { return (uint32_t)((p1 >> 11) & kIdxMask); }
__device__ __forceinline__ uint32_t r3_gi(uint64_t p1) { return (uint32_t)((p1 >> 36) & 0x3FFFFFFull); }

// r2 -> r3, QUAD RECORD (MXBM_R3_QUAD): 24 B instead of the 72 B above. Same argument as
// the pair record one round up -- a round-2 output element is a combine of two round-2
// inputs, each of which is determined by two seed indices, so FOUR indices determine it
// and those four ARE its leaves (sIn = 4). The 7 work words are then redundant: round 3
// rebuilds them with rebuild_r3 instead of reading them.
//   w0 = key | l0<<24      w1 = l1 | l2<<25      w2 = l3 | gi<<25
// w0 is bit-identical to the pair record's, so the staging loop's `rec0 & 0xFFFFFF` key
// extraction is unchanged and shared with every other mode.
__device__ __forceinline__ uint64_t quad_w0(uint32_t key, uint32_t l0) { return (uint64_t)key | ((uint64_t)l0 << 24); }
__device__ __forceinline__ uint64_t quad_w1(uint32_t l1, uint32_t l2)  { return (uint64_t)l1  | ((uint64_t)l2 << 25); }
__device__ __forceinline__ uint64_t quad_w2(uint32_t l3, uint32_t gi)  { return (uint64_t)l3  | ((uint64_t)gi << 25); }
__device__ __forceinline__ uint32_t quad_l0(uint64_t w0) { return (uint32_t)(w0 >> 24) & kIdxMask; }
__device__ __forceinline__ uint32_t quad_l1(uint64_t w1) { return (uint32_t)(w1)       & kIdxMask; }
__device__ __forceinline__ uint32_t quad_l2(uint64_t w1) { return (uint32_t)(w1 >> 25) & kIdxMask; }
__device__ __forceinline__ uint32_t quad_l3(uint64_t w2) { return (uint32_t)(w2)       & kIdxMask; }
__device__ __forceinline__ uint32_t quad_gi(uint64_t w2) { return (uint32_t)(w2 >> 25); }

// Rebuild a ROUND-2 element from its two parent seed indices: two seeds mixed at
// Lmix(1)=448, combined at Lout(1)=424, then mixed at Lmix(2)=424 over the 2-leaf tree.
__device__ __forceinline__ void rebuild_r2(const uint64_t pp[4], uint32_t li, uint32_t ri,
                                           bh3::Elem& out) {
    bh3::Elem a, b;
    uint32_t t1[1], t2[2];
    bh3::seed_element(pp, li, a); t1[0] = li; bh3::apply_mix(a, t1, 1u, 448u);
    bh3::seed_element(pp, ri, b); t1[0] = ri; bh3::apply_mix(b, t1, 1u, 448u);
    bh3::combine(a, b, 424u, out);
    t2[0] = li; t2[1] = ri; bh3::apply_mix(out, t2, 2u, 424u);
}

// Rebuild a ROUND-2 OUTPUT element (round 3's input) from its four leaves: two round-2
// rebuilds, combined at Lout(2)=400, then mixed at Lmix(3)=400 over the 4-leaf tree.
// Fourteen siphash rounds against the 48 B it saves reading -- which side wins is the
// measurement, and it moved once already when compile-time round constants sped the
// kernel up and the rebuild stopped hiding in its memory stalls.
__device__ __forceinline__ void rebuild_r3(const uint64_t pp[4], const uint32_t l[4],
                                           bh3::Elem& out) {
    bh3::Elem a, b;
    rebuild_r2(pp, l[0], l[1], a);
    rebuild_r2(pp, l[2], l[3], b);
    bh3::combine(a, b, 400u, out);
    uint32_t t4[4] = { l[0], l[1], l[2], l[3] };
    bh3::apply_mix(out, t4, 4u, 400u);
}

}} // namespace mxbm::cuda
