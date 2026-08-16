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

// r1 -> r2, W0-CHECKPOINT RECORD (MXBM_PAIR_W0): the same 16 B, carrying the child's
// post-mix work word 0 where the plain record carries only its 24-bit key. apply_mix
// writes word 0 alone and combine is XOR-plus-shift, so words 1..6 of a round-2 element
// are linear in the parents' SEED words 1..6: storing word 0 lets round 2 skip both
// parent mixes, the child mix and two of the fourteen siphashes -- see rebuild_r2_lane.
//
// Word 0's extra 24 bits come out of the bucket ADDRESS, the same resource the r2 -> r3
// pack spends: IMPB of the key's 24 bits ARE the bucket index, so word 0 is stored with
// them removed and the consumer reinserts them from its own bucket. The kept low
// 24-IMPB key bits stay at the bottom of word 0, so the staging loop's sub-mask filter
// and tab hash read them where they always were -- which is why this needs IMPB, and
// with it the perfect-tab compare.
//   w0 = key low (24-IMPB) | word-0 bits 63..24 | gi bits 25..14
//   w1 = li | ri<<25 | (gi low 14)<<50
// 48 + 25 + 25 + 26 = 124 bits of 128 at IMPB=16, 123 at 17. gi is 26 bits for the same
// reason it is in r3_p1: a round emits ~2^25 children and kCapacity is 34,603,008.
//
// CUDA only -- unlike the records above, this one has no counterpart in lds.cl. The two
// backends never read each other's buffers, so they may differ here; OpenCL has no
// implicit-bits pack and so no address bits to spend.
template<uint32_t IMPB>
__device__ __forceinline__ uint64_t pw0_w0(uint64_t w0, uint32_t gi) {
    constexpr uint32_t impKB = 24u - IMPB;              // kept key bits
    return (w0 & ((1ull << impKB) - 1u)) | ((w0 >> 24) << impKB)
         | ((uint64_t)(gi >> 14) << (64u - IMPB));
}
__device__ __forceinline__ uint64_t pw0_w1(uint32_t li, uint32_t ri, uint32_t gi) {
    return (uint64_t)li | ((uint64_t)ri << 25) | ((uint64_t)(gi & 0x3FFFu) << 50);
}
// Inverse: word 0 whole again, with the block's bucket back in the dropped bits.
template<uint32_t IMPB>
__device__ __forceinline__ uint64_t pw0_word0(uint64_t r0, uint32_t bucket) {
    constexpr uint32_t impKB = 24u - IMPB;
    return (r0 & ((1ull << impKB) - 1u)) | ((uint64_t)bucket << impKB)
         | (((r0 >> impKB) & 0xFFFFFFFFFFull) << 24);
}
__device__ __forceinline__ uint32_t pw0_left (uint64_t r1) { return (uint32_t)(r1)       & kIdxMask; }
__device__ __forceinline__ uint32_t pw0_right(uint64_t r1) { return (uint32_t)(r1 >> 25) & kIdxMask; }
template<uint32_t IMPB>
__device__ __forceinline__ uint32_t pw0_gi(uint64_t r0, uint64_t r1) {
    return (uint32_t)(r1 >> 50) | ((((uint32_t)(r0 >> (64u - IMPB))) & 0xFFFu) << 14);
}

// The linear lane: words 1..6 of a round-2 element from 12 siphashes (k = 1..6 per
// parent) and the combine's shift-XOR, masked at Lout(1) = 424. Word 0 is not written --
// the caller stages the stored checkpoint there.
__device__ __forceinline__ void rebuild_r2_lane(const uint64_t pp[4], uint32_t li, uint32_t ri,
                                                uint64_t* w16 /* w16[1..6] written */) {
    uint64_t x[7];
    #pragma unroll
    for (int k = 1; k < 7; ++k)
        x[k] = bh3::siphash24(pp[0], pp[1], pp[2], pp[3], ((uint64_t)li << 3) + (uint64_t)k)
             ^ bh3::siphash24(pp[0], pp[1], pp[2], pp[3], ((uint64_t)ri << 3) + (uint64_t)k);
    #pragma unroll
    for (int i = 1; i < 6; ++i) w16[i] = (x[i] >> 24) | (x[i + 1] << 40);
    w16[6] = (x[6] >> 24) & ((1ull << 40) - 1);        // Lout(1) = 424: word 6 keeps 40 bits
}

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

// r2 -> r3, QUAD RECORD WITHOUT gi: 16 B instead of 24, for the octo rungs only. There,
// nothing reads a round-2 element's gi -- the reference rows it would index are not
// allocated, and the only other consumer is round 3's left/right tiebreak, which takes
// the slot instead. Dropping it leaves key 24 + 4 x 25 = 124 bits, which fits 2 u64.
//   w0 = key | l0<<24 | l1<<49        w1 = l1>>15 | l2<<10 | l3<<35
__device__ __forceinline__ uint64_t quad16_w0(uint32_t key, const uint32_t l[4]) {
    return (uint64_t)key | ((uint64_t)l[0] << 24) | ((uint64_t)l[1] << 49);
}
__device__ __forceinline__ uint64_t quad16_w1(const uint32_t l[4]) {
    return ((uint64_t)l[1] >> 15) | ((uint64_t)l[2] << 10) | ((uint64_t)l[3] << 35);
}
__device__ __forceinline__ void quad16_leaves(uint64_t w0, uint64_t w1, uint32_t l[4]) {
    l[0] = (uint32_t)(w0 >> 24) & kIdxMask;
    l[1] = (uint32_t)(((w0 >> 49) | (w1 << 15)) & kIdxMask);
    l[2] = (uint32_t)(w1 >> 10) & kIdxMask;
    l[3] = (uint32_t)(w1 >> 35) & kIdxMask;
}

// r3 -> r4, OCTO RECORD: 32 B instead of 64. The same argument one round further down --
// a round-3 output element is a combine of two round-3 inputs, each determined by four
// seed indices, so EIGHT indices determine it and those eight ARE its leaves (sBuild = 8).
// The 6 work words, the lead and the leftContrib are then all redundant: leaf 0 IS the
// lead, the leftContrib is a mix of the eight leaves alone, and round 4 rebuilds the work
// words with rebuild_r4.
//
// No slacker layout exists: key 24 + 8 x 25 + gi 26 = 250 bits of 256, so three leaves
// straddle a word boundary and every field is a shift.
//   w0 = key      | l0<<24 | l1<<49
//   w1 = l1>>15   | l2<<10 | l3<<35 | l4<<60
//   w2 = l4>>4    | l5<<21 | l6<<46
//   w3 = l6>>18   | l7<<7  | gi<<32
// w0's low 24 bits are the key, exactly as in the pair and quad records, so the staging
// loop's `rec0 & 0xFFFFFF` extraction is shared with every other mode.
__device__ __forceinline__ uint64_t octo_w0(uint32_t key, const uint32_t l[8]) {
    return (uint64_t)key | ((uint64_t)l[0] << 24) | ((uint64_t)l[1] << 49);
}
__device__ __forceinline__ uint64_t octo_w1(const uint32_t l[8]) {
    return ((uint64_t)l[1] >> 15) | ((uint64_t)l[2] << 10) | ((uint64_t)l[3] << 35)
         | ((uint64_t)l[4] << 60);
}
__device__ __forceinline__ uint64_t octo_w2(const uint32_t l[8]) {
    return ((uint64_t)l[4] >> 4) | ((uint64_t)l[5] << 21) | ((uint64_t)l[6] << 46);
}
__device__ __forceinline__ uint64_t octo_w3(const uint32_t l[8], uint32_t gi) {
    return ((uint64_t)l[6] >> 18) | ((uint64_t)l[7] << 7) | ((uint64_t)gi << 32);
}
__device__ __forceinline__ void octo_leaves(uint64_t w0, uint64_t w1, uint64_t w2,
                                            uint64_t w3, uint32_t l[8]) {
    l[0] = (uint32_t)(w0 >> 24) & kIdxMask;
    l[1] = (uint32_t)(((w0 >> 49) | (w1 << 15)) & kIdxMask);
    l[2] = (uint32_t)(w1 >> 10) & kIdxMask;
    l[3] = (uint32_t)(w1 >> 35) & kIdxMask;
    l[4] = (uint32_t)(((w1 >> 60) | (w2 << 4)) & kIdxMask);
    l[5] = (uint32_t)(w2 >> 21) & kIdxMask;
    l[6] = (uint32_t)(((w2 >> 46) | (w3 << 18)) & kIdxMask);
    l[7] = (uint32_t)(w3 >> 7) & kIdxMask;
}
__device__ __forceinline__ uint32_t octo_lead(uint64_t w0) { return (uint32_t)(w0 >> 24) & kIdxMask; }
__device__ __forceinline__ uint32_t octo_gi  (uint64_t w3) { return (uint32_t)(w3 >> 32); }

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

// Rebuild a ROUND-3 OUTPUT element (round 4's input) from its eight leaves: two round-3
// rebuilds, combined at Lout(3)=376, then mixed at Lmix(4)=376 over the 8-leaf tree. The
// 6 is what apply_mix computes for itself at Lmix=376 ((512-376+24)/25), so the tree
// length is not a second thing to keep in step. Twenty-eight siphash rounds against the
// 32 B it saves reading, which is why it is a reach rung and never a default.
__device__ __forceinline__ void rebuild_r4(const uint64_t pp[4], const uint32_t l[8],
                                           bh3::Elem& out) {
    bh3::Elem a, b;
    rebuild_r3(pp, l,     a);
    rebuild_r3(pp, l + 4, b);
    bh3::combine(a, b, 376u, out);
    bh3::apply_mix(out, l, 6u, 376u);
}

// r3 -> r4, OCTO W0-CHECKPOINT RECORD (MXBM_OCTO_W0): the same 32 B, carrying the child's
// post-mix work word 0 where the plain octo record carries a 24-bit key and a gi. Word 0
// is what deletes the mixes -- see rebuild_r4_lane -- and it is paid for out of two fields
// that were carrying less than they cost:
//
//   * gi, all 26 bits. Nothing indexes a reference row by a round-3 element's gi any more,
//     and its one remaining consumer is round 4's left/right tiebreak, which takes the
//     input slot instead. The same substitution the quad16 record makes one round up.
//   * 14 of the key's 24 bits. Only the low 24 - bb reach the sub-mask filter, the chain
//     hash and the in-group key compare, and bb + sm = 17 makes that exactly 7 + sm <= 10
//     on every rung the octo record runs on. The rest are the bucket address.
//
// Word 0's own low 24 bits are the key, and round 4's combine reads bits 24..63 alone
// (`out.w[0] = (x[0] >> 24) | (x[1] << 40)`), so the 14 dropped bits are read by nothing.
// 10 + 40 + 8 x 25 = 250 bits of 256, and the layout is fixed -- no template parameter and
// no address arithmetic, unlike the r2 -> r3 implicit-bits pack.
//   w0 = key low 10 | w0[24..63]<<10 | l0<<50
//   w1 = l0>>14 | l1<<11 | l2<<36 | l3<<61
//   w2 = l3>>3  | l4<<22 | l5<<47
//   w3 = l5>>17 | l6<<8  | l7<<33
__device__ __forceinline__ uint64_t ow0_w0(uint64_t w0, const uint32_t l[8]) {
    return (w0 & 0x3FFull) | (((w0 >> 24) & 0xFFFFFFFFFFull) << 10) | ((uint64_t)l[0] << 50);
}
__device__ __forceinline__ uint64_t ow0_w1(const uint32_t l[8]) {
    return ((uint64_t)l[0] >> 14) | ((uint64_t)l[1] << 11) | ((uint64_t)l[2] << 36)
         | ((uint64_t)l[3] << 61);
}
__device__ __forceinline__ uint64_t ow0_w2(const uint32_t l[8]) {
    return ((uint64_t)l[3] >> 3) | ((uint64_t)l[4] << 22) | ((uint64_t)l[5] << 47);
}
__device__ __forceinline__ uint64_t ow0_w3(const uint32_t l[8]) {
    return ((uint64_t)l[5] >> 17) | ((uint64_t)l[6] << 8) | ((uint64_t)l[7] << 33);
}
__device__ __forceinline__ void ow0_leaves(uint64_t w0, uint64_t w1, uint64_t w2,
                                           uint64_t w3, uint32_t l[8]) {
    l[0] = (uint32_t)(((w0 >> 50) | (w1 << 14)) & kIdxMask);
    l[1] = (uint32_t)(w1 >> 11) & kIdxMask;
    l[2] = (uint32_t)(w1 >> 36) & kIdxMask;
    l[3] = (uint32_t)(((w1 >> 61) | (w2 << 3)) & kIdxMask);
    l[4] = (uint32_t)(w2 >> 22) & kIdxMask;
    l[5] = (uint32_t)(((w2 >> 47) | (w3 << 17)) & kIdxMask);
    l[6] = (uint32_t)(w3 >> 8) & kIdxMask;
    l[7] = (uint32_t)(w3 >> 33) & kIdxMask;
}
// Word 0 as round 4 stages it: the kept key bits where every mode puts them, and bits
// 24..63 where combine reads them. Bits 10..23 come back zero and nothing consults them --
// the same trade the terminal record's t5_work makes.
__device__ __forceinline__ uint64_t ow0_word0(uint64_t w0) {
    return (w0 & 0x3FFull) | (((w0 >> 10) & 0xFFFFFFFFFFull) << 24);
}
__device__ __forceinline__ uint32_t ow0_lead(uint64_t w0, uint64_t w1) {
    return (uint32_t)(((w0 >> 50) | (w1 << 14)) & kIdxMask);      // leaf 0 IS the lead
}

// Words 1..6 of a node from its two children's words 1..6. combine's word i reads x[i] and
// x[i+1] alone, so word 0 never enters the lane; the masking is combine's, term for term.
__device__ __forceinline__ void combine_hi(const uint64_t a[7], const uint64_t b[7],
                                           uint32_t Lout, uint64_t out[7]) {
    uint64_t x[7];
    #pragma unroll
    for (int i = 1; i < 7; ++i) x[i] = a[i] ^ b[i];
    #pragma unroll
    for (int i = 1; i < 7; ++i) {
        uint64_t v = (x[i] >> 24) | (i < 6 ? (x[i + 1] << 40) : 0);
        const uint32_t base = 64u * (uint32_t)i;
        if (base >= Lout)           v = 0;
        else if (Lout - base < 64u) v &= (((uint64_t)1 << (Lout - base)) - 1);
        out[i] = v;
    }
}

// Words 1..6 of a ROUND-3 OUTPUT element from its eight leaves, with no `apply_mix` at all
// and six siphashes a leaf instead of seven. The same argument rebuild_r2_lane rests on,
// three levels deep: apply_mix writes work word 0 alone, so the whole 15-mix chain and each
// leaf's k = 0 seed word reach word 0 and nothing else. The caller stages word 0 from the
// record; everything below word 0 is linear in the leaves' seed words 1..6.
__device__ __forceinline__ void rebuild_r4_lane(const uint64_t pp[4], const uint32_t l[8],
                                                uint64_t* w16 /* w16[1..6] written */) {
    uint64_t a[7], b[7], r0[7], r1[7];
    rebuild_r2_lane(pp, l[0], l[1], a);
    rebuild_r2_lane(pp, l[2], l[3], b);
    combine_hi(a, b, 400u, r0);
    rebuild_r2_lane(pp, l[4], l[5], a);
    rebuild_r2_lane(pp, l[6], l[7], b);
    combine_hi(a, b, 400u, r1);
    combine_hi(r0, r1, 376u, w16);
}

// The leftContrib round 3 stored: a mix of the eight leaves ALONE, over a zero element,
// so it needs no work state and is the cheap half of what the octo record gives up.
__device__ __forceinline__ uint64_t octo_contrib(const uint32_t l[8]) {
    bh3::Elem z{};
    bh3::apply_mix(z, l, 8u, 288u);
    return bh3::rotl64(z.w[0], 40);
}

// r4 -> terminal, 8 B instead of 16. The terminal round combines at Lout = 24, so only
// word 0's bits 24..47 reach the acceptance test and only the 24 - bb key bits the bucket
// address does not already carry reach the sub-mask filter and the chain hash -- 32 bits
// of information in what was a 16 B record. gi and lead are not stored at all: nothing
// indexes a reference row by them any more, and combine is symmetric, so the left/right
// order is settled by the replay two levels down instead of here.
//   bits  0..9   word 0's low 10 key bits (>= 24 - bb for every rung on the ladder)
//   bits 10..33  word 0 bits 24..47
//   bits 34..50  the emitting block's input bucket, for replay_r4
//   bits 51..63  word 0 bits 48..60, widening the replay's content match to 47 bits
__device__ __forceinline__ uint64_t t5_rec(uint64_t w0, uint32_t bucket) {
    return (w0 & 0x3FFull) | (((w0 >> 24) & 0xFFFFFFull) << 10)
         | ((uint64_t)bucket << 34) | (((w0 >> 48) & 0x1FFFull) << 51);
}
// Word 0 as the terminal stages it: key in the low bits where every mode puts it, and
// bits 24..47 where combine reads them. The bucket bits of the key are not restored --
// nothing downstream of here consults them.
__device__ __forceinline__ uint64_t t5_work(uint64_t r) {
    return (r & 0x3FFull) | (((r >> 10) & 0xFFFFFFull) << 24);
}
__device__ __forceinline__ uint32_t t5_bucket(uint64_t r) {
    return (uint32_t)((r >> 34) & 0x1FFFFu);
}
// The 47 bits replay_r4 matches a candidate child against, from the record and from a
// freshly combined word 0 respectively.
__device__ __forceinline__ uint64_t t5_ident(uint64_t r) {
    return (r & 0x3FFull) | (((r >> 10) & 0xFFFFFFull) << 10) | ((r >> 51) << 34);
}
__device__ __forceinline__ uint64_t t5_ident_w0(uint64_t w0) {
    return (w0 & 0x3FFull) | (((w0 >> 24) & 0xFFFFFFull) << 10)
         | (((w0 >> 48) & 0x1FFFull) << 34);
}

}} // namespace mxbm::cuda
