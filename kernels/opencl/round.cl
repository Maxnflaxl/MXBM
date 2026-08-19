// BeamHash III Wagner-round kernels (mix / scatter / match).
// Consumes bh3.cl primitives (compiled together; bh3.cl listed first).

// E3a leaf-prefix layout: AoS with fixed stride = max prefix (9 = padNum(5)).
// element g's leaves are the contiguous block [g*9 .. g*9+9). Contiguous per-child
// writes in the match (the hot producer) beat SoA's 9 scattered cache lines.
#define BH3_MAX_LEAVES 9u

// round1_mix_seeds: seed_element(pp,begin+g) mixed with the single-leaf tree
// {begin+g} at Lmix=448 (round 1's shape), written to work[] at the ABSOLUTE
// index (begin+g)*7 -- not the batch-local g*7 -- since callers may drive
// this over multiple begin/count batches that all land in the same
// consolidated work buffer.
__kernel void round1_mix_seeds(__global const ulong* pp4, uint begin, uint count,
                               __global ulong* work /* seed-work buffer, absolute */,
                               __global uint* leaves_out /* AoS leaves for work[1]: leaf i at g*BH3_MAX_LEAVES+i */,
                               uint capacity,
                               __global ulong* pairs_out /* D3: (key,index) pairs for the sort, fused */) {
    uint g = (uint)get_global_id(0);
    if (g >= count) return;
    uint idx = begin + g;
    ulong pp[4] = { pp4[0], pp4[1], pp4[2], pp4[3] };
    ulong e[7];
    bh3_seed_element(pp, idx, e);
    uint tree1[1] = { idx };
    e[0] = bh3_apply_mix(e, tree1, 1u, 448u);
    for (int k = 0; k < 7; ++k) work[(size_t)idx*7 + k] = e[k];
    // E3a: round-1 elements have a single leaf (the seed index itself). Stored so
    // the match can grow leaf lists incrementally and round_mix skips the DFS.
    leaves_out[(size_t)idx*BH3_MAX_LEAVES] = idx;   // leaf 0 of element idx (AoS: idx*9+0)
    // D3 fusion: emit the sort pair here (mix already holds the mixed key in e[0]),
    // so the sort path needs no separate key_extract pass over work.
    pairs_out[(size_t)idx] = ((ulong)idx << 32) | (uint)(e[0] & 0xFFFFFFu);
}
// r>=2: mix in place using the element's pre-order leaf prefix. E3a: the prefix
// was materialized by the previous round's match (concat of the two parents'
// prefixes), stored AoS as leaves_in[g*BH3_MAX_LEAVES + i], so this reads padNum
// leaves directly instead of the old latency-bound back-ref DFS.
__kernel void round_mix(uint N, uint capacity, uint padNum, uint Lmix,
                        __global ulong* work,               // level's work buffer
                        __global const uint* leaves_in,     // AoS leaf prefix (g*9+i) for these elems
                        __global ulong* pairs_out) {        // D3: fused (key,index) sort pairs
    uint g = (uint)get_global_id(0);
    if (g >= N) return;
    uint tree[9];
    for (uint i = 0; i < padNum; ++i) tree[i] = leaves_in[(size_t)g*BH3_MAX_LEAVES + i];
    ulong e[7];
    for (int k = 0; k < 7; ++k) e[k] = work[(size_t)g*7 + k];
    e[0] = bh3_apply_mix(e, tree, padNum, Lmix);
    work[(size_t)g*7] = e[0];
    // D3 fusion: emit the sort pair from the freshly-mixed key (no key_extract pass).
    pairs_out[(size_t)g] = ((ulong)g << 32) | (uint)(e[0] & 0xFFFFFFu);
}
// COMPACTED, FULLY CONSTANT variants: work stored at INW = inwords_for(r)
// significant words, with padNum and Lmix baked in as well. The upper (7-INW)
// words are 0 in the uncompacted layout (prior round's bh3_combine masked them
// to Lout) and apply_mix folds only e[0..6] (rotl(0)=0), so reading INW words +
// zero-filling is bit-exact. All three MUST be compile-time constants so the
// loops fully unroll -- a runtime stride cost +24 ms (mix regressed 50->74),
// and a runtime Lmix cost round 4 alone 27.0 ms against 7.6 / 7.6 / 7.8 for
// r2 / r3 / r5, while r5 does MORE of the work (padNum 9 against 6).
//
// The Lmix mechanism is inside bh3_apply_mix, which then sees a runtime Lmix:
//
//     uint word = (Lmix + i*25u) >> 6;      t[word] |= v << sh;
//
// `t` is a private ulong[8] indexed by a value the compiler cannot resolve, so it
// cannot be kept in registers and goes to scratch -- every |= becomes a scratch
// load-modify-store. With Lmix and padNum constant the loop unrolls, every `word`
// and `sh` folds to a literal, and t[8] becomes eight registers.
//
// bh3_apply_mix itself is NOT touched (it is on the never-modify list with combine and
// siphash24) -- it is `inline` and simply gets literals at the call site.
#define ROUND_MIX_K(NAME, INW, PADN, LMIX)                                      \
__kernel void NAME(uint N, uint capacity, uint padNum, uint Lmix,               \
                   __global ulong* work,                                        \
                   __global const uint* leaves_in,                             \
                   __global ulong* pairs_out) {                                 \
    uint g = (uint)get_global_id(0);                                           \
    if (g >= N) return;                                                        \
    uint tree[9];                                                             \
    for (uint i = 0; i < (PADN); ++i) tree[i] = leaves_in[(size_t)g*BH3_MAX_LEAVES + i]; \
    ulong e[7];                                                               \
    for (int k = 0; k < 7; ++k) e[k] = 0ul;                                   \
    for (int k = 0; k < (INW); ++k) e[k] = work[(size_t)g*(INW) + k];          \
    e[0] = bh3_apply_mix(e, tree, (PADN), (LMIX));                            \
    work[(size_t)g*(INW)] = e[0];                                             \
    pairs_out[(size_t)g] = ((ulong)g << 32) | (uint)(e[0] & 0xFFFFFFu);        \
}
// (INW, padNum, Lmix) per round, from inwords_for / padnum_for / lmix_for. The kernel
// signature is unchanged so mix_level still only swaps a name; the three arguments are
// simply ignored by these variants.
ROUND_MIX_K(round_mix_k2, 7, 2, 424)
ROUND_MIX_K(round_mix_k3, 7, 4, 400)
ROUND_MIX_K(round_mix_k4, 6, 6, 376)
ROUND_MIX_K(round_mix_k5, 5, 9, 288)
// survivor_scan: scans round 5's children (in work[0] -- the r==5 special
// case, which has no round 6 / work[6] to feed) for the ALL-ZERO
// work vector. bh3_combine is XOR-based: a byte-identical pair (a==b in all
// 7 words) XORs to zero, and Lout=24's masking of an already-zero word stays
// zero -- so an all-zero child is EXACTLY "two round-4 elements collided AND
// were byte-identical", a deterministic marker rather than a probabilistic
// one. One work-item per candidate; counters[3] is the atomic (uncapped)
// survivor count, repurposed from the otherwise-unused "spare" counter slot
// -- the HOST must zero it before every launch (it is not self-resetting,
// so a second call on the same counters buffer would otherwise accumulate).
// Entries at oi>=cap are silently dropped -- no separate overflow counter,
// since survivors are expected to be vanishingly rare against real data.
__kernel void survivor_scan_s(uint N, uint stride, __global const ulong* work,
                              __global uint* out_slots, uint cap,
                              __global uint* counters /* [3]=survivors */) {
    uint g = (uint)get_global_id(0);
    if (g >= N) return;
    if (work[(size_t)g*stride] != 0ul) return;
    uint oi = atomic_inc(&counters[3]);
    if (oi < cap) out_slots[oi] = g;
}
// Seed indices are 25-bit (2^25 seeds); every record below packs them at that width,
// and so do the pair/quad records in lds.cl, which is compiled after this file.
#define RD2_IDXMASK 0x1FFFFFFu

// QUAD RECORD WITHOUT gi, 16 B, for the octo rungs only. Nothing reads a round-2
// element's gi there: the reference rows it would index are not allocated, and round 3's
// left/right tiebreak takes the staged slot instead. That leaves key 24 + 4 x 25 = 124
// bits, which fits 2 u64 where the record above needs 3.
//   w0 = key | l0<<24 | l1<<49        w1 = l1>>15 | l2<<10 | l3<<35
inline ulong q16_w0(uint key, uint l0, uint l1) {
    return (ulong)key | ((ulong)l0 << 24) | ((ulong)l1 << 49);
}
inline ulong q16_w1(uint l1, uint l2, uint l3) {
    return ((ulong)l1 >> 15) | ((ulong)l2 << 10) | ((ulong)l3 << 35);
}
inline uint q16_l0(ulong w0)           { return (uint)(w0 >> 24) & RD2_IDXMASK; }
inline uint q16_l1(ulong w0, ulong w1) { return (uint)(((w0 >> 49) | (w1 << 15)) & RD2_IDXMASK); }
inline uint q16_l2(ulong w1)           { return (uint)(w1 >> 10) & RD2_IDXMASK; }
inline uint q16_l3(ulong w1)           { return (uint)(w1 >> 35) & RD2_IDXMASK; }

// OCTO RECORD (round 3 -> round 4), 32 B instead of 64. The quad argument one round
// further down: a round-3 OUTPUT element combines two round-3 inputs, each determined by
// four seed indices, so eight indices determine it and those eight ARE its leaves. Its
// six work words, its lead (= leaf 0) and its leftContrib (a mix of the eight leaves over
// a zero element) are then all derivable, and round 4 rebuilds them with rd_elem4.
//
// key 24 + 8 x 25 + gi 26 = 250 bits of the 256 that 4 u64 give, so three leaves straddle
// a word boundary and no layout avoids it. w0's low 24 bits are the key, as in every
// other record, so the staging loop's extraction is shared.
//   w0 = key    | l0<<24 | l1<<49
//   w1 = l1>>15 | l2<<10 | l3<<35 | l4<<60
//   w2 = l4>>4  | l5<<21 | l6<<46
//   w3 = l6>>18 | l7<<7  | gi<<32
inline ulong oc_w0(uint key, uint l0, uint l1) {
    return (ulong)key | ((ulong)l0 << 24) | ((ulong)l1 << 49);
}
inline ulong oc_w1(uint l1, uint l2, uint l3, uint l4) {
    return ((ulong)l1 >> 15) | ((ulong)l2 << 10) | ((ulong)l3 << 35) | ((ulong)l4 << 60);
}
inline ulong oc_w2(uint l4, uint l5, uint l6) {
    return ((ulong)l4 >> 4) | ((ulong)l5 << 21) | ((ulong)l6 << 46);
}
inline ulong oc_w3(uint l6, uint l7, uint gi) {
    return ((ulong)l6 >> 18) | ((ulong)l7 << 7) | ((ulong)gi << 32);
}
// Round 4's back-references name a round-3 parent by SLOT, and a slot is HALF-LOCAL: a
// split record set numbers both halves from zero. This bit rides in the reference to say
// which half, leaving 31 bits for a slot number that never exceeds nb/2 * cap + pool.
#define LDS_OCTO_HI 0x80000000u
inline void oc_leaves(ulong w0, ulong w1, ulong w2, ulong w3, uint l[8]) {
    l[0] = (uint)(w0 >> 24) & RD2_IDXMASK;
    l[1] = (uint)(((w0 >> 49) | (w1 << 15)) & RD2_IDXMASK);
    l[2] = (uint)(w1 >> 10) & RD2_IDXMASK;
    l[3] = (uint)(w1 >> 35) & RD2_IDXMASK;
    l[4] = (uint)(((w1 >> 60) | (w2 << 4)) & RD2_IDXMASK);
    l[5] = (uint)(w2 >> 21) & RD2_IDXMASK;
    l[6] = (uint)(((w2 >> 46) | (w3 << 18)) & RD2_IDXMASK);
    l[7] = (uint)(w3 >> 7) & RD2_IDXMASK;
}

// W0-CHECKPOINT OCTO RECORD: the same 32 B carrying the child's post-mix work word 0
// where the record above carries a key and a gi, so round 4 derives only the linear lane
// -- every apply_mix in the rebuild and 8 of its 56 siphashes go. It lives here rather
// than in lds.cl because recover reads it too, and one definition cannot drift.
// -DLDS_OW0=0 moves the emit, the rebuild and recover back together.
#ifndef LDS_OW0
#define LDS_OW0 1
#endif
// Word 0's 40 bits are paid for out of two fields carrying less than they cost:
//   * gi, all 26. Nothing indexes a round-3 element by gi on an octo build -- the rows
//     that did are not allocated -- and round 4's tiebreak takes the staged slot instead.
//   * 14 of the 24 key bits. Only the low 24 - bb reach the sub-mask filter, the chain
//     hash and the spill rescan, and bb + sm = 17 makes that 7 + sm <= 10 on all three
//     octo rungs. The rest are the bucket address.
// Round 4's combine reads word 0's bits 24..63 alone, so the dropped key bits are read by
// nothing. 10 + 40 + 8 x 25 = 250 bits of 256, and no field is a build parameter.
//   w0 = key low 10 | w0[24..63]<<10 | l0<<50
//   w1 = l0>>14 | l1<<11 | l2<<36 | l3<<61
//   w2 = l3>>3  | l4<<22 | l5<<47
//   w3 = l5>>17 | l6<<8  | l7<<33
inline ulong ow0_w0(ulong w0, uint l0) {
    return (w0 & 0x3FFul) | (((w0 >> 24) & 0xFFFFFFFFFFul) << 10) | ((ulong)l0 << 50);
}
inline ulong ow0_w1(uint l0, uint l1, uint l2, uint l3) {
    return ((ulong)l0 >> 14) | ((ulong)l1 << 11) | ((ulong)l2 << 36) | ((ulong)l3 << 61);
}
inline ulong ow0_w2(uint l3, uint l4, uint l5) {
    return ((ulong)l3 >> 3) | ((ulong)l4 << 22) | ((ulong)l5 << 47);
}
inline ulong ow0_w3(uint l5, uint l6, uint l7) {
    return ((ulong)l5 >> 17) | ((ulong)l6 << 8) | ((ulong)l7 << 33);
}
inline void ow0_leaves(ulong w0, ulong w1, ulong w2, ulong w3, uint l[8]) {
    l[0] = (uint)(((w0 >> 50) | (w1 << 14)) & RD2_IDXMASK);
    l[1] = (uint)(w1 >> 11) & RD2_IDXMASK;
    l[2] = (uint)(w1 >> 36) & RD2_IDXMASK;
    l[3] = (uint)(((w1 >> 61) | (w2 << 3)) & RD2_IDXMASK);
    l[4] = (uint)(w2 >> 22) & RD2_IDXMASK;
    l[5] = (uint)(((w2 >> 47) | (w3 << 17)) & RD2_IDXMASK);
    l[6] = (uint)(w3 >> 8) & RD2_IDXMASK;
    l[7] = (uint)(w3 >> 33) & RD2_IDXMASK;
}
// Word 0 as round 4 stages it: the kept key bits where every mode puts them, and bits
// 24..63 where combine reads them. Bits 10..23 come back zero and nothing consults them.
inline ulong ow0_word0(ulong r0) {
    return (r0 & 0x3FFul) | (((r0 >> 10) & 0xFFFFFFFFFFul) << 24);
}

// recover: one work-item per survivor. Walk the consolidated back-refs from the
// round-5 survivor (slot in work[0], back-ref at row 4 = (5-1)*capacity) down to
// round 1 (row 0, whose left/right entries ARE leaf indices), collecting all 32
// leaves in tree order (pre-order: left subtree fully, then right -- the same
// order round_mix's DFS uses and the verifier expects). No apply_mix.
// OCTO: only rows 4 and 5 exist, row 4 at offset 0 and row 5 at `capacity`. Row 4 names
// its parents by SLOT in round 3's output set, and each of those records carries the
// eight seed indices below it -- in the same left-to-right order this walk would have
// produced, because round 3 built its leaf tree and its references from the same
// leftPos/rightPos. So the walk is two levels deep instead of five, and rows 1-3 are
// never allocated. r3_elem/r3_stride bind that set; off an octo build they are unread.
__kernel void recover(uint nSurv, __global const uint* surv_slots, uint capacity,
                      __global const uint* all_left, __global const uint* all_right,
                      __global uint* out /* [nSurv*32] */,
                      __global const ulong* r3_elem, uint r3_stride,
                      __global const ulong* r3_elem_hi) {
    uint i = (uint)get_global_id(0);
    if (i >= nSurv) return;
    uint got = 0u;
#if LDS_OCTO
    uint s5 = surv_slots[i];
    uint g4[2] = { all_left[capacity + s5], all_right[capacity + s5] };
    for (int k = 0; k < 2; ++k) {
        uint sl[2] = { all_left[g4[k]], all_right[g4[k]] };
        for (int j = 0; j < 2; ++j) {
            __global const ulong* hset = (sl[j] & LDS_OCTO_HI) ? r3_elem_hi : r3_elem;
            __global const ulong* r = hset + (size_t)(sl[j] & ~LDS_OCTO_HI) * r3_stride;
            uint l[8];
            if (LDS_OW0) ow0_leaves(r[0], r[1], r[2], r[3], l);
            else         oc_leaves (r[0], r[1], r[2], r[3], l);
            for (int t = 0; t < 8 && got < 32u; ++t)
                out[(size_t)i*32u + got++] = l[t];
        }
    }
    return;
#endif
    uint lvl[8], slt[8]; int sp = 0;
    lvl[0] = 5u; slt[0] = surv_slots[i]; sp = 1;
    while (sp > 0 && got < 32u) {
        --sp;
        uint lv = lvl[sp], sl = slt[sp];
        uint off = (lv - 1u) * capacity;
        uint L = all_left[off + sl], R = all_right[off + sl];
        if (lv == 1u) {
            out[(size_t)i*32u + got] = L; ++got;
            if (got < 32u) { out[(size_t)i*32u + got] = R; ++got; }
        } else {
            lvl[sp] = lv - 1u; slt[sp] = R; ++sp;   // right pushed first,
            lvl[sp] = lv - 1u; slt[sp] = L; ++sp;   // left pops first (pre-order)
        }
    }
}
