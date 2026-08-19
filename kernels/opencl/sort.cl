// Phase-D3 sort-based collision finder (the coalescing rewrite).
//
// Replaces `scatter (atomic bucket) + match (O(k^2) all-pairs)` with a stable
// LSD radix sort of (key,index) pairs by the 24-bit collision key, followed by
// a coalesced run-scan that emits collisions in order. Building block kernels
// here are unit-tested in ISOLATION (tests/test_gpu_sort.cpp) against a CPU
// std::sort / std::stable_sort oracle before being wired into the pipeline.
//
// Pair layout (u64): low 32 bits = key (only low 24 used), high 32 bits = index.
//   key(pair)   = (uint)(pair & 0xFFFFFFFF)      // collision key, 24-bit
//   index(pair) = (uint)(pair >> 32)             // element slot in this round
// Radix: tiled 4-bit digit x 6 passes over the low 24 bits => sorts the 24-bit
// key exactly. The sort is STABLE (LSD requires per-digit stability), so equal-key
// runs preserve input order -- the oracle uses std::stable_sort by key.
//
// WG (work-items per group) and the scan BLOCK are both SORT_WG. Launches are
// rounded up to a multiple of SORT_WG and every kernel bounds-checks g < n, so
// any n works. Validated in isolation in tests/test_gpu_sort.cpp.

#ifndef SORT_WG
#define SORT_WG 256u
#endif

inline uint sort_key(ulong pair)   { return (uint)(pair & 0xFFFFFFFFul); }
inline uint sort_index(ulong pair)  { return (uint)(pair >> 32); }

// ===========================================================================
// Tiled CUB-style radix (the production sort): 4-bit digit x 6 passes. Each workgroup owns a
// TILE = SORT_WG * SORT_IPT block; a coalesced strided load stages the tile into
// __local, then threads process BLOCKED sub-ranges so a per-thread private
// histogram + a cross-thread scan gives each element its stable rank in local
// (= global) order WITHOUT the O(256)-per-element loop the 1-elem reorder used.
// 16 bins keep the private histogram + bin-major scan tiny. Validated against the
// std::stable_sort oracle in test_gpu_sort before wiring into the pipeline.
// ===========================================================================
#ifndef SORT_IPT
#define SORT_IPT 8u
#endif
#define SORT_TILE (SORT_WG * SORT_IPT)
#define SORT_RADIX4 16u
#define SORT_DIG4(pair,shift) ((uint)((sort_key(pair) >> (shift)) & 0xFu))

// Tiled histogram of one 4-bit digit: 16-bin local histogram over the tile,
// written BIN-MAJOR (wghist[bin*ngroups+wg]) for the same exclusive-scan-to-base
// scheme as the 8-bit path. Reads are strided (coalesced); order is irrelevant.
__kernel void radix4_histogram(__global const ulong* pairs,
                               __global uint* wghist,
                               uint shift, uint n, uint ngroups) {
    __local uint h[SORT_RADIX4];
    uint l = get_local_id(0);
    if (l < SORT_RADIX4) h[l] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    uint wg = get_group_id(0);
    uint base = wg * SORT_TILE;
    for (uint i = 0; i < SORT_IPT; ++i) {
        uint g = base + i * SORT_WG + l;
        if (g < n) atomic_inc(&h[SORT_DIG4(pairs[g], shift)]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (l < SORT_RADIX4) wghist[l * ngroups + wg] = h[l];
}

// Tiled stable reorder. wgoffset[bin*ngroups+wg] is the global base for this
// (tile,bin). Each element's dest = that base + (same-bin elements in earlier
// threads of this tile) + (same-bin earlier elements within this thread) -- a
// stable rank in local order. Cross-thread bin bases come from a 16-bin scan
// over the SORT_WG per-thread histograms (done by 16 lanes, serial over WG).
__kernel void radix4_reorder(__global const ulong* pairs_in,
                             __global ulong* pairs_out,
                             __global const uint* wgoffset,
                             uint shift, uint n, uint ngroups) {
    __local ulong tile[SORT_TILE];
    __local uint  thist[SORT_WG * SORT_RADIX4];   // per-thread histograms -> in-place exclusive prefix
    uint l = get_local_id(0);
    uint wg = get_group_id(0);
    uint base = wg * SORT_TILE;
    uint count = (base < n) ? (n - base) : 0;
    if (count > SORT_TILE) count = SORT_TILE;

    // Coalesced strided load into __local.
    for (uint lp = l; lp < SORT_TILE; lp += SORT_WG)
        if (base + lp < n) tile[lp] = pairs_in[base + lp];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Per-thread private histogram over this thread's BLOCKED sub-range.
    uint cnt[SORT_RADIX4];
    for (uint b = 0; b < SORT_RADIX4; ++b) cnt[b] = 0;
    for (uint i = 0; i < SORT_IPT; ++i) {
        uint lp = l * SORT_IPT + i;
        if (lp < count) ++cnt[SORT_DIG4(tile[lp], shift)];
    }
    for (uint b = 0; b < SORT_RADIX4; ++b) thist[l * SORT_RADIX4 + b] = cnt[b];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Exclusive prefix across threads for each bin, IN PLACE (16 lanes, serial
    // over WG): thist[t*16+l] becomes the count of same-bin elements in threads
    // < t. Each lane owns bin column l, so the in-place rewrite is race-free.
    if (l < SORT_RADIX4) {
        uint acc = 0;
        for (uint t = 0; t < SORT_WG; ++t) {
            uint v = thist[t * SORT_RADIX4 + l];
            thist[t * SORT_RADIX4 + l] = acc;
            acc += v;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Scatter: stable rank = wgoffset[bin] + cross-thread base + within-thread run.
    uint run[SORT_RADIX4];
    for (uint b = 0; b < SORT_RADIX4; ++b) run[b] = thist[l * SORT_RADIX4 + b];
    for (uint i = 0; i < SORT_IPT; ++i) {
        uint lp = l * SORT_IPT + i;
        if (lp < count) {
            ulong pr = tile[lp];
            uint b = SORT_DIG4(pr, shift);
            pairs_out[wgoffset[b * ngroups + wg] + run[b]] = pr;
            ++run[b];
        }
    }
}

// --- generic exclusive prefix scan over a uint array ---------------------------
// Blelloch work-efficient scan of one SORT_WG-sized block; writes each block's
// total to blocksums[wg]. Host recursively scans blocksums, then scan_add folds
// the block offsets back in. Bounds-checked so any length works (tail padded 0).
__kernel void scan_block(__global const uint* in, __global uint* out,
                         __global uint* blocksums, uint n) {
    __local uint t[SORT_WG];
    uint l = get_local_id(0);
    uint base = get_group_id(0) * SORT_WG;
    t[l] = (base + l < n) ? in[base + l] : 0u;
    barrier(CLK_LOCAL_MEM_FENCE);

    // up-sweep (reduce)
    for (uint d = 1; d < SORT_WG; d <<= 1) {
        uint idx = (l + 1) * (d << 1) - 1;
        if (idx < SORT_WG) t[idx] += t[idx - d];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // clear last, down-sweep
    if (l == 0) { if (blocksums) blocksums[get_group_id(0)] = t[SORT_WG - 1]; t[SORT_WG - 1] = 0u; }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint d = SORT_WG >> 1; d >= 1; d >>= 1) {
        uint idx = (l + 1) * (d << 1) - 1;
        if (idx < SORT_WG) { uint tmp = t[idx - d]; t[idx - d] = t[idx]; t[idx] += tmp; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (base + l < n) out[base + l] = t[l];
}

// Add each block's exclusive offset (from the scanned blocksums) to its elements.
__kernel void scan_add(__global uint* out, __global const uint* blockoff, uint n) {
    uint g = get_global_id(0);
    if (g < n) out[g] += blockoff[get_group_id(0)];
}

// --- collision run-scan: count phase -------------------------------------------
// Over sorted pairs, a maximal run of equal keys of length m emits C(m,2)
// collision children. Only the run-START lane computes a count (others 0), so an
// exclusive scan of counts gives each start its coalesced output base with no
// stream compaction. Mean run ~2 (33M keys / 16M distinct) => ~1 child each.
__kernel void collision_count(__global const ulong* pairs, __global uint* counts, uint n) {
    uint g = get_global_id(0);
    if (g >= n) return;
    uint k = sort_key(pairs[g]);
    if (g > 0 && sort_key(pairs[g - 1]) == k) { counts[g] = 0u; return; }  // not a run start
    uint m = 1;
    while (g + m < n && sort_key(pairs[g + m]) == k) ++m;
    counts[g] = (m * (m - 1u)) >> 1;   // C(m,2)
}

// --- collision run-scan: emit phase --------------------------------------------
// Each run-start lane writes its C(m,2) index pairs at its scanned offset, in
// order (a<b by sorted position => coalesced output). Emitted as packed u64
// (indexA<<32 | indexB) for the isolation test; the pipeline swaps this body for
// bh3_combine + leaf/back-ref writes but keeps the identical offset scheme.
__kernel void collision_emit(__global const ulong* pairs, __global const uint* offsets,
                             __global ulong* out, uint n) {
    uint g = get_global_id(0);
    if (g >= n) return;
    uint k = sort_key(pairs[g]);
    if (g > 0 && sort_key(pairs[g - 1]) == k) return;   // only run starts emit
    uint m = 1;
    while (g + m < n && sort_key(pairs[g + m]) == k) ++m;
    uint o = offsets[g];
    for (uint a = 0; a < m; ++a)
        for (uint b = a + 1; b < m; ++b)
            out[o++] = ((ulong)sort_index(pairs[g + a]) << 32) | sort_index(pairs[g + b]);
}

// ===========================================================================
// Pipeline wiring: the sort-based replacement for round_scatter + round_match.
// Depends on bh3.cl (bh3_combine) and round.cl (BH3_MAX_LEAVES) -- compiled in
// the order {bh3, round, sort}, so both symbols are visible here.
// ===========================================================================

// round_match_sorted: the coalescing match. Over the key-sorted pairs, each
// equal-key run's START lane emits its C(m,2) children at the run's prefix-sum
// offset -- so children land DENSE and IN ORDER (out_work / left / right / leaves
// all coalesced), the entire point of the sort rewrite. The per-pair body is
// byte-for-byte the same contract as round_match: order the pair by (lead,slot)
// (lead = the element's first materialized leaf, or its own slot at r==1), combine
// at Lout (XOR-symmetric, so left/right choice doesn't change the child work),
// and grow the child's pre-order leaf prefix (left parent's s_in leaves then the
// right parent's, truncated to s_out; s_out==0 at r==5). Emission SLOT differs
// from round_match's atomic order, but every downstream consumer (mix, survivor,
// recover) is order-independent, so the solution set is identical.
__kernel void round_match_sorted(uint N, uint Lout, uint lead_identity,
                                 uint out_off, uint out_capacity,
                                 __global const ulong* sorted_pairs,
                                 __global const uint* offsets,
                                 __global const ulong* in_work,
                                 __global ulong* out_work,
                                 __global uint* all_left, __global uint* all_right,
                                 __global uint* counters,          // [2]=pair_drops
                                 __global const uint* leaves_in,   // AoS prefix (slot*9+i) of work[r]
                                 __global uint* leaves_out,        // AoS prefix of the children
                                 uint s_in, uint s_out) {          // leaves/elem in / out (s_out==0 -> skip)
    uint g = get_global_id(0);
    if (g >= N) return;
    uint ka = sort_key(sorted_pairs[g]);
    if (g > 0 && sort_key(sorted_pairs[g - 1]) == ka) return;      // only run starts emit
    uint m = 1;
    while (g + m < N && sort_key(sorted_pairs[g + m]) == ka) ++m;

    uint o = offsets[g];
    for (uint a = 0; a < m; ++a) {
        uint sa = sort_index(sorted_pairs[g + a]);
        uint la = lead_identity ? sa : leaves_in[(size_t)sa*BH3_MAX_LEAVES];
        for (uint bb = a + 1; bb < m; ++bb) {
            uint sb = sort_index(sorted_pairs[g + bb]);
            uint lb = lead_identity ? sb : leaves_in[(size_t)sb*BH3_MAX_LEAVES];
            uint left = sa, right = sb;
            if (lb < la || (lb == la && sb < sa)) { left = sb; right = sa; }
            ulong ea[7], eb[7], ec[7];
            for (int w = 0; w < 7; ++w) { ea[w] = in_work[(size_t)left*7+w]; eb[w] = in_work[(size_t)right*7+w]; }
            bh3_combine(ea, eb, Lout, ec);
            uint oi = o++;
            if (oi < out_capacity) {
                for (int w = 0; w < 7; ++w) out_work[(size_t)oi*7 + w] = ec[w];
                all_left[out_off + oi] = left; all_right[out_off + oi] = right;
                for (uint i = 0; i < s_out; ++i) {
                    uint leaf = (i < s_in) ? leaves_in[(size_t)left*BH3_MAX_LEAVES + i]
                                           : leaves_in[(size_t)right*BH3_MAX_LEAVES + (i - s_in)];
                    leaves_out[(size_t)oi*BH3_MAX_LEAVES + i] = leaf;
                }
            } else atomic_inc(&counters[2]);
        }
    }
}

// COMPACTED, FULLY CONSTANT match: the work gather/emit widths, Lout,
// lead_identity and the leaf-prefix widths are all COMPILE-TIME constants so
// every loop fully unrolls (the runtime-bound loop was the measured loss --
// test_compaction_arch: fixed-width AoS recovers the full ~25% the runtime
// version threw away; round 4's mix cost 27 ms against 8 for the same reason,
// ROUND_MIX_K in round.cl). Parents are stored at INW = inwords_for(r) words
// ([7,7,7,6,5]); children at OUTW = outwords_for(r) ([7,7,6,5,1]). Zero-filling
// ea/eb[INW..6] is bit-exact (prior Lout masked those to 0), and bh3_combine's
// child word i depends only on parent words i,i+1, so the first OUTW child words
// are identical to the full-width combine. Leaves are unaffected (BH3_MAX_LEAVES
// stride, orthogonal to work width). The constant arguments, mechanically:
//
//  * LOUT reaches bh3_combine, whose tail is `if (64*i >= Lout) out[i] = 0;` plus a
//    partial-word mask -- seven runtime comparisons per combine, in the innermost loop.
//  * LEADID picks between `sa` and a global leaf load, twice per candidate pair. As a
//    constant, round 1 loses the load entirely and rounds 2-5 lose the branch.
//  * SIN/SOUT bound the leaf-prefix copy -- a RUNTIME-bounded loop containing a RUNTIME
//    branch (`i < s_in` chooses which parent to read), executed s_out times per emitted
//    child, up to 9. Constant, it unrolls into nine unconditional loads from a known
//    parent; at SOUT == 0 (round 5) it vanishes.
//
// Nothing else changes: this is the same contract, the same emission order, and the
// goldens are byte-identical.
//
// PINU (2026-07-31): __local BALLAST, in uints, to pin OCCUPANCY -- 0 for the shipped
// kernels. The k1/k2 regression's named mechanism is that constant-folding dropped
// this kernel from the generic's 48 registers to 40, raising residency from 5 to 6
// workgroups/SM, and the extra warps cost a gather-bound kernel more in cache
// locality than they buy in latency hiding (the same shape as the capped-generic
// probe: forcing the GENERIC to 40 regs cost r1 +4.9 / r2 +9.5 ms). The untried
// corollary: keep the constants, give the occupancy back. The ballast array is
// touched only under a runtime condition that never holds (out_capacity is never 0
// on a real launch), so it is live to the compiler and free at run time. Sweep the
// size with MXBM_CL_OPTS="-DKPIN_UINTS=N": the per-SM local-memory budget this
// driver schedules against is not documented, so the pin point is found by sweep --
// 2400 uints (9.6 KB) caps at 5 workgroups under a 48 KB/SM budget, 4352 (17.4 KB)
// under a 100 KB one.
#ifndef KPIN_UINTS
#define KPIN_UINTS 2400u
#endif
#define MATCH_SORTED_K(NAME, INW, OUTW, LOUT, LEADID, SIN, SOUT, PINU)            \
__kernel void NAME(uint N, uint Lout, uint lead_identity,                         \
                   uint out_off, uint out_capacity,                              \
                   __global const ulong* sorted_pairs,                           \
                   __global const uint* offsets,                                 \
                   __global const ulong* in_work,                                \
                   __global ulong* out_work,                                     \
                   __global uint* all_left, __global uint* all_right,            \
                   __global uint* counters,                                      \
                   __global const uint* leaves_in,                              \
                   __global uint* leaves_out,                                    \
                   uint s_in, uint s_out) {                                       \
    __local uint pin_[(PINU) != 0 ? (PINU) : 1];                                  \
    uint g = get_global_id(0);                                                   \
    if ((PINU) != 0 && out_capacity == 0u) {   /* never at run time; keeps pin_ live */ \
        pin_[g % ((PINU) != 0 ? (PINU) : 1)] = N;                                \
        counters[3] = pin_[0];                                                   \
    }                                                                            \
    if (g >= N) return;                                                          \
    uint ka = sort_key(sorted_pairs[g]);                                         \
    if (g > 0 && sort_key(sorted_pairs[g - 1]) == ka) return;                    \
    uint m = 1;                                                                  \
    while (g + m < N && sort_key(sorted_pairs[g + m]) == ka) ++m;                \
    uint o = offsets[g];                                                         \
    for (uint a = 0; a < m; ++a) {                                               \
        uint sa = sort_index(sorted_pairs[g + a]);                              \
        uint la = (LEADID) ? sa : leaves_in[(size_t)sa*BH3_MAX_LEAVES];         \
        for (uint bb = a + 1; bb < m; ++bb) {                                   \
            uint sb = sort_index(sorted_pairs[g + bb]);                         \
            uint lb = (LEADID) ? sb : leaves_in[(size_t)sb*BH3_MAX_LEAVES];     \
            uint left = sa, right = sb;                                         \
            if (lb < la || (lb == la && sb < sa)) { left = sb; right = sa; }    \
            ulong ea[7], eb[7], ec[7];                                          \
            /* Zero-fill only where there is something to fill. At INW == 7 the  \
               next loop writes every word, and the redundant pass measured +2 ms \
               per round rather than being eliminated -- the constant condition   \
               folds away, an unconditional dead store apparently did not. */     \
            if ((INW) < 7) for (int w = (INW); w < 7; ++w) { ea[w] = 0ul; eb[w] = 0ul; } \
            for (int w = 0; w < (INW); ++w) {                                   \
                ea[w] = in_work[(size_t)left*(INW)+w];                          \
                eb[w] = in_work[(size_t)right*(INW)+w];                         \
            }                                                                   \
            bh3_combine(ea, eb, (LOUT), ec);                                    \
            uint oi = o++;                                                      \
            if (oi < out_capacity) {                                            \
                for (int w = 0; w < (OUTW); ++w) out_work[(size_t)oi*(OUTW) + w] = ec[w]; \
                all_left[out_off + oi] = left; all_right[out_off + oi] = right; \
                for (uint i = 0; i < (SOUT); ++i) {                             \
                    uint leaf = (i < (SIN)) ? leaves_in[(size_t)left*BH3_MAX_LEAVES + i]  \
                                            : leaves_in[(size_t)right*BH3_MAX_LEAVES + (i - (SIN))]; \
                    leaves_out[(size_t)oi*BH3_MAX_LEAVES + i] = leaf;          \
                }                                                              \
            } else atomic_inc(&counters[2]);                                   \
        }                                                                      \
    }                                                                          \
}

// MEASURED, and NOT what the mix result predicted: constant-folding these wins for
// r3/r4/r5 and LOSES for r1/r2, reproducibly, on every interleaved run.
//
//        r1 +2.8   r2 +2.3   |   r3 -1.6   r4 -0.9   r5 -1.2   ms
//
// So only k3/k4/k5 are selected (round_pipeline.cpp); r1 and r2 stay on the generic
// round_match_sorted. k1 and k2 are kept, compiled and unused, so the null stays
// re-measurable -- their presence costs nothing measurable.
//
// The split tracks s_out, the leaf-copy loop: 8 / 9 / 0 for r3 / r4 / r5, where
// unrolling it (or deleting it entirely at 0) pays, against 2 / 4 for r1 / r2 where
// it does not. What costs r1/r2 the 2-3 ms is NOT explained -- it is not the zero-fill
// (guarding it changed nothing) and LEADID=1 should if anything have helped r1, since
// it removes two global leaf loads per candidate pair.
//
// The general lesson, which the mix result on its own would have got wrong: baking a
// constant pays where it lets a PRIVATE ARRAY escape scratch (round.cl's t[8], 27 -> 8 ms)
// and is roughly free-to-negative where it only removes comparisons. Match has no
// dynamically-indexed private array -- ea/eb/ec are indexed by unrolled loop counters --
// so there was never a cliff here to find.
//                     name                  INW OUTW  Lout  leadId sIn sOut PINU
MATCH_SORTED_K(round_match_k1, 7, 7, 424u, 1, 1, 2, 0u)   // r1: LOSES, not selected
MATCH_SORTED_K(round_match_k2, 7, 7, 400u, 0, 2, 4, 0u)   // r2: LOSES, not selected
MATCH_SORTED_K(round_match_k3, 7, 6, 376u, 0, 4, 8, 0u)
MATCH_SORTED_K(round_match_k4, 6, 5, 288u, 0, 8, 9, 0u)
MATCH_SORTED_K(round_match_k5, 5, 1,  24u, 0, 9, 0, 0u)   // r5: no child leaves at all
// The occupancy-pinned pair (the ballast experiment above). Selected only under
// MXBM_MATCH_K12=2; MXBM_MATCH_K12=1 selects the unpinned k1/k2 so the original
// regression stays re-measurable in the same binary.
MATCH_SORTED_K(round_match_k1p, 7, 7, 424u, 1, 1, 2, KPIN_UINTS)
MATCH_SORTED_K(round_match_k2p, 7, 7, 400u, 0, 2, 4, KPIN_UINTS)
