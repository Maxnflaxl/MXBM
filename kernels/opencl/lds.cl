// P2a: LDS-local bucketed collision finder (isolated mechanism validation).
//
// Mirrors beam's equihash_150_5.cl structure (verified 2026-07-24): bucket the
// elements by the TOP bits of the 24-bit collision key, split each bucket by a
// low SUB-MASK so one workgroup owns a small group, stage that group into LDS,
// build a hash table over the middle key bits via atomic_xchg linked lists, then
// walk the chains to emit every equal-FULL-key pair. NO global parent-data gather
// (parents live in LDS). Validated against a CPU group-by-key oracle in
// tests/test_lds_collide.cpp before any pipeline wiring.
//
// This isolated version carries a lightweight element = (index<<32)|key (8 B) so
// it validates the bucketing + chaining MECHANISM independent of the real 56 B
// element (whose LDS-staging strategy is a P2b fork). key = bits[0,24).

#ifndef LDS_WG
#define LDS_WG 256u
#endif
#ifndef LDS_GROUPCAP
#define LDS_GROUPCAP 1216u   // max survivors staged per (bucket,sub-mask) group
#endif
#ifndef LDS_TABSIZE
#define LDS_TABSIZE 512u     // LDS hash-table entries (power of two)
#endif
#define LDS_EMPTY 0xFFFFFFFFu

inline uint lds_key(ulong e)  { return (uint)(e & 0xFFFFFFu); }
inline uint lds_idx(ulong e)  { return (uint)(e >> 32); }

// p2_scatter: bucket N (index<<32|key) elements by the top `bucket_bits` of the
// 24-bit key into a bucket-contiguous buffer. counts[b] is the raw arrival count
// (uncapped); arrivals past bucket_cap are dropped and tallied in drops[0].
__kernel void p2_scatter(uint N, uint bucket_bits, uint bucket_cap,
                         __global const ulong* elems,
                         __global uint* counts,       // [num_buckets], pre-zeroed
                         __global ulong* buckets,      // [num_buckets*bucket_cap]
                         __global uint* drops) {       // [0]=bucket overflow
    uint g = get_global_id(0);
    if (g >= N) return;
    ulong e = elems[g];
    uint b = lds_key(e) >> (24u - bucket_bits);
    uint pos = atomic_inc(&counts[b]);
    if (pos < bucket_cap) buckets[(size_t)b*bucket_cap + pos] = e;
    else atomic_inc(&drops[0]);
}

// p2_collide: one workgroup per (bucket, sub-mask) pair. Grid = num_buckets *
// 2^submask_bits workgroups. Stage the bucket's elements whose low submask_bits
// == this group's mask into LDS, chain them by the middle key bits via
// atomic_xchg, then walk each chain emitting (min,max) index pairs for equal
// full keys. Every unordered colliding pair is emitted exactly once (by the
// later-inserted member walking back to the earlier one).
__kernel void p2_collide(uint bucket_bits, uint submask_bits, uint bucket_cap,
                         __global const uint* counts,
                         __global const ulong* buckets,
                         __global ulong* out_pairs,    // (min<<32)|max
                         __global uint* out_count,     // [0] atomic cursor
                         uint out_cap,
                         __global uint* drops) {        // [1]=group ovf,[2]=out ovf,[3]=chain cap
    __local uint lkey[LDS_GROUPCAP];
    __local uint lidx[LDS_GROUPCAP];
    __local uint lchain[LDS_GROUPCAP];
    __local uint tab[LDS_TABSIZE];
    __local uint gcount;

    uint lId = get_local_id(0);
    uint submaskCount = 1u << submask_bits;
    uint bucket = get_group_id(0) / submaskCount;
    uint mask   = get_group_id(0) % submaskCount;

    if (lId == 0) gcount = 0;
    for (uint i = lId; i < LDS_TABSIZE; i += LDS_WG) tab[i] = LDS_EMPTY;
    barrier(CLK_LOCAL_MEM_FENCE);

    uint cnt = counts[bucket];
    if (cnt > bucket_cap) cnt = bucket_cap;
    size_t base = (size_t)bucket * bucket_cap;

    // Stage this sub-mask's elements into LDS (coalesced read over the bucket).
    for (uint p = lId; p < cnt; p += LDS_WG) {
        ulong e = buckets[base + p];
        uint key = lds_key(e);
        if ((key & (submaskCount - 1u)) != mask) continue;
        uint pos = atomic_inc(&gcount);
        if (pos < LDS_GROUPCAP) { lkey[pos] = key; lidx[pos] = lds_idx(e); }
        else atomic_inc(&drops[1]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint total = gcount < LDS_GROUPCAP ? gcount : LDS_GROUPCAP;

    // Chain: hash on the middle bits (above the sub-mask), insert via atomic_xchg
    // (returns the previous head -> singly-linked collision chain per hash key).
    for (uint pos = lId; pos < total; pos += LDS_WG) {
        uint hk = (lkey[pos] >> submask_bits) & (LDS_TABSIZE - 1u);
        lchain[pos] = atomic_xchg(&tab[hk], pos);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Walk: each slot follows its chain back to EARLIER same-hash slots; emit on
    // equal full key (verify guards hash-collision false pairs).
    for (uint pos = lId; pos < total; pos += LDS_WG) {
        uint key = lkey[pos], a = lidx[pos];
        uint oth = lchain[pos], walk = 0;
        while (oth != LDS_EMPTY) {
            if (++walk > 64u) { atomic_inc(&drops[3]); break; }
            if (lkey[oth] == key) {
                uint b = lidx[oth];
                uint lo = a < b ? a : b, hi = a < b ? b : a;
                uint oi = atomic_inc(out_count);
                if (oi < out_cap) out_pairs[oi] = ((ulong)lo << 32) | hi;
                else atomic_inc(&drops[2]);
            }
            oth = lchain[oth];
        }
    }
}

// ===========================================================================
// P2b: stage the REAL 7-u64 element into LDS and bh3_combine colliding parents
// locally -- no per-collision global gather. Needs bh3.cl (bh3_combine); compile
// {bh3, lds}. Strategy A (full element in LDS): forces smaller groups (LDS holds
// 7 u64/elem), so use a finer sub-mask. Validated vs a CPU combine oracle.
// ===========================================================================
#define LDS_EW 7u                 // element work words (u64)
#ifndef LDS_ECAP
#define LDS_ECAP 576u             // group cap for full-element staging (LDS-bound)
#endif

// Scatter full elements (work[g*7..]) + index into bucket-contiguous storage.
__kernel void p2_scatter_elem(uint N, uint bucket_bits, uint bucket_cap,
                              __global const ulong* work,   // [N*7]
                              __global uint* counts,         // [numBuckets], pre-zeroed
                              __global ulong* bwork,          // [numBuckets*bucket_cap*7]
                              __global uint* bidx,            // [numBuckets*bucket_cap]
                              __global uint* drops) {         // [0]=bucket overflow
    uint g = get_global_id(0);
    if (g >= N) return;
    uint key = (uint)(work[(size_t)g*7] & 0xFFFFFFu);
    uint b = key >> (24u - bucket_bits);
    uint pos = atomic_inc(&counts[b]);
    if (pos < bucket_cap) {
        size_t d = (size_t)b*bucket_cap + pos;
        for (uint w = 0; w < LDS_EW; ++w) bwork[d*LDS_EW + w] = work[(size_t)g*LDS_EW + w];
        bidx[d] = g;
    } else atomic_inc(&drops[0]);
}

// ===========================================================================
// P2c: PIPELINE LDS match -- replaces round_scatter+radix sort+round_match_sorted.
// Reproduces round_match_sorted's EXACT contract (combine at Lout, (lead,slot)
// canonical ordering, leaf-prefix materialization, left/right back-refs) so the
// KAT goldens stay byte-identical, but finds collisions in LDS (bucket + chain)
// with parents combined from LDS -- no global parent-work gather. Needs bh3.cl
// (bh3_combine) + round.cl (BH3_MAX_LEAVES); compile {bh3, round, sort, lds}.
// ===========================================================================
#define LDS_PW 7u   // pipeline element work words (u64)

// round_scatter_lds: bucket work[r]'s N mixed elements by the top bucket_bits of
// their key into bucket-contiguous storage: full work (7 u64) + original slot +
// lead (leaves_in[slot*9], or the slot itself at r==1). This fat write is the
// price of LDS-local combine (no per-collision gather); measured ~16 ms/round.
__kernel void round_scatter_lds(uint N, uint bucket_bits, uint bucket_cap, uint lead_identity,
                                __global const ulong* work, __global const uint* leaves_in,
                                __global uint* counts, __global ulong* bwork,
                                __global uint* bslot, __global uint* blead,
                                __global uint* drops) {
    uint g = get_global_id(0);
    if (g >= N) return;
    uint key = (uint)(work[(size_t)g*LDS_PW] & 0xFFFFFFu);
    uint b = key >> (24u - bucket_bits);
    uint pos = atomic_inc(&counts[b]);
    if (pos < bucket_cap) {
        size_t d = (size_t)b*bucket_cap + pos;
        for (uint w = 0; w < LDS_PW; ++w) bwork[d*LDS_PW + w] = work[(size_t)g*LDS_PW + w];
        bslot[d] = g;
        blead[d] = lead_identity ? g : leaves_in[(size_t)g*BH3_MAX_LEAVES];
    } else atomic_inc(&drops[1]);   // counters[1] = bucket_drops (pipeline convention)
}

// round_collide_lds: one workgroup per (bucket, sub-mask). Stage the group into
// LDS (work + slot + lead), chain by middle key bits, and for each equal-full-key
// collision emit the SAME child round_match_sorted would: combine the (lead,slot)-
// ordered pair at Lout, child -> out_work[oi], back-refs -> all_left/all_right
// [out_off+oi], leaf prefix -> leaves_out[oi] (concat of parents' prefixes read
// from leaves_in). oi is a global atomic (emission order free; goldens are sets).
__kernel void round_collide_lds(uint bucket_bits, uint submask_bits, uint bucket_cap, uint Lout,
                                uint out_off, uint out_capacity,
                                __global const uint* counts,
                                __global const ulong* bwork, __global const uint* bslot, __global const uint* blead,
                                __global ulong* out_work, __global uint* all_left, __global uint* all_right,
                                __global const uint* leaves_in, __global uint* leaves_out,
                                uint s_in, uint s_out,
                                __global uint* counters) {   // [0]=out_count,[2]=pair_drops
    __local ulong lwork[LDS_PW * LDS_ECAP];
    __local uint lslot[LDS_ECAP];
    __local uint llead[LDS_ECAP];
    __local uint lkey[LDS_ECAP];
    __local uint lchain[LDS_ECAP];
    __local uint tab[LDS_TABSIZE];
    __local uint gcount;

    uint lId = get_local_id(0);
    uint submaskCount = 1u << submask_bits;
    uint bucket = get_group_id(0) / submaskCount;
    uint mask   = get_group_id(0) % submaskCount;

    if (lId == 0) gcount = 0;
    for (uint i = lId; i < LDS_TABSIZE; i += LDS_WG) tab[i] = LDS_EMPTY;
    barrier(CLK_LOCAL_MEM_FENCE);

    uint cnt = counts[bucket];
    if (cnt > bucket_cap) cnt = bucket_cap;
    size_t base = (size_t)bucket * bucket_cap;
    for (uint p = lId; p < cnt; p += LDS_WG) {
        size_t d = base + p;
        uint key = (uint)(bwork[d*LDS_PW] & 0xFFFFFFu);
        if ((key & (submaskCount - 1u)) != mask) continue;
        uint pos = atomic_inc(&gcount);
        if (pos < LDS_ECAP) {
            for (uint w = 0; w < LDS_PW; ++w) lwork[pos*LDS_PW + w] = bwork[d*LDS_PW + w];
            lslot[pos] = bslot[d]; llead[pos] = blead[d]; lkey[pos] = key;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint total = gcount < LDS_ECAP ? gcount : LDS_ECAP;

    for (uint pos = lId; pos < total; pos += LDS_WG) {
        uint hk = (lkey[pos] >> submask_bits) & (LDS_TABSIZE - 1u);
        lchain[pos] = atomic_xchg(&tab[hk], pos);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint pos = lId; pos < total; pos += LDS_WG) {
        uint key = lkey[pos];
        uint oth = lchain[pos], walk = 0;
        while (oth != LDS_EMPTY) {
            if (++walk > 64u) break;
            if (lkey[oth] == key) {
                // (lead,slot) canonical order -- identical rule to round_match_sorted.
                uint la = llead[pos], lb = llead[oth], sa = lslot[pos], sb = lslot[oth];
                uint leftPos = pos, rightPos = oth, left = sa, right = sb;
                if (lb < la || (lb == la && sb < sa)) { leftPos = oth; rightPos = pos; left = sb; right = sa; }
                ulong a[7], b[7], c[7];
                for (uint w = 0; w < LDS_PW; ++w) { a[w] = lwork[leftPos*LDS_PW+w]; b[w] = lwork[rightPos*LDS_PW+w]; }
                bh3_combine(a, b, Lout, c);
                uint oi = atomic_inc(&counters[0]);
                if (oi < out_capacity) {
                    for (uint w = 0; w < LDS_PW; ++w) out_work[(size_t)oi*LDS_PW + w] = c[w];
                    all_left[out_off + oi] = left; all_right[out_off + oi] = right;
                    for (uint i = 0; i < s_out; ++i) {
                        uint leaf = (i < s_in) ? leaves_in[(size_t)left*BH3_MAX_LEAVES + i]
                                               : leaves_in[(size_t)right*BH3_MAX_LEAVES + (i - s_in)];
                        leaves_out[(size_t)oi*BH3_MAX_LEAVES + i] = leaf;
                    }
                } else atomic_inc(&counters[2]);
            }
            oth = lchain[oth];
        }
    }
}

// ===========================================================================
// P2 STEP B: FUSED row-bucket round kernel. ONE kernel does round r's match +
// round (r+1)'s mix + round (r+1)'s scatter, so the coalesced-gather win (crux
// 30bc4fe) is NOT eaten by a separate fat scatter pass (the un-fused LDS path's
// downfall @ 244 ms). Per workgroup (one (bucket, sub-mask) group):
//   1. coalesced-load round-r FAT bucket into LDS (work[7] + gi + lead + leaves[sIn])
//   2. chain by middle key bits (atomic_xchg), same as round_collide_lds
//   3. per equal-full-key collision: order by (lead, gi); bh3_combine at Lout(r);
//      materialize the child's leaf prefix = concat(parents' leaves); fold
//      bh3_apply_mix(Lmix_next, padnum_next) to set the child's NEXT-round key;
//      emit the child FAT into round-(r+1) buckets (bucketed by that key), record
//      all_left/all_right[child_gi] = parent gi's.
// FAT bucket per STEP A (9ba5e80): leaves travel inline (coalesced leaf read).
// gi is a stable dense per-level id: array index for this level's back-ref row
// AND the (lead,gi) tiebreak. Validated vs ref::cpu_round in test_lds_collide.
// Full-width (7 u64) first for correctness; per-round compaction layers on later.
// ===========================================================================
#ifndef LDS_FCAP
#define LDS_FCAP 384u        // fat-element group cap (work+leaves in LDS -> smaller than LDS_ECAP)
#endif
#define LDS_FLEAF 8u         // max parent leaf prefix staged (sleaves_for(<=4) = 8)

// ===========================================================================
// DECOUPLED-SCATTER OCCUPANCY PROBE. Question: does the SAME uncoalesced fat
// scatter run faster at high occupancy (standalone kernel, tiny LDS) than at the
// fused kernel's low occupancy (~41 KB LDS => 1 wg/SM)? If yes, splitting the
// scatter out of the fused find into its own high-occupancy kernel is the lever.
// DUMMYW __local uints are filled + read (runtime index) so the compiler must
// allocate them, pinning occupancy the way the fused kernel's real LDS does.
// ===========================================================================
#define SCATTER_PROBE(NAME, DUMMYW)                                                  \
__kernel void NAME(uint N, uint bucket_bits, uint bucket_cap, uint ew,               \
                   __global const ulong* src, __global const uint* keys,             \
                   __global uint* counts, __global ulong* buckets, __global uint* drops) { \
    __local uint dummy[(DUMMYW)];                                                    \
    uint li = get_local_id(0);                                                       \
    for (uint i = li; i < (DUMMYW); i += 256u) dummy[i] = i ^ get_group_id(0);       \
    barrier(CLK_LOCAL_MEM_FENCE);                                                    \
    uint g = get_global_id(0); if (g >= N) return;                                   \
    ulong contam = (ulong)dummy[g % (DUMMYW)];                                       \
    uint key = keys[g];                                                              \
    uint b = key >> (24u - bucket_bits);                                             \
    uint pos = atomic_inc(&counts[b]);                                               \
    if (pos < bucket_cap) { size_t d = (size_t)b*bucket_cap + pos;                   \
        buckets[d*ew] = src[(size_t)g*ew] ^ contam;                                  \
        for (uint w = 1; w < ew; ++w) buckets[d*ew + w] = src[(size_t)g*ew + w]; }   \
    else atomic_inc(&drops[0]);                                                      \
}
SCATTER_PROBE(scatter_probe_hi, 4u)        // ~tiny LDS -> max occupancy
SCATTER_PROBE(scatter_probe_mid, 6144u)    // 24 KB -> ~2 wg/SM
SCATTER_PROBE(scatter_probe_lo, 10240u)    // 40 KB -> ~1 wg/SM (fused-kernel occupancy)

// RUN-LENGTH SWEEP: the coalescing benefit curve for a key-random fat scatter.
// Any local-reorder / two-level / magazine scheme's ONLY payoff is making writes to
// the same destination CONTIGUOUS. This measures that payoff directly, without
// building the machinery: elements are written in runs of `runLen` consecutive slots
// within one bucket, while consecutive runs land in different (far-apart) buckets.
//   runLen=1  == today's per-lane scatter (each lane a different bucket)
//   runLen=32 == a fully coalesced burst per warp
// Deterministic slot mapping (no atomics, no collisions): run r -> bucket r%nb,
// wave r/nb -> slot base wave*runLen. Read side is sequential so this isolates writes.
__kernel void scatter_runs(uint N, uint ew, uint runLen, uint num_buckets, uint bucket_cap,
                           __global const ulong* src, __global ulong* dst) {
    uint g = get_global_id(0); if (g >= N) return;
    uint runId = g / runLen;
    uint rank  = g - runId * runLen;
    uint b     = runId % num_buckets;
    uint wave  = runId / num_buckets;
    size_t d = (size_t)b * bucket_cap + (size_t)wave * runLen + rank;
    for (uint w = 0; w < ew; ++w) dst[d*ew + w] = src[(size_t)g*ew + w];
}

// ATOMIC-COST ISOLATION: same TRUE-RANDOM bucket destination as the real emit, but
// the slot comes from a deterministic formula instead of atomic_inc on a per-bucket
// counter. (Slots may collide -- irrelevant, this measures write bandwidth only.)
// Compared against scatter_probe_hi (same randomness, WITH the atomic) this isolates
// exactly what the 16384 contended per-bucket counters cost us.
__kernel void scatter_noatomic(uint N, uint bucket_bits, uint bucket_cap, uint ew,
                               __global const ulong* src, __global const uint* keys,
                               __global ulong* buckets) {
    uint g = get_global_id(0); if (g >= N) return;
    uint b = keys[g] >> (24u - bucket_bits);
    size_t d = (size_t)b*bucket_cap + (g % bucket_cap);
    for (uint w = 0; w < ew; ++w) buckets[d*ew + w] = src[(size_t)g*ew + w];
}

// SPLIT-vs-PACKED EMIT. The fused emit writes each child to FOUR separate arrays at
// the same random index (work, gi, lead, leaves). Global writes are serviced in 32 B
// sectors, so each 4 B field costs a whole sector: ~5 sectors (160 B of traffic) for
// 72 B of useful data. These two kernels write the SAME useful bytes -- split across
// four arrays vs packed into one contiguous record -- to measure that waste.
__kernel void emit_split(uint N, uint bucket_bits, uint bucket_cap,
                         __global const uint* keys, __global uint* counts,
                         __global ulong* bwork, __global uint* bgi,
                         __global uint* blead, __global uint* bleaves,
                         __global uint* drops) {
    uint g = get_global_id(0); if (g >= N) return;
    uint key = keys[g];
    uint b = key >> (24u - bucket_bits);
    uint pos = atomic_inc(&counts[b]);
    if (pos >= bucket_cap) { atomic_inc(&drops[0]); return; }
    size_t d = (size_t)b*bucket_cap + pos;
    for (uint w = 0; w < 7u; ++w) bwork[d*7 + w] = (ulong)(g + w);   // 56 B
    bgi[d]   = g;                                                     // 4 B
    blead[d] = key;                                                   // 4 B
    bleaves[d*2 + 0] = g; bleaves[d*2 + 1] = key;                     // 8 B
}
// Same 72 B, one contiguous 9-ulong record: work[0..6], (gi|lead), leaves.
__kernel void emit_packed(uint N, uint bucket_bits, uint bucket_cap,
                          __global const uint* keys, __global uint* counts,
                          __global ulong* belem, __global uint* drops) {
    uint g = get_global_id(0); if (g >= N) return;
    uint key = keys[g];
    uint b = key >> (24u - bucket_bits);
    uint pos = atomic_inc(&counts[b]);
    if (pos >= bucket_cap) { atomic_inc(&drops[0]); return; }
    size_t d = (size_t)b*bucket_cap + pos;
    for (uint w = 0; w < 7u; ++w) belem[d*9 + w] = (ulong)(g + w);   // 56 B
    belem[d*9 + 7] = ((ulong)g << 32) | (ulong)key;                   // gi|lead
    belem[d*9 + 8] = ((ulong)g << 32) | (ulong)key;                   // leaves
}
// Packed AND padded to 96 B (12 ulong = exactly 3 x 32 B sectors) so each record is
// sector-aligned: trades 24 B of padding for zero straddle.
__kernel void emit_packed_pad(uint N, uint bucket_bits, uint bucket_cap,
                              __global const uint* keys, __global uint* counts,
                              __global ulong* belem, __global uint* drops) {
    uint g = get_global_id(0); if (g >= N) return;
    uint key = keys[g];
    uint b = key >> (24u - bucket_bits);
    uint pos = atomic_inc(&counts[b]);
    if (pos >= bucket_cap) { atomic_inc(&drops[0]); return; }
    size_t d = (size_t)b*bucket_cap + pos;
    for (uint w = 0; w < 7u; ++w) belem[d*12 + w] = (ulong)(g + w);
    belem[d*12 + 7] = ((ulong)g << 32) | (ulong)key;
    belem[d*12 + 8] = ((ulong)g << 32) | (ulong)key;
}

// SEED RE-DERIVATION PROBE. Round-1 elements are SEEDS: fully determined by a 25-bit
// index, so they need not be stored at all -- the 56 B work state can be recomputed
// from the index (7 x siphash24 + apply_mix). This measures whether that recompute is
// cheaper than the memory it replaces (storing + re-reading a 72 B record).
//   seed_compute_only : the recompute cost alone (result folded to defeat DCE)
//   seed_scatter_thin : entry pass storing only an 8 B (index,key) record
__kernel void seed_compute_only(__global const ulong* pp4, uint begin, uint count,
                                __global ulong* sink) {
    uint g = get_global_id(0); if (g >= count) return;
    uint idx = begin + g;
    ulong pp[4] = { pp4[0], pp4[1], pp4[2], pp4[3] };
    ulong e[7];
    bh3_seed_element(pp, idx, e);
    uint tree1[1] = { idx };
    e[0] = bh3_apply_mix(e, tree1, 1u, 448u);
    if ((e[0] & 0xFFFFFFFFFFFFul) == 0ul) sink[0] = e[0];   // never taken; defeats DCE
}
__kernel void seed_scatter_thin(__global const ulong* pp4, uint begin, uint count,
                                uint bucket_bits, uint bucket_cap,
                                __global uint* counts, __global ulong* belem,
                                __global uint* drops) {
    uint g = get_global_id(0); if (g >= count) return;
    uint idx = begin + g;
    ulong pp[4] = { pp4[0], pp4[1], pp4[2], pp4[3] };
    ulong e[7];
    bh3_seed_element(pp, idx, e);
    uint tree1[1] = { idx };
    e[0] = bh3_apply_mix(e, tree1, 1u, 448u);
    uint key = (uint)(e[0] & 0xFFFFFFu);
    uint b = key >> (24u - bucket_bits);
    uint pos = atomic_inc(&counts[b]);
    if (pos < bucket_cap) belem[(size_t)b*bucket_cap + pos] = ((ulong)idx << 32) | (ulong)key;
    else atomic_inc(&drops[0]);
}

// Coalesced copy: peak-ish sequential write reference (each thread writes ew
// contiguous words; threads tile the array -> fully coalesced overall).
__kernel void bw_copy(uint N, uint ew, __global const ulong* src, __global ulong* dst) {
    uint g = get_global_id(0); if (g >= N) return;
    for (uint w = 0; w < ew; ++w) dst[(size_t)g*ew + w] = src[(size_t)g*ew + w];
}

// COMPACTION variants: work carried at per-round significant widths. INW = input
// parent words (inwords_for(r)), OUTW = emitted child words (outwords_for(r)). Both
// are COMPILE-TIME so the stage/combine/emit loops fully unroll (a runtime width cost
// +24 ms on the sort path -- the unrolling killer). LDS lwork stride = INW (also
// shrinks LDS for late rounds). Upper words are 0 in the compacted layout (the prior
// bh3_combine masked to Lout, and apply_mix touches only e[0]), so zero-filling
// a/b[INW..7) before bh3_combine is bit-exact. Widths: r1,r2=(7,7); r3=(7,6); r4=(6,5).
// LEAF MODE (LMODE) -- how the leaf payload travels. Round 4's mix needs padNum(5)=9
// leaves: all 8 of the LEFT parent's, but only the RIGHT parent's FIRST (already carried
// as `lead`). apply_mix decomposes exactly as
//     mix = rotl24( workPart + indexPart ),  indexPart additive over leaves
// (index bits sit at positions >= Lmix, element bits below it -- disjoint, and rotl
// distributes over a disjoint OR). So a round-4 element can carry ONE u64
//     leftContrib = rotl40( apply_mix(0, itsOwn8Leaves, 8, Lmix(5)) )
// instead of its 8 raw leaves, and round 4 reconstructs the exact child key as
//     rotl24( rotl40( apply_mix(child, [0x8, rightLead], 9, 288) ) + leftContrib ).
// Verified bit-exact vs the full mix (500k random trials). Shrinks round 3's emit
// 88 -> 64 B/child and round 4's stage read by the same.
#define LMODE_RAW  0   // stage raw leaves, emit raw leaves            (r1, r2)
#define LMODE_EMIT 1   // stage raw leaves, emit the compact contrib   (r3)
#define LMODE_USE  2   // stage the contrib, no leaf emit              (r4)
#define LMODE_SEED 3   // RE-DERIVE the element from its index          (r1)
#define BH3_LMIX5  288u   // Lmix(5) -- the round the contrib is precomputed against

// PACKED ELEMENT LAYOUT. Each element is ONE contiguous record instead of four
// parallel arrays, because global writes are serviced in 32 B sectors: a 4 B gi and a
// 4 B lead in separate arrays each burned a whole sector, so a 72 B child cost ~5
// sectors of traffic. Measured 1.20x faster packed (test_emit_packing); padding to
// sector alignment adds nothing beyond contiguity, so the record has no wasted bytes.
//   [0 .. W-1]  work words        (W = INW reading / OUTW writing)
//   [W]         meta = (gi << 32) | lead
//   [W+1 ..]    leaf payload, two u32 packed per u64 (or the u64 leftContrib)
// Strides are runtime args; every LOOP BOUND stays compile-time so the word loops
// still fully unroll (a runtime loop bound cost +24 ms here before).
#define FUSED_LDS(NAME, INW, OUTW, LEAFW, LMODE)                                      \
__kernel void NAME(                                                                   \
    uint bucket_bits, uint submask_bits, uint in_bucket_cap, uint out_bucket_cap,     \
    uint Lout, uint Lmix_next, uint padnum_next, uint sIn, uint sOut, uint out_off,   \
    uint sBuild, uint in_stride, uint out_stride,                                     \
    __global const uint* in_counts, __global const ulong* in_belem,                   \
    __global uint* out_counts, __global ulong* out_belem,                             \
    __global uint* all_left, __global uint* all_right,                                \
    __global uint* gi_counter, __global uint* drops,                                  \
    __global const ulong* pp4) {                                                     \
    __local ulong lwork[(INW) * LDS_FCAP];                                            \
    __local uint  lgi[LDS_FCAP]; __local uint llead[LDS_FCAP];                        \
    __local uint  lleaf[(LEAFW) * LDS_FCAP];                                          \
    __local uint  lkey[LDS_FCAP]; __local uint lchain[LDS_FCAP];                      \
    __local uint  tab[LDS_TABSIZE]; __local uint gcount;                              \
    uint lId = get_local_id(0);                                                       \
    uint submaskCount = 1u << submask_bits;                                           \
    uint bucket = get_group_id(0) / submaskCount;                                     \
    uint mask   = get_group_id(0) % submaskCount;                                     \
    if (lId == 0) gcount = 0;                                                         \
    for (uint i = lId; i < LDS_TABSIZE; i += LDS_WG) tab[i] = LDS_EMPTY;              \
    barrier(CLK_LOCAL_MEM_FENCE);                                                     \
    uint cnt = in_counts[bucket];                                                     \
    if (cnt > in_bucket_cap) cnt = in_bucket_cap;                                     \
    size_t base = (size_t)bucket * in_bucket_cap;                                     \
    for (uint p = lId; p < cnt; p += LDS_WG) {                                        \
        size_t d = (base + p) * in_stride;                                            \
        ulong rec0 = in_belem[d];                                                     \
        uint key = (uint)(rec0 & 0xFFFFFFu);                                          \
        if ((key & (submaskCount - 1u)) != mask) continue;                            \
        uint pos = atomic_inc(&gcount);                                               \
        if (pos < LDS_FCAP) {                                                         \
            if ((LMODE) == LMODE_SEED) {                                              \
                /* Round-1 elements are SEEDS: the record is just (index,key) and the \
                   56 B work state is RE-DERIVED. Recompute measured 2.5 ms vs 18.2 ms \
                   to store and re-read the full record (test_seed_rederivation).     \
                   Only the INDEX is captured here -- the expensive derivation happens \
                   in the DEFERRED EXPAND below, where every lane is active. This      \
                   staging loop runs 8 sub-mask passes at ~4/32 lanes; the expand loop \
                   runs once at 32/32. See "non-divergent expand" in docs/performance.md. */ \
                lgi[pos] = (uint)(rec0 >> 32); lkey[pos] = key;                       \
            } else {                                                                  \
                for (uint w = 0; w < (INW); ++w) lwork[pos*(INW) + w] = in_belem[d + w]; \
                ulong meta = in_belem[d + (INW)];                                     \
                lgi[pos] = (uint)(meta >> 32); llead[pos] = (uint)meta; lkey[pos] = key; \
                for (uint i = 0; i < sIn; ++i)                                        \
                    lleaf[pos*(LEAFW) + i] =                                          \
                        (uint)(in_belem[d + (INW) + 1u + (i >> 1)] >> ((i & 1u) * 32u)); \
            }                                                                         \
        } else atomic_inc(&drops[1]);                                                \
    }                                                                                \
    barrier(CLK_LOCAL_MEM_FENCE);                                                     \
    uint total = gcount < LDS_FCAP ? gcount : LDS_FCAP;                               \
    for (uint pos = lId; pos < total; pos += LDS_WG) {                               \
        if ((LMODE) == LMODE_SEED) {                                                  \
            /* DEFERRED EXPAND -- the whole point of splitting the stage. Every lane  \
               here is active (total ~= 256 == LDS_WG, one iteration), whereas the    \
               staging loop above is sub-mask filtered to ~1/8 of its lanes. Same     \
               number of seeds derived, 8x the SIMD utilization. */                   \
            uint idx = lgi[pos];                                                      \
            ulong pp[4] = { pp4[0], pp4[1], pp4[2], pp4[3] };                         \
            ulong se[7]; bh3_seed_element(pp, idx, se);                               \
            uint t1[1] = { idx };                                                     \
            se[0] = bh3_apply_mix(se, t1, 1u, 448u);                                  \
            for (uint w = 0; w < (INW); ++w) lwork[pos*(INW) + w] = se[w];            \
            llead[pos] = idx; lleaf[pos*(LEAFW) + 0] = idx;                           \
        }                                                                             \
        uint hk = (lkey[pos] >> submask_bits) & (LDS_TABSIZE - 1u);                   \
        lchain[pos] = atomic_xchg(&tab[hk], pos);                                     \
    }                                                                                \
    barrier(CLK_LOCAL_MEM_FENCE);                                                     \
    for (uint pos = lId; pos < total; pos += LDS_WG) {                               \
        uint key = lkey[pos];                                                         \
        uint oth = lchain[pos], walk = 0;                                             \
        while (oth != LDS_EMPTY) {                                                    \
            if (++walk > 64u) { atomic_inc(&drops[3]); break; }                      \
            if (lkey[oth] == key) {                                                   \
                uint la = llead[pos], lb = llead[oth], ga = lgi[pos], gb = lgi[oth];  \
                uint leftPos = pos, rightPos = oth;                                   \
                if (lb < la || (lb == la && gb < ga)) { leftPos = oth; rightPos = pos; } \
                ulong a[7], b[7], c[7];                                               \
                for (uint w = 0; w < 7; ++w) { a[w] = 0ul; b[w] = 0ul; }              \
                for (uint w = 0; w < (INW); ++w) { a[w] = lwork[leftPos*(INW)+w]; b[w] = lwork[rightPos*(INW)+w]; } \
                bh3_combine(a, b, Lout, c);                                           \
                uint ctree[9]; ulong contribOut = 0ul;                                \
                if ((LMODE) == LMODE_USE) {                                           \
                    for (uint i = 0; i < 8u; ++i) ctree[i] = 0u;                      \
                    ctree[8] = llead[rightPos];                                        \
                    ulong lc = ((ulong)lleaf[leftPos*(LEAFW)+1] << 32)                 \
                             |  (ulong)lleaf[leftPos*(LEAFW)+0];                       \
                    c[0] = bh3_rotl64(bh3_rotl64(                                      \
                               bh3_apply_mix(c, ctree, padnum_next, Lmix_next), 40)    \
                             + lc, 24);                                                \
                    ctree[0] = llead[leftPos];                                         \
                } else {                                                              \
                    for (uint i = 0; i < sBuild; ++i)                                 \
                        ctree[i] = (i < sIn) ? lleaf[leftPos*(LEAFW) + i]             \
                                             : lleaf[rightPos*(LEAFW) + (i - sIn)];    \
                    c[0] = bh3_apply_mix(c, ctree, padnum_next, Lmix_next);           \
                    if ((LMODE) == LMODE_EMIT) {                                      \
                        ulong zw[7]; for (uint w = 0; w < 7u; ++w) zw[w] = 0ul;        \
                        contribOut = bh3_rotl64(                                       \
                            bh3_apply_mix(zw, ctree, 8u, BH3_LMIX5), 40);              \
                    }                                                                 \
                }                                                                     \
                uint ckey = (uint)(c[0] & 0xFFFFFFu);                                 \
                uint cb = ckey >> (24u - bucket_bits);                                \
                uint cpos = atomic_inc(&out_counts[cb]);                              \
                if (cpos < out_bucket_cap) {                                          \
                    uint cgi = atomic_inc(gi_counter);                                \
                    size_t od = ((size_t)cb * out_bucket_cap + cpos) * out_stride;    \
                    for (uint w = 0; w < (OUTW); ++w) out_belem[od + w] = c[w];       \
                    out_belem[od + (OUTW)] = ((ulong)cgi << 32) | (ulong)ctree[0];    \
                    if ((LMODE) == LMODE_EMIT) {                                      \
                        out_belem[od + (OUTW) + 1u] = contribOut;                     \
                    } else {                                                          \
                        for (uint j = 0; j*2u < sOut; ++j) {                          \
                            uint lo = ctree[j*2u];                                     \
                            uint hi = (j*2u + 1u < sOut) ? ctree[j*2u + 1u] : 0u;      \
                            out_belem[od + (OUTW) + 1u + j] = ((ulong)hi << 32) | (ulong)lo; \
                        }                                                             \
                    }                                                                 \
                    all_left[out_off + cgi] = lgi[leftPos]; all_right[out_off + cgi] = lgi[rightPos]; \
                } else atomic_inc(&drops[2]);                                         \
            }                                                                        \
            oth = lchain[oth];                                                        \
        }                                                                            \
    }                                                                                \
}
// (INW, OUTW, LEAFW = LDS leaf-payload uints/elem, LMODE). LEAFW is per-round rather
// than a fixed 8, which also trims LDS: r1/r2 stage <=2 leaves, r3 stages 4, r4 the
// 2-uint contrib.
FUSED_LDS(round_fused_seed, 7, 7, 2, LMODE_SEED)  // r1: re-derives seeds from indices
FUSED_LDS(round_fused_lds, 7, 7, 2, LMODE_RAW)    // r2 (sIn=2)
FUSED_LDS(round_fused_7_6, 7, 6, 4, LMODE_EMIT)   // r3: stages 4 leaves, emits contrib
FUSED_LDS(round_fused_6_5, 6, 5, 2, LMODE_USE)    // r4: stages contrib, no leaf emit

// ENTRY (round 1) for the fused row-bucket path: seed_element + apply_mix(Lmix=448,
// single-leaf tree {idx}) -- exactly round1_mix_seeds -- then scatter the mixed
// element DIRECTLY into round-1 FAT buckets (no flat work[1]/leaves[1]). gi = seed
// index (recover reads it as a leaf at row 0); lead = leaves[0] = idx.
__kernel void round1_mix_scatter_fat(__global const ulong* pp4, uint begin, uint count,
                                     uint bucket_bits, uint bucket_cap, uint stride,
                                     __global uint* counts, __global ulong* belem,
                                     __global uint* drops) {
    uint g = (uint)get_global_id(0);
    if (g >= count) return;
    uint idx = begin + g;
    ulong pp[4] = { pp4[0], pp4[1], pp4[2], pp4[3] };
    ulong e[7];
    bh3_seed_element(pp, idx, e);
    uint tree1[1] = { idx };
    e[0] = bh3_apply_mix(e, tree1, 1u, 448u);
    uint key = (uint)(e[0] & 0xFFFFFFu);
    uint b = key >> (24u - bucket_bits);
    uint pos = atomic_inc(&counts[b]);
    if (pos < bucket_cap) {
        // THIN round-1 record: (index << 32) | key, 8 B total. The 56 B work state is
        // NOT stored -- round 1 re-derives it from the index while staging into LDS
        // (LMODE_SEED). gi, lead and leaf0 are all the index for a seed element, so
        // nothing else needs carrying. Measured 2.5 ms vs 18.2 ms for the 72 B record.
        size_t d = ((size_t)b*bucket_cap + pos) * stride;
        belem[d] = ((ulong)idx << 32) | (ulong)key;
    } else atomic_inc(&drops[1]);
}

// TERMINAL (round 5) for the fused row-bucket path: like round_fused_lds but NO
// child mix/scatter -- combine colliding round-5 pairs at Lout(5)=24 and detect the
// all-zero survivor (two byte-identical round-5 elements XOR to 0). No leaves needed
// (recover walks back-refs). Survivor slot = a dense atomic index; its back-ref row
// (out_off = 4*capacity) holds the two round-5 parent gi's so recover can descend.
__kernel void round5_fused_lds(
    uint bucket_bits, uint submask_bits, uint in_bucket_cap, uint Lout, uint out_off,
    uint in_stride,
    __global const uint*  in_counts,
    __global const ulong* in_belem,    // packed: work[0..4], meta=(gi<<32)|lead
    __global uint*  all_left, __global uint* all_right,   // indexed by survivor slot (row out_off)
    __global uint*  surv_slots, __global uint* surv_count, uint surv_cap,
    __global uint*  drops) {            // [1]=group ovf, [3]=chain cap, [2]=surv ovf
    // Round-5 input = round-4 output = inwords_for(5) = 5 significant words.
    #define LDS_R5W 5u
    __local ulong lwork[LDS_R5W * LDS_FCAP];
    __local uint  lgi[LDS_FCAP];
    __local uint  llead[LDS_FCAP];
    __local uint  lkey[LDS_FCAP];
    __local uint  lchain[LDS_FCAP];
    __local uint  tab[LDS_TABSIZE];
    __local uint  gcount;

    uint lId = get_local_id(0);
    uint submaskCount = 1u << submask_bits;
    uint bucket = get_group_id(0) / submaskCount;
    uint mask   = get_group_id(0) % submaskCount;

    if (lId == 0) gcount = 0;
    for (uint i = lId; i < LDS_TABSIZE; i += LDS_WG) tab[i] = LDS_EMPTY;
    barrier(CLK_LOCAL_MEM_FENCE);

    uint cnt = in_counts[bucket];
    if (cnt > in_bucket_cap) cnt = in_bucket_cap;
    size_t base = (size_t)bucket * in_bucket_cap;
    for (uint p = lId; p < cnt; p += LDS_WG) {
        size_t d = (base + p) * in_stride;
        uint key = (uint)(in_belem[d] & 0xFFFFFFu);
        if ((key & (submaskCount - 1u)) != mask) continue;
        uint pos = atomic_inc(&gcount);
        if (pos < LDS_FCAP) {
            for (uint w = 0; w < LDS_R5W; ++w) lwork[pos*LDS_R5W + w] = in_belem[d + w];
            ulong meta = in_belem[d + LDS_R5W];
            lgi[pos] = (uint)(meta >> 32); llead[pos] = (uint)meta; lkey[pos] = key;
        } else atomic_inc(&drops[1]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint total = gcount < LDS_FCAP ? gcount : LDS_FCAP;

    for (uint pos = lId; pos < total; pos += LDS_WG) {
        uint hk = (lkey[pos] >> submask_bits) & (LDS_TABSIZE - 1u);
        lchain[pos] = atomic_xchg(&tab[hk], pos);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint pos = lId; pos < total; pos += LDS_WG) {
        uint key = lkey[pos];
        uint oth = lchain[pos], walk = 0;
        while (oth != LDS_EMPTY) {
            if (++walk > 64u) { atomic_inc(&drops[3]); break; }
            if (lkey[oth] == key) {
                uint la = llead[pos], lb = llead[oth], ga = lgi[pos], gb = lgi[oth];
                uint leftPos = pos, rightPos = oth;
                if (lb < la || (lb == la && gb < ga)) { leftPos = oth; rightPos = pos; }
                ulong a[7], b[7], c[7];
                for (uint w = 0; w < 7; ++w) { a[w] = 0ul; b[w] = 0ul; }
                for (uint w = 0; w < LDS_R5W; ++w) { a[w] = lwork[leftPos*LDS_R5W+w]; b[w] = lwork[rightPos*LDS_R5W+w]; }
                bh3_combine(a, b, Lout, c);
                ulong z = 0ul; for (uint w = 0; w < 7; ++w) z |= c[w];
                if (z == 0ul) {   // survivor: byte-identical round-5 elements
                    uint si = atomic_inc(surv_count);
                    if (si < surv_cap) {
                        all_left[out_off + si] = lgi[leftPos]; all_right[out_off + si] = lgi[rightPos];
                        surv_slots[si] = si;
                    } else atomic_inc(&drops[2]);
                }
            }
            oth = lchain[oth];
        }
    }
}

// ===========================================================================
// ROW-BUCKET REWRITE, crux measurement: parent-gather LOCALITY. The sort path's
// match cost is dominated by reading 2 parents (7 u64) by ARBITRARY slot -- a
// scattered gather. A bucket-local layout would instead read parents from
// bucket-contiguous storage (coalesced). gather_combine_bench does M combines,
// each reading 2 parents by index + bh3_combine; feeding it a SCATTERED vs a
// COALESCED index list quantifies the ceiling of the whole rewrite. `ew` = element
// width so we can also see the win at the compacted widths [7,6,5].
// ===========================================================================
__kernel void gather_combine_bench(uint M, uint Lout, uint ew,
                                   __global const ulong* parents,   // [N*ew]
                                   __global const uint* idxL, __global const uint* idxR,
                                   __global ulong* out) {           // [M*ew]
    uint g = get_global_id(0);
    if (g >= M) return;
    uint l = idxL[g], r = idxR[g];
    ulong a[7], b[7], c[7];
    for (int w = 0; w < 7; ++w) { a[w] = 0ul; b[w] = 0ul; }
    for (uint w = 0; w < ew; ++w) { a[w] = parents[(size_t)l*ew + w]; b[w] = parents[(size_t)r*ew + w]; }
    bh3_combine(a, b, Lout, c);
    for (uint w = 0; w < ew; ++w) out[(size_t)g*ew + w] = c[w];
}

// ===========================================================================
// ROW-BUCKET STEP A: fat-vs-thin bucket layout. The fused round kernel must
// materialize each child's leaf prefix (concat of its two parents' prefixes) to
// feed apply_mix and the NEXT round's combine. Two layouts:
//   FAT  = leaves travel INSIDE the bucket element -> parents' leaves arrive in
//          the same coalesced bucket load (cheap read) but the emit write is
//          wider by ceil(sleaves/2) u64.
//   THIN = leaves live in a gi-indexed global array read by arbitrary parent
//          index (the scattered 48 ms/solve we measured in match today) but the
//          emit stays narrow.
// gather_leaves_bench races the READ side (coalesced vs scattered leaf gather at
// the real per-round leaf widths); bucket_emit_bench measures the WRITE side
// (emit cost vs element width, so fat_ew - thin_ew is the fat penalty). Netting
// the two per round decides the bucket layout for STEP B (round_fused_lds).
// ===========================================================================

// READ side: build a child leaf prefix = concat(leftParent[sIn], rightParent[sIn])
// capped at sOut, reading parents' leaves by index. Feed a SCATTERED index list
// (thin, gi-indexed) vs a COALESCED one (fat, bucket-local) to race the two.
__kernel void gather_leaves_bench(uint M, uint sIn, uint sOut,
                                  __global const uint* leaves,   // [N*sIn]
                                  __global const uint* idxL, __global const uint* idxR,
                                  __global uint* out) {          // [M*sOut]
    uint g = get_global_id(0);
    if (g >= M) return;
    uint l = idxL[g], r = idxR[g];
    uint tree[9];
    for (uint i = 0; i < sIn; ++i)                 tree[i]      = leaves[(size_t)l*sIn + i];
    for (uint i = 0; i < sIn && sIn + i < sOut; ++i) tree[sIn + i] = leaves[(size_t)r*sIn + i];
    for (uint i = 0; i < sOut; ++i) out[(size_t)g*sOut + i] = tree[i];
}

// WRITE side: pure bucketed-emit cost vs element width `ew` (u64), payload
// synthesized on the fly so ew is unbounded by any source buffer (fat elements
// exceed 7 u64). Same atomic-bucket scatter shape as the real emit.
__kernel void bucket_emit_bench(uint N, uint bucket_bits, uint bucket_cap, uint ew,
                                __global uint* counts, __global ulong* bwork, __global uint* drops) {
    uint g = get_global_id(0);
    if (g >= N) return;
    uint key = (uint)(((ulong)g * 2654435761ul) & 0xFFFFFFul);   // cheap uniform key
    uint b = key >> (24u - bucket_bits);
    uint pos = atomic_inc(&counts[b]);
    if (pos >= bucket_cap) { atomic_inc(&drops[0]); return; }
    size_t d = (size_t)b*bucket_cap + pos;
    for (uint w = 0; w < ew; ++w) bwork[d*ew + w] = (ulong)(g + w);
}

// Measurement: scatter with a variable element width `ew` (u64 words, stride ew)
// to characterize the emit-to-bucket cost vs element size -- i.e. how much
// compaction to the significant-word schedule [7,7,6,5,1] actually saves.
__kernel void p2_scatter_w(uint N, uint bucket_bits, uint bucket_cap, uint ew,
                           __global const ulong* work,   // [N*7], only ew words used
                           __global uint* counts, __global ulong* bwork, __global uint* drops) {
    uint g = get_global_id(0);
    if (g >= N) return;
    uint key = (uint)(work[(size_t)g*7] & 0xFFFFFFu);
    uint b = key >> (24u - bucket_bits);
    uint pos = atomic_inc(&counts[b]);
    if (pos < bucket_cap) { size_t d=(size_t)b*bucket_cap+pos; for (uint w=0;w<ew;++w) bwork[d*ew+w]=work[(size_t)g*7+w]; }
    else atomic_inc(&drops[0]);
}

// ===========================================================================
// COMPACTION ARCHITECTURE DECISION (measurement only, not wired).
// The AoS runtime-width scatter (p2_scatter_w, `for w<ew`) measured a compaction
// LOSS because the runtime loop bound defeats unrolling. Two rival fixes:
//   (H1) AoS with COMPILE-TIME width  -> fully unrolled, no SoA needed.
//   (H2) SoA word-planes              -> one buffer per word; skip zero planes.
// SoA also risks LOSING scatter: one AoS element write touches ~1 cache line
// (56 B contiguous); the same element in SoA fans across W plane buffers = up to
// W scattered lines. test_compaction_arch() races H1 vs H2 vs the AoS-7 baseline.
// ===========================================================================

// H1: AoS scatter with a COMPILE-TIME word count W (loop unrolls; bucket stride
// = W, so late-round elements occupy exactly W u64). Source stays 7-wide AoS.
#define AOS_SCATTER_W(NAME, W)                                                  \
__kernel void NAME(uint N, uint bucket_bits, uint bucket_cap,                   \
                   __global const ulong* work, __global uint* counts,          \
                   __global ulong* bwork, __global uint* drops) {              \
    uint g = get_global_id(0); if (g >= N) return;                             \
    uint key = (uint)(work[(size_t)g*7] & 0xFFFFFFu);                          \
    uint b = key >> (24u - bucket_bits);                                       \
    uint pos = atomic_inc(&counts[b]);                                         \
    if (pos >= bucket_cap) { atomic_inc(&drops[0]); return; }                  \
    size_t d = (size_t)b*bucket_cap + pos;                                     \
    for (uint w = 0; w < W; ++w) bwork[d*W + w] = work[(size_t)g*7 + w];       \
}
AOS_SCATTER_W(aos_scatter1, 1)
AOS_SCATTER_W(aos_scatter5, 5)
AOS_SCATTER_W(aos_scatter6, 6)
AOS_SCATTER_W(aos_scatter7, 7)

// H2: SoA scatter -- element g is (p0[g]..p6[g]) across 7 source planes; write
// the W significant planes to W bucket planes at the same slot d. W is a
// compile-time constant so `if (W>k)` dead-code-eliminates the unused planes.
#define SOA_SCATTER_W(NAME, W)                                                  \
__kernel void NAME(uint N, uint bucket_bits, uint bucket_cap,                   \
    __global const ulong* p0, __global const ulong* p1, __global const ulong* p2, \
    __global const ulong* p3, __global const ulong* p4, __global const ulong* p5, \
    __global const ulong* p6, __global uint* counts,                           \
    __global ulong* b0, __global ulong* b1, __global ulong* b2,                \
    __global ulong* b3, __global ulong* b4, __global ulong* b5,                \
    __global ulong* b6, __global uint* drops) {                                \
    uint g = get_global_id(0); if (g >= N) return;                             \
    uint key = (uint)(p0[g] & 0xFFFFFFu);                                      \
    uint b = key >> (24u - bucket_bits);                                       \
    uint pos = atomic_inc(&counts[b]);                                         \
    if (pos >= bucket_cap) { atomic_inc(&drops[0]); return; }                  \
    size_t d = (size_t)b*bucket_cap + pos;                                     \
    b0[d] = p0[g];                                                             \
    if (W > 1) b1[d] = p1[g];  if (W > 2) b2[d] = p2[g];                       \
    if (W > 3) b3[d] = p3[g];  if (W > 4) b4[d] = p4[g];                       \
    if (W > 5) b5[d] = p5[g];  if (W > 6) b6[d] = p6[g];                       \
}
SOA_SCATTER_W(soa_scatter1, 1)
SOA_SCATTER_W(soa_scatter5, 5)
SOA_SCATTER_W(soa_scatter6, 6)
SOA_SCATTER_W(soa_scatter7, 7)

// Collide + combine: one workgroup per (bucket, sub-mask). Stage full elements
// into LDS, chain by middle key bits, and for each equal-full-key pair emit the
// bh3_combine child work (7 u64) + the two parent indices. Parents read from LDS.
__kernel void p2_collide_combine(uint bucket_bits, uint submask_bits, uint bucket_cap, uint Lout,
                                 __global const uint* counts,
                                 __global const ulong* bwork,
                                 __global const uint* bidx,
                                 __global ulong* out_work,    // [out_cap*7] child work
                                 __global uint* out_left, __global uint* out_right,
                                 __global uint* out_count, uint out_cap,
                                 __global uint* drops) {       // [1]=grp ovf,[2]=out ovf,[3]=chain cap
    __local ulong lwork[LDS_EW * LDS_ECAP];
    __local uint lkey[LDS_ECAP];
    __local uint lidx[LDS_ECAP];
    __local uint lchain[LDS_ECAP];
    __local uint tab[LDS_TABSIZE];
    __local uint gcount;

    uint lId = get_local_id(0);
    uint submaskCount = 1u << submask_bits;
    uint bucket = get_group_id(0) / submaskCount;
    uint mask   = get_group_id(0) % submaskCount;

    if (lId == 0) gcount = 0;
    for (uint i = lId; i < LDS_TABSIZE; i += LDS_WG) tab[i] = LDS_EMPTY;
    barrier(CLK_LOCAL_MEM_FENCE);

    uint cnt = counts[bucket];
    if (cnt > bucket_cap) cnt = bucket_cap;
    size_t base = (size_t)bucket * bucket_cap;
    for (uint p = lId; p < cnt; p += LDS_WG) {
        size_t d = base + p;
        uint key = (uint)(bwork[d*LDS_EW] & 0xFFFFFFu);
        if ((key & (submaskCount - 1u)) != mask) continue;
        uint pos = atomic_inc(&gcount);
        if (pos < LDS_ECAP) {
            for (uint w = 0; w < LDS_EW; ++w) lwork[pos*LDS_EW + w] = bwork[d*LDS_EW + w];
            lkey[pos] = key; lidx[pos] = bidx[d];
        } else atomic_inc(&drops[1]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint total = gcount < LDS_ECAP ? gcount : LDS_ECAP;

    for (uint pos = lId; pos < total; pos += LDS_WG) {
        uint hk = (lkey[pos] >> submask_bits) & (LDS_TABSIZE - 1u);
        lchain[pos] = atomic_xchg(&tab[hk], pos);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint pos = lId; pos < total; pos += LDS_WG) {
        uint key = lkey[pos];
        uint oth = lchain[pos], walk = 0;
        while (oth != LDS_EMPTY) {
            if (++walk > 64u) { atomic_inc(&drops[3]); break; }
            if (lkey[oth] == key) {
                ulong a[7], b[7], c[7];
                for (uint w = 0; w < LDS_EW; ++w) { a[w] = lwork[pos*LDS_EW+w]; b[w] = lwork[oth*LDS_EW+w]; }
                bh3_combine(a, b, Lout, c);
                uint oi = atomic_inc(out_count);
                if (oi < out_cap) {
                    for (uint w = 0; w < LDS_EW; ++w) out_work[(size_t)oi*LDS_EW + w] = c[w];
                    out_left[oi] = lidx[pos]; out_right[oi] = lidx[oth];
                } else atomic_inc(&drops[2]);
            }
            oth = lchain[oth];
        }
    }
}
