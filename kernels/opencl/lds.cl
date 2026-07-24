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
