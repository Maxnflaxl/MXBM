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

__kernel void round_fused_lds(
    uint bucket_bits, uint submask_bits, uint in_bucket_cap, uint out_bucket_cap,
    uint Lout, uint Lmix_next, uint padnum_next, uint sIn, uint sOut, uint out_off,
    __global const uint*  in_counts,
    __global const ulong* in_bwork,    // [nb*in_cap*7]
    __global const uint*  in_bgi,      // [nb*in_cap]      stable per-level id (tiebreak + backref)
    __global const uint*  in_blead,    // [nb*in_cap]      first leaf (pair ordering)
    __global const uint*  in_bleaves,  // [nb*in_cap*sIn]  leaf prefix (fat)
    __global uint*  out_counts,        // [nb], pre-zeroed
    __global ulong* out_bwork,         // [nb*out_cap*7]
    __global uint*  out_bgi,           // [nb*out_cap]
    __global uint*  out_blead,         // [nb*out_cap]
    __global uint*  out_bleaves,       // [nb*out_cap*sOut]
    __global uint*  all_left, __global uint* all_right,  // indexed by child gi
    __global uint*  gi_counter,        // global atomic -> dense child gi
    __global uint*  drops) {           // [1]=group ovf, [2]=out-bucket ovf, [3]=chain cap
    __local ulong lwork[LDS_PW * LDS_FCAP];
    __local uint  lgi[LDS_FCAP];
    __local uint  llead[LDS_FCAP];
    __local uint  lleaf[LDS_FLEAF * LDS_FCAP];
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
        size_t d = base + p;
        uint key = (uint)(in_bwork[d*LDS_PW] & 0xFFFFFFu);
        if ((key & (submaskCount - 1u)) != mask) continue;
        uint pos = atomic_inc(&gcount);
        if (pos < LDS_FCAP) {
            for (uint w = 0; w < LDS_PW; ++w) lwork[pos*LDS_PW + w] = in_bwork[d*LDS_PW + w];
            lgi[pos] = in_bgi[d]; llead[pos] = in_blead[d]; lkey[pos] = key;
            for (uint i = 0; i < sIn; ++i) lleaf[pos*LDS_FLEAF + i] = in_bleaves[d*sIn + i];
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
                // (lead, gi) canonical order -- lead primary, gi tiebreak. Distinct
                // leads (every valid solution) are decided by lead alone; equal-lead
                // pairs are overlapping-tree cases the CPU verify gate drops anyway.
                uint la = llead[pos], lb = llead[oth], ga = lgi[pos], gb = lgi[oth];
                uint leftPos = pos, rightPos = oth;
                if (lb < la || (lb == la && gb < ga)) { leftPos = oth; rightPos = pos; }
                ulong a[7], b[7], c[7];
                for (uint w = 0; w < LDS_PW; ++w) { a[w] = lwork[leftPos*LDS_PW+w]; b[w] = lwork[rightPos*LDS_PW+w]; }
                bh3_combine(a, b, Lout, c);
                // Child leaf prefix = concat(left leaves, right leaves), capped sOut.
                uint ctree[9];
                for (uint i = 0; i < sOut; ++i)
                    ctree[i] = (i < sIn) ? lleaf[leftPos*LDS_FLEAF + i]
                                         : lleaf[rightPos*LDS_FLEAF + (i - sIn)];
                // Fold round-(r+1) mix into the child -> sets its next collision key.
                c[0] = bh3_apply_mix(c, ctree, padnum_next, Lmix_next);
                uint ckey = (uint)(c[0] & 0xFFFFFFu);
                uint cb = ckey >> (24u - bucket_bits);
                uint cpos = atomic_inc(&out_counts[cb]);
                if (cpos < out_bucket_cap) {
                    uint cgi = atomic_inc(gi_counter);
                    size_t od = (size_t)cb * out_bucket_cap + cpos;
                    for (uint w = 0; w < LDS_PW; ++w) out_bwork[od*LDS_PW + w] = c[w];
                    out_bgi[od] = cgi; out_blead[od] = ctree[0];
                    for (uint i = 0; i < sOut; ++i) out_bleaves[od*sOut + i] = ctree[i];
                    all_left[out_off + cgi] = lgi[leftPos]; all_right[out_off + cgi] = lgi[rightPos];
                } else atomic_inc(&drops[2]);
            }
            oth = lchain[oth];
        }
    }
}

// ENTRY (round 1) for the fused row-bucket path: seed_element + apply_mix(Lmix=448,
// single-leaf tree {idx}) -- exactly round1_mix_seeds -- then scatter the mixed
// element DIRECTLY into round-1 FAT buckets (no flat work[1]/leaves[1]). gi = seed
// index (recover reads it as a leaf at row 0); lead = leaves[0] = idx.
__kernel void round1_mix_scatter_fat(__global const ulong* pp4, uint begin, uint count,
                                     uint bucket_bits, uint bucket_cap,
                                     __global uint* counts, __global ulong* bwork,
                                     __global uint* bgi, __global uint* blead,
                                     __global uint* bleaves, __global uint* drops) {
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
        size_t d = (size_t)b*bucket_cap + pos;
        for (int k = 0; k < 7; ++k) bwork[d*LDS_PW + k] = e[k];
        bgi[d] = idx; blead[d] = idx; bleaves[d*1 + 0] = idx;   // sIn=1 for round 1
    } else atomic_inc(&drops[1]);
}

// TERMINAL (round 5) for the fused row-bucket path: like round_fused_lds but NO
// child mix/scatter -- combine colliding round-5 pairs at Lout(5)=24 and detect the
// all-zero survivor (two byte-identical round-5 elements XOR to 0). No leaves needed
// (recover walks back-refs). Survivor slot = a dense atomic index; its back-ref row
// (out_off = 4*capacity) holds the two round-5 parent gi's so recover can descend.
__kernel void round5_fused_lds(
    uint bucket_bits, uint submask_bits, uint in_bucket_cap, uint Lout, uint out_off,
    __global const uint*  in_counts,
    __global const ulong* in_bwork,    // [nb*in_cap*7]
    __global const uint*  in_bgi,
    __global const uint*  in_blead,
    __global uint*  all_left, __global uint* all_right,   // indexed by survivor slot (row out_off)
    __global uint*  surv_slots, __global uint* surv_count, uint surv_cap,
    __global uint*  drops) {            // [1]=group ovf, [3]=chain cap, [2]=surv ovf
    __local ulong lwork[LDS_PW * LDS_FCAP];
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
        size_t d = base + p;
        uint key = (uint)(in_bwork[d*LDS_PW] & 0xFFFFFFu);
        if ((key & (submaskCount - 1u)) != mask) continue;
        uint pos = atomic_inc(&gcount);
        if (pos < LDS_FCAP) {
            for (uint w = 0; w < LDS_PW; ++w) lwork[pos*LDS_PW + w] = in_bwork[d*LDS_PW + w];
            lgi[pos] = in_bgi[d]; llead[pos] = in_blead[d]; lkey[pos] = key;
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
                for (uint w = 0; w < LDS_PW; ++w) { a[w] = lwork[leftPos*LDS_PW+w]; b[w] = lwork[rightPos*LDS_PW+w]; }
                bh3_combine(a, b, Lout, c);
                ulong z = 0ul; for (uint w = 0; w < LDS_PW; ++w) z |= c[w];
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
