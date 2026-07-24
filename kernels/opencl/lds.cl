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
