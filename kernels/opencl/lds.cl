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

// 128-bit record access on the even-stride paths (staging loads + emit stores).
// -DLDS_V2=0 restores the scalar loops for A/Bs; the r2->r3 side plane is
// STRUCTURAL (reader and writer agree via the record stride) and does not toggle.
#ifndef LDS_V2
#define LDS_V2 1
#endif
// Geometry baked at build time: the host picks bb/sm/caps BEFORE it compiles the
// program, so the per-child bucket shift and the 64-bit slot multiplies stop
// carrying runtime operands. MEASURED NULL on Ada (20 paired arms, solve-count
// instrument at 0.038 % reproducibility, diff -0.008 %); kept because sm_61 does
// 32-bit multiply as multi-instruction XMAD, where a runtime operand should cost
// real time. GEO_BAKED=0 keeps the runtime args -- the A/B channel, and what the
// off-line-geometry tests drive.
#ifndef GEO_BAKED
#define GEO_BAKED 0
#define GEO_BB 16u
#define GEO_SM 1u
#define GEO_INCAP 1u
#define GEO_OUTCAP 1u
#endif
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
// Group cap for the shipped fused kernels. It used to be 384 because it had to cover
// the group's TAIL; LDS_SPILL removes that obligation by splitting an overflowing
// group on one more key bit, so it can sit near the MEAN (264 on the bb + sm = 17
// line). Smaller cap -> smaller __local footprint -> a second workgroup per SM
// becomes reachable inside the 48 KB OpenCL exposes. Values mirror the CUDA side
// (kFCap 320, MXBM_R1_FCAP 288); sweep via MXBM_CL_OPTS="-DLDS_FCAP=...".
// The generic round_fused_lds keeps its own pinned 384 (see the instantiations).
#ifndef LDS_FCAP
#define LDS_FCAP 320u        // fat-element group cap, r2..r4 (near the group mean; spill covers the tail)
#endif
#ifndef LDS_FCAP_R1
#define LDS_FCAP_R1 288u     // round 1 stages only 64 B/element, so it affords a tighter cap
#endif
// Chain table for the shipped kernels: 24 - bb - sm = 7 varying key bits on the
// bb + sm = 17 line, so 128 entries is a PERFECT hash (see LDS_PERFECT_TAB below).
// 512 never addressed more than 128 of them. round_fused_lds keeps LDS_TABSIZE.
#ifndef LDS_FTAB
#define LDS_FTAB 128u
#endif
// LDS_PERFECT_TAB: drop lkey entirely, and with it the key comparison in the walk.
// Within a group the bucket fixes the key's top bb bits and the sub-mask its bottom
// sm; when the table covers the 24 - bb - sm bits left varying, two elements share a
// chain if and only if they share the full key, so `lkey[oth] == key` is a tautology
// and the array re-derives a value word 0 of lwork already carries. Worth 4 B/element
// of __local AND a shared load + branch out of the walk's innermost loop.
// UNSAFE off the bb + sm = 17 line -- run_pipeline_rowbucket checks and refuses.
#ifndef LDS_PERFECT_TAB
#define LDS_PERFECT_TAB 1
#endif
// LDS_SPILL: when a group overflows FCAP, SPLIT it on one more key bit and redo,
// instead of dropping the overflow. Splitting is exactly safe -- two elements whose
// keys differ in the added bit can never collide, so no pair is lost or doubled --
// and it is self-limiting: the split halves the group and overflow is rare, so the
// redone work lands on a few per cent of groups. gcount counts UNCLAMPED (staging
// stops writing past FCAP but keeps counting), so the true group size is known and
// the split decision is uniform across the workgroup.
#ifndef LDS_SPILL
#define LDS_SPILL 1
#endif
// Terminal round (round5_fused_lds) stages only 24 B/element -- its cap never
// constrained occupancy and must not be dragged down when LDS_FCAP is tuned.
#ifndef LDS_TCAP
#define LDS_TCAP 384u
#endif
#define LDS_FLEAF 8u         // max parent leaf prefix staged (sleaves_for(<=4) = 8)

// OVERFLOW ARENA. Per-bucket capacities are cut from a mean + 8 sigma tail bound to
// mean + 2 sigma, and a bucket that fills appends to a global pool instead of dropping;
// arena_link threads the pool onto per-bucket chains between producer and consumer, and
// the consumer walks its chain alongside its own records. That trades 3.6 % of time
// (measured, (14,3) quad, interleaved) for ~22 % of the record slots -- what lets a card
// take a finer geometry, or run at all. The pool sits BEHIND the bucket records in the
// same buffer, so one slot index addresses both regions and no store has to know which
// it hit.
//
// A -D rather than a per-kernel parameter: the program is compiled after the ladder
// settles (GEO_BAKED does the same), so an arena rung compiles the path in and every
// other rung compiles byte-identical kernels. LDS_ARENA_CAP counts slots per record-set
// HALF -- a split set has a pool behind each half, and the two must sum to the pool the
// footprint arithmetic budgeted (gpu/rowbucket_geom.h, kRbArenaSlots).
#ifndef LDS_ARENA
#define LDS_ARENA 0
#endif
#ifndef LDS_ARENA_CAP
#define LDS_ARENA_CAP 65536u
#endif
// Pool entries staged per bucket. At the dense cap ~95 % of chains are empty, so this
// bounds a list that is almost always length 0; overrunning it loses elements, which
// drops[1] reports rather than hiding.
#define LDS_AMAX 64u

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
// ABL: temporary phase-attribution ablation, off unless built with -DABL=n via
// MXBM_CL_OPTS. Results are INTENTIONALLY WRONG when enabled -- it exists to
// attribute time, not to compute. bit0 = skip the emit payload write (counters
// and back-refs kept so element counts stay representative), bit1 = skip the
// re-derivation rebuild, bit2 = skip combine+mix+emit entirely.
#ifndef ABL
#define ABL 0
#endif
#ifndef LEADTIE_PROBE
#define LEADTIE_PROBE 0
#endif
#ifndef ABL_MODE            // which LMODE to ablate; -1 = all rounds
#define ABL_MODE (-1)
#endif
// True only for the round being ablated. Ablating a round corrupts every round
// AFTER it (garbage keys skew the next bucket distribution), so ablate ONE round
// and read only that round's number.
#define ABL_HIT(bit, lm) (((ABL) & (bit)) && ((ABL_MODE) < 0 || (lm) == (ABL_MODE)))
#define LMODE_RAW  0   // stage raw leaves, emit raw leaves            (unused; r2 fallback)
#define LMODE_EMIT 1   // stage raw leaves, emit the compact contrib   (r3)
#define LMODE_USE  2   // stage the contrib, no leaf emit              (r4)
#define LMODE_SEED 3   // RE-DERIVE from index; emit a 16 B PAIR record (r1)
#define LMODE_RD2  4   // RE-DERIVE from that pair record                (r2)
#define LMODE_RD3  5   // RE-DERIVE from a 24 B QUAD record              (r3)
#define BH3_LMIX5  288u   // Lmix(5) -- the round the contrib is precomputed against

// PAIR RECORD (round 1 -> round 2), 16 B instead of the 72 B packed element. A
// round-2 element is one combine away from two seeds, so its two parent indices
// determine it completely -- and those indices ARE its two leaves. Round 1 emits
// only (key, leftIdx, rightIdx, gi) and round 2 rebuilds the 56 B work state in
// its non-divergent expand loop, where the recompute measured +0.4 ms (see
// docs/performance.md). Seed indices are 25-bit (2^25 seeds); gi is 26-bit.
//   word0 = key[0,24) | leftIdx  << 24
//   word1 = rightIdx  | gi       << 25
#define RD2_IDXMASK 0x1FFFFFFu
inline uint  rd2_left (ulong w0) { return (uint)(w0 >> 24) & RD2_IDXMASK; }
inline uint  rd2_right(ulong w1) { return (uint)(w1)       & RD2_IDXMASK; }
inline uint  rd2_gi   (ulong w1) { return (uint)(w1 >> 25); }
inline ulong rd2_w0(uint key, uint li) { return (ulong)key | ((ulong)li << 24); }
inline ulong rd2_w1(uint ri, uint gi)  { return (ulong)ri  | ((ulong)gi << 25); }

// ROUND-2 -> ROUND-3 record, 72 B (9 u64) rather than 80. The `lead` field the generic
// packed layout puts in `meta` is REDUNDANT here: lead is ctree[0], which is already
// leaf 0 of the payload. Dropping it leaves 4 leaves (25 bits each) + gi (26) = 126
// bits, which fits two u64 with the meta word deleted entirely:
//   p0 = l0 | l1<<25 | (l2 low 14)<<50        p1 = l2>>14 | l3<<11 | gi<<36
inline ulong r3_p0(uint l0, uint l1, uint l2) {
    return (ulong)l0 | ((ulong)l1 << 25) | (((ulong)l2 & 0x3FFFul) << 50);
}
inline ulong r3_p1(uint l2, uint l3, uint gi) {
    return ((ulong)l2 >> 14) | ((ulong)l3 << 11) | ((ulong)gi << 36);
}
inline uint r3_l0(ulong p0)           { return (uint)(p0 & RD2_IDXMASK); }
inline uint r3_l1(ulong p0)           { return (uint)((p0 >> 25) & RD2_IDXMASK); }
inline uint r3_l2(ulong p0, ulong p1) { return (uint)(((p0 >> 50) & 0x3FFFul) | ((p1 & 0x7FFul) << 14)); }
inline uint r3_l3(ulong p1)           { return (uint)((p1 >> 11) & RD2_IDXMASK); }
inline uint r3_gi(ulong p1)           { return (uint)((p1 >> 36) & 0x3FFFFFFul); }

// QUAD RECORD (round 2 -> round 3), 24 B instead of the 80 B packed element. Same
// argument one level up: a round-3 element is a combine of two round-2 elements, so
// four seed indices determine it -- and those four ARE its leaves (sIn = 4).
//   word0 = key[0,24) | i0 << 24     word1 = i1 | i2 << 25     word2 = i3 | gi << 25
inline uint  rd3_i1(ulong w1) { return (uint)(w1)       & RD2_IDXMASK; }
inline uint  rd3_i2(ulong w1) { return (uint)(w1 >> 25) & RD2_IDXMASK; }
inline uint  rd3_i3(ulong w2) { return (uint)(w2)       & RD2_IDXMASK; }
inline uint  rd3_gi(ulong w2) { return (uint)(w2 >> 25); }
inline ulong rd3_w1(uint i1, uint i2) { return (ulong)i1 | ((ulong)i2 << 25); }
inline ulong rd3_w2(uint i3, uint gi) { return (ulong)i3 | ((ulong)gi << 25); }

// Rebuild a ROUND-2 element (the round-1 child of seeds li, ri) from scratch:
// two seeds, each mixed at Lmix(1)=448, combined at Lout(1)=424, then mixed at
// Lmix(2)=424 over the 2-leaf tree. This is the unit of re-derivation for both
// round 2 (one call) and round 3 (two calls).
inline void rd_elem2(const ulong pp[4], uint li, uint ri, ulong out[7]) {
    ulong sa[7], sb[7]; uint t1[1]; uint t2[2];
    bh3_seed_element(pp, li, sa); t1[0] = li; sa[0] = bh3_apply_mix(sa, t1, 1u, 448u);
    bh3_seed_element(pp, ri, sb); t1[0] = ri; sb[0] = bh3_apply_mix(sb, t1, 1u, 448u);
    bh3_combine(sa, sb, 424u, out);
    t2[0] = li; t2[1] = ri; out[0] = bh3_apply_mix(out, t2, 2u, 424u);
}

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
// FCAP / TAB / PERFECT / SPILL are per-instantiation parameters (the 2026-07-26 CUDA
// match wins, backported): the shipped per-round kernels run the perfect table and
// spill at a near-mean cap, while the generic round_fused_lds -- validated by tests
// at arbitrary geometries where the perfect-hash argument does not hold -- keeps the
// legacy behavior (tail-covering 384, keyed walk, drop-on-overflow).
// CO-BLOCKS (COB): round 4 hosts the NEXT nonce's entry pass as separate workgroups
// interleaved through its grid -- every co_stride-th group is an entry group.
// Interleaved, not appended, so every dispatch wave carries the mix. Entry groups
// return before the first barrier, and barriers are per-workgroup, so the round
// groups are unaffected. From CUDA's COBLOCKS (fused_round.cuh): round 4 is 82 % of
// DRAM peak but 29 % SM, and hosting the entry there measured cheaper than running it
// alone. COB=0 adds neither args nor dispatch, so the other kernels are unchanged.
#define COB_ARGS_0
#define COB_ARGS_1 , __global const ulong* restrict co_pp4, uint co_count,            \
    uint co_bucket_cap, uint co_stride, __global uint* restrict co_counts,            \
    __global ulong* restrict co_lo, __global ulong* restrict co_hi, uint co_half,     \
    __global uint* restrict co_actr, __global uint* restrict co_atag
#define COB_DISPATCH_0
#define COB_DISPATCH_1                                                                \
    if (grp % co_stride == co_stride - 1u) {                                          \
        uint eb = grp / co_stride;                    /* rank among entry groups */   \
        uint nE = (uint)get_num_groups(0) / co_stride;                                \
        if (nE == 0u) nE = 1u;                        /* a zero stride would hang */  \
        for (uint g = eb * LDS_WG + lId; g < co_count; g += nE * LDS_WG)              \
            bh3_entry_body(g, co_pp4, bb_, co_bucket_cap, 1u, co_counts,              \
                           co_lo, co_hi, co_half, drops, co_actr, co_atag);           \
        return;                                                                       \
    }                                                                                 \
    grp -= grp / co_stride;                           /* rank among round groups */
#define FUSED_LDS(NAME, ...)     FUSED_LDS_C(NAME, 0, __VA_ARGS__)
#define FUSED_LDS_COB(NAME, ...) FUSED_LDS_C(NAME, 1, __VA_ARGS__)
#define FUSED_LDS_C(NAME, COB, INW, OUTW, LEAFW, LMODE, LOUT, PADN, SIN, SOUT, SBUILD, INSTR, OUTSTR, FCAP, TAB, PERFECT, SPILL) \
__kernel __attribute__((reqd_work_group_size(LDS_WG, 1, 1)))                          \
void NAME(                                                                            \
    uint bucket_bits, uint submask_bits, uint in_bucket_cap, uint out_bucket_cap,     \
    uint Lout, uint Lmix_next, uint padnum_next, uint sIn, uint sOut, uint out_off,   \
    uint sBuild, uint in_stride, uint out_stride,                                     \
    /* restrict: the ping-pong sets are distinct buffers and every output array is    \
       disjoint from every input, so the compiler may reorder loads across stores.    \
       Without it each store barriers the loads that follow.                          \
       lo/hi + half: a record set over the driver's single-allocation cap arrives as  \
       two bucket-halves. Unsplit, the SAME buffer arrives twice with half == nb, so  \
       hi is never dereferenced and the restrict promise holds either way. */         \
    /* side_in/side_out: the r2->r3 side plane (packed only) -- record stride 8    \
       keeps 16 B alignment, the 9th word lives at the element's global slot in    \
       its own never-split buffer. Other rounds get a dummy binding, untouched. */ \
    __global const uint* restrict in_counts,                                          \
    __global const ulong* restrict in_lo, __global const ulong* restrict in_hi,       \
    uint in_half,                                                                     \
    __global const ulong* restrict side_in,                                           \
    __global uint* restrict out_counts,                                               \
    __global ulong* restrict out_lo, __global ulong* restrict out_hi,                 \
    uint out_half,                                                                    \
    __global ulong* restrict side_out,                                                \
    __global uint* restrict all_left, __global uint* restrict all_right,              \
    __global uint* restrict gi_counter, __global uint* restrict drops,                \
    __global const ulong* restrict pp4,                                              \
    /* arena: the input set's per-bucket chains, and the output set's pool cursor.    \
       Bound to a dummy off the arena rungs, where LDS_ARENA compiles the uses out. */\
    __global const uint* restrict in_ahead, __global const uint* restrict in_anext,   \
    __global uint* restrict out_actr, __global uint* restrict out_atag                \
    COB_ARGS_##COB ) {                                                               \
    /* Mode-dependent sizes. Round 1's gi IS its leaf (both are the seed index), so  \
       LMODE_SEED keeps ONE copy in lleaf and lgi collapses. `lead` is leaf 0 of the \
       element's own prefix in every mode whose payload is leaves, so only LMODE_USE \
       (contrib payload) and the generic RAW keep a separate llead. lkey exists only \
       when the table is not a perfect hash. */                                       \
    __local ulong lwork[(INW) * (FCAP)];                                              \
    __local uint  lgi[((LMODE) == LMODE_SEED) ? 1 : (FCAP)];                          \
    __local uint  llead[((LMODE) == LMODE_USE || (LMODE) == LMODE_RAW) ? (FCAP) : 1]; \
    __local uint  lleaf[(LEAFW) * (FCAP)];                                            \
    __local uint  lkey[(PERFECT) ? 1 : (FCAP)];                                       \
    __local uint  lchain[(FCAP)];                                                     \
    __local uint  tab[(TAB)]; __local uint gcount;                                    \
    __local uint  cnt8[(SPILL) ? 8 : 1];                                              \
    __local uint  aidx[(LDS_ARENA) ? LDS_AMAX : 1]; __local uint acnt_sh;             \
    uint lId = get_local_id(0);                                                       \
    /* GEO_BAKED folds these to literals; otherwise they are the kernel args. */      \
    const uint bb_ = (GEO_BAKED) ? GEO_BB : bucket_bits;                              \
    const uint sm_ = (GEO_BAKED) ? GEO_SM : submask_bits;                             \
    const uint incap_  = (GEO_BAKED) ? GEO_INCAP  : in_bucket_cap;                    \
    const uint outcap_ = (GEO_BAKED) ? GEO_OUTCAP : out_bucket_cap;                   \
    uint submaskCount = 1u << sm_;                                                    \
    uint grp = (uint)get_group_id(0);                                                 \
    COB_DISPATCH_##COB                                                                \
    uint bucket = grp / submaskCount;                                                 \
    uint mask   = grp % submaskCount;                                                 \
    uint cnt = in_counts[bucket];                                                     \
    if (cnt > incap_) cnt = incap_;                                                   \
    /* Bucket is uniform per workgroup, so the half select costs nothing per element. */ \
    __global const ulong* restrict in_belem = (bucket < in_half) ? in_lo : in_hi;     \
    size_t base = (size_t)(bucket < in_half ? bucket : bucket - in_half) * incap_; \
    /* Side plane indexes by GLOBAL slot: the plane never splits. */                  \
    size_t side_base = (size_t)bucket * incap_;                                \
    /* This bucket's pool entries, chain-walked ONCE into local before the sweep      \
       loop -- at the dense cap ~95 % of chains are empty, so the common case is one  \
       global load. The pool for this bucket's half starts at in_half * incap_. */    \
    uint acnt = 0u;                                                                   \
    size_t apool = 0u; uint ah_ = 0u;                                                 \
    if (LDS_ARENA) {                                                                  \
        ah_ = (bucket < in_half) ? 0u : 1u;                                           \
        apool = (size_t)in_half * incap_;                                             \
        if (lId == 0) {                                                               \
            uint n_ = 0u;                                                             \
            for (uint a_ = in_ahead[bucket]; a_ != LDS_EMPTY;                         \
                 a_ = in_anext[ah_ * LDS_ARENA_CAP + a_]) {                           \
                if (n_ >= LDS_AMAX) { atomic_inc(&drops[1]); break; }                 \
                aidx[n_++] = a_;                                                      \
            }                                                                         \
            acnt_sh = n_;                                                             \
        }                                                                             \
        barrier(CLK_LOCAL_MEM_FENCE);                                                 \
        acnt = acnt_sh;                                                               \
    }                                                                                 \
    /* SPILL state: xb = extra split bits (level chosen ONCE, never escalated -- an  \
       escalation after a walk would re-emit the coarser parts' children), xp = the  \
       part being processed this pass, nparts = 1 << xb. Without SPILL the loop runs \
       exactly once and this is the old kernel. */                                    \
    uint xb = 0u, xp = 0u, nparts = 1u, counted = 0u;                                 \
    while (xp < nparts) {                                                             \
    uint sbits = sm_ + xb;                                                   \
    uint pmask = mask | (xp << sm_);                                         \
    if (lId == 0) gcount = 0;                                                         \
    for (uint i = lId; i < (TAB); i += LDS_WG) tab[i] = LDS_EMPTY;                    \
    barrier(CLK_LOCAL_MEM_FENCE);                                                     \
    for (uint p = lId; p < cnt + acnt; p += LDS_WG) {                                  \
        /* Pool entries are staged after the bucket's own, from the same buffer. */    \
        size_t sl_ = (LDS_ARENA && p >= cnt) ? apool + aidx[p - cnt] : base + p;       \
        size_t d = sl_ * (INSTR);                                                      \
        ulong rec0; ulong rec1 = 0ul;                                                 \
        if ((LDS_V2) && (LMODE) == LMODE_RD2) {                                       \
            /* PAIR128: both words of the 16 B pair record in one aligned load,       \
               BEFORE the filter -- mirrors CUDA's MXBM_PAIR128 (-0.145 ms there). */ \
            ulong2 q = *((__global const ulong2*)(in_belem + d));                     \
            rec0 = q.x; rec1 = q.y;                                                   \
        } else rec0 = in_belem[d];                                                    \
        uint key = (uint)(rec0 & 0xFFFFFFu);                                          \
        if ((key & ((1u << sbits) - 1u)) != pmask) continue;                          \
        uint pos = atomic_inc(&gcount);                                               \
        if (pos < (FCAP)) {                                                           \
            if ((LMODE) == LMODE_SEED) {                                              \
                /* Round-1 elements are SEEDS: the record is just (index,key) and the \
                   56 B work state is RE-DERIVED. Recompute measured 2.5 ms vs 18.2 ms \
                   to store and re-read the full record (test_seed_rederivation).     \
                   Only the INDEX is captured here -- the expensive derivation happens \
                   in the DEFERRED EXPAND below, where every lane is active. This      \
                   staging loop runs 8 sub-mask passes at ~4/32 lanes; the expand loop \
                   runs once at 32/32. See "non-divergent expand" in docs/performance.md. \
                   gi == lead == leaf 0 for a seed, so ONE lleaf slot carries all      \
                   three (LEAFW is 1 for this mode). */                                \
                lleaf[pos*(LEAFW)] = (uint)(rec0 >> 32);                              \
                if (!(PERFECT)) lkey[pos] = key;                                      \
            } else if ((LMODE) == LMODE_RD2) {                                        \
                /* Unpack the 16 B pair record. Cheap field extraction only -- the     \
                   two seed derivations are deferred to the expand loop below. */      \
                if (!(LDS_V2)) rec1 = in_belem[d + 1u];                               \
                uint li = rd2_left(rec0), ri = rd2_right(rec1);                       \
                lgi[pos] = rd2_gi(rec1);                                              \
                if (!(PERFECT)) lkey[pos] = key;                                      \
                lleaf[pos*(LEAFW) + 0] = li; lleaf[pos*(LEAFW) + 1] = ri;             \
            } else if ((LMODE) == LMODE_RD3) {                                        \
                /* 24 B QUAD record: key, four leaves, gi -- no work words. Three     \
                   scalar loads where LMODE_EMIT below reads nine. Field extraction    \
                   only; the two round-2 rebuilds are deferred to the expand loop. */  \
                ulong w1 = in_belem[d + 1u], w2 = in_belem[d + 2u];                   \
                lgi[pos] = rd3_gi(w2);                                                 \
                if (!(PERFECT)) lkey[pos] = key;                                       \
                lleaf[pos*(LEAFW) + 0] = rd2_left(rec0);                               \
                lleaf[pos*(LEAFW) + 1] = rd3_i1(w1);                                   \
                lleaf[pos*(LEAFW) + 2] = rd3_i2(w1);                                   \
                lleaf[pos*(LEAFW) + 3] = rd3_i3(w2);                                   \
            } else if ((LMODE) == LMODE_EMIT) {                                       \
                /* Stride-8 r2 record [work0..6 | p0], p1 in the side plane: 4 x 16 B \
                   loads where 9 scalars were (CUDA: the MIO-queue win, 41.5->35.3). \
                   Indexed by p, not by the staged slot: the plane has no pool region \
                   behind it, which is why the arena rungs are quad-only and the host \
                   refuses the pairing rather than reaching this line with p >= cnt. */ \
                ulong p0, p1 = side_in[side_base + p];                                \
                if (LDS_V2) {                                                         \
                    __global const ulong2* v = (__global const ulong2*)(in_belem + d); \
                    ulong2 q0 = v[0], q1 = v[1], q2 = v[2], q3 = v[3];                \
                    lwork[pos*(INW) + 0] = q0.x; lwork[pos*(INW) + 1] = q0.y;         \
                    lwork[pos*(INW) + 2] = q1.x; lwork[pos*(INW) + 3] = q1.y;         \
                    lwork[pos*(INW) + 4] = q2.x; lwork[pos*(INW) + 5] = q2.y;         \
                    lwork[pos*(INW) + 6] = q3.x; p0 = q3.y;                           \
                } else {                                                              \
                    for (uint w = 0; w < (INW); ++w) lwork[pos*(INW) + w] = in_belem[d + w]; \
                    p0 = in_belem[d + (INW)];                                         \
                }                                                                     \
                lgi[pos] = r3_gi(p1);                                                 \
                if (!(PERFECT)) lkey[pos] = key;                                      \
                lleaf[pos*(LEAFW) + 0] = r3_l0(p0);                                   \
                lleaf[pos*(LEAFW) + 1] = r3_l1(p0);                                   \
                lleaf[pos*(LEAFW) + 2] = r3_l2(p0, p1);                               \
                lleaf[pos*(LEAFW) + 3] = r3_l3(p1);                                   \
            } else {                                                                  \
                /* NOT deferred, unlike the seed expand: this is a LOAD, not compute.
                   Tried parking p and reading the record in the all-lanes loop --
                   measured SLOWER (r2 24.3->26.2, r4 25.1->27.5). These loads already
                   overlap across the 8 staging iterations, so moving them into a
                   single-iteration loop removes that memory-level parallelism and adds
                   an LDS round-trip. Divergence costs compute, not bandwidth. */      \
                if ((LDS_V2) && (LMODE) == LMODE_USE) {                               \
                    /* Stride-8 r3 record: the 6 work words as 3 x 16 B loads. RAW    \
                       keeps the scalar loop (runtime stride, no alignment claim). */  \
                    __global const ulong2* v = (__global const ulong2*)(in_belem + d); \
                    ulong2 q0 = v[0], q1 = v[1], q2 = v[2];                           \
                    lwork[pos*(INW) + 0] = q0.x; lwork[pos*(INW) + 1] = q0.y;         \
                    lwork[pos*(INW) + 2] = q1.x; lwork[pos*(INW) + 3] = q1.y;         \
                    lwork[pos*(INW) + 4] = q2.x; lwork[pos*(INW) + 5] = q2.y;         \
                } else                                                                \
                    for (uint w = 0; w < (INW); ++w) lwork[pos*(INW) + w] = in_belem[d + w]; \
                ulong meta = in_belem[d + (INW)];                                     \
                lgi[pos] = (uint)(meta >> 32); llead[pos] = (uint)meta;               \
                if (!(PERFECT)) lkey[pos] = key;                                      \
                for (uint i = 0; i < (SIN); ++i)                                      \
                    lleaf[pos*(LEAFW) + i] =                                          \
                        (uint)(in_belem[d + (INW) + 1u + (i >> 1)] >> ((i & 1u) * 32u)); \
            }                                                                         \
        } else if (!(SPILL)) atomic_inc(&drops[1]);                                   \
    }                                                                                \
    barrier(CLK_LOCAL_MEM_FENCE);                                                     \
    if ((SPILL) && gcount > (FCAP)) {                                                 \
        if (!counted) {                                                               \
            /* One word-0 pass counts all 8 possible sub-parts at once, so the split  \
               level is picked from real sizes rather than discovered by trial. Paid  \
               only by the few per cent of groups that overflow. The reads are the    \
               same rescan word-0 reads the staging loop makes -- L2-resident. */     \
            if (lId < 8u) cnt8[lId] = 0u;                                             \
            barrier(CLK_LOCAL_MEM_FENCE);                                             \
            for (uint p = lId; p < cnt; p += LDS_WG) {                                \
                uint k2 = (uint)(in_belem[(base + p) * (INSTR)] & 0xFFu);             \
                if ((k2 & (submaskCount - 1u)) != mask) continue;                     \
                atomic_inc(&cnt8[(k2 >> sm_) & 7u]);                         \
            }                                                                         \
            barrier(CLK_LOCAL_MEM_FENCE);                                             \
            uint lv = 1u;                                                             \
            for (; lv <= 3u; ++lv) {                                                  \
                uint mx = 0u;                                                         \
                for (uint q = 0u; q < (1u << lv); ++q) {                              \
                    uint sz = 0u;                                                     \
                    for (uint i = 0u; i < 8u; ++i)                                    \
                        if ((i & ((1u << lv) - 1u)) == q) sz += cnt8[i];              \
                    if (sz > mx) mx = sz;                                             \
                }                                                                     \
                if (mx <= (FCAP)) break;                                              \
            }                                                                         \
            xb = lv > 3u ? 3u : lv;                                                   \
            nparts = 1u << xb; xp = 0u; counted = 1u;                                 \
            barrier(CLK_LOCAL_MEM_FENCE);                                             \
            continue;                                                                 \
        }                                                                             \
        /* Only reachable if the chosen split still overflows, which the count rules  \
           out at any cap the shipped geometries use. */                              \
        if (lId == 0) atomic_add(&drops[1], gcount - (FCAP));                         \
    }                                                                                 \
    uint total = gcount < (FCAP) ? gcount : (FCAP);                                   \
    for (uint pos = lId; pos < total; pos += LDS_WG) {                               \
        if (!ABL_HIT(2, LMODE) && (LMODE) == LMODE_SEED) {                                    \
            /* DEFERRED EXPAND -- the whole point of splitting the stage. Every lane  \
               here is active (total ~= 256 == LDS_WG, one iteration), whereas the    \
               staging loop above is sub-mask filtered to ~1/8 of its lanes. Same     \
               number of seeds derived, 8x the SIMD utilization. */                   \
            uint idx = lleaf[pos*(LEAFW)];                                            \
            ulong pp[4] = { pp4[0], pp4[1], pp4[2], pp4[3] };                         \
            ulong se[7]; bh3_seed_element(pp, idx, se);                               \
            uint t1[1] = { idx };                                                     \
            se[0] = bh3_apply_mix(se, t1, 1u, 448u);                                  \
            for (uint w = 0; w < (INW); ++w) lwork[pos*(INW) + w] = se[w];            \
        }                                                                             \
        if (!ABL_HIT(2, LMODE) && (LMODE) == LMODE_RD2) {                                     \
            /* DEFERRED EXPAND, round 2: rebuild the 56 B work state from the two      \
               parent seed indices. Measured +0.4 ms here versus +23.5 ms for the same \
               arithmetic in the sub-mask-filtered staging loop. */                    \
            ulong pp[4] = { pp4[0], pp4[1], pp4[2], pp4[3] };                         \
            ulong cc[7];                                                              \
            rd_elem2(pp, lleaf[pos*(LEAFW) + 0], lleaf[pos*(LEAFW) + 1], cc);         \
            for (uint w = 0; w < (INW); ++w) lwork[pos*(INW) + w] = cc[w];            \
        }                                                                             \
        if (!ABL_HIT(2, LMODE) && (LMODE) == LMODE_RD3) {                                     \
            /* DEFERRED EXPAND, round 3: TWO round-2 rebuilds, combined at Lout(2)=400 \
               and mixed at Lmix(3)=400 over the 4-leaf tree. Deferred for the same     \
               reason round 2's is -- this loop runs every lane, the staging loop runs  \
               ~4/32 of them. Fourteen siphash rounds per parent against the 48 B it    \
               saves reading; which side wins is measured, not argued. */                \
            ulong pp[4] = { pp4[0], pp4[1], pp4[2], pp4[3] };                         \
            uint t4[4];                                                                \
            t4[0] = lleaf[pos*(LEAFW) + 0]; t4[1] = lleaf[pos*(LEAFW) + 1];            \
            t4[2] = lleaf[pos*(LEAFW) + 2]; t4[3] = lleaf[pos*(LEAFW) + 3];            \
            ulong a2[7], b2[7], cc[7];                                                 \
            rd_elem2(pp, t4[0], t4[1], a2);                                            \
            rd_elem2(pp, t4[2], t4[3], b2);                                            \
            bh3_combine(a2, b2, 400u, cc);                                             \
            cc[0] = bh3_apply_mix(cc, t4, 4u, 400u);                                   \
            for (uint w = 0; w < (INW); ++w) lwork[pos*(INW) + w] = cc[w];            \
        }                                                                             \
        /* lwork[pos*INW] is word 0, whose low 24 bits ARE the key in every mode --   \
           and it is filled by this point (the derive modes fill it just above). */   \
        uint hkk = (PERFECT) ? (uint)(lwork[pos*(INW)] & 0xFFFFFFu) : lkey[pos];      \
        uint hk = (hkk >> sm_) & ((TAB) - 1u);                               \
        lchain[pos] = atomic_xchg(&tab[hk], pos);                                     \
    }                                                                                \
    barrier(CLK_LOCAL_MEM_FENCE);                                                     \
    for (uint pos = lId; pos < total; pos += LDS_WG) {                               \
        uint key = (PERFECT) ? 0u : lkey[pos];                                        \
        uint oth = lchain[pos], walk = 0;                                             \
        while (oth != LDS_EMPTY) {                                                    \
            if (++walk > 64u) { atomic_inc(&drops[3]); break; }                      \
            /* Perfect table => same chain means same full key; the compare folds. */ \
            if (!ABL_HIT(4, LMODE) && ((PERFECT) || lkey[oth] == key)) {              \
                uint la = ((LMODE) == LMODE_USE || (LMODE) == LMODE_RAW)              \
                              ? llead[pos] : lleaf[pos*(LEAFW)];                      \
                uint lb = ((LMODE) == LMODE_USE || (LMODE) == LMODE_RAW)              \
                              ? llead[oth] : lleaf[oth*(LEAFW)];                      \
                uint ga = ((LMODE) == LMODE_SEED) ? lleaf[pos*(LEAFW)] : lgi[pos];    \
                uint gb = ((LMODE) == LMODE_SEED) ? lleaf[oth*(LEAFW)] : lgi[oth];    \
                uint leftPos = pos, rightPos = oth;                                   \
                if (lb < la || (lb == la && gb < ga)) { leftPos = oth; rightPos = pos; } \
                if (LEADTIE_PROBE && lb == la) atomic_inc(&drops[3]);                 \
                ulong a[7], b[7], c[7];                                               \
                for (uint w = 0; w < 7; ++w) { a[w] = 0ul; b[w] = 0ul; }              \
                for (uint w = 0; w < (INW); ++w) { a[w] = lwork[leftPos*(INW)+w]; b[w] = lwork[rightPos*(INW)+w]; } \
                if (ABL_HIT(128, LMODE)) { for (uint w = 0; w < 7; ++w) c[w] = a[w]; } \
                else bh3_combine(a, b, (LOUT), c);                                    \
                uint ctree[9]; ulong contribOut = 0ul;                                \
                if (ABL_HIT(16, LMODE)) { ctree[0] = llead[leftPos]; }                \
                else if ((LMODE) == LMODE_USE) {                                      \
                    for (uint i = 0; i < 8u; ++i) ctree[i] = 0u;                      \
                    ctree[8] = llead[rightPos];                                        \
                    ulong lc = ((ulong)lleaf[leftPos*(LEAFW)+1] << 32)                 \
                             |  (ulong)lleaf[leftPos*(LEAFW)+0];                       \
                    c[0] = bh3_rotl64(bh3_rotl64(                                      \
                               bh3_apply_mix(c, ctree, (PADN), (LOUT)), 40)            \
                             + lc, 24);                                                \
                    ctree[0] = llead[leftPos];                                         \
                } else {                                                              \
                    for (uint i = 0; i < (SBUILD); ++i)                               \
                        ctree[i] = (i < (SIN)) ? lleaf[leftPos*(LEAFW) + i]           \
                                               : lleaf[rightPos*(LEAFW) + (i-(SIN))];  \
                    c[0] = bh3_apply_mix(c, ctree, (PADN), (LOUT));                   \
                    if ((LMODE) == LMODE_EMIT || (LMODE) == LMODE_RD3) {                                      \
                        ulong zw[7]; for (uint w = 0; w < 7u; ++w) zw[w] = 0ul;        \
                        contribOut = bh3_rotl64(                                       \
                            bh3_apply_mix(zw, ctree, 8u, BH3_LMIX5), 40);              \
                    }                                                                 \
                }                                                                     \
                uint ckey = (uint)(c[0] & 0xFFFFFFu);                                 \
                uint cb = ckey >> (24u - bb_);                                \
                uint cpos = ABL_HIT(32, LMODE) ? ((pos * 7u + walk) & 1023u)          \
                                              : atomic_inc(&out_counts[cb]);          \
                /* A full bucket appends to its half's pool instead of dropping. The  \
                   pool sits behind that half's records, so the slot below addresses  \
                   both regions and every store past it is unchanged. */              \
                size_t oslot = (size_t)(cb < out_half ? cb : cb - out_half)           \
                             * outcap_ + cpos;                                        \
                bool okeep = cpos < outcap_;                                          \
                if (LDS_ARENA && cpos >= outcap_) {                                   \
                    uint oh_ = (cb < out_half) ? 0u : 1u;                             \
                    uint a_  = atomic_inc(&out_actr[oh_]);                            \
                    if (a_ < LDS_ARENA_CAP) {                                         \
                        out_atag[oh_ * LDS_ARENA_CAP + a_] = cb;                      \
                        oslot = (size_t)out_half * outcap_ + a_;                      \
                        okeep = true;                                                 \
                    }                                                                 \
                }                                                                     \
                if (okeep) {                                                          \
                    /* The dense gi is NOT just a cost. It removes a global atomic to \
                       replace it with the bucket slot (cb*cap + cpos) -- measured     \
                       4.5 ms SLOWER, because consecutive gi values make the two       \
                       back-ref writes below coalesce across a warp, while slots       \
                       scatter them over a 190 MB row. Same reason the per-bucket      \
                       atomic_inc beats a computed slot. Do not retry. */              \
                    uint cgi = ABL_HIT(64, LMODE) ? pos : atomic_inc(gi_counter);     \
                    /* One predictable select per child; free (measured 2026-08-01). */ \
                    __global ulong* restrict out_belem =                              \
                        (cb < out_half) ? out_lo : out_hi;                            \
                    size_t od = oslot * (OUTSTR);                                     \
                    if (ABL_HIT(1, LMODE)) { /* payload write ablated */ }                      \
                    else if ((LMODE) == LMODE_SEED) {                                 \
                        /* 16 B PAIR RECORD instead of the 72 B element: round 2       \
                           re-derives the work state from these two indices, so the    \
                           work words and the leaf payload need not be stored at all.  \
                           ctree[0]/ctree[1] are the two parent seed indices. */       \
                        if (LDS_V2)                                                   \
                            *((__global ulong2*)(out_belem + od)) =                   \
                                (ulong2)(rd2_w0(ckey, ctree[0]), rd2_w1(ctree[1], cgi)); \
                        else {                                                        \
                            out_belem[od + 0u] = rd2_w0(ckey, ctree[0]);              \
                            out_belem[od + 1u] = rd2_w1(ctree[1], cgi);               \
                        }                                                             \
                    } else if ((LMODE) == LMODE_RD2 && (OUTSTR) == 3u) {              \
                        /* QUAD RECORD: key, four leaves, gi in 3 u64 where the packed \
                           record below takes 9. The work words are dropped entirely;  \
                           round 3 rebuilds them from these same four leaves. Keyed on \
                           OUTSTR rather than a mode of its own, so r2's staging and    \
                           expand stay byte-identical between the two record formats. */ \
                        out_belem[od + 0u] = rd2_w0(ckey, ctree[0]);                  \
                        out_belem[od + 1u] = rd3_w1(ctree[1], ctree[2]);              \
                        out_belem[od + 2u] = rd3_w2(ctree[3], cgi);                   \
                    } else if ((LMODE) == LMODE_RD2) {                                \
                        /* Stride-8 record [work0..6 | p0]; p1 to the side plane at   \
                           the GLOBAL slot (the plane never splits). lead == ctree[0] \
                           == leaf 0, so storing it again was waste. */               \
                        if (LDS_V2) {                                                 \
                            __global ulong2* v = (__global ulong2*)(out_belem + od);  \
                            v[0] = (ulong2)(c[0], c[1]);                              \
                            v[1] = (ulong2)(c[2], c[3]);                              \
                            v[2] = (ulong2)(c[4], c[5]);                              \
                            v[3] = (ulong2)(c[6], r3_p0(ctree[0], ctree[1], ctree[2])); \
                        } else {                                                      \
                            for (uint w = 0; w < (OUTW); ++w) out_belem[od + w] = c[w]; \
                            out_belem[od + (OUTW)] = r3_p0(ctree[0], ctree[1], ctree[2]); \
                        }                                                             \
                        side_out[(size_t)cb * outcap_ + cpos] =                \
                            r3_p1(ctree[2], ctree[3], cgi);                           \
                    } else if ((LDS_V2) && ((LMODE) == LMODE_EMIT || (LMODE) == LMODE_RD3)) { \
                        __global ulong2* v = (__global ulong2*)(out_belem + od);      \
                        v[0] = (ulong2)(c[0], c[1]);                                  \
                        v[1] = (ulong2)(c[2], c[3]);                                  \
                        v[2] = (ulong2)(c[4], c[5]);                                  \
                        v[3] = (ulong2)(((ulong)cgi << 32) | (ulong)ctree[0], contribOut); \
                    } else if ((LDS_V2) && (LMODE) == LMODE_USE) {                    \
                        *((__global ulong2*)(out_belem + od)) =                       \
                            (ulong2)(c[0], ((ulong)cgi << 32) | (ulong)ctree[0]);     \
                    } else {                                                          \
                        for (uint w = 0; w < (OUTW); ++w) out_belem[od + w] = c[w];   \
                        out_belem[od + (OUTW)] = ((ulong)cgi << 32) | (ulong)ctree[0]; \
                        if ((LMODE) == LMODE_EMIT || (LMODE) == LMODE_RD3) {                                  \
                            out_belem[od + (OUTW) + 1u] = contribOut;                 \
                        } else {                                                      \
                            for (uint j = 0; j*2u < (SOUT); ++j) {                    \
                                uint lo = ctree[j*2u];                                 \
                                uint hi = (j*2u+1u < (SOUT)) ? ctree[j*2u+1u] : 0u;    \
                                out_belem[od + (OUTW) + 1u + j] = ((ulong)hi << 32) | (ulong)lo; \
                            }                                                         \
                        }                                                             \
                    }                                                                 \
                    if (!ABL_HIT(8, LMODE)) {                                         \
                      all_left[out_off + cgi]  = ((LMODE) == LMODE_SEED)              \
                          ? lleaf[leftPos*(LEAFW)]  : lgi[leftPos];                   \
                      all_right[out_off + cgi] = ((LMODE) == LMODE_SEED)              \
                          ? lleaf[rightPos*(LEAFW)] : lgi[rightPos];                  \
                    }                                                                 \
                } else atomic_inc(&drops[2]);                                         \
            }                                                                        \
            oth = lchain[oth];                                                        \
        }                                                                            \
    }                                                                                \
    barrier(CLK_LOCAL_MEM_FENCE);       /* next spill pass reuses every array above */ \
    ++xp;                                                                             \
    }                                                                                 \
}
// (INW, OUTW, LEAFW = LDS leaf-payload uints/elem, LMODE, ..., FCAP, TAB, PERFECT,
// SPILL). LEAFW is per-round rather than a fixed 8, which also trims LDS: r1 stages
// ONE uint (gi == lead == leaf 0 == the seed index), r2 two leaves, r3 four, r4 the
// 2-uint contrib. The shipped kernels run the perfect 128-entry table and spill at a
// near-mean cap (r1 288, rest 320); MXBM_CL_OPTS -D overrides restore the old
// configuration at runtime for A/Bs.
FUSED_LDS(round_fused_seed, 7, 7, 1, LMODE_SEED, 424u, 2u, 1u, 2u, 2u, 1u, 2u,
          LDS_FCAP_R1, LDS_FTAB, LDS_PERFECT_TAB, LDS_SPILL)  // r1: seeds in, 16 B pair record out
FUSED_LDS(round_fused_rd2,  7, 7, 2, LMODE_RD2, 400u, 4u, 2u, 4u, 4u, 2u, 8u,
          LDS_FCAP, LDS_FTAB, LDS_PERFECT_TAB, LDS_SPILL)     // r2: stride-8 record + side plane
// RUNTIME-PARAMETERISED variant: the constants are the kernel's own runtime args, so
// this one kernel still validates the fused mechanism generically across r=1..4
// (tests/test_lds_collide.cpp, test_fused_round). Not on the pipeline path -- the four
// shipped kernels above bake their per-round constants in, which is worth ~26 ms.
// Lout(r) == Lmix(r+1) for every round, so `Lout` serves both roles. It keeps the
// LEGACY configuration -- tail cap 384, 512-entry keyed table, no spill -- because its
// tests run it at geometries (e.g. bits=11, submask=0) where the perfect-hash argument
// does not hold.
FUSED_LDS(round_fused_lds,  7, 7, 2, LMODE_RAW, Lout, padnum_next, sIn, sOut, sBuild, in_stride, out_stride,
          384u, LDS_TABSIZE, 0, 0)
FUSED_LDS(round_fused_7_6,  7, 6, 4, LMODE_EMIT, 376u, 6u, 4u, 2u, 8u, 8u, 8u,
          LDS_FCAP, LDS_FTAB, LDS_PERFECT_TAB, LDS_SPILL)     // r3 fallback (stride-8 + side plane)
// QUAD-RECORD PAIR (rowbucket_geom.h `quad`). Only the strides and round 3's mode
// change: the record carries the same INFORMATION either way, just not the same bytes.
// r2 emits 3 u64 instead of 9, and set 0's stride collapses 9 -> 3 with it, which is
// what takes the largest single allocation from 2.76 to 2.45 GiB -- under the
// CL_DEVICE_MAX_MEM_ALLOC_SIZE of an 11 GB card, which is why this exists on OpenCL.
FUSED_LDS(round_fused_rd2q, 7, 7, 2, LMODE_RD2, 400u, 4u, 2u, 4u, 4u, 2u, 3u,
          LDS_FCAP, LDS_FTAB, LDS_PERFECT_TAB, LDS_SPILL)     // r2: emits the 24 B quad record
FUSED_LDS(round_fused_rd3,  7, 6, 4, LMODE_RD3, 376u, 6u, 4u, 2u, 8u, 3u, 8u,
          LDS_FCAP, LDS_FTAB, LDS_PERFECT_TAB, LDS_SPILL)     // r3: rebuilds from it
FUSED_LDS(round_fused_6_5,  6, 1, 2, LMODE_USE, 288u, 9u, 2u, 0u, 0u, 8u, 2u,
          LDS_FCAP, LDS_FTAB, LDS_PERFECT_TAB, LDS_SPILL)     // r4: stages contrib, no leaf emit

// ENTRY (round 1) for the fused row-bucket path: seed_element + apply_mix(Lmix=448,
// single-leaf tree {idx}) -- exactly round1_mix_seeds -- then scatter the mixed
// element DIRECTLY into round-1 FAT buckets (no flat work[1]/leaves[1]). gi = seed
// index (recover reads it as a leaf at row 0); lead = leaves[0] = idx.
// The per-element half lives in its own function so round 4's co-block variant can
// run it from its own workgroups (COB_DISPATCH_1) instead of duplicating it.
static void bh3_entry_body(uint idx, __global const ulong* restrict pp4,
                           uint bucket_bits, uint bucket_cap, uint stride,
                           __global uint* restrict counts,
                           __global ulong* restrict belem_lo,
                           __global ulong* restrict belem_hi,
                           uint half_buckets,
                           __global uint* restrict drops,
                           __global uint* restrict actr,
                           __global uint* restrict atag) {
    ulong pp[4] = { pp4[0], pp4[1], pp4[2], pp4[3] };
    ulong e[7];
    bh3_seed_element(pp, idx, e);
    uint tree1[1] = { idx };
    e[0] = bh3_apply_mix(e, tree1, 1u, 448u);
    uint key = (uint)(e[0] & 0xFFFFFFu);
    uint b = key >> (24u - bucket_bits);
    uint pos = atomic_inc(&counts[b]);
    // THIN round-1 record: (index << 32) | key, 8 B total. The 56 B work state is
    // NOT stored -- round 1 re-derives it from the index while staging into LDS
    // (LMODE_SEED). gi, lead and leaf0 are all the index for a seed element, so
    // nothing else needs carrying. Measured 2.5 ms vs 18.2 ms for the 72 B record.
    // lo/hi + half_buckets: split record set, see the FUSED_LDS note.
    __global ulong* restrict belem = (b < half_buckets) ? belem_lo : belem_hi;
    size_t slot = (size_t)(b < half_buckets ? b : b - half_buckets)*bucket_cap + pos;
#if LDS_ARENA
    if (pos >= bucket_cap) {
        // Overflow goes to this half's pool, which lives behind its bucket records.
        uint h = (b < half_buckets) ? 0u : 1u;
        uint a = atomic_inc(&actr[h]);
        if (a >= LDS_ARENA_CAP) { atomic_inc(&drops[1]); return; }
        atag[h * LDS_ARENA_CAP + a] = b;
        slot = (size_t)half_buckets * bucket_cap + a;
    }
#else
    if (pos >= bucket_cap) { atomic_inc(&drops[1]); return; }
#endif
    belem[slot * stride] = ((ulong)idx << 32) | (ulong)key;
}

// No reqd_work_group_size here, unlike the fused kernels: pinning it to 256 (what
// CUDA's entry_scatter uses) measured NULL on Ada across two bracketed ABBA rounds,
// and this kernel is a flat 1-D map with no LDS, so the driver's own choice is the
// better default on the cards we cannot measure.
__kernel void round1_mix_scatter_fat(__global const ulong* restrict pp4, uint begin, uint count,
                                     uint bucket_bits, uint bucket_cap, uint stride,
                                     __global uint* restrict counts,
                                     __global ulong* restrict belem_lo,
                                     __global ulong* restrict belem_hi,
                                     uint half_buckets,
                                     __global uint* restrict drops,
                                     __global uint* restrict actr,
                                     __global uint* restrict atag) {
    uint g = (uint)get_global_id(0);
    if (g >= count) return;
    bh3_entry_body(begin + g, pp4, bucket_bits, bucket_cap, stride, counts,
                   belem_lo, belem_hi, half_buckets, drops, actr, atag);
}

// Between a producer and its consumer: thread this round's pool entries onto per-bucket
// chains. ahead must be 0xFF-memset first. spill_total is the run-long POSITIVE CONTROL
// -- a pool that never fills means the dense cap was never exceeded and the arena
// measured nothing, which reads identically to "the arena works" without it.
__kernel void arena_link(__global const uint* restrict actr,
                         __global const uint* restrict atag,
                         __global uint* restrict ahead, __global uint* restrict anext,
                         __global uint* restrict spill_total, uint halves) {
    uint i = (uint)get_global_id(0);
    if (i == 0u) {
        uint t = 0u;
        for (uint h = 0u; h < halves; ++h)
            t += min(actr[h], LDS_ARENA_CAP);
        atomic_add(spill_total, t);
    }
    uint h = i / LDS_ARENA_CAP, a = i % LDS_ARENA_CAP;
    if (h >= halves || a >= min(actr[h], LDS_ARENA_CAP)) return;
    anext[i] = atomic_xchg(&ahead[atag[i]], a);
}

// Round 4 again, hosting the next nonce's entry as co-blocks. Instantiated here, not
// beside its siblings, because the dispatch calls bh3_entry_body. Parameters must
// match round_fused_6_5 exactly -- a drifting copy would mine a different round 4.
FUSED_LDS_COB(round_fused_6_5_cob, 6, 1, 2, LMODE_USE, 288u, 9u, 2u, 0u, 0u, 8u, 2u,
              LDS_FCAP, LDS_FTAB, LDS_PERFECT_TAB, LDS_SPILL)

// TERMINAL (round 5) for the fused row-bucket path: like round_fused_lds but NO
// child mix/scatter -- combine colliding round-5 pairs at Lout(5)=24 and detect the
// all-zero survivor (two byte-identical round-5 elements XOR to 0). No leaves needed
// (recover walks back-refs). Survivor slot = a dense atomic index; its back-ref row
// (out_off = 4*capacity) holds the two round-5 parent gi's so recover can descend.
__kernel __attribute__((reqd_work_group_size(LDS_WG, 1, 1)))
void round5_fused_lds(
    uint bucket_bits, uint submask_bits, uint in_bucket_cap, uint Lout, uint out_off,
    uint in_stride,
    __global const uint*  restrict in_counts,
    __global const ulong* restrict in_lo,   // packed: work[0..4], meta=(gi<<32)|lead --
    __global const ulong* restrict in_hi,   // as bucket-halves when split, see FUSED_LDS
    uint in_half,
    __global uint* restrict all_left, __global uint* restrict all_right,  // by survivor slot
    __global uint* restrict surv_slots, __global uint* restrict surv_count, uint surv_cap,
    __global uint* restrict drops,      // [1]=group ovf, [3]=chain cap, [2]=surv ovf
    __global const uint* restrict in_ahead, __global const uint* restrict in_anext) {
    // Round-5 input = ONE work word, not the 5 significant ones round 4 produces.
    // The terminal test is z = OR(c[0..6]) after bh3_combine at Lout(5) = 24, which
    // forces c[1..6] to zero and masks c[0] to 24 bits; and
    //   c[0] = ((x[0] >> 24) | (x[1] << 40)) & 0xFFFFFF
    // where the x[1] term shifts zeros into bits 0..23. So the test reduces to
    // "bits 24..47 of a[0]^b[0] are zero" -- words 1..4 are never consulted. The key
    // (word 0's low 24 bits) comes from the same word. Round 4 therefore emits 1 work
    // word + meta = 16 B instead of 48 B. Checked over 200k random pairs and gated by
    // the survivor count (must stay 3 on the KAT) plus the goldens.
    #define LDS_R5W 1u
    #define LDS_R5STR 2u     // kFbStride[5]: 1 work word + meta
    #define LDS_R5LOUT 24u   // Lout(5); see the note at the combine below
    // LDS_TCAP, not LDS_FCAP: this kernel stages 24 B/element, an eighth of a round
    // block, so its cap never constrained occupancy -- it keeps covering the group
    // TAIL rather than inheriting the spill-backed near-mean cap of the rounds.
    // Same perfect-hash argument as the fused rounds (LDS_PERFECT_TAB): on the
    // bb + sm = 17 line only 7 key bits vary inside a group, so the 128-entry table
    // covers them exactly -- same chain <=> same full key, and lkey plus the walk's
    // compare are dead. run_pipeline_rowbucket's geometry check guards this kernel
    // too, and -DLDS_PERFECT_TAB=0 restores the keyed walk at 512 entries.
    #define LDS_R5TAB (LDS_PERFECT_TAB ? LDS_FTAB : LDS_TABSIZE)
    __local ulong lwork[LDS_R5W * LDS_TCAP];
    __local uint  lgi[LDS_TCAP];
    __local uint  llead[LDS_TCAP];
    __local uint  lkey[LDS_PERFECT_TAB ? 1 : LDS_TCAP];
    __local uint  lchain[LDS_TCAP];
    __local uint  tab[LDS_R5TAB];
    __local uint  gcount;
    __local uint  aidx[LDS_ARENA ? LDS_AMAX : 1];
    __local uint  acnt_sh;

    uint lId = get_local_id(0);
    uint submaskCount = 1u << submask_bits;
    uint bucket = get_group_id(0) / submaskCount;
    uint mask   = get_group_id(0) % submaskCount;

    if (lId == 0) gcount = 0;
    for (uint i = lId; i < LDS_R5TAB; i += LDS_WG) tab[i] = LDS_EMPTY;
    barrier(CLK_LOCAL_MEM_FENCE);

    uint cnt = in_counts[bucket];
    if (cnt > in_bucket_cap) cnt = in_bucket_cap;
    __global const ulong* in_belem = (bucket < in_half) ? in_lo : in_hi;
    size_t base = (size_t)(bucket < in_half ? bucket : bucket - in_half) * in_bucket_cap;
    // Same chain walk as the fused rounds': round 4's output carries a pool too, and
    // a survivor's ancestor can sit in it.
    uint acnt = 0u;
    size_t apool = 0u;
    if (LDS_ARENA) {
        const uint ah_ = (bucket < in_half) ? 0u : 1u;
        apool = (size_t)in_half * in_bucket_cap;
        if (lId == 0) {
            uint n_ = 0u;
            for (uint a_ = in_ahead[bucket]; a_ != LDS_EMPTY;
                 a_ = in_anext[ah_ * LDS_ARENA_CAP + a_]) {
                if (n_ >= LDS_AMAX) { atomic_inc(&drops[1]); break; }
                aidx[n_++] = a_;
            }
            acnt_sh = n_;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        acnt = acnt_sh;
    }
    for (uint p = lId; p < cnt + acnt; p += LDS_WG) {
        size_t d = ((LDS_ARENA && p >= cnt) ? apool + aidx[p - cnt] : base + p) * LDS_R5STR;
#if LDS_V2
        ulong2 q = *((__global const ulong2*)(in_belem + d));
        uint key = (uint)(q.x & 0xFFFFFFu);
        if ((key & (submaskCount - 1u)) != mask) continue;
        uint pos = atomic_inc(&gcount);
        if (pos < LDS_TCAP) {
            lwork[pos*LDS_R5W] = q.x;
            lgi[pos] = (uint)(q.y >> 32); llead[pos] = (uint)q.y;
            if (!LDS_PERFECT_TAB) lkey[pos] = key;
        } else atomic_inc(&drops[1]);
#else
        uint key = (uint)(in_belem[d] & 0xFFFFFFu);
        if ((key & (submaskCount - 1u)) != mask) continue;
        uint pos = atomic_inc(&gcount);
        if (pos < LDS_TCAP) {
            for (uint w = 0; w < LDS_R5W; ++w) lwork[pos*LDS_R5W + w] = in_belem[d + w];
            ulong meta = in_belem[d + LDS_R5W];
            lgi[pos] = (uint)(meta >> 32); llead[pos] = (uint)meta;
            if (!LDS_PERFECT_TAB) lkey[pos] = key;
        } else atomic_inc(&drops[1]);
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint total = gcount < LDS_TCAP ? gcount : LDS_TCAP;

    for (uint pos = lId; pos < total; pos += LDS_WG) {
        // lwork[pos] is work word 0, whose low 24 bits ARE the key.
        uint kk = LDS_PERFECT_TAB ? (uint)(lwork[pos*LDS_R5W] & 0xFFFFFFu) : lkey[pos];
        uint hk = (kk >> submask_bits) & (LDS_R5TAB - 1u);
        lchain[pos] = atomic_xchg(&tab[hk], pos);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint pos = lId; pos < total; pos += LDS_WG) {
        uint key = LDS_PERFECT_TAB ? 0u : lkey[pos];
        uint oth = lchain[pos], walk = 0;
        while (oth != LDS_EMPTY) {
            if (++walk > 64u) { atomic_inc(&drops[3]); break; }
            if (LDS_PERFECT_TAB || lkey[oth] == key) {
                uint la = llead[pos], lb = llead[oth], ga = lgi[pos], gb = lgi[oth];
                uint leftPos = pos, rightPos = oth;
                if (lb < la || (lb == la && gb < ga)) { leftPos = oth; rightPos = pos; }
                ulong a[7], b[7], c[7];
                for (uint w = 0; w < 7; ++w) { a[w] = 0ul; b[w] = 0ul; }
                for (uint w = 0; w < LDS_R5W; ++w) { a[w] = lwork[leftPos*LDS_R5W+w]; b[w] = lwork[rightPos*LDS_R5W+w]; }
                /* LDS_R5LOUT, not the runtime `Lout` arg: a compile-time Lout folds
                   bh3_combine's masking loop away entirely (at 24 it is "mask word 0,
                   zero words 1..6"). Same reason the fused rounds bake theirs in --
                   see the FUSED_LDS constants. Pinned by test_gpu_rounds. */
                bh3_combine(a, b, LDS_R5LOUT, c);
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
