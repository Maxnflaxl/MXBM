// BeamHash III Wagner-round kernels (mix / scatter / match).
// Consumes bh3.cl primitives (compiled together; bh3.cl listed first).

// E3a leaf-prefix layout: AoS with fixed stride = max prefix (9 = padNum(5)).
// element g's leaves are the contiguous block [g*9 .. g*9+9). Contiguous per-child
// writes in round_match (the hot producer) beat SoA's 9 scattered cache lines.
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
    // round_match can grow leaf lists incrementally and round_mix skips the DFS.
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
// COMPACTION variants: work stored at INW = inwords_for(r) significant words. The
// upper (7-INW) words are 0 in the uncompacted layout (prior round's bh3_combine
// masked them to Lout) and apply_mix folds only e[0..6] (rotl(0)=0), so reading INW
// words + zero-filling is bit-exact. INW is a COMPILE-TIME constant so the read
// loop fully unrolls -- a runtime stride here cost +24 ms (mix regressed 50->74),
// the same unrolling-killer that sank runtime-width match. r2,r3 (INW=7) reuse the
// stock round_mix unchanged; only r4 (6) and r5 (5) need a narrowed variant.
#define ROUND_MIX_C(NAME, INW)                                                  \
__kernel void NAME(uint N, uint capacity, uint padNum, uint Lmix,               \
                   __global ulong* work,                                        \
                   __global const uint* leaves_in,                             \
                   __global ulong* pairs_out) {                                 \
    uint g = (uint)get_global_id(0);                                           \
    if (g >= N) return;                                                        \
    uint tree[9];                                                             \
    for (uint i = 0; i < padNum; ++i) tree[i] = leaves_in[(size_t)g*BH3_MAX_LEAVES + i]; \
    ulong e[7];                                                               \
    for (int k = 0; k < 7; ++k) e[k] = 0ul;                                   \
    for (int k = 0; k < (INW); ++k) e[k] = work[(size_t)g*(INW) + k];          \
    e[0] = bh3_apply_mix(e, tree, padNum, Lmix);                              \
    work[(size_t)g*(INW)] = e[0];                                             \
    pairs_out[(size_t)g] = ((ulong)g << 32) | (uint)(e[0] & 0xFFFFFFu);        \
}
ROUND_MIX_C(round_mix_c6, 6)   // r4 input: 6 significant words
ROUND_MIX_C(round_mix_c5, 5)   // r5 input: 5 significant words
// round_scatter: radix-bucket N mixed elements by the top bucket_bits of
// their 24-bit collision key, so the sortless all-pairs match (T4) only
// searches within a bucket. bucket_count[b] is the raw (uncapped) atomic
// arrival count for bucket b; only the first `slots` arrivals are recorded
// into bucket_slots -- the rest are dropped and tallied in counters[1].
// Bucket order is nondeterministic under atomic_inc (host must compare
// membership as SETS, not sequences). bucket_count/counters must be zeroed
// by the caller (fill_u32) before this kernel runs.
__kernel void round_scatter(uint N, uint bucket_bits, uint slots,
                            __global const ulong* work,
                            __global uint* bucket_count,
                            __global uint* bucket_slots,
                            __global uint* counters /* [1]=bucket_drops */) {
    uint g = (uint)get_global_id(0);
    if (g >= N) return;
    uint key = (uint)(work[(size_t)g*7] & 0xFFFFFFu);
    uint b = key >> (24u - bucket_bits);
    uint pos = atomic_inc(&bucket_count[b]);
    if (pos < slots) bucket_slots[(size_t)b*slots + pos] = g;
    else atomic_inc(&counters[1]);
}
// round_match: one work-item per bucket. Sortless all-pairs within the
// bucket (k = min(bucket_count[b], slots) members recorded by scatter);
// every colliding pair (equal 24-bit key, re-checked here from in_work --
// scatter only bucketed by the top bucket_bits) is emitted, ordered
// left=smaller (lead,slot) -- lead is either the slot itself (round 1,
// lead_identity!=0, no back-ref history yet) or looked up in all_lead_in at
// in_lead_off (round >=2). No distinct/indexAfter gate: equal-lead and
// overlapping-tree pairs ARE emitted (a later Phase-C gate filters those).
// Combines the ordered pair at Lout, writes the child into out_work[oi] (0
// based) plus a consolidated back-ref triple at all_left/all_right/all_lead
// [out_off+oi]. counters[0] is the raw (uncapped) atomic child count;
// entries at oi>=out_capacity are dropped and tallied in counters[2].
// LOCAL-MEMORY BUCKET STAGING variant: one WORKGROUP per bucket. The group
// cooperatively stages this bucket's per-slot 24-bit collision keys and element
// indices into __local memory with a single coalesced pass over global memory
// (one element-key read per slot instead of O(k^2) re-reads), then runs the
// all-pairs collision scan entirely out of __local. The expensive 56-byte
// element read + bh3_combine only fires for ACTUAL colliding pairs (rare), so
// the ~2B uncoalesced key re-reads of the per-work-item version collapse to
// ~33M staged loads. Output multiset is identical: every i<j pair is owned by
// exactly one work-item (the one whose strided row set contains i), the same
// (lead,slot) tie-break picks left/right/lead, and children/back-refs are
// appended via the same atomic counters[0] cursor (emission order is free --
// the tests compare as multisets). lkeys/lidx are __local scratch sized to
// `slots` uints each, supplied by the host as dynamic local args.
__kernel void round_match(uint Lout, uint bucket_bits, uint slots, uint lead_identity,
                          uint in_lead_off, uint out_off, uint out_capacity,
                          __global const uint* bucket_count,
                          __global const uint* bucket_slots,
                          __global const ulong* in_work,
                          __global ulong* out_work,
                          __global const uint* all_lead_in,   // same buffer as lead; read at in_lead_off
                          __global uint* all_left, __global uint* all_right, __global uint* all_lead,
                          __global uint* counters /* [0]=out_count, [2]=pair_drops */,
                          __local uint* lkeys, __local uint* lidx,
                          __global const uint* leaves_in,  // AoS leaf prefix (slot*9+i) of INPUT elems (work[r])
                          __global uint* leaves_out,       // SoA leaf prefix of the CHILDREN (work[r+1])
                          uint s_in, uint s_out) {         // leaves/elem in / out (s_out==0 -> skip, r==5)
    uint b = (uint)get_group_id(0);
    uint t = (uint)get_local_id(0);
    uint W = (uint)get_local_size(0);
    uint cnt = bucket_count[b];
    uint k = cnt < slots ? cnt : slots;

    // Stage: coalesced read of each member's slot index + 24-bit key into local.
    size_t base = (size_t)b * slots;
    for (uint p = t; p < k; p += W) {
        uint g = bucket_slots[base + p];
        lidx[p]  = g;
        lkeys[p] = (uint)(in_work[(size_t)g*7] & 0xFFFFFFu);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // All-pairs scan in local memory. Work-item t owns rows i = t, t+W, t+2W...
    // so each unordered pair (i<j) is handled exactly once.
    for (uint i = t; i + 1u < k; i += W) {
        uint sa = lidx[i];
        uint ka = lkeys[i];
        for (uint j = i + 1u; j < k; ++j) {
            if (ka != lkeys[j]) continue;
            uint sb = lidx[j];
            // Lead == the element's tree[0] == its first materialized leaf, so
            // read it straight from leaves_in (all_lead is redundant with E3a and
            // no longer written -- see below). in_lead_off/all_lead_in unused now.
            uint la = lead_identity ? sa : leaves_in[(size_t)sa*BH3_MAX_LEAVES];
            uint lb = lead_identity ? sb : leaves_in[(size_t)sb*BH3_MAX_LEAVES];
            uint left = sa, right = sb, ll = la;
            if (lb < la || (lb == la && sb < sa)) { left = sb; right = sa; ll = lb; }
            ulong ea[7], eb[7], ec[7];
            for (int w = 0; w < 7; ++w) { ea[w] = in_work[(size_t)left*7+w]; eb[w] = in_work[(size_t)right*7+w]; }
            bh3_combine(ea, eb, Lout, ec);
            uint oi = atomic_inc(&counters[0]);
            if (oi < out_capacity) {
                for (int w = 0; w < 7; ++w) out_work[(size_t)oi*7 + w] = ec[w];
                all_left[out_off + oi] = left; all_right[out_off + oi] = right;
                // all_lead[out_off + oi] = ll;  // DROPPED: == leaves_out[oi*9] (redundant)
                // E3a: child's pre-order leaf prefix = left parent's full prefix
                // (s_in leaves) then the right parent's, truncated to s_out. s_in
                // is the FULL leaf count for work[r] (r<=4), so no left leaf is
                // missing. Stored SoA (stride out_capacity) so round_mix reads it
                // coalesced. s_out==0 for r==5 (its output is never mixed).
                for (uint i = 0; i < s_out; ++i) {
                    uint leaf = (i < s_in) ? leaves_in[(size_t)left*BH3_MAX_LEAVES + i]
                                           : leaves_in[(size_t)right*BH3_MAX_LEAVES + (i - s_in)];
                    leaves_out[(size_t)oi*BH3_MAX_LEAVES + i] = leaf;
                }
            } else atomic_inc(&counters[2]);
        }
    }
}
// survivor_scan: scans round 5's children (in work[0] -- round_match's r==5
// special case, which has no round 6 / work[6] to feed) for the ALL-ZERO
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
__kernel void survivor_scan(uint N, __global const ulong* work,
                            __global uint* out_slots, uint cap,
                            __global uint* counters /* [3]=survivors */) {
    uint g = (uint)get_global_id(0);
    if (g >= N) return;
    for (int w = 0; w < 7; ++w) if (work[(size_t)g*7 + w] != 0ul) return;
    uint oi = atomic_inc(&counters[3]);
    if (oi < cap) out_slots[oi] = g;
}
// COMPACTION variant: r5 children stored at `stride` = outwords_for(5) = 1 word.
// Lout=24 keeps only word0's low 24 bits (upper 40 bits of word0 and all of
// words 1..6 are 0), so the all-7-zero solution marker collapses to word0 == 0.
__kernel void survivor_scan_s(uint N, uint stride, __global const ulong* work,
                              __global uint* out_slots, uint cap,
                              __global uint* counters /* [3]=survivors */) {
    uint g = (uint)get_global_id(0);
    if (g >= N) return;
    if (work[(size_t)g*stride] != 0ul) return;
    uint oi = atomic_inc(&counters[3]);
    if (oi < cap) out_slots[oi] = g;
}
// recover: one work-item per survivor. Walk the consolidated back-refs from the
// round-5 survivor (slot in work[0], back-ref at row 4 = (5-1)*capacity) down to
// round 1 (row 0, whose left/right entries ARE leaf indices), collecting all 32
// leaves in tree order (pre-order: left subtree fully, then right -- the same
// order round_mix's DFS uses and the verifier expects). No apply_mix.
__kernel void recover(uint nSurv, __global const uint* surv_slots, uint capacity,
                      __global const uint* all_left, __global const uint* all_right,
                      __global uint* out /* [nSurv*32] */) {
    uint i = (uint)get_global_id(0);
    if (i >= nSurv) return;
    uint got = 0u;
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
