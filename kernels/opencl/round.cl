// BeamHash III Wagner-round kernels (mix / scatter / match).
// Consumes bh3.cl primitives (compiled together; bh3.cl listed first).

// round1_mix_seeds: seed_element(pp,begin+g) mixed with the single-leaf tree
// {begin+g} at Lmix=448 (round 1's shape), written to work[] at the ABSOLUTE
// index (begin+g)*7 -- not the batch-local g*7 -- since callers may drive
// this over multiple begin/count batches that all land in the same
// consolidated work buffer.
__kernel void round1_mix_seeds(__global const ulong* pp4, uint begin, uint count,
                               __global ulong* work /* seed-work buffer, absolute */) {
    uint g = (uint)get_global_id(0);
    if (g >= count) return;
    uint idx = begin + g;
    ulong pp[4] = { pp4[0], pp4[1], pp4[2], pp4[3] };
    ulong e[7];
    bh3_seed_element(pp, idx, e);
    uint tree1[1] = { idx };
    e[0] = bh3_apply_mix(e, tree1, 1u, 448u);
    for (int k = 0; k < 7; ++k) work[(size_t)idx*7 + k] = e[k];
}
// r>=2: mix level-(r-1) elements in place, reconstructing their leaf prefix.
__kernel void round_mix(uint level /* = r-1, in 1..4 */, uint N, uint capacity,
                        uint padNum, uint Lmix,
                        __global ulong* work,               // level's work buffer
                        __global const uint* all_left,      // consolidated
                        __global const uint* all_right) {
    uint g = (uint)get_global_id(0);
    if (g >= N) return;
    uint tree[9]; uint got = 0u;
    uint lvl[8]; uint slt[8]; int sp = 0;      // explicit pre-order stack
    lvl[0] = level; slt[0] = g; sp = 1;
    while (sp > 0 && got < padNum) {
        --sp;
        uint lv = lvl[sp], sl = slt[sp];
        uint off = (lv - 1u) * capacity;
        uint L = all_left[off + sl], R = all_right[off + sl];
        if (lv == 1u) { tree[got++] = L; if (got < padNum) tree[got++] = R; }
        else { lvl[sp] = lv-1u; slt[sp] = R; ++sp;   // right pushed first,
               lvl[sp] = lv-1u; slt[sp] = L; ++sp; } // left pops first
    }
    ulong e[7];
    for (int k = 0; k < 7; ++k) e[k] = work[(size_t)g*7 + k];
    e[0] = bh3_apply_mix(e, tree, padNum, Lmix);
    work[(size_t)g*7] = e[0];
}
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
__kernel void round_match(uint Lout, uint bucket_bits, uint slots, uint lead_identity,
                          uint in_lead_off, uint out_off, uint out_capacity,
                          __global const uint* bucket_count,
                          __global const uint* bucket_slots,
                          __global const ulong* in_work,
                          __global ulong* out_work,
                          __global const uint* all_lead_in,   // same buffer as lead; read at in_lead_off
                          __global uint* all_left, __global uint* all_right, __global uint* all_lead,
                          __global uint* counters /* [0]=out_count, [2]=pair_drops */) {
    uint b = (uint)get_global_id(0);
    uint cnt = bucket_count[b];
    uint k = cnt < slots ? cnt : slots;
    for (uint i = 0; i + 1u < k; ++i) {
        uint sa = bucket_slots[(size_t)b*slots + i];
        uint ka = (uint)(in_work[(size_t)sa*7] & 0xFFFFFFu);
        for (uint j = i + 1u; j < k; ++j) {
            uint sb = bucket_slots[(size_t)b*slots + j];
            uint kb = (uint)(in_work[(size_t)sb*7] & 0xFFFFFFu);
            if (ka != kb) continue;
            uint la = lead_identity ? sa : all_lead_in[in_lead_off + sa];
            uint lb = lead_identity ? sb : all_lead_in[in_lead_off + sb];
            uint left = sa, right = sb, ll = la;
            if (lb < la || (lb == la && sb < sa)) { left = sb; right = sa; ll = lb; }
            ulong ea[7], eb[7], ec[7];
            for (int w = 0; w < 7; ++w) { ea[w] = in_work[(size_t)left*7+w]; eb[w] = in_work[(size_t)right*7+w]; }
            bh3_combine(ea, eb, Lout, ec);
            uint oi = atomic_inc(&counters[0]);
            if (oi < out_capacity) {
                for (int w = 0; w < 7; ++w) out_work[(size_t)oi*7 + w] = ec[w];
                all_left[out_off + oi] = left; all_right[out_off + oi] = right;
                all_lead[out_off + oi] = ll;
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
