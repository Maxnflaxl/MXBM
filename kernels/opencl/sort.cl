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
// Radix: 8-bit digit x 3 passes over bytes 0,1,2 of the low word => sorts the
// 24-bit key exactly. The sort is STABLE (LSD requires per-digit stability), so
// equal-key runs preserve input order -- the oracle uses std::stable_sort by key.
//
// WG (work-items per group) and the scan BLOCK are both SORT_WG. One element per
// work-item in P1 (simple + correct; P4 tiles for speed). Launches are rounded up
// to a multiple of SORT_WG and every kernel bounds-checks g < n, so any n works.

#ifndef SORT_WG
#define SORT_WG 256u
#endif
#define SORT_RADIX 256u          // 8-bit digit
#define SORT_INVALID_DIGIT 0x100u // padding lanes (g>=n): never match a real digit

inline uint sort_key(ulong pair)   { return (uint)(pair & 0xFFFFFFFFul); }
inline uint sort_index(ulong pair)  { return (uint)(pair >> 32); }

// --- radix pass, step 1: per-workgroup 256-bin histogram of one digit ----------
// Writes the group's bin counts BIN-MAJOR: wghist[bin*ngroups + wg] = count.
// Exclusive-scanning that bin-major array (scan_block/scan_add below) yields, at
// [bin*ngroups+wg], the global base for this (wg,bin): (all elements in smaller
// bins) + (this bin's elements in earlier workgroups). That is exactly the sorted
// destination base the reorder needs.
__kernel void radix_histogram(__global const ulong* pairs,
                              __global uint* wghist,
                              uint shift, uint n, uint ngroups) {
    __local uint h[SORT_RADIX];
    uint l = get_local_id(0);
    for (uint i = l; i < SORT_RADIX; i += get_local_size(0)) h[i] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    uint g = get_global_id(0);
    if (g < n) {
        uint d = (sort_key(pairs[g]) >> shift) & 0xFFu;
        atomic_inc(&h[d]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint wg = get_group_id(0);
    for (uint b = l; b < SORT_RADIX; b += get_local_size(0))
        wghist[b * ngroups + wg] = h[b];
}

// --- radix pass, step 3: stable scatter to sorted positions --------------------
// wgoffset[bin*ngroups+wg] is the exclusive-scanned base (from step 2). Each
// element's destination = base(bin) + its stable rank among same-digit elements
// earlier in THIS workgroup. Rank is an O(WG) count over the group's digits
// (mean group is tiny; correctness-first -- P4 replaces with a local scan).
__kernel void radix_reorder(__global const ulong* pairs_in,
                            __global ulong* pairs_out,
                            __global const uint* wgoffset,
                            uint shift, uint n, uint ngroups) {
    __local uint dig[SORT_WG];
    uint l = get_local_id(0);
    uint g = get_global_id(0);
    ulong pair = 0;
    uint d = SORT_INVALID_DIGIT;
    if (g < n) { pair = pairs_in[g]; d = (sort_key(pair) >> shift) & 0xFFu; }
    dig[l] = d;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (g < n) {
        uint rank = 0;
        for (uint j = 0; j < l; ++j) if (dig[j] == d) ++rank;   // stable: earlier lanes first
        uint wg = get_group_id(0);
        pairs_out[wgoffset[d * ngroups + wg] + rank] = pair;
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
