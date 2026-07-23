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
