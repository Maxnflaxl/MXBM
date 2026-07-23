// BeamHash III primitives — OpenCL C. Authored from src/beamhash/bh3_primitives.h.
// Bit-identical to the C++ primitives; pinned by tests/test_cl_primitives.cpp.

inline ulong bh3_rotl64(ulong x, int b) { return (x << b) | (x >> (64 - b)); }

// Non-standard SipHash-2-4: direct state init, no magic constants, no length block.
inline ulong bh3_siphash24(ulong k0, ulong k1, ulong k2, ulong k3, ulong nonce) {
    ulong v0 = k0, v1 = k1, v2 = k2, v3 = k3;
    v3 ^= nonce;
    for (int r = 0; r < 2; ++r) {
        v0 += v1; v2 += v3; v1 = bh3_rotl64(v1,13); v3 = bh3_rotl64(v3,16);
        v1 ^= v0; v3 ^= v2; v0 = bh3_rotl64(v0,32);
        v2 += v1; v0 += v3; v1 = bh3_rotl64(v1,17); v3 = bh3_rotl64(v3,21);
        v1 ^= v2; v3 ^= v0; v2 = bh3_rotl64(v2,32);
    }
    v0 ^= nonce; v2 ^= 0xffUL;
    for (int r = 0; r < 4; ++r) {
        v0 += v1; v2 += v3; v1 = bh3_rotl64(v1,13); v3 = bh3_rotl64(v3,16);
        v1 ^= v0; v3 ^= v2; v0 = bh3_rotl64(v0,32);
        v2 += v1; v0 += v3; v1 = bh3_rotl64(v1,17); v3 = bh3_rotl64(v3,21);
        v1 ^= v2; v3 ^= v0; v2 = bh3_rotl64(v2,32);
    }
    return v0 ^ v1 ^ v2 ^ v3;
}

inline void bh3_seed_element(const ulong pp[4], uint index, ulong out[7]) {
    for (int k = 0; k < 7; ++k)
        out[k] = bh3_siphash24(pp[0], pp[1], pp[2], pp[3], ((ulong)index << 3) + (ulong)k);
}

inline uint bh3_collision_bits(ulong w0) { return (uint)(w0 & 0xFFFFFFu); }

// apply_mix: fold the element's index tree into w[0]. Writes only w[0].
// Mirrors bh3_primitives.h::apply_mix exactly.
inline ulong bh3_apply_mix(const ulong e[7], const uint* tree, uint treeLen, uint Lmix) {
    ulong t[8];
    for (int i = 0; i < 7; ++i) t[i] = e[i];
    t[7] = 0;
    uint padNum = ((512u - Lmix) + 24u) / 25u;
    if (padNum > treeLen) padNum = treeLen;
    for (uint i = 0; i < padNum; ++i) {
        uint pos = Lmix + i * 25u;
        if (pos >= 512u) break;
        ulong v = (ulong)tree[i];
        uint word = pos >> 6;
        uint sh   = pos & 63u;
        t[word] |= v << sh;
        if (sh > 39u && word < 7u) t[word + 1] |= v >> (64u - sh);
    }
    const int ROT[8] = {29,58,23,52,17,46,11,40};
    ulong result = 0;
    for (int i = 0; i < 8; ++i) result += bh3_rotl64(t[i], ROT[i]);
    result = bh3_rotl64(result, 24);
    return result;
}

// combine: XOR, shift down 24 (drop the collided lane), mask to Lout bits.
inline void bh3_combine(const ulong a[7], const ulong b[7], uint Lout, ulong out[7]) {
    ulong x[7];
    for (int i = 0; i < 7; ++i) x[i] = a[i] ^ b[i];
    for (int i = 0; i < 7; ++i)
        out[i] = (x[i] >> 24) | (i < 6 ? (x[i + 1] << 40) : 0);
    for (int i = 0; i < 7; ++i) {
        uint base = 64u * (uint)i;
        if (base >= Lout) out[i] = 0;
        else if (Lout - base < 64u) out[i] &= (((ulong)1 << (Lout - base)) - 1);
    }
}
