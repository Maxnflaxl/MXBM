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
