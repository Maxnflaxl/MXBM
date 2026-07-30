#pragma once
#include "bh3_types.h"
namespace mxbm { namespace bh3 {

MXBM_HD inline uint64_t rotl64(uint64_t x, int b) { return (x << b) | (x >> (64 - b)); }

// Non-standard SipHash-2-4: direct state init, no magic constants, no length block.
MXBM_HD inline uint64_t siphash24(uint64_t k0, uint64_t k1, uint64_t k2, uint64_t k3, uint64_t nonce) {
    uint64_t v0 = k0, v1 = k1, v2 = k2, v3 = k3;
    v3 ^= nonce;
    for (int r = 0; r < 2; ++r) {
        v0 += v1; v2 += v3; v1 = rotl64(v1,13); v3 = rotl64(v3,16);
        v1 ^= v0; v3 ^= v2; v0 = rotl64(v0,32);
        v2 += v1; v0 += v3; v1 = rotl64(v1,17); v3 = rotl64(v3,21);
        v1 ^= v2; v3 ^= v0; v2 = rotl64(v2,32);
    }
    v0 ^= nonce; v2 ^= 0xffULL;
    for (int r = 0; r < 4; ++r) {
        v0 += v1; v2 += v3; v1 = rotl64(v1,13); v3 = rotl64(v3,16);
        v1 ^= v0; v3 ^= v2; v0 = rotl64(v0,32);
        v2 += v1; v0 += v3; v1 = rotl64(v1,17); v3 = rotl64(v3,21);
        v1 ^= v2; v3 ^= v0; v2 = rotl64(v2,32);
    }
    return v0 ^ v1 ^ v2 ^ v3;
}

MXBM_HD inline void seed_element(const MXBM_THREAD uint64_t* prePow, uint32_t index, MXBM_THREAD Elem& out) {
    for (int k = 0; k < kWorkWords; ++k)
        out.w[k] = siphash24(prePow[0], prePow[1], prePow[2], prePow[3],
                             ((uint64_t)index << 3) + (uint64_t)k);
}

MXBM_HD inline uint32_t collision_bits(const MXBM_THREAD Elem& e) { return (uint32_t)(e.w[0] & 0xFFFFFFu); }

MXBM_HD inline void apply_mix(MXBM_THREAD Elem& e, const MXBM_THREAD uint32_t* tree, uint32_t treeLen, uint32_t Lmix) {
    uint64_t t[8];
    for (int i = 0; i < kWorkWords; ++i) t[i] = e.w[i];
    t[7] = 0;
    uint32_t padNum = ((512u - Lmix) + 24u) / 25u;
    if (padNum > treeLen) padNum = treeLen;
    for (uint32_t i = 0; i < padNum; ++i) {
        uint32_t pos = Lmix + i * 25u;
        if (pos >= 512u) break;
        uint64_t v   = (uint64_t)tree[i];
        uint32_t word = pos >> 6;
        uint32_t sh   = pos & 63u;
        t[word] |= v << sh;
        if (sh > 39u && word < 7u) t[word + 1] |= v >> (64u - sh);
    }
    const int ROT[8] = {29,58,23,52,17,46,11,40};
    uint64_t result = 0;
    for (int i = 0; i < 8; ++i) result += rotl64(t[i], ROT[i]);
    result = rotl64(result, 24);
    e.w[0] = result;
}

MXBM_HD inline void combine(const MXBM_THREAD Elem& a, const MXBM_THREAD Elem& b, uint32_t Lout, MXBM_THREAD Elem& out) {
    uint64_t x[kWorkWords];
    for (int i = 0; i < kWorkWords; ++i) x[i] = a.w[i] ^ b.w[i];
    for (int i = 0; i < kWorkWords; ++i)
        out.w[i] = (x[i] >> 24) | (i < kWorkWords - 1 ? (x[i + 1] << 40) : 0);
    for (int i = 0; i < kWorkWords; ++i) {
        uint32_t base = 64u * (uint32_t)i;
        if (base >= Lout) out.w[i] = 0;
        else if (Lout - base < 64u) out.w[i] &= (((uint64_t)1 << (Lout - base)) - 1);
    }
}

MXBM_HD inline void pack_indices(const MXBM_THREAD uint32_t* idx, MXBM_THREAD uint8_t* out) {
    for (int i = 0; i < 100; ++i) out[i] = 0;
    for (int k = 0; k < kNumIndices; ++k)
        for (int b = 0; b < kIndexPackBits; ++b)
            if ((idx[k] >> b) & 1u) {
                int pos = kIndexPackBits * k + b;
                out[pos >> 3] |= (uint8_t)(1u << (pos & 7));
            }
}

MXBM_HD inline void unpack_indices(const MXBM_THREAD uint8_t* soln, MXBM_THREAD uint32_t* idx) {
    for (int k = 0; k < kNumIndices; ++k) {
        uint32_t v = 0;
        for (int b = 0; b < kIndexPackBits; ++b) {
            int pos = kIndexPackBits * k + b;
            v |= (uint32_t)((soln[pos >> 3] >> (pos & 7)) & 1u) << b;
        }
        idx[k] = v;
    }
}

} } // namespace mxbm::bh3
