#pragma once
#include <cstdint>
#include <cstddef>
namespace mxbm { namespace bh3 {
// Personalized Blake2b: "Beam-PoW"|LE32(448)|LE32(5), digest 32, fanout/depth 1,
// over input||nonce||extraNonce; output as 4 little-endian u64.
void compute_prepow(const uint8_t* input, size_t inLen,
                    const uint8_t nonce[8], const uint8_t extraNonce[4],
                    uint64_t prePow[4]);
} } // namespace mxbm::bh3
