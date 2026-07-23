#include "bh3_blake2b.h"
#include <cstring>
extern "C" {
#include "blake2.h"
}
namespace mxbm { namespace bh3 {
void compute_prepow(const uint8_t* input, size_t inLen,
                    const uint8_t nonce[8], const uint8_t extraNonce[4],
                    uint64_t prePow[4]) {
    static const uint8_t personal[16] = {
        'B','e','a','m','-','P','o','W',
        0xc0,0x01,0x00,0x00,   // LE32(448)
        0x05,0x00,0x00,0x00 }; // LE32(5)
    blake2b_param P;
    std::memset(&P, 0, sizeof(P));
    P.digest_length = 32;
    P.fanout = 1;
    P.depth = 1;
    std::memcpy(P.personal, personal, 16);
    blake2b_state S;
    blake2b_init_param(&S, &P);
    blake2b_update(&S, input, inLen);
    blake2b_update(&S, nonce, 8);
    blake2b_update(&S, extraNonce, 4);
    uint8_t out[32];
    blake2b_final(&S, out, 32);
    for (int i = 0; i < 4; ++i) {
        uint64_t v = 0;
        for (int j = 0; j < 8; ++j) v |= (uint64_t)out[i*8 + j] << (8*j);
        prePow[i] = v;
    }
}
} } // namespace mxbm::bh3
