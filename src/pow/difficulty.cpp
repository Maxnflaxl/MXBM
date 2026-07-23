#include "pow/difficulty.h"
#include "sha256/sha256.h"   // vendored: void sha256(const uint8_t* d, size_t n, uint8_t out[32]);
#include <cmath>
#include <cstring>
namespace mxbm { namespace pow {
static bool mul_lt_pow2(const uint8_t hash[32], uint32_t mantissa, unsigned bitpos) {
    // Compute hash(256-bit BE) * mantissa (<=25 bit) as a 288-bit number, compare < 2^bitpos.
    // 288-bit little-endian accumulator in 9 x 32-bit limbs.
    uint32_t H[8];
    for (int i=0;i<8;++i) H[i] = (uint32_t(hash[i*4])<<24)|(uint32_t(hash[i*4+1])<<16)|
                                 (uint32_t(hash[i*4+2])<<8)|uint32_t(hash[i*4+3]);   // H[0]=most significant
    uint32_t P[9] = {0};        // P[0]=least significant limb
    uint64_t carry=0;
    for (int i=7;i>=0;--i) {    // multiply, LSB->MSB
        uint64_t cur = (uint64_t)H[i]*mantissa + carry;
        P[7-i] = (uint32_t)cur; carry = cur>>32;
    }
    P[8] = (uint32_t)carry;
    // compare P (288-bit) < 2^bitpos
    for (int limb=8; limb>=0; --limb) {
        unsigned lo = limb*32, hi = lo+32;
        uint32_t v = P[limb];
        if (bitpos >= hi) { if (v) return true /*bits below cutoff*/; else continue; }
        if (bitpos <= lo) { if (v) return false; else continue; }
        uint32_t mask = (bitpos-lo>=32)?0xFFFFFFFFu:((1u<<(bitpos-lo))-1u);
        if (v & ~mask) return false;    // a set bit at/above cutoff
    }
    return true;
}
bool is_target_reached(const uint8_t hash[32], uint32_t packed) {
    const uint32_t s_Inf = (uint32_t)232 << 24;     // (s_MaxOrder+1)<<24, s_MaxOrder=231
    if (packed > s_Inf) return false;
    uint32_t order = packed >> 24;
    uint32_t mantissa = (1u<<24) | (packed & 0xFFFFFF);
    // clears iff hash*mantissa < 2^(256+24-order)
    return mul_lt_pow2(hash, mantissa, 280u - order);
}
bool clears_difficulty(const uint8_t soln[104], uint32_t packed) {
    uint8_t h[32]; sha256(soln, 104, h); return is_target_reached(h, packed);
}
double to_display_units(uint32_t packed) {
    // Beam Difficulty::ToFloat(): mantissa with the implicit top bit, scaled by
    // 2^(order - s_MantissaBits). Exact for power-of-two difficulties.
    uint32_t order = packed >> 24;
    uint32_t m = (1u << 24) | (packed & 0xFFFFFF);
    return std::ldexp((double)m, (int)order - 24);
}
double achieved_units(const uint8_t hash[32]) {
    int lz = 0;
    while (lz < 32 && hash[lz] == 0) ++lz;
    if (lz == 32) return HUGE_VAL;
    uint64_t top = 0;
    for (int i = 0; i < 8; ++i)
        top = (top << 8) | (lz + i < 32 ? (uint64_t)hash[lz + i] : 0u);
    // hash ~= top * 2^(8*(24-lz)); units = 2^256/hash = (2^64/top) * 2^(24 - 8*(24-lz) ... )
    // exponent: 256 - 64 - 8*(32 - lz - 8)
    return std::ldexp(std::ldexp(1.0, 64) / (double)top, 256 - 64 - 8 * (32 - lz - 8));
}
} } // namespace mxbm::pow
