#include "check.h"
#include "pow/difficulty.h"
#include "sha256/sha256.h"
#include <cmath>
#include <cstring>
using namespace mxbm; using namespace mxbm::pow;

// ---------------------------------------------------------------------------
// Beam parity vectors (PV1..PV9 below) were NOT derived from this port. They
// come from an independent arbitrary-precision oracle (python3, see
// .superpowers/sdd/task-3-report.md for the script) implementing Beam's exact
// predicate as specified in core/difficulty.h/.cpp:
//   order    = packed >> 24
//   mantissa = (1 << 24) | (packed & 0xFFFFFF)
//   s_Inf    = 232 << 24                      # core/difficulty.h:33-34
//   if packed > s_Inf: return False           # core/difficulty.cpp:101-102 (strict '>')
//   bitpos   = 256 + 24 - order = 280 - order  # core/difficulty.cpp:108-109, Number::nBits=256
//   return int.from_bytes(hash32, 'big') * mantissa < (1 << bitpos)   # strict '<', numeric_utils.h:295-313
// hash bytes are big-endian (core/uintBig.h:179, uintBig.cpp _ToNum byte-swap-to-host).
// Any failure below means the C++ port (or this comment's oracle) is wrong -- investigate,
// do not adjust a vector to force green.
// ---------------------------------------------------------------------------

int main() {
    // -- starter checks (given) --
    uint8_t zero[32]; std::memset(zero,0,32);
    check(is_target_reached(zero, (8u<<24)|0), "hash 0 always clears");   // 0*mantissa < 2^n
    uint8_t big[32]; std::memset(big,0xFF,32);
    check(!is_target_reached(big, (200u<<24)|0xFFFFFF), "max hash, hard target -> false");

    // -- PV1: product == 2^bitpos EXACTLY -> must NOT clear (strict '<' boundary) --
    // order=24, mantissa=2^24 exactly (packed low24 bits=0), bitpos=280-24=256;
    // hash = 2^256 / mantissa = 2^232, so hash*mantissa == 2^bitpos exactly.
    {
        static const uint8_t h[32] = {
            0x00,0x00,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
            0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 };
        check(!is_target_reached(h, 0x18000000u), "PV1 product==2^bitpos exactly -> not clear (boundary)");
    }

    // -- PV2: product == 2^bitpos - 1 EXACTLY -> must clear (strict '<' boundary) --
    // order=24, mantissa=257*65537=16843009=0x1010101 (an exact divisor of 2^256-1 via the
    // difference-of-squares chain (2^16-1)(2^16+1)(2^32+1)(2^64+1)(2^128+1), since 257 divides
    // 2^16-1 and 65537=2^16+1); hash=(2^256-1)/mantissa so hash*mantissa == 2^bitpos-1 exactly.
    {
        static const uint8_t h[32] = {
            0x00,0x00,0x00,0xff,0x00,0x00,0x00,0xff,0x00,0x00,0x00,0xff,0x00,0x00,0x00,0xff,
            0x00,0x00,0x00,0xff,0x00,0x00,0x00,0xff,0x00,0x00,0x00,0xff,0x00,0x00,0x00,0xff };
        check(is_target_reached(h, 0x18010101u), "PV2 product==2^bitpos-1 exactly -> clears (boundary)");
    }

    // -- PV3: small order extreme (order=8), small non-trivial hash -> clears --
    // At order=8 the target is extremely generous (bitpos=272); a 32-bit-magnitude hash is
    // far below it, unlike an "always false" stub this cannot pass by coincidence.
    {
        static const uint8_t h[32] = {
            0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
            0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xde,0xad,0xbe,0xef };
        check(is_target_reached(h, 0x08123456u), "PV3 order=8 (small extreme), small hash -> clears");
    }

    // -- PV4: max legal order (Beam s_MaxOrder = Raw::nBits - s_MantissaBits - 1 = 256-24-1 = 231) --
    {
        static const uint8_t h[32] = {
            0x01,0x0e,0x1b,0x28,0x35,0x42,0x4f,0x5c,0x69,0x76,0x83,0x90,0x9d,0xaa,0xb7,0xc4,
            0xd1,0xde,0xeb,0xf8,0x05,0x12,0x1f,0x2c,0x39,0x46,0x53,0x60,0x6d,0x7a,0x87,0x94 };
        check(!is_target_reached(h, 0xe70f0f0fu), "PV4 order=231 (s_MaxOrder), typical hash -> not clear");
    }

    // -- PV5: mid-range realistic pool difficulty -- 50331701=0x03000035, the `difficulty`
    // field of tests/vectors/stratum_wire.h's wire::kJob fixture. hash = SHA-256(104 zero
    // bytes); reused below both for the SHA-256 KAT and the clears_difficulty composition check.
    static const uint8_t zero104[104] = {0};
    static const uint8_t hash_zero104[32] = {
        0x39,0xf3,0x7f,0x8d,0x19,0x31,0xb3,0xbd,0xf7,0x67,0xe7,0x51,0x0d,0xd6,0x95,0x09,
        0xfb,0xf2,0x3a,0xf1,0xf7,0x65,0x49,0x33,0xd0,0xa4,0xd2,0x91,0xcb,0xdd,0x44,0x18 };
    check(!is_target_reached(hash_zero104, 50331701u), "PV5 mid-range wire-doc difficulty 50331701 -> not clear");

    // -- PV6/PV7: packed > s_Inf -> false regardless of hash --
    check(!is_target_reached(zero, 233u<<24),
          "PV6 packed=233<<24 (>s_Inf), zero hash would trivially clear otherwise -> false regardless");
    {
        uint8_t hFF[32]; std::memset(hFF, 0xFF, 32);
        check(!is_target_reached(hFF, 0xFFFFFFFFu), "PV7 packed=0xFFFFFFFF (>s_Inf), max hash -> false");
    }

    // -- PV8/PV9: random hash + random legal packed (python random.Random(1337) draws #0,#1) --
    {
        static const uint8_t h[32] = {
            0x54,0xaa,0xc4,0xb8,0x9d,0xc8,0x68,0xba,0x37,0xd9,0xcc,0x21,0xb2,0xce,0xce,0x9f,
            0x09,0xb4,0x3c,0xeb,0x7e,0x57,0xa0,0xea,0x87,0x66,0x22,0x16,0x24,0xd0,0x1b,0x08 };
        check(!is_target_reached(h, 0x9ebb5079u), "PV8 random vector #0 (order=158, mantissa=0xbb5079)");
    }
    {
        static const uint8_t h[32] = {
            0x64,0x35,0x91,0x64,0xe7,0xa0,0x06,0xaa,0xdd,0x75,0x17,0x9d,0x6d,0x5c,0x5e,0x19,
            0xfd,0xe9,0x0c,0xf9,0xb4,0x83,0x86,0x22,0x42,0x1e,0x57,0xa1,0x28,0x62,0xe1,0x81 };
        check(!is_target_reached(h, 0xa264c58fu), "PV9 random vector #1 (order=162, mantissa=0x64c58f)");
    }

    // -- clears_difficulty end-to-end composition check --
    // Same 104-byte buffer and packed value as PV5: confirm clears_difficulty(soln, packed)
    // -- which internally does SHA-256(soln) then is_target_reached -- agrees with calling
    // is_target_reached directly on the independently-known SHA-256(soln). This verifies the
    // *composition* (hash-then-compare), not just the two parts in isolation.
    check(clears_difficulty(zero104, 50331701u) == is_target_reached(hash_zero104, 50331701u),
          "clears_difficulty composes sha256(soln) + is_target_reached correctly");

    // ---- SHA-256 KATs (FIPS 180-4 known-answer tests, verified against python3 hashlib) ----
    {
        uint8_t out[32];
        sha256(reinterpret_cast<const uint8_t*>("abc"), 3, out);
        static const uint8_t want[32] = {
            0xba,0x78,0x16,0xbf,0x8f,0x01,0xcf,0xea,0x41,0x41,0x40,0xde,0x5d,0xae,0x22,0x23,
            0xb0,0x03,0x61,0xa3,0x96,0x17,0x7a,0x9c,0xb4,0x10,0xff,0x61,0xf2,0x00,0x15,0xad };
        check_eq_bytes(out, want, 32, "sha256(\"abc\") KAT");
    }
    {
        uint8_t out[32];
        sha256(reinterpret_cast<const uint8_t*>(""), 0, out);
        static const uint8_t want[32] = {
            0xe3,0xb0,0xc4,0x42,0x98,0xfc,0x1c,0x14,0x9a,0xfb,0xf4,0xc8,0x99,0x6f,0xb9,0x24,
            0x27,0xae,0x41,0xe4,0x64,0x9b,0x93,0x4c,0xa4,0x95,0x99,0x1b,0x78,0x52,0xb8,0x55 };
        check_eq_bytes(out, want, 32, "sha256(\"\") KAT");
    }
    {
        // Pins the exact 104-byte input-length path that clears_difficulty() drives sha256()
        // through; expected value is hash_zero104 above (also cross-checked in PV5).
        uint8_t out[32];
        sha256(zero104, 104, out);
        check_eq_bytes(out, hash_zero104, 32, "sha256(104 zero bytes) KAT");
    }

    // ---- to_display_units (Beam Difficulty::ToFloat port) ----
    // 9<<24: order 9, zero mantissa -> exactly 2^9 = 512 units (HeroMiners'
    // vardiff floor; miners display "Difficulty: 512" for the same jobs).
    check(to_display_units(9u << 24) == 512.0, "units: 9<<24 -> 512");
    check(to_display_units(0u) == 1.0, "units: packed 0 -> 1");
    // Wire-doc example 50331701 = 0x03000035: 16777269 * 2^-21 = 8.0000252...
    {
        double u = to_display_units(50331701u);
        check(u > 8.0000252 && u < 8.0000254, "units: 0x03000035 -> ~8.000025");
    }

    // ---- achieved_units KATs (python-derived; see script + full derivation
    // log in the task-2 report) -- 2^256/hash in Beam display units. ----
    {
        // hash = 2^255 (0x80 then 31 zero bytes) -> exactly 2^256/2^255 = 2.0.
        static const uint8_t h[32] = {
            0x80,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
            0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 };
        check(achieved_units(h) == 2.0, "achieved_units: 2^255 -> exactly 2.0");
    }
    {
        // hash = 2^224 (3 zero bytes, 0x01, 28 zero bytes) -> exactly
        // 2^256/2^224 = 2^32 = 4294967296.0.
        static const uint8_t h[32] = {
            0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
            0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 };
        check(achieved_units(h) == 4294967296.0, "achieved_units: 2^224 -> exactly 2^32");
    }
    {
        // all-0xFF hash (= 2^256-1): 2^256/(2^256-1) is within 2^-256 of 1.0
        // -- far tighter than a double's ~2^-52 resolution near 1.0 -- so
        // that exact ratio rounds to precisely 1.0 in double precision;
        // `want` below is that value, not an approximation of it. (The port
        // itself also lands on exactly 1.0 here for a related but distinct
        // reason: it only reads the leading 64 significant bits (top), and
        // for an all-0xFF hash top=2^64-1 as a uint64_t -- which itself
        // rounds to exactly 2^64 on conversion to double, since doubles are
        // ~4096-spaced at that magnitude -- making the computed ratio
        // 2^64/2^64 exactly. Verified in python3, see derivation script.)
        // The tolerance check is kept per spec rather than hard equality,
        // since that exact-1.0 landing is an implementation detail, not a
        // guarantee this KAT should encode.
        uint8_t all_ff[32];
        std::memset(all_ff, 0xFF, 32);
        double got = achieved_units(all_ff);
        double want = 1.0;   // 2^256/(2^256-1), correctly rounded to double precision
        check(std::fabs(got - want) < 1e-9,
              "achieved_units: all-0xFF -> ~1.0, within 1e-9 of 2^256/(2^256-1)");
    }
    {
        // Fixed pseudorandom 32-byte vector (python3 random.Random(20260722)
        // -- seed = today's date per CLAUDE.md currentDate -- draw #0, see
        // /private/tmp/claude-501/-Users-manuel-Developer-Github-maxnflaxl/
        // b28224a0-2487-4305-b6b5-ed602e0a2e6b/scratchpad/achieved_units_kat.py).
        // Expected = (2**256)/int.from_bytes(h,'big') computed with python
        // arbitrary-precision ints (correctly-rounded int/int true division);
        // the script also replicates the C++ top-64-bit approximation and
        // confirms (via assert) it lands within 1e-9 relative of this value
        // before the vector was trusted -- here it landed exactly equal to
        // full double precision (relative error 0.0 in the derivation run).
        static const uint8_t h[32] = {
            0xcd,0x22,0xcc,0x1b,0x7d,0x3a,0x28,0xd6,0x7d,0xd2,0x8d,0x45,0x91,0xbe,0xbe,0x5b,
            0x16,0x8d,0x4a,0xb8,0xc2,0xe7,0xcf,0xea,0xed,0xd0,0x65,0x3a,0xad,0x77,0xad,0x7c,
        };
        double got = achieved_units(h);
        double want = 1.2479530223174709;   // repr()'d python double, see derivation script
        check(std::fabs(got - want) < 1e-9 * want,
              "achieved_units: pseudorandom vector matches python (2**256)/int(h) within 1e-9 relative");
    }

    return summary("difficulty");
}
