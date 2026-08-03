#pragma once
#include <cstdint>
namespace mxbm { namespace pow {
// Ported from Beam core/difficulty.cpp IsTargetReached (Apache-2.0). hash is the
// 32-byte value big-endian; packed is Difficulty::m_Packed (order<<24 | mantissa24).
bool is_target_reached(const uint8_t hash[32], uint32_t packed);
bool clears_difficulty(const uint8_t soln[104], uint32_t packed);   // SHA256(soln) then target test
// Human-readable difficulty units, ported from Beam Difficulty::ToFloat():
// ((1<<24)|mantissa) * 2^(order-24). E.g. packed 9<<24 -> 512 (what miners
// and pool UIs display).
double to_display_units(uint32_t packed);
// Achieved difficulty of a share hash, in the same display units as
// to_display_units(): 2^256 / hash (a share "of difficulty d" clears any
// target <= d; see is_target_reached). hash is 32 bytes, big-endian. An
// all-zero hash returns HUGE_VAL. Approximated from the leading 64
// significant bits of hash (see difficulty.cpp) -- exact to within a
// relative error on the order of 2^-64.
double achieved_units(const uint8_t hash[32]);
} } // namespace mxbm::pow
