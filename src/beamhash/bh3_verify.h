#pragma once
#include <cstdint>
#include <cstddef>
namespace mxbm { namespace bh3 {
// Mirrors BeamHash_III::IsValidSolution. Recomputes prePow from the solution's
// own extraNonce (soln[100..103]). Returns true iff soln is Equihash-valid.
bool is_valid_solution(const uint8_t* input, size_t inLen,
                       const uint8_t nonce[8], const uint8_t soln[104]);
} } // namespace mxbm::bh3
