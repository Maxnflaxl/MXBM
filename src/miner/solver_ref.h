#pragma once
#include "miner/solver.h"

namespace mxbm { namespace miner {

// Reference solver: wraps Beam's own BeamHash_III::OptimisedSolve (see
// BEAM_SOURCE_DIR/3rdparty/crypto/beamHashIII_impl.cpp) via the same call
// pattern tests/test_oracle_spike.cpp and tests/test_differential.cpp use --
// InitialiseState, blake2b_update(input, 32), blake2b_update(nonce, 8),
// then OptimisedSolve with a collector callback that always returns false
// so the search runs to completion for this one nonce and every reported
// solution is captured, not just the first.
//
// CPU-only and unabridged: one solve() call is the full BeamHash III search
// for a single nonce, on the order of minutes. Compiled (and mxbm_miner
// linked against beam_oracle) only when MXBM_HAVE_BEAM_ORACLE is set -- see
// cmake/beam_oracle.cmake. Not exercised by ctest (too slow for the suite);
// live mining runs are its coverage.
class SolverRef : public Solver {
public:
    std::vector<std::array<uint8_t, 104>> solve(const uint8_t input[32], const uint8_t nonce[8]) override;
};

} } // namespace mxbm::miner
