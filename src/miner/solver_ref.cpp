#include "miner/solver_ref.h"

#include <algorithm>

#include "beamHashIII.h"   // BEAM_SOURCE_DIR/3rdparty/crypto -- see cmake/beam_oracle.cmake

namespace mxbm { namespace miner {

std::vector<std::array<uint8_t, 104>> SolverRef::solve(const uint8_t input[32], const uint8_t nonce[8]) {
    std::vector<std::array<uint8_t, 104>> out;

    BeamHash_III bh;
    blake2b_state base;
    bh.InitialiseState(base);
    blake2b_update(&base, input, 32);
    blake2b_update(&base, nonce, 8);

    // Collector: OptimisedSolve stops early (and returns true) the moment
    // this returns true, so returning false unconditionally makes it
    // enumerate every solution found for this (input, nonce) pair before
    // it exhausts the search and returns false itself.
    auto collect = [&out](const std::vector<uint8_t>& soln) -> bool {
        if (soln.size() == 104) {   // can't-happen for BeamHash III; skip defensively, don't crash
            std::array<uint8_t, 104> cand{};
            std::copy(soln.begin(), soln.end(), cand.begin());
            out.push_back(cand);
        }
        return false;
    };
    auto never_cancel = [](SolverCancelCheck) { return false; };

    bh.OptimisedSolve(base, collect, never_cancel);

    return out;
}

} } // namespace mxbm::miner
