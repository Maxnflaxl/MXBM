#pragma once
#include <array>
#include <cstdint>
#include <vector>

namespace mxbm { namespace miner {

// Implemented by SolverRef (Beam's reference OptimisedSolve, CPU, on the
// order of minutes per call -- see solver_ref.h) and, in M3, by a GPU
// solver. Given a 32-byte block input and an 8-byte nonce, returns every
// 104-byte BeamHash III candidate solution found for that exact (input,
// nonce) pair: zero, one, or many -- the algorithm's last round can report
// several colliding pairs before the search for that one nonce is
// exhausted, and Engine checks each candidate against difficulty itself.
struct Solver {
    virtual std::vector<std::array<uint8_t, 104>> solve(const uint8_t input[32], const uint8_t nonce[8]) = 0;
    // Ask an in-flight (or future) solve() to return early. Default no-op:
    // SolverRef's CPU search cannot be interrupted mid-call and inherits
    // this as-is. GpuSolver (src/gpu/gpu_solver.h) overrides it to flip an
    // std::atomic<bool> that run_pipeline polls between rounds.
    virtual void request_abort() {}
    virtual ~Solver() = default;
};

} } // namespace mxbm::miner
