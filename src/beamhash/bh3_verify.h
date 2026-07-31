#pragma once
#include <cstdint>
#include <cstddef>
namespace mxbm { namespace bh3 {
// Mirrors BeamHash_III::IsValidSolution. Recomputes prePow from the solution's
// own extraNonce (soln[100..103]). Returns true iff soln is Equihash-valid.
bool is_valid_solution(const uint8_t* input, size_t inLen,
                       const uint8_t nonce[8], const uint8_t soln[104]);

// Diagnostic companion: the SAME checks in the SAME order, but reporting which
// one failed and at which round, so the found-vs-verified gap (~2.29 survivors
// against ~1.99 verified per solve) can be attributed instead of assumed.
// kind == None means valid; is_valid_solution() is defined as exactly that.
struct VerifyReason {
    enum Kind : uint8_t { None = 0, Collision, DupIndex, Order, NonZero };
    Kind    kind;
    uint8_t round;   // 1..5 for the per-round checks; 0 for None / NonZero
};
VerifyReason classify_solution(const uint8_t* input, size_t inLen,
                               const uint8_t nonce[8], const uint8_t soln[104]);

// MXBM_VERIFY_STATS=1 instrumentation: process-wide tallies over a run, printed
// to stderr at exit. The solvers call note_solve() once per solve() and tally()
// once per candidate when the env var is set; zero cost otherwise.
void verify_stats_note_solve();
void verify_stats_tally(const VerifyReason& r);
} } // namespace mxbm::bh3
