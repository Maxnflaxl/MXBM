#pragma once
#include <cstdint>
namespace mxbm { namespace gpu {

constexpr uint32_t kNumRounds        = 5;
constexpr uint32_t kSeedElemBytes     = 56;   // 7 x u64
constexpr uint32_t kResidentElemBytes = 68;   // 56 work + 12 back-ref
constexpr uint32_t kTargetElemsLog2   = 25;

// Per-element VRAM cost of the resident pipeline, used to decide how large a seed
// layer a device can host. The two collision paths cost very different amounts, and
// using one figure for both means every card is sized for the hungrier one:
//   row-bucket : packed records x2 sets (per-set stride) + left/right   = 239 B
//   sort       : work x2 + leaves x2 + left/right + sort scratch        = 264 B
// Both are measured, and tests/test_gpu_rounds.cpp pins them against the real
// allocation so they cannot drift silently. ~7% is added for bucket_slots and
// rounding.
//
// Which path runs is decided by rowbucket_viable(), which needs a capacity, which
// needs one of these -- so compute_budget() is called twice: once optimistically
// for the row-bucket path, and again for the sort path if that is rejected. Sizing
// with the wrong one is not safe in either direction: too small and the sort path
// OOMs, too large and cards that could host a full seed layer are cut down to a
// reduced one, which per budget_can_find_solutions() mines nothing at all.
constexpr uint32_t kBytesPerElementRowbucket = 256;
constexpr uint32_t kBytesPerElementSort      = 284;
constexpr uint32_t kBytesPerElement          = kBytesPerElementSort;   // safe default

// A BeamHash III solution is 32 indices drawn from the FULL [0, 2^25) space, so a
// solver that generates only [0, epr) can find a solution only if all 32 of its
// indices land below epr -- probability (epr / 2^25)^32. At epr = 2^24 that is
// 2e-10: a reduced seed layer does not mine slowly, it mines nothing. Callers must
// treat elems_per_round < target_elems as non-functional rather than degraded.
inline bool budget_can_find_solutions(uint32_t elems_per_round) {
    return elems_per_round >= (1u << kTargetElemsLog2);
}

struct Budget {
    uint64_t global_mem = 0;
    uint64_t max_alloc  = 0;
    uint64_t usable     = 0;
    uint32_t bytes_per_element = 0;  // the divisor this budget was sized with, so
                                     // consumers can check against the right figure
    uint32_t target_elems     = 0;
    uint32_t elems_per_round  = 0;   // seed count / round-1 input N (2^25 on a full-search card)
    uint32_t capacity         = 0;   // BUFFER + out_capacity size = elems_per_round + headroom.
                                     // A round's genuine collision count fluctuates a few thousand
                                     // ABOVE the 2^25 seed count (Poisson ~sqrt(2^25) std); sizing
                                     // buffers to exactly elems_per_round forces out_capacity to
                                     // clamp that excess in nondeterministic atomic order, silently
                                     // dropping ~10% of solves' solutions. capacity gives headroom so
                                     // pairDrops == 0 and the search is deterministic. 0 => callers
                                     // (synthetic tests) fall back to elems_per_round, see alloc_pipeline.
    uint32_t bucket_bits      = 0;
    uint32_t num_buckets      = 0;
    uint32_t slots_per_bucket = 0;
    uint32_t seed_batch       = 0;
};

Budget compute_budget(uint64_t global_mem, uint64_t max_alloc, double headroom = 0.85,
                      uint32_t bytes_per_element = kBytesPerElement);

// The full-2^25-layer budget for the row-bucket path, where the GEOMETRY LADDER
// (rowbucket_viable), not the flat divisor above, is the authority. Callers must
// ask rowbucket_viable() whether it stands -- using it unchecked would allocate
// an unfittable pipeline.
Budget budget_full_rowbucket(uint64_t global_mem, uint64_t max_alloc);

}} // namespace mxbm::gpu
