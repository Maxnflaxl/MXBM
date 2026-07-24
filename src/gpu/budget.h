#pragma once
#include <cstdint>
namespace mxbm { namespace gpu {

constexpr uint32_t kNumRounds        = 5;
constexpr uint32_t kSeedElemBytes     = 56;   // 7 x u64
constexpr uint32_t kResidentElemBytes = 68;   // 56 work + 12 back-ref
constexpr uint32_t kTargetElemsLog2   = 25;

// Per-element VRAM cost of the resident pipeline, used to decide how large a seed
// layer a device can host. Must cover the LARGER of the two collision paths, since
// which one alloc_pipeline picks is decided later (see want_rowbucket):
//   row-bucket : packed record x2 sets (1.25x bucket slack) + left/right  ~= 268 B
//   sort       : work x2 + leaves x2 + left/right/lead + sort scratch     ~= 285 B
// 304 B adds ~7% margin over the sort path for bucket_slots and rounding.
// tests/test_gpu_rounds.cpp pins this against the real allocation, so the two
// cannot drift apart silently.
constexpr uint32_t kBytesPerElement = 304;

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

Budget compute_budget(uint64_t global_mem, uint64_t max_alloc, double headroom = 0.85);

}} // namespace mxbm::gpu
