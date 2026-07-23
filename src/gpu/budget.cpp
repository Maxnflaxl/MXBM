#include "gpu/budget.h"
namespace mxbm { namespace gpu {

static uint64_t min_u64(uint64_t a, uint64_t b) { return a < b ? a : b; }

Budget compute_budget(uint64_t global_mem, uint64_t max_alloc, double headroom) {
    Budget b;
    b.global_mem = global_mem;
    b.max_alloc  = max_alloc;
    b.usable     = (uint64_t)((double)global_mem * headroom);
    b.target_elems = 1u << kTargetElemsLog2;

    uint64_t rounds_cap = b.usable / ((uint64_t)kNumRounds * kResidentElemBytes);
    uint64_t epr = min_u64(b.target_elems, rounds_cap);
    if (epr == 0) epr = 1;
    b.elems_per_round = (uint32_t)epr;

    // Small buckets keep the sortless all-pairs match cheap: mean ~32/bucket.
    uint32_t lg = 0; while ((1ull << (lg+1)) <= b.elems_per_round) ++lg;   // floor(log2)
    b.bucket_bits = lg >= 15 ? (lg - 5) : 10;         // clamp low
    if (b.bucket_bits > 20) b.bucket_bits = 20;       // clamp high
    b.num_buckets = 1u << b.bucket_bits;

    uint64_t mean = (uint64_t)b.elems_per_round / b.num_buckets;
    uint64_t spb  = (mean * 14 + 9) / 10;           // 1.4x mean, ceil
    if (spb < 64) spb = 64;
    b.slots_per_bucket = (uint32_t)spb;

    uint64_t batch = min_u64(b.target_elems, max_alloc / kSeedElemBytes);
    batch = min_u64(batch, b.elems_per_round);
    if (batch == 0) batch = 1;
    b.seed_batch = (uint32_t)batch;
    return b;
}

}} // namespace mxbm::gpu
