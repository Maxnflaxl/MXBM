#include "gpu/budget.h"
#include "gpu/rowbucket_geom.h"   // kRbCapacity: the ladder's own capacity figure
namespace mxbm { namespace gpu {

static uint64_t min_u64(uint64_t a, uint64_t b) { return a < b ? a : b; }

namespace {
uint64_t g_keepfree = 0;
bool     g_keepfree_given = false;
}

void set_keepfree_bytes(uint64_t bytes) { g_keepfree = bytes; g_keepfree_given = true; }
void clear_keepfree() { g_keepfree = 0; g_keepfree_given = false; }
bool keepfree_given() { return g_keepfree_given; }

uint64_t reserve_bytes(bool display_attached) {
    if (g_keepfree_given) return g_keepfree;
    return display_attached ? kReserveDisplayBytes : kReserveHeadlessBytes;
}

uint64_t usable_vram(uint64_t free_bytes, uint64_t total_bytes, uint64_t reserve) {
    if (free_bytes == 0) return (uint64_t)((double)total_bytes * kUnknownFreeHeadroom);
    const uint64_t capped = min_u64(free_bytes, total_bytes ? total_bytes : free_bytes);
    return capped > reserve ? capped - reserve : 0;
}

Budget compute_budget(uint64_t global_mem, uint64_t max_alloc, double headroom,
                      uint32_t bytes_per_element) {
    Budget b;
    b.global_mem = global_mem;
    b.max_alloc  = max_alloc;
    b.usable     = (uint64_t)((double)global_mem * headroom);
    b.bytes_per_element = bytes_per_element;
    b.target_elems = 1u << kTargetElemsLog2;

    // Per-element resident divisor (kBytesPerElement, see budget.h). This used to be
    // 5*68+56 = 396 B/elem, a leftover from the pre-reclaim six-buffer layout. That
    // over-estimate is not free: it only grants a full 2^25 seed layer above ~14.6 GiB
    // of reported VRAM, so 12 GB cards were downgraded to a REDUCED layer -- which,
    // per budget_can_find_solutions(), finds essentially no solutions at all. Sizing
    // to what the pipeline actually allocates makes those cards usable.
    uint64_t rounds_cap = b.usable / (uint64_t)bytes_per_element;
    uint64_t epr = min_u64(b.target_elems, rounds_cap);
    if (epr == 0) epr = 1;
    b.elems_per_round = (uint32_t)epr;

    // Buffer / out_capacity headroom over the seed count. Each round's genuine
    // collision count fluctuates a few thousand ABOVE elems_per_round (Poisson,
    // std ~ sqrt(2^25) ~ 5800); without headroom, out_capacity clamps that
    // excess in nondeterministic atomic order and a clamped child can be a
    // valid solution's ancestor -> ~10% of solves silently lose a solution.
    // epr/32 (~3.1%, ~1M elements at 2^25) is >100x the observed excess, and is
    // itself capped by what VRAM allows (rounds_cap) so smaller cards that only
    // just fit the seed layer keep capacity == elems_per_round (no headroom, but
    // also no worse than before). Never below epr.
    uint64_t cap = min_u64(rounds_cap, epr + epr / 32);
    if (cap < epr) cap = epr;
    b.capacity = (uint32_t)cap;

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

Budget budget_full_rowbucket(uint64_t usable_mem, uint64_t max_alloc) {
    Budget b;
    b.global_mem = usable_mem;
    b.max_alloc  = max_alloc;
    b.usable     = usable_mem;
    b.bytes_per_element = kBytesPerElementRowbucket;
    b.target_elems = 1u << kTargetElemsLog2;
    b.elems_per_round = b.target_elems;
    b.capacity = kRbCapacity;      // 2^25 + 2^25/32, what the ladder sizes against
    // The sort-path bucket fields, at exactly compute_budget()'s full-layer values
    // -- unused on the row-bucket path, but the legacy/LDS opt-in paths read them
    // and a zeroed Budget would surprise those.
    b.bucket_bits = 20u; b.num_buckets = 1u << 20;
    b.slots_per_bucket = 64u;
    uint64_t batch = max_alloc ? min_u64(b.target_elems, max_alloc / kSeedElemBytes)
                               : b.target_elems;
    if (batch == 0) batch = 1;
    b.seed_batch = (uint32_t)batch;
    return b;
}

}} // namespace mxbm::gpu
