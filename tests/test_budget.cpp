#include "gpu/budget.h"
#include "check.h"
#include <cstdint>
using namespace mxbm;
using namespace mxbm::gpu;

int main() {
    section("budget sizing");
    const uint64_t GiB = 1024ull*1024*1024;

    // 16 GB NVIDIA-like: max_alloc = 1/4 global (a common driver cap).
    Budget b16 = compute_budget(16*GiB, 4*GiB);
    check(b16.elems_per_round == (1u<<25), "16GB keeps full 2^25 elems/round");
    check(b16.seed_batch*56ull <= 4*GiB, "16GB seed_batch fits max_alloc");
    check(b16.seed_batch <= b16.elems_per_round, "16GB seed_batch <= elems_per_round");
    // Sortless all-pairs match (T4) costs ~N^2/num_buckets, so buckets must be
    // SMALL: bucket_bits = clamp(floor(log2(elems_per_round))-5, 10, 20).
    // elems=2^25 -> floor(log2)=25 -> bits=20 (clamped high) -> 2^20 buckets.
    check(b16.bucket_bits == 20u, "16GB bucket_bits == 20 (clamped high)");
    check(b16.num_buckets == 1048576u, "16GB uses 2^20 = 1048576 buckets (sortless-match sizing)");
    check(b16.slots_per_bucket == 64u, "16GB slots_per_bucket == 64 (mean 32/bucket -> 64 floor)");
    check((uint64_t)b16.slots_per_bucket*b16.num_buckets >= b16.elems_per_round, "16GB total slots cover elems");

    // 8 GB card: full resident 2^25 (11.4 GB) can't fit -> must clamp below 2^25.
    Budget b8 = compute_budget(8*GiB, 2*GiB);
    check(b8.elems_per_round < (1u<<25), "8GB clamps elems/round below 2^25 (honest drop)");
    check(b8.elems_per_round > 0, "8GB still positive elems/round");
    check(b8.seed_batch*56ull <= 2*GiB, "8GB seed_batch fits max_alloc");
    // elems_per_round == 21,474,836 -> floor(log2) == 24 -> bucket_bits == 19.
    check(b8.elems_per_round == 21474836u, "8GB elems_per_round == 21,474,836 (anchor for the bucket calc below)");
    check(b8.bucket_bits == 19u, "8GB bucket_bits == 19 (floor(log2(21474836))=24, 24-5=19)");
    check(b8.num_buckets == 524288u, "8GB uses 2^19 = 524288 buckets");
    check((uint64_t)b8.slots_per_bucket*b8.num_buckets >= b8.elems_per_round, "8GB total slots cover elems");

    // Apple M3 Max-like: huge unified memory, generous everything.
    Budget bm = compute_budget(110ull*GiB, 27ull*GiB);
    check(bm.elems_per_round == (1u<<25), "M3Max keeps full 2^25");
    check(bm.seed_batch == (1u<<25), "M3Max seed_batch = full 2^25 (max_alloc ample)");
    check(bm.num_buckets == 1048576u, "M3Max uses 2^20 = 1048576 buckets (same as 16GB: elems=2^25)");

    return summary("budget");
}
