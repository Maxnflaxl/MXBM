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
    check(b16.num_buckets == 4096u, "16GB uses 4096 buckets");
    check((uint64_t)b16.slots_per_bucket*b16.num_buckets >= b16.elems_per_round, "16GB total slots cover elems");

    // 8 GB card: full resident 2^25 (11.4 GB) can't fit -> must clamp below 2^25.
    Budget b8 = compute_budget(8*GiB, 2*GiB);
    check(b8.elems_per_round < (1u<<25), "8GB clamps elems/round below 2^25 (honest drop)");
    check(b8.elems_per_round > 0, "8GB still positive elems/round");
    check(b8.seed_batch*56ull <= 2*GiB, "8GB seed_batch fits max_alloc");
    check(b8.num_buckets == 4096u, "8GB uses 4096 buckets (elems >= 2^22 -> bucket_bits=12)");

    // Apple M3 Max-like: huge unified memory, generous everything.
    Budget bm = compute_budget(110ull*GiB, 27ull*GiB);
    check(bm.elems_per_round == (1u<<25), "M3Max keeps full 2^25");
    check(bm.seed_batch == (1u<<25), "M3Max seed_batch = full 2^25 (max_alloc ample)");

    return summary("budget");
}
