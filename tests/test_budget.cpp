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
    // capacity = seed count + collision headroom so rounds 1-2 never clamp
    // (out_capacity drops would nondeterministically lose solutions). The larger
    // buffers must still fit the usable VRAM budget.
    check(b16.capacity > b16.elems_per_round, "16GB reserves collision headroom (capacity > 2^25 seed count)");
    check((uint64_t)b16.capacity * ((uint64_t)kNumRounds*kResidentElemBytes + kSeedElemBytes) <= b16.usable,
          "16GB headroom buffers still fit usable VRAM");

    // 8 GB card: full resident 2^25 (11.4 GB) can't fit -> must clamp below 2^25.
    Budget b8 = compute_budget(8*GiB, 2*GiB);
    check(b8.elems_per_round < (1u<<25), "8GB clamps elems/round below 2^25 (honest drop)");
    check(b8.elems_per_round > 0, "8GB still positive elems/round");
    check(b8.seed_batch*56ull <= 2*GiB, "8GB seed_batch fits max_alloc");
    // usable (8 GiB * 0.85) / kBytesPerElement. DERIVED from the constant rather than
    // pinned to a literal: the per-element cost is measured against the real allocation
    // (test_gpu_rounds) and has been retuned several times, and a hard-coded expectation
    // here just breaks on every retune without catching anything. What matters is that
    // compute_budget divides by the constant it is handed. This was 396 B/elem, which
    // under-provisioned so far
    // that 12 GB cards were denied a full seed layer; see HW_REQUIREMENTS.md.
    const uint32_t expect8 = (uint32_t)(((uint64_t)((double)(8ull<<30) * 0.85)) / kBytesPerElement);
    check(b8.elems_per_round == expect8, "8GB elems_per_round == usable / kBytesPerElement");
    uint32_t lg8 = 0; while ((1ull << (lg8 + 1)) <= expect8) ++lg8;
    check(b8.bucket_bits == (lg8 >= 15 ? lg8 - 5 : 10), "8GB bucket_bits tracks floor(log2(epr)) - 5");
    check(b8.num_buckets == 524288u, "8GB uses 2^19 = 524288 buckets");
    check((uint64_t)b8.slots_per_bucket*b8.num_buckets >= b8.elems_per_round, "8GB total slots cover elems");
    // Memory-constrained: the seed layer already maxes out VRAM, so there is no
    // room for headroom -- capacity pins to elems_per_round (no worse than before).
    check(b8.capacity == b8.elems_per_round, "8GB has no headroom room (capacity == elems_per_round)");

    // The per-device thresholds published in HW_REQUIREMENTS.md. Real callers pass the
    // path's own per-element figure (gpu_solver.cpp tries row-bucket, then sort), so
    // pinning the DEFAULT divisor alone would not catch the table going stale -- which
    // it did: the summary claimed "~16 GB today" for months after the budget was fixed.
    //
    // These are the SEED-LAYER thresholds only. Whether the row-bucket path actually
    // runs is a second question answered by rb_pick_geometry(), which needs a real
    // device and so lives in the gpu_rounds tests: below ~12 GB, NVIDIA's OpenCL
    // max_alloc of 1/4 VRAM rejects every row-bucket geometry even though the budget
    // here would grant a full layer, and the sort budget is what the device ends up
    // sized by. That is exactly why a 10 GB card refuses under OpenCL but runs under
    // CUDA, which has no max_alloc ceiling.
    section("per-path VRAM thresholds");
    struct { const char* name; double gib; bool rb_full, sort_full; } cards[] = {
        { "16 GB", 15.59, true,  true  },
        { "12 GB", 11.60, true,  true  },
        { "11 GB", 10.60, true,  true  },
        { "10 GB",  9.70, true,  false },   // row-bucket budget fits, sort budget does not
        { "8 GB",   7.70, false, false },   // neither -> the miner refuses to start
    };
    for (const auto& c : cards) {
        const uint64_t gm = (uint64_t)(c.gib * (double)GiB);
        Budget rb = compute_budget(gm, gm/4, 0.85, kBytesPerElementRowbucket);
        Budget so = compute_budget(gm, gm/4, 0.85, kBytesPerElementSort);
        char msg[128];
        std::snprintf(msg, sizeof msg, "%s: row-bucket hosts a full seed layer = %s",
                      c.name, c.rb_full ? "yes" : "no");
        check(budget_can_find_solutions(rb.elems_per_round) == c.rb_full, msg);
        std::snprintf(msg, sizeof msg, "%s: sort path hosts a full seed layer = %s",
                      c.name, c.sort_full ? "yes" : "no");
        check(budget_can_find_solutions(so.elems_per_round) == c.sort_full, msg);
    }
    // The thresholds themselves, so a change to either constant shows up as a size class
    // rather than as a silently different table.
    const double rb_need   = kBytesPerElementRowbucket * (double)(1u<<25) / 0.85 / (double)GiB;
    const double sort_need = kBytesPerElementSort      * (double)(1u<<25) / 0.85 / (double)GiB;
    check(rb_need > 9.0 && rb_need < 9.8,
          "row-bucket seed layer needs ~9.4 GiB reported VRAM");
    check(sort_need > 10.1 && sort_need < 10.8,
          "sort-path seed layer needs ~10.4 GiB reported VRAM");

    // Apple M3 Max-like: huge unified memory, generous everything.
    Budget bm = compute_budget(110ull*GiB, 27ull*GiB);
    check(bm.elems_per_round == (1u<<25), "M3Max keeps full 2^25");
    check(bm.seed_batch == (1u<<25), "M3Max seed_batch = full 2^25 (max_alloc ample)");
    check(bm.num_buckets == 1048576u, "M3Max uses 2^20 = 1048576 buckets (same as 16GB: elems=2^25)");
    check(bm.capacity > bm.elems_per_round, "M3Max reserves collision headroom (capacity > 2^25 seed count)");

    return summary("budget");
}
