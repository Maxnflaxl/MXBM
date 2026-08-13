#include "gpu/budget.h"
#include "gpu/round_pipeline.h"
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

    // The geometry step-down, which is what ACTUALLY decides whether a card gets
    // the fast path: the seed-layer budget above says yes well below 12 GB, but
    // NVIDIA's OpenCL caps a single allocation at 1/4 of VRAM and the coarsest
    // row-bucket layout still wants 2.76 GiB. This is the half of
    // HW_REQUIREMENTS.md's table that the budget checks cannot see.
    section("row-bucket geometry step-down");
    {
        const uint32_t cap = (1u<<25) + (1u<<25)/32;   // the shipping capacity
        struct { const char* name; double gib; uint32_t bb; bool viable; } cards[] = {
            { "16 GB", 15.59, 16, true  },   // finest geometry, fastest
            { "12 GB", 11.60, 14, true  },   // steps down twice, ~5 % slower
            { "11 GB", 10.60, 14, false },   // no geometry fits -> sort path
            { "10 GB",  9.70, 14, false },
        };
        for (const auto& c : cards) {
            const uint64_t gm = (uint64_t)(c.gib * (double)GiB);
            const RbGeometry g = rb_geometry_for(cap, gm/4, gm);   // NVIDIA: max_alloc = VRAM/4
            char msg[128];
            std::snprintf(msg, sizeof msg, "%s: row-bucket viable = %s",
                          c.name, c.viable ? "yes" : "no");
            check(g.viable == c.viable, msg);
            if (c.viable) {
                std::snprintf(msg, sizeof msg, "%s: geometry (%u, %u)", c.name, c.bb, 17u - c.bb);
                check(g.bb == c.bb && g.sm == 17u - c.bb, msg);
            }
        }
        // bucket_bits + submask_bits == 17 is the constraint the whole geometry
        // family sits on -- see docs/performance.md, "Row-bucket geometry".
        for (double gib : {15.59, 11.60}) {
            const uint64_t gm = (uint64_t)(gib * (double)GiB);
            const RbGeometry g = rb_geometry_for(cap, gm/4, gm);
            check(g.bb + g.sm == 17u, "geometry stays on the bb + sm == 17 line");
        }
        // The single allocation is what binds, not the total: give a 12 GB card
        // an unlimited max_alloc and it takes the finest geometry after all.
        const RbGeometry unlimited = rb_geometry_for(cap, 0, (uint64_t)(11.60 * (double)GiB));
        check(unlimited.viable && unlimited.bb == 16,
              "with no single-allocation cap, 12 GB takes the finest geometry (this is CUDA)");
        // Total VRAM is a SEPARATE gate from the single-allocation one, and on
        // every card above the single-alloc check happens to fire first -- so
        // without this case the total check could be deleted unnoticed. A card
        // with no allocation cap but only 8 GiB (the CUDA-style situation) must
        // still be refused: 7.46 GiB of footprint plus a GiB of headroom does
        // not fit.
        {
            const RbGeometry g8 = rb_geometry_for(cap, 0, (uint64_t)(8.0 * (double)GiB));
            check(g8.viable && g8.bb == 15,
                  "8 GiB total steps down to (15,2) on total size alone, not max_alloc");
            check(!rb_geometry_for(cap, 0, (uint64_t)(7.1 * (double)GiB)).viable,
                  "7.1 GiB is below even (14,3)'s 6.24 GiB + 1 GiB headroom");
            check(rb_geometry_for(cap, 0, (uint64_t)(7.6 * (double)GiB)).viable,
                  "7.6 GiB clears (14,3) -- the true floor for the fast path");
        }

        // And the footprint figures the docs quote.
        size_t total = 0, single = 0;
        rowbucket_bytes(cap, 16, total, single);
        check(single > 3.2*GiB && single < 3.35*GiB, "geometry (16,1) largest alloc ~3.27 GiB");
        check(total  > 7.15*GiB && total  < 7.30*GiB, "geometry (16,1) total ~7.20 GiB");
        rowbucket_bytes(cap, 14, total, single);
        check(single > 2.7*GiB && single < 2.85*GiB, "geometry (14,3) largest alloc ~2.76 GiB");
    }

    // Apple M3 Max-like: huge unified memory, generous everything.
    Budget bm = compute_budget(110ull*GiB, 27ull*GiB);
    check(bm.elems_per_round == (1u<<25), "M3Max keeps full 2^25");
    check(bm.seed_batch == (1u<<25), "M3Max seed_batch = full 2^25 (max_alloc ample)");
    check(bm.num_buckets == 1048576u, "M3Max uses 2^20 = 1048576 buckets (same as 16GB: elems=2^25)");
    check(bm.capacity > bm.elems_per_round, "M3Max reserves collision headroom (capacity > 2^25 seed count)");

    // -- the sizing basis: free VRAM less a fixed reserve --
    section("vram reserve");
    const uint64_t MiB = 1024ull*1024;
    {
        clear_keepfree();
        check(reserve_bytes(true)  == kReserveDisplayBytes,  "a display attached reserves 256 MB");
        check(reserve_bytes(false) == kReserveHeadlessBytes, "headless reserves 64 MB");

        // 16 GiB card, 15.6 GiB free, desktop up.
        const uint64_t total = 16376ull*MiB, freeb = 15620ull*MiB;
        check(usable_vram(freeb, total, reserve_bytes(true)) == freeb - 256*MiB,
              "usable is free minus the reserve, never a share of the total");

        // Free is what the driver already netted of other processes, so a busy
        // card yields less without any extra arithmetic here.
        check(usable_vram(4ull*GiB, total, 64*MiB) == 4ull*GiB - 64*MiB,
              "a card with other tenants sizes off what is actually free");

        check(usable_vram(0, total, 64*MiB) == (uint64_t)((double)total * kUnknownFreeHeadroom),
              "an unreadable free figure falls back to total times the headroom");
        check(usable_vram(100*MiB, total, 256*MiB) == 0,
              "a reserve larger than free yields nothing rather than underflowing");
        check(usable_vram(total + GiB, total, 0) == total,
              "usable never exceeds the card");
    }
    {
        // --keepfree replaces the reserve outright, 0 included.
        set_keepfree_bytes(0);
        check(keepfree_given(), "--keepfree is recorded as given");
        check(reserve_bytes(true) == 0 && reserve_bytes(false) == 0,
              "--keepfree 0 takes everything free, display or not");
        set_keepfree_bytes(4ull*GiB);
        check(reserve_bytes(true) == 4ull*GiB, "--keepfree is not capped or clamped");
        clear_keepfree();
        check(!keepfree_given() && reserve_bytes(true) == kReserveDisplayBytes,
              "clearing it restores the built-in reserve");
    }

    return summary("budget");
}
