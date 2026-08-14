// The row-bucket geometry decision, tested where BOTH backends can see it.
//
// test_budget.cpp also exercises rb_geometry_for(), but links mxbm_gpu and so only
// builds with OpenCL present -- on a CUDA-only machine the shipping geometry choice
// would go untested. This links nothing but mxbm_geom, and pins the claim that motivated
// sharing the arithmetic: CUDA reaches cards OpenCL cannot, by stepping DOWN the ladder
// rather than refusing.
#include "gpu/rowbucket_geom.h"
#include "check.h"
#include <cstdint>
#include <cstdio>
#include <initializer_list>
using namespace mxbm;
using namespace mxbm::gpu;

int main() {
    const double GiB = 1024.0*1024*1024;
    const uint32_t cap = (1u<<25) + (1u<<25)/32;      // 34,603,008, the shipping capacity
    auto gib = [&](double g) { return (uint64_t)(g * GiB); };

    section("footprint by geometry");
    {
        // The direction that makes the ladder work at all: COARSER buckets need LESS
        // memory, because the mean + 8 sd + 32 reservation is paid once per bucket and
        // sd/mean shrinks as the mean grows.
        struct { uint32_t bb; double total, single; } want[] = {
            // Back-ref row 5 is survivor-cap-sized, not capacity-sized.
            { 17, 8.09, 3.74 }, { 16, 7.20, 3.27 }, { 15, 6.62, 2.96 }, { 14, 6.24, 2.76 },
        };
        double prev = 1e9;
        for (const auto& w : want) {
            size_t total = 0, single = 0;
            rowbucket_bytes(cap, w.bb, total, single);
            char msg[160];
            std::snprintf(msg, sizeof msg, "(%u,%u) total %.2f GiB (expect %.2f), single %.2f (expect %.2f)",
                          w.bb, 17u - w.bb, total/GiB, w.total, single/GiB, w.single);
            check(total/GiB > w.total - 0.05 && total/GiB < w.total + 0.05
               && single/GiB > w.single - 0.05 && single/GiB < w.single + 0.05, msg);
            check(total/GiB < prev, "coarser buckets cost strictly less memory");
            prev = total/GiB;
        }
    }

    section("the quad record's footprint");
    {
        // Round 2 emits 3 u64 instead of 9, so set 0 stops being the larger of the two
        // sets and the largest single allocation becomes set 1. These are the numbers
        // cuda/pipeline prints under -DMXBM_R3_QUAD=1, to two decimals.
        struct { uint32_t bb; double total, single; } want[] = {
            { 16, 5.02, 2.90 }, { 15, 4.65, 2.63 }, { 14, 4.40, 2.45 },
        };
        for (const auto& w : want) {
            size_t total = 0, single = 0;
            rowbucket_bytes(cap, w.bb, total, single, /*quad=*/true);
            char msg[160];
            std::snprintf(msg, sizeof msg, "quad (%u,%u) total %.2f GiB (expect %.2f), single %.2f (expect %.2f)",
                          w.bb, 17u - w.bb, total/GiB, w.total, single/GiB, w.single);
            check(total/GiB > w.total - 0.05 && total/GiB < w.total + 0.05
               && single/GiB > w.single - 0.05 && single/GiB < w.single + 0.05, msg);
        }
        // THE FACT THE LADDER'S ORDER TURNS ON. quad (16,1) is smaller than packed
        // (14,3) -- 5.28 against 6.50 -- AND faster, 38.39 ms against 40.02. So the
        // rungs cannot be sorted by footprint: doing that would put packed (14,3) first
        // and hand a card the slower of the two.
        size_t q16 = 0, p14 = 0, ignore = 0;
        rowbucket_bytes(cap, 16, q16, ignore, /*quad=*/true);
        rowbucket_bytes(cap, 14, p14, ignore, /*quad=*/false);
        check(q16 < p14, "quad (16,1) is SMALLER than packed (14,3) -- and it is also faster");
    }

    section("the implicit-bits record's footprint");
    {
        // Not a rung -- a narrower set 0 on the rungs that admit it. The bucket address
        // already carries bb of the 24 collision bits, so 400 - bb work bits fit 6 u64
        // and the 9th-word plane has no writer: 9 u64 per slot becomes 8.
        check( rb_impb_ok(16, 1, false) &&  rb_impb_ok(17, 0, false),
               "(16,1) and (17,0) admit the pack");
        check(!rb_impb_ok(15, 2, false) && !rb_impb_ok(14, 3, false),
               "coarser rungs do not: 400 - bb overflows 6 u64");
        check(!rb_impb_ok(16, 1, true), "the quad record has no plane to delete");

        struct { uint32_t bb, sm; double total, single; } want[] = {
            { 17, 0, 7.67, 3.32 }, { 16, 1, 6.84, 2.90 },
        };
        for (const auto& w : want) {
            size_t total = 0, single = 0, was = 0, ignore = 0;
            rowbucket_bytes(cap, w.bb, total, single, /*quad=*/false, /*impb=*/true);
            rowbucket_bytes(cap, w.bb, was, ignore);
            char msg[160];
            std::snprintf(msg, sizeof msg,
                          "implicit-bits (%u,%u) total %.2f GiB (expect %.2f), single %.2f (expect %.2f)",
                          w.bb, w.sm, total/GiB, w.total, single/GiB, w.single);
            check(total/GiB > w.total - 0.05 && total/GiB < w.total + 0.05
               && single/GiB > w.single - 0.05 && single/GiB < w.single + 0.05, msg);
            check((was - total)/GiB > 0.35, "the deleted plane is worth at least 0.36 GiB");
        }

        // What it buys the ladder: the finest geometry's requirement drops by exactly the
        // deleted plane, so a card with 6.84-7.20 GiB usable runs (16,1) instead of
        // stepping down to (15,2) -- 33.67 ms against 36.01. It does NOT move the floor:
        // the rungs a small card lands on are quad, which has no plane to delete.
        auto pick = [&](double v, bool impb) {
            return rb_geometry_for(cap, 0, gib(v), /*allow_quad=*/true, 0,
                                   /*allow_split=*/false, /*slack=*/0, impb);
        };
        const RbGeometry with = pick(7.0, true), without = pick(7.0, false);
        check(with.bb == 16 && !with.quad && without.bb == 15 && !without.quad,
              "7.0 GiB usable: implicit bits keep (16,1) where the ladder stepped to (15,2)");
        check(pick(6.83, true).bb == 15 && pick(7.21, false).bb == 16,
              "and the band it wins is exactly the 0.36 GiB the plane took");
        // It never changes a rung it does not apply to, floor included.
        for (double v : {6.20, 5.70, 5.30, 4.45}) {
            const RbGeometry a = pick(v, true), b = pick(v, false);
            check(a.bb == b.bb && a.quad == b.quad && a.viable == b.viable,
                  "the quad rungs are sized identically with and without the pack");
        }
    }

    section("CUDA reaches cards OpenCL cannot");
    {
        // max_alloc = 0 means "no per-allocation limit", which is the honest CUDA value;
        // NVIDIA's OpenCL reports CL_DEVICE_MAX_MEM_ALLOC_SIZE = VRAM/4.
        struct { const char* card; double vram; int ocl_bb; int cuda_bb; } cards[] = {
            //                       OpenCL (VRAM/4 cap)     CUDA (no cap)
            { "16 GB", 15.59,  16, 16 },   // both take the finest geometry
            { "12 GB", 11.60,  14, 16 },   // OpenCL is single-alloc bound down to (14,3)
            { "11 GB", 10.60,  -1, 16 },   // OpenCL: nothing fits -> 215 ms sort path
            { "10 GB",  9.70,  -1, 16 },   // CUDA still hosts the FINEST geometry here
            {  "8 GB",  8.00,  -1, 15 },   // and steps down rather than refusing
        };
        for (const auto& c : cards) {
            const RbGeometry o = rb_geometry_for(cap, gib(c.vram)/4, gib(c.vram));
            const RbGeometry u = rb_geometry_for(cap, 0,             gib(c.vram));
            char msg[160];
            std::snprintf(msg, sizeof msg, "%s: OpenCL %s, CUDA %s", c.card,
                          c.ocl_bb  < 0 ? "sort path" : "row-bucket",
                          c.cuda_bb < 0 ? "sort path" : "row-bucket");
            check(o.viable == (c.ocl_bb >= 0) && u.viable == (c.cuda_bb >= 0), msg);
            if (c.ocl_bb  >= 0) { std::snprintf(msg, sizeof msg, "%s: OpenCL geometry (%d,%d)",
                                                c.card, c.ocl_bb, 17 - c.ocl_bb);
                                  check(o.bb == (uint32_t)c.ocl_bb, msg); }
            if (c.cuda_bb >= 0) { std::snprintf(msg, sizeof msg, "%s: CUDA geometry (%d,%d)",
                                                c.card, c.cuda_bb, 17 - c.cuda_bb);
                                  check(u.bb == (uint32_t)c.cuda_bb, msg); }
        }
        // The floor, either side of it. Below this the fast path is genuinely gone --
        // and so, on the CUDA side, is mining: there is no CUDA sort path.
        check(!rb_geometry_for(cap, 0, gib(7.1)).viable, "7.1 GiB: no geometry fits");
        check( rb_geometry_for(cap, 0, gib(7.35)).viable, "7.35 GiB: (14,3) fits -- the floor");
    }

    section("the quad record is a second axis, and it lowers the floor");
    {
        // allow_quad defaults to FALSE, which is what keeps the OpenCL backend and every
        // caller above on exactly the ladder they had. Only CudaSolver passes true.
        for (double v : {15.59, 11.60, 9.70, 8.00, 7.60, 7.00, 6.20, 5.70}) {
            const RbGeometry g = rb_geometry_for(cap, 0, gib(v));
            check(!g.quad, "allow_quad defaults false: the packed ladder is unchanged");
        }

        struct { double vram; int bb; bool quad; bool viable; const char* why; } want[] = {
            { 15.59, 16, false, true,  "16 GB: finest packed geometry, as before" },
            {  9.70, 16, false, true,  "10 GB: still the finest packed geometry" },
            {  8.00, 15, false, true,  "8 GB: packed (15,2) -- quad is slower and not needed" },
            {  7.60, 16, true,  true,  "7.6 GiB: quad (16,1) BEATS packed (14,3), 38.4 ms vs 40.0" },
            {  7.00, 16, true,  true,  "7.0 GiB: packed refuses entirely; quad (16,1) runs" },
            {  6.20, 16, true,  true,  "6.2 GiB: quad (16,1) -- a rung up since row 5 shrank" },
            {  5.70, 15, true,  true,  "5.7 GiB: quad (15,2) -- the 6 GB card class, a rung up" },
            {  5.00, 14, false, false, "5.0 GiB: below the new floor, nothing fits" },
        };
        for (const auto& w : want) {
            const RbGeometry g = rb_geometry_for(cap, 0, gib(w.vram), /*allow_quad=*/true);
            check(g.viable == w.viable && (!w.viable || (g.bb == (uint32_t)w.bb && g.quad == w.quad)),
                  w.why);
        }

        // The improvement stated as the two claims it rests on, so a regression in
        // either is named rather than showing up as a changed table row.
        const RbGeometry a = rb_geometry_for(cap, 0, gib(7.6), true);
        const RbGeometry b = rb_geometry_for(cap, 0, gib(7.6), false);
        check(a.quad && a.bb == 16 && !b.quad && b.bb == 14,
              "at 7.6 GiB the quad ladder picks a FASTER rung than the packed one");
        check(!rb_geometry_for(cap, 0, gib(7.0), false).viable
           &&  rb_geometry_for(cap, 0, gib(7.0), true ).viable,
              "the quad record takes the CUDA floor from 7.6 GiB down past 7.0");

        // WHAT THE QUAD RECORD BUYS ON OPENCL, which is the point of porting it there:
        // OpenCL is bound by CL_DEVICE_MAX_MEM_ALLOC_SIZE (VRAM/4 on NVIDIA), and the
        // quad record takes the largest single allocation from 2.76 GiB to 2.45. A card
        // that clears the smaller number runs the row-bucket path instead of falling
        // back to the sort path, which measures ~190 ms against ~42.
        struct { double vram; bool packed_ok; bool quad_ok; const char* why; } ocl[] = {
            { 15.59, true,  true,  "16 GB: row-bucket either way" },
            { 11.60, true,  true,  "12 GB: row-bucket either way" },
            { 10.60, false, true,  "11 GB: SORT PATH without the quad record, row-bucket with it" },
            {  9.70, false, false, "10 GB: max_alloc 2.42 GiB misses even quad (14,3)'s 2.45 -- still sort" },
        };
        for (const auto& c : ocl) {
            const uint64_t ma = gib(c.vram) / 4;             // NVIDIA OpenCL reports VRAM/4
            check(rb_geometry_for(cap, ma, gib(c.vram), false).viable == c.packed_ok
               && rb_geometry_for(cap, ma, gib(c.vram), true ).viable == c.quad_ok, c.why);
        }
        // The 11 GB card specifically: name the rung, so a regression in the ordering
        // shows up as this assertion rather than as a silent 4x slowdown for that class.
        const RbGeometry g11 = rb_geometry_for(cap, gib(10.60)/4, gib(10.60), true);
        check(g11.viable && g11.quad && g11.bb == 15,
              "11 GB OpenCL lands on quad (15,2) -- the first rung under its 2.65 GiB ceiling");

        // Every rung stays on the line the compile-time kernel constants assume.
        int n = 0;
        const RbRung* rungs = rb_rungs(n);
        check(n == 11, "eleven rungs: three geometries x two record formats x the "
                       "dense-cap rows, less the ones a faster row already dominates");
        for (int i = 0; i < n; ++i)
            check(rungs[i].bb + rungs[i].sm == 17u, "every rung is on the bb + sm == 17 line");
        // The ladder is walked in order and the first fit wins, so a rung that is both
        // slower and larger than one above it can never be chosen. Nothing here asserts
        // the times -- only that footprint alone does not order the list, which is the
        // trap a footprint-sorted ladder falls into.
        bool monotonic = true;
        size_t prev = 0;
        for (int i = 0; i < n; ++i) {
            size_t t = 0, s = 0;
            rowbucket_bytes(cap, rungs[i].bb, t, s, rungs[i].quad,
                            rb_impb_ok(rungs[i].bb, rungs[i].sm, rungs[i].quad),
                            rungs[i].arena);
            if (i && t > prev) monotonic = false;
            prev = t;
        }
        check(!monotonic, "the ladder is ordered by time, not by footprint");
    }

    section("split record sets take the OpenCL floor from 11 GB to CUDA's ~5.7 GiB");
    {
        // allow_split: the OpenCL backend can allocate each record set as two
        // bucket-halves (fb_elem_hi), so CL_DEVICE_MAX_MEM_ALLOC_SIZE binds on
        // rowbucket_single_split() -- half the larger set, or the never-split
        // back-ref row when bigger -- instead of on the whole set. Default false:
        // every row above is unchanged, and CUDA (max_alloc 0) never wants it.
        const size_t backrefs = (size_t)5 * cap * 4;                // 0.64 GiB
        check(rowbucket_single_split(cap, 16, false) > 1.60*GiB
           && rowbucket_single_split(cap, 16, false) < 1.70*GiB,
              "(16,1) split piece ~1.63 GiB (half of 3.27)");
        check(rowbucket_single_split(cap, 14, true) > backrefs,
              "even the smallest rung's half-set still exceeds the back-ref row");

        // The NVIDIA-OpenCL decision (max_alloc = VRAM/4), quad + split allowed --
        // what rb_pick_geometry actually asks since the split shipped. Every card
        // class that used to be sort-path or refused now gets a fast-path rung,
        // and the 11-12 GB class climbs to the FINEST geometry.
        struct { const char* card; double vram; int bb; bool quad; bool viable; const char* why; } want[] = {
            { "16 GB", 15.59, 16, false, true,  "16 GB: finest packed, split never engages" },
            { "12 GB", 11.60, 16, false, true,  "12 GB: (16,1) via split -- was (14,3), two rungs finer" },
            { "11 GB", 10.60, 16, false, true,  "11 GB: (16,1) via split -- was quad (15,2)" },
            { "8 GB",   8.00, 15, false, true,  "8 GB: packed (15,2), same rung CUDA picks -- was REFUSED" },
            { "6 GB",   6.00, 15, true,  true,  "6 GB: quad (15,2) -- was refused" },
            { "5.7 GB", 5.70, 15, true,  true,  "5.7 GiB: quad (15,2) -- a rung up since row 5 shrank" },
            { "5 GB",   5.00,  0, false, false, "5.0 GiB: below the floor either way" },
        };
        for (const auto& w : want) {
            const RbGeometry g = rb_geometry_for(cap, gib(w.vram)/4, gib(w.vram),
                                                 /*allow_quad=*/true, 0, /*allow_split=*/true);
            check(g.viable == w.viable && (!w.viable || (g.bb == (uint32_t)w.bb && g.quad == w.quad)),
                  w.why);
        }
        // The claim stated as its two halves, so a regression is named:
        check(!rb_geometry_for(cap, gib(8.0)/4, gib(8.0), true, 0, false).viable
           &&  rb_geometry_for(cap, gib(8.0)/4, gib(8.0), true, 0, true ).viable,
              "8 GB: the split alone is what makes the card viable");
        check(rb_geometry_for(cap, gib(10.60)/4, gib(10.60), true, 0, true).bb == 16u
           && rb_geometry_for(cap, gib(10.60)/4, gib(10.60), true, 0, false).bb == 15u,
              "11 GB: the split alone is what buys the finer rung");
    }

    section("the power-cap policy is ARMED below 130 W");
    {
        // The band is kSpecMinPowerW's: (17,0) and speculative entry both act on round
        // 4's block population, so only their composition was measured and only the
        // composition ships. An earlier arming at a wider band was disarmed when its
        // prize did not reproduce; see docs/performance-research.md for both.
        check(kRbLowPowerW == 130, "the (17,0) band is armed at the spec-entry threshold");
        for (unsigned w : {0u, 130u, 160u, 189u, 285u}) {
            const RbGeometry g = rb_geometry_for(cap, 0, gib(15.59), true, w);
            check(g.viable && g.bb == 16 && !g.quad,
                  "at or above the band (and with no hint), the ladder answers unchanged");
        }
        for (unsigned w : {100u, 120u, 129u}) {
            const RbGeometry g = rb_geometry_for(cap, 0, gib(15.59), true, w);
            check(g.viable && g.bb == 17 && !g.quad,
                  "below the band, a 16 GB card takes (17,0)");
        }
        // (17,0) needs 8.09 GiB; a card that cannot host it falls back to the ladder
        // rather than refusing -- the policy is an exception, not a requirement.
        check(rb_geometry_for(cap, 0, gib(8.00), true, 100).bb == 15,
              "capped 8 GB card: (17,0) does not fit, so the ladder answers as before");
        // main.cpp's restart notice compares selections at the startup and current
        // limits; armed, a cap crossing the band is exactly what makes them differ.
        check(rb_geometry_for(kRbCapacity, 0, gib(15.59), true, 285).bb
           != rb_geometry_for(kRbCapacity, 0, gib(15.59), true, 100).bb,
              "armed: crossing the band changes the selection -> the notice can fire");
        check(kRbCapacity == cap, "the exposed capacity is the shipping capacity");
    }

    section("the dense-cap rungs and their overflow pool");
    {
        // The cap stops being the tail bound and becomes "tight enough that the pool
        // absorbs the rest": mean + 8 sigma -> mean + 2 sigma, and zero drops moves to a
        // pool-full check. At (16,1) that is 743 slots per bucket down to 605.
        check(fb_cap_for(528u) == 743u && fb_cap_for(528u, kRbDenseSigma) == 605u,
              "(16,1): the dense cap is 605 slots against the tail bound's 743");
        check(fb_cap_for(2112u) == 2511u && fb_cap_for(2112u, kRbDenseSigma) == 2235u,
              "(14,3): 2235 against 2511 -- coarser buckets had less slack to give back");

        struct { uint32_t bb; bool quad; double total; } want[] = {
            { 16, false, 5.77 },     // with the implicit-bits pack: the new best rung
            { 15, false, 5.82 },
            { 16, true,  4.29 },
            { 14, true,  4.03 },     // the floor
        };
        for (const auto& w : want) {
            size_t total = 0, single = 0, plain = 0, ignore = 0;
            const bool im = rb_impb_ok(w.bb, 17u - w.bb, w.quad);
            rowbucket_bytes(cap, w.bb, total, single, w.quad, im, /*arena=*/true);
            rowbucket_bytes(cap, w.bb, plain, ignore, w.quad, im);
            char msg[160];
            std::snprintf(msg, sizeof msg, "%s(%u,%u) + arena is %.2f GiB (expect %.2f)",
                          w.quad ? "quad " : "", w.bb, 17u - w.bb, total/GiB, w.total);
            check(total/GiB > w.total - 0.05 && total/GiB < w.total + 0.05, msg);
            check(total < plain, "the dense cap always gives memory back");
        }

        // What it buys the ladder. allow_arena defaults FALSE, so OpenCL and Metal never
        // see these rungs and their answers are exactly what they were.
        auto pick = [&](double v, bool arena) {
            return rb_geometry_for(cap, 0, gib(v), /*allow_quad=*/true, 0,
                                   /*allow_split=*/false, /*slack=*/0,
                                   /*allow_impb=*/true, arena);
        };
        for (double v : {15.59, 9.70, 8.00, 7.00, 6.20, 5.00}) {
            const RbGeometry g = rb_geometry_for(cap, 0, gib(v), /*allow_quad=*/true);
            check(!g.arena, "allow_arena defaults false: the plain ladder is unchanged");
        }
        const RbGeometry a = pick(6.0, true), b = pick(6.0, false);
        check(a.bb == 16 && !a.quad && a.arena && b.bb == 16 && b.quad && !b.arena,
              "6.0 GiB usable: dense caps keep the PACKED (16,1) record where the ladder "
              "had to take the quad one");
        // And the floor, either side of it.
        check( pick(4.05, true).viable && !pick(4.00, true).viable,
               "the arena floor is quad (14,3) at 4.03 GiB");
        check(!pick(4.35, false).viable,
               "without it the floor is quad (14,3) at 4.40 -- 0.37 GiB higher");
        // In card terms, through CUDA's availability rule: total VRAM less a fixed
        // 640 MiB driver-plus-context allowance (cuda_solver.cu). A 5 GB card clears the
        // floor and a 4 GB one does not, which is where reach now stops.
        check( pick(4.70 - 0.625, true).viable, "a 5 GB card clears the arena floor");
        check(!pick(3.70 - 0.625, true).viable, "a 4 GB card does not");
    }

    section("invariants the kernels depend on");
    {
        // Everything compile-time in the CUDA kernels is invariant along bb + sm = 17.
        // A geometry off this line silently breaks all three of these.
        for (double v : {15.59, 11.60, 9.70, 8.00, 7.60}) {
            const RbGeometry g = rb_geometry_for(cap, 0, gib(v));
            check(g.viable && g.bb + g.sm == 17u, "geometry stays on the bb + sm == 17 line");
            check((1ull << g.bb) << g.sm == (1ull << 17), "grid is 2^17 blocks at every step");
            check(24u - g.bb - g.sm == 7u, "7 key bits vary within a group -> 128 is a perfect hash");
        }
        // The CUDA solver static_asserts against these widths; diverging makes the
        // footprint arithmetic wrong for one of the two backends.
        check(fb_set_stride(0) == 9u && fb_set_stride(1) == 8u,
              "ping-pong set widths are 9 and 8 u64");
        // Under the quad record set 0 carries round 2's 3 u64 and round 4's 2, so it is
        // 3 -- and set 1 is untouched, since only round 2's output format changed.
        check(fb_set_stride(0, true) == 3u && fb_set_stride(1, true) == 8u,
              "quad set widths are 3 and 8 u64");
        check(fb_round_stride(2, true) == 3u && fb_round_stride(2, false) == 9u,
              "only round 2's output stride differs between the formats");
        // The implicit-bits pack is the third format for the same round: the record is
        // still 8 u64, but there is no plane behind it, so set 0 is the record alone.
        check(fb_round_stride(2, false, true) == 8u && fb_set_stride(0, false, true) == 8u
           && fb_set_stride(1, false, true) == 8u,
              "implicit-bits set widths are 8 and 8 u64");
        check(fb_round_stride(3, true) == fb_round_stride(3, false)
           && fb_round_stride(4, true) == fb_round_stride(4, false),
              "rounds 3 and 4 write the same record either way");
        check(fb_cap_for(528u) == 743u, "fb_cap_for(528) == 743 (mean + 8 sd + 32)");
        check(fb_cap_for(2112u) == 2511u, "fb_cap_for(2112) == 2511");
    }

    return summary("rowbucket_geom");
}
