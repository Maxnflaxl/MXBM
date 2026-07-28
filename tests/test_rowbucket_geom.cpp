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
            { 16, 7.46, 3.27 }, { 15, 6.88, 2.96 }, { 14, 6.50, 2.76 },
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
            { 16, 5.28, 2.90 }, { 15, 4.91, 2.63 }, { 14, 4.66, 2.45 },
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
        check(!rb_geometry_for(cap, 0, gib(7.4)).viable, "7.4 GiB: no geometry fits");
        check( rb_geometry_for(cap, 0, gib(7.6)).viable, "7.6 GiB: (14,3) fits -- the floor");
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
            {  6.20, 15, true,  true,  "6.2 GiB: quad (15,2)" },
            {  5.70, 14, true,  true,  "5.7 GiB: quad (14,3) -- the 6 GB card class" },
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
        check(n == 6, "six rungs: three geometries x two record formats");
        for (int i = 0; i < n; ++i)
            check(rungs[i].bb + rungs[i].sm == 17u, "every rung is on the bb + sm == 17 line");
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
        check(fb_round_stride(3, true) == fb_round_stride(3, false)
           && fb_round_stride(4, true) == fb_round_stride(4, false),
              "rounds 3 and 4 write the same record either way");
        check(fb_cap_for(528u) == 743u, "fb_cap_for(528) == 743 (mean + 8 sd + 32)");
        check(fb_cap_for(2112u) == 2511u, "fb_cap_for(2112) == 2511");
    }

    return summary("rowbucket_geom");
}
