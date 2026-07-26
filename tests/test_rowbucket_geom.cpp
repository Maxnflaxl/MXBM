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
        check(fb_cap_for(528u) == 743u, "fb_cap_for(528) == 743 (mean + 8 sd + 32)");
        check(fb_cap_for(2112u) == 2511u, "fb_cap_for(2112) == 2511");
    }

    return summary("rowbucket_geom");
}
