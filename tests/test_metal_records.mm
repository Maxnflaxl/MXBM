// The Metal record helpers must produce byte-identical layouts to the CUDA ones.
//
// Two distinct properties are checked, and only the second one catches the bug that
// matters. Round-tripping (unpack(pack(x)) == x) passes even when pack and unpack
// share a matched pair of errors -- the record would then be self-consistent on
// Metal and incompatible with every other backend. So the packed WORDS are also
// compared against the layout formulas from kernels/cuda/bh3_records.cuh,
// recomputed here on the host.
#include "metal_harness.h"
#include <cstdint>

using namespace mxbm;

static const uint32_t kN = 65536;
static const uint32_t kIdxMask = 0x1FFFFFFu;

// A cheap deterministic generator: the test must be reproducible, and a fixed
// seed beats a golden file for values this mechanical.
static uint32_t xs(uint32_t& s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s;
}

int main() {
    @autoreleasepool {
        section("Metal record layouts match the CUDA formulas");
        mtest::Harness h;
        if (!h.init()) return summary("metal_records");

        std::vector<uint32_t> in(kN * 4);
        uint32_t s = 0x9e3779b9u;
        for (auto& v : in) v = xs(s);

        id<MTLBuffer> obuf = h.buffer(kN * 12 * sizeof(uint32_t));
        id<MTLBuffer> ibuf = h.buffer_with(in.data(), in.size() * sizeof(uint32_t));
        id<MTLBuffer> pbuf = h.buffer(kN * 8 * sizeof(uint64_t));

        auto ps = h.pipeline("records_probe");
        if (!h.run(ps, kN, @[ obuf, ibuf, pbuf ])) return summary("metal_records");

        const uint32_t* got = (const uint32_t*)obuf.contents;
        const uint64_t* pk  = (const uint64_t*)pbuf.contents;

        size_t rt_bad = 0, layout_bad = 0;
        for (uint32_t i = 0; i < kN; ++i) {
            const uint32_t key = in[i * 4 + 0] & 0xFFFFFFu;
            const uint32_t a   = in[i * 4 + 1] & kIdxMask;
            const uint32_t b   = in[i * 4 + 2] & kIdxMask;
            const uint32_t gi  = in[i * 4 + 3] & 0x3FFFFFFu;
            const uint32_t* g  = got + i * 12;

            // --- property 1: the round trip recovers every field ---
            const uint32_t want[12] = {
                a, b, gi,              // pair_left / pair_right / pair_gi
                a, b, key, a, gi,      // r3_l0 l1 l2 l3 gi
                a, b, a, b             // quad_l0 l1 l2 l3
            };
            for (int k = 0; k < 12; ++k)
                if (g[k] != want[k]) {
                    if (rt_bad == 0) {
                        char m[64]; std::snprintf(m, sizeof m, "round-trip i=%u field=%d", i, k);
                        check_eq_u64(g[k], want[k], m);
                    }
                    ++rt_bad;
                }

            // --- property 2: the packed words match the CUDA layout exactly ---
            const uint64_t w0 = (uint64_t)key | ((uint64_t)a << 24);
            const uint64_t w1 = (uint64_t)b   | ((uint64_t)gi << 25);
            const uint64_t p0 = (uint64_t)a | ((uint64_t)b << 25) | (((uint64_t)key & 0x3FFFull) << 50);
            const uint64_t p1 = ((uint64_t)key >> 14) | ((uint64_t)a << 11) | ((uint64_t)gi << 36);
            const uint64_t q0 = (uint64_t)key | ((uint64_t)a << 24);
            const uint64_t q1 = (uint64_t)b   | ((uint64_t)a << 25);
            const uint64_t q2 = (uint64_t)b   | ((uint64_t)gi << 25);
            const uint64_t wantp[8] = { w0, w1, p0, p1, q0, q1, q2, gi };
            for (int k = 0; k < 8; ++k)
                if (pk[i * 8 + k] != wantp[k]) {
                    if (layout_bad == 0) {
                        char m[64]; std::snprintf(m, sizeof m, "layout i=%u word=%d", i, k);
                        check_eq_u64(pk[i * 8 + k], wantp[k], m);
                    }
                    ++layout_bad;
                }
        }
        check(rt_bad == 0, "every field round-trips at its exact bit width");
        check(layout_bad == 0, "packed words are bit-identical to the CUDA layout");
        if (rt_bad)     std::printf("  (%zu round-trip mismatches)\n", rt_bad);
        if (layout_bad) std::printf("  (%zu layout mismatches)\n", layout_bad);

        return summary("metal_records");
    }
}
