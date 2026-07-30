// The BeamHash III primitives, compiled as Metal Shading Language, must agree
// bit-for-bit with the host implementation -- because they ARE the host
// implementation. src/beamhash/bh3_primitives.h is compiled three times (host,
// nvcc, Metal) and this test is what stops the Metal instantiation from drifting.
//
// A mismatch here would be invisible downstream: the rounds would still find
// collisions, just the WRONG ones, and the CPU verify gate would silently reject
// every solution. So this runs over a wide index sweep rather than a token few.
#include "metal_harness.h"
#include "beamhash/bh3_primitives.h"
#include <vector>

using namespace mxbm;

static const uint32_t kN = 4096;

// The same shape of prePow the CPU KAT vectors use, so a failure here is
// comparable to a failure in test_primitives rather than its own private universe.
static const uint64_t kPrePow[4] = {
    0x0123456789abcdefull, 0xfedcba9876543210ull,
    0x00ff00ff00ff00ffull, 0xdeadbeefcafebabeull
};

int main() {
    @autoreleasepool {
        section("MSL primitives are bit-identical to the host");
        mtest::Harness h;
        if (!h.init()) return summary("metal_primitives");

        const size_t words = (size_t)kN * bh3::kWorkWords;
        id<MTLBuffer> obuf = h.buffer(words * sizeof(uint64_t));
        id<MTLBuffer> pbuf = h.buffer_with(kPrePow, sizeof(kPrePow));
        if (!h.run(h.pipeline("prim_probe"), kN, @[ obuf, pbuf ]))
            return summary("metal_primitives");
        const uint64_t* got = (const uint64_t*)obuf.contents;

        // Report only the FIRST divergence in full, then count the rest: a
        // systematic mismatch would otherwise print 28k lines and bury the
        // one fact that matters (which index and which word broke first).
        size_t bad = 0;
        for (uint32_t i = 0; i < kN; ++i) {
            bh3::Elem a, b, c;
            uint32_t t[1];
            bh3::seed_element(kPrePow, i, a);      t[0] = i;      bh3::apply_mix(a, t, 1u, 448u);
            bh3::seed_element(kPrePow, i + 1u, b); t[0] = i + 1u; bh3::apply_mix(b, t, 1u, 448u);
            bh3::combine(a, b, 424u, c);

            for (int w = 0; w < bh3::kWorkWords; ++w) {
                const uint64_t g = got[(size_t)i * bh3::kWorkWords + w];
                if (g != c.w[w]) {
                    if (bad == 0) {
                        char msg[64];
                        std::snprintf(msg, sizeof msg, "index %u word %d", i, w);
                        check_eq_u64(g, c.w[w], msg);
                    }
                    ++bad;
                }
            }
        }
        check(bad == 0, "all 4096 x 7 work words match the host");
        if (bad) std::printf("  (%zu of %zu words diverged)\n",
                             bad, (size_t)kN * bh3::kWorkWords);
        return summary("metal_primitives");
    }
}
