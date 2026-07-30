// The entry pass on Metal vs the CPU reference.
//
// entry_scatter appends to buckets with an atomic, so the ORDER within a bucket is
// nondeterministic by construction -- comparing position-by-position would fail on a
// correct kernel. What must match exactly is the multiset: every element lands in the
// bucket its key selects, exactly once, with no drops.
//
// Also times the pass at the full 2^25 so there is an Apple-vs-Ada datapoint for the
// emit-bandwidth question before the fused rounds are written.
#include "metal_harness.h"
#include "beamhash/bh3_primitives.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>

using namespace mxbm;

static const uint64_t kPrePow[4] = {
    0x0123456789abcdefull, 0xfedcba9876543210ull,
    0x00ff00ff00ff00ffull, 0xdeadbeefcafebabeull
};

// Small enough to hold a full CPU reference in memory, large enough that every
// bucket is populated many times over: 2^20 elements across 2^10 buckets.
static const uint32_t kN          = 1u << 20;
static const uint32_t kBucketBits = 10;
static const uint32_t kNumBuckets = 1u << kBucketBits;
// Generous: the expected occupancy is 1024, and Poisson spread around it is ~32.
static const uint32_t kBucketCap  = 4096;

struct Run {
    std::vector<uint32_t> counts;
    std::vector<uint64_t> belem;
    std::vector<uint32_t> drops;
};

int main() {
    @autoreleasepool {
        section("Metal entry pass matches the CPU reference");
        mtest::Harness h;
        if (!h.init()) return summary("metal_entry");

        auto ps = h.pipeline("entry_scatter");
        if (!ps) return summary("metal_entry");

        id<MTLBuffer> pbuf   = h.buffer_with(kPrePow, sizeof(kPrePow));
        id<MTLBuffer> cbuf   = h.buffer(kNumBuckets * sizeof(uint32_t));
        id<MTLBuffer> ebuf   = h.buffer((size_t)kNumBuckets * kBucketCap * sizeof(uint64_t));
        id<MTLBuffer> dbuf   = h.buffer(8 * sizeof(uint32_t));
        std::memset(cbuf.contents, 0, kNumBuckets * sizeof(uint32_t));
        std::memset(dbuf.contents, 0, 8 * sizeof(uint32_t));

        uint32_t begin = 0, count = kN, bits = kBucketBits, cap = kBucketCap;
        id<MTLBuffer> b1 = h.buffer_with(&begin, sizeof begin);
        id<MTLBuffer> b2 = h.buffer_with(&count, sizeof count);
        id<MTLBuffer> b3 = h.buffer_with(&bits,  sizeof bits);
        id<MTLBuffer> b4 = h.buffer_with(&cap,   sizeof cap);

        if (!h.run(ps, kN, @[ pbuf, b1, b2, b3, b4, cbuf, ebuf, dbuf ]))
            return summary("metal_entry");

        const uint32_t* gcounts = (const uint32_t*)cbuf.contents;
        const uint64_t* gbelem  = (const uint64_t*)ebuf.contents;
        const uint32_t* gdrops  = (const uint32_t*)dbuf.contents;

        // ---- CPU reference ----
        std::vector<std::vector<uint64_t>> want(kNumBuckets);
        for (uint32_t i = 0; i < kN; ++i) {
            bh3::Elem e;
            bh3::seed_element(kPrePow, i, e);
            uint32_t t1[1] = { i };
            bh3::apply_mix(e, t1, 1u, 448u);
            const uint32_t key = (uint32_t)(e.w[0] & 0xFFFFFFu);
            want[key >> (24u - kBucketBits)].push_back(((uint64_t)i << 32) | key);
        }

        check(gdrops[1] == 0, "no bucket overflowed");

        size_t bad_count = 0, bad_set = 0, total = 0;
        for (uint32_t b = 0; b < kNumBuckets; ++b) {
            if (gcounts[b] != want[b].size()) {
                if (bad_count == 0) {
                    char m[64]; std::snprintf(m, sizeof m, "bucket %u count", b);
                    check_eq_u64(gcounts[b], want[b].size(), m);
                }
                ++bad_count;
                continue;
            }
            total += want[b].size();
            // Sorted multiset comparison: atomic append order is not defined.
            std::vector<uint64_t> got(gbelem + (size_t)b * kBucketCap,
                                      gbelem + (size_t)b * kBucketCap + gcounts[b]);
            std::sort(got.begin(), got.end());
            std::sort(want[b].begin(), want[b].end());
            if (got != want[b]) {
                if (bad_set == 0) {
                    char m[64]; std::snprintf(m, sizeof m, "bucket %u contents", b);
                    check(false, m);
                }
                ++bad_set;
            }
        }
        check(bad_count == 0, "every bucket count matches the CPU reference");
        check(bad_set == 0, "every bucket holds exactly the right records");
        check(total == kN, "every element landed somewhere exactly once");
        if (verbose())
            std::printf("     %zu records across %u buckets, drops=%u\n",
                        total, kNumBuckets, gdrops[1]);

        // ---- optional: the full 2^25 timing, for the emit-bandwidth question ----
        // Off by default so ctest stays fast; MXBM_BENCH=1 turns it on. This is the
        // first Apple-vs-Ada datapoint and it is what says whether the row-bucket
        // record widths need re-tuning before the fused rounds are written.
        if (std::getenv("MXBM_BENCH")) {
            const uint32_t bigN    = 1u << 25;
            const uint32_t bigBits = 20;
            const uint32_t bigNB   = 1u << bigBits;
            const uint32_t bigCap  = 64;
            id<MTLBuffer> bc = h.buffer((size_t)bigNB * sizeof(uint32_t));
            id<MTLBuffer> be = h.buffer((size_t)bigNB * bigCap * sizeof(uint64_t));
            uint32_t z = 0, n2 = bigN, bb = bigBits, bcap = bigCap;
            id<MTLBuffer> q1 = h.buffer_with(&z,    sizeof z);
            id<MTLBuffer> q2 = h.buffer_with(&n2,   sizeof n2);
            id<MTLBuffer> q3 = h.buffer_with(&bb,   sizeof bb);
            id<MTLBuffer> q4 = h.buffer_with(&bcap, sizeof bcap);

            double best = 1e30;
            for (int rep = 0; rep < 5; ++rep) {
                std::memset(bc.contents, 0, (size_t)bigNB * sizeof(uint32_t));
                std::memset(dbuf.contents, 0, 8 * sizeof(uint32_t));
                const auto t0 = std::chrono::steady_clock::now();
                h.run(ps, bigN, @[ pbuf, q1, q2, q3, q4, bc, be, dbuf ]);
                const auto t1 = std::chrono::steady_clock::now();
                const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                if (ms < best) best = ms;
            }
            std::printf("  entry_scatter 2^25: %.1f ms (best of 5), drops=%u\n",
                        best, ((const uint32_t*)dbuf.contents)[1]);
        }

        return summary("metal_entry");
    }
}
