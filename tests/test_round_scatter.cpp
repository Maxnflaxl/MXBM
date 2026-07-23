// round_scatter kernel (+ host scatter()) vs a hand-computed host oracle.
// Uses a HAND-CONSTRUCTED small Budget (bucket_bits=6 -> 64 buckets,
// slots_per_bucket=8) -- NOT compute_budget -- so a bucket can be driven past
// capacity under full control: bucket_count[b] = key(work[g*7]) >> (24-6).
// Elements: 63 of the 64 buckets each get a small (1..3), non-overflowing
// "background" group with controlled/distinct keys (exercises ordinary
// placement + multi-member set comparison); the remaining bucket is forced to
// receive exactly 20 elements against its 8-slot capacity, so exactly 8
// survive and 12 are dropped. Bucket order is nondeterministic under
// atomic_inc, so membership is compared as SETS, never sequences.
#include "gpu/cl_runtime.h"
#include "gpu/round_pipeline.h"
#include "generated/mxbm_kernels.h"
#include "check.h"
#include <cstdint>
#include <cstdio>
#include <set>
#include <string>
#include <vector>
using namespace mxbm;
using namespace mxbm::gpu;

int main() {
    if (!Runtime::any_device_available()) { std::printf("SKIP: no OpenCL device\n"); return 0; }
    Runtime rt;
    std::printf("  device: %s | CUs=%u\n", rt.device().name.c_str(), rt.device().compute_units);
    section("round_scatter: bucket placement (set-compared) + forced-overflow drop count");

    const uint32_t BUCKET_BITS     = 6u;
    const uint32_t NUM_BUCKETS     = 1u << BUCKET_BITS;   // 64
    const uint32_t SLOTS           = 8u;
    const uint32_t OVERFLOW_BUCKET = 31u;                 // arbitrary, mid-range
    const uint32_t OVERFLOW_N      = 20u;                 // forced into OVERFLOW_BUCKET
    const uint32_t EXPECT_DROPS    = OVERFLOW_N - SLOTS;  // 20-8 = 12
    const uint32_t KEY_LOW_BITS    = 24u - BUCKET_BITS;   // 18 low bits are free (disambiguator)

    std::vector<std::vector<uint32_t>> expected(NUM_BUCKETS);  // bucket -> expected member g's (non-overflow buckets only)
    std::vector<uint32_t> keys;                                 // key per element g (index == g)

    auto push_elem = [&](uint32_t bucket, uint32_t disambiguator) -> uint32_t {
        uint32_t g = (uint32_t)keys.size();
        uint32_t key = (bucket << KEY_LOW_BITS) | (disambiguator & ((1u << KEY_LOW_BITS) - 1u));
        keys.push_back(key);
        return g;
    };

    // Background: every bucket except OVERFLOW_BUCKET gets 1..3 members,
    // always well under SLOTS, so no drops occur there.
    for (uint32_t b = 0; b < NUM_BUCKETS; ++b) {
        if (b == OVERFLOW_BUCKET) continue;
        uint32_t count = 1u + (b % 3u);
        for (uint32_t i = 0; i < count; ++i) expected[b].push_back(push_elem(b, i));
    }
    // Forced overflow: exactly OVERFLOW_N elements, all mapping to OVERFLOW_BUCKET.
    std::vector<uint32_t> overflow_candidates;
    for (uint32_t i = 0; i < OVERFLOW_N; ++i)
        overflow_candidates.push_back(push_elem(OVERFLOW_BUCKET, 1000u + i));

    const uint32_t N = (uint32_t)keys.size();
    std::printf("  N=%u num_buckets=%u overflow_bucket=%u overflow_n=%u expect_drops=%u\n",
                N, NUM_BUCKETS, OVERFLOW_BUCKET, OVERFLOW_N, EXPECT_DROPS);

    // Hand-built Budget -- deliberately NOT compute_budget, so the overflow
    // case above is exercised under full, deterministic control.
    Budget bud;
    bud.bucket_bits      = BUCKET_BITS;
    bud.num_buckets      = NUM_BUCKETS;
    bud.slots_per_bucket = SLOTS;
    bud.elems_per_round  = N;
    bud.target_elems     = N;
    bud.seed_batch       = N;

    PipelineBuffers pb = alloc_pipeline(rt, bud);
    check(pb.capacity == N, "pipeline capacity == N");

    // Stage work[1]: round_scatter only reads work[g*7] (the collision-key word).
    std::vector<uint64_t> work((size_t)N * 7, 0);
    for (uint32_t g = 0; g < N; ++g) work[(size_t)g*7] = (uint64_t)keys[g];
    rt.write(pb.work[1].get(), work.size() * 8, work.data());

    // Verifies the current on-device scatter state against `expected` /
    // `overflow_candidates` above. Shared by both scatter passes below so
    // pass 2 is checked with the exact same logic as pass 1, not a
    // hand-duplicated copy that could quietly drift.
    auto verify_scatter_state = [&](const char* pass) {
        std::vector<uint32_t> bucket_count(NUM_BUCKETS);
        rt.read(pb.bucket_count.get(), bucket_count.size() * 4, bucket_count.data());
        std::vector<uint32_t> bucket_slots((size_t)NUM_BUCKETS * SLOTS);
        rt.read(pb.bucket_slots.get(), bucket_slots.size() * 4, bucket_slots.data());
        std::vector<uint32_t> counters(4);
        rt.read(pb.counters.get(), counters.size() * 4, counters.data());

        // --- Non-overflow buckets: exact set equality (order-independent). ---
        int mismatches = 0;
        for (uint32_t b = 0; b < NUM_BUCKETS; ++b) {
            if (b == OVERFLOW_BUCKET) continue;
            std::set<uint32_t> want(expected[b].begin(), expected[b].end());
            uint32_t cnt = bucket_count[b];
            bool countOk = (cnt == want.size());
            uint32_t readCnt = (cnt <= SLOTS) ? cnt : SLOTS;   // guard against OOB read on unexpected overflow
            std::set<uint32_t> got;
            for (uint32_t i = 0; i < readCnt; ++i) got.insert(bucket_slots[(size_t)b*SLOTS + i]);
            if (!countOk || got != want) {
                if (mismatches < 5) std::printf("  [%s] MISMATCH bucket=%u count=%u expected_size=%zu\n",
                                                 pass, b, bucket_count[b], want.size());
                ++mismatches;
            }
        }
        check(mismatches == 0,
              (std::string(pass) + ": all 63 non-overflow buckets: contents match expected set exactly").c_str());

        // --- Overflow bucket: bucket_count is the RAW (uncapped) atomic
        //     arrival tally; only the first SLOTS winners (nondeterministic
        //     race order) are actually written into bucket_slots. ---
        check(bucket_count[OVERFLOW_BUCKET] == OVERFLOW_N,
              (std::string(pass) + ": overflow bucket_count == 20 (raw atomic arrivals, uncapped by slots)").c_str());
        std::set<uint32_t> overflow_want(overflow_candidates.begin(), overflow_candidates.end());
        std::set<uint32_t> overflow_got;
        for (uint32_t i = 0; i < SLOTS; ++i) overflow_got.insert(bucket_slots[(size_t)OVERFLOW_BUCKET*SLOTS + i]);
        check(overflow_got.size() == SLOTS,
              (std::string(pass) + ": overflow bucket: exactly 8 distinct slots written (atomic_inc gives unique pos)").c_str());
        bool subset = true;
        for (uint32_t g : overflow_got) if (!overflow_want.count(g)) subset = false;
        check(subset, (std::string(pass) + ": overflow bucket: every survivor is one of the 20 forced candidates").c_str());

        // --- Drop counter: exact. ---
        std::printf("  [%s] counters[1] (bucket_drops) = %u (expected %u)\n", pass, counters[1], EXPECT_DROPS);
        check(counters[1] == EXPECT_DROPS,
              (std::string(pass) + ": counters[1] (bucket_drops) == 12 (20 forced into an 8-slot bucket)").c_str());
        check(counters[0] == 0u && counters[2] == 0u && counters[3] == 0u,
              (std::string(pass) + ": counters[0],[2],[3] untouched by scatter (zeroed by fill_u32, not written by round_scatter)").c_str());
    };

    scatter(rt, pb, bud, N, /*workIndex=*/1);
    verify_scatter_state("pass 1 (fresh buffers)");

    // --- Second scatter pass on the SAME PipelineBuffers, without
    // reallocating. This is what makes the test load-bearing against
    // fill_u32, and models the real production pattern: bucket_count/
    // bucket_slots/counters are reused round-over-round, never freshly
    // allocated per round, so a fresh never-written cl_mem's OS zero-page
    // luck (which made bucket_count/counters look zeroed above regardless
    // of whether fill_u32 actually ran) cannot mask a broken fill_u32 here.
    // Pass 1 above left bucket_count/counters populated with real non-zero
    // values via the kernel's own atomic writes (e.g.
    // bucket_count[OVERFLOW_BUCKET]==20, counters[1]==12) -- not a poison
    // value fill_u32 itself wrote, which would be circular (a broken
    // fill_u32 wouldn't land the poison either). scatter() calls fill_u32
    // to re-zero bucket_count/counters before round_scatter's atomic_inc
    // runs again; round_scatter only reads work[], never writes it, so
    // work[1] is byte-identical to pass 1. With a correct fill_u32, pass 2
    // must therefore reproduce the same bucket membership sets and the same
    // drop count as pass 1. If fill_u32 were a no-op, pass 2's atomic_inc
    // would accumulate ON TOP of pass 1's stale counts instead of starting
    // from zero: e.g. bucket_count[OVERFLOW_BUCKET] would start pass 2's
    // atomic_inc sequence at 20 instead of 0, so all 20 of pass 2's overflow
    // arrivals would land past the 8-slot cutoff, and counters[1] would
    // read 32 (12 stale carried over + 20 newly dropped), not 12.
    scatter(rt, pb, bud, N, /*workIndex=*/1);
    verify_scatter_state("pass 2 (reused buffers, must be re-zeroed by fill_u32)");

    // Sanity: the kernel source really was embedded.
    std::string src(kRoundClSource);
    check(src.find("round_scatter") != std::string::npos, "kRoundClSource contains round_scatter");

    return summary("round_scatter");
}
