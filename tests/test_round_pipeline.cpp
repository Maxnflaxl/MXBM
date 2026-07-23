// The 5-round pipeline driver (run_pipeline) + survivor scan, vs:
//  (a) a deterministic, hand-crafted r=5 survivor (exercises the r=5
//      special path -- Lmix=288, Lout=24, work[0] output, is_zero -- and
//      survivor_scan itself -- end to end, on real hardware);
//  (b) a chained mini-search: run_pipeline() over 2^20 REAL seeds vs a
//      from-scratch CPU chain of 5x cpu_round, comparing every round's GPU
//      `out` count against the CPU count EXACTLY, plus 0 bucket/pair drops.
//
// Both sections build on already-reviewed, unmodified T2-T4 building blocks
// (mix_seeds/mix_level/scatter/match/run_single_round) -- this file adds and
// exercises ONLY survivor_scan (kernel + host wrapper) and run_pipeline (the
// new driver). Per the task's own framing: if (b) ever mismatches, the bug
// is in THIS file's driver/chaining, not in T2-T4's kernels.
#include "gpu/cl_runtime.h"
#include "gpu/round_pipeline.h"
#include "generated/mxbm_kernels.h"
#include "round_ref.h"
#include "bh3_primitives.h"
#include "kat_vectors.h"
#include "check.h"
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
using namespace mxbm;
using namespace mxbm::gpu;

// ---------------------------------------------------------------------
// Section (a): deterministic r=5 survivor via full round-4 duplication.
// ---------------------------------------------------------------------
//
// round_match's r==5 special case (Lmix=288, padNum=9, Lout=24) writes its
// children into pb.work[0] (there is no round 6 / work[6] -- see
// round_pipeline.h's doc comment on match()). bh3_combine is XOR-based
// (kernels/opencl/bh3.cl / src/beamhash/bh3_primitives.h): combine(a,b,24)
// with a and b byte-identical across all 7 words XORs to exactly zero, and
// Lout=24's masking of an already-zero word stays zero -- so an all-zero
// child is the deterministic (not probabilistic) signature of "two round-4
// elements collided AND were byte-identical". To get there we need TWO
// round-4 (level-4, 16-leaf) elements with IDENTICAL raw work AND an
// IDENTICAL reconstructed leaf prefix -- NOT merely identical work with
// distinct leaves (apply_mix folds the leaf indices into w[0] bijectively;
// see tests/test_round_reconstruct.cpp's test_cpu_round_forced_collisions
// for the same lesson at r=1 -- distinct leaves + equal work does NOT
// collide).
//
// build_level4_history() below builds ONE small, consistent 4-level
// back-ref chain (leaves -> level-1 -> level-2 -> level-3), fully populated
// for all 16 leaves (0..15) even though round 5's DFS (padNum(5)==9) only
// ever needs the first 9 in pre-order -- built in full anyway so this is a
// genuine tree, not a truncated stub. The caller then stages TWO level-4
// slots (0 and 1) that point at the SAME two level-3 children: a SHARED
// back-ref subtree, so round_mix's on-device DFS reconstructs byte-identical
// leaves for both BY CONSTRUCTION (no risk of the two views drifting apart,
// the same reasoning test_round_match.cpp's build_level2_fixture uses one
// source of truth for). Their raw (pre-mix) work is staged identically too.
struct Level4History {
    std::vector<uint32_t> left, right;  // 5*capacity each; rows 0..2 populated here
    uint64_t w[7];                      // raw (pre-mix) work, shared by both level-4 slots
};

static Level4History build_level4_history(uint32_t capacity) {
    Level4History h;
    h.left.assign(5 * (size_t)capacity, 0u);
    h.right.assign(5 * (size_t)capacity, 0u);

    // Level-1 row (offset 0): 8 slots, each a literal leaf pair {2s,2s+1}.
    for (uint32_t s = 0; s < 8; ++s) {
        h.left [0 * (size_t)capacity + s] = 2 * s;
        h.right[0 * (size_t)capacity + s] = 2 * s + 1;
    }
    // Level-2 row (offset capacity): 4 slots, each 2 level-1 slots {2s,2s+1}.
    for (uint32_t s = 0; s < 4; ++s) {
        h.left [1 * (size_t)capacity + s] = 2 * s;
        h.right[1 * (size_t)capacity + s] = 2 * s + 1;
    }
    // Level-3 row (offset 2*capacity): 2 slots, each 2 level-2 slots {2s,2s+1}.
    for (uint32_t s = 0; s < 2; ++s) {
        h.left [2 * (size_t)capacity + s] = 2 * s;
        h.right[2 * (size_t)capacity + s] = 2 * s + 1;
    }
    // Level-4 row (offset 3*capacity) is intentionally NOT populated here --
    // staging its two (duplicated) slots is the test's whole point, done by
    // the caller.

    bh3::Elem seed;
    bh3::seed_element(kat::prePow, 555555u, seed);   // deterministic, KAT-based raw work
    for (int w = 0; w < 7; ++w) h.w[w] = seed.w[w];
    return h;
}

static void test_r5_survivor(Runtime& rt) {
    section("survivor_scan r=5: two FULLY duplicated round-4 elements (shared back-ref subtree) -> is_zero child");

    const uint32_t capacity = 16;   // >= 8 (level-1 row's max slot); small on purpose
    Level4History h = build_level4_history(capacity);

    // Level-4 row: slots 0 and 1 BOTH point at level-3 slots {0,1} -- a
    // SHARED subtree (not two independently-built-but-equal subtrees), so
    // the DFS reconstructs byte-identical leaves for both by construction.
    h.left [3 * (size_t)capacity + 0] = 0; h.right[3 * (size_t)capacity + 0] = 1;
    h.left [3 * (size_t)capacity + 1] = 0; h.right[3 * (size_t)capacity + 1] = 1;

    // Independent CPU-side sanity check (reuses tests/round_ref.h's
    // first_leaves -- the SAME pre-order-DFS-with-early-stop oracle
    // test_round_reconstruct.cpp already validates against round_mix -- not
    // reimplemented here) BEFORE touching the GPU: both level-4 slots must
    // reconstruct the identical 9-leaf prefix.
    check(ref::padNum(5) == 9u, "r=5: padNum(5) == 9 (sanity)");
    check(ref::Lmix(5) == 288u, "r=5: Lmix(5) == 288 (sanity)");
    check(ref::Lout(5) == 24u, "r=5: Lout(5) == 24 (sanity)");
    std::vector<uint32_t> leaves0 = ref::first_leaves(h.left, h.right, capacity, /*level=*/4, /*slot=*/0, ref::padNum(5));
    std::vector<uint32_t> leaves1 = ref::first_leaves(h.left, h.right, capacity, /*level=*/4, /*slot=*/1, ref::padNum(5));
    check(leaves0.size() == 9 && leaves1.size() == 9, "r=5: reconstructed prefix has exactly 9 leaves (both slots)");
    check(leaves0 == leaves1, "r=5: both staged round-4 slots reconstruct the IDENTICAL 9-leaf prefix (sanity, CPU-side)");

    // Independent CPU-side mix check: identical raw work + identical
    // reconstructed tree -> identical mixed w[0] (bh3::apply_mix called
    // directly, not reimplemented).
    uint64_t cpuMixedW0 = 0;
    {
        bh3::Elem e0, e1;
        for (int w = 0; w < 7; ++w) { e0.w[w] = h.w[w]; e1.w[w] = h.w[w]; }
        bh3::apply_mix(e0, leaves0.data(), (uint32_t)leaves0.size(), ref::Lmix(5));
        bh3::apply_mix(e1, leaves1.data(), (uint32_t)leaves1.size(), ref::Lmix(5));
        check_eq_u64(e0.w[0], e1.w[0], "r=5: both staged elements mix to the exact same w[0] (sanity, CPU-side)");
        cpuMixedW0 = e0.w[0];
    }
    std::printf("  r=5: CPU-side sanity: both staged elements' mixed w[0] = 0x%016llx (identical)\n",
                (unsigned long long)cpuMixedW0);

    Budget bud;
    bud.bucket_bits      = 2u;
    bud.num_buckets       = 1u << bud.bucket_bits;   // 4
    bud.slots_per_bucket = 4u;
    bud.elems_per_round  = capacity;
    bud.target_elems     = capacity;
    bud.seed_batch       = capacity;

    PipelineBuffers pb = alloc_pipeline(rt, bud);
    check(pb.capacity == capacity, "r=5: pipeline capacity == 16");

    // Stage raw (pre-mix) work at work[5] slots 0 and 1 -- IDENTICAL (per
    // CORRECTION 3: full duplication, not merely equal work with distinct
    // leaves).
    std::vector<uint64_t> work5((size_t)capacity * 7, 0);
    for (int w = 0; w < 7; ++w) { work5[0 * 7 + w] = h.w[w]; work5[1 * 7 + w] = h.w[w]; }
    rt.write(pb.work[5].get(), work5.size() * 8, work5.data());
    rt.write(pb.left.get(),  h.left.size()  * 4, h.left.data());
    rt.write(pb.right.get(), h.right.size() * 4, h.right.data());

    // Level-4 row's own lead (row 3 == in_lead_off for r=5, since
    // in_lead_off=(r-2)*capacity=3*capacity): both slots share the identical
    // subtree, so their lead (tree[0] == the first leaf in pre-order ==
    // leaves0[0]) is identical too -- written explicitly (not left at the
    // zero-init default) so match()'s tie-break reads a deliberate value.
    std::vector<uint32_t> lead(5 * (size_t)capacity, 0u);
    lead[3 * (size_t)capacity + 0] = leaves0[0];
    lead[3 * (size_t)capacity + 1] = leaves0[0];
    rt.write(pb.lead.get(), lead.size() * 4, lead.data());

    const uint32_t N = 2;   // round 5's own resident element count (the 2 staged slots)
    mix_level(rt, pb, /*r=*/5, N);

    // Confirm the GPU's own mix agrees with the CPU sanity check above --
    // both slots' full 7-word Elem must be byte-identical after mixing
    // (round_mix only rewrites w[0]; w[1..6] were staged identical and are
    // untouched).
    {
        std::vector<uint64_t> mixed(2 * 7);
        rt.read(pb.work[5].get(), mixed.size() * 8, mixed.data());
        bool identical = true;
        for (int w = 0; w < 7; ++w) if (mixed[0 * 7 + w] != mixed[1 * 7 + w]) identical = false;
        check(identical, "r=5: GPU-mixed slot0 and slot1 are byte-identical (all 7 words)");
        check_eq_u64(mixed[0], cpuMixedW0, "r=5: GPU-mixed w[0] == the CPU sanity check's w[0]");
        std::printf("  r=5: GPU-mixed w[0] slot0=0x%016llx slot1=0x%016llx\n",
                    (unsigned long long)mixed[0], (unsigned long long)mixed[7]);
    }

    scatter(rt, pb, bud, N, /*workIndex=*/5);
    {
        uint32_t counters[4];
        rt.read(pb.counters.get(), sizeof counters, counters);
        check(counters[1] == 0u, "r=5: scatter counters[1] (bucket_drops) == 0");
    }

    uint32_t pair_drops = 0;
    uint32_t outN = match(rt, pb, bud, /*r=*/5, /*inN=*/N, pair_drops);
    std::printf("  r=5: match produced outN=%u pair_drops=%u (expect outN=1: the one colliding pair)\n", outN, pair_drops);
    check(pair_drops == 0u, "r=5: match counters[2] (pair_drops) == 0");
    check(outN == 1u, "r=5: match produced EXACTLY 1 child (the one colliding pair)");

    // The child landed in pb.work[0] (r==5's special-case out_work), NOT
    // pb.work[6] (which doesn't exist -- CORRECTION 1).
    std::vector<uint64_t> childWork(7, 0);
    rt.read(pb.work[0].get(), childWork.size() * 8, childWork.data());
    bool isZero = true;
    for (int w = 0; w < 7; ++w) if (childWork[w] != 0ull) isZero = false;
    check(isZero, "r=5: the one child's work is ALL-ZERO across all 7 words (bh3_combine(x,x,24): XOR==0)");

    // Back-ref row 4 (out_off=(r-1)*capacity=4*capacity): left/right must be
    // {0,1} with left = the SMALLER slot (both leads tie, so match()'s
    // tie-break falls to slot order).
    std::vector<uint32_t> allLeft(5 * (size_t)capacity), allRight(5 * (size_t)capacity), allLead(5 * (size_t)capacity);
    rt.read(pb.left.get(),  allLeft.size()  * 4, allLeft.data());
    rt.read(pb.right.get(), allRight.size() * 4, allRight.data());
    rt.read(pb.lead.get(),  allLead.size()  * 4, allLead.data());
    const uint32_t outOff = (5 - 1) * capacity;   // 4*capacity
    check(allLeft[outOff + 0] == 0u && allRight[outOff + 0] == 1u,
          "r=5: survivor's back-ref row 4 == {left=0, right=1} (left = smaller slot, tie-break on equal lead)");
    std::printf("  r=5: survivor back-ref left=%u right=%u lead=%u\n",
                allLeft[outOff + 0], allRight[outOff + 0], allLead[outOff + 0]);

    // -----------------------------------------------------------------
    // survivor_scan over pb.work[0] (CORRECTION 1: the round-5 OUTPUT, NOT
    // work[5]), N = the round-5 child count (outN, == 1 here).
    // -----------------------------------------------------------------
    std::vector<uint32_t> survivorSlots;
    uint32_t survivors = survivor_scan(rt, pb, outN, survivorSlots);
    std::printf("  r=5: survivor_scan found %u survivor(s) (expect exactly 1)\n", survivors);
    check(survivors == 1u, "r=5: survivor_scan found EXACTLY 1 survivor");
    check(survivorSlots.size() == 1 && survivorSlots[0] == 0u,
          "r=5: the survivor's work[0] slot == 0 (the only child match() produced)");

    // Re-run survivor_scan on the SAME PipelineBuffers to confirm
    // counters[3] is zeroed each call (CORRECTION 2), not accumulated.
    std::vector<uint32_t> survivorSlots2;
    uint32_t survivors2 = survivor_scan(rt, pb, outN, survivorSlots2);
    check(survivors2 == 1u,
          "r=5: survivor_scan is idempotent -- a second call on the same buffers still reports EXACTLY 1 (counters[3] zeroed each call, not accumulated)");

    std::string src(kRoundClSource);
    check(src.find("survivor_scan") != std::string::npos, "kRoundClSource contains survivor_scan");
}

// ---------------------------------------------------------------------
// Section (b): chained mini-search vs CPU (run_pipeline end to end).
// ---------------------------------------------------------------------
static void test_full_pipeline_vs_cpu(Runtime& rt) {
    section("run_pipeline: 2^20 real seeds, full 5-round GPU driver vs chained cpu_round x5");

    const uint32_t N = 1u << 20;

    Budget bud;
    bud.bucket_bits      = 15u;
    bud.num_buckets       = 1u << bud.bucket_bits;   // 32768
    bud.slots_per_bucket = 64u;                      // mean 32/bucket -> no drops (per task)
    bud.elems_per_round  = N;
    bud.target_elems     = N;
    bud.seed_batch       = N;

    PipelineBuffers pb = alloc_pipeline(rt, bud);
    check(pb.capacity == N, "b: pipeline capacity == 2^20");

    std::printf("  b: running GPU run_pipeline() over %u seeds...\n", N);
    PipelineResult res = run_pipeline(rt, pb, bud, kat::prePow);

    std::printf("  b: computing CPU reference chain (cpu_round x5) over %u seeds...\n", N);
    std::vector<uint32_t> indices(N);
    for (uint32_t i = 0; i < N; ++i) indices[i] = i;
    std::vector<ref::RElem> cpuElems = ref::make_seeds(kat::prePow, indices);

    uint32_t cpuCounts[5];
    for (int r = 1; r <= 5; ++r) {
        cpuElems = ref::cpu_round(cpuElems, r);
        cpuCounts[r - 1] = (uint32_t)cpuElems.size();
        std::printf("  b: CPU round %d -> %u children\n", r, cpuCounts[r - 1]);
    }

    std::printf("  round |    GPU out |    CPU out | bucketDrops |   pairDrops\n");
    for (int r = 1; r <= 5; ++r) {
        const RoundStats& st = res.rounds[r - 1];
        std::printf("    r%d  | %10u | %10u | %12u | %12u\n", r, st.out, cpuCounts[r - 1], st.bucket_drops, st.pair_drops);

        char lbl[80];
        // GPU's own chaining: round r's `in` must equal round (r-1)'s CPU
        // (== GPU, by the exact-count check below, inductively) out count --
        // the task's own named failure mode ("driver's round chaining").
        uint32_t expectedIn = (r == 1) ? N : cpuCounts[r - 2];
        std::snprintf(lbl, sizeof lbl, "b: round %d GPU in == previous round's out count (chaining)", r);
        check(st.in == expectedIn, lbl);

        std::snprintf(lbl, sizeof lbl, "b: round %d GPU out == CPU out EXACTLY", r);
        check(st.out == cpuCounts[r - 1], lbl);
        std::snprintf(lbl, sizeof lbl, "b: round %d bucket_drops == 0", r);
        check(st.bucket_drops == 0u, lbl);
        std::snprintf(lbl, sizeof lbl, "b: round %d pair_drops == 0", r);
        check(st.pair_drops == 0u, lbl);
    }

    std::printf("  b: run_pipeline reported %u survivor(s) (is_zero children among round 5's %u; expected ~0 for real, distinct data)\n",
                res.survivors, res.rounds[4].out);
    check(res.survivor_slots.size() == res.survivors, "b: survivor_slots.size() == the reported survivor count");
}

int main() {
    if (!Runtime::any_device_available()) { std::printf("SKIP: no OpenCL device\n"); return 0; }
    Runtime rt;
    std::printf("  device: %s | CUs=%u\n", rt.device().name.c_str(), rt.device().compute_units);
    test_r5_survivor(rt);
    test_full_pipeline_vs_cpu(rt);
    return summary("round_pipeline");
}
