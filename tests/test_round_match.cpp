// round_match kernel (+ host match()/run_single_round()) vs the CPU oracle
// (tests/round_ref.h::cpu_round), run as a full per-round GPU-vs-CPU
// cross-check. Two sections:
//  (a) r=1: 4096 REAL seeds, forced collisions by duplicating WHOLE elements
//      (work AND tree -- copying work alone does NOT collide: apply_mix
//      folds the leaf index into w[0] bijectively, so equal-w-different-leaf
//      pairs mix to DIFFERENT keys; see test_round_reconstruct.cpp's
//      test_cpu_round_forced_collisions for the same technique), PLUS one
//      algebraically-forced triple of genuinely DISTINCT leaves (same
//      w[0]-solving technique as test_round_reconstruct.cpp's
//      test_cpu_round_all_pairs_in_group -- see the comment at its
//      construction below for why a *duplicated* triple would not do: with
//      lead_identity=1, "lead" is read as the element's own SLOT, which is
//      only a valid stand-in for the true lead (tree[0]) when every member
//      is a genuinely distinct leaf, exactly as production round 1 always
//      is). Mix is done on the CPU (cpu_mix_w0) and the MIXED work uploaded
//      directly, so this section isolates scatter+match from
//      mix_seeds/round_mix.
//  (b) r=3: a synthetic 2-level history (256 level-2 elements, 4 leaves
//      each, PLUS 2 more forming a lead-inversion pair -- see below), built
//      by build_level2_fixture() below as ONE source of truth that emits
//      BOTH the RElem-with-.tree view cpu_round needs AND the flat
//      consolidated left/right/lead device arrays round_mix's on-device
//      back-ref DFS needs -- so the two views cannot structurally drift.
//      Forces 128 colliding pairs by duplicating whole level-2 elements
//      (RElem AND the matching flat back-ref/lead entries together), which
//      makes every forced pair an EQUAL-LEAD pair too, exercising match's
//      (lead,slot) tie-break end-to-end, 128 times over -- but never its
//      strict lead COMPARISON (the `lb < la` branch of match's ordering
//      condition), since whole-element duplication always copies tree[0]
//      along with everything else, so slot-order and lead-order can never
//      disagree for any of those 128 pairs. Two further elements (slots
//      256,257, appended after the loop) close that gap: algebraically
//      forced (same w[0]-solving technique as the r=1 triple below,
//      generalized to r=3's Lmix=400/padNum=4 tree fold) to collide despite
//      DIFFERENT leads, with the HIGHER slot given the SMALLER lead -- the
//      one shape a kernel that ordered by raw slot instead of looked-up lead
//      cannot be told apart from correct by any of the 128 tie pairs above.
//      Exercises round_mix's reconstruction + a mid-round Lout (376) +
//      lead_identity=0 (all_lead_in lookup), none of which section (a)
//      touches.
//
// Both sections compare GPU vs CPU as MULTISETS (bucket/scatter emission
// order is nondeterministic under atomic_inc) -- sort both by
// (work[0..6],left,right) and compare exactly; never compare emission order.
#include "gpu/cl_runtime.h"
#include "gpu/round_pipeline.h"
#include "generated/mxbm_kernels.h"
#include "round_ref.h"
#include "bh3_primitives.h"
#include "kat_vectors.h"
#include "check.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <map>
#include <string>
#include <utility>
#include <vector>
using namespace mxbm;
using namespace mxbm::gpu;

// Deterministic pseudo-random (no <random> dependence): splitmix64.
static uint64_t sm(uint64_t& s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Multiset comparison key: a child's full work vector + the (left,right)
// slot pair that produced it. Using the ORDERED pair (not a {left,right}
// set) is the STRONGER check -- both cpu_round and round_match order pairs
// by the identical (lead, then slot) rule, so a correct kernel's (left,right)
// already agrees with the CPU oracle's (leftSlot,rightSlot); a pure
// left/right swap bug would show up here as a mismatch rather than being
// masked by set-style comparison. The standalone ordering-invariant checks
// below additionally verify lead(left) <= lead(right) directly.
struct ChildKey {
    uint64_t w[7];
    uint32_t left, right;
    bool operator<(const ChildKey& o) const {
        for (int i = 0; i < 7; ++i) if (w[i] != o.w[i]) return w[i] < o.w[i];
        if (left != o.left) return left < o.left;
        return right < o.right;
    }
    bool operator==(const ChildKey& o) const {
        for (int i = 0; i < 7; ++i) if (w[i] != o.w[i]) return false;
        return left == o.left && right == o.right;
    }
};

static std::vector<ChildKey> cpu_keys(const std::vector<ref::RElem>& children) {
    std::vector<ChildKey> out;
    out.reserve(children.size());
    for (const auto& c : children) {
        ChildKey k{};
        for (int w = 0; w < 7; ++w) k.w[w] = c.w[w];
        k.left = c.leftSlot; k.right = c.rightSlot;
        out.push_back(k);
    }
    return out;
}
static std::vector<ChildKey> gpu_keys(const std::vector<uint64_t>& work,
                                       const std::vector<uint32_t>& left,
                                       const std::vector<uint32_t>& right,
                                       uint32_t n) {
    std::vector<ChildKey> out;
    out.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        ChildKey k{};
        for (int w = 0; w < 7; ++w) k.w[w] = work[(size_t)i*7 + w];
        k.left = left[i]; k.right = right[i];
        out.push_back(k);
    }
    return out;
}
// Prints up to maxPrint entries present in `a` but not `b`, honoring
// multiplicity (std::set_difference on sorted ranges IS multiset
// difference). Debug aid for the "exact multiset diff" the task asks for if
// a mismatch is ever hit; both inputs must already be sorted.
static size_t report_diff(const char* label, const std::vector<ChildKey>& a, const std::vector<ChildKey>& b, int maxPrint) {
    std::vector<ChildKey> diff;
    std::set_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(diff));
    for (size_t i = 0; i < diff.size() && (int)i < maxPrint; ++i)
        std::printf("    %s[%zu]: left=%u right=%u w0=0x%016llx\n",
                    label, i, diff[i].left, diff[i].right, (unsigned long long)diff[i].w[0]);
    return diff.size();
}
// Shared multiset-equality assertion: sorts copies of both key vectors,
// compares exactly, and on mismatch prints the symmetric difference (which
// children one side has that the other doesn't) before failing.
static void check_multiset_eq(std::vector<ChildKey> cpu, std::vector<ChildKey> gpu, const char* label) {
    std::sort(cpu.begin(), cpu.end());
    std::sort(gpu.begin(), gpu.end());
    bool sizeOk = (cpu.size() == gpu.size());
    bool contentOk = sizeOk && std::equal(cpu.begin(), cpu.end(), gpu.begin());
    if (!contentOk) {
        std::printf("  %s MISMATCH: cpu_count=%zu gpu_count=%zu\n", label, cpu.size(), gpu.size());
        size_t onlyCpu = report_diff("cpu-only", cpu, gpu, 10);
        size_t onlyGpu = report_diff("gpu-only", gpu, cpu, 10);
        std::printf("  %s cpu-only=%zu gpu-only=%zu\n", label, onlyCpu, onlyGpu);
    }
    char msg[160];
    std::snprintf(msg, sizeof msg, "%s: GPU child multiset == CPU child multiset (sorted, exact)", label);
    check(contentOk, msg);
}

// ---------------------------------------------------------------------
// Section (a): r=1.
// ---------------------------------------------------------------------
static void test_r1_match(Runtime& rt) {
    section("round_match r=1: 4096 real seeds, forced pairs+triple, vs cpu_round(.,1)");

    const uint32_t N = 4096;
    std::vector<uint32_t> indices(N);
    for (uint32_t i = 0; i < N; ++i) indices[i] = i;
    std::vector<ref::RElem> in = ref::make_seeds(kat::prePow, indices);

    // Force 1024 colliding PAIRS by duplicating WHOLE elements (work AND
    // tree -- the T1-proven technique; work alone does not collide). Every
    // pair copies the LARGER slot from the SMALLER (its "anchor"), so the
    // anchor is always one of the two members being compared -- this is
    // what keeps GPU's lead_identity=1 (lead==own slot) exactly equal to
    // the shared tree[0] for every one of these pairs (min(slot) == the
    // anchor's slot == the anchor's tree[0], since the anchor is a real,
    // untouched seed with tree[0]==its own slot).
    for (uint32_t i = 0; i < 1024; ++i) in[2*i + 1] = in[2*i];

    // One TRIPLE, algebraically forced (NOT a duplicated element): 3
    // genuinely DISTINCT real seeds, picked from the untouched background
    // range (2048..4095, never touched by the pairing loop above), held to
    // one shared round-1 mixed w[0] by solving each element's w[0] against a
    // common target T (same technique as test_round_reconstruct.cpp's
    // test_cpu_round_all_pairs_in_group) -- round 1's Lmix=448 fold is
    // result=rotl64(S(w),..)+rotl64(tree[0],40) with S(w) depending only on
    // w[1..6] (held equal across the three) and w[0] (solved per-element),
    // so this is exactly invertible. .tree is left untouched (still
    // {idx} from make_seeds), so tree[0]==own slot holds for EACH of the
    // three, exactly like production round 1. A duplicated triple (copying
    // a whole element twice, as this test originally tried) would instead
    // make two non-anchor slots share the anchor's tree[0] while keeping
    // THEIR OWN distinct slot numbers -- for the pair formed by those two
    // non-anchor slots, GPU's min(slot) recovers neither slot's own
    // (meaningless, post-copy) tree[0] nor the shared anchor lead, a
    // genuine cross-check failure that traces to the test construction
    // violating round 1's slot==tree[0] invariant, not to the kernel: this
    // 3-way accidental-collision shape is normal in a real run (birthday
    // multi-collisions among 2^25 DISTINCT leaves), where lead_identity=1
    // is exact for any group size precisely because every member
    // independently satisfies tree[0]==own slot.
    {
        const uint32_t tripleIdx[3] = { 3000u, 3001u, 3002u };
        uint64_t wRest[7];
        for (int w = 0; w < 7; ++w) wRest[w] = in[tripleIdx[0]].w[w];
        uint64_t T = bh3::rotl64(wRest[0], 29) + bh3::rotl64((uint64_t)tripleIdx[0], 40);
        for (int t = 0; t < 3; ++t) {
            for (int w = 1; w < 7; ++w) in[tripleIdx[t]].w[w] = wRest[w];
            uint64_t rotatedW0 = T - bh3::rotl64((uint64_t)tripleIdx[t], 40);
            in[tripleIdx[t]].w[0] = bh3::rotl64(rotatedW0, 64 - 29);
            // in[tripleIdx[t]].tree is untouched: still {tripleIdx[t]}.
        }
        uint64_t m0 = ref::cpu_mix_w0(in[tripleIdx[0]], 1);
        uint64_t m1 = ref::cpu_mix_w0(in[tripleIdx[1]], 1);
        uint64_t m2 = ref::cpu_mix_w0(in[tripleIdx[2]], 1);
        check(m0 == m1 && m1 == m2, "r=1: algebraic triple mixes to the exact same w[0] (sanity)");
    }

    std::vector<ref::RElem> cpuChildren = ref::cpu_round(in, 1);
    std::printf("  r=1: cpu_round produced %zu children (1027 expected: 1024 forced pairs + C(3,2)=3 from the triple)\n", cpuChildren.size());
    // Exact, not a lower bound: the fixture is fully deterministic (fixed
    // indices, no randomness) so the designed count is knowable in advance.
    // A `>=` here would silently accept a future accidental EXTRA collision
    // among the background seeds (2048..4095, minus the triple's 3) -- which
    // is exactly the failure class this test exists to catch (a stray
    // collision reintroduces an untested pair shape); `==` fails loudly
    // instead.
    check(cpuChildren.size() == 1027, "r=1: cpu_round produced EXACTLY the 1027 designed children (1024 forced pairs + 3 from the triple)");

    // GPU: mix on the CPU (cpu_mix_w0) so this section isolates scatter+match.
    std::vector<uint64_t> mixedWork((size_t)N * 7);
    for (uint32_t i = 0; i < N; ++i) {
        mixedWork[(size_t)i*7 + 0] = ref::cpu_mix_w0(in[i], 1);
        for (int w = 1; w < 7; ++w) mixedWork[(size_t)i*7 + w] = in[i].w[w];
    }

    Budget bud;
    bud.bucket_bits      = 10u;
    bud.num_buckets       = 1u << bud.bucket_bits;   // 1024
    bud.slots_per_bucket = 64u;
    bud.elems_per_round  = N;
    bud.target_elems     = N;
    bud.seed_batch       = N;

    PipelineBuffers pb = alloc_pipeline(rt, bud);
    check(pb.capacity == N, "r=1: pipeline capacity == 4096");

    rt.write(pb.work[1].get(), mixedWork.size() * 8, mixedWork.data());

    scatter(rt, pb, bud, N, /*workIndex=*/1);
    {
        uint32_t counters[4];
        rt.read(pb.counters.get(), sizeof counters, counters);
        std::printf("  r=1: scatter bucket_drops=%u (expect 0 -- 1024 buckets x 64 slots for 4096 elems)\n", counters[1]);
        check(counters[1] == 0u, "r=1: scatter counters[1] (bucket_drops) == 0");
    }

    uint32_t pair_drops = 0;
    uint32_t outN = match(rt, pb, bud, /*r=*/1, /*inN=*/N, pair_drops);
    std::printf("  r=1: match produced outN=%u pair_drops=%u\n", outN, pair_drops);
    check(pair_drops == 0u, "r=1: match counters[2] (pair_drops) == 0");
    check(outN == cpuChildren.size(), "r=1: GPU child count == CPU child count");

    // Round 1's children land in pb.work[2] (r+1), back-ref row (r-1)*capacity == 0.
    std::vector<uint64_t> gpuWork((size_t)outN * 7);
    rt.read(pb.work[2].get(), gpuWork.size() * 8, gpuWork.data());
    std::vector<uint32_t> allLeft(5 * (size_t)pb.capacity), allRight(5 * (size_t)pb.capacity), allLead(5 * (size_t)pb.capacity);
    rt.read(pb.left.get(),  allLeft.size()  * 4, allLeft.data());
    rt.read(pb.right.get(), allRight.size() * 4, allRight.data());
    rt.read(pb.lead.get(),  allLead.size()  * 4, allLead.data());
    std::vector<uint32_t> gpuLeft(outN), gpuRight(outN), gpuLead(outN);
    for (uint32_t i = 0; i < outN; ++i) {
        gpuLeft[i]  = allLeft[i];
        gpuRight[i] = allRight[i];
        gpuLead[i]  = allLead[i];
    }

    check_multiset_eq(cpu_keys(cpuChildren), gpu_keys(gpuWork, gpuLeft, gpuRight, outN), "r=1");

    // Cross-reference GPU lead[oi] against the CPU child's tree[0] for the
    // SAME (left,right) pair, and verify the ordering invariant directly
    // (r=1: lead_identity=1 -> lead(slot)==slot, and slots within one bucket
    // are always distinct, so left < right strictly -- brief's own note that
    // an equal-lead pair is impossible at r=1).
    std::map<std::pair<uint32_t,uint32_t>, uint32_t> cpuLeadByPair;
    for (const auto& c : cpuChildren) cpuLeadByPair[{c.leftSlot, c.rightSlot}] = c.tree[0];
    int leadMismatches = 0, orderMismatches = 0, notFound = 0;
    for (uint32_t i = 0; i < outN; ++i) {
        if (!(gpuLeft[i] < gpuRight[i])) ++orderMismatches;
        auto it = cpuLeadByPair.find({gpuLeft[i], gpuRight[i]});
        if (it == cpuLeadByPair.end()) { ++notFound; continue; }
        if (gpuLead[i] != it->second) ++leadMismatches;
    }
    check(notFound == 0, "r=1: every GPU (left,right) pair matches some CPU child (same pair identity)");
    check(orderMismatches == 0, "r=1: every GPU child has left < right (lead_identity: lead==slot, all distinct)");
    check(leadMismatches == 0, "r=1: every GPU child's lead == the CPU child's tree[0] for the same pair");

    // Smoke-exercise run_single_round() too, not just the split-out
    // scatter()+match() calls above -- r==1 exercises its "mix already done
    // by the caller" branch (run_single_round's own `if (r>=2)` skips
    // mix_level for r==1, matching mix_seeds/round1_mix_seeds being the
    // CALLER's responsibility per its contract).
    {
        PipelineBuffers pb2 = alloc_pipeline(rt, bud);
        rt.write(pb2.work[1].get(), mixedWork.size() * 8, mixedWork.data());
        RoundStats stats;
        uint32_t outN2 = run_single_round(rt, pb2, bud, /*r=*/1, N, stats);
        check(outN2 == outN, "r=1: run_single_round() child count == direct match() child count");
        check(stats.in == N && stats.out == outN && stats.bucket_drops == 0 && stats.pair_drops == 0,
              "r=1: run_single_round() RoundStats {in,out,bucket_drops,pair_drops} correct");
    }

    std::string src(kRoundClSource);
    check(src.find("round_match") != std::string::npos, "kRoundClSource contains round_match");
}

// ---------------------------------------------------------------------
// Section (b): r=3.
// ---------------------------------------------------------------------

// Single source of truth for the r=3 cross-check's synthetic 2-level
// history. Builds N level-2 RElems (4 leaves each, tree order
// [leaf0,leaf1,leaf2,leaf3]) for cpu_round's consumption, AND the matching
// flat consolidated back-ref rows: level-1 row (offset 0*capacity) holds
// N*2 leaf-pairs; level-2 row (offset 1*capacity) holds each element's own
// two level-1 "child" slots, plus (in a separate array at the same row) its
// own lead (== tree[0]). Element k's 4 leaves are {4k,4k+1,4k+2,4k+3}
// (globally distinct across the fixture) with random (splitmix64) work.
// round_mix's on-device DFS, given level=2 and slot=k, decodes to EXACTLY
// {4k,4k+1,4k+2,4k+3} (left subtree first) -- the same order as elems[k].tree
// -- by construction, so the two views cannot drift apart.
struct Level2Fixture {
    std::vector<ref::RElem> elems;      // for cpu_round(elems, 3)
    std::vector<uint32_t> left, right;  // consolidated, 5*capacity each (rows 0,1 populated)
    std::vector<uint32_t> lead;         // consolidated, 5*capacity (row 1 populated)
    uint32_t capacity = 0;
};

static Level2Fixture build_level2_fixture(uint32_t N, uint32_t capacity, uint64_t& rngState) {
    Level2Fixture fx;
    fx.capacity = capacity;
    fx.elems.resize(N);
    fx.left.assign(5 * (size_t)capacity, 0u);
    fx.right.assign(5 * (size_t)capacity, 0u);
    fx.lead.assign(5 * (size_t)capacity, 0u);
    for (uint32_t k = 0; k < N; ++k) {
        uint32_t leafBase = k * 4u;
        ref::RElem& e = fx.elems[k];
        for (int w = 0; w < 7; ++w) e.w[w] = sm(rngState);
        e.tree = { leafBase + 0, leafBase + 1, leafBase + 2, leafBase + 3 };

        uint32_t s0 = 2*k, s1 = 2*k + 1;   // level-1 "child" slots for element k
        fx.left [0*(size_t)capacity + s0] = leafBase + 0; fx.right[0*(size_t)capacity + s0] = leafBase + 1;
        fx.left [0*(size_t)capacity + s1] = leafBase + 2; fx.right[0*(size_t)capacity + s1] = leafBase + 3;

        fx.left [1*(size_t)capacity + k] = s0;
        fx.right[1*(size_t)capacity + k] = s1;
        fx.lead [1*(size_t)capacity + k] = leafBase + 0;   // == e.tree[0]
    }
    return fx;
}

static void test_r3_match(Runtime& rt) {
    section("round_match r=3: synthetic 2-level history (256 level-2 elems + 1 lead-inversion pair), vs cpu_round(.,3)");

    const uint32_t N = 256;
    const uint32_t capacity = 1024;   // >= 2*N (level-1 row needs 512 slots) with headroom
    uint64_t rngState = 0xBEAC0Du ^ 0xF00DFACEu;

    Level2Fixture fx = build_level2_fixture(N, capacity, rngState);

    // Force 128 colliding PAIRS by duplicating WHOLE level-2 elements --
    // BOTH representations, from the fixture's own fields, so the two views
    // stay structurally identical by construction: the RElem (work+tree) for
    // cpu_round, AND the flat level-2 row (children slots + lead) for the
    // GPU. Because the lead is copied too, every one of these 128 pairs is
    // an EQUAL-LEAD pair -- this exercises match's (lead,slot) tie-break
    // end-to-end, not just once.
    for (uint32_t i = 0; i < 128; ++i) {
        uint32_t dst = 2*i + 1, src = 2*i;
        fx.elems[dst] = fx.elems[src];
        fx.left [1*(size_t)capacity + dst] = fx.left [1*(size_t)capacity + src];
        fx.right[1*(size_t)capacity + dst] = fx.right[1*(size_t)capacity + src];
        fx.lead [1*(size_t)capacity + dst] = fx.lead [1*(size_t)capacity + src];
    }

    // ------------------------------------------------------------------
    // Lead-inversion pair (slots N, N+1): the ONE collision shape the 128
    // forced WHOLE-element duplicates above cannot produce. Duplication
    // always copies tree[0] together with the rest of the element, so every
    // forced pair above is an EQUAL-lead pair (a tie-break, never a
    // comparison) -- a round_match that ordered pairs by raw SLOT instead of
    // the looked-up LEAD (all_lead_in[in_lead_off+slot]) would still produce
    // byte-identical output against all 128 of them. This pair closes that
    // gap: A sits at the LOWER slot (N) but is given the LARGER lead (2000);
    // B sits at the HIGHER slot (N+1) but is given the SMALLER lead (1024)
    // -- a strict inversion, not a tie. The correct kernel must pick left=B
    // (smaller lead) even though B has the higher slot; a slot-ordering bug
    // picks left=A instead, producing a DIFFERENT (left,right,lead) triple
    // that the checks below (and the load-bearing proof in the task report)
    // catch.
    //
    // Forcing the exact 24-bit collision (rather than leaving it to chance)
    // uses the same w[0]-solving technique test_round_reconstruct.cpp's
    // test_cpu_round_orders_by_lead_not_slot uses at r=1, generalized to
    // r=3's shape. apply_mix's round-3 fold (Lmix=400, padNum=4; see
    // bh3_primitives.h) places tree[0..3] at pos 400/425/450/475, landing on
    // (word6,sh16), (word6,sh41, +carry into word7 at sh23), (word7,sh2),
    // (word7,sh27) respectively -- i.e. the tree folds into t[6] and t[7],
    // not t[7] alone like r=1's single-leaf fold does. Pinning w[6]=0 on
    // BOTH A and B (rather than a real random value) makes that fold land on
    // a clean zero background instead of an unpredictable OR against real
    // w[6] bits:
    //   t[6] = tree[0]<<16 | tree[1]<<41        (exact -- see below)
    //   t[7] = tree[2]<<2  | tree[3]<<27
    // (tree[1]'s word-7 carry, tree[1]>>23, is exactly 0 since every leaf
    // used here is < 2^16 << 2^23). Holding w[1..5] AND w[6](=0) equal
    // between A and B then collapses "mixed(A) == mixed(B)" (the FULL 64-bit
    // apply_mix result, not just its low 24 bits -- an exact, not
    // probabilistic, collision) to one linear equation in the single
    // remaining unknown A.w[0] (ROT[0]=29 is a bijective rotation, so this is
    // exactly solvable, same as the r=1 constructions):
    //   rotl64(A.w0,29) + rotl64(t6_A,11) + rotl64(t7_A,40)
    //     == rotl64(B.w0,29) + rotl64(t6_B,11) + rotl64(t7_B,40)
    //   =>  A.w0 = rotl64(target, 64-29)
    // w[1..5] come from a real seed_element() call (kat::prePow-based) to
    // stay grounded in deterministic KAT data, same spirit as the r=1
    // triple's wRest above. Independently verified offline against all 256
    // base elements' mixed keys: zero accidental extra collisions, and this
    // pair's bucket (bucket_bits=8) has no other member, so it lands as an
    // isolated 2-member group -- see the task report for the verification
    // output.
    const uint32_t slotA = N;        // lower slot,  LARGER lead (2000)
    const uint32_t slotB = N + 1u;   // higher slot, SMALLER lead (1024) -- the inversion
    const uint32_t Ntotal = N + 2u;  // total live round-3 elements (fixture N + this pair)

    const uint32_t leafA[4] = { 2000u, 2001u, 2002u, 2003u };   // lead(A) = 2000
    const uint32_t leafB[4] = { 1024u, 1025u, 1026u, 1027u };   // lead(B) = 1024 < lead(A)

    bh3::Elem seedForRest;
    bh3::seed_element(kat::prePow, 777777u, seedForRest);   // deterministic, KAT-based wRest source

    ref::RElem A, B;
    A.tree.assign(leafA, leafA + 4);
    B.tree.assign(leafB, leafB + 4);
    for (int w = 1; w <= 5; ++w) { A.w[w] = seedForRest.w[w]; B.w[w] = seedForRest.w[w]; }
    A.w[6] = 0; B.w[6] = 0;
    B.w[0] = seedForRest.w[0];

    auto foldTree67 = [](const uint32_t leaf[4], uint64_t& t6, uint64_t& t7) {
        t6 = ((uint64_t)leaf[0] << 16) | ((uint64_t)leaf[1] << 41);
        t7 = ((uint64_t)leaf[2] << 2)  | ((uint64_t)leaf[3] << 27);
    };
    uint64_t t6A, t7A, t6B, t7B;
    foldTree67(leafA, t6A, t7A);
    foldTree67(leafB, t6B, t7B);
    uint64_t target = bh3::rotl64(B.w[0], 29)
                     + bh3::rotl64(t6B, 11) + bh3::rotl64(t7B, 40)
                     - bh3::rotl64(t6A, 11) - bh3::rotl64(t7A, 40);
    A.w[0] = bh3::rotl64(target, 64 - 29);

    check_eq_u64(ref::cpu_mix_w0(A, 3), ref::cpu_mix_w0(B, 3),
                 "r=3 inversion pair: A and B mix to the exact same w[0] (forced collision, not luck)");

    fx.elems.push_back(A);   // slot N   (slotA)
    fx.elems.push_back(B);   // slot N+1 (slotB)

    // Flat consolidated view: A/B each get 2 fresh level-1 "child" slots
    // (512..515, unused by the k=0..255 base fixture, which only claims
    // 0..511) holding their 4 leaves as 2 pairs, plus their own level-2 row
    // entry (children slots + lead) -- built from the SAME leafA/leafB/A/B
    // values as the RElem view above, so the two views cannot drift, exactly
    // like build_level2_fixture()'s own per-k loop does.
    const uint32_t s0A = 2*slotA, s1A = 2*slotA + 1;
    const uint32_t s0B = 2*slotB, s1B = 2*slotB + 1;
    fx.left [0*(size_t)capacity + s0A] = leafA[0]; fx.right[0*(size_t)capacity + s0A] = leafA[1];
    fx.left [0*(size_t)capacity + s1A] = leafA[2]; fx.right[0*(size_t)capacity + s1A] = leafA[3];
    fx.left [1*(size_t)capacity + slotA] = s0A;    fx.right[1*(size_t)capacity + slotA] = s1A;
    fx.lead [1*(size_t)capacity + slotA] = leafA[0];   // == A.tree[0]

    fx.left [0*(size_t)capacity + s0B] = leafB[0]; fx.right[0*(size_t)capacity + s0B] = leafB[1];
    fx.left [0*(size_t)capacity + s1B] = leafB[2]; fx.right[0*(size_t)capacity + s1B] = leafB[3];
    fx.left [1*(size_t)capacity + slotB] = s0B;    fx.right[1*(size_t)capacity + slotB] = s1B;
    fx.lead [1*(size_t)capacity + slotB] = leafB[0];   // == B.tree[0]

    std::vector<ref::RElem> cpuChildren = ref::cpu_round(fx.elems, 3);
    std::printf("  r=3: cpu_round produced %zu children (>=129 expected: 128 forced pairs + 1 lead-inversion pair)\n", cpuChildren.size());
    check(cpuChildren.size() >= 129, "r=3: cpu_round produced at least the 129 designed children");

    // The crux, on the CPU oracle side: cpu_round must pair the inversion
    // element by LEAD, not slot -- left=B (slot257, lead1024), right=A
    // (slot256, lead2000) -- and must NEVER produce the slot-ordered
    // (256,257) variant a raw-slot-comparison bug would emit instead.
    {
        const ref::RElem* invChild = nullptr;
        int wrongOrderCount = 0;
        for (const auto& c : cpuChildren) {
            if (c.leftSlot == slotB && c.rightSlot == slotA) invChild = &c;
            if (c.leftSlot == slotA && c.rightSlot == slotB) ++wrongOrderCount;
        }
        check(invChild != nullptr, "r=3 inversion pair: cpu_round pairs left=B(slot257,lead1024) right=A(slot256,lead2000) -- LEAD order");
        check(wrongOrderCount == 0, "r=3 inversion pair: cpu_round never emits the slot-ordered (256,257) pairing");
        if (invChild) {
            check(invChild->tree.size() == 8 &&
                  invChild->tree[0]==leafB[0] && invChild->tree[1]==leafB[1] &&
                  invChild->tree[2]==leafB[2] && invChild->tree[3]==leafB[3] &&
                  invChild->tree[4]==leafA[0] && invChild->tree[5]==leafA[1] &&
                  invChild->tree[6]==leafA[2] && invChild->tree[7]==leafA[3],
                  "r=3 inversion pair: child.tree == B.tree ++ A.tree (lead-ordered concat, B's leaves first)");
        }
    }

    Budget bud;
    bud.bucket_bits      = 8u;
    bud.num_buckets       = 1u << bud.bucket_bits;   // 256
    bud.slots_per_bucket = 32u;
    bud.elems_per_round  = capacity;
    bud.target_elems     = capacity;
    bud.seed_batch       = capacity;

    PipelineBuffers pb = alloc_pipeline(rt, bud);
    check(pb.capacity == capacity, "r=3: pipeline capacity == 1024");

    // pb.work[3]: round 3's own resident elements are the level-2 elements
    // (unmixed raw work -- round_mix mixes in place below). Ntotal (not N):
    // picks up the 2 lead-inversion elements appended above too.
    std::vector<uint64_t> work3((size_t)capacity * 7, 0);
    for (uint32_t k = 0; k < Ntotal; ++k)
        for (int w = 0; w < 7; ++w) work3[(size_t)k*7 + w] = fx.elems[k].w[w];
    rt.write(pb.work[3].get(), work3.size() * 8, work3.data());
    rt.write(pb.left.get(),  fx.left.size()  * 4, fx.left.data());
    rt.write(pb.right.get(), fx.right.size() * 4, fx.right.data());
    rt.write(pb.lead.get(),  fx.lead.size()  * 4, fx.lead.data());

    const uint32_t padNum3 = ref::padNum(3), Lmix3 = ref::Lmix(3), Lout3 = ref::Lout(3);
    check(padNum3 == 4u && Lmix3 == 400u, "r=3: padNum(3)==4, Lmix(3)==400 (sanity)");
    check(Lout3 == 376u, "r=3: Lout(3) == 376 (sanity)");

    mix_level(rt, pb, /*r=*/3, /*N=*/Ntotal);
    scatter(rt, pb, bud, Ntotal, /*workIndex=*/3);
    {
        uint32_t counters[4];
        rt.read(pb.counters.get(), sizeof counters, counters);
        std::printf("  r=3: scatter bucket_drops=%u (expect 0 -- 256 buckets x 32 slots for %u elems)\n", counters[1], Ntotal);
        check(counters[1] == 0u, "r=3: scatter counters[1] (bucket_drops) == 0");
    }

    uint32_t pair_drops = 0;
    uint32_t outN = match(rt, pb, bud, /*r=*/3, /*inN=*/Ntotal, pair_drops);
    std::printf("  r=3: match produced outN=%u pair_drops=%u\n", outN, pair_drops);
    check(pair_drops == 0u, "r=3: match counters[2] (pair_drops) == 0");
    check(outN == cpuChildren.size(), "r=3: GPU child count == CPU child count");

    // Round 3's children land in pb.work[4] (r+1); its own back-ref row is
    // out_off=(r-1)*capacity==2*capacity; the row this round's INPUT
    // elements' leads were read from is in_lead_off=(r-2)*capacity==capacity
    // (the level-2 row build_level2_fixture staged).
    const uint32_t outOff = (3 - 1) * capacity;      // 2*capacity
    const uint32_t inLeadOff = (3 - 2) * capacity;   // capacity

    std::vector<uint64_t> gpuWork((size_t)outN * 7);
    rt.read(pb.work[4].get(), gpuWork.size() * 8, gpuWork.data());
    std::vector<uint32_t> allLeft(5 * (size_t)capacity), allRight(5 * (size_t)capacity), allLead(5 * (size_t)capacity);
    rt.read(pb.left.get(),  allLeft.size()  * 4, allLeft.data());
    rt.read(pb.right.get(), allRight.size() * 4, allRight.data());
    rt.read(pb.lead.get(),  allLead.size()  * 4, allLead.data());
    std::vector<uint32_t> gpuLeft(outN), gpuRight(outN), gpuLead(outN);
    for (uint32_t i = 0; i < outN; ++i) {
        gpuLeft[i]  = allLeft[outOff + i];
        gpuRight[i] = allRight[outOff + i];
        gpuLead[i]  = allLead[outOff + i];
    }

    check_multiset_eq(cpu_keys(cpuChildren), gpu_keys(gpuWork, gpuLeft, gpuRight, outN), "r=3");

    // Structural checks against the SAME lead row match() itself read
    // (allLead at in_lead_off), independent of the CPU side: ordering
    // invariant (lead(left) < lead(right), or slot tie-break on equal
    // lead) plus a direct tally of how many children came from an
    // equal-lead pair -- must be >=1 (task requirement), and given ALL 128
    // forced pairs are equal-lead by construction, expected to be exactly
    // the forced-pair survivor count.
    int orderMismatches = 0, equalLeadCount = 0;
    for (uint32_t i = 0; i < outN; ++i) {
        uint32_t leadL = allLead[inLeadOff + gpuLeft[i]];
        uint32_t leadR = allLead[inLeadOff + gpuRight[i]];
        bool ok = (leadL < leadR) || (leadL == leadR && gpuLeft[i] < gpuRight[i]);
        if (!ok) ++orderMismatches;
        if (leadL == leadR) ++equalLeadCount;
    }
    std::printf("  r=3: %d of %u children came from an EQUAL-LEAD pair (tie-break path)\n", equalLeadCount, outN);
    check(orderMismatches == 0, "r=3: every GPU child ordered lead(left)<lead(right), or slot tie-break on equal lead");
    check(equalLeadCount >= 1, "r=3: at least one equal-lead pair exercised the (lead,slot) tie-break");
    check(equalLeadCount >= 128, "r=3: all 128 forced (fully-duplicated, equal-lead) pairs took the tie-break path");

    // Cross-reference GPU lead[oi] against the CPU child's tree[0] for the
    // SAME (left,right) pair.
    std::map<std::pair<uint32_t,uint32_t>, uint32_t> cpuLeadByPair;
    for (const auto& c : cpuChildren) cpuLeadByPair[{c.leftSlot, c.rightSlot}] = c.tree[0];
    int leadMismatches = 0, notFound = 0;
    for (uint32_t i = 0; i < outN; ++i) {
        auto it = cpuLeadByPair.find({gpuLeft[i], gpuRight[i]});
        if (it == cpuLeadByPair.end()) { ++notFound; continue; }
        if (gpuLead[i] != it->second) ++leadMismatches;
    }
    check(notFound == 0, "r=3: every GPU (left,right) pair matches some CPU child (same pair identity)");
    check(leadMismatches == 0, "r=3: every GPU child's lead == the CPU child's tree[0] for the same pair");

    // The crux, on the GPU kernel side (independent of the generic multiset
    // check above): round_match must pair the inversion element by LEAD, not
    // slot -- left=B (smaller lead, higher slot), right=A -- exactly the
    // distinction a kernel that ordered by raw slot instead of the
    // looked-up lead gets wrong. This is the specific assertion the task
    // report's load-bearing proof (temporarily patching round_match to
    // order by slot only) is expected to break.
    {
        int gpuLeadOrderCount = 0, gpuSlotOrderCount = 0;
        for (uint32_t i = 0; i < outN; ++i) {
            if (gpuLeft[i] == slotB && gpuRight[i] == slotA) {
                ++gpuLeadOrderCount;
                check(gpuLead[i] == leafB[0], "r=3 inversion pair: GPU child's lead == lead(B)=1024 (the smaller lead)");
            }
            if (gpuLeft[i] == slotA && gpuRight[i] == slotB) ++gpuSlotOrderCount;
        }
        check(gpuLeadOrderCount == 1, "r=3 inversion pair: GPU emits exactly one left=B(slot257) right=A(slot256) child (LEAD order, not slot order)");
        check(gpuSlotOrderCount == 0, "r=3 inversion pair: GPU never emits the slot-ordered (256,257) pairing");
    }

    // Smoke-exercise run_single_round() too, not just the split-out
    // mix_level()+scatter()+match() calls above -- r==3 exercises its
    // "r>=2, mix in place" branch (the counterpart to r=1's above).
    {
        PipelineBuffers pb2 = alloc_pipeline(rt, bud);
        rt.write(pb2.work[3].get(), work3.size() * 8, work3.data());
        rt.write(pb2.left.get(),  fx.left.size()  * 4, fx.left.data());
        rt.write(pb2.right.get(), fx.right.size() * 4, fx.right.data());
        rt.write(pb2.lead.get(),  fx.lead.size()  * 4, fx.lead.data());
        RoundStats stats;
        uint32_t outN2 = run_single_round(rt, pb2, bud, /*r=*/3, Ntotal, stats);
        check(outN2 == outN, "r=3: run_single_round() child count == direct match() child count");
        check(stats.in == Ntotal && stats.out == outN && stats.bucket_drops == 0 && stats.pair_drops == 0,
              "r=3: run_single_round() RoundStats {in,out,bucket_drops,pair_drops} correct");
    }

    std::string src(kRoundClSource);
    check(src.find("round_mix") != std::string::npos, "kRoundClSource contains round_mix");
}

int main() {
    if (!Runtime::any_device_available()) { std::printf("SKIP: no OpenCL device\n"); return 0; }
    Runtime rt;
    std::printf("  device: %s | CUs=%u\n", rt.device().name.c_str(), rt.device().compute_units);
    test_r1_match(rt);
    test_r3_match(rt);
    return summary("round_match");
}
