// Host-only: exercises tests/round_ref.h (the CPU round oracle) and does a
// compile/shape smoke check of src/gpu/round_pipeline.h. No OpenCL device
// needed -- must run everywhere OpenCL headers exist, like test_budget.
#include "check.h"
#include "kat_vectors.h"
#include "round_ref.h"
#include "gpu/round_pipeline.h"
#include <cstdint>
#include <cstdio>
#include <vector>
using namespace mxbm;

static void test_round_table() {
    section("round table: Lmix/Lout/padNum for r=1..5");
    const uint32_t wantLmix[5]   = {448,424,400,376,288};
    const uint32_t wantLout[5]   = {424,400,376,288,24};
    const uint32_t wantPadNum[5] = {1,2,4,6,9};
    for (int r = 1; r <= 5; ++r) {
        char lbl[48];
        std::snprintf(lbl, sizeof lbl, "Lmix(%d) == %u", r, wantLmix[r-1]);
        check(ref::Lmix(r) == wantLmix[r-1], lbl);
        std::snprintf(lbl, sizeof lbl, "Lout(%d) == %u", r, wantLout[r-1]);
        check(ref::Lout(r) == wantLout[r-1], lbl);
        std::snprintf(lbl, sizeof lbl, "padNum(%d) == %u", r, wantPadNum[r-1]);
        check(ref::padNum(r) == wantPadNum[r-1], lbl);
    }
}

// Section 1: hand-built level-1/level-2 back-ref chain, no hashing involved --
// pure exercise of first_leaves' pre-order DFS + early-stop over flat arrays.
// Leaves used (conceptually the 8 raw indices {5,3,9,1,12,7,20,2}, grouped):
//   level-1  A={3,5}  B={1,9}  C={7,12}  D={2,20}   (left = smaller leaf)
//   level-2  E={B,A}  (lead(B)=1 < lead(A)=3 -> left=B)
//            F={D,C}  (lead(D)=2 < lead(C)=7 -> left=D)
static void test_first_leaves_hand_built() {
    section("first_leaves: hand-built level-1/level-2 back-ref chain (capacity 8)");
    const uint32_t capacity = 8;
    std::vector<uint32_t> left(5 * capacity, 0), right(5 * capacity, 0);

    // Level-1 row: offset 0. Entries ARE leaf indices (no level-0 row exists).
    const uint32_t slotA = 0, slotB = 1, slotC = 2, slotD = 3;
    left[0*capacity + slotA] = 3;  right[0*capacity + slotA] = 5;    // A={3,5}
    left[0*capacity + slotB] = 1;  right[0*capacity + slotB] = 9;    // B={1,9}
    left[0*capacity + slotC] = 7;  right[0*capacity + slotC] = 12;   // C={7,12}
    left[0*capacity + slotD] = 2;  right[0*capacity + slotD] = 20;   // D={2,20}

    // Level-2 row: offset capacity. Entries are level-1 SLOT indices.
    const uint32_t slotE = 0, slotF = 1;
    left[1*capacity + slotE] = slotB;  right[1*capacity + slotE] = slotA;  // E={B,A}
    left[1*capacity + slotF] = slotD;  right[1*capacity + slotF] = slotC;  // F={D,C}

    std::vector<uint32_t> a2 = ref::first_leaves(left, right, capacity, /*level*/1, slotA, 2);
    check(a2.size() == 2 && a2[0] == 3 && a2[1] == 5,
          "first_leaves(level=1,slotA,need=2) == {3,5}");

    std::vector<uint32_t> e4 = ref::first_leaves(left, right, capacity, /*level*/2, slotE, 4);
    check(e4.size() == 4 && e4[0] == 1 && e4[1] == 9 && e4[2] == 3 && e4[3] == 5,
          "first_leaves(level=2,slotE,need=4) == {1,9,3,5}");

    std::vector<uint32_t> e3 = ref::first_leaves(left, right, capacity, /*level*/2, slotE, 3);
    check(e3.size() == 3 && e3[0] == 1 && e3[1] == 9 && e3[2] == 3,
          "first_leaves(level=2,slotE,need=3) == {1,9,3} (prefix truncation)");

    std::vector<uint32_t> f4 = ref::first_leaves(left, right, capacity, /*level*/2, slotF, 4);
    check(f4.size() == 4 && f4[0] == 2 && f4[1] == 20 && f4[2] == 7 && f4[3] == 12,
          "first_leaves(level=2,slotF,need=4) == {2,20,7,12}");
}

// Section 2: 4 REAL seeds (kat::prePow), forced collisions via full-input
// duplication, run through cpu_round for real.
//
// apply_mix's round-1 fold (Lmix=448 -> word 7, shift 0; see bh3_primitives.h)
// computes result = rotl64(S(w) + rotl64(tree[0],40), 24) -- the lone folded
// leaf (tree[0]) enters ADDITIVELY and independently of w, through a bijective
// rotation. So two elements with equal w but DIFFERENT tree[0] provably mix to
// DIFFERENT w[0] (never a coincidence away from it) -- matching "work" alone
// cannot guarantee a collision. To get a genuinely guaranteed (not
// probabilistic) collision we duplicate the full mixing input -- w AND tree --
// from the pair's first element into its second.
static void test_cpu_round_forced_collisions() {
    section("cpu_round: real seeds, forced collisions (grouping/ordering/concat)");
    const uint32_t six[4] = {11u, 23u, 37u, 41u};
    std::vector<ref::RElem> seeds =
        ref::make_seeds(kat::prePow, {six[0], six[1], six[2], six[3]});

    seeds[1] = seeds[0];   // force pair (0,1) to share w AND tree
    seeds[3] = seeds[2];   // force pair (2,3) to share w AND tree

    uint64_t mix0 = ref::cpu_mix_w0(seeds[0], 1);
    uint64_t mix1 = ref::cpu_mix_w0(seeds[1], 1);
    uint64_t mix2 = ref::cpu_mix_w0(seeds[2], 1);
    uint64_t mix3 = ref::cpu_mix_w0(seeds[3], 1);
    check(mix0 == mix1, "forced pair (0,1): identical mixed w[0] (guaranteed collision)");
    check(mix2 == mix3, "forced pair (2,3): identical mixed w[0] (guaranteed collision)");
    check((mix0 & 0xFFFFFFu) != (mix2 & 0xFFFFFFu),
          "the two forced groups do not also collide with each other");

    std::vector<ref::RElem> children = ref::cpu_round(seeds, 1);
    check(children.size() == 2, "cpu_round emits exactly 2 children (one pair per forced group)");

    const ref::RElem* c01 = nullptr;
    const ref::RElem* c23 = nullptr;
    for (const auto& c : children) {
        if (c.leftSlot == 0 && c.rightSlot == 1) c01 = &c;
        if (c.leftSlot == 2 && c.rightSlot == 3) c23 = &c;
    }
    check(c01 != nullptr, "child(0,1) present: leftSlot/rightSlot == the ordered input slots");
    check(c23 != nullptr, "child(2,3) present: leftSlot/rightSlot == the ordered input slots");

    if (c01) {
        check(c01->tree.size() == 2 && c01->tree[0] == six[0] && c01->tree[1] == six[0],
              "child(0,1).tree == left.tree ++ right.tree (ordered concat)");
    }
    if (c23) {
        check(c23->tree.size() == 2 && c23->tree[0] == six[2] && c23->tree[1] == six[2],
              "child(2,3).tree == left.tree ++ right.tree (ordered concat)");
    }

    // cpu_mix_w0 at r=2 must equal a hand-rolled apply_mix call over the
    // child's own 2-leaf tree (padNum(2)==2 -> both leaves fold in).
    if (c01) {
        check(ref::padNum(2) == 2, "padNum(2) == 2 (both child leaves fold into round 2)");
        bh3::Elem manual;
        for (int w = 0; w < bh3::kWorkWords; ++w) manual.w[w] = c01->w[w];
        bh3::apply_mix(manual, c01->tree.data(), (uint32_t)c01->tree.size(), ref::Lmix(2));
        uint64_t viaWrapper = ref::cpu_mix_w0(*c01, 2);
        check_eq_u64(viaWrapper, manual.w[0],
                     "cpu_mix_w0(child,2) == manual apply_mix over its 2-leaf tree");
    }
}

// The forced-collision pairs above always tie on lead (both members of a pair
// share the exact same tree[0], since the pair is a full duplicate), so they
// only exercise cpu_round's slot tie-break, never its lead comparison. This
// closes that gap: two DIFFERENT leads, deliberately placed so slot-order and
// lead-order disagree (the smaller lead sits at the LARGER slot), forced to
// collide by exact algebraic construction rather than duplication. Round-1's
// mix (Lmix=448 -> word 7, shift 0) is result = rotl64(S(w) + rotl64(tree[0],40), 24)
// where S(w) = sum of rotl64(w[i],ROT[i]) for i=0..6, ROT[0]=29. Holding
// w[1..6] equal between the pair collapses the equation to one unknown
// (w[0]), solvable exactly -- so the two elements mix to the SAME w[0] despite
// different leaves, by construction rather than luck.
static void test_cpu_round_orders_by_lead_not_slot() {
    section("cpu_round: orders by LEAD first, not by slot, when leads differ");
    const uint32_t idxP = 9000u, idxQ = 4000u;   // idxQ < idxP: Q must sort first

    bh3::Elem eP; bh3::seed_element(kat::prePow, idxP, eP);
    ref::RElem P; for (int w = 0; w < bh3::kWorkWords; ++w) P.w[w] = eP.w[w];
    P.tree = { idxP };

    ref::RElem Q = P;              // start from P's real work words, w[1..6] kept equal
    Q.tree = { idxQ };
    uint64_t target = bh3::rotl64(P.w[0], 29)
                     + bh3::rotl64((uint64_t)idxP, 40)
                     - bh3::rotl64((uint64_t)idxQ, 40);
    Q.w[0] = bh3::rotl64(target, 64 - 29);   // solved so P and Q mix to the same w[0]

    // P at slot 0, Q at slot 1: slot-order says "P first"; lead-order (Q=4000 < P=9000) says "Q first".
    std::vector<ref::RElem> pair = { P, Q };
    uint64_t mixP = ref::cpu_mix_w0(pair[0], 1);
    uint64_t mixQ = ref::cpu_mix_w0(pair[1], 1);
    check_eq_u64(mixP, mixQ, "constructed differing-lead pair mixes to the exact same w[0]");

    std::vector<ref::RElem> children = ref::cpu_round(pair, 1);
    check(children.size() == 1, "differing-lead collision yields exactly 1 child");
    if (!children.empty()) {
        const ref::RElem& c = children[0];
        check(c.leftSlot == 1 && c.rightSlot == 0,
              "left = smaller-LEAD Q (slot1), right = P (slot0) -- lead beats slot order");
        check(c.tree.size() == 2 && c.tree[0] == idxQ && c.tree[1] == idxP,
              "child.tree == left(Q).tree ++ right(P).tree (lead-ordered concat, not slot-ordered)");
    }
}

// A 2-member collision group can't discriminate "emit ALL i<j pairs" (what
// cpu_round does -- a real search) from "emit only adjacent pairs" (what
// bh3_verify.cpp does -- it only ever CHECKS a pre-paired sequence): both
// produce exactly 1 pair either way. This generalizes the algebraic
// construction above to a genuine 3-member group -- 3 distinct leads forced
// to the exact same round-1 mixed w[0] by sharing w[1..6] and solving each
// w[0] against one common target T -- and checks cpu_round emits all
// C(3,2)=3 pairs, each correctly lead-ordered.
static void test_cpu_round_all_pairs_in_group() {
    section("cpu_round: emits ALL i<j pairs within a 3-member collision group");
    const uint32_t idx[3] = { 5000u, 1000u, 9000u };   // deliberately unsorted

    bh3::Elem seed0; bh3::seed_element(kat::prePow, idx[0], seed0);
    uint64_t wRest[7]; for (int w = 0; w < bh3::kWorkWords; ++w) wRest[w] = seed0.w[w];

    // Common target: rotl64(w0,29) + rotl64(idx,40) == T for every member.
    // Anchored on element 0's own real w[0] -- so group[0] falls out of the
    // uniform formula below already equal to wRest[0] unmodified.
    uint64_t T = bh3::rotl64(wRest[0], 29) + bh3::rotl64((uint64_t)idx[0], 40);

    std::vector<ref::RElem> group(3);
    for (int k = 0; k < 3; ++k) {
        for (int w = 1; w < bh3::kWorkWords; ++w) group[k].w[w] = wRest[w];
        uint64_t rotatedW0 = T - bh3::rotl64((uint64_t)idx[k], 40);
        group[k].w[0] = bh3::rotl64(rotatedW0, 64 - 29);
        group[k].tree = { idx[k] };
    }

    uint64_t m0 = ref::cpu_mix_w0(group[0], 1);
    uint64_t m1 = ref::cpu_mix_w0(group[1], 1);
    uint64_t m2 = ref::cpu_mix_w0(group[2], 1);
    check(m0 == m1 && m1 == m2, "all 3 constructed elements mix to the exact same w[0]");
    check_eq_u64(group[0].w[0], wRest[0], "element 0's solved w[0] reproduces its real, unmodified value");

    std::vector<ref::RElem> children = ref::cpu_round(group, 1);
    check(children.size() == 3, "3-member group yields all C(3,2)=3 pairs, not just 2 adjacent ones");

    // Lead order: idx[1]=1000 < idx[0]=5000 < idx[2]=9000 -> sorted slots (1,0,2).
    // All i<j pairs over that order: (1,0) (1,2) (0,2).
    int seen_1_0 = 0, seen_1_2 = 0, seen_0_2 = 0;
    for (const auto& c : children) {
        if (c.leftSlot == 1 && c.rightSlot == 0) {
            ++seen_1_0;
            check(c.tree.size()==2 && c.tree[0]==idx[1] && c.tree[1]==idx[0], "pair(1,0) lead-ordered concat");
        }
        if (c.leftSlot == 1 && c.rightSlot == 2) {
            ++seen_1_2;
            check(c.tree.size()==2 && c.tree[0]==idx[1] && c.tree[1]==idx[2], "pair(1,2) lead-ordered concat");
        }
        if (c.leftSlot == 0 && c.rightSlot == 2) {
            ++seen_0_2;
            check(c.tree.size()==2 && c.tree[0]==idx[0] && c.tree[1]==idx[2], "pair(0,2) lead-ordered concat");
        }
    }
    check(seen_1_0==1 && seen_1_2==1 && seen_0_2==1, "exactly the 3 expected lead-ordered pairs are present, each once");
}

// Compile/shape smoke check for src/gpu/round_pipeline.h -- no Runtime/device
// needed for default construction, so this stays host-only.
static void test_pipeline_buffers_shape() {
    section("PipelineBuffers: default-constructible (host-only smoke)");
    gpu::PipelineBuffers pb;
    check(pb.capacity == 0, "PipelineBuffers default capacity is 0");
}

int main() {
    test_round_table();
    test_first_leaves_hand_built();
    test_cpu_round_forced_collisions();
    test_cpu_round_orders_by_lead_not_slot();
    test_cpu_round_all_pairs_in_group();
    test_pipeline_buffers_shape();
    return summary("round_reconstruct");
}
