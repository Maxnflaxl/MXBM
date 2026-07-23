// GPU round-mix kernels (round1_mix_seeds, round_mix) vs the CPU oracle
// (tests/round_ref.h). Two sections:
//  (a) round1_mix_seeds over a small-but-real range [0,65536) vs
//      bh3::apply_mix(seed_element(pp,idx), {idx}, 1, 448).
//  (b) round_mix (level>=1, i.e. mixing round-r>=2 elements in place)
//      reconstructing leaves from the SAME hand-built capacity-8 back-ref
//      chain Task 1 used (tests/test_round_reconstruct.cpp's
//      test_first_leaves_hand_built), vs
//      bh3::apply_mix(work, ref::first_leaves(...).data(), padNum, Lmix) --
//      both the natural padNum(3)=4 shape (via mix_level) and a padNum=3
//      truncation case (direct kernel call, exercising the DFS early-stop).
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

// Deterministic pseudo-random (no <random> dependence): splitmix64.
static uint64_t sm(uint64_t& s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Section (a): round1_mix_seeds over [0,65536) vs CPU seed_element+apply_mix.
static void test_round1_mix_seeds(Runtime& rt) {
    section("round1_mix_seeds == C++ seed_element+apply_mix(tree={idx},Lmix=448) over [0,65536)");

    // Small-but-real Budget: elems_per_round == seed_batch == 65536, so
    // mix_seeds' own batching loop drives exactly one batch (begin=0,count=65536).
    Budget bud;
    bud.elems_per_round = 65536u;
    bud.seed_batch      = 65536u;
    bud.bucket_bits = 10u; bud.num_buckets = 1024u; bud.slots_per_bucket = 64u;  // unused here; must be non-zero for alloc_pipeline
    bud.target_elems = 1u << 25;

    PipelineBuffers pb = alloc_pipeline(rt, bud);
    check(pb.capacity == 65536u, "pipeline capacity == 65536");

    mix_seeds(rt, pb, bud, kat::prePow);

    std::vector<uint64_t> work((size_t)pb.capacity * 7);
    rt.read(pb.work[1].get(), work.size() * 8, work.data());

    const uint32_t STRIDE = 128;   // 65536/128 == 512 samples, exactly
    int bad = 0; long checked = 0;
    for (uint32_t idx = 0; idx < pb.capacity; idx += STRIDE) {
        bh3::Elem seed; bh3::seed_element(kat::prePow, idx, seed);
        bh3::Elem mixed = seed;
        uint32_t tree1[1] = { idx };
        bh3::apply_mix(mixed, tree1, 1u, 448u);
        bool ok = (work[(size_t)idx*7] == mixed.w[0]);
        for (int w = 1; w < 7 && ok; ++w) ok = (work[(size_t)idx*7 + w] == seed.w[w]);
        if (!ok) {
            if (bad < 3) std::printf("  MISMATCH idx=%u got=%llx want=%llx\n",
                idx, (unsigned long long)work[(size_t)idx*7], (unsigned long long)mixed.w[0]);
            ++bad;
        }
        ++checked;
    }
    std::printf("  round1_mix_seeds: %ld sampled indices, %d mismatches\n", checked, bad);
    check(checked == 512, "sampled exactly 512 indices");
    check(bad == 0, "round1_mix_seeds bit-identical to CPU over the sampled range");
}

// Section (b): round_mix over the Task-1 synthetic capacity-8 back-ref chain.
static void test_round_mix_reconstruct(Runtime& rt) {
    section("round_mix(level>=1) == CPU first_leaves + apply_mix (r=3 shape, incl padNum-truncation)");
    const uint32_t capacity = 8;

    // Same hand-built level-1/level-2 chain as
    // tests/test_round_reconstruct.cpp's test_first_leaves_hand_built:
    //   level-1  A={3,5}  B={1,9}  C={7,12}  D={2,20}   (left = smaller leaf)
    //   level-2  E={B,A}  F={D,C}
    std::vector<uint32_t> left(5 * capacity, 0), right(5 * capacity, 0);
    const uint32_t slotA = 0, slotB = 1, slotC = 2, slotD = 3;
    left[0*capacity + slotA] = 3;  right[0*capacity + slotA] = 5;
    left[0*capacity + slotB] = 1;  right[0*capacity + slotB] = 9;
    left[0*capacity + slotC] = 7;  right[0*capacity + slotC] = 12;
    left[0*capacity + slotD] = 2;  right[0*capacity + slotD] = 20;
    const uint32_t slotE = 0, slotF = 1;
    left[1*capacity + slotE] = slotB;  right[1*capacity + slotE] = slotA;
    left[1*capacity + slotF] = slotD;  right[1*capacity + slotF] = slotC;

    Budget bud;
    bud.elems_per_round = capacity;
    bud.seed_batch      = capacity;
    bud.bucket_bits = 1u; bud.num_buckets = 2u; bud.slots_per_bucket = 1u;  // unused here; must be non-zero for alloc_pipeline

    PipelineBuffers pb = alloc_pipeline(rt, bud);
    check(pb.capacity == capacity, "synthetic pipeline capacity == 8");

    // (E3a) Back-refs are NOT uploaded -- round_mix no longer walks them. They
    // live only on the CPU to derive the EXPECTED leaf prefix (ref::first_leaves),
    // which is exactly what the previous round's match would have materialized.
    const uint32_t LSTRIDE = 9;   // AoS leaf stride (== BH3_MAX_LEAVES); element g at g*9+i

    // 2 level-2 elements' work values (random via splitmix64), staged at slots E,F.
    uint64_t s = 0xDEC0DEu;
    std::vector<uint64_t> workE(7), workF(7);
    for (int w = 0; w < 7; ++w) workE[w] = sm(s);
    for (int w = 0; w < 7; ++w) workF[w] = sm(s);
    auto stage_work3 = [&]() {
        std::vector<uint64_t> work3(pb.capacity * 7, 0);
        for (int w = 0; w < 7; ++w) { work3[slotE*7 + w] = workE[w]; work3[slotF*7 + w] = workF[w]; }
        rt.write(pb.work[3 & 1].get(), work3.size() * 8, work3.data());
    };
    auto expected_leaves = [&](uint32_t slot, uint32_t need) {
        return ref::first_leaves(left, right, capacity, /*level*/2, slot, need);
    };
    auto stage_leaves = [&](uint32_t need) {   // materialize the prefix E3a-style (AoS)
        std::vector<uint32_t> lv((size_t)pb.capacity * LSTRIDE, 0);
        auto le = expected_leaves(slotE, need), lf = expected_leaves(slotF, need);
        for (uint32_t i = 0; i < need; ++i) { lv[slotE*LSTRIDE + i] = le[i]; lv[slotF*LSTRIDE + i] = lf[i]; }
        rt.write(pb.leaves[3 & 1].get(), lv.size() * 4, lv.data());
    };

    const uint32_t padNum3 = ref::padNum(3), Lmix3 = ref::Lmix(3);
    check(padNum3 == 4u, "padNum(3) == 4 (sanity)");
    check(Lmix3 == 400u, "Lmix(3) == 400 (sanity)");

    auto cpu_mix = [&](uint32_t slot, const std::vector<uint64_t>& w, uint32_t need) {
        std::vector<uint32_t> leaves = expected_leaves(slot, need);
        bh3::Elem e{}; for (int k = 0; k < 7; ++k) e.w[k] = w[k];
        bh3::apply_mix(e, leaves.data(), (uint32_t)leaves.size(), Lmix3);
        return e.w[0];
    };

    // (b1) natural shape via mix_level(r=3): padNum=4, Lmix=400.
    stage_work3();
    stage_leaves(padNum3);
    mix_level(rt, pb, /*r=*/3, /*N=*/2);
    {
        std::vector<uint64_t> got(pb.capacity * 7);
        rt.read(pb.work[3 & 1].get(), got.size() * 8, got.data());
        check_eq_u64(got[slotE*7], cpu_mix(slotE, workE, padNum3), "mix_level(r=3) slot E: w[0] == CPU mix from materialized leaves");
        check_eq_u64(got[slotF*7], cpu_mix(slotF, workF, padNum3), "mix_level(r=3) slot F: w[0] == CPU mix from materialized leaves");
        int bad = 0;
        for (int k = 1; k < 7; ++k) {
            if (got[slotE*7+k] != workE[k]) ++bad;
            if (got[slotF*7+k] != workF[k]) ++bad;
        }
        check(bad == 0, "mix_level(r=3): w[1..6] unchanged (only w[0] overwritten in place)");
    }

    // (b2) padNum-truncation: padNum=3 (< the natural 4). round_mix reads only
    // the first 3 leaves of the stored prefix (pre-order is prefix-consistent).
    // Direct kernel call for the padNum override (mix_level has no such knob).
    stage_work3();
    stage_leaves(padNum3);   // full 4-leaf prefix stored; the kernel reads the first 3
    {
        Program prog = rt.build({std::string(kBh3ClSource), std::string(kRoundClSource)}, "");
        Kernel k = rt.kernel(prog.get(), "round_mix");
        uint32_t N = 2u, cap = capacity, padNum = 3u, Lmix = Lmix3;
        cl_mem workMem = pb.work[3 & 1].get(), leavesMem = pb.leaves[3 & 1].get();
        rt.set_arg(k.get(), 0, N);
        rt.set_arg(k.get(), 1, cap);            // capacity arg (unused by AoS round_mix, kept in signature)
        rt.set_arg(k.get(), 2, padNum);
        rt.set_arg(k.get(), 3, Lmix);
        cl_mem pairsMem = pb.sort_pairs[0].get();   // D3: fused sort-pair output arg
        rt.set_arg(k.get(), 4, sizeof(cl_mem), &workMem);
        rt.set_arg(k.get(), 5, sizeof(cl_mem), &leavesMem);
        rt.set_arg(k.get(), 6, sizeof(cl_mem), &pairsMem);
        rt.run1d(k.get(), N);

        std::vector<uint64_t> got(pb.capacity * 7);
        rt.read(pb.work[3 & 1].get(), got.size() * 8, got.data());
        check_eq_u64(got[slotE*7], cpu_mix(slotE, workE, padNum), "truncation(padNum=3) slot E: w[0] == CPU mix");
        check_eq_u64(got[slotF*7], cpu_mix(slotF, workF, padNum), "truncation(padNum=3) slot F: w[0] == CPU mix");
    }

    // Sanity: the kernel source really was embedded (both kernel names present).
    std::string src(kRoundClSource);
    check(src.find("round1_mix_seeds") != std::string::npos, "kRoundClSource contains round1_mix_seeds");
    check(src.find("round_mix") != std::string::npos, "kRoundClSource contains round_mix");
}

int main() {
    if (!Runtime::any_device_available()) { std::printf("SKIP: no OpenCL device\n"); return 0; }
    Runtime rt;
    std::printf("  device: %s | CUs=%u\n", rt.device().name.c_str(), rt.device().compute_units);
    test_round1_mix_seeds(rt);
    test_round_mix_reconstruct(rt);
    return summary("round_mix");
}
