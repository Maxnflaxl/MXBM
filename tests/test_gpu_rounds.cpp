// Task 6 -- Phase-B exit / milestone: a REAL full 2^25 GPU Wagner search
// over the KAT input, proving run_pipeline() (T5's 5-round driver --
// mix_seeds -> 5x run_single_round -> survivor_scan, src/gpu/round_pipeline.
// {h,cpp}, reused here completely unmodified) finds genuine is_zero round-5
// survivors on real device hardware. This is neither the synthetic/crafted
// fixture of tests/test_round_pipeline.cpp section (a) (a hand-built
// duplicated-history pair, exercising the r=5 special path in isolation) nor
// that file's memory-limited 2^20 mini-search vs a CPU oracle (section (b));
// it is the actual, full-scale search this whole phase has been building
// toward.
//
// kat::input32/kat::nonce0 -> kat::prePow (tests/kat_vectors.h) is known to
// have exactly 3 golden 104-byte solutions (tests/vectors/beamhash3-kat.md
// section 5). Each golden's ancestry culminates in one round-5 collision
// between two BYTE-IDENTICAL round-4 elements -- bh3_combine is XOR-based
// (see kernels/opencl/round.cl's survivor_scan doc comment), so an all-zero
// child is the deterministic signature of exactly that event. A genuine
// is_zero round-5 survivor found by a real, non-crafted search over real
// seed data therefore IS (for all practical purposes) a golden solution:
// two DIFFERENT round-4 elements coincidentally XORing to all-zero across
// all 7 words (448 bits) would require a 448-bit collision among ~2^25
// candidates, astronomically implausible. This test's job is to prove the
// search FINDS >=1 of them on real hardware; recovering and fully verifying
// each survivor's 104-byte encoding against tests/vectors/beamhash3-kat.md
// is deferred to Phase C.
//
// compute_budget() sizes every round's work/backref buffers AND match()'s
// out_capacity to the SAME elems_per_round (== 2^25 on any device with
// enough memory for the full resident set -- see tests/test_budget.cpp's
// "M3Max keeps full 2^25" case), so a round whose TRUE collision count
// exceeds that capacity drops the excess (honest pair_drops, never silently
// resized or hidden). No headroom decoupling was added to alloc_pipeline /
// run_pipeline for this task: the UNMODIFIED T5 pipeline already finds all 3
// known goldens on real M3 Max hardware (see .superpowers/sdd/task-6-report.
// md for the full per-round log), so the task brief's "if survivors==0, add
// a search_capacity" fallback never triggers here.
#include "gpu/cl_runtime.h"
#include "gpu/round_pipeline.h"
#include "kat_vectors.h"
#include "round_ref.h"
#include "check.h"
#include <cstdint>
#include <cstdio>
#include <chrono>
using namespace mxbm;
using namespace mxbm::gpu;

int main() {
    if (!Runtime::any_device_available()) { std::printf("SKIP: no OpenCL device\n"); return 0; }
    Runtime rt;
    const DeviceInfo& d = rt.device();
    section("Phase-B exit: real 2^25 GPU Wagner search finds is_zero survivor(s) on the KAT input");
    std::printf("  device: %s | gmem=%.2fGiB | max_alloc=%.2fGiB | CUs=%u\n",
                d.name.c_str(), d.global_mem / 1073741824.0, d.max_alloc / 1073741824.0, d.compute_units);

    Budget b = compute_budget(d.global_mem, d.max_alloc);
    std::printf("  budget: elems_per_round=%u bucket_bits=%u num_buckets=%u slots_per_bucket=%u seed_batch=%u\n",
                b.elems_per_round, b.bucket_bits, b.num_buckets, b.slots_per_bucket, b.seed_batch);
    // BeamHash III indices are 25-bit ([0,2^25)) -- the seed layer's size is
    // a protocol constant, not a memory-scaling knob. On any device with
    // enough memory for the full resident set (compute_budget's headroom
    // math), elems_per_round == the full 2^25 -- confirmed here so a future
    // run on a memory-constrained device fails LOUDLY (via this check) rather
    // than silently searching a truncated seed range and calling it done.
    check(b.elems_per_round == (1u << 25), "device budget keeps the full 2^25 seed layer (no memory-driven truncation on this hardware)");

    PipelineBuffers pb = alloc_pipeline(rt, b);
    // capacity carries collision headroom ABOVE the 2^25 seed count so rounds 1-2
    // never clamp (see Budget::capacity / the pairDrops==0 assertion below).
    check(pb.capacity == b.capacity, "pipeline capacity == b.capacity (seed count + headroom)");

    // BUDGET vs REALITY. compute_budget decides how large a seed layer fits using
    // kBytesPerElement; alloc_pipeline then makes the real allocations. If the
    // estimate ever drops below what is actually allocated, a device would be told
    // it can host a full layer and then OOM. Pin them together here: sum the real
    // per-element resident cost for the ACTIVE path and require the budget's
    // per-element figure to cover it.
    {
        const uint64_t C = pb.capacity;
        uint64_t actual;
        if (pb.rowbucket) {
            const uint64_t ns = (uint64_t)pb.fb_num_buckets * pb.fb_bucket_cap;
            actual = ns*(pb.fb_stride[0] + pb.fb_stride[1])*8
                   + 2*(uint64_t)pb.fb_num_buckets*4 + 2*5*C*4;
        } else {
            actual = 2*C*7*8 + 2*9*C*4 + 3*5*C*4 + (uint64_t)b.num_buckets*(1 + b.slots_per_bucket)*4
                   + 2*C*8 + C*4 + 256*((C+255)/256)*4;
        }
        const double perElem = (double)actual / (double)b.elems_per_round;
        std::printf("  budget check: %s path allocates %.2f GiB = %.0f B/element; "
                    "budget assumes %u B/element\n",
                    pb.rowbucket ? "row-bucket" : "sort",
                    (double)actual / 1073741824.0, perElem, kBytesPerElement);
        check(perElem <= (double)kBytesPerElement,
              "compute_budget's per-element estimate covers the real allocation (no OOM risk)");
    }
    // The fused kernels BAKE their per-round constants in at compile time (see the
    // FUSED_LDS instantiations in kernels/opencl/lds.cl) and therefore IGNORE the
    // matching runtime arguments the host still sets. Pin the two together: these
    // literals must stay identical to the ones in lds.cl, and must agree with the
    // round table the rest of the pipeline derives everything else from.
    {
        struct { int r; uint32_t Lout, padNext, sIn, sOut, sBuild; } kKernelLiterals[4] = {
            {1, 424u, 2u, 1u, 2u, 2u},   // round_fused_seed
            {2, 400u, 4u, 2u, 4u, 4u},   // round_fused_rd2
            {3, 376u, 6u, 4u, 2u, 8u},   // round_fused_rd3
            {4, 288u, 9u, 2u, 0u, 0u},   // round_fused_6_5
        };
        for (const auto& k : kKernelLiterals) {
            const FusedConsts fc = fused_consts_for(k.r);
            check(fc.Lout == k.Lout && fc.padNext == k.padNext && fc.sIn == k.sIn &&
                  fc.sOut == k.sOut && fc.sBuild == k.sBuild,
                  "fused_consts_for matches the literals compiled into lds.cl");
            // Checked against the independent reference table, not the pipeline's
            // own copy. Lout(r) == Lmix(r+1) is what lets one constant serve both.
            check(k.Lout == ref::Lout(k.r) && k.Lout == ref::Lmix(k.r + 1) &&
                  k.padNext == ref::padNum(k.r + 1),
                  "baked-in constants agree with the reference round table");
        }
        // round5_fused_lds bakes Lout(5) in the same way (LDS_R5LOUT in lds.cl).
        check(ref::Lout(5) == 24u, "round 5's baked-in LDS_R5LOUT (24) matches the round table");
    }
    check(b.capacity > b.elems_per_round, "budget reserves collision headroom (capacity > 2^25 seed count) on this card");

    std::printf("  running run_pipeline() over the full 2^25 seed layer on the KAT prePow...\n");
    auto t0 = std::chrono::steady_clock::now();
    PipelineResult res = run_pipeline(rt, pb, b, kat::prePow);
    auto t1 = std::chrono::steady_clock::now();
    double wallSecs = std::chrono::duration<double>(t1 - t0).count();

    std::printf("  round |        in |       out | bucketDrops | pairDrops |   mix ms | scatr ms | match ms\n");
    uint32_t totalBucketDrops = 0, totalPairDrops = 0;
    for (int r = 0; r < 5; ++r) {
        const RoundStats& st = res.rounds[r];
        std::printf("    r%d  | %9u | %9u | %11u | %9u | %8.1f | %8.1f | %8.1f\n",
                    r + 1, st.in, st.out, st.bucket_drops, st.pair_drops,
                    st.t_mix_ms, st.t_scatter_ms, st.t_match_ms);
        totalBucketDrops += st.bucket_drops;
        totalPairDrops   += st.pair_drops;

        // Sanity against silent early-collapse: rounds 1-4 (i<4) must each
        // produce output, or a downstream 0 (and thus 0 survivors) would be
        // indistinguishable from "the search legitimately petered out" vs "an
        // earlier-task bug only exercised at full 2^25 scale". Round 5 (i==4)
        // is covered by the survivors>=1 assertion below instead: it needs a
        // non-zero survivor_scan hit, a strictly stronger condition than out>0.
        if (r < 4) {
            char lbl[96];
            std::snprintf(lbl, sizeof lbl, "round %d produced output (out > 0) -- sanity vs silent early-collapse", r + 1);
            check(st.out > 0, lbl);
        }
    }
    std::printf("  totals: bucketDrops=%u pairDrops=%u\n", totalBucketDrops, totalPairDrops);
    std::printf("  wall-clock: %.3fs\n", wallSecs);

    // NO GENUINE COLLISION MAY BE DROPPED. A round that emits more children than
    // out_capacity clamps the excess in nondeterministic atomic-cursor order, so
    // WHICH children survive varies run-to-run -- and a dropped child can be a
    // valid solution's ancestor, silently losing that solution (~10% of solves).
    // The verify gate keeps that SAFE (never a bad submit) but LOSSY. A correct
    // full search must size the buffers with enough headroom over the 2^25 seed
    // count that every genuine collision fits: bucketDrops == pairDrops == 0.
    check(totalBucketDrops == 0, "no element dropped by scatter overflow (bucketDrops == 0)");
    check(totalPairDrops == 0, "no genuine collision dropped by out_capacity clamp (pairDrops == 0) -- else solutions are lost nondeterministically");

    // ---- PERF (deliberately unmissable) --------------------------------
    // One solve's cost, printed next to the reference miner's reference so "1.27s" reads
    // as "~35x too slow", not "test passed". Timing is hardware-dependent, so
    // this is LOGGED, never asserted -- a wall-time gate would flake across
    // GPUs (M3 Max vs 4070 Ti vs a CI software rasterizer). The number beside
    // the reference is the catch; a human reads it.
    {
        const double kRefSolPerSec  = 53.0;   // the reference miner ~53 sol/s, RTX 4070 Ti SUPER
        const double kSolsPerNonce  = 1.9;    // Equihash <150,5> statistical avg solutions/nonce
        const double kRefSolvePerSec = kRefSolPerSec / kSolsPerNonce;   // ~28 solve/s
        double solvePerSec = wallSecs > 0 ? 1.0 / wallSecs : 0.0;
        std::printf("\n  ==================== PERF ====================\n");
        std::printf("  this GPU : 1 solve in %.3fs  ->  %.2f solve/s  (~%.1f sol/s at ~%.1f sol/solve)\n",
                    wallSecs, solvePerSec, solvePerSec * kSolsPerNonce, kSolsPerNonce);
        std::printf("  reference: the reference miner ~%.0f sol/s  (~%.0f solve/s)  on a 4070 Ti SUPER\n",
                    kRefSolPerSec, kRefSolvePerSec);
        if (solvePerSec > 0.0)
            std::printf("  verdict  : ~%.0fx slower than reference  (unoptimized kernels = Phase D)\n",
                        kRefSolvePerSec / solvePerSec);
        std::printf("  breakdown: seed=%.1fms  survivor=%.1fms  pipeline-total=%.1fms\n",
                    res.t_seed_ms, res.t_survivor_ms, res.t_total_ms);
        std::printf("  =============================================\n\n");
    }
    // Smoke check on the instrumentation itself (NOT a perf gate): the timers
    // must actually be populated, so a future refactor that silently drops the
    // per-phase timing fails here instead of going dark.
    check(res.t_total_ms > 0.0, "pipeline timing is populated (t_total_ms > 0)");

    std::printf("  survivors=%u survivor_slots.size()=%zu (tests/vectors/beamhash3-kat.md documents exactly 3 known goldens for this KAT input)\n",
                res.survivors, res.survivor_slots.size());
    check(res.survivor_slots.size() == res.survivors, "survivor_slots.size() == the reported survivor count");

    // THE MILESTONE. Golden expectation: with a full, memory-generous budget
    // (drops honestly logged above, not hidden), >=1 of the KAT input's 3
    // known solutions must survive the search. Do NOT weaken this to >=0 or
    // catch/ignore a failure -- a 0 here is a real, reportable bug.
    check(res.survivors >= 1, "GPU 5-round search yields is_zero survivor(s) on KAT input");

    return summary("gpu_rounds");
}
