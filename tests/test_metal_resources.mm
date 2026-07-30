// The Metal occupancy contract.
//
// This is the counterpart to tests/test_cuda_resources.cpp, and it is deliberately
// WEAKER, for a reason worth stating rather than discovering later.
//
// The CUDA contract's load-bearing half is the REGISTER count -- it catches a 64 -> 65
// regression that costs a resident block and does not spill or warn. Metal exposes no
// register count, and the alternatives were exhausted rather than assumed:
//
//   * maxTotalThreadsPerThreadgroup does not move with register pressure (measured
//     across 4/64/256 live values: 1024 throughout). The compiler spills instead.
//   * applegpu-nt, the AIR -> native translator, loads a metallib only at AIR 2.5 and
//     then stops at "Unsupported function script" -- an undocumented intermediate that
//     Xcode's build system generates and no shipped tool emits standalone.
//   * MTLBinaryArchive DOES yield the native AGX binary, but its __descriptor and
//     __reflection sections are byte-identical across register-pressure variants
//     (they carry argument reflection), and __compute is raw ISA with no disassembler
//     shipped ("no instruction printer for target agx3").
//
// So this pins what IS observable: threadgroup memory, exactly, per kernel. That is the
// resource the fused rounds are actually bound by -- r3 sits at 79.8% of the 32 KB
// ceiling -- and it is readable at runtime with no external tool.
//
// Register regressions are caught by their EFFECT instead: the median-ms/solve figure
// under MXBM_BENCH in test_metal_solver.mm.
#include "metal_harness.h"
#include <cstdint>

using namespace mxbm;

// Apple's hard ceiling, measured on an M3 Max (maxThreadgroupMemoryLength).
static const uint32_t kCeiling = 32768;

struct Expect {
    const char* kernel;
    uint32_t    tgmem;      // exact, in bytes
    const char* why;
};

// Measured 2026-07-30 on an M3 Max. The CUDA figures for the same kernels are in
// tests/test_cuda_resources.cpp; r1 and r3 agree to within 12 B, which is the
// gcount/cnt8 atomics rounding, and is the strongest evidence available that the two
// ports allocate the same shape.
static const Expect kExpected[] = {
    { "fused_round_r1", 18992, "LM_SEED, FCAP 288, LEAFW 1, gi folded into the leaf "
                               "(CUDA: 18980)" },
    { "fused_round_r2", 23600, "LM_RD2, FCAP 320, LEAFW 2" },
    { "fused_round_r3", 26160, "LM_EMIT, FCAP 320, LEAFW 4 -- the tightest round, "
                               "79.8% of the ceiling (CUDA: 26148)" },
    { "fused_round_r4", 22320, "LM_USE, FCAP 320, INW 6, needs a separate llead" },
    { "terminal_round",  9744, "stages word 0 + 4 meta u32 only; never the constraint" },
    { "entry_scatter",      0, "no threadgroup memory: one element per thread" },
    { "recover",            0, "no threadgroup memory: scalar DFS per survivor" },
};

int main() {
    @autoreleasepool {
        section("Metal threadgroup-memory contract");
        mtest::Harness h;
        if (!h.init()) return summary("metal_resources");

        for (const auto& e : kExpected) {
            auto ps = h.pipeline(e.kernel);
            if (!ps) continue;
            const uint32_t got = (uint32_t)ps.staticThreadgroupMemoryLength;

            char m[96];
            std::snprintf(m, sizeof m, "%s threadgroup bytes", e.kernel);
            check_eq_u64(got, e.tgmem, m);

            std::snprintf(m, sizeof m, "%s fits Apple's 32 KB ceiling", e.kernel);
            check(got <= kCeiling, m);

            if (verbose())
                std::printf("     %-16s %6u B  (%4.1f%%)  %s\n",
                            e.kernel, got, 100.0 * got / kCeiling, e.why);
        }

        // A kernel that stopped being dispatchable at kWG would silently lose its
        // launch configuration, so pin that too. 256 is what fused_round.metal and
        // metal_solver.mm both assume; they must not drift apart.
        for (const char* k : { "fused_round_r1", "fused_round_r2",
                               "fused_round_r3", "fused_round_r4", "terminal_round" }) {
            auto ps = h.pipeline(k);
            if (!ps) continue;
            char m[96];
            std::snprintf(m, sizeof m, "%s can be dispatched at 256 threads", k);
            check(ps.maxTotalThreadsPerThreadgroup >= 256, m);
        }

        // The SIMD width is what makes gi_alloc's simd_prefix_exclusive_sum equivalent
        // to CUDA's warp-aggregated atomic. If Apple ever ships a device with a
        // different width the aggregation is still correct, but the comment claiming
        // 1:1 equivalence with the CUDA idiom would stop being true.
        auto ps1 = h.pipeline("fused_round_r1");
        if (ps1) check(ps1.threadExecutionWidth == 32, "SIMD width is 32, as gi_alloc assumes");

        return summary("metal_resources");
    }
}
