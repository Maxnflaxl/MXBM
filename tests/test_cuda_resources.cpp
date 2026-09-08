// The register / shared-memory cliff contract for the CUDA kernels.
//
// WHY THIS EXISTS. Four of the nine kernels sit EXACTLY on an occupancy cliff -- one
// more register and they lose a resident block per SM:
//
//   r1        48 registers of 48, AND 18 980 B of shared of 19 456
//   r2        64 of 64, both record formats
//   r3-quad   80 of 80
//   entry_scatter  40 of 40
//
// Losing a block is measured time (r1's fifth block -0.15 ms, docs/performance-research
// .md:1719; r1 and r2 crossing 3 -> 4 together -1.13 ms, :1609) and it is COMPLETELY
// SILENT: 64 -> 65 registers does not spill, emits no ptxas diagnostic, and presents
// downstream as a mysterious few-percent slowdown on a benchmark nobody re-ran.
//
// WHY NOT __launch_bounds__(kWG, minBlocks). Because it is a hint, not an assertion.
// When ptxas cannot meet the register target it does not fail the build, it spills to
// local memory -- trading measured time for an unmeasured loss. And it cannot see
// the 64 -> 65 case at all, because that does not spill. A launch bound silently
// breaking a kernel is already on this project's record (terminal_round bounded at 256
// while launched with 288, docs/performance-research.md:1877-1880). Worse, a WELL-MEANT
// bound is itself a way to lose the blocks: __launch_bounds__(kWG, 3) is satisfied
// exactly, with zero spills and zero warnings, by letting r1 and r2 grow to 80
// registers -- which costs r1 two blocks and r2 one. This test catches that; ptxas does
// not. fused_round does carry __launch_bounds__(kWG) with NO minBlocks argument
// (kernels/cuda/fused_round.cuh:290); that is a maxThreadsPerBlock declaration, and is
// a different thing.
//
// WHAT THIS DOES. Reads the resource usage ptxas actually produced, out of the linked
// library, with cuobjdump -res-usage -- measuring the shipped binary rather than asking
// the compiler for a promise. Then it recomputes blocks/SM from first principles and
// asserts the contract. It needs no GPU: cuobjdump is a static ELF reader.
//
// TWO FAILURE CLASSES, deliberately distinguished:
//   OCCUPANCY REGRESSION - blocks/SM fell. This is the banked time. Never re-baseline;
//                          fix the kernel, or measure and accept the loss.
//   RESOURCE DRIFT       - REG/SHARED/STACK moved but blocks/SM held. Re-baselining is
//                          legitimate here; the message prints the new values and the
//                          headroom left to the next cliff, so the decision is made with
//                          the distance-to-cliff in front of you.
//
// AND IT PROVES ITSELF. A guard never seen to fail is not known to work. After checking
// the real library, this test takes that same cuobjdump output, adds ONE register to
// each kernel in turn, and requires the checker to report a new violation every time. If
// the detector has gone blind -- a parse change, a renamed kernel, a table edited to
// match whatever the compiler happened to emit -- the self-check fails even though the
// build is clean.

#include "check.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>

#ifdef _WIN32
#include <io.h>
#define popen  _popen
#define pclose _pclose
#else
#include <unistd.h>
#endif

#ifndef MXBM_CUOBJDUMP
#define MXBM_CUOBJDUMP "cuobjdump"
#endif
#ifndef MXBM_CUDA_LIB
#define MXBM_CUDA_LIB ""
#endif
#ifndef MXBM_KERNEL_DIR
#define MXBM_KERNEL_DIR ""
#endif

namespace {

// ---------------------------------------------------------------------------------
// 1. The SM model.
//
// Architectural constants of compute capability 8.9 (Ada; `nvidia-smi --query-gpu=
// compute_cap` reports 8.9 for the RTX 4070 Ti SUPER). Documented constants, not
// measurements: the CUDA C Programming Guide's "Technical Specifications per Compute
// Capability" for the per-SM totals, and the CUDA occupancy calculator's own
// granularities -- /opt/cuda/include/cuda_occupancy.h,
// cudaOccSMemAllocationGranularity (128 B for computeMajor 8),
// cudaOccRegAllocationGranularity (256 registers, allocated per warp),
// cudaOccSubPartitionsPerMultiprocessor (4).
//
// The repository corroborates the parts that carry the result:
//   100 KB shared/SM              docs/performance-research.md:13, :1891
//   ~1 KB/block driver reserve    docs/performance-research.md:1771 -- "4 blocks/SM
//                                 needs <= 24 576 B of shared per block (25 600 minus
//                                 ~1 KB the driver reserves)", which this model
//                                 reproduces exactly
//   80 -> 64 registers is 3 -> 4 blocks at 256 threads   docs/performance-research.md:1587-1592
//   entry_scatter gets 6 blocks/SM                       docs/performance-research.md:951
// ---------------------------------------------------------------------------------
struct SmModel {
    int regsPerSM;              // 65 536 32-bit registers
    int maxWarpsPerSM;          // 48 resident warps
    int maxBlocksPerSM;         // 24 resident blocks
    int sharedPerSM;            // 102 400 B addressable as shared at max carveout, which
                                //   the solver asks for (src/gpu/cuda_solver.cu:214-222)
    int subPartitions;          // 4 -- registers are allocated per sub-partition
    int regAllocUnit;           // 256 registers, allocated per warp
    int smemAllocUnit;          // 128 B
    int reservedSmemPerBlock;   // 1024 B taken out of every block's shared by the driver
    int warpSize;               // 32
};
constexpr SmModel kSm89 = { 65536, 48, 24, 102400, 4, 256, 128, 1024, 32 };
constexpr const char* kArch = "sm_89";

int round_up(int x, int unit) { return unit * ((x + unit - 1) / unit); }
int warps_per_cta(const SmModel& m, int blockSize) {
    return (blockSize + m.warpSize - 1) / m.warpSize;
}

// Registers. This is cuda_occupancy.h's cudaOccMaxBlocksPerSMRegsLimit with partitioned
// global caching off (the default for compute capability >= 7): registers are allocated
// per warp rounded up to 256, and warps are placed per SUB-PARTITION, so the floor is
// taken on regsPerSM/4 rather than on regsPerSM. The simpler
// floor(65536 / (warpsPerBlock * regsPerWarp)) agrees with this on all nine kernels
// shipped today but is not equivalent in general (e.g. 3 warps/block at 1536 reg/warp:
// 14 blocks by the simple form, 13 by this one).
int blocks_from_regs(const SmModel& m, int regs, int blockSize) {
    if (regs <= 0) return m.maxBlocksPerSM;
    const int wpc          = warps_per_cta(m, blockSize);
    const int allocPerWarp = round_up(regs * m.warpSize, m.regAllocUnit);
    // Hardware check: the launch is verified as if all sub-partitions were allocated.
    if (allocPerWarp * round_up(wpc, m.subPartitions) > m.regsPerSM) return 0;
    const int warpsPerSubPart = (m.regsPerSM / m.subPartitions) / allocPerWarp;
    return (warpsPerSubPart * m.subPartitions) / wpc;
}

// Shared memory. The driver's per-block reservation is added BEFORE the 128 B rounding.
int blocks_from_smem(const SmModel& m, int smem) {
    const int perBlock = round_up(smem + m.reservedSmemPerBlock, m.smemAllocUnit);
    return perBlock > 0 ? m.sharedPerSM / perBlock : m.maxBlocksPerSM;
}

int blocks_from_warps(const SmModel& m, int blockSize) {
    return m.maxWarpsPerSM / warps_per_cta(m, blockSize);
}

int occupancy(const SmModel& m, int regs, int smem, int blockSize) {
    int b = blocks_from_regs(m, regs, blockSize);
    const int s = blocks_from_smem(m, smem);
    const int w = blocks_from_warps(m, blockSize);
    if (s < b) b = s;
    if (w < b) b = w;
    if (m.maxBlocksPerSM < b) b = m.maxBlocksPerSM;
    return b;
}

// The cliffs, as the largest value that still reaches `want` blocks/SM with the other
// resource held at its measured value. Searched rather than solved in closed form, so
// the printed limit cannot drift away from the model above.
int max_regs_for(const SmModel& m, int blockSize, int smem, int want) {
    for (int r = 255; r >= 1; --r)
        if (occupancy(m, r, smem, blockSize) >= want) return r;
    return 0;
}
int max_smem_for(const SmModel& m, int blockSize, int regs, int want) {
    for (int s = m.sharedPerSM; s >= 0; --s)
        if (occupancy(m, regs, s, blockSize) >= want) return s;
    return -1;
}

// ---------------------------------------------------------------------------------
// 2. The contract.
//
// blockSize is the launch configuration, not a guess:
//   entry_scatter   256   src/gpu/cuda_solver.cu:243  <<<(kElems+255)/256, 256>>>
//   fused_round     kWG   src/gpu/cuda_solver.cu:251  <<<I.nb << I.sm, kWG>>>
//   terminal_round  kWG   src/gpu/cuda_solver.cu:265  <<<I.nb << I.sm, kWG>>>
//   recover          64   src/gpu/cuda_solver.cu:281  <<<(hs+63)/64, 64>>>
// kWG is MXBM_WG = 256 (kernels/cuda/fused_round.cuh:27-29); check_wg() re-reads that
// #define from the source, so this table cannot go stale behind a -DMXBM_WG sweep.
//
// reg/smem/stack are what ptxas produced for sm_89 under CUDA 13.3.73 -- reproduce with
//   cuobjdump -res-usage build/libmxbm_cuda.a
// minBlocks is the CONTRACT: the occupancy that was paid for and measured.
// ---------------------------------------------------------------------------------
struct Kernel {
    const char* name;
    // fused_round is found by the six template arguments that identify the round:
    // INW, OUTW, LEAFW, LMODE, INSTR, OUTSTR. (LMODE alone does not separate r2's two
    // record formats, and FCAP is deliberately NOT part of the identity, so a cap sweep
    // reports a resource change on a known kernel rather than an unknown kernel.)
    // The other three are found by the <length><name> token of the Itanium mangling,
    // which is stable under a change to their parameter list.
    bool        templated;
    // id[6] is the COBLOCKS template argument: the speculative-entry variant of a round
    // is a distinct kernel with the same six round numbers, so the identity needs the
    // bool. Rows written with six values zero-initialise it, which matches the plain
    // kernels.
    int         id[11];
    const char* token;
    int         blockSize;
    int         reg;
    int         smem;
    int         stack;
    int         minBlocks;
    const char* note;
};

// Template-argument order (kernels/cuda/fused_round.cuh:281-289):
//   INW OUTW LEAFW LMODE | LOUT PADN SIN SOUT SBUILD | INSTR OUTSTR | FCAP | COTENANT SUBPASS
// LMode values (kernels/cuda/fused_round.cuh:212-220):
//   LM_EMIT 1, LM_USE 2, LM_SEED 3, LM_RD2 4, LM_SEEDF 5, LM_RAW 6, LM_RD3 7
//
// Shared figures re-baselined 2026-07-31 for the RoundShared refactor (the arrays moved
// into one struct so fused_pair can union two rounds' layouts): each round pays +12-20 B
// for the struct's 1-sized placeholder members plus padding, blocks/SM unchanged on
// every kernel and registers unchanged or lower (r4 47 -> 46). Re-baselined again
// 2026-08-14 when the arena and LM_RD4 scratch joined that convention: +8 B on the 34
// rows where their mode is off, registers, stack and blocks/SM unchanged on all of them.
const Kernel kContract[] = {
    // name                    tmpl   INW OUT LEAF LM IN OUT   token                wg  reg   smem stk min
    { "entry_scatter",         false, {0,0,0,0,0,0}, "13entry_scatterE",  256,  40,     0,  0, 6,
      "ON A CLIFF: 40 registers is EXACTLY the limit for 6 blocks/SM. Measured at 6 "
      "blocks/SM standalone (docs/performance-research.md:951); the pass costs 2.77 ms" },
    { "r1 (LM_SEED, FCAP 288)", true, {7,7,1,3,1,2,0,0,0,0,1}, nullptr,             256,  48, 19016,  0, 5,
      "ON TWO CLIFFS: 48 registers of 48 AND 19016 B of 19456. r1's fifth block is "
      "worth 0.15 ms (docs/performance-research.md:1719)" },
    { "r2 (packed record)",     true, {7,7,2,4,2,8,0,0,0,0,1}, nullptr,             256,  64, 23624,  0, 4,
      "ON A CLIFF: 64 registers is EXACTLY the limit for 4 blocks/SM. r1 and r2 "
      "crossing 3 -> 4 together was worth 1.13 ms (docs/performance-research.md:1609)" },
    { "r2 (quad record)",       true, {7,7,2,4,2,3,0,0,0,0,1}, nullptr,             256,  64, 23624,  0, 4,
      "ON A CLIFF: 64 registers is EXACTLY the limit for 4 blocks/SM" },
    { "r3 (packed record)",     true, {7,6,4,1,8,8,0,0,0,0,1}, nullptr,             256,  56, 26184,  0, 3,
      "shared-bound at 3 blocks; r3 does not want a fourth "
      "(docs/performance-research.md:1731-1745)" },
    { "r3 (quad record)",       true, {7,6,4,7,3,8,0,0,0,0,1}, nullptr,             256,  80, 26184,  0, 3,
      "ON A CLIFF: 80 registers is EXACTLY the limit for 3 blocks/SM" },
    { "r4 (LM_USE)",            true, {6,1,2,2,8,1,0,0,0,0,1}, nullptr,             256,  47, 22336,  0, 4,
      "47 of 64: the reference row and the gi atomic both went, and the 4-block line\n"
      "      is where it was" },
    // The match-first variants: same rounds, chain built at staging so the rebuild can
    // skip the elements no walk reads. mlist costs ~640 B of shared per round, which is
    // what takes r1 off its fifth block -- the reason the variant is selected only in
    // the low-power band. See docs/performance-research.md.
    { "r1 match-first",         true, {7,7,1,3,1,2,0,1,0,0,1}, nullptr,         256,  64, 19592,  0, 4,
      "the fifth block is GONE (5 -> 4): mlist's 640 B crosses r1's 19456 B line, and "
      "registers go 48 -> 64 with it" },
    { "r2 match-first",         true, {7,7,2,4,2,8,0,1,0,0,1}, nullptr,         256,  64, 24256,  0, 4, "" },
    { "r2 implicit-bits",       true, {7,7,2,4,2,8,0,0,16,0,1}, nullptr,       256,  64, 23624,  0, 4,
      "the packed stores fold the repack; resources identical to the base record, "
      "still exactly on the 64-register cliff" },
    { "r2 implicit-bits mf",    true, {7,7,2,4,2,8,0,1,16,0,1}, nullptr,       256,  64, 24256,  0, 4, "" },
    { "r3 implicit-bits",       true, {7,6,4,1,8,8,0,0,16,0,1}, nullptr,       256,  60, 26184,  0, 3, "" },
    { "r3 implicit-bits mf",    true, {7,6,4,1,8,8,0,1,16,0,1}, nullptr,       256,  60, 26816,  0, 3, "" },
    // The (17,0) pack: same code with 17 address-implied bits, selected under the
    // low-power gate. Resources identical to the 16-bit pack on every variant --
    // r2 stays exactly on the 64-register cliff.
    { "r2 implicit-bits 17",    true, {7,7,2,4,2,8,0,0,17,0,1}, nullptr,       256,  64, 23624,  0, 4, "" },
    { "r2 implicit-bits 17 mf", true, {7,7,2,4,2,8,0,1,17,0,1}, nullptr,       256,  64, 24256,  0, 4, "" },
    { "r3 implicit-bits 17",    true, {7,6,4,1,8,8,0,0,17,0,1}, nullptr,       256,  60, 26184,  0, 3, "" },
    { "r3 implicit-bits 17 mf", true, {7,6,4,1,8,8,0,1,17,0,1}, nullptr,       256,  60, 26816,  0, 3, "" },
    // Round 1 at the rung's IMPB. With MXBM_PAIR_W0 the emit packs the w0-checkpoint
    // record from registers it already holds, so every variant matches its IMPB=0 twin
    // exactly and the same rows hold with the checkpoint off -- r1 keeps both of its
    // cliffs either way. Round 2 reads the record and is unchanged above: same template
    // arguments, one branch swapped inside.
    { "r1 implicit-bits",       true, {7,7,1,3,1,2,0,0,16,0,1}, nullptr,       256,  48, 19016,  0, 5,
      "STILL ON TWO CLIFFS: 48 registers of 48, 19016 B of 19456 -- the checkpoint is "
      "packed from registers the emit already holds" },
    { "r1 implicit-bits mf",    true, {7,7,1,3,1,2,0,1,16,0,1}, nullptr,       256,  64, 19592,  0, 4, "" },
    { "r1 implicit-bits 17",    true, {7,7,1,3,1,2,0,0,17,0,1}, nullptr,       256,  48, 19016,  0, 5, "" },
    { "r1 implicit-bits 17 mf", true, {7,7,1,3,1,2,0,1,17,0,1}, nullptr,       256,  64, 19592,  0, 4, "" },
    { "r1 implicit-bits arena", true, {7,7,1,3,1,2,0,0,16,1,1}, nullptr,       256,  48, 19144,  0, 5, "" },
    { "r1 implicit-bits mf arena", true, {7,7,1,3,1,2,0,1,16,1,1}, nullptr,    256,  64, 19720,  0, 4, "" },
    { "r1 implicit-bits 17 arena", true, {7,7,1,3,1,2,0,0,17,1,1}, nullptr,    256,  48, 19144,  0, 5, "" },
    { "r1 implicit-bits 17 mf arena", true, {7,7,1,3,1,2,0,1,17,1,1}, nullptr, 256,  64, 19720,  0, 4, "" },
    { "r2 match-first (quad)",  true, {7,7,2,4,2,3,0,1,0,0,1}, nullptr,         256,  64, 24256,  0, 4, "" },
    { "r3 match-first",         true, {7,6,4,1,8,8,0,1,0,0,1}, nullptr,         256,  54, 26816,  0, 3, "" },
    { "r3 match-first (quad)",  true, {7,6,4,7,3,8,0,1,0,0,1}, nullptr,         256,  80, 26816,  0, 3, "" },
    { "r4 match-first",         true, {6,1,2,2,8,1,0,1,0,0,1}, nullptr,         256,  47, 22976,  0, 4, "" },
    { "r4 (entry co-blocks)",   true, {6,1,2,2,8,1,1,0,0,0,1}, nullptr,           256,  64, 22336,  0, 4,
      "ON A CLIFF: hosting the speculative entry pass costs 18 registers (46 -> 64), "
      "landing EXACTLY on the 4-blocks/SM limit. One more and the whole launch -- the "
      "round AND the co-scheduled entry -- drops to 3 blocks" },
    // Re-baselined 2026-07-31 for the perfect chain table (MXBM_PERFECT_TAB reaching
    // terminal_round): lkey's 4 B x 384 of shared removed, REG 22 -> 24. blocks/SM held.
    // The arena rungs: every round again with dense per-bucket caps and one overflow
    // pool per set. The pool's per-bucket shared list costs 128-136 B, and THE POINT OF
    // THESE ROWS IS THAT NO ROUND LOSES A BLOCK TO IT -- r1 keeps its fifth (19144 B of
    // 19456), r2 its fourth on exactly 64 registers, r3 and r4 their third and fourth.
    // Registers move only on r2/r3's plain record (56 -> 60 at r3), which is shared-bound
    // anyway. The IMPB 17 pairs are instantiated by the round macro and reachable through
    // MXBM_BB=17 with MXBM_ARENA=1, which is how the rung is A/B'd.
    { "r1 arena",               true, {7,7,1,3,1,2,0,0,0,1,1}, nullptr,    256,  48, 19144,  0, 5,
      "STILL ON BOTH CLIFFS: 48 registers of 48, 19144 B of 19456 -- the pool's shared "
      "list fits in r1's 448 B of headroom and the fifth block survives" },
    { "r1 mf arena",            true, {7,7,1,3,1,2,0,1,0,1,1}, nullptr,    256,  64, 19720,  0, 4, "" },
    { "r2 arena",               true, {7,7,2,4,2,8,0,0,0,1,1}, nullptr,    256,  64, 23752,  0, 4, "" },
    { "r2 mf arena",            true, {7,7,2,4,2,8,0,1,0,1,1}, nullptr,    256,  64, 24384,  0, 4, "" },
    { "r3 arena",               true, {7,6,4,1,8,8,0,0,0,1,1}, nullptr,    256,  60, 26312,  0, 3, "" },
    { "r3 mf arena",            true, {7,6,4,1,8,8,0,1,0,1,1}, nullptr,    256,  62, 26944,  0, 3, "" },
    { "r2 implicit-bits arena", true, {7,7,2,4,2,8,0,0,16,1,1}, nullptr,   256,  64, 23752,  0, 4, "" },
    { "r2 implicit-bits mf arena", true, {7,7,2,4,2,8,0,1,16,1,1}, nullptr,256,  64, 24384,  0, 4, "" },
    { "r3 implicit-bits arena", true, {7,6,4,1,8,8,0,0,16,1,1}, nullptr,   256,  58, 26312,  0, 3, "" },
    { "r3 implicit-bits mf arena", true, {7,6,4,1,8,8,0,1,16,1,1}, nullptr,256,  58, 26944,  0, 3, "" },
    { "r2 implicit-bits 17 arena", true, {7,7,2,4,2,8,0,0,17,1,1}, nullptr,256,  64, 23752,  0, 4, "" },
    { "r2 implicit-bits 17 mf arena", true, {7,7,2,4,2,8,0,1,17,1,1}, nullptr, 256, 64, 24384, 0, 4, "" },
    { "r3 implicit-bits 17 arena", true, {7,6,4,1,8,8,0,0,17,1,1}, nullptr,256,  58, 26312,  0, 3, "" },
    { "r3 implicit-bits 17 mf arena", true, {7,6,4,1,8,8,0,1,17,1,1}, nullptr, 256, 58, 26944, 0, 3, "" },
    { "r2 quad arena",          true, {7,7,2,4,2,3,0,0,0,1,1}, nullptr,    256,  64, 23752,  0, 4, "" },
    { "r2 quad mf arena",       true, {7,7,2,4,2,3,0,1,0,1,1}, nullptr,    256,  64, 24384,  0, 4, "" },
    { "r3 quad arena",          true, {7,6,4,7,3,8,0,0,0,1,1}, nullptr,    256,  80, 26312,  0, 3,
      "ON A CLIFF: 80 registers is EXACTLY the limit for 3 blocks/SM" },
    { "r3 quad mf arena",       true, {7,6,4,7,3,8,0,1,0,1,1}, nullptr,    256,  80, 26944,  0, 3, "" },
    { "r4 arena",               true, {6,1,2,2,8,1,0,0,0,1,1}, nullptr,    256,  47, 22464,  0, 4, "" },
    { "r4 mf arena",            true, {6,1,2,2,8,1,0,1,0,1,1}, nullptr,    256,  47, 23104,  0, 4, "" },
    // The octo record. Round 3's side is free -- same kernel, a narrower store, and the
    // 80-register cliff it already sat on; its rows are all no-refs, below. Round 4's side is where it is paid: rebuilding
    // six work words from eight leaves costs 46 -> 128 registers, which is EXACTLY the
    // limit for 2 blocks/SM, so the round halves its residency as well as its input
    // bytes. That trade is the rung's whole question and it is measured, not assumed.
    { "r4 octo arena",          true, {6,1,2,8,4,2,0,0,0,1,1}, nullptr,  256,  80, 23744,  0, 3,
      "ON A CLIFF: 80 registers is EXACTLY the limit for 3 blocks/SM. Left to itself the "
      "rebuild takes 128 and 2 blocks; MXBM_MB_OCTO asks for the third and ptxas finds it. "
      "Zero spill: with the mix chain gone the lane's live state fits those 80 registers" },
    { "r4 octo mf arena",       true, {6,1,2,8,4,2,0,1,0,1,1}, nullptr,  256,  80, 24376,  0, 3,
      "+1280 B over its reference-by-gi twin: one u32 per staged element recording the\n      slot it was read from, which is what round 4 names its parents by" },
    // Rounds 1-3 of an octo rung, which write no back-references because rows 1-3 are not
    // allocated there. Resources are identical to their reference-writing twins on every
    // one -- r1 keeps its fifth block, r2 its fourth, r3 its third -- so what the two
    // deleted stores per element buy is traffic, not occupancy.
    { "r1 arena no-refs",       true, {7,7,1,3,1,2,0,0,0,1,0}, nullptr,  256,  48, 19144,  0, 5, "" },
    { "r1 mf arena no-refs",    true, {7,7,1,3,1,2,0,1,0,1,0}, nullptr,  256,  64, 19720,  0, 4, "" },
    { "r2 quad16 arena no-refs", true, {7,7,2,4,2,2,0,0,0,1,0}, nullptr, 256,  64, 23752,  0, 4,
      "the gi-less 16 B quad record: same kernel, one u64 less stored, and the same 64-\n      register cliff" },
    { "r2 quad16 mf arena no-refs", true, {7,7,2,4,2,2,0,1,0,1,0}, nullptr, 256, 64, 24384, 0, 4, "" },
    { "r3 quad16 octo no-refs", true, {7,6,4,7,2,4,0,0,0,1,0}, nullptr,  256,  80, 26312,  0, 3, "" },
    { "r3 quad16 octo mf no-refs", true, {7,6,4,7,2,4,0,1,0,1,0}, nullptr, 256, 80, 26944,  0, 3, "" },
    { "terminal_round",        false, {0,0,0,0,0,0}, "14terminal_roundILb0ELj1EE", 256, 32, 14340, 0, 6,
      "warp-capped at 6 (48 warps/SM / 8 warps per block), not resource-bound. One block\n"
      "      per bucket (kTermCap 832, a 256-entry table) with its record loads hoisted\n"
      "      ahead of the filter: 20 -> 32 registers, 6660 -> 14340 B, blocks held" },
    { "terminal_round (arena)", false, {0,0,0,0,0,0}, "14terminal_roundILb1ELj1EE", 256, 29, 14472, 0, 6,
      "the pool chain's shared list costs 132 B and no block: still warp-capped at 6" },
    { "terminal_round (octo)",  false, {0,0,0,0,0,0}, "14terminal_roundILb1ELj2EE", 256, 33, 21128, 0, 4,
      "the 16 B arm: an octo round 4 keeps its reference row, so it keeps gi and lead;\n"
      "      at one block per bucket that is 21 KB and four blocks, measured -6 % on the\n"
      "      round against six" },
    { "arena_link",            false, {0,0,0,0,0,0}, "10arena_linkE",      256,  12,     0,  0, 6,
      "threads the overflow pool onto per-bucket chains between rounds; 256 blocks" },
    // recover's 64 B of stack is a genuine local array, not a spill: ptxas -v reports
    // "64 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads". Its grid is at
    // most (1024+63)/64 = 17 blocks over 66 SMs, so occupancy is not a lever here and
    // minBlocks is only the hardware cap; the load-bearing assertion is the stack size.
    // The five-level recover<false> ships only under MXBM_R4_ROWS, where it is the
    // replay's reference arm; the packed rungs enter at level 3 instead.
    { "recover (octo)",        false, {0,0,0,0,0,0}, "7recoverILb1EE",     64,  40,     0,  0, 24,
      "no stack at all: the octo walk is two levels deep and flat, so the explicit DFS "
      "stack the five-level form needs is gone" },
    { "recover_from_l3",       false, {0,0,0,0,0,0}, "15recover_from_l3E",  64,  26,     0, 64, 24,
      "the same walk entered at level 3; same 64 B DFS stack, same <= 17-block grid" },
    { "recover_from_l2",       false, {0,0,0,0,0,0}, "15recover_from_l2E",  64,  40,     0,  0, 24,
      "no walk and no stack: the eight round-2 records ARE the 32 leaves" },
    { "replay_r3",             false, {0,0,0,0,0,0}, "9replay_r3E",        256,  62, 16644, 144, 4,
      "four blocks per survivor, so ~8 in the whole grid; the 144 B stack is the two\n"
      "      unpacked 7-word records, not a spill of the hot path" },
    { "replay_r4",             false, {0,0,0,0,0,0}, "9replay_r4E",        256,  48, 16644,  0, 5,
      "two blocks per survivor, so ~4 in the whole grid: occupancy is not a lever here,\n"
      "      and the 4096-key staging that costs the fifth block is what covers bb = 14" },
};
constexpr int kNumKernels = (int)(sizeof(kContract) / sizeof(kContract[0]));

// ---------------------------------------------------------------------------------
// 3. Reading what ptxas produced.
// ---------------------------------------------------------------------------------
struct Measured {
    std::string mangled;
    std::vector<long long> targs;
    int reg = -1, smem = -1, stack = -1;
    int match = -1;                     // index into kContract, -1 unknown, -2 ambiguous
};

void failf(const char* fmt, ...) {
    ++mxbm::fail_count();
    std::va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stdout, fmt, ap);
    va_end(ap);
}

const char* cuobjdump_path() {
#ifdef _WIN32
    // The CMake-configured path has no extension and forward slashes; cmd.exe
    // wants both fixed before it will execute it.
    static std::string p = [] {
        std::string s = std::string(MXBM_CUOBJDUMP) + ".exe";
        for (char& c : s) if (c == '/') c = '\\';
        return s;
    }();
    if (_access(p.c_str(), 0) == 0) return p.c_str();
#else
    static std::string p = MXBM_CUOBJDUMP;
    if (!p.empty() && ::access(p.c_str(), X_OK) == 0) return p.c_str();
#endif
    return "cuobjdump";
}

bool run_cuobjdump(std::string& out) {
#ifdef _WIN32
    // cmd.exe quoting: one extra outer pair -- cmd strips the first and last
    // quote and runs the rest verbatim, spaces in Program Files included.
    const std::string cmd = std::string("\"\"") + cuobjdump_path() + "\" -res-usage \"" +
                            MXBM_CUDA_LIB + "\" 2>&1\"";
#else
    const std::string cmd = std::string("'") + cuobjdump_path() + "' -res-usage '" +
                            MXBM_CUDA_LIB + "' 2>&1";
#endif
    FILE* f = ::popen(cmd.c_str(), "r");
    if (!f) return false;
    char buf[4096];
    while (std::fgets(buf, sizeof buf, f)) out += buf;
    return ::pclose(f) == 0;
}

// "  REG:80 STACK:0 SHARED:26148 LOCAL:0 ..." -> the integer after `key`.
int field(const std::string& line, const char* key) {
    const size_t p = line.find(key);
    if (p == std::string::npos) return -1;
    return std::atoi(line.c_str() + p + std::strlen(key));
}

// Pull the L<type><value>E template-argument list out of an Itanium-mangled
// fused_round instantiation. Returns false if `s` is not a fused_round.
bool parse_targs(const std::string& s, std::vector<long long>& out) {
    static const char kMark[] = "11fused_roundI";
    size_t p = s.find(kMark);
    if (p == std::string::npos) return false;
    p += sizeof(kMark) - 1;
    while (p < s.size() && s[p] == 'L') {
        p += 2;                                     // 'L' and the type letter (i/j/b)
        bool neg = false;
        if (p < s.size() && s[p] == 'n') { neg = true; ++p; }
        long long v = 0;
        const size_t d0 = p;
        while (p < s.size() && s[p] >= '0' && s[p] <= '9') v = v * 10 + (s[p++] - '0');
        if (p == d0 || p >= s.size() || s[p] != 'E') return false;
        ++p;
        out.push_back(neg ? -v : v);
    }
    return true;
}

// Split cuobjdump's listing into (mangled name, REG, STACK, SHARED), restricted to the
// sm_89 code section. `sawArch` reports whether such a section existed at all.
std::vector<Measured> parse_res_usage(const std::string& out, bool& sawArch) {
    std::vector<Measured> found;
    sawArch = false;
    bool inArch = false;
    Measured* pending = nullptr;
    std::string line;
    for (size_t i = 0; i <= out.size(); ++i) {
        if (i != out.size() && out[i] != '\n') { line += out[i]; continue; }
        if (line.find("arch =") != std::string::npos) {
            inArch = line.find(kArch) != std::string::npos;
            sawArch = sawArch || inArch;
            pending = nullptr;
        } else if (inArch && line.compare(0, 10, " Function ") == 0) {
            const size_t e = line.find_last_of(':');
            found.push_back(Measured{});
            found.back().mangled = line.substr(10, e - 10);
            parse_targs(found.back().mangled, found.back().targs);
            pending = &found.back();
        } else if (pending && line.find("REG:") != std::string::npos) {
            pending->reg   = field(line, "REG:");
            pending->stack = field(line, "STACK:");
            pending->smem  = field(line, "SHARED:");
            pending = nullptr;
        }
        line.clear();
    }
    // The kFCap64K instantiations (rounds 2 and 4 staged at 280, round 3 at 272 with
    // its narrow word-6 plane) launch only on a card with 64 KB of shared memory per
    // SM. The reference card never runs them, so they carry no row and are set aside
    // before matching; their cliff is Turing's.
    found.erase(std::remove_if(found.begin(), found.end(), [](const Measured& k) {
                    return k.targs.size() >= 19 &&
                           (k.targs[11] == 280 || k.targs[11] == 272); }),
                found.end());
    for (Measured& k : found) {
        for (int i = 0; i < kNumKernels; ++i) {
            const Kernel& c = kContract[i];
            bool hit;
            if (c.templated)
                hit = k.targs.size() >= 19 &&
                      k.targs[0] == c.id[0] && k.targs[1] == c.id[1] &&
                      k.targs[2] == c.id[2] && k.targs[3] == c.id[3] &&
                      k.targs[9] == c.id[4] && k.targs[10] == c.id[5] &&
                      k.targs[14] == c.id[6] &&    // COBLOCKS variant is its own row
                      k.targs[15] == c.id[7] &&    // so is the match-first variant
                      k.targs[16] == c.id[8] &&    // the implicit-bits record
                      k.targs[17] == c.id[9] &&    // the dense-cap / arena rungs
                      k.targs[18] == c.id[10];     // and whether it writes back-refs
            else
                hit = k.targs.empty() && k.mangled.find(c.token) != std::string::npos;
            if (hit) { k.match = i; break; }
        }
    }
    return found;
}

void print_limits(const SmModel& m, int regs, int smem, int blockSize, const char* pad) {
    const int wpc = warps_per_cta(m, blockSize);
    const int aw  = round_up(regs * m.warpSize, m.regAllocUnit);
    const int wsp = aw > 0 ? (m.regsPerSM / m.subPartitions) / aw : 0;
    const int sb  = round_up(smem + m.reservedSmemPerBlock, m.smemAllocUnit);
    std::printf("%sregisters: ceil%d(%d x %d = %d) = %d reg/warp -> floor(%d/%d) = %d "
                "warps/sub-partition\n"
                "%s           x %d sub-partitions = %d warps/SM -> floor(%d/%d) = %d blocks/SM\n",
                pad, m.regAllocUnit, regs, m.warpSize, regs * m.warpSize, aw,
                m.regsPerSM / m.subPartitions, aw, wsp,
                pad, m.subPartitions, wsp * m.subPartitions, wsp * m.subPartitions, wpc,
                blocks_from_regs(m, regs, blockSize));
    std::printf("%sshared:    ceil%d(%d + %d reserved) = %d B/block -> floor(%d/%d) = %d blocks/SM\n",
                pad, m.smemAllocUnit, smem, m.reservedSmemPerBlock, sb,
                m.sharedPerSM, sb, blocks_from_smem(m, smem));
    std::printf("%swarps:     floor(%d warps/SM / %d warps/block) = %d blocks/SM ; hardware cap %d\n",
                pad, m.maxWarpsPerSM, wpc, blocks_from_warps(m, blockSize),
                m.maxBlocksPerSM);
}

// The whole contract, applied to one cuobjdump listing. Returns the number of
// violations. `report` distinguishes the real run (print, and count into fail_count)
// from the self-check below (silent, count only).
int check_listing(const SmModel& m, const std::string& out, bool report) {
    int bad = 0;
    auto note = [&](const char* fmt, ...) {
        ++bad;
        if (!report) return;
        ++mxbm::fail_count();
        std::va_list ap; va_start(ap, fmt); std::vfprintf(stdout, fmt, ap); va_end(ap);
    };

    bool sawArch = false;
    const std::vector<Measured> found = parse_res_usage(out, sawArch);

    std::vector<bool> seen(kNumKernels, false);
    for (const Measured& k : found)
        if (k.match >= 0) {
            if (seen[k.match])
                note("  FAIL: two kernels match the contract entry '%s'; its identity is "
                     "ambiguous.\n        %s\n",
                     kContract[k.match].name, k.mangled.c_str());
            seen[k.match] = true;
        }

    for (int i = 0; i < kNumKernels; ++i)
        if (!seen[i])
            note("  FAIL: kernel '%s' is NOT in %s.\n"
                 "        Either it was renamed or re-parameterised (update its identity "
                 "in kContract, and re-derive its cliff), or it is no longer "
                 "instantiated.\n",
                 kContract[i].name, MXBM_CUDA_LIB);

    for (const Measured& k : found) {
        if (k.reg <= 0 || k.smem < 0 || k.stack < 0) {
            note("  FAIL: could not read REG/STACK/SHARED for %s -- has cuobjdump's "
                 "output format changed?\n", k.mangled.c_str());
            continue;
        }
        if (k.match < 0) {
            note("  FAIL: UNDECLARED KERNEL in %s\n"
                 "        %s\n"
                 "        REG %d  SHARED %d B  STACK %d -> %d blocks/SM at 256 threads\n"
                 "        Every kernel that ships declares the occupancy it was measured "
                 "at. Add it to kContract with its minBlocks.\n",
                 MXBM_CUDA_LIB, k.mangled.c_str(), k.reg, k.smem, k.stack,
                 occupancy(m, k.reg, k.smem, 256));
            continue;
        }
        const Kernel& c = kContract[k.match];
        const int blocks = occupancy(m, k.reg, k.smem, c.blockSize);
        const int rl = max_regs_for(m, c.blockSize, k.smem, c.minBlocks);
        const int sl = max_smem_for(m, c.blockSize, k.reg, c.minBlocks);

        // 3a. The load-bearing assertion.
        if (blocks < c.minBlocks) {
            note("  FAIL: OCCUPANCY REGRESSION in '%s'\n"
                 "        blocks/SM fell %d -> %d ; the contract requires >= %d\n"
                 "        measured  REG %d  SHARED %d B  STACK %d   (block = %d threads "
                 "= %d warps)\n",
                 c.name, c.minBlocks, blocks, c.minBlocks,
                 k.reg, k.smem, k.stack, c.blockSize, warps_per_cta(m, c.blockSize));
            if (report) {
                std::printf("        on %s, %d reg/SM, %d B shared/SM, %d warps/SM:\n",
                            kArch, m.regsPerSM, m.sharedPerSM, m.maxWarpsPerSM);
                print_limits(m, k.reg, k.smem, c.blockSize, "          ");
                // Each cliff quoted against the OTHER resource's baseline, so the line
                // stays where the design put it even when both resources moved.
                const int rlim = max_regs_for(m, c.blockSize, c.smem, c.minBlocks);
                const int slim = max_smem_for(m, c.blockSize, c.reg, c.minBlocks);
                std::printf("        the lines for %d blocks/SM: REG <= %d (at the "
                            "baseline SHARED %d B) and SHARED <= %d B (at the baseline "
                            "REG %d)\n",
                            c.minBlocks, rlim, c.smem, slim, c.reg);
                if (k.reg > rlim)
                    std::printf("        --> REGISTERS over by %d (%d, limit %d)\n",
                                k.reg - rlim, k.reg, rlim);
                if (k.smem > slim)
                    std::printf("        --> SHARED over by %d B (%d, limit %d)\n",
                                k.smem - slim, k.smem, slim);
                if (k.stack > c.stack)
                    std::printf("        --> SPILL: STACK %d -> %d, ptxas ran out of "
                                "registers\n", c.stack, k.stack);
                if (c.note[0]) std::printf("        %s\n", c.note);
                std::printf("        This is banked time. Do NOT re-baseline it.\n");
            }
            continue;
        }

        // 3b. Drift that has not (yet) cost a block.
        if (k.reg != c.reg || k.smem != c.smem || k.stack != c.stack) {
            note("  FAIL: RESOURCE DRIFT in '%s' (blocks/SM held at %d)\n"
                 "        REG    %d -> %d\n"
                 "        SHARED %d -> %d B\n"
                 "        STACK  %d -> %d%s\n"
                 "        headroom left at %d blocks/SM: %d registers, %d B of shared\n"
                 "        If this is intended, update this kernel's row in kContract to "
                 "reg %d, smem %d, stack %d (minBlocks stays %d).\n",
                 c.name, blocks,
                 c.reg, k.reg, c.smem, k.smem, c.stack, k.stack,
                 (k.stack > c.stack) ? "   <-- SPILL: ptxas ran out of registers" : "",
                 c.minBlocks, rl - k.reg, sl - k.smem,
                 k.reg, k.smem, k.stack, c.minBlocks);
            continue;
        }

        if (report && mxbm::verbose()) {
            std::printf("  ok   %-22s REG %3d/%3d  SHARED %6d/%6d B  STACK %2d  -> %d "
                        "blocks/SM (>= %d)\n",
                        c.name, k.reg, rl, k.smem, sl, k.stack, blocks, c.minBlocks);
            print_limits(m, k.reg, k.smem, c.blockSize, "         ");
        }
    }
    return bad;
}

// Add `delta` to the REG: value of one named function in a cuobjdump listing,
// INSIDE the kArch section -- a multi-arch fatbin lists every kernel once per
// architecture and the first textual occurrence is another arch's copy, which
// the parser rightly ignores. Perturbing there made the self-check declare the
// guard blind on the first multi-arch build, so this walks sections exactly
// like parse_res_usage. Used only by the self-check.
bool bump_reg(std::string& s, const std::string& mangled, int delta) {
    const std::string fn = " Function " + mangled + ":";
    bool inArch = false, atFunc = false;
    std::string line;
    size_t lineStart = 0;
    for (size_t i = 0; i <= s.size(); ++i) {
        if (i != s.size() && s[i] != '\n') { line += s[i]; continue; }
        if (line.find("arch =") != std::string::npos) {
            inArch = line.find(kArch) != std::string::npos;
            atFunc = false;
        } else if (inArch && line.compare(0, fn.size(), fn) == 0) {
            atFunc = true;
        } else if (atFunc && line.find("REG:") != std::string::npos) {
            const size_t v = lineStart + line.find("REG:") + 4;
            size_t e = v;
            while (e < s.size() && s[e] >= '0' && s[e] <= '9') ++e;
            if (e == v) return false;
            const int now = std::atoi(s.c_str() + v);
            s.replace(v, e - v, std::to_string(now + delta));
            return true;
        }
        line.clear();
        lineStart = i + 1;
    }
    return false;
}

// kWG is the launch width for six of the nine kernels and it is a #define, so it can be
// swept from the command line without any mangled name changing. Re-read it.
bool check_wg(int expect) {
    const std::string path = std::string(MXBM_KERNEL_DIR) + "/fused_round.cuh";
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) { failf("  FAIL: cannot read %s to confirm MXBM_WG\n", path.c_str()); return false; }
    char buf[512];
    int got = -1;
    while (std::fgets(buf, sizeof buf, f)) {
        const char* p = std::strstr(buf, "#define MXBM_WG");
        if (p) { got = std::atoi(p + std::strlen("#define MXBM_WG")); break; }
    }
    std::fclose(f);
    if (got != expect) {
        failf("  FAIL: MXBM_WG is %d in %s, but this contract's blocks/SM arithmetic "
              "assumes %d\n        threads per block. Re-derive the table.\n",
              got, path.c_str(), expect);
        return false;
    }
    return true;
}

} // namespace

int main() {
    const SmModel& m = kSm89;

    // ---- 0. The contract table must be internally consistent: if minBlocks does not
    //         FOLLOW from the baseline triple by the arithmetic above, it is a magic
    //         number and the whole test means nothing.
    for (int i = 0; i < kNumKernels; ++i) {
        const Kernel& k = kContract[i];
        const int b = occupancy(m, k.reg, k.smem, k.blockSize);
        if (b != k.minBlocks)
            failf("  FAIL: contract table is self-inconsistent for '%s': the baseline "
                  "(REG %d, SHARED %d, %d threads) yields %d blocks/SM, the table claims "
                  "%d\n", k.name, k.reg, k.smem, k.blockSize, b, k.minBlocks);
    }
    check_wg(256);

    // ---- 1. Read what ptxas actually produced.
    std::string out;
    if (!run_cuobjdump(out) || out.empty()) {
        failf("  FAIL: '%s -res-usage %s' produced nothing.\n%s\n",
              cuobjdump_path(), MXBM_CUDA_LIB, out.c_str());
        return mxbm::summary("cuda_resources");
    }
    bool sawArch = false;
    (void)parse_res_usage(out, sawArch);
    if (!sawArch) {
        std::printf("SKIPPED: cuda_resources -- %s carries no %s code. This contract is "
                    "the reference card's (Ada, compute capability 8.9); another "
                    "architecture has a different occupancy model and different "
                    "baselines.\n", MXBM_CUDA_LIB, kArch);
        return mxbm::summary("cuda_resources");
    }

    // ---- 2. The contract.
    const int violations = check_listing(m, out, /*report=*/true);

    // ---- 3. POSITIVE CONTROL. A guard never seen to fail is not known to work. One
    //         register onto each kernel in turn must produce a new violation -- that is
    //         the exact regression this exists to catch (64 -> 65 on r2 costs the fourth
    //         block, does not spill, and warns about nothing).
    if (violations == 0) {
        for (const Measured& k : parse_res_usage(out, sawArch)) {
            if (k.match < 0) continue;
            std::string mutated = out;
            if (!bump_reg(mutated, k.mangled, 1)) {
                failf("  FAIL: self-check could not perturb '%s'\n",
                      kContract[k.match].name);
                continue;
            }
            if (check_listing(m, mutated, /*report=*/false) < 1)
                failf("  FAIL: SELF-CHECK -- the guard is BLIND. '%s' at REG %d + 1 = %d "
                      "was accepted.\n"
                      "        A one-register regression is exactly what this test "
                      "exists to catch; it would not have caught it.\n",
                      kContract[k.match].name, k.reg, k.reg + 1);
            else if (mxbm::verbose()) {
                const Kernel& c = kContract[k.match];
                const int b1 = occupancy(m, k.reg + 1, k.smem, c.blockSize);
                std::printf("  ok   self-check: %-22s REG %d -> %d is caught (%s)\n",
                            c.name, k.reg, k.reg + 1,
                            b1 < c.minBlocks
                              ? "occupancy regression: on a register cliff"
                              : "resource drift: blocks/SM would still hold");
            }
        }
    } else {
        std::printf("  NOTE: self-check skipped -- the contract is already violated, so "
                    "a perturbation proves nothing.\n");
    }

    return mxbm::summary("cuda_resources");
}
