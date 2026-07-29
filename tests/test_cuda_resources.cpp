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
#include <string>
#include <vector>

#include <unistd.h>

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
    int         id[6];
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
const Kernel kContract[] = {
    // name                    tmpl   INW OUT LEAF LM IN OUT   token                wg  reg   smem stk min
    { "entry_scatter",         false, {0,0,0,0,0,0}, "13entry_scatterE",  256,  40,     0,  0, 6,
      "ON A CLIFF: 40 registers is EXACTLY the limit for 6 blocks/SM. Measured at 6 "
      "blocks/SM standalone (docs/performance-research.md:951); the pass costs 2.77 ms" },
    { "r1 (LM_SEED, FCAP 288)", true, {7,7,1,3,1,2}, nullptr,             256,  48, 18980,  0, 5,
      "ON TWO CLIFFS: 48 registers of 48 AND 18980 B of 19456. r1's fifth block is "
      "worth 0.15 ms (docs/performance-research.md:1719)" },
    { "r2 (packed record)",     true, {7,7,2,4,2,8}, nullptr,             256,  64, 23588,  0, 4,
      "ON A CLIFF: 64 registers is EXACTLY the limit for 4 blocks/SM. r1 and r2 "
      "crossing 3 -> 4 together was worth 1.13 ms (docs/performance-research.md:1609)" },
    { "r2 (quad record)",       true, {7,7,2,4,2,3}, nullptr,             256,  64, 23588,  0, 4,
      "ON A CLIFF: 64 registers is EXACTLY the limit for 4 blocks/SM" },
    { "r3 (packed record)",     true, {7,6,4,1,8,8}, nullptr,             256,  56, 26148,  0, 3,
      "shared-bound at 3 blocks; r3 does not want a fourth "
      "(docs/performance-research.md:1731-1745)" },
    { "r3 (quad record)",       true, {7,6,4,7,3,8}, nullptr,             256,  80, 26148,  0, 3,
      "ON A CLIFF: 80 registers is EXACTLY the limit for 3 blocks/SM" },
    { "r4 (LM_USE)",            true, {6,1,2,2,8,2}, nullptr,             256,  47, 22308,  0, 4, "" },
    { "terminal_round",        false, {0,0,0,0,0,0}, "14terminal_roundE", 256,  22,  9732,  0, 6,
      "warp-capped at 6 (48 warps/SM / 8 warps per block), not resource-bound" },
    // recover's 64 B of stack is a genuine local array, not a spill: ptxas -v reports
    // "64 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads". Its grid is at
    // most (1024+63)/64 = 17 blocks over 66 SMs, so occupancy is not a lever here and
    // minBlocks is only the hardware cap; the load-bearing assertion is the stack size.
    { "recover",               false, {0,0,0,0,0,0}, "7recoverE",          64,  20,     0, 64, 24,
      "grid is <= 17 blocks; occupancy is not a lever here, the stack size is" },
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
    static std::string p = MXBM_CUOBJDUMP;
    if (!p.empty() && ::access(p.c_str(), X_OK) == 0) return p.c_str();
    return "cuobjdump";
}

bool run_cuobjdump(std::string& out) {
    const std::string cmd = std::string("'") + cuobjdump_path() + "' -res-usage '" +
                            MXBM_CUDA_LIB + "' 2>&1";
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
    for (Measured& k : found) {
        for (int i = 0; i < kNumKernels; ++i) {
            const Kernel& c = kContract[i];
            bool hit;
            if (c.templated)
                hit = k.targs.size() >= 12 &&
                      k.targs[0] == c.id[0] && k.targs[1] == c.id[1] &&
                      k.targs[2] == c.id[2] && k.targs[3] == c.id[3] &&
                      k.targs[9] == c.id[4] && k.targs[10] == c.id[5];
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

// Add `delta` to the REG: value of one named function in a cuobjdump listing. Used only
// by the self-check.
bool bump_reg(std::string& s, const std::string& mangled, int delta) {
    const size_t f = s.find(" Function " + mangled + ":");
    if (f == std::string::npos) return false;
    const size_t r = s.find("REG:", f);
    if (r == std::string::npos) return false;
    const size_t v = r + 4;
    size_t e = v;
    while (e < s.size() && s[e] >= '0' && s[e] <= '9') ++e;
    if (e == v) return false;
    const int now = std::atoi(s.c_str() + v);
    s.replace(v, e - v, std::to_string(now + delta));
    return true;
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
