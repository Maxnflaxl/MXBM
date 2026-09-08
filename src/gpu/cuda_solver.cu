// CUDA backend implementation. The kernels live in kernels/cuda/ and are shared with the
// standalone bench (cuda/pipeline.cu), so there is one copy of each.
#include "gpu/cuda_solver.h"
#include "gpu/rowbucket_geom.h"
#include "gpu/budget.h"
#include "gpu/nvml.h"
#include "pipeline_kernels.cuh"
#include "beamhash/bh3_blake2b.h"
#include "beamhash/bh3_verify.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace mxbm { namespace gpu {
using namespace mxbm::cuda;

namespace {
template<class T> T* dalloc(size_t n) { void* p=nullptr; return cudaMalloc(&p, n*sizeof(T))==cudaSuccess ? (T*)p : nullptr; }

constexpr uint32_t kElems    = 1u << 25;
constexpr uint32_t kCapacity = kElems + kElems/32;      // 34,603,008
static_assert(kCapacity == kRbCapacity,
              "main.cpp's restart notice asks geometry questions with kRbCapacity; "
              "the two must be the same number");
// Every record that bit-packs a gi gives it 26 bits (r3_p1, and the w0-checkpoint pair
// record). A round emits at most one child per output slot, so the capacity bounds it.
static_assert(kCapacity < (1u << 26),
              "the packed records give gi 26 bits; a larger capacity silently truncates "
              "the back-reference a solution is recovered through");
constexpr uint32_t kSurvCap = 1024;
// set 0 carries the round-2 and round-4 outputs, set 1 the round-1 and round-3 outputs.
// Round 2's record is 9 u64: an 8-u64 record at a 16 B-aligned stride, plus a 9th-word
// plane laid out after all the records (fused_round derives its offset from the same
// geometry). Set 0 is therefore sized 8 + 1 rather than the 10 it took when the record
// was padded to keep the 128-bit accesses aligned -- see docs/performance.md.
constexpr uint32_t kSetStride[2] = { 9u, 8u };
constexpr uint32_t kR2RecStride  = 8u;   // round 2's record; the 9th word is the plane

constexpr uint32_t kPairStride = 2u;     // r1 -> r2 pair record, 16 B

// Each round's template arguments, named once. They were written out separately at the
// launch and at the carveout call, and had drifted: the carveout was applied to LM_RD2
// with OUTSTR 10 and LM_EMIT with INSTR 10, instantiations that no longer run, so the
// kernels that DO run never received the preference.
#define MXBM_R1_ARGS 7,7,1,LM_SEED, 424u,2u,1u,2u,2u, 1u,kPairStride,MXBM_R1_FCAP
#define MXBM_R2_ARGS 7,7,2,LM_RD2,  400u,4u,2u,4u,4u, kPairStride,kR2RecStride,kFCap
#define MXBM_R3_ARGS 7,6,4,LM_EMIT, 376u,6u,4u,2u,8u, kR2RecStride,8u,kFCap
// Round 4's output stride: 8 B (t5_rec) on a shipping build, 16 B where MXBM_R4_ROWS
// keeps gi and lead for the reference recovery arm.
#define MXBM_R4_ARGS 6,1,2,LM_USE,  288u,9u,2u,0u,0u, 8u,(MXBM_R4_ROWS?2u:1u),kFCap
// The quad-record pair. Only rounds 2 and 3 differ, and only in their strides and round
// 3's mode: the record carries the same INFORMATION either way, so every L value, tree
// width and cap is identical. Both pairs are instantiated and the choice is made at
// RUNTIME from the geometry, because which one a card wants depends on its VRAM.
constexpr uint32_t kQuadStride = 3u;
#define MXBM_R2Q_ARGS 7,7,2,LM_RD2, 400u,4u,2u,4u,4u, kPairStride,kQuadStride,kFCap
#define MXBM_R3Q_ARGS 7,6,4,LM_RD3, 376u,6u,4u,2u,8u, kQuadStride,8u,kFCap
// The OCTO pair. Round 3 emits its eight leaves in 4 u64 instead of the 64 B record, and
// round 4 rebuilds the work words, the lead and the leftContrib from them (LM_RD4). Only
// round 3's OUTSTR and round 4's INSTR and mode change; every L value, tree width and cap
// is the same, because the record carries the same INFORMATION either way. Offered only
// on quad rungs, so round 3's octo form pairs with the quad input and nothing else.
constexpr uint32_t kOctoStride = 4u;
constexpr uint32_t kQuad16Stride = 2u;   // the quad record with its gi retired
#define MXBM_R2QO_ARGS 7,7,2,LM_RD2, 400u,4u,2u,4u,4u, kPairStride,kQuad16Stride,kFCap
#define MXBM_R3QO_ARGS 7,6,4,LM_RD3, 376u,6u,4u,2u,8u, kQuad16Stride,kOctoStride,kFCap
#define MXBM_R4O_ARGS  6,1,2,LM_RD4, 288u,9u,2u,0u,0u, kOctoStride,2u,kFCap
// The same rounds staged at 280 for cards with 64 KB of shared memory per SM (Turing):
// under 21,845 B a block, rounds 1, 2 and 4 hold three resident blocks there instead of
// two, and ptxas fits their registers to match. Round 3 carries 80 B per staged element
// and reaches the line only with its word 6 in a u16 plane (N6) and a cap of 272; the
// dense-cap variant is 21,344 B there. Every card with a larger budget keeps kFCap,
// where the smaller caps buy no block and cost time. Chosen at runtime from the
// device's budget (Impl::smallShared).
constexpr uint32_t kFCap64K  = 280u;
constexpr uint32_t kFCap64K3 = 272u;
#define MXBM_R2_ARGS_S  7,7,2,LM_RD2, 400u,4u,2u,4u,4u, kPairStride,kR2RecStride,kFCap64K
#define MXBM_R4_ARGS_S  6,1,2,LM_USE, 288u,9u,2u,0u,0u, 8u,(MXBM_R4_ROWS?2u:1u),kFCap64K
#define MXBM_R2Q_ARGS_S 7,7,2,LM_RD2, 400u,4u,2u,4u,4u, kPairStride,kQuadStride,kFCap64K
#define MXBM_R2QO_ARGS_S 7,7,2,LM_RD2, 400u,4u,2u,4u,4u, kPairStride,kQuad16Stride,kFCap64K
#define MXBM_R4O_ARGS_S  6,1,2,LM_RD4, 288u,9u,2u,0u,0u, kOctoStride,2u,kFCap64K
#define MXBM_R3_ARGS_S   7,6,4,LM_EMIT, 376u,6u,4u,2u,8u, kR2RecStride,8u,kFCap64K3
#define MXBM_R3Q_ARGS_S  7,6,4,LM_RD3, 376u,6u,4u,2u,8u, kQuadStride,8u,kFCap64K3
#define MXBM_R3QO_ARGS_S 7,6,4,LM_RD3, 376u,6u,4u,2u,8u, kQuad16Stride,kOctoStride,kFCap64K3
// Left alone the rebuild takes 128 registers, exactly 2 blocks/SM. Asking for a third
// costs 88-96 B of spill and pays -2 %; 0 leaves the choice to ptxas.
#ifndef MXBM_MB_OCTO
#define MXBM_MB_OCTO 3
#endif
static_assert(kSetStride[0] == fb_set_stride(0) && kSetStride[1] == fb_set_stride(1),
              "CUDA record widths must match the shared footprint arithmetic");
// Under the implicit-bits pack the 9th word has no writer, so set 0 is the record and
// nothing else. The allocation and the ladder both take this from fb_set_stride.
static_assert(fb_set_stride(0, false, true) == kR2RecStride,
              "implicit-bits set 0 is the record alone -- the plane has no writer");
static_assert(kArenaCap == kRbArenaSlots,
              "the pool the kernels index must be the pool the footprint reserves");
static_assert(kQuadStride == fb_round_stride(2, true) &&
              fb_set_stride(0, true) == kQuadStride && fb_set_stride(1, true) == 8u,
              "quad record widths must match the shared footprint arithmetic");
static_assert(kOctoStride == fb_round_stride(3, true, false, true) &&
              fb_set_stride(1, true, false, true) == kOctoStride,
              "octo record widths must match the shared footprint arithmetic");
static_assert(kQuad16Stride == fb_round_stride(2, true, false, true) &&
              fb_set_stride(0, true, false, true) == kQuad16Stride,
              "the gi-less quad record must match the shared footprint arithmetic");

// Was hardcoded (16,1), so a card that could not host the finest rung was refused outright --
// even though the same kernels run unchanged at 6.50 GiB. bb and sm are kernel
// ARGUMENTS, and everything compile-time is invariant along the bb + sm = 17 line: the
// grid is 2^17 blocks, the staged group is capacity / 2^17 = 264, and the 128-entry
// chain table is a perfect hash for the 24 - bb - sm = 7 bits left varying.
// max_alloc = 0: CUDA has no per-allocation limit (see rowbucket_geom.h).
// power_limit_w = 0 (the availability/enumeration callers): viability must not depend
// on the cap -- (17,0) is a preference among geometries that fit, never a requirement.
// The implicit-bits record is decided WITH the geometry, not after it: it narrows set 0
// from 9 u64 to 8, so the ladder's arithmetic and the allocator have to answer the same
// question. MXBM_NO_IMPB turns it off on both sides at once.
bool impb_allowed() {
    static const bool off = std::getenv("MXBM_NO_IMPB") != nullptr;
    return !off;
}

// What availability holds back from TOTAL VRAM. The enumeration callers cannot ask
// cudaMemGetInfo without creating a context on every device, so they size against total
// and subtract this. Measured on the reference card: `totalGlobalMem` is 15.598 GiB
// against nvidia-smi's 15.99, so the driver's own carve-out is ALREADY excluded from the
// figure this subtracts from -- and the one cost that is not is this process's context,
// at 0.214 GiB. The rest is margin for a card that is not idle when we ask. The flat
// 1 GiB this replaced was ~0.75 GiB of margin, which is the difference between an offered
// device and a refused one on the small cards. Being optimistic here is safe: the
// constructor re-asks against FREE memory, steps down the ladder on a real allocation
// failure, and main.cpp reports a throw per-GPU and falls back to the OpenCL path.
constexpr uint64_t kAvailSlackBytes = 640ull << 20;

RbGeometry pick_geometry(uint64_t usable_mem, unsigned power_limit_w = 0,
                         uint64_t slack = kAvailSlackBytes) {
    // allow_quad: costs +14 % time and buys no watts at any cap (measured), so the
    // ladder only reaches for it when a card cannot host a packed rung -- which is
    // exactly what takes the CUDA path below 6.5 GiB. (Both backends carry the quad
    // kernels since 2026-07-28; each caller states its capability itself.)
    RbGeometry g = rb_geometry_for(kCapacity, /*max_alloc=*/0, usable_mem,
                                   /*allow_quad=*/true, power_limit_w,
                                   /*allow_split=*/false, slack,
                                   /*allow_impb=*/impb_allowed(),
                                   /*allow_arena=*/true, /*allow_octo=*/true,
                                   /*allow_replay=*/MXBM_R4_ROWS == 0);
    if (const char* e = std::getenv("MXBM_BB")) { g.bb = (uint32_t)atoi(e); g.sm = 17u - g.bb;
                                                 g.viable = true; }
    if (const char* e = std::getenv("MXBM_SM")) { g.sm = (uint32_t)atoi(e); g.viable = true; }
    if (const char* e = std::getenv("MXBM_QUAD")) { g.quad = atoi(e) != 0; g.viable = true; }
    if (const char* e = std::getenv("MXBM_ARENA")) { g.arena = atoi(e) != 0; g.viable = true; }
    if (const char* e = std::getenv("MXBM_OCTO")) { g.octo = atoi(e) != 0; g.viable = true; }
    // The octo record is offered on quad dense-cap rungs and nowhere else, so a forced
    // octo has to bring both: round 3's dispatch emits the octo form from the QUAD input,
    // and rounds 1-3 go through ROUND_NR, which instantiates the dense-cap kernels only.
    // Without this a forced MXBM_OCTO writes 16 B records into a set sized for 64 and
    // hands the chain walk null pointers.
    if (g.octo) { g.quad = true; g.arena = true; }
    return g;
}
} // namespace

struct CudaSolver::Impl {
    uint32_t bb = 16, sm = 1, nb = 1u << 16;
    bool quad = false;                  // 24 B round-2 record; see rowbucket_geom.h
    bool impb = false;                  // 64 B round-2 record with address-implied key
                                        // bits packed out and the side plane deleted.
                                        // Set with the geometry (rb_impb_ok), because it
                                        // is also what set 0 is sized against.
                                        // Figures: docs/performance-research.md.
    // Dense per-bucket caps backed by one overflow pool per record set (rowbucket_geom's
    // arena rungs). The pool lives behind the bucket records in the same buffer; ahead/
    // anext thread it onto per-bucket chains between a producer round and its consumer.
    // Null pointers are what make the plain kernels' path the one that runs.
    bool arena = false;
    // Round 3's output as its eight leaves, round 4 rebuilding from them. See
    // rowbucket_geom.h; only ever set together with quad and arena.
    bool octo = false;
    uint32_t refRows = 4;               // capacity-sized back-ref rows actually allocated
    uint32_t *ahead[2] = {nullptr,nullptr}, *anext[2] = {nullptr,nullptr};
    uint32_t *atag[2]  = {nullptr,nullptr}, *actr[2]  = {nullptr,nullptr};
    // Run-long spill count. A pool that never fills means the dense cap was never
    // exceeded, so a zero here and a zero in drops[] mean opposite things: this is the
    // positive control that says the overflow path ran at all (MXBM_DROP_STATS).
    uint32_t *spillTot = nullptr;
    uint32_t cap = 0; size_t nslots = 0;
    // MXBM_SMEM_PAD=<bytes>: dynamic shared memory added to every round launch, used by
    // nothing. It lowers resident blocks per SM and changes nothing else, so it prices
    // occupancy on this card at another card's shared-memory budget.
    size_t smemPad = 0;
    // Rounds 2 and 4 launch their kFCap64K instantiations: the device has 64 KB of
    // shared memory per SM. MXBM_FCAP_SMALL=0|1 overrides, for the gate on other cards.
    bool smallShared = false;
    // Stage timing (Solver::stage_timing): one event per stage boundary on the launch
    // stream, read back after the solve's own synchronising copies.
    static constexpr int kStages = 7;
    bool timing = false, evReady = false;
    cudaEvent_t ev[kStages + 1] = {};
    std::vector<float> stageMs[kStages];
    void mark(int i) { if (timing) cudaEventRecord(ev[i], 0); }
    uint64_t *elem[2] = {nullptr,nullptr}, *dpp = nullptr;
    uint32_t *counts[2] = {nullptr,nullptr}, *gictr=nullptr, *drops=nullptr;
#if MXBM_GCEN
    uint32_t* kcen = nullptr;          // the key census, see MXBM_GCEN
#endif
    uint32_t *left=nullptr, *right=nullptr, *survSlots=nullptr, *survCount=nullptr, *dleaves=nullptr;
    // Recovery by replay: the survivors' round-4 parents by slot, and the round-3
    // ancestors replay_r4 derives from them.
    uint32_t *survL4=nullptr, *l3Slots=nullptr, *l4Lead=nullptr, *l2Slots=nullptr;
    // Round 4's own output plane. Without it round 4 would land on set 0 and destroy
    // round 2's output, which is the record recovery reads its leaves out of once rows
    // 1-3 are gone. 8 B a slot against the 24 B a row pair costs.
    uint64_t *elem4 = nullptr;
    uint32_t *counts4 = nullptr;
    uint32_t *ahead4=nullptr, *anext4=nullptr, *atag4=nullptr, *actr4=nullptr;
    std::atomic<bool> abort_{false};
    DeviceInfo info;
    int device_index = 0;

    // Speculative entry co-scheduling. The NEXT nonce's entry pass rides inside round
    // 4's launch as separate interleaved blocks (COBLOCKS in fused_round.cuh), where it
    // measures ~0.4 ms cheaper than standing alone -- round 4 is 82 % of DRAM peak and
    // 29 % SM, and displacing a third of its blocks is measured free. The engine
    // iterates nonces at a fixed stride within a job, so the solver LEARNS the delta
    // between consecutive calls and seeds (nonce + delta); a job change mis-speculates
    // one solve, which then simply runs its own entry pass. Entry output moves to a
    // dedicated dense buffer (+0.39 GiB) so it survives round 2 overwriting elem[0];
    // if that buffer does not fit, spec* stay null and everything runs as before.
    uint64_t *specElem = nullptr, *specPp = nullptr;
    uint32_t *specCounts = nullptr;
    bool specOn = false;                // buffers allocated and MXBM_NO_SPEC unset
    bool matchFirst = false;            // skip the childless rebuilds; floor-only, see below
    bool specValid = false;             // specElem holds entry(specNonce) of specInput
    bool haveLast = false;
    bool haveDelta = false;             // persistent: the stride survives job changes,
                                        // so a new job costs one miss, not two
    uint64_t specNonceLE = 0, lastNonceLE = 0, learnedDelta = 0;
    uint8_t specInput[32] = {0}, lastInput[32] = {0};

    // The geometry-dependent half of the allocation: the two record sets and their
    // bucket counters. Split out so a step-down retries only what changes size.
    bool alloc_geometry(uint32_t bb_, uint32_t sm_, bool quad_, bool arena_, bool octo_) {
        for (auto*& q : elem)   { cudaFree(q); q = nullptr; }
        for (auto*& q : counts) { cudaFree(q); q = nullptr; }
        for (auto* a : {ahead, anext, atag, actr})
            for (int k = 0; k < 2; ++k) { cudaFree(a[k]); a[k] = nullptr; }
        cudaFree(left); left = nullptr; cudaFree(right); right = nullptr;
        cudaFree(elem4); elem4 = nullptr; cudaFree(counts4); counts4 = nullptr;
        for (auto** q : {&ahead4, &anext4, &atag4, &actr4}) { cudaFree(*q); *q = nullptr; }
        bb = bb_; sm = sm_; nb = 1u << bb_; quad = quad_; arena = arena_; octo = octo_;
        impb = impb_allowed() && rb_impb_ok(bb, sm, quad);
        // Back-refs: the surviving rows are gi-indexed and need full capacity. Rows 4 and
        // 5 are not written at all off the octo rungs -- replay_r4 reconstructs round 4's
        // pairing from the bucket the record carries -- so three rows are allocated, and
        // an octo rung reads only rows 4 and 5 (recover() explains why) so it allocates
        // one. Sized here rather than beside the geometry-independent buffers because a
        // step-down can cross onto an octo rung, and holding rows it will never read is
        // 0.77 GiB withheld exactly where memory is tightest.
        refRows = octo ? 1u : (MXBM_R4_ROWS ? 4u : (impb ? 0u : 3u));
        if (refRows) {
            left  = dalloc<uint32_t>((size_t)refRows*kCapacity + kSurvCap);
            right = dalloc<uint32_t>((size_t)refRows*kCapacity + kSurvCap);
            if (!left || !right) return false;
        }
        cap    = fb_cap_for(kCapacity / nb, arena ? kRbDenseSigma : kRbCapSigma);
        nslots = (size_t)nb * cap;
        // Set 0 is 9 u64 for the packed record, 8 once the implicit-bits pack deletes the
        // plane, and 3 for the quad one -- taken from the shared helper rather than a
        // local constant so the allocation cannot disagree with what rb_geometry_for
        // sized the ladder against. The pool follows the bucket records in the same
        // buffer, so pool slot a is record index nslots + a and no store has to branch.
        const size_t slots = nslots + (arena ? kArenaCap : 0u);
        elem[0] = dalloc<uint64_t>(slots*fb_set_stride(0, quad, impb, octo));
        elem[1] = dalloc<uint64_t>(slots*fb_set_stride(1, quad, impb, octo));
        counts[0] = dalloc<uint32_t>(nb); counts[1] = dalloc<uint32_t>(nb);
        if (!elem[0] || !elem[1] || !counts[0] || !counts[1]) return false;
        for (int k = 0; arena && k < 2; ++k) {
            ahead[k] = dalloc<uint32_t>(nb);          anext[k] = dalloc<uint32_t>(kArenaCap);
            atag[k]  = dalloc<uint32_t>(kArenaCap);   actr[k]  = dalloc<uint32_t>(1);
            if (!ahead[k] || !anext[k] || !atag[k] || !actr[k]) return false;
        }
        // On the implicit-bits rungs; under MXBM_R4_ROWS as well, so the rows and the
        // replays can both run and be compared.
        if (impb) {
            elem4   = dalloc<uint64_t>(slots * (MXBM_R4_ROWS ? 2u : 1u));
            counts4 = dalloc<uint32_t>(nb);
            if (!elem4 || !counts4) return false;
            if (arena) {
                ahead4 = dalloc<uint32_t>(nb);        anext4 = dalloc<uint32_t>(kArenaCap);
                atag4  = dalloc<uint32_t>(kArenaCap); actr4  = dalloc<uint32_t>(1);
                if (!ahead4 || !anext4 || !atag4 || !actr4) return false;
            }
        }
        return true;
    }

    // The frees belong here, not in ~CudaSolver: that constructor throws when a device
    // is too small, and a throwing constructor never runs its own destructor, so every
    // cudaMalloc before the throw leaked for the life of the process. A caller catching
    // the throw and retrying smaller therefore got LESS memory on each attempt. ~Impl
    // runs either way, p_ being a fully-constructed member.
    ~Impl() {
        if (evReady) for (auto e : ev) cudaEventDestroy(e);
        for (auto* q : {elem[0], elem[1], dpp, specElem, specPp, elem4}) cudaFree(q);
#if MXBM_GCEN
        cudaFree(kcen);
#endif
        for (auto* q : {counts[0], counts[1], gictr, drops, spillTot,
                        left, right, survSlots, survCount, dleaves, specCounts,
                        survL4, l3Slots, l4Lead, l2Slots,
                        counts4, ahead4, anext4, atag4, actr4}) cudaFree(q);
        for (auto* a : {ahead, anext, atag, actr})
            for (int k = 0; k < 2; ++k) cudaFree(a[k]);
    }
};

// Tracks the release fatbin's floor: compute_75 is its oldest image, and a card below
// it finds none at launch. Nothing in the kernels needs Ampere (no cp.async, no async
// barriers, no clusters), but Turing's 64 KB shared/SM hosts two 320-thread blocks per
// SM where Ada hosts three, so the ISA is not what sets the pace there.
constexpr int kMinComputeCapability = 75;

bool cc_supported(const cudaDeviceProp& p) {
    return p.major * 10 + p.minor >= kMinComputeCapability;
}

bool CudaSolver::available(int index) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return false;
    if (index < 0 || index >= n) return false;
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, index) != cudaSuccess) return false;
    if (!cc_supported(p)) return false;
    // Refuse only what no geometry on the ladder can host. The seed layer itself is
    // never reduced -- a reduced layer mines nothing (budget.h, budget_can_find_solutions)
    // -- so the ladder trades speed for footprint and nothing else.
    return pick_geometry(p.totalGlobalMem).viable;
}

CudaSolver::DeviceInfo CudaSolver::device_info(int index) {
    DeviceInfo d; cudaDeviceProp p{};
    d.index = index;
    if (cudaGetDeviceProperties(&p, index) == cudaSuccess) {
        d.name = p.name; d.global_mem = p.totalGlobalMem; d.compute_units = p.multiProcessorCount;
        d.viable = cc_supported(p) && pick_geometry(p.totalGlobalMem).viable;
        char buf[32];
        std::snprintf(buf, sizeof buf, "%x:%x", p.pciBusID, p.pciDeviceID);
        d.pci = buf;
    }
    return d;
}

std::vector<CudaSolver::DeviceInfo> CudaSolver::enumerate() {
    std::vector<DeviceInfo> out;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return out;
    for (int i = 0; i < n; ++i) out.push_back(device_info(i));
    // PCI order, not CUDA's. CUDA defaults to CUDA_DEVICE_ORDER=FASTEST_FIRST, so
    // its index 0 is not necessarily bus 1 -- and --pl's per-GPU list is applied
    // through NVML, which orders by bus. Sorting here is what stops "GPU 1" from
    // meaning two different cards in two different flags on the same rig.
    //
    // NUMERICALLY, on the parsed address: the string is "%x:%x", so a string compare
    // puts bus 0x41 before bus 0x9 -- on any board with buses either side of 0x10 that
    // disagrees with NVML's own bus order, which is what lands --pl on the wrong card.
    auto pci_key = [](const DeviceInfo& d) {
        const size_t c = d.pci.find(':');
        if (c == std::string::npos) return 0ull;
        const unsigned long bus = std::strtoul(d.pci.substr(0, c).c_str(), nullptr, 16);
        const unsigned long dev = std::strtoul(d.pci.substr(c + 1).c_str(), nullptr, 16);
        return ((unsigned long long)bus << 16) | (dev & 0xffffull);
    };
    std::sort(out.begin(), out.end(), [&](const DeviceInfo& a, const DeviceInfo& b) {
        return pci_key(a) < pci_key(b);
    });
    return out;
}

const CudaSolver::DeviceInfo& CudaSolver::device() const { return p_->info; }
unsigned CudaSolver::bucket_bits() const { return p_->bb; }
void CudaSolver::request_abort() { p_->abort_.store(true, std::memory_order_relaxed); }
// The rung that ALLOCATED, read off the pipeline rather than recomputed: the ladder
// steps down when the allocator refuses, and it is that landing point a report needs.
bool CudaSolver::stage_timing(bool on) {
    Impl& I = *p_;
    cudaSetDevice(I.device_index);
    if (on && !I.evReady) {
        for (auto& e : I.ev) if (cudaEventCreate(&e) != cudaSuccess) return false;
        I.evReady = true;
    }
    for (auto& v : I.stageMs) v.clear();
    I.timing = on && I.evReady;
    return I.timing;
}

std::vector<miner::Solver::StageTime> CudaSolver::stage_times() const {
    static const char* const kNames[Impl::kStages] =
        {"entry", "round 1", "round 2", "round 3", "round 4", "terminal", "recovery"};
    std::vector<miner::Solver::StageTime> out;
    for (int i = 0; i < Impl::kStages; ++i) {
        std::vector<float> v = p_->stageMs[i];
        if (v.empty()) return {};
        std::sort(v.begin(), v.end());
        out.push_back({kNames[i], (double)v[v.size() / 2]});
    }
    return out;
}

std::string CudaSolver::geometry() const {
    return rb_describe(p_->bb, p_->sm, p_->quad, p_->impb, p_->arena, p_->octo,
                       /*replay=*/true) + (p_->smallShared ? ", staging 280/272" : "");
}

CudaSolver::CudaSolver(int index, unsigned power_limit_w, unsigned mem_clock_mhz)
    : p_(new Impl) {
    // Bind THIS thread to the chosen device before any allocation. CUDA's current
    // device is per-thread, which is why solve() sets it again below: the engine
    // runs solve() on a worker thread that never saw this constructor, and would
    // otherwise use device 0's context against this device's pointers.
    if (cudaSetDevice(index) != cudaSuccess)
        throw std::runtime_error("cudaSetDevice(" + std::to_string(index) + ") failed");
    p_->device_index = index;
    p_->info = device_info(index);
    {
        cudaDeviceProp dp{};
        if (cudaGetDeviceProperties(&dp, index) == cudaSuccess)
            p_->smallShared = dp.sharedMemPerMultiprocessor <= 64u * 1024u;
        if (const char* e = std::getenv("MXBM_FCAP_SMALL")) p_->smallShared = atoi(e) != 0;
    }
    // Geometry-independent first, so the step-down below is not retrying these.
    p_->gictr = dalloc<uint32_t>(1); p_->drops = dalloc<uint32_t>(4);
#if MXBM_GCEN
    // 2 bits per 24-bit key. Geometry-independent: the key is 24 bits at every rung.
    p_->kcen = dalloc<uint32_t>(1u << 20);
    cudaMemcpyToSymbol(g_kcen, &p_->kcen, sizeof(uint32_t*));
#endif
    p_->spillTot = dalloc<uint32_t>(1);
    if (p_->spillTot) cudaMemset(p_->spillTot, 0, 4);
    p_->survSlots = dalloc<uint32_t>(kSurvCap); p_->survCount = dalloc<uint32_t>(1);
    p_->dleaves = dalloc<uint32_t>((size_t)kSurvCap*32);
    p_->survL4  = dalloc<uint32_t>((size_t)kSurvCap*2);
    p_->l3Slots = dalloc<uint32_t>((size_t)kSurvCap*4);
    p_->l4Lead  = dalloc<uint32_t>((size_t)kSurvCap*2);
    p_->l2Slots = dalloc<uint32_t>((size_t)kSurvCap*8);
    p_->dpp = dalloc<uint64_t>(4);
    if (!p_->dpp || !p_->survSlots)
        throw std::runtime_error("CUDA allocation failed (device out of memory)");

    // Then walk the ladder for real, against what the driver says is FREE less the
    // reserve rather than against total VRAM: a desktop compositor sitting on a
    // gigabyte is invisible in totalGlobalMem. Whether a footprint fits *today* is
    // still only knowable by asking the allocator, which the retry below does. An
    // explicit MXBM_BB / MXBM_SM is the user's choice and is not stepped down.
    size_t cuda_free = 0, cuda_total = 0;
    const bool have_free = cudaMemGetInfo(&cuda_free, &cuda_total) == cudaSuccess;
    const uint64_t usable = usable_vram(have_free ? (uint64_t)cuda_free : 0,
                                        p_->info.global_mem,
                                        reserve_bytes(nvml_display_active((unsigned)index)));
    // slack 0: the reserve above is the whole of what is held back.
    const RbGeometry g = pick_geometry(usable, power_limit_w, /*slack=*/0);
    if (!g.viable)
        throw std::runtime_error("CUDA: no row-bucket geometry fits this device");
    // MXBM_PERFECT_TAB drops the key comparison in the chain walk, which is only sound
    // while the table can separate every key that varies inside a group: 24 - bb - sm
    // bits. Every geometry on the bb + sm = 17 line leaves exactly 7 and kTabSize is 128,
    // so this holds for all of them -- but MXBM_BB/MXBM_SM can be set off the line by
    // hand, and silently combining unequal keys would corrupt results rather than fail.
    if (MXBM_PERFECT_TAB && (24u - g.bb - g.sm) > 7u)
        throw std::runtime_error("CUDA: geometry leaves more key bits than the chain "
                                 "table can separate (bb + sm must be >= 17)");
    const bool forced = std::getenv("MXBM_BB") || std::getenv("MXBM_SM")
                     || std::getenv("MXBM_QUAD") || std::getenv("MXBM_ARENA")
                     || std::getenv("MXBM_OCTO");
    // bb == 17 without a force can only mean the power-limit preference fired; say so,
    // because a user comparing against a stock run will see different memory and speed.
    if (!forced && g.bb == 17u)
        std::fprintf(stderr, "CUDA: board power limit %u W is below %u W: selecting the "
                     "low-power (17,0) geometry\n", power_limit_w, kRbLowPowerW);
    bool ok = p_->alloc_geometry(g.bb, g.sm, g.quad, g.arena, g.octo);
    // Keep stepping down the SAME ladder rb_geometry_for walked, rather than
    // decrementing bb: the rungs interleave the record formats and the dense-cap rows,
    // so a bb-only retry would stop at the bottom of the packed half and refuse a card
    // the quad or arena rungs would have hosted.
    if (!ok && !forced) {
        int n = 0;
        const RbRung* rungs = rb_rungs(n);
        auto describe = [](const RbRung& r) {
            return r.octo ? " (quad record, dense caps, octo)"
                 : r.quad ? (r.arena ? " (quad record, dense caps)" : " (quad record)")
                          : (r.arena ? " (dense caps)" : "");
        };
        int i = 0;
        while (i < n && !(rungs[i].bb == g.bb && rungs[i].quad == g.quad
                          && rungs[i].arena == g.arena && rungs[i].octo == g.octo)) ++i;
        // A cap-selected (17,0) is not ON the ladder, so the scan above runs off the
        // end; when the allocator refuses it, retry from the top rung, not from i == n.
        if (i == n) i = -1;
        for (++i; !ok && i < n; ++i) {
            std::fprintf(stderr, "CUDA: %u buckets%s did not fit, retrying at %u%s\n",
                         1u << g.bb, describe({g.bb, g.sm, g.quad, g.arena, g.octo}),
                         1u << rungs[i].bb, describe(rungs[i]));
            ok = p_->alloc_geometry(rungs[i].bb, rungs[i].sm, rungs[i].quad,
                                    rungs[i].arena, rungs[i].octo);
        }
    }
    if (!ok) throw std::runtime_error("CUDA allocation failed (device out of memory)");
    // Ada splits L1/shared by a carveout that defaults to favouring L1; ask for shared.
    #define CARVE(K) cudaFuncSetAttribute(K, cudaFuncAttributePreferredSharedMemoryCarveout, \
                                          cudaSharedmemCarveoutMaxShared)
    CARVE((fused_round<MXBM_R1_ARGS>)); CARVE((fused_round<MXBM_R2_ARGS>));
    CARVE((fused_round<MXBM_R3_ARGS>)); CARVE((fused_round<MXBM_R4_ARGS>));
    // Every record format and rebuild variant that can launch is instantiated; the
    // carveout has to reach the pair that actually runs, and applying it to a
    // kernel that never launches is what the MXBM_Rn_ARGS naming exists to prevent
    // (see the drift bug in cad8fde). The packed pairs and the match-first
    // variants launch in the shipping configs -- (16,1) packed at stock, match-
    // first x packed-17 under the low-power gate -- and each needs its own line.
    CARVE((fused_round<MXBM_R2Q_ARGS>)); CARVE((fused_round<MXBM_R3Q_ARGS>));
    CARVE((fused_round<MXBM_R2_ARGS, false, false, false, false, 16u>));
    CARVE((fused_round<MXBM_R3_ARGS, false, false, false, false, 16u>));
    CARVE((fused_round<MXBM_R2_ARGS, false, false, false, true, 16u>));
    CARVE((fused_round<MXBM_R3_ARGS, false, false, false, true, 16u>));
    CARVE((fused_round<MXBM_R2_ARGS, false, false, false, true, 17u>));
    CARVE((fused_round<MXBM_R3_ARGS, false, false, false, true, 17u>));
    CARVE((fused_round<MXBM_R2_ARGS, false, false, false, false, 17u>));
    CARVE((fused_round<MXBM_R3_ARGS, false, false, false, false, 17u>));
    CARVE((fused_round<MXBM_R1_ARGS, false, false, false, true, 0u>));
    // Round 1 at the rung's IMPB -- the instantiations that launch at stock. IMPB keys
    // both the w0-checkpoint format (with MXBM_PAIR_W0) and the reference-row elision,
    // so these exist whether or not the checkpoint does.
    CARVE((fused_round<MXBM_R1_ARGS, false, false, false, false, 16u>));
    CARVE((fused_round<MXBM_R1_ARGS, false, false, false, true,  16u>));
    CARVE((fused_round<MXBM_R1_ARGS, false, false, false, false, 17u>));
    CARVE((fused_round<MXBM_R1_ARGS, false, false, false, true,  17u>));
    CARVE((fused_round<MXBM_R4_ARGS, false, false, false, true>));
    // The round-4 variant that hosts the next nonce's entry as co-blocks.
    CARVE((fused_round<MXBM_R4_ARGS, false, false, true>));
    CARVE((terminal_round<false>));
    if (p_->smallShared) {
        CARVE((fused_round<MXBM_R4_ARGS_S, false, false, false, true>));
        CARVE((fused_round<MXBM_R4_ARGS_S, false, false, true>));
        CARVE((fused_round<MXBM_R2_ARGS_S, false, false, false, false, 16u>));
        CARVE((fused_round<MXBM_R2_ARGS_S, false, false, false, true,  16u>));
        CARVE((fused_round<MXBM_R2_ARGS_S, false, false, false, false, 17u>));
        CARVE((fused_round<MXBM_R2_ARGS_S, false, false, false, true,  17u>));
        CARVE((fused_round<MXBM_R2_ARGS_S, false, false, false, false,  0u>));
        CARVE((fused_round<MXBM_R2_ARGS_S, false, false, false, true,   0u>));
        CARVE((fused_round<MXBM_R4_ARGS_S, false, false, false, false>));
        CARVE((fused_round<MXBM_R3_ARGS_S, false, false, false, false, 16u, false, true, true>));
        CARVE((fused_round<MXBM_R3_ARGS_S, false, false, false, true,  16u, false, true, true>));
        CARVE((fused_round<MXBM_R3_ARGS_S, false, false, false, false, 17u, false, true, true>));
        CARVE((fused_round<MXBM_R3_ARGS_S, false, false, false, true,  17u, false, true, true>));
        CARVE((fused_round<MXBM_R3Q_ARGS_S, false, false, false, false, 0u, false, true, true>));
        CARVE((fused_round<MXBM_R3Q_ARGS_S, false, false, false, true,  0u, false, true, true>));
    }
    // The arena rungs run every round again with dense caps and the overflow pool. Set
    // only when one was picked, since these kernels are otherwise never launched.
    if (p_->arena) {
        CARVE((fused_round<MXBM_R1_ARGS,  false, false, false, false, 0u, true>));
        CARVE((fused_round<MXBM_R1_ARGS,  false, false, false, true,  0u, true>));
        CARVE((fused_round<MXBM_R1_ARGS,  false, false, false, false, 16u, true>));
        CARVE((fused_round<MXBM_R1_ARGS,  false, false, false, true,  16u, true>));
        CARVE((fused_round<MXBM_R1_ARGS,  false, false, false, false, 17u, true>));
        CARVE((fused_round<MXBM_R1_ARGS,  false, false, false, true,  17u, true>));
        CARVE((fused_round<MXBM_R4_ARGS,  false, false, false, false, 0u, true>));
        CARVE((fused_round<MXBM_R4_ARGS,  false, false, false, true,  0u, true>));
        CARVE((fused_round<MXBM_R2Q_ARGS, false, false, false, false, 0u, true>));
        CARVE((fused_round<MXBM_R3Q_ARGS, false, false, false, false, 0u, true>));
        CARVE((fused_round<MXBM_R2Q_ARGS, false, false, false, true,  0u, true>));
        CARVE((fused_round<MXBM_R3Q_ARGS, false, false, false, true,  0u, true>));
        if (p_->smallShared) {
            CARVE((fused_round<MXBM_R4_ARGS_S,  false, false, false, false, 0u, true>));
            CARVE((fused_round<MXBM_R4_ARGS_S,  false, false, false, true,  0u, true>));
            CARVE((fused_round<MXBM_R2Q_ARGS_S, false, false, false, false, 0u, true>));
            CARVE((fused_round<MXBM_R2Q_ARGS_S, false, false, false, true,  0u, true>));
            CARVE((fused_round<MXBM_R2_ARGS_S,  false, false, false, false, 16u, true>));
            CARVE((fused_round<MXBM_R2_ARGS_S,  false, false, false, true,  16u, true>));
            CARVE((fused_round<MXBM_R2_ARGS_S,  false, false, false, false,  0u, true>));
            CARVE((fused_round<MXBM_R2_ARGS_S,  false, false, false, true,   0u, true>));
            CARVE((fused_round<MXBM_R3Q_ARGS_S, false, false, false, false, 0u, true, true, true>));
            CARVE((fused_round<MXBM_R3Q_ARGS_S, false, false, false, true,  0u, true, true, true>));
            CARVE((fused_round<MXBM_R3_ARGS_S,  false, false, false, false, 16u, true, true, true>));
            CARVE((fused_round<MXBM_R3_ARGS_S,  false, false, false, true,  16u, true, true, true>));
            CARVE((fused_round<MXBM_R3_ARGS_S,  false, false, false, false,  0u, true, true, true>));
            CARVE((fused_round<MXBM_R3_ARGS_S,  false, false, false, true,   0u, true, true, true>));
        }
    }
    if (p_->octo) {
        CARVE((fused_round<MXBM_R3QO_ARGS, false, false, false, false, 0u, true, false>));
        CARVE((fused_round<MXBM_R3QO_ARGS, false, false, false, true,  0u, true, false>));
        CARVE((fused_round<MXBM_R1_ARGS,  false, false, false, false, 0u, true, false>));
        CARVE((fused_round<MXBM_R1_ARGS,  false, false, false, true,  0u, true, false>));
        CARVE((fused_round<MXBM_R2QO_ARGS, false, false, false, false, 0u, true, false>));
        CARVE((fused_round<MXBM_R2QO_ARGS, false, false, false, true,  0u, true, false>));
        CARVE((fused_round<MXBM_R4O_ARGS,  false, false, false, false, 0u, true, true, false, MXBM_MB_OCTO>));
        CARVE((fused_round<MXBM_R4O_ARGS,  false, false, false, true,  0u, true, true, false, MXBM_MB_OCTO>));
        if (p_->smallShared) {
            CARVE((fused_round<MXBM_R2QO_ARGS_S, false, false, false, false, 0u, true, false>));
            CARVE((fused_round<MXBM_R2QO_ARGS_S, false, false, false, true,  0u, true, false>));
            CARVE((fused_round<MXBM_R4O_ARGS_S,  false, false, false, false, 0u, true, true, false, MXBM_MB_OCTO>));
            CARVE((fused_round<MXBM_R4O_ARGS_S,  false, false, false, true,  0u, true, true, false, MXBM_MB_OCTO>));
            CARVE((fused_round<MXBM_R3QO_ARGS_S, false, false, false, false, 0u, true, false, true>));
            CARVE((fused_round<MXBM_R3QO_ARGS_S, false, false, false, true,  0u, true, false, true>));
        }
        CARVE((fused_round<MXBM_R2_ARGS,  false, false, false, false, 16u, true>));
        CARVE((fused_round<MXBM_R3_ARGS,  false, false, false, false, 16u, true>));
        CARVE((fused_round<MXBM_R2_ARGS,  false, false, false, true,  16u, true>));
        CARVE((fused_round<MXBM_R3_ARGS,  false, false, false, true,  16u, true>));
        CARVE((fused_round<MXBM_R2_ARGS,  false, false, false, false,  0u, true>));
        CARVE((fused_round<MXBM_R3_ARGS,  false, false, false, false,  0u, true>));
        CARVE((fused_round<MXBM_R2_ARGS,  false, false, false, true,   0u, true>));
        CARVE((fused_round<MXBM_R3_ARGS,  false, false, false, true,   0u, true>));
        CARVE((terminal_round<true>)); CARVE((terminal_round<true, 2u>));
    }
    #undef CARVE

    // Speculative-entry buffers are best-effort: a card without the extra 0.39 GiB (or
    // a user setting MXBM_NO_SPEC=1) simply runs the entry pass standalone, as before.
    // Under a cap the co-blocks displace r4 work the card can no longer spare -- the
    // rounds turn issue-bound -- so the observed power limit gates it too, with the
    // observed memory clock choosing which crossover applies: a down-locked rung
    // un-starves the core (spec pays again above kSpecMinPowerRungW), while stock
    // memory keeps every round issue-bound up to kSpecMinPowerW. mem_clock_mhz > 600
    // distinguishes a held rung, which reports its lock even before kernels run, from
    // an unlocked card idling near 405.
    const bool lowRungMem = mem_clock_mhz > 600 && mem_clock_mhz <= kSpecRungMclkMHz;
    const unsigned specMinW = lowRungMem ? kSpecMinPowerRungW : kSpecMinPowerW;
    const bool specOff =
        specMinW != 0 && power_limit_w != 0 && power_limit_w < specMinW;
    if (specOff)
        std::fprintf(stderr, "CUDA: board power limit %u W is below %u W%s: speculative "
                     "entry off (it costs 1-2%% there)\n", power_limit_w, specMinW,
                     lowRungMem ? " on this memory rung" : "");
    // Match-first has its own band: the elements alone in their chain slot pay a
    // rebuild no walk reads, and skipping them costs the shared memory that holds r1's
    // fifth block. That trade loses at stock and wins once the rebuild bills at full
    // issue rate. Both variants are instantiated; MXBM_MATCH_FIRST=1 forces it on.
    // ...and on the octo rungs, for a third reason: what it skips for an element alone in
    // its chain slot is there the eight-leaf rebuild, which is most of the round. Measured
    // -6.5 % there, against a loss at stock.
    const bool mfLowPower = kMatchFirstMaxPowerW != 0 && power_limit_w != 0
                         && power_limit_w < kMatchFirstMaxPowerW;
    p_->matchFirst = mfLowPower || p_->octo || (MXBM_MATCH_FIRST != 0);
    // MXBM_MFIRST=0|1 forces it either way; without it the variant is only reachable by
    // presenting a board limit under kMatchFirstMaxPowerW.
    if (const char* e = std::getenv("MXBM_MFIRST")) p_->matchFirst = atoi(e) != 0;
    if (const char* e = std::getenv("MXBM_SMEM_PAD")) p_->smemPad = (size_t)atol(e);
    if (mfLowPower)
        std::fprintf(stderr, "CUDA: ... and match-first rebuild skipping on\n");
    // ...and off on the arena rungs too, for the opposite reason: a card takes one of
    // those because it is short of memory, and speculation's dedicated entry buffer is
    // 0.36 GiB of exactly that. Its co-blocks would also need pool metadata of their own.
    if (!std::getenv("MXBM_NO_SPEC") && !specOff && !p_->arena) {
        p_->specElem   = dalloc<uint64_t>(p_->nslots);
        p_->specCounts = dalloc<uint32_t>(p_->nb);
        p_->specPp     = dalloc<uint64_t>(4);
        p_->specOn = p_->specElem && p_->specCounts && p_->specPp;
        if (!p_->specOn) {
            cudaFree(p_->specElem);   p_->specElem = nullptr;
            cudaFree(p_->specCounts); p_->specCounts = nullptr;
            cudaFree(p_->specPp);     p_->specPp = nullptr;
        }
    }
}

// Defined here rather than defaulted in the header: ~unique_ptr<Impl> needs Impl to be
// a complete type, and the whole point of the pimpl is that it is not one there.
CudaSolver::~CudaSolver() = default;

std::vector<std::array<uint8_t,104>> CudaSolver::solve(const uint8_t input[32], const uint8_t nonce[8]) {
    std::vector<std::array<uint8_t,104>> out;
    {   // MXBM_VERIFY_STATS: candidates/solve needs the solve count as denominator.
        static const bool kVerifyStats = std::getenv("MXBM_VERIFY_STATS") != nullptr;
        if (kVerifyStats) bh3::verify_stats_note_solve();
    }
    Impl& I = *p_;
    // Per-thread, and cheap when it is already current. See the constructor.
    cudaSetDevice(I.device_index);
    I.abort_.store(false, std::memory_order_relaxed);

    uint64_t pp[4]; const uint8_t extra0[4] = {0,0,0,0};
    bh3::compute_prepow(input, 32, nonce, extra0, pp);
    cudaMemcpy(I.dpp, pp, 32, cudaMemcpyHostToDevice);
    cudaMemset(I.drops, 0, 16); cudaMemset(I.survCount, 0, 4);

    // Speculative entry co-scheduling. The engine walks nonces at a fixed stride
    // within a job, so consecutive calls teach the solver the delta; each solve then
    // seeds (nonce + delta) inside round 4's launch, and the next call skips its
    // entry pass if the prediction was right. A job change simply misses once.
    uint64_t nLE = 0; std::memcpy(&nLE, nonce, 8);
    const bool sameJob = I.haveLast && std::memcmp(I.lastInput, input, 32) == 0;
    if (sameJob) { I.learnedDelta = nLE - I.lastNonceLE; I.haveDelta = true; }
    I.haveLast = true; I.lastNonceLE = nLE; std::memcpy(I.lastInput, input, 32);

    const bool hit = I.specOn && I.specValid && I.specNonceLE == nLE
                  && std::memcmp(I.specInput, input, 32) == 0;
    I.specValid = false;
    const bool spec = I.specOn && I.haveDelta;  // seed nonce + delta during round 4
    // Entry output lives in the dedicated dense buffer when speculation is available,
    // because elem[0] is overwritten by round 2 and could not carry data across solves.
    uint32_t* r1Counts = I.specOn ? I.specCounts : I.counts[0];
    uint64_t* r1Elem   = I.specOn ? I.specElem   : I.elem[0];
    // Arena bookkeeping, both no-ops off the arena rungs. A set's pool is emptied before
    // the round that writes into it and threaded onto per-bucket chains after, so the
    // consumer finds the few elements its bucket could not hold. Speculation is off on
    // these rungs, so the entry pass always writes set 0.
    auto arenaClear = [&](int set) {
        if (!I.arena) return;
        cudaMemset(I.ahead[set], 0xFF, (size_t)I.nb*4);
        cudaMemset(I.actr[set], 0, 4);
    };
    auto arenaLink = [&](int set) {
        if (!I.arena) return;
        arena_link<<<(kArenaCap+255)/256, 256>>>(I.actr[set], I.atag[set],
                                                 I.ahead[set], I.anext[set], I.spillTot);
    };
    I.mark(0);
    if (!hit) {
        cudaMemset(r1Counts, 0, (size_t)I.nb*4);
#if MXBM_GCEN
        cudaMemset(I.kcen, 0, (size_t)1u << 22);
#endif
        arenaClear(0);
        entry_scatter<<<(kElems+255)/256,256>>>(I.dpp, 0, kElems, I.bb, I.cap,
                                                r1Counts, r1Elem, I.drops,
                                                I.arena ? I.actr[0] : nullptr,
                                                I.arena ? I.atag[0] : nullptr);
        arenaLink(0);
    }
    I.mark(1);
    if (spec) {
        uint64_t nextLE = nLE + I.learnedDelta;
        uint8_t nn[8]; std::memcpy(nn, &nextLE, 8);
        uint64_t npp[4];
        bh3::compute_prepow(input, 32, nn, extra0, npp);
        cudaMemcpy(I.specPp, npp, 32, cudaMemcpyHostToDevice);
        // Zeroing specCounts here is safe: round 1 reads it before round 4 runs, and
        // the stream is in-order -- but only when r1's input was already consumed by a
        // HIT above. On a miss the standalone entry just filled these buffers, so the
        // zeroing must wait until round 1 has read them; it is issued before round 4
        // below instead.
    }
    int inSet = 0;
    // What the terminal round reads: round 4's output, wherever this rung put it.
    const uint32_t* termC = nullptr; const uint64_t* termE = nullptr;
    const uint32_t* termAh = nullptr; const uint32_t* termAn = nullptr;
    // Where each round's back-reference row starts. Rounds 1-3 write none on an octo
    // rung, so round 4's row is the first and the terminal round's the second.
    const uint32_t r4Off = I.octo ? 0u : 3u*kCapacity;
    const uint32_t r5Off = I.octo ? kCapacity : 4u*kCapacity;
#if MXBM_GCEN
    #define KCEN_CLEAR cudaMemset(I.kcen, 0, (size_t)1u << 22);
#else
    #define KCEN_CLEAR
#endif
    // Variadic so a round can carry the optional trailing template arguments
    // (COTENANT, SUBPASS, FCAP) without every other round having to name them.
    // Round 1 reads the entry buffer, wherever this solve put it.
    // MFIRST and ARENA are template parameters, so every combination that can run needs
    // its own launch line; ROUND_L is that line, with only those two spelled out.
    #define ROUND_L(MF, AR, RF, N6, R, IMPB, ...)                                    \
            fused_round<__VA_ARGS__, false, false, false, MF, IMPB, AR, RF, N6>      \
              <<<I.nb << I.sm, kWG, I.smemPad>>>(I.bb, I.sm, I.cap, I.cap, (uint32_t)((R)-1)*kCapacity,\
                  inC, inE, I.counts[o], I.elem[o],                                  \
                  I.left, I.right, I.gictr, I.drops, I.dpp,                          \
                  arIn ? I.ahead[inSet] : nullptr, arIn ? I.anext[inSet] : nullptr,  \
                  arIn ? I.actr[o] : nullptr, arIn ? I.atag[o] : nullptr)
    #define ROUND_I(N6, R, IMPB, ...)                                                \
        { const int o = inSet ^ 1;                                                   \
          const uint32_t* inC = ((R)==1) ? r1Counts : I.counts[inSet];               \
          const uint64_t* inE = ((R)==1) ? r1Elem   : I.elem[inSet];                 \
          const bool arIn = I.arena;                                                 \
          cudaMemset(I.counts[o], 0, (size_t)I.nb*4); cudaMemset(I.gictr, 0, 4);     \
          KCEN_CLEAR                                                             \
          arenaClear(o);                                                             \
          if (arIn) { if (I.matchFirst) ROUND_L(true,  true,  true, N6, R, IMPB, __VA_ARGS__); \
                      else              ROUND_L(false, true,  true, N6, R, IMPB, __VA_ARGS__); }\
          else      { if (I.matchFirst) ROUND_L(true,  false, true, N6, R, IMPB, __VA_ARGS__); \
                      else              ROUND_L(false, false, true, N6, R, IMPB, __VA_ARGS__); }\
          arenaLink(o);                                                              \
          inSet = o; }
    // Rounds 1-3 of an octo rung, which write no references. Two arms rather than four:
    // an octo rung is always a dense-cap rung, so the plain-cap kernels cannot run here
    // and instantiating them would put kernels in the binary that never launch.
    #define ROUND_NRI(N6, R, IMPB, ...)                                              \
        { const int o = inSet ^ 1;                                                   \
          const uint32_t* inC = ((R)==1) ? r1Counts : I.counts[inSet];               \
          const uint64_t* inE = ((R)==1) ? r1Elem   : I.elem[inSet];                 \
          const bool arIn = true;                                                    \
          cudaMemset(I.counts[o], 0, (size_t)I.nb*4); cudaMemset(I.gictr, 0, 4);     \
          KCEN_CLEAR                                                             \
          arenaClear(o);                                                             \
          if (I.matchFirst) ROUND_L(true,  true, false, N6, R, IMPB, __VA_ARGS__);   \
          else              ROUND_L(false, true, false, N6, R, IMPB, __VA_ARGS__);   \
          arenaLink(o);                                                              \
          inSet = o; }
    #define ROUND(R, IMPB, ...)    ROUND_I((MXBM_NARROW6 != 0), R, IMPB, __VA_ARGS__)
    #define ROUND_NR(R, IMPB, ...) ROUND_NRI((MXBM_NARROW6 != 0), R, IMPB, __VA_ARGS__)
    // ROUND_X expands MXBM_Rn_ARGS before ROUND counts its arguments. The _6 forms
    // stage word 6 narrow (round 3 on a 64 KB-shared card).
    #define ROUND_X(...)    ROUND(__VA_ARGS__)
    #define ROUND_XNR(...)  ROUND_NR(__VA_ARGS__)
    #define ROUND_X6(...)   ROUND_I(true, __VA_ARGS__)
    #define ROUND_XNR6(...) ROUND_NRI(true, __VA_ARGS__)
    // Round 1 rides the rung's implied-bit count, derived from the condition round 2's
    // own arm below tests, so writer and reader agree on the r1 -> r2 layout: the
    // w0-checkpoint record with MXBM_PAIR_W0, the plain pair record without. IMPB != 0
    // also tells round 1 the rung writes no reference row (refRows is 0), so it stays
    // nonzero either way.
    const uint32_t r1b = (!I.octo && !I.quad && I.impb) ? (I.bb == 17u ? 17u : 16u) : 0u;
    if (I.octo)          ROUND_XNR(1, 0u,  MXBM_R1_ARGS)
    else if (r1b == 17u) ROUND_X(1, 17u, MXBM_R1_ARGS)
    else if (r1b == 16u) ROUND_X(1, 16u, MXBM_R1_ARGS)
    else                 ROUND_X(1, 0u,  MXBM_R1_ARGS)
    I.mark(2);
    // Rounds 2 and 3 come in a matched pair -- r2's OUTSTR is r3's INSTR -- so they
    // switch together or the second reads a record the first never wrote. The
    // implicit-bits pairs are the same strides with the pack folded into the
    // stores; the IMPB value is the bucket-bit count the pack drops, so each
    // geometry needs its own instantiation.
    // Both rounds at the device's staging caps: kFCap64K / kFCap64K3 with round 3's
    // narrow word-6 plane on a 64 KB-shared card, kFCap elsewhere.
    #define ROUND2(IMPB, A, AS) { if (I.smallShared) ROUND_X(2, IMPB, AS) else ROUND_X(2, IMPB, A) }
    #define ROUND2NR(IMPB, A, AS) { if (I.smallShared) ROUND_XNR(2, IMPB, AS) else ROUND_XNR(2, IMPB, A) }
    #define ROUND3(IMPB, A, AS) { if (I.smallShared) ROUND_X6(3, IMPB, AS) else ROUND_X(3, IMPB, A) }
    #define ROUND3NR(IMPB, A, AS) { if (I.smallShared) ROUND_XNR6(3, IMPB, AS) else ROUND_XNR(3, IMPB, A) }
    if (I.octo)          { ROUND2NR(0u, MXBM_R2QO_ARGS, MXBM_R2QO_ARGS_S) I.mark(3); ROUND3NR(0u, MXBM_R3QO_ARGS, MXBM_R3QO_ARGS_S) }
    else if (I.quad)     { ROUND2(0u,  MXBM_R2Q_ARGS, MXBM_R2Q_ARGS_S) I.mark(3); ROUND3(0u,  MXBM_R3Q_ARGS, MXBM_R3Q_ARGS_S) }
    else if (I.impb && I.bb == 17u)
                         { ROUND2(17u, MXBM_R2_ARGS, MXBM_R2_ARGS_S)  I.mark(3); ROUND3(17u, MXBM_R3_ARGS, MXBM_R3_ARGS_S)  }
    else if (I.impb)     { ROUND2(16u, MXBM_R2_ARGS, MXBM_R2_ARGS_S)  I.mark(3); ROUND3(16u, MXBM_R3_ARGS, MXBM_R3_ARGS_S)  }
    else                 { ROUND2(0u,  MXBM_R2_ARGS, MXBM_R2_ARGS_S)  I.mark(3); ROUND3(0u,  MXBM_R3_ARGS, MXBM_R3_ARGS_S)  }
    #undef ROUND3NR
    #undef ROUND3
    #undef ROUND2NR
    #undef ROUND2
    I.mark(4);
    #undef ROUND_XNR
    #undef ROUND_X
    #undef ROUND_NR
    #undef ROUND
    // Round 4, outside the macro so the co-blocks variant is instantiated for this
    // round only. When speculating it hosts the next nonce's entry pass as interleaved
    // co-blocks: grid *3/2, every third block seeds into the spec buffers.
    { const int o = inSet ^ 1;
      // Where round 4 puts its output. On its own plane wherever recovery needs round
      // 2's, which is still sitting in set 0; on set 0 otherwise, as before.
      const bool own = I.elem4 != nullptr;
      uint64_t* const r4E = own ? I.elem4   : I.elem[o];
      uint32_t* const r4C = own ? I.counts4 : I.counts[o];
      uint32_t* const r4Ah = own ? I.ahead4 : I.ahead[o];
      uint32_t* const r4An = own ? I.anext4 : I.anext[o];
      uint32_t* const r4At = own ? I.atag4  : I.atag[o];
      uint32_t* const r4Ac = own ? I.actr4  : I.actr[o];
      cudaMemset(r4C, 0, (size_t)I.nb*4); cudaMemset(I.gictr, 0, 4);
      KCEN_CLEAR
#undef KCEN_CLEAR
      if (I.arena) { cudaMemset(r4Ah, 0xFF, (size_t)I.nb*4); cudaMemset(r4Ac, 0, 4); }
      // Round 4's own chain arguments, spelled once: the arms below differ only in
      // which instantiation they name.
      #define R4_ARENA_ARGS I.arena ? I.ahead[inSet] : nullptr,                      \
                            I.arena ? I.anext[inSet] : nullptr,                      \
                            I.arena ? r4Ac : nullptr, I.arena ? r4At : nullptr
      #define R4_ARMS(R4A, R4OA) \
      if (spec) { \
          cudaMemset(I.specCounts, 0, (size_t)I.nb*4); \
          fused_round<R4A, false, false, true> \
            <<<(I.nb << I.sm) + (I.nb << I.sm)/2u, kWG, I.smemPad>>>(I.bb, I.sm, I.cap, I.cap, \
                r4Off, I.counts[inSet], I.elem[inSet], r4C, r4E, \
                I.left, I.right, I.gictr, I.drops, I.dpp, \
                nullptr, nullptr, nullptr, nullptr, \
                I.specPp, kElems, I.cap, I.specCounts, I.specElem, 3u); \
      } else if (I.octo && I.matchFirst) { \
          fused_round<R4OA, false, false, false, true, 0u, true, true, false, MXBM_MB_OCTO> \
            <<<I.nb << I.sm, kWG, I.smemPad>>>(I.bb, I.sm, I.cap, I.cap, r4Off, \
                I.counts[inSet], I.elem[inSet], r4C, r4E, \
                I.left, I.right, I.gictr, I.drops, I.dpp, R4_ARENA_ARGS); \
      } else if (I.octo) { \
          fused_round<R4OA, false, false, false, false, 0u, true, true, false, MXBM_MB_OCTO> \
            <<<I.nb << I.sm, kWG, I.smemPad>>>(I.bb, I.sm, I.cap, I.cap, r4Off, \
                I.counts[inSet], I.elem[inSet], r4C, r4E, \
                I.left, I.right, I.gictr, I.drops, I.dpp, R4_ARENA_ARGS); \
      } else if (I.arena && I.matchFirst) { \
          fused_round<R4A, false, false, false, true, 0u, true> \
            <<<I.nb << I.sm, kWG, I.smemPad>>>(I.bb, I.sm, I.cap, I.cap, r4Off, \
                I.counts[inSet], I.elem[inSet], r4C, r4E, \
                I.left, I.right, I.gictr, I.drops, I.dpp, R4_ARENA_ARGS); \
      } else if (I.arena) { \
          fused_round<R4A, false, false, false, false, 0u, true> \
            <<<I.nb << I.sm, kWG, I.smemPad>>>(I.bb, I.sm, I.cap, I.cap, r4Off, \
                I.counts[inSet], I.elem[inSet], r4C, r4E, \
                I.left, I.right, I.gictr, I.drops, I.dpp, R4_ARENA_ARGS); \
      } else if (I.matchFirst) { \
          fused_round<R4A, false, false, false, true> \
            <<<I.nb << I.sm, kWG, I.smemPad>>>(I.bb, I.sm, I.cap, I.cap, r4Off, \
                I.counts[inSet], I.elem[inSet], r4C, r4E, \
                I.left, I.right, I.gictr, I.drops, I.dpp); \
      } else { \
          fused_round<R4A> \
            <<<I.nb << I.sm, kWG, I.smemPad>>>(I.bb, I.sm, I.cap, I.cap, r4Off, \
                I.counts[inSet], I.elem[inSet], r4C, r4E, \
                I.left, I.right, I.gictr, I.drops, I.dpp); \
      }
      // Round 4 at the device's staging cap, like round 2. The arms above are one
      // macro so both caps name them once.
      if (I.smallShared) { R4_ARMS(MXBM_R4_ARGS_S, MXBM_R4O_ARGS_S) }
      else               { R4_ARMS(MXBM_R4_ARGS,   MXBM_R4O_ARGS) }
      #undef R4_ARMS
      #undef R4_ARENA_ARGS
      if (I.arena)
          arena_link<<<(kArenaCap+255)/256, 256>>>(r4Ac, r4At, r4Ah, r4An, I.spillTot);
      termC = r4C; termE = r4E; termAh = r4Ah; termAn = r4An;
      inSet = o; }
    I.mark(5);
    // Row 5 is written only where recovery reads it: an octo rung, or a build that keeps
    // round 4's row as the replay's reference arm. Off those, the terminal names its two
    // parents by slot in survL4 and the rows stop at three.
    const bool r5 = I.octo || MXBM_R4_ROWS;
    // One block per bucket where the whole bucket fits the terminal's staging and its
    // table stays a perfect hash; the sub-mask split otherwise (see kTermCap).
    const uint32_t tsm = (I.bb >= 16u && I.cap + kAMax <= kTermCap) ? 0u : I.sm;
    if (I.octo)
        terminal_round<true, 2u><<<I.nb << tsm, kWG, I.smemPad>>>(I.bb, tsm, I.cap, r5Off,
                                        termC, termE, I.left, I.right,
                                        I.survSlots, I.survCount, kSurvCap, I.drops,
                                        termAh, termAn);
    else if (I.arena)
        terminal_round<true><<<I.nb << tsm, kWG, I.smemPad>>>(I.bb, tsm, I.cap, r5Off,
                                        termC, termE,
                                        r5 ? I.left : nullptr, r5 ? I.right : nullptr,
                                        I.survSlots, I.survCount, kSurvCap, I.drops,
                                        termAh, termAn, I.survL4);
    else
        terminal_round<false><<<I.nb << tsm, kWG, I.smemPad>>>(I.bb, tsm, I.cap, r5Off,
                                        termC, termE,
                                        r5 ? I.left : nullptr, r5 ? I.right : nullptr,
                                        I.survSlots, I.survCount, kSurvCap, I.drops,
                                        nullptr, nullptr, I.survL4);

    I.mark(6);
    const cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) {
        // A failed launch otherwise just yields zero survivors, which reads like a
        // correctness bug rather than a configuration one.
        std::fprintf(stderr, "CUDA launch error: %s\n", cudaGetErrorString(le));
        return out;
    }
    if (spec) {   // round 4's co-blocks seeded (nonce + delta) for the next call
        I.specValid = true;
        I.specNonceLE = nLE + I.learnedDelta;
        std::memcpy(I.specInput, input, 32);
    }
    uint32_t hs = 0;
    cudaMemcpy(&hs, I.survCount, 4, cudaMemcpyDeviceToHost);
    // The copy above drained the stream, so every stage event up to the terminal round
    // has landed. Recovery's is read after its own copy below.
    if (I.timing) {
        for (int i = 0; i < 6; ++i) {
            float ms = 0.f;
            cudaEventElapsedTime(&ms, I.ev[i], I.ev[i + 1]);
            I.stageMs[i].push_back(ms);
        }
    }
    // MXBM_DROP_STATS=1: the drop counters and the arena's run-long spill total, per
    // solve. Every A/B is gated on drops == 0, and on an arena rung that gate only means
    // something beside a spill count that is NOT zero -- otherwise "no drops" and "the
    // overflow path never ran" read the same. Off the hot path when unset.
#if MXBM_GCEN
    // MXBM_GCEN_STATS=1: read the census back and count what it says. This is the
    // applied-assert AND an instrument independent of the drop counters -- the keys seen
    // exactly once ARE the elements with no partner, so the figure must land on the
    // 12.8 % the shared-memory prepass measured by a different route entirely. The table
    // read here is the LAST producer's, round 4's.
    static const bool kCenStats = std::getenv("MXBM_GCEN_STATS") != nullptr;
    if (kCenStats && I.kcen) {
        static std::vector<uint32_t> h(1u << 20);
        cudaMemcpy(h.data(), I.kcen, (size_t)1u << 22, cudaMemcpyDeviceToHost);
        uint64_t seen = 0, multi = 0;
        for (uint32_t w : h) {
            seen  += (uint32_t)__builtin_popcount(w & 0x55555555u);
            multi += (uint32_t)__builtin_popcount(w & 0xAAAAAAAAu);
        }
        std::fprintf(stderr, "kcen seen=%llu multi=%llu single=%llu (%.2f %% of elements)\n",
                     (unsigned long long)seen, (unsigned long long)multi,
                     (unsigned long long)(seen - multi),
                     100.0 * (double)(seen - multi) / (double)(1u << 25));
    }
#endif
    static const bool kDropStats = std::getenv("MXBM_DROP_STATS") != nullptr;
    if (kDropStats) {
        uint32_t d[4] = {0,0,0,0}, sp = 0;
        cudaMemcpy(d, I.drops, 16, cudaMemcpyDeviceToHost);
        if (I.spillTot) cudaMemcpy(&sp, I.spillTot, 4, cudaMemcpyDeviceToHost);
        // One capacity per field: entry's scatter, shared staging (group cap, arena
        // chain, terminal stage), output bucket or survivor cap, and the chain walk.
        std::fprintf(stderr, "drops entry=%u stage=%u out=%u walk=%u | spills=%u\n",
                     d[0], d[1], d[2], d[3], sp);
    }
    if (hs > kSurvCap) hs = kSurvCap;
    if (hs == 0 || I.abort_.load(std::memory_order_relaxed)) {
        if (I.timing) I.stageMs[6].push_back(0.f);
        return out;
    }

    // Set 1 still holds round 3's output: round 4 wrote set 0 and the terminal round read
    // it, so the octo records are untouched and their leaves are what recovery reads.
    if (I.octo)
        recover<true><<<(hs+63)/64, 64>>>(hs, I.survSlots, kCapacity, I.left, I.right,
                                          I.dleaves, I.elem[1], kOctoStride);
    else {
        // No reference row is written for round 4: replay_r4 re-runs it over the one
        // bucket of round 3's output the record names and hands back the survivor's four
        // round-3 ancestors by slot. Two blocks per survivor, ~4 in the whole grid.
        cudaMemset(I.l3Slots, 0xFF, (size_t)hs*4*4);
        replay_r4<<<2*hs, 256>>>(hs, I.survL4, I.elem4 ? I.elem4 : I.elem[0], I.elem[1],
                                 8u, I.counts[1], I.cap, I.l3Slots, I.l4Lead, I.drops,
                                 I.bb, I.arena ? I.ahead[1] : nullptr,
                                 I.arena ? I.anext[1] : nullptr);
        if (I.elem4) {
            // ...and none for rounds 1-3 either. replay_r3 does the same one level down,
            // reaching round 2's output, whose record carries four leaves in tree order.
            cudaMemset(I.l2Slots, 0xFF, (size_t)hs*8*4);
            replay_r3<<<4*hs, 256>>>(hs, I.l3Slots, I.elem[1], 8u, I.elem[0], 8u, I.bb,
                                     I.counts[0], I.cap, I.l2Slots, I.drops, I.bb,
                                     I.arena ? I.ahead[0] : nullptr,
                                     I.arena ? I.anext[0] : nullptr);
            recover_from_l2<<<(hs+63)/64, 64>>>(hs, I.l2Slots, I.l4Lead,
                                                I.elem[0], 8u, I.dleaves);
        } else {
            recover_from_l3<<<(hs+63)/64, 64>>>(hs, kCapacity, I.left, I.right, I.l3Slots,
                                                MXBM_R4_ROWS ? nullptr : I.l4Lead,
                                                I.elem[1], 8u, I.dleaves);
        }
    }
    I.mark(7);
    std::vector<uint32_t> hl((size_t)hs*32);
    cudaMemcpy(hl.data(), I.dleaves, (size_t)hs*32*4, cudaMemcpyDeviceToHost);
    if (I.timing) {
        float ms = 0.f;
        cudaEventElapsedTime(&ms, I.ev[6], I.ev[7]);
        I.stageMs[6].push_back(ms);
    }
#if MXBM_R4_ROWS
    // The replay's gate: with the rows restored, the two paths must produce the same 32
    // leaves for every survivor. Never on a shipping build -- the rows are the thing
    // being deleted -- so this is a rebuild, not a runtime flag.
    if (!I.octo && hs) {
        recover<false><<<(hs+63)/64, 64>>>(hs, I.survSlots, kCapacity, I.left, I.right,
                                           I.dleaves);
        std::vector<uint32_t> h2((size_t)hs*32);
        cudaMemcpy(h2.data(), I.dleaves, (size_t)hs*32*4, cudaMemcpyDeviceToHost);
        static uint64_t nOk = 0, nBad = 0;
        if (h2 == hl) ++nOk; else ++nBad;
        std::fprintf(stderr, "replay-check: %llu match, %llu differ\n",
                     (unsigned long long)nOk, (unsigned long long)nBad);
    }
#endif
    // MXBM_VERIFY_STATS=1: classify every candidate's verify outcome instead of
    // only pass/fail, so the found-vs-verified gap is attributed by reject reason
    // (tallies print at exit). The mining path is unchanged when unset.
    static const bool kVerifyStats = std::getenv("MXBM_VERIFY_STATS") != nullptr;
    for (uint32_t i = 0; i < hs; ++i) {
        std::array<uint8_t,104> sol{};
        bh3::pack_indices(&hl[(size_t)i*32], sol.data());
        if (!kVerifyStats) {
            if (bh3::is_valid_solution(input, 32, nonce, sol.data())) out.push_back(sol);
        } else {
            const bh3::VerifyReason r = bh3::classify_solution(input, 32, nonce, sol.data());
            bh3::verify_stats_tally(r);
            if (r.kind == bh3::VerifyReason::None) out.push_back(sol);
        }
    }
    return out;
}

}} // namespace mxbm::gpu
