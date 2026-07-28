// CUDA backend implementation. The kernels live in kernels/cuda/ and are shared with the
// standalone bench (cuda/pipeline.cu), so there is one copy of each.
#include "gpu/cuda_solver.h"
#include "gpu/rowbucket_geom.h"
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
constexpr uint32_t kSurvCap = 1024;
// set 0 carries the round-2 and round-4 outputs, set 1 the round-1 and round-3 outputs.
// Round 2's record is 9 u64: an 8-u64 record at a 16 B-aligned stride, plus a 9th-word
// plane laid out after all the records (fused_round derives its offset from the same
// geometry). Set 0 is therefore sized 8 + 1 rather than the 10 it took when the record
// was padded to keep the 128-bit accesses aligned -- see docs/performance.md.
constexpr uint32_t kSetStride[2] = { 9u, 8u };
constexpr uint32_t kR2RecStride  = 8u;   // round 2's record; the 9th word is the plane

// Each round's template arguments, named once. They were written out separately at the
// launch and at the carveout call, and had drifted: the carveout was applied to LM_RD2
// with OUTSTR 10 and LM_EMIT with INSTR 10, instantiations that no longer run, so the
// kernels that DO run never received the preference.
#define MXBM_R1_ARGS 7,7,1,LM_SEED, 424u,2u,1u,2u,2u, 1u,2u,MXBM_R1_FCAP
#define MXBM_R2_ARGS 7,7,2,LM_RD2,  400u,4u,2u,4u,4u, 2u,kR2RecStride,kFCap
#define MXBM_R3_ARGS 7,6,4,LM_EMIT, 376u,6u,4u,2u,8u, kR2RecStride,8u,kFCap
#define MXBM_R4_ARGS 6,1,2,LM_USE,  288u,9u,2u,0u,0u, 8u,2u,kFCap
// The quad-record pair. Only rounds 2 and 3 differ, and only in their strides and round
// 3's mode: the record carries the same INFORMATION either way, so every L value, tree
// width and cap is identical. Both pairs are instantiated and the choice is made at
// RUNTIME from the geometry, because which one a card wants depends on its VRAM.
constexpr uint32_t kQuadStride = 3u;
#define MXBM_R2Q_ARGS 7,7,2,LM_RD2, 400u,4u,2u,4u,4u, 2u,kQuadStride,kFCap
#define MXBM_R3Q_ARGS 7,6,4,LM_RD3, 376u,6u,4u,2u,8u, kQuadStride,8u,kFCap
static_assert(kSetStride[0] == fb_set_stride(0) && kSetStride[1] == fb_set_stride(1),
              "CUDA record widths must match the shared footprint arithmetic");
static_assert(kQuadStride == fb_round_stride(2, true) &&
              fb_set_stride(0, true) == kQuadStride && fb_set_stride(1, true) == 8u,
              "quad record widths must match the shared footprint arithmetic");

// Was hardcoded (16,1), so a card that could not host 7.46 GiB was refused outright --
// even though the same kernels run unchanged at 6.50 GiB. bb and sm are kernel
// ARGUMENTS, and everything compile-time is invariant along the bb + sm = 17 line: the
// grid is 2^17 blocks, the staged group is capacity / 2^17 = 264, and the 128-entry
// chain table is a perfect hash for the 24 - bb - sm = 7 bits left varying.
// max_alloc = 0: CUDA has no per-allocation limit (see rowbucket_geom.h).
RbGeometry pick_geometry(uint64_t global_mem) {
    // allow_quad: the 24 B record is implemented in the CUDA kernels only, so this is
    // the one caller that passes true. It costs +14 % time and buys no watts at any cap
    // (measured), so the ladder only reaches for it when a card cannot host a packed
    // rung -- which is exactly what takes the CUDA path below 6.5 GiB.
    RbGeometry g = rb_geometry_for(kCapacity, /*max_alloc=*/0, global_mem, /*allow_quad=*/true);
    if (const char* e = std::getenv("MXBM_BB")) { g.bb = (uint32_t)atoi(e); g.sm = 17u - g.bb;
                                                 g.viable = true; }
    if (const char* e = std::getenv("MXBM_SM")) { g.sm = (uint32_t)atoi(e); g.viable = true; }
    if (const char* e = std::getenv("MXBM_QUAD")) { g.quad = atoi(e) != 0; g.viable = true; }
    return g;
}
} // namespace

struct CudaSolver::Impl {
    uint32_t bb = 16, sm = 1, nb = 1u << 16;
    bool quad = false;                  // 24 B round-2 record; see rowbucket_geom.h
    uint32_t cap = 0; size_t nslots = 0;
    uint64_t *elem[2] = {nullptr,nullptr}, *dpp = nullptr;
    uint32_t *counts[2] = {nullptr,nullptr}, *gictr=nullptr, *drops=nullptr;
    uint32_t *left=nullptr, *right=nullptr, *survSlots=nullptr, *survCount=nullptr, *dleaves=nullptr;
    std::atomic<bool> abort_{false};
    DeviceInfo info;
    int device_index = 0;

    // The geometry-dependent half of the allocation: the two record sets and their
    // bucket counters. Split out so a step-down retries only what changes size.
    bool alloc_geometry(uint32_t bb_, uint32_t sm_, bool quad_) {
        for (auto*& q : elem)   { cudaFree(q); q = nullptr; }
        for (auto*& q : counts) { cudaFree(q); q = nullptr; }
        bb = bb_; sm = sm_; nb = 1u << bb_; quad = quad_;
        cap    = fb_cap_for(kCapacity / nb);
        nslots = (size_t)nb * cap;
        // Set 0 is 9 u64 for the packed record and 3 for the quad one -- taken from the
        // shared helper rather than a local constant so the allocation cannot disagree
        // with what rb_geometry_for sized the ladder against.
        elem[0] = dalloc<uint64_t>(nslots*fb_set_stride(0, quad));
        elem[1] = dalloc<uint64_t>(nslots*fb_set_stride(1, quad));
        counts[0] = dalloc<uint32_t>(nb); counts[1] = dalloc<uint32_t>(nb);
        return elem[0] && elem[1] && counts[0] && counts[1];
    }

    // The frees belong here, not in ~CudaSolver: that constructor throws when a device
    // is too small, and a throwing constructor never runs its own destructor, so every
    // cudaMalloc before the throw leaked for the life of the process. A caller catching
    // the throw and retrying smaller therefore got LESS memory on each attempt. ~Impl
    // runs either way, p_ being a fully-constructed member.
    ~Impl() {
        for (auto* q : {elem[0], elem[1], dpp}) cudaFree(q);
        for (auto* q : {counts[0], counts[1], gictr, drops,
                        left, right, survSlots, survCount, dleaves}) cudaFree(q);
    }
};

bool CudaSolver::available(int index) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return false;
    if (index < 0 || index >= n) return false;
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, index) != cudaSuccess) return false;
    if (p.major < 8) return false;                      // needs Ampere+ shared-mem budget
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
        d.viable = p.major >= 8 && pick_geometry(p.totalGlobalMem).viable;
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
    std::sort(out.begin(), out.end(), [](const DeviceInfo& a, const DeviceInfo& b) {
        return a.pci < b.pci;
    });
    return out;
}

const CudaSolver::DeviceInfo& CudaSolver::device() const { return p_->info; }
void CudaSolver::request_abort() { p_->abort_.store(true, std::memory_order_relaxed); }

CudaSolver::CudaSolver(int index) : p_(new Impl) {
    // Bind THIS thread to the chosen device before any allocation. CUDA's current
    // device is per-thread, which is why solve() sets it again below: the engine
    // runs solve() on a worker thread that never saw this constructor, and would
    // otherwise use device 0's context against this device's pointers.
    if (cudaSetDevice(index) != cudaSuccess)
        throw std::runtime_error("cudaSetDevice(" + std::to_string(index) + ") failed");
    p_->device_index = index;
    p_->info = device_info(index);
    // Geometry-independent first, so the step-down below is not retrying these.
    p_->gictr = dalloc<uint32_t>(1); p_->drops = dalloc<uint32_t>(4);
    p_->left  = dalloc<uint32_t>((size_t)5*kCapacity);
    p_->right = dalloc<uint32_t>((size_t)5*kCapacity);
    p_->survSlots = dalloc<uint32_t>(kSurvCap); p_->survCount = dalloc<uint32_t>(1);
    p_->dleaves = dalloc<uint32_t>((size_t)kSurvCap*32);
    p_->dpp = dalloc<uint64_t>(4);
    if (!p_->left || !p_->right || !p_->dpp)
        throw std::runtime_error("CUDA allocation failed (device out of memory)");

    // Then walk the ladder for real: rb_geometry_for() sizes against TOTAL VRAM, but a
    // desktop compositor can be sitting on a gigabyte of it, and whether a footprint
    // fits *today* is only knowable by asking the allocator. An explicit MXBM_BB /
    // MXBM_SM is the user's choice and is not stepped down under them.
    const RbGeometry g = pick_geometry(p_->info.global_mem);
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
                     || std::getenv("MXBM_QUAD");
    bool ok = p_->alloc_geometry(g.bb, g.sm, g.quad);
    // Keep stepping down the SAME ladder rb_geometry_for walked, rather than
    // decrementing bb: the rungs interleave the two record formats, so a bb-only retry
    // would stop at the bottom of the packed half and refuse a card the quad rungs
    // would have hosted.
    if (!ok && !forced) {
        int n = 0;
        const RbRung* rungs = rb_rungs(n);
        int i = 0;
        while (i < n && !(rungs[i].bb == g.bb && rungs[i].quad == g.quad)) ++i;
        for (++i; !ok && i < n; ++i) {
            std::fprintf(stderr, "CUDA: %u buckets%s did not fit, retrying at %u%s\n",
                         1u << g.bb, g.quad ? " (quad record)" : "",
                         1u << rungs[i].bb, rungs[i].quad ? " (quad record)" : "");
            ok = p_->alloc_geometry(rungs[i].bb, rungs[i].sm, rungs[i].quad);
        }
    }
    if (!ok) throw std::runtime_error("CUDA allocation failed (device out of memory)");
    // Ada splits L1/shared by a carveout that defaults to favouring L1; ask for shared.
    #define CARVE(K) cudaFuncSetAttribute(K, cudaFuncAttributePreferredSharedMemoryCarveout, \
                                          cudaSharedmemCarveoutMaxShared)
    CARVE((fused_round<MXBM_R1_ARGS>)); CARVE((fused_round<MXBM_R2_ARGS>));
    CARVE((fused_round<MXBM_R3_ARGS>)); CARVE((fused_round<MXBM_R4_ARGS>));
    // Both record formats are instantiated; the carveout has to reach the pair that
    // actually runs, and applying it to a kernel that never launches is what the
    // MXBM_Rn_ARGS naming exists to prevent (see the drift bug in cad8fde).
    CARVE((fused_round<MXBM_R2Q_ARGS>)); CARVE((fused_round<MXBM_R3Q_ARGS>));
    CARVE(terminal_round);
    #undef CARVE
}

// Defined here rather than defaulted in the header: ~unique_ptr<Impl> needs Impl to be
// a complete type, and the whole point of the pimpl is that it is not one there.
CudaSolver::~CudaSolver() = default;

std::vector<std::array<uint8_t,104>> CudaSolver::solve(const uint8_t input[32], const uint8_t nonce[8]) {
    std::vector<std::array<uint8_t,104>> out;
    Impl& I = *p_;
    // Per-thread, and cheap when it is already current. See the constructor.
    cudaSetDevice(I.device_index);
    I.abort_.store(false, std::memory_order_relaxed);

    uint64_t pp[4]; const uint8_t extra0[4] = {0,0,0,0};
    bh3::compute_prepow(input, 32, nonce, extra0, pp);
    cudaMemcpy(I.dpp, pp, 32, cudaMemcpyHostToDevice);
    cudaMemset(I.drops, 0, 16); cudaMemset(I.survCount, 0, 4);
    cudaMemset(I.counts[0], 0, (size_t)I.nb*4);

    entry_scatter<<<(kElems+255)/256,256>>>(I.dpp, 0, kElems, I.bb, I.cap, I.counts[0], I.elem[0], I.drops);
    int inSet = 0;
    // Variadic so a round can carry the optional trailing template arguments
    // (COTENANT, SUBPASS, FCAP) without every other round having to name them.
    #define ROUND(R, ...)                                                            \
        { const int o = inSet ^ 1;                                                   \
          cudaMemset(I.counts[o], 0, (size_t)I.nb*4); cudaMemset(I.gictr, 0, 4);     \
          fused_round<__VA_ARGS__>                                                   \
            <<<I.nb << I.sm, kWG>>>(I.bb, I.sm, I.cap, I.cap, (uint32_t)((R)-1)*kCapacity,\
                I.counts[inSet], I.elem[inSet], I.counts[o], I.elem[o],               \
                I.left, I.right, I.gictr, I.drops, I.dpp);                            \
          inSet = o; }
    // ROUND_X expands MXBM_Rn_ARGS before ROUND counts its arguments.
    #define ROUND_X(...) ROUND(__VA_ARGS__)
    ROUND_X(1, MXBM_R1_ARGS)
    // Rounds 2 and 3 come in a matched pair -- r2's OUTSTR is r3's INSTR -- so they
    // switch together or the second reads a record the first never wrote.
    if (I.quad) { ROUND_X(2, MXBM_R2Q_ARGS) ROUND_X(3, MXBM_R3Q_ARGS) }
    else        { ROUND_X(2, MXBM_R2_ARGS)  ROUND_X(3, MXBM_R3_ARGS)  }
    ROUND_X(4, MXBM_R4_ARGS)
    #undef ROUND_X
    #undef ROUND
    terminal_round<<<I.nb << I.sm, kWG>>>(I.bb, I.sm, I.cap, 4u*kCapacity, I.counts[inSet],
                                        I.elem[inSet], I.left, I.right, I.survSlots,
                                        I.survCount, kSurvCap, I.drops);

    const cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) {
        // A failed launch otherwise just yields zero survivors, which reads like a
        // correctness bug rather than a configuration one.
        std::fprintf(stderr, "CUDA launch error: %s\n", cudaGetErrorString(le));
        return out;
    }
    uint32_t hs = 0;
    cudaMemcpy(&hs, I.survCount, 4, cudaMemcpyDeviceToHost);
    if (hs > kSurvCap) hs = kSurvCap;
    if (hs == 0 || I.abort_.load(std::memory_order_relaxed)) return out;

    recover<<<(hs+63)/64, 64>>>(hs, I.survSlots, kCapacity, I.left, I.right, I.dleaves);
    std::vector<uint32_t> hl((size_t)hs*32);
    cudaMemcpy(hl.data(), I.dleaves, (size_t)hs*32*4, cudaMemcpyDeviceToHost);
    for (uint32_t i = 0; i < hs; ++i) {
        std::array<uint8_t,104> sol{};
        bh3::pack_indices(&hl[(size_t)i*32], sol.data());
        if (bh3::is_valid_solution(input, 32, nonce, sol.data())) out.push_back(sol);
    }
    return out;
}

}} // namespace mxbm::gpu
