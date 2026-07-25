// CUDA backend implementation. The kernels live in kernels/cuda/ and are shared with the
// standalone bench (cuda/pipeline.cu), so there is one copy of each.
#include "gpu/cuda_solver.h"
#include "pipeline_kernels.cuh"
#include "beamhash/bh3_blake2b.h"
#include "beamhash/bh3_verify.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace mxbm { namespace gpu {
using namespace mxbm::cuda;

namespace {
template<class T> T* dalloc(size_t n) { void* p=nullptr; return cudaMalloc(&p, n*sizeof(T))==cudaSuccess ? (T*)p : nullptr; }

constexpr uint32_t kElems    = 1u << 25;
constexpr uint32_t kCapacity = kElems + kElems/32;      // 34,603,008
constexpr uint32_t kBB = 16, kSM = 1, kNB = 1u << kBB;
constexpr uint32_t kSurvCap = 1024;
// set 0 carries the round-2 and round-4 outputs, set 1 the round-1 and round-3 outputs.
// Round 2's record is 9 u64: an 8-u64 record at a 16 B-aligned stride, plus a 9th-word
// plane laid out after all the records (fused_round derives its offset from the same
// geometry). Set 0 is therefore sized 8 + 1 rather than the 10 it took when the record
// was padded to keep the 128-bit accesses aligned -- see docs/performance.md.
constexpr uint32_t kSetStride[2] = { 9u, 8u };
constexpr uint32_t kR2RecStride  = 8u;   // round 2's record; the 9th word is the plane

// Total device memory the solver needs, so available() can refuse a device that would
// only fit a reduced seed layer.
size_t footprint_bytes() {
    const uint32_t mean = kCapacity / kNB;
    const uint32_t cap  = mean + (uint32_t)(8.0*std::sqrt((double)mean)) + 32u;
    const size_t nslots = (size_t)kNB * cap;
    return nslots*(kSetStride[0]+kSetStride[1])*8 + (size_t)5*kCapacity*4*2 + (size_t)2*kNB*4;
}
} // namespace

struct CudaSolver::Impl {
    uint32_t cap = 0; size_t nslots = 0;
    uint64_t *elem[2] = {nullptr,nullptr}, *dpp = nullptr;
    uint32_t *counts[2] = {nullptr,nullptr}, *gictr=nullptr, *drops=nullptr;
    uint32_t *left=nullptr, *right=nullptr, *survSlots=nullptr, *survCount=nullptr, *dleaves=nullptr;
    std::atomic<bool> abort_{false};
    DeviceInfo info;
};

bool CudaSolver::available() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return false;
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return false;
    if (p.major < 8) return false;                      // needs Ampere+ shared-mem budget
    // Refuse anything that cannot host the full seed layer: a reduced layer mines nothing.
    return (size_t)p.totalGlobalMem > footprint_bytes() + (size_t)(1ull<<30);
}

CudaSolver::DeviceInfo CudaSolver::device_info() {
    DeviceInfo d; cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, 0) == cudaSuccess) {
        d.name = p.name; d.global_mem = p.totalGlobalMem; d.compute_units = p.multiProcessorCount;
    }
    return d;
}

const CudaSolver::DeviceInfo& CudaSolver::device() const { return p_->info; }
void CudaSolver::request_abort() { p_->abort_.store(true, std::memory_order_relaxed); }

CudaSolver::CudaSolver() : p_(new Impl) {
    p_->info = device_info();
    const uint32_t mean = kCapacity / kNB;
    p_->cap    = mean + (uint32_t)(8.0*std::sqrt((double)mean)) + 32u;
    p_->nslots = (size_t)kNB * p_->cap;
    p_->elem[0] = dalloc<uint64_t>(p_->nslots*kSetStride[0]);
    p_->elem[1] = dalloc<uint64_t>(p_->nslots*kSetStride[1]);
    p_->counts[0] = dalloc<uint32_t>(kNB); p_->counts[1] = dalloc<uint32_t>(kNB);
    p_->gictr = dalloc<uint32_t>(1); p_->drops = dalloc<uint32_t>(4);
    p_->left  = dalloc<uint32_t>((size_t)5*kCapacity);
    p_->right = dalloc<uint32_t>((size_t)5*kCapacity);
    p_->survSlots = dalloc<uint32_t>(kSurvCap); p_->survCount = dalloc<uint32_t>(1);
    p_->dleaves = dalloc<uint32_t>((size_t)kSurvCap*32);
    p_->dpp = dalloc<uint64_t>(4);
    if (!p_->elem[0] || !p_->elem[1] || !p_->left || !p_->right || !p_->dpp)
        throw std::runtime_error("CUDA allocation failed (device out of memory)");
    // Ada splits L1/shared by a carveout that defaults to favouring L1; ask for shared.
    #define CARVE(K) cudaFuncSetAttribute(K, cudaFuncAttributePreferredSharedMemoryCarveout, \
                                          cudaSharedmemCarveoutMaxShared)
    CARVE((fused_round<7,7,2,LM_SEED,424u,2u,1u,2u,2u,1u,2u>));
    CARVE((fused_round<7,7,2,LM_RD2,400u,4u,2u,4u,4u,2u,10u>));
    CARVE((fused_round<7,6,4,LM_EMIT,376u,6u,4u,2u,8u,10u,8u>));
    CARVE((fused_round<6,1,2,LM_USE,288u,9u,2u,0u,0u,8u,2u>));
    CARVE(terminal_round);
    #undef CARVE
}

CudaSolver::~CudaSolver() {
    if (!p_) return;
    for (auto* q : {p_->elem[0], p_->elem[1], p_->dpp}) cudaFree(q);
    for (auto* q : {p_->counts[0], p_->counts[1], p_->gictr, p_->drops,
                    p_->left, p_->right, p_->survSlots, p_->survCount, p_->dleaves}) cudaFree(q);
}

std::vector<std::array<uint8_t,104>> CudaSolver::solve(const uint8_t input[32], const uint8_t nonce[8]) {
    std::vector<std::array<uint8_t,104>> out;
    Impl& I = *p_;
    I.abort_.store(false, std::memory_order_relaxed);

    uint64_t pp[4]; const uint8_t extra0[4] = {0,0,0,0};
    bh3::compute_prepow(input, 32, nonce, extra0, pp);
    cudaMemcpy(I.dpp, pp, 32, cudaMemcpyHostToDevice);
    cudaMemset(I.drops, 0, 16); cudaMemset(I.survCount, 0, 4);
    cudaMemset(I.counts[0], 0, (size_t)kNB*4);

    entry_scatter<<<(kElems+255)/256,256>>>(I.dpp, 0, kElems, kBB, I.cap, I.counts[0], I.elem[0], I.drops);
    int inSet = 0;
    #define ROUND(R, INW,OUTW,LEAFW,MODE, LOUT,PADN,SIN,SOUT,SBUILD, INSTR,OUTSTR)   \
        { const int o = inSet ^ 1;                                                   \
          cudaMemset(I.counts[o], 0, (size_t)kNB*4); cudaMemset(I.gictr, 0, 4);      \
          fused_round<INW,OUTW,LEAFW,MODE,LOUT,PADN,SIN,SOUT,SBUILD,INSTR,OUTSTR>    \
            <<<kNB << kSM, kWG>>>(kBB, kSM, I.cap, I.cap, (uint32_t)((R)-1)*kCapacity,\
                I.counts[inSet], I.elem[inSet], I.counts[o], I.elem[o],               \
                I.left, I.right, I.gictr, I.drops, I.dpp);                            \
          inSet = o; }
    ROUND(1, 7,7,2,LM_SEED, 424u,2u,1u,2u,2u, 1u,2u)
    ROUND(2, 7,7,2,LM_RD2,  400u,4u,2u,4u,4u, 2u,kR2RecStride)
    ROUND(3, 7,6,4,LM_EMIT, 376u,6u,4u,2u,8u, kR2RecStride,8u)
    ROUND(4, 6,1,2,LM_USE,  288u,9u,2u,0u,0u, 8u,2u)
    #undef ROUND
    terminal_round<<<kNB << kSM, kWG>>>(kBB, kSM, I.cap, 4u*kCapacity, I.counts[inSet],
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
