// CUDA row-bucket pipeline, gated on the KAT. Standalone: builds with nvcc, not wired
// into CMake, so an incomplete backend cannot destabilise the shipping OpenCL path.
//
//   nvcc -O3 -arch=sm_89 -I src -I kernels/cuda -I tests cuda/pipeline.cu -o cuda/pipeline
//
// The gate is the same one the OpenCL path uses and is non-negotiable: survivors == 3 on
// the KAT prePow, bucketDrops == 0, pairDrops == 0.
#include "fused_round.cuh"
#include "kat_vectors.h"
#include <cstdio>
#include <cmath>
#ifndef MXBM_R2_FULL
#define MXBM_R2_FULL 0
#endif
#include <vector>
#include "beamhash/bh3_verify.h"
#include "beamhash/bh3_blake2b.h"
#include <algorithm>
#include <chrono>
#include <array>
#include <cstdlib>
#include <cstring>
using namespace mxbm;
using namespace mxbm::cuda;

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__); return 1; } }while(0)

// ---- entry: seed + mix + scatter into round-1 buckets (8 B thin record) -------------
__global__ __launch_bounds__(256)
void entry_scatter(const uint64_t* __restrict__ pp4, uint32_t begin, uint32_t count,
                   uint32_t bucket_bits, uint32_t bucket_cap,
                   uint32_t* __restrict__ counts, uint64_t* __restrict__ belem,
                   uint32_t* __restrict__ drops) {
    const uint32_t g = blockIdx.x*blockDim.x + threadIdx.x;
    if (g >= count) return;
    const uint32_t idx = begin + g;
    uint64_t pp[4] = { pp4[0], pp4[1], pp4[2], pp4[3] };
    bh3::Elem e; bh3::seed_element(pp, idx, e);
    uint32_t t1[1] = { idx };
    bh3::apply_mix(e, t1, 1u, 448u);
    const uint32_t key = (uint32_t)(e.w[0] & 0xFFFFFFu);
    const uint32_t b = key >> (24u - bucket_bits);
    const uint32_t pos = atomicAdd(&counts[b], 1u);
    if (pos < bucket_cap) belem[(size_t)b*bucket_cap + pos] = ((uint64_t)idx << 32) | key;
    else atomicAdd(&drops[1], 1u);
}

// ---- terminal: combine at Lout(5)=24 and keep the all-zero survivors ----------------
// Only work word 0 is consulted: at Lout=24 combine zeroes c[1..6] and masks c[0] to 24
// bits, and the x[1]<<40 term contributes nothing below bit 40. See "the terminal
// round's dead work words" in docs/performance.md.
__global__ __launch_bounds__(kWG)
void terminal_round(uint32_t bucket_bits, uint32_t submask_bits, uint32_t in_bucket_cap,
                    uint32_t out_off,
                    const uint32_t* __restrict__ in_counts,
                    const uint64_t* __restrict__ in_belem,
                    uint32_t* __restrict__ all_left, uint32_t* __restrict__ all_right,
                    uint32_t* __restrict__ surv_slots, uint32_t* __restrict__ surv_count,
                    uint32_t surv_cap, uint32_t* __restrict__ drops) {
    __shared__ uint64_t lwork[kFCap];
    __shared__ uint32_t lgi[kFCap], llead[kFCap], lkey[kFCap], lchain[kFCap], tab[kTabSize];
    __shared__ uint32_t gcount;
    const uint32_t lId = threadIdx.x;
    const uint32_t submaskCount = 1u << submask_bits;
    const uint32_t bucket = blockIdx.x / submaskCount, mask = blockIdx.x % submaskCount;
    if (lId == 0) gcount = 0;
    for (uint32_t i = lId; i < kTabSize; i += kWG) tab[i] = kEmpty;
    __syncthreads();
    uint32_t cnt = in_counts[bucket];
    if (cnt > in_bucket_cap) cnt = in_bucket_cap;
    const size_t base = (size_t)bucket * in_bucket_cap;
    for (uint32_t p = lId; p < cnt; p += kWG) {
        const size_t d = (base + p) * 2u;                 // kFbStride[5]
        const uint64_t w0 = in_belem[d];
        const uint32_t key = (uint32_t)(w0 & 0xFFFFFFu);
        if ((key & (submaskCount - 1u)) != mask) continue;
        const uint32_t pos = atomicAdd(&gcount, 1u);
        if (pos >= kFCap) { atomicAdd(&drops[1], 1u); continue; }
        const uint64_t meta = in_belem[d + 1];
        lwork[pos] = w0; lgi[pos] = (uint32_t)(meta >> 32);
        llead[pos] = (uint32_t)meta; lkey[pos] = key;
    }
    __syncthreads();
    const uint32_t total = gcount < kFCap ? gcount : kFCap;
    for (uint32_t pos = lId; pos < total; pos += kWG) {
        const uint32_t hk = (lkey[pos] >> submask_bits) & (kTabSize - 1u);
        lchain[pos] = atomicExch(&tab[hk], pos);
    }
    __syncthreads();
    for (uint32_t pos = lId; pos < total; pos += kWG) {
        const uint32_t key = lkey[pos];
        uint32_t oth = lchain[pos], walk = 0;
        while (oth != kEmpty) {
            if (++walk > 64u) { atomicAdd(&drops[3], 1u); break; }
            if (lkey[oth] == key) {
                const uint32_t la = llead[pos], lb = llead[oth];
                const uint32_t ga = lgi[pos],  gb = lgi[oth];
                uint32_t L = pos, R = oth;
                if (lb < la || (lb == la && gb < ga)) { L = oth; R = pos; }
                bh3::Elem a{}, b{}, c;
                a.w[0] = lwork[L]; b.w[0] = lwork[R];
                bh3::combine(a, b, 24u, c);
                uint64_t z = 0; for (int w = 0; w < 7; ++w) z |= c.w[w];
                if (z == 0) {
                    const uint32_t si = atomicAdd(surv_count, 1u);
                    if (si < surv_cap) {
                        all_left[out_off + si] = lgi[L]; all_right[out_off + si] = lgi[R];
                        surv_slots[si] = si;
                    } else atomicAdd(&drops[2], 1u);
                }
            }
            oth = lchain[oth];
        }
    }
}

// ---- recover: walk a survivor's back-ref ancestry to its 32 leaf indices -----------
// Explicit-stack pre-order DFS from level 5 down to level 1, whose left/right entries ARE
// leaf indices. Right is pushed first so left pops first, which is what makes the leaves
// come out in tree order -- get that backwards and the indices are a valid-looking
// permutation that fails verification.
__global__ void recover(uint32_t nSurv, const uint32_t* __restrict__ surv_slots,
                        uint32_t capacity, const uint32_t* __restrict__ all_left,
                        const uint32_t* __restrict__ all_right, uint32_t* __restrict__ out) {
    const uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nSurv) return;
    uint32_t got = 0, lvl[8], slt[8]; int sp = 0;
    lvl[0] = 5u; slt[0] = surv_slots[i]; sp = 1;
    while (sp > 0 && got < 32u) {
        --sp;
        const uint32_t lv = lvl[sp], sl = slt[sp];
        const uint32_t off = (lv - 1u) * capacity;
        const uint32_t L = all_left[off + sl], R = all_right[off + sl];
        if (lv == 1u) {
            out[(size_t)i*32u + got] = L; ++got;
            if (got < 32u) { out[(size_t)i*32u + got] = R; ++got; }
        } else {
            lvl[sp] = lv - 1u; slt[sp] = R; ++sp;
            lvl[sp] = lv - 1u; slt[sp] = L; ++sp;
        }
    }
}

// ---- solver: persistent buffers, one solve() per (input, nonce) ---------------------
// Mirrors src/gpu/gpu_solver.cpp so the timing is like-for-like: buffers are allocated
// ONCE and reused, and solve() returns only CPU-VERIFIED solutions. Comparing a
// kernel-time figure against the OpenCL solver's end-to-end number would flatter this
// side, which is the whole reason for building this.
template<class T> static T* dalloc(size_t n) { void* p=nullptr; cudaMalloc(&p, n*sizeof(T)); return (T*)p; }

struct CudaSolver {
    static constexpr uint32_t elems    = 1u << 25;
    static constexpr uint32_t capacity = elems + elems/32;    // 34,603,008
    static constexpr uint32_t bb = 16, sm = 1, nb = 1u << bb;
    static constexpr uint32_t survCap = 1024;
    uint32_t cap = 0; size_t nslots = 0;
    uint64_t *elem[2] = {nullptr,nullptr}, *dpp = nullptr;
    uint32_t *counts[2] = {nullptr,nullptr}, *gictr=nullptr, *drops=nullptr;
    uint32_t *left=nullptr, *right=nullptr, *survSlots=nullptr, *survCount=nullptr, *dleaves=nullptr;

    bool init() {
        const uint32_t mean = capacity / nb;
        cap    = mean + (uint32_t)(8.0*std::sqrt((double)mean)) + 32u;
        nslots = (size_t)nb * cap;
        const uint32_t setStride[2] = { 10u, MXBM_R2_FULL ? 10u : 8u };
        elem[0] = dalloc<uint64_t>(nslots*setStride[0]);
        elem[1] = dalloc<uint64_t>(nslots*setStride[1]);
        counts[0] = dalloc<uint32_t>(nb); counts[1] = dalloc<uint32_t>(nb);
        gictr = dalloc<uint32_t>(1); drops = dalloc<uint32_t>(4);
        left  = dalloc<uint32_t>((size_t)5*capacity);
        right = dalloc<uint32_t>((size_t)5*capacity);
        survSlots = dalloc<uint32_t>(survCap); survCount = dalloc<uint32_t>(1);
        dleaves = dalloc<uint32_t>((size_t)survCap*32);
        dpp = dalloc<uint64_t>(4);
        #define CARVE(K) cudaFuncSetAttribute(K, cudaFuncAttributePreferredSharedMemoryCarveout, \
                                              cudaSharedmemCarveoutMaxShared)
        CARVE((fused_round<7,7,2,LM_SEED,424u,2u,1u,2u,2u,1u,2u>));
        CARVE((fused_round<7,7,2,LM_RD2,400u,4u,2u,4u,4u,2u,10u>));
        CARVE((fused_round<7,6,4,LM_EMIT,376u,6u,4u,2u,8u,10u,8u>));
        CARVE((fused_round<6,1,2,LM_USE,288u,9u,2u,0u,0u,8u,2u>));
        CARVE(terminal_round);
        #undef CARVE
        return elem[0] && elem[1] && left && right && dpp;
    }

    // Full solve: prePow -> pipeline -> recover -> CPU verify. Returns verified solutions.
    std::vector<std::array<uint8_t,104>> solve(const uint8_t input[32], const uint8_t nonce[8],
                                               uint32_t* survOut = nullptr, uint32_t* dropOut = nullptr) {
        std::vector<std::array<uint8_t,104>> out;
        uint64_t pp[4]; const uint8_t extra0[4] = {0,0,0,0};
        bh3::compute_prepow(input, 32, nonce, extra0, pp);
        cudaMemcpy(dpp, pp, 32, cudaMemcpyHostToDevice);
        cudaMemset(drops, 0, 16); cudaMemset(survCount, 0, 4);
        cudaMemset(counts[0], 0, (size_t)nb*4);

        entry_scatter<<<(elems+255)/256,256>>>(dpp, 0, elems, bb, cap, counts[0], elem[0], drops);
        int inSet = 0;
        #define ROUND(R, INW,OUTW,LEAFW,MODE, LOUT,PADN,SIN,SOUT,SBUILD, INSTR,OUTSTR)   \
            { const int o = inSet ^ 1;                                                   \
              cudaMemset(counts[o], 0, (size_t)nb*4); cudaMemset(gictr, 0, 4);           \
              fused_round<INW,OUTW,LEAFW,MODE,LOUT,PADN,SIN,SOUT,SBUILD,INSTR,OUTSTR>    \
                <<<nb << sm, kWG>>>(bb, sm, cap, cap, (uint32_t)((R)-1)*capacity,        \
                    counts[inSet], elem[inSet], counts[o], elem[o],                      \
                    left, right, gictr, drops, dpp);                                     \
              inSet = o; }
#if MXBM_R2_FULL
        ROUND(1, 7,7,2,LM_SEEDF,424u,2u,1u,2u,2u, 1u,10u)
        ROUND(2, 7,7,2,LM_RAW,  400u,4u,2u,4u,4u, 10u,10u)
#else
        ROUND(1, 7,7,2,LM_SEED, 424u,2u,1u,2u,2u, 1u,2u)
        ROUND(2, 7,7,2,LM_RD2,  400u,4u,2u,4u,4u, 2u,10u)
#endif
        ROUND(3, 7,6,4,LM_EMIT, 376u,6u,4u,2u,8u, 10u,8u)
        ROUND(4, 6,1,2,LM_USE,  288u,9u,2u,0u,0u, 8u,2u)
        #undef ROUND
        terminal_round<<<nb << sm, kWG>>>(bb, sm, cap, 4u*capacity, counts[inSet],
                                          elem[inSet], left, right, survSlots, survCount,
                                          survCap, drops);

        cudaError_t le = cudaGetLastError();
        if (le != cudaSuccess) {
            // A launch that fails (bad launch bounds, too many threads) otherwise just
            // yields zero survivors, which reads like a correctness bug in the kernels.
            printf("CUDA launch error: %s\n", cudaGetErrorString(le));
            return out;
        }
        uint32_t hs = 0, hd[4] = {0,0,0,0};
        cudaMemcpy(&hs, survCount, 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(hd, drops, 16, cudaMemcpyDeviceToHost);
        if (hs > survCap) hs = survCap;
        if (survOut) *survOut = hs;
        if (dropOut) *dropOut = hd[1] | hd[2] | hd[3];
        if (hs == 0) return out;

        recover<<<(hs+63)/64, 64>>>(hs, survSlots, capacity, left, right, dleaves);
        std::vector<uint32_t> hl((size_t)hs*32);
        cudaMemcpy(hl.data(), dleaves, (size_t)hs*32*4, cudaMemcpyDeviceToHost);
        for (uint32_t i = 0; i < hs; ++i) {
            std::array<uint8_t,104> sol{};
            bh3::pack_indices(&hl[(size_t)i*32], sol.data());
            if (bh3::is_valid_solution(input, 32, nonce, sol.data())) out.push_back(sol);
        }
        return out;
    }
};

int main(int argc, char** argv) {
    CudaSolver s;
    if (!s.init()) { printf("FAIL: allocation\n"); return 1; }
    printf("geometry : bb=%u sm=%u nb=%u cap=%u\n", s.bb, s.sm, s.nb, s.cap);
    printf("footprint: %.2f GiB\n",
           ((double)s.nslots*18*8 + (double)5*s.capacity*4*2)/(double)(1u<<30));

    // --- correctness gate: the KAT input has exactly 3 known goldens ---
    uint32_t surv=0, drop=0;
    auto sols = s.solve(kat::input32, kat::nonce0, &surv, &drop);
    int matched = 0;
    for (auto& sol : sols)
        for (int g = 0; g < 3; ++g)
            if (memcmp(sol.data(), kat::golden[g], 104) == 0) { ++matched; break; }
    const bool gate = (surv == 3) && (drop == 0) && (sols.size() == 3) && (matched == 3);
    printf("KAT      : survivors=%u verified=%zu goldens=%d/3 drops=%u -> %s\n",
           surv, sols.size(), matched, drop, gate ? "PASS" : "FAIL");
    if (!gate) return 1;

    // --- like-for-like timing: distinct nonces, end-to-end, verified solutions ---
    const int n = (argc > 1) ? atoi(argv[1]) : 20;
    uint8_t nonce[8]; memcpy(nonce, kat::nonce0, 8);
    std::vector<double> ms; size_t verified = 0;
    for (int i = 0; i < n; ++i) {
        nonce[7] = (uint8_t)(kat::nonce0[7] + i + 1);
        auto t0 = std::chrono::steady_clock::now();
        auto v = s.solve(kat::input32, nonce);
        cudaDeviceSynchronize();
        ms.push_back(std::chrono::duration<double,std::milli>(
                         std::chrono::steady_clock::now() - t0).count());
        verified += v.size();
    }
    std::sort(ms.begin(), ms.end());
    const double med = ms[ms.size()/2];
    const double spersolve = (double)verified / n;
    printf("end-to-end: %.1f ms median over %d distinct nonces (incl. recover + CPU verify)\n", med, n);
    printf("solutions : %.2f verified/solve  =>  %.1f sol/s\n", spersolve, spersolve*1000.0/med);
    return 0;
}
