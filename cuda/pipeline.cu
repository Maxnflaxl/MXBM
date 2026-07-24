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
#include <vector>
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
__global__ __launch_bounds__(256)
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

// ---- host driver -------------------------------------------------------------------
template<class T> static T* dalloc(size_t n) { void* p=nullptr; cudaMalloc(&p, n*sizeof(T)); return (T*)p; }

int main() {
    const uint32_t elems    = 1u << 25;
    const uint32_t capacity = elems + elems/32;              // 34,603,008
    const uint32_t bb = 16, sm = 1, nb = 1u << bb;
    const uint32_t mean = capacity / nb;
    const uint32_t cap  = mean + (uint32_t)(8.0*std::sqrt((double)mean)) + 32u;
    const size_t   nslots = (size_t)nb * cap;
    const uint32_t setStride[2] = { 9u, 8u };                // set0: r2/r4 out; set1: r1/r3 out
    const uint32_t survCap = 1024;

    printf("geometry : bb=%u sm=%u nb=%u cap=%u nslots=%zu\n", bb, sm, nb, cap, nslots);
    printf("footprint: sets %.2f GiB + backrefs %.2f GiB\n",
           (double)nslots*(setStride[0]+setStride[1])*8/(double)(1u<<30),
           (double)5*capacity*4*2/(double)(1u<<30));

    uint64_t *elem[2] = { dalloc<uint64_t>(nslots*setStride[0]), dalloc<uint64_t>(nslots*setStride[1]) };
    uint32_t *counts[2] = { dalloc<uint32_t>(nb), dalloc<uint32_t>(nb) };
    uint32_t *gictr = dalloc<uint32_t>(1), *drops = dalloc<uint32_t>(4);
    uint32_t *left = dalloc<uint32_t>((size_t)5*capacity), *right = dalloc<uint32_t>((size_t)5*capacity);
    uint32_t *survSlots = dalloc<uint32_t>(survCap), *survCount = dalloc<uint32_t>(1);
    uint64_t *dpp = dalloc<uint64_t>(4);
    if (!elem[0] || !elem[1] || !dpp || !left) { printf("FAIL: allocation\n"); return 1; }
    CK(cudaMemcpy(dpp, kat::prePow, 32, cudaMemcpyHostToDevice));

    // MXBM_CUDA_ITERS=1 for profiling: Nsight replays every launch several times to
    // collect counters, so three solves is three times the wait for no extra signal.
    const int iters = getenv("MXBM_CUDA_ITERS") ? atoi(getenv("MXBM_CUDA_ITERS")) : 3;
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    for (int iter = 0; iter < iters; ++iter) {
        CK(cudaMemset(drops, 0, 16)); CK(cudaMemset(survCount, 0, 4));
        CK(cudaMemset(counts[0], 0, (size_t)nb*4));
        cudaEventRecord(t0);

        entry_scatter<<<(elems+255)/256,256>>>(dpp, 0, elems, bb, cap, counts[0], elem[0], drops);

        int inSet = 0;
        #define ROUND(R, INW,OUTW,LEAFW,MODE, LOUT,PADN,SIN,SOUT,SBUILD, INSTR,OUTSTR)      \
            { const int outSet = inSet ^ 1;                                                 \
              CK(cudaMemset(counts[outSet], 0, (size_t)nb*4)); CK(cudaMemset(gictr, 0, 4)); \
              fused_round<INW,OUTW,LEAFW,MODE,LOUT,PADN,SIN,SOUT,SBUILD,INSTR,OUTSTR>       \
                <<<nb << sm, kWG>>>(bb, sm, cap, cap, (uint32_t)((R)-1)*capacity,           \
                    counts[inSet], elem[inSet], counts[outSet], elem[outSet],               \
                    left, right, gictr, drops, dpp);                                        \
              inSet = outSet; }
        ROUND(1, 7,7,2,LM_SEED, 424u,2u,1u,2u,2u, 1u,2u)
        ROUND(2, 7,7,2,LM_RD2,  400u,4u,2u,4u,4u, 2u,9u)
        ROUND(3, 7,6,4,LM_EMIT, 376u,6u,4u,2u,8u, 9u,8u)
        ROUND(4, 6,1,2,LM_USE,  288u,9u,2u,0u,0u, 8u,2u)
        #undef ROUND

        terminal_round<<<nb << sm, kWG>>>(bb, sm, cap, 4u*capacity, counts[inSet],
                                          elem[inSet], left, right, survSlots, survCount,
                                          survCap, drops);
        cudaEventRecord(t1); CK(cudaEventSynchronize(t1));
        CK(cudaGetLastError());

        uint32_t hs = 0, hd[4] = {0,0,0,0};
        CK(cudaMemcpy(&hs, survCount, 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hd, drops, 16, cudaMemcpyDeviceToHost));
        float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
        printf("solve %d : %6.1f ms  survivors=%u  drops{group=%u out=%u chain=%u}\n",
               iter, ms, hs, hd[1], hd[2], hd[3]);
        if (iter == iters - 1) {
            bool ok = (hs == 3) && (hd[1] == 0) && (hd[2] == 0) && (hd[3] == 0);
            // Full gate, matching the OpenCL path: recover each survivor's 32 indices,
            // pack them, and byte-compare against the KAT goldens. Survivor COUNT alone
            // would not catch a wrong tree order or a mis-packed index.
            if (ok && hs > 0) {
                uint32_t* dleaves = dalloc<uint32_t>((size_t)hs*32);
                recover<<<(hs+63)/64, 64>>>(hs, survSlots, capacity, left, right, dleaves);
                CK(cudaDeviceSynchronize());
                std::vector<uint32_t> hl((size_t)hs*32);
                CK(cudaMemcpy(hl.data(), dleaves, (size_t)hs*32*4, cudaMemcpyDeviceToHost));
                int matched = 0;
                for (uint32_t sidx = 0; sidx < hs; ++sidx) {
                    uint8_t sol[104] = {0};
                    bh3::pack_indices(&hl[(size_t)sidx*32], sol);
                    for (int g = 0; g < 3; ++g)
                        if (memcmp(sol, kat::golden[g], 104) == 0) { ++matched; break; }
                }
                printf("goldens  : %d of 3 recovered solutions match byte-for-byte\n", matched);
                ok = ok && (matched == 3);
            }
            printf("%s: cuda pipeline -- 3 KAT survivors, drop-free, goldens byte-identical\n",
                   ok ? "PASS" : "FAIL");
            return ok ? 0 : 1;
        }
    }
    return 1;
}
