#pragma once
// Entry, terminal and recover kernels for the CUDA row-bucket pipeline. Shared by the
// standalone bench (cuda/pipeline.cu) and the miner-facing solver
// (src/gpu/cuda_solver.cu) so there is exactly one copy of each.
#include "fused_round.cuh"

namespace mxbm { namespace cuda {

// ---- entry: seed + mix + scatter into round-1 buckets (8 B thin record) -------------
__global__ __launch_bounds__(256)
void entry_scatter(const uint64_t* __restrict__ pp4, uint32_t begin, uint32_t count,
                   uint32_t bucket_bits, uint32_t bucket_cap,
                   uint32_t* __restrict__ counts, uint64_t* __restrict__ belem,
                   uint32_t* __restrict__ drops) {
    const uint32_t g = blockIdx.x*blockDim.x + threadIdx.x;
    if (g >= count) return;
    // Body shared with fused_round's COTENANT path, so the co-resident entry cannot
    // drift from the standalone one.
    entry_body(begin + g, pp4, bucket_bits, bucket_cap, counts, belem, drops);
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
    __shared__ uint64_t lwork[kTCap];
    __shared__ uint32_t lgi[kTCap], llead[kTCap], lkey[kTCap], lchain[kTCap], tab[kTabSize];
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
        if (pos >= kTCap) { atomicAdd(&drops[1], 1u); continue; }
        const uint64_t meta = in_belem[d + 1];
        lwork[pos] = w0; lgi[pos] = (uint32_t)(meta >> 32);
        llead[pos] = (uint32_t)meta; lkey[pos] = key;
    }
    __syncthreads();
    const uint32_t total = gcount < kTCap ? gcount : kTCap;
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


}} // namespace mxbm::cuda
