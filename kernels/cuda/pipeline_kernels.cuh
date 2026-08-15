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
                   uint32_t* __restrict__ drops,
                   uint32_t* __restrict__ actr = nullptr,
                   uint32_t* __restrict__ atag = nullptr) {
    const uint32_t g = blockIdx.x*blockDim.x + threadIdx.x;
    if (g >= count) return;
    // Body shared with fused_round's COTENANT path, so the co-resident entry cannot
    // drift from the standalone one.
    entry_body(begin + g, pp4, bucket_bits, bucket_cap, counts, belem, drops, actr, atag);
}

// Between a producer round and its consumer: thread the pool entries onto per-bucket
// chains. ahead must be 0xFF-memset first; spill_total is the run-long positive
// control (a pool that never fills means the dense cap was never exceeded, and the
// experiment measured nothing).
__global__ void arena_link(const uint32_t* __restrict__ actr,
                           const uint32_t* __restrict__ atag,
                           uint32_t* __restrict__ ahead, uint32_t* __restrict__ anext,
                           uint32_t* __restrict__ spill_total) {
    const uint32_t n = min(*actr, kArenaCap);
    const uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i == 0 && spill_total) atomicAdd(spill_total, n);
    if (i < n) anext[i] = atomicExch(&ahead[atag[i]], i);
}

// ---- terminal: combine at Lout(5)=24 and keep the all-zero survivors ----------------
// Only work word 0 is consulted: at Lout=24 combine zeroes c[1..6] and masks c[0] to 24
// bits, and the x[1]<<40 term contributes nothing below bit 40. See "the terminal
// round's dead work words" in docs/performance.md. That is also why round 4's record is
// 8 B (t5_rec in bh3_records.cuh): 48 bits of word 0 can reach this test, of which the
// bucket address already carries bb, and nothing else here is read at all.
// An octo rung's round 4 is LM_RD4: it keeps its slot-indexed reference row, so it keeps
// gi and lead and the 16 B record with them. STRIDE is what tells the two apart.
constexpr uint32_t kTermStride = MXBM_R4_ROWS ? 2u : 1u;
template<bool ARENA = (MXBM_ARENA != 0), uint32_t STRIDE = kTermStride>
__global__ __launch_bounds__(kWG)
void terminal_round(uint32_t bucket_bits, uint32_t submask_bits, uint32_t in_bucket_cap,
                    uint32_t out_off,
                    const uint32_t* __restrict__ in_counts,
                    const uint64_t* __restrict__ in_belem,
                    uint32_t* __restrict__ all_left, uint32_t* __restrict__ all_right,
                    uint32_t* __restrict__ surv_slots, uint32_t* __restrict__ surv_count,
                    uint32_t surv_cap, uint32_t* __restrict__ drops,
                    const uint32_t* __restrict__ in_ahead = nullptr,
                    const uint32_t* __restrict__ in_anext = nullptr,
                    // Where each survivor's two round-4 parents sit in round 4's output,
                    // by SLOT rather than by gi: the replay reads their records to get
                    // the bucket hint, and a slot is what addresses one.
                    uint32_t* __restrict__ surv_l4 = nullptr) {
    __shared__ uint64_t lwork[kTCap];
    __shared__ uint32_t lchain[kTCap], tab[kTabSize], lslot[kTCap];
    // Only the reference-row arm orders left from right here; the 8 B record carries
    // neither field and the order is settled in recovery, off the lead replay_r4 reads
    // out of the left parent's record.
    __shared__ uint32_t lgi[STRIDE == 2 ? kTCap : 1];
    __shared__ uint32_t llead[STRIDE == 2 ? kTCap : 1];
    // Same perfect-hash argument as the fused rounds (MXBM_PERFECT_TAB in
    // fused_round.cuh): on the bb + sm = 17 line only 7 key bits vary inside a
    // group and the 128-entry table covers them, so same chain <=> same full key
    // and lkey plus the walk's compare are dead. The launcher's geometry check in
    // cuda_solver.cu guards this kernel too (same bb/sm).
    __shared__ uint32_t lkey[MXBM_PERFECT_TAB ? 1 : kTCap];
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
    // Same chain walk as the fused rounds'. Declared at 1 rather than 0 off the arena
    // rungs -- nvcc rejects a zero-sized shared variable -- and unused there, so ptxas
    // drops it exactly as it drops lkey above.
    [[maybe_unused]] const size_t nbcap_in = ((size_t)1u << bucket_bits) * in_bucket_cap;
    __shared__ uint16_t aidx[ARENA ? kAMax : 1];
    __shared__ uint32_t acnt_sh[1];
    uint32_t acnt = 0u;
    if constexpr (ARENA) {
        if (lId == 0) {
            uint32_t n_ = 0;
            if (in_ahead)
                for (uint32_t a_ = in_ahead[bucket]; a_ != 0xFFFFFFFFu; a_ = in_anext[a_]) {
                    if (n_ >= kAMax) { atomicAdd(&drops[1], 1u); break; }
                    aidx[n_++] = (uint16_t)a_;
                }
            acnt_sh[0] = n_;
        }
        __syncthreads();
        acnt = acnt_sh[0];
    }
    for (uint32_t p = lId; p < cnt + acnt; p += kWG) {
        size_t idx_ = base + p;
        if constexpr (ARENA)
            if (p >= cnt) idx_ = nbcap_in + aidx[p - cnt];
        const size_t d = idx_ * (size_t)STRIDE;
        const uint64_t rec = in_belem[d];
        const uint64_t w0  = STRIDE == 1 ? t5_work(rec) : rec;
        const uint32_t key = (uint32_t)(w0 & 0xFFFFFFu);
        if ((key & (submaskCount - 1u)) != mask) continue;
        const uint32_t pos = atomicAdd(&gcount, 1u);
        if (pos >= kTCap) { atomicAdd(&drops[1], 1u); continue; }
        lwork[pos] = w0; lslot[pos] = (uint32_t)idx_;
        if constexpr (STRIDE == 2) {
            const uint64_t meta = in_belem[d + 1];
            // gi is 26 bits; bit 63 carries the replay's 17th bucket bit, so mask it off.
            lgi[pos] = (uint32_t)(meta >> 32) & 0x3FFFFFFu;
            llead[pos] = (uint32_t)meta;
        }
        if constexpr (!MXBM_PERFECT_TAB) lkey[pos] = key;
    }
    __syncthreads();
    const uint32_t total = gcount < kTCap ? gcount : kTCap;
    for (uint32_t pos = lId; pos < total; pos += kWG) {
        // lwork[pos] is word 0, whose low 24 bits ARE the key.
        const uint32_t k_ = MXBM_PERFECT_TAB ? (uint32_t)(lwork[pos] & 0xFFFFFFu)
                                             : lkey[pos];
        const uint32_t hk = (k_ >> submask_bits) & (kTabSize - 1u);
        lchain[pos] = atomicExch(&tab[hk], pos);
    }
    __syncthreads();
    for (uint32_t pos = lId; pos < total; pos += kWG) {
        const uint32_t key = MXBM_PERFECT_TAB ? 0u : lkey[pos];
        uint32_t oth = lchain[pos], walk = 0;
        while (oth != kEmpty) {
            if (++walk > 64u) { atomicAdd(&drops[3], 1u); break; }
            if (MXBM_PERFECT_TAB || lkey[oth] == key) {
                uint32_t L = pos, R = oth;
                if constexpr (STRIDE == 2) {
                    const uint32_t la = llead[pos], lb = llead[oth];
                    const uint32_t ga = lgi[pos],  gb = lgi[oth];
                    if (lb < la || (lb == la && gb < ga)) { L = oth; R = pos; }
                }
                bh3::Elem a{}, b{}, c;
                a.w[0] = lwork[L]; b.w[0] = lwork[R];
                bh3::combine(a, b, 24u, c);
                uint64_t z = 0; for (int w = 0; w < 7; ++w) z |= c.w[w];
                if (z == 0) {
                    const uint32_t si = atomicAdd(surv_count, 1u);
                    if (si < surv_cap) {
                        // Null off the reference rows: the replay path names the two
                        // round-4 parents by slot instead, and rows 4 and 5 do not exist.
                        if constexpr (STRIDE == 2)
                            if (all_left) {
                                all_left[out_off + si] = lgi[L];
                                all_right[out_off + si] = lgi[R];
                            }
                        surv_slots[si] = si;
                        if (surv_l4) { surv_l4[2u*si] = lslot[L]; surv_l4[2u*si + 1u] = lslot[R]; }
                    } else atomicAdd(&drops[2], 1u);
                }
            }
            oth = lchain[oth];
        }
    }
}

// ---- replay_r4: rebuild one survivor-ancestor's round-4 pairing from its bucket -----
// Round 4's back-reference row names its parents for all 33.5 M children and is read for
// four. Instead the child's record carries the emitting block's input bucket, and this
// kernel re-runs round 4 over that ONE bucket of round 3's output, identifying the pair by
// the 48 bits of word 0 the terminal round consumes. Slot assignment is nondeterministic
// run to run and this does not depend on it: the pair is found by content, and left/right
// is the same (lead, gi) rule the round applied. Two blocks per survivor.
__global__ __launch_bounds__(256)
void replay_r4(uint32_t nSurv, const uint32_t* __restrict__ surv_l4,
               const uint64_t* __restrict__ r4_elem,
               const uint64_t* __restrict__ r3_elem, uint32_t r3_stride,
               const uint32_t* __restrict__ r3_counts, uint32_t r3_cap,
               uint32_t* __restrict__ out_slots, uint32_t* __restrict__ out_lead,
               uint32_t* __restrict__ drops, uint32_t bucket_bits = 0u,
               const uint32_t* __restrict__ r3_ahead = nullptr,
               const uint32_t* __restrict__ r3_anext = nullptr) {
    // The whole bucket, not one (bucket, sub-mask) group: elements in different sub-masks
    // have different keys and can never pair, so the hint stays 17 bits. kMaxN covers the
    // widest rung -- bb = 14 caps a bucket near 2400 -- and excess is counted, not dropped.
    constexpr uint32_t kMaxN = 4096;
    __shared__ uint32_t skey[kMaxN];
    __shared__ uint32_t apool[kAMax];
    __shared__ uint32_t ntot;
    if (blockIdx.x >= 2u*nSurv) return;
    const size_t d4 = (size_t)surv_l4[blockIdx.x] * kTermStride;
    const uint64_t rec = r4_elem[d4];
    const uint64_t target = kTermStride == 1 ? t5_ident(rec) : (rec & 0xFFFFFFFFFFFFull);
    const uint32_t b3 = kTermStride == 1
        ? t5_bucket(rec)
        : ((uint32_t)(rec >> 48) | (uint32_t)((r4_elem[d4 + 1] >> 63) << 16));
    uint32_t cnt = r3_counts[b3];
    if (cnt > r3_cap) cnt = r3_cap;
    if (cnt > kMaxN) { if (threadIdx.x == 0) atomicAdd(&drops[1], 1u); cnt = kMaxN; }
    const size_t base  = (size_t)b3 * r3_cap;
    const size_t nbcap = ((size_t)1u << bucket_bits) * r3_cap;
    auto slot_of = [&](uint32_t p) -> size_t {
        return p < cnt ? base + p : nbcap + apool[p - cnt];
    };
    if (threadIdx.x == 0) {
        // The bucket's overflow-pool members, which the round staged after the dense ones.
        uint32_t n_ = 0;
        if (r3_ahead)
            for (uint32_t a_ = r3_ahead[b3]; a_ != 0xFFFFFFFFu && n_ < kAMax;
                 a_ = r3_anext[a_])
                apool[n_++] = a_;
        ntot = cnt + n_;
    }
    __syncthreads();
    const uint32_t n = ntot;
    for (uint32_t p = threadIdx.x; p < n; p += blockDim.x)
        skey[p] = (uint32_t)(r3_elem[slot_of(p) * r3_stride] & 0xFFFFFFu);
    __syncthreads();
    for (uint32_t p = threadIdx.x; p < n; p += blockDim.x) {
        for (uint32_t q = p + 1u; q < n; ++q) {
            if (skey[q] != skey[p]) continue;
            const size_t dp = slot_of(p) * r3_stride, dq = slot_of(q) * r3_stride;
            const uint64_t mp = r3_elem[dp + 6], mq = r3_elem[dq + 6];
            const uint32_t la = (uint32_t)mp, lb = (uint32_t)mq;
            const uint32_t ga = (uint32_t)(mp >> 32), gb = (uint32_t)(mq >> 32);
            const bool swap = (lb < la) || (lb == la && gb < ga);
            const size_t dL = swap ? dq : dp, dR = swap ? dp : dq;
            bh3::Elem a{}, b{}, c;
            #pragma unroll
            for (int w = 0; w < 6; ++w) { a.w[w] = r3_elem[dL + w]; b.w[w] = r3_elem[dR + w]; }
            bh3::combine(a, b, 288u, c);
            uint32_t ctree[9] = {0,0,0,0,0,0,0,0,0};
            ctree[8] = (uint32_t)(swap ? mp : mq);          // lead of the right parent
            bh3::apply_mix(c, ctree, 9u, 288u);
            c.w[0] = bh3::rotl64(bh3::rotl64(c.w[0], 40) + r3_elem[dL + 7], 24);
            const uint64_t got = kTermStride == 1 ? t5_ident_w0(c.w[0])
                                                  : (c.w[0] & 0xFFFFFFFFFFFFull);
            if (got == target) {
                out_slots[2u*blockIdx.x]      = (uint32_t)(dL / r3_stride);
                out_slots[2u*blockIdx.x + 1u] = (uint32_t)(dR / r3_stride);
                // This element's own lead, which is its left parent's: the terminal round
                // no longer stores one, so the survivor's two halves are ordered here.
                out_lead[blockIdx.x] = (uint32_t)(swap ? mq : mp);
            }
        }
    }
}

// ---- replay_r3: the same trick one level down, on the implicit-bits record -----------
// Round 3's output record spends a work word on nothing: at Lout(4) = 288 the shift drops
// everything above bit 311, so word 5 cannot reach round 4's output ("round 3 stores a
// work word round 4 never reads" in docs/performance.md). The emitting block's input
// bucket rides there, and this re-runs round 3 over that one bucket of round 2's output,
// matching a candidate child on work words 0..4. That deletes rows 1, 2 and 3 together:
// once the survivor's eight round-2 ancestors are known by slot, their records carry the
// four leaves each, in tree order.
// One block per (survivor, round-3 ancestor); the grid is four blocks per survivor.
__global__ __launch_bounds__(256)
void replay_r3(uint32_t nSurv, const uint32_t* __restrict__ l3_slots,
               const uint64_t* __restrict__ r3_elem, uint32_t r3_stride,
               const uint64_t* __restrict__ r2_elem, uint32_t r2_stride, uint32_t impb,
               const uint32_t* __restrict__ r2_counts, uint32_t r2_cap,
               uint32_t* __restrict__ out_slots, uint32_t* __restrict__ drops,
               uint32_t bucket_bits,
               const uint32_t* __restrict__ r2_ahead = nullptr,
               const uint32_t* __restrict__ r2_anext = nullptr) {
    constexpr uint32_t kMaxN = 4096;
    __shared__ uint32_t skey[kMaxN];
    __shared__ uint32_t apool[kAMax];
    __shared__ uint32_t ntot;
    if (blockIdx.x >= 4u*nSurv) return;
    const uint32_t s3 = l3_slots[blockIdx.x];
    if (s3 == 0xFFFFFFFFu) return;                  // the round-4 replay found no pair
    const size_t d3 = (size_t)s3 * r3_stride;
    uint64_t tw[5];
    #pragma unroll
    for (int w = 0; w < 5; ++w) tw[w] = r3_elem[d3 + w];
    const uint32_t b2 = (uint32_t)r3_elem[d3 + 5];
    // The record keeps the 24 - impb key bits the bucket address does not carry, so
    // inside one bucket those bits alone decide whether two full keys are equal.
    const uint32_t impKB = 24u - impb;
    const uint32_t impM  = (1u << impKB) - 1u;
    uint32_t cnt = r2_counts[b2];
    if (cnt > r2_cap) cnt = r2_cap;
    if (cnt > kMaxN) { if (threadIdx.x == 0) atomicAdd(&drops[1], 1u); cnt = kMaxN; }
    const size_t base  = (size_t)b2 * r2_cap;
    const size_t nbcap = ((size_t)1u << bucket_bits) * r2_cap;
    auto slot_of = [&](uint32_t p) -> size_t {
        return p < cnt ? base + p : nbcap + apool[p - cnt];
    };
    if (threadIdx.x == 0) {
        uint32_t n_ = 0;
        if (r2_ahead)
            for (uint32_t a_ = r2_ahead[b2]; a_ != 0xFFFFFFFFu && n_ < kAMax;
                 a_ = r2_anext[a_])
                apool[n_++] = a_;
        ntot = cnt + n_;
    }
    __syncthreads();
    const uint32_t n = ntot;
    for (uint32_t p = threadIdx.x; p < n; p += blockDim.x)
        skey[p] = (uint32_t)(r2_elem[slot_of(p) * r2_stride]) & impM;
    __syncthreads();
    // Unpack one record: seven work words with the block's bucket put back into word 0,
    // four leaves and gi. Inverse of the implicit-bits pack in fused_round's LM_RD2 emit.
    auto load = [&](size_t d, uint64_t w[7], uint32_t l[4], uint32_t& gi) {
        const ulonglong2* v = reinterpret_cast<const ulonglong2*>(r2_elem + d);
        const ulonglong2 q0 = v[0], q1 = v[1], q2 = v[2], q3 = v[3];
        w[0] = (q0.x & impM) | ((uint64_t)b2 << impKB)
             | (((q0.x >> impKB) & 0xFFFFFFFFFFull) << 24);
        w[1] = (q0.x >> (64u - impb)) | (q0.y << impb);
        w[2] = (q0.y >> (64u - impb)) | (q1.x << impb);
        w[3] = (q1.x >> (64u - impb)) | (q1.y << impb);
        w[4] = (q1.y >> (64u - impb)) | (q2.x << impb);
        w[5] = (q2.x >> (64u - impb)) | (q2.y << impb);
        w[6] = (q2.y >> (64u - impb)) & 0xFFFFull;
        l[0] = r3_l0(q3.x); l[1] = r3_l1(q3.x);
        l[2] = r3_l2(q3.x, q3.y); l[3] = r3_l3(q3.y);
        gi   = r3_gi(q3.y);
    };
    for (uint32_t p = threadIdx.x; p < n; p += blockDim.x) {
        for (uint32_t q = p + 1u; q < n; ++q) {
            if (skey[q] != skey[p]) continue;
            uint64_t wp[7], wq[7]; uint32_t lp[4], lq[4], gp, gq;
            const size_t dp = slot_of(p) * r2_stride, dq = slot_of(q) * r2_stride;
            load(dp, wp, lp, gp); load(dq, wq, lq, gq);
            // Leaf 0 IS the lead, and the tiebreak is the round's own.
            const bool swap = (lq[0] < lp[0]) || (lq[0] == lp[0] && gq < gp);
            const uint64_t* wL = swap ? wq : wp; const uint32_t* lL = swap ? lq : lp;
            const uint64_t* wR = swap ? wp : wq; const uint32_t* lR = swap ? lp : lq;
            bh3::Elem a{}, b{}, c;
            #pragma unroll
            for (int w = 0; w < 7; ++w) { a.w[w] = wL[w]; b.w[w] = wR[w]; }
            bh3::combine(a, b, 376u, c);
            uint32_t ctree[8];
            #pragma unroll
            for (int i = 0; i < 4; ++i) { ctree[i] = lL[i]; ctree[i + 4] = lR[i]; }
            bh3::apply_mix(c, ctree, 6u, 376u);
            bool hit = true;
            #pragma unroll
            for (int w = 0; w < 5; ++w) if (c.w[w] != tw[w]) hit = false;
            if (hit) {
                out_slots[2u*blockIdx.x]      = (uint32_t)((swap ? dq : dp) / r2_stride);
                out_slots[2u*blockIdx.x + 1u] = (uint32_t)((swap ? dp : dq) / r2_stride);
            }
        }
    }
}

// ---- recover: walk a survivor's back-ref ancestry to its 32 leaf indices -----------
// Explicit-stack pre-order DFS from level 5 down to level 1, whose left/right entries ARE
// leaf indices. Right is pushed first so left pops first, which is what makes the leaves
// come out in tree order -- get that backwards and the indices are a valid-looking
// permutation that fails verification.
// OCTO: only rows 4 and 5 exist, row 4 at offset 0 and row 5 at `capacity`. Row 4 names
// its parents by SLOT in round 3's output set, and each of those records carries the eight
// seed indices below it -- in the same left-to-right order this walk would have produced,
// because round 3 built its leaf tree and its references from the same leftPos/rightPos.
// So the walk is two levels deep instead of five, and rows 1-3 are never allocated.
template<bool OCTO = false>
__global__ void recover(uint32_t nSurv, const uint32_t* __restrict__ surv_slots,
                        uint32_t capacity, const uint32_t* __restrict__ all_left,
                        const uint32_t* __restrict__ all_right, uint32_t* __restrict__ out,
                        const uint64_t* __restrict__ r3_elem = nullptr,
                        uint32_t r3_stride = 0) {
    const uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nSurv) return;
    if constexpr (OCTO) {
        const uint32_t s5 = surv_slots[i];
        const uint32_t g4[2] = { all_left[capacity + s5], all_right[capacity + s5] };
        uint32_t got = 0;
        for (int k = 0; k < 2; ++k) {
            const uint32_t sl[2] = { all_left[g4[k]], all_right[g4[k]] };
            for (int j = 0; j < 2; ++j) {
                const uint64_t* r = r3_elem + (size_t)sl[j] * r3_stride;
                uint32_t l[8];
                octo_leaves(r[0], r[1], r[2], r[3], l);
                for (int t = 0; t < 8 && got < 32u; ++t)
                    out[(size_t)i*32u + got++] = l[t];
            }
        }
        return;
    }
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

// The same walk entered two levels down: replay_r4 has already produced the survivor's
// four round-3 ancestors as SLOTS, so rows 4 and 5 are not read. A round-3 record carries
// its own gi in the meta word, which is what rows 1-3 are still indexed by.
__global__ void recover_from_l3(uint32_t nSurv, uint32_t capacity,
                                const uint32_t* __restrict__ all_left,
                                const uint32_t* __restrict__ all_right,
                                const uint32_t* __restrict__ l3_slots,
                                const uint32_t* __restrict__ l4_lead,
                                const uint64_t* __restrict__ r3_elem, uint32_t r3_stride,
                                uint32_t* __restrict__ out) {
    const uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nSurv) return;
    // Equal leads mean a shared seed index, so that pair can never be a valid solution
    // and the order it is given does not matter.
    const uint32_t flip = (l4_lead && l4_lead[2u*i + 1u] < l4_lead[2u*i]) ? 2u : 0u;
    uint32_t got = 0, lvl[8], slt[8];
    for (uint32_t k0 = 0; k0 < 4u; ++k0) {
        const uint32_t k = (k0 ^ flip);
        const uint32_t sr = l3_slots[i*4u + k];
        // A replay that found no pair leaves its sentinel here. Emitting zeros makes the
        // candidate fail verification, which is what a lost solution should look like --
        // dereferencing the sentinel would take the context down instead.
        if (sr == 0xFFFFFFFFu) {
            while (got < 8u*(k0 + 1u) && got < 32u) out[(size_t)i*32u + got++] = 0u;
            continue;
        }
        const size_t s = (size_t)sr;
        int sp = 1;
        lvl[0] = 3u; slt[0] = (uint32_t)(r3_elem[s * r3_stride + 6] >> 32);
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
}


// Recovery with no reference row left to read: replay_r3 has named the survivor's eight
// round-2 ancestors by slot, and a round-2 output record carries the four seed indices
// below it in the order this walk would have produced -- ctree[0..1] came from the left
// parent and [2..3] from the right.
__global__ void recover_from_l2(uint32_t nSurv,
                                const uint32_t* __restrict__ l2_slots,
                                const uint32_t* __restrict__ l4_lead,
                                const uint64_t* __restrict__ r2_elem, uint32_t r2_stride,
                                uint32_t* __restrict__ out) {
    const uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nSurv) return;
    // Equal leads mean a shared seed index, so that pair can never be a valid solution
    // and the order it is given does not matter.
    const uint32_t flip = (l4_lead && l4_lead[2u*i + 1u] < l4_lead[2u*i]) ? 4u : 0u;
    for (uint32_t k0 = 0; k0 < 8u; ++k0) {
        const uint32_t sr = l2_slots[i*8u + (k0 ^ flip)];
        const size_t o = (size_t)i*32u + k0*4u;
        if (sr == 0xFFFFFFFFu) {          // a replay that found no pair; fails the CPU gate
            for (int t = 0; t < 4; ++t) out[o + t] = 0u;
            continue;
        }
        const size_t d = (size_t)sr * r2_stride;
        const uint64_t p0 = r2_elem[d + 6], p1 = r2_elem[d + 7];
        out[o + 0] = r3_l0(p0);       out[o + 1] = r3_l1(p0);
        out[o + 2] = r3_l2(p0, p1);   out[o + 3] = r3_l3(p1);
    }
}

}} // namespace mxbm::cuda
