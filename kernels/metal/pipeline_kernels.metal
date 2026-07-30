// Terminal and recover kernels for the Metal row-bucket pipeline.
// Ported from kernels/cuda/pipeline_kernels.cuh.
#include "bh3_records.metalh"

using namespace metal;

namespace mxbm { namespace metalk {

// terminal_round stages only word 0 + 4 meta u32 = 24 B/element, an eighth of a round
// block, so its group cap was never what constrained occupancy and must not be dragged
// down when the round cap is tuned. It keeps the full tail cap.
#ifndef MXBM_METAL_WG
#define MXBM_METAL_WG 256
#endif
constant constexpr uint32_t kWGT      = MXBM_METAL_WG;
constant constexpr uint32_t kTCap     = 384;
constant constexpr uint32_t kTabSizeT = 128;
constant constexpr uint32_t kEmptyT   = 0xFFFFFFFFu;
}}

// ---- terminal: combine at Lout(5)=24 and keep the all-zero survivors ----------------
// Only work word 0 is consulted: at Lout=24 combine zeroes c[1..6] and masks c[0] to 24
// bits, and the x[1]<<40 term contributes nothing below bit 40. See "the terminal
// round's dead work words" in docs/performance.md.
//
// Unlike the fused rounds this one CANNOT drop its key comparison: it stages on a
// separate lkey array and its table is not guaranteed perfect at every geometry.
kernel void terminal_round(
        constant uint32_t*     cfg        [[buffer(0)]],   // bb, sm, in_cap, out_off, surv_cap
        const device uint32_t* in_counts  [[buffer(1)]],
        const device uint64_t* in_belem   [[buffer(2)]],
        device uint32_t*       all_left   [[buffer(3)]],
        device uint32_t*       all_right  [[buffer(4)]],
        device uint32_t*       surv_slots [[buffer(5)]],
        device atomic_uint*    surv_count [[buffer(6)]],
        device atomic_uint*    drops      [[buffer(7)]],
        uint lId     [[thread_position_in_threadgroup]],
        uint groupId [[threadgroup_position_in_grid]])
{
    using namespace mxbm;
    using namespace mxbm::metalk;

    const uint32_t bucket_bits   = cfg[0];
    const uint32_t submask_bits  = cfg[1];
    const uint32_t in_bucket_cap = cfg[2];
    const uint32_t out_off       = cfg[3];
    const uint32_t surv_cap      = cfg[4];
    (void)bucket_bits;

    threadgroup uint64_t    lwork[kTCap];
    threadgroup uint32_t    lgi[kTCap], llead[kTCap], lkey[kTCap], lchain[kTCap];
    threadgroup atomic_uint tab[kTabSizeT];
    threadgroup atomic_uint gcount;

    const uint32_t submaskCount = 1u << submask_bits;
    const uint32_t bucket = groupId / submaskCount;
    const uint32_t mask   = groupId % submaskCount;

    if (lId == 0) atomic_store_explicit(&gcount, 0u, memory_order_relaxed);
    for (uint32_t i = lId; i < kTabSizeT; i += kWGT)
        atomic_store_explicit(&tab[i], kEmptyT, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint32_t cnt = in_counts[bucket];
    if (cnt > in_bucket_cap) cnt = in_bucket_cap;
    const size_t base = (size_t)bucket * in_bucket_cap;

    for (uint32_t p = lId; p < cnt; p += kWGT) {
        const size_t d = (base + p) * 2u;                 // the round-5 record stride
        const uint64_t w0 = in_belem[d];
        const uint32_t key = (uint32_t)(w0 & 0xFFFFFFu);
        if ((key & (submaskCount - 1u)) != mask) continue;
        const uint32_t pos = atomic_fetch_add_explicit(&gcount, 1u, memory_order_relaxed);
        if (pos >= kTCap) {
            atomic_fetch_add_explicit(&drops[1], 1u, memory_order_relaxed);
            continue;
        }
        const uint64_t meta = in_belem[d + 1];
        lwork[pos] = w0;
        lgi[pos]   = (uint32_t)(meta >> 32);
        llead[pos] = (uint32_t)meta;
        lkey[pos]  = key;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint32_t g = atomic_load_explicit(&gcount, memory_order_relaxed);
    const uint32_t total = g < kTCap ? g : kTCap;

    for (uint32_t pos = lId; pos < total; pos += kWGT) {
        const uint32_t hk = (lkey[pos] >> submask_bits) & (kTabSizeT - 1u);
        lchain[pos] = atomic_exchange_explicit(&tab[hk], pos, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint32_t pos = lId; pos < total; pos += kWGT) {
        const uint32_t key = lkey[pos];
        uint32_t oth = lchain[pos], walk = 0;
        while (oth != kEmptyT) {
            if (++walk > 64u) {
                atomic_fetch_add_explicit(&drops[3], 1u, memory_order_relaxed);
                break;
            }
            if (lkey[oth] == key) {
                const uint32_t la = llead[pos], lb = llead[oth];
                const uint32_t ga = lgi[pos],   gb = lgi[oth];
                uint32_t L = pos, R = oth;
                if (lb < la || (lb == la && gb < ga)) { L = oth; R = pos; }
                bh3::Elem a, b, c;
                for (int w = 0; w < 7; ++w) { a.w[w] = 0; b.w[w] = 0; }
                a.w[0] = lwork[L]; b.w[0] = lwork[R];
                bh3::combine(a, b, 24u, c);
                uint64_t z = 0;
                for (int w = 0; w < 7; ++w) z |= c.w[w];
                if (z == 0) {
                    const uint32_t si = atomic_fetch_add_explicit(surv_count, 1u,
                                                                  memory_order_relaxed);
                    if (si < surv_cap) {
                        all_left [out_off + si] = lgi[L];
                        all_right[out_off + si] = lgi[R];
                        surv_slots[si] = si;
                    } else {
                        atomic_fetch_add_explicit(&drops[2], 1u, memory_order_relaxed);
                    }
                }
            }
            oth = lchain[oth];
        }
    }
}

// ---- recover: walk a survivor's back-ref ancestry to its 32 leaf indices ------------
// Explicit-stack pre-order DFS from level 5 down to level 1, whose left/right entries ARE
// leaf indices. RIGHT IS PUSHED FIRST so left pops first, which is what makes the leaves
// come out in tree order -- get that backwards and the indices are a valid-looking
// permutation that fails verification.
kernel void recover(constant uint32_t&     nSurv      [[buffer(0)]],
                    constant uint32_t&     capacity   [[buffer(1)]],
                    const device uint32_t* surv_slots [[buffer(2)]],
                    const device uint32_t* all_left   [[buffer(3)]],
                    const device uint32_t* all_right  [[buffer(4)]],
                    device uint32_t*       out        [[buffer(5)]],
                    uint i [[thread_position_in_grid]])
{
    if (i >= nSurv) return;
    uint32_t got = 0, lvl[8], slt[8];
    int sp = 0;
    lvl[0] = 5u; slt[0] = surv_slots[i]; sp = 1;
    while (sp > 0 && got < 32u) {
        --sp;
        const uint32_t lv = lvl[sp], sl = slt[sp];
        const uint32_t off = (lv - 1u) * capacity;
        const uint32_t L = all_left[off + sl], R = all_right[off + sl];
        if (lv == 1u) {
            out[(size_t)i * 32u + got] = L; ++got;
            if (got < 32u) { out[(size_t)i * 32u + got] = R; ++got; }
        } else {
            lvl[sp] = lv - 1u; slt[sp] = R; ++sp;
            lvl[sp] = lv - 1u; slt[sp] = L; ++sp;
        }
    }
}
