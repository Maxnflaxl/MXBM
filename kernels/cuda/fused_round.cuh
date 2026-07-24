#pragma once
// CUDA port of kernels/opencl/lds.cl's FUSED_LDS. One block per (bucket, sub-mask):
// stage the group into shared memory, chain equal keys, walk the chains, and emit each
// child straight into the next round's buckets.
//
// The OpenCL macro's parameters become TEMPLATE parameters. They must stay compile-time:
// passing them as kernel arguments cost 27 ms on the OpenCL side, because a runtime Lmix
// makes apply_mix's tree loop runtime-bounded, which stops it unrolling, which turns
// t[word] into a dynamic index and spills an 8-element array to local memory once per
// emitted child. See "compile-time round constants" in docs/performance.md.
#include "bh3_records.cuh"

namespace mxbm { namespace cuda {

constexpr uint32_t kWG      = 256u;
constexpr uint32_t kFCap    = 384u;        // group cap; must cover the group-size TAIL
// 128, down from OpenCL's 512. Within a group the bucket fixes the key's high bits and
// the sub-mask its low ones, leaving exactly 24 - bucketBits - submaskBits = 7 varying
// bits -- and hk = (key >> submaskBits) & 127 selects precisely those, so 128 entries is
// a PERFECT hash here: every distinct key gets its own chain, and a chain holds exactly
// the elements sharing a key. 512 entries never addressed more than 128 of them.
//
// This holds for any geometry on the bucketBits + submaskBits = 17 line, which is the
// only line the group-size constraint allows (see docs/performance.md, row-bucket
// geometry). The 1.5 KB it frees against 512 is what buys round 3 a third block per SM.
constexpr uint32_t kTabSize = 128u;
constexpr uint32_t kEmpty   = 0xFFFFFFFFu;

enum LMode { LM_EMIT = 1, LM_USE = 2, LM_SEED = 3, LM_RD2 = 4 };

template<int INW, int OUTW, int LEAFW, int LMODE,
         uint32_t LOUT, uint32_t PADN, uint32_t SIN, uint32_t SOUT, uint32_t SBUILD,
         uint32_t INSTR, uint32_t OUTSTR>
__global__ __launch_bounds__(kWG)
void fused_round(uint32_t bucket_bits, uint32_t submask_bits,
                 uint32_t in_bucket_cap, uint32_t out_bucket_cap, uint32_t out_off,
                 const uint32_t* __restrict__ in_counts,
                 const uint64_t* __restrict__ in_belem,
                 uint32_t* __restrict__ out_counts,
                 uint64_t* __restrict__ out_belem,
                 uint32_t* __restrict__ all_left,
                 uint32_t* __restrict__ all_right,
                 uint32_t* __restrict__ gi_counter,
                 uint32_t* __restrict__ drops,
                 const uint64_t* __restrict__ pp4) {
    // `lead` is leaf 0 of the element's own prefix, so for every mode that stages real
    // leaves it is already in lleaf and a separate array is pure waste. LM_USE is the
    // exception: its leaf payload is the packed leftContrib, not leaves.
    constexpr bool kNeedLead = (LMODE == LM_USE);
    __shared__ uint64_t lwork[INW * kFCap];
    __shared__ uint32_t lgi[kFCap], lleaf[LEAFW * kFCap];
    __shared__ uint32_t llead[kNeedLead ? kFCap : 1];
    __shared__ uint32_t lkey[kFCap], lchain[kFCap], tab[kTabSize];
    __shared__ uint32_t gcount;
    auto lead_of = [&](uint32_t p) -> uint32_t {
        return kNeedLead ? llead[p] : lleaf[p * LEAFW + 0];
    };

    const uint32_t lId = threadIdx.x;
    const uint32_t submaskCount = 1u << submask_bits;
    const uint32_t bucket = blockIdx.x / submaskCount;
    const uint32_t mask   = blockIdx.x % submaskCount;

    if (lId == 0) gcount = 0;
    for (uint32_t i = lId; i < kTabSize; i += kWG) tab[i] = kEmpty;
    __syncthreads();

    uint32_t cnt = in_counts[bucket];
    if (cnt > in_bucket_cap) cnt = in_bucket_cap;
    const size_t base = (size_t)bucket * in_bucket_cap;

    // STAGE. Sub-mask filtered, so only ~1/2^submask_bits of the lanes survive: keep this
    // loop cheap and defer anything expensive to the all-lanes loop below.
    for (uint32_t p = lId; p < cnt; p += kWG) {
        const size_t d = (base + p) * INSTR;
        const uint64_t rec0 = in_belem[d];
        const uint32_t key = (uint32_t)(rec0 & 0xFFFFFFu);
        if ((key & (submaskCount - 1u)) != mask) continue;
        const uint32_t pos = atomicAdd(&gcount, 1u);
        if (pos >= kFCap) { atomicAdd(&drops[1], 1u); continue; }
        if constexpr (LMODE == LM_SEED) {
            lgi[pos] = (uint32_t)(rec0 >> 32); lkey[pos] = key;      // derive later
        } else if constexpr (LMODE == LM_RD2) {
            const uint64_t rec1 = in_belem[d + 1];
            const uint32_t li = pair_left(rec0), ri = pair_right(rec1);
            lgi[pos] = pair_gi(rec1); lkey[pos] = key;
            lleaf[pos*LEAFW + 0] = li; lleaf[pos*LEAFW + 1] = ri;   // leaf 0 IS the lead
        } else if constexpr (LMODE == LM_EMIT) {
            // 128-bit loads. The record stride is padded to an even number of u64, so
            // d*8 is 16 B aligned and cudaMalloc's base is 256 B aligned -- but the
            // compiler cannot prove either, and emitted 9 x LD.64 where 5 x LD.128 do.
            // Round 3 stalls 22% of its warp-active cycles on the MIO queue, which is
            // memory-INSTRUCTION issue rather than bandwidth, so instruction count is
            // the thing to cut here.
            static_assert(INSTR % 2 == 0, "vectorised path needs an even record stride");
            const ulonglong2* v = reinterpret_cast<const ulonglong2*>(in_belem + d);
            ulonglong2 q0 = v[0], q1 = v[1], q2 = v[2], q3 = v[3];
            lwork[pos*INW + 0] = q0.x; lwork[pos*INW + 1] = q0.y;
            lwork[pos*INW + 2] = q1.x; lwork[pos*INW + 3] = q1.y;
            lwork[pos*INW + 4] = q2.x; lwork[pos*INW + 5] = q2.y;
            lwork[pos*INW + 6] = q3.x;
            const uint64_t p0 = q3.y, p1 = in_belem[d + INW + 1];
            const uint32_t l0 = r3_l0(p0);
            lgi[pos] = r3_gi(p1); lkey[pos] = key;
            lleaf[pos*LEAFW + 0] = l0;          lleaf[pos*LEAFW + 1] = r3_l1(p0);
            lleaf[pos*LEAFW + 2] = r3_l2(p0,p1); lleaf[pos*LEAFW + 3] = r3_l3(p1);
        } else {   // LM_USE: work words, meta, then the leftContrib as the leaf payload
            static_assert(INSTR % 2 == 0, "vectorised path needs an even record stride");
            const ulonglong2* v = reinterpret_cast<const ulonglong2*>(in_belem + d);
            ulonglong2 q0 = v[0], q1 = v[1], q2 = v[2];
            lwork[pos*INW + 0] = q0.x; lwork[pos*INW + 1] = q0.y;
            lwork[pos*INW + 2] = q1.x; lwork[pos*INW + 3] = q1.y;
            lwork[pos*INW + 4] = q2.x; lwork[pos*INW + 5] = q2.y;
            const uint64_t meta = in_belem[d + INW];
            lgi[pos] = (uint32_t)(meta >> 32); llead[pos] = (uint32_t)meta; lkey[pos] = key;
            for (uint32_t i = 0; i < SIN; ++i)
                lleaf[pos*LEAFW + i] =
                    (uint32_t)(in_belem[d + INW + 1 + (i >> 1)] >> ((i & 1u) * 32u));
        }
    }
    __syncthreads();
    const uint32_t total = gcount < kFCap ? gcount : kFCap;

    // EXPAND + CHAIN. Every lane is active here, which is why the re-derivations live in
    // this loop and not the staging one (worth 8x their SIMD utilisation).
    for (uint32_t pos = lId; pos < total; pos += kWG) {
        if constexpr (LMODE == LM_SEED) {
            const uint32_t idx = lgi[pos];
            uint64_t pp[4] = { pp4[0], pp4[1], pp4[2], pp4[3] };
            bh3::Elem e; bh3::seed_element(pp, idx, e);
            uint32_t t1[1] = { idx };
            bh3::apply_mix(e, t1, 1u, 448u);
            for (int w = 0; w < INW; ++w) lwork[pos*INW + w] = e.w[w];
            lleaf[pos*LEAFW + 0] = idx;
        } else if constexpr (LMODE == LM_RD2) {
            uint64_t pp[4] = { pp4[0], pp4[1], pp4[2], pp4[3] };
            bh3::Elem e;
            rebuild_r2(pp, lleaf[pos*LEAFW + 0], lleaf[pos*LEAFW + 1], e);
            for (int w = 0; w < INW; ++w) lwork[pos*INW + w] = e.w[w];
        }
        const uint32_t hk = (lkey[pos] >> submask_bits) & (kTabSize - 1u);
        lchain[pos] = atomicExch(&tab[hk], pos);
    }
    __syncthreads();

    // WALK. Each slot follows its chain back to earlier same-hash slots and emits on an
    // equal full key, so every unordered colliding pair is emitted exactly once.
    for (uint32_t pos = lId; pos < total; pos += kWG) {
        const uint32_t key = lkey[pos];
        uint32_t oth = lchain[pos], walk = 0;
        while (oth != kEmpty) {
            if (++walk > 64u) { atomicAdd(&drops[3], 1u); break; }
            if (lkey[oth] == key) {
                const uint32_t la = lead_of(pos), lb = lead_of(oth);
                const uint32_t ga = lgi[pos],  gb = lgi[oth];
                uint32_t leftPos = pos, rightPos = oth;
                if (lb < la || (lb == la && gb < ga)) { leftPos = oth; rightPos = pos; }

                bh3::Elem a{}, b{}, c;
                for (int w = 0; w < INW; ++w) { a.w[w] = lwork[leftPos*INW+w];
                                                b.w[w] = lwork[rightPos*INW+w]; }
                bh3::combine(a, b, LOUT, c);

                uint32_t ctree[9]; uint64_t contribOut = 0;
                if constexpr (LMODE == LM_USE) {
                    for (int i = 0; i < 8; ++i) ctree[i] = 0u;
                    ctree[8] = lead_of(rightPos);
                    const uint64_t lc = ((uint64_t)lleaf[leftPos*LEAFW+1] << 32)
                                      |  (uint64_t)lleaf[leftPos*LEAFW+0];
                    bh3::apply_mix(c, ctree, PADN, LOUT);
                    c.w[0] = bh3::rotl64(bh3::rotl64(c.w[0], 40) + lc, 24);
                    ctree[0] = lead_of(leftPos);
                } else {
                    for (uint32_t i = 0; i < SBUILD; ++i)
                        ctree[i] = (i < SIN) ? lleaf[leftPos*LEAFW + i]
                                             : lleaf[rightPos*LEAFW + (i - SIN)];
                    bh3::apply_mix(c, ctree, PADN, LOUT);
                    if constexpr (LMODE == LM_EMIT) {
                        bh3::Elem z{};
                        bh3::apply_mix(z, ctree, 8u, 288u);
                        contribOut = bh3::rotl64(z.w[0], 40);
                    }
                }

                const uint32_t ckey = (uint32_t)(c.w[0] & 0xFFFFFFu);
                const uint32_t cb   = ckey >> (24u - bucket_bits);
                const uint32_t cpos = atomicAdd(&out_counts[cb], 1u);
                if (cpos < out_bucket_cap) {
                    const uint32_t cgi = atomicAdd(gi_counter, 1u);
                    const size_t od = ((size_t)cb * out_bucket_cap + cpos) * OUTSTR;
                    if constexpr (LMODE == LM_SEED) {
                        static_assert(OUTSTR % 2 == 0, "vectorised path needs an even stride");
                        *reinterpret_cast<ulonglong2*>(out_belem + od) =
                            make_ulonglong2(pair_w0(ckey, ctree[0]), pair_w1(ctree[1], cgi));
                    } else if constexpr (LMODE == LM_RD2) {
                        // Matching 128-bit stores; the padded stride makes od*8 16 B
                        // aligned. 5 x ST.128 instead of 9 x ST.64.
                        static_assert(OUTSTR % 2 == 0, "vectorised path needs an even stride");
                        ulonglong2* v = reinterpret_cast<ulonglong2*>(out_belem + od);
                        v[0] = make_ulonglong2(c.w[0], c.w[1]);
                        v[1] = make_ulonglong2(c.w[2], c.w[3]);
                        v[2] = make_ulonglong2(c.w[4], c.w[5]);
                        v[3] = make_ulonglong2(c.w[6], r3_p0(ctree[0], ctree[1], ctree[2]));
                        out_belem[od + OUTW + 1] = r3_p1(ctree[2], ctree[3], cgi);
                    } else if constexpr (LMODE == LM_EMIT) {
                        static_assert(OUTSTR % 2 == 0, "vectorised path needs an even stride");
                        ulonglong2* v = reinterpret_cast<ulonglong2*>(out_belem + od);
                        v[0] = make_ulonglong2(c.w[0], c.w[1]);
                        v[1] = make_ulonglong2(c.w[2], c.w[3]);
                        v[2] = make_ulonglong2(c.w[4], c.w[5]);
                        v[3] = make_ulonglong2(((uint64_t)cgi << 32) | (uint64_t)ctree[0],
                                               contribOut);
                    } else {
                        static_assert(OUTSTR % 2 == 0, "vectorised path needs an even stride");
                        *reinterpret_cast<ulonglong2*>(out_belem + od) =
                            make_ulonglong2(c.w[0],
                                            ((uint64_t)cgi << 32) | (uint64_t)ctree[0]);
                    }
                    all_left [out_off + cgi] = lgi[leftPos];
                    all_right[out_off + cgi] = lgi[rightPos];
                } else atomicAdd(&drops[2], 1u);
            }
            oth = lchain[oth];
        }
    }
}

}} // namespace mxbm::cuda
