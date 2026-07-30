// The fused row-bucket round, ported from kernels/cuda/fused_round.cuh.
//
// One threadgroup per (bucket, sub-mask): stage the group into threadgroup memory,
// chain equal keys through a perfect hash table, walk the chains, and emit each
// child straight into the next round's buckets.
//
// SHIPPING CONFIGURATION ONLY. The CUDA source carries ~20 experiment macros --
// MXBM_ABL_* (which deliberately produce WRONG results for attribution), MXBM_OCC,
// MXBM_ROUND_REPS, MXBM_STCS/WARPAGG/CPASYNC/NARROW6 (all settled and left off),
// plus the COTENANT and SUBPASS paths. Every one of those was tuned against Ada
// occupancy numbers that do not transfer, so carrying them here would be carrying
// dead code whose settings are meaningless until re-measured on Apple. What is
// ported is the path that actually runs: PERFECT_TAB on, SPILL on, plain stores.
//
// The template parameters must stay COMPILE-TIME. Passing them as kernel arguments
// cost 27 ms on the OpenCL side: a runtime Lmix makes apply_mix's tree loop
// runtime-bounded, which stops it unrolling, which turns t[word] into a dynamic
// index and spills an 8-element array to thread-local memory once per emitted
// child. See "compile-time round constants" in docs/performance.md.
#include "bh3_records.metalh"

using namespace metal;

namespace mxbm { namespace metalk {

// Threadgroup size. The staged group is mean_bucket / 2^submaskBits = 264 elements,
// so a 256-thread group runs a second loop iteration with 8 of 256 lanes busy. Kept
// at CUDA's value as a STARTING POINT; Apple's optimum is unmeasured.
constant constexpr uint32_t kWG = 256;

// Group cap -- the single number that sets threadgroup memory and therefore occupancy.
// Ada wants 320 (and 288 for round 1). Carried over as a starting value only.
constant constexpr uint32_t kFCap   = 320;
constant constexpr uint32_t kR1FCap = 288;

// 128 entries is a PERFECT hash on the bucket_bits + submask_bits = 17 line: the
// bucket fixes the key's high bits and the sub-mask its low ones, leaving exactly
// 24 - bb - sm = 7 varying bits, and hk = (key >> sm) & 127 selects precisely those.
// So two elements share a chain if and only if they share the full 24-bit key, which
// is what lets the walk skip the key comparison entirely (PERFECT_TAB).
//
// UNSAFE if the geometry ever makes kTabSize < 2^(24-bb-sm): distinct keys would then
// share a chain and be combined as if equal. The launcher must check; the goldens catch it.
constant constexpr uint32_t kTabSize = 128;
constant constexpr uint32_t kEmpty   = 0xFFFFFFFFu;

// 8-way max split. The level must be chosen BEFORE any part is walked: escalating
// after a walk would re-emit everything the coarser parts already emitted, which
// inflates the next round's bucket counts and evicts real elements -- it shows up as
// LOST goldens with a zero drop counter, not as an obvious failure.
constant constexpr uint32_t kMaxSpill = 3;

enum LMode { LM_EMIT = 1, LM_USE = 2, LM_SEED = 3, LM_RD2 = 4, LM_RD3 = 5 };

// Dense per-round id allocation, one atomic per SIMD group instead of one per lane.
// Every emit needs a unique id and they all come from a single u32, so a fully active
// SIMD group would otherwise serialise into 32 round-trips on one address.
//
// Unlike CUDA -- where ptxas generates this idiom automatically from a plain
// atomicAdd (verified in the SASS, see fused_round.cuh) -- Apple's compiler is NOT
// known to do so, so it is written out. simd_prefix_exclusive_sum over an active
// mask gives each lane its rank; the first active lane takes the whole group's worth
// in one atomic and broadcasts the base. gi stays a permutation of [0, count) and the
// ids are CONSECUTIVE within a group, which is strictly better for the back-ref
// writes that index on them.
inline uint32_t gi_alloc(device atomic_uint* ctr) {
    const uint32_t rank  = simd_prefix_exclusive_sum(1u);
    const uint32_t total = simd_sum(1u);
    uint32_t base = 0;
    if (rank == 0)
        base = atomic_fetch_add_explicit(ctr, total, memory_order_relaxed);
    base = simd_broadcast_first(base);
    return base + rank;
}

}} // namespace mxbm::metalk

// ---------------------------------------------------------------------------------
// The round body. Instantiated per round below with [[host_name]], because MSL kernel
// entry points cannot themselves be templates.
// ---------------------------------------------------------------------------------
template<int INW, int OUTW, int LEAFW, int LMODE,
         uint32_t LOUT, uint32_t PADN, uint32_t SIN, uint32_t SOUT, uint32_t SBUILD,
         uint32_t INSTR, uint32_t OUTSTR, uint32_t FCAP>
void fused_round_body(
        uint32_t bucket_bits, uint32_t submask_bits,
        uint32_t in_bucket_cap, uint32_t out_bucket_cap, uint32_t out_off,
        const device uint32_t* in_counts,
        const device uint64_t* in_belem,
        device atomic_uint*    out_counts,
        device uint64_t*       out_belem,
        device uint32_t*       all_left,
        device uint32_t*       all_right,
        device atomic_uint*    gi_counter,
        device atomic_uint*    drops,
        const device uint64_t* pp4,
        threadgroup uint64_t*  lwork,
        threadgroup uint32_t*  lgi,
        threadgroup uint32_t*  lleaf,
        threadgroup uint32_t*  llead,
        threadgroup uint32_t*  lchain,
        threadgroup atomic_uint* tab,
        threadgroup atomic_uint& gcount,
        threadgroup atomic_uint* cnt8,
        uint lId, uint groupId)
{
    using namespace mxbm;
    using namespace mxbm::metalk;

    // `lead` is leaf 0 of the element's own prefix, so for every mode that stages real
    // leaves it is already in lleaf and a separate array is pure waste. LM_USE is the
    // exception: its leaf payload is the packed leftContrib, not leaves.
    constexpr bool kNeedLead = (LMODE == LM_USE);
    // Round 1's gi IS its leaf: both are the seed index, written from the same
    // rec0 >> 32. So LM_SEED keeps only one copy and reads gi through lleaf[0].
    constexpr bool kGiIsLead = (LMODE == LM_SEED);

    const uint32_t submaskCount = 1u << submask_bits;
    const uint32_t bucket = groupId / submaskCount;
    const uint32_t mask_s = groupId % submaskCount;

    // Round 2's record is 9 u64. A 9-u64 stride puts every odd slot on an 8 B boundary,
    // which the 128-bit accesses cannot use; padding to 10 bought alignment at 8 B per
    // element of traffic in BOTH directions (r2 writes, r3 reads back) -- 537 MB per
    // solve, 4.0% of all traffic, carrying nothing. So the 9th word lives in its own
    // plane after the records and the record stride stays 8.
    const size_t side_in  = ((size_t)1u << bucket_bits) * in_bucket_cap  * INSTR;
    const size_t side_out = ((size_t)1u << bucket_bits) * out_bucket_cap * OUTSTR;

    uint32_t cnt = in_counts[bucket];
    if (cnt > in_bucket_cap) cnt = in_bucket_cap;
    const size_t base = (size_t)bucket * in_bucket_cap;

    uint32_t xb = 0, xp = 0, nparts = 1;   // split level, current part, part count
    bool counted = false;                  // the level is chosen once, never escalated

    while (xp < nparts) {
        const uint32_t sbits = submask_bits + xb;
        const uint32_t mask  = mask_s | (xp << submask_bits);
        if (lId == 0) atomic_store_explicit(&gcount, 0u, memory_order_relaxed);
        for (uint32_t i = lId; i < kTabSize; i += kWG)
            atomic_store_explicit(&tab[i], kEmpty, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- STAGE. Sub-mask filtered, so only ~1/2^submask_bits of the lanes
        // survive: keep this loop cheap and defer anything expensive to the
        // all-lanes loop below (worth 8x their SIMD utilisation).
        for (uint32_t p = lId; p < cnt; p += kWG) {
            const size_t d = (base + p) * INSTR;
            const uint64_t rec0 = in_belem[d];
            const uint32_t key = (uint32_t)(rec0 & 0xFFFFFFu);
            if ((key & ((1u << sbits) - 1u)) != mask) continue;
            // Keep counting past FCAP: gcount is what decides whether to split.
            const uint32_t pos = atomic_fetch_add_explicit(&gcount, 1u, memory_order_relaxed);
            if (pos >= FCAP) continue;

            if (LMODE == LM_SEED) {
                lleaf[pos * LEAFW + 0] = (uint32_t)(rec0 >> 32);      // derive later
            } else if (LMODE == LM_RD2) {
                const uint64_t rec1 = in_belem[d + 1];
                lgi[pos] = pair_gi(rec1);
                lleaf[pos * LEAFW + 0] = pair_left(rec0);             // leaf 0 IS the lead
                lleaf[pos * LEAFW + 1] = pair_right(rec1);
            } else if (LMODE == LM_RD3) {
                // 24 B quad record: three scalar loads, against LM_EMIT's five wide ones.
                const uint64_t w1 = in_belem[d + 1], w2 = in_belem[d + 2];
                lgi[pos] = quad_gi(w2);
                lleaf[pos * LEAFW + 0] = quad_l0(rec0);               // leaf 0 IS the lead
                lleaf[pos * LEAFW + 1] = quad_l1(w1);
                lleaf[pos * LEAFW + 2] = quad_l2(w1);
                lleaf[pos * LEAFW + 3] = quad_l3(w2);
            } else if (LMODE == LM_EMIT) {
                for (int w = 0; w < INW; ++w) lwork[pos * INW + w] = in_belem[d + w];
                const uint64_t p0 = in_belem[d + INW];
                const uint64_t p1 = (INSTR >= INW + 2) ? in_belem[d + INW + 1]
                                                       : in_belem[side_in + base + p];
                lgi[pos] = r3_gi(p1);
                lleaf[pos * LEAFW + 0] = r3_l0(p0);
                lleaf[pos * LEAFW + 1] = r3_l1(p0);
                lleaf[pos * LEAFW + 2] = r3_l2(p0, p1);
                lleaf[pos * LEAFW + 3] = r3_l3(p1);
            } else {   // LM_USE: work words, meta, then leftContrib as the leaf payload
                for (int w = 0; w < INW; ++w) lwork[pos * INW + w] = in_belem[d + w];
                const uint64_t meta = in_belem[d + INW];
                lgi[pos]   = (uint32_t)(meta >> 32);
                llead[pos] = (uint32_t)meta;
                for (uint32_t i = 0; i < SIN; ++i)
                    lleaf[pos * LEAFW + i] =
                        (uint32_t)(in_belem[d + INW + 1 + (i >> 1)] >> ((i & 1u) * 32u));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- SPILL. When a group overflows FCAP, SPLIT it on one more key bit and
        // redo rather than dropping the overflow. Exactly safe: two elements whose keys
        // differ in the added bit can never collide, so no pair is lost and none is
        // emitted twice. Self-limiting -- the split halves the group and overflow is
        // rare, so the redone work is paid by a few per cent of groups.
        uint32_t gtotal = atomic_load_explicit(&gcount, memory_order_relaxed);
        if (gtotal > FCAP && !counted) {
            // One word-0 pass counts all 8 possible sub-parts at once, so the split
            // level is picked from real sizes rather than discovered by trial.
            if (lId < 8) atomic_store_explicit(&cnt8[lId], 0u, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint32_t p = lId; p < cnt; p += kWG) {
                const uint32_t k2 = (uint32_t)(in_belem[(base + p) * INSTR] & 0xFFu);
                if ((k2 & (submaskCount - 1u)) != mask_s) continue;
                atomic_fetch_add_explicit(&cnt8[(k2 >> submask_bits) & 7u], 1u,
                                          memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint32_t lv = 1;
            for (; lv <= kMaxSpill; ++lv) {
                uint32_t mx = 0;
                for (uint32_t q = 0; q < (1u << lv); ++q) {
                    uint32_t sz = 0;
                    for (uint32_t i = 0; i < 8u; ++i)
                        if ((i & ((1u << lv) - 1u)) == q)
                            sz += atomic_load_explicit(&cnt8[i], memory_order_relaxed);
                    if (sz > mx) mx = sz;
                }
                if (mx <= FCAP) break;
            }
            xb = lv > kMaxSpill ? kMaxSpill : lv;
            nparts = 1u << xb; xp = 0; counted = true;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            continue;
        }
        // Only reachable if the chosen split still overflows, which the count rules out.
        if (gtotal > FCAP && lId == 0)
            atomic_fetch_add_explicit(&drops[1], gtotal - FCAP, memory_order_relaxed);

        const uint32_t total = gtotal < FCAP ? gtotal : FCAP;

        // ---- EXPAND + CHAIN. Every lane is active here, which is why the
        // re-derivations live in this loop and not the staging one. Round 3's rebuild
        // measured +23.5 ms in the filtered loop against +0.4 ms here.
        for (uint32_t pos = lId; pos < total; pos += kWG) {
            uint64_t pp[4] = { pp4[0], pp4[1], pp4[2], pp4[3] };
            if (LMODE == LM_SEED) {
                const uint32_t idx = lleaf[pos * LEAFW + 0];
                bh3::Elem e;
                bh3::seed_element(pp, idx, e);
                uint32_t t1[1] = { idx };
                bh3::apply_mix(e, t1, 1u, 448u);
                for (int w = 0; w < INW; ++w) lwork[pos * INW + w] = e.w[w];
            } else if (LMODE == LM_RD2) {
                bh3::Elem e;
                rebuild_r2(pp, lleaf[pos * LEAFW + 0], lleaf[pos * LEAFW + 1], e);
                for (int w = 0; w < INW; ++w) lwork[pos * INW + w] = e.w[w];
            } else if (LMODE == LM_RD3) {
                const uint32_t l[4] = { lleaf[pos * LEAFW + 0], lleaf[pos * LEAFW + 1],
                                        lleaf[pos * LEAFW + 2], lleaf[pos * LEAFW + 3] };
                bh3::Elem e;
                rebuild_r3(pp, l, e);
                for (int w = 0; w < INW; ++w) lwork[pos * INW + w] = e.w[w];
            }
            // lwork[pos*INW] is word 0, whose low 24 bits ARE the key in every mode --
            // and it is filled by this point. That is what makes lkey unnecessary.
            const uint32_t k_ = (uint32_t)(lwork[pos * INW] & 0xFFFFFFu);
            const uint32_t hk = (k_ >> submask_bits) & (kTabSize - 1u);
            // atomic_exchange on a threadgroup uint: the chain head swap.
            lchain[pos] = atomic_exchange_explicit(&tab[hk], pos, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- WALK. Each slot follows its chain back to earlier same-hash slots and
        // emits on an equal full key, so every unordered colliding pair is emitted
        // exactly once. The table is a perfect hash here, so "same chain" already means
        // "same key" and the comparison is a tautology -- it is omitted.
        for (uint32_t pos = lId; pos < total; pos += kWG) {
            uint32_t oth = lchain[pos], walk = 0;
            while (oth != kEmpty) {
                if (++walk > 64u) {
                    atomic_fetch_add_explicit(&drops[3], 1u, memory_order_relaxed);
                    break;
                }
                const uint32_t la = kNeedLead ? llead[pos] : lleaf[pos * LEAFW + 0];
                const uint32_t lb = kNeedLead ? llead[oth] : lleaf[oth * LEAFW + 0];
                const uint32_t ga = kGiIsLead ? lleaf[pos * LEAFW + 0] : lgi[pos];
                const uint32_t gb = kGiIsLead ? lleaf[oth * LEAFW + 0] : lgi[oth];
                uint32_t leftPos = pos, rightPos = oth;
                if (lb < la || (lb == la && gb < ga)) { leftPos = oth; rightPos = pos; }

                bh3::Elem a, b, c;
                for (int w = 0; w < 7; ++w) { a.w[w] = 0; b.w[w] = 0; }
                for (int w = 0; w < INW; ++w) {
                    a.w[w] = lwork[leftPos  * INW + w];
                    b.w[w] = lwork[rightPos * INW + w];
                }
                bh3::combine(a, b, LOUT, c);

                uint32_t ctree[9];
                uint64_t contribOut = 0;
                if (LMODE == LM_USE) {
                    for (int i = 0; i < 8; ++i) ctree[i] = 0u;
                    ctree[8] = kNeedLead ? llead[rightPos] : lleaf[rightPos * LEAFW + 0];
                    const uint64_t lc = ((uint64_t)lleaf[leftPos * LEAFW + 1] << 32)
                                      |  (uint64_t)lleaf[leftPos * LEAFW + 0];
                    bh3::apply_mix(c, ctree, PADN, LOUT);
                    c.w[0] = bh3::rotl64(bh3::rotl64(c.w[0], 40) + lc, 24);
                    ctree[0] = kNeedLead ? llead[leftPos] : lleaf[leftPos * LEAFW + 0];
                } else {
                    for (uint32_t i = 0; i < SBUILD; ++i)
                        ctree[i] = (i < SIN) ? lleaf[leftPos  * LEAFW + i]
                                             : lleaf[rightPos * LEAFW + (i - SIN)];
                    bh3::apply_mix(c, ctree, PADN, LOUT);
                    if (LMODE == LM_EMIT || LMODE == LM_RD3) {
                        bh3::Elem z;
                        for (int w = 0; w < 7; ++w) z.w[w] = 0;
                        bh3::apply_mix(z, ctree, 8u, 288u);
                        contribOut = bh3::rotl64(z.w[0], 40);
                    }
                }

                const uint32_t ckey = (uint32_t)(c.w[0] & 0xFFFFFFu);
                const uint32_t cb   = ckey >> (24u - bucket_bits);
                const uint32_t cpos = atomic_fetch_add_explicit(&out_counts[cb], 1u,
                                                                memory_order_relaxed);
                if (cpos < out_bucket_cap) {
                    const uint32_t cgi = gi_alloc(gi_counter);
                    const size_t oslot = (size_t)cb * out_bucket_cap + cpos;
                    const size_t od    = oslot * OUTSTR;
                    if (LMODE == LM_SEED) {
                        out_belem[od + 0] = pair_w0(ckey, ctree[0]);
                        out_belem[od + 1] = pair_w1(ctree[1], cgi);
                    } else if (LMODE == LM_RD2 && OUTSTR == 3) {
                        // Keyed on the STRIDE, not on a macro: the miner instantiates
                        // both record formats and picks between them at runtime from the
                        // card's memory. OUTSTR == 3 is what "this is the quad record"
                        // means. The work words are dropped entirely; round 3 rebuilds
                        // them from these same four leaves.
                        out_belem[od + 0] = quad_w0(ckey, ctree[0]);
                        out_belem[od + 1] = quad_w1(ctree[1], ctree[2]);
                        out_belem[od + 2] = quad_w2(ctree[3], cgi);
                    } else if (LMODE == LM_RD2) {
                        for (int w = 0; w < 7; ++w) out_belem[od + w] = c.w[w];
                        out_belem[od + 7] = r3_p0(ctree[0], ctree[1], ctree[2]);
                        out_belem[side_out + oslot] = r3_p1(ctree[2], ctree[3], cgi);
                    } else if (LMODE == LM_EMIT || LMODE == LM_RD3) {
                        // LM_RD3 differs from LM_EMIT only in how it READ its input;
                        // what round 3 writes for round 4 is byte-identical either way.
                        for (int w = 0; w < 6; ++w) out_belem[od + w] = c.w[w];
                        out_belem[od + 6] = ((uint64_t)cgi << 32) | (uint64_t)ctree[0];
                        out_belem[od + 7] = contribOut;
                    } else {   // LM_USE -> the terminal round's 2 u64 input
                        out_belem[od + 0] = c.w[0];
                        out_belem[od + 1] = ((uint64_t)cgi << 32) | (uint64_t)ctree[0];
                    }
                    all_left [out_off + cgi] = kGiIsLead ? lleaf[leftPos  * LEAFW + 0] : lgi[leftPos];
                    all_right[out_off + cgi] = kGiIsLead ? lleaf[rightPos * LEAFW + 0] : lgi[rightPos];
                } else {
                    atomic_fetch_add_explicit(&drops[2], 1u, memory_order_relaxed);
                }
                oth = lchain[oth];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);  // next pass reuses every array
        ++xp;
    }
}

// ---------------------------------------------------------------------------------
// Per-round entry points.
//
// MSL kernel entry points cannot be templates, so each round declares its own
// statically-sized threadgroup arrays and calls the shared body. The array sizes are
// what set occupancy, so each kernel asserts its own total against Apple's 32 KB
// ceiling -- measured, not assumed (see the spec's "Measured constraints").
//
// The per-round parameter packs are the same ones cuda_solver.cu names in
// MXBM_R{1,2,3,4}_ARGS. They are repeated rather than macro'd because a silent drift
// between the launch site and the instantiation is exactly the bug the CUDA source
// documents having had.
// ---------------------------------------------------------------------------------

// Round 1: LM_SEED. INW=7 OUTW=7 LEAFW=1 LOUT=424 PADN=2 SIN=1 SOUT=2 SBUILD=2
// INSTR=1 OUTSTR=2 FCAP=288. gi IS the leaf, so lgi collapses to one word.
kernel void fused_round_r1(
        constant uint32_t*     cfg        [[buffer(0)]],
        const device uint32_t* in_counts  [[buffer(1)]],
        const device uint64_t* in_belem   [[buffer(2)]],
        device atomic_uint*    out_counts [[buffer(3)]],
        device uint64_t*       out_belem  [[buffer(4)]],
        device uint32_t*       all_left   [[buffer(5)]],
        device uint32_t*       all_right  [[buffer(6)]],
        device atomic_uint*    gi_counter [[buffer(7)]],
        device atomic_uint*    drops      [[buffer(8)]],
        const device uint64_t* pp4        [[buffer(9)]],
        uint lId     [[thread_position_in_threadgroup]],
        uint groupId [[threadgroup_position_in_grid]])
{
    using namespace mxbm::metalk;
    constexpr uint32_t F = kR1FCap;
    threadgroup uint64_t    lwork[7 * F];
    threadgroup uint32_t    lgi[1];
    threadgroup uint32_t    lleaf[1 * F];
    threadgroup uint32_t    llead[1];
    threadgroup uint32_t    lchain[F];
    threadgroup atomic_uint tab[kTabSize];
    threadgroup atomic_uint gcount;
    threadgroup atomic_uint cnt8[8];
    static_assert(sizeof(lwork) + sizeof(lgi) + sizeof(lleaf) + sizeof(llead)
                  + sizeof(lchain) + sizeof(tab) + 36u <= 32768u,
                  "r1 threadgroup allocation exceeds Apple's 32 KB ceiling");
    fused_round_body<7, 7, 1, LM_SEED, 424u, 2u, 1u, 2u, 2u, 1u, 2u, F>(
        cfg[0], cfg[1], cfg[2], cfg[3], cfg[4],
        in_counts, in_belem, out_counts, out_belem, all_left, all_right,
        gi_counter, drops, pp4,
        lwork, lgi, lleaf, llead, lchain, tab, gcount, cnt8, lId, groupId);
}

// Round 2: LM_RD2. Packed 9-u64 output record (8 inline + a 9th-word plane).
kernel void fused_round_r2(
        constant uint32_t*     cfg        [[buffer(0)]],
        const device uint32_t* in_counts  [[buffer(1)]],
        const device uint64_t* in_belem   [[buffer(2)]],
        device atomic_uint*    out_counts [[buffer(3)]],
        device uint64_t*       out_belem  [[buffer(4)]],
        device uint32_t*       all_left   [[buffer(5)]],
        device uint32_t*       all_right  [[buffer(6)]],
        device atomic_uint*    gi_counter [[buffer(7)]],
        device atomic_uint*    drops      [[buffer(8)]],
        const device uint64_t* pp4        [[buffer(9)]],
        uint lId     [[thread_position_in_threadgroup]],
        uint groupId [[threadgroup_position_in_grid]])
{
    using namespace mxbm::metalk;
    constexpr uint32_t F = kFCap;
    threadgroup uint64_t    lwork[7 * F];
    threadgroup uint32_t    lgi[F];
    threadgroup uint32_t    lleaf[2 * F];
    threadgroup uint32_t    llead[1];
    threadgroup uint32_t    lchain[F];
    threadgroup atomic_uint tab[kTabSize];
    threadgroup atomic_uint gcount;
    threadgroup atomic_uint cnt8[8];
    static_assert(sizeof(lwork) + sizeof(lgi) + sizeof(lleaf) + sizeof(llead)
                  + sizeof(lchain) + sizeof(tab) + 36u <= 32768u,
                  "r2 threadgroup allocation exceeds Apple's 32 KB ceiling");
    fused_round_body<7, 7, 2, LM_RD2, 400u, 4u, 2u, 4u, 4u, 2u, 8u, F>(
        cfg[0], cfg[1], cfg[2], cfg[3], cfg[4],
        in_counts, in_belem, out_counts, out_belem, all_left, all_right,
        gi_counter, drops, pp4,
        lwork, lgi, lleaf, llead, lchain, tab, gcount, cnt8, lId, groupId);
}

// Round 3: LM_EMIT. The tightest round -- 26 148 B on CUDA, the figure that pinned
// the 32 KB check in the first place.
kernel void fused_round_r3(
        constant uint32_t*     cfg        [[buffer(0)]],
        const device uint32_t* in_counts  [[buffer(1)]],
        const device uint64_t* in_belem   [[buffer(2)]],
        device atomic_uint*    out_counts [[buffer(3)]],
        device uint64_t*       out_belem  [[buffer(4)]],
        device uint32_t*       all_left   [[buffer(5)]],
        device uint32_t*       all_right  [[buffer(6)]],
        device atomic_uint*    gi_counter [[buffer(7)]],
        device atomic_uint*    drops      [[buffer(8)]],
        const device uint64_t* pp4        [[buffer(9)]],
        uint lId     [[thread_position_in_threadgroup]],
        uint groupId [[threadgroup_position_in_grid]])
{
    using namespace mxbm::metalk;
    constexpr uint32_t F = kFCap;
    threadgroup uint64_t    lwork[7 * F];
    threadgroup uint32_t    lgi[F];
    threadgroup uint32_t    lleaf[4 * F];
    threadgroup uint32_t    llead[1];
    threadgroup uint32_t    lchain[F];
    threadgroup atomic_uint tab[kTabSize];
    threadgroup atomic_uint gcount;
    threadgroup atomic_uint cnt8[8];
    static_assert(sizeof(lwork) + sizeof(lgi) + sizeof(lleaf) + sizeof(llead)
                  + sizeof(lchain) + sizeof(tab) + 36u <= 32768u,
                  "r3 threadgroup allocation exceeds Apple's 32 KB ceiling");
    fused_round_body<7, 6, 4, LM_EMIT, 376u, 6u, 4u, 2u, 8u, 8u, 8u, F>(
        cfg[0], cfg[1], cfg[2], cfg[3], cfg[4],
        in_counts, in_belem, out_counts, out_belem, all_left, all_right,
        gi_counter, drops, pp4,
        lwork, lgi, lleaf, llead, lchain, tab, gcount, cnt8, lId, groupId);
}

// Round 4: LM_USE. INW=6, and the leaf payload is the packed leftContrib rather than
// leaves -- the one mode that needs a separate llead array.
kernel void fused_round_r4(
        constant uint32_t*     cfg        [[buffer(0)]],
        const device uint32_t* in_counts  [[buffer(1)]],
        const device uint64_t* in_belem   [[buffer(2)]],
        device atomic_uint*    out_counts [[buffer(3)]],
        device uint64_t*       out_belem  [[buffer(4)]],
        device uint32_t*       all_left   [[buffer(5)]],
        device uint32_t*       all_right  [[buffer(6)]],
        device atomic_uint*    gi_counter [[buffer(7)]],
        device atomic_uint*    drops      [[buffer(8)]],
        const device uint64_t* pp4        [[buffer(9)]],
        uint lId     [[thread_position_in_threadgroup]],
        uint groupId [[threadgroup_position_in_grid]])
{
    using namespace mxbm::metalk;
    constexpr uint32_t F = kFCap;
    threadgroup uint64_t    lwork[6 * F];
    threadgroup uint32_t    lgi[F];
    threadgroup uint32_t    lleaf[2 * F];
    threadgroup uint32_t    llead[F];
    threadgroup uint32_t    lchain[F];
    threadgroup atomic_uint tab[kTabSize];
    threadgroup atomic_uint gcount;
    threadgroup atomic_uint cnt8[8];
    static_assert(sizeof(lwork) + sizeof(lgi) + sizeof(lleaf) + sizeof(llead)
                  + sizeof(lchain) + sizeof(tab) + 36u <= 32768u,
                  "r4 threadgroup allocation exceeds Apple's 32 KB ceiling");
    fused_round_body<6, 1, 2, LM_USE, 288u, 9u, 2u, 0u, 0u, 8u, 2u, F>(
        cfg[0], cfg[1], cfg[2], cfg[3], cfg[4],
        in_counts, in_belem, out_counts, out_belem, all_left, all_right,
        gi_counter, drops, pp4,
        lwork, lgi, lleaf, llead, lchain, tab, gcount, cnt8, lId, groupId);
}
