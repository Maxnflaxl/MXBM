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
#include <cooperative_groups.h>
#if MXBM_CPASYNC
#include <cuda_pipeline.h>
#endif
#ifndef MXBM_CPASYNC
#define MXBM_CPASYNC 0
#endif

namespace mxbm { namespace cuda {

// The staged group is mean_bucket / 2^submaskBits = 264 elements, so a 256-thread block
// runs a SECOND loop iteration with 8 of 256 lanes busy while the other seven warps wait
// at the barrier -- which is where the 15-16% barrier stall comes from. Sizing the block
// at or above the group removes it. Sweepable with -DMXBM_WG.
#ifndef MXBM_WG
#define MXBM_WG 256
#endif
constexpr uint32_t kWG      = (uint32_t)MXBM_WG;
// Group cap, and the single number that sets shared memory per block and therefore
// occupancy. It used to be 384 because it had to cover the group's TAIL; MXBM_SPILL
// removes that obligation by splitting an overflowing group on one more key bit, so it
// can now sit near the MEAN (264). At 320 rounds 1 and 2 cross from 3 to 4 blocks/SM --
// both their shared memory AND their register count fall below the threshold -- which is
// worth 1.25 ms. Round 3 needs a further cut to follow (see docs/performance.md).
// Sweepable with -DMXBM_FCAP; drops[1] catches a value set too low if MXBM_SPILL is off.
#ifndef MXBM_FCAP
#define MXBM_FCAP 320
#endif
constexpr uint32_t kFCap    = (uint32_t)MXBM_FCAP;
// 128, down from OpenCL's 512. Within a group the bucket fixes the key's high bits and
// the sub-mask its low ones, leaving exactly 24 - bucketBits - submaskBits = 7 varying
// bits -- and hk = (key >> submaskBits) & 127 selects precisely those, so 128 entries is
// a PERFECT hash here: every distinct key gets its own chain, and a chain holds exactly
// the elements sharing a key. 512 entries never addressed more than 128 of them.
//
// This holds for any geometry on the bucketBits + submaskBits = 17 line, which is the
// only line the group-size constraint allows (see docs/performance.md, row-bucket
// geometry). The 1.5 KB it frees against 512 is what buys round 3 a third block per SM.
// Sized for the bits that actually vary inside a group: 24 - bb - sm of them, so 128 is
// a perfect hash on the bb+sm=17 line. Raising bb+sm shrinks that, and the table with it
// -- MXBM_TAB. It must stay >= 2^(24-bb-sm) or distinct keys start sharing a chain
// (still correct, the walk compares full keys, but the walk gets longer).
#ifndef MXBM_TAB
#define MXBM_TAB 128
#endif
// Upper bound on a bucket's occupancy, i.e. fb_cap_for(capacity >> bb). Only the SUBPASS
// path allocates it, one BYTE per element. 768 covers bb=16 (cap 743).
#ifndef MXBM_SKEY
#define MXBM_SKEY 768
#endif
// MXBM_PERFECT_TAB: drop lkey entirely, and with it the key comparison in the walk.
//
// Inside a group the bucket fixes the key's top bb bits and the sub-mask its bottom sm,
// so exactly 24 - bb - sm bits vary -- and hk = (key >> sm) & (kTabSize-1) selects
// precisely those. When kTabSize >= 2^(24-bb-sm) the table is a PERFECT hash: two
// elements share a chain if and only if they share the full 24-bit key. So `lkey[oth] ==
// key` is a tautology, and the array exists only to re-derive a value that word 0 of
// lwork already carries in its low 24 bits (true in every LMODE -- the key was computed
// from that word by the producing round).
//
// Worth two things at once: 4 B/element of shared memory, and a shared load plus branch
// out of the innermost loop of the chain walk, which runs ~2 steps per element x 33.5 M
// elements per round.
//
// UNSAFE if the geometry ever makes kTabSize < 2^(24-bb-sm) -- distinct keys would then
// share a chain and be combined as if equal. The launcher must check; the KAT catches it.
#ifndef MXBM_PERFECT_TAB
#define MXBM_PERFECT_TAB 1
#endif

// MXBM_SPILL: when a group overflows kFCap, SPLIT it on one more key bit and redo,
// instead of dropping the overflow.
//
// kFCap has to cover the group's TAIL (352 at a mean of 264) while only the mean is ever
// staged, so ~25 % of the largest array in the kernel is a reservation that is almost
// never used -- and that reservation is what caps blocks/SM. Splitting is exactly safe:
// two elements whose keys differ in the added bit can never collide, so no pair is lost
// and none is emitted twice. It is also self-limiting -- the split halves the group, and
// the overflow that triggers it is rare, so the redone work is paid on a few per cent of
// blocks rather than all of them.
//
// gcount counts UNCLAMPED (the staging loop stops writing past kFCap but keeps counting),
// so it is the true group size and the decision is uniform across the block.
#ifndef MXBM_SPILL
#define MXBM_SPILL 1
#endif

// The ARENA template parameter: dense per-bucket caps backed by ONE global overflow
// pool per element set. A bucket that overflows its cap appends to the pool instead of
// dropping, and each consumer block stages its bucket's few pool entries from a
// per-bucket chain (arena_link builds it between rounds). The pool replaces the
// per-bucket tail reserve, which is the largest slack term in the row-bucket footprint;
// the pool itself is megabytes. Both variants are instantiated and CudaSolver picks per
// rung -- the mechanism costs a fixed few per cent, so it is what a card takes when the
// alternative is the quad record or a refusal. MXBM_ARENA sets only the DEFAULT, for
// single-config probe builds.
#ifndef MXBM_ARENA
#define MXBM_ARENA 0
#endif
constexpr uint32_t kArenaCap = 65536;  // pool slots per set; pool indices fit uint16_t
constexpr uint32_t kAMax     = 64;     // staged pool entries per bucket (shared list)

// MXBM_IMPBITS: stop storing the 16 key bits the bucket ADDRESS already encodes.
// The r2->r3 record's 400 work bits drop to 384 = 6 u64; the record becomes 8 u64
// inline ([w0'..w5', p0, p1]) and the side plane is deleted, which also removes one
// load and one store instruction per element. The kept low 8 key bits carry the
// sub-mask, the split bits and the 7 tab-hash bits, so matching is unchanged.
// Hardwired to bucket_bits == 16 (the launcher must enforce it) and to the
// perfect-tab compare (bits 23..8 of a packed rec0 are work bits, so a full-key
// compare would reject true partners).
#ifndef MXBM_IMPBITS
#define MXBM_IMPBITS 0
#endif
// (Guarded per instantiation in fused_round_body: IMPB needs the perfect-tab
// compare and cannot ride the cp.async raw copy.)
//
// The dropped-bit count rides the IMPB template parameter (0 = plain record,
// else bucket_bits): the packed record carries 400-IMPB bits in 6 u64 — 384
// exactly at 16, 383 with a spare bit at 17 — and every pack/unpack shift
// derives from it at compile time. The kept low 24-IMPB key bits must still
// cover the sub-mask and the 7 tab-hash bits, which holds on (16,1) and (17,0).
// MXBM_IMPDB sets only the DEFAULT, for single-config probe builds.
#ifndef MXBM_IMPDB
#define MXBM_IMPDB 16
#endif
constexpr uint32_t kImpDB = (uint32_t)MXBM_IMPDB;

// MXBM_PAIR_W0: the r1 -> r2 record carries the child's post-mix work word 0, so round 2
// derives only words 1..6 -- 12 siphashes and no mixes, against 14 siphashes and 3 mixes.
// See the w0-checkpoint record in bh3_records.cuh for the algebra and the bit layout.
//
// The record stays 16 B: word 0's extra 24 bits are paid for out of the bucket address,
// which is why this rides IMPB and is off wherever the packed r2 -> r3 record is off.
// Carrying it in a WIDER record instead costs more than the arithmetic it removes.
// r1's emit and r2's staging must be given the same IMPB or they disagree about the
// format; the launcher derives both from one condition.
// Worth -0.76 ms (-2.4 %) at stock, with r1's and r2's registers, shared memory and
// blocks/SM all unchanged. Set to 0 for the A/B.
#ifndef MXBM_PAIR_W0
#define MXBM_PAIR_W0 1
#endif

// MXBM_NARROW6: round 3 stages its 7th work word in 2 bytes instead of 8.
//
// r3's input is r2's output, which combine() masked to LOUT = 400 bits -- so word 6
// carries bits 384..399, sixteen significant bits in a u64. Storing it in its own u16
// plane takes r3 from 80 to 74 B per staged element, which is what it needs to reach
// 4 blocks/SM at a kFCap that keeps the spill rare. Only LM_EMIT qualifies: r1's input
// is a full 448 bits, r2's word 6 holds 40, and r4's word 5 holds 56.
//
// The cost is bank conflicts, and it is predictable: the conflict degree of a u64 stride
// S is 32/gcd(2S,32), so ODD strides give 2-way and even ones 4-way or worse. Dropping
// lwork from 7 to 6 u64 therefore doubles the conflicts on the walk's dominant shared
// access. Whether the extra block pays for that is the measurement.
//
// Measured null at stock (16,1), and measured LOSS composed with (17,0) under power
// caps -- +0.9 % at 180 W growing to +3 % at 100 W (2026-07-31): the split lw6 plane
// costs instructions on the staging store and every walk ldw, and instructions are
// the one thing a capped card cannot afford. Keep it off; do not re-propose.
#ifndef MXBM_NARROW6
#define MXBM_NARROW6 0
#endif

// Round 1's own cap. Folding its gi into its leaf and narrowing LEAFW to 1 took it to
// 64 B/element AND its register count from 64 to 48, so at 288 it fits 5 blocks/SM where
// 320 fits 4 -- 48 registers and 18980 B against the 48/19456 that 5 blocks needs, so no
// launch bound is required to hold it there. Note that is exactly ZERO register headroom:
// 49 registers rounds to 1792 per warp (granularity 256), which is 9 warps per
// sub-partition, 36 per SM, 4 blocks. tests/test_cuda_resources.cpp asserts it. Only r1 pays the higher spill rate a tighter
// cap implies; it is worth 0.15 ms of r1's 5.28.
#ifndef MXBM_R1_FCAP
#define MXBM_R1_FCAP 288
#endif
// 8-way max split. The level must be chosen BEFORE any part is walked: escalating after
// a walk would re-emit everything the earlier, coarser parts already emitted, which
// inflates the next round's bucket counts and evicts real elements -- it shows up as
// LOST goldens with a zero drop counter, not as an obvious failure.
constexpr uint32_t kMaxSpill = 3;

// MXBM_R4_ROWS: restore round 4's back-reference row and the terminal round's, so both
// recovery paths run side by side and are required to agree leaf for leaf. A shipping
// build writes neither -- replay_r4 reconstructs round 4's pairing from the input bucket
// the record carries. Off the octo rungs only: octo's round 4 is LM_RD4 and keeps its
// slot-indexed row.
#ifndef MXBM_R4_ROWS
#define MXBM_R4_ROWS 0
#endif
// terminal_round stages only word 0 + 4 meta u32 = 24 B/element, an eighth of a round
// block, so its group cap was never what constrained occupancy and must not be dragged
// down when kFCap is tuned for the rounds. It keeps the full tail cap.
constexpr uint32_t kTCap = kFCap > 384u ? kFCap : 384u;
constexpr uint32_t kSKey = (uint32_t)MXBM_SKEY;
constexpr uint32_t kTabSize = (uint32_t)MXBM_TAB;
constexpr uint32_t kEmpty   = 0xFFFFFFFFu;

// MXBM_STCS: emit through __stcs (streaming, evict-first in L2). Emitted records are
// read exactly ONCE, by the next round; back-refs are read only for the <=1024 survivors
// out of 33 M. Both still allocate L2 normally and evict the staging reads that DO have
// reuse -- the sub-mask rescan reads word 0 of every record twice.
#ifndef MXBM_STCS
#define MXBM_STCS 0
#endif
#if MXBM_STCS
__device__ __forceinline__ void st_em(ulonglong2* p, ulonglong2 v) { __stcs(p, v); }
__device__ __forceinline__ void st_em(uint64_t* p, uint64_t v) {
    __stcs((unsigned long long*)p, (unsigned long long)v); }
__device__ __forceinline__ void st_em(uint32_t* p, uint32_t v) { __stcs((unsigned*)p, v); }
#else
template<class T> __device__ __forceinline__ void st_em(T* p, T v) { *p = v; }
#endif

// MXBM_ABL_DERIVE: attribution only, RESULTS ARE INTENTIONALLY WRONG. Bit 0 replaces
// round 1's seed derivation, bit 1 round 2's 14-siphash rebuild, with a cheap spread.
// This is the one ablation that answers "how much of r1/r2's distance from their DRAM
// floor is compute" -- and it is safe to ablate because both re-derivations read only
// shared memory and registers, so the GLOBAL access footprint is untouched (the trap
// that mispriced the dense-gi experiment). Two controls have to be checked with it:
// drops must stay 0 and MXBM_OCC's max occupancy must stay near the real one, or the
// substitute has collapsed the bucket distribution and the number measures contention.
#ifndef MXBM_ABL_DERIVE
#define MXBM_ABL_DERIVE 0
#endif
__device__ __forceinline__ void abl_spread(uint32_t a, uint32_t b, bh3::Elem& e) {
    uint64_t x = ((uint64_t)a << 32 | b) * 0x9E3779B97F4A7C15ull;
    for (int w = 0; w < 7; ++w) { x ^= x >> 29; x *= 0xBF58476D1CE4E5B9ull; e.w[w] = x; }
}

// MXBM_WARPAGG: hand out the dense per-round id `gi` one atomic per WARP instead of one
// per lane. Every emit needs a unique id and they all come from a single u32, so a fully
// active warp would serialise into 32 L2 round-trips on one address. Aggregating keeps gi
// a permutation of [0,count) and makes the ids CONSECUTIVE within a warp, which is
// strictly better for the back-ref writes that index on them.
//
// BUT PTXAS ALREADY DOES IT (verified 2026-07-29 with cuobjdump -sass on the shipping
// archive; an earlier version of this comment said the SASS showed "a plain ATOMG.E.ADD
// per lane, with no aggregation prologue", which is false). Round 2 packed, 0x98f0-0x9a90:
// VOTEU.ANY -> FLO.U32 (leader) -> POPC (group size) -> @P0 ATOMG.E.ADD of the popcount
// -> SR_LTMASK + POPC (rank) -> SHFL.IDX (broadcast). ptxas and not NVVM: -ptx emits a
// bare atom.global.add.u32, -cubin emits the idiom. The negative control is the bucket
// atomic at 0x9810, whose address is lane-varying and which gets no prologue at all.
//
// So MXBM_WARPAGG=1 stacks a SECOND aggregation on ptxas's, which is why it measures
// +0.3 ms. Leave it off; it is not a lead.
//
// It does change which id a given emit gets, and gi is the tie-break when two elements
// share a lead (LEADTIE_PROBE: ~17 per solve out of 134 M). The KAT gate is what settles
// that -- note the existing order is already nondeterministic run to run.
#ifndef MXBM_WARPAGG
#define MXBM_WARPAGG 0
#endif
__device__ __forceinline__ uint32_t gi_alloc(uint32_t* __restrict__ ctr) {
#if MXBM_WARPAGG
    auto g = cooperative_groups::coalesced_threads();
    uint32_t base = 0;
    if (g.thread_rank() == 0) base = atomicAdd(ctr, g.size());
    return g.shfl(base, 0) + g.thread_rank();
#else
    return atomicAdd(ctr, 1u);
#endif
}

// MXBM_R3_QUAD: round 2 emits a 24 B QUAD RECORD -- key, four leaves and gi -- and round
// 3 rebuilds the 7 work words from those leaves instead of reading them. One level up
// from what LM_RD2 already does, and the largest single traffic lever in the pipeline:
// that record is written by r2 and read by r3, ~5.1 GB of the solve's 13.0, so at 24 B
// instead of 72 it removes ~26 % of ALL DRAM traffic. It also collapses set 0's stride
// from 9 u64 to 3 (see fb_set_stride), which is a ~2.1 GiB footprint cut -- the only
// change in the tree that moves the reach lever and the efficiency lever at once.
//
// It existed once on the OpenCL path (b360695) and was retired (2584518) at "+2.2 ms at
// 56", a NET figure that already contained whatever clock the freed watts bought, since
// the card is power-capped ~100 % of the time. What was never measured is the trade
// under a LOW cap, where a byte is worth 5x the clock it is worth at stock -- see
// docs/performance.md, "bytes are not free in watts".
//
// Off by default: it is expected to LOSE at stock and the point is to measure where it
// crosses over. Every figure it produces is still gated on the KAT and drops == 0.
#ifndef MXBM_R3_QUAD
#define MXBM_R3_QUAD 0
#endif

// Register-budget probes; see MINBLOCKS below. 0 = compiler default.
#ifndef MXBM_MB_SEED
#define MXBM_MB_SEED 0
#endif
#ifndef MXBM_MB_RD2
#define MXBM_MB_RD2 0
#endif
// Round 3's budget. Shared memory caps it at 3 resident blocks whatever this says, so
// the point is not occupancy but the REGISTER FILE: at 80 registers r3's three blocks
// hold 61,440 of the SM's 65,536, and the 4,096 left cannot host the co-resident entry
// block that speculative entry wants (256 threads x 46 = 11,776). Forcing 64 leaves
// 16,384. 0 = compiler default.
#ifndef MXBM_MB_EMIT
#define MXBM_MB_EMIT 0
#endif

// MXBM_POISON_W=N poisons work word N-1 of the r3->r4 record at round 3's emit; the KAT
// is the readout. N is 1-based over a 0-based array, so the dead word round 4 cannot
// reach (Lout(4)=288 masks c.w[4] to 32 bits, deleting w5's only term) is N=6, and N=5
// and N=4 are its positive controls. An attribution instrument: results are
// intentionally wrong for any N a consumer reaches. 0 = off.
#ifndef MXBM_POISON_W
#define MXBM_POISON_W 0
#endif

// MXBM_PAIR128: round 2 stages its 16 B pair record with one LD.128 instead of two
// LD.64 (INSTR is 2, so d*8 is 16 B aligned -- same argument as the record loads in
// LM_EMIT). The rescan lanes that fail the sub-mask filter fetch 16 B where they used
// to fetch 8, but those bytes are ~98 % L2-absorbed; what every staged element saves
// is a memory INSTRUCTION, the currency the MIO queue is priced in. Priced at the
// project's 1 %-of-a-solve floor -- expected <= ~0.2 ms; ships only if it clears noise.
#ifndef MXBM_PAIR128
#define MXBM_PAIR128 1
#endif

// MXBM_LWORK_SOA=1 lays lwork out column-major -- word w of element p at
// w*FCAP + p instead of p*LWS + w -- so lanes walking consecutive p read
// consecutive u64: conflict-free, where the record layout's odd u64 stride is
// 2-way (the 2026-07-31 r2 census measured 39 % of shared-load wavefronts as
// conflict replays at exactly the record reads). Same bytes, same shared
// budget, only the index changes; FCAP is compile-time, so per-word offsets
// stay immediates and each element still costs one base computation.
#ifndef MXBM_LWORK_SOA
#define MXBM_LWORK_SOA 0
#endif

// MXBM_PP_CONST: the siphash key (prePow) reaches the fused rounds through __constant__
// memory instead of a global pointer dereferenced in the all-lanes loop. MEASURED NULL
// 2026-08-02 (-0.07 %, 12 ABBA arms), and the premise was wrong with it: the key looked
// like 8 pinned registers of the 64 r2 is capped at, but register counts came back
// byte-identical, so ptxas was re-materialising it rather than pinning it. Kept as the
// closure's instrument -- off by default, codegen-neutral when off.
#ifndef MXBM_PP_CONST
#define MXBM_PP_CONST 0
#endif

#ifndef MXBM_ABL_TAIL
#define MXBM_ABL_TAIL 0
#endif

// MXBM_MATCH_FIRST: build the chain from the STAGED key, then derive only the elements
// the walk will actually read. An element alone in its chain slot is never loaded by the
// walk -- at 264 staged elements over a 128-entry perfect table that is e^-2.06 = 12.8 %
// of every group, each currently paying a full 7- or 14-siphash rebuild for nothing.
// Skipping them in place would buy nothing (the lanes idle inside the same warp), so the
// survivors are compacted into mlist first and the rebuild runs over that.
#ifndef MXBM_MATCH_FIRST
#define MXBM_MATCH_FIRST 0
#endif
#if MXBM_PP_CONST
__constant__ uint64_t c_pp[4];
#define MXBM_PP_LOAD(dst) uint64_t dst[4] = { c_pp[0], c_pp[1], c_pp[2], c_pp[3] }
#else
#define MXBM_PP_LOAD(dst) uint64_t dst[4] = { pp4[0], pp4[1], pp4[2], pp4[3] }
#endif

enum LMode { LM_EMIT = 1, LM_USE = 2, LM_SEED = 3, LM_RD2 = 4,
             // A/B pair (MXBM_R2_FULL): round 1 emits a FULL packed record and round 2
             // reads it instead of rebuilding from two seed indices. Trades bytes for
             // compute in the two kernels that still have DRAM headroom.
             LM_SEEDF = 5, LM_RAW = 6,
             // MXBM_R3_QUAD: round 3's mode. Reads a 24 B quad record and rebuilds; its
             // OUTPUT is byte-identical to LM_EMIT's, so every emit-side branch that
             // names LM_EMIT names this too.
             LM_RD3 = 7,
             // The OCTO RECORD: round 4's mode. Reads a 32 B record of eight leaves and
             // rebuilds the work words, the lead and the leftContrib; its OUTPUT is
             // byte-identical to LM_USE's, so every emit-side branch that names LM_USE
             // names this too. Round 3 emits the octo form when its OUTSTR is 4.
             LM_RD4 = 8 };

// MXBM_ABL_EMIT=R (1..4): skip round R's scattered record payload store. The bucket
// atomic, the gi atomic, the back-refs, combine and apply_mix ALL still run, so element
// counts and every other access stay representative and only the scattered write is
// removed. One round at a time, because the round after an ablated one reads garbage --
// pair it with MXBM_ROUND_REPS=R:N, whose replay input comes from the unablated R-1.
// Results are intentionally wrong.
#ifndef MXBM_ABL_EMIT
#define MXBM_ABL_EMIT 0
#endif
constexpr int kAblEmit = MXBM_ABL_EMIT == 1 ? LM_SEED : MXBM_ABL_EMIT == 2 ? LM_RD2
                       : MXBM_ABL_EMIT == 3 ? LM_EMIT : MXBM_ABL_EMIT == 4 ? LM_USE : 0;

// MXBM_ABL_MIX=R (1..4): skip round R's apply_mix, keeping combine. The child key then
// comes straight out of combine, which still mixes, so the bucket distribution stays
// healthy -- check MXBM_OCC anyway. c is still stored, so nothing is dead. Results wrong.
#ifndef MXBM_ABL_MIX
#define MXBM_ABL_MIX 0
#endif
#ifndef MXBM_CO_NOSCATTER
#define MXBM_CO_NOSCATTER 0
#endif
constexpr int kAblMix = MXBM_ABL_MIX == 1 ? LM_SEED : MXBM_ABL_MIX == 2 ? LM_RD2
                      : MXBM_ABL_MIX == 3 ? LM_EMIT : MXBM_ABL_MIX == 4 ? LM_USE : 0;

// The entry pass, as a device function so the standalone entry_scatter kernel and the
// co-tenant path below run byte-identical code.
__device__ __forceinline__
// The arena pointers are runtime rather than a template parameter here: with actr null
// an overflowing element takes the same drop it always did, and entry is 98 % SM-bound,
// so the extra branch is free where a second instantiation would not be.
void entry_body(uint32_t idx, const uint64_t* __restrict__ pp4, uint32_t bucket_bits,
                uint32_t bucket_cap, uint32_t* __restrict__ counts,
                uint64_t* __restrict__ belem, uint32_t* __restrict__ drops,
                uint32_t* __restrict__ actr = nullptr,
                uint32_t* __restrict__ atag = nullptr) {
    uint64_t pp[4] = { pp4[0], pp4[1], pp4[2], pp4[3] };
    bh3::Elem e; bh3::seed_element(pp, idx, e);
    uint32_t t1[1] = { idx };
    bh3::apply_mix(e, t1, 1u, 448u);
    const uint32_t key = (uint32_t)(e.w[0] & 0xFFFFFFu);
    const uint32_t b = key >> (24u - bucket_bits);
    const uint32_t pos = atomicAdd(&counts[b], 1u);
    size_t slot = (size_t)b*bucket_cap + pos;
    if (pos >= bucket_cap) {
        const uint32_t a = actr ? atomicAdd(actr, 1u) : kArenaCap;
        // One counter per capacity -- [0] entry scatter, [1] shared staging, [2] output
        // bucket, [3] chain cap -- so a nonzero one names which reservation was short.
        if (a >= kArenaCap) { atomicAdd(&drops[0], 1u); return; }
        atag[a] = b;
        slot = ((size_t)1u << bucket_bits)*bucket_cap + a;
    }
    belem[slot] = ((uint64_t)idx << 32) | key;
}

// COTENANT: host the NEXT nonce's entry pass inside this round's blocks.
//
// entry is 98 % SM / 16 % DRAM and r3/r4 are 21-29 % SM / 78-80 % DRAM, so they are
// complementary -- but two STREAMS cannot exploit that, and this was measured, not
// argued: a round enqueues 131072 blocks against 198 resident, so it is 662 waves deep
// and the work distributor never has a free slot to give the second kernel. The fix is
// not better scheduling, it is putting the work in the SAME launch, where it needs no
// slot of its own.
//
// The grids make this almost free: a round launches (nb << sm) x kWG = 2^17 x 256 = 2^25
// threads, which is exactly one per entry element. The co-tenant work sits after the
// round's last barrier, and since blocks finish their chain walks at different times
// (chain length varies with bucket occupancy), an SM naturally holds a mix of
// still-walking and already-seeding warps rather than alternating between phases.
//
// Cost is registers, and there is room: r3 uses 48 of the 85/thread that 3 blocks/SM
// allows, and it is shared-memory-bound at 32772 B, not register-bound. If the fused
// kernel pushes past 85 the occupancy drops to 2 and this will lose -- check ptxas -v.
// Everything a fused_round launch takes at runtime, as one POD so a heterogeneous
// kernel can carry TWO rounds' argument sets. Field order mirrors the kernel signature.
struct RoundArgs {
    uint32_t bucket_bits, submask_bits, in_bucket_cap, out_bucket_cap, out_off;
    const uint32_t* in_counts; const uint64_t* in_belem;
    uint32_t* out_counts; uint64_t* out_belem;
    uint32_t* all_left; uint32_t* all_right;
    uint32_t* gi_counter; uint32_t* drops;
    const uint64_t* pp4;
};

// One round's shared-memory arrays as a STRUCT, so a heterogeneous kernel can put two
// rounds' layouts in a union and pay max() of the two rather than their sum. The member
// order and sizes are exactly the arrays fused_round declared individually before this
// existed; test_cuda_resources pins the totals against cuobjdump.
template<int INW, int LEAFW, int LMODE, uint32_t FCAP, bool SUBPASS,
         bool MFIRST = (MXBM_MATCH_FIRST != 0), bool ARENA = (MXBM_ARENA != 0)>
struct RoundShared {
    static constexpr bool kNeedLead = (LMODE == LM_USE || LMODE == LM_RD4);
    static constexpr bool kNarrow6 =
        (MXBM_NARROW6 != 0) && (LMODE == LM_EMIT || LMODE == LM_RD3);
    static constexpr int  kLws = kNarrow6 ? INW - 1 : INW;
    static constexpr bool kGiIsLead = (LMODE == LM_SEED || LMODE == LM_SEEDF);
    uint64_t lwork[kLws * FCAP];
    uint16_t lw6[kNarrow6 ? FCAP : 1];
    uint32_t lgi[kGiIsLead ? 1 : FCAP];
    uint32_t lleaf[LEAFW * FCAP];
    uint32_t llead[kNeedLead ? FCAP : 1];
    uint32_t lkey[MXBM_PERFECT_TAB ? 1 : FCAP];
    uint32_t lchain[FCAP];
    uint32_t tab[kTabSize];
    uint16_t mlist[MFIRST ? FCAP : 1];             // staged slots with a partner
    uint32_t nmatch;
    uint32_t gcount;
    uint32_t cnt8[MXBM_SPILL ? 8 : 1];
    uint8_t  skey[SUBPASS ? kSKey : 1];
    // This bucket's pool entries, chain-walked once. One element rather than zero off
    // their rungs, matching the other placeholders here: a zero-length member is a GNU
    // extension MSVC rejects. Costs 8 B where the mode is off, which no round's
    // resident-block cliff notices -- test_cuda_resources holds every one of them.
    uint16_t aidx[ARENA ? kAMax : 1];
    uint32_t acnt[1];
    // Where each staged element was read from. LM_RD4 only: its back-references name a
    // parent by slot rather than by gi -- see recover() in pipeline_kernels.cuh.
    uint32_t lslot[LMODE == LM_RD4 ? FCAP : 1];
};

// The round body, extracted so a kernel can run DIFFERENT rounds in different blocks of
// one launch (see fused_pair below / cuda/pipeline.cu). `bid` is the block's rank among
// this round's blocks -- blockIdx.x for a homogeneous launch, a remapped rank otherwise.
// Shared memory comes in as a RoundShared& so the caller controls where it lives.
template<int INW, int OUTW, int LEAFW, int LMODE,
         uint32_t LOUT, uint32_t PADN, uint32_t SIN, uint32_t SOUT, uint32_t SBUILD,
         uint32_t INSTR, uint32_t OUTSTR, uint32_t FCAP, bool SUBPASS = false,
         bool MFIRST = (MXBM_MATCH_FIRST != 0),
         // 0 = plain record; else the count of address-implied key bits the pack
         // drops (16 for (16,1), 17 for (17,0)) -- every shift derives from it.
         uint32_t IMPB = (MXBM_IMPBITS ? kImpDB : 0u),
         bool ARENA = (MXBM_ARENA != 0),
         // Write the two back-reference rows. False on rounds 1-3 of an octo rung,
         // whose rows recovery never reads and the host never allocates.
         bool REFS = true>
__device__ __forceinline__
void fused_round_body(RoundShared<INW, LEAFW, LMODE, FCAP, SUBPASS, MFIRST, ARENA>& shm,
                 uint32_t bid,
                 uint32_t bucket_bits, uint32_t submask_bits,
                 uint32_t in_bucket_cap, uint32_t out_bucket_cap, uint32_t out_off,
                 const uint32_t* __restrict__ in_counts,
                 const uint64_t* __restrict__ in_belem,
                 uint32_t* __restrict__ out_counts,
                 uint64_t* __restrict__ out_belem,
                 uint32_t* __restrict__ all_left,
                 uint32_t* __restrict__ all_right,
                 uint32_t* __restrict__ gi_counter,
                 uint32_t* __restrict__ drops,
                 const uint64_t* __restrict__ pp4,
                 const uint32_t* __restrict__ in_ahead = nullptr,
                 const uint32_t* __restrict__ in_anext = nullptr,
                 uint32_t* __restrict__ out_actr = nullptr,
                 uint32_t* __restrict__ out_atag = nullptr) {
    static_assert(!IMPB || MXBM_PERFECT_TAB,
                  "IMPB: bits 23..8 of a packed rec0 are work bits, so a full-key "
                  "compare would reject true partners");
    static_assert(!IMPB || !MXBM_CPASYNC,
                  "IMPB repacks the record; the cp.async raw copy cannot carry it");
    static_assert(!IMPB || LMODE != LM_EMIT || INSTR == 8,
                  "IMPB's unpack is a fixed 4 x ulonglong2, so the record it reads must "
                  "have been written at stride 8. A wider one (MXBM_R2_FULL) compiles "
                  "and then fails the KAT");
    // The w0-checkpoint pair record (MXBM_PAIR_W0). It spends the address-implied key
    // bits, so it exists only where IMPB does -- and both sides of the r1/r2 boundary
    // read the same IMPB, which is what keeps writer and reader on one layout.
    constexpr bool PW0 = (MXBM_PAIR_W0 != 0) && IMPB != 0;
    static_assert(!PW0 || LMODE != LM_RD2 || INSTR == 2,
                  "the w0-checkpoint record is 2 u64; a wider input stride is a "
                  "different format");
    // `lead` is leaf 0 of the element's own prefix, so for every mode that stages real
    // leaves it is already in lleaf and a separate array is pure waste. LM_USE is the
    // exception: its leaf payload is the packed leftContrib, not leaves -- and LM_RD4
    // derives that payload rather than reading it, but stores it in the same place.
    constexpr bool kNeedLead = (LMODE == LM_USE || LMODE == LM_RD4);
    // Derived, not passed: LM_EMIT is round 3's mode and nothing else uses it, so the
    // launch sites need no new argument.
    constexpr bool NARROW6 = (MXBM_NARROW6 != 0) && (LMODE == LM_EMIT || LMODE == LM_RD3);
    constexpr int  LWS     = NARROW6 ? INW - 1 : INW;      // lwork stride, u64
    // Round 1's gi IS its leaf: both are the seed index, written from the same
    // rec0 >> 32. So LM_SEED keeps only one copy and reads gi through lleaf[0].
    constexpr bool kGiIsLead = (LMODE == LM_SEED || LMODE == LM_SEEDF);
    // Reference bindings: the body below is textually what it was when these were the
    // kernel's own __shared__ declarations, and the bindings keep it that way.
    auto& lwork  = shm.lwork;
    auto& lw6    = shm.lw6;
    auto& lgi    = shm.lgi;
    auto& lleaf  = shm.lleaf;
    auto& llead  = shm.llead;
    auto& lkey   = shm.lkey;
    auto& lchain = shm.lchain;
    [[maybe_unused]] auto& lslot = shm.lslot;
    [[maybe_unused]] auto& mlist  = shm.mlist;
    [[maybe_unused]] auto& nmatch = shm.nmatch;
    auto& tab    = shm.tab;
    auto& gcount = shm.gcount;
    auto& cnt8   = shm.cnt8;
    auto& skey   = shm.skey;
    auto lead_of = [&](uint32_t p) -> uint32_t {
        return kNeedLead ? llead[p] : lleaf[p * LEAFW + 0];
    };
    auto gi_of = [&](uint32_t p) -> uint32_t {
        return kGiIsLead ? lleaf[p * LEAFW + 0] : lgi[p];
    };
    // What a back-reference names: a gi, except on the octo round 4, which names the slot
    // its parent's record sits in so recovery can read the eight leaves straight out.
    auto ref_of = [&](uint32_t p) -> uint32_t {
        if constexpr (LMODE == LM_RD4) return lslot[p];
        else                           return gi_of(p);
    };
    // The one place lwork's layout is spelled out; see MXBM_LWORK_SOA above.
    auto lwx = [&](uint32_t p, int w) -> uint32_t {
#if MXBM_LWORK_SOA
        return (uint32_t)w * FCAP + p;
#else
        return p * (uint32_t)LWS + (uint32_t)w;
#endif
    };
    // w is a compile-time-bounded loop index at every call site, so the branch folds.
    auto ldw = [&](uint32_t p, int w) -> uint64_t {
        if constexpr (NARROW6) return (w == INW - 1) ? (uint64_t)lw6[p] : lwork[lwx(p, w)];
        else                   return lwork[lwx(p, w)];
    };
    auto stw = [&](uint32_t p, int w, uint64_t v) {
        if constexpr (NARROW6) { if (w == INW - 1) lw6[p] = (uint16_t)v;
                                 else              lwork[lwx(p, w)] = v; }
        else lwork[lwx(p, w)] = v;
    };

    const uint32_t lId = threadIdx.x;
    const uint32_t submaskCount = 1u << submask_bits;
    // Without SUBPASS a block is one (bucket, sub-mask) and the grid is nb << sm. With it
    // a block is one BUCKET, the grid is nb, and the sub-masks become an inner loop.
    const uint32_t bucket = SUBPASS ? bid : bid / submaskCount;
    const uint32_t mask0  = SUBPASS ? 0u : bid % submaskCount;
    const uint32_t nsweep = SUBPASS ? submaskCount : 1u;

    // Round 2's record is 9 u64, and a 9-u64 stride puts every odd slot on an 8 B
    // boundary, which the 128-bit accesses below cannot use. Padding the stride to 10
    // bought the alignment at 8 B/element of DRAM traffic in BOTH directions -- round 2
    // writes it and round 3 reads it back -- which measured 537 MB per solve, 4.0 % of
    // all traffic, carrying nothing. Instead the 9th word lives in its own plane after
    // the records, so the record stride is 8 (always 16 B aligned) and the odd word is
    // one coalesced u64 per element. Same instruction count, 72 B/element instead of 80.
    // The plane offset is derived from the geometry rather than passed: nb = 1<<bb.
    // With the arena, each set's layout is [bucket records][pool records][plane]:
    // pool slot a lives at record index nbcap + a, so ONE slot index addresses both
    // regions and every store below is unchanged; only the plane offset moves.
    const size_t nbcap_in  = ((size_t)1u << bucket_bits) * in_bucket_cap;
    const size_t nbcap_out = ((size_t)1u << bucket_bits) * out_bucket_cap;
    [[maybe_unused]] const size_t side_in  = (nbcap_in  + (ARENA ? kArenaCap : 0u)) * INSTR;
    [[maybe_unused]] const size_t side_out = (nbcap_out + (ARENA ? kArenaCap : 0u)) * OUTSTR;
    // The 9-word record needs INW work words + 2 meta, so a stride of INW+1 means the
    // last one is in the plane and anything wider means it is still inline. Reader and
    // writer must agree, and only the stride tells them apart -- MXBM_R2_FULL's LM_RAW
    // producer keeps the old padded 10-u64 record with no plane behind it.
    auto r3_word8 = [&](const uint64_t* __restrict__ src, size_t inline_d, size_t plane_i)
                        -> uint64_t {
        if constexpr (INSTR >= INW + 2) return src[inline_d + INW + 1];
        else                            return src[plane_i];
    };

    uint32_t cnt = in_counts[bucket];
    if (cnt > in_bucket_cap) cnt = in_bucket_cap;
    const size_t base = (size_t)bucket * in_bucket_cap;

    // KEY PREPASS (SUBPASS only). This is the read the sub-mask geometry otherwise pays
    // 2^sm times over -- once per (bucket, sub-mask) block. Doing it once here means the
    // sweep below costs shared reads instead of global ones, so a FINER sub-mask (smaller
    // group, less shared memory, more resident blocks) no longer costs more global
    // traffic. That coupling is what pinned bb + sm = 17.
    if constexpr (SUBPASS) {
        for (uint32_t p = lId; p < cnt && p < kSKey; p += kWG)
            skey[p] = (uint8_t)(in_belem[(base + p) * INSTR] & 0xFFu);
        if (cnt > kSKey) cnt = kSKey;      // cannot happen: kSKey covers in_bucket_cap
        __syncthreads();
    }

    // This bucket's pool entries, chain-walked ONCE into shared. ~95 % of buckets
    // have an empty chain at the dense cap, so the walk is one global load; a chain
    // past kAMax loses elements, which drops[1] makes loud rather than silent.
    uint32_t acnt = 0u;
    if constexpr (ARENA) {
        if (lId == 0) {
            uint32_t n_ = 0;
            if (in_ahead)
                for (uint32_t a_ = in_ahead[bucket]; a_ != 0xFFFFFFFFu; a_ = in_anext[a_]) {
                    if (n_ >= kAMax) { atomicAdd(&drops[1], 1u); break; }
                    shm.aidx[n_++] = (uint16_t)a_;
                }
            shm.acnt[0] = n_;
        }
        __syncthreads();
        acnt = shm.acnt[0];
    }

  for (uint32_t sweep = 0; sweep < nsweep; ++sweep) {
    const uint32_t mask_s = SUBPASS ? sweep : mask0;
    uint32_t xb = 0, xp = 0, nparts = 1;     // split level, current part, part count
    bool counted = false;                    // the level is chosen once, never escalated
    while (xp < nparts) {
    const uint32_t sbits = submask_bits + xb;
    const uint32_t mask  = mask_s | (xp << submask_bits);
    if (lId == 0) { gcount = 0; if constexpr (MFIRST) nmatch = 0; }
    for (uint32_t i = lId; i < kTabSize; i += kWG) tab[i] = kEmpty;
    __syncthreads();

    // STAGE. Sub-mask filtered, so only ~1/2^submask_bits of the lanes survive: keep this
    // loop cheap and defer anything expensive to the all-lanes loop below.
    for (uint32_t p = lId; p < cnt + acnt; p += kWG) {
        const bool ar_ = ARENA && p >= cnt;
        // The filter now costs a shared byte, not a global word. The full record is still
        // read exactly once across the whole sweep, by whichever sweep claims it.
        if constexpr (SUBPASS)
            if (!ar_ && (skey[p] & ((1u << sbits) - 1u)) != mask) continue;
        size_t idx_ = base + p;
        if constexpr (ARENA)
            if (ar_) idx_ = nbcap_in + shm.aidx[p - cnt];
        const size_t d = idx_ * INSTR;
        uint64_t rec0;
        [[maybe_unused]] uint64_t rec1 = 0;
        if constexpr (LMODE == LM_RD2 && MXBM_PAIR128) {
            // Both words of the pair record in one LD.128 -- see MXBM_PAIR128 above.
            const ulonglong2 q = *reinterpret_cast<const ulonglong2*>(in_belem + d);
            rec0 = q.x; rec1 = q.y;
        } else {
            rec0 = in_belem[d];
        }
        const uint32_t key = (uint32_t)(rec0 & 0xFFFFFFu);
        if constexpr (!SUBPASS)
            if ((key & ((1u << sbits) - 1u)) != mask) continue;
        // Pool entries bypass the skey prepass, so under SUBPASS they filter here.
        if constexpr (ARENA && SUBPASS)
            if (ar_ && (key & ((1u << sbits) - 1u)) != mask) continue;
        const uint32_t pos = atomicAdd(&gcount, 1u);
        // Keep counting past FCAP: gcount is what decides whether to split.
        if (pos >= FCAP) { if constexpr (!MXBM_SPILL) atomicAdd(&drops[1], 1u); continue; }
        if constexpr (MFIRST) {
            // The chain, built here rather than after the rebuild: this loop already
            // holds the key, and knowing the chain first is what lets the rebuild skip
            // the elements no walk will read. Same hash and insert as the loop below.
            lchain[pos] = atomicExch(&tab[(key >> submask_bits) & (kTabSize - 1u)], pos);
        }
        if constexpr (LMODE == LM_SEED || LMODE == LM_SEEDF) {
            lleaf[pos*LEAFW + 0] = (uint32_t)(rec0 >> 32);           // derive later
            if constexpr (!MXBM_PERFECT_TAB) lkey[pos] = key;
        } else if constexpr (LMODE == LM_RD2) {
            if constexpr (!MXBM_PAIR128) rec1 = in_belem[d + 1];
            if constexpr (PW0) {
                // Word 0 is read, not derived: stage it here with the bucket's key bits
                // put back, and the all-lanes loop below rebuilds the linear lane only.
                stw(pos, 0, pw0_word0<IMPB>(rec0, bucket));
                lgi[pos] = pw0_gi<IMPB>(rec0, rec1);
                lleaf[pos*LEAFW + 0] = pw0_left(rec1);              // leaf 0 IS the lead
                lleaf[pos*LEAFW + 1] = pw0_right(rec1);
            } else {
                const uint32_t li = pair_left(rec0), ri = pair_right(rec1);
                lgi[pos] = pair_gi(rec1);
                lleaf[pos*LEAFW + 0] = li; lleaf[pos*LEAFW + 1] = ri;  // leaf 0 IS the lead
            }
            if constexpr (!MXBM_PERFECT_TAB) lkey[pos] = key;
        } else if constexpr (LMODE == LM_RD3 && INSTR == 2) {
            // 16 B quad record, one LD.128. No gi in it: on an octo rung the only thing
            // that would read one is this round's own tiebreak, and the slot orders the
            // staged elements just as well -- see quad16 in bh3_records.cuh.
            const ulonglong2 q = *reinterpret_cast<const ulonglong2*>(in_belem + d);
            uint32_t l4[4];
            quad16_leaves(q.x, q.y, l4);
            lgi[pos] = (uint32_t)idx_;
            if constexpr (!MXBM_PERFECT_TAB) lkey[pos] = key;
            #pragma unroll
            for (int t = 0; t < 4; ++t) lleaf[pos*LEAFW + t] = l4[t];   // leaf 0 IS the lead
        } else if constexpr (LMODE == LM_RD3) {
            // 24 B quad record: three scalar loads, against LM_EMIT's five LD.128 below.
            // Not vectorised, and deliberately not padded to make it so -- a 4th u64
            // would buy one 128-bit load and give back a third of the bytes this mode
            // exists to save. Fewer INSTRUCTIONS as well as fewer bytes, which matters
            // because round 3's top stall is the MIO queue and not bandwidth.
            const uint64_t w1 = in_belem[d + 1], w2 = in_belem[d + 2];
            lgi[pos] = quad_gi(w2);
            if constexpr (!MXBM_PERFECT_TAB) lkey[pos] = key;
            lleaf[pos*LEAFW + 0] = quad_l0(rec0);   // leaf 0 IS the lead
            lleaf[pos*LEAFW + 1] = quad_l1(w1);
            lleaf[pos*LEAFW + 2] = quad_l2(w1);
            lleaf[pos*LEAFW + 3] = quad_l3(w2);
        } else if constexpr (LMODE == LM_EMIT) {
#if MXBM_ABL_TAIL == 3
            // Prices the OTHER half of "materialise only what the walk reads": in a
            // bandwidth-bound round the saving is DRAM sectors, so a predicate is enough
            // and no compaction is needed. Skips one element in eight -- near the 12.8 %
            // that is alone in its chain slot -- leaving its work state garbage. Timing
            // probe: results WRONG, needs MXBM_FORCE_TIMING.
            if ((p & 7u) == 0u) { lgi[pos] = 0; lleaf[pos*LEAFW] = 0; continue; }
#endif
            // 128-bit loads. The record stride is an even number of u64, so d*8 is 16 B
            // aligned and cudaMalloc's base is 256 B aligned -- but the compiler cannot
            // prove either, and emitted 9 x LD.64 where 5 x LD.128 do. Round 3 stalls
            // 22% of its warp-active cycles on the MIO queue, which is memory-INSTRUCTION
            // issue rather than bandwidth, so instruction count is the thing to cut here.
            static_assert(INSTR % 2 == 0, "vectorised path needs an even record stride");
#if MXBM_CPASYNC
            // cp.async: global -> shared without a register round-trip, and without the
            // warp stalling on the load before it can store. Global-latency stalls are
            // now the top stall in this kernel (52.8% long_scoreboard), which is exactly
            // what this is for. 8 B granularity: lwork's stride is INW u64, so the shared
            // destination is only 16 B aligned for even pos.
            #pragma unroll
            for (int w = 0; w < INW; ++w)
                __pipeline_memcpy_async(&lwork[lwx(pos, w)], &in_belem[d + w], 8);
            __pipeline_commit();
            const uint64_t p0 = in_belem[d + INW], p1 = r3_word8(in_belem, d, side_in + idx_);
#else
            const ulonglong2* v = reinterpret_cast<const ulonglong2*>(in_belem + d);
            ulonglong2 q0 = v[0], q1 = v[1], q2 = v[2], q3 = v[3];
            uint64_t p0, p1;
            if constexpr (IMPB) {
                // Unpack the packed record, reinserting the IMPB address bits
                // from the block's own bucket index. Inverse of the LM_RD2 pack.
                constexpr uint32_t impKB = 24u - IMPB;
                constexpr uint64_t impM  = (1ull << impKB) - 1u;
                stw(pos,0,(q0.x & impM) | ((uint64_t)bucket << impKB)
                          | (((q0.x >> impKB) & 0xFFFFFFFFFFull) << 24));
                stw(pos,1,(q0.x >> (64 - IMPB)) | (q0.y << IMPB));
                stw(pos,2,(q0.y >> (64 - IMPB)) | (q1.x << IMPB));
                stw(pos,3,(q1.x >> (64 - IMPB)) | (q1.y << IMPB));
                stw(pos,4,(q1.y >> (64 - IMPB)) | (q2.x << IMPB));
                stw(pos,5,(q2.x >> (64 - IMPB)) | (q2.y << IMPB));
                stw(pos,6,(q2.y >> (64 - IMPB)) & 0xFFFFull);
                p0 = q3.x; p1 = q3.y;
            } else {
                stw(pos,0,q0.x); stw(pos,1,q0.y);
                stw(pos,2,q1.x); stw(pos,3,q1.y);
                stw(pos,4,q2.x); stw(pos,5,q2.y);
                stw(pos,6,q3.x);
                p0 = q3.y; p1 = r3_word8(in_belem, d, side_in + idx_);
            }
#endif
            const uint32_t l0 = r3_l0(p0);
            lgi[pos] = r3_gi(p1);
            if constexpr (!MXBM_PERFECT_TAB) lkey[pos] = key;
            lleaf[pos*LEAFW + 0] = l0;          lleaf[pos*LEAFW + 1] = r3_l1(p0);
            lleaf[pos*LEAFW + 2] = r3_l2(p0,p1); lleaf[pos*LEAFW + 3] = r3_l3(p1);
        } else if constexpr (LMODE == LM_RD4) {
            // 32 B octo record: two LD.128 where LM_USE reads eight u64. The raw words
            // go into lwork's own slots, which the rebuild below overwrites with the six
            // it derives, so this mode costs no shared memory over LM_USE. lwork word 0
            // carries the key in its low 24 bits both before and after the rebuild, so
            // the chain build downstream needs no special case.
            static_assert(INSTR == 4 && INW >= 4, "the octo record is 4 u64 in lwork");
            const ulonglong2* v = reinterpret_cast<const ulonglong2*>(in_belem + d);
            const ulonglong2 q0 = v[0], q1 = v[1];
            stw(pos, 0, q0.x); stw(pos, 1, q0.y);
            stw(pos, 2, q1.x); stw(pos, 3, q1.y);
            lgi[pos]   = octo_gi(q1.y);
            llead[pos] = octo_lead(q0.x);          // leaf 0 IS the lead
            lslot[pos] = (uint32_t)idx_;           // what this round's references name
            if constexpr (!MXBM_PERFECT_TAB) lkey[pos] = key;
        } else {   // LM_USE: work words, meta, then the leftContrib as the leaf payload
            // INW-generic: this branch serves LM_USE (INW=6) and LM_RAW (INW=7), and
            // hardcoding three ulonglong2 loads silently dropped word 6 for the latter.
            static_assert(INSTR % 2 == 0, "vectorised path needs an even record stride");
#if MXBM_CPASYNC
            #pragma unroll
            for (int w = 0; w < INW; ++w)
                __pipeline_memcpy_async(&lwork[lwx(pos, w)], &in_belem[d + w], 8);
            __pipeline_commit();
#else
            const ulonglong2* v = reinterpret_cast<const ulonglong2*>(in_belem + d);
            #pragma unroll
            for (int j = 0; j < INW/2; ++j) {
                const ulonglong2 q = v[j];
                lwork[lwx(pos, 2*j)] = q.x; lwork[lwx(pos, 2*j + 1)] = q.y;
            }
            if constexpr (INW & 1) lwork[lwx(pos, INW-1)] = in_belem[d + INW-1];
#endif
            const uint64_t meta = in_belem[d + INW];
            lgi[pos] = (uint32_t)(meta >> 32); llead[pos] = (uint32_t)meta;
            if constexpr (!MXBM_PERFECT_TAB) lkey[pos] = key;
            for (uint32_t i = 0; i < SIN; ++i)
                lleaf[pos*LEAFW + i] =
                    (uint32_t)(in_belem[d + INW + 1 + (i >> 1)] >> ((i & 1u) * 32u));
        }
    }
#if MXBM_CPASYNC
    if constexpr (LMODE == LM_EMIT || LMODE == LM_USE || LMODE == LM_RAW)
        __pipeline_wait_prior(0);      // all in-flight copies land before the barrier
#endif
    __syncthreads();
    if constexpr (MXBM_SPILL) {
        if (gcount > FCAP && !counted) {
            // One word-0 pass counts all 8 possible sub-parts at once, so the split level
            // is picked from real sizes rather than discovered by trial. Paid only by the
            // few per cent of groups that overflow; the common path never runs it.
            if (lId < 8) cnt8[lId] = 0;
            __syncthreads();
            for (uint32_t p = lId; p < cnt; p += kWG) {
                uint32_t k2;
                if constexpr (SUBPASS) k2 = skey[p];
                else                   k2 = (uint32_t)(in_belem[(base + p) * INSTR] & 0xFFu);
                if ((k2 & (submaskCount - 1u)) != mask_s) continue;
                atomicAdd(&cnt8[(k2 >> submask_bits) & 7u], 1u);
            }
            __syncthreads();
            uint32_t lv = 1;
            for (; lv <= kMaxSpill; ++lv) {
                uint32_t mx = 0;
                for (uint32_t q = 0; q < (1u << lv); ++q) {
                    uint32_t sz = 0;
                    for (uint32_t i = 0; i < 8u; ++i)
                        if ((i & ((1u << lv) - 1u)) == q) sz += cnt8[i];
                    if (sz > mx) mx = sz;
                }
                if (mx <= FCAP) break;
            }
            xb = lv > kMaxSpill ? kMaxSpill : lv;
            nparts = 1u << xb; xp = 0; counted = true;
            __syncthreads();
            continue;
        }
        // Only reachable if the chosen split still overflows, which the count rules out.
        if (gcount > FCAP && lId == 0) atomicAdd(&drops[1], gcount - FCAP);
    }
#if MXBM_ABL_TAIL == 1
    uint32_t total = gcount < FCAP ? gcount : FCAP;
#else
    const uint32_t total = gcount < FCAP ? gcount : FCAP;
#endif
    // MXBM_ABL_TAIL: drop the elements past one block-wide pass. Timing probe only --
    // it loses ~3 % of every group, so results are WRONG -- and it exists to price the
    // ragged tail of the expand loop below: the staged group is ~264 against kWG 256,
    // so 8 lanes run a second derivation while the other 248 wait at the barrier.
#if MXBM_ABL_TAIL == 1
    if (total > kWG) total = kWG;
#endif
    // MXBM_ABL_TAIL=2: derive only the leading 7/8 of the group, leaving the rest's work
    // state garbage while the chain, the walk and the emit stay whole. That is the shape
    // of "derive only the elements the walk will read" -- ~12.8 % of a group is alone in
    // its 7-bit chain slot at 264 elements over 128 slots -- without the compaction such
    // a change needs. Timing probe: results WRONG, and it needs MXBM_PERFECT_TAB=0 so the
    // chain is built from the staged key rather than from the work state it corrupts.
#if MXBM_ABL_TAIL == 2
    const uint32_t nDer = (total * 7u) / 8u;
#endif

    uint32_t nWork = total;
    if constexpr (MFIRST) {
        // COMPACT. One thread per chain slot walks its members; a slot holding a single
        // element contributes nothing to the walk, so its element is never materialised.
        // 128 slots against kWG lanes, ~2 members each: one short pass, no barrier inside.
        for (uint32_t h = lId; h < kTabSize; h += kWG) {
            const uint32_t head = tab[h];
            if (head == kEmpty || lchain[head] == kEmpty) continue; // empty, or singleton
            uint32_t n = 0, e = head;
            while (e != kEmpty && n < FCAP) { ++n; e = lchain[e]; }
            uint32_t w = atomicAdd(&nmatch, n);
            for (e = head; e != kEmpty && w < FCAP; e = lchain[e]) mlist[w++] = (uint16_t)e;
        }
        __syncthreads();
        nWork = nmatch;
    }

    // EXPAND + CHAIN. Every lane is active here, which is why the re-derivations live in
    // this loop and not the staging one (worth 8x their SIMD utilisation).
    for (uint32_t i_ = lId; i_ < nWork; i_ += kWG) {
      const uint32_t pos = MFIRST ? (uint32_t)mlist[i_] : i_;
#if MXBM_ABL_TAIL == 2
      if (pos < nDer) {
#endif
        if constexpr (LMODE == LM_SEED || LMODE == LM_SEEDF) {
            const uint32_t idx = lleaf[pos*LEAFW + 0];
            MXBM_PP_LOAD(pp);
            bh3::Elem e;
            if constexpr (MXBM_ABL_DERIVE & 1) abl_spread(idx, 0u, e);
            else {
                bh3::seed_element(pp, idx, e);
                uint32_t t1[1] = { idx };
                bh3::apply_mix(e, t1, 1u, 448u);
            }
            for (int w = 0; w < INW; ++w) lwork[lwx(pos, w)] = e.w[w];
        } else if constexpr (LMODE == LM_RD2 && PW0) {
            // Word 0 came out of the record, so only the linear lane is derived. Every
            // word here is the one rebuild_r2 would have produced; word 0 is left alone.
            MXBM_PP_LOAD(pp);
            uint64_t w16[7];
            if constexpr (MXBM_ABL_DERIVE & 2) {
                bh3::Elem e;
                abl_spread(lleaf[pos*LEAFW + 0], lleaf[pos*LEAFW + 1], e);
                #pragma unroll
                for (int w = 1; w < INW; ++w) w16[w] = e.w[w];
            } else {
                rebuild_r2_lane(pp, lleaf[pos*LEAFW + 0], lleaf[pos*LEAFW + 1], w16);
            }
            #pragma unroll
            for (int w = 1; w < INW; ++w) lwork[lwx(pos, w)] = w16[w];
        } else if constexpr (LMODE == LM_RD2) {
            MXBM_PP_LOAD(pp);
            bh3::Elem e;
            if constexpr (MXBM_ABL_DERIVE & 2)
                abl_spread(lleaf[pos*LEAFW + 0], lleaf[pos*LEAFW + 1], e);
            else
                rebuild_r2(pp, lleaf[pos*LEAFW + 0], lleaf[pos*LEAFW + 1], e);
            for (int w = 0; w < INW; ++w) lwork[lwx(pos, w)] = e.w[w];
        } else if constexpr (LMODE == LM_RD3) {
            // Deferred, in the all-lanes loop, for the same reason round 2's rebuild is:
            // the staging loop above is sub-mask filtered, so only ~1/2^submask_bits of
            // the lanes are live in it, and the same arithmetic there measured +23.5 ms
            // against +0.4 ms here.
            MXBM_PP_LOAD(pp);
            const uint32_t l[4] = { lleaf[pos*LEAFW + 0], lleaf[pos*LEAFW + 1],
                                    lleaf[pos*LEAFW + 2], lleaf[pos*LEAFW + 3] };
            bh3::Elem e;
            rebuild_r3(pp, l, e);
            for (int w = 0; w < INW; ++w) lwork[lwx(pos, w)] = e.w[w];
        } else if constexpr (LMODE == LM_RD4) {
            // Deferred here for the same reason as the two rebuilds above: every lane is
            // live, and only 1/2^submask_bits of them were in the staging loop. The leaves
            // are unpacked into registers before the overwrite, so the aliasing on lwork
            // is only apparent.
            MXBM_PP_LOAD(pp);
            uint32_t l[8];
            octo_leaves(lwork[lwx(pos,0)], lwork[lwx(pos,1)],
                        lwork[lwx(pos,2)], lwork[lwx(pos,3)], l);
            bh3::Elem e;
            rebuild_r4(pp, l, e);
            const uint64_t lc = octo_contrib(l);   // the leftContrib round 3 did not store
            for (int w = 0; w < INW; ++w) lwork[lwx(pos, w)] = e.w[w];
            lleaf[pos*LEAFW + 0] = (uint32_t)lc;
            lleaf[pos*LEAFW + 1] = (uint32_t)(lc >> 32);
        }
#if MXBM_ABL_TAIL == 2
      }
#endif
        if constexpr (!MFIRST) {
            // lwork[pos*INW] is word 0, whose low 24 bits ARE the key in every mode --
            // and it is filled by this point (the two derive modes fill it just above).
            const uint32_t k_ = MXBM_PERFECT_TAB ? (uint32_t)(lwork[lwx(pos, 0)] & 0xFFFFFFu)
                                                 : lkey[pos];
            const uint32_t hk = (k_ >> submask_bits) & (kTabSize - 1u);
            lchain[pos] = atomicExch(&tab[hk], pos);
        }
    }
    __syncthreads();

    // WALK. Each slot follows its chain back to earlier same-hash slots and emits on an
    // equal full key, so every unordered colliding pair is emitted exactly once.
    for (uint32_t pos = lId; pos < total; pos += kWG) {
        uint32_t oth = lchain[pos], walk = 0;
        while (oth != kEmpty) {
            if (++walk > 64u) { atomicAdd(&drops[3], 1u); break; }
            // Perfect table => same chain means same key, so the test is a tautology.
            if (MXBM_PERFECT_TAB || lkey[oth] == lkey[pos]) {
                const uint32_t la = lead_of(pos), lb = lead_of(oth);
                const uint32_t ga = gi_of(pos), gb = gi_of(oth);
                uint32_t leftPos = pos, rightPos = oth;
                if (lb < la || (lb == la && gb < ga)) { leftPos = oth; rightPos = pos; }

                bh3::Elem a{}, b{}, c;
                for (int w = 0; w < INW; ++w) { a.w[w] = ldw(leftPos, w);
                                                b.w[w] = ldw(rightPos, w); }
                bh3::combine(a, b, LOUT, c);

                uint32_t ctree[9]; uint64_t contribOut = 0;
                if constexpr (LMODE == LM_USE || LMODE == LM_RD4) {
                    for (int i = 0; i < 8; ++i) ctree[i] = 0u;
                    ctree[8] = lead_of(rightPos);
                    const uint64_t lc = ((uint64_t)lleaf[leftPos*LEAFW+1] << 32)
                                      |  (uint64_t)lleaf[leftPos*LEAFW+0];
                    if constexpr (LMODE != kAblMix) bh3::apply_mix(c, ctree, PADN, LOUT);
                    c.w[0] = bh3::rotl64(bh3::rotl64(c.w[0], 40) + lc, 24);
                    ctree[0] = lead_of(leftPos);
                } else {
                    for (uint32_t i = 0; i < SBUILD; ++i)
                        ctree[i] = (i < SIN) ? lleaf[leftPos*LEAFW + i]
                                             : lleaf[rightPos*LEAFW + (i - SIN)];
                    if constexpr (LMODE != kAblMix) {
                        bh3::apply_mix(c, ctree, PADN, LOUT);
                        // The leftContrib round 4 consumes. Under the octo record round 4
                        // derives it from the same eight leaves, so it is not computed
                        // here rather than computed and discarded.
                        if constexpr ((LMODE == LM_EMIT || LMODE == LM_RD3) && OUTSTR != 4) {
                            bh3::Elem z{};
                            bh3::apply_mix(z, ctree, 8u, 288u);
                            contribOut = bh3::rotl64(z.w[0], 40);
                        }
                    }
                }

                const uint32_t ckey = (uint32_t)(c.w[0] & 0xFFFFFFu);
                const uint32_t cb   = ckey >> (24u - bucket_bits);
                const uint32_t cpos = atomicAdd(&out_counts[cb], 1u);
                size_t oslot = (size_t)cb * out_bucket_cap + cpos;
                if constexpr (ARENA)
                    if (cpos >= out_bucket_cap) {
                        const uint32_t a_ = out_actr ? atomicAdd(out_actr, 1u) : kArenaCap;
                        if (a_ < kArenaCap) { out_atag[a_] = cb; oslot = nbcap_out + a_; }
                        else oslot = ~(size_t)0;
                    }
                if (ARENA ? oslot != ~(size_t)0 : cpos < out_bucket_cap) {
                    // The gi this child is given. Dead on an octo rung's round 2: no
                    // reference row is indexed by it and the 16 B record does not carry
                    // it, so the atomic is not issued at all.
                    // Round 4's 8 B record drops gi with everything else it stopped
                    // needing, and no row is indexed by one there, so the atomic goes.
                    constexpr bool kNeedGi = (LMODE == LM_USE && OUTSTR == 1)
                                           ? false
                                           : (REFS || !(LMODE == LM_RD2 && OUTSTR == 2));
                    const uint32_t cgi = kNeedGi ? gi_alloc(gi_counter) : 0u;
                    const size_t od    = oslot * OUTSTR;
                    if constexpr (LMODE == kAblEmit) {
                        // NARROW the payload to 16 B rather than skipping it. Skipping
                        // outright makes combine, apply_mix and the whole ctree build
                        // dead code, and the compiler duly deletes them -- which prices
                        // "scatter removed AND compute removed" as one number (r1 then
                        // measures NEGATIVE). Folding every word the real store would
                        // have written keeps all of it live, so the only variable left
                        // is bytes. r1 and r4 already store 16 B, so they are the
                        // positive control: their delta must come out ~0.
                        uint64_t f = contribOut;
                        #pragma unroll
                        for (int w = 0; w < 7; ++w) f ^= c.w[w];
                        constexpr uint32_t NF = (LMODE == LM_USE || LMODE == LM_RD4) ? 9u : SBUILD;
                        #pragma unroll
                        for (uint32_t i = 0; i < NF; ++i) f ^= (uint64_t)ctree[i];
                        st_em(reinterpret_cast<ulonglong2*>(out_belem + od),
                              make_ulonglong2(f, (uint64_t)cgi));
                    } else if constexpr (LMODE == LM_SEED) {
                        static_assert(OUTSTR % 2 == 0, "vectorised path needs an even stride");
                        if constexpr (PW0)
                            st_em(reinterpret_cast<ulonglong2*>(out_belem + od),
                                  make_ulonglong2(pw0_w0<IMPB>(c.w[0], cgi),
                                                  pw0_w1(ctree[0], ctree[1], cgi)));
                        else
                            st_em(reinterpret_cast<ulonglong2*>(out_belem + od),
                                  make_ulonglong2(pair_w0(ckey, ctree[0]),
                                                  pair_w1(ctree[1], cgi)));
                    } else if constexpr (LMODE == LM_RD2 && OUTSTR == 2) {
                        st_em(reinterpret_cast<ulonglong2*>(out_belem + od),
                              make_ulonglong2(quad16_w0(ckey, ctree), quad16_w1(ctree)));
                    } else if constexpr (LMODE == LM_RD2 && OUTSTR == 3) {
                        // Keyed on the STRIDE, not on MXBM_R3_QUAD: the miner
                        // instantiates both record formats and picks between them at
                        // runtime from the card's VRAM, so the macro is not a constant
                        // there. OUTSTR == 3 is what "this is the quad record" means.
                        // QUAD RECORD: key, four leaves, gi -- 3 u64 where the packed
                        // record below takes 9. The work words are dropped entirely;
                        // round 3 rebuilds them from these same four leaves. Three
                        // scalar stores rather than 4 x ST.128 + 1 x ST.64, so this is
                        // fewer store INSTRUCTIONS as well as fewer bytes.
                        st_em(out_belem + od + 0, quad_w0(ckey, ctree[0]));
                        st_em(out_belem + od + 1, quad_w1(ctree[1], ctree[2]));
                        st_em(out_belem + od + 2, quad_w2(ctree[3], cgi));
                    } else if constexpr (LMODE == LM_RD2 || LMODE == LM_RAW) {
                        // Matching 128-bit stores; the even stride makes od*8 16 B
                        // aligned. 4 x ST.128 + 1 x ST.64 instead of 9 x ST.64.
                        static_assert(OUTSTR % 2 == 0, "vectorised path needs an even stride");
                        ulonglong2* v = reinterpret_cast<ulonglong2*>(out_belem + od);
                        if constexpr (IMPB && LMODE == LM_RD2) {
                            // Pack out the IMPB key bits above the kept low bits
                            // (they ARE cb): w0' = low kept bits | w0 bits 63..24
                            // at impKB | w1's low bits on top; then
                            // wi' = wi>>IMPB | w(i+1)<<(64-IMPB). p1 rides inline
                            // where the dropped word was, so the side plane has no
                            // writer.
                            static_assert(OUTW == 7 || !IMPB, "pack is 400-bit");
                            constexpr uint32_t impKB = 24u - IMPB;
                            constexpr uint64_t impM  = (1ull << impKB) - 1u;
                            st_em(v + 0, make_ulonglong2(
                                (c.w[0] & impM) | ((c.w[0] >> 24) << impKB)
                                                | (c.w[1] << (impKB + 40)),
                                (c.w[1] >> IMPB) | (c.w[2] << (64 - IMPB))));
                            st_em(v + 1, make_ulonglong2(
                                (c.w[2] >> IMPB) | (c.w[3] << (64 - IMPB)),
                                (c.w[3] >> IMPB) | (c.w[4] << (64 - IMPB))));
                            st_em(v + 2, make_ulonglong2(
                                (c.w[4] >> IMPB) | (c.w[5] << (64 - IMPB)),
                                (c.w[5] >> IMPB) | (c.w[6] << (64 - IMPB))));
                            st_em(v + 3, make_ulonglong2(r3_p0(ctree[0], ctree[1], ctree[2]),
                                                         r3_p1(ctree[2], ctree[3], cgi)));
                        } else {
                            st_em(v + 0, make_ulonglong2(c.w[0], c.w[1]));
                            st_em(v + 1, make_ulonglong2(c.w[2], c.w[3]));
                            st_em(v + 2, make_ulonglong2(c.w[4], c.w[5]));
                            st_em(v + 3, make_ulonglong2(c.w[6], r3_p0(ctree[0], ctree[1], ctree[2])));
                            // LM_RAW is the MXBM_R2_FULL experiment, which keeps the
                            // padded 10-u64 record and so has no side plane behind it.
                            if constexpr (LMODE == LM_RD2)
                                st_em(out_belem + side_out + oslot, r3_p1(ctree[2], ctree[3], cgi));
                            else
                                st_em(out_belem + od + OUTW + 1, r3_p1(ctree[2], ctree[3], cgi));
                        }
                    } else if constexpr (LMODE == LM_SEEDF) {
                        // Full packed record: 7 work + meta + 1 leaf u64 (sOut = 2).
                        static_assert(OUTSTR % 2 == 0, "vectorised path needs an even stride");
                        ulonglong2* v = reinterpret_cast<ulonglong2*>(out_belem + od);
                        st_em(v + 0, make_ulonglong2(c.w[0], c.w[1]));
                        st_em(v + 1, make_ulonglong2(c.w[2], c.w[3]));
                        st_em(v + 2, make_ulonglong2(c.w[4], c.w[5]));
                        st_em(v + 3, make_ulonglong2(c.w[6],
                                   ((uint64_t)cgi << 32) | (uint64_t)ctree[0]));
                        st_em(out_belem + od + 8, ((uint64_t)ctree[1] << 32) | (uint64_t)ctree[0]);
                    } else if constexpr ((LMODE == LM_EMIT || LMODE == LM_RD3)
                                         && OUTSTR == 4) {
                        // THE OCTO RECORD. ctree already holds the eight leaves that
                        // determine this child, so the six work words, the lead and the
                        // leftContrib are all derivable and none is stored: 32 B instead
                        // of 64, in one ST.128 pair.
                        static_assert(SBUILD == 8, "the octo record is the 8-leaf tree");
                        ulonglong2* v = reinterpret_cast<ulonglong2*>(out_belem + od);
                        st_em(v + 0, make_ulonglong2(octo_w0(ckey, ctree), octo_w1(ctree)));
                        st_em(v + 1, make_ulonglong2(octo_w2(ctree), octo_w3(ctree, cgi)));
                    } else if constexpr (LMODE == LM_EMIT || LMODE == LM_RD3) {
                        // LM_RD3 differs from LM_EMIT only in how it READ its input; what
                        // round 3 writes for round 4 is byte-identical either way.
                        static_assert(OUTSTR % 2 == 0, "vectorised path needs an even stride");
                        ulonglong2* v = reinterpret_cast<ulonglong2*>(out_belem + od);
#if MXBM_POISON_W
                        // Poison rather than zero: a reader must corrupt, not coincide.
                        uint64_t cw[7]; for (int q = 0; q < 7; ++q) cw[q] = c.w[q];
                        cw[MXBM_POISON_W - 1] = 0xDEADBEEFDEADBEEFull;
                        st_em(v + 0, make_ulonglong2(cw[0], cw[1]));
                        st_em(v + 1, make_ulonglong2(cw[2], cw[3]));
                        st_em(v + 2, make_ulonglong2(cw[4], cw[5]));
#else
                        st_em(v + 0, make_ulonglong2(c.w[0], c.w[1]));
                        st_em(v + 1, make_ulonglong2(c.w[2], c.w[3]));
                        // Word 5 in the emitting block's INPUT bucket's place: at
                        // Lout(4) = 288 the shift drops everything above bit 311, so this
                        // word cannot reach round 4's output and round 4 reads it only to
                        // XOR it into a result that masks it away. replay_r3 reads it.
                        st_em(v + 2, make_ulonglong2(c.w[4], (uint64_t)bucket));
#endif
                        st_em(v + 3, make_ulonglong2(((uint64_t)cgi << 32) | (uint64_t)ctree[0],
                                               contribOut));
                    } else if constexpr (LMODE == LM_USE && OUTSTR == 1) {
                        st_em(out_belem + od, t5_rec(c.w[0], bucket));
                    } else {
                        static_assert(OUTSTR % 2 == 0, "vectorised path needs an even stride");
                        // Round 4's record carries the emitting block's INPUT bucket, so
                        // recovery replays this round over that one bucket of round 3's
                        // output instead of reading a row written for all 33.5 M children.
                        // The terminal combines at Lout = 24, keeping only word 0's bits
                        // 24..47, so 48..63 are dead there; bucket_bits reaches 17 and the
                        // 17th bit goes above gi's 26 in the meta word.
                        constexpr bool kBHint = (LMODE == LM_USE);
                        const uint64_t w0o = kBHint
                            ? ((c.w[0] & 0xFFFFFFFFFFFFull) | ((uint64_t)(bucket & 0xFFFFu) << 48))
                            : c.w[0];
                        const uint64_t m1o = ((uint64_t)cgi << 32) | (uint64_t)ctree[0]
                                           | (kBHint ? ((uint64_t)(bucket >> 16) << 63) : 0ull);
                        st_em(reinterpret_cast<ulonglong2*>(out_belem + od),
                              make_ulonglong2(w0o, m1o));
                    }
                    // Which rounds still write a reference row. Off the octo rungs
                    // round 4 never does -- replay_r4 reconstructs its pairing -- and on
                    // the implicit-bits rungs rounds 1-3 do not either: replay_r3 reaches
                    // round 2's output, whose record already carries four leaves in tree
                    // order. IMPB is nonzero on exactly those rungs. MXBM_R4_ROWS puts
                    // every row back so the two recoveries can be compared.
                    constexpr bool kReplayed = (LMODE == LM_USE)
                        || (IMPB != 0 && (LMODE == LM_SEED || LMODE == LM_RD2
                                          || LMODE == LM_EMIT));
                    if constexpr (REFS && (MXBM_R4_ROWS || !kReplayed)) {
                        st_em(all_left  + out_off + cgi, ref_of(leftPos));
                        st_em(all_right + out_off + cgi, ref_of(rightPos));
                    }
                } else atomicAdd(&drops[2], 1u);
            }
            oth = lchain[oth];
        }
    }
    __syncthreads();       // next pass reuses every shared array above
    ++xp;
    }
  }

}

template<int INW, int OUTW, int LEAFW, int LMODE,
         uint32_t LOUT, uint32_t PADN, uint32_t SIN, uint32_t SOUT, uint32_t SBUILD,
         uint32_t INSTR, uint32_t OUTSTR,
         // Per-round group cap. Every shared array scales with it, and B (shared bytes
         // per staged element) differs by round -- r3 80, r2 72, r1 64 -- so one global
         // value cannot put them all at their best occupancy. Lowering it for a round
         // costs only that round's own spill rate. Ahead of the two bools because the
         // co-tenant and sub-pass launches append to the argument list.
         uint32_t FCAP = kFCap, bool COTENANT = false, bool SUBPASS = false,
         // COBLOCKS: co-schedule the NEXT nonce's entry pass as SEPARATE BLOCKS
         // interleaved through this round's grid -- every co_stride-th block is an entry
         // block. Distinct from both measured overlap nulls: a second STREAM never runs
         // (662 waves of backlog), and COTENANT runs entry in the round's own warps
         // AFTER their round work, where a memory stall serialises rather than overlaps.
         // Here entry's instructions issue from their own warps, resident BESIDE the
         // round's stalled ones -- at the price of displacing ~1/co_stride of the
         // round's blocks (each entry block still reserves the round's static shared).
         bool COBLOCKS = false,
         // Minimum resident blocks/SM the compiler must budget registers for. 0 keeps
         // the default. Probed and NULL: r1/r2 are shared-memory-capped at 5/4 blocks
         // before the freed registers can matter (5 x 23.6 KB > 100 KB for r2), so
         // occupancy there is closed from BOTH resources. Kept as an instrument.
         // Skip the rebuild for elements alone in their chain slot: a loss at stock,
         // a win once a cap makes the rebuild bill at full issue rate, so both
         // variants are instantiated and CudaSolver picks. Placed BEFORE MINBLOCKS so
         // naming it at a launch site keeps MINBLOCKS's per-round default -- a
         // spelled-out 0 would silently drop an MXBM_MB_* probe budget.
         bool MFIRST = (MXBM_MATCH_FIRST != 0),
         uint32_t IMPB = (MXBM_IMPBITS ? kImpDB : 0u),
         // Dense per-bucket caps plus a global overflow pool -- the fit-ladder rung, see
         // MXBM_ARENA above. Also before MINBLOCKS, for the same reason.
         bool ARENA = (MXBM_ARENA != 0),
         // See fused_round_body: rounds 1-3 of an octo rung write no references.
         bool REFS = true,
         int MINBLOCKS = (LMODE == LM_SEED ? MXBM_MB_SEED
                        : LMODE == LM_RD2  ? MXBM_MB_RD2
                        : LMODE == LM_EMIT ? MXBM_MB_EMIT : 0)>
__global__ __launch_bounds__(kWG, MINBLOCKS)
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
                 const uint64_t* __restrict__ pp4,
                 const uint32_t* __restrict__ in_ahead = nullptr,
                 const uint32_t* __restrict__ in_anext = nullptr,
                 uint32_t* __restrict__ out_actr = nullptr,
                 uint32_t* __restrict__ out_atag = nullptr,
                 // co-tenant entry pass for the NEXT nonce; ignored unless COTENANT
                 const uint64_t* __restrict__ co_pp4 = nullptr,
                 uint32_t co_count = 0, uint32_t co_bucket_cap = 0,
                 uint32_t* __restrict__ co_counts = nullptr,
                 uint64_t* __restrict__ co_belem = nullptr,
                 uint32_t co_stride = 0, uint32_t co_begin = 0) {
    // COBLOCKS role dispatch. The grid is (nb << sm) round blocks plus one entry block
    // per (co_stride - 1) of them, interleaved by index so every dispatch wave carries
    // the mix -- appending them instead would schedule them all in the tail, which is
    // the second-stream null all over again. Entry blocks return before the first
    // barrier; a barrier is per-block, so the round blocks are unaffected.
    uint32_t bid = blockIdx.x;
    if constexpr (COBLOCKS) {
        if (bid % co_stride == co_stride - 1u) {
            const uint32_t eb = bid / co_stride;            // 0-based among entry blocks
            const uint32_t nE = gridDim.x / co_stride;
#if MXBM_CO_NOSCATTER
            // Diagnostic (results wrong without a separate real entry launch): compute
            // only, no scatter -- isolates issue interference from DRAM interference.
            uint64_t acc = 0;
            for (uint32_t g = eb * blockDim.x + threadIdx.x; g < co_count;
                 g += nE * blockDim.x) {
                uint64_t pp[4] = { co_pp4[0], co_pp4[1], co_pp4[2], co_pp4[3] };
                bh3::Elem e; bh3::seed_element(pp, co_begin + g, e);
                uint32_t t1[1] = { co_begin + g };
                bh3::apply_mix(e, t1, 1u, 448u);
                acc ^= e.w[0];
            }
            if (acc == 0xDEADBEEFDEADBEEFull) atomicAdd(&drops[3], 1u);
#else
            for (uint32_t g = eb * blockDim.x + threadIdx.x; g < co_count;
                 g += nE * blockDim.x)
                entry_body(co_begin + g, co_pp4, bucket_bits, co_bucket_cap, co_counts,
                           co_belem, drops);
#endif
            return;
        }
        bid -= bid / co_stride;                             // rank among round blocks
    }

    __shared__ RoundShared<INW, LEAFW, LMODE, FCAP, SUBPASS, MFIRST, ARENA> shm;
    fused_round_body<INW, OUTW, LEAFW, LMODE, LOUT, PADN, SIN, SOUT, SBUILD,
                     INSTR, OUTSTR, FCAP, SUBPASS, MFIRST, IMPB, ARENA, REFS>(
        shm, bid, bucket_bits, submask_bits, in_bucket_cap, out_bucket_cap, out_off,
        in_counts, in_belem, out_counts, out_belem, all_left, all_right, gi_counter,
        drops, pp4, in_ahead, in_anext, out_actr, out_atag);

    // Co-tenant: the next nonce's entry pass, in THIS launch. Grid-strided rather than
    // one-element-per-thread, so it stays correct if the geometry or kWG ever makes
    // gridDim*blockDim differ from 2^25 (today they are equal).
    if constexpr (COTENANT) {
#if MXBM_CO_NOSCATTER
        // DIAGNOSTIC, result wrong: run entry's COMPUTE only, no scatter. entry reads as
        // "98 % SM, 16 % DRAM", but that 16 % is 33.5 M scattered 8 B writes plus 33.5 M
        // bucket atomics -- and the host round is limited by scatter TRANSACTIONS, not
        // bytes. If compute-only is nearly free while the full co-tenant is not, then
        // what does not fit is the scatter, and no amount of idle SM will take it.
        // The predicated store keeps the compute live without adding traffic.
        uint64_t acc = 0;
        for (uint32_t g = blockIdx.x*blockDim.x + threadIdx.x; g < co_count;
             g += gridDim.x*blockDim.x) {
            uint64_t pp[4] = { co_pp4[0], co_pp4[1], co_pp4[2], co_pp4[3] };
            bh3::Elem e; bh3::seed_element(pp, g, e);
            uint32_t t1[1] = { g };
            bh3::apply_mix(e, t1, 1u, 448u);
            acc ^= e.w[0];
        }
        if (acc == 0xDEADBEEFDEADBEEFull) atomicAdd(&drops[3], 1u);
#else
        // MXBM_CO_STAGGER: half the blocks seed BEFORE their round work. Without it every
        // block runs round-then-entry, so a wave of blocks moves through the two phases
        // together and the SM alternates instruction mixes instead of blending them.
        for (uint32_t g = blockIdx.x*blockDim.x + threadIdx.x; g < co_count;
             g += gridDim.x*blockDim.x)
            entry_body(g, co_pp4, bucket_bits, co_bucket_cap, co_counts, co_belem, drops);
#endif
    }
}

// ---- fused_pair: TWO ROUNDS OF TWO SOLVES IN ONE LAUNCH ------------------------------
//
// The roofline said the SM is idle half the wall clock and DRAM the other half, at
// different moments -- and both prior attempts to overlap phases were null: a second
// stream never runs (662 waves of backlog) and same-warp hosting serialises behind the
// stall. This is the third mechanism: solve A's round in the even blocks and solve B's
// round in the odd blocks of ONE grid, so each SM holds a mix of A-blocks and B-blocks
// and B's instructions issue from their own warps while A's warps sit in their memory
// stalls. The COBLOCKS probe measured the mechanism real: co-scheduled compute hides at
// ~2/3, and displacing blocks of a DRAM-bound round is free.
//
// The shared-memory union is the point of the RoundShared refactor: the block pays
// max(A, B), not A + B, so residency stays at the shipping 4 blocks/SM when r4 hosts r1.
// The register file is the hazard instead -- the merged kernel takes the max of both
// paths, and at 256 threads anything past 64 registers drops a block. Check the
// occupancy probe before believing any number from this kernel.
//
// bEvery: every bEvery-th block is a B block; (gridDim / bEvery) must equal B's block
// count and the rest must equal A's. For two full rounds on the bb+sm=17 line both need
// nb << sm blocks, so the grid is 2*(nb << sm) and bEvery is 2.
template<int AINW, int AOUTW, int ALEAFW, int ALMODE,
         uint32_t ALOUT, uint32_t APADN, uint32_t ASIN, uint32_t ASOUT, uint32_t ASBUILD,
         uint32_t AINSTR, uint32_t AOUTSTR, uint32_t AFCAP,
         int BINW, int BOUTW, int BLEAFW, int BLMODE,
         uint32_t BLOUT, uint32_t BPADN, uint32_t BSIN, uint32_t BSOUT, uint32_t BSBUILD,
         uint32_t BINSTR, uint32_t BOUTSTR, uint32_t BFCAP,
         // Each B block runs BMULT consecutive B ranks back to back. This is how the
         // RESIDENCY split is steered away from the 1:1 the equal block counts force:
         // with BMULT=2 and bEvery=3, two thirds of the resident blocks are A's.
         uint32_t BMULT = 1>
__global__ __launch_bounds__(kWG)
void fused_pair(RoundArgs a, RoundArgs b, uint32_t bEvery) {
    __shared__ union PairShared {
        RoundShared<AINW, ALEAFW, ALMODE, AFCAP, false> a;
        RoundShared<BINW, BLEAFW, BLMODE, BFCAP, false> b;
    } shm;
    const uint32_t bid = blockIdx.x;
    if (bid % bEvery == bEvery - 1u) {
        const uint32_t eb = bid / bEvery;
        #pragma unroll
        for (uint32_t k = 0; k < BMULT; ++k)
            fused_round_body<BINW, BOUTW, BLEAFW, BLMODE, BLOUT, BPADN, BSIN, BSOUT,
                             BSBUILD, BINSTR, BOUTSTR, BFCAP, false>(
                shm.b, eb * BMULT + k, b.bucket_bits, b.submask_bits, b.in_bucket_cap,
                b.out_bucket_cap, b.out_off, b.in_counts, b.in_belem, b.out_counts,
                b.out_belem, b.all_left, b.all_right, b.gi_counter, b.drops, b.pp4);
    } else {
        fused_round_body<AINW, AOUTW, ALEAFW, ALMODE, ALOUT, APADN, ASIN, ASOUT, ASBUILD,
                         AINSTR, AOUTSTR, AFCAP, false>(
            shm.a, bid - bid / bEvery,      // rank among A blocks: B blocks before bid
            a.bucket_bits, a.submask_bits, a.in_bucket_cap,
            a.out_bucket_cap, a.out_off, a.in_counts, a.in_belem, a.out_counts,
            a.out_belem, a.all_left, a.all_right, a.gi_counter, a.drops, a.pp4);
    }
}

}} // namespace mxbm::cuda
