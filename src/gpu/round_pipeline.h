#pragma once
#include "gpu/cl_runtime.h"
#include "gpu/budget.h"
#include "gpu/rowbucket_geom.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <vector>

namespace mxbm { namespace gpu {

struct PipelineBuffers {
    uint32_t capacity = 0;      // elems per round (= budget.elems_per_round)
    // Compaction: when true, the sort path stores work at the significant-word
    // schedule [7,7,6,5,1] and dispatches the fixed-width mix/match variants
    // (round_mix_c*, round_match_sorted_*, survivor_scan_s). Set once at
    // alloc_pipeline from the active path so every kernel agrees on the stride;
    // white-box tests that hand-drive the legacy stride-7 kernels clear it.
    bool compact = false;
    // Fused row-bucket path selected for this allocation (see want_rowbucket in
    // round_pipeline.cpp: MXBM_ROWBUCKET forces on, MXBM_NO_ROWBUCKET forces off,
    // else auto-select when device memory clears the ~12 GB / 3.3 GB single-alloc
    // need). Recorded here so run_pipeline routes exactly as alloc_pipeline built.
    bool rowbucket = false;
    Mem work[2];                // ping-pong: round r reads work[r&1], writes work[(r+1)&1] (each
                                // capacity*7*8 B). Only 2 of the logical 6 round-slots are ever live
                                // at once, so 2 physical buffers suffice -- frees ~7.7 GB vs the old 6.
    Mem leaves[2];              // E3a: ping-pong AoS leaf prefixes (leaf i at slot*BH3_MAX_LEAVES+i),
                                // sized 9*capacity uint (max S). leaves[r&1] holds work[r]'s prefix.
    Mem left, right;            // consolidated: uint[5*capacity] each; round k at (k-1)*capacity
    // NOTE: there is no `lead` array. It used to be a third row of the same shape, but
    // an element's lead IS leaf 0 of its own prefix, so round_match stopped writing it
    // ("all_lead dropped") -- the allocation just outlived the write by several rounds
    // of cleanup. Worth 20 B/element on the sort path.
    Mem pp;                     // ulong[4] prePow. Persistent because the index-only
                                // round-1 match derives its parents and so needs the
                                // same prePow the seed kernel used; mix_seeds writes it.
    Mem bucket_count;           // uint[num_buckets]
    Mem bucket_slots;           // uint[num_buckets*slots_per_bucket]
    Mem counters;               // uint[4]: {out_count, bucket_drops, pair_drops, spare}

    // Phase-D3 sort-based collision finder scratch (allocated only when
    // MXBM_SORT_MATCH is set; see match_sorted() in round_pipeline.cpp). The
    // sort replaces scatter+match with a stable radix sort of (key,index) pairs
    // then a coalesced run-scan emit -- see kernels/opencl/sort.cl.
    Mem sort_pairs[2];          // ulong[capacity] ping-pong (key,index) pairs
    Mem sort_scan;              // uint[capacity] collision counts -> prefix-sum offsets
    Mem sort_hist;             // uint[256 * ceil(capacity/256)] radix bin-major histogram

    // P2c LDS-match scratch (allocated only under MXBM_LDS_MATCH): bucket-contiguous
    // storage for round_scatter_lds -> round_collide_lds (kernels/opencl/lds.cl).
    uint32_t lds_num_buckets = 0, lds_bucket_cap = 0;
    Mem lds_bwork;              // ulong[num_buckets*bucket_cap*7] bucketed work
    Mem lds_bslot;             // uint[num_buckets*bucket_cap] original slot
    Mem lds_blead;             // uint[num_buckets*bucket_cap] lead (first leaf)
    Mem lds_counts;            // uint[num_buckets] arrival counters

    // MXBM_ROWBUCKET fused row-bucket path (round_fused_lds). FAT bucket ping-pong:
    // the entry (round1_mix_scatter_fat) writes set 0; round r reads set S, emits
    // round-(r+1) children into set S^1 (bucketed by the child's freshly-mixed key),
    // then S^=1. Each element = work[7] + gi (dense per-level id: back-ref index +
    // (lead,gi) tiebreak) + lead + leaves[<=9] (materialized child prefix, carried
    // coalesced -- STEP A: fat wins +17 ms/solve). No flat work[]/leaves[]/sort
    // scratch is allocated on this path. left/right stay for recover (same row
    // convention as the sort path: round r fills row (r-1)*capacity, indexed by gi).
    // PACKED element record (see the layout note above FUSED_LDS in lds.cl): one
    // contiguous [work | meta=(gi<<32)|lead | leaf payload] block per element instead
    // of four parallel arrays, because a 4 B field in its own array costs a whole 32 B
    // write sector (measured 1.20x, test_emit_packing). Stride varies per round; the
    // buffer is sized for the widest round (fb_stride, = kFbMaxStride).
    uint32_t fb_num_buckets = 0, fb_bucket_cap = 0;
    uint32_t fb_submask_bits = 0;   // chosen with fb_num_buckets; see rb_pick_geometry

    bool fb_quad = false;            // 24 B quad record for r2 -> r3; see rowbucket_geom.h
    uint32_t fb_stride[2] = {0, 0};  // u64/element actually allocated PER SET (the two
                               // differ) -- read this rather than re-deriving the width
    Mem fb_elem[2];            // ulong[nb*cap*fb_stride]
    Mem fb_counts[2];          // uint[nb] arrival counters
    Mem fb_gictr;              // uint[1] per-round dense child-gi counter
};

// Per-round constants for the fused row-bucket kernels. These are BAKED IN at
// compile time in each FUSED_LDS instantiation in kernels/opencl/lds.cl -- making
// them compile-time (so apply_mix's tree loop unrolls and its t[8] stays in
// registers instead of local memory) is worth ~26 ms, but it means the kernels
// IGNORE the matching runtime arguments. tests/test_gpu_rounds.cpp pins this table
// against the literals in lds.cl so the two cannot drift apart silently.
// Note Lout(r) == Lmix(r+1) for every round, so one constant serves both roles.
struct FusedConsts { uint32_t Lout, padNext, sIn, sOut, sBuild; };
FusedConsts fused_consts_for(int r);

// Can this device host the fused row-bucket path at the given budget's capacity?
// Checks the real footprint against global memory AND the single-allocation limit,
// and honours the MXBM_ROWBUCKET / MXBM_NO_ROWBUCKET overrides. Exposed because the
// per-element budget differs per path, so the caller must know which path will run
// before it can size the seed layer (see kBytesPerElement* in budget.h).
bool rowbucket_viable(Runtime& rt, const Budget& b);

// RbGeometry, rb_geometry_for() and rowbucket_bytes() are declared in
// gpu/rowbucket_geom.h (included above) -- backend-neutral, because the CUDA solver
// makes the same decision from the same arithmetic and cannot see OpenCL's headers.

PipelineBuffers alloc_pipeline(Runtime& rt, const Budget& b);

// Round 1: seed + mix in one pass (round1_mix_seeds kernel), writing
// pb.work[1] at each index's ABSOLUTE slot. Drives the full [0,pb.capacity)
// range in b.seed_batch-sized batches.
void mix_seeds(Runtime& rt, PipelineBuffers& pb, const Budget& b, const uint64_t pp[4]);

// Rounds r=2..5: mix pb.work[r]'s N resident elements in place (round_mix
// kernel), reconstructing each element's first padNum(r) leaves via a
// back-ref DFS over pb.left/pb.right at level=r-1 -- the row where round r's
// own elements were written by round (r-1)'s combine step. padNum/Lmix are
// computed internally from r via the round table.
void mix_level(Runtime& rt, PipelineBuffers& pb, int r, uint32_t N);

// Scatter: radix-bucket pb.work[workIndex]'s N mixed elements by the top
// b.bucket_bits of their 24-bit collision key (round_scatter kernel), so the
// sortless all-pairs match (T4) only searches within a bucket. Zeroes
// pb.bucket_count and pb.counters via fill_u32 first; counters[1] accumulates
// bucket_drops -- elements whose bucket was already at b.slots_per_bucket
// capacity when they arrived.
void scatter(Runtime& rt, PipelineBuffers& pb, const Budget& b, uint32_t N, int workIndex);

// Per-round counts + per-phase GPU timing (ms). run1d() clFinish()es after
// every kernel, so a host-side steady_clock around each phase call is an
// accurate GPU time, not just an enqueue time. t_mix_ms is 0 for r==1 (its
// mix is round1_mix_seeds, timed as the pipeline's t_seed_ms instead) and for
// any round short-circuited by the prevN==0 empty-round path.
struct RoundStats {
    uint32_t in = 0, out = 0;
    uint32_t bucket_drops = 0, pair_drops = 0;
    double t_mix_ms = 0.0, t_scatter_ms = 0.0, t_match_ms = 0.0;
};

// Match: sortless all-pairs collision search within every scattered bucket
// (round_match kernel). Reads pb.work[r] (round r's fully mixed resident
// elements -- already scatter()'d into pb.bucket_count/pb.bucket_slots over
// the SAME buffer); for every colliding pair within a bucket, combines the
// (lead,slot)-ordered pair at Lout(r) and writes the child into pb.work[r+1]
// (round r's children become round r+1's own resident elements) plus a
// consolidated back-ref triple (left/right slot + lead) at
// pb.left/right/lead row (r-1)*capacity. r==5 has no round 6 to feed -- its
// children land in pb.work[0], the pipeline's only otherwise-idle work slot
// (mix_seeds writes seeds directly into pb.work[1], never uses work[0]);
// this keeps PipelineBuffers' shape untouched, deferring what (if anything)
// further consumes r==5's children to a later task. Zeroes counters[0]
// (out_count) and counters[2] (pair_drops) before launching -- counters[1]
// (scatter's bucket_drops) and [3] are left untouched; launches over
// b.num_buckets work-items; returns the capacity-clamped child count and
// reports the raw pair_drops (children beyond pb.capacity, silently dropped
// by the kernel) via the out-param. `inN` is accepted for interface symmetry
// with mix_level/scatter -- round_match itself iterates buckets (already
// sized by the prior scatter() call), not raw input elements.
uint32_t match(Runtime& rt, PipelineBuffers& pb, const Budget& b, int r, uint32_t inN, uint32_t& pair_drops);

// One full round: mix (r==1's mix -- round1_mix_seeds -- is the CALLER's
// responsibility, already landed in pb.work[1] before this runs; r>=2 mixes
// pb.work[r] in place via mix_level) -> scatter(pb.work[r]) -> match. Fills
// `stats` (in/out/bucket_drops/pair_drops) and returns the same child count
// as match().
uint32_t run_single_round(Runtime& rt, PipelineBuffers& pb, const Budget& b, int r, uint32_t inN, RoundStats& stats);

// Survivor scan: after round 5's match() writes its children into
// pb.work[0] (its r==5 special-case out_work -- see match()'s doc comment
// above), scan those N children for the ALL-ZERO work vector (survivor_scan
// kernel, kernels/opencl/round.cl) -- the deterministic signature of two
// byte-identical round-4 elements having collided (bh3_combine is XOR-based:
// a==b -> zero, at any Lout). Zeroes ONLY counters[3] (repurposed here as
// the survivor count) via a targeted read-modify-write first, mirroring
// match()'s own idiom for counters[0]/[2] -- else a second call on the same
// PipelineBuffers would accumulate onto a stale count; counters[0..2] are
// left untouched. Allocates its own `cap`-sized device buffer for the
// survivor slot list (a one-shot, end-of-run scan, not a per-round hot
// path, so this does not warrant a permanent PipelineBuffers member).
// Returns the capacity-clamped survivor count (== survivor_slots.size()
// after this call) and fills `survivor_slots` with the work[0] slot index
// of every survivor, read back from the GPU.
uint32_t survivor_scan(Runtime& rt, PipelineBuffers& pb, uint32_t N,
                        std::vector<uint32_t>& survivor_slots, uint32_t cap = 1024);

// Full 5-round pipeline: mix_seeds (the FULL [0,pb.capacity) seed range,
// batched by b.seed_batch) -> for r=1..5, run_single_round(r, prevN) with
// each round's outN threaded as the next round's inN -> survivor_scan over
// pb.work[0] (round 5's output) with N = round 5's child count. Empty-round
// robustness: once prevN reaches 0 (real searches decay to 0 well before
// round 5 -- see tests/test_round_pipeline.cpp section (b)), every
// remaining round is recorded as an honest {in=0,out=0,bucket_drops=0,
// pair_drops=0} WITHOUT dispatching a kernel -- a global_work_size==0
// launch is invalid OpenCL, and mix_level/scatter (T2/T3, unmodified) do
// not themselves guard against it. Logs one line per round
// ("  r%d: in=%u out=%u bucketDrops=%u pairDrops=%u\n") -- honest drop
// reporting, not just success counts.
//
// Abortable: `abort`, if non-null, is polled (relaxed load) after each
// round -- and so, since survivor_scan runs only once the loop finishes,
// also before survivor_scan. The instant it reads true, run_pipeline
// returns a default-constructed (empty) PipelineResult immediately --
// honest "found nothing" rather than a partial/truncated candidate set.
// Default nullptr, so every pre-existing caller (Phase B, C-T1) is
// unaffected -- this is purely additive.
//
// `verbose` (default true, so every pre-existing caller is unaffected)
// gates the per-round "  r%d: in=... out=... bucketDrops=... pairDrops=...\n"
// printf above: continuous production mining (GpuSolver::solve(), called
// once per attempted nonce) passes false to avoid spamming stdout every
// round of every solve() call; the test callers driving this directly
// (test_gpu_rounds.cpp, test_gpu_recover.cpp, test_round_pipeline.cpp) keep
// the default and still print, since their asserts are on the returned
// PipelineResult, not on stdout.
struct PipelineResult {
    uint32_t survivors = 0;
    std::vector<uint32_t> survivor_slots;
    RoundStats rounds[5];
    // GPU timing (ms) for the phases that live outside the per-round loop, plus
    // the whole-pipeline total (seed -> 5 rounds -> survivor_scan). Together
    // with rounds[*].t_*_ms this fully accounts for one solve's device time.
    double t_seed_ms = 0.0, t_survivor_ms = 0.0, t_total_ms = 0.0;
};
PipelineResult run_pipeline(Runtime& rt, PipelineBuffers& pb, const Budget& b, const uint64_t pp[4],
                             const std::atomic<bool>* abort = nullptr, bool verbose = true);

// Recover: for each survivor (a work[0] slot index from run_pipeline's
// survivor_scan), walk its consolidated back-ref ancestry on-device (the
// `recover` kernel, kernels/opencl/round.cl -- the SAME explicit-stack
// pre-order DFS round_mix uses, but from level 5 down to level 1, collecting
// all 32 leaves in tree order instead of feeding them to apply_mix) and pack
// each leaf set into a 104-byte candidate via bh3::pack_indices (src/
// beamhash/bh3_primitives.h, proven bit-identical in Phase A -- never
// reimplemented here). Only `survivor_slots.size()*32` uints are read back
// from the GPU -- never the multi-GB pb.left/right back-ref arrays
// themselves. soln[0..99] is the packed index bitfield; soln[100..103] is
// the extraNonce, left at 0 (a default-constructed std::array<uint8_t,104>
// value-initializes to all-zero, so this falls out for free -- see
// pack_indices, which only ever writes out[0..99]). No gate yet (e.g.
// distinct-leaf / non-overlapping-tree checks) -- these are raw candidates,
// deferred to a later Phase-C task.
std::vector<std::array<uint8_t,104>> recover_candidates(Runtime& rt, PipelineBuffers& pb,
                                                          const std::vector<uint32_t>& survivor_slots);

}} // namespace mxbm::gpu
