#pragma once
#include "gpu/cl_runtime.h"
#include "gpu/budget.h"
#include <array>
#include <cstdint>
#include <vector>

namespace mxbm { namespace gpu {

struct PipelineBuffers {
    uint32_t capacity = 0;      // elems per round (= budget.elems_per_round)
    Mem work[6];                // [0]=transient seed work; [1..5]=round r work (each capacity*7*8 B)
    Mem left, right, lead;      // consolidated: uint[5*capacity] each; round k at (k-1)*capacity
    Mem bucket_count;           // uint[num_buckets]
    Mem bucket_slots;           // uint[num_buckets*slots_per_bucket]
    Mem counters;               // uint[4]: {out_count, bucket_drops, pair_drops, spare}
};

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

struct RoundStats { uint32_t in = 0, out = 0; uint32_t bucket_drops = 0, pair_drops = 0; };

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
struct PipelineResult {
    uint32_t survivors = 0;
    std::vector<uint32_t> survivor_slots;
    RoundStats rounds[5];
};
PipelineResult run_pipeline(Runtime& rt, PipelineBuffers& pb, const Budget& b, const uint64_t pp[4]);

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
