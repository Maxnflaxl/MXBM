#pragma once
#include "gpu/cl_runtime.h"
#include "gpu/budget.h"

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

}} // namespace mxbm::gpu
