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

}} // namespace mxbm::gpu
