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

}} // namespace mxbm::gpu
