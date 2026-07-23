#include "gpu/round_pipeline.h"

namespace mxbm { namespace gpu {

PipelineBuffers alloc_pipeline(Runtime& rt, const Budget& b) {
    PipelineBuffers p;
    p.capacity = b.elems_per_round;

    // Each resident element is 7 x uint64 (56 B) of work state, matching
    // kSeedElemBytes; work[0] (transient seed stage) and work[1..5] (round
    // r=1..5 output) all share this per-element shape.
    const size_t elemBytes = (size_t)p.capacity * 7 * 8;
    for (int i = 0; i < 6; ++i)
        p.work[i] = rt.alloc(CL_MEM_READ_WRITE, elemBytes);

    // Consolidated flat back-ref arrays: 5 rounds x capacity slots, uint32 each.
    const size_t backrefBytes = (size_t)5 * p.capacity * 4;
    p.left  = rt.alloc(CL_MEM_READ_WRITE, backrefBytes);
    p.right = rt.alloc(CL_MEM_READ_WRITE, backrefBytes);
    p.lead  = rt.alloc(CL_MEM_READ_WRITE, backrefBytes);

    p.bucket_count = rt.alloc(CL_MEM_READ_WRITE, (size_t)b.num_buckets * 4);
    p.bucket_slots = rt.alloc(CL_MEM_READ_WRITE, (size_t)b.num_buckets * b.slots_per_bucket * 4);

    p.counters = rt.alloc(CL_MEM_READ_WRITE, 4 * 4);

    return p;
}

}} // namespace mxbm::gpu
