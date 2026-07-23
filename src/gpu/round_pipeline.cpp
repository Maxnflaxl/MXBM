#include "gpu/round_pipeline.h"
#include "generated/mxbm_kernels.h"
#include <string>

namespace mxbm { namespace gpu {

namespace {
// Round table (r=1..5). Mirrors tests/round_ref.h's ref::Lmix/ref::padNum
// formulas (pinned equal by tests/test_round_reconstruct.cpp's
// test_round_table) -- copied here as literal values since round_pipeline.cpp
// is shipped code and must not depend on a test header.
uint32_t lmix_for(int r) {
    static const uint32_t T[5] = {448u, 424u, 400u, 376u, 288u};
    return T[r - 1];
}
uint32_t padnum_for(int r) {
    static const uint32_t T[5] = {1u, 2u, 4u, 6u, 9u};
    return T[r - 1];
}
} // namespace

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

void mix_seeds(Runtime& rt, PipelineBuffers& pb, const Budget& b, const uint64_t pp[4]) {
    Program prog = rt.build({std::string(kBh3ClSource), std::string(kRoundClSource)}, "");
    Kernel k = rt.kernel(prog.get(), "round1_mix_seeds");

    uint64_t pp4[4] = { pp[0], pp[1], pp[2], pp[3] };
    Mem mp = rt.alloc(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof pp4, pp4);
    cl_mem ppMem   = mp.get();
    cl_mem workMem = pb.work[1].get();

    const uint32_t total = pb.capacity;
    const uint32_t batch = (b.seed_batch != 0) ? b.seed_batch : total;
    for (uint32_t begin = 0; begin < total; begin += batch) {
        uint32_t count = (total - begin < batch) ? (total - begin) : batch;
        rt.set_arg(k.get(), 0, sizeof(cl_mem), &ppMem);
        rt.set_arg(k.get(), 1, begin);
        rt.set_arg(k.get(), 2, count);
        rt.set_arg(k.get(), 3, sizeof(cl_mem), &workMem);
        rt.run1d(k.get(), count);
    }
}

void mix_level(Runtime& rt, PipelineBuffers& pb, int r, uint32_t N) {
    Program prog = rt.build({std::string(kBh3ClSource), std::string(kRoundClSource)}, "");
    Kernel k = rt.kernel(prog.get(), "round_mix");

    uint32_t level    = (uint32_t)(r - 1);
    uint32_t capacity = pb.capacity;
    uint32_t padNum   = padnum_for(r);
    uint32_t Lmix     = lmix_for(r);
    cl_mem workMem  = pb.work[r].get();
    cl_mem leftMem  = pb.left.get();
    cl_mem rightMem = pb.right.get();

    rt.set_arg(k.get(), 0, level);
    rt.set_arg(k.get(), 1, N);
    rt.set_arg(k.get(), 2, capacity);
    rt.set_arg(k.get(), 3, padNum);
    rt.set_arg(k.get(), 4, Lmix);
    rt.set_arg(k.get(), 5, sizeof(cl_mem), &workMem);
    rt.set_arg(k.get(), 6, sizeof(cl_mem), &leftMem);
    rt.set_arg(k.get(), 7, sizeof(cl_mem), &rightMem);
    rt.run1d(k.get(), N);
}

}} // namespace mxbm::gpu
