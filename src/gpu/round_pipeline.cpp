#include "gpu/round_pipeline.h"
#include "generated/mxbm_kernels.h"
#include "beamhash/bh3_primitives.h"
#include <cstdio>
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
uint32_t lout_for(int r) {
    static const uint32_t T[5] = {424u, 400u, 376u, 288u, 24u};
    return T[r - 1];
}
} // namespace

PipelineBuffers alloc_pipeline(Runtime& rt, const Budget& b) {
    PipelineBuffers p;
    p.capacity = b.elems_per_round;

    // Each resident element is 7 x uint64 (56 B) of work state, matching
    // kSeedElemBytes; all 6 slots (work[0..5]) share this per-element shape.
    // Usage: mix_seeds() writes round 1's mixed input into work[1]; for
    // round r, match() reads round r's input from work[r] and writes round
    // r's children (round r+1's input) into work[r+1] -- except r==5, which
    // has no round 6 to feed and writes its children into work[0], the only
    // otherwise-idle slot (mix_seeds never touches work[0]).
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

void scatter(Runtime& rt, PipelineBuffers& pb, const Budget& b, uint32_t N, int workIndex) {
    // Zero bucket_count (arrival counters) and counters (incl. counters[1] =
    // bucket_drops) before every scatter pass -- round_scatter's atomic_inc
    // accumulates onto whatever is already there.
    rt.fill_u32(pb.bucket_count.get(), 0u, b.num_buckets);
    rt.fill_u32(pb.counters.get(), 0u, 4);

    Program prog = rt.build({std::string(kBh3ClSource), std::string(kRoundClSource)}, "");
    Kernel k = rt.kernel(prog.get(), "round_scatter");

    uint32_t bucketBits = b.bucket_bits;
    uint32_t slots       = b.slots_per_bucket;
    cl_mem workMem        = pb.work[workIndex].get();
    cl_mem bucketCountMem = pb.bucket_count.get();
    cl_mem bucketSlotsMem = pb.bucket_slots.get();
    cl_mem countersMem    = pb.counters.get();

    rt.set_arg(k.get(), 0, N);
    rt.set_arg(k.get(), 1, bucketBits);
    rt.set_arg(k.get(), 2, slots);
    rt.set_arg(k.get(), 3, sizeof(cl_mem), &workMem);
    rt.set_arg(k.get(), 4, sizeof(cl_mem), &bucketCountMem);
    rt.set_arg(k.get(), 5, sizeof(cl_mem), &bucketSlotsMem);
    rt.set_arg(k.get(), 6, sizeof(cl_mem), &countersMem);
    rt.run1d(k.get(), N);
}

uint32_t match(Runtime& rt, PipelineBuffers& pb, const Budget& b, int r, uint32_t inN, uint32_t& pair_drops) {
    (void)inN;  // round_match iterates buckets (already sized by scatter()), not raw elements.

    // Zero only counters[0] (out_count) and counters[2] (pair_drops); leave
    // counters[1] (bucket_drops, owned by scatter) and [3] (spare) as-is.
    uint32_t counters[4];
    rt.read(pb.counters.get(), sizeof counters, counters);
    counters[0] = 0u;
    counters[2] = 0u;
    rt.write(pb.counters.get(), sizeof counters, counters);

    Program prog = rt.build({std::string(kBh3ClSource), std::string(kRoundClSource)}, "");
    Kernel k = rt.kernel(prog.get(), "round_match");

    uint32_t Lout         = lout_for(r);
    uint32_t bucketBits   = b.bucket_bits;
    uint32_t slots        = b.slots_per_bucket;
    uint32_t leadIdentity = (r == 1) ? 1u : 0u;
    uint32_t inLeadOff    = (r >= 2) ? (uint32_t)(r - 2) * pb.capacity : 0u;
    uint32_t outOff       = (uint32_t)(r - 1) * pb.capacity;
    uint32_t outCapacity  = pb.capacity;

    // Round r's own resident elements live in pb.work[r]; its children
    // become round r+1's resident elements, in pb.work[r+1] -- except r==5,
    // which has no round 6 and reuses pb.work[0] (see match()'s doc comment
    // in round_pipeline.h).
    cl_mem bucketCountMem = pb.bucket_count.get();
    cl_mem bucketSlotsMem = pb.bucket_slots.get();
    cl_mem inWorkMem      = pb.work[r].get();
    cl_mem outWorkMem     = pb.work[(r < (int)kNumRounds) ? r + 1 : 0].get();
    cl_mem leadMem        = pb.lead.get();   // aliased: all_lead_in (read) + all_lead (write) -- disjoint row ranges per round
    cl_mem leftMem        = pb.left.get();
    cl_mem rightMem       = pb.right.get();
    cl_mem countersMem    = pb.counters.get();

    rt.set_arg(k.get(), 0, Lout);
    rt.set_arg(k.get(), 1, bucketBits);
    rt.set_arg(k.get(), 2, slots);
    rt.set_arg(k.get(), 3, leadIdentity);
    rt.set_arg(k.get(), 4, inLeadOff);
    rt.set_arg(k.get(), 5, outOff);
    rt.set_arg(k.get(), 6, outCapacity);
    rt.set_arg(k.get(), 7, sizeof(cl_mem), &bucketCountMem);
    rt.set_arg(k.get(), 8, sizeof(cl_mem), &bucketSlotsMem);
    rt.set_arg(k.get(), 9, sizeof(cl_mem), &inWorkMem);
    rt.set_arg(k.get(), 10, sizeof(cl_mem), &outWorkMem);
    rt.set_arg(k.get(), 11, sizeof(cl_mem), &leadMem);
    rt.set_arg(k.get(), 12, sizeof(cl_mem), &leftMem);
    rt.set_arg(k.get(), 13, sizeof(cl_mem), &rightMem);
    rt.set_arg(k.get(), 14, sizeof(cl_mem), &leadMem);
    rt.set_arg(k.get(), 15, sizeof(cl_mem), &countersMem);

    rt.run1d(k.get(), b.num_buckets);

    rt.read(pb.counters.get(), sizeof counters, counters);
    pair_drops = counters[2];
    return (counters[0] < outCapacity) ? counters[0] : outCapacity;
}

uint32_t run_single_round(Runtime& rt, PipelineBuffers& pb, const Budget& b, int r, uint32_t inN, RoundStats& stats) {
    if (r >= 2) mix_level(rt, pb, r, inN);
    scatter(rt, pb, b, inN, /*workIndex=*/r);

    uint32_t counters[4];
    rt.read(pb.counters.get(), sizeof counters, counters);
    uint32_t bucket_drops = counters[1];

    uint32_t pair_drops = 0;
    uint32_t outN = match(rt, pb, b, r, inN, pair_drops);

    stats.in = inN;
    stats.out = outN;
    stats.bucket_drops = bucket_drops;
    stats.pair_drops = pair_drops;
    return outN;
}

uint32_t survivor_scan(Runtime& rt, PipelineBuffers& pb, uint32_t N,
                        std::vector<uint32_t>& survivor_slots, uint32_t cap) {
    survivor_slots.clear();
    // A global_work_size==0 launch is invalid OpenCL; 0 candidates -> 0
    // survivors, trivially, with no GPU round-trip needed.
    if (N == 0) return 0u;

    // Zero only counters[3] (repurposed here as the survivor count); leave
    // [0..2] (owned by scatter()/match()) untouched -- same read-modify-write
    // idiom match() already uses for its own slots.
    uint32_t counters[4];
    rt.read(pb.counters.get(), sizeof counters, counters);
    counters[3] = 0u;
    rt.write(pb.counters.get(), sizeof counters, counters);

    Program prog = rt.build({std::string(kBh3ClSource), std::string(kRoundClSource)}, "");
    Kernel k = rt.kernel(prog.get(), "survivor_scan");

    Mem outSlots = rt.alloc(CL_MEM_READ_WRITE, (size_t)cap * 4);
    // Round 5's children live in pb.work[0] (match()'s r==5 special case --
    // CORRECTION 1 vs the brief's original text, which said work[5]).
    cl_mem workMem     = pb.work[0].get();
    cl_mem outSlotsMem = outSlots.get();
    cl_mem countersMem = pb.counters.get();

    rt.set_arg(k.get(), 0, N);
    rt.set_arg(k.get(), 1, sizeof(cl_mem), &workMem);
    rt.set_arg(k.get(), 2, sizeof(cl_mem), &outSlotsMem);
    rt.set_arg(k.get(), 3, cap);
    rt.set_arg(k.get(), 4, sizeof(cl_mem), &countersMem);
    rt.run1d(k.get(), N);

    rt.read(pb.counters.get(), sizeof counters, counters);
    uint32_t survivors = counters[3];
    uint32_t clamped = (survivors < cap) ? survivors : cap;
    survivor_slots.resize(clamped);
    if (clamped > 0) rt.read(outSlots.get(), (size_t)clamped * 4, survivor_slots.data());
    return clamped;
}

PipelineResult run_pipeline(Runtime& rt, PipelineBuffers& pb, const Budget& b, const uint64_t pp[4]) {
    PipelineResult result;
    mix_seeds(rt, pb, b, pp);

    uint32_t prevN = pb.capacity;   // mix_seeds writes the FULL [0,capacity) range into work[1]
    for (int r = 1; r <= 5; ++r) {
        RoundStats& st = result.rounds[r - 1];
        if (prevN == 0) {
            // Empty-round robustness: once the chain runs dry, every
            // remaining round is honestly {0,0,0,0} -- 0 input can never
            // produce a collision, and dispatching mix_level/scatter with
            // N==0 would be an invalid (global_work_size==0) OpenCL launch.
            st = RoundStats{};
        } else {
            prevN = run_single_round(rt, pb, b, r, prevN, st);
        }
        std::printf("  r%d: in=%u out=%u bucketDrops=%u pairDrops=%u\n",
                    r, st.in, st.out, st.bucket_drops, st.pair_drops);
    }

    result.survivors = survivor_scan(rt, pb, prevN, result.survivor_slots);
    return result;
}

std::vector<std::array<uint8_t,104>> recover_candidates(Runtime& rt, PipelineBuffers& pb,
                                                          const std::vector<uint32_t>& survivor_slots) {
    const size_t n = survivor_slots.size();
    // Value-initialized: every byte starts at 0, so soln[100..103] (the
    // extraNonce) is 0 without touching it below -- pack_indices only ever
    // writes out[0..99].
    std::vector<std::array<uint8_t,104>> out(n);
    // A global_work_size==0 launch is invalid OpenCL; 0 survivors -> 0
    // candidates, trivially, with no GPU round-trip needed (mirrors
    // survivor_scan's own N==0 guard).
    if (n == 0) return out;

    Program prog = rt.build({std::string(kBh3ClSource), std::string(kRoundClSource)}, "");
    Kernel k = rt.kernel(prog.get(), "recover");

    Mem slotsMem  = rt.alloc(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             n * sizeof(uint32_t), (void*)survivor_slots.data());
    Mem leavesMem = rt.alloc(CL_MEM_READ_WRITE, n * 32 * sizeof(uint32_t));

    uint32_t nSurv    = (uint32_t)n;
    uint32_t capacity = pb.capacity;
    cl_mem slotsRaw  = slotsMem.get();
    cl_mem leftMem   = pb.left.get();
    cl_mem rightMem  = pb.right.get();
    cl_mem leavesRaw = leavesMem.get();

    rt.set_arg(k.get(), 0, nSurv);
    rt.set_arg(k.get(), 1, sizeof(cl_mem), &slotsRaw);
    rt.set_arg(k.get(), 2, capacity);
    rt.set_arg(k.get(), 3, sizeof(cl_mem), &leftMem);
    rt.set_arg(k.get(), 4, sizeof(cl_mem), &rightMem);
    rt.set_arg(k.get(), 5, sizeof(cl_mem), &leavesRaw);
    rt.run1d(k.get(), n);

    // Read back ONLY n*32 uints (the recovered leaves) -- never the
    // multi-GB pb.left/right consolidated back-ref arrays themselves.
    std::vector<uint32_t> leaves(n * 32);
    rt.read(leavesMem.get(), leaves.size() * sizeof(uint32_t), leaves.data());

    for (size_t i = 0; i < n; ++i)
        bh3::pack_indices(&leaves[i * 32], out[i].data());

    return out;
}

}} // namespace mxbm::gpu
