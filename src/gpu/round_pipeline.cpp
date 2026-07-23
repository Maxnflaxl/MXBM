#include "gpu/round_pipeline.h"
#include "generated/mxbm_kernels.h"
#include "beamhash/bh3_primitives.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace mxbm { namespace gpu {

namespace {
// GPU phase timing. run1d() clFinish()es after each kernel launch (see
// cl_runtime.cpp), so wall-clock around a phase call == that phase's device
// time; no CL event profiling needed.
using clk = std::chrono::steady_clock;
inline double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}
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
// E3a: leaves stored per work[r] element = its FULL pre-order leaf prefix
// (2^(r-1), capped by what the next round needs). r<=4 are full so a child's
// prefix = concat of its two parents' prefixes; work[5] needs only padNum(5)=9.
uint32_t sleaves_for(int r) {
    static const uint32_t T[5] = {1u, 2u, 4u, 8u, 9u};
    return T[r - 1];
}
uint32_t lout_for(int r) {
    static const uint32_t T[5] = {424u, 400u, 376u, 288u, 24u};
    return T[r - 1];
}
} // namespace

PipelineBuffers alloc_pipeline(Runtime& rt, const Budget& b) {
    PipelineBuffers p;
    // Buffers/out_capacity are sized to b.capacity (= seed count + headroom) so a
    // round's genuine collision count, which fluctuates a few thousand above the
    // 2^25 seed count, is never clamped (see Budget::capacity). b.capacity == 0
    // (synthetic tests that hand-build a Budget) falls back to elems_per_round --
    // those tests supply their own non-overflowing data, so no headroom is needed.
    p.capacity = b.capacity != 0 ? b.capacity : b.elems_per_round;

    // Each resident element is 7 x uint64 (56 B) of work state, matching
    // kSeedElemBytes. PING-PONG: round r reads work[r&1] and writes work[(r+1)&1];
    // r==5 writes work[0] (== work[(5+1)&1]). Only 2 logical round-slots are live
    // at any instant (round r's input and its output) -- round r's input is dead
    // the moment match(r) finishes, so its buffer is reused as round r+1's output.
    // mix_seeds writes work[1]; survivor_scan/recover read the r5 output in work[0].
    const size_t elemBytes = (size_t)p.capacity * 7 * 8;
    for (int i = 0; i < 2; ++i)
        p.work[i] = rt.alloc(CL_MEM_READ_WRITE, elemBytes);

    // E3a leaf prefixes: ping-pong SoA, sized to the max prefix (9 leaves/elem at
    // work[5]). ~9*capacity*4 = 1.25 GB each; the 6->2 work-buffer reclaim leaves
    // ample room. leaves[r&1] holds work[r]'s prefix (mirrors the work ping-pong).
    const size_t leavesBytes = (size_t)9 * p.capacity * sizeof(uint32_t);
    for (int i = 0; i < 2; ++i)
        p.leaves[i] = rt.alloc(CL_MEM_READ_WRITE, leavesBytes);

    // Consolidated flat back-ref arrays: 5 rounds x capacity slots, uint32 each.
    const size_t backrefBytes = (size_t)5 * p.capacity * 4;
    p.left  = rt.alloc(CL_MEM_READ_WRITE, backrefBytes);
    p.right = rt.alloc(CL_MEM_READ_WRITE, backrefBytes);
    p.lead  = rt.alloc(CL_MEM_READ_WRITE, backrefBytes);

    p.bucket_count = rt.alloc(CL_MEM_READ_WRITE, (size_t)b.num_buckets * 4);
    p.bucket_slots = rt.alloc(CL_MEM_READ_WRITE, (size_t)b.num_buckets * b.slots_per_bucket * 4);

    p.counters = rt.alloc(CL_MEM_READ_WRITE, 4 * 4);

    // Phase-D3 sort scratch, allocated only when the sort-based match is enabled
    // (env MXBM_SORT_MATCH). ~512 MB (pairs) + 128 MB (hist) at 2^25 capacity,
    // comfortably inside the headroom the 6->2 work-buffer reclaim freed. Left
    // null otherwise so the default path costs nothing extra.
    if (std::getenv("MXBM_SORT_MATCH")) {
        const size_t pairBytes = (size_t)p.capacity * sizeof(uint64_t);
        for (int i = 0; i < 2; ++i) p.sort_pairs[i] = rt.alloc(CL_MEM_READ_WRITE, pairBytes);
        p.sort_scan = rt.alloc(CL_MEM_READ_WRITE, (size_t)p.capacity * 4);
        const size_t ngroups = ((size_t)p.capacity + 255) / 256;
        p.sort_hist = rt.alloc(CL_MEM_READ_WRITE, (size_t)256 * ngroups * 4);
    }

    return p;
}

void mix_seeds(Runtime& rt, PipelineBuffers& pb, const Budget& b, const uint64_t pp[4]) {
    cl_program prog = rt.cached_program({std::string(kBh3ClSource), std::string(kRoundClSource)}, "");
    Kernel k = rt.kernel(prog, "round1_mix_seeds");

    uint64_t pp4[4] = { pp[0], pp[1], pp[2], pp[3] };
    Mem mp = rt.alloc(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof pp4, pp4);
    cl_mem ppMem   = mp.get();
    cl_mem workMem = pb.work[1].get();
    cl_mem leavesMem = pb.leaves[1].get();   // leaves for work[1] (== leaves[1&1])
    uint32_t capacity = pb.capacity;

    // Generate exactly the SEED COUNT (elems_per_round, 2^25 on a full-search
    // card), NOT pb.capacity -- the buffers are larger than the seed count by
    // the collision headroom, and BeamHash III has exactly 2^25 seed indices.
    const uint32_t total = (b.elems_per_round != 0) ? b.elems_per_round : pb.capacity;
    const uint32_t batch = (b.seed_batch != 0) ? b.seed_batch : total;
    for (uint32_t begin = 0; begin < total; begin += batch) {
        uint32_t count = (total - begin < batch) ? (total - begin) : batch;
        rt.set_arg(k.get(), 0, sizeof(cl_mem), &ppMem);
        rt.set_arg(k.get(), 1, begin);
        rt.set_arg(k.get(), 2, count);
        rt.set_arg(k.get(), 3, sizeof(cl_mem), &workMem);
        rt.set_arg(k.get(), 4, sizeof(cl_mem), &leavesMem);
        rt.set_arg(k.get(), 5, capacity);
        rt.run1d(k.get(), count);
    }
}

void mix_level(Runtime& rt, PipelineBuffers& pb, int r, uint32_t N) {
    cl_program prog = rt.cached_program({std::string(kBh3ClSource), std::string(kRoundClSource)}, "");
    Kernel k = rt.kernel(prog, "round_mix");

    uint32_t capacity = pb.capacity;
    uint32_t padNum   = padnum_for(r);
    uint32_t Lmix     = lmix_for(r);
    cl_mem workMem   = pb.work[r & 1].get();
    cl_mem leavesMem = pb.leaves[r & 1].get();   // E3a: work[r]'s materialized prefix

    rt.set_arg(k.get(), 0, N);
    rt.set_arg(k.get(), 1, capacity);
    rt.set_arg(k.get(), 2, padNum);
    rt.set_arg(k.get(), 3, Lmix);
    rt.set_arg(k.get(), 4, sizeof(cl_mem), &workMem);
    rt.set_arg(k.get(), 5, sizeof(cl_mem), &leavesMem);
    rt.run1d(k.get(), N);
}

void scatter(Runtime& rt, PipelineBuffers& pb, const Budget& b, uint32_t N, int workIndex) {
    // Zero bucket_count (arrival counters) and counters (incl. counters[1] =
    // bucket_drops) before every scatter pass -- round_scatter's atomic_inc
    // accumulates onto whatever is already there.
    rt.fill_u32(pb.bucket_count.get(), 0u, b.num_buckets);
    rt.fill_u32(pb.counters.get(), 0u, 4);

    cl_program prog = rt.cached_program({std::string(kBh3ClSource), std::string(kRoundClSource)}, "");
    Kernel k = rt.kernel(prog, "round_scatter");

    uint32_t bucketBits = b.bucket_bits;
    uint32_t slots       = b.slots_per_bucket;
    cl_mem workMem        = pb.work[workIndex & 1].get();
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

    cl_program prog = rt.cached_program({std::string(kBh3ClSource), std::string(kRoundClSource)}, "");
    Kernel k = rt.kernel(prog, "round_match");

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
    cl_mem inWorkMem      = pb.work[r & 1].get();
    cl_mem outWorkMem     = pb.work[((r < (int)kNumRounds) ? r + 1 : 0) & 1].get();
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

    // LOCAL-MEMORY BUCKET STAGING (Phase-D winner): one workgroup per bucket
    // cooperatively stages this bucket's `slots` keys + element indices into two
    // dynamic __local uint arrays (args 16,17) with a coalesced pass, then the W
    // work-items split the O(k^2) all-pairs scan (row i owned by work-item i%W)
    // entirely out of __local. Split width W is set to ~mean bucket occupancy so
    // there is about one outer row per work-item with minimal idle lanes: D0
    // found W=mean (32) beats the naive W=slots (64), which left half the group
    // idle since mean k = elems/num_buckets ~ 32 << slots. global = num_buckets*W
    // is a whole multiple of W as OpenCL requires.
    const DeviceInfo& dev = rt.device();

    // Guard the dynamic __local request (two uint arrays of `slots`) against the
    // device's per-workgroup local memory. At slots 64 this is 512 B vs a 48 KB
    // limit -- trivially safe -- but a hand-built Budget with huge slots must
    // fail loudly here rather than launch with a silently-clamped shared buffer.
    const size_t localBytes = (size_t)slots * sizeof(uint32_t) * 2u;
    if (dev.local_mem != 0 && localBytes > dev.local_mem)
        throw ClError(CL_INVALID_WORK_GROUP_SIZE,
                      "round_match staging: __local " + std::to_string(localBytes) +
                      " B exceeds device local mem " + std::to_string(dev.local_mem) + " B");

    uint32_t mean = (b.num_buckets != 0u) ? (b.elems_per_round / b.num_buckets) : slots;
    uint32_t W = (mean != 0u) ? mean : 1u;
    if (W > slots) W = slots;                                  // never exceed staged slots
    if (dev.max_work_group != 0 && W > dev.max_work_group)     // respect device group cap
        W = (uint32_t)dev.max_work_group;
    if (W == 0u) W = 1u;

    rt.set_arg(k.get(), 16, (size_t)slots * sizeof(uint32_t), nullptr);
    rt.set_arg(k.get(), 17, (size_t)slots * sizeof(uint32_t), nullptr);

    // E3a: grow leaf prefixes. Parents live in work[r] -> leaves[r&1]; children go
    // to work[r+1] -> leaves[(r+1)&1]. s_out==0 for r==5 (its output isn't mixed).
    cl_mem leavesInMem  = pb.leaves[r & 1].get();
    cl_mem leavesOutMem = pb.leaves[(r + 1) & 1].get();
    uint32_t sIn  = sleaves_for(r);
    uint32_t sOut = (r < (int)kNumRounds) ? sleaves_for(r + 1) : 0u;
    rt.set_arg(k.get(), 18, sizeof(cl_mem), &leavesInMem);
    rt.set_arg(k.get(), 19, sizeof(cl_mem), &leavesOutMem);
    rt.set_arg(k.get(), 20, sIn);
    rt.set_arg(k.get(), 21, sOut);

    rt.run1d(k.get(), (size_t)b.num_buckets * W, W);

    rt.read(pb.counters.get(), sizeof counters, counters);
    pair_drops = counters[2];
    return (counters[0] < outCapacity) ? counters[0] : outCapacity;
}

// ===========================================================================
// Phase-D3: sort-based collision finder (host driver for kernels/opencl/sort.cl).
// Enabled per-solve by env MXBM_SORT_MATCH; replaces scatter()+match() with
// key_extract -> stable radix sort -> collision run-scan -> coalesced emit.
// ===========================================================================
namespace {
constexpr uint32_t kSortWG = 256u;

cl_program sort_prog(Runtime& rt) {
    return rt.cached_program({std::string(kBh3ClSource), std::string(kRoundClSource),
                              std::string(kSortClSource)}, "");
}

// Generic exclusive prefix scan of `n` uints in `buf` (in place), via
// scan_block + recursion on the per-block sums + scan_add (see sort.cl).
void scan_exclusive(Runtime& rt, cl_program prog, cl_mem buf, uint32_t n) {
    if (n == 0) return;
    uint32_t nblocks = (n + kSortWG - 1) / kSortWG;
    Mem blocksums = rt.alloc(CL_MEM_READ_WRITE, (size_t)nblocks * 4);
    cl_mem bs = blocksums.get();
    Kernel kb = rt.kernel(prog, "scan_block");
    rt.set_arg(kb.get(), 0, sizeof(cl_mem), &buf);
    rt.set_arg(kb.get(), 1, sizeof(cl_mem), &buf);
    rt.set_arg(kb.get(), 2, sizeof(cl_mem), &bs);
    rt.set_arg(kb.get(), 3, n);
    rt.run1d(kb.get(), (size_t)nblocks * kSortWG, kSortWG);
    if (nblocks > 1) {
        scan_exclusive(rt, prog, bs, nblocks);
        Kernel ka = rt.kernel(prog, "scan_add");
        rt.set_arg(ka.get(), 0, sizeof(cl_mem), &buf);
        rt.set_arg(ka.get(), 1, sizeof(cl_mem), &bs);
        rt.set_arg(ka.get(), 2, n);
        rt.run1d(ka.get(), (size_t)nblocks * kSortWG, kSortWG);
    }
}

// Stable LSD radix sort of `n` (key,index) pairs by the low 24 bits (3 x 8-bit
// passes). Ping-pongs pb.sort_pairs[0]<->[1]; caller must have loaded the input
// into pairs[0]. Returns the cl_mem holding the sorted result (pairs[1] after 3
// swaps). Uses pb.sort_hist for the per-pass bin-major histogram.
cl_mem radix_sort_pairs(Runtime& rt, cl_program prog, PipelineBuffers& pb, uint32_t n) {
    uint32_t ngroups = (n + kSortWG - 1) / kSortWG;
    cl_mem cur = pb.sort_pairs[0].get(), other = pb.sort_pairs[1].get(), wh = pb.sort_hist.get();
    Kernel kh = rt.kernel(prog, "radix_histogram");
    Kernel kr = rt.kernel(prog, "radix_reorder");
    for (uint32_t pass = 0; pass < 3; ++pass) {
        uint32_t shift = pass * 8;
        rt.set_arg(kh.get(), 0, sizeof(cl_mem), &cur);
        rt.set_arg(kh.get(), 1, sizeof(cl_mem), &wh);
        rt.set_arg(kh.get(), 2, shift);
        rt.set_arg(kh.get(), 3, n);
        rt.set_arg(kh.get(), 4, ngroups);
        rt.run1d(kh.get(), (size_t)ngroups * kSortWG, kSortWG);

        scan_exclusive(rt, prog, wh, 256u * ngroups);

        rt.set_arg(kr.get(), 0, sizeof(cl_mem), &cur);
        rt.set_arg(kr.get(), 1, sizeof(cl_mem), &other);
        rt.set_arg(kr.get(), 2, sizeof(cl_mem), &wh);
        rt.set_arg(kr.get(), 3, shift);
        rt.set_arg(kr.get(), 4, n);
        rt.set_arg(kr.get(), 5, ngroups);
        rt.run1d(kr.get(), (size_t)ngroups * kSortWG, kSortWG);

        cl_mem t = cur; cur = other; other = t;
    }
    return cur;
}
} // namespace

// Sort-based replacement for scatter()+match(): identical child multiset (same
// equal-24-bit-key collisions, same (lead,slot)-canonical children), just emitted
// coalesced from sorted runs instead of scattered from atomic buckets. `bucket_
// drops` is always 0 (the sort caps nothing per key); pair_drops reports children
// beyond pb.capacity, same as match(). Returns the capacity-clamped child count.
uint32_t match_sorted(Runtime& rt, PipelineBuffers& pb, const Budget& b, int r,
                      uint32_t N, uint32_t& pair_drops) {
    (void)b;
    cl_program prog = sort_prog(rt);
    uint32_t outCapacity = pb.capacity;

    // 1. key_extract: build (key,index) pairs into sort_pairs[0].
    {
        Kernel k = rt.kernel(prog, "key_extract");
        cl_mem inWork = pb.work[r & 1].get();
        cl_mem pairs0 = pb.sort_pairs[0].get();
        rt.set_arg(k.get(), 0, N);
        rt.set_arg(k.get(), 1, sizeof(cl_mem), &inWork);
        rt.set_arg(k.get(), 2, sizeof(cl_mem), &pairs0);
        rt.run1d(k.get(), N);
    }

    // 2. stable radix sort by 24-bit key.
    cl_mem sorted = radix_sort_pairs(rt, prog, pb, N);

    // 3. collision_count over the sorted runs -> sort_scan.
    cl_mem scan = pb.sort_scan.get();
    {
        Kernel k = rt.kernel(prog, "collision_count");
        rt.set_arg(k.get(), 0, sizeof(cl_mem), &sorted);
        rt.set_arg(k.get(), 1, sizeof(cl_mem), &scan);
        rt.set_arg(k.get(), 2, N);
        rt.run1d(k.get(), (size_t)((N + kSortWG - 1) / kSortWG) * kSortWG, kSortWG);
    }
    // total children = exclusive-offset(last) + count(last). Grab count(last)
    // BEFORE the in-place scan clobbers it (one 4-byte read), then offset(last).
    uint32_t lastCount = 0;
    rt.read_at(scan, (size_t)(N - 1) * 4, 4, &lastCount);
    scan_exclusive(rt, prog, scan, N);
    uint32_t lastOff = 0;
    rt.read_at(scan, (size_t)(N - 1) * 4, 4, &lastOff);
    uint32_t total = lastOff + lastCount;

    // 4. emit. Zero counters[2] (pair_drops) first; leave the rest as-is.
    uint32_t counters[4];
    rt.read(pb.counters.get(), sizeof counters, counters);
    counters[2] = 0u;
    rt.write(pb.counters.get(), sizeof counters, counters);

    {
        Kernel k = rt.kernel(prog, "round_match_sorted");
        uint32_t Lout = lout_for(r);
        uint32_t leadIdentity = (r == 1) ? 1u : 0u;
        uint32_t outOff = (uint32_t)(r - 1) * pb.capacity;
        cl_mem inWork  = pb.work[r & 1].get();
        cl_mem outWork = pb.work[((r < (int)kNumRounds) ? r + 1 : 0) & 1].get();
        cl_mem leftMem = pb.left.get(), rightMem = pb.right.get(), cnt = pb.counters.get();
        cl_mem leavesIn = pb.leaves[r & 1].get(), leavesOut = pb.leaves[(r + 1) & 1].get();
        uint32_t sIn = sleaves_for(r);
        uint32_t sOut = (r < (int)kNumRounds) ? sleaves_for(r + 1) : 0u;
        rt.set_arg(k.get(), 0, N);
        rt.set_arg(k.get(), 1, Lout);
        rt.set_arg(k.get(), 2, leadIdentity);
        rt.set_arg(k.get(), 3, outOff);
        rt.set_arg(k.get(), 4, outCapacity);
        rt.set_arg(k.get(), 5, sizeof(cl_mem), &sorted);
        rt.set_arg(k.get(), 6, sizeof(cl_mem), &scan);
        rt.set_arg(k.get(), 7, sizeof(cl_mem), &inWork);
        rt.set_arg(k.get(), 8, sizeof(cl_mem), &outWork);
        rt.set_arg(k.get(), 9, sizeof(cl_mem), &leftMem);
        rt.set_arg(k.get(), 10, sizeof(cl_mem), &rightMem);
        rt.set_arg(k.get(), 11, sizeof(cl_mem), &cnt);
        rt.set_arg(k.get(), 12, sizeof(cl_mem), &leavesIn);
        rt.set_arg(k.get(), 13, sizeof(cl_mem), &leavesOut);
        rt.set_arg(k.get(), 14, sIn);
        rt.set_arg(k.get(), 15, sOut);
        rt.run1d(k.get(), (size_t)((N + kSortWG - 1) / kSortWG) * kSortWG, kSortWG);
    }

    rt.read(pb.counters.get(), sizeof counters, counters);
    pair_drops = counters[2];
    return (total < outCapacity) ? total : outCapacity;
}

uint32_t run_single_round(Runtime& rt, PipelineBuffers& pb, const Budget& b, int r, uint32_t inN, RoundStats& stats) {
    // Phase-D3: route to the sort-based collision finder when enabled. Mix is
    // unchanged (still E3a leaf-prefix mix); only scatter+match is replaced.
    static const bool useSort = (std::getenv("MXBM_SORT_MATCH") != nullptr);
    if (useSort) {
        auto tMixS = clk::now();
        if (r >= 2) mix_level(rt, pb, r, inN);
        stats.t_mix_ms = (r >= 2) ? ms_since(tMixS) : 0.0;
        auto tMatchS = clk::now();
        uint32_t pd = 0;
        uint32_t outNs = match_sorted(rt, pb, b, r, inN, pd);
        stats.t_match_ms = ms_since(tMatchS);
        stats.t_scatter_ms = 0.0;
        stats.in = inN; stats.out = outNs; stats.bucket_drops = 0u; stats.pair_drops = pd;
        return outNs;
    }

    auto tMix = clk::now();
    if (r >= 2) mix_level(rt, pb, r, inN);
    stats.t_mix_ms = (r >= 2) ? ms_since(tMix) : 0.0;

    auto tScatter = clk::now();
    scatter(rt, pb, b, inN, /*workIndex=*/r);
    stats.t_scatter_ms = ms_since(tScatter);

    uint32_t counters[4];
    rt.read(pb.counters.get(), sizeof counters, counters);
    uint32_t bucket_drops = counters[1];

    auto tMatch = clk::now();
    uint32_t pair_drops = 0;
    uint32_t outN = match(rt, pb, b, r, inN, pair_drops);
    stats.t_match_ms = ms_since(tMatch);

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

    cl_program prog = rt.cached_program({std::string(kBh3ClSource), std::string(kRoundClSource)}, "");
    Kernel k = rt.kernel(prog, "survivor_scan");

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

PipelineResult run_pipeline(Runtime& rt, PipelineBuffers& pb, const Budget& b, const uint64_t pp[4],
                             const std::atomic<bool>* abort, bool verbose) {
    PipelineResult result;
    auto tPipeline = clk::now();

    auto tSeed = clk::now();
    mix_seeds(rt, pb, b, pp);
    result.t_seed_ms = ms_since(tSeed);

    // Round 1 processes exactly the SEED COUNT (elems_per_round), not pb.capacity
    // -- mix_seeds populated work[1][0, seed_count); the buffers' headroom tail
    // [seed_count, capacity) is unwritten and unread here (only rounds 1-2's
    // OUTPUTS grow into it, which is the whole point of the headroom).
    uint32_t prevN = (b.elems_per_round != 0) ? b.elems_per_round : pb.capacity;
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
        if (verbose) {
            std::printf("  r%d: in=%u out=%u bucketDrops=%u pairDrops=%u | mix=%.1f scatter=%.1f match=%.1f ms\n",
                        r, st.in, st.out, st.bucket_drops, st.pair_drops,
                        st.t_mix_ms, st.t_scatter_ms, st.t_match_ms);
        }

        // Abort check: after every round (so, for r==5, also strictly
        // before survivor_scan below). An honest empty result -- no
        // partial/truncated candidate set is ever produced from a search
        // that was cut short.
        if (abort && abort->load(std::memory_order_relaxed)) return PipelineResult{};
    }

    auto tSurvivor = clk::now();
    result.survivors = survivor_scan(rt, pb, prevN, result.survivor_slots);
    result.t_survivor_ms = ms_since(tSurvivor);
    result.t_total_ms = ms_since(tPipeline);

    if (verbose) {
        double sps = result.t_total_ms > 0.0 ? 1000.0 / result.t_total_ms : 0.0;
        std::printf("  [pipeline] seed=%.1f survivor=%.1f total=%.1f ms  ->  %.2f solve/s\n",
                    result.t_seed_ms, result.t_survivor_ms, result.t_total_ms, sps);
    }
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

    cl_program prog = rt.cached_program({std::string(kBh3ClSource), std::string(kRoundClSource)}, "");
    Kernel k = rt.kernel(prog, "recover");

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
