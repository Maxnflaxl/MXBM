#include "gpu/round_pipeline.h"
#include "generated/mxbm_kernels.h"
#include "beamhash/bh3_primitives.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
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
// COMPACTION significant-word schedule (== ceil(Lout/64)). A round's OUTPUT child
// carries outwords_for(r) meaningful u64; its INPUT parents carry inwords_for(r) =
// outwords_for(r-1) (round 1's seeds are full 7). Words above these are 0 (masked
// by the prior round's bh3_combine), so storing/reading only the significant words
// is bit-exact. Used to pick the fixed-width match variant + strided mix per round.
uint32_t outwords_for(int r) {
    static const uint32_t T[5] = {7u, 7u, 6u, 5u, 1u};
    return T[r - 1];
}
uint32_t inwords_for(int r) {
    static const uint32_t T[5] = {7u, 7u, 7u, 6u, 5u};
    return T[r - 1];
}
// Compacted sort path: fixed-width mix and match variants shrink the dominant
// gather/emit to the [7,7,6,5,1] significant-word schedule -- measured
// 224->214 ms (~4.4%), byte-identical goldens.

// Row-bucket geometry. bucketBits sets the bucket count; submaskBits sets how many
// times each bucket is rescanned by the staging loop (2^submaskBits) and hence the
// group size (mean bucket / 2^submaskBits), which LDS_FCAP must cover *including its
// tail*. Halving the bucket and the sub-mask together keeps the group size -- and so
// the LDS footprint -- fixed while halving the rescan.
// Two constraints. The staging loop rescans each bucket 2^submaskBits times, so
// submaskBits should be as small as possible; but the group it stages is
// mean_bucket / 2^submaskBits, and that wants to be ~256 -- LDS_FCAP=384 must cover its
// tail, and a group much *below* the 256-thread workgroup leaves lanes idle. Together
// those pin bucketBits + submaskBits ~ 17, and then the smallest submaskBits wins.
//
// Measured (all drop-free unless noted):
//     (14,3) 47.5 ms 8x rescan   (15,2) 42.6 ms 4x   (16,1) 40.4 ms 2x   <- here
//     (17,0) 40.4 ms 1x but 8.35 GiB -- the rescan has stopped mattering by then
//     (16,2) 46.6, (17,1) 45.4 -- group falls to 132, half the workgroup idle
//     (18,0) single allocation exceeds the device limit -> sort-path fallback
//
// (16,1) only became reachable once fb_cap_for stopped over-reserving: at the old
// capacity its single allocation was 4.3 GB, past the device's 3.9 GB limit.
uint32_t rb_bucket_bits() {
    static uint32_t v = std::getenv("MXBM_BB") ? (uint32_t)atoi(std::getenv("MXBM_BB")) : 16u;
    return v;
}
uint32_t rb_submask_bits() {
    static uint32_t v = std::getenv("MXBM_SM") ? (uint32_t)atoi(std::getenv("MXBM_SM")) : 1u;
    return v;
}

// Extra OpenCL build options for the fused row-bucket program (MXBM_CL_OPTS).
// The fused kernels are LDS-bound to 1 workgroup/SM, so the compiler's default
// register budget -- chosen for an occupancy this kernel can never reach -- is
// far tighter than necessary and spills the re-derivation's live state.
const char* rowbucket_cl_opts() {
    static const char* v = std::getenv("MXBM_CL_OPTS");
    return v ? v : "";
}
} // namespace

// Mirrors the literals baked into the FUSED_LDS instantiations in lds.cl; pinned by
// tests/test_gpu_rounds.cpp (test_fused_consts_match_kernels).
FusedConsts fused_consts_for(int r) {
    switch (r) {
        case 1:  return {424u, 2u, 1u, 2u, 2u};
        case 2:  return {400u, 4u, 2u, 4u, 4u};
        case 3:  return {376u, 6u, 4u, 2u, 8u};   // builds 8 leaves, stores the contrib
        default: return {288u, 9u, 2u, 0u, 0u};   // r4: contrib in, nothing out
    }
}

// Packed row-bucket element strides (u64 per element), indexed by the round that
// READS them. Record = [work: inwords | meta: 1 | leaf payload: ceil(uints/2)]:
//   r1 THIN (index,key) = 1   r2 PAIR = 2   r3 7+2=9 (no meta)   r4 6+1+1=8   r5 1+1=2
// Round r's OUTPUT stride is round r+1's input stride. Rounds 1 and 2 both read
// RE-DERIVABLE records rather than stored work state:
//   r1 reads a bare 8 B (index,key)      -- seed re-derived in-kernel (2.5 vs 18.2 ms)
//   r2 reads a 16 B (key,left,right,gi)  -- rebuilt from its two parent seeds
// Both derivations live in the kernel's non-divergent expand loop; done in the
// sub-mask-filtered staging loop instead, round 2's cost 23.5 ms rather than 0.4.
// kFbStride, fb_set_stride, fb_cap_for, rowbucket_bytes and rb_geometry_for now live in
// gpu/rowbucket_geom.h -- the CUDA backend picks its geometry from the same arithmetic,
// and it cannot include this file (OpenCL headers). See there for the sigma reasoning
// and the measured ladder.

// Device-memory-aware default: MXBM_ROWBUCKET forces the fused path on (even if it
// might not fit -- the user's explicit choice), MXBM_NO_ROWBUCKET forces it off, and
// otherwise auto-select when BOTH the largest single alloc clears max_alloc AND the
// total leaves ~1 GB headroom under global_mem. Falls back to the sort path on any
// device that can't hold the ~12 GB working set (8-12 GB cards). capacity==0 (some
// synthetic Budgets) only honors an explicit force. MXBM_BB / MXBM_SM override.
// `quad` rides along because it is a second axis of the same ladder, not a tuning
// knob: it changes the record format, hence the strides, hence the allocation.
struct RbGeom { uint32_t bb, sm; bool quad; bool arena; bool octo; };

// The IMPLICIT-BITS record follows from the geometry (rb_impb_ok), not from a rung of
// its own. MXBM_NO_IMPB turns it off on both sides at once -- the arithmetic that sizes
// the set and the kernels that read it -- so the two cannot disagree.
static bool impb_allowed() {
    static const bool off = std::getenv("MXBM_NO_IMPB") != nullptr;
    return !off;
}

// `usable` is the budget's free-derived figure (driver free less gpu/budget.h's
// reserve), which is what rowbucket_viable checks against; the ladder has to answer
// against the same number or the two disagree and the caller falls back to the sort
// path. The slack goes to 0 with it -- the reserve has already come off, and a flat
// gigabyte on top would be held back twice. 0 means the caller has no free figure.
static RbGeom rb_pick_geometry(Runtime& rt, uint32_t capacity, uint64_t usable) {
    if (std::getenv("MXBM_BB") || std::getenv("MXBM_SM")) {
        // MXBM_BB alone used to leave sm at its default of 1, putting the geometry OFF
        // the bb + sm == 17 line every compile-time kernel constant assumes -- and the
        // OpenCL path, unlike CUDA's, did not check: `MXBM_BB=15` silently produced
        // wrong results (0/3 goldens) rather than refusing. Derive sm from bb the way
        // CudaSolver::pick_geometry does, and let an explicit MXBM_SM override it.
        const uint32_t bb = rb_bucket_bits();
        const uint32_t sm = std::getenv("MXBM_SM") ? rb_submask_bits() : (17u - bb);
        return { bb, sm, std::getenv("MXBM_QUAD") != nullptr,
                 std::getenv("MXBM_ARENA") != nullptr,
                 std::getenv("MXBM_OCTO") != nullptr };
    }
    const DeviceInfo& d = rt.device();
    // allow_quad: the OpenCL row-bucket kernels now carry the 24 B quad record
    // (round_fused_rd2q / round_fused_rd3), so this backend can reach the rungs that
    // used to be CUDA-only. It matters far more here than on CUDA: the quad record
    // takes the largest single allocation from 2.76 to 2.45 GiB, which is under the
    // CL_DEVICE_MAX_MEM_ALLOC_SIZE of an 11 GB card -- a whole device class that was
    // falling back to the ~5x slower sort path.
    // allow_split: sets over CL_DEVICE_MAX_MEM_ALLOC_SIZE go in as bucket-halves
    // (fb_elem_hi) -- what took the OpenCL floor from 11 GB to CUDA's ~5.7 GiB.
    const uint64_t mem   = usable ? usable : (uint64_t)d.global_mem;
    const uint64_t slack = usable ? 0ull   : ((uint64_t)1 << 30);
    // allow_arena: dense per-bucket caps plus an overflow pool per record set
    // (LDS_ARENA in lds.cl). A rung, not a default -- it costs time for ~22 % of the
    // slots, so the ladder only reaches for it when the rows above do not fit.
    const RbGeometry g = rb_geometry_for(capacity, d.max_alloc, mem,
                                         /*allow_quad=*/true, 0, /*allow_split=*/true,
                                         slack, /*allow_impb=*/impb_allowed(),
                                         /*allow_arena=*/true,
                                         // allow_octo: round 3's record as its eight
                                         // leaves, round 4 rebuilding from them. The
                                         // bottom three rungs, and the 3-4 GB classes.
                                         /*allow_octo=*/true);
    // MXBM_QUAD / MXBM_ARENA force a format a card would not otherwise need, which is
    // the only way to exercise those paths on a 16 GB device.
    return { g.bb, g.sm, g.quad || std::getenv("MXBM_QUAD") != nullptr,
             g.arena || std::getenv("MXBM_ARENA") != nullptr,
             g.octo  || std::getenv("MXBM_OCTO")  != nullptr };
}

bool rowbucket_viable(Runtime& rt, const Budget& b) {
    if (std::getenv("MXBM_NO_ROWBUCKET")) return false;
    const bool forced = (std::getenv("MXBM_ROWBUCKET") != nullptr);
    if (forced) return true;
    // An explicit alternate collision path (legacy / LDS) opts out of the auto
    // default -- those paths read the flat work[] layout the row-bucket alloc omits.
    if (std::getenv("MXBM_LEGACY_MATCH") || std::getenv("MXBM_LDS_MATCH")) return false;
    uint32_t capacity = b.capacity != 0 ? b.capacity : b.elems_per_round;
    if (capacity == 0) return false;
    const DeviceInfo& d = rt.device();
    if (d.global_mem == 0 || d.max_alloc == 0) return false;    // unknown -> sort (safe)
    const RbGeom g = rb_pick_geometry(rt, capacity, b.usable);
    size_t total = 0, single = 0;
    const bool impb = impb_allowed() && rb_impb_ok(g.bb, g.sm, g.quad);
    rowbucket_bytes(capacity, g.bb, total, single, g.quad, impb, g.arena, g.octo);
    // max_alloc binds on the split figure when smaller. Checked here too, not just
    // in rb_geometry_for: MXBM_BB/MXBM_SM bypass the ladder yet still get the split.
    size_t bind = single;
    { const size_t s = rowbucket_single_split(capacity, g.bb, g.quad, impb, g.arena, g.octo);
      if (s < bind) bind = s; }
    if (bind > (size_t)d.max_alloc) return false;
    // Against the budget's usable figure -- free VRAM less the reserve (gpu/budget.h)
    // -- not the card's total, which says nothing about what is available today. The
    // reserve is the whole of the slack, so nothing further is held back here.
    const uint64_t mem_limit = b.usable ? b.usable : (uint64_t)d.global_mem;
    if (total > (size_t)mem_limit) return false;
    return true;
}

// Fused row-bucket allocation: FAT bucket ping-pong + left/right for recover. No
// flat work[]/leaves[]/sort scratch (the buckets ARE the resident storage). Sized
// so bucketDrops==0 at 2^25 (cap ~1.75x mean).
static void try_alloc_rowbucket(Runtime& rt, PipelineBuffers& p,
                                uint32_t bb, uint32_t sm, bool quad, bool arena = false,
                                bool octo = false) {
    p.fb_num_buckets  = 1u << bb;
    p.fb_submask_bits = sm;
    p.fb_bucket_cap = fb_cap_for(p.capacity / p.fb_num_buckets,
                                 arena ? kRbDenseSigma : kRbCapSigma);
    p.fb_quad = quad;
    p.fb_arena = arena;
    p.fb_octo = octo;
    p.fb_impb = impb_allowed() && rb_impb_ok(bb, sm, quad);
    // The pool lives BEHIND the bucket records in the record buffer, and the packed
    // record's side plane has none -- so a packed arena rung is only legal once the
    // implicit-bits pack has folded the plane away. Refused here rather than at first
    // use, because a ClError from this function is what alloc_rowbucket steps down on.
    if (arena && !quad && !p.fb_impb)
        throw ClError(CL_INVALID_VALUE,
                      "the overflow arena needs the quad record or the implicit-bits "
                      "pack: the packed r2->r3 side plane carries no pool region");
    // The pool is split between the two halves, so the total matches what the footprint
    // arithmetic budgeted whether or not the set ends up split.
    p.fb_arena_cap = arena ? kRbArenaSlots / 2 : 0u;
    const size_t nslots = (size_t)p.fb_num_buckets * p.fb_bucket_cap;
    const uint64_t maxAlloc = rt.device().max_alloc;
    const bool forceSplit = std::getenv("MXBM_SPLIT") != nullptr;
    for (int i = 0; i < 2; ++i) {
        p.fb_stride[i] = fb_set_stride((int)i, quad, p.fb_impb, octo);
        // Packed set 0 allocates stride-8 records; the 9th word (r3_p1) lives in
        // fb_side so every record access is 16 B aligned. Slot cost stays 9 u64
        // (rowbucket_bytes' figure), split 8 + 1. Under the implicit-bits pack the
        // plane has no writer and fb_set_stride already says 8.
        if (i == 0 && !quad) p.fb_stride[i] = 8u;
        // Each buffer carries its half's pool region behind the bucket records, which
        // is what lets one slot index address both.
        const size_t pool = (size_t)p.fb_arena_cap * p.fb_stride[i] * 8;
        const size_t bytes = nslots * p.fb_stride[i] * 8 + pool;
        // A set the driver's single-allocation cap refuses whole goes in as two
        // bucket-halves; the kernels select by child bucket (fb_half). MXBM_SPLIT
        // forces it, which is how the 16 GB dev rig exercises this path at all.
        if (forceSplit || (maxAlloc != 0 && bytes > (size_t)maxAlloc)) {
            const size_t halfBytes = (nslots / 2) * p.fb_stride[i] * 8 + pool;
            p.fb_elem[i]    = rt.alloc(CL_MEM_READ_WRITE, halfBytes);
            p.fb_elem_hi[i] = rt.alloc(CL_MEM_READ_WRITE, halfBytes);
            p.fb_half[i]    = p.fb_num_buckets / 2;
        } else {
            p.fb_elem[i]    = rt.alloc(CL_MEM_READ_WRITE, bytes);
            p.fb_elem_hi[i] = Mem();
            p.fb_half[i]    = p.fb_num_buckets;
        }
        p.fb_counts[i] = rt.alloc(CL_MEM_READ_WRITE, (size_t)p.fb_num_buckets * 4);
        if (arena) {
            p.fb_ahead[i] = rt.alloc(CL_MEM_READ_WRITE, (size_t)p.fb_num_buckets * 4);
            p.fb_anext[i] = rt.alloc(CL_MEM_READ_WRITE, (size_t)p.fb_arena_cap * 2 * 4);
            p.fb_atag[i]  = rt.alloc(CL_MEM_READ_WRITE, (size_t)p.fb_arena_cap * 2 * 4);
            p.fb_actr[i]  = rt.alloc(CL_MEM_READ_WRITE, 2 * 4);
        }
    }
    if (arena) { p.fb_spill = rt.alloc(CL_MEM_READ_WRITE, 4); rt.fill_u32(p.fb_spill.get(), 0u, 1); }
    p.fb_gictr = rt.alloc(CL_MEM_READ_WRITE, 4);
    if (!quad && !p.fb_impb) p.fb_side = rt.alloc(CL_MEM_READ_WRITE, nslots * 8);
    // Consolidated back-ref rows for recover. Rows 1-4 are gi-indexed and need full
    // capacity; row 5 is written only by the terminal round at SURVIVOR indices
    // (round5_fused_lds stores at out_off + si, si < the 1024 survivor cap), so a
    // capacity-sized row 5 reserved ~264 MB for <= 8 KB of use. The sort path's rows
    // stay 5 x capacity: its round-5 match writes per-child at gi.
    // Rows 1-3 are neither written nor read on an octo build: recovery reaches the
    // leaves through round 3's surviving output, so the walk starts at row 4.
    const size_t backrefBytes = ((size_t)(octo ? 1u : 4u) * p.capacity + 1024) * 4;
    p.left  = rt.alloc(CL_MEM_READ_WRITE, backrefBytes);
    p.right = rt.alloc(CL_MEM_READ_WRITE, backrefBytes);
    p.counters = rt.alloc(CL_MEM_READ_WRITE, 4 * 4);
    // clCreateBuffer is lazy on NVIDIA: a rung the card cannot really hold fails at
    // first USE, long after the arithmetic said yes. Touch every multi-GB allocation
    // so the commit happens HERE, where alloc_rowbucket's ladder can still step down.
    auto touch = [&](const Mem& m, size_t bytes) { if (m) rt.fill_u32(m.get(), 0u, bytes / 4); };
    for (int i = 0; i < 2; ++i) {
        const size_t nsl = (p.fb_half[i] == p.fb_num_buckets) ? nslots : nslots / 2;
        const size_t b = nsl * p.fb_stride[i] * 8 + (size_t)p.fb_arena_cap * p.fb_stride[i] * 8;
        touch(p.fb_elem[i], b);
        touch(p.fb_elem_hi[i], b);
    }
    touch(p.fb_side, nslots * 8);
    touch(p.left, backrefBytes); touch(p.right, backrefBytes);
}

static void release_rowbucket(PipelineBuffers& p) {
    for (int i = 0; i < 2; ++i) {
        p.fb_elem[i] = Mem(); p.fb_elem_hi[i] = Mem(); p.fb_counts[i] = Mem();
        p.fb_half[i] = 0; p.fb_stride[i] = 0;
        p.fb_ahead[i] = Mem(); p.fb_anext[i] = Mem();
        p.fb_atag[i] = Mem(); p.fb_actr[i] = Mem();
    }
    p.fb_arena = false; p.fb_arena_cap = 0; p.fb_spill = Mem(); p.fb_impb = false;
    p.fb_gictr = Mem(); p.fb_side = Mem();
    p.left = Mem(); p.right = Mem(); p.counters = Mem();
    p.spec_elem = Mem(); p.spec_counts = Mem(); p.spec_on = false;
}

// Speculative-entry buffers, allocated only once the ladder has SETTLED. Throughput is
// a luxury; reach is not. Sizing them into the rung choice would let a card lose a rung
// for them -- a measured ~1 ms of geometry for a hoped-for fraction of one. Touched so
// a lazy allocation cannot fail later, during mining.
static void alloc_spec_entry(Runtime& rt, PipelineBuffers& p) {
    if (std::getenv("MXBM_NO_SPEC")) return;
    // Not on an arena rung. The spec set is its own buffer with no pool region behind
    // it, so at the dense cap a full bucket there would have nowhere to append; giving
    // it a pool would cost reach on exactly the cards that took this rung to get it.
    if (p.fb_arena) return;
    const size_t nslots = (size_t)p.fb_num_buckets * p.fb_bucket_cap;
    try {
        p.spec_elem   = rt.alloc(CL_MEM_READ_WRITE, nslots * 8);
        p.spec_counts = rt.alloc(CL_MEM_READ_WRITE, (size_t)p.fb_num_buckets * 4);
        rt.fill_u32(p.spec_elem.get(), 0u, nslots * 2);
        rt.fill_u32(p.spec_counts.get(), 0u, p.fb_num_buckets);
        p.spec_on = true;
    } catch (const ClError&) {
        p.spec_elem = Mem(); p.spec_counts = Mem(); p.spec_on = false;
    }
}

static void alloc_rowbucket(Runtime& rt, const Budget& b, PipelineBuffers& p) {
    const RbGeom g = rb_pick_geometry(rt, p.capacity, b.usable);
    // An explicitly forced geometry or record format is the user's choice: allocate it
    // as asked and never step down. Without this the ladder re-walk below would look
    // for a rung matching the forced combination, fail to find one, and quietly hand
    // back the top rung -- which reads as "the flag did nothing".
    if (std::getenv("MXBM_BB") || std::getenv("MXBM_SM") ||
        std::getenv("MXBM_QUAD") || std::getenv("MXBM_ARENA") ||
        std::getenv("MXBM_OCTO")) {
        try_alloc_rowbucket(rt, p, g.bb, g.sm, g.quad, g.arena, g.octo);
        return;
    }
    // The allocator can refuse a rung the arithmetic accepted (a compositor's
    // gigabyte is invisible in global_mem): step DOWN, as CudaSolver does.
    int n = 0;
    const RbRung* rungs = rb_rungs(n);
    int start = 0;
    for (int i = 0; i < n; ++i)
        if (rungs[i].bb == g.bb && rungs[i].sm == g.sm && rungs[i].quad == g.quad
            && rungs[i].arena == g.arena && rungs[i].octo == g.octo) { start = i; break; }
    for (int i = start; i < n; ++i) {
        try {
            try_alloc_rowbucket(rt, p, rungs[i].bb, rungs[i].sm, rungs[i].quad,
                                rungs[i].arena, rungs[i].octo);
            if (i != start)
                std::fprintf(stderr, "[mxbm] row-bucket rung (%u,%u)%s refused by the "
                             "allocator; stepped down to (%u,%u)%s\n",
                             rungs[start].bb, rungs[start].sm, rungs[start].quad ? " quad" : "",
                             rungs[i].bb, rungs[i].sm, rungs[i].quad ? " quad" : "");
            // Silent on the top rung, which is what every large card takes; below it
            // the rung is what decides the card's speed.
            else if (i != 0) {
                size_t tot = 0, sng = 0;
                rowbucket_bytes(p.capacity, rungs[i].bb, tot, sng, rungs[i].quad,
                                false, rungs[i].arena, rungs[i].octo);
                std::fprintf(stderr, "[mxbm] row-bucket rung (%u,%u)%s%s -- %.2f GiB, "
                             "chosen for %.2f GiB of usable VRAM\n",
                             rungs[i].bb, rungs[i].sm, rungs[i].quad ? " quad" : "",
                             rungs[i].arena ? (rungs[i].octo ? " arena octo" : " arena") : "",
                             tot / 1073741824.0, (double)b.usable / 1073741824.0);
            }
            return;
        } catch (const ClError&) {
            release_rowbucket(p);
        }
    }
    throw ClError(CL_MEM_OBJECT_ALLOCATION_FAILURE,
                  "no row-bucket rung fits at allocation time (VRAM held by other "
                  "processes?); free device memory and retry");
}

PipelineBuffers alloc_pipeline(Runtime& rt, const Budget& b) {
    PipelineBuffers p;
    if (rowbucket_viable(rt, b)) {
        p.capacity = b.capacity != 0 ? b.capacity : b.elems_per_round;
        p.rowbucket = true;
        alloc_rowbucket(rt, b, p);
        alloc_spec_entry(rt, p);
        return p;
    }
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

    // E3a leaf prefixes: ping-pong AoS (leaf i of elem g at g*BH3_MAX_LEAVES+i),
    // sized to the max prefix (9 leaves/elem at work[5]). ~9*capacity*4 = 1.25 GB
    // each; the 6->2 work-buffer reclaim leaves ample room. leaves[r&1] holds
    // work[r]'s prefix (mirrors the work ping-pong).
    const size_t leavesBytes = (size_t)9 * p.capacity * sizeof(uint32_t);
    for (int i = 0; i < 2; ++i)
        p.leaves[i] = rt.alloc(CL_MEM_READ_WRITE, leavesBytes);

    // Consolidated flat back-ref arrays: 5 rounds x capacity slots, uint32 each.
    const size_t backrefBytes = (size_t)5 * p.capacity * 4;
    p.left  = rt.alloc(CL_MEM_READ_WRITE, backrefBytes);
    p.right = rt.alloc(CL_MEM_READ_WRITE, backrefBytes);

    p.bucket_count = rt.alloc(CL_MEM_READ_WRITE, (size_t)b.num_buckets * 4);
    p.bucket_slots = rt.alloc(CL_MEM_READ_WRITE, (size_t)b.num_buckets * b.slots_per_bucket * 4);

    p.counters = rt.alloc(CL_MEM_READ_WRITE, 4 * 4);

    // Phase-D3 sort scratch. Always allocated: mix_seeds/round_mix now fuse the
    // (key,index) sort pair into sort_pairs[0] (no separate key_extract pass), so
    // even the legacy path writes it. ~512 MB (pairs) + 256 MB (hist+scan) at 2^25
    // capacity, comfortably inside the headroom the 6->2 work-buffer reclaim freed.
    {
        const size_t pairBytes = (size_t)p.capacity * sizeof(uint64_t);
        for (int i = 0; i < 2; ++i) p.sort_pairs[i] = rt.alloc(CL_MEM_READ_WRITE, pairBytes);
        p.sort_scan = rt.alloc(CL_MEM_READ_WRITE, (size_t)p.capacity * 4);
        const size_t ngroups = ((size_t)p.capacity + 255) / 256;
        p.sort_hist = rt.alloc(CL_MEM_READ_WRITE, (size_t)256 * ngroups * 4);
    }

    // P2c LDS-match bucket storage (only under MXBM_LDS_MATCH). bucket_bits=14 ->
    // 16384 buckets; cap ~1.5x mean so bucketDrops==0; each holds 7-word work +
    // slot + lead. ~2.8 GB at 2^25; a sub-mask of 3 keeps per-group survivors
    // (~256) well inside LDS_ECAP.
    if (std::getenv("MXBM_LDS_MATCH")) {
        p.lds_num_buckets = 1u << 14;
        uint32_t mean = p.capacity / p.lds_num_buckets;
        p.lds_bucket_cap = mean + (mean >> 1) + 512;
        const size_t nslots = (size_t)p.lds_num_buckets * p.lds_bucket_cap;
        p.lds_bwork  = rt.alloc(CL_MEM_READ_WRITE, nslots * 7 * 8);
        p.lds_bslot  = rt.alloc(CL_MEM_READ_WRITE, nslots * 4);
        p.lds_blead  = rt.alloc(CL_MEM_READ_WRITE, nslots * 4);
        p.lds_counts = rt.alloc(CL_MEM_READ_WRITE, (size_t)p.lds_num_buckets * 4);
    }

    return p;
}

// Index-only round 1. Opt-in while it is being measured; the seed kernel and the round-1
// match read this same function so they cannot disagree.
bool idxonly_r1() { static bool v = (std::getenv("MXBM_IDXONLY") != nullptr); return v; }

void mix_seeds(Runtime& rt, PipelineBuffers& pb, const Budget& b, const uint64_t pp[4]) {
    cl_program prog = rt.cached_program({std::string(kBh3ClSource), std::string(kRoundClSource)}, "");
    // Index-only round 1: the seed kernel stores only the sort pair, and the match that
    // consumes it derives both parents. The two MUST switch together -- see
    // idxonly_r1() -- or the match reads a work buffer nobody wrote.
    Kernel k = rt.kernel(prog, idxonly_r1() ? "round1_mix_seeds_idx" : "round1_mix_seeds");

    uint64_t pp4[4] = { pp[0], pp[1], pp[2], pp[3] };
    // Kept on the buffers rather than local to this call: the index-only round-1 match
    // derives its parents and needs the SAME prePow this kernel seeded from.
    pb.pp = rt.alloc(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof pp4, pp4);
    cl_mem ppMem   = pb.pp.get();
    cl_mem workMem = pb.work[1].get();
    cl_mem leavesMem = pb.leaves[1].get();   // leaves for work[1] (== leaves[1&1])
    cl_mem pairsMem = pb.sort_pairs[0].get();   // D3: fused (key,index) sort pairs
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
        rt.set_arg(k.get(), 6, sizeof(cl_mem), &pairsMem);
        rt.run1d(k.get(), count);
    }
}

void mix_level(Runtime& rt, PipelineBuffers& pb, int r, uint32_t N) {
    cl_program prog = rt.cached_program({std::string(kBh3ClSource), std::string(kRoundClSource)}, "");
    uint32_t capacity = pb.capacity;
    uint32_t padNum   = padnum_for(r);
    uint32_t Lmix     = lmix_for(r);
    cl_mem workMem   = pb.work[r & 1].get();
    cl_mem leavesMem = pb.leaves[r & 1].get();   // E3a: work[r]'s materialized prefix
    cl_mem pairsMem  = pb.sort_pairs[0].get();   // D3: fused (key,index) sort pairs

    // Fixed-width in-place mix by round: fully-constant variants (INW *and* padNum
    // *and* Lmix baked in) -- see ROUND_MIX_K in round.cl for why, and for the 27 ms
    // a runtime Lmix was costing round 4.
    static const char* kMixK[6] = { "", "", "round_mix_k2", "round_mix_k3",
                                    "round_mix_k4", "round_mix_k5" };
    Kernel k = rt.kernel(prog, kMixK[r]);
    rt.set_arg(k.get(), 0, N);
    rt.set_arg(k.get(), 1, capacity);
    rt.set_arg(k.get(), 2, padNum);
    rt.set_arg(k.get(), 3, Lmix);
    rt.set_arg(k.get(), 4, sizeof(cl_mem), &workMem);
    rt.set_arg(k.get(), 5, sizeof(cl_mem), &leavesMem);
    rt.set_arg(k.get(), 6, sizeof(cl_mem), &pairsMem);
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
    // round_match still declares all_lead_in / all_lead but reads and writes neither
    // (see round.cl: "all_lead dropped ... in_lead_off/all_lead_in unused now"), so
    // both are bound to `left` rather than keeping a 5*capacity array alive for them.
    cl_mem leadMem        = pb.left.get();
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
// The DEFAULT collision path (MXBM_LEGACY_MATCH opts back into scatter+match);
// replaces scatter()+match() with (sort pair fused into mix) -> tiled radix sort
// -> collision run-scan -> coalesced emit.
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
bool sort_profile() { static bool v = (std::getenv("MXBM_SORT_PROFILE") != nullptr); return v; }

// Tiled CUB-style radix sort (P4): 6 x 4-bit passes, TILE = kSortWG*kSortIPT per
// workgroup, 16-bin bin-major histogram + local stable multisplit -- replaces the
// P1 one-elem/thread sort (validated head-to-head in test_gpu_sort: ~9.5 ms vs
// ~40 ms GPU-side at 2^25). 6 passes end in sort_pairs[0] (even count), so callers
// that loaded the input into pairs[0] get the sorted result there.
constexpr uint32_t kSortIPT = 8u;
constexpr uint32_t kSortTile = kSortWG * kSortIPT;

cl_mem radix_sort_pairs(Runtime& rt, cl_program prog, PipelineBuffers& pb, uint32_t n) {
    uint32_t ngroups = (n + kSortTile - 1) / kSortTile;
    cl_mem cur = pb.sort_pairs[0].get(), other = pb.sort_pairs[1].get(), wh = pb.sort_hist.get();
    Kernel kh = rt.kernel(prog, "radix4_histogram");
    Kernel kr = rt.kernel(prog, "radix4_reorder");
    double tHist = 0, tScan = 0, tReo = 0;
    const bool prof = sort_profile();
    for (uint32_t pass = 0; pass < 6; ++pass) {
        uint32_t shift = pass * 4;
        auto t0 = clk::now();
        rt.set_arg(kh.get(), 0, sizeof(cl_mem), &cur);
        rt.set_arg(kh.get(), 1, sizeof(cl_mem), &wh);
        rt.set_arg(kh.get(), 2, shift);
        rt.set_arg(kh.get(), 3, n);
        rt.set_arg(kh.get(), 4, ngroups);
        rt.run1d(kh.get(), (size_t)ngroups * kSortWG, kSortWG);
        if (prof) tHist += ms_since(t0);

        auto t1 = clk::now();
        scan_exclusive(rt, prog, wh, 16u * ngroups);
        if (prof) tScan += ms_since(t1);

        auto t2 = clk::now();
        rt.set_arg(kr.get(), 0, sizeof(cl_mem), &cur);
        rt.set_arg(kr.get(), 1, sizeof(cl_mem), &other);
        rt.set_arg(kr.get(), 2, sizeof(cl_mem), &wh);
        rt.set_arg(kr.get(), 3, shift);
        rt.set_arg(kr.get(), 4, n);
        rt.set_arg(kr.get(), 5, ngroups);
        rt.run1d(kr.get(), (size_t)ngroups * kSortWG, kSortWG);
        if (prof) tReo += ms_since(t2);

        cl_mem t = cur; cur = other; other = t;
    }
    if (prof) std::printf("      [sort] hist=%.1f binscan=%.1f reorder=%.1f ms (6x4-bit, ngroups=%u)\n",
                          tHist, tScan, tReo, ngroups);
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
    const bool prof = sort_profile();

    // 1. (key,index) pairs are already in sort_pairs[0] -- FUSED into the mix that
    // ran just before this (round1_mix_seeds / round_mix write the pair from the
    // freshly-mixed key), so there is no separate key_extract pass.
    double tKeyExtract = 0.0;

    // 2. stable radix sort by 24-bit key.
    cl_mem sorted = radix_sort_pairs(rt, prog, pb, N);

    // 3. collision_count over the sorted runs -> sort_scan.
    auto tCnt = clk::now();
    cl_mem scan = pb.sort_scan.get();
    {
        Kernel k = rt.kernel(prog, "collision_count");
        rt.set_arg(k.get(), 0, sizeof(cl_mem), &sorted);
        rt.set_arg(k.get(), 1, sizeof(cl_mem), &scan);
        rt.set_arg(k.get(), 2, N);
        rt.run1d(k.get(), (size_t)((N + kSortWG - 1) / kSortWG) * kSortWG, kSortWG);
    }
    double tCount = prof ? ms_since(tCnt) : 0.0;
    // total children = exclusive-offset(last) + count(last). Grab count(last)
    // BEFORE the in-place scan clobbers it (one 4-byte read), then offset(last).
    uint32_t lastCount = 0;
    rt.read_at(scan, (size_t)(N - 1) * 4, 4, &lastCount);
    scan_exclusive(rt, prog, scan, N);
    uint32_t lastOff = 0;
    rt.read_at(scan, (size_t)(N - 1) * 4, 4, &lastOff);
    uint32_t total = lastOff + lastCount;

    double tScanTotal = prof ? ms_since(tCnt) - tCount : 0.0;

    // 4. emit. Zero counters[2] (pair_drops) first; leave the rest as-is.
    uint32_t counters[4];
    rt.read(pb.counters.get(), sizeof counters, counters);
    counters[2] = 0u;
    rt.write(pb.counters.get(), sizeof counters, counters);

    auto tEm = clk::now();
    {
        // Fixed-width match variant by round (r3:7->6, r4:6->5, r5:5->1), fully
        // constant: Lout, lead_identity and the leaf widths are as compile-time-
        // knowable as the work widths already were. See MATCH_SORTED_K in sort.cl.
        const char* matchName = "round_match_sorted";
        {
            // MEASURED, and not what was expected: the constants win for r3/r4/r5
            // (-1.6 / -0.9 / -1.2 ms) and LOSE for r1/r2 (+2.8 / +2.3), reproducibly,
            // across every interleaved run. They are therefore used only where they pay.
            // The split tracks the leaf-copy loop -- s_out is 8/9/0 for r3/r4/r5, where
            // unrolling it (or deleting it, at 0) is worth real time, and 2/4 for r1/r2,
            // where it is not and something else costs more. That "something else" is
            // unexplained; see docs/performance.md.
            if (r >= 3) {
                static const char* kK[6] = { "", "", "", "round_match_k3",
                                             "round_match_k4", "round_match_k5" };
                matchName = kK[r];
            }
            // MXBM_MATCH_K12: opt r1/r2 into the constant kernels the measured
            // regression keeps un-selected. 1 = plain k1/k2 (re-measures the
            // regression); 2 = the occupancy-pinned k1p/k2p (the ballast
            // experiment -- same constants, the generic kernel's residency).
            else {
                static const int k12 = std::getenv("MXBM_MATCH_K12")
                                     ? atoi(std::getenv("MXBM_MATCH_K12")) : 0;
                if (k12 == 1) matchName = (r == 1) ? "round_match_k1"  : "round_match_k2";
                if (k12 == 2) matchName = (r == 1) ? "round_match_k1p" : "round_match_k2p";
            }
        }
        // Round 1 index-only: derives both parents from their seed indices instead of
        // reading a work buffer round1_mix_seeds_idx never wrote. Takes one extra
        // argument (the prePow), so it is dispatched after the shared set_arg block.
        const bool seedMatch = (r == 1) && idxonly_r1();
        if (seedMatch) matchName = "round_match_seed";
        Kernel k = rt.kernel(prog, matchName);
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
        if (seedMatch) { cl_mem ppMem = pb.pp.get(); rt.set_arg(k.get(), 16, sizeof(cl_mem), &ppMem); }
        rt.run1d(k.get(), (size_t)((N + kSortWG - 1) / kSortWG) * kSortWG, kSortWG);
    }

    double tEmit = prof ? ms_since(tEm) : 0.0;
    if (prof) std::printf("    [match_sorted r%d N=%u] keyx=%.1f count=%.1f scan=%.1f emit=%.1f ms (children=%u)\n",
                          r, N, tKeyExtract, tCount, tScanTotal, tEmit, total);

    rt.read(pb.counters.get(), sizeof counters, counters);
    pair_drops = counters[2];
    return (total < outCapacity) ? total : outCapacity;
}

// Phase-D3: the sort-based collision finder is the DEFAULT collision path
// (it beats scatter+match and is the coalescing foundation for further tuning);
// set MXBM_LEGACY_MATCH to fall back to the old atomic-bucket scatter+match.
// One source of truth, read once, shared by alloc_pipeline and run_single_round.
bool use_sort_match() { static bool v = (std::getenv("MXBM_LEGACY_MATCH") == nullptr); return v; }
bool use_lds_match()  { static bool v = (std::getenv("MXBM_LDS_MATCH") != nullptr); return v; }

// P2c: LDS-local match. round_scatter_lds buckets work[r] (+slot+lead) into
// bucket-contiguous storage; round_collide_lds finds collisions in LDS and emits
// the SAME children as match_sorted (byte-identical goldens), combining parents
// from LDS instead of a global gather. Keeps mix separate (run before this).
uint32_t match_lds(Runtime& rt, PipelineBuffers& pb, const Budget& b, int r,
                   uint32_t N, uint32_t& bucket_drops, uint32_t& pair_drops) {
    (void)b;
    cl_program prog = rt.cached_program({std::string(kBh3ClSource), std::string(kRoundClSource),
                                         std::string(kLdsClSource)}, "");
    const uint32_t bucketBits = 14, submaskBits = 3, cap = pb.lds_bucket_cap;
    uint32_t outCapacity = pb.capacity;
    rt.fill_u32(pb.lds_counts.get(), 0u, pb.lds_num_buckets);
    rt.fill_u32(pb.counters.get(), 0u, 4);

    {   // scatter work[r] (+slot+lead) into buckets
        Kernel k = rt.kernel(prog, "round_scatter_lds");
        uint32_t leadIdentity = (r == 1) ? 1u : 0u;
        cl_mem work = pb.work[r & 1].get(), leavesIn = pb.leaves[r & 1].get();
        cl_mem counts = pb.lds_counts.get(), bwork = pb.lds_bwork.get();
        cl_mem bslot = pb.lds_bslot.get(), blead = pb.lds_blead.get(), cnt = pb.counters.get();
        rt.set_arg(k.get(), 0, N); rt.set_arg(k.get(), 1, bucketBits); rt.set_arg(k.get(), 2, cap);
        rt.set_arg(k.get(), 3, leadIdentity);
        rt.set_arg(k.get(), 4, sizeof(cl_mem), &work); rt.set_arg(k.get(), 5, sizeof(cl_mem), &leavesIn);
        rt.set_arg(k.get(), 6, sizeof(cl_mem), &counts); rt.set_arg(k.get(), 7, sizeof(cl_mem), &bwork);
        rt.set_arg(k.get(), 8, sizeof(cl_mem), &bslot); rt.set_arg(k.get(), 9, sizeof(cl_mem), &blead);
        rt.set_arg(k.get(), 10, sizeof(cl_mem), &cnt);
        rt.run1d(k.get(), N);
    }
    {   // LDS collide + combine + emit
        Kernel k = rt.kernel(prog, "round_collide_lds");
        uint32_t Lout = lout_for(r), outOff = (uint32_t)(r - 1) * pb.capacity;
        cl_mem counts = pb.lds_counts.get(), bwork = pb.lds_bwork.get();
        cl_mem bslot = pb.lds_bslot.get(), blead = pb.lds_blead.get();
        cl_mem outWork = pb.work[((r < (int)kNumRounds) ? r + 1 : 0) & 1].get();
        cl_mem leftMem = pb.left.get(), rightMem = pb.right.get();
        cl_mem leavesIn = pb.leaves[r & 1].get(), leavesOut = pb.leaves[(r + 1) & 1].get(), cnt = pb.counters.get();
        uint32_t sIn = sleaves_for(r), sOut = (r < (int)kNumRounds) ? sleaves_for(r + 1) : 0u;
        rt.set_arg(k.get(), 0, bucketBits); rt.set_arg(k.get(), 1, submaskBits); rt.set_arg(k.get(), 2, cap);
        rt.set_arg(k.get(), 3, Lout); rt.set_arg(k.get(), 4, outOff); rt.set_arg(k.get(), 5, outCapacity);
        rt.set_arg(k.get(), 6, sizeof(cl_mem), &counts); rt.set_arg(k.get(), 7, sizeof(cl_mem), &bwork);
        rt.set_arg(k.get(), 8, sizeof(cl_mem), &bslot); rt.set_arg(k.get(), 9, sizeof(cl_mem), &blead);
        rt.set_arg(k.get(), 10, sizeof(cl_mem), &outWork); rt.set_arg(k.get(), 11, sizeof(cl_mem), &leftMem);
        rt.set_arg(k.get(), 12, sizeof(cl_mem), &rightMem); rt.set_arg(k.get(), 13, sizeof(cl_mem), &leavesIn);
        rt.set_arg(k.get(), 14, sizeof(cl_mem), &leavesOut); rt.set_arg(k.get(), 15, sIn); rt.set_arg(k.get(), 16, sOut);
        rt.set_arg(k.get(), 17, sizeof(cl_mem), &cnt);
        uint32_t groups = pb.lds_num_buckets << submaskBits;
        rt.run1d(k.get(), (size_t)groups * 256u, 256u);
    }
    uint32_t counters[4];
    rt.read(pb.counters.get(), sizeof counters, counters);
    bucket_drops = counters[1]; pair_drops = counters[2];
    return (counters[0] < outCapacity) ? counters[0] : outCapacity;
}

uint32_t run_single_round(Runtime& rt, PipelineBuffers& pb, const Budget& b, int r, uint32_t inN, RoundStats& stats) {
    if (use_lds_match()) {   // P2c LDS-local match (mix stays separate)
        auto tMix = clk::now();
        if (r >= 2) mix_level(rt, pb, r, inN);
        stats.t_mix_ms = (r >= 2) ? ms_since(tMix) : 0.0;
        auto tM = clk::now();
        uint32_t bd = 0, pd = 0;
        uint32_t outN = match_lds(rt, pb, b, r, inN, bd, pd);
        stats.t_match_ms = ms_since(tM);
        stats.t_scatter_ms = 0.0;
        stats.in = inN; stats.out = outN; stats.bucket_drops = bd; stats.pair_drops = pd;
        return outN;
    }
    // Phase-D3: route to the sort-based collision finder by default. Mix is
    // unchanged (still E3a leaf-prefix mix); only scatter+match is replaced.
    const bool useSort = use_sort_match();
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

    Mem outSlots = rt.alloc(CL_MEM_READ_WRITE, (size_t)cap * 4);
    // Round 5's children live in pb.work[0] (match()'s r==5 special case --
    // CORRECTION 1 vs the brief's original text, which said work[5]).
    cl_mem workMem     = pb.work[0].get();
    cl_mem outSlotsMem = outSlots.get();
    cl_mem countersMem = pb.counters.get();

    {
        // r5 children are stored at outwords_for(5) = 1 word; the all-zero marker
        // collapses to word0 == 0 (upper words were 0 under Lout=24 anyway).
        Kernel k = rt.kernel(prog, "survivor_scan_s");
        uint32_t stride = outwords_for(5);
        rt.set_arg(k.get(), 0, N);
        rt.set_arg(k.get(), 1, stride);
        rt.set_arg(k.get(), 2, sizeof(cl_mem), &workMem);
        rt.set_arg(k.get(), 3, sizeof(cl_mem), &outSlotsMem);
        rt.set_arg(k.get(), 4, cap);
        rt.set_arg(k.get(), 5, sizeof(cl_mem), &countersMem);
        rt.run1d(k.get(), N);
    }

    rt.read(pb.counters.get(), sizeof counters, counters);
    uint32_t survivors = counters[3];
    uint32_t clamped = (survivors < cap) ? survivors : cap;
    survivor_slots.resize(clamped);
    if (clamped > 0) rt.read(outSlots.get(), (size_t)clamped * 4, survivor_slots.data());
    return clamped;
}

// MXBM_ROWBUCKET: the fused row-bucket pipeline. ENTRY (round1_mix_scatter_fat)
// mixes seeds and scatters into round-1 FAT buckets; round_fused_lds runs r=1..4,
// each fusing round r's match + round (r+1)'s mix + scatter into the next FAT
// bucket set (ping-pong); round5_fused_lds combines at Lout=24 and emits the
// all-zero survivors. Back-refs (pb.left/right, row (r-1)*capacity by gi) and
// recover are shared with the sort path. Same PipelineResult contract.
static PipelineResult run_pipeline_rowbucket(Runtime& rt, PipelineBuffers& pb, const Budget& b,
                             const uint64_t pp[4], const std::atomic<bool>* abort, bool verbose,
                             SpecEntry* spec) {
    PipelineResult result;
    auto tPipeline = clk::now();
    const uint32_t nb = pb.fb_num_buckets, cap = pb.fb_bucket_cap, capacity = pb.capacity;
    // GEO_BAKED (see lds.cl for the measurement): cached_program keys on the options
    // string, so each geometry gets its own build. MXBM_NO_GEOBAKE restores the
    // runtime args, giving both arms from one binary.
    std::string opts = rowbucket_cl_opts();
    // The arena is a build option for the same reason the geometry is: this program is
    // compiled after the ladder has settled, so an arena rung compiles the pool paths in
    // and every other rung compiles the kernels it always had. Which rungs may pair with
    // it is decided in try_alloc_rowbucket, where a refusal can still step down.
    if (pb.fb_arena)
        opts += " -DLDS_ARENA=1 -DLDS_ARENA_CAP=" + std::to_string(pb.fb_arena_cap) + "u";
    if (pb.fb_octo) opts += " -DLDS_OCTO=1";
    uint32_t bbv0 = 0; for (uint32_t t = nb; t > 1u; t >>= 1) ++bbv0;
    // The dropped-bit count IS bucket_bits, so the pack is a build option like the
    // rest: writer and reader take it from the same string.
    if (pb.fb_impb) opts += " -DLDS_IMPB=" + std::to_string(bbv0) + "u";
    if (!std::getenv("MXBM_NO_GEOBAKE")) {
        opts += " -DGEO_BAKED=1 -DGEO_BB=" + std::to_string(bbv0) + "u"
              + " -DGEO_SM=" + std::to_string(pb.fb_submask_bits) + "u"
              + " -DGEO_INCAP=" + std::to_string(cap) + "u"
              + " -DGEO_OUTCAP=" + std::to_string(cap) + "u";
    }
    cl_program prog = rt.cached_program({std::string(kBh3ClSource), std::string(kRoundClSource),
                                         std::string(kLdsClSource)}, opts);
    // Sweepable for tuning: the sub-mask split sets how many times each bucket is
    // rescanned (2^submaskBits) against the LDS a group needs. See MXBM_BB/MXBM_SM.
    uint32_t bbv = 0; for (uint32_t t = pb.fb_num_buckets; t > 1u; t >>= 1) ++bbv;
    const uint32_t bucketBits = bbv;                  // as chosen by alloc_rowbucket
    const uint32_t submaskBits = pb.fb_submask_bits;
    // LDS_PERFECT_TAB (default on) drops the walk's key compare on the strength of
    // the 128-entry chain table being a PERFECT hash over the 24 - bb - sm bits that
    // vary inside a group. Off the bb + sm >= 17 line that assumption breaks
    // SILENTLY -- distinct keys would be combined as equal -- so refuse rather than
    // mine wrong, the same check CudaSolver makes. An off-line experiment can pass
    // MXBM_CL_OPTS="-DLDS_PERFECT_TAB=0" to restore the keyed walk (and this check
    // steps aside when it sees that).
    {
        const std::string opts = rowbucket_cl_opts();
        const bool perfectOff = opts.find("LDS_PERFECT_TAB=0") != std::string::npos;
        if (!perfectOff && (int)(24 - bucketBits - submaskBits) > 7)
            throw ClError(CL_INVALID_VALUE,
                "geometry (" + std::to_string(bucketBits) + "," + std::to_string(submaskBits) +
                ") leaves more than 7 varying key bits per group; the perfect chain table "
                "needs bucket_bits + submask_bits >= 17. Use a geometry on that line, or pass "
                "MXBM_CL_OPTS=-DLDS_PERFECT_TAB=0 to keep the keyed walk.");
    }
    const uint32_t survCap = 1024;
    const uint32_t total = (b.elems_per_round != 0) ? b.elems_per_round : capacity;
    const uint32_t batch = (b.seed_batch != 0) ? b.seed_batch : total;

    Mem mdrops = rt.alloc(CL_MEM_READ_WRITE, 4 * 4);
    rt.fill_u32(mdrops.get(), 0u, 4);
    // prePow, needed by the entry AND by round 1 (which re-derives seeds from indices).
    uint64_t pp4[4] = { pp[0], pp[1], pp[2], pp[3] };
    Mem mpp = rt.alloc(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof pp4, pp4);
    // The speculated nonce's prePow. Created before the first enqueue: a blocking write
    // later would drain the in-order queue and undo the de-bubble. Per-solve, not in
    // PipelineBuffers -- round 4 is its only reader.
    Mem mSpecPp;
    if (spec && pb.spec_on && spec->seed_next)
        mSpecPp = rt.alloc(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof pp4, spec->next_pp);

    // DE-BUBBLE (#23): in production (verbose=false) enqueue fills+kernels ASYNC on
    // the in-order queue and let the final survivor/drops readbacks drain once,
    // instead of a clFinish after every kernel. In verbose mode keep the blocking
    // calls so the per-round steady_clock timers stay accurate GPU time.
    auto RUN  = [&](cl_kernel k, size_t g, size_t l) { if (verbose) rt.run1d(k, g, l); else rt.run1d_async(k, g, l); };
    auto FILL = [&](cl_mem bm, uint32_t v, size_t c) { if (verbose) rt.fill_u32(bm, v, c); else rt.fill_u32_async(bm, v, c); };
    // Arena bookkeeping for one record set. RESET clears the pool cursor and empties
    // every chain before the producer writes; LINK threads what the producer appended
    // onto per-bucket chains, and must run between the producer and its consumer.
    auto ARENA_RESET = [&](int set) {
        if (!pb.fb_arena) return;
        FILL(pb.fb_actr[set].get(), 0u, 2);
        FILL(pb.fb_ahead[set].get(), 0xFFFFFFFFu, nb);
    };
    auto ARENA_LINK = [&](int set) {
        if (!pb.fb_arena) return;
        Kernel kl = rt.kernel(prog, "arena_link");
        cl_mem ac = pb.fb_actr[set].get(), at = pb.fb_atag[set].get();
        cl_mem ah = pb.fb_ahead[set].get(), an = pb.fb_anext[set].get();
        cl_mem sp = pb.fb_spill.get();
        const uint32_t halves = (pb.fb_half[set] == nb) ? 1u : 2u;
        rt.set_arg(kl.get(), 0, sizeof(cl_mem), &ac); rt.set_arg(kl.get(), 1, sizeof(cl_mem), &at);
        rt.set_arg(kl.get(), 2, sizeof(cl_mem), &ah); rt.set_arg(kl.get(), 3, sizeof(cl_mem), &an);
        rt.set_arg(kl.get(), 4, sizeof(cl_mem), &sp); rt.set_arg(kl.get(), 5, halves);
        RUN(kl.get(), (size_t)pb.fb_arena_cap * halves, 0);
    };
    // Bound to every kernel off the arena rungs, where LDS_ARENA compiles the uses out.
    cl_mem aDummy = pb.fb_gictr.get();

    // ENTRY: mix seeds + scatter into round-1 FAT buckets. Under speculation the
    // destination is the spec set instead of set 0 -- one code path, so a hit and a
    // miss differ only in whether the pass runs. That set is never split, so binding
    // it twice with half == nb keeps hi undereferenced.
    const bool specOn = pb.spec_on && spec != nullptr;
    const bool specHit = specOn && spec->hit;
    cl_mem r1Counts = specOn ? pb.spec_counts.get() : pb.fb_counts[0].get();
    cl_mem r1ElemLo = specOn ? pb.spec_elem.get()   : pb.fb_elem[0].get();
    cl_mem r1ElemHi = specOn ? r1ElemLo
                            : (pb.fb_elem_hi[0] ? pb.fb_elem_hi[0].get() : r1ElemLo);
    const uint32_t r1Half = specOn ? nb : pb.fb_half[0];
    auto tSeed = clk::now();
    if (!specHit) {
        FILL(r1Counts, 0u, nb);
        ARENA_RESET(0);
        Kernel k = rt.kernel(prog, "round1_mix_scatter_fat");
        cl_mem ppMem = mpp.get(), dr = mdrops.get();
        cl_mem eActr = pb.fb_arena ? pb.fb_actr[0].get() : aDummy;
        cl_mem eAtag = pb.fb_arena ? pb.fb_atag[0].get() : aDummy;
        for (uint32_t begin = 0; begin < total; begin += batch) {
            uint32_t count = (total - begin < batch) ? (total - begin) : batch;
            rt.set_arg(k.get(), 0, sizeof(cl_mem), &ppMem); rt.set_arg(k.get(), 1, begin); rt.set_arg(k.get(), 2, count);
            rt.set_arg(k.get(), 3, bucketBits); rt.set_arg(k.get(), 4, cap); rt.set_arg(k.get(), 5, kFbStride[1]);
            rt.set_arg(k.get(), 6, sizeof(cl_mem), &r1Counts);
            rt.set_arg(k.get(), 7, sizeof(cl_mem), &r1ElemLo); rt.set_arg(k.get(), 8, sizeof(cl_mem), &r1ElemHi);
            rt.set_arg(k.get(), 9, r1Half);
            rt.set_arg(k.get(), 10, sizeof(cl_mem), &dr);
            rt.set_arg(k.get(), 11, sizeof(cl_mem), &eActr);
            rt.set_arg(k.get(), 12, sizeof(cl_mem), &eAtag);
            RUN(k.get(), count, 0);
        }
        ARENA_LINK(0);
    }
    if (verbose) rt.finish();
    result.t_seed_ms = ms_since(tSeed);

    // MIDDLE r=1..4: fused match + child-mix + child-scatter, ping-ponging FAT sets.
    int inSet = 0;
    uint32_t prevOut = total;
    for (int r = 1; r <= 4; ++r) {
        RoundStats& st = result.rounds[r - 1];
        int outSet = inSet ^ 1;
        auto tM = clk::now();
        FILL(pb.fb_counts[outSet].get(), 0u, nb);
        FILL(pb.fb_gictr.get(), 0u, 1);
        uint32_t Lout = lout_for(r), LmixNext = lmix_for(r + 1), padNext = padnum_for(r + 1);
        // Leaf payload per round (see LMODE in lds.cl):
        //   r1,r2 RAW  : carry raw leaves (sleaves_for).
        //   r3    EMIT : the round-4 child carries ONE u64 leftContrib (2 uints), folded
        //                from its 8 leaves, instead of the 8 raw leaves -- it still
        //                BUILDS 8 (sBuild) but STORES 2. Emit 88 -> 64 B/child.
        //   r4    USE  : stages that 2-uint contrib (sIn=2) and stores nothing
        //                (L5-thin: round 5 is terminal; recover walks back-refs).
        const FusedConsts fc = fused_consts_for(r);
        const uint32_t sIn = fc.sIn, sOut = fc.sOut, sBuild = fc.sBuild;
        // Back-reference row. An octo build allocates ONE row plus the survivor tail and
        // writes it from round 4 alone -- rounds 1-3 write none -- so round 4's row sits
        // at 0, where recover() reads it, not at the (r-1) the five-row layout gives it.
        uint32_t outOff = pb.fb_octo ? 0u : (uint32_t)(r - 1) * capacity;
        // Per-round work compaction (inwords->outwords): r1,r2=(7,7); r3=(7,6); r4=(6,5).
        // Rounds 2 and 3 switch TOGETHER between record formats -- r2's out-stride is
        // r3's in-stride, so a mismatch would have r3 read a record r2 never wrote.
        // Round 4 hosts the next nonce's entry as co-blocks when this solve is
        // speculating; every other round, and an unspeculated round 4, is unchanged.
        const bool cob = (r == 4) && specOn && spec->seed_next;
        const char* fusedName = "round_fused_lds";
        // Round 1's variant differs only in the record it emits: the w0 checkpoint, which
        // exists exactly where the implicit-bits pack does, since the address bits pay for it.
        if      (r == 1) fusedName = pb.fb_impb ? "round_fused_seedi" : "round_fused_seed";
        else if (r == 2) fusedName = pb.fb_octo ? "round_fused_rd2q16"
                                   : (pb.fb_quad ? "round_fused_rd2q"
                                   : (pb.fb_impb ? "round_fused_rd2i" : "round_fused_rd2"));
        else if (r == 3) fusedName = pb.fb_octo ? "round_fused_rd3o"
                                   : (pb.fb_quad ? "round_fused_rd3"
                                   : (pb.fb_impb ? "round_fused_7_6i" : "round_fused_7_6"));
        else if (r == 4) fusedName = pb.fb_octo ? "round_fused_rd4"
                                   : (cob ? "round_fused_6_5_cob" : "round_fused_6_5");
        Kernel k = rt.kernel(prog, fusedName);
        // Round 1 reads whatever the entry pass wrote into -- the spec set under
        // speculation, set 0 otherwise (r1Half is nb for the never-split spec set).
        cl_mem ic = (r == 1) ? r1Counts : pb.fb_counts[inSet].get();
        cl_mem oc = pb.fb_counts[outSet].get();
        cl_mem ieLo = (r == 1) ? r1ElemLo : pb.fb_elem[inSet].get();
        cl_mem ieHi = (r == 1) ? r1ElemHi
                     : (pb.fb_elem_hi[inSet] ? pb.fb_elem_hi[inSet].get() : ieLo);
        const uint32_t inHalf = (r == 1) ? r1Half : pb.fb_half[inSet];
        cl_mem oeLo = pb.fb_elem[outSet].get();
        cl_mem oeHi = pb.fb_elem_hi[outSet] ? pb.fb_elem_hi[outSet].get() : oeLo;
        cl_mem aL = pb.left.get(), aR = pb.right.get(), gc = pb.fb_gictr.get(), dr = mdrops.get();
        // Side plane: r2 writes it, r3 reads it; other rounds (and quad) never touch
        // it, so the dummy binding is safe (same precedent as round_match's all_lead).
        cl_mem sp = pb.fb_side ? pb.fb_side.get() : gc;
        // The packed r2->r3 record allocates stride 8 (9th word in the plane); the
        // slot-cost table still says 9, so patch the runtime stride args to match.
        auto recStride = [&](int rr) {
            const uint32_t s = fb_round_stride(rr, pb.fb_quad, pb.fb_impb, pb.fb_octo);
            return s == 9u ? 8u : s;
        };
        int a = 0;
        rt.set_arg(k.get(), a++, bucketBits); rt.set_arg(k.get(), a++, submaskBits);
        rt.set_arg(k.get(), a++, cap); rt.set_arg(k.get(), a++, cap);
        rt.set_arg(k.get(), a++, Lout); rt.set_arg(k.get(), a++, LmixNext); rt.set_arg(k.get(), a++, padNext);
        rt.set_arg(k.get(), a++, sIn); rt.set_arg(k.get(), a++, sOut); rt.set_arg(k.get(), a++, outOff);
        rt.set_arg(k.get(), a++, sBuild);
        rt.set_arg(k.get(), a++, recStride(r - 1));
        rt.set_arg(k.get(), a++, recStride(r));
        rt.set_arg(k.get(), a++, sizeof(cl_mem), &ic);
        rt.set_arg(k.get(), a++, sizeof(cl_mem), &ieLo); rt.set_arg(k.get(), a++, sizeof(cl_mem), &ieHi);
        rt.set_arg(k.get(), a++, inHalf);
        rt.set_arg(k.get(), a++, sizeof(cl_mem), &sp);
        rt.set_arg(k.get(), a++, sizeof(cl_mem), &oc);
        rt.set_arg(k.get(), a++, sizeof(cl_mem), &oeLo); rt.set_arg(k.get(), a++, sizeof(cl_mem), &oeHi);
        rt.set_arg(k.get(), a++, pb.fb_half[outSet]);
        rt.set_arg(k.get(), a++, sizeof(cl_mem), &sp);
        rt.set_arg(k.get(), a++, sizeof(cl_mem), &aL); rt.set_arg(k.get(), a++, sizeof(cl_mem), &aR);
        rt.set_arg(k.get(), a++, sizeof(cl_mem), &gc); rt.set_arg(k.get(), a++, sizeof(cl_mem), &dr);
        cl_mem ppm = mpp.get(); rt.set_arg(k.get(), a++, sizeof(cl_mem), &ppm);
        // Arena: this round consumes the input set's chains and fills the output set's
        // pool. The output set's bookkeeping is cleared here, after the previous round
        // finished reading it as ITS input, and linked once this round has written it.
        ARENA_RESET(outSet);
        cl_mem iAhead = pb.fb_arena ? pb.fb_ahead[inSet].get() : aDummy;
        cl_mem iAnext = pb.fb_arena ? pb.fb_anext[inSet].get() : aDummy;
        cl_mem oActr  = pb.fb_arena ? pb.fb_actr[outSet].get() : aDummy;
        cl_mem oAtag  = pb.fb_arena ? pb.fb_atag[outSet].get() : aDummy;
        rt.set_arg(k.get(), a++, sizeof(cl_mem), &iAhead);
        rt.set_arg(k.get(), a++, sizeof(cl_mem), &iAnext);
        rt.set_arg(k.get(), a++, sizeof(cl_mem), &oActr);
        rt.set_arg(k.get(), a++, sizeof(cl_mem), &oAtag);
        size_t groups = (size_t)nb << submaskBits;
        if (cob) {
            // Zeroing the spec counters belongs HERE: after round 1 read them on the
            // in-order queue, before round 4 refills them. Same ordering CudaSolver uses.
            FILL(pb.spec_counts.get(), 0u, nb);
            // One entry group per (stride - 1) round groups, so the default 3 displaces
            // a third of the grid -- CUDA's ratio. Which ratio pays depends on how much
            // idle round 4 has, a per-card property, so MXBM_CO_STRIDE sweeps it.
            static const uint32_t coStride = [] {
                const char* e = std::getenv("MXBM_CO_STRIDE");
                const int v = e ? std::atoi(e) : 0;
                return (v >= 2 && v <= 64) ? (uint32_t)v : 3u;
            }();
            cl_mem cpp = mSpecPp.get(), ccnt = pb.spec_counts.get(), cel = pb.spec_elem.get();
            rt.set_arg(k.get(), a++, sizeof(cl_mem), &cpp);
            rt.set_arg(k.get(), a++, total); rt.set_arg(k.get(), a++, cap);
            rt.set_arg(k.get(), a++, coStride);
            rt.set_arg(k.get(), a++, sizeof(cl_mem), &ccnt);
            rt.set_arg(k.get(), a++, sizeof(cl_mem), &cel);
            rt.set_arg(k.get(), a++, sizeof(cl_mem), &cel);   // never split: bound twice
            rt.set_arg(k.get(), a++, nb);
            // The co-entry writes the spec set, which has no pool -- and speculation is
            // off on the arena rungs for exactly that reason, so these are never read.
            rt.set_arg(k.get(), a++, sizeof(cl_mem), &aDummy);
            rt.set_arg(k.get(), a++, sizeof(cl_mem), &aDummy);
            groups += groups / (coStride - 1u);
            spec->seeded = true;
        }
        RUN(k.get(), groups * 256u, 256u);
        ARENA_LINK(outSet);
        // MXBM_OCC=1: read this round's arrival counts and report the occupancy
        // distribution. fb_cap_for's headroom is a guess; this measures the tail it
        // actually has to cover.
        if (std::getenv("MXBM_OCC")) {
            rt.finish();
            std::vector<uint32_t> occ(nb);
            rt.read(pb.fb_counts[outSet].get(), (size_t)nb * 4, occ.data());
            std::vector<uint32_t> sorted(occ);
            std::sort(sorted.begin(), sorted.end());
            double mean = 0; for (uint32_t v : occ) mean += v; mean /= nb;
            double var = 0; for (uint32_t v : occ) var += (v - mean) * (v - mean); var /= nb;
            const double sd = std::sqrt(var);
            std::printf("  [occ] r%d n=%u mean=%.1f sd=%.1f max=%u (mean+%.2f sd)  "
                        "p99.9=%u  cap=%u (mean+%.2f sd)\n",
                        r, nb, mean, sd, sorted.back(), (sorted.back() - mean) / sd,
                        sorted[(size_t)(nb * 0.999)], cap, (cap - mean) / sd);
        }
        // childCount is stats-only; read it (draining the queue) only in verbose mode.
        uint32_t childCount = prevOut;
        if (verbose) { rt.read(pb.fb_gictr.get(), 4, &childCount);
                       std::printf("  r%d: in=%u out=%u | fused=%.1f ms\n", r, prevOut, childCount, ms_since(tM)); }
        st.in = prevOut; st.out = childCount; st.bucket_drops = 0; st.pair_drops = 0;
        st.t_match_ms = ms_since(tM);
        prevOut = childCount; inSet = outSet;
        if (abort && abort->load(std::memory_order_relaxed)) return PipelineResult{};
    }

    // TERMINAL r=5: combine at Lout(5)=24, emit all-zero survivors (row 4 back-refs).
    auto tSurv = clk::now();
    Mem mSurvSlots = rt.alloc(CL_MEM_READ_WRITE, (size_t)survCap * 4);
    Mem mSurvCount = rt.alloc(CL_MEM_READ_WRITE, 4);
    FILL(mSurvCount.get(), 0u, 1);
    {
        Kernel k = rt.kernel(prog, "round5_fused_lds");
        uint32_t outOff5 = pb.fb_octo ? capacity : (uint32_t)(5 - 1) * capacity;
        uint32_t Lout5 = lout_for(5);
        cl_mem ic = pb.fb_counts[inSet].get();
        cl_mem ieLo = pb.fb_elem[inSet].get();
        cl_mem ieHi = pb.fb_elem_hi[inSet] ? pb.fb_elem_hi[inSet].get() : ieLo;
        cl_mem aL = pb.left.get(), aR = pb.right.get(), ss = mSurvSlots.get(), sc = mSurvCount.get(), dr = mdrops.get();
        int a = 0;
        rt.set_arg(k.get(), a++, bucketBits); rt.set_arg(k.get(), a++, submaskBits); rt.set_arg(k.get(), a++, cap);
        rt.set_arg(k.get(), a++, Lout5); rt.set_arg(k.get(), a++, outOff5);
        rt.set_arg(k.get(), a++, kFbStride[5]);
        rt.set_arg(k.get(), a++, sizeof(cl_mem), &ic);
        rt.set_arg(k.get(), a++, sizeof(cl_mem), &ieLo); rt.set_arg(k.get(), a++, sizeof(cl_mem), &ieHi);
        rt.set_arg(k.get(), a++, pb.fb_half[inSet]);
        rt.set_arg(k.get(), a++, sizeof(cl_mem), &aL); rt.set_arg(k.get(), a++, sizeof(cl_mem), &aR);
        rt.set_arg(k.get(), a++, sizeof(cl_mem), &ss); rt.set_arg(k.get(), a++, sizeof(cl_mem), &sc);
        rt.set_arg(k.get(), a++, survCap); rt.set_arg(k.get(), a++, sizeof(cl_mem), &dr);
        cl_mem iAhead = pb.fb_arena ? pb.fb_ahead[inSet].get() : aDummy;
        cl_mem iAnext = pb.fb_arena ? pb.fb_anext[inSet].get() : aDummy;
        rt.set_arg(k.get(), a++, sizeof(cl_mem), &iAhead);
        rt.set_arg(k.get(), a++, sizeof(cl_mem), &iAnext);
        RUN(k.get(), (size_t)(nb << submaskBits) * 256u, 256u);
    }
    // Blocking read drains the whole async chain (entry -> r1..4 -> terminal).
    uint32_t survCount = 0; rt.read(mSurvCount.get(), 4, &survCount);
    uint32_t clamped = survCount < survCap ? survCount : survCap;
    result.survivor_slots.resize(clamped);
    if (clamped) rt.read(mSurvSlots.get(), (size_t)clamped * 4, result.survivor_slots.data());
    result.survivors = clamped;
    result.rounds[4].in = prevOut; result.rounds[4].out = clamped;
    result.t_survivor_ms = ms_since(tSurv);
    result.t_total_ms = ms_since(tPipeline);

    uint32_t dr[4]; rt.read(mdrops.get(), sizeof dr, dr);
    // The arena's POSITIVE CONTROL, on the same instrument channel as the occupancy
    // histogram. A run-long spill total of zero means no bucket ever exceeded the dense
    // cap, so the pool carried nothing and the arm measured the plain rung under another
    // name -- indistinguishable from success without this line. Also prints the drops,
    // which is the other half of the gate every arena A/B is held to.
    if (pb.fb_arena && (verbose || std::getenv("MXBM_OCC"))) {
        uint32_t sp = 0; rt.read(pb.fb_spill.get(), 4, &sp);
        std::printf("  [arena] dense cap %u, pool %u/half, spilled %u so far | "
                    "drops{bucket=%u out=%u chain=%u}\n",
                    pb.fb_bucket_cap, pb.fb_arena_cap, sp, dr[1], dr[2], dr[3]);
    }
    if (verbose) {
        double sps = result.t_total_ms > 0.0 ? 1000.0 / result.t_total_ms : 0.0;
        std::printf("  r5: survivors=%u  drops{bucket/group=%u out=%u chain=%u}\n", clamped, dr[1], dr[2], dr[3]);
        std::printf("  [rowbucket] seed=%.1f survivor=%.1f total=%.1f ms  ->  %.2f solve/s\n",
                    result.t_seed_ms, result.t_survivor_ms, result.t_total_ms, sps);
    }
    return result;
}

PipelineResult run_pipeline(Runtime& rt, PipelineBuffers& pb, const Budget& b, const uint64_t pp[4],
                             const std::atomic<bool>* abort, bool verbose, SpecEntry* spec) {
    if (pb.rowbucket) return run_pipeline_rowbucket(rt, pb, b, pp, abort, verbose, spec);
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

    // The octo walk is two levels deep instead of five and reads its leaves out of
    // round 3's surviving output, so this program needs the same build option the
    // pipeline's does -- and the rows it walks are laid out differently. It takes
    // MXBM_CL_OPTS with it because that is where the record's own -D lives: a walk
    // reading one layout while round 3 writes the other would decode noise.
    cl_program prog = rt.cached_program({std::string(kBh3ClSource), std::string(kRoundClSource)},
                                        std::string(rowbucket_cl_opts())
                                            + (pb.fb_octo ? " -DLDS_OCTO=1" : ""));
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
    // Round 3's output set, which the octo walk reads the eight leaves from -- both
    // halves, because a reference names a half-local slot and carries its half in the
    // top bit. Off an octo build the kernel never dereferences either, so any live
    // buffer will do.
    cl_mem r3e = pb.fb_octo ? pb.fb_elem[1].get() : leavesRaw;
    cl_mem r3eHi = (pb.fb_octo && pb.fb_elem_hi[1]) ? pb.fb_elem_hi[1].get() : r3e;
    rt.set_arg(k.get(), 6, sizeof(cl_mem), &r3e);
    rt.set_arg(k.get(), 7, pb.fb_octo ? pb.fb_stride[1] : 0u);
    rt.set_arg(k.get(), 8, sizeof(cl_mem), &r3eHi);
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
