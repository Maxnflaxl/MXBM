// CUDA row-bucket pipeline, gated on the KAT. Standalone: builds with nvcc, not wired
// into CMake, so an incomplete backend cannot destabilise the shipping OpenCL path.
//
//   nvcc -O3 -arch=sm_89 -I src -I kernels/cuda -I tests cuda/pipeline.cu -o cuda/pipeline
//
// The gate is the same one the OpenCL path uses and is non-negotiable: survivors == 3 on
// the KAT prePow, bucketDrops == 0, pairDrops == 0.
#include "pipeline_kernels.cuh"
#ifndef MXBM_ABL_DERIVE
#define MXBM_ABL_DERIVE 0
#endif
#ifndef MXBM_ABL_EMIT
#define MXBM_ABL_EMIT 0
#endif
#ifndef MXBM_CO_NOSCATTER
#define MXBM_CO_NOSCATTER 0
#endif
#ifndef MXBM_SUBPASS
#define MXBM_SUBPASS 0
#endif
#define MXBM_ABL (MXBM_ABL_DERIVE | MXBM_ABL_EMIT | MXBM_ABL_MIX | MXBM_CO_NOSCATTER)
#include "kat_vectors.h"
#include <cstdio>
#include <cmath>
#ifndef MXBM_R2_FULL
#define MXBM_R2_FULL 0
#endif
#include <vector>
#include "beamhash/bh3_verify.h"
#include "beamhash/bh3_blake2b.h"
#include <algorithm>
#include <chrono>
#include <array>
#include <cstdlib>
#include <cstring>
using namespace mxbm;
using namespace mxbm::cuda;

// Each round's template arguments, named ONCE. They were previously written out
// separately at the launch, at the carveout call and (in the bench) at the occupancy
// probe, and had drifted in three of the four: the carveout was being applied to
// LM_RD2 with OUTSTR 10 and LM_EMIT with INSTR 10 -- instantiations that no longer
// exist at runtime -- so the kernels that DO run never received the preference, and
// the bench's blocks/SM column was reporting a different kernel than it timed.
#if MXBM_R2_FULL
#define MXBM_R1_ARGS 7,7,2,LM_SEEDF,424u,2u,1u,2u,2u, 1u,10u,kFCap
#define MXBM_R2_ARGS 7,7,2,LM_RAW,  400u,4u,2u,4u,4u, 10u,10u,kFCap
#define MXBM_R3_ARGS 7,6,4,LM_EMIT, 376u,6u,4u,2u,8u, 10u,8u,kFCap
#elif MXBM_R3_QUAD
// Round 2 emits 3 u64 (key + 4 leaves + gi) and round 3 reads 3 and rebuilds. Only the
// two strides and round 3's mode change; every L value, tree width and cap is the same,
// because the record carries the same INFORMATION either way -- just not the same bytes.
#define MXBM_R1_ARGS 7,7,1,LM_SEED, 424u,2u,1u,2u,2u, 1u,2u,MXBM_R1_FCAP
#define MXBM_R2_ARGS 7,7,2,LM_RD2,  400u,4u,2u,4u,4u, 2u,3u,kFCap
#define MXBM_R3_ARGS 7,6,4,LM_RD3,  376u,6u,4u,2u,8u, 3u,8u,kFCap
#else
#define MXBM_R1_ARGS 7,7,1,LM_SEED, 424u,2u,1u,2u,2u, 1u,2u,MXBM_R1_FCAP
#define MXBM_R2_ARGS 7,7,2,LM_RD2,  400u,4u,2u,4u,4u, 2u,8u,kFCap
#define MXBM_R3_ARGS 7,6,4,LM_EMIT, 376u,6u,4u,2u,8u, 8u,8u,kFCap
#endif
#define MXBM_R4_ARGS 6,1,2,LM_USE,  288u,9u,2u,0u,0u, 8u,2u,kFCap


#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__); return 1; } }while(0)

// ---- MXBM_OCC: what the per-bucket capacity actually has to cover --------------------
// The 8 in fb_cap_for's mean + 8*sqrt(mean) + 32 is the largest term in the footprint:
// 215 of 743 slots at bb=16. counts[] is atomicAdd'ed WITHOUT being clamped (only the
// emit is gated on cpos < cap), so it is the true unclamped occupancy and reducing it
// gives the tail directly. Running max over every round of every solve.
__global__ void occ_reduce(const uint32_t* __restrict__ counts, uint32_t nb,
                           uint32_t* __restrict__ out) {
    const uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < nb) atomicMax(out, counts[i]);
}
static bool occ_on() { static const bool v = getenv("MXBM_OCC") != nullptr; return v; }

// ---- solver: persistent buffers, one solve() per (input, nonce) ---------------------
// Mirrors src/gpu/gpu_solver.cpp so the timing is like-for-like: buffers are allocated
// ONCE and reused, and solve() returns only CPU-VERIFIED solutions. Comparing a
// kernel-time figure against the OpenCL solver's end-to-end number would flatter this
// side, which is the whole reason for building this.
template<class T> static T* dalloc(size_t n) { void* p=nullptr; cudaMalloc(&p, n*sizeof(T)); return (T*)p; }

struct CudaSolver {
    static constexpr uint32_t elems    = 1u << 25;
    static constexpr uint32_t capacity = elems + elems/32;    // 34,603,008
    static constexpr uint32_t survCap = 1024;
    // Geometry is a RUNTIME choice on the bb + sm = 17 line. Coarser buckets pay the
    // per-bucket capacity slack fewer times (smaller footprint) but multiply the
    // redundant staging rescan by 2^sm. MXBM_BB/MXBM_SM sweep it.
    uint32_t bb = 16, sm = 1, nb = 1u << 16;
    uint32_t cap = 0; size_t nslots = 0;
    uint64_t *elem[2] = {nullptr,nullptr}, *dpp = nullptr;
    uint32_t *counts[2] = {nullptr,nullptr}, *gictr=nullptr, *drops=nullptr;
    uint32_t *left=nullptr, *right=nullptr, *survSlots=nullptr, *survCount=nullptr, *dleaves=nullptr;

    // --- phase-overlap experiment (docs/performance.md lead 5) ---------------
    // entry_scatter writes ONE u64 per element, densely (idx<<32|key), so its output
    // needs a stride-1 buffer of nslots -- 0.363 GiB -- not the 10-u64-strided elem[0]
    // it currently shares. Given its own buffer it can run on a second stream against
    // the previous nonce's r3/r4, which leave 71-79 % of the SM idle.
    // Double-buffered: entry(i+1) writes one slot while r1(i) still reads the other.
    uint32_t *occmax = nullptr;                  // [0]=entry, [1..4]=rounds 1..4
    uint64_t *entryOut[2] = {nullptr,nullptr};
    uint32_t *entryCounts[2] = {nullptr,nullptr};
    uint64_t *dpp2[2] = {nullptr,nullptr};       // rounds of solve i still use solve i's pp
    cudaStream_t sMain = nullptr, sEntry = nullptr;
    cudaEvent_t entryDone[2] = {nullptr,nullptr};   // entry(slot) finished writing
    cudaEvent_t r1Done[2]    = {nullptr,nullptr};   // r1 finished READING slot (WAR guard)
    bool r1Ever[2] = {false,false};

    bool init() {
        if (const char* e = getenv("MXBM_BB")) bb = (uint32_t)atoi(e);
        sm = 17u - bb;
        if (const char* e = getenv("MXBM_SM")) sm = (uint32_t)atoi(e);
        nb = 1u << bb;
        const uint32_t mean = capacity / nb;
        cap    = mean + (uint32_t)(8.0*std::sqrt((double)mean)) + 32u;
        nslots = (size_t)nb * cap;
        // set 0: 8-u64 records + the 9th-word plane behind them (see kSetStride in
        // src/gpu/cuda_solver.cu). MXBM_R2_FULL keeps the old padded 10-u64 record.
        // MXBM_R3_QUAD takes set 0 to 3 u64 instead: round 2's output is then the whole
        // of it (r4's is 2), so the set's stride is round 2's record and nothing else.
        // This is where the ~2.1 GiB of the footprint cut actually happens.
        const uint32_t setStride[2] = {
            MXBM_R2_FULL ? 10u : (MXBM_R3_QUAD ? 3u : 9u),
            MXBM_R2_FULL ? 10u : 8u };
        elem[0] = dalloc<uint64_t>(nslots*setStride[0]);
        elem[1] = dalloc<uint64_t>(nslots*setStride[1]);
        counts[0] = dalloc<uint32_t>(nb); counts[1] = dalloc<uint32_t>(nb);
        gictr = dalloc<uint32_t>(1); drops = dalloc<uint32_t>(4);
        left  = dalloc<uint32_t>((size_t)5*capacity);
        right = dalloc<uint32_t>((size_t)5*capacity);
        survSlots = dalloc<uint32_t>(survCap); survCount = dalloc<uint32_t>(1);
        dleaves = dalloc<uint32_t>((size_t)survCap*32);
        dpp = dalloc<uint64_t>(4);
        occmax = dalloc<uint32_t>(5); cudaMemset(occmax, 0, 20);
        for (int k = 0; k < 2; ++k) {
            entryOut[k]    = dalloc<uint64_t>(nslots);
            entryCounts[k] = dalloc<uint32_t>(nb);
            dpp2[k]        = dalloc<uint64_t>(4);
            cudaEventCreateWithFlags(&entryDone[k], cudaEventDisableTiming);
            cudaEventCreateWithFlags(&r1Done[k],    cudaEventDisableTiming);
        }
        cudaStreamCreate(&sMain); cudaStreamCreate(&sEntry);
        if (cob_s() && (cob_s() < 2u || ((nb << sm) % (cob_s() - 1u)) != 0u)) {
            printf("FAIL: MXBM_COB_S-1 must divide nb<<sm (%u)\n", nb << sm);
            return false;
        }
        #define CARVE(K) cudaFuncSetAttribute(K, cudaFuncAttributePreferredSharedMemoryCarveout, \
                                              cudaSharedmemCarveoutMaxShared)
        CARVE((fused_round<MXBM_R1_ARGS>)); CARVE((fused_round<MXBM_R2_ARGS>));
        CARVE((fused_round<MXBM_R3_ARGS>)); CARVE((fused_round<MXBM_R4_ARGS>));
        CARVE(terminal_round);
        #undef CARVE
        return elem[0] && elem[1] && left && right && dpp
            && entryOut[0] && entryOut[1] && entryCounts[0] && entryCounts[1];
    }

    // entry for `nonce` into slot, on `st`. Split out of solve() so it can be issued on a
    // second stream against the PREVIOUS nonce's r3/r4. Waits on r1Done[slot] first: that
    // slot's last reader was r1 of an earlier solve, and overwriting it before that read
    // retires is a write-after-read hazard the streams do not order for us.
    void launch_entry(const uint8_t input[32], const uint8_t nonce[8], int slot, cudaStream_t st) {
        uint64_t pp[4]; const uint8_t extra0[4] = {0,0,0,0};
        bh3::compute_prepow(input, 32, nonce, extra0, pp);
        if (r1Ever[slot]) cudaStreamWaitEvent(st, r1Done[slot], 0);
        cudaMemcpyAsync(dpp2[slot], pp, 32, cudaMemcpyHostToDevice, st);
        static const int reps = [] {
            const char* e = getenv("MXBM_ENTRY_REPS"); int v = e ? atoi(e) : 1;
            return v < 1 ? 1 : v;
        }();
        // Re-zero BEFORE each rep: entry_scatter atomically appends, so repeating it over
        // a single zeroing would overflow every bucket and trip the drop counters. Each
        // rep is therefore a complete, correct pass and the last one leaves valid state --
        // the only thing that changes is how long entry occupies the stream.
        for (int r = 0; r < reps; ++r) {
            cudaMemsetAsync(entryCounts[slot], 0, (size_t)nb*4, st);
            entry_scatter<<<(elems+255)/256,256,0,st>>>(dpp2[slot], 0, elems, bb, cap,
                                                        entryCounts[slot], entryOut[slot], drops);
        }
        if (occ_on()) occ_reduce<<<(nb+255)/256,256,0,st>>>(entryCounts[slot], nb, occmax);
        cudaEventRecord(entryDone[slot], st);
    }

    // MXBM_ROUND_REPS="R:N" replays round R (1..4, or 5 = terminal) N times inside the
    // solve that produced its input, so an external NVML sampler can attribute BOARD
    // POWER to one stage. Replaying in place is what makes it honest: the input bytes,
    // the bucket occupancies and the output addresses are the ones the round really
    // sees, so the memory traffic per rep is identical to the real thing. Each rep
    // re-zeros the output counters first, exactly as MXBM_ENTRY_REPS does, so every rep
    // is a complete correct pass and the last leaves valid state for the next round.
    // Which round hosts the co-tenant entry. r3 by default: longest of the DRAM-bound
    // rounds (9.5 ms at 21 % SM), so the most idle issue slots to fill. MXBM_CO_ROUND.
    static int co_round() { static const int v = []{ const char* e=getenv("MXBM_CO_ROUND");
                                return e ? atoi(e) : 3; }(); return v; }
    // MXBM_COB_S=S: run the co-tenant entry as SEPARATE INTERLEAVED BLOCKS (every S-th
    // block of the host round's grid) instead of in the round's own warps. S-1 must
    // divide nb << sm. 0 = off (classic COTENANT tail-hosting).
    static uint32_t cob_s() { static const uint32_t v = []{ const char* e=getenv("MXBM_COB_S");
                                  return e ? (uint32_t)atoi(e) : 0u; }(); return v; }
    // MXBM_CO_ROUND=34: split the entry work across TWO hosts -- r3 seeds the first half
    // of the index space and r4 the second -- so each host carries half the co-work and
    // the entry side has both rounds' combined duration to finish in.
    static bool co_hosts(int R) { const int cr = co_round();
        return cr == 34 ? (R == 3 || R == 4) : (R == cr); }
    // MXBM_COB_DUMMY=1: co-blocks are launched but given ZERO work, so they return
    // immediately -- measures pure block displacement + grid overhead. The real entry is
    // launched separately by the driver, exactly as the NOSCATTER diagnostic does.
    static bool cob_dummy() { static const bool v = getenv("MXBM_COB_DUMMY") != nullptr;
        return v; }
    static void co_slice(int R, uint32_t total, uint32_t& begin, uint32_t& count) {
        if (co_round() == 34) { const uint32_t h = total / 2u;
            begin = (R == 3) ? 0u : h; count = (R == 3) ? h : total - h; }
        else { begin = 0u; count = total; }
    }
    static int rep_round() { static const int v = []{ const char* e=getenv("MXBM_ROUND_REPS");
                                 return e ? atoi(e) : 0; }(); return v; }
    static int rep_count() { static const int v = []{ const char* e=getenv("MXBM_ROUND_REPS");
                                 const char* c = e ? strchr(e,':') : nullptr;
                                 int n = c ? atoi(c+1) : 1; return n<1?1:n; }(); return v; }

    // Rounds 1..terminal + recover + CPU verify, consuming entry's output from `slot`.
    // `after_r1` is invoked once r1 has been ISSUED (not completed), which is the point at
    // which the next nonce's entry may be queued on the other stream.
    // coNonce != nullptr: host that nonce's entry pass inside round `coRound`'s blocks,
    // instead of launching entry_scatter for it separately. See COTENANT in fused_round.
    template<class F>
    std::vector<std::array<uint8_t,104>> finish_solve(const uint8_t input[32], const uint8_t nonce[8],
                                                      int slot, cudaStream_t st, F after_r1,
                                                      uint32_t* survOut = nullptr, uint32_t* dropOut = nullptr,
                                                      const uint8_t* coNonce = nullptr, int coSlot = 0) {
        std::vector<std::array<uint8_t,104>> out;
        uint64_t pp[4]; const uint8_t extra0[4] = {0,0,0,0};
        bh3::compute_prepow(input, 32, nonce, extra0, pp);
        cudaMemcpyAsync(dpp, pp, 32, cudaMemcpyHostToDevice, st);
        if (coNonce) {
            uint64_t cpp[4];
            bh3::compute_prepow(input, 32, coNonce, extra0, cpp);
            cudaMemcpyAsync(dpp2[coSlot], cpp, 32, cudaMemcpyHostToDevice, st);
            // entry_scatter appends atomically, so its counters must be zeroed first --
            // exactly as launch_entry does before its own launch.
            cudaMemsetAsync(entryCounts[coSlot], 0, (size_t)nb*4, st);
        }
        cudaMemsetAsync(drops, 0, 16, st); cudaMemsetAsync(survCount, 0, 4, st);
        cudaStreamWaitEvent(st, entryDone[slot], 0);
        int inSet = 0;
        // R==1 reads entry's dedicated stride-1 buffer instead of elem[0]; every later
        // round ping-pongs exactly as before, so the only change is r1's input pointer.
        #define ROUND(R, ...)                                                            \
            { const int o = inSet ^ 1;                                                   \
              const uint32_t* inC = ((R)==1) ? entryCounts[slot] : counts[inSet];        \
              const uint64_t* inE = ((R)==1) ? entryOut[slot]    : elem[inSet];          \
              const int reps = (rep_round() == (R)) ? rep_count() : 1;                   \
              for (int rp = 0; rp < reps; ++rp) {                                         \
                cudaMemsetAsync(counts[o], 0, (size_t)nb*4, st);                         \
                cudaMemsetAsync(gictr, 0, 4, st);                                        \
                if (MXBM_SUBPASS)                                                        \
                  fused_round<__VA_ARGS__,false,true>\
                    <<<nb, kWG, 0, st>>>(bb, sm, cap, cap, (uint32_t)((R)-1)*capacity,   \
                        (uint32_t*)inC, (uint64_t*)inE, counts[o], elem[o],               \
                        left, right, gictr, drops, dpp);                                 \
                else if (coNonce && co_hosts(R) && cob_s()) {                            \
                  uint32_t cb_ = 0, cc_ = 0; co_slice((R), elems, cb_, cc_);             \
                  if (cob_dummy()) cc_ = 0;                                              \
                  fused_round<__VA_ARGS__,false,false,true>\
                    <<<(nb << sm) + (nb << sm)/(cob_s()-1u), kWG, 0, st>>>(               \
                        bb, sm, cap, cap, (uint32_t)((R)-1)*capacity,                    \
                        (uint32_t*)inC, (uint64_t*)inE, counts[o], elem[o],               \
                        left, right, gictr, drops, dpp,                                  \
                        dpp2[coSlot], cc_, cap, entryCounts[coSlot], entryOut[coSlot],   \
                        cob_s(), cb_); }                                                 \
                else if (coNonce && (R) == co_round())                                   \
                  fused_round<__VA_ARGS__,true>\
                    <<<nb << sm, kWG, 0, st>>>(bb, sm, cap, cap, (uint32_t)((R)-1)*capacity,\
                        (uint32_t*)inC, (uint64_t*)inE, counts[o], elem[o],               \
                        left, right, gictr, drops, dpp,                                  \
                        dpp2[coSlot], elems, cap, entryCounts[coSlot], entryOut[coSlot]); \
                else                                                                     \
                  fused_round<__VA_ARGS__>                                                 \
                    <<<nb << sm, kWG, 0, st>>>(bb, sm, cap, cap, (uint32_t)((R)-1)*capacity,\
                        (uint32_t*)inC, (uint64_t*)inE, counts[o], elem[o],               \
                        left, right, gictr, drops, dpp); }                                \
              if (occ_on())                                                             \
                occ_reduce<<<(nb+255)/256,256,0,st>>>(counts[o], nb, occmax + (R));      \
              if ((R)==1) { cudaEventRecord(r1Done[slot], st); r1Ever[slot] = true;       \
                            after_r1(); }                                                \
              inSet = o; }
        // ROUND_X exists only to expand MXBM_Rn_ARGS first: a function-like macro counts
        // its arguments before expanding them, so ROUND(1, MXBM_R1_ARGS) reads as 2 args.
        #define ROUND_X(...) ROUND(__VA_ARGS__)
#if MXBM_R2_FULL
        ROUND_X(1, MXBM_R1_ARGS)
        ROUND_X(2, MXBM_R2_ARGS)
#else
        ROUND_X(1, MXBM_R1_ARGS)
        ROUND_X(2, MXBM_R2_ARGS)
#endif
        ROUND_X(3, MXBM_R3_ARGS)
        ROUND_X(4, MXBM_R4_ARGS)
        #undef ROUND
        #undef ROUND_X
        for (int rp = 0, treps = (rep_round() == 5) ? rep_count() : 1; rp < treps; ++rp) {
            cudaMemsetAsync(survCount, 0, 4, st);
            terminal_round<<<nb << sm, kWG, 0, st>>>(bb, sm, cap, 4u*capacity, counts[inSet],
                                              elem[inSet], left, right, survSlots, survCount,
                                              survCap, drops);
        }

        cudaError_t le = cudaGetLastError();
        if (le != cudaSuccess) {
            // A launch that fails (bad launch bounds, too many threads) otherwise just
            // yields zero survivors, which reads like a correctness bug in the kernels.
            printf("CUDA launch error: %s\n", cudaGetErrorString(le));
            return out;
        }
        uint32_t hs = 0, hd[4] = {0,0,0,0};
        cudaMemcpyAsync(&hs, survCount, 4, cudaMemcpyDeviceToHost, st);
        cudaMemcpyAsync(hd, drops, 16, cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);
        if (hs > survCap) hs = survCap;
        if (survOut) *survOut = hs;
        if (dropOut) *dropOut = hd[1] | hd[2] | hd[3];
        if (hs == 0) return out;

        recover<<<(hs+63)/64, 64, 0, st>>>(hs, survSlots, capacity, left, right, dleaves);
        std::vector<uint32_t> hl((size_t)hs*32);
        cudaMemcpyAsync(hl.data(), dleaves, (size_t)hs*32*4, cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);
        for (uint32_t i = 0; i < hs; ++i) {
            std::array<uint8_t,104> sol{};
            bh3::pack_indices(&hl[(size_t)i*32], sol.data());
            if (bh3::is_valid_solution(input, 32, nonce, sol.data())) out.push_back(sol);
        }
        return out;
    }

    // Sequential convenience wrapper -- same work, no overlap. The KAT gate uses this.
    std::vector<std::array<uint8_t,104>> solve(const uint8_t input[32], const uint8_t nonce[8],
                                               uint32_t* survOut = nullptr, uint32_t* dropOut = nullptr) {
        launch_entry(input, nonce, 0, sMain);
        return finish_solve(input, nonce, 0, sMain, []{}, survOut, dropOut);
    }

    // ---- pipe2: two solves in flight; r4(i) and r1(i+1) share ONE launch -------------
    //
    // r1's output lives in its own double-buffered bucket set (pair records, stride 2)
    // and its back-refs in a second BR set, so nothing it writes collides with solve i
    // still in flight. The main elem/counts ping-pong is HANDED OVER between solves:
    // r2(i+1) writes elem[0] only after terminal(i) has read it, which the in-order
    // stream guarantees. Extra footprint: 2 x 0.73 GiB pair buckets + 1.29 GiB BR set.
    uint64_t *pairOut[2] = {nullptr,nullptr};
    uint32_t *pairCounts[2] = {nullptr,nullptr};
    uint32_t *br2L = nullptr, *br2R = nullptr;      // back-ref set for odd-parity solves
    uint32_t *gictrB = nullptr;                     // r1(i+1)'s gi counter
    uint32_t* brL(int p) { return p ? br2L : left; }
    uint32_t* brR(int p) { return p ? br2R : right; }

    bool init_pipe2() {
        for (int k = 0; k < 2; ++k) {
            pairOut[k]    = dalloc<uint64_t>(nslots * 2);
            pairCounts[k] = dalloc<uint32_t>(nb);
        }
        br2L = dalloc<uint32_t>((size_t)5*capacity);
        br2R = dalloc<uint32_t>((size_t)5*capacity);
        gictrB = dalloc<uint32_t>(1);
        return pairOut[0] && pairOut[1] && pairCounts[0] && pairCounts[1]
            && br2L && br2R && gictrB;
    }

    // Standalone r1 for the solve in slot p: entryOut[p] -> pairOut[p], BR(p) level 0.
    // Used by the prologue; in steady state r1 rides inside fused_pair instead.
    void launch_r1(int p, cudaStream_t st) {
        cudaMemsetAsync(pairCounts[p], 0, (size_t)nb*4, st);
        cudaMemsetAsync(gictrB, 0, 4, st);
        fused_round<MXBM_R1_ARGS><<<nb << sm, kWG, 0, st>>>(bb, sm, cap, cap, 0u,
            entryCounts[p], entryOut[p], pairCounts[p], pairOut[p],
            brL(p), brR(p), gictrB, drops, dpp2[p]);
    }

    // One steady-state iteration: r2(i) r3(i) entry(i+1) [r4(i) || r1(i+1)] term(i),
    // then readback + recover + CPU verify for solve i. nextNonce == nullptr drains the
    // pipeline with a plain r4.
    std::vector<std::array<uint8_t,104>> pipe2_iter(const uint8_t input[32],
            const uint8_t nonce[8], const uint8_t* nextNonce, int p, uint32_t* dropOut) {
        cudaStream_t st = sMain;
        std::vector<std::array<uint8_t,104>> out;
        uint64_t pp[4]; const uint8_t extra0[4] = {0,0,0,0};
        bh3::compute_prepow(input, 32, nonce, extra0, pp);
        cudaMemcpyAsync(dpp, pp, 32, cudaMemcpyHostToDevice, st);
        cudaMemsetAsync(drops, 0, 16, st);
        cudaMemsetAsync(survCount, 0, 4, st);
        // r2(i): pair records -> packed records
        cudaMemsetAsync(counts[0], 0, (size_t)nb*4, st); cudaMemsetAsync(gictr, 0, 4, st);
        fused_round<MXBM_R2_ARGS><<<nb << sm, kWG, 0, st>>>(bb, sm, cap, cap, 1u*capacity,
            pairCounts[p], pairOut[p], counts[0], elem[0],
            brL(p), brR(p), gictr, drops, dpp);
        // r3(i)
        cudaMemsetAsync(counts[1], 0, (size_t)nb*4, st); cudaMemsetAsync(gictr, 0, 4, st);
        fused_round<MXBM_R3_ARGS><<<nb << sm, kWG, 0, st>>>(bb, sm, cap, cap, 2u*capacity,
            counts[0], elem[0], counts[1], elem[1],
            brL(p), brR(p), gictr, drops, dpp);
        // entry(i+1), then the heterogeneous launch. MXBM_P2_VAR picks the pairing:
        //   0  r4(i) || r1(i+1), 1:1 residency          (grid 2R, alternating)
        //   1  r1(i+1)-major: r4 blocks doubled, so 2/3 of residency is r1's
        static const int var = []{ const char* e=getenv("MXBM_P2_VAR");
                                   return e ? atoi(e) : 0; }();
        if (nextNonce) launch_entry(input, nextNonce, p ^ 1, st);
        cudaMemsetAsync(counts[0], 0, (size_t)nb*4, st); cudaMemsetAsync(gictr, 0, 4, st);
        if (nextNonce) {
            cudaMemsetAsync(pairCounts[p^1], 0, (size_t)nb*4, st);
            cudaMemsetAsync(gictrB, 0, 4, st);
            RoundArgs A{bb, sm, cap, cap, 3u*capacity, counts[1], elem[1],
                        counts[0], elem[0], brL(p), brR(p), gictr, drops, dpp};
            RoundArgs B{bb, sm, cap, cap, 0u, entryCounts[p^1], entryOut[p^1],
                        pairCounts[p^1], pairOut[p^1], brL(p^1), brR(p^1), gictrB, drops,
                        dpp2[p^1]};
            if (var == 1)
                // A = r1(i+1) gets 2 of every 3 resident blocks; B = r4(i), each B block
                // runs two consecutive buckets. Grid: R r1-blocks + R/2 r4-blocks.
                fused_pair<MXBM_R1_ARGS, MXBM_R4_ARGS, 2u>
                    <<<(nb << sm) + (nb << sm)/2u, kWG, 0, st>>>(B, A, 3u);
            else
                fused_pair<MXBM_R4_ARGS, MXBM_R1_ARGS>
                    <<<2u*(nb << sm), kWG, 0, st>>>(A, B, 2u);
        } else {
            fused_round<MXBM_R4_ARGS><<<nb << sm, kWG, 0, st>>>(bb, sm, cap, cap,
                3u*capacity, counts[1], elem[1], counts[0], elem[0],
                brL(p), brR(p), gictr, drops, dpp);
        }
        terminal_round<<<nb << sm, kWG, 0, st>>>(bb, sm, cap, 4u*capacity, counts[0],
            elem[0], brL(p), brR(p), survSlots, survCount, survCap, drops);

        cudaError_t le = cudaGetLastError();
        if (le != cudaSuccess) {
            printf("CUDA launch error: %s\n", cudaGetErrorString(le));
            return out;
        }
        uint32_t hs = 0, hd[4] = {0,0,0,0};
        cudaMemcpyAsync(&hs, survCount, 4, cudaMemcpyDeviceToHost, st);
        cudaMemcpyAsync(hd, drops, 16, cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);
        if (hs > survCap) hs = survCap;
        if (dropOut) *dropOut = hd[1] | hd[2] | hd[3];
        if (hs == 0) return out;
        recover<<<(hs+63)/64, 64, 0, st>>>(hs, survSlots, capacity, brL(p), brR(p), dleaves);
        std::vector<uint32_t> hl((size_t)hs*32);
        cudaMemcpyAsync(hl.data(), dleaves, (size_t)hs*32*4, cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);
        for (uint32_t i = 0; i < hs; ++i) {
            std::array<uint8_t,104> sol{};
            bh3::pack_indices(&hl[(size_t)i*32], sol.data());
            if (bh3::is_valid_solution(input, 32, nonce, sol.data())) out.push_back(sol);
        }
        return out;
    }
};

int main(int argc, char** argv) {
    CudaSolver s;
    if (!s.init()) { printf("FAIL: allocation\n"); return 1; }
    printf("geometry : bb=%u sm=%u nb=%u cap=%u\n", s.bb, s.sm, s.nb, s.cap);
    // The two set strides summed: 9+8 normally, 20 under R2_FULL's padded record, and
    // 3+8 under R3_QUAD -- which is the whole of that variant's footprint claim, so it
    // is printed rather than asserted from a comment.
    printf("footprint: %.2f GiB\n",
           ((double)s.nslots*(MXBM_R2_FULL ? 20 : (MXBM_R3_QUAD ? 11 : 17))*8
            + (double)5*s.capacity*4*2)/(double)(1u<<30));

    // Why a second stream cannot overlap these kernels: concurrent execution needs the
    // FIRST kernel to run out of blocks before the work distributor will schedule the
    // second one's. Print resident capacity vs blocks pending, i.e. the wave count.
    {
        cudaDeviceProp pr{}; cudaGetDeviceProperties(&pr, 0);
        auto show = [&](const char* nm, const void* fn, int grid) {
            int bpsm = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsm, fn, kWG, 0);
            const int resident = bpsm * pr.multiProcessorCount;
            printf("  %-10s blocks/SM=%d  resident=%-6d grid=%-7d waves=%.0f\n",
                   nm, bpsm, resident, grid, resident ? (double)grid/resident : 0.0);
        };
        printf("occupancy (%d SMs):\n", pr.multiProcessorCount);
        show("entry", (const void*)entry_scatter, (s.elems+255)/256);
        show("r1", (const void*)(fused_round<MXBM_R1_ARGS>), s.nb << s.sm);
        show("r2", (const void*)(fused_round<MXBM_R2_ARGS>), s.nb << s.sm);
        show("r3", (const void*)(fused_round<MXBM_R3_ARGS>), s.nb << s.sm);
        show("r4", (const void*)(fused_round<MXBM_R4_ARGS>), s.nb << s.sm);
    }

    // --- correctness gate: the KAT input has exactly 3 known goldens ---
    uint32_t surv=0, drop=0;
    auto sols = s.solve(kat::input32, kat::nonce0, &surv, &drop);
    int matched = 0;
    for (auto& sol : sols)
        for (int g = 0; g < 3; ++g)
            if (memcmp(sol.data(), kat::golden[g], 104) == 0) { ++matched; break; }
    const bool gate = (surv == 3) && (drop == 0) && (sols.size() == 3) && (matched == 3);
    printf("KAT      : survivors=%u verified=%zu goldens=%d/3 drops=%u -> %s\n",
           surv, sols.size(), matched, drop, gate ? "PASS" : "FAIL");
    // An attribution build computes the wrong answer BY DESIGN, so the gate cannot also
    // be the exit condition for it. Everything else still has to pass it.
    // MXBM_FORCE_TIMING: time a configuration that fails the gate. Only ever valid when
    // the drop count is reported alongside, since drops REMOVE downstream work and thus
    // flatter the result -- a config that is not faster despite dropping is conclusive.
    const bool force = getenv("MXBM_FORCE_TIMING") != nullptr;
    if (!gate && !MXBM_ABL && !force) return 1;
    if (MXBM_ABL) printf("  *** ablation build (derive=%d emit=%d): timings only, results WRONG\n",
                         MXBM_ABL_DERIVE, MXBM_ABL_EMIT);

    // --- like-for-like timing: distinct nonces, end-to-end, verified solutions ---
    //
    // --overlap runs entry(i+1) on a second stream against r2..r4 of nonce i. The A/B is
    // in ONE binary so the two paths share a compilation, the same allocations and the
    // same kernels -- only the scheduling differs.
    //
    // Throughput is reported as total wall / n, not a per-solve median, because under
    // overlap solves intentionally share the clock and a per-solve figure would
    // double-count the shared time. The same figure is printed for both paths so the
    // comparison is like-for-like.
    const int n = (argc > 1) ? atoi(argv[1]) : 20;
    bool overlap = false, fuse = false, pipe2 = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--overlap")) overlap = true;
        if (!strcmp(argv[i], "--fuse"))    fuse = true;
        if (!strcmp(argv[i], "--pipe2"))   pipe2 = true;
    }
    if (pipe2) {
        if (!s.init_pipe2()) { printf("FAIL: pipe2 allocation\n"); return 1; }
        cudaDeviceProp pr{}; cudaGetDeviceProperties(&pr, 0);
        int bpsm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsm,
            (const void*)(fused_pair<MXBM_R4_ARGS, MXBM_R1_ARGS>), kWG, 0);
        printf("  r4||r1 pair blocks/SM=%d (needs 4 to match the homogeneous launches)\n",
               bpsm);
    }

    uint8_t nonce[8]; memcpy(nonce, kat::nonce0, 8);
    auto nonce_for = [&](int i, uint8_t* out) {
        memcpy(out, kat::nonce0, 8);
        out[7] = (uint8_t)(kat::nonce0[7] + i + 1);
    };

    size_t verified = 0; uint32_t worstDrop = 0;
    std::vector<double> ms;
    cudaDeviceSynchronize();
    auto T0 = std::chrono::steady_clock::now();

    if (pipe2) {
        // Prologue: entry(0) + standalone r1(0); each iteration then finishes solve i
        // and runs r1(i+1) inside r4(i)'s launch.
        uint8_t cur[8], nxt[8];
        nonce_for(0, cur);
        s.launch_entry(kat::input32, cur, 0, s.sMain);
        s.launch_r1(0, s.sMain);
        for (int i = 0; i < n; ++i) {
            nonce_for(i, cur);
            const bool more = (i + 1 < n);
            if (more) nonce_for(i + 1, nxt);
            auto t0 = std::chrono::steady_clock::now();
            uint32_t d = 0;
            auto v = s.pipe2_iter(kat::input32, cur, more ? nxt : nullptr, i & 1, &d);
            cudaDeviceSynchronize();
            ms.push_back(std::chrono::duration<double,std::milli>(
                             std::chrono::steady_clock::now() - t0).count());
            verified += v.size(); worstDrop |= d;
        }
    } else if (fuse) {
        // Solve i's round `co_round` hosts solve i+1's entry pass, so entry_scatter is
        // launched exactly ONCE (to prime solve 0) instead of once per solve. If the
        // co-residency does nothing, this costs the same as sequential; if it works, the
        // entry pass is free from solve 1 onward.
        uint8_t cur[8], nxt[8];
        nonce_for(0, cur);
        s.launch_entry(kat::input32, cur, 0, s.sMain);          // prime solve 0 only
        for (int i = 0; i < n; ++i) {
            nonce_for(i, cur);
            const int slot = i & 1;
            const bool more = (i + 1 < n);
            if (more) nonce_for(i + 1, nxt);
            auto t0 = std::chrono::steady_clock::now();
            uint32_t d = 0;
            auto v = s.finish_solve(kat::input32, cur, slot, s.sMain, []{}, nullptr, &d,
                                    more ? nxt : nullptr, (i + 1) & 1);
            // NOSCATTER is a diagnostic: the co-tenant computes but stores nothing, so it
            // cannot feed the next solve. Launch the real entry as well, making the
            // co-tenant PURELY ADDITIVE compute -- the delta against sequential is then
            // exactly what it costs to host entry's arithmetic inside a round, with its
            // scatter left out. (Without this the pipeline runs on an empty buffer and
            // "3.84 ms/solve" measures nothing at all.)
            if ((MXBM_CO_NOSCATTER || CudaSolver::cob_dummy()) && more)
                s.launch_entry(kat::input32, nxt, (i + 1) & 1, s.sMain);
            cudaDeviceSynchronize();
            ms.push_back(std::chrono::duration<double,std::milli>(
                             std::chrono::steady_clock::now() - t0).count());
            verified += v.size(); worstDrop |= d;
        }
    } else if (!overlap) {
        for (int i = 0; i < n; ++i) {
            nonce_for(i, nonce);
            auto t0 = std::chrono::steady_clock::now();
            uint32_t d = 0;
            auto v = s.solve(kat::input32, nonce, nullptr, &d);
            cudaDeviceSynchronize();
            ms.push_back(std::chrono::duration<double,std::milli>(
                             std::chrono::steady_clock::now() - t0).count());
            verified += v.size(); worstDrop |= d;
        }
    } else {
        uint8_t cur[8], nxt[8];
        nonce_for(0, cur);
        s.launch_entry(kat::input32, cur, 0, s.sEntry);        // prime slot 0
        for (int i = 0; i < n; ++i) {
            nonce_for(i, cur);
            const int slot = i & 1;
            auto t0 = std::chrono::steady_clock::now();
            uint32_t d = 0;
            // after_r1 fires once r1(i) is ISSUED and its read of `slot` is ordered by
            // r1Done -- the earliest point the next entry can be safely queued.
            auto v = s.finish_solve(kat::input32, cur, slot, s.sMain,
                [&]{
                    if (i + 1 < n) {
                        uint8_t nn[8]; nonce_for(i + 1, nn);
                        s.launch_entry(kat::input32, nn, (i + 1) & 1, s.sEntry);
                    }
                }, nullptr, &d);
            ms.push_back(std::chrono::duration<double,std::milli>(
                             std::chrono::steady_clock::now() - t0).count());
            verified += v.size(); worstDrop |= d;
            (void)nxt;
        }
        cudaDeviceSynchronize();
    }

    const double wall = std::chrono::duration<double,std::milli>(
                            std::chrono::steady_clock::now() - T0).count();
    std::sort(ms.begin(), ms.end());
    const double med = ms[ms.size()/2];
    const double per = wall / n;
    const double spersolve = (double)verified / n;
    printf("mode      : %s\n", pipe2   ? "PIPE2 (r4(i) || r1(i+1) in one launch)"
                            : fuse    ? "FUSED (entry co-tenant in a round's blocks)"
                            : overlap ? "OVERLAP (entry on 2nd stream)"
                                      : "sequential (baseline)");
    printf("end-to-end: %.2f ms/solve (wall/%d)   [per-solve median %.2f ms]\n", per, n, med);
    printf("solutions : %.2f verified/solve  =>  %.1f sol/s\n", spersolve, spersolve*1000.0/per);
    printf("drops     : %u  %s\n", worstDrop, worstDrop ? "*** NONZERO -- RESULT INVALID ***" : "(clean)");

    if (occ_on()) {
        // Normalised the way fb_cap_for is written -- (max - mean)/sqrt(mean) with
        // mean = capacity/nb -- so the sd column compares directly against its 8.
        uint32_t occ[5] = {0,0,0,0,0};
        cudaMemcpy(occ, s.occmax, 20, cudaMemcpyDeviceToHost);
        const double mean = (double)s.capacity / (double)s.nb, sd = std::sqrt(mean);
        printf("occupancy : cap=%u (mean %.0f + 8 sd + 32), over %d solves x %u buckets\n",
               s.cap, mean, n + 1, s.nb);
        static const char* nm[5] = {"entry","r1","r2","r3","r4"};
        for (int i = 0; i < 5; ++i)
            printf("  %-6s max=%-6u  = mean + %.2f sd    (%.0f%% of cap)\n",
                   nm[i], occ[i], ((double)occ[i] - mean)/sd, 100.0*occ[i]/s.cap);
    }
    return 0;
}
