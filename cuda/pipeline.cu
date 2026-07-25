// CUDA row-bucket pipeline, gated on the KAT. Standalone: builds with nvcc, not wired
// into CMake, so an incomplete backend cannot destabilise the shipping OpenCL path.
//
//   nvcc -O3 -arch=sm_89 -I src -I kernels/cuda -I tests cuda/pipeline.cu -o cuda/pipeline
//
// The gate is the same one the OpenCL path uses and is non-negotiable: survivors == 3 on
// the KAT prePow, bucketDrops == 0, pairDrops == 0.
#include "pipeline_kernels.cuh"
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

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__); return 1; } }while(0)

// ---- solver: persistent buffers, one solve() per (input, nonce) ---------------------
// Mirrors src/gpu/gpu_solver.cpp so the timing is like-for-like: buffers are allocated
// ONCE and reused, and solve() returns only CPU-VERIFIED solutions. Comparing a
// kernel-time figure against the OpenCL solver's end-to-end number would flatter this
// side, which is the whole reason for building this.
template<class T> static T* dalloc(size_t n) { void* p=nullptr; cudaMalloc(&p, n*sizeof(T)); return (T*)p; }

struct CudaSolver {
    static constexpr uint32_t elems    = 1u << 25;
    static constexpr uint32_t capacity = elems + elems/32;    // 34,603,008
    static constexpr uint32_t bb = 16, sm = 1, nb = 1u << bb;
    static constexpr uint32_t survCap = 1024;
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
    uint64_t *entryOut[2] = {nullptr,nullptr};
    uint32_t *entryCounts[2] = {nullptr,nullptr};
    uint64_t *dpp2[2] = {nullptr,nullptr};       // rounds of solve i still use solve i's pp
    cudaStream_t sMain = nullptr, sEntry = nullptr;
    cudaEvent_t entryDone[2] = {nullptr,nullptr};   // entry(slot) finished writing
    cudaEvent_t r1Done[2]    = {nullptr,nullptr};   // r1 finished READING slot (WAR guard)
    bool r1Ever[2] = {false,false};

    bool init() {
        const uint32_t mean = capacity / nb;
        cap    = mean + (uint32_t)(8.0*std::sqrt((double)mean)) + 32u;
        nslots = (size_t)nb * cap;
        // set 0: 8-u64 records + the 9th-word plane behind them (see kSetStride in
        // src/gpu/cuda_solver.cu). MXBM_R2_FULL keeps the old padded 10-u64 record.
        const uint32_t setStride[2] = { MXBM_R2_FULL ? 10u : 9u, MXBM_R2_FULL ? 10u : 8u };
        elem[0] = dalloc<uint64_t>(nslots*setStride[0]);
        elem[1] = dalloc<uint64_t>(nslots*setStride[1]);
        counts[0] = dalloc<uint32_t>(nb); counts[1] = dalloc<uint32_t>(nb);
        gictr = dalloc<uint32_t>(1); drops = dalloc<uint32_t>(4);
        left  = dalloc<uint32_t>((size_t)5*capacity);
        right = dalloc<uint32_t>((size_t)5*capacity);
        survSlots = dalloc<uint32_t>(survCap); survCount = dalloc<uint32_t>(1);
        dleaves = dalloc<uint32_t>((size_t)survCap*32);
        dpp = dalloc<uint64_t>(4);
        for (int k = 0; k < 2; ++k) {
            entryOut[k]    = dalloc<uint64_t>(nslots);
            entryCounts[k] = dalloc<uint32_t>(nb);
            dpp2[k]        = dalloc<uint64_t>(4);
            cudaEventCreateWithFlags(&entryDone[k], cudaEventDisableTiming);
            cudaEventCreateWithFlags(&r1Done[k],    cudaEventDisableTiming);
        }
        cudaStreamCreate(&sMain); cudaStreamCreate(&sEntry);
        #define CARVE(K) cudaFuncSetAttribute(K, cudaFuncAttributePreferredSharedMemoryCarveout, \
                                              cudaSharedmemCarveoutMaxShared)
        CARVE((fused_round<7,7,2,LM_SEED,424u,2u,1u,2u,2u,1u,2u>));
        CARVE((fused_round<7,7,2,LM_RD2,400u,4u,2u,4u,4u,2u,10u>));
        CARVE((fused_round<7,6,4,LM_EMIT,376u,6u,4u,2u,8u,10u,8u>));
        CARVE((fused_round<6,1,2,LM_USE,288u,9u,2u,0u,0u,8u,2u>));
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
        cudaEventRecord(entryDone[slot], st);
    }

    // MXBM_ROUND_REPS="R:N" replays round R (1..4, or 5 = terminal) N times inside the
    // solve that produced its input, so an external NVML sampler can attribute BOARD
    // POWER to one stage. Replaying in place is what makes it honest: the input bytes,
    // the bucket occupancies and the output addresses are the ones the round really
    // sees, so the memory traffic per rep is identical to the real thing. Each rep
    // re-zeros the output counters first, exactly as MXBM_ENTRY_REPS does, so every rep
    // is a complete correct pass and the last leaves valid state for the next round.
    static int rep_round() { static const int v = []{ const char* e=getenv("MXBM_ROUND_REPS");
                                 return e ? atoi(e) : 0; }(); return v; }
    static int rep_count() { static const int v = []{ const char* e=getenv("MXBM_ROUND_REPS");
                                 const char* c = e ? strchr(e,':') : nullptr;
                                 int n = c ? atoi(c+1) : 1; return n<1?1:n; }(); return v; }

    // Rounds 1..terminal + recover + CPU verify, consuming entry's output from `slot`.
    // `after_r1` is invoked once r1 has been ISSUED (not completed), which is the point at
    // which the next nonce's entry may be queued on the other stream.
    template<class F>
    std::vector<std::array<uint8_t,104>> finish_solve(const uint8_t input[32], const uint8_t nonce[8],
                                                      int slot, cudaStream_t st, F after_r1,
                                                      uint32_t* survOut = nullptr, uint32_t* dropOut = nullptr) {
        std::vector<std::array<uint8_t,104>> out;
        uint64_t pp[4]; const uint8_t extra0[4] = {0,0,0,0};
        bh3::compute_prepow(input, 32, nonce, extra0, pp);
        cudaMemcpyAsync(dpp, pp, 32, cudaMemcpyHostToDevice, st);
        cudaMemsetAsync(drops, 0, 16, st); cudaMemsetAsync(survCount, 0, 4, st);
        cudaStreamWaitEvent(st, entryDone[slot], 0);
        int inSet = 0;
        // R==1 reads entry's dedicated stride-1 buffer instead of elem[0]; every later
        // round ping-pongs exactly as before, so the only change is r1's input pointer.
        #define ROUND(R, INW,OUTW,LEAFW,MODE, LOUT,PADN,SIN,SOUT,SBUILD, INSTR,OUTSTR)   \
            { const int o = inSet ^ 1;                                                   \
              const uint32_t* inC = ((R)==1) ? entryCounts[slot] : counts[inSet];        \
              const uint64_t* inE = ((R)==1) ? entryOut[slot]    : elem[inSet];          \
              const int reps = (rep_round() == (R)) ? rep_count() : 1;                   \
              for (int rp = 0; rp < reps; ++rp) {                                         \
                cudaMemsetAsync(counts[o], 0, (size_t)nb*4, st);                         \
                cudaMemsetAsync(gictr, 0, 4, st);                                        \
                fused_round<INW,OUTW,LEAFW,MODE,LOUT,PADN,SIN,SOUT,SBUILD,INSTR,OUTSTR>  \
                  <<<nb << sm, kWG, 0, st>>>(bb, sm, cap, cap, (uint32_t)((R)-1)*capacity,\
                      (uint32_t*)inC, (uint64_t*)inE, counts[o], elem[o],                \
                      left, right, gictr, drops, dpp); }                                 \
              if ((R)==1) { cudaEventRecord(r1Done[slot], st); r1Ever[slot] = true;       \
                            after_r1(); }                                                \
              inSet = o; }
#if MXBM_R2_FULL
        ROUND(1, 7,7,2,LM_SEEDF,424u,2u,1u,2u,2u, 1u,10u)
        ROUND(2, 7,7,2,LM_RAW,  400u,4u,2u,4u,4u, 10u,10u)
#else
        ROUND(1, 7,7,2,LM_SEED, 424u,2u,1u,2u,2u, 1u,2u)
        ROUND(2, 7,7,2,LM_RD2,  400u,4u,2u,4u,4u, 2u,8u)
#endif
        ROUND(3, 7,6,4,LM_EMIT, 376u,6u,4u,2u,8u, MXBM_R2_FULL ? 10u : 8u, 8u)
        ROUND(4, 6,1,2,LM_USE,  288u,9u,2u,0u,0u, 8u,2u)
        #undef ROUND
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
};

int main(int argc, char** argv) {
    CudaSolver s;
    if (!s.init()) { printf("FAIL: allocation\n"); return 1; }
    printf("geometry : bb=%u sm=%u nb=%u cap=%u\n", s.bb, s.sm, s.nb, s.cap);
    printf("footprint: %.2f GiB\n",
           ((double)s.nslots*(MXBM_R2_FULL ? 20 : 17)*8
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
        show("r1", (const void*)(fused_round<7,7,2,LM_SEED,424u,2u,1u,2u,2u,1u,2u>), s.nb << s.sm);
        show("r2", (const void*)(fused_round<7,7,2,LM_RD2,400u,4u,2u,4u,4u,2u,10u>), s.nb << s.sm);
        show("r3", (const void*)(fused_round<7,6,4,LM_EMIT,376u,6u,4u,2u,8u,10u,8u>), s.nb << s.sm);
        show("r4", (const void*)(fused_round<6,1,2,LM_USE,288u,9u,2u,0u,0u,8u,2u>), s.nb << s.sm);
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
    if (!gate) return 1;

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
    bool overlap = false;
    for (int i = 1; i < argc; ++i) if (!strcmp(argv[i], "--overlap")) overlap = true;

    uint8_t nonce[8]; memcpy(nonce, kat::nonce0, 8);
    auto nonce_for = [&](int i, uint8_t* out) {
        memcpy(out, kat::nonce0, 8);
        out[7] = (uint8_t)(kat::nonce0[7] + i + 1);
    };

    size_t verified = 0; uint32_t worstDrop = 0;
    std::vector<double> ms;
    cudaDeviceSynchronize();
    auto T0 = std::chrono::steady_clock::now();

    if (!overlap) {
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
    printf("mode      : %s\n", overlap ? "OVERLAP (entry on 2nd stream)" : "sequential (baseline)");
    printf("end-to-end: %.2f ms/solve (wall/%d)   [per-solve median %.2f ms]\n", per, n, med);
    printf("solutions : %.2f verified/solve  =>  %.1f sol/s\n", spersolve, spersolve*1000.0/per);
    printf("drops     : %u  %s\n", worstDrop, worstDrop ? "*** NONZERO -- RESULT INVALID ***" : "(clean)");
    return 0;
}
