// P3 of the solver-reorg plan (docs-internal/SOLVER_REORG.md): does a sorted-runs match
// core beat the chain-table round kernel on cycles, at how many registers?
//
// The reorganized r1 stages 8 B keys-and-seeds instead of 56 B work records: one block
// per bucket, a 256-bin counting sort over the 8 key bits that vary inside a 2^16
// bucket, then each lane derives ITS sorted slot's element into registers and pairs
// with run-mates by statically-unrolled constant-distance __shfl_up_sync (d = 1..4) --
// 15 register moves where the shipping kernel does 14 u64 shared loads. The sort knows
// each run's start (bins[k-1] after placement), so no __match_any is needed; partners
// beyond distance 4 or across a warp boundary walk sorted[] backwards and re-derive.
//
// Output is byte-identical pair records into the same 2^16 buckets, so P2's thin-record
// scatter cliff is never touched. Correctness here is multiset equality against the
// shipping fused_round on the same real entry output: per-bucket out-counts memcmp,
// emitted-pair count, and order-invariant XOR+ADD folds of the gi-masked records and
// the (left,right) back-ref tuples, across several nonces.
//
//   nvcc -O3 -arch=sm_89 -std=c++17 -diag-suppress 186 -I src -I kernels/cuda -I tests \
//        -I third_party/blake2b cuda/sorted_r1_probe.cu src/beamhash/bh3_blake2b.cpp \
//        src/beamhash/bh3_verify.cpp third_party/blake2b/blake2b-ref.c -o cuda/sorted_r1_probe
//
// RESULT (RTX 4070 Ti SUPER, 2026-07-31): **the gate FAILS, and not narrowly.**
// Multisets equal across every nonce tried, drops 0 -- the sorted core is correct --
// but it is 1.75x SLOWER than the chain kernel (8.56 vs 4.89 ms), against a <= 0.5x
// gate. The MXBM_P3_ABL carve decomposes it (compile-time carves, results wrong):
//
//     sort (count+scan+place+staged read)  0.44 ms   <- the sort itself is nearly free
//   + derive into registers               +2.5       <- same seed_element+mix entry runs
//   + match (shuffles, no emit)           +3.6       <- of which ~3.3 is the LEFTOVER
//   + emit (combine+mix+scatter)          +2.0          WALK, see below
//
// Three codegen traps found on the way, each worth remembering:
//   1. `const Elem* left = cond ? &a : &b` demotes BOTH elements to local memory
//      (a runtime-selected pointer cannot name registers): 168 B stack, 3x slower.
//      combine is a symmetric XOR, so order the TREE, not the operands.
//   2. Warp collectives inside a data-dependent loop (__match_any + REDUX'd trip
//      count + lane-indexed __shfl_sync) each compile to a WARPSYNC trampoline CALL.
//      Constant-distance shuffles in straight-line unrolled code compile bare.
//   3. The killer: a DIVERGENT re-derive costs ~32x its converged price. The leftover
//      walk re-derives ~2 partners per 32-slot window with ONE lane active -- ~2800
//      warp-issue slots against 1400 for the whole converged 32-element derive. The
//      walk serves 6.3% of pairs and costs 3.3 of 8.56 ms (measured by ABL=4).
//
// The verdict stands even at the design's floor: sort 0.44 + derive 2.5 + bare
// shuffles ~0.3 + perfectly converged emit ~0.6 = ~3.9 ms = 0.80x the chain kernel,
// nowhere near 0.5x. The premise behind the reorganization -- that chain walking,
// staging and bookkeeping dominate r1's 5.2 ms of core cycles -- is REFUTED: the
// chain kernel's own machinery costs ~0.6 ms (4.89 = derive 2.5 + emit ~1.8 + ~0.6).
// What actually fills r1's cycle budget is seed derivation and emit arithmetic,
// which no reorganization of the match can remove. See SOLVER_REORG.md for what
// this implies about where the reference miner's work-per-clock advantage cannot come from.
#include "pipeline_kernels.cuh"
#include "kat_vectors.h"
#include "beamhash/bh3_blake2b.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
using namespace mxbm;
using namespace mxbm::cuda;

#define MXBM_R1_ARGS 7,7,1,LM_SEED, 424u,2u,1u,2u,2u, 1u,2u,MXBM_R1_FCAP
#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__); return 1; } }while(0)

// Covers the bb=16 bucket cap (743). The input count is clamped to the cap before
// staging, so unlike the chain kernel there is no spill path and no drop path here.
constexpr uint32_t kSortCap = 768;

// LM_SEED's emit, verbatim semantics: left = smaller (lead, gi) which for r1 is one
// seed compare; combine at Lout 424; two-leaf mix; pair record + back-refs.
__device__ __forceinline__ void emit_pair_r1(
        const bh3::Elem& ea, uint32_t ia, const bh3::Elem& eb, uint32_t ib,
        uint32_t bucket_bits, uint32_t out_cap,
        uint32_t* __restrict__ out_counts, uint64_t* __restrict__ out_belem,
        uint32_t* __restrict__ all_left, uint32_t* __restrict__ all_right,
        uint32_t* __restrict__ gi_counter, uint32_t* __restrict__ drops) {
    // combine is a symmetric XOR, so only the (lead, gi) order of the TREE matters --
    // selecting between the two elements through pointers would demote both to local
    // memory (a runtime-selected pointer cannot be a register), which costs 3x the
    // whole kernel. Never reintroduce a `const Elem* left = cond ? &a : &b` here.
    uint32_t L = ia, R = ib;
    if (ib < ia) { L = ib; R = ia; }
    bh3::Elem c;
    bh3::combine(ea, eb, 424u, c);
    uint32_t ctree[2] = { L, R };
    bh3::apply_mix(c, ctree, 2u, 424u);
    const uint32_t ckey = (uint32_t)(c.w[0] & 0xFFFFFFu);
    const uint32_t cb   = ckey >> (24u - bucket_bits);
    const uint32_t cpos = atomicAdd(&out_counts[cb], 1u);
    if (cpos < out_cap) {
        const uint32_t cgi = atomicAdd(gi_counter, 1u);
        const size_t od = ((size_t)cb * out_cap + cpos) * 2u;
        out_belem[od + 0] = pair_w0(ckey, ctree[0]);
        out_belem[od + 1] = pair_w1(ctree[1], cgi);
        all_left[cgi] = L; all_right[cgi] = R;
    } else atomicAdd(&drops[2], 1u);
}

// ABL (MXBM_P3_ABL) carves the kernel for attribution -- results intentionally wrong:
//   1 = sort only (count + scan + place + staged read)   2 = sort + derive
//   3 = everything but the emit body (combine/mix/stores skipped, shuffles live)
// A compile-time parameter, not a runtime one: the carved code must be ABSENT, not
// branched around, or the ablation also measures fetch/layout effects of dead code.
template<int ABL>
__global__ __launch_bounds__(256)
void sorted_r1(uint32_t bucket_bits, uint32_t in_bucket_cap, uint32_t out_bucket_cap,
               const uint32_t* __restrict__ in_counts,
               const uint64_t* __restrict__ in_belem,
               uint32_t* __restrict__ out_counts, uint64_t* __restrict__ out_belem,
               uint32_t* __restrict__ all_left, uint32_t* __restrict__ all_right,
               uint32_t* __restrict__ gi_counter, uint32_t* __restrict__ drops,
               const uint64_t* __restrict__ pp4) {
    __shared__ uint64_t sorted[kSortCap];
    __shared__ uint32_t bins[256];
    __shared__ uint32_t wsum[8];
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t bucket = blockIdx.x;
    uint32_t cnt = in_counts[bucket];
    if (cnt > in_bucket_cap) cnt = in_bucket_cap;
    const size_t base = (size_t)bucket * in_bucket_cap;

    // COUNT: one shared atomic per element on the 8 varying key bits.
    bins[tid] = 0;
    __syncthreads();
    for (uint32_t p = tid; p < cnt; p += 256u)
        atomicAdd(&bins[(uint32_t)in_belem[base + p] & 0xFFu], 1u);
    __syncthreads();

    // SCAN: 256-bin exclusive prefix (warp scan + 8-way combine), bins become cursors.
    {
        const uint32_t c0 = bins[tid];
        uint32_t v = c0;
        #pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const uint32_t n = __shfl_up_sync(0xFFFFFFFFu, v, d);
            if (lane >= (uint32_t)d) v += n;
        }
        if (lane == 31u) wsum[warp] = v;
        __syncthreads();
        if (warp == 0) {
            uint32_t s = (lane < 8u) ? wsum[lane] : 0u;
            #pragma unroll
            for (int d = 1; d < 8; d <<= 1) {
                const uint32_t n = __shfl_up_sync(0xFFFFFFFFu, s, d);
                if (lane >= (uint32_t)d) s += n;
            }
            if (lane < 8u) wsum[lane] = s;
        }
        __syncthreads();
        bins[tid] = v + (warp ? wsum[warp - 1u] : 0u) - c0;
        __syncthreads();
    }

    // PLACE: re-read the 8 B record (L2-resident since COUNT) to its sorted slot.
    // Equal keys are now contiguous runs; order inside a run is arbitrary, exactly as
    // the chain order is in the shipping kernel.
    for (uint32_t p = tid; p < cnt; p += 256u) {
        const uint64_t rec = in_belem[base + p];
        sorted[atomicAdd(&bins[(uint32_t)rec & 0xFFu], 1u)] = rec;
    }
    __syncthreads();

    // MATCH: derive own element into registers, pair with run-mates.
    const uint64_t pp[4] = { pp4[0], pp4[1], pp4[2], pp4[3] };
    const uint32_t nwaves = (cnt + 255u) / 256u;
    uint64_t sink = 0;
    for (uint32_t wv = 0; wv < nwaves; ++wv) {
        const uint32_t s = wv * 256u + tid;
        const bool live = s < cnt;
        const uint64_t rec = live ? sorted[s] : 0u;
        // Dead lanes carry a per-lane sentinel >= 256 so they match nobody and can
        // still participate in every warp collective below.
        const uint32_t k9 = live ? ((uint32_t)rec & 0xFFu) : 256u + lane;
        const uint32_t idx = (uint32_t)(rec >> 32);
        sink ^= rec;
        if constexpr (ABL == 1) continue;
        bh3::Elem e{};
        if (live) {
            bh3::seed_element(pp, idx, e);
            uint32_t t1[1] = { idx };
            bh3::apply_mix(e, t1, 1u, 448u);
        }
        sink ^= e.w[0];
        if constexpr (ABL == 2) continue;
        // Pairing needs no __match_any: the sort itself knows the runs. After PLACE,
        // bins[k] is run k's END cursor, so run k's start is bins[k-1]. A first draft
        // used __match_any + a REDUX'd dynamic loop of lane-indexed shuffles -- and
        // every collective in that data-dependent loop compiled to a WARPSYNC
        // trampoline CALL, which alone cost 4.0 ms of a 9.1 ms kernel. Statically
        // unrolled constant-distance shuffles in straight-line code compile bare.
        const uint32_t run_start = live ? (k9 ? bins[k9 - 1u] : 0u) : s;
        const uint32_t rank = s - run_start;
        constexpr uint32_t kDMax = 4;   // covers rank<=4 in-warp; Poisson(2.06) beyond
        #pragma unroll
        for (uint32_t d = 1; d <= kDMax; ++d) {
            bh3::Elem b;
            #pragma unroll
            for (int w = 0; w < 7; ++w) b.w[w] = __shfl_up_sync(0xFFFFFFFFu, e.w[w], d);
            const uint32_t jidx = __shfl_up_sync(0xFFFFFFFFu, idx, d);
            if (live && d <= rank && d <= lane) {
                sink ^= b.w[0];
                if constexpr (ABL != 3)
                    emit_pair_r1(e, idx, b, jidx, bucket_bits, out_bucket_cap, out_counts,
                                 out_belem, all_left, all_right, gi_counter, drops);
            }
        }
        // Leftovers -- partners beyond kDMax or across the warp boundary -- walk
        // sorted[] backwards and re-derive (~2 per window: boundary straddles at mean
        // run 2, plus the Poisson tail past rank 4). One code path for both.
        const uint32_t covered = lane < kDMax ? lane : kDMax;
        if (ABL != 4 && live && rank > covered) {   // ABL 4: no leftover walk (timing only)
            uint32_t j = s - covered;
            while (j > run_start) {
                const uint64_t rj = sorted[--j];
                const uint32_t jidx = (uint32_t)(rj >> 32);
                bh3::Elem b;
                bh3::seed_element(pp, jidx, b);
                uint32_t t1[1] = { jidx };
                bh3::apply_mix(b, t1, 1u, 448u);
                sink ^= b.w[0];
                if constexpr (ABL != 3)
                    emit_pair_r1(e, idx, b, jidx, bucket_bits, out_bucket_cap, out_counts,
                                 out_belem, all_left, all_right, gi_counter, drops);
            }
        }
    }
    // Keeps the carved builds' arithmetic live; never fires on real data paths.
    if constexpr (ABL != 0)
        if (sink == 0x123456789ABCDEFull) atomicAdd(&drops[0], 1u);
}

// Order-invariant folds. gi assignment is nondeterministic in BOTH kernels, so the
// record fold masks it out of w1; the back-ref fold hashes the (left,right) tuple,
// which gi only indexes. XOR alone lets duplicate pairs cancel, so ADD rides along.
__global__ void fold_records(const uint32_t* __restrict__ counts, uint32_t cap,
                             const uint64_t* __restrict__ belem,
                             unsigned long long* __restrict__ acc) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = counts[b]; if (n > cap) n = cap;
    unsigned long long x = 0, a = 0;
    for (uint32_t p = 0; p < n; ++p) {
        const size_t d = ((size_t)b * cap + p) * 2u;
        const unsigned long long v = belem[d] ^ ((belem[d + 1] & 0x1FFFFFFull) << 1);
        x ^= v; a += v;
    }
    if (n) { atomicXor(&acc[0], x); atomicAdd(&acc[1], a); }
}
__global__ void fold_backrefs(const uint32_t* __restrict__ L,
                              const uint32_t* __restrict__ R, uint32_t n,
                              unsigned long long* __restrict__ acc) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned long long v = ((unsigned long long)L[i] << 32) | R[i];
    atomicXor(&acc[2], v); atomicAdd(&acc[3], v);
}

template<class T> static T* dalloc(size_t n) { void* p=nullptr; cudaMalloc(&p, n*sizeof(T)); return (T*)p; }

int main(int argc, char** argv) {
    const uint32_t elems = 1u << 25, capacity = elems + elems/32;
    const uint32_t bb = 16, sm = 1, nb = 1u << bb;
    const uint32_t mean = capacity / nb;
    const uint32_t cap  = mean + (uint32_t)(8.0*std::sqrt((double)mean)) + 32u;
    const size_t nslots = (size_t)nb * cap;
    const int nonces = (argc > 1) ? atoi(argv[1]) : 5;
    const int reps   = 15;
    const uint32_t abl = getenv("MXBM_P3_ABL") ? (uint32_t)atoi(getenv("MXBM_P3_ABL")) : 0u;
    if (abl) printf("*** MXBM_P3_ABL=%u: carved build, results WRONG, timing only\n", abl);

    uint64_t *entryOut = dalloc<uint64_t>(nslots), *dpp = dalloc<uint64_t>(4);
    uint32_t *entryCounts = dalloc<uint32_t>(nb), *drops = dalloc<uint32_t>(4);
    uint64_t *outA = dalloc<uint64_t>(nslots*2), *outB = dalloc<uint64_t>(nslots*2);
    uint32_t *cntA = dalloc<uint32_t>(nb), *cntB = dalloc<uint32_t>(nb);
    uint32_t *lA = dalloc<uint32_t>(capacity), *rA = dalloc<uint32_t>(capacity);
    uint32_t *lB = dalloc<uint32_t>(capacity), *rB = dalloc<uint32_t>(capacity);
    uint32_t *gi = dalloc<uint32_t>(1);
    unsigned long long* acc = dalloc<unsigned long long>(8);
    if (!entryOut || !outA || !outB || !lA || !rA || !lB || !rB) { printf("FAIL: alloc\n"); return 1; }
    cudaFuncSetAttribute((const void*)(fused_round<MXBM_R1_ARGS>),
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    cudaFuncSetAttribute((const void*)(sorted_r1<0>),
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);

    { // resources + occupancy, the register gate's raw numbers
        cudaDeviceProp pr{}; cudaGetDeviceProperties(&pr, 0);
        auto show = [&](const char* nm, const void* fn) {
            cudaFuncAttributes at{}; cudaFuncGetAttributes(&at, fn);
            int bpsm = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsm, fn, 256, 0);
            printf("  %-10s regs=%-3d shared=%-6zu blocks/SM=%d\n", nm, at.numRegs,
                   at.sharedSizeBytes, bpsm);
        };
        printf("resources (%d SMs):\n", pr.multiProcessorCount);
        show("chain r1", (const void*)(fused_round<MXBM_R1_ARGS>));
        show("sorted r1", (const void*)(sorted_r1<0>));
    }

    std::vector<double> msA, msB;
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    bool all_ok = true;
    unsigned long long worstDrops = 0;

    for (int ni = 0; ni < nonces; ++ni) {
        uint8_t nonce[8]; memcpy(nonce, kat::nonce0, 8);
        nonce[7] = (uint8_t)(kat::nonce0[7] + ni);
        uint64_t pp[4]; const uint8_t extra0[4] = {0,0,0,0};
        bh3::compute_prepow(kat::input32, 32, nonce, extra0, pp);
        CK(cudaMemcpy(dpp, pp, 32, cudaMemcpyHostToDevice));
        CK(cudaMemset(entryCounts, 0, (size_t)nb*4));
        CK(cudaMemset(drops, 0, 16));
        entry_scatter<<<(elems+255)/256,256>>>(dpp, 0, elems, bb, cap, entryCounts,
                                               entryOut, drops);

        auto run = [&](bool sorted_path, uint32_t* oc, uint64_t* ob, uint32_t* ol,
                       uint32_t* orr, std::vector<double>& ms) {
            for (int rp = 0; rp < reps; ++rp) {
                cudaMemset(oc, 0, (size_t)nb*4);
                cudaMemset(gi, 0, 4);
                cudaEventRecord(t0);
                if (sorted_path) {
                    #define SR1(A) sorted_r1<A><<<nb, 256>>>(bb, cap, cap, entryCounts, \
                        entryOut, oc, ob, ol, orr, gi, drops, dpp)
                    switch (abl) { case 1: SR1(1); break; case 2: SR1(2); break;
                                   case 3: SR1(3); break; case 4: SR1(4); break;
                                   default: SR1(0); }
                    #undef SR1
                }
                else
                    fused_round<MXBM_R1_ARGS><<<nb << sm, kWG>>>(bb, sm, cap, cap, 0u,
                        entryCounts, entryOut, oc, ob, ol, orr, gi, drops, dpp);
                cudaEventRecord(t1); cudaEventSynchronize(t1);
                float f = 0; cudaEventElapsedTime(&f, t0, t1); ms.push_back(f);
            }
            uint32_t n = 0; cudaMemcpy(&n, gi, 4, cudaMemcpyDeviceToHost);
            return n;
        };
        const uint32_t nA = run(false, cntA, outA, lA, rA, msA);
        const uint32_t nB = run(true,  cntB, outB, lB, rB, msB);

        // multiset comparison
        std::vector<uint32_t> hA(nb), hB(nb);
        cudaMemcpy(hA.data(), cntA, (size_t)nb*4, cudaMemcpyDeviceToHost);
        cudaMemcpy(hB.data(), cntB, (size_t)nb*4, cudaMemcpyDeviceToHost);
        const bool counts_eq = memcmp(hA.data(), hB.data(), (size_t)nb*4) == 0;
        CK(cudaMemset(acc, 0, 64));
        fold_records <<<nb/256, 256>>>(cntA, cap, outA, acc);
        fold_backrefs<<<(nA+255)/256, 256>>>(lA, rA, nA, acc);
        fold_records <<<nb/256, 256>>>(cntB, cap, outB, acc + 4);
        fold_backrefs<<<(nB+255)/256, 256>>>(lB, rB, nB, acc + 4);
        unsigned long long h[8]; cudaMemcpy(h, acc, 64, cudaMemcpyDeviceToHost);
        uint32_t hd[4]; cudaMemcpy(hd, drops, 16, cudaMemcpyDeviceToHost);
        worstDrops |= hd[1] | hd[2] | hd[3];
        const bool ok = counts_eq && nA == nB && !memcmp(h, h + 4, 32);
        all_ok = all_ok && ok;
        printf("nonce %d  : pairs %u vs %u, per-bucket counts %s, folds %s, drops %u -> %s\n",
               ni, nA, nB, counts_eq ? "EQ" : "DIFF",
               memcmp(h, h + 4, 32) ? "DIFF" : "EQ", hd[1] | hd[2] | hd[3],
               ok ? "PASS" : "FAIL");
    }

    auto med = [](std::vector<double>& v) {
        std::sort(v.begin(), v.end()); return v[v.size()/2]; };
    const double mA = med(msA), mB = med(msB);
    printf("\nchain r1  : %.3f ms   (in-pipeline marginal is 5.29; standalone differs)\n", mA);
    printf("sorted r1 : %.3f ms\n", mB);
    printf("ratio     : %.3f   (P3 gate: <= 0.5 to proceed; upper bound per in-situ rule)\n", mB/mA);
    printf("verdict   : %s%s\n", all_ok ? "MULTISETS EQUAL" : "*** MISMATCH ***",
           worstDrops ? " (drops nonzero!)" : "");
    return all_ok ? 0 : 1;
}
