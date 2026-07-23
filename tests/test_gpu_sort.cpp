// Phase-D3 P1: isolate + verify the sort-based collision-finder building blocks
// (kernels/opencl/sort.cl) against CPU oracles, BEFORE wiring them into the
// pipeline. Radix correctness (stable order + prefix-sum/reorder) and the
// run-scan emission offsets are the crux, so we pin them here on small/medium N:
//   1. scan_block/scan_add          == std::exclusive_scan
//   2. radix_sort_pairs (3 passes)  == std::stable_sort by 24-bit key
//   3. collision_count+emit         == CPU group-by-key C(m,2) emission
#include "gpu/cl_runtime.h"
#include "generated/mxbm_kernels.h"
#include "check.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>
using namespace mxbm;
using namespace mxbm::gpu;

static const uint32_t SORT_WG = 256;
static const uint32_t SORT_RADIX_HOST = 256;

// splitmix64 — deterministic, no <random> dependence (matches the suite's style).
static uint64_t sm(uint64_t& s){ uint64_t z=(s+=0x9E3779B97F4A7C15ULL);
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31); }

static inline uint32_t pkey(uint64_t pair){ return (uint32_t)(pair & 0xFFFFFFFFu); }
static inline uint32_t pidx(uint64_t pair){ return (uint32_t)(pair >> 32); }

// Host-side generic exclusive scan over a device uint buffer of `n` elements,
// using scan_block + recursion on the per-block sums + scan_add. Mirrors the
// driver the pipeline will call.
static void scan_exclusive(Runtime& rt, cl_program prog, cl_mem buf, uint32_t n) {
    if (n == 0) return;
    uint32_t nblocks = (n + SORT_WG - 1) / SORT_WG;
    Mem blocksums = rt.alloc(CL_MEM_READ_WRITE, (size_t)nblocks * 4);
    Kernel kb = rt.kernel(prog, "scan_block");
    cl_mem bs = blocksums.get();
    rt.set_arg(kb.get(), 0, sizeof(cl_mem), &buf);   // in
    rt.set_arg(kb.get(), 1, sizeof(cl_mem), &buf);   // out (in place)
    rt.set_arg(kb.get(), 2, sizeof(cl_mem), &bs);
    rt.set_arg(kb.get(), 3, n);
    rt.run1d(kb.get(), (size_t)nblocks * SORT_WG, SORT_WG);
    if (nblocks > 1) {
        scan_exclusive(rt, prog, bs, nblocks);       // exclusive-scan the block sums
        Kernel ka = rt.kernel(prog, "scan_add");
        rt.set_arg(ka.get(), 0, sizeof(cl_mem), &buf);
        rt.set_arg(ka.get(), 1, sizeof(cl_mem), &bs);
        rt.set_arg(ka.get(), 2, n);
        rt.run1d(ka.get(), (size_t)nblocks * SORT_WG, SORT_WG);
    }
}

// Host driver: stable LSD radix sort of `n` (key,index) pairs in `buf` by the
// low 24 bits of the key (3 x 8-bit passes). Ping-pongs buf<->tmp; result ends
// back in `buf`. Returns nothing (in place from the caller's view).
static void radix_sort_pairs(Runtime& rt, cl_program prog, cl_mem buf, uint32_t n) {
    uint32_t ngroups = (n + SORT_WG - 1) / SORT_WG;
    Mem tmp     = rt.alloc(CL_MEM_READ_WRITE, (size_t)n * 8);
    Mem wghist  = rt.alloc(CL_MEM_READ_WRITE, (size_t)SORT_RADIX_HOST * ngroups * 4);
    Kernel kh = rt.kernel(prog, "radix_histogram");
    Kernel kr = rt.kernel(prog, "radix_reorder");
    cl_mem cur = buf, other = tmp.get(), wh = wghist.get();
    for (uint32_t pass = 0; pass < 3; ++pass) {
        uint32_t shift = pass * 8;
        rt.set_arg(kh.get(), 0, sizeof(cl_mem), &cur);
        rt.set_arg(kh.get(), 1, sizeof(cl_mem), &wh);
        rt.set_arg(kh.get(), 2, shift);
        rt.set_arg(kh.get(), 3, n);
        rt.set_arg(kh.get(), 4, ngroups);
        rt.run1d(kh.get(), (size_t)ngroups * SORT_WG, SORT_WG);

        scan_exclusive(rt, prog, wh, SORT_RADIX_HOST * ngroups);

        rt.set_arg(kr.get(), 0, sizeof(cl_mem), &cur);
        rt.set_arg(kr.get(), 1, sizeof(cl_mem), &other);
        rt.set_arg(kr.get(), 2, sizeof(cl_mem), &wh);
        rt.set_arg(kr.get(), 3, shift);
        rt.set_arg(kr.get(), 4, n);
        rt.set_arg(kr.get(), 5, ngroups);
        rt.run1d(kr.get(), (size_t)ngroups * SORT_WG, SORT_WG);

        cl_mem t = cur; cur = other; other = t;   // ping-pong
    }
    // After 3 passes result is in `cur`. If that's not `buf`, copy back.
    if (cur != buf) {
        std::vector<uint64_t> h((size_t)n);
        rt.read(cur, (size_t)n * 8, h.data());
        rt.write(buf, (size_t)n * 8, h.data());
    }
}

// --- test 1: generic exclusive scan --------------------------------------------
static void test_scan(Runtime& rt, cl_program prog) {
    section("scan_block/scan_add == exclusive_scan");
    for (uint32_t n : {1u, 255u, 256u, 257u, 1000u, 65536u, 100003u, (1u<<20)}) {
        std::vector<uint32_t> in(n);
        uint64_t s = 0x1234 + n;
        for (uint32_t i = 0; i < n; ++i) in[i] = (uint32_t)(sm(s) & 0xFFu);   // small so no overflow
        Mem buf = rt.alloc(CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, (size_t)n*4, in.data());
        scan_exclusive(rt, prog, buf.get(), n);
        std::vector<uint32_t> got(n);
        rt.read(buf.get(), (size_t)n*4, got.data());
        uint32_t acc = 0; int bad = 0;
        for (uint32_t i = 0; i < n; ++i) { if (got[i] != acc) { if (bad<3) std::printf("  scan n=%u i=%u got=%u want=%u\n",n,i,got[i],acc); ++bad; } acc += in[i]; }
        check(bad == 0, (std::string("exclusive scan n=") + std::to_string(n)).c_str());
    }
}

// --- test 2: stable radix sort by 24-bit key -----------------------------------
static void test_radix(Runtime& rt, cl_program prog) {
    section("radix_sort_pairs == std::stable_sort by 24-bit key");
    for (uint32_t n : {1u, 2u, 300u, 4096u, 100000u, (1u<<20)}) {
        std::vector<uint64_t> pairs(n);
        uint64_t s = 0xBEEF + n;
        // Mix of key distributions: mostly small key space (dense collisions,
        // stresses stability + long runs) plus some wide-key cases.
        uint32_t keymask = (n <= 4096u) ? 0xFFu : 0xFFFFFFu;   // small n -> heavy collisions
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t key = (uint32_t)(sm(s) & keymask);
            uint32_t idx = i;                                  // index = original position
            pairs[i] = ((uint64_t)idx << 32) | key;
        }
        std::vector<uint64_t> oracle = pairs;
        std::stable_sort(oracle.begin(), oracle.end(),
            [](uint64_t a, uint64_t b){ return pkey(a) < pkey(b); });

        Mem buf = rt.alloc(CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, (size_t)n*8, pairs.data());
        radix_sort_pairs(rt, prog, buf.get(), n);
        std::vector<uint64_t> got(n);
        rt.read(buf.get(), (size_t)n*8, got.data());

        int bad = 0;
        for (uint32_t i = 0; i < n; ++i) {
            // Stability: exact pair (key AND index) must match the stable oracle.
            if (got[i] != oracle[i]) { if (bad<3) std::printf("  radix n=%u i=%u got key=%x idx=%u want key=%x idx=%u\n",
                n,i,pkey(got[i]),pidx(got[i]),pkey(oracle[i]),pidx(oracle[i])); ++bad; }
        }
        check(bad == 0, (std::string("stable radix sort n=") + std::to_string(n)).c_str());
    }
}

// --- test 3: collision run-scan emission ---------------------------------------
static void test_collision(Runtime& rt, cl_program prog) {
    section("collision_count+emit == CPU group-by-key C(m,2)");
    for (uint32_t n : {2u, 300u, 4096u, 100000u}) {
        std::vector<uint64_t> pairs(n);
        uint64_t s = 0xC0DE + n;
        uint32_t keymask = 0xFFu;    // dense -> long runs, many collisions
        for (uint32_t i = 0; i < n; ++i)
            pairs[i] = ((uint64_t)i << 32) | (uint32_t)(sm(s) & keymask);
        // Sort first (device), then run the collision scan on the sorted array.
        Mem buf = rt.alloc(CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, (size_t)n*8, pairs.data());
        radix_sort_pairs(rt, prog, buf.get(), n);
        std::vector<uint64_t> sorted(n);
        rt.read(buf.get(), (size_t)n*8, sorted.data());

        // CPU oracle: over the sorted array, each equal-key run of length m emits
        // C(m,2) pairs (a<b by position) => (idx[a]<<32 | idx[b]).
        std::vector<uint64_t> oracle;
        for (uint32_t i = 0; i < n; ) {
            uint32_t j = i; while (j < n && pkey(sorted[j]) == pkey(sorted[i])) ++j;
            for (uint32_t a = i; a < j; ++a)
                for (uint32_t b = a+1; b < j; ++b)
                    oracle.push_back(((uint64_t)pidx(sorted[a]) << 32) | pidx(sorted[b]));
            i = j;
        }

        // Device: count -> exclusive scan -> emit.
        Mem counts = rt.alloc(CL_MEM_READ_WRITE, (size_t)n*4);
        Kernel kc = rt.kernel(prog, "collision_count");
        cl_mem sb = buf.get(), cb = counts.get();
        rt.set_arg(kc.get(),0,sizeof(cl_mem),&sb);
        rt.set_arg(kc.get(),1,sizeof(cl_mem),&cb);
        rt.set_arg(kc.get(),2,n);
        rt.run1d(kc.get(), ((n+SORT_WG-1)/SORT_WG)*SORT_WG, SORT_WG);
        // total emit = last-offset + last-count; grab counts before scan clobbers.
        std::vector<uint32_t> hc(n); rt.read(cb, (size_t)n*4, hc.data());
        uint32_t total = 0; for (uint32_t i=0;i<n;++i) total += hc[i];
        check(total == oracle.size(), (std::string("emit total n=")+std::to_string(n)).c_str());

        scan_exclusive(rt, prog, cb, n);
        Mem outp = rt.alloc(CL_MEM_READ_WRITE, (size_t)(total?total:1)*8);
        Kernel ke = rt.kernel(prog, "collision_emit");
        cl_mem ob = outp.get();
        rt.set_arg(ke.get(),0,sizeof(cl_mem),&sb);
        rt.set_arg(ke.get(),1,sizeof(cl_mem),&cb);
        rt.set_arg(ke.get(),2,sizeof(cl_mem),&ob);
        rt.set_arg(ke.get(),3,n);
        rt.run1d(ke.get(), ((n+SORT_WG-1)/SORT_WG)*SORT_WG, SORT_WG);
        std::vector<uint64_t> got(total);
        if (total) rt.read(ob, (size_t)total*8, got.data());

        // Emission order is deterministic (runs in sorted order, pairs a<b), so
        // compare directly against the oracle sequence.
        int bad = 0;
        for (size_t i = 0; i < oracle.size(); ++i)
            if (i >= got.size() || got[i] != oracle[i]) { if(bad<3) std::printf("  emit n=%u i=%zu got=%llx want=%llx\n",n,i,(unsigned long long)(i<got.size()?got[i]:0),(unsigned long long)oracle[i]); ++bad; }
        check(bad == 0, (std::string("collision emit n=")+std::to_string(n)).c_str());
    }
}

int main() {
    if (!Runtime::any_device_available()) { std::printf("SKIP: no OpenCL device\n"); return 0; }
    Runtime rt;
    std::printf("  device: %s\n", rt.device().name.c_str());
    // sort.cl's pipeline kernels reference bh3_combine (bh3.cl) + BH3_MAX_LEAVES
    // (round.cl); compile all three in the {bh3, round, sort} order the pipeline uses.
    Program prog = rt.build({std::string(kBh3ClSource), std::string(kRoundClSource),
                             std::string(kSortClSource)}, "");
    test_scan(rt, prog.get());
    test_radix(rt, prog.get());
    test_collision(rt, prog.get());

    // Full pipeline-scale case: 2^25 wide-key pairs (the real per-round size),
    // correctness vs std::stable_sort + a wall-clock number to steer P4 tuning.
    {
        section("radix sort at pipeline scale (N=2^25, wide 24-bit keys)");
        const uint32_t n = 1u << 25;
        std::vector<uint64_t> pairs(n);
        uint64_t s = 0x5127;
        for (uint32_t i = 0; i < n; ++i)
            pairs[i] = ((uint64_t)i << 32) | (uint32_t)(sm(s) & 0xFFFFFFu);
        std::vector<uint64_t> oracle = pairs;
        std::stable_sort(oracle.begin(), oracle.end(),
            [](uint64_t a, uint64_t b){ return pkey(a) < pkey(b); });
        Mem buf = rt.alloc(CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, (size_t)n*8, pairs.data());
        // Warm + timed run (program already built; alloc reused across the 2 runs).
        radix_sort_pairs(rt, prog.get(), buf.get(), n);
        rt.write(buf.get(), (size_t)n*8, pairs.data());
        auto t0 = std::chrono::steady_clock::now();
        radix_sort_pairs(rt, prog.get(), buf.get(), n);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::vector<uint64_t> got(n);
        rt.read(buf.get(), (size_t)n*8, got.data());
        int bad = 0;
        for (uint32_t i = 0; i < n; ++i) if (got[i] != oracle[i]) { if(bad<3) std::printf("  2^25 i=%u got key=%x idx=%u want key=%x idx=%u\n",i,pkey(got[i]),pidx(got[i]),pkey(oracle[i]),pidx(oracle[i])); ++bad; }
        std::printf("  3-pass radix sort of 2^25 pairs: %.2f ms  (P1 one-elem/thread, untuned)\n", ms);
        check(bad == 0, "stable radix sort n=2^25");
    }
    return summary("gpu_sort");
}
