// P2a: validate the LDS-local bucketed collision finder (kernels/opencl/lds.cl)
// against a CPU group-by-key oracle, then empirically sweep the design forks
// (bucket-bit / sub-mask split) at pipeline scale to pick the fastest config
// that keeps drops == 0. This isolates the collision-find MECHANISM (bucketing +
// atomic_xchg LDS chaining) before it is wired into the pipeline.
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

static const uint32_t WG = 256;

static uint64_t sm(uint64_t& s){ uint64_t z=(s+=0x9E3779B97F4A7C15ULL);
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31); }

static inline uint32_t ekey(uint64_t e){ return (uint32_t)(e & 0xFFFFFFu); }

// Host driver: scatter + collide. Returns the emitted (min<<32|max) index pairs
// (unsorted); fills drops[0..3] = {bucketOvf, groupOvf, outOvf, chainCap} and, if
// non-null, the collide-kernel ms.
static std::vector<uint64_t> run_collide(Runtime& rt, cl_program prog,
        const std::vector<uint64_t>& elems, uint32_t bucketBits, uint32_t submaskBits,
        uint32_t bucketCap, uint32_t outCap, uint32_t drops[4], double* ms) {
    uint32_t N = (uint32_t)elems.size();
    uint32_t numBuckets = 1u << bucketBits;
    Mem mElems = rt.alloc(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, (size_t)N*8, (void*)elems.data());
    Mem mCounts = rt.alloc(CL_MEM_READ_WRITE, (size_t)numBuckets*4);
    Mem mBuckets = rt.alloc(CL_MEM_READ_WRITE, (size_t)numBuckets*bucketCap*8);
    Mem mDrops = rt.alloc(CL_MEM_READ_WRITE, 4*4);
    Mem mOut = rt.alloc(CL_MEM_READ_WRITE, (size_t)outCap*8);
    Mem mOutCount = rt.alloc(CL_MEM_READ_WRITE, 4);
    rt.fill_u32(mCounts.get(), 0u, numBuckets);
    rt.fill_u32(mDrops.get(), 0u, 4);
    rt.fill_u32(mOutCount.get(), 0u, 1);

    { Kernel k = rt.kernel(prog, "p2_scatter");
      cl_mem e=mElems.get(), c=mCounts.get(), b=mBuckets.get(), d=mDrops.get();
      rt.set_arg(k.get(),0,N); rt.set_arg(k.get(),1,bucketBits); rt.set_arg(k.get(),2,bucketCap);
      rt.set_arg(k.get(),3,sizeof(cl_mem),&e); rt.set_arg(k.get(),4,sizeof(cl_mem),&c);
      rt.set_arg(k.get(),5,sizeof(cl_mem),&b); rt.set_arg(k.get(),6,sizeof(cl_mem),&d);
      rt.run1d(k.get(), N); }

    uint32_t groups = numBuckets << submaskBits;
    { Kernel k = rt.kernel(prog, "p2_collide");
      cl_mem c=mCounts.get(), b=mBuckets.get(), o=mOut.get(), oc=mOutCount.get(), d=mDrops.get();
      rt.set_arg(k.get(),0,bucketBits); rt.set_arg(k.get(),1,submaskBits); rt.set_arg(k.get(),2,bucketCap);
      rt.set_arg(k.get(),3,sizeof(cl_mem),&c); rt.set_arg(k.get(),4,sizeof(cl_mem),&b);
      rt.set_arg(k.get(),5,sizeof(cl_mem),&o); rt.set_arg(k.get(),6,sizeof(cl_mem),&oc);
      rt.set_arg(k.get(),7,outCap); rt.set_arg(k.get(),8,sizeof(cl_mem),&d);
      auto t0 = std::chrono::steady_clock::now();
      rt.run1d(k.get(), (size_t)groups*WG, WG);
      if (ms) *ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count(); }

    rt.read(mDrops.get(), 4*4, drops);
    uint32_t outN = 0; rt.read(mOutCount.get(), 4, &outN);
    uint32_t clamped = outN < outCap ? outN : outCap;
    std::vector<uint64_t> out(clamped);
    if (clamped) rt.read(mOut.get(), (size_t)clamped*8, out.data());
    return out;
}

// CPU oracle: group elems by full 24-bit key, emit C(m,2) (min<<32|max) index pairs.
static std::vector<uint64_t> oracle(const std::vector<uint64_t>& elems) {
    std::vector<uint64_t> s = elems;   // sort by (key,index) so groups are contiguous
    std::sort(s.begin(), s.end(), [](uint64_t a, uint64_t b){
        uint32_t ka=ekey(a), kb=ekey(b); if(ka!=kb) return ka<kb; return a<b; });
    std::vector<uint64_t> out;
    for (size_t i = 0; i < s.size(); ) {
        size_t j = i; while (j < s.size() && ekey(s[j]) == ekey(s[i])) ++j;
        for (size_t a = i; a < j; ++a) for (size_t b = a+1; b < j; ++b) {
            uint32_t ia=(uint32_t)(s[a]>>32), ib=(uint32_t)(s[b]>>32);
            uint32_t lo = ia<ib?ia:ib, hi = ia<ib?ib:ia;
            out.push_back(((uint64_t)lo<<32)|hi);
        }
        i = j;
    }
    return out;
}

static bool multiset_eq(std::vector<uint64_t> a, std::vector<uint64_t> b) {
    std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
    return a.size()==b.size() && std::equal(a.begin(),a.end(),b.begin());
}

// Generate N elements with `distinctBits` of key entropy spread across the full
// 24-bit range (bijective odd-multiply hash), index = position. Fewer distinct
// bits => denser collisions; spread => buckets populate across the top bits.
static std::vector<uint64_t> gen(uint32_t N, uint32_t distinctBits, uint64_t seed) {
    std::vector<uint64_t> v(N);
    uint64_t s = seed;
    uint32_t mask = (distinctBits >= 24) ? 0xFFFFFFu : ((1u<<distinctBits)-1u);
    for (uint32_t i = 0; i < N; ++i) {
        uint32_t base = (uint32_t)(sm(s) & mask);
        uint32_t key = (uint32_t)(((uint64_t)base * 2654435761u) & 0xFFFFFFu);  // spread over 24 bits
        v[i] = ((uint64_t)i << 32) | key;
    }
    return v;
}

static void test_correctness(Runtime& rt, cl_program prog) {
    section("p2 LDS collide == CPU group-by-key oracle (drops==0)");
    struct Cfg { uint32_t N, distinctBits, bucketBits, submaskBits; };
    Cfg cfgs[] = {
        {1u<<14, 24, 8, 0},    // sparse: ~0 collisions (no false positives)
        {1u<<18, 20, 10, 0},   // moderate collisions
        {1u<<20, 20, 11, 2},   // denser + sub-mask split
        {1u<<20, 16, 12, 3},   // heavy collisions, exercises chains
        {1u<<22, 22, 13, 3},   // near pipeline scale
    };
    for (auto c : cfgs) {
        auto elems = gen(c.N, c.distinctBits, 0xABCD ^ c.N ^ (c.distinctBits<<8));
        auto orc = oracle(elems);
        uint32_t meanBucket = c.N >> c.bucketBits;
        uint32_t bucketCap = meanBucket*4 + 256;             // generous, expect no ovf
        uint32_t outCap = (uint32_t)orc.size() + 4096u;
        uint32_t drops[4];
        auto got = run_collide(rt, prog, elems, c.bucketBits, c.submaskBits, bucketCap, outCap, drops, nullptr);
        char msg[160];
        std::snprintf(msg, sizeof msg, "N=2^%d distinctBits=%u b=%u s=%u: %zu pairs, drops={%u,%u,%u,%u}",
            (int)__builtin_ctz(c.N), c.distinctBits, c.bucketBits, c.submaskBits, orc.size(),
            drops[0],drops[1],drops[2],drops[3]);
        bool dropsZero = (drops[0]|drops[1]|drops[2]|drops[3])==0;
        check(dropsZero, (std::string("no drops -- ")+msg).c_str());
        check(multiset_eq(orc, got), (std::string("multiset == oracle -- ")+msg).c_str());
    }
}

int main() {
    if (!Runtime::any_device_available()) { std::printf("SKIP: no OpenCL device\n"); return 0; }
    Runtime rt;
    std::printf("  device: %s | local_mem=%llu KB\n", rt.device().name.c_str(),
        (unsigned long long)(rt.device().local_mem/1024));
    Program prog = rt.build({std::string(kLdsClSource)}, "");

    test_correctness(rt, prog.get());

    // Empirical fork sweep at pipeline scale: which (bucketBits, submaskBits)
    // split is fastest while drops==0? Realistic uniform 24-bit keys, N=2^25.
    {
        section("fork sweep: (bucketBits, submaskBits) at N=2^25, uniform keys");
        const uint32_t N = 1u << 25;
        auto elems = gen(N, 24, 0x5127);
        // total collisions ~ N for load-2; oracle count for outCap sizing.
        uint32_t outCap = N + (N>>2);
        struct Split { uint32_t b, s; };
        Split splits[] = {{12,4},{13,3},{13,4},{14,2},{14,3},{11,5}};
        double best = 1e9; uint32_t bb=0, bs=0;
        for (auto sp : splits) {
            uint32_t meanBucket = N >> sp.b;
            uint32_t bucketCap = meanBucket + (meanBucket>>2) + 512;   // 25% headroom
            uint32_t drops[4]; double ms = 0;
            auto got = run_collide(rt, prog.get(), elems, sp.b, sp.s, bucketCap, outCap, drops, &ms);
            bool ok = (drops[0]|drops[1]|drops[2]|drops[3])==0;
            std::printf("  b=%u s=%u groups=2^%u meanGrp=%u : collide=%.2f ms, pairs=%zu, drops={%u,%u,%u,%u} %s\n",
                sp.b, sp.s, sp.b+sp.s, meanBucket>>sp.s, ms, got.size(),
                drops[0],drops[1],drops[2],drops[3], ok?"OK":"DROPS!");
            if (ok && ms < best) { best = ms; bb=sp.b; bs=sp.s; }
        }
        std::printf("  --> fastest drop-free split: b=%u s=%u @ %.2f ms\n", bb, bs, best);
        check(best < 1e9, "at least one split is drop-free at 2^25");
    }
    return summary("lds_collide");
}
