// P2a: validate the LDS-local bucketed collision finder (kernels/opencl/lds.cl)
// against a CPU group-by-key oracle, then empirically sweep the design forks
// (bucket-bit / sub-mask split) at pipeline scale to pick the fastest config
// that keeps drops == 0. This isolates the collision-find MECHANISM (bucketing +
// atomic_xchg LDS chaining) before it is wired into the pipeline.
#include "gpu/cl_runtime.h"
#include "generated/mxbm_kernels.h"
#include "check.h"
#include "round_ref.h"
#include <algorithm>
#include <chrono>
#include <functional>
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

// ---- P2b: real 7-u64 element staging + bh3_combine in LDS ----
// CPU combine matching kernels/opencl/bh3.cl bh3_combine (XOR, >>24 restitch, mask to Lout).
static void cpu_combine(const uint64_t a[7], const uint64_t b[7], uint32_t Lout, uint64_t c[7]) {
    uint64_t x[7]; for (int i=0;i<7;++i) x[i]=a[i]^b[i];
    for (int i=0;i<7;++i) c[i] = (x[i]>>24) | (i<6 ? (x[i+1]<<40) : 0);
    for (int i=0;i<7;++i) { uint32_t base=64u*i;
        if (base>=Lout) c[i]=0; else if (Lout-base<64u) c[i] &= (((uint64_t)1<<(Lout-base))-1); }
}
// Order-independent digest of a child-work multiset: per-word XOR fold + count.
struct Digest { uint64_t x[7]; uint64_t n; bool operator==(const Digest&o)const{
    if(n!=o.n)return false; for(int i=0;i<7;++i) if(x[i]!=o.x[i]) return false; return true; } };

static std::vector<uint64_t> gen_elem(uint32_t N, uint32_t distinctBits, uint64_t seed) {
    std::vector<uint64_t> w((size_t)N*7);
    uint64_t s = seed;
    uint32_t mask = (distinctBits>=24)?0xFFFFFFu:((1u<<distinctBits)-1u);
    for (uint32_t g=0; g<N; ++g) {
        uint32_t base=(uint32_t)(sm(s)&mask);
        uint32_t key=(uint32_t)(((uint64_t)base*2654435761u)&0xFFFFFFu);
        w[(size_t)g*7] = ((sm(s) & 0xFFFFFFFFFFull) << 24) | key;   // low 24 = key
        for (int i=1;i<7;++i) w[(size_t)g*7+i]=sm(s);
    }
    return w;
}
static Digest oracle_combine(const std::vector<uint64_t>& w, uint32_t Lout) {
    uint32_t N=(uint32_t)(w.size()/7);
    std::vector<std::pair<uint32_t,uint32_t>> ki(N);  // (key, slot)
    for (uint32_t g=0;g<N;++g) ki[g]={(uint32_t)(w[(size_t)g*7]&0xFFFFFFu),g};
    std::sort(ki.begin(),ki.end());
    Digest d{}; d.n=0;
    for (uint32_t i=0;i<N;){ uint32_t j=i; while(j<N&&ki[j].first==ki[i].first)++j;
        for(uint32_t a=i;a<j;++a)for(uint32_t b=a+1;b<j;++b){
            uint64_t c[7]; cpu_combine(&w[(size_t)ki[a].second*7],&w[(size_t)ki[b].second*7],Lout,c);
            for(int k=0;k<7;++k) d.x[k]^=c[k]; ++d.n; }
        i=j; }
    return d;
}
static Digest run_combine(Runtime& rt, cl_program prog, const std::vector<uint64_t>& w,
        uint32_t bucketBits, uint32_t submaskBits, uint32_t bucketCap, uint32_t Lout,
        uint32_t outCap, uint32_t drops[4], double* ms, double* scatterMs = nullptr) {
    uint32_t N=(uint32_t)(w.size()/7), numBuckets=1u<<bucketBits;
    Mem mW=rt.alloc(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,(size_t)N*7*8,(void*)w.data());
    Mem mCounts=rt.alloc(CL_MEM_READ_WRITE,(size_t)numBuckets*4);
    Mem mBW=rt.alloc(CL_MEM_READ_WRITE,(size_t)numBuckets*bucketCap*7*8);
    Mem mBI=rt.alloc(CL_MEM_READ_WRITE,(size_t)numBuckets*bucketCap*4);
    Mem mDrops=rt.alloc(CL_MEM_READ_WRITE,4*4);
    Mem mOW=rt.alloc(CL_MEM_READ_WRITE,(size_t)outCap*7*8);
    Mem mOL=rt.alloc(CL_MEM_READ_WRITE,(size_t)outCap*4), mOR=rt.alloc(CL_MEM_READ_WRITE,(size_t)outCap*4);
    Mem mOC=rt.alloc(CL_MEM_READ_WRITE,4);
    rt.fill_u32(mCounts.get(),0u,numBuckets); rt.fill_u32(mDrops.get(),0u,4); rt.fill_u32(mOC.get(),0u,1);
    { Kernel k=rt.kernel(prog,"p2_scatter_elem"); cl_mem a=mW.get(),c=mCounts.get(),bw=mBW.get(),bi=mBI.get(),d=mDrops.get();
      rt.set_arg(k.get(),0,N);rt.set_arg(k.get(),1,bucketBits);rt.set_arg(k.get(),2,bucketCap);
      rt.set_arg(k.get(),3,sizeof(cl_mem),&a);rt.set_arg(k.get(),4,sizeof(cl_mem),&c);rt.set_arg(k.get(),5,sizeof(cl_mem),&bw);
      rt.set_arg(k.get(),6,sizeof(cl_mem),&bi);rt.set_arg(k.get(),7,sizeof(cl_mem),&d);
      auto ts=std::chrono::steady_clock::now(); rt.run1d(k.get(),N);
      if(scatterMs)*scatterMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-ts).count(); }
    uint32_t groups=numBuckets<<submaskBits;
    { Kernel k=rt.kernel(prog,"p2_collide_combine");
      cl_mem c=mCounts.get(),bw=mBW.get(),bi=mBI.get(),ow=mOW.get(),ol=mOL.get(),orr=mOR.get(),oc=mOC.get(),d=mDrops.get();
      rt.set_arg(k.get(),0,bucketBits);rt.set_arg(k.get(),1,submaskBits);rt.set_arg(k.get(),2,bucketCap);rt.set_arg(k.get(),3,Lout);
      rt.set_arg(k.get(),4,sizeof(cl_mem),&c);rt.set_arg(k.get(),5,sizeof(cl_mem),&bw);rt.set_arg(k.get(),6,sizeof(cl_mem),&bi);
      rt.set_arg(k.get(),7,sizeof(cl_mem),&ow);rt.set_arg(k.get(),8,sizeof(cl_mem),&ol);rt.set_arg(k.get(),9,sizeof(cl_mem),&orr);
      rt.set_arg(k.get(),10,sizeof(cl_mem),&oc);rt.set_arg(k.get(),11,outCap);rt.set_arg(k.get(),12,sizeof(cl_mem),&d);
      auto t0=std::chrono::steady_clock::now(); rt.run1d(k.get(),(size_t)groups*WG,WG);
      if(ms)*ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count(); }
    rt.read(mDrops.get(),4*4,drops);
    uint32_t outN=0; rt.read(mOC.get(),4,&outN); uint32_t clamped=outN<outCap?outN:outCap;
    std::vector<uint64_t> ow((size_t)clamped*7); if(clamped) rt.read(mOW.get(),(size_t)clamped*7*8,ow.data());
    Digest d{}; d.n=clamped; for(uint32_t i=0;i<clamped;++i) for(int k=0;k<7;++k) d.x[k]^=ow[(size_t)i*7+k];
    return d;
}
static void test_combine(Runtime& rt, cl_program prog) {
    section("p2 LDS collide+combine child-work digest == CPU combine oracle (drops==0)");
    const uint32_t Lout=424;  // round-1 Lout; the mechanism is Lout-agnostic
    struct Cfg{uint32_t N,distinctBits,bucketBits,submaskBits;};
    Cfg cfgs[]={ {1u<<18,20,11,3}, {1u<<20,20,12,3}, {1u<<20,16,12,4} };
    for(auto c:cfgs){
        auto w=gen_elem(c.N,c.distinctBits,0x1234^c.N);
        Digest orc=oracle_combine(w,Lout);
        uint32_t meanBucket=c.N>>c.bucketBits, bucketCap=meanBucket*4+256;
        uint32_t outCap=(uint32_t)orc.n+4096u, drops[4];
        Digest got=run_combine(rt,prog,w,c.bucketBits,c.submaskBits,bucketCap,Lout,outCap,drops,nullptr);
        char msg[160]; std::snprintf(msg,sizeof msg,"N=2^%d db=%u b=%u s=%u: %llu children, drops={%u,%u,%u,%u}",
            (int)__builtin_ctz(c.N),c.distinctBits,c.bucketBits,c.submaskBits,(unsigned long long)orc.n,drops[0],drops[1],drops[2],drops[3]);
        check((drops[0]|drops[1]|drops[2]|drops[3])==0,(std::string("no drops -- ")+msg).c_str());
        check(got==orc,(std::string("child digest == oracle -- ")+msg).c_str());
    }
    // WARM steady-state breakdown at 2^25 (preallocated buffers reused across
    // iterations, so no cold first-touch): best-of-6 for scatter and collide.
    const uint32_t N=1u<<25; auto w=gen_elem(N,24,0x99);
    Digest orc=oracle_combine(w,Lout);
    const uint32_t bb=14, sm=3, numBuckets=1u<<bb;
    uint32_t bucketCap=(N>>bb)+(N>>(bb+2))+512, outCap=N+(N>>2);
    Mem mW=rt.alloc(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,(size_t)N*7*8,(void*)w.data());
    Mem mC=rt.alloc(CL_MEM_READ_WRITE,(size_t)numBuckets*4);
    Mem mBW=rt.alloc(CL_MEM_READ_WRITE,(size_t)numBuckets*bucketCap*7*8);
    Mem mBI=rt.alloc(CL_MEM_READ_WRITE,(size_t)numBuckets*bucketCap*4);
    Mem mD=rt.alloc(CL_MEM_READ_WRITE,4*4);
    Mem mOW=rt.alloc(CL_MEM_READ_WRITE,(size_t)outCap*7*8);
    Mem mOL=rt.alloc(CL_MEM_READ_WRITE,(size_t)outCap*4), mOR=rt.alloc(CL_MEM_READ_WRITE,(size_t)outCap*4);
    Mem mOC=rt.alloc(CL_MEM_READ_WRITE,4);
    uint32_t drops[4]={0,0,0,0}; double bestSc=1e9, bestCo=1e9; uint32_t children=0;
    for (int it=0; it<6; ++it) {
        rt.fill_u32(mC.get(),0u,numBuckets); rt.fill_u32(mD.get(),0u,4); rt.fill_u32(mOC.get(),0u,1);
        { Kernel k=rt.kernel(prog,"p2_scatter_elem"); cl_mem a=mW.get(),c=mC.get(),bw=mBW.get(),bi=mBI.get(),d=mD.get();
          rt.set_arg(k.get(),0,N);rt.set_arg(k.get(),1,bb);rt.set_arg(k.get(),2,bucketCap);
          rt.set_arg(k.get(),3,sizeof(cl_mem),&a);rt.set_arg(k.get(),4,sizeof(cl_mem),&c);rt.set_arg(k.get(),5,sizeof(cl_mem),&bw);
          rt.set_arg(k.get(),6,sizeof(cl_mem),&bi);rt.set_arg(k.get(),7,sizeof(cl_mem),&d);
          auto t0=std::chrono::steady_clock::now(); rt.run1d(k.get(),N);
          double m=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count(); if(it&&m<bestSc)bestSc=m; }
        { Kernel k=rt.kernel(prog,"p2_collide_combine");
          cl_mem c=mC.get(),bw=mBW.get(),bi=mBI.get(),ow=mOW.get(),ol=mOL.get(),orr=mOR.get(),oc=mOC.get(),d=mD.get();
          rt.set_arg(k.get(),0,bb);rt.set_arg(k.get(),1,sm);rt.set_arg(k.get(),2,bucketCap);rt.set_arg(k.get(),3,Lout);
          rt.set_arg(k.get(),4,sizeof(cl_mem),&c);rt.set_arg(k.get(),5,sizeof(cl_mem),&bw);rt.set_arg(k.get(),6,sizeof(cl_mem),&bi);
          rt.set_arg(k.get(),7,sizeof(cl_mem),&ow);rt.set_arg(k.get(),8,sizeof(cl_mem),&ol);rt.set_arg(k.get(),9,sizeof(cl_mem),&orr);
          rt.set_arg(k.get(),10,sizeof(cl_mem),&oc);rt.set_arg(k.get(),11,outCap);rt.set_arg(k.get(),12,sizeof(cl_mem),&d);
          auto t0=std::chrono::steady_clock::now(); rt.run1d(k.get(),(size_t)(numBuckets<<sm)*WG,WG);
          double m=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count(); if(it&&m<bestCo)bestCo=m; }
        rt.read(mOC.get(),4,&children);
    }
    rt.read(mD.get(),4*4,drops);
    Digest got{}; got.n=children<outCap?children:outCap;
    { std::vector<uint64_t> ow((size_t)got.n*7); if(got.n) rt.read(mOW.get(),(size_t)got.n*7*8,ow.data());
      for(uint64_t i=0;i<got.n;++i) for(int k=0;k<7;++k) got.x[k]^=ow[i*7+k]; }
    std::printf("  [P2 WARM @ 2^25, b=%u s=%u, real 56B elem] scatter(=emit proxy)=%.2f ms  stage+find+combine+flatemit=%.2f ms"
                "  %llu children  drops={%u,%u,%u,%u}\n",
        bb, sm, bestSc, bestCo, (unsigned long long)got.n, drops[0],drops[1],drops[2],drops[3]);
    check((drops[0]|drops[1]|drops[2]|drops[3])==0,"2^25 warm drop-free");
    check(got==orc,"2^25 warm child digest == oracle");
}

// Characterize the SCATTER cost: is 56B-element bucketing slow due to atomic
// contention (helped by more buckets) or scattered write size (56B vs 8B slot)?
static void test_scatter_cost(Runtime& rt, cl_program prog) {
    section("scatter cost sweep (contention vs write-size) at N=2^25");
    const uint32_t N=1u<<25;
    auto wElem = gen_elem(N,24,0x7);       // 56B elements for p2_scatter_elem
    auto pairs = gen(N,24,0x7);            // 8B (key,index) for p2_scatter
    for (uint32_t bb : {14u,16u,18u}) {
        uint32_t numBuckets=1u<<bb, meanBucket=N>>bb;
        // Cap so the 56B bucket buffer stays under the 4.18 GB max-alloc limit.
        uint32_t capLimit=(uint32_t)(3.6e9/((double)numBuckets*56.0));
        uint32_t bucketCap=meanBucket*2+64; if (bucketCap>capLimit) bucketCap=capLimit;
        // --- 8B slot/key scatter (lightweight) ---
        double t8=1e9;
        { Mem mE=rt.alloc(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,(size_t)N*8,(void*)pairs.data());
          Mem mC=rt.alloc(CL_MEM_READ_WRITE,(size_t)numBuckets*4), mB=rt.alloc(CL_MEM_READ_WRITE,(size_t)numBuckets*bucketCap*8);
          Mem mD=rt.alloc(CL_MEM_READ_WRITE,4*4);
          for(int it=0;it<3;++it){ rt.fill_u32(mC.get(),0u,numBuckets); rt.fill_u32(mD.get(),0u,4);
            Kernel k=rt.kernel(prog,"p2_scatter"); cl_mem e=mE.get(),c=mC.get(),b=mB.get(),d=mD.get();
            rt.set_arg(k.get(),0,N);rt.set_arg(k.get(),1,bb);rt.set_arg(k.get(),2,bucketCap);
            rt.set_arg(k.get(),3,sizeof(cl_mem),&e);rt.set_arg(k.get(),4,sizeof(cl_mem),&c);
            rt.set_arg(k.get(),5,sizeof(cl_mem),&b);rt.set_arg(k.get(),6,sizeof(cl_mem),&d);
            auto t0=std::chrono::steady_clock::now(); rt.run1d(k.get(),N);
            double m=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count(); if(m<t8)t8=m; } }
        // --- 56B full-element scatter ---
        double t56=1e9;
        { Mem mW=rt.alloc(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,(size_t)N*7*8,(void*)wElem.data());
          Mem mC=rt.alloc(CL_MEM_READ_WRITE,(size_t)numBuckets*4), mBW=rt.alloc(CL_MEM_READ_WRITE,(size_t)numBuckets*bucketCap*7*8);
          Mem mBI=rt.alloc(CL_MEM_READ_WRITE,(size_t)numBuckets*bucketCap*4), mD=rt.alloc(CL_MEM_READ_WRITE,4*4);
          for(int it=0;it<3;++it){ rt.fill_u32(mC.get(),0u,numBuckets); rt.fill_u32(mD.get(),0u,4);
            Kernel k=rt.kernel(prog,"p2_scatter_elem"); cl_mem a=mW.get(),c=mC.get(),bw=mBW.get(),bi=mBI.get(),d=mD.get();
            rt.set_arg(k.get(),0,N);rt.set_arg(k.get(),1,bb);rt.set_arg(k.get(),2,bucketCap);
            rt.set_arg(k.get(),3,sizeof(cl_mem),&a);rt.set_arg(k.get(),4,sizeof(cl_mem),&c);rt.set_arg(k.get(),5,sizeof(cl_mem),&bw);
            rt.set_arg(k.get(),6,sizeof(cl_mem),&bi);rt.set_arg(k.get(),7,sizeof(cl_mem),&d);
            auto t0=std::chrono::steady_clock::now(); rt.run1d(k.get(),N);
            double m=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count(); if(m<t56)t56=m; } }
        std::printf("  bucket_bits=%u (%u buckets, ~%u/bucket): 8B-slot scatter=%.2f ms | 56B-elem scatter=%.2f ms\n",
            bb, numBuckets, meanBucket, t8, t56);
    }
    check(true,"scatter cost characterized");
}

// Compaction ARCHITECTURE decision: does compaction win with (H1) compile-time
// fixed-width AoS, or does it need (H2) SoA word-planes -- and does SoA even beat
// the AoS-7 baseline on scatter? Races all three at N=2^25, warm best-of-N.
// Decides the whole rewrite: H1 wins -> a few fixed-width kernels; H2 wins -> full
// SoA; neither beats AoS-7 -> compaction is dead, pivot.
static void test_compaction_arch(Runtime& rt, cl_program prog) {
    section("compaction architecture: AoS-fixed (H1) vs SoA-planes (H2) vs AoS-7 baseline @ 2^25");
    const uint32_t N=1u<<25, bb=14, numBuckets=1u<<bb;
    const uint32_t bucketCap=(N>>bb)+(N>>(bb+2))+512;   // ~3072, mean ~2048 -> drop-free
    const uint32_t sched[5]={7,7,6,5,1};                // emit width per round [7,7,6,5,1]
    const int ITERS=6;
    auto best=[&](std::function<double()> once){ double b=1e9; for(int i=0;i<ITERS;++i){ double m=once(); if(i&&m<b)b=m; } return b; };

    auto w = gen_elem(N,24,0x3);   // AoS 7-wide source

    // --- H1 + baseline: AoS fixed-width scatter (aos_scatterW) ---
    double aos[8]; for(int i=0;i<8;++i) aos[i]=0;
    { Mem mW=rt.alloc(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,(size_t)N*7*8,(void*)w.data());
      Mem mC=rt.alloc(CL_MEM_READ_WRITE,(size_t)numBuckets*4);
      Mem mBW=rt.alloc(CL_MEM_READ_WRITE,(size_t)numBuckets*bucketCap*7*8);  // 7-word max
      Mem mD=rt.alloc(CL_MEM_READ_WRITE,4*4);
      uint32_t maxDrop=0;
      auto runW=[&](const char* name){ return best([&]{ rt.fill_u32(mC.get(),0u,numBuckets); rt.fill_u32(mD.get(),0u,4);
          Kernel k=rt.kernel(prog,name); cl_mem a=mW.get(),c=mC.get(),bw=mBW.get(),d=mD.get();
          rt.set_arg(k.get(),0,N);rt.set_arg(k.get(),1,bb);rt.set_arg(k.get(),2,bucketCap);
          rt.set_arg(k.get(),3,sizeof(cl_mem),&a);rt.set_arg(k.get(),4,sizeof(cl_mem),&c);
          rt.set_arg(k.get(),5,sizeof(cl_mem),&bw);rt.set_arg(k.get(),6,sizeof(cl_mem),&d);
          auto t0=std::chrono::steady_clock::now(); rt.run1d(k.get(),N);
          double m=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
          uint32_t dr[4]; rt.read(mD.get(),16,dr); if(dr[0]>maxDrop)maxDrop=dr[0]; return m; }); };
      aos[1]=runW("aos_scatter1"); aos[5]=runW("aos_scatter5"); aos[6]=runW("aos_scatter6"); aos[7]=runW("aos_scatter7");
      std::printf("  AoS-fixed emit: w1=%.2f w5=%.2f w6=%.2f w7=%.2f ms  (maxBucketDrop=%u)\n",aos[1],aos[5],aos[6],aos[7],maxDrop);
      check(maxDrop==0,"AoS-fixed scatter drop-free");
    }

    // --- H2: SoA plane scatter (soa_scatterW). Transpose source into 7 planes. ---
    double soa[8]; for(int i=0;i<8;++i) soa[i]=0;
    { std::vector<Mem> sp, bp;   // 7 source planes (N u64), 7 bucket planes
      for(int p=0;p<7;++p){ std::vector<uint64_t> plane(N); for(uint32_t g=0;g<N;++g) plane[g]=w[(size_t)g*7+p];
        sp.push_back(rt.alloc(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,(size_t)N*8,plane.data())); }
      for(int p=0;p<7;++p) bp.push_back(rt.alloc(CL_MEM_READ_WRITE,(size_t)numBuckets*bucketCap*8));
      Mem mC=rt.alloc(CL_MEM_READ_WRITE,(size_t)numBuckets*4), mD=rt.alloc(CL_MEM_READ_WRITE,4*4);
      uint32_t maxDrop=0;
      auto runW=[&](const char* name){ return best([&]{ rt.fill_u32(mC.get(),0u,numBuckets); rt.fill_u32(mD.get(),0u,4);
          Kernel k=rt.kernel(prog,name); cl_mem c=mC.get(),d=mD.get();
          rt.set_arg(k.get(),0,N);rt.set_arg(k.get(),1,bb);rt.set_arg(k.get(),2,bucketCap);
          for(int p=0;p<7;++p){ cl_mem s=sp[p].get(); rt.set_arg(k.get(),3+p,sizeof(cl_mem),&s); }
          rt.set_arg(k.get(),10,sizeof(cl_mem),&c);
          for(int p=0;p<7;++p){ cl_mem b=bp[p].get(); rt.set_arg(k.get(),11+p,sizeof(cl_mem),&b); }
          rt.set_arg(k.get(),18,sizeof(cl_mem),&d);
          auto t0=std::chrono::steady_clock::now(); rt.run1d(k.get(),N);
          double m=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
          uint32_t dr[4]; rt.read(mD.get(),16,dr); if(dr[0]>maxDrop)maxDrop=dr[0]; return m; }); };
      soa[1]=runW("soa_scatter1"); soa[5]=runW("soa_scatter5"); soa[6]=runW("soa_scatter6"); soa[7]=runW("soa_scatter7");
      std::printf("  SoA-plane emit: w1=%.2f w5=%.2f w6=%.2f w7=%.2f ms  (maxBucketDrop=%u)\n",soa[1],soa[5],soa[6],soa[7],maxDrop);
      check(maxDrop==0,"SoA-plane scatter drop-free");
    }

    // Per-solve emit totals (5 rounds, schedule [7,7,6,5,1]).
    double baseline=5*aos[7];
    double h1=aos[sched[0]]+aos[sched[1]]+aos[sched[2]]+aos[sched[3]]+aos[sched[4]];
    double h2=soa[sched[0]]+soa[sched[1]]+soa[sched[2]]+soa[sched[3]]+soa[sched[4]];
    std::printf("  --> per-solve emit: AoS-7 baseline=%.1f ms | H1 AoS-compacted=%.1f ms (%+.0f%%) | H2 SoA-compacted=%.1f ms (%+.0f%%)\n",
        baseline, h1, 100.0*(h1-baseline)/baseline, h2, 100.0*(h2-baseline)/baseline);
    std::printf("  --> SoA-7 vs AoS-7 (pure layout, no compaction): %.2f vs %.2f ms (%+.0f%%)\n",
        soa[7], aos[7], 100.0*(soa[7]-aos[7])/aos[7]);
    check(true,"compaction architecture characterized");
}

// ROW-BUCKET crux: does a COALESCED (bucket-local) parent gather beat the sort
// path's SCATTERED-by-slot gather, and by how much? M combines, each reads 2
// parents by index + bh3_combine. Scattered index list = random slots (today's
// match); coalesced = adjacent slots (bucket-local pairs). The delta is the
// ceiling of the whole rewrite. Swept over element width [7,6,5] (compaction).
static void test_gather_locality(Runtime& rt, cl_program prog) {
    section("row-bucket crux: scattered vs coalesced parent gather @ M=2^25");
    const uint32_t N=1u<<25, M=1u<<25;
    const int ITERS=6;
    auto best=[&](std::function<double()> once){ double b=1e9; for(int i=0;i<ITERS;++i){ double m=once(); if(i&&m<b)b=m; } return b; };

    // Index lists: scattered (random) and coalesced (adjacent pairs 2g,2g+1).
    std::vector<uint32_t> sL(M), sR(M), cL(M), cR(M);
    uint64_t s=0xC0FFEE;
    for (uint32_t g=0; g<M; ++g) {
        sL[g]=(uint32_t)(sm(s)%N); sR[g]=(uint32_t)(sm(s)%N);
        cL[g]=(2u*g)%N; cR[g]=(2u*g+1u)%N;
    }
    Mem mIsL=rt.alloc(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,(size_t)M*4,sL.data());
    Mem mIsR=rt.alloc(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,(size_t)M*4,sR.data());
    Mem mIcL=rt.alloc(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,(size_t)M*4,cL.data());
    Mem mIcR=rt.alloc(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,(size_t)M*4,cR.data());
    auto w=gen_elem(N,24,0x9);
    Mem mP=rt.alloc(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,(size_t)N*7*8,w.data());
    Mem mOut=rt.alloc(CL_MEM_READ_WRITE,(size_t)M*7*8);

    const uint32_t Lout[3]={424,376,288};   // r1/r3/r4-ish, widths 7/6/5
    const uint32_t ews[3]={7,6,5};
    for (int wi=0; wi<3; ++wi) {
        uint32_t ew=ews[wi], lo=Lout[wi];
        auto run=[&](cl_mem iL, cl_mem iR){ return best([&]{
            Kernel k=rt.kernel(prog,"gather_combine_bench");
            cl_mem p=mP.get(), o=mOut.get();
            rt.set_arg(k.get(),0,M); rt.set_arg(k.get(),1,lo); rt.set_arg(k.get(),2,ew);
            rt.set_arg(k.get(),3,sizeof(cl_mem),&p);
            rt.set_arg(k.get(),4,sizeof(cl_mem),&iL); rt.set_arg(k.get(),5,sizeof(cl_mem),&iR);
            rt.set_arg(k.get(),6,sizeof(cl_mem),&o);
            auto t0=std::chrono::steady_clock::now(); rt.run1d(k.get(),M);
            return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count(); }); };
        double scat=run(mIsL.get(),mIsR.get());
        double coal=run(mIcL.get(),mIcR.get());
        std::printf("  ew=%u (%uB): scattered gather=%.2f ms | coalesced=%.2f ms  (%.0f%% faster, %.1fx)\n",
            ew, ew*8, scat, coal, 100.0*(scat-coal)/scat, scat/coal);
    }
    check(true,"gather locality characterized");
}

// ROW-BUCKET STEP A: fat-vs-thin bucket layout decision. A fused round kernel must
// materialize each child's leaf prefix. FAT carries leaves inside the bucket
// element (coalesced leaf read, wider emit); THIN keeps leaves in a gi-indexed
// global array (scattered read = the 48 ms/solve match leaf gather, narrow emit).
// Net per level = (scattered_leafread - coalesced_leafread) [fat's READ win]
//               - (fat_emit - thin_emit)                     [fat's WRITE penalty].
// Positive net => fat wins. Levels 2..5 carry leaves of width sleaves_for(L).
static void test_leaf_locality(Runtime& rt, cl_program prog) {
    section("row-bucket STEP A: fat vs thin bucket (leaf read locality vs emit width)");
    const uint32_t N=1u<<25, M=1u<<25;
    const int ITERS=6;
    auto best=[&](std::function<double()> once){ double b=1e9; for(int i=0;i<ITERS;++i){ double m=once(); if(i&&m<b)b=m; } return b; };

    // Index lists shared across levels: scattered (random) vs coalesced (adjacent).
    std::vector<uint32_t> sL(M), sR(M), cL(M), cR(M);
    uint64_t s=0xBADF00D;
    for (uint32_t g=0; g<M; ++g){ sL[g]=(uint32_t)(sm(s)%N); sR[g]=(uint32_t)(sm(s)%N); cL[g]=(2u*g)%N; cR[g]=(2u*g+1u)%N; }
    Mem mIsL=rt.alloc(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,(size_t)M*4,sL.data());
    Mem mIsR=rt.alloc(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,(size_t)M*4,sR.data());
    Mem mIcL=rt.alloc(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,(size_t)M*4,cL.data());
    Mem mIcR=rt.alloc(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,(size_t)M*4,cR.data());
    // Leaf source (uint), max stride 9; and a wide bucket sink for emit (up to 12 u64).
    std::vector<uint32_t> lf((size_t)N*9); for(size_t i=0;i<lf.size();++i) lf[i]=(uint32_t)i;
    Mem mLf=rt.alloc(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,(size_t)N*9*4,lf.data());
    Mem mLOut=rt.alloc(CL_MEM_READ_WRITE,(size_t)M*9*4);
    // Sink kept under the ~4 GB single-alloc limit: 11-word max fat elem, lean cap
    // (uniform keys -> max bucket occupancy ~avg+5sigma, well under this).
    const uint32_t bb=14, numBuckets=1u<<bb, bucketCap=(N>>bb)+(N>>(bb+4))+256;
    Mem mC=rt.alloc(CL_MEM_READ_WRITE,(size_t)numBuckets*4);
    Mem mBW=rt.alloc(CL_MEM_READ_WRITE,(size_t)numBuckets*bucketCap*11*8);   // 11-word max fat elem
    Mem mD=rt.alloc(CL_MEM_READ_WRITE,4*4);

    auto readSide=[&](uint32_t sIn, uint32_t sOut, cl_mem iL, cl_mem iR){ return best([&]{
        Kernel k=rt.kernel(prog,"gather_leaves_bench"); cl_mem lfp=mLf.get(), o=mLOut.get();
        rt.set_arg(k.get(),0,M); rt.set_arg(k.get(),1,sIn); rt.set_arg(k.get(),2,sOut);
        rt.set_arg(k.get(),3,sizeof(cl_mem),&lfp);
        rt.set_arg(k.get(),4,sizeof(cl_mem),&iL); rt.set_arg(k.get(),5,sizeof(cl_mem),&iR);
        rt.set_arg(k.get(),6,sizeof(cl_mem),&o);
        auto t0=std::chrono::steady_clock::now(); rt.run1d(k.get(),M);
        return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count(); }); };
    auto emit=[&](uint32_t ew){ return best([&]{
        rt.fill_u32(mC.get(),0u,numBuckets); rt.fill_u32(mD.get(),0u,4);
        Kernel k=rt.kernel(prog,"bucket_emit_bench"); cl_mem c=mC.get(),bw=mBW.get(),d=mD.get();
        rt.set_arg(k.get(),0,N); rt.set_arg(k.get(),1,bb); rt.set_arg(k.get(),2,bucketCap); rt.set_arg(k.get(),3,ew);
        rt.set_arg(k.get(),4,sizeof(cl_mem),&c); rt.set_arg(k.get(),5,sizeof(cl_mem),&bw); rt.set_arg(k.get(),6,sizeof(cl_mem),&d);
        auto t0=std::chrono::steady_clock::now(); rt.run1d(k.get(),N);
        return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count(); }); };

    // Per level L (=2..5, output of round L-1). sleaves=[_,2,4,8,9], work width
    // = outwords_for(L-1) = [7,7,6,5]. control (gi+lead) = 1 u64.
    const uint32_t levels[4]={2,3,4,5};
    const uint32_t sleaves[6]={0,1,2,4,8,9};   // sleaves_for(1..5)
    const uint32_t owords[6] ={0,7,7,6,5,1};   // outwords_for(1..5)
    double totRead=0, totWrite=0;
    for (int li=0; li<4; ++li) {
        uint32_t L=levels[li];
        uint32_t sIn=sleaves[L];
        uint32_t sOut = (L<5) ? sleaves[L+1] : sleaves[L];   // L=5: self-read (no child build)
        double scat=readSide(sIn,sOut,mIsL.get(),mIsR.get());
        double coal=readSide(sIn,sOut,mIcL.get(),mIcR.get());
        uint32_t control=1, leafU64=(sIn+1)/2;               // ceil(sIn/2)
        uint32_t thin_ew=owords[L-1]+control, fat_ew=thin_ew+leafU64;
        double eThin=emit(thin_ew), eFat=emit(fat_ew);
        double readWin=scat-coal, writePen=eFat-eThin, net=readWin-writePen;
        std::printf("  L%u (leaves=%u u32): read scat=%.2f coal=%.2f (win %.2f) | emit thin=%.2f(ew%u) fat=%.2f(ew%u) (pen %.2f) | NET %+.2f ms\n",
            L, sIn, scat, coal, readWin, eThin, thin_ew, eFat, fat_ew, writePen, net);
        totRead+=readWin; totWrite+=writePen;
    }
    std::printf("  --> per-solve: fat READ win=%.1f ms | fat WRITE penalty=%.1f ms | NET fat %+.1f ms  => %s bucket\n",
        totRead, totWrite, totRead-totWrite, (totRead-totWrite>0)?"FAT":"THIN");
    check(true,"fat-vs-thin bucket layout characterized");
}

// ROW-BUCKET STEP B: validate round_fused_lds (fused match + child-mix + child-
// scatter) against ref::cpu_round. Synthesizes N round-r elements (random work +
// a distinct 2^(r-1)-leaf tree), mixes them (apply_mix Lmix(r)) to set collision
// keys, host-scatters into FAT input buckets, runs the fused kernel, and compares
// the emitted round-(r+1) children -- (mixed work[7], leaf prefix) -- as a multiset
// against cpu_round(in,r) with each child re-mixed for round r+1. sIn=2^(r-1) is
// the FULL subtree for r<=4, so no leaf information is lost.
struct FChild { uint64_t w[7]; uint32_t lv[9]; uint32_t nlv; };
static bool fchild_lt(const FChild&a,const FChild&b){
    for(int i=0;i<7;++i) if(a.w[i]!=b.w[i]) return a.w[i]<b.w[i];
    if(a.nlv!=b.nlv) return a.nlv<b.nlv;
    for(uint32_t i=0;i<a.nlv;++i) if(a.lv[i]!=b.lv[i]) return a.lv[i]<b.lv[i];
    return false;
}
static bool fchild_eq(const FChild&a,const FChild&b){
    for(int i=0;i<7;++i) if(a.w[i]!=b.w[i]) return false;
    if(a.nlv!=b.nlv) return false;
    for(uint32_t i=0;i<a.nlv;++i) if(a.lv[i]!=b.lv[i]) return false;
    return true;
}
static void test_fused_round(Runtime& rt, cl_program prog, int r, uint32_t N,
                             uint32_t bucketBits, uint32_t submaskBits,
                             uint32_t inCap, uint32_t outCap, bool exact) {
    char title[128]; std::snprintf(title,sizeof title,
        "STEP B: round_fused_lds r=%d->%d, N=2^%d (%s)", r, r+1, (int)__builtin_ctz(N),
        exact?"exact multiset vs cpu_round":"scale: drops==0 + count");
    section(title);
    const uint32_t sIn = 1u << (r-1);          // = sleaves_for(r) for r<=4 (full subtree)
    const uint32_t sOut = (r < 4) ? (1u<<r) : 9u;   // sleaves_for(r+1)
    const uint32_t Lout = ref::Lout(r), LmixNext = ref::Lmix(r+1), padNext = ref::padNum(r+1);
    const uint32_t numBuckets = 1u << bucketBits;

    // Synthesize N round-r inputs: random work, distinct random leaf tree.
    uint64_t s = 0xF00D5EED ^ ((uint64_t)r<<40) ^ N;
    std::vector<ref::RElem> in(N);
    for (uint32_t i=0;i<N;++i){
        for(int w=0;w<7;++w) in[i].w[w]=sm(s);
        in[i].tree.resize(sIn);
        for(uint32_t j=0;j<sIn;++j) in[i].tree[j]=(uint32_t)(sm(s)&0x1FFFFFFu);  // 25-bit leaf ids
    }
    // Mix (sets collision key), then host-scatter into FAT input buckets.
    std::vector<uint32_t> counts(numBuckets,0);
    std::vector<uint64_t> bwork((size_t)numBuckets*inCap*7,0);
    std::vector<uint32_t> bgi((size_t)numBuckets*inCap,0), blead((size_t)numBuckets*inCap,0);
    std::vector<uint32_t> bleaves((size_t)numBuckets*inCap*sIn,0);
    uint32_t hostDrop=0;
    for (uint32_t i=0;i<N;++i){
        uint64_t mw0 = ref::cpu_mix_w0(in[i], r);
        uint32_t key = (uint32_t)(mw0 & 0xFFFFFFu);
        uint32_t b = key >> (24u - bucketBits);
        uint32_t pos = counts[b]++;
        if (pos>=inCap){ hostDrop++; continue; }
        size_t d=(size_t)b*inCap+pos;
        bwork[d*7]=mw0; for(int w=1;w<7;++w) bwork[d*7+w]=in[i].w[w];
        bgi[d]=i; blead[d]=in[i].tree[0];
        for(uint32_t j=0;j<sIn;++j) bleaves[d*sIn+j]=in[i].tree[j];
    }
    check(hostDrop==0,"host input scatter drop-free (raise inCap if this fails)");

    // Upload + run round_fused_lds.
    Mem mInCounts=rt.alloc(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,(size_t)numBuckets*4,counts.data());
    Mem mInWork=rt.alloc(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,bwork.size()*8,bwork.data());
    Mem mInGi=rt.alloc(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,bgi.size()*4,bgi.data());
    Mem mInLead=rt.alloc(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,blead.size()*4,blead.data());
    Mem mInLeaves=rt.alloc(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,bleaves.size()*4,bleaves.data());
    Mem mOutCounts=rt.alloc(CL_MEM_READ_WRITE,(size_t)numBuckets*4);
    Mem mOutWork=rt.alloc(CL_MEM_READ_WRITE,(size_t)numBuckets*outCap*7*8);
    Mem mOutGi=rt.alloc(CL_MEM_READ_WRITE,(size_t)numBuckets*outCap*4);
    Mem mOutLead=rt.alloc(CL_MEM_READ_WRITE,(size_t)numBuckets*outCap*4);
    Mem mOutLeaves=rt.alloc(CL_MEM_READ_WRITE,(size_t)numBuckets*outCap*sOut*4);
    Mem mAllL=rt.alloc(CL_MEM_READ_WRITE,(size_t)numBuckets*outCap*4);
    Mem mAllR=rt.alloc(CL_MEM_READ_WRITE,(size_t)numBuckets*outCap*4);
    Mem mGiCtr=rt.alloc(CL_MEM_READ_WRITE,4);
    Mem mDrops=rt.alloc(CL_MEM_READ_WRITE,4*4);
    rt.fill_u32(mOutCounts.get(),0u,numBuckets); rt.fill_u32(mGiCtr.get(),0u,1); rt.fill_u32(mDrops.get(),0u,4);

    { Kernel k=rt.kernel(prog,"round_fused_lds");
      cl_mem ic=mInCounts.get(),iw=mInWork.get(),ig=mInGi.get(),il=mInLead.get(),ilv=mInLeaves.get();
      cl_mem oc=mOutCounts.get(),ow=mOutWork.get(),og=mOutGi.get(),ol=mOutLead.get(),olv=mOutLeaves.get();
      cl_mem aL=mAllL.get(),aR=mAllR.get(),gc=mGiCtr.get(),dr=mDrops.get();
      int a=0;
      rt.set_arg(k.get(),a++,bucketBits); rt.set_arg(k.get(),a++,submaskBits);
      rt.set_arg(k.get(),a++,inCap); rt.set_arg(k.get(),a++,outCap);
      rt.set_arg(k.get(),a++,Lout); rt.set_arg(k.get(),a++,LmixNext); rt.set_arg(k.get(),a++,padNext);
      rt.set_arg(k.get(),a++,sIn); rt.set_arg(k.get(),a++,sOut); rt.set_arg(k.get(),a++,0u/*out_off*/);
      rt.set_arg(k.get(),a++,sOut/*sBuild: RAW variant builds exactly what it stores*/);
      rt.set_arg(k.get(),a++,sizeof(cl_mem),&ic); rt.set_arg(k.get(),a++,sizeof(cl_mem),&iw);
      rt.set_arg(k.get(),a++,sizeof(cl_mem),&ig); rt.set_arg(k.get(),a++,sizeof(cl_mem),&il);
      rt.set_arg(k.get(),a++,sizeof(cl_mem),&ilv);
      rt.set_arg(k.get(),a++,sizeof(cl_mem),&oc); rt.set_arg(k.get(),a++,sizeof(cl_mem),&ow);
      rt.set_arg(k.get(),a++,sizeof(cl_mem),&og); rt.set_arg(k.get(),a++,sizeof(cl_mem),&ol);
      rt.set_arg(k.get(),a++,sizeof(cl_mem),&olv);
      rt.set_arg(k.get(),a++,sizeof(cl_mem),&aL); rt.set_arg(k.get(),a++,sizeof(cl_mem),&aR);
      rt.set_arg(k.get(),a++,sizeof(cl_mem),&gc); rt.set_arg(k.get(),a++,sizeof(cl_mem),&dr);
      rt.run1d(k.get(), ((size_t)numBuckets << submaskBits)*WG, WG); }

    uint32_t drops[4]; rt.read(mDrops.get(),16,drops);
    std::printf("  drops: grpOvf=%u outBucketOvf=%u chainCap=%u\n",drops[1],drops[2],drops[3]);
    check(drops[1]==0 && drops[2]==0 && drops[3]==0, "fused kernel drop-free");

    // Read back GPU children.
    std::vector<uint32_t> oc(numBuckets); rt.read(mOutCounts.get(),(size_t)numBuckets*4,oc.data());
    std::vector<uint64_t> ow((size_t)numBuckets*outCap*7); rt.read(mOutWork.get(),ow.size()*8,ow.data());
    std::vector<uint32_t> olv((size_t)numBuckets*outCap*sOut); rt.read(mOutLeaves.get(),olv.size()*4,olv.data());
    std::vector<FChild> gpu;
    for(uint32_t b=0;b<numBuckets;++b){ uint32_t c=oc[b]<outCap?oc[b]:outCap;
        for(uint32_t p=0;p<c;++p){ size_t d=(size_t)b*outCap+p; FChild fc; fc.nlv=sOut;
            for(int w=0;w<7;++w) fc.w[w]=ow[d*7+w];
            for(uint32_t i=0;i<sOut;++i) fc.lv[i]=olv[d*sOut+i]; gpu.push_back(fc); } }

    if (exact) {
        // Oracle: cpu_round then re-mix each child for round r+1.
        std::vector<ref::RElem> rc = ref::cpu_round(in, r);
        std::vector<FChild> ref; ref.reserve(rc.size());
        for(auto& ch: rc){ FChild fc; uint64_t mw0=ref::cpu_mix_w0(ch,r+1);
            fc.w[0]=mw0; for(int w=1;w<7;++w) fc.w[w]=ch.w[w];
            fc.nlv=sOut; for(uint32_t i=0;i<sOut;++i) fc.lv[i]=(i<ch.tree.size())?ch.tree[i]:0u;
            ref.push_back(fc); }
        std::printf("  children: gpu=%zu ref=%zu\n", gpu.size(), ref.size());
        check(gpu.size()==ref.size(),"child count matches oracle");
        std::sort(gpu.begin(),gpu.end(),fchild_lt); std::sort(ref.begin(),ref.end(),fchild_lt);
        bool eq = gpu.size()==ref.size();
        for(size_t i=0;eq&&i<gpu.size();++i) eq = fchild_eq(gpu[i],ref[i]);
        check(eq,"fused children == cpu_round (mixed) as a multiset");
    } else {
        // Scale: oracle CHILD COUNT only (group by key, sum m*(m-1)/2), + GPU drop-free.
        std::unordered_map<uint32_t,uint32_t> grp; grp.reserve(N*2);
        for(uint32_t i=0;i<N;++i){ uint64_t mw0=ref::cpu_mix_w0(in[i],r); grp[(uint32_t)(mw0&0xFFFFFFu)]++; }
        uint64_t expPairs=0; for(auto&kv:grp){ uint64_t m=kv.second; expPairs+=m*(m-1)/2; }
        std::printf("  children: gpu=%zu expected(pairs)=%llu\n", gpu.size(),(unsigned long long)expPairs);
        check(gpu.size()==expPairs,"child count matches oracle pair count at scale");
    }
}

// DECOUPLED-SCATTER OCCUPANCY PROBE: race the SAME uncoalesced fat scatter (2^25
// 56B elems -> 16384 buckets) at three occupancy levels via a compile-time dummy
// __local array (tiny=max occ, 24KB=~2wg/SM, 40KB=~1wg/SM like the fused kernel),
// plus a coalesced-copy peak reference. If lo(40KB) is much slower than hi(tiny),
// the fused kernel's scatter is occupancy-starved and splitting it into its own
// high-occupancy kernel is the real lever; if they're equal, decoupling won't help.
static void test_scatter_occupancy(Runtime& rt, cl_program prog) {
    section("decoupled-scatter occupancy probe @ 2^25, 56B elems -> 16384 buckets");
    const uint32_t N=1u<<25, ew=7, bb=14, nb=1u<<bb;
    const uint32_t cap=(N>>bb)+(N>>(bb+1))+512;
    const int ITERS=6;
    auto best=[&](std::function<double()> once){ double b=1e9; for(int i=0;i<ITERS;++i){ double m=once(); if(i&&m<b)b=m;} return b; };
    std::vector<uint64_t> src((size_t)N*ew); uint64_t s=0x5CA77E4ULL;
    for(size_t i=0;i<src.size();++i) src[i]=sm(s);
    std::vector<uint32_t> keys(N); for(uint32_t i=0;i<N;++i) keys[i]=(uint32_t)(sm(s)&0xFFFFFFu);
    Mem mSrc=rt.alloc(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,src.size()*8,src.data());
    Mem mKeys=rt.alloc(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,keys.size()*4,keys.data());
    Mem mCounts=rt.alloc(CL_MEM_READ_WRITE,(size_t)nb*4);
    Mem mBuckets=rt.alloc(CL_MEM_READ_WRITE,(size_t)nb*cap*ew*8);
    Mem mDst=rt.alloc(CL_MEM_READ_WRITE,src.size()*8);
    Mem mDrops=rt.alloc(CL_MEM_READ_WRITE,4*4);
    const double GB=(double)(2.0*(double)N*ew*8)/1e9;   // read src + write buckets
    double tcopy=best([&]{ Kernel k=rt.kernel(prog,"bw_copy"); cl_mem a=mSrc.get(),b=mDst.get();
        rt.set_arg(k.get(),0,N); rt.set_arg(k.get(),1,ew); rt.set_arg(k.get(),2,sizeof(cl_mem),&a); rt.set_arg(k.get(),3,sizeof(cl_mem),&b);
        auto t0=std::chrono::steady_clock::now(); rt.run1d(k.get(),N);
        return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count(); });
    std::printf("  peak coalesced copy   : %.2f ms  (%.0f GB/s)\n", tcopy, GB/tcopy*1000.0);
    auto runScatter=[&](const char* name){ uint32_t md=0; double t=best([&]{
        rt.fill_u32(mCounts.get(),0u,nb); rt.fill_u32(mDrops.get(),0u,4);
        Kernel k=rt.kernel(prog,name); cl_mem sr=mSrc.get(),ky=mKeys.get(),c=mCounts.get(),bk=mBuckets.get(),dr=mDrops.get();
        rt.set_arg(k.get(),0,N); rt.set_arg(k.get(),1,bb); rt.set_arg(k.get(),2,cap); rt.set_arg(k.get(),3,ew);
        rt.set_arg(k.get(),4,sizeof(cl_mem),&sr); rt.set_arg(k.get(),5,sizeof(cl_mem),&ky);
        rt.set_arg(k.get(),6,sizeof(cl_mem),&c); rt.set_arg(k.get(),7,sizeof(cl_mem),&bk); rt.set_arg(k.get(),8,sizeof(cl_mem),&dr);
        auto t0=std::chrono::steady_clock::now(); rt.run1d(k.get(),N,256);
        double m=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
        rt.read(mDrops.get(),4,&md); return m; });
        std::printf("  %-24s: %.2f ms  (%.0f GB/s)  drops=%u\n", name, t, GB/t*1000.0, md); return t; };
    // Isolate the per-bucket atomic: same true-random destinations, no atomic_inc.
    {
        double t=best([&]{ Kernel k=rt.kernel(prog,"scatter_noatomic");
            cl_mem sr=mSrc.get(),ky=mKeys.get(),bk=mBuckets.get();
            rt.set_arg(k.get(),0,N); rt.set_arg(k.get(),1,bb); rt.set_arg(k.get(),2,cap); rt.set_arg(k.get(),3,ew);
            rt.set_arg(k.get(),4,sizeof(cl_mem),&sr); rt.set_arg(k.get(),5,sizeof(cl_mem),&ky);
            rt.set_arg(k.get(),6,sizeof(cl_mem),&bk);
            auto t0=std::chrono::steady_clock::now(); rt.run1d(k.get(),N,256);
            return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count(); });
        std::printf("  %-24s: %.2f ms  (%.0f GB/s)   <- random dest, NO per-bucket atomic\n",
                    "scatter_noatomic", t, GB/t*1000.0);
    }
    double thi =runScatter("scatter_probe_hi");
    double tmid=runScatter("scatter_probe_mid");
    double tlo =runScatter("scatter_probe_lo");
    std::printf("  --> SAME scatter vs occupancy: hi(tiny)=%.2f  mid(24KB)=%.2f  lo(40KB=fused)=%.2f ms  (lo/hi=%.2fx)\n",
        thi,tmid,tlo, tlo/thi);

    // Does scatter BW depend on the number of output buckets (DRAM row-buffer locality)?
    // Coarser buckets = fewer open rows = more sequential writes. mBuckets is sized for
    // bb=14; coarser bb has a smaller footprint so it fits the same buffer.
    std::printf("  scatter BW vs #output buckets (scatter_probe_hi):\n");
    for (uint32_t bbv : {8u,10u,12u,14u}) {
        uint32_t nbv=1u<<bbv, capv=(N>>bbv)+(N>>(bbv+1))+512;
        uint32_t md=0; double t=best([&]{
            rt.fill_u32(mCounts.get(),0u,nbv); rt.fill_u32(mDrops.get(),0u,4);
            Kernel k=rt.kernel(prog,"scatter_probe_hi"); cl_mem sr=mSrc.get(),ky=mKeys.get(),c=mCounts.get(),bk=mBuckets.get(),dr=mDrops.get();
            rt.set_arg(k.get(),0,N); rt.set_arg(k.get(),1,bbv); rt.set_arg(k.get(),2,capv); rt.set_arg(k.get(),3,ew);
            rt.set_arg(k.get(),4,sizeof(cl_mem),&sr); rt.set_arg(k.get(),5,sizeof(cl_mem),&ky);
            rt.set_arg(k.get(),6,sizeof(cl_mem),&c); rt.set_arg(k.get(),7,sizeof(cl_mem),&bk); rt.set_arg(k.get(),8,sizeof(cl_mem),&dr);
            auto t0=std::chrono::steady_clock::now(); rt.run1d(k.get(),N,256);
            double m=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
            rt.read(mDrops.get(),4,&md); return m; });
        std::printf("    bb=%2u (%7u buckets): %.2f ms  (%.0f GB/s)  drops=%u\n", bbv, nbv, t, GB/t*1000.0, md);
    }
    check(true,"scatter occupancy characterized");
}

// COALESCING BENEFIT CURVE -- the decisive test for every reorder scheme (two-level
// bucketing, magazine, local-reorder-then-flush). Their ONLY effect is making
// same-destination writes contiguous, so measure that payoff directly instead of
// building the machinery: runLen=1 is today's per-lane scatter, runLen=32 a fully
// coalesced warp burst. Swept at the real per-round emit widths (r1/r2=7, r3=6,
// r4=5 u64), which also answers whether a late-round ("thinner element") hybrid
// could flip the math.
static void test_coalescing_curve(Runtime& rt, cl_program prog) {
    section("coalescing benefit curve: scatter throughput vs run length @ 2^25");
    const uint32_t N=1u<<25, nb=1u<<14, cap=(N/nb)+64;
    const int ITERS=6;
    auto best=[&](std::function<double()> once){ double b=1e9; for(int i=0;i<ITERS;++i){ double m=once(); if(i&&m<b)b=m;} return b; };
    std::vector<uint64_t> src((size_t)N*7); uint64_t s=0xC0A1E5CEULL;
    for(size_t i=0;i<src.size();++i) src[i]=sm(s);
    Mem mSrc=rt.alloc(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,src.size()*8,src.data());
    Mem mDst=rt.alloc(CL_MEM_READ_WRITE,(size_t)nb*cap*7*8);
    std::vector<uint32_t> keys(N); for(uint32_t i=0;i<N;++i) keys[i]=(uint32_t)(sm(s)&0xFFFFFFu);
    Mem mKeys=rt.alloc(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,keys.size()*4,keys.data());
    Mem mCnt=rt.alloc(CL_MEM_READ_WRITE,(size_t)nb*4);
    Mem mDr=rt.alloc(CL_MEM_READ_WRITE,4*4);
    const uint32_t RUNS[6]={1,2,4,8,16,32};
    for (uint32_t ew : {7u,6u,5u}) {
        const double GB=(double)(2.0*(double)N*ew*8)/1e9;
        double t1=0, t32=0;
        std::printf("  ew=%u (%2uB):", ew, ew*8);
        for (int i=0;i<6;++i) { uint32_t R=RUNS[i];
            double t=best([&]{ Kernel k=rt.kernel(prog,"scatter_runs");
                cl_mem a=mSrc.get(), b2=mDst.get();
                rt.set_arg(k.get(),0,N); rt.set_arg(k.get(),1,ew); rt.set_arg(k.get(),2,R);
                rt.set_arg(k.get(),3,nb); rt.set_arg(k.get(),4,cap);
                rt.set_arg(k.get(),5,sizeof(cl_mem),&a); rt.set_arg(k.get(),6,sizeof(cl_mem),&b2);
                auto t0=std::chrono::steady_clock::now(); rt.run1d(k.get(),N,256);
                return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count(); });
            if (R==1) t1=t; if (R==32) t32=t;
            std::printf("  R%u=%.1f(%.0f)", R, t, GB/t*1000.0);
        }
        std::printf("   [ms(GB/s)]\n");
        // TRUE-RANDOM destination baseline at this width (what the real emit does),
        // vs R=32 (perfect reordering). Two-level bucketing needs TWO passes, so it
        // can only pay if this ratio exceeds 2.0x.
        uint32_t md=0; double trand=best([&]{
            rt.fill_u32(mCnt.get(),0u,nb);
            Kernel k=rt.kernel(prog,"scatter_probe_hi"); cl_mem sr=mSrc.get(),ky=mKeys.get(),c=mCnt.get(),bk=mDst.get(),dr=mDr.get();
            rt.set_arg(k.get(),0,N); rt.set_arg(k.get(),1,14u); rt.set_arg(k.get(),2,cap); rt.set_arg(k.get(),3,ew);
            rt.set_arg(k.get(),4,sizeof(cl_mem),&sr); rt.set_arg(k.get(),5,sizeof(cl_mem),&ky);
            rt.set_arg(k.get(),6,sizeof(cl_mem),&c); rt.set_arg(k.get(),7,sizeof(cl_mem),&bk); rt.set_arg(k.get(),8,sizeof(cl_mem),&dr);
            auto t0=std::chrono::steady_clock::now(); rt.run1d(k.get(),N,256);
            double m=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
            rt.read(mDr.get(),4,&md); return m; });
        std::printf("    -> REAL(random dest)=%.1f ms (%.0f GB/s) | perfect(R=32)=%.1f ms (%.0f GB/s)"
                    " | max reorder gain %.2fx  %s 2.0x two-pass breakeven\n",
                    trand, GB/trand*1000.0, t32, GB/t32*1000.0, trand/t32,
                    (trand/t32 > 2.0) ? "CLEARS" : "below");
    }
    check(true,"coalescing curve characterized");
}

// SPLIT vs PACKED emit: the real fused emit scatters each child across FOUR arrays
// (work, gi, lead, leaves) at the same random index. Global writes are serviced in
// 32 B sectors, so each 4 B field burns a whole sector. Same useful bytes (72 B),
// three layouts: split / packed 72 B / packed+padded to 96 B (sector-aligned).
static void test_emit_packing(Runtime& rt, cl_program prog) {
    section("emit layout: 4 split arrays vs 1 packed record @ 2^25");
    // Tight cap: uniform keys give mean 2048/bucket with sigma ~45, so +256 is ~5.7
    // sigma and stays drop-free. Needed to keep the 96 B padded array under the
    // device's 3.9 GB single-allocation limit.
    const uint32_t N=1u<<25, bb=14, nb=1u<<bb, cap=(N>>bb)+256;
    const int ITERS=6;
    auto best=[&](std::function<double()> once){ double b=1e9; for(int i=0;i<ITERS;++i){ double m=once(); if(i&&m<b)b=m;} return b; };
    uint64_t s=0xE311770ULL;
    std::vector<uint32_t> keys(N); for(uint32_t i=0;i<N;++i) keys[i]=(uint32_t)(sm(s)&0xFFFFFFu);
    Mem mKeys=rt.alloc(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,keys.size()*4,keys.data());
    Mem mCnt=rt.alloc(CL_MEM_READ_WRITE,(size_t)nb*4);
    Mem mDr=rt.alloc(CL_MEM_READ_WRITE,4*4);
    const size_t nslots=(size_t)nb*cap;
    Mem mW=rt.alloc(CL_MEM_READ_WRITE,nslots*7*8);
    Mem mG=rt.alloc(CL_MEM_READ_WRITE,nslots*4);
    Mem mL=rt.alloc(CL_MEM_READ_WRITE,nslots*4);
    Mem mLv=rt.alloc(CL_MEM_READ_WRITE,nslots*2*4);
    Mem mP=rt.alloc(CL_MEM_READ_WRITE,nslots*9*8);
    Mem mPP=rt.alloc(CL_MEM_READ_WRITE,nslots*12*8);
    const double USEFUL=(double)((double)N*72.0)/1e9;   // 72 B of real payload per child
    auto run=[&](const char* name, bool split, cl_mem packed){ double t=best([&]{
        rt.fill_u32(mCnt.get(),0u,nb); rt.fill_u32(mDr.get(),0u,4);
        Kernel k=rt.kernel(prog,name);
        cl_mem ky=mKeys.get(),c=mCnt.get(),dr=mDr.get();
        rt.set_arg(k.get(),0,N); rt.set_arg(k.get(),1,bb); rt.set_arg(k.get(),2,cap);
        rt.set_arg(k.get(),3,sizeof(cl_mem),&ky); rt.set_arg(k.get(),4,sizeof(cl_mem),&c);
        if (split) { cl_mem w=mW.get(),g2=mG.get(),l=mL.get(),lv=mLv.get();
            rt.set_arg(k.get(),5,sizeof(cl_mem),&w); rt.set_arg(k.get(),6,sizeof(cl_mem),&g2);
            rt.set_arg(k.get(),7,sizeof(cl_mem),&l); rt.set_arg(k.get(),8,sizeof(cl_mem),&lv);
            rt.set_arg(k.get(),9,sizeof(cl_mem),&dr); }
        else { rt.set_arg(k.get(),5,sizeof(cl_mem),&packed); rt.set_arg(k.get(),6,sizeof(cl_mem),&dr); }
        auto t0=std::chrono::steady_clock::now(); rt.run1d(k.get(),N,256);
        return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count(); });
        std::printf("  %-16s: %.2f ms  (%.0f GB/s of useful payload)\n", name, t, USEFUL/t*1000.0);
        return t; };
    double tS=run("emit_split",true,nullptr);
    double tP=run("emit_packed",false,mP.get());
    double tPP=run("emit_packed_pad",false,mPP.get());
    std::printf("  --> packed %.2fx faster than split | padded-96B %.2fx faster than split\n", tS/tP, tS/tPP);
    check(true,"emit packing characterized");
}

// Pins the algebra the round-3/round-4 compact leftContrib rests on (CPU-only, no
// GPU): apply_mix decomposes as rotl24(workPart + indexPart) with indexPart additive
// over leaves, so the LEFT parent's 8 leaves can be pre-folded into ONE u64 by round 3
// and round 4 reconstructs the exact round-5 key from that plus the RIGHT parent's
// lead. If this ever fails, the compact leaf payload in lds.cl (LMODE_EMIT/LMODE_USE)
// is invalid.
static void test_contrib_identity() {
    section("compact leftContrib identity (CPU algebra behind LMODE_EMIT/LMODE_USE)");
    auto rotl=[](uint64_t x,int b){ return (uint64_t)((x<<b)|(x>>(64-b))); };
    const int ROT[8]={29,58,23,52,17,46,11,40};
    auto mix=[&](const uint64_t e[7], const uint32_t* tree, uint32_t treeLen, uint32_t Lmix){
        uint64_t t[8]; for(int i=0;i<7;++i)t[i]=e[i]; t[7]=0;
        uint32_t padNum=((512u-Lmix)+24u)/25u; if(padNum>treeLen)padNum=treeLen;
        for(uint32_t i=0;i<padNum;++i){ uint32_t pos=Lmix+i*25u; if(pos>=512u)break; uint64_t v=tree[i];
            uint32_t w=pos>>6,sh=pos&63u; t[w]|=v<<sh; if(sh>39u&&w<7u)t[w+1]|=v>>(64u-sh); }
        uint64_t r=0; for(int i=0;i<8;++i)r+=rotl(t[i],ROT[i]); return rotl(r,24); };
    const uint64_t ZERO[7]={0,0,0,0,0,0,0};
    uint64_t s=0xC0FFEE99ULL; auto rnd=[&](){ s^=s<<13; s^=s>>7; s^=s<<17; return s; };
    int fails=0; const int T=200000;
    for(int trial=0;trial<T;++trial){
        uint64_t c[7]; for(int i=0;i<7;++i)c[i]=rnd();
        for(int i=0;i<7;++i){ uint32_t base=64u*i;                     // mask to Lout(4)=288
            if(base>=288u)c[i]=0; else if(288u-base<64u)c[i]&=((uint64_t)1<<(288u-base))-1; }
        uint32_t L[8],R0; for(int i=0;i<8;++i)L[i]=(uint32_t)(rnd()&0x1FFFFFF); R0=(uint32_t)(rnd()&0x1FFFFFF);
        uint32_t tree9[9]; for(int i=0;i<8;++i)tree9[i]=L[i]; tree9[8]=R0;
        uint64_t truth=mix(c,tree9,9,288);
        uint64_t leftContrib=rotl(mix(ZERO,L,8,288),40);               // what round 3 emits
        uint32_t rightOnly[9]={0,0,0,0,0,0,0,0,R0};
        uint64_t got=rotl(rotl(mix(c,rightOnly,9,288),40)+leftContrib,24);   // what round 4 does
        if(truth!=got)++fails;
    }
    std::printf("  %d/%d mismatches over random (work, 8 left leaves, right lead)\n",fails,T);
    check(fails==0,"compact leftContrib reproduces the full mix bit-exactly");
}

int main() {
    if (!Runtime::any_device_available()) { std::printf("SKIP: no OpenCL device\n"); return 0; }
    Runtime rt;
    std::printf("  device: %s | local_mem=%llu KB\n", rt.device().name.c_str(),
        (unsigned long long)(rt.device().local_mem/1024));
    // lds.cl's pipeline kernels reference BH3_MAX_LEAVES (round.cl) + bh3_combine
    // (bh3.cl); compile {bh3, round, lds} as the pipeline does.
    Program prog = rt.build({std::string(kBh3ClSource), std::string(kRoundClSource),
                             std::string(kLdsClSource)}, "");

    test_contrib_identity();
    test_correctness(rt, prog.get());
    test_combine(rt, prog.get());
    test_scatter_cost(rt, prog.get());
    test_compaction_arch(rt, prog.get());
    test_gather_locality(rt, prog.get());
    test_leaf_locality(rt, prog.get());
    test_scatter_occupancy(rt, prog.get());
    test_coalescing_curve(rt, prog.get());
    test_emit_packing(rt, prog.get());

    // STEP B: fused round kernel correctness (exact vs oracle) + scale/drop-free.
    // Only r=1,2 use the RAW leaf variant (round_fused_lds, LEAFW=2); r3 emits the
    // compact leftContrib and r4 consumes it (different kernels/semantics), covered
    // by test_contrib_identity below + the byte-exact KAT goldens end-to-end.
    for (int r=1; r<=2; ++r)
        test_fused_round(rt, prog.get(), r, 1u<<18, /*bits*/11, /*submask*/0, /*inCap*/384, /*outCap*/384, /*exact*/true);
    test_fused_round(rt, prog.get(), 1, 1u<<24, /*bits*/15, /*submask*/2, /*inCap*/768, /*outCap*/768, /*exact*/false);

    // Compaction prototype: how much does shrinking the element to its
    // significant-word schedule actually save on the dominant emit-to-bucket cost?
    {
        section("compaction: emit-to-bucket cost vs element width (u64 words)");
        // BeamHash III Lout schedule -> significant words = ceil(Lout/64) for the
        // OUTPUT of each round (= what that round emits).
        const uint32_t Lout[5]={424,400,376,288,24};
        uint32_t emitWords[5];
        std::printf("  sig-word schedule (emit width per round): ");
        for(int r=0;r<5;++r){ emitWords[r]=(Lout[r]+63)/64; std::printf("r%d=%u(%ub) ",r+1,emitWords[r],Lout[r]); }
        std::printf("\n");
        const uint32_t N=1u<<25; auto w=gen_elem(N,24,0x3);
        const uint32_t bb=14, numBuckets=1u<<bb, bucketCap=(N>>bb)+(N>>(bb+2))+512;
        Mem mW=rt.alloc(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,(size_t)N*7*8,(void*)w.data());
        Mem mC=rt.alloc(CL_MEM_READ_WRITE,(size_t)numBuckets*4);
        Mem mBW=rt.alloc(CL_MEM_READ_WRITE,(size_t)numBuckets*bucketCap*7*8);  // 7-word max
        Mem mD=rt.alloc(CL_MEM_READ_WRITE,4*4);
        double emit[8]={0}; for(int i=0;i<8;++i) emit[i]=1e9;
        for(uint32_t ew=1; ew<=7; ++ew){
            for(int it=0;it<4;++it){ rt.fill_u32(mC.get(),0u,numBuckets); rt.fill_u32(mD.get(),0u,4);
                Kernel k=rt.kernel(prog.get(),"p2_scatter_w"); cl_mem a=mW.get(),c=mC.get(),bw=mBW.get(),d=mD.get();
                rt.set_arg(k.get(),0,N);rt.set_arg(k.get(),1,bb);rt.set_arg(k.get(),2,bucketCap);rt.set_arg(k.get(),3,ew);
                rt.set_arg(k.get(),4,sizeof(cl_mem),&a);rt.set_arg(k.get(),5,sizeof(cl_mem),&c);
                rt.set_arg(k.get(),6,sizeof(cl_mem),&bw);rt.set_arg(k.get(),7,sizeof(cl_mem),&d);
                auto t0=std::chrono::steady_clock::now(); rt.run1d(k.get(),N);
                double m=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
                if(it&&m<emit[ew])emit[ew]=m; }
            std::printf("  emit-to-bucket ew=%u words (%u B): %.2f ms\n", ew, ew*8, emit[ew]);
        }
        double uncompacted=5*emit[7];
        double compacted=emit[emitWords[0]]+emit[emitWords[1]]+emit[emitWords[2]]+emit[emitWords[3]]+emit[emitWords[4]];
        std::printf("  --> emit total/solve: uncompacted(5x7w)=%.1f ms  vs  compacted[%u,%u,%u,%u,%u]=%.1f ms  (%.0f%% saved)\n",
            uncompacted, emitWords[0],emitWords[1],emitWords[2],emitWords[3],emitWords[4], compacted,
            100.0*(uncompacted-compacted)/uncompacted);
        check(true,"compaction emit curve measured");
    }

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
