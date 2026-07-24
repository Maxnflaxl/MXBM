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

int main() {
    if (!Runtime::any_device_available()) { std::printf("SKIP: no OpenCL device\n"); return 0; }
    Runtime rt;
    std::printf("  device: %s | local_mem=%llu KB\n", rt.device().name.c_str(),
        (unsigned long long)(rt.device().local_mem/1024));
    Program prog = rt.build({std::string(kBh3ClSource), std::string(kLdsClSource)}, "");

    test_correctness(rt, prog.get());
    test_combine(rt, prog.get());
    test_scatter_cost(rt, prog.get());

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
