#include "gpu/cl_runtime.h"
#include "generated/mxbm_kernels.h"
#include "gpu_probe_kernels.h"
#include "check.h"
#include "kat_vectors.h"
#include "bh3_primitives.h"
#include <cstdint>
#include <string>
#include <vector>
using namespace mxbm;
using namespace mxbm::gpu;

// Deterministic pseudo-random (no <random> dependence): splitmix64.
static uint64_t sm(uint64_t& s){ uint64_t z=(s+=0x9E3779B97F4A7C15ULL);
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31); }

static Program build_probe(Runtime& rt) {
    return rt.build({std::string(kBh3ClSource), std::string(kProbeSrc)}, "");
}

static void test_siphash(Runtime& rt, Program& prog) {
    section("cl siphash24 == C++ siphash24");
    // Anchored KAT keys: prePow, plus differential over random nonces.
    const uint64_t* K = kat::prePow;
    const size_t N = 4096;
    std::vector<uint64_t> nonce(N), out(N);
    uint64_t s = 0xC0FFEE;
    nonce[0]=0; nonce[1]=1; nonce[2]=7; nonce[3]=8; nonce[4]=0xffffff;   // KAT anchors
    for (size_t i=5;i<N;++i) nonce[i]=sm(s);
    uint64_t key4[4] = {K[0],K[1],K[2],K[3]};
    Kernel k = rt.kernel(prog.get(), "probe_siphash");
    Mem mk = rt.alloc(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof key4, key4);
    Mem mn = rt.alloc(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, N*8, nonce.data());
    Mem mo = rt.alloc(CL_MEM_WRITE_ONLY, N*8);
    cl_mem a=mk.get(),b=mn.get(),c=mo.get();
    rt.set_arg(k.get(),0,sizeof(cl_mem),&a); rt.set_arg(k.get(),1,sizeof(cl_mem),&b); rt.set_arg(k.get(),2,sizeof(cl_mem),&c);
    rt.run1d(k.get(), N);
    rt.read(mo.get(), N*8, out.data());
    int bad=0;
    for (size_t i=0;i<N;++i) {
        uint64_t want = bh3::siphash24(K[0],K[1],K[2],K[3], nonce[i]);
        if (out[i]!=want){ if(bad<3) std::printf("  MISMATCH nonce=%llu got=%llx want=%llx\n",
            (unsigned long long)nonce[i],(unsigned long long)out[i],(unsigned long long)want); ++bad; }
    }
    check(bad==0, "siphash24 differential (4096 cases incl KAT anchors)");
    // Explicit KAT value anchor:
    check(out[0]==0x10b04caf51290da9ULL, "siphash prePow n=0 == KAT");
}

static void test_seed(Runtime& rt, Program& prog) {
    section("cl seed_element == C++ seed_element");
    const size_t N = 4096;
    std::vector<uint32_t> idx(N);
    std::vector<uint64_t> out(N*7);
    idx[0]=0; idx[1]=1; idx[2]=12345678u; idx[3]=0xffffffu; idx[4]=(1u<<25)-1u;  // anchors
    uint64_t s=0xABCDEF; for(size_t i=5;i<N;++i) idx[i]=(uint32_t)(sm(s) & 0x1FFFFFFu);
    uint64_t pp[4]={kat::prePow[0],kat::prePow[1],kat::prePow[2],kat::prePow[3]};
    Kernel k = rt.kernel(prog.get(), "probe_seed");
    Mem mp = rt.alloc(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof pp, pp);
    Mem mi = rt.alloc(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, N*4, idx.data());
    Mem mo = rt.alloc(CL_MEM_WRITE_ONLY, N*7*8);
    cl_mem a=mp.get(),b=mi.get(),c=mo.get();
    rt.set_arg(k.get(),0,sizeof(cl_mem),&a); rt.set_arg(k.get(),1,sizeof(cl_mem),&b); rt.set_arg(k.get(),2,sizeof(cl_mem),&c);
    rt.run1d(k.get(), N);
    rt.read(mo.get(), N*7*8, out.data());
    int bad=0;
    for (size_t i=0;i<N;++i) {
        bh3::Elem e; bh3::seed_element(pp, idx[i], e);
        for (int w=0; w<7; ++w) if (out[i*7+w]!=e.w[w]) { ++bad; break; }
    }
    check(bad==0, "seed_element differential (4096 cases incl KAT anchors)");
    // KAT anchor idx0 word0:
    check(out[0]==kat::seed_index0[0], "seed idx0 word0 == KAT");
}

static void test_collision(Runtime& rt, Program& prog) {
    section("cl collision_bits == C++ collision_bits");
    const size_t N = 4096;
    std::vector<uint64_t> w0(N);
    std::vector<uint32_t> out(N);
    w0[0] = kat::mix448_index0;   // 0x99a961e498026291 -> lane 0x026291 (KAT anchor)
    uint64_t s=0x5151; for(size_t i=1;i<N;++i) w0[i]=sm(s);
    Kernel k = rt.kernel(prog.get(), "probe_collision");
    Mem mi = rt.alloc(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, N*8, w0.data());
    Mem mo = rt.alloc(CL_MEM_WRITE_ONLY, N*4);
    cl_mem a=mi.get(),b=mo.get();
    rt.set_arg(k.get(),0,sizeof(cl_mem),&a); rt.set_arg(k.get(),1,sizeof(cl_mem),&b);
    rt.run1d(k.get(), N);
    rt.read(mo.get(), N*4, out.data());
    int bad=0;
    for (size_t i=0;i<N;++i) {
        bh3::Elem e{}; e.w[0]=w0[i];
        if (out[i]!=bh3::collision_bits(e)) { ++bad; }
    }
    check(bad==0, "collision_bits differential (4096 cases)");
    check(out[0]==0x026291u, "collision_bits KAT anchor (mix448 idx0 lane)");
}

int main() {
    if (!Runtime::any_device_available()) { std::printf("SKIP: no OpenCL device\n"); return 0; }
    Runtime rt;
    Program prog = build_probe(rt);
    test_siphash(rt, prog);
    test_seed(rt, prog);
    test_collision(rt, prog);
    return summary("cl_primitives");
}
