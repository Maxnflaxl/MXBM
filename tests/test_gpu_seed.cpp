#include "gpu/cl_runtime.h"
#include "gpu/budget.h"
#include "generated/mxbm_kernels.h"
#include "check.h"
#include "kat_vectors.h"
#include "bh3_primitives.h"
#include <cstdint>
#include <string>
#include <vector>
using namespace mxbm;
using namespace mxbm::gpu;

int main() {
    if (!Runtime::any_device_available()) { std::printf("SKIP: no OpenCL device\n"); return 0; }
    Runtime rt;
    Budget bud = compute_budget(rt.device().global_mem, rt.device().max_alloc);
    section("GPU seed layer == C++ seed_element over full 2^25 range");
    std::printf("  device: %s | seed_batch=%u | elems/round=%u\n",
                rt.device().name.c_str(), bud.seed_batch, bud.elems_per_round);

    Program prog = rt.build({std::string(kBh3ClSource), std::string(kSeedClSource)}, "");
    Kernel k = rt.kernel(prog.get(), "seed_layer");

    uint64_t pp[4] = {kat::prePow[0],kat::prePow[1],kat::prePow[2],kat::prePow[3]};
    Mem mp = rt.alloc(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof pp, pp);

    const uint32_t TOTAL = 1u<<25;
    const uint32_t batch = bud.seed_batch;
    Mem mo = rt.alloc(CL_MEM_WRITE_ONLY, (size_t)batch*7*8);
    std::vector<uint64_t> out((size_t)batch*7);

    // Verify all anchors exactly + a strided sample across the WHOLE range,
    // driving the kernel over every batch so it covers all 2^25 indices.
    const uint32_t anchors[5] = {0u,1u,12345678u,0xffffffu,TOTAL-1u};
    const uint32_t STRIDE = 9973;   // prime; ~3363 samples per batch
    long checked=0; int bad=0;
    for (uint32_t begin=0; begin<TOTAL; begin+=batch) {
        uint32_t count = (TOTAL-begin < batch) ? (TOTAL-begin) : batch;
        cl_mem a=mp.get(), c=mo.get();
        rt.set_arg(k.get(),0,sizeof(cl_mem),&a);
        rt.set_arg(k.get(),1,begin);
        rt.set_arg(k.get(),2,count);
        rt.set_arg(k.get(),3,sizeof(cl_mem),&c);
        rt.run1d(k.get(), count);
        rt.read(mo.get(), (size_t)count*7*8, out.data());
        for (uint32_t g=0; g<count; ++g) {
            uint32_t idx = begin+g;
            bool is_anchor=false; for (uint32_t an : anchors) if (idx==an) is_anchor=true;
            if (!is_anchor && (idx % STRIDE)!=0) continue;
            bh3::Elem e; bh3::seed_element(pp, idx, e);
            for (int w=0; w<7; ++w) if (out[(size_t)g*7+w]!=e.w[w]) {
                if (bad<3) std::printf("  MISMATCH idx=%u w=%d got=%llx want=%llx\n",
                    idx,w,(unsigned long long)out[(size_t)g*7+w],(unsigned long long)e.w[w]);
                ++bad; break;
            }
            ++checked;
        }
    }
    std::printf("  checked %ld sampled indices across full 2^25 range\n", checked);
    check(bad==0, "GPU seed bit-identical to C++ seed_element (anchors + strided sample)");
    check(checked > 3000, "sample coverage spans the whole range");
    return summary("gpu_seed");
}
