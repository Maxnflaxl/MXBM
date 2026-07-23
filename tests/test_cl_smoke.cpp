#include "gpu/cl_runtime.h"
#include "check.h"
#include <cstdint>
#include <string>
#include <vector>
using namespace mxbm;
using namespace mxbm::gpu;

static const char* kIncSrc =
    "__kernel void inc(__global uint* b){ size_t i=get_global_id(0); b[i]=b[i]+1u; }\n";

int main() {
    if (!Runtime::any_device_available()) { std::printf("SKIP: no OpenCL device\n"); return 0; }
    Runtime rt;
    const DeviceInfo& d = rt.device();
    section("OpenCL runtime smoke");
    std::printf("  device: %s | gmem=%lluMB | CUs=%u\n", d.name.c_str(),
                (unsigned long long)(d.global_mem >> 20), d.compute_units);
    check(d.global_mem > 0, "device reports global memory");

    Program prog = rt.build({std::string(kIncSrc)}, "");
    Kernel k = rt.kernel(prog.get(), "inc");

    const size_t N = 1024;
    std::vector<uint32_t> host(N);
    for (size_t i = 0; i < N; ++i) host[i] = (uint32_t)i;
    Mem buf = rt.alloc(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N * sizeof(uint32_t), host.data());
    cl_mem raw = buf.get();
    rt.set_arg(k.get(), 0, sizeof(cl_mem), &raw);
    rt.run1d(k.get(), N);
    rt.read(buf.get(), N * sizeof(uint32_t), host.data());

    bool ok = true;
    for (size_t i = 0; i < N; ++i) if (host[i] != (uint32_t)i + 1) { ok = false; break; }
    check(ok, "trivial kernel incremented every element");
    return summary("cl_smoke");
}
