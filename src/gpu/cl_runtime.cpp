#include "gpu/cl_runtime.h"
#include <string>
#include <vector>

namespace mxbm { namespace gpu {
namespace {

// First platform that has at least one device; prefer GPU, else any type.
bool pick_device(cl_platform_id* plat_out, cl_device_id* dev_out) {
    *dev_out = nullptr;
    cl_uint nplat = 0;
    if (clGetPlatformIDs(0, nullptr, &nplat) != CL_SUCCESS || nplat == 0) return false;
    std::vector<cl_platform_id> plats(nplat);
    if (clGetPlatformIDs(nplat, plats.data(), nullptr) != CL_SUCCESS) return false;
    for (cl_device_type type : {(cl_device_type)CL_DEVICE_TYPE_GPU, (cl_device_type)CL_DEVICE_TYPE_ALL}) {
        for (cl_platform_id p : plats) {
            cl_uint ndev = 0;
            if (clGetDeviceIDs(p, type, 0, nullptr, &ndev) == CL_SUCCESS && ndev > 0) {
                if (clGetDeviceIDs(p, type, 1, dev_out, nullptr) != CL_SUCCESS) continue;
                *plat_out = p;
                return true;
            }
        }
    }
    return false;
}

std::string str_info(cl_device_id d, cl_device_info k) {
    size_t n = 0;
    if (clGetDeviceInfo(d, k, 0, nullptr, &n) != CL_SUCCESS || n == 0) return {};
    std::string s(n, '\0');
    clGetDeviceInfo(d, k, n, &s[0], nullptr);
    if (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
}

void check_cl(cl_int err, const char* what) {
    if (err != CL_SUCCESS) throw ClError(err, std::string(what) + " (cl err " + std::to_string(err) + ")");
}

} // namespace

bool Runtime::any_device_available() {
    cl_platform_id p; cl_device_id d;
    return pick_device(&p, &d);
}

Runtime::Runtime() {
    if (!pick_device(&plat_, &dev_)) throw ClError(CL_DEVICE_NOT_FOUND, "no OpenCL device");
    info_.name        = str_info(dev_, CL_DEVICE_NAME);
    info_.version     = str_info(dev_, CL_DEVICE_VERSION);
    info_.clc_version = str_info(dev_, CL_DEVICE_OPENCL_C_VERSION);
    clGetDeviceInfo(dev_, CL_DEVICE_GLOBAL_MEM_SIZE,     sizeof info_.global_mem,    &info_.global_mem,    nullptr);
    clGetDeviceInfo(dev_, CL_DEVICE_MAX_MEM_ALLOC_SIZE,  sizeof info_.max_alloc,     &info_.max_alloc,     nullptr);
    clGetDeviceInfo(dev_, CL_DEVICE_MAX_COMPUTE_UNITS,   sizeof info_.compute_units, &info_.compute_units, nullptr);
    cl_int err = CL_SUCCESS;
    ctx_ = clCreateContext(nullptr, 1, &dev_, nullptr, nullptr, &err);
    check_cl(err, "clCreateContext");
    q_ = clCreateCommandQueue(ctx_, dev_, 0, &err);  // OpenCL 1.2 baseline
    if (err != CL_SUCCESS) {
        clReleaseContext(ctx_);
        ctx_ = nullptr;
        check_cl(err, "clCreateCommandQueue");
    }
}

Runtime::~Runtime() {
    if (q_)   clReleaseCommandQueue(q_);
    if (ctx_) clReleaseContext(ctx_);
}

Program Runtime::build(const std::vector<std::string>& sources, const std::string& options) {
    std::vector<const char*> ptrs;
    std::vector<size_t> lens;
    ptrs.reserve(sources.size());
    lens.reserve(sources.size());
    for (const auto& s : sources) { ptrs.push_back(s.c_str()); lens.push_back(s.size()); }
    cl_int err = CL_SUCCESS;
    cl_program prog = clCreateProgramWithSource(ctx_, (cl_uint)ptrs.size(), ptrs.data(), lens.data(), &err);
    check_cl(err, "clCreateProgramWithSource");
    err = clBuildProgram(prog, 1, &dev_, options.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t n = 0;
        clGetProgramBuildInfo(prog, dev_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
        std::string log(n, '\0');
        clGetProgramBuildInfo(prog, dev_, CL_PROGRAM_BUILD_LOG, n, &log[0], nullptr);
        clReleaseProgram(prog);
        throw ClError(err, "clBuildProgram failed:\n" + log);
    }
    return Program(prog);
}

Kernel Runtime::kernel(cl_program prog, const char* name) {
    cl_int err = CL_SUCCESS;
    cl_kernel k = clCreateKernel(prog, name, &err);
    check_cl(err, "clCreateKernel");
    return Kernel(k);
}

Mem Runtime::alloc(cl_mem_flags flags, size_t bytes, void* host_ptr) {
    cl_int err = CL_SUCCESS;
    cl_mem m = clCreateBuffer(ctx_, flags, bytes, host_ptr, &err);
    check_cl(err, "clCreateBuffer");
    return Mem(m);
}

void Runtime::write(cl_mem buf, size_t bytes, const void* src) {
    check_cl(clEnqueueWriteBuffer(q_, buf, CL_TRUE, 0, bytes, src, 0, nullptr, nullptr), "clEnqueueWriteBuffer");
}
void Runtime::read(cl_mem buf, size_t bytes, void* dst) {
    check_cl(clEnqueueReadBuffer(q_, buf, CL_TRUE, 0, bytes, dst, 0, nullptr, nullptr), "clEnqueueReadBuffer");
}
void Runtime::fill_u32(cl_mem buf, uint32_t value, size_t count) {
    // NOTE: event=nullptr here is silently unreliable on at least one real
    // OpenCL implementation (Apple Silicon / macOS's OpenCL-over-Metal
    // shim): clEnqueueFillBuffer and the trailing clFinish both report
    // CL_SUCCESS, yet the fill never lands -- the buffer is left with
    // whatever it held before. Capturing the completion event and blocking
    // on it explicitly (clWaitForEvents) is what actually forces the fill to
    // execute; clFinish alone is not a sufficient barrier for this command
    // on that stack. Confirmed by direct repro against real device memory.
    cl_event evt = nullptr;
    check_cl(clEnqueueFillBuffer(q_, buf, &value, sizeof(value), 0, count * sizeof(value), 0, nullptr, &evt),
              "clEnqueueFillBuffer");
    // Release the event before checking/throwing on its wait result (mirrors
    // build()'s release-then-throw of `prog` on a failed clBuildProgram) so a
    // failing clWaitForEvents can't leak `evt` past the throw; release runs
    // exactly once either way.
    cl_int waitErr = clWaitForEvents(1, &evt);
    clReleaseEvent(evt);
    check_cl(waitErr, "clWaitForEvents(fill_u32)");
    check_cl(clFinish(q_), "clFinish");
}
void Runtime::set_arg(cl_kernel k, cl_uint i, size_t size, const void* val) {
    check_cl(clSetKernelArg(k, i, size, val), "clSetKernelArg");
}
void Runtime::run1d(cl_kernel k, size_t global, size_t local) {
    const size_t* lp = (local != 0) ? &local : nullptr;
    check_cl(clEnqueueNDRangeKernel(q_, k, 1, nullptr, &global, lp, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
    check_cl(clFinish(q_), "clFinish");
}

}} // namespace mxbm::gpu
