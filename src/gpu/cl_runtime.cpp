#include "gpu/cl_runtime.h"
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

namespace mxbm { namespace gpu {
namespace {

// The `index`-th device across every platform; prefer GPUs, else any type.
//
// Indexing runs over the FLATTENED list -- platform 0's devices, then platform
// 1's -- because that is the only ordering available before a device handle
// exists to ask about. It matches CUDA's PCI-sorted index only on a
// single-vendor rig; on a mixed one the two backends can disagree, which is why
// --list-devices prints what each backend actually found rather than promising
// one numbering for both.
bool pick_device(cl_platform_id* plat_out, cl_device_id* dev_out, unsigned index) {
    *dev_out = nullptr;
    cl_uint nplat = 0;
    if (clGetPlatformIDs(0, nullptr, &nplat) != CL_SUCCESS || nplat == 0) return false;
    std::vector<cl_platform_id> plats(nplat);
    if (clGetPlatformIDs(nplat, plats.data(), nullptr) != CL_SUCCESS) return false;
    for (cl_device_type type : {(cl_device_type)CL_DEVICE_TYPE_GPU, (cl_device_type)CL_DEVICE_TYPE_ALL}) {
        unsigned seen = 0;
        for (cl_platform_id p : plats) {
            cl_uint ndev = 0;
            if (clGetDeviceIDs(p, type, 0, nullptr, &ndev) != CL_SUCCESS || ndev == 0) continue;
            std::vector<cl_device_id> devs(ndev);
            if (clGetDeviceIDs(p, type, ndev, devs.data(), nullptr) != CL_SUCCESS) continue;
            if (index < seen + ndev) {
                *dev_out = devs[index - seen];
                *plat_out = p;
                return true;
            }
            seen += ndev;
        }
        // A GPU pass that found devices but not ENOUGH of them falls through to
        // the ALL pass, which is the same widening the original did -- an index
        // past the end of both is simply not available.
    }
    return false;
}

// cl_nv_device_attribute_query, spelled out rather than included: its header ships
// with NVIDIA's SDK and this file must build against any OpenCL 1.2 header set.
// A platform that does not know these values just fails the query.
#ifndef CL_DEVICE_PCI_BUS_ID_NV
  #define CL_DEVICE_PCI_BUS_ID_NV  0x4008
#endif
#ifndef CL_DEVICE_PCI_SLOT_ID_NV
  #define CL_DEVICE_PCI_SLOT_ID_NV 0x4009
#endif

// "bus:device" in CUDA's and NVML's hex short form, or "" when the device does not
// answer. The join keys on the BUS alone: NVIDIA documents SLOT_ID without pinning
// its encoding (the device number on some drivers, device << 3 | function on
// others), while a bus is unique per card without that assumption.
std::string pci_of(cl_device_id d) {
    cl_uint bus = 0, slot = 0;
    if (clGetDeviceInfo(d, CL_DEVICE_PCI_BUS_ID_NV, sizeof bus, &bus, nullptr) != CL_SUCCESS)
        return {};
    if (clGetDeviceInfo(d, CL_DEVICE_PCI_SLOT_ID_NV, sizeof slot, &slot, nullptr) != CL_SUCCESS)
        slot = 0;
    char buf[32];
    std::snprintf(buf, sizeof buf, "%x:%x", (unsigned)bus, (unsigned)slot);
    return buf;
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

bool Runtime::any_device_available(unsigned index) {
    cl_platform_id p; cl_device_id d;
    return pick_device(&p, &d, index);
}

std::vector<DeviceInfo> Runtime::enumerate() {
    std::vector<DeviceInfo> out;
    cl_uint nplat = 0;
    if (clGetPlatformIDs(0, nullptr, &nplat) != CL_SUCCESS || nplat == 0) return out;
    std::vector<cl_platform_id> plats(nplat);
    if (clGetPlatformIDs(nplat, plats.data(), nullptr) != CL_SUCCESS) return out;
    // The GPU pass only, in platform order -- the same walk pick_device does
    // first, so entry N here is the device Runtime(N) selects.
    for (cl_platform_id p : plats) {
        cl_uint ndev = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &ndev) != CL_SUCCESS || ndev == 0)
            continue;
        std::vector<cl_device_id> devs(ndev);
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, ndev, devs.data(), nullptr) != CL_SUCCESS)
            continue;
        for (cl_device_id d : devs) {
            DeviceInfo i;
            i.index = (unsigned)out.size();
            i.name   = str_info(d, CL_DEVICE_NAME);
            i.vendor = str_info(d, CL_DEVICE_VENDOR);
            i.pci   = pci_of(d);
            clGetDeviceInfo(d, CL_DEVICE_GLOBAL_MEM_SIZE,    sizeof i.global_mem, &i.global_mem, nullptr);
            clGetDeviceInfo(d, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof i.max_alloc,  &i.max_alloc,  nullptr);
            clGetDeviceInfo(d, CL_DEVICE_MAX_COMPUTE_UNITS,  sizeof i.compute_units, &i.compute_units, nullptr);
            out.push_back(std::move(i));
        }
    }
    return out;
}

Runtime::Runtime(unsigned index) {
    if (!pick_device(&plat_, &dev_, index))
        throw ClError(CL_DEVICE_NOT_FOUND, "no OpenCL device at index "
                      + std::to_string(index));
    info_.index       = index;
    info_.pci         = pci_of(dev_);
    info_.name        = str_info(dev_, CL_DEVICE_NAME);
    info_.version     = str_info(dev_, CL_DEVICE_VERSION);
    info_.clc_version = str_info(dev_, CL_DEVICE_OPENCL_C_VERSION);
    clGetDeviceInfo(dev_, CL_DEVICE_GLOBAL_MEM_SIZE,     sizeof info_.global_mem,    &info_.global_mem,    nullptr);
    clGetDeviceInfo(dev_, CL_DEVICE_MAX_MEM_ALLOC_SIZE,  sizeof info_.max_alloc,     &info_.max_alloc,     nullptr);
    clGetDeviceInfo(dev_, CL_DEVICE_MAX_COMPUTE_UNITS,   sizeof info_.compute_units, &info_.compute_units, nullptr);
    clGetDeviceInfo(dev_, CL_DEVICE_LOCAL_MEM_SIZE,      sizeof info_.local_mem,     &info_.local_mem,     nullptr);
    clGetDeviceInfo(dev_, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof info_.max_work_group,&info_.max_work_group,nullptr);
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
    for (auto& e : prog_cache_) if (e.second) clReleaseProgram(e.second);
    if (q_)   clReleaseCommandQueue(q_);
    if (ctx_) clReleaseContext(ctx_);
}

cl_program Runtime::compile(const std::vector<std::string>& sources, const std::string& options) {
    std::vector<const char*> ptrs;
    std::vector<size_t> lens;
    ptrs.reserve(sources.size());
    lens.reserve(sources.size());
    for (const auto& s : sources) { ptrs.push_back(s.c_str()); lens.push_back(s.size()); }
    cl_int err = CL_SUCCESS;
    cl_program prog = clCreateProgramWithSource(ctx_, (cl_uint)ptrs.size(), ptrs.data(), lens.data(), &err);
    check_cl(err, "clCreateProgramWithSource");
    // MXBM_CL_VERBOSE: ask NVIDIA's OpenCL compiler for the ptxas report and print it.
    // This is the ONLY register/spill instrument available on this path -- Nsight cannot
    // profile OpenCL -- and spilling is the one mechanism with a demonstrated 3x behind
    // it here (the round-4 mix anomaly: a private ulong[8] in scratch, 27 -> 8 ms).
    // The report names, per kernel: registers used, "bytes stack frame" (the scratch
    // allocation) and "bytes spill stores/loads".
    std::string opts = options;
    const bool verbose = std::getenv("MXBM_CL_VERBOSE") != nullptr;
    if (verbose) opts += " -cl-nv-verbose";
    // MXBM_CL_MAXREG=N caps registers program-wide. Diagnostic only: it is how you test
    // whether a kernel's register count -- and so its occupancy -- is what is costing it,
    // by forcing a fast kernel DOWN to a slow one's count and watching it slow down.
    if (const char* r = std::getenv("MXBM_CL_MAXREG")) {
        opts += " -cl-nv-maxrregcount=";
        opts += r;
    }
    err = clBuildProgram(prog, 1, &dev_, opts.c_str(), nullptr, nullptr);
    if (verbose && err == CL_SUCCESS) {
        size_t n = 0;
        clGetProgramBuildInfo(prog, dev_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
        std::string log(n, '\0');
        clGetProgramBuildInfo(prog, dev_, CL_PROGRAM_BUILD_LOG, n, &log[0], nullptr);
        std::fprintf(stderr, "---- ptxas report ----\n%s\n----------------------\n", log.c_str());
    }
    if (err != CL_SUCCESS) {
        size_t n = 0;
        clGetProgramBuildInfo(prog, dev_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
        std::string log(n, '\0');
        clGetProgramBuildInfo(prog, dev_, CL_PROGRAM_BUILD_LOG, n, &log[0], nullptr);
        clReleaseProgram(prog);
        throw ClError(err, "clBuildProgram failed:\n" + log);
    }
    return prog;
}

Program Runtime::build(const std::vector<std::string>& sources, const std::string& options) {
    return Program(compile(sources, options));
}

cl_program Runtime::cached_program(const std::vector<std::string>& sources, const std::string& options) {
    // Key = every source concatenated (with a separator that can't appear
    // mid-source in a way that would alias two different lists) + options.
    std::string key;
    for (const auto& s : sources) { key += s; key.push_back('\x01'); }
    key += '\x02';
    key += options;
    for (auto& e : prog_cache_) if (e.first == key) return e.second;
    // Cache miss: compile once (the expensive step) and time it, so the
    // one-off cost is visible -- on some drivers (NVIDIA) it is hundreds of ms
    // and was previously being paid on every kernel launch of every solve.
    auto t0 = std::chrono::steady_clock::now();
    cl_program prog = compile(sources, options);   // owned by the cache
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("  [gpu] compiled OpenCL program in %.0f ms (cached for reuse across all solves)\n", ms);
    prog_cache_.emplace_back(std::move(key), prog);
    return prog;
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
void Runtime::read_at(cl_mem buf, size_t offset, size_t bytes, void* dst) {
    check_cl(clEnqueueReadBuffer(q_, buf, CL_TRUE, offset, bytes, dst, 0, nullptr, nullptr), "clEnqueueReadBuffer");
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
void Runtime::finish() { check_cl(clFinish(q_), "clFinish"); }
// Enqueue-only fill: no event wait / clFinish. Ordering is guaranteed by the
// in-order queue -- a later dependent kernel (or an explicit finish()) sees it.
void Runtime::fill_u32_async(cl_mem buf, uint32_t value, size_t count) {
    check_cl(clEnqueueFillBuffer(q_, buf, &value, sizeof(value), 0, count * sizeof(value), 0, nullptr, nullptr),
             "clEnqueueFillBuffer(async)");
}
void Runtime::run1d_async(cl_kernel k, size_t global, size_t local) {
    const size_t* lp = (local != 0) ? &local : nullptr;
    check_cl(clEnqueueNDRangeKernel(q_, k, 1, nullptr, &global, lp, 0, nullptr, nullptr), "clEnqueueNDRangeKernel(async)");
}
void Runtime::run1d(cl_kernel k, size_t global, size_t local) {
    const size_t* lp = (local != 0) ? &local : nullptr;
    check_cl(clEnqueueNDRangeKernel(q_, k, 1, nullptr, &global, lp, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
    check_cl(clFinish(q_), "clFinish");
}

}} // namespace mxbm::gpu
