#pragma once
#if defined(__APPLE__)
  #define CL_SILENCE_DEPRECATION
  #include <OpenCL/cl.h>
#else
  #ifndef CL_TARGET_OPENCL_VERSION
    #define CL_TARGET_OPENCL_VERSION 120
  #endif
  #include <CL/cl.h>
#endif
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mxbm { namespace gpu {

struct ClError : std::runtime_error {
    cl_int code;
    ClError(cl_int c, const std::string& what) : std::runtime_error(what), code(c) {}
};

struct DeviceInfo {
    std::string name, version, clc_version;
    cl_ulong global_mem = 0;   // CL_DEVICE_GLOBAL_MEM_SIZE
    cl_ulong max_alloc  = 0;   // CL_DEVICE_MAX_MEM_ALLOC_SIZE
    cl_uint  compute_units = 0;
    cl_ulong local_mem  = 0;     // CL_DEVICE_LOCAL_MEM_SIZE (bytes of __local per workgroup)
    size_t   max_work_group = 0; // CL_DEVICE_MAX_WORK_GROUP_SIZE (work-items per group)
};

// Move-only RAII owner for an OpenCL handle type T released by Rel.
template <class T, cl_int (*Rel)(T)>
class Handle {
public:
    Handle() = default;
    explicit Handle(T h) : h_(h) {}
    ~Handle() { if (h_) Rel(h_); }
    Handle(Handle&& o) noexcept : h_(o.h_) { o.h_ = nullptr; }
    Handle& operator=(Handle&& o) noexcept {
        if (this != &o) { if (h_) Rel(h_); h_ = o.h_; o.h_ = nullptr; }
        return *this;
    }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    T get() const { return h_; }
    explicit operator bool() const { return h_ != nullptr; }
private:
    T h_ = nullptr;
};
using Program = Handle<cl_program, clReleaseProgram>;
using Kernel  = Handle<cl_kernel,  clReleaseKernel>;
using Mem     = Handle<cl_mem,     clReleaseMemObject>;

class Runtime {
public:
    explicit Runtime(unsigned index = 0);
    ~Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    // `index` selects among the devices every platform reports, flattened.
    static bool any_device_available(unsigned index = 0);

    const DeviceInfo& device() const { return info_; }

    Program build(const std::vector<std::string>& sources, const std::string& options);
    // Like build(), but memoizes the compiled program (keyed by sources+options)
    // for this Runtime's lifetime and returns a NON-owning cl_program (the
    // Runtime owns it, released in ~Runtime). clBuildProgram is expensive
    // (hundreds of ms on some drivers, e.g. NVIDIA); the round pipeline calls it
    // many times per solve over the SAME kernel sources, so caching turns dozens
    // of recompiles per search into one. clCreateKernel on the cached program
    // stays cheap and is still done per call.
    cl_program cached_program(const std::vector<std::string>& sources, const std::string& options);
    Kernel  kernel(cl_program prog, const char* name);
    Mem     alloc(cl_mem_flags flags, size_t bytes, void* host_ptr = nullptr);
    void    write(cl_mem buf, size_t bytes, const void* src);
    void    read(cl_mem buf, size_t bytes, void* dst);
    // Blocking read of `bytes` from `offset` into buf (used for the sort path's
    // total-child-count: two 4-byte element reads instead of a full readback).
    void    read_at(cl_mem buf, size_t offset, size_t bytes, void* dst);
    // Fills the first `count` uint32_t elements of buf with `value` via
    // clEnqueueFillBuffer, then blocks (clFinish) until complete.
    void    fill_u32(cl_mem buf, uint32_t value, size_t count);
    void    set_arg(cl_kernel k, cl_uint i, size_t size, const void* val);
    template <class T> void set_arg(cl_kernel k, cl_uint i, const T& v) { set_arg(k, i, sizeof(T), &v); }
    void    run1d(cl_kernel k, size_t global, size_t local = 0);
    // ASYNC variants: enqueue WITHOUT the trailing clFinish. Valid only on the
    // single in-order command queue, where a later dependent op (or an explicit
    // finish()) observes the result -- used by the production row-bucket pipeline to
    // fuse ~15 per-kernel GPU drains into one clFinish before the survivor readback.
    void    fill_u32_async(cl_mem buf, uint32_t value, size_t count);
    void    run1d_async(cl_kernel k, size_t global, size_t local = 0);
    void    finish();

    cl_context context() const { return ctx_; }
    cl_command_queue queue() const { return q_; }
    cl_device_id device_id() const { return dev_; }

private:
    cl_platform_id   plat_ = nullptr;
    cl_device_id     dev_  = nullptr;
    cl_context       ctx_  = nullptr;
    cl_command_queue q_    = nullptr;
    DeviceInfo       info_;
    // Program cache for cached_program(): {key, owned cl_program}. Small (the
    // pipeline uses one source set), a linear scan is fine. Released in ~Runtime.
    std::vector<std::pair<std::string, cl_program>> prog_cache_;

    // Shared compile path for build()/cached_program(): clCreateProgramWithSource
    // + clBuildProgram, throwing ClError with the build log on failure. Returns a
    // raw cl_program the caller takes ownership of.
    cl_program compile(const std::vector<std::string>& sources, const std::string& options);
};

}} // namespace mxbm::gpu
