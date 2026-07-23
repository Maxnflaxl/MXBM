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
#include <stdexcept>
#include <string>
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
    Runtime();
    ~Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    static bool any_device_available();

    const DeviceInfo& device() const { return info_; }

    Program build(const std::vector<std::string>& sources, const std::string& options);
    Kernel  kernel(cl_program prog, const char* name);
    Mem     alloc(cl_mem_flags flags, size_t bytes, void* host_ptr = nullptr);
    void    write(cl_mem buf, size_t bytes, const void* src);
    void    read(cl_mem buf, size_t bytes, void* dst);
    void    set_arg(cl_kernel k, cl_uint i, size_t size, const void* val);
    template <class T> void set_arg(cl_kernel k, cl_uint i, const T& v) { set_arg(k, i, sizeof(T), &v); }
    void    run1d(cl_kernel k, size_t global, size_t local = 0);

    cl_context context() const { return ctx_; }
    cl_command_queue queue() const { return q_; }
    cl_device_id device_id() const { return dev_; }

private:
    cl_platform_id   plat_ = nullptr;
    cl_device_id     dev_  = nullptr;
    cl_context       ctx_  = nullptr;
    cl_command_queue q_    = nullptr;
    DeviceInfo       info_;
};

}} // namespace mxbm::gpu
