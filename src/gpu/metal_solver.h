#pragma once
// Metal backend for the BeamHash III solver. Same contract as CudaSolver: buffers are
// allocated once in the constructor and reused across every solve(), and solve() returns
// only CPU-VERIFIED solutions.
//
// No Metal or Objective-C types appear here -- the implementation is behind a pimpl so
// main.cpp and the engine keep compiling as plain C++ with no Objective-C++ toolchain,
// exactly as cuda_solver.h keeps them free of nvcc.
#include "miner/solver.h"
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mxbm { namespace gpu {

class MetalSolver : public miner::Solver {
public:
    // True if a Metal device is present AND has room for the full 2^25 seed layer. A
    // reduced layer finds essentially nothing (see budget_can_find_solutions), so a
    // device that cannot host the whole thing must not claim the job. On Apple Silicon
    // this is effectively always true -- unified memory and an 80 GB max buffer length
    // mean the layer is never the binding constraint -- but the check stays, because a
    // backend that skips it is a backend that mines nothing and says nothing.
    static bool available(int index = 0);

    struct DeviceInfo {
        std::string name;
        unsigned long long global_mem = 0;   // recommendedMaxWorkingSetSize, not
                                             // hw.memsize: on unified memory the latter
                                             // is a number the GPU cannot actually have.
        unsigned compute_units = 0;          // GPU core count via IOKit; 0 when unknown
        // No `pci` field, unlike CudaSolver::DeviceInfo. That exists to make an index
        // mean the same card across CUDA, OpenCL and NVML on a mixed rig; Apple Silicon
        // has one integrated GPU and no PCI address, so the problem does not arise.
        int  index  = 0;
        bool viable = false;                 // has room for the full 2^25 seed layer
    };
    static DeviceInfo device_info(int index = 0);
    // Every Metal device the system reports, in MTLCopyAllDevices order.
    static std::vector<DeviceInfo> enumerate();

    // `index` is a Metal device index. Throws std::runtime_error on failure.
    explicit MetalSolver(int index = 0);
    ~MetalSolver() override;
    MetalSolver(const MetalSolver&) = delete;
    MetalSolver& operator=(const MetalSolver&) = delete;

    std::vector<std::array<uint8_t, 104>> solve(const uint8_t input[32], const uint8_t nonce[8]) override;
    void request_abort() override;

    const DeviceInfo& device() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}} // namespace mxbm::gpu
