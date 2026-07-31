#pragma once
// CUDA backend for the BeamHash III solver. Same contract as GpuSolver (OpenCL): buffers
// are allocated once in the constructor and reused across every solve(), and solve()
// returns only CPU-VERIFIED solutions.
//
// No CUDA types appear here -- the implementation is behind a pimpl so main.cpp and the
// engine can be compiled by the host compiler without nvcc.
#include "miner/solver.h"
#include <atomic>
#include <memory>
#include <string>
#include <vector>

namespace mxbm { namespace gpu {

class CudaSolver : public miner::Solver {
public:
    // True if a CUDA device is present AND has room for the full 2^25 seed layer. A
    // reduced layer finds essentially nothing (see budget_can_find_solutions), so a
    // device that cannot host the whole thing must not claim the job.
    static bool available(int index = 0);
    struct DeviceInfo {
        std::string name; unsigned long long global_mem = 0; unsigned compute_units = 0;
        // PCI bus:device, the same short form NVML reports. This is what makes an
        // index mean the same card across CUDA, OpenCL and NVML: CUDA's own
        // ordering is "fastest first" by default while NVML's is by bus id, so
        // two backends' index N are NOT the same card on a mixed rig. Ordering
        // by PCI address is the only key all three agree on.
        std::string pci;
        int  index = 0;                 // this backend's own index for the card
        bool viable = false;            // has room for the full 2^25 seed layer
    };
    static DeviceInfo device_info(int index = 0);
    // Every CUDA device the driver reports, in PCI order.
    static std::vector<DeviceInfo> enumerate();

    // `index` is a CUDA device index. Throws std::runtime_error on failure.
    // `power_limit_w` is the board power limit observed at startup (0 = unknown or
    // uncapped) -- the hint the geometry policy reads. The policy is currently
    // DISARMED (kRbLowPowerW == 0 in rowbucket_geom.h; its comment tells the story),
    // so every value is inert; the plumbing stays because selection, when armed,
    // happens once, here -- a later cap change never re-selects (multi-GiB realloc).
    explicit CudaSolver(int index = 0, unsigned power_limit_w = 0);
    ~CudaSolver() override;
    CudaSolver(const CudaSolver&) = delete;
    CudaSolver& operator=(const CudaSolver&) = delete;

    std::vector<std::array<uint8_t, 104>> solve(const uint8_t input[32], const uint8_t nonce[8]) override;
    void request_abort() override;

    const DeviceInfo& device() const;

    // The chosen bucket bits: bb of the (bb, 17-bb) geometry actually allocated,
    // AFTER any allocator step-down. What the power-limit preference selects --
    // exposed so tests can assert the hint reached the solver, not just the picker.
    unsigned bucket_bits() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}} // namespace mxbm::gpu
