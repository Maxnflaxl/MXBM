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

namespace mxbm { namespace gpu {

class CudaSolver : public miner::Solver {
public:
    // True if a CUDA device is present AND has room for the full 2^25 seed layer. A
    // reduced layer finds essentially nothing (see budget_can_find_solutions), so a
    // device that cannot host the whole thing must not claim the job.
    static bool available();
    struct DeviceInfo { std::string name; unsigned long long global_mem = 0; unsigned compute_units = 0; };
    static DeviceInfo device_info();

    CudaSolver();                       // throws std::runtime_error on failure
    ~CudaSolver() override;
    CudaSolver(const CudaSolver&) = delete;
    CudaSolver& operator=(const CudaSolver&) = delete;

    std::vector<std::array<uint8_t, 104>> solve(const uint8_t input[32], const uint8_t nonce[8]) override;
    void request_abort() override;

    const DeviceInfo& device() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}} // namespace mxbm::gpu
