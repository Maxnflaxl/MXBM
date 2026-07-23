#pragma once
#include "miner/solver.h"
#include "gpu/cl_runtime.h"
#include "gpu/budget.h"
#include "gpu/round_pipeline.h"
#include <atomic>
namespace mxbm { namespace gpu {

// The Phase-C GPU Solver: wraps the full search (mix -> scatter -> match x5
// -> survivor_scan, src/gpu/round_pipeline.{h,cpp}) plus on-device recovery
// (recover_candidates) plus the CPU verify gate (bh3::is_valid_solution,
// src/beamhash/bh3_verify.h -- the M1-proven Equihash validity check the GPU
// rounds deliberately skip: distinct-leaf and indexAfter) behind the
// miner::Solver interface. Holds one Runtime/Budget/PipelineBuffers,
// allocated ONCE in the constructor and reused across every solve() call --
// never reallocated per solve.
class GpuSolver : public miner::Solver {
public:
    GpuSolver();                       // selects device, budget, alloc_pipeline
    static bool available();           // Runtime::any_device_available()
    std::vector<std::array<uint8_t,104>> solve(const uint8_t input[32], const uint8_t nonce[8]) override;
    void request_abort() override { abort_.store(true, std::memory_order_relaxed); }
    const DeviceInfo& device() const { return rt_.device(); }
private:
    Runtime rt_;
    Budget  budget_;
    PipelineBuffers pb_;
    std::atomic<bool> abort_{false};
};

}} // namespace mxbm::gpu
