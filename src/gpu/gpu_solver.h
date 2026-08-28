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
    // `index` selects among the OpenCL devices, flattened across platforms.
    explicit GpuSolver(unsigned index = 0);   // selects device, budget, alloc_pipeline
    static bool available(unsigned index = 0); // Runtime::any_device_available()
    std::vector<std::array<uint8_t,104>> solve(const uint8_t input[32], const uint8_t nonce[8]) override;
    void request_abort() override { abort_.store(true, std::memory_order_relaxed); }
    std::string geometry() const override;
    const DeviceInfo& device() const { return rt_.device(); }
    // What the pipeline was sized against: driver-reported free VRAM less the
    // reserve, or the total-times-headroom fallback when free was unreadable.
    uint64_t usable_vram_bytes() const { return usable_; }
    // Observable so a test can assert a solve was actually SERVED by speculation,
    // rather than merely producing the right answer anyway.
    bool speculation_available() const { return pb_.spec_on; }
    bool last_solve_speculated() const { return spec_hit_; }
private:
    Runtime rt_;
    uint64_t usable_ = 0;
    Budget  budget_;
    PipelineBuffers pb_;
    std::atomic<bool> abort_{false};

    // Speculative entry. The engine walks nonces at a FIXED stride within a job, so
    // consecutive calls teach this solver the delta; each solve seeds (nonce + delta)'s
    // entry pass inside round 4 and the next call skips its own. A job change
    // mis-speculates exactly once. The delta survives job changes -- the stride belongs
    // to the engine, not the job -- so a new job costs one miss, not two.
    bool spec_valid_ = false;      // spec_elem holds entry(spec_nonce_) of spec_input_
    bool spec_hit_ = false;        // the last solve skipped its entry pass
    bool have_last_ = false, have_delta_ = false;
    uint64_t spec_nonce_ = 0, last_nonce_ = 0, learned_delta_ = 0;
    uint8_t spec_input_[32] = {0}, last_input_[32] = {0};
};

}} // namespace mxbm::gpu
