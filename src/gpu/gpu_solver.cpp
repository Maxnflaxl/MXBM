#include "gpu/gpu_solver.h"
#include "beamhash/bh3_blake2b.h"
#include "beamhash/bh3_verify.h"

namespace mxbm { namespace gpu {

GpuSolver::GpuSolver() {
    // rt_ is already constructed (device selected, context/queue created --
    // Runtime's own default constructor, run implicitly before this body)
    // by the time we get here; throws ClError if no device is present, so
    // callers must guard with available() first, exactly as every device
    // test in this suite guards with Runtime::any_device_available().
    budget_ = compute_budget(rt_.device().global_mem, rt_.device().max_alloc);
    pb_ = alloc_pipeline(rt_, budget_);
}

bool GpuSolver::available() { return Runtime::any_device_available(); }

std::vector<std::array<uint8_t, 104>> GpuSolver::solve(const uint8_t input[32], const uint8_t nonce[8]) {
    // Clear any abort left set by a prior call's request_abort() -- each
    // solve() starts clean, exactly like run_pipeline's own doc contract.
    abort_.store(false, std::memory_order_relaxed);

    // extraNonce is fixed at 0 (matches the goldens, which all end in
    // 0x00000000, and matches SolverRef -- see the Oracle's InitialiseState
    // call pattern, which never varies extraNonce either).
    const uint8_t extra0[4] = {0, 0, 0, 0};
    uint64_t prePow[4];
    bh3::compute_prepow(input, 32, nonce, extra0, prePow);

    // The full 5-round search, reusing this solver's PERSISTENT pb_/budget_
    // (allocated once in the constructor, never reallocated here) -- and
    // abortable via &abort_, polled by run_pipeline between rounds.
    PipelineResult res = run_pipeline(rt_, pb_, budget_, prePow, &abort_);
    if (res.survivors == 0) return {};

    // Raw candidates: on-device back-ref recovery + host-side pack_indices
    // (C-T1, unmodified) -- NOT yet gated; the rounds only enforced the
    // 24-bit collision, so some of these may fail full Equihash validity.
    std::vector<std::array<uint8_t, 104>> cand = recover_candidates(rt_, pb_, res.survivor_slots);

    // THE CPU VERIFY GATE (permanent anti-cheat): only candidates that pass
    // bh3::is_valid_solution -- which recomputes prePow from the candidate's
    // own extraNonce and enforces the FULL Equihash validity, including the
    // distinct-leaf and indexAfter checks the GPU rounds deliberately skip
    // -- are ever returned. No candidate is exempted or returned unchecked.
    std::vector<std::array<uint8_t, 104>> out;
    out.reserve(cand.size());
    for (const auto& c : cand)
        if (bh3::is_valid_solution(input, 32, nonce, c.data()))
            out.push_back(c);
    return out;
}

}} // namespace mxbm::gpu
