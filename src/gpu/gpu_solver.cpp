#include "gpu/gpu_solver.h"
#include "beamhash/bh3_blake2b.h"
#include "beamhash/bh3_verify.h"
#include "gpu/nvml.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace mxbm { namespace gpu {

namespace {

// Free VRAM for the card this OpenCL device is, matched through NVML on the PCI
// address -- OpenCL's own index is not NVML's on a mixed rig. Returns 0 when the
// address is unknown or NVML has nothing to say, which sizes off the total.
uint64_t free_vram_for(const std::string& pci, bool& display_attached) {
    display_attached = false;
    if (pci.empty()) return 0;
    for (unsigned i = 0, n = nvml_device_count(); i < n; ++i) {
        if (nvml_pci_address(i) != pci) continue;
        uint64_t f = 0, t = 0;
        display_attached = nvml_display_active(i);
        return nvml_memory_info(i, f, t) ? f : 0;
    }
    return 0;
}

} // namespace

GpuSolver::GpuSolver(unsigned index) : rt_(index) {
    // rt_ is already constructed (device selected, context/queue created --
    // Runtime's constructor, run before this body)
    // by the time we get here; throws ClError if no device is present, so
    // callers must guard with available() first, exactly as every device
    // test in this suite guards with Runtime::any_device_available().
    // Size the seed layer for the path that will actually run. The row-bucket path
    // costs 239 B/element against the sort path's 264, so budgeting both at the sort
    // figure needlessly cuts down cards that could host a full 2^25 layer on the fast
    // path -- and a reduced layer mines nothing (budget_can_find_solutions). Try the
    // optimistic budget first; if the device cannot host the row-bucket footprint at
    // that capacity, fall back to the sort-path budget, which is what will run.
    bool display = false;
    const uint64_t free_b = free_vram_for(rt_.device().pci, display);
    usable_ = usable_vram(free_b, rt_.device().global_mem, reserve_bytes(display));
    budget_ = compute_budget(usable_, rt_.device().max_alloc, 1.0, kBytesPerElementRowbucket);
    // The flat divisor is the full-fat (16,1) figure; a card it refuses may still
    // host the FULL 2^25 layer on a coarser/quad rung with split record sets. The
    // ladder, not the divisor, is the row-bucket authority -- ask it before refusing.
    if (!budget_can_find_solutions(budget_.elems_per_round)) {
        Budget full = budget_full_rowbucket(usable_, rt_.device().max_alloc);
        if (rowbucket_viable(rt_, full)) budget_ = full;
    }
    if (!rowbucket_viable(rt_, budget_))
        budget_ = compute_budget(usable_, rt_.device().max_alloc, 1.0, kBytesPerElementSort);

    // REFUSE a partial seed layer rather than mine nothing. A BeamHash III solution
    // is 32 indices from the full [0, 2^25) space, so a device that can only host
    // elems_per_round < 2^25 can only find solutions whose every index happens to
    // fall below that -- probability (epr/2^25)^32, i.e. 2e-10 at half a layer. Such
    // a miner runs, reports a hashrate, and finds nothing; failing loudly here is the
    // honest behaviour. MXBM_ALLOW_PARTIAL_SEARCH overrides for experiments.
    if (!budget_can_find_solutions(budget_.elems_per_round) &&
        std::getenv("MXBM_ALLOW_PARTIAL_SEARCH") == nullptr) {
        // Quote the LADDER's floor, not the sort path's flat divisor. The divisor is
        // what this budget was sized by once the ladder refused, and it is ~4.7x the
        // real requirement -- a figure that would tell a 3 GB owner to give up.
        size_t floor_b = 0, ignore = 0;
        { int n = 0; const RbRung* r = rb_rungs(n);
          rowbucket_bytes(kRbCapacity, r[n-1].bb, floor_b, ignore,
                          r[n-1].quad, false, r[n-1].arena, r[n-1].octo); }
        char msg[352];
        std::snprintf(msg, sizeof msg,
            "GPU has too little memory for BeamHash III: it can host only %u of the "
            "required %u seed elements. A partial seed layer cannot find solutions "
            "(probability ~(%.2f)^32). The coarsest geometry needs ~%.2f GiB of usable "
            "VRAM; this device offers %.2f GiB after the reserve (see --keepfree). Set "
            "MXBM_ALLOW_PARTIAL_SEARCH=1 to override for testing.",
            budget_.elems_per_round, 1u << kTargetElemsLog2,
            (double)budget_.elems_per_round / (double)(1u << kTargetElemsLog2),
            (double)floor_b / 1073741824.0,
            (double)usable_ / 1073741824.0);
        throw ClError(CL_OUT_OF_RESOURCES, msg);
    }

    pb_ = alloc_pipeline(rt_, budget_);
}

bool GpuSolver::available(unsigned index) { return Runtime::any_device_available(index); }

std::vector<std::array<uint8_t, 104>> GpuSolver::solve(const uint8_t input[32], const uint8_t nonce[8]) {
    {   // MXBM_VERIFY_STATS: candidates/solve needs the solve count as denominator.
        static const bool kVerifyStatsSolve = std::getenv("MXBM_VERIFY_STATS") != nullptr;
        if (kVerifyStatsSolve) bh3::verify_stats_note_solve();
    }
    // Opt-in per-solve profiling: run with MXBM_PROFILE=1 to print a one-line
    // GPU-time breakdown (seed | per-round mix/scatter/match | survivor |
    // recover | verify) every solve, plus the derived solve/s next to
    // the reference figure. Off by default (read once) so production mining
    // stays quiet.
    static const bool profile = (std::getenv("MXBM_PROFILE") != nullptr);
    auto tSolve = std::chrono::steady_clock::now();

    // Clear any abort left set by a prior call's request_abort() -- each
    // solve() starts clean, exactly like run_pipeline's own doc contract.
    abort_.store(false, std::memory_order_relaxed);

    // extraNonce is fixed at 0 (matches the goldens, which all end in
    // 0x00000000, and matches SolverRef -- see the Oracle's InitialiseState
    // call pattern, which never varies extraNonce either).
    const uint8_t extra0[4] = {0, 0, 0, 0};
    uint64_t prePow[4];
    bh3::compute_prepow(input, 32, nonce, extra0, prePow);

    // Speculative entry: learn the engine's nonce stride, then tell the pipeline whether
    // last solve's round 4 already produced THIS solve's entry output, and which prePow
    // to seed for the next. See GpuSolver's members and PipelineBuffers::spec_elem.
    SpecEntry spec;
    uint64_t nonceLE = 0; std::memcpy(&nonceLE, nonce, 8);
    const bool sameJob = have_last_ && std::memcmp(last_input_, input, 32) == 0;
    if (sameJob) { learned_delta_ = nonceLE - last_nonce_; have_delta_ = true; }
    have_last_ = true; last_nonce_ = nonceLE; std::memcpy(last_input_, input, 32);
    spec.hit = spec_valid_ && spec_nonce_ == nonceLE
            && std::memcmp(spec_input_, input, 32) == 0;
    spec_hit_ = spec.hit && pb_.spec_on;
    spec.seed_next = have_delta_;
    spec_valid_ = false;
    if (spec.seed_next) {
        const uint64_t nextLE = nonceLE + learned_delta_;
        uint8_t nextNonce[8]; std::memcpy(nextNonce, &nextLE, 8);
        bh3::compute_prepow(input, 32, nextNonce, extra0, spec.next_pp);
    }

    // The full 5-round search, reusing this solver's PERSISTENT pb_/budget_
    // (allocated once in the constructor, never reallocated here) -- and
    // abortable via &abort_, polled by run_pipeline between rounds.
    // verbose=false: continuous production mining calls solve() once per
    // nonce attempt, so the per-round printf (default-on for the direct
    // pipeline tests) would spam stdout every round of every attempt.
    PipelineResult res = run_pipeline(rt_, pb_, budget_, prePow, &abort_, /*verbose=*/false, &spec);
    if (spec.seeded) {                     // round 4 seeded (nonce + delta) for next time
        spec_valid_ = true;
        spec_nonce_ = nonceLE + learned_delta_;
        std::memcpy(spec_input_, input, 32);
    }

    // THE CPU VERIFY GATE (permanent anti-cheat): only candidates that pass
    // bh3::is_valid_solution -- which recomputes prePow from the candidate's
    // own extraNonce and enforces the FULL Equihash validity, including the
    // distinct-leaf and indexAfter checks the GPU rounds deliberately skip
    // -- are ever returned. No candidate is exempted or returned unchecked.
    // (The common per-nonce case is 0 survivors -> empty `out`, same as the
    // old early return; recover/verify only run when there's something to gate.)
    std::vector<std::array<uint8_t, 104>> out;
    double tRecoverMs = 0.0, tVerifyMs = 0.0;
    uint32_t nCand = 0;
    if (res.survivors != 0) {
        // Raw candidates: on-device back-ref recovery + host-side pack_indices
        // (C-T1, unmodified) -- NOT yet gated; the rounds only enforced the
        // 24-bit collision, so some of these may fail full Equihash validity.
        auto tRec = std::chrono::steady_clock::now();
        std::vector<std::array<uint8_t, 104>> cand = recover_candidates(rt_, pb_, res.survivor_slots);
        tRecoverMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tRec).count();
        nCand = (uint32_t)cand.size();

        auto tVer = std::chrono::steady_clock::now();
        out.reserve(cand.size());
        // MXBM_VERIFY_STATS=1: classify each candidate's verify outcome so the
        // found-vs-verified gap is attributed by reject reason (printed at exit).
        static const bool kVerifyStats = std::getenv("MXBM_VERIFY_STATS") != nullptr;
        for (const auto& c : cand) {
            if (!kVerifyStats) {
                if (bh3::is_valid_solution(input, 32, nonce, c.data()))
                    out.push_back(c);
            } else {
                const bh3::VerifyReason r = bh3::classify_solution(input, 32, nonce, c.data());
                bh3::verify_stats_tally(r);
                if (r.kind == bh3::VerifyReason::None) out.push_back(c);
            }
        }
        tVerifyMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tVer).count();
    }

    if (profile) {
        double total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tSolve).count();
        double sps = total > 0.0 ? 1000.0 / total : 0.0;
        // spec: did this solve's entry pass come free from the previous solve's round 4.
        // "off" = buffers did not fit, or MXBM_NO_SPEC; a steady run of "miss" means the
        // nonce stride is not being predicted.
        std::printf("[prof] total=%.1fms  %.2f solve/s (~%.1f sol/s | ref ~53) | spec=%s seed=%.1f",
                    total, sps, sps * 1.9,
                    !pb_.spec_on ? "off" : spec.hit ? "hit" : "miss", res.t_seed_ms);
        for (int r = 0; r < 5; ++r) {
            const RoundStats& st = res.rounds[r];
            std::printf(" r%d=%.1f[mx%.1f sc%.1f mt%.1f]", r + 1,
                        st.t_mix_ms + st.t_scatter_ms + st.t_match_ms,
                        st.t_mix_ms, st.t_scatter_ms, st.t_match_ms);
        }
        std::printf(" | surv=%.1f rec=%.1f ver=%.1f | survivors=%u cand=%u sols=%zu\n",
                    res.t_survivor_ms, tRecoverMs, tVerifyMs, res.survivors, nCand, out.size());
    }
    return out;
}

}} // namespace mxbm::gpu
