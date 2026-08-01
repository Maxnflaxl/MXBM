// CUDA backend driven purely through the miner::Solver interface, exactly as the engine
// drives it. Same contract as test_gpu_solver: the KAT input has exactly 3 golden
// solutions, and solve() must return those 3, verified, byte-for-byte.
#include "gpu/cuda_solver.h"
#include "kat_vectors.h"
#include "check.h"
#include "beamhash/bh3_verify.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifdef _WIN32
// MSVC has no setenv/unsetenv; _putenv_s covers both (empty value = remove).
static int setenv(const char* k, const char* v, int) { return _putenv_s(k, v); }
static int unsetenv(const char* k) { return _putenv_s(k, ""); }
#endif

using namespace mxbm;

static void solve_the_kat(gpu::CudaSolver& s, const char* geom) {
    for (int pass = 1; pass <= 2; ++pass) {
        // Pass 2 reuses the same solver: the engine calls solve() back to back over
        // incrementing nonces, so persistent-buffer reuse has to be clean or the miner
        // would produce nothing from the second nonce onward.
        auto sols = s.solve(kat::input32, kat::nonce0);
        char lbl[96]; std::snprintf(lbl, sizeof lbl, "%s pass %d: solve() returns EXACTLY 3 solutions", geom, pass);
        check(sols.size() == 3, lbl);
        for (size_t i = 0; i < sols.size(); ++i) {
            std::snprintf(lbl, sizeof lbl, "%s pass %d: solution %zu re-verifies", geom, pass, i);
            check(bh3::is_valid_solution(kat::input32, 32, kat::nonce0, sols[i].data()), lbl);
        }
        bool matched[3] = {false,false,false};
        for (const auto& sol : sols)
            for (int g = 0; g < 3; ++g)
                if (!matched[g] && std::memcmp(sol.data(), kat::golden[g], 104) == 0) { matched[g]=true; break; }
        std::snprintf(lbl, sizeof lbl, "%s pass %d: all 3 KAT goldens matched byte-for-byte", geom, pass);
        check(matched[0] && matched[1] && matched[2], lbl);
    }
}

int main() {
    if (!gpu::CudaSolver::available()) { std::printf("SKIP: no suitable CUDA device\n"); return 0; }
    section("CudaSolver::solve(): full pipeline + recovery + CPU gate -> the 3 KAT goldens");

    // Scoped: a solver holds ~7 GiB, so two of them do not fit on the card this runs on.
    {
        gpu::CudaSolver s;
        std::printf("  device: %s (%.1f GiB, %u SMs)\n", s.device().name.c_str(),
                    s.device().global_mem / 1073741824.0, s.device().compute_units);
        solve_the_kat(s, "auto");

        // A different nonce must produce a different (and self-consistent) result, which
        // is what the miner does. Solution count varies by nonce; correctness does not.
        uint8_t nonce[8]; std::memcpy(nonce, kat::nonce0, 8); nonce[7] ^= 0x5A;
        auto other = s.solve(kat::input32, nonce);
        bool allValid = true;
        for (const auto& sol : other)
            if (!bh3::is_valid_solution(kat::input32, 32, nonce, sol.data())) allValid = false;
        std::printf("  distinct nonce: %zu solution(s), all verified\n", other.size());
        check(allValid, "every solution from a different nonce also verifies");
    }

    // Every rung of the ladder, through the SHIPPING solver. A card too small for (16,1)
    // runs one of the others, so a coarser geometry has to produce the same goldens, not
    // merely allocate. On a 16 GB card auto is always (16,1) -- which is exactly why the
    // other two would otherwise ship untested. bb = 17 is off-ladder (users reach it by
    // MXBM_BB=17 while the power-cap policy stays disarmed) and keeps its kernels covered.
    for (uint32_t bb = 17; bb >= 14; --bb) {
        char v[8], geom[32];
        std::snprintf(v, sizeof v, "%u", bb);
        std::snprintf(geom, sizeof geom, "(%u,%u)", bb, 17u - bb);
        setenv("MXBM_BB", v, 1);
        // MXBM_BB is an explicit override, so the solver does NOT step down under it --
        // which is right for a user who asked for a geometry, and means a rung can
        // legitimately not fit on a busy card. Not fitting is not a correctness failure;
        // producing the wrong answer would be.
        try {
            gpu::CudaSolver g;
            solve_the_kat(g, geom);
        } catch (const std::exception& e) {
            std::printf("  SKIP %s: %s\n", geom, e.what());
        }
    }
    unsetenv("MXBM_BB");

    // The power-limit hint through the ctor -- exactly what main.cpp passes after
    // observing the board limit. The policy is DISARMED (kRbLowPowerW == 0; the
    // low-band prize failed reproduction -- rowbucket_geom.h tells the story), so the
    // plumbing must be live and the value must change NOTHING.
    {
        gpu::CudaSolver g(0, /*power_limit_w=*/160);
        check(g.bucket_bits() == 16u, "disarmed: a 160 W hint is inert, (16,1) as always");
    }

    return summary("cuda_solver");
}
