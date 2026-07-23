// Our verifier must agree with Beam's reference on every input.
#include <vector>
#include <cstdint>
#include <random>
#include <cstdio>
#include "beamHashIII.h"          // Beam reference
#include "bh3_verify.h"           // ours
#include "kat_vectors.h"
#include "check.h"
using namespace mxbm;

static bool beam_valid(const uint8_t* input, size_t inLen,
                       const uint8_t nonce[8], const uint8_t soln[104]) {
    BeamHash_III bh;
    blake2b_state base;
    bh.InitialiseState(base);
    blake2b_update(&base, input, inLen);
    blake2b_update(&base, nonce, 8);
    std::vector<uint8_t> s(soln, soln + 104);
    return bh.IsValidSolution(base, s);
}

int main() {
    // 1) Golden solutions: our verdict and Beam's must both accept.
    section("goldens: ours vs Beam's IsValidSolution");
    int golden_agree = 0;
    for (int s = 0; s < 3; ++s) {
        bool a = bh3::is_valid_solution(kat::input32, 32, kat::nonce0, kat::golden[s]);
        bool b = beam_valid(kat::input32, 32, kat::nonce0, kat::golden[s]);
        if (a && b) ++golden_agree;
        if (verbose())
            std::printf("     golden[%d]  ours=%s  beam=%s\n", s,
                        a ? "accept" : "reject", b ? "accept" : "reject");
        check(a && b, "golden: both accept");
    }

    // 2) Fuzz: random headers + random / near-miss blobs; verdicts must agree.
    //    Counted (not per-iteration checked) so verbose mode doesn't emit 20k lines.
    section("fuzz: 20000 inputs, our verdict must equal Beam's");
    std::mt19937_64 rng(0xC0FFEE);
    int total = 0, agree = 0, ours_accept = 0, beam_accept = 0;
    for (int n = 0; n < 20000; ++n) {
        uint8_t input[32], nonce[8], soln[104];
        for (auto& x : input) x = (uint8_t)rng();
        for (auto& x : nonce) x = (uint8_t)rng();
        bool near_miss = (n % 3 == 0);
        if (near_miss) {                           // mutate a golden solution
            const uint8_t* g = kat::golden[(n / 3) % 3];
            for (int i = 0; i < 104; ++i) soln[i] = g[i];
            soln[rng() % 100] ^= (uint8_t)(1u << (rng() % 8));
            for (int i = 0; i < 32; ++i) input[i] = kat::input32[i];
            for (int i = 0; i < 8;  ++i) nonce[i] = kat::nonce0[i];
        } else {
            for (auto& x : soln) x = (uint8_t)rng();
        }
        bool a = bh3::is_valid_solution(input, 32, nonce, soln);
        bool b = beam_valid(input, 32, nonce, soln);
        ++total;
        if (a == b) ++agree;
        if (a) ++ours_accept;
        if (b) ++beam_accept;
        if (verbose() && n < 5)
            std::printf("     fuzz[%d] %-9s ours=%s beam=%s  agree=%s\n", n,
                        near_miss ? "near-miss" : "random",
                        a ? "accept" : "reject", b ? "accept" : "reject",
                        (a == b) ? "yes" : "NO");
        if (a != b) {
            std::printf("  FAIL: fuzz disagreement at n=%d (ours=%d beam=%d)\n", n, a, b);
            break;
        }
    }
    // Summary printed always (this is the headline evidence).
    std::printf("     goldens: %d/3 both-accept;  fuzz: %d/%d verdicts agree "
                "(ours accepted %d, beam accepted %d)\n",
                golden_agree, agree, total, ours_accept, beam_accept);
    check(agree == total, "fuzz: all verdicts agree");
    return summary("differential");
}
