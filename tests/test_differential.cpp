// Our verifier must agree with Beam's reference on every input.
#include <vector>
#include <cstdint>
#include <random>
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
    // 1) Golden solutions: both must accept.
    for (int s = 0; s < 3; ++s) {
        bool a = bh3::is_valid_solution(kat::input32, 32, kat::nonce0, kat::golden[s]);
        bool b = beam_valid(kat::input32, 32, kat::nonce0, kat::golden[s]);
        check(a && b, "golden: both accept");
    }
    // 2) Fuzz: random headers + random / near-miss blobs; verdicts must agree.
    std::mt19937_64 rng(0xC0FFEE);
    for (int n = 0; n < 20000; ++n) {
        uint8_t input[32], nonce[8], soln[104];
        for (auto& x : input) x = (uint8_t)rng();
        for (auto& x : nonce) x = (uint8_t)rng();
        if (n % 3 == 0) {                          // near-miss: mutate a golden solution
            const uint8_t* g = kat::golden[(n / 3) % 3];
            for (int i = 0; i < 104; ++i) soln[i] = g[i];
            soln[rng() % 100] ^= (uint8_t)(1u << (rng() % 8));
            // near-misses use the golden input/nonce so prePow matches
            for (int i = 0; i < 32; ++i) input[i] = kat::input32[i];
            for (int i = 0; i < 8;  ++i) nonce[i] = kat::nonce0[i];
        } else {
            for (auto& x : soln) x = (uint8_t)rng();
        }
        bool a = bh3::is_valid_solution(input, 32, nonce, soln);
        bool b = beam_valid(input, 32, nonce, soln);
        check(a == b, "fuzz: verdict agrees");
        if (a != b) break;   // stop flooding on first disagreement
    }
    return summary("differential");
}
