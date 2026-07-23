#include "check.h"
#include "kat_vectors.h"
#include "bh3_verify.h"
#include "bh3_primitives.h"
#include <cstdio>
using namespace mxbm;
int main() {
    section("verify golden solutions (accept) — Equihash validity of the 104-byte solution");
    for (int s = 0; s < 3; ++s) {
        bool ok = bh3::is_valid_solution(kat::input32, 32, kat::nonce0, kat::golden[s]);
        char lbl[32]; std::snprintf(lbl, sizeof lbl, "golden[%d] accepted", s);
        check(ok, lbl);
    }
    if (verbose()) {
        // Show what a solution actually decodes to: the 32 x 25-bit indices of golden[0].
        uint32_t idx[32]; bh3::unpack_indices(kat::golden[0], idx);
        std::printf("     golden[0] 32 indices    =");
        for (int i = 0; i < 32; ++i) std::printf(" %u", idx[i]);
        std::printf("\n");
        show_hex("golden[0] extraNonce", &kat::golden[0][100], 4);
    }
    section("verify tampered solutions (reject) — flip one bit, must fail");
    for (int s = 0; s < 3; ++s) {
        uint8_t bad[104];
        for (int i = 0; i < 104; ++i) bad[i] = kat::golden[s][i];
        bad[0] ^= 0x01;
        bool rejected = !bh3::is_valid_solution(kat::input32, 32, kat::nonce0, bad);
        char lbl[32]; std::snprintf(lbl, sizeof lbl, "golden[%d] bit-flip rejected", s);
        check(rejected, lbl);
    }
    return summary("verify");
}
