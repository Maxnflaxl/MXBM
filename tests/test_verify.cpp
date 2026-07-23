#include "check.h"
#include "kat_vectors.h"
#include "bh3_verify.h"
using namespace mxbm;
int main() {
    for (int s = 0; s < 3; ++s)
        check(bh3::is_valid_solution(kat::input32, 32, kat::nonce0, kat::golden[s]),
              "golden accepted");
    // Flip one bit in each solution's work-index region -> must reject.
    for (int s = 0; s < 3; ++s) {
        uint8_t bad[104];
        for (int i = 0; i < 104; ++i) bad[i] = kat::golden[s][i];
        bad[0] ^= 0x01;
        check(!bh3::is_valid_solution(kat::input32, 32, kat::nonce0, bad),
              "bit-flip rejected");
    }
    return summary("verify");
}
