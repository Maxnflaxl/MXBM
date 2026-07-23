// Proves the Beam island compiles + links against the vendored Blake2b.
#include <vector>
#include "beamHashIII.h"   // from BEAM_SOURCE_DIR/3rdparty/crypto
#include "check.h"
int main() {
    using namespace mxbm;
    BeamHash_III bh;
    blake2b_state base;
    int rc = bh.InitialiseState(base);
    check(rc == 0, "InitialiseState returns 0");
    // A 104-byte all-zero solution must be rejected (not crash).
    std::vector<uint8_t> soln(104, 0);
    bool ok = bh.IsValidSolution(base, soln);
    check(ok == false, "all-zero solution rejected");
    return summary("oracle_spike");
}
