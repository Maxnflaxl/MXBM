#include "check.h"
#include "bh3_types.h"
int main() {
    using namespace mxbm;
    check(bh3::kWorkWords == 7, "kWorkWords");
    check(bh3::kSolutionBytes == 104, "kSolutionBytes");
    bh3::Elem e{}; e.w[0] = 0x123;
    check_eq_u64(e.w[0], 0x123, "Elem store");
    return summary("smoke");
}
