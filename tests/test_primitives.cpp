#include "check.h"
#include "kat_vectors.h"
#include "bh3_primitives.h"
#include "bh3_blake2b.h"
using namespace mxbm;
using namespace mxbm::bh3;
static void test_siphash() {
    section("SipHash-2-4 (non-standard variant)");
    check_eq_u64(siphash24(0,1,2,3,0), 0xdd8748de678c744eULL, "siphash trivial n=0");
    check_eq_u64(siphash24(0,1,2,3,1), 0x55112546032352d8ULL, "siphash trivial n=1");
    const uint64_t* k = kat::prePow;
    check_eq_u64(siphash24(k[0],k[1],k[2],k[3],0),        0x10b04caf51290da9ULL, "siphash prePow n=0");
    check_eq_u64(siphash24(k[0],k[1],k[2],k[3],1),        0xe41fc5314952c418ULL, "siphash prePow n=1");
    check_eq_u64(siphash24(k[0],k[1],k[2],k[3],7),        0xb1514093d1c4f4b4ULL, "siphash prePow n=7");
    check_eq_u64(siphash24(k[0],k[1],k[2],k[3],8),        0xc80e82627af2913aULL, "siphash prePow n=8");
    check_eq_u64(siphash24(k[0],k[1],k[2],k[3],0xffffff), 0xe196386a335549bfULL, "siphash prePow n=ffffff");
}
static void test_prepow() {
    section("Blake2b prePow (SipHash key) from input||nonce||extraNonce");
    uint64_t pp[4];
    compute_prepow(kat::input32, 32, kat::nonce0, kat::extra0, pp);
    for (int i = 0; i < 4; ++i)
        check_eq_u64(pp[i], kat::prePow[i], "prePow word");
}
static void test_seed() {
    section("seed_element: 7x SipHash -> 448-bit element (indices 0 and 1)");
    Elem e0; seed_element(kat::prePow, 0, e0);
    for (int k = 0; k < 7; ++k) check_eq_u64(e0.w[k], kat::seed_index0[k], "seed idx0 word");
    Elem e1; seed_element(kat::prePow, 1, e1);
    for (int k = 0; k < 7; ++k) check_eq_u64(e1.w[k], kat::seed_index1[k], "seed idx1 word");
}
static void test_apply_mix() {
    section("apply_mix(Lmix=448): mixed low word + 24-bit collision lane");
    uint32_t t0[1] = {0}; Elem e0; seed_element(kat::prePow, 0, e0);
    apply_mix(e0, t0, 1, 448);
    check_eq_u64(e0.w[0], kat::mix448_index0, "apply_mix idx0 w0");
    check_eq_u64(collision_bits(e0), 0x026291u, "apply_mix idx0 lane");
    uint32_t t1[1] = {1}; Elem e1; seed_element(kat::prePow, 1, e1);
    apply_mix(e1, t1, 1, 448);
    check_eq_u64(e1.w[0], kat::mix448_index1, "apply_mix idx1 w0");
}
static void test_pack() {
    section("25-bit index pack/unpack (32 indices <-> 100 bytes)");
    uint32_t idx[32];
    for (int i = 0; i < 32; ++i) idx[i] = ((uint32_t)(i * 2654435761u)) & 0x1FFFFFFu;
    show_u64("sample index idx[1]", idx[1]);
    uint8_t packed[100]; pack_indices(idx, packed);
    check_eq_bytes(packed, kat::pack_expected, 100, "pack bytes");
    uint32_t rt[32]; unpack_indices(packed, rt);
    bool same = true; for (int i = 0; i < 32; ++i) if (rt[i] != idx[i]) same = false;
    check(same, "pack/unpack round-trip");
}
static void test_combine() {
    section("combine: XOR + drop 24-bit lane + mask to Lout");
    // combine(a, a, Lout) XORs to zero -> all work words zero.
    Elem a; seed_element(kat::prePow, 5, a);
    Elem out; combine(a, a, 424, out);
    bool z = true; for (int i = 0; i < 7; ++i) if (out.w[i]) z = false;
    check(z, "combine self -> zero");
}
int main() { test_siphash(); test_prepow(); test_seed(); test_apply_mix(); test_pack(); test_combine(); return summary("primitives"); }
