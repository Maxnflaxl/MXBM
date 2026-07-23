#include "bh3_verify.h"
#include "bh3_primitives.h"
#include "bh3_blake2b.h"
#include <vector>
namespace mxbm { namespace bh3 {

static bool is_zero(const Elem& e) {
    for (int i = 0; i < kWorkWords; ++i) if (e.w[i]) return false;
    return true;
}
static bool distinct(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
    for (uint32_t x : a) for (uint32_t y : b) if (x == y) return false;
    return true;
}

bool is_valid_solution(const uint8_t* input, size_t inLen,
                       const uint8_t nonce[8], const uint8_t soln[104]) {
    uint64_t prePow[4];
    compute_prepow(input, inLen, nonce, &soln[100], prePow);

    uint32_t idx[32];
    unpack_indices(soln, idx);   // reads soln[0..99]

    struct Group { Elem e; std::vector<uint32_t> tree; };
    std::vector<Group> X(kNumIndices);
    for (int k = 0; k < kNumIndices; ++k) {
        seed_element(prePow, idx[k], X[k].e);
        X[k].tree = { idx[k] };
    }

    int round = 1;
    while (X.size() > 1) {
        std::vector<Group> Xtmp;
        for (size_t i = 0; i < X.size(); i += 2) {
            uint32_t Lmix = (uint32_t)(448 - (round - 1) * 24);
            if (round == 5) Lmix -= 64;
            apply_mix(X[i].e,   X[i].tree.data(),   (uint32_t)X[i].tree.size(),   Lmix);
            apply_mix(X[i+1].e, X[i+1].tree.data(), (uint32_t)X[i+1].tree.size(), Lmix);

            if (collision_bits(X[i].e) != collision_bits(X[i+1].e)) return false; // hasCollision
            if (!distinct(X[i].tree, X[i+1].tree))                  return false; // distinctIndices
            if (!(X[i].tree[0] < X[i+1].tree[0]))                   return false; // indexAfter

            uint32_t Lout = (uint32_t)(448 - round * 24);
            if (round == 4) Lout -= 64;
            if (round == 5) Lout = 24;

            Group g;
            combine(X[i].e, X[i+1].e, Lout, g.e);
            g.tree = X[i].tree;
            g.tree.insert(g.tree.end(), X[i+1].tree.begin(), X[i+1].tree.end());
            Xtmp.push_back(std::move(g));
        }
        X = std::move(Xtmp);
        ++round;
    }
    return is_zero(X[0].e);
}
} } // namespace mxbm::bh3
