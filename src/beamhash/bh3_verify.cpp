#include "bh3_verify.h"
#include "bh3_primitives.h"
#include "bh3_blake2b.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
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

VerifyReason classify_solution(const uint8_t* input, size_t inLen,
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

            if (collision_bits(X[i].e) != collision_bits(X[i+1].e))
                return { VerifyReason::Collision, (uint8_t)round };   // hasCollision
            if (!distinct(X[i].tree, X[i+1].tree))
                return { VerifyReason::DupIndex, (uint8_t)round };    // distinctIndices
            if (!(X[i].tree[0] < X[i+1].tree[0]))
                return { VerifyReason::Order, (uint8_t)round };       // indexAfter

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
    if (!is_zero(X[0].e)) return { VerifyReason::NonZero, 0 };
    return { VerifyReason::None, 0 };
}

bool is_valid_solution(const uint8_t* input, size_t inLen,
                       const uint8_t nonce[8], const uint8_t soln[104]) {
    return classify_solution(input, inLen, nonce, soln).kind == VerifyReason::None;
}

// ---- MXBM_VERIFY_STATS tallies --------------------------------------------------
// Atomics because a rig can run one solver per GPU; printed once at process exit.
namespace {
struct VStats {
    std::atomic<uint64_t> solves{0}, cand{0}, ok{0}, nonzero{0};
    std::atomic<uint64_t> coll[6]{}, dup[6]{}, order[6]{};   // [1..5] by round
};
VStats g;
void print_stats() {
    const uint64_t s = g.solves.load(), c = g.cand.load(), k = g.ok.load();
    if (c == 0) return;
    uint64_t coll = 0, dup = 0, ord = 0;
    for (int r = 1; r <= 5; ++r) { coll += g.coll[r]; dup += g.dup[r]; ord += g.order[r]; }
    const uint64_t rej = c - k;
    std::fprintf(stderr,
        "[verify-stats] solves=%llu candidates=%llu (%.3f/solve) verified=%llu (%.3f/solve)\n"
        "[verify-stats] rejected=%llu (%.3f/solve): dup-index=%llu  order=%llu  "
        "collision=%llu  nonzero-final=%llu\n",
        (unsigned long long)s, (unsigned long long)c, s ? (double)c / s : 0.0,
        (unsigned long long)k, s ? (double)k / s : 0.0,
        (unsigned long long)rej, s ? (double)rej / s : 0.0,
        (unsigned long long)dup, (unsigned long long)ord,
        (unsigned long long)coll, (unsigned long long)g.nonzero.load());
    auto row = [](const char* name, const std::atomic<uint64_t>* a) {
        uint64_t t = 0; for (int r = 1; r <= 5; ++r) t += a[r];
        if (!t) return;
        std::fprintf(stderr, "[verify-stats]   %s by round:", name);
        for (int r = 1; r <= 5; ++r)
            std::fprintf(stderr, "  r%d=%llu", r, (unsigned long long)a[r].load());
        std::fprintf(stderr, "\n");
    };
    row("dup-index", g.dup);
    row("order", g.order);
    row("collision", g.coll);
}
void ensure_printer() {
    static bool once = [] { std::atexit(print_stats); return true; }();
    (void)once;
}
} // namespace

void verify_stats_note_solve() { ensure_printer(); g.solves.fetch_add(1); }
void verify_stats_tally(const VerifyReason& r) {
    ensure_printer();
    g.cand.fetch_add(1);
    const int rd = (r.round >= 1 && r.round <= 5) ? r.round : 0;
    switch (r.kind) {
        case VerifyReason::None:      g.ok.fetch_add(1); break;
        case VerifyReason::Collision: if (rd) g.coll[rd].fetch_add(1); break;
        case VerifyReason::DupIndex:  if (rd) g.dup[rd].fetch_add(1); break;
        case VerifyReason::Order:     if (rd) g.order[rd].fetch_add(1); break;
        case VerifyReason::NonZero:   g.nonzero.fetch_add(1); break;
    }
}

} } // namespace mxbm::bh3
