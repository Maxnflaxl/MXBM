#pragma once
// Host-only CPU reference for one Wagner round of BeamHash III. Mirrors
// src/beamhash/bh3_verify.cpp's per-round math (apply_mix -> collision check
// -> combine) exactly, but -- unlike the verifier, which only CHECKS a given
// pair -- cpu_round SEARCHES: it groups every mixed input by its 24-bit
// collision key and emits every ordered pair within each group. It does NOT
// apply the verifier's distinctIndices/indexAfter gates (see bh3_verify.cpp
// lines 40-42): equal-lead pairs and overlapping trees are emitted here; a
// later Phase-C gate is responsible for filtering those.
//
// This is the ground-truth oracle later GPU-round tasks differential-test
// their kernels against.
#include "bh3_primitives.h"
#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace mxbm { namespace ref {

inline uint32_t Lmix(int r)   { return (r==5) ? 288u : 448u - (uint32_t)(r-1)*24u; }
inline uint32_t Lout(int r)   { if (r==5) return 24u; uint32_t L = 448u - (uint32_t)r*24u; return (r==4) ? L-64u : L; }
inline uint32_t padNum(int r) { uint32_t p = ((512u - Lmix(r)) + 24u)/25u, t = 1u<<(r-1); return p<t?p:t; }

struct RElem {
    uint64_t w[7];
    std::vector<uint32_t> tree;   // full leaf list, tree order
    uint32_t leftSlot=0, rightSlot=0;  // parents in the previous level (round>=1)
};

// seeds: RElem{ seed_element(pp,i), tree={i} }
inline std::vector<RElem> make_seeds(const uint64_t pp[4], const std::vector<uint32_t>& indices) {
    std::vector<RElem> out;
    out.reserve(indices.size());
    for (uint32_t idx : indices) {
        bh3::Elem e;
        bh3::seed_element(pp, idx, e);
        RElem r;
        for (int w = 0; w < bh3::kWorkWords; ++w) r.w[w] = e.w[w];
        r.tree = { idx };
        out.push_back(std::move(r));
    }
    return out;
}

// apply_mix on a copy: folds first padNum(r) leaves at Lmix(r); returns mixed
// w[0]. Passes treeLen = e.tree.size() (== 2^(r-1) for a properly-built
// round-r input) exactly as bh3_verify.cpp does -- apply_mix clamps padNum to
// treeLen internally, so this is equivalent to passing padNum(r) directly.
inline uint64_t cpu_mix_w0(const RElem& e, int r) {
    bh3::Elem copy;
    for (int w = 0; w < bh3::kWorkWords; ++w) copy.w[w] = e.w[w];
    bh3::apply_mix(copy, e.tree.data(), (uint32_t)e.tree.size(), Lmix(r));
    return copy.w[0];
}

// One full round, mirroring bh3_verify.cpp WITHOUT the distinct/indexAfter
// gates: mix all inputs; group by (mixed w[0] & 0xFFFFFF); per group emit ALL
// pairs (i<j over the group's members ordered by (lead=tree[0], then input
// slot)); pair ordered left=smaller (lead, slot); combine(Lout(r)) of MIXED
// values; child.tree = left.tree ++ right.tree; child.leftSlot/rightSlot =
// input slots.
inline std::vector<RElem> cpu_round(const std::vector<RElem>& in, int r) {
    const uint32_t Lm = Lmix(r);
    const uint32_t Lo = Lout(r);
    const size_t n = in.size();

    // Mixed copies: w[0] replaced by apply_mix's result, w[1..6] untouched
    // (apply_mix mutates only w[0] -- see bh3_primitives.h apply_mix).
    std::vector<bh3::Elem> mixed(n);
    for (size_t i = 0; i < n; ++i) {
        bh3::Elem e;
        for (int w = 0; w < bh3::kWorkWords; ++w) e.w[w] = in[i].w[w];
        bh3::apply_mix(e, in[i].tree.data(), (uint32_t)in[i].tree.size(), Lm);
        mixed[i] = e;
    }

    std::unordered_map<uint32_t, std::vector<size_t>> groups;
    for (size_t i = 0; i < n; ++i)
        groups[bh3::collision_bits(mixed[i])].push_back(i);

    std::vector<RElem> out;
    for (auto& kv : groups) {
        std::vector<size_t>& members = kv.second;
        std::sort(members.begin(), members.end(), [&](size_t a, size_t b) {
            uint32_t leadA = in[a].tree[0], leadB = in[b].tree[0];
            return (leadA != leadB) ? (leadA < leadB) : (a < b);
        });
        const size_t m = members.size();
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = i + 1; j < m; ++j) {
                size_t li = members[i], ri = members[j];
                bh3::Elem combined;
                bh3::combine(mixed[li], mixed[ri], Lo, combined);
                RElem child;
                for (int w = 0; w < bh3::kWorkWords; ++w) child.w[w] = combined.w[w];
                child.tree = in[li].tree;
                child.tree.insert(child.tree.end(), in[ri].tree.begin(), in[ri].tree.end());
                child.leftSlot  = (uint32_t)li;
                child.rightSlot = (uint32_t)ri;
                out.push_back(std::move(child));
            }
        }
    }
    return out;
}

namespace detail {
// Pre-order DFS over the flat consolidated back-ref arrays: level L's row
// lives at offset (L-1)*capacity; level-1 entries ARE leaf indices (there is
// no level-0 row to dereference), level>=2 entries are slot indices into the
// row directly below. Recursion decrements level by 1 each step; the base
// case (level==0) means the value being carried IS already a leaf, so it is
// appended as-is. Early-stops the instant `need` leaves are collected,
// including mid-subtree (checked between every pair of recursive calls).
inline void walk_leaves(const std::vector<uint32_t>& left, const std::vector<uint32_t>& right,
                        uint32_t capacity, int level, uint32_t slotOrLeaf, uint32_t need,
                        std::vector<uint32_t>& out) {
    if (out.size() >= need) return;
    if (level == 0) { out.push_back(slotOrLeaf); return; }
    uint32_t off = (uint32_t)(level - 1) * capacity;
    uint32_t l = left[off + slotOrLeaf];
    uint32_t r = right[off + slotOrLeaf];
    walk_leaves(left, right, capacity, level - 1, l, need, out);
    if (out.size() >= need) return;
    walk_leaves(left, right, capacity, level - 1, r, need, out);
}
} // namespace detail

// First `need` leaves of element `slot` at level L (round-L element), walking
// flat consolidated backref arrays (offset (k-1)*capacity for level k; level-1
// entries ARE leaf indices). Pre-order: left subtree then right.
inline std::vector<uint32_t> first_leaves(const std::vector<uint32_t>& left,
                                          const std::vector<uint32_t>& right,
                                          uint32_t capacity, int L, uint32_t slot, uint32_t need) {
    std::vector<uint32_t> out;
    out.reserve(need);
    detail::walk_leaves(left, right, capacity, L, slot, need, out);
    return out;
}

} } // namespace mxbm::ref
