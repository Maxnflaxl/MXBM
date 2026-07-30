// Differential probe for the BeamHash III primitives compiled as MSL.
//
// This kernel exists only to prove that src/beamhash/bh3_primitives.h -- the
// SAME header the CPU verifier and the CUDA kernels use -- produces bit-identical
// results on the GPU. It is not part of the solver pipeline. tests/test_metal_
// primitives.mm runs it and compares against the host implementation.
#include "beamhash/bh3_primitives.h"

using namespace metal;

// One thread per index i: seed i and i+1, mix both at Lmix(1) = 448 over their
// one-leaf trees, combine at Lout(1) = 424. That path exercises siphash24,
// seed_element, apply_mix and combine -- every primitive the rounds depend on.
kernel void prim_probe(device uint64_t*       out    [[buffer(0)]],
                       const device uint64_t* prePow [[buffer(1)]],
                       uint i [[thread_position_in_grid]])
{
    // Copy the prePow into thread space: the primitives take `thread` pointers,
    // because that is what the host and CUDA versions get and we are deliberately
    // not forking the signatures per backend.
    uint64_t pp[4] = { prePow[0], prePow[1], prePow[2], prePow[3] };

    mxbm::bh3::Elem a, b, c;
    uint32_t t[1];

    mxbm::bh3::seed_element(pp, i, a);
    t[0] = i;
    mxbm::bh3::apply_mix(a, t, 1u, 448u);

    mxbm::bh3::seed_element(pp, i + 1u, b);
    t[0] = i + 1u;
    mxbm::bh3::apply_mix(b, t, 1u, 448u);

    mxbm::bh3::combine(a, b, 424u, c);

    for (int w = 0; w < mxbm::bh3::kWorkWords; ++w)
        out[(uint)i * mxbm::bh3::kWorkWords + w] = c.w[w];
}
