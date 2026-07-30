// The entry pass: seed + mix + scatter into round-1 buckets.
//
// Translated from entry_body / entry_scatter in kernels/cuda/{fused_round,
// pipeline_kernels}.cuh. Each thread derives one seed element, mixes it at
// Lmix(1) = 448, and appends an 8 B thin record to the bucket its 24-bit
// collision key selects.
//
// No st_em() shim here, unlike the CUDA source. MSL has no cache-hint store, but
// that costs nothing: MXBM_STCS defaults to 0 in fused_round.cuh, so the shipping
// CUDA build emits through plain stores too, and performance-research.md records
// the streaming variant as "verified applied in the SASS, and moved nothing".
#include "bh3_records.metalh"

using namespace metal;

namespace mxbm { namespace metalk {

// Shared by the standalone entry kernel and (later) the co-tenant path inside the
// fused rounds, so the two cannot drift -- the same reason the CUDA version is a
// device function rather than kernel-body code.
inline void entry_body(uint32_t idx,
                       const device uint64_t* pp4,
                       uint32_t bucket_bits, uint32_t bucket_cap,
                       device atomic_uint*   counts,
                       device uint64_t*      belem,
                       device atomic_uint*   drops)
{
    uint64_t pp[4] = { pp4[0], pp4[1], pp4[2], pp4[3] };
    bh3::Elem e;
    bh3::seed_element(pp, idx, e);
    uint32_t t1[1] = { idx };
    bh3::apply_mix(e, t1, 1u, 448u);

    const uint32_t key = (uint32_t)(e.w[0] & 0xFFFFFFu);
    const uint32_t b   = key >> (24u - bucket_bits);
    const uint32_t pos = atomic_fetch_add_explicit(&counts[b], 1u, memory_order_relaxed);
    if (pos < bucket_cap)
        belem[(size_t)b * bucket_cap + pos] = ((uint64_t)idx << 32) | key;
    else
        // drops[1] is the bucket-overflow counter. A silently clamped bucket is how
        // solutions go missing nondeterministically, so it is counted, never ignored.
        atomic_fetch_add_explicit(&drops[1], 1u, memory_order_relaxed);
}

}} // namespace mxbm::metalk

kernel void entry_scatter(const device uint64_t* pp4      [[buffer(0)]],
                          constant uint32_t&     begin    [[buffer(1)]],
                          constant uint32_t&     count    [[buffer(2)]],
                          constant uint32_t&     bkt_bits [[buffer(3)]],
                          constant uint32_t&     bkt_cap  [[buffer(4)]],
                          device atomic_uint*    counts   [[buffer(5)]],
                          device uint64_t*       belem    [[buffer(6)]],
                          device atomic_uint*    drops    [[buffer(7)]],
                          uint g [[thread_position_in_grid]])
{
    if (g >= count) return;
    mxbm::metalk::entry_body(begin + g, pp4, bkt_bits, bkt_cap, counts, belem, drops);
}
