// Round-trip probe for the Metal record helpers (bh3_records.metalh).
//
// The record bit layouts decide whether round 3 can rebuild its input instead of
// reading it, so a packing bug would not crash -- it would quietly corrupt leaf
// indices and every solution would fail the CPU verify gate with no clue why.
// tests/test_metal_records.mm drives this and compares against the CUDA formulas
// recomputed on the host.
#include "bh3_records.metalh"

using namespace metal;
using namespace mxbm::metalk;

// Each thread takes one (key, a, b, gi) tuple, packs it every way the pipeline
// packs records, unpacks it again, and writes back what it recovered. The host
// checks both that the round trip is lossless AND that the packed words match
// the layout the CUDA backend produces bit-for-bit.
kernel void records_probe(device uint32_t*       out  [[buffer(0)]],
                          const device uint32_t* in   [[buffer(1)]],
                          device uint64_t*       packed [[buffer(2)]],
                          uint i [[thread_position_in_grid]])
{
    const uint32_t key = in[i * 4 + 0] & 0xFFFFFFu;   // 24-bit collision key
    const uint32_t a   = in[i * 4 + 1] & kIdxMask;    // 25-bit index
    const uint32_t b   = in[i * 4 + 2] & kIdxMask;
    const uint32_t gi  = in[i * 4 + 3] & 0x3FFFFFFu;  // 26-bit global id

    // --- pair record (r1 -> r2) ---
    const uint64_t w0 = pair_w0(key, a);
    const uint64_t w1 = pair_w1(b, gi);

    // --- r3 pair-of-leaves record (r2 -> r3) ---
    const uint64_t p0 = r3_p0(a, b, key);
    const uint64_t p1 = r3_p1(key, a, gi);

    // --- quad record (r2 -> r3, 24 B form) ---
    const uint64_t q0 = quad_w0(key, a);
    const uint64_t q1 = quad_w1(b, a);
    const uint64_t q2 = quad_w2(b, gi);

    uint32_t o = i * 12u;
    out[o +  0] = pair_left(w0);
    out[o +  1] = pair_right(w1);
    out[o +  2] = pair_gi(w1);
    out[o +  3] = r3_l0(p0);
    out[o +  4] = r3_l1(p0);
    out[o +  5] = r3_l2(p0, p1);
    out[o +  6] = r3_l3(p1);
    out[o +  7] = r3_gi(p1);
    out[o +  8] = quad_l0(q0);
    out[o +  9] = quad_l1(q1);
    out[o + 10] = quad_l2(q1);
    out[o + 11] = quad_l3(q2);

    // The packed words themselves, so the host can assert the LAYOUT and not
    // merely that pack/unpack are mutually consistent -- a matched pair of bugs
    // would round-trip perfectly and still be wrong across backends.
    uint32_t q = i * 8u;
    packed[q + 0] = w0; packed[q + 1] = w1;
    packed[q + 2] = p0; packed[q + 3] = p1;
    packed[q + 4] = q0; packed[q + 5] = q1; packed[q + 6] = q2;
    packed[q + 7] = (uint64_t)quad_gi(q2);
}
