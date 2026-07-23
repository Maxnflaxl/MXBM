#pragma once
namespace mxbm { namespace gpu {
// Probe kernels: one work-item per case; read flat inputs, call a bh3.cl
// primitive, write flat outputs. Built together with kBh3ClSource.
inline const char* kProbeSrc = R"CLSRC(
// probe_siphash: for each case i, out[i] = siphash24(k0..k3, nonce[i])
__kernel void probe_siphash(__global const ulong* key4,   // [k0,k1,k2,k3]
                            __global const ulong* nonce,  // [n]
                            __global ulong* out) {        // [n]
    size_t i = get_global_id(0);
    out[i] = bh3_siphash24(key4[0], key4[1], key4[2], key4[3], nonce[i]);
}
// probe_seed: for each case i, write 7 words of seed_element(pp, index[i])
__kernel void probe_seed(__global const ulong* pp4,     // [p0..p3]
                         __global const uint* index,    // [n]
                         __global ulong* out) {         // [n*7]
    size_t i = get_global_id(0);
    ulong pp[4] = { pp4[0], pp4[1], pp4[2], pp4[3] };
    ulong e[7];
    bh3_seed_element(pp, index[i], e);
    for (int k = 0; k < 7; ++k) out[i*7 + k] = e[k];
}
// probe_collision: for each case i, out[i] = collision_bits(w0[i]) (low 24 bits)
__kernel void probe_collision(__global const ulong* w0,   // [n]
                              __global uint* out) {        // [n]
    size_t i = get_global_id(0);
    out[i] = bh3_collision_bits(w0[i]);
}
)CLSRC";
}} // namespace mxbm::gpu
