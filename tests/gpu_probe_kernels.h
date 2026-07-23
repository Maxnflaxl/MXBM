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
// probe_apply_mix: case i has e at ework[i*8..i*8+6] (8-stride, pad slot unused
// on input), tree at etree[i*16..], plus params[i*2]=treeLen, params[i*2+1]=Lmix.
// Writes out[i] = apply_mix(e, tree, treeLen, Lmix).
__kernel void probe_apply_mix(__global const ulong* ework,   // [n*8]
                              __global const uint*  etree,   // [n*16]
                              __global const uint*  params,  // [n*2]
                              __global ulong* out) {         // [n]
    size_t i = get_global_id(0);
    ulong e[7]; for (int k=0;k<7;++k) e[k]=ework[i*8+k];
    uint treeLen = params[i*2+0];
    uint Lmix    = params[i*2+1];
    uint tree[16]; for (int k=0;k<16;++k) tree[k]=etree[i*16+k];
    out[i] = bh3_apply_mix(e, tree, treeLen, Lmix);
}
// probe_combine: a at aw[i*7..], b at bw[i*7..], Lout at lout[i]; out at o[i*7..].
__kernel void probe_combine(__global const ulong* aw,    // [n*7]
                            __global const ulong* bw,    // [n*7]
                            __global const uint*  lout,  // [n]
                            __global ulong* o) {         // [n*7]
    size_t i = get_global_id(0);
    ulong a[7],b[7],r[7];
    for (int k=0;k<7;++k){ a[k]=aw[i*7+k]; b[k]=bw[i*7+k]; }
    bh3_combine(a, b, lout[i], r);
    for (int k=0;k<7;++k) o[i*7+k]=r[k];
}
// probe_pack: case i packs idx[i*32..i*32+31] -> out[i*100..i*100+99].
__kernel void probe_pack(__global const uint* idx,   // [n*32]
                         __global uchar* out) {       // [n*100]
    size_t i = get_global_id(0);
    uint ix[32]; for (int k=0;k<32;++k) ix[k]=idx[i*32+k];
    uchar packed[100];
    bh3_pack_indices(ix, packed);
    for (int k=0;k<100;++k) out[i*100+k]=packed[k];
}
)CLSRC";
}} // namespace mxbm::gpu
