// Seed-generation kernel. Uses bh3_seed_element from bh3.cl (built together).
__kernel void seed_layer(__global const ulong* pp4,   // [p0..p3]
                         uint begin, uint count,
                         __global ulong* out) {        // [count*7]
    uint g = (uint)get_global_id(0);
    if (g >= count) return;
    ulong pp[4] = { pp4[0], pp4[1], pp4[2], pp4[3] };
    ulong e[7];
    bh3_seed_element(pp, begin + g, e);
    for (int k = 0; k < 7; ++k) out[(size_t)g*7 + k] = e[k];
}
