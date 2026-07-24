// Step 1 of the CUDA backend: prove the BeamHash III primitives run on-device and are
// bit-identical to the host path.
//
// No port was needed. src/beamhash/bh3_primitives.h is annotated MXBM_HD, which expands
// to __host__ __device__ under nvcc, so the CUDA kernels call THE SAME SOURCE the CPU
// verifier uses. Bit-exactness is therefore structural rather than something a test has
// to keep chasing -- which removes the part of this port that usually consumes the time.
#include "beamhash/bh3_primitives.h"
#include <cstdio>
#include <cstdint>
#include <vector>
using namespace mxbm;

__global__ void dev_seed_mix(const uint64_t* pp, const uint32_t* idx, uint64_t* out, uint32_t n) {
    uint32_t i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= n) return;
    bh3::Elem e; bh3::seed_element(pp, idx[i], e);
    uint32_t tree[1] = { idx[i] };
    bh3::apply_mix(e, tree, 1u, 448u);
    for (int w = 0; w < bh3::kWorkWords; ++w) out[(size_t)i*bh3::kWorkWords + w] = e.w[w];
}
__global__ void dev_combine(const uint64_t* a, const uint64_t* b, uint64_t* out, uint32_t n) {
    uint32_t i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= n) return;
    bh3::Elem ea, eb, ec;
    for (int w = 0; w < bh3::kWorkWords; ++w) {
        ea.w[w] = a[(size_t)i*bh3::kWorkWords + w];
        eb.w[w] = b[(size_t)i*bh3::kWorkWords + w];
    }
    bh3::combine(ea, eb, 424u, ec);
    for (int w = 0; w < bh3::kWorkWords; ++w) out[(size_t)i*bh3::kWorkWords + w] = ec.w[w];
}

int main() {
    const uint32_t n = 1u << 20;
    const uint64_t pp[4] = { 0x0123456789abcdefull, 0xfedcba9876543210ull,
                             0xdeadbeefcafebabeull, 0x1337c0de5eedfaceull };
    std::vector<uint32_t> idx(n);
    uint64_t s = 12345;
    for (uint32_t i = 0; i < n; ++i) { s^=s<<13; s^=s>>7; s^=s<<17; idx[i] = (uint32_t)(s & 0x1FFFFFF); }

    uint64_t *d_pp, *d_out, *d_out2; uint32_t* d_idx;
    cudaMalloc(&d_pp, 32); cudaMalloc(&d_idx, (size_t)n*4);
    cudaMalloc(&d_out, (size_t)n*7*8); cudaMalloc(&d_out2, (size_t)n*7*8);
    cudaMemcpy(d_pp, pp, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx, idx.data(), (size_t)n*4, cudaMemcpyHostToDevice);

    dev_seed_mix<<<(n+255)/256,256>>>(d_pp, d_idx, d_out, n);
    dev_combine<<<(n+255)/256,256>>>(d_out, d_out, d_out2, n);
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) { printf("FAIL: kernel error\n"); return 1; }

    std::vector<uint64_t> gpu((size_t)n*7), gpu2((size_t)n*7);
    cudaMemcpy(gpu.data(),  d_out,  (size_t)n*7*8, cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu2.data(), d_out2, (size_t)n*7*8, cudaMemcpyDeviceToHost);

    size_t bad1 = 0, bad2 = 0;
    for (uint32_t i = 0; i < n; ++i) {
        bh3::Elem e; bh3::seed_element(pp, idx[i], e);
        uint32_t tree[1] = { idx[i] };
        bh3::apply_mix(e, tree, 1u, 448u);
        bh3::Elem c; bh3::combine(e, e, 424u, c);
        for (int w = 0; w < 7; ++w) {
            if (gpu [(size_t)i*7+w] != e.w[w]) ++bad1;
            if (gpu2[(size_t)i*7+w] != c.w[w]) ++bad2;
        }
    }
    printf("seed_element + apply_mix : %zu mismatches over %u elements\n", bad1, n);
    printf("combine                  : %zu mismatches over %u elements\n", bad2, n);
    bool ok = (bad1 == 0 && bad2 == 0);
    printf("%s: cuda primitives bit-identical to host\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
