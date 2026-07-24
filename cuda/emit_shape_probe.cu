// Two questions the OpenCL side cannot answer (Nsight will not profile OpenCL):
//   1. Do sibling partial-sector writes combine? If a 72 B record really costs three
//      32 B sectors, MXBM's write traffic is 7.00 GiB, not the modelled 5.25.
//   2. Is the emit bound by bytes, or by the per-record atomic that hands out slots?
//
// Both answerable by timing, no counters needed. ATOMIC variants take their slot from
// atomicAdd exactly as the real emit does; PRECOMP variants write to the SAME addresses
// from a precomputed slot table, so the memory pattern is identical and only the atomic
// differs.
#include <cstdio>
#include <cstdint>
#include <vector>
#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 1;} } while(0)

template<int W> __global__ void emit_atomic(const uint32_t* __restrict__ key,
        uint32_t* __restrict__ counts, uint64_t* __restrict__ out,
        uint32_t n, uint32_t nb, uint32_t cap) {
    uint32_t i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= n) return;
    uint32_t b = key[i] & (nb-1u);
    uint32_t pos = atomicAdd(&counts[b], 1u); if (pos >= cap) return;
    uint64_t* p = out + ((uint64_t)b*cap + pos)*W;
    #pragma unroll
    for (int w = 0; w < W; ++w) p[w] = (uint64_t)i*0x9E3779B97F4A7C15ull + w;
}
template<int W> __global__ void emit_precomp(const uint64_t* __restrict__ slot,
        uint64_t* __restrict__ out, uint32_t n) {
    uint32_t i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= n) return;
    uint64_t s = slot[i]; if (s == ~0ull) return;
    uint64_t* p = out + s*W;
    #pragma unroll
    for (int w = 0; w < W; ++w) p[w] = (uint64_t)i*0x9E3779B97F4A7C15ull + w;
}
// Calibration: the same bytes written SEQUENTIALLY. Without this the scatter numbers
// have no scale -- and this probe's absolute rates turned out not to match the real
// pipeline anyway (which interleaves reads, writes and compute), so only the RATIOS
// between variants here are meaningful.
__global__ void seq_write(uint64_t* __restrict__ out, uint64_t n64) {
    uint64_t i = (uint64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n64) out[i] = i;
}
template<typename F> static double best_of(F f) {
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); double best=1e30;
    for (int r=0;r<5;++r){ cudaEventRecord(a); f(); cudaEventRecord(b); cudaEventSynchronize(b);
        float ms=0; cudaEventElapsedTime(&ms,a,b); if(ms<best) best=ms; }
    cudaEventDestroy(a); cudaEventDestroy(b); return best;
}
int main() {
    const uint32_t n=33554432, nb=65536, cap=744;
    std::vector<uint32_t> hk(n); std::vector<uint64_t> hs(n); std::vector<uint32_t> cnt(nb,0);
    uint64_t s=88172645463325252ull;
    for (uint32_t i=0;i<n;++i){ s^=s<<13; s^=s>>7; s^=s<<17; hk[i]=(uint32_t)s; }
    for (uint32_t i=0;i<n;++i){ uint32_t b=hk[i]&(nb-1u); uint32_t p=cnt[b]++;
        hs[i] = (p<cap) ? ((uint64_t)b*cap+p) : ~0ull; }          // identical addresses
    uint32_t *dk,*dc; uint64_t *ds,*dout;
    CK(cudaMalloc(&dk,(size_t)n*4)); CK(cudaMalloc(&dc,(size_t)nb*4));
    CK(cudaMalloc(&ds,(size_t)n*8)); CK(cudaMalloc(&dout,(size_t)nb*cap*9*8));
    CK(cudaMemcpy(dk,hk.data(),(size_t)n*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ds,hs.data(),(size_t)n*8,cudaMemcpyHostToDevice));
    printf("rec        atomic ms  GB/s | precomp ms  GB/s | atomic costs\n");
    auto row=[&](int wb,double ma,double mp){ double u=(double)n*wb;
        printf("%2d B      %8.2f %6.1f | %9.2f %6.1f | %+6.1f%%\n", wb, ma, u/(ma*1e-3)/1e9,
               mp, u/(mp*1e-3)/1e9, (ma/mp-1)*100); };
    row(16, best_of([&]{ cudaMemset(dc,0,(size_t)nb*4); emit_atomic<2><<<(n+255)/256,256>>>(dk,dc,dout,n,nb,cap); }),
            best_of([&]{ emit_precomp<2><<<(n+255)/256,256>>>(ds,dout,n); }));
    row(64, best_of([&]{ cudaMemset(dc,0,(size_t)nb*4); emit_atomic<8><<<(n+255)/256,256>>>(dk,dc,dout,n,nb,cap); }),
            best_of([&]{ emit_precomp<8><<<(n+255)/256,256>>>(ds,dout,n); }));
    row(72, best_of([&]{ cudaMemset(dc,0,(size_t)nb*4); emit_atomic<9><<<(n+255)/256,256>>>(dk,dc,dout,n,nb,cap); }),
            best_of([&]{ emit_precomp<9><<<(n+255)/256,256>>>(ds,dout,n); }));
    // sequential baseline over the same volume as the 72 B case
    {
        uint64_t n64 = (uint64_t)n*9;
        double ms = best_of([&]{ seq_write<<<(unsigned)((n64+255)/256),256>>>(dout, n64); });
        printf("\nsequential  %8.2f %6.1f GB/s  (same volume, dense)\n",
               ms, (double)n64*8/(ms*1e-3)/1e9);
    }
    return 0;
}
// RESULTS (RTX 4070 Ti SUPER, 2026-07-25)
//   record   atomic     precomp   atomic costs
//   16 B     128.1 GB/s 130.6     +2.0%
//   64 B     132.9      133.7     +0.6%
//   72 B     132.8      133.7     +0.6%
//   sequential dense                640.1 GB/s
//
// 1. The slot atomic costs ~1%. It is not the emit's bottleneck.
// 2. Useful GB/s is FLAT across 16/64/72 B, and 72 B is not the 25% worse that
//    3-sectors-per-record would predict. Sibling partial-sector writes combine, so
//    MXBM's roofline (which counts useful bytes) is right and the "write traffic is
//    really 7.00 GiB" hypothesis is dead.
// 3. Random scatter runs at 21% of a dense write of the same volume.
//
// CAVEAT, and it is a big one: at 133 GB/s the pipeline's 6.25 GiB of writes would take
// 50 ms on their own, against a 40.3 ms whole solve. So the real emit is materially
// faster than this probe, which is a pure write burst with no reads or compute to
// overlap. Trust the RATIOS here, not the absolute rates. Getting trustworthy absolutes
// is what the CUDA port plus Nsight is for.
