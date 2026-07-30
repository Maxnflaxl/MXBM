// Two questions the OpenCL side cannot answer (Nsight will not profile OpenCL):
//   1. Do sibling partial-sector writes combine? If a 72 B record really costs three
//      32 B sectors, MXBM's write traffic is 7.00 GiB, not the modelled 5.25.
//   2. Is the emit bound by bytes, or by the per-record atomic that hands out slots?
//
// Both answerable by timing, no counters needed. ATOMIC variants take their slot from
// atomicAdd exactly as the real emit does; PRECOMP variants write to the SAME addresses
// from a precomputed slot table, so the memory pattern is identical and only the atomic
// differs.
//
// `sweep` mode (P2 of the solver-reorg plan): same atomic emit, but across bucket
// counts 2^16..2^21 at the three record widths the pipeline actually writes
// (8 B seed / 16 B pair / 72 B packed). The warp-per-bucket design needs ~2^21
// buckets (mean 16); its 2M bucket-tail lines are ~64 MB, larger than L2, so the
// question is how much of today's scatter rate survives losing tail-line residency.
// Same caveat as above: only the ratio of each width's rate to its own bb=16 row
// transfers to the real pipeline.
#include <cstdio>
#include <cstdint>
#include <cstring>
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
static int sweep() {
    const uint32_t n = 33554432u;                       // 2^25, entry-stage population
    // cap tracks the design at each bucket count: 744 is today's real bb=16 layout,
    // 32 is the warp-per-bucket target (mean 16 + 4 sigma; Poisson drop ~1e-5).
    // Intermediates keep >=4 sigma headroom so drops never skew the byte counts.
    struct Cfg { int bb; uint32_t cap; };
    const Cfg cfgs[] = { {16,744},{17,380},{18,196},{19,104},{20,56},{21,32} };
    std::vector<uint32_t> hk(n);
    uint64_t s = 88172645463325252ull;
    for (uint32_t i=0;i<n;++i){ s^=s<<13; s^=s>>7; s^=s<<17; hk[i]=(uint32_t)s; }
    uint32_t *dk,*dc; uint64_t *dout;
    const uint64_t maxSlots = (uint64_t)(1u<<21)*32u;   // largest nb*cap in cfgs
    CK(cudaMalloc(&dk,(size_t)n*4));
    CK(cudaMalloc(&dc,(size_t)(1u<<21)*4));
    CK(cudaMalloc(&dout, maxSlots*9*8));
    CK(cudaMemcpy(dk,hk.data(),(size_t)n*4,cudaMemcpyHostToDevice));
    printf("P2 sweep: atomic scatter emit, n=2^25, best of 5. vs16 = rate / same width's bb=16 rate\n");
    printf("bb   mean  cap |  8B ms   GB/s  vs16 | 16B ms   GB/s  vs16 | 72B ms   GB/s  vs16 | drop%%\n");
    double base[3] = {0,0,0};
    std::vector<uint32_t> hc(1u<<21);
    for (const Cfg& c : cfgs) {
        const uint32_t nb = 1u<<c.bb, cap = c.cap;
        double ms[3];
        ms[0] = best_of([&]{ cudaMemset(dc,0,(size_t)nb*4); emit_atomic<1><<<(n+255)/256,256>>>(dk,dc,dout,n,nb,cap); });
        ms[1] = best_of([&]{ cudaMemset(dc,0,(size_t)nb*4); emit_atomic<2><<<(n+255)/256,256>>>(dk,dc,dout,n,nb,cap); });
        ms[2] = best_of([&]{ cudaMemset(dc,0,(size_t)nb*4); emit_atomic<9><<<(n+255)/256,256>>>(dk,dc,dout,n,nb,cap); });
        if (c.bb == 16) { base[0]=ms[0]; base[1]=ms[1]; base[2]=ms[2]; }
        // counts run free past cap, so the last rep's counters give the exact drop total
        CK(cudaMemcpy(hc.data(),dc,(size_t)nb*4,cudaMemcpyDeviceToHost));
        uint64_t drops = 0;
        for (uint32_t b=0;b<nb;++b) if (hc[b] > cap) drops += hc[b]-cap;
        const int wb[3] = {8,16,72};
        printf("%2d %6u %4u |", c.bb, n/nb, cap);
        for (int w=0;w<3;++w)
            printf(" %6.2f %6.1f %5.2f |", ms[w], (double)n*wb[w]/(ms[w]*1e-3)/1e9, base[w]/ms[w]);
        printf(" %.4f\n", 100.0*drops/n);
    }
    return 0;
}
int main(int argc, char** argv) {
    if (argc > 1 && !strcmp(argv[1],"sweep")) return sweep();
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
//
// SWEEP RESULTS (RTX 4070 Ti SUPER, 2026-07-31) — P2 of the solver-reorg plan
//   bb   mean  cap |  8B GB/s vs16 | 16B GB/s vs16 | 72B GB/s vs16 | drop%
//   16    512  744 |  131.1  1.00  |  128.1  1.00  |  127.8  1.00  | 0.0000
//   17    256  380 |  131.1  1.00  |  132.7  1.04  |  132.8  1.04  | 0.0000
//   18    128  196 |   70.6  0.54  |  117.0  0.91  |  132.8  1.04  | 0.0000
//   19     64  104 |   43.4  0.33  |   79.8  0.62  |  132.7  1.04  | 0.0000
//   20     32   56 |   34.7  0.26  |   65.7  0.51  |  132.3  1.03  | 0.0003
//   21     16   32 |   28.6  0.22  |   54.4  0.42  |  130.5  1.02  | 0.0016
//
// Verdict: warp-per-bucket's global-memory form (bb=21) is DEAD at the pre-registered
// <0.7x gate. Thin scattered records need their bucket-tail sectors resident in L2 so
// sibling writes merge; 2^21 tails x 32 B = 64 MB > 48 MB L2, and once merging fails
// every 8 B write pays a read-modify-write sector — a 4.5x rate collapse that no match-
// side saving can buy back (entry's emit alone would grow by ~7 ms against a 5.2 ms
// prize). The cliff starts at bb=18; only bb=17 is free. Wide records never merge much
// to begin with (72 B is flat), which is why today's coarse layout never saw this.
// Consequence: fine bucketing must live in SHARED at match time (counting-sort runs
// inside a coarse bucket), not in the global layout. The drop column doubles as P4:
// measured overflow matches Poisson (0.0016% at mean 16, cap 32).
