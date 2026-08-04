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
#include "beamhash/bh3_primitives.h"   // real seed derivation for the `pipe` mode
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
// The eco kill-test: one round of a store-everything pipeline, stripped to its traffic.
// Coalesced 64 B read (the stored layer), the key from word 0, atomic slot, scattered
// 64 B write into bb=16 buckets -- no derive, no rebuild, no shared memory, which is
// the whole design premise. What it measures is whether OUR rig sustains the combined
// rd+wr rate that design needs; the pure-scatter shape rows cannot answer that because
// reads share the bus and real interleaved kernels run far above a pure write burst.
__global__ void stream_round(const uint64_t* __restrict__ in, uint32_t* __restrict__ counts,
                             uint64_t* __restrict__ out, uint32_t n, uint32_t nb, uint32_t cap) {
    uint32_t i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= n) return;
    const ulonglong2* v = reinterpret_cast<const ulonglong2*>(in + (size_t)i*8);
    ulonglong2 q0=v[0], q1=v[1], q2=v[2], q3=v[3];
    const uint32_t key = (uint32_t)(q0.x & 0xFFFFFFu);
    const uint32_t b = key >> 8;                       // top 16 of 24 bits: bb=16
    const uint32_t pos = atomicAdd(&counts[b], 1u); if (pos >= cap) return;
    ulonglong2* o = reinterpret_cast<ulonglong2*>(out + ((size_t)b*cap + pos)*8);
    o[0]=q0; o[1]=q1; o[2]=q2; o[3]=q3;
}
template<typename F> static double best_of(F f) {
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); double best=1e30;
    for (int r=0;r<5;++r){ cudaEventRecord(a); f(); cudaEventRecord(b); cudaEventSynchronize(b);
        float ms=0; cudaEventElapsedTime(&ms,a,b); if(ms<best) best=ms; }
    cudaEventDestroy(a); cudaEventDestroy(b); return best;
}
// `shape W` mode: ONE access shape in a sustained ~2.5 s loop, for measurement under a
// power cap. best_of takes a MINIMUM of short bursts, and under a cap the minimum is
// the rep before the governor clamps -- biased optimistic exactly where the cap matters.
// A sustained window contains the governor's steady state instead. W is the record's
// useful bytes (8/16/64/72, scattered atomic emit at today's bb=16 layout) or "seq"
// (the same store instructions, dense). Prints wall seconds and useful GB so a caller
// can bracket the energy counter around the process and divide.
template<int W> static double shape_loop(const uint32_t* dk, uint32_t* dc, uint64_t* dout,
                                         uint32_t n, uint32_t nb, uint32_t cap, long& reps) {
    cudaEvent_t a,m; cudaEventCreate(&a); cudaEventCreate(&m);
    for (int r=0;r<3;++r){ cudaMemset(dc,0,(size_t)nb*4);
        emit_atomic<W><<<(n+255)/256,256>>>(dk,dc,dout,n,nb,cap); }   // warm, untimed
    cudaEventRecord(a); float ms=0; reps=0;
    while (ms < 2500.0f) {
        for (int r=0;r<4;++r){ cudaMemset(dc,0,(size_t)nb*4);
            emit_atomic<W><<<(n+255)/256,256>>>(dk,dc,dout,n,nb,cap); }
        reps += 4; cudaEventRecord(m); cudaEventSynchronize(m); cudaEventElapsedTime(&ms,a,m);
    }
    cudaEventDestroy(a); cudaEventDestroy(m); return ms;
}
// `shape pipe`: the WHOLE eco pipeline as a traffic-faithful mock -- seed pass plus
// four chained rounds plus a thin terminal read, ping-ponging real bucket layers, so
// the full solve's floor is measured rather than extrapolated from one round. The seed
// derivation is REAL (7 siphashes + mix per element -- the one derive a store-everything
// design keeps, and at a cap compute is the expensive currency, so mocking it cheap
// would flatter the design). The rounds carry only a splitmix key update per element,
// which is faithful: the design under test has no derive or rebuild arithmetic in its
// rounds. Population is conserved 1:1 (a round consumes an element and emits a child),
// keys stay pseudorandom so the bucket distribution matches the scatter modes.
// Traffic per solve: n*(64 + 3*128 + 72 + 8) B = 17.45 GB, the competitor's schedule.
__constant__ uint64_t c_pipe_pp[4];
__global__ void pipe_seed(uint32_t* __restrict__ counts, uint64_t* __restrict__ out,
                          uint32_t n, uint32_t cap) {
    uint32_t i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= n) return;
    uint64_t pp[4] = { c_pipe_pp[0], c_pipe_pp[1], c_pipe_pp[2], c_pipe_pp[3] };
    mxbm::bh3::Elem e; mxbm::bh3::seed_element(pp, i, e);
    uint32_t t1[1] = { i };
    mxbm::bh3::apply_mix(e, t1, 1u, 448u);
    const uint32_t key = (uint32_t)(e.w[0] & 0xFFFFFFu);
    const uint32_t b = key >> 8;                        // bb = 16
    const uint32_t pos = atomicAdd(&counts[b], 1u); if (pos >= cap) return;
    ulonglong2* o = reinterpret_cast<ulonglong2*>(out + ((size_t)b*cap + pos)*8);
    o[0] = make_ulonglong2(e.w[0], e.w[1]); o[1] = make_ulonglong2(e.w[2], e.w[3]);
    o[2] = make_ulonglong2(e.w[4], e.w[5]); o[3] = make_ulonglong2(e.w[6], (uint64_t)i);
}
__device__ __forceinline__ uint64_t pipe_mix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull; x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27; x *= 0x94D049BB133111EBull; return x ^ (x >> 31);
}
// THIN = 1: round 4's 8 B child (the terminal round's input), else a full 64 B child.
template<int THIN>
__global__ void pipe_round(const uint64_t* __restrict__ in, const uint32_t* __restrict__ icnt,
                           uint32_t* __restrict__ ocnt, uint64_t* __restrict__ out,
                           uint32_t nb, uint32_t cap, uint32_t rnd) {
    const uint32_t tid = blockIdx.x*blockDim.x + threadIdx.x;
    const uint32_t b = tid / cap, s = tid % cap; if (b >= nb) return;
    uint32_t cnt = icnt[b]; if (cnt > cap) cnt = cap;
    if (s >= cnt) return;
    const ulonglong2* v = reinterpret_cast<const ulonglong2*>(in + ((size_t)b*cap + s)*8);
    ulonglong2 q0=v[0], q1=v[1], q2=v[2], q3=v[3];
    const uint64_t nk = pipe_mix(q0.x ^ q1.y ^ q3.x ^ ((uint64_t)rnd << 56));
    const uint32_t cb = (uint32_t)(nk >> 8) & 0xFFFFu;
    const uint32_t pos = atomicAdd(&ocnt[cb], 1u); if (pos >= cap) return;
    if constexpr (THIN) {
        out[(size_t)cb*cap + pos] = nk ^ q2.x;
    } else {
        ulonglong2* o = reinterpret_cast<ulonglong2*>(out + ((size_t)cb*cap + pos)*8);
        o[0] = make_ulonglong2((q0.x & ~0xFFFFFFull) | (nk & 0xFFFFFFu), q0.y);
        o[1] = q1; o[2] = q2; o[3] = q3;
    }
}
__global__ void pipe_sink(const uint64_t* __restrict__ in, const uint32_t* __restrict__ icnt,
                          uint32_t* __restrict__ drops, uint32_t nb, uint32_t cap) {
    const uint32_t tid = blockIdx.x*blockDim.x + threadIdx.x;
    const uint32_t b = tid / cap, s = tid % cap; if (b >= nb) return;
    uint32_t cnt = icnt[b]; if (cnt > cap) cnt = cap;
    if (s >= cnt) return;
    if (in[(size_t)b*cap + s] == 0xDEADBEEFDEADBEEFull) atomicAdd(drops, 1u);
}
static int pipemode() {
    const uint32_t n = 33554432u, nb = 65536u, cap = 744u;
    const size_t slots = (size_t)nb*cap;
    uint64_t *A,*B,*T; uint32_t *cA,*cB,*cT,*dsink;
    CK(cudaMalloc(&A, slots*64)); CK(cudaMalloc(&B, slots*64)); CK(cudaMalloc(&T, slots*8));
    CK(cudaMalloc(&cA,(size_t)nb*4)); CK(cudaMalloc(&cB,(size_t)nb*4));
    CK(cudaMalloc(&cT,(size_t)nb*4)); CK(cudaMalloc(&dsink,4));
    const uint64_t pp[4] = {0x0123456789ABCDEFull, 0xFEDCBA9876543210ull,
                            0xA5A5A5A55A5A5A5Aull, 0x1122334455667788ull};
    CK(cudaMemcpyToSymbol(c_pipe_pp, pp, 32));
    const unsigned rg = (unsigned)((slots + 511)/512);
    auto solve = [&]{
        cudaMemset(cA,0,(size_t)nb*4);
        pipe_seed<<<(n+511)/512,512>>>(cA, A, n, cap);
        uint64_t *src=A, *dst=B; uint32_t *cs=cA, *cd=cB;
        for (uint32_t r=1; r<=3; ++r) {
            cudaMemset(cd,0,(size_t)nb*4);
            pipe_round<0><<<rg,512>>>(src, cs, cd, dst, nb, cap, r);
            uint64_t* t=src; src=dst; dst=t; uint32_t* c=cs; cs=cd; cd=c;
        }
        cudaMemset(cT,0,(size_t)nb*4);
        pipe_round<1><<<rg,512>>>(src, cs, cT, T, nb, cap, 4);
        pipe_sink<<<rg,512>>>(T, cT, dsink, nb, cap);
    };
    solve(); solve();                                   // warm, untimed
    cudaEvent_t a,m; cudaEventCreate(&a); cudaEventCreate(&m);
    cudaEventRecord(a); float ms=0; long reps=0;
    while (ms < 2500.0f) {
        solve(); ++reps;
        cudaEventRecord(m); cudaEventSynchronize(m); cudaEventElapsedTime(&ms,a,m);
    }
    cudaEventDestroy(a); cudaEventDestroy(m);
    const double gbPerRep = (double)n*(64 + 3*128 + 72 + 8)/1e9;
    const double gb = gbPerRep*reps;
    printf("shape pipe: reps=%ld wall=%.3f s useful=%.2f GB rate=%.1f GB/s solve=%.2f ms\n",
           reps, ms/1e3, gb, gb/(ms*1e-3), ms/reps);
    return 0;
}
static int shapemode(const char* warg) {
    if (!strcmp(warg,"pipe")) return pipemode();
    const uint32_t n = 33554432u, nb = 65536u, cap = 744u;
    std::vector<uint32_t> hk(n);
    uint64_t s = 88172645463325252ull;
    for (uint32_t i=0;i<n;++i){ s^=s<<13; s^=s>>7; s^=s<<17; hk[i]=(uint32_t)s; }
    uint32_t *dk,*dc; uint64_t *dout;
    CK(cudaMalloc(&dk,(size_t)n*4)); CK(cudaMalloc(&dc,(size_t)nb*4));
    CK(cudaMalloc(&dout,(size_t)nb*cap*9*8));
    CK(cudaMemcpy(dk,hk.data(),(size_t)n*4,cudaMemcpyHostToDevice));
    long reps=0; double ms=0; double gbPerRep=0;
    if (!strcmp(warg,"round")) {
        // Layer in, layer out: rd n*64 + wr n*64 = 4.29 GB useful per rep. The input's
        // word 0 carries the same xorshift key stream the scatter modes use, so the
        // bucket distribution matches them and the drop rate stays ~0 at cap 744.
        uint64_t* din; CK(cudaMalloc(&din,(size_t)n*64));
        std::vector<uint64_t> hin((size_t)n*8);
        for (uint32_t i=0;i<n;++i){ hin[(size_t)i*8]=hk[i]; for(int w=1;w<8;++w) hin[(size_t)i*8+w]=hk[i]^w; }
        CK(cudaMemcpy(din,hin.data(),(size_t)n*64,cudaMemcpyHostToDevice));
        gbPerRep=(double)n*128/1e9;
        cudaEvent_t a,m; cudaEventCreate(&a); cudaEventCreate(&m);
        for (int r=0;r<3;++r){ cudaMemset(dc,0,(size_t)nb*4);
            stream_round<<<(n+255)/256,256>>>(din,dc,dout,n,nb,cap); }
        cudaEventRecord(a); float f=0;
        while (f < 2500.0f) {
            for (int r=0;r<4;++r){ cudaMemset(dc,0,(size_t)nb*4);
                stream_round<<<(n+255)/256,256>>>(din,dc,dout,n,nb,cap); }
            reps += 4; cudaEventRecord(m); cudaEventSynchronize(m); cudaEventElapsedTime(&f,a,m);
        }
        ms=f; cudaEventDestroy(a); cudaEventDestroy(m);
    } else if (!strcmp(warg,"seq")) {
        const uint64_t n64=(uint64_t)n*9; gbPerRep=(double)n64*8/1e9;
        cudaEvent_t a,m; cudaEventCreate(&a); cudaEventCreate(&m);
        for (int r=0;r<3;++r) seq_write<<<(unsigned)((n64+255)/256),256>>>(dout,n64);
        cudaEventRecord(a); float f=0;
        while (f < 2500.0f) {
            for (int r=0;r<4;++r) seq_write<<<(unsigned)((n64+255)/256),256>>>(dout,n64);
            reps += 4; cudaEventRecord(m); cudaEventSynchronize(m); cudaEventElapsedTime(&f,a,m);
        }
        ms=f; cudaEventDestroy(a); cudaEventDestroy(m);
    } else {
        const int wb=atoi(warg); gbPerRep=(double)n*wb/1e9;
        switch (wb) {
            case  8: ms=shape_loop<1>(dk,dc,dout,n,nb,cap,reps); break;
            case 16: ms=shape_loop<2>(dk,dc,dout,n,nb,cap,reps); break;
            case 64: ms=shape_loop<8>(dk,dc,dout,n,nb,cap,reps); break;
            case 72: ms=shape_loop<9>(dk,dc,dout,n,nb,cap,reps); break;
            default: printf("shape: width must be 8|16|64|72|seq|round|pipe\n"); return 1;
        }
    }
    const double gb = gbPerRep*reps;
    printf("shape %s: reps=%ld wall=%.3f s useful=%.2f GB rate=%.1f GB/s\n",
           warg, reps, ms/1e3, gb, gb/(ms*1e-3));
    return 0;
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
    if (argc > 2 && !strcmp(argv[1],"shape")) return shapemode(argv[2]);
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
//
// SHAPE RESULTS (RTX 4070 Ti SUPER, 2026-08-04, `shape` mode under power caps,
// docs-internal/rootruns/byte_shape_caps.sh, stock memory, energy from the counter)
//   cap    seq GB/s @SM   | 64B scatter GB/s @SM | J/GB seq | J/GB 64B
//   285 W  639.5  2775    | 132.9  2775          | 0.26     | 1.09
//   160 W  639.4  2730    | 132.9  2775          | 0.26     | 1.11
//   120 W  633.6  1005    | 127.2  2670          | 0.20     | 1.02
//   100 W  435.3   465    | 118.2  2505          | 0.24     | 0.93
//
// 1. DRAM work survives a core cap. Sequential holds 633 GB/s at 120 W on a 1005 MHz
//    core, and still 435 at the 100 W floor on 465 MHz. Scattered wide records lose
//    only 12% floor-vs-stock. The cap's currency is core cycles, not bytes -- the
//    store-design mechanism, confirmed from a third direction.
// 2. The clock column is a work-per-cycle readout: dense stores cost so much power per
//    cycle the governor crushes seq to 465 MHz at 100 W, while the stall-heavy scatter
//    keeps 2505 MHz inside the same budget.
// 3. J/GB is shape-dominated (~4.3x scatter vs seq, every cap), not cap-dominated.
//
// Consequence: MXBM_R2_FULL's loss at every cap CANNOT have been the shape of its
// bytes -- scattered bytes stay nearly cap-immune -- so that result no longer prices a
// ground-up streaming design's byte bill. The eco question stands on byte count vs
// arithmetic alone; the prize stays as the rung sweeps bounded it.
//
// KILL-PROBE RESULTS (`shape round`, same harness, 2026-08-04). One streaming round --
// coalesced 64 B read, key, atomic slot, scattered 64 B write, no derive, no shared:
//   285 W  512.2 GB/s @2775   (the competitor's measured kernel class, reproduced)
//   120 W  343.2 GB/s @1740   gate was 259 break-even / 350 alive
//   100 W  250.0 GB/s @1200   gate was 190 break-even
// A 17.7 GB/solve design's traffic floor is 51.6 ms at 120 W and 70.8 at 100 W, ~24 %
// under MXBM+rung at both -- 1.3x break-even, pre-registered before the run. The probe
// did NOT kill: the instruction-light kernel keeps 1740 MHz inside 120 W where MXBM's
// real rounds hold ~1155, which is the design thesis measured on this rig. Caveat: the
// W and J/GB columns for this mode are biased high (the energy bracket spans the whole
// process, including a 2.1 GB host-to-device copy, over the timed wall only); the rate
// and clock columns are clean and are the result.
//
// PIPE RESULTS (`shape pipe`, 2026-08-04): the full six-kernel mock, 17.45 GB/solve.
//   cap    stock mem        + 5001 rung      MXBM+rung comparator
//   285 W  37.4 ms          59.0 (roofline)  -- (real competitor: 35.8 on this schedule)
//   120 W  60.4             59.6             68.3  ->  best -12.7 %, THIN
//   100 W  82.1             66.1             92.9  ->  best -28.9 %, ALIVE
// The one-round -24 % halves to -12 % once the seed's real arithmetic and the thin
// legs are in; the rung crossover for a store design lands between 120 and 100 W (the
// competitor's ~126, reproduced). No match arithmetic in the mock, so these floors are
// ceilings for a real design. Verdict: eco is a deep-floor (<=~110 W) case only.
