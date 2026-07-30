// Metal backend implementation. The kernels live in kernels/metal/ and are compiled to a
// .metallib at build time, embedded by cmake/EmbedBinary.cmake.
//
// This file is Objective-C++ and is the ONLY place Metal types appear; metal_solver.h is
// deliberately free of them so the rest of the miner needs no Objective-C++ toolchain.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <IOKit/IOKitLib.h>

#include "gpu/metal_solver.h"
#include "gpu/budget.h"
#include "gpu/rowbucket_geom.h"
#include "beamhash/bh3_blake2b.h"
#include "beamhash/bh3_verify.h"
#include "beamhash/bh3_primitives.h"
#include "mxbm_metallib.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <stdexcept>

namespace {
constexpr uint32_t kElems    = 1u << 25;
constexpr uint32_t kCapacity = kElems + kElems / 32;      // 34,603,008
constexpr uint32_t kSurvCap  = 1024;
// Threadgroup size. MUST equal the kWG the kernels were compiled with -- it is the
// stride of their staging loops, and a mismatch silently skips elements rather than
// failing. CMake passes the same MXBM_METAL_WG to both this file and the metallib;
// the constructor then asserts it against the pipeline rather than trusting that.
#ifndef MXBM_METAL_WG
#define MXBM_METAL_WG 256
#endif
constexpr uint32_t kWG = MXBM_METAL_WG;
}

namespace mxbm { namespace gpu {

namespace {

// Apple exposes no public API for GPU core count -- MTLDevice has nothing, and
// maxThreadsPerThreadgroup is a threadgroup limit, not a core count. IOKit's
// AGXAccelerator service carries "gpu-core-count", which is what powermetrics and
// every system tool reads. Returns 0 when the property is absent rather than
// guessing: a wrong core count in the stats block is worse than a blank one.
unsigned query_gpu_core_count() {
    unsigned cores = 0;
    io_iterator_t it = 0;
    if (IOServiceGetMatchingServices(kIOMainPortDefault,
                                     IOServiceMatching("AGXAccelerator"), &it) != KERN_SUCCESS)
        return 0;
    io_object_t svc = IOIteratorNext(it);
    if (svc) {
        CFTypeRef v = IORegistryEntryCreateCFProperty(svc, CFSTR("gpu-core-count"),
                                                      kCFAllocatorDefault, 0);
        if (v) {
            if (CFGetTypeID(v) == CFNumberGetTypeID()) {
                int n = 0;
                if (CFNumberGetValue((CFNumberRef)v, kCFNumberIntType, &n) && n > 0)
                    cores = (unsigned)n;
            }
            CFRelease(v);
        }
        IOObjectRelease(svc);
    }
    IOObjectRelease(it);
    return cores;
}

// MTLCopyAllDevices order, stable within a boot. On Apple Silicon this is a
// one-element list; the vector shape exists to match CudaSolver::enumerate() so
// main.cpp's --devices handling needs no special case.
NSArray<id<MTLDevice>>* all_devices() {
    NSArray<id<MTLDevice>>* devs = MTLCopyAllDevices();
    if (devs.count > 0) return devs;
    // MTLCopyAllDevices can come back empty in odd sandboxes even when a default
    // device exists; fall back rather than reporting "no GPU" on a Mac that has one.
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    return d ? @[ d ] : @[];
}

MetalSolver::DeviceInfo info_for(id<MTLDevice> dev, int index) {
    MetalSolver::DeviceInfo di;
    di.name          = dev.name.UTF8String;
    di.global_mem    = (unsigned long long)dev.recommendedMaxWorkingSetSize;
    di.compute_units = query_gpu_core_count();
    di.index         = index;
    // Viability is the SHARED rule, not a Metal-private one: size the row-bucket
    // budget and ask whether the resulting seed layer can find anything at all.
    const Budget b = compute_budget(di.global_mem, (uint64_t)dev.maxBufferLength,
                                    0.85, kBytesPerElementRowbucket);
    di.viable = budget_can_find_solutions(b.elems_per_round);
    return di;
}

} // namespace

std::vector<MetalSolver::DeviceInfo> MetalSolver::enumerate() {
    @autoreleasepool {
        std::vector<DeviceInfo> out;
        NSArray<id<MTLDevice>>* devs = all_devices();
        out.reserve(devs.count);
        for (NSUInteger i = 0; i < devs.count; ++i)
            out.push_back(info_for(devs[i], (int)i));
        return out;
    }
}

MetalSolver::DeviceInfo MetalSolver::device_info(int index) {
    @autoreleasepool {
        NSArray<id<MTLDevice>>* devs = all_devices();
        if (index < 0 || (NSUInteger)index >= devs.count) return DeviceInfo{};
        return info_for(devs[index], index);
    }
}

bool MetalSolver::available(int index) {
    return device_info(index).viable;
}


// ---------------------------------------------------------------------------------
// The solver proper.
// ---------------------------------------------------------------------------------

namespace {
// Same ladder the CUDA path walks. allow_quad is FALSE for now: the quad record's
// r2/r3 kernel pair is not instantiated in fused_round.metal yet, and reaching for a
// rung whose kernels do not exist would fail at pipeline creation rather than fall
// back. Apple Silicon has no memory pressure that would need it (80 GB max buffer),
// so nothing is lost -- see the spec's open questions.
RbGeometry pick_geometry(uint64_t global_mem) {
    RbGeometry g = rb_geometry_for(kCapacity, /*max_alloc=*/0, global_mem,
                                   /*allow_quad=*/false);
    if (const char* e = std::getenv("MXBM_BB")) { g.bb = (uint32_t)atoi(e); g.sm = 17u - g.bb;
                                                  g.viable = true; }
    if (const char* e = std::getenv("MXBM_SM")) { g.sm = (uint32_t)atoi(e); g.viable = true; }
    return g;
}
} // namespace

struct MetalSolver::Impl {
    id<MTLDevice>       dev   = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> psEntry = nil, psR1 = nil, psR2 = nil, psR3 = nil,
                                psR4 = nil, psTerm = nil, psRecover = nil;

    uint32_t bb = 16, sm = 1, nb = 1u << 16;
    uint32_t cap = 0; size_t nslots = 0;

    // Every buffer is MTLStorageModeShared. On unified memory that is not a compromise:
    // the GPU reads it at full speed and the CPU sees the results with no blit, which is
    // how the CUDA path's cudaMemcpy of survivors and back-refs disappears entirely.
    id<MTLBuffer> elem[2] = {nil, nil};
    id<MTLBuffer> counts[2] = {nil, nil};
    id<MTLBuffer> dpp = nil, gictr = nil, drops = nil;
    id<MTLBuffer> left = nil, right = nil, survSlots = nil, survCount = nil, dleaves = nil;
    // One cfg buffer per round: {bucket_bits, submask_bits, in_cap, out_cap, out_off}.
    id<MTLBuffer> cfg[5] = {nil, nil, nil, nil, nil};
    id<MTLBuffer> entryCfg[4] = {nil, nil, nil, nil};
    id<MTLBuffer> recCfg[2] = {nil, nil};

    std::atomic<bool> abort_{false};
    DeviceInfo info;

    // Per-phase GPU time, in ms, from MTLCommandBuffer's GPUStartTime/GPUEndTime --
    // the time the GPU was actually executing, not the wall clock around the submit.
    // Populated only when MXBM_METAL_TIMING is set; zero cost otherwise.
    bool  timing = false;
    double phase_ms[7] = {0,0,0,0,0,0,0};   // entry, r1..r4, terminal, recover
    double wall_ms = 0.0;

    bool alloc_geometry(uint32_t bb_, uint32_t sm_) {
        bb = bb_; sm = sm_; nb = 1u << bb_;
        cap    = fb_cap_for(kCapacity / nb);
        nslots = (size_t)nb * cap;
        // Set 0 carries the round-2 and round-4 outputs, set 1 the round-1 and round-3
        // outputs. Strides come from the shared helper rather than local constants so the
        // allocation cannot disagree with what rb_geometry_for sized the ladder against.
        elem[0] = [dev newBufferWithLength:nslots * fb_set_stride(0) * sizeof(uint64_t)
                                   options:MTLResourceStorageModeShared];
        elem[1] = [dev newBufferWithLength:nslots * fb_set_stride(1) * sizeof(uint64_t)
                                   options:MTLResourceStorageModeShared];
        counts[0] = [dev newBufferWithLength:nb * sizeof(uint32_t)
                                     options:MTLResourceStorageModeShared];
        counts[1] = [dev newBufferWithLength:nb * sizeof(uint32_t)
                                     options:MTLResourceStorageModeShared];
        return elem[0] && elem[1] && counts[0] && counts[1];
    }
};

MetalSolver::MetalSolver(int index) : p_(new Impl) {
    @autoreleasepool {
        NSArray<id<MTLDevice>>* devs = all_devices();
        if (index < 0 || (NSUInteger)index >= devs.count)
            throw std::runtime_error("Metal: no device at index " + std::to_string(index));
        p_->timing = std::getenv("MXBM_METAL_TIMING") != nullptr;
        p_->dev   = devs[index];
        p_->queue = [p_->dev newCommandQueue];
        p_->info  = info_for(p_->dev, index);

        NSError* err = nil;
        dispatch_data_t blob = dispatch_data_create(kMxbmMetallib, kMxbmMetallibSize,
                                                    dispatch_get_main_queue(),
                                                    DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        id<MTLLibrary> lib = [p_->dev newLibraryWithData:blob error:&err];
        if (!lib)
            throw std::runtime_error(std::string("Metal: metallib load failed: ")
                                     + err.localizedDescription.UTF8String);

        auto make = [&](const char* name) -> id<MTLComputePipelineState> {
            NSError* e2 = nil;
            id<MTLFunction> fn = [lib newFunctionWithName:@(name)];
            if (!fn) throw std::runtime_error(std::string("Metal: kernel missing: ") + name);
            id<MTLComputePipelineState> ps =
                [p_->dev newComputePipelineStateWithFunction:fn error:&e2];
            if (!ps) throw std::runtime_error(std::string("Metal: pipeline ") + name + ": "
                                              + e2.localizedDescription.UTF8String);
            return ps;
        };
        p_->psEntry   = make("entry_scatter");
        p_->psR1      = make("fused_round_r1");
        p_->psR2      = make("fused_round_r2");
        p_->psR3      = make("fused_round_r3");
        p_->psR4      = make("fused_round_r4");
        p_->psTerm    = make("terminal_round");
        // The kernels stride their staging loops by their compile-time kWG. If this
        // binary and the metallib were built with different values the loops skip
        // elements and solutions vanish with no error -- so refuse to run rather than
        // mine quietly wrong. maxTotalThreadsPerThreadgroup is the one cross-check
        // available: a kernel that cannot host kWG threads certainly was not built
        // for it.
        for (id<MTLComputePipelineState> ps : { p_->psR1, p_->psR2, p_->psR3,
                                                p_->psR4, p_->psTerm }) {
            if (ps.maxTotalThreadsPerThreadgroup < kWG)
                throw std::runtime_error("Metal: kernel cannot host the configured "
                                         "threadgroup size (" + std::to_string(kWG) + ")");
        }
        p_->psRecover = make("recover");

        auto buf = [&](size_t bytes) {
            return [p_->dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        };
        p_->gictr     = buf(sizeof(uint32_t));
        p_->drops     = buf(4 * sizeof(uint32_t));
        p_->left      = buf((size_t)5 * kCapacity * sizeof(uint32_t));
        p_->right     = buf((size_t)5 * kCapacity * sizeof(uint32_t));
        p_->survSlots = buf(kSurvCap * sizeof(uint32_t));
        p_->survCount = buf(sizeof(uint32_t));
        p_->dleaves   = buf((size_t)kSurvCap * 32 * sizeof(uint32_t));
        p_->dpp       = buf(4 * sizeof(uint64_t));
        if (!p_->left || !p_->right || !p_->dpp)
            throw std::runtime_error("Metal allocation failed (out of memory)");

        const RbGeometry g = pick_geometry(p_->info.global_mem);
        if (!g.viable)
            throw std::runtime_error("Metal: no row-bucket geometry fits this device");
        // The chain table drops its key comparison, which is only sound while the table
        // can separate every key that varies inside a group: 24 - bb - sm bits. Every
        // geometry on the bb + sm = 17 line leaves exactly 7 and kTabSize is 128 -- but
        // MXBM_BB/MXBM_SM can be set off the line by hand, and silently combining
        // unequal keys would corrupt results rather than fail.
        if ((24u - g.bb - g.sm) > 7u)
            throw std::runtime_error("Metal: geometry leaves more key bits than the chain "
                                     "table can separate (bb + sm must be >= 17)");
        if (!p_->alloc_geometry(g.bb, g.sm))
            throw std::runtime_error("Metal allocation failed (out of memory)");

        // Per-round config. out_off is (round - 1) * kCapacity, the back-ref plane base.
        for (int r = 0; r < 5; ++r) {
            uint32_t c[5] = { p_->bb, p_->sm, p_->cap, p_->cap, (uint32_t)r * kCapacity };
            if (r == 4) { c[3] = 0; c[4] = 4u * kCapacity; }   // terminal: out_off, surv_cap
            p_->cfg[r] = [p_->dev newBufferWithBytes:c length:sizeof c
                                             options:MTLResourceStorageModeShared];
        }
        // terminal_round's cfg is {bb, sm, in_cap, out_off, surv_cap}.
        {
            uint32_t c[5] = { p_->bb, p_->sm, p_->cap, 4u * kCapacity, kSurvCap };
            p_->cfg[4] = [p_->dev newBufferWithBytes:c length:sizeof c
                                             options:MTLResourceStorageModeShared];
        }
        {
            uint32_t z = 0, n = kElems, bits = p_->bb, cp = p_->cap;
            p_->entryCfg[0] = [p_->dev newBufferWithBytes:&z    length:4 options:MTLResourceStorageModeShared];
            p_->entryCfg[1] = [p_->dev newBufferWithBytes:&n    length:4 options:MTLResourceStorageModeShared];
            p_->entryCfg[2] = [p_->dev newBufferWithBytes:&bits length:4 options:MTLResourceStorageModeShared];
            p_->entryCfg[3] = [p_->dev newBufferWithBytes:&cp   length:4 options:MTLResourceStorageModeShared];
        }
        {
            uint32_t z = 0, capv = kCapacity;
            p_->recCfg[0] = [p_->dev newBufferWithBytes:&z    length:4 options:MTLResourceStorageModeShared];
            p_->recCfg[1] = [p_->dev newBufferWithBytes:&capv length:4 options:MTLResourceStorageModeShared];
        }
    }
}

// Defined here rather than defaulted in the header: ~unique_ptr<Impl> needs Impl to be a
// complete type, and the whole point of the pimpl is that it is not one there.
MetalSolver::~MetalSolver() = default;

const MetalSolver::DeviceInfo& MetalSolver::device() const { return p_->info; }
void MetalSolver::request_abort() { p_->abort_.store(true, std::memory_order_relaxed); }

std::vector<std::array<uint8_t, 104>> MetalSolver::solve(const uint8_t input[32],
                                                          const uint8_t nonce[8]) {
    std::vector<std::array<uint8_t, 104>> out;
    @autoreleasepool {
        Impl& I = *p_;
        I.abort_.store(false, std::memory_order_relaxed);

        uint64_t pp[4];
        const uint8_t extra0[4] = {0, 0, 0, 0};
        bh3::compute_prepow(input, 32, nonce, extra0, pp);
        std::memcpy(I.dpp.contents, pp, sizeof pp);           // unified: no staging copy
        std::memset(I.drops.contents, 0, 16);
        std::memset(I.survCount.contents, 0, 4);
        std::memset(I.counts[0].contents, 0, (size_t)I.nb * 4);

        // One command buffer per round. That granularity is what gives the abort poll
        // its seam, and it matches the CUDA path's implicit stream ordering.
        int phase = 0;
        auto encode = [&](id<MTLComputePipelineState> ps, NSUInteger groups,
                          NSUInteger threadsPerGroup, NSArray* bufs) {
            id<MTLCommandBuffer> cb = [I.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ps];
            for (NSUInteger i = 0; i < bufs.count; ++i) {
                id b = bufs[i];
                [enc setBuffer:(b == [NSNull null] ? nil : (id<MTLBuffer>)b)
                        offset:0 atIndex:i];
            }
            [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.error) {
                // A failed dispatch otherwise just yields zero survivors, which reads
                // like a correctness bug rather than a configuration one.
                std::fprintf(stderr, "Metal dispatch error: %s\n",
                             cb.error.localizedDescription.UTF8String);
                return false;
            }
            if (I.timing && phase < 7)
                I.phase_ms[phase] = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
            ++phase;
            return true;
        };
        const auto wall0 = std::chrono::steady_clock::now();

        if (!encode(I.psEntry, (kElems + 255) / 256, 256,
                    @[ I.dpp, I.entryCfg[0], I.entryCfg[1], I.entryCfg[2], I.entryCfg[3],
                       I.counts[0], I.elem[0], I.drops ]))
            return out;

        const NSUInteger groups = (NSUInteger)I.nb << I.sm;
        int inSet = 0;
        id<MTLComputePipelineState> rounds[4] = { I.psR1, I.psR2, I.psR3, I.psR4 };
        for (int r = 0; r < 4; ++r) {
            const int o = inSet ^ 1;
            std::memset(I.counts[o].contents, 0, (size_t)I.nb * 4);
            std::memset(I.gictr.contents, 0, 4);
            if (!encode(rounds[r], groups, kWG,
                        @[ I.cfg[r], I.counts[inSet], I.elem[inSet], I.counts[o],
                           I.elem[o], I.left, I.right, I.gictr, I.drops, I.dpp ]))
                return out;
            inSet = o;
            if (I.abort_.load(std::memory_order_relaxed)) return out;
        }

        if (!encode(I.psTerm, groups, kWG,
                    @[ I.cfg[4], I.counts[inSet], I.elem[inSet], I.left, I.right,
                       I.survSlots, I.survCount, I.drops ]))
            return out;

        // Report BEFORE the survivor check: a solve that finds nothing still took the
        // same five rounds, and an attribution build (which deliberately finds nothing)
        // is exactly when the per-phase numbers are wanted.
        if (I.timing) {
            I.wall_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - wall0).count();
            const char* names[7] = { "entry", "r1", "r2", "r3", "r4", "terminal", "recover" };
            double gpu = 0;
            for (int i = 0; i < 7; ++i) gpu += I.phase_ms[i];
            std::fprintf(stderr, "  [metal] ");
            for (int i = 0; i < 7; ++i)
                std::fprintf(stderr, "%s %.1f  ", names[i], I.phase_ms[i]);
            std::fprintf(stderr, "| GPU %.1f  wall %.1f  (%.1f outside)\n",
                         gpu, I.wall_ms, I.wall_ms - gpu);
        }

        uint32_t hs = *(const uint32_t*)I.survCount.contents;   // unified: a plain read
        if (hs > kSurvCap) hs = kSurvCap;
        if (hs == 0 || I.abort_.load(std::memory_order_relaxed)) return out;

        std::memcpy(I.recCfg[0].contents, &hs, 4);
        if (!encode(I.psRecover, (hs + 63) / 64, 64,
                    @[ I.recCfg[0], I.recCfg[1], I.survSlots, I.left, I.right, I.dleaves ]))
            return out;

        // The CPU verify gate: the distinct-leaf and indexAfter checks the GPU rounds
        // deliberately skip. Nothing leaves this function unverified.
        const uint32_t* hl = (const uint32_t*)I.dleaves.contents;
        for (uint32_t i = 0; i < hs; ++i) {
            std::array<uint8_t, 104> sol{};
            uint32_t idx[32];
            std::memcpy(idx, hl + (size_t)i * 32, sizeof idx);
            bh3::pack_indices(idx, sol.data());
            if (bh3::is_valid_solution(input, 32, nonce, sol.data()))
                out.push_back(sol);
        }
    }
    return out;
}

}} // namespace mxbm::gpu
