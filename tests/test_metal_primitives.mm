// The BeamHash III primitives, compiled as Metal Shading Language, must agree
// bit-for-bit with the host implementation -- because they ARE the host
// implementation. src/beamhash/bh3_primitives.h is compiled three times (host,
// nvcc, Metal) and this test is what stops the Metal instantiation from drifting.
//
// A mismatch here would be invisible downstream: the rounds would still find
// collisions, just the WRONG ones, and the CPU verify gate would silently reject
// every solution. So this runs over a wide index sweep rather than a token few.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "beamhash/bh3_primitives.h"
#include "check.h"
#include <vector>

#include "mxbm_metallib.h"   // kMxbmMetallib / kMxbmMetallibSize (generated)

using namespace mxbm;

static const uint32_t kN = 4096;

// The same prePow the CPU KAT vectors use, so a failure here is comparable to a
// failure in test_primitives rather than being its own private universe.
static const uint64_t kPrePow[4] = {
    0x0123456789abcdefull, 0xfedcba9876543210ull,
    0x00ff00ff00ff00ffull, 0xdeadbeefcafebabeull
};

// Run kernels/metal/prim_probe.metal over [0, kN) and return the 7 work words
// each thread produced. Returns false if Metal is unavailable, which is not a
// test failure -- it is a machine without a GPU.
static bool run_probe(std::vector<uint64_t>& out) {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { std::printf("  SKIP: no Metal device\n"); return false; }

    NSError* err = nil;
    dispatch_data_t blob = dispatch_data_create(kMxbmMetallib, kMxbmMetallibSize,
                                                dispatch_get_main_queue(),
                                                DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    id<MTLLibrary> lib = [dev newLibraryWithData:blob error:&err];
    if (!lib) { std::printf("  FAIL: metallib load: %s\n",
                            err.localizedDescription.UTF8String); return false; }

    id<MTLFunction> fn = [lib newFunctionWithName:@"prim_probe"];
    if (!fn) { std::printf("  FAIL: prim_probe not found in metallib\n"); return false; }
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!ps) { std::printf("  FAIL: pipeline: %s\n",
                           err.localizedDescription.UTF8String); return false; }

    const size_t words = (size_t)kN * bh3::kWorkWords;
    // Shared storage: unified memory, so the result is readable with no blit.
    id<MTLBuffer> obuf = [dev newBufferWithLength:words * sizeof(uint64_t)
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> pbuf = [dev newBufferWithBytes:kPrePow length:sizeof(kPrePow)
                                         options:MTLResourceStorageModeShared];

    id<MTLCommandQueue> q  = [dev newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:obuf offset:0 atIndex:0];
    [enc setBuffer:pbuf offset:0 atIndex:1];
    const NSUInteger tg = ps.maxTotalThreadsPerThreadgroup < 256
                        ? ps.maxTotalThreadsPerThreadgroup : 256;
    [enc dispatchThreads:MTLSizeMake(kN, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) { std::printf("  FAIL: dispatch: %s\n",
                                cb.error.localizedDescription.UTF8String); return false; }

    out.assign((const uint64_t*)obuf.contents, (const uint64_t*)obuf.contents + words);
    return true;
}

int main() {
    @autoreleasepool {
        section("MSL primitives are bit-identical to the host");
        std::vector<uint64_t> got;
        if (!run_probe(got)) return summary("metal_primitives");

        // Report only the FIRST divergence in full, then count the rest: a
        // systematic mismatch would otherwise print 28k lines and bury the
        // one fact that matters (which index and which word broke first).
        size_t bad = 0;
        for (uint32_t i = 0; i < kN; ++i) {
            bh3::Elem a, b, c;
            uint32_t t[1];
            bh3::seed_element(kPrePow, i, a);      t[0] = i;      bh3::apply_mix(a, t, 1u, 448u);
            bh3::seed_element(kPrePow, i + 1u, b); t[0] = i + 1u; bh3::apply_mix(b, t, 1u, 448u);
            bh3::combine(a, b, 424u, c);

            for (int w = 0; w < bh3::kWorkWords; ++w) {
                const uint64_t g = got[(size_t)i * bh3::kWorkWords + w];
                if (g != c.w[w]) {
                    if (bad == 0) {
                        char msg[64];
                        std::snprintf(msg, sizeof msg, "index %u word %d", i, w);
                        check_eq_u64(g, c.w[w], msg);
                    }
                    ++bad;
                }
            }
        }
        check(bad == 0, "all 4096 x 7 work words match the host");
        if (bad) std::printf("  (%zu of %zu words diverged)\n",
                             bad, (size_t)kN * bh3::kWorkWords);
        return summary("metal_primitives");
    }
}
