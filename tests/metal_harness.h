#pragma once
// Shared Metal boilerplate for the kernel tests.
//
// Every Metal test needs the same six steps -- device, embedded metallib, function,
// pipeline, shared buffers, dispatch-and-wait -- and repeating them per test file is
// how the OpenCL tests ended up with several subtly different copies. One copy here,
// with the error paths actually reported rather than swallowed: a failed dispatch
// otherwise just leaves the output buffer zeroed, which reads like a legitimate
// result and fails the comparison for the wrong reason.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "check.h"
#include "mxbm_metallib.h"
#include <string>
#include <vector>

namespace mxbm { namespace mtest {

struct Harness {
    id<MTLDevice>       dev = nil;
    id<MTLLibrary>      lib = nil;
    id<MTLCommandQueue> queue = nil;
    bool ok = false;

    // Returns false (with a printed SKIP) on a machine with no Metal device -- not a
    // failure. Any other problem IS a failure and says so.
    bool init() {
        dev = MTLCreateSystemDefaultDevice();
        if (!dev) { std::printf("  SKIP: no Metal device\n"); return false; }
        NSError* err = nil;
        dispatch_data_t blob = dispatch_data_create(kMxbmMetallib, kMxbmMetallibSize,
                                                    dispatch_get_main_queue(),
                                                    DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        lib = [dev newLibraryWithData:blob error:&err];
        if (!lib) {
            std::printf("  FAIL: metallib load: %s\n", err.localizedDescription.UTF8String);
            ++fail_count(); return false;
        }
        queue = [dev newCommandQueue];
        ok = true;
        return true;
    }

    id<MTLComputePipelineState> pipeline(const char* name) {
        NSError* err = nil;
        id<MTLFunction> fn = [lib newFunctionWithName:@(name)];
        if (!fn) {
            std::printf("  FAIL: kernel '%s' not in metallib\n", name);
            ++fail_count(); return nil;
        }
        id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!ps) {
            std::printf("  FAIL: pipeline '%s': %s\n", name,
                        err.localizedDescription.UTF8String);
            ++fail_count(); return nil;
        }
        return ps;
    }

    // Storage mode Shared throughout: unified memory means the CPU reads results with
    // no blit, which is one of the concrete wins Metal has over the OpenCL path.
    id<MTLBuffer> buffer(size_t bytes) {
        return [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> buffer_with(const void* src, size_t bytes) {
        return [dev newBufferWithBytes:src length:bytes options:MTLResourceStorageModeShared];
    }

    // Dispatch `threads` threads over the given buffers and block until done.
    bool run(id<MTLComputePipelineState> ps, NSUInteger threads,
             NSArray<id<MTLBuffer>>* buffers) {
        if (!ps) return false;
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ps];
        for (NSUInteger i = 0; i < buffers.count; ++i)
            [enc setBuffer:buffers[i] offset:0 atIndex:i];
        const NSUInteger tg = ps.maxTotalThreadsPerThreadgroup < 256
                            ? ps.maxTotalThreadsPerThreadgroup : 256;
        [enc dispatchThreads:MTLSizeMake(threads, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) {
            std::printf("  FAIL: dispatch: %s\n", cb.error.localizedDescription.UTF8String);
            ++fail_count(); return false;
        }
        return true;
    }
};

}} // namespace mxbm::mtest
