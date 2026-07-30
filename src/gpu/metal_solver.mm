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
#include <stdexcept>

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

}} // namespace mxbm::gpu
