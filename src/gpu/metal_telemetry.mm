// Apple Silicon GPU telemetry through the public IORegistry.
//
// The IOAccelerator service publishes a PerformanceStatistics dictionary carrying
// "Device Utilization %" and "In use system memory". That is the whole of what is
// available without private APIs -- see metal_telemetry.h for why the missing fields
// stay missing rather than being scraped.
#import <Foundation/Foundation.h>
#include <IOKit/IOKitLib.h>

#include "gpu/metal_telemetry.h"

namespace mxbm { namespace gpu {

namespace {

// Reads one integer out of the first IOAccelerator's PerformanceStatistics.
// `found` distinguishes "absent" from "genuinely zero" -- an idle GPU really does
// report 0% utilization, and collapsing the two would blank the field whenever the
// miner was between jobs.
bool perf_stat(const char* key, uint64_t& value) {
    @autoreleasepool {
        io_iterator_t it = 0;
        if (IOServiceGetMatchingServices(kIOMainPortDefault,
                                         IOServiceMatching("IOAccelerator"),
                                         &it) != KERN_SUCCESS)
            return false;
        bool found = false;
        io_object_t svc = IOIteratorNext(it);
        if (svc) {
            CFMutableDictionaryRef props = nullptr;
            if (IORegistryEntryCreateCFProperties(svc, &props, kCFAllocatorDefault, 0)
                    == KERN_SUCCESS && props) {
                NSDictionary* d = (__bridge NSDictionary*)props;
                NSDictionary* ps = d[@"PerformanceStatistics"];
                if ([ps isKindOfClass:[NSDictionary class]]) {
                    id v = ps[@(key)];
                    if ([v isKindOfClass:[NSNumber class]]) {
                        value = (uint64_t)[(NSNumber*)v unsignedLongLongValue];
                        found = true;
                    }
                }
                CFRelease(props);
            }
            IOObjectRelease(svc);
        }
        IOObjectRelease(it);
        return found;
    }
}

} // namespace

bool metal_telemetry_init() {
    uint64_t v = 0;
    // Any readable statistic proves the service is reachable. Utilization is the one
    // this backend actually reports, so probe that rather than something else.
    return perf_stat("Device Utilization %", v);
}

Telemetry metal_sample(unsigned index) {
    Telemetry t;
    // Apple Silicon has exactly one integrated GPU. A per-device row showing the same
    // card's numbers under a different index is worse than a blank one.
    if (index != 0) return t;
    uint64_t v = 0;
    if (perf_stat("Device Utilization %", v)) {
        t.have_util = true;
        t.util_pct  = (unsigned)(v > 100 ? 100 : v);
    }
    // have_power / have_sm / have_mem / have_temp / have_fan stay false by design.
    return t;
}

uint64_t metal_memory_in_use(unsigned index) {
    if (index != 0) return 0;
    uint64_t v = 0;
    return perf_stat("In use system memory", v) ? v : 0;
}

}} // namespace mxbm::gpu
