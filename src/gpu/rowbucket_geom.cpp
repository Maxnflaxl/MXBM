#include "gpu/rowbucket_geom.h"
#include <cmath>

namespace mxbm { namespace gpu {

uint32_t fb_cap_for(uint32_t mean) {
    const double sd = std::sqrt((double)mean);
    return mean + (uint32_t)(8.0 * sd) + 32u;
}

void rowbucket_bytes(uint32_t capacity, uint32_t bb, size_t& total, size_t& single) {
    const uint32_t nb = 1u << bb;
    const uint32_t cap = fb_cap_for(capacity / nb);
    const size_t nslots = (size_t)nb * cap;
    single = nslots * fb_set_stride(0) * 8;                     // fb_elem[0] -- largest
    total = nslots * (fb_set_stride(0) + fb_set_stride(1)) * 8  // both packed record sets
          + 2*(size_t)nb*4 + (size_t)5*capacity*4*2 + 64;       // +counts/left/right/counters
}

RbGeometry rb_geometry_for(uint32_t capacity, uint64_t max_alloc, uint64_t global_mem) {
    for (uint32_t bb = 16; bb >= 14; --bb) {
        size_t total = 0, single = 0;
        rowbucket_bytes(capacity, bb, total, single);
        if (max_alloc && single > (size_t)max_alloc) continue;
        if (global_mem && total + (size_t)(1ull << 30) > (size_t)global_mem) continue;
        return { bb, 17u - bb, true };
    }
    return { 14u, 3u, false };   // nothing fits; callers fall back to the sort path
}

}} // namespace mxbm::gpu
