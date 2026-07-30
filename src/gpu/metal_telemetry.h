#pragma once
// GPU telemetry on Apple Silicon -- the nvml.h analogue, and deliberately a much
// smaller one, because Apple publishes much less.
//
// What is NOT here, and why: power, core/memory clocks, temperature and fan speed
// have no public API on Apple Silicon. powermetrics reads them through IOReport, a
// private framework whose layout is undocumented and changes between OS releases.
// Reporting a guessed or scraped number in the stats block is worse than reporting
// none, so these stay unavailable and the corresponding have_* flags stay false --
// the same "0 or the truth, never a guess" rule the GPU core count follows.
//
// What IS available, through the public IORegistry (IOAccelerator's
// PerformanceStatistics), is utilization and driver memory in use.
#include "gpu/nvml.h"   // gpu::Telemetry, shared with the NVML path
#include <cstdint>

namespace mxbm { namespace gpu {

// True when the IOAccelerator service can be read. Safe to call at startup and
// ignore; everything below degrades to "nothing available" when it returns false.
bool metal_telemetry_init();

// Samples the integrated GPU. On Apple Silicon there is exactly one, so `index` is
// accepted for symmetry with nvml_sample() and anything non-zero returns an empty
// Telemetry rather than the same card's numbers under a different label.
Telemetry metal_sample(unsigned index = 0);

// Driver memory in use, in bytes. 0 when unavailable. Not part of Telemetry because
// no other backend reports it and a field only one caller fills is a field that rots.
uint64_t metal_memory_in_use(unsigned index = 0);

}} // namespace mxbm::gpu
