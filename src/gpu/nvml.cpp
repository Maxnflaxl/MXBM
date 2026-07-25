#include "gpu/nvml.h"
#include <dlfcn.h>
#include <cstring>
#include <initializer_list>

namespace mxbm { namespace gpu {
namespace {

// Minimal NVML surface, declared here so we need no CUDA headers at build time.
using nvmlReturn_t = int;
constexpr nvmlReturn_t NVML_SUCCESS = 0;
using nvmlDevice_t = void*;

void*        g_lib = nullptr;
nvmlDevice_t g_dev = nullptr;
bool         g_ready = false;

nvmlReturn_t (*p_init)(void) = nullptr;
nvmlReturn_t (*p_shutdown)(void) = nullptr;
nvmlReturn_t (*p_handle)(unsigned, nvmlDevice_t*) = nullptr;
nvmlReturn_t (*p_power)(nvmlDevice_t, unsigned*) = nullptr;
nvmlReturn_t (*p_clock)(nvmlDevice_t, int, unsigned*) = nullptr;
nvmlReturn_t (*p_temp)(nvmlDevice_t, int, unsigned*) = nullptr;
nvmlReturn_t (*p_fan)(nvmlDevice_t, unsigned*) = nullptr;

template<class F> void bind(F& fn, const char* name) { fn = (F)dlsym(g_lib, name); }

} // namespace

bool nvml_init() {
    if (g_ready) return true;
    // .so.1 is the versioned name the driver installs; the unversioned one only exists
    // when a toolkit is present.
    for (const char* so : {"libnvidia-ml.so.1", "libnvidia-ml.so"}) {
        g_lib = dlopen(so, RTLD_LAZY | RTLD_LOCAL);
        if (g_lib) break;
    }
    if (!g_lib) return false;

    bind(p_init,     "nvmlInit_v2");
    bind(p_shutdown, "nvmlShutdown");
    bind(p_handle,   "nvmlDeviceGetHandleByIndex_v2");
    bind(p_power,    "nvmlDeviceGetPowerUsage");
    bind(p_clock,    "nvmlDeviceGetClockInfo");
    bind(p_temp,     "nvmlDeviceGetTemperature");
    bind(p_fan,      "nvmlDeviceGetFanSpeed");
    if (!p_init || !p_handle) { dlclose(g_lib); g_lib = nullptr; return false; }

    if (p_init() != NVML_SUCCESS)              { dlclose(g_lib); g_lib = nullptr; return false; }
    if (p_handle(0, &g_dev) != NVML_SUCCESS)   { if (p_shutdown) p_shutdown();
                                                 dlclose(g_lib); g_lib = nullptr; return false; }
    g_ready = true;
    return true;
}

void nvml_shutdown() {
    if (!g_ready) return;
    if (p_shutdown) p_shutdown();
    if (g_lib) dlclose(g_lib);
    g_lib = nullptr; g_dev = nullptr; g_ready = false;
}

Telemetry nvml_sample() {
    Telemetry t;
    if (!g_ready) return t;
    unsigned v = 0;
    if (p_power && p_power(g_dev, &v) == NVML_SUCCESS) { t.have_power = true; t.power_w = v / 1000.0; }
    if (p_clock && p_clock(g_dev, /*NVML_CLOCK_SM=*/1,  &v) == NVML_SUCCESS) { t.have_sm  = true; t.sm_clock_mhz  = v; }
    if (p_clock && p_clock(g_dev, /*NVML_CLOCK_MEM=*/2, &v) == NVML_SUCCESS) { t.have_mem = true; t.mem_clock_mhz = v; }
    if (p_temp  && p_temp (g_dev, /*NVML_TEMPERATURE_GPU=*/0, &v) == NVML_SUCCESS) { t.have_temp = true; t.temp_c = v; }
    if (p_fan   && p_fan  (g_dev, &v) == NVML_SUCCESS) { t.have_fan = true; t.fan_pct = v; }
    return t;
}

}} // namespace mxbm::gpu
