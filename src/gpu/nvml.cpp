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
nvmlReturn_t (*p_driver)(char*, unsigned) = nullptr;
// Takes an nvmlPciInfo_t*; see nvml_pci_address() for how it is passed.
nvmlReturn_t (*p_pci)(nvmlDevice_t, void*) = nullptr;

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
    bind(p_driver,   "nvmlSystemGetDriverVersion");
    bind(p_pci,      "nvmlDeviceGetPciInfo_v3");
    if (!p_pci) bind(p_pci, "nvmlDeviceGetPciInfo_v2");
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

std::string nvml_driver_version() {
    if (!g_ready || !p_driver) return std::string();
    char buf[96] = {0};
    if (p_driver(buf, sizeof buf - 1) != NVML_SUCCESS) return std::string();
    buf[sizeof buf - 1] = 0;
    return buf;
}

std::string nvml_pci_address() {
    if (!g_ready || !p_pci) return std::string();
    // nvmlPciInfo_t is not declared here and has grown across NVML versions, so
    // pass more room than any known version needs -- under-sizing it would let
    // NVML write past the buffer -- and read only the leading busId string.
    char info[512] = {0};
    if (p_pci(g_dev, info) != NVML_SUCCESS) return std::string();
    info[sizeof info - 1] = 0;
    // "00000000:01:00.0" -> "1:0": drop the domain, and the .function suffix,
    // and strip the leading zeros each field is padded with.
    std::string id(info);
    size_t c1 = id.find(':');
    if (c1 == std::string::npos) return std::string();
    size_t c2 = id.find(':', c1 + 1);
    if (c2 == std::string::npos) return std::string();
    size_t dot = id.find('.', c2 + 1);
    std::string bus = id.substr(c1 + 1, c2 - c1 - 1);
    std::string dev = id.substr(c2 + 1, (dot == std::string::npos ? id.size() : dot) - c2 - 1);
    auto strip = [](std::string v) {
        size_t i = v.find_first_not_of('0');
        return i == std::string::npos ? std::string("0") : v.substr(i);
    };
    return strip(bus) + ":" + strip(dev);
}

}} // namespace mxbm::gpu
