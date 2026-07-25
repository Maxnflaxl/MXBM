#include "gpu/nvml.h"
#include <dlfcn.h>
#include <cstring>
#include <initializer_list>

namespace mxbm { namespace gpu {
namespace {

// Minimal NVML surface, declared here so we need no CUDA headers at build time.
using nvmlReturn_t = int;
constexpr nvmlReturn_t NVML_SUCCESS        = 0;
constexpr nvmlReturn_t NVML_ERROR_NOT_SUPPORTED = 3;
constexpr nvmlReturn_t NVML_ERROR_NO_PERMISSION = 4;
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
// Power limit: values are milliwatts throughout NVML's interface.
nvmlReturn_t (*p_pl_get)(nvmlDevice_t, unsigned*) = nullptr;
nvmlReturn_t (*p_pl_default)(nvmlDevice_t, unsigned*) = nullptr;
nvmlReturn_t (*p_pl_constraints)(nvmlDevice_t, unsigned*, unsigned*) = nullptr;
nvmlReturn_t (*p_pl_set)(nvmlDevice_t, unsigned) = nullptr;

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
    bind(p_pl_get,         "nvmlDeviceGetPowerManagementLimit");
    bind(p_pl_default,     "nvmlDeviceGetPowerManagementDefaultLimit");
    bind(p_pl_constraints, "nvmlDeviceGetPowerManagementLimitConstraints");
    bind(p_pl_set,         "nvmlDeviceSetPowerManagementLimit");
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

// mW -> W, rounded to nearest: the driver reports 285000 for a 285 W card but
// 284999 would truncate to 284 and make a no-op write look like a change.
static unsigned mw_to_w(unsigned mw) { return (mw + 500u) / 1000u; }

PowerLimit nvml_power_limit() {
    PowerLimit p;
    if (!g_ready || !p_pl_get) return p;
    unsigned v = 0;
    if (p_pl_get(g_dev, &v) != NVML_SUCCESS) return p;
    p.current_w = mw_to_w(v);
    p.valid = true;
    // The default and the constraints are separately optional: a card can
    // report its current limit and refuse the rest. Leaving them 0 lets the
    // caller tell "unknown" from "known and equal to the current value".
    if (p_pl_default && p_pl_default(g_dev, &v) == NVML_SUCCESS) p.default_w = mw_to_w(v);
    unsigned lo = 0, hi = 0;
    if (p_pl_constraints && p_pl_constraints(g_dev, &lo, &hi) == NVML_SUCCESS) {
        p.min_w = mw_to_w(lo);
        p.max_w = mw_to_w(hi);
    }
    return p;
}

NvmlWrite nvml_set_power_limit(unsigned watts) {
    if (!g_ready || !p_pl_set) return NvmlWrite::Unsupported;
    const nvmlReturn_t rc = p_pl_set(g_dev, watts * 1000u);
    if (rc == NVML_SUCCESS)                 return NvmlWrite::Ok;
    if (rc == NVML_ERROR_NO_PERMISSION)     return NvmlWrite::NoPermission;
    if (rc == NVML_ERROR_NOT_SUPPORTED)     return NvmlWrite::Unsupported;
    return NvmlWrite::Failed;
}

}} // namespace mxbm::gpu
