#include "gpu/nvml.h"
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <cstring>
#include <initializer_list>
#include <vector>

namespace mxbm { namespace gpu {
namespace {

// Minimal NVML surface, declared here so we need no CUDA headers at build time.
using nvmlReturn_t = int;
constexpr nvmlReturn_t NVML_SUCCESS        = 0;
constexpr nvmlReturn_t NVML_ERROR_NOT_SUPPORTED = 3;
constexpr nvmlReturn_t NVML_ERROR_NO_PERMISSION = 4;
constexpr nvmlReturn_t NVML_ERROR_INSUFFICIENT_SIZE = 7;
using nvmlDevice_t = void*;

void*        g_lib = nullptr;
nvmlDevice_t g_dev = nullptr;          // device 0; init bookkeeping only -- every
                                       // read/write resolves its own dev_at(index)
std::vector<nvmlDevice_t> g_devs;      // every device, for per-card telemetry
bool         g_ready = false;

// The handle for `index`, or nullptr when it does not exist. Callers return an
// empty result on nullptr rather than falling back to device 0 -- a per-device
// row is worse than useless if it can silently show another card.
nvmlDevice_t dev_at(unsigned index) {
    return index < g_devs.size() ? g_devs[index] : nullptr;
}

nvmlReturn_t (*p_init)(void) = nullptr;
nvmlReturn_t (*p_shutdown)(void) = nullptr;
nvmlReturn_t (*p_handle)(unsigned, nvmlDevice_t*) = nullptr;
nvmlReturn_t (*p_count)(unsigned*) = nullptr;
nvmlReturn_t (*p_power)(nvmlDevice_t, unsigned*) = nullptr;
nvmlReturn_t (*p_clock)(nvmlDevice_t, int, unsigned*) = nullptr;
nvmlReturn_t (*p_temp)(nvmlDevice_t, int, unsigned*) = nullptr;
nvmlReturn_t (*p_fan)(nvmlDevice_t, unsigned*) = nullptr;
nvmlReturn_t (*p_driver)(char*, unsigned) = nullptr;
nvmlReturn_t (*p_name)(nvmlDevice_t, char*, unsigned) = nullptr;
// Takes an nvmlPciInfo_t*; see nvml_pci_address() for how it is passed.
nvmlReturn_t (*p_pci)(nvmlDevice_t, void*) = nullptr;
// Cumulative energy since driver load, in millijoules.
nvmlReturn_t (*p_energy)(nvmlDevice_t, unsigned long long*) = nullptr;
// Supported memory clocks: count is in/out (capacity in, entries written out).
nvmlReturn_t (*p_memclocks)(nvmlDevice_t, unsigned*, unsigned*) = nullptr;
// Power limit: values are milliwatts throughout NVML's interface.
nvmlReturn_t (*p_pl_get)(nvmlDevice_t, unsigned*) = nullptr;
nvmlReturn_t (*p_pl_default)(nvmlDevice_t, unsigned*) = nullptr;
nvmlReturn_t (*p_pl_constraints)(nvmlDevice_t, unsigned*, unsigned*) = nullptr;
nvmlReturn_t (*p_pl_set)(nvmlDevice_t, unsigned) = nullptr;
// V/F curve offsets. Signed: a negative offset is a downclock, which is a
// legitimate thing to ask for and the reason these are int and not unsigned.
nvmlReturn_t (*p_coff_get)(nvmlDevice_t, int*) = nullptr;
nvmlReturn_t (*p_coff_set)(nvmlDevice_t, int) = nullptr;
nvmlReturn_t (*p_coff_range)(nvmlDevice_t, int*, int*) = nullptr;
nvmlReturn_t (*p_moff_get)(nvmlDevice_t, int*) = nullptr;
nvmlReturn_t (*p_moff_set)(nvmlDevice_t, int) = nullptr;
nvmlReturn_t (*p_moff_range)(nvmlDevice_t, int*, int*) = nullptr;
// Locked clocks, and the max the domain supports.
nvmlReturn_t (*p_maxclock)(nvmlDevice_t, int, unsigned*) = nullptr;
nvmlReturn_t (*p_lock_core)(nvmlDevice_t, unsigned, unsigned) = nullptr;
nvmlReturn_t (*p_unlock_core)(nvmlDevice_t) = nullptr;
nvmlReturn_t (*p_lock_mem)(nvmlDevice_t, unsigned, unsigned) = nullptr;
nvmlReturn_t (*p_unlock_mem)(nvmlDevice_t) = nullptr;
// Fans.
nvmlReturn_t (*p_numfans)(nvmlDevice_t, unsigned*) = nullptr;
nvmlReturn_t (*p_fan_set)(nvmlDevice_t, unsigned, unsigned) = nullptr;
nvmlReturn_t (*p_fan_auto)(nvmlDevice_t, unsigned) = nullptr;
nvmlReturn_t (*p_fan_range)(nvmlDevice_t, unsigned*, unsigned*) = nullptr;

// Every write goes through the same mapping, so one place decides what
// "NoPermission" means and callers cannot each invent their own reading of it.
NvmlWrite map_write(nvmlReturn_t rc) {
    if (rc == NVML_SUCCESS)             return NvmlWrite::Ok;
    if (rc == NVML_ERROR_NO_PERMISSION) return NvmlWrite::NoPermission;
    if (rc == NVML_ERROR_NOT_SUPPORTED) return NvmlWrite::Unsupported;
    return NvmlWrite::Failed;
}

// dlopen/LoadLibrary behind one set of names, so the bind list below reads
// the same on both platforms.
#ifdef _WIN32
void* lib_open() {
    // nvml.dll ships with the driver into System32; very old drivers parked it
    // in NVSMI, which is not on PATH.
    if (HMODULE h = LoadLibraryA("nvml.dll")) return (void*)h;
    return (void*)LoadLibraryA("C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll");
}
void  lib_close(void* h) { FreeLibrary((HMODULE)h); }
void* lib_sym(void* h, const char* name) { return (void*)GetProcAddress((HMODULE)h, name); }
#else
void* lib_open() {
    // .so.1 is the versioned name the driver installs; the unversioned one only exists
    // when a toolkit is present.
    for (const char* so : {"libnvidia-ml.so.1", "libnvidia-ml.so"})
        if (void* h = dlopen(so, RTLD_LAZY | RTLD_LOCAL)) return h;
    return nullptr;
}
void  lib_close(void* h) { dlclose(h); }
void* lib_sym(void* h, const char* name) { return dlsym(h, name); }
#endif

template<class F> void bind(F& fn, const char* name) { fn = (F)lib_sym(g_lib, name); }

} // namespace

bool nvml_init() {
    if (g_ready) return true;
    g_lib = lib_open();
    if (!g_lib) return false;

    bind(p_init,     "nvmlInit_v2");
    bind(p_shutdown, "nvmlShutdown");
    bind(p_handle,   "nvmlDeviceGetHandleByIndex_v2");
    bind(p_count,    "nvmlDeviceGetCount_v2");
    bind(p_power,    "nvmlDeviceGetPowerUsage");
    bind(p_clock,    "nvmlDeviceGetClockInfo");
    bind(p_temp,     "nvmlDeviceGetTemperature");
    bind(p_fan,      "nvmlDeviceGetFanSpeed");
    bind(p_driver,   "nvmlSystemGetDriverVersion");
    bind(p_name,     "nvmlDeviceGetName");
    bind(p_pci,      "nvmlDeviceGetPciInfo_v3");
    if (!p_pci) bind(p_pci, "nvmlDeviceGetPciInfo_v2");
    bind(p_energy,         "nvmlDeviceGetTotalEnergyConsumption");
    bind(p_memclocks,      "nvmlDeviceGetSupportedMemoryClocks");
    bind(p_pl_get,         "nvmlDeviceGetPowerManagementLimit");
    bind(p_pl_default,     "nvmlDeviceGetPowerManagementDefaultLimit");
    bind(p_pl_constraints, "nvmlDeviceGetPowerManagementLimitConstraints");
    bind(p_pl_set,         "nvmlDeviceSetPowerManagementLimit");
    // Clock and fan control. All optional: an older driver simply will not have
    // some of these symbols, and the knob then reports Unsupported rather than
    // failing the whole NVML init -- telemetry must keep working on a card that
    // cannot be overclocked.
    bind(p_coff_get,   "nvmlDeviceGetGpcClkVfOffset");
    bind(p_coff_set,   "nvmlDeviceSetGpcClkVfOffset");
    bind(p_coff_range, "nvmlDeviceGetGpcClkMinMaxVfOffset");
    bind(p_moff_get,   "nvmlDeviceGetMemClkVfOffset");
    bind(p_moff_set,   "nvmlDeviceSetMemClkVfOffset");
    bind(p_moff_range, "nvmlDeviceGetMemClkMinMaxVfOffset");
    bind(p_maxclock,   "nvmlDeviceGetMaxClockInfo");
    bind(p_lock_core,  "nvmlDeviceSetGpuLockedClocks");
    bind(p_unlock_core,"nvmlDeviceResetGpuLockedClocks");
    bind(p_lock_mem,   "nvmlDeviceSetMemoryLockedClocks");
    bind(p_unlock_mem, "nvmlDeviceResetMemoryLockedClocks");
    bind(p_numfans,    "nvmlDeviceGetNumFans");
    bind(p_fan_set,    "nvmlDeviceSetFanSpeed_v2");
    bind(p_fan_auto,   "nvmlDeviceSetDefaultFanSpeed_v2");
    bind(p_fan_range,  "nvmlDeviceGetMinMaxFanSpeed");
    if (!p_init || !p_handle) { lib_close(g_lib); g_lib = nullptr; return false; }

    if (p_init() != NVML_SUCCESS)              { lib_close(g_lib); g_lib = nullptr; return false; }
    if (p_handle(0, &g_dev) != NVML_SUCCESS)   { if (p_shutdown) p_shutdown();
                                                 lib_close(g_lib); g_lib = nullptr; return false; }
    // Every device, in NVML's own order -- which is by PCI bus id, the same key
    // CudaSolver::enumerate() sorts on, so index N means the same card in both.
    g_devs.clear();
    g_devs.push_back(g_dev);
    if (p_count) {
        unsigned n = 0;
        if (p_count(&n) == NVML_SUCCESS) {
            for (unsigned i = 1; i < n; ++i) {
                nvmlDevice_t d = nullptr;
                if (p_handle(i, &d) == NVML_SUCCESS) g_devs.push_back(d);
            }
        }
    }
    g_ready = true;
    return true;
}

void nvml_shutdown() {
    if (!g_ready) return;
    if (p_shutdown) p_shutdown();
    if (g_lib) lib_close(g_lib);
    g_lib = nullptr; g_dev = nullptr; g_devs.clear(); g_ready = false;
}

unsigned nvml_device_count() { return g_ready ? (unsigned)g_devs.size() : 0u; }

Telemetry nvml_sample(unsigned index) {
    Telemetry t;
    if (!g_ready) return t;
    nvmlDevice_t g_dev = dev_at(index);
    if (!g_dev) return t;
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

std::string nvml_pci_address(unsigned index) {
    nvmlDevice_t g_dev = dev_at(index);
    if (!g_ready || !p_pci || !g_dev) return std::string();
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

namespace {
PowerLimit read_power_limit(nvmlDevice_t dev) {
    PowerLimit p;
    if (!g_ready || !p_pl_get || !dev) return p;
    unsigned v = 0;
    if (p_pl_get(dev, &v) != NVML_SUCCESS) return p;
    p.current_w = nvml_mw_to_w(v);
    p.valid = true;
    // The default and the constraints are separately optional: a card can
    // report its current limit and refuse the rest. Leaving them 0 lets the
    // caller tell "unknown" from "known and equal to the current value".
    if (p_pl_default && p_pl_default(dev, &v) == NVML_SUCCESS) p.default_w = nvml_mw_to_w(v);
    unsigned lo = 0, hi = 0;
    if (p_pl_constraints && p_pl_constraints(dev, &lo, &hi) == NVML_SUCCESS) {
        p.min_w = nvml_mw_to_w(lo);
        p.max_w = nvml_mw_to_w(hi);
    }
    return p;
}
} // namespace

PowerLimit nvml_power_limit(unsigned index) { return read_power_limit(dev_at(index)); }

std::string nvml_device_name(unsigned index) {
    nvmlDevice_t d = dev_at(index);
    if (!g_ready || !p_name || !d) return "";
    char buf[96] = {0};                       // NVML_DEVICE_NAME_V2_BUFFER_SIZE
    if (p_name(d, buf, sizeof buf) != NVML_SUCCESS) return "";
    return buf;
}

std::vector<unsigned> nvml_supported_mem_clocks(unsigned index) {
    std::vector<unsigned> out;
    nvmlDevice_t d = dev_at(index);
    if (!g_ready || !p_memclocks || !d) return out;
    // The count is in/out: on INSUFFICIENT_SIZE it holds the required capacity,
    // so retry once at that size instead of silently returning nothing (which
    // would make --tune skip its rung pass on a card with a long clock list).
    unsigned n = 32;
    std::vector<unsigned> clocks(n, 0);
    nvmlReturn_t rc = p_memclocks(d, &n, clocks.data());
    if (rc == NVML_ERROR_INSUFFICIENT_SIZE && n > 32) {
        clocks.assign(n, 0);
        rc = p_memclocks(d, &n, clocks.data());
    }
    if (rc != NVML_SUCCESS || n > clocks.size()) return out;
    out.assign(clocks.begin(), clocks.begin() + n);
    return out;
}

bool nvml_total_energy_mj(unsigned long long& mj, unsigned index) {
    nvmlDevice_t d = dev_at(index);
    if (!g_ready || !p_energy || !d) return false;
    unsigned long long v = 0;
    if (p_energy(d, &v) != NVML_SUCCESS) return false;
    mj = v;
    return true;
}

NvmlWrite nvml_set_power_limit(unsigned index, unsigned watts) {
    nvmlDevice_t d = dev_at(index);
    if (!g_ready || !p_pl_set || !d) return NvmlWrite::Unsupported;
    return map_write(p_pl_set(d, watts * 1000u));
}

// --- clock offsets -------------------------------------------------------

namespace {
// Offsets are read the same way for both domains; only the symbols differ.
ClockOffset read_offset(nvmlDevice_t dev,
                        nvmlReturn_t (*get)(nvmlDevice_t, int*),
                        nvmlReturn_t (*range)(nvmlDevice_t, int*, int*)) {
    ClockOffset o;
    if (!g_ready || !get || !dev) return o;
    int v = 0;
    if (get(dev, &v) != NVML_SUCCESS) return o;
    o.current_mhz = v;
    o.valid = true;
    // The band is separately optional. Leaving it 0/0 lets the caller tell
    // "no clamp available" from "clamped to zero", which are not the same --
    // and an unclamped write is the driver's problem to reject, not ours to
    // guess at.
    int lo = 0, hi = 0;
    if (range && range(dev, &lo, &hi) == NVML_SUCCESS) { o.min_mhz = lo; o.max_mhz = hi; }
    return o;
}
} // namespace

ClockOffset nvml_core_clock_offset(unsigned index) { return read_offset(dev_at(index), p_coff_get, p_coff_range); }
ClockOffset nvml_mem_clock_offset(unsigned index)  { return read_offset(dev_at(index), p_moff_get, p_moff_range); }

NvmlWrite nvml_set_core_clock_offset(unsigned index, int mhz) {
    nvmlDevice_t d = dev_at(index);
    if (!g_ready || !p_coff_set || !d) return NvmlWrite::Unsupported;
    return map_write(p_coff_set(d, mhz));
}
NvmlWrite nvml_set_mem_clock_offset(unsigned index, int mhz) {
    nvmlDevice_t d = dev_at(index);
    if (!g_ready || !p_moff_set || !d) return NvmlWrite::Unsupported;
    return map_write(p_moff_set(d, mhz));
}

// --- locked clocks -------------------------------------------------------

unsigned nvml_max_core_clock_mhz(unsigned index) {
    unsigned v = 0;
    nvmlDevice_t d = dev_at(index);
    if (!g_ready || !p_maxclock || !d) return 0;
    return p_maxclock(d, /*NVML_CLOCK_SM=*/1, &v) == NVML_SUCCESS ? v : 0;
}
unsigned nvml_max_mem_clock_mhz(unsigned index) {
    unsigned v = 0;
    nvmlDevice_t d = dev_at(index);
    if (!g_ready || !p_maxclock || !d) return 0;
    return p_maxclock(d, /*NVML_CLOCK_MEM=*/2, &v) == NVML_SUCCESS ? v : 0;
}

NvmlWrite nvml_set_locked_core_clock(unsigned index, unsigned lo, unsigned hi) {
    nvmlDevice_t d = dev_at(index);
    if (!g_ready || !p_lock_core || !d) return NvmlWrite::Unsupported;
    return map_write(p_lock_core(d, lo, hi));
}
NvmlWrite nvml_set_locked_mem_clock(unsigned index, unsigned lo, unsigned hi) {
    nvmlDevice_t d = dev_at(index);
    if (!g_ready || !p_lock_mem || !d) return NvmlWrite::Unsupported;
    return map_write(p_lock_mem(d, lo, hi));
}
NvmlWrite nvml_reset_locked_core_clock(unsigned index) {
    nvmlDevice_t d = dev_at(index);
    if (!g_ready || !p_unlock_core || !d) return NvmlWrite::Unsupported;
    return map_write(p_unlock_core(d));
}
NvmlWrite nvml_reset_locked_mem_clock(unsigned index) {
    nvmlDevice_t d = dev_at(index);
    if (!g_ready || !p_unlock_mem || !d) return NvmlWrite::Unsupported;
    return map_write(p_unlock_mem(d));
}

// --- fans ----------------------------------------------------------------

FanInfo nvml_fans(unsigned index) {
    FanInfo f;
    nvmlDevice_t g_dev = dev_at(index);
    if (!g_ready || !g_dev) return f;
    unsigned n = 0;
    // A card with no readable fan count but a readable speed is still a card
    // with one fan -- laptops and blower cards do this -- so fall back rather
    // than declaring the knob unavailable.
    if (p_numfans && p_numfans(g_dev, &n) == NVML_SUCCESS && n > 0) f.count = n;
    unsigned pct = 0;
    if (p_fan && p_fan(g_dev, &pct) == NVML_SUCCESS) {
        f.pct = pct;
        if (f.count == 0) f.count = 1;
    }
    unsigned lo = 0, hi = 0;
    if (p_fan_range && p_fan_range(g_dev, &lo, &hi) == NVML_SUCCESS) {
        f.min_pct = lo; f.max_pct = hi;
    }
    f.valid = f.count > 0;
    return f;
}

NvmlWrite nvml_set_fan_speed(unsigned index, unsigned fan, unsigned pct) {
    nvmlDevice_t d = dev_at(index);
    if (!g_ready || !p_fan_set || !d) return NvmlWrite::Unsupported;
    return map_write(p_fan_set(d, fan, pct));
}
NvmlWrite nvml_reset_fan(unsigned index, unsigned fan) {
    nvmlDevice_t d = dev_at(index);
    if (!g_ready || !p_fan_auto || !d) return NvmlWrite::Unsupported;
    return map_write(p_fan_auto(d, fan));
}

}} // namespace mxbm::gpu
