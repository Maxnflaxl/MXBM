/* Prints device 0's cumulative energy counter in millijoules and exits 0, or
 * exits 1 printing nothing. The whole program exists because this driver's
 * nvidia-smi does not expose the counter (checked 2026-07-31: `-q -d POWER`
 * prints draw and limits only), and the sweeps live in bash. Reading needs no
 * root. Built on demand by bench_energy_mj in lib.sh; kept dependency-free --
 * dlopen, five symbols, no NVML headers -- for the same reason src/gpu/nvml.cpp
 * is: the library ships with the driver, not with any toolkit. */
#include <dlfcn.h>
#include <stdio.h>

int main(void) {
    void* lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
    if (!lib) lib = dlopen("libnvidia-ml.so", RTLD_LAZY);
    if (!lib) return 1;
    int (*init)(void) = (int (*)(void))dlsym(lib, "nvmlInit_v2");
    int (*handle)(unsigned, void**) =
        (int (*)(unsigned, void**))dlsym(lib, "nvmlDeviceGetHandleByIndex_v2");
    int (*energy)(void*, unsigned long long*) =
        (int (*)(void*, unsigned long long*))dlsym(lib, "nvmlDeviceGetTotalEnergyConsumption");
    if (!init || !handle || !energy || init() != 0) return 1;
    void* dev = 0;
    unsigned long long mj = 0;
    if (handle(0, &dev) != 0 || energy(dev, &mj) != 0) return 1;
    printf("%llu\n", mj);
    return 0;
}
