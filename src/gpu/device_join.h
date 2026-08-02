#pragma once
// One physical card, seen by every subsystem at once.
//
// CUDA enumerates in PCI order, NVML by bus id, OpenCL per platform with no PCI
// notion at all unless the NVIDIA attribute query answers. Driving a mixed rig
// from one process means knowing when index N in one list and index M in another
// are the SAME card: a power cap or a tune verdict landing on the wrong card is
// the silent-wrongness class this codebase works hardest to exclude.
//
// A pure function of three plain lists, so every case -- clean join, missing
// extension, name mismatch, two identical cards, empty lists -- is testable on a
// machine with no GPU in it. Plan of record: docs-internal/MIXED_RIG.md.
#include <string>
#include <vector>

namespace mxbm { namespace gpu {

enum class Backend { None, Cuda, OpenCL };

// CudaSolver::enumerate(), which is already PCI-sorted. `pci` is "bus:device" in
// hex with no leading zeros ("1:0"), the short form NVML prints.
struct CudaCard {
    std::string name, pci;
    int index = 0;                      // CUDA's own device index for this card
    unsigned long long global_mem = 0;
    bool viable = false;                // compute >= 8.0 AND a geometry that fits
};

// Runtime::enumerate(), in the flattened order GpuSolver's index counts. `pci` is
// empty when cl_nv_device_attribute_query is absent (any non-NVIDIA platform).
struct ClCard {
    std::string name, pci;
    unsigned index = 0;
    unsigned long long global_mem = 0;
};

// nvml_device_name()/nvml_pci_address() at each NVML position; position is the
// vector index. Empty strings where NVML did not answer.
struct NvmlCard {
    std::string name, pci;
};

struct JoinedCard {
    std::string name, pci;
    unsigned long long global_mem = 0;
    Backend backend = Backend::None;    // what will actually drive it; None = skipped
    int  cuda_index = -1;               // >= 0 when the CUDA path can construct on it
    int  cl_index   = -1;               // >= 0 when the OpenCL path can construct on it
    bool viable = false;                // the CUDA path can drive it
    bool degraded = false;              // identity ambiguous: CUDA-list identity only
    // Why this card is skipped, or why its identity is degraded. Empty on a clean
    // card. Written to be printed after "Device N (name): ".
    std::string reason;
};

// Joins the three views on PCI bus id and applies the per-card backend rule.
//
// `solver` is the --solver value: "auto"/"gpu" pick per card (CUDA where it can
// drive the card, else OpenCL), "cuda" runs CUDA-capable cards and skips the rest,
// "opencl" forces the portable path everywhere, anything else claims nothing.
// Indexed the way --devices and --pl already are: the CUDA list's PCI order when
// there is one, else the OpenCL list's own order.
std::vector<JoinedCard> join_devices(const std::vector<CudaCard>& cuda,
                                     const std::vector<ClCard>& cl,
                                     const std::vector<NvmlCard>& nvml,
                                     const std::string& solver);

// The bus field of a "bus:device" address ("1:0" -> "1"), or "" if there is none.
// Exposed because it is the join key: a test that cannot see it can only check the
// join's conclusions, not its premise.
std::string pci_bus(const std::string& pci);

}} // namespace mxbm::gpu
