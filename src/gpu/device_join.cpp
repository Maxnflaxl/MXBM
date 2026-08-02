#include "gpu/device_join.h"
#include <cctype>

namespace mxbm { namespace gpu {
namespace {

std::string lower(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

// Trim and case-fold: the same card's name arrives as three different vendors'
// strings and none of them is ours.
std::string norm_name(const std::string& s) {
    size_t a = s.find_first_not_of(" \t");
    if (a == std::string::npos) return {};
    size_t b = s.find_last_not_of(" \t");
    return lower(s.substr(a, b - a + 1));
}

bool same_bus(const std::string& a, const std::string& b) {
    const std::string x = pci_bus(a), y = pci_bus(b);
    return !x.empty() && !y.empty() && lower(x) == lower(y);
}

const char* kNoPciFromCl =
    "the OpenCL driver reports no PCI address for its devices "
    "(cl_nv_device_attribute_query is missing), so no OpenCL device can be matched "
    "to this card";
const char* kNotCudaCapable =
    "the CUDA path needs compute capability 8.0 or newer and a geometry that fits";

} // namespace

std::string pci_bus(const std::string& pci) {
    const size_t c = pci.find(':');
    // No colon means the address is not the "bus:device" form every producer here
    // emits, and half an address is not a key.
    return c == std::string::npos ? std::string() : pci.substr(0, c);
}

std::vector<JoinedCard> join_devices(const std::vector<CudaCard>& cuda,
                                     const std::vector<ClCard>& cl,
                                     const std::vector<NvmlCard>& nvml,
                                     const std::string& solver) {
    const bool want_cuda   = (solver == "auto" || solver == "gpu" || solver == "cuda");
    const bool want_opencl = (solver == "auto" || solver == "gpu" || solver == "opencl");
    const bool cl_has_pci = [&] {
        for (const ClCard& c : cl) if (!c.pci.empty()) return true;
        return false;
    }();

    std::vector<JoinedCard> out;

    // NVML is consulted by POSITION, as --pl, --tune and the telemetry rows already
    // do. Disagreement there means a cap aimed at this card lands on another one.
    auto nvml_disagrees = [&](size_t pos, const std::string& pci) {
        return pos < nvml.size() && !nvml[pos].pci.empty() && !pci.empty()
            && !same_bus(nvml[pos].pci, pci);
    };

    if (!cuda.empty()) {
        for (size_t i = 0; i < cuda.size(); ++i) {
            const CudaCard& c = cuda[i];
            JoinedCard j;
            j.name = c.name;
            j.pci = c.pci;
            j.global_mem = c.global_mem;
            j.viable = c.viable;
            j.cuda_index = c.index;

            // Recorded before the backend rule, so the rule can report it as the
            // reason a card is skipped.
            std::string amb;
            if (nvml_disagrees(i, c.pci)) {
                j.degraded = true;
                amb = "NVML reports PCI " + nvml[i].pci + " at this position but the card is at "
                      + c.pci + " - the two device orders disagree, so per-card power settings "
                      "and any OpenCL match for this card cannot be trusted";
            } else if (want_opencl) {
                // The bus alone: unique per card, where the slot field's encoding
                // is driver-dependent (cl_runtime's pci_of). The name backs it up.
                int found = -1, matches = 0;
                for (const ClCard& d : cl)
                    if (same_bus(d.pci, c.pci)) { ++matches; if (found < 0) found = (int)d.index; }
                if (matches > 1) {
                    j.degraded = true;
                    amb = "two OpenCL devices report PCI bus " + pci_bus(c.pci)
                        + " - identity is ambiguous, so neither is used for this card";
                } else if (matches == 1) {
                    const ClCard* m = nullptr;
                    for (const ClCard& d : cl) if ((int)d.index == found) m = &d;
                    if (m && norm_name(m->name) == norm_name(c.name)) {
                        j.cl_index = found;
                    } else {
                        j.degraded = true;
                        amb = "the OpenCL device on PCI bus " + pci_bus(c.pci) + " calls itself \""
                            + (m ? m->name : std::string()) + "\" where CUDA reports \"" + c.name
                            + "\" - identity is ambiguous, so it is not used for this card";
                    }
                } else if (!cl.empty() && !cl_has_pci) {
                    amb = kNoPciFromCl;
                }
            }

            // --solver cuda skips what CUDA cannot drive rather than quietly using
            // the slower path; --solver opencl forces it even where CUDA would win.
            if (solver == "cuda") {
                if (j.viable) j.backend = Backend::Cuda;
                else j.reason = std::string(kNotCudaCapable)
                              + " - --solver cuda was requested, so this card is skipped";
            } else if (solver == "opencl") {
                if (j.cl_index >= 0) j.backend = Backend::OpenCL;
                else j.reason = amb.empty()
                        ? "no OpenCL device could be matched to this card"
                        : amb;
            } else if (want_cuda && j.viable) {
                j.backend = Backend::Cuda;
                if (j.degraded) j.reason = amb;      // mines, but the identity is noted
            } else if (want_opencl && j.cl_index >= 0) {
                j.backend = Backend::OpenCL;
            } else if (want_cuda || want_opencl) {
                j.reason = amb.empty()
                    ? std::string(kNotCudaCapable) + ", and no OpenCL device could be matched "
                      "to this card"
                    : std::string(kNotCudaCapable) + ", and " + amb;
            }
            out.push_back(std::move(j));
        }
        return out;
    }

    // No CUDA devices: the OpenCL list is the table, in its own flattened order --
    // the order --devices already means on a rig with no CUDA in it.
    for (size_t i = 0; i < cl.size(); ++i) {
        const ClCard& c = cl[i];
        JoinedCard j;
        j.name = c.name;
        j.pci = c.pci;
        j.global_mem = c.global_mem;
        j.cl_index = (int)c.index;
        if (nvml_disagrees(i, c.pci)) {
            j.degraded = true;
            j.reason = "NVML reports PCI " + nvml[i].pci + " at this position but the device is at "
                     + c.pci + " - per-card power settings may not land on this card";
        }
        if (want_opencl) j.backend = Backend::OpenCL;
        else if (solver == "cuda")
            j.reason = "--solver cuda was requested but no CUDA device is present";
        out.push_back(std::move(j));
    }
    return out;
}

}} // namespace mxbm::gpu
