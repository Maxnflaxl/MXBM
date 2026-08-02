// Device identity across CUDA, OpenCL and NVML -- the one part of the mixed-rig
// plan that can be proven on a machine with no GPU in it, which is exactly why it
// is the part that carries the risk. A misjoin lands a power cap or a tune verdict
// on the wrong card, so every case below is a case where the join must REFUSE
// rather than guess: missing PCI extension, mismatched names, two cards on one
// bus, NVML in a different order.
#include "gpu/device_join.h"
#include "check.h"
#include <string>
#include <vector>
using namespace mxbm;
using namespace mxbm::gpu;

namespace {

CudaCard cu(const char* name, const char* pci, int index, bool viable) {
    CudaCard c; c.name = name; c.pci = pci; c.index = index; c.viable = viable;
    c.global_mem = 16ull << 30;
    return c;
}
ClCard clc(const char* name, const char* pci, unsigned index) {
    ClCard c; c.name = name; c.pci = pci; c.index = index;
    c.global_mem = 16ull << 30;
    return c;
}
NvmlCard nv(const char* name, const char* pci) {
    NvmlCard c; c.name = name; c.pci = pci;
    return c;
}

const char* kAda = "NVIDIA GeForce RTX 4070 Ti SUPER";
const char* kTuring = "NVIDIA GeForce GTX 1660 Ti";
const char* k3080 = "NVIDIA GeForce RTX 3080";

} // namespace

int main() {
    section("pci_bus");
    {
        check(pci_bus("1:0") == "1", "bus of 1:0");
        check(pci_bus("a:0") == "a", "bus of a:0 (hex)");
        // Half an address is not a key: a producer that emits no colon gets no
        // match rather than a guess about which half it meant.
        check(pci_bus("1").empty(), "an address with no colon yields no bus");
        check(pci_bus("").empty(), "an empty address yields no bus");
    }

    section("clean join: one CUDA card, one OpenCL device, NVML agreeing");
    {
        const auto t = join_devices({cu(kAda, "1:0", 0, true)}, {clc(kAda, "1:0", 0)},
                                    {nv(kAda, "1:0")}, "auto");
        check(t.size() == 1, "one card in, one card out");
        check(t[0].backend == Backend::Cuda, "a viable card goes to CUDA");
        check(t[0].cuda_index == 0 && t[0].cl_index == 0, "both backends' indices are kept");
        check(!t[0].degraded && t[0].reason.empty(), "a clean join reports nothing");
    }

    section("the mixed rig: viable card on CUDA, old card on OpenCL, one process");
    {
        const auto t = join_devices({cu(kAda, "1:0", 0, true), cu(kTuring, "2:0", 1, false)},
                                    {clc(kAda, "1:0", 0), clc(kTuring, "2:0", 1)},
                                    {nv(kAda, "1:0"), nv(kTuring, "2:0")}, "auto");
        check(t.size() == 2, "both cards are in the table");
        check(t[0].backend == Backend::Cuda, "the Ampere+ card mines on CUDA");
        check(t[1].backend == Backend::OpenCL && t[1].cl_index == 1,
              "the card CUDA cannot drive mines on OpenCL, at ITS OpenCL index");
        check(t[1].reason.empty(), "and it is not reported as a problem, because it mines");
    }

    section("the watchdog trap: a non-viable card with no OpenCL identity is skipped");
    {
        const auto t = join_devices({cu(kAda, "1:0", 0, true), cu(kTuring, "2:0", 1, false)},
                                    {clc(kAda, "1:0", 0)},   // no OpenCL device on bus 2
                                    {}, "auto");
        check(t[1].backend == Backend::None, "no backend claims it");
        check(!t[1].reason.empty(), "and the skip says why");
        // Constructing it anyway is the defect this whole table exists to prevent:
        // the ctor succeeds, the first kernel launch fails, and the watchdog takes
        // the healthy cards down with it.
        check(t[1].cuda_index == 1, "its CUDA index is still recorded, for the message");
    }

    section("missing extension: OpenCL reports no PCI, so nothing is joined");
    {
        const auto t = join_devices({cu(kAda, "1:0", 0, true), cu(kTuring, "2:0", 1, false)},
                                    {clc(kAda, "", 0), clc(kTuring, "", 1)}, {}, "auto");
        check(t[0].backend == Backend::Cuda && t[0].cl_index < 0,
              "the viable card still mines, on CUDA, with no OpenCL match");
        check(t[1].backend == Backend::None, "the non-viable card is NOT guessed onto index 1");
        check(t[1].reason.find("cl_nv_device_attribute_query") != std::string::npos,
              "and the reason names the missing extension");
    }

    section("name mismatch on the same bus: refuse, never guess");
    {
        const auto t = join_devices({cu(kAda, "1:0", 0, false)},
                                    {clc(k3080, "1:0", 0)}, {}, "auto");
        check(t[0].cl_index < 0, "a bus match with a different name is not a match");
        check(t[0].degraded, "the card is marked degraded");
        check(t[0].backend == Backend::None && !t[0].reason.empty(),
              "and it is skipped with the ambiguity named");
    }

    section("two identical cards: the bus id tells them apart");
    {
        const auto t = join_devices({cu(k3080, "1:0", 0, true), cu(k3080, "41:0", 1, true)},
                                    {clc(k3080, "41:0", 0), clc(k3080, "1:0", 1)},
                                    {nv(k3080, "1:0"), nv(k3080, "41:0")}, "auto");
        // The OpenCL list is in the OTHER order, which is the whole point: same
        // name, different bus, and the indices must not be paired by position.
        check(t[0].cl_index == 1, "card at bus 1 joins the OpenCL device at bus 1");
        check(t[1].cl_index == 0, "card at bus 41 joins the OpenCL device at bus 41");
        check(!t[0].degraded && !t[1].degraded, "neither is ambiguous");
    }

    section("two OpenCL devices on one bus: ambiguous, so neither is used");
    {
        const auto t = join_devices({cu(kAda, "1:0", 0, false)},
                                    {clc(kAda, "1:0", 0), clc(kAda, "1:1", 1)}, {}, "auto");
        check(t[0].cl_index < 0 && t[0].degraded, "the match is refused");
        check(t[0].reason.find("bus 1") != std::string::npos, "the reason names the bus");
    }

    section("hex case: NVML's busId is not lowercase and CUDA's %x is");
    {
        const auto t = join_devices({cu(k3080, "a:0", 0, true)}, {clc(k3080, "A:0", 0)},
                                    {nv(k3080, "A:0")}, "auto");
        check(t[0].cl_index == 0, "bus a and bus A are one bus");
        check(!t[0].degraded, "and NVML at the same address does not read as disagreement");
    }

    section("NVML in a different order: the card still mines, the identity is noted");
    {
        const auto t = join_devices({cu(kAda, "1:0", 0, true), cu(k3080, "41:0", 1, true)},
                                    {clc(kAda, "1:0", 0), clc(k3080, "41:0", 1)},
                                    {nv(k3080, "41:0"), nv(kAda, "1:0")}, "auto");
        check(t[0].degraded && t[1].degraded, "both positions disagree with NVML");
        check(t[0].backend == Backend::Cuda && t[1].backend == Backend::Cuda,
              "CUDA-viable cards keep mining -- CUDA identity does not depend on NVML");
        check(t[0].cl_index < 0, "but the OpenCL join is refused while identity is in doubt");
        check(t[0].reason.find("NVML") != std::string::npos,
              "and the reason says so, because --pl aims by position");
    }

    section("--solver cuda: CUDA-capable cards only, the rest skipped");
    {
        const auto t = join_devices({cu(kAda, "1:0", 0, true), cu(kTuring, "2:0", 1, false)},
                                    {clc(kAda, "1:0", 0), clc(kTuring, "2:0", 1)}, {}, "cuda");
        check(t[0].backend == Backend::Cuda, "the viable card runs");
        check(t[1].backend == Backend::None, "the other is skipped even though OpenCL could");
        check(t[1].reason.find("--solver cuda") != std::string::npos, "and the reason says why");
    }

    section("--solver opencl: the portable path everywhere, including on viable cards");
    {
        const auto t = join_devices({cu(kAda, "1:0", 0, true), cu(kTuring, "2:0", 1, false)},
                                    {clc(kAda, "1:0", 0), clc(kTuring, "2:0", 1)}, {}, "opencl");
        check(t[0].backend == Backend::OpenCL && t[0].cl_index == 0,
              "a CUDA-viable card is forced onto OpenCL");
        check(t[1].backend == Backend::OpenCL, "so is the old one");
    }

    section("--solver opencl with no OpenCL identity: skipped, not silently CUDA");
    {
        const auto t = join_devices({cu(kAda, "1:0", 0, true)}, {}, {}, "opencl");
        check(t[0].backend == Backend::None && !t[0].reason.empty(),
              "forcing a path the card has no identity for is a skip, not a fallback");
    }

    section("no CUDA at all: the OpenCL list IS the table, in its own order");
    {
        const auto t = join_devices({}, {clc("gfx1100", "3:0", 0), clc("gfx1100", "c:0", 1)},
                                    {}, "auto");
        check(t.size() == 2, "both OpenCL devices are listed");
        check(t[0].backend == Backend::OpenCL && t[0].cl_index == 0
           && t[1].cl_index == 1, "indices are the OpenCL ones, unreordered");
        check(t[0].cuda_index < 0, "with no CUDA index to offer");
    }

    section("no CUDA at all, --solver cuda: nothing runs, and it says so");
    {
        const auto t = join_devices({}, {clc("gfx1100", "3:0", 0)}, {}, "cuda");
        check(t[0].backend == Backend::None && !t[0].reason.empty(),
              "--solver cuda on a rig with no CUDA card is a skip with a reason");
    }

    section("empty everything");
    {
        check(join_devices({}, {}, {}, "auto").empty(), "no devices in, no devices out");
        check(join_devices({}, {}, {nv(kAda, "1:0")}, "auto").empty(),
              "NVML alone is not a device list: it cannot construct a solver");
    }

    section("an unknown --solver value claims nothing");
    {
        // --solver ref is the CPU oracle: the table must not hand it a card.
        const auto t = join_devices({cu(kAda, "1:0", 0, true)}, {clc(kAda, "1:0", 0)}, {}, "ref");
        check(t[0].backend == Backend::None, "no GPU backend claims a card for --solver ref");
    }

    return summary("device_join");
}
