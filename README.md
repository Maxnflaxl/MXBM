# MXBM

**An open-source GPU miner for [Beam](https://beam.mw/) (BeamHash III).**

MXBM is a clean-room, from-scratch implementation of the BeamHash III
proof-of-work — the Equihash ⟨150,5⟩ variant Beam uses — built to be a fully
open, auditable alternative to the closed-source miners in the ecosystem. The
proof-of-work core is validated bit-for-bit against Beam's own reference
implementation.

> **Status: GPU solver working, optimization ongoing.** MXBM connects to a real
> Beam pool over TLS, authenticates with your wallet address, receives live jobs,
> and runs the full job → solve → difficulty → submit pipeline on an **OpenCL GPU
> solver**. On an RTX 4070 Ti SUPER it does **22.8 sol/s** (83.2 ms per solve) — up
> from 1.8 sol/s when the solver first found a share — verified against the BeamHash III
> known-answer vectors. That is
> roughly 2.3× off the fastest closed-source miner, and closing that gap is the
> current focus — see [docs/performance.md](docs/performance.md) for the full
> measured optimization history.
>
> A CPU reference solver remains available (`--solver ref`) for validating the
> pipeline. **Note the VRAM requirement is currently high** (~16 GB for a search
> that finds solutions, against a real footprint of 6.95 GiB) — see
> [HW_REQUIREMENTS.md](docs/HW_REQUIREMENTS.md), which also
> documents the known limitations behind that number. Builds and runs on Linux and
> macOS.

Licensed under the [Apache License 2.0](LICENSE).

---

## Features

- **BeamHash III proof-of-work** implemented from a verified specification,
  differentially tested against Beam's reference `IsValidSolution` over tens of
  thousands of fuzzed inputs.
- **Real-pool stratum client** — TLS-first, wallet-address authentication,
  nonce-prefix partitioning, automatic reconnect with backoff. Verified live
  against [HeroMiners](https://beam.herominers.com/).
- **lolMiner-shaped UX** — a familiar console (per-interval speed line, periodic
  statistics block, share/accept lines), a compatible command-line surface, and
  both configuration-file formats.
- **`/summary` HTTP API** for monitoring.
- **OpenCL GPU solver** — a fused row-bucket Wagner pipeline that finds all five
  rounds' collisions in local memory. Selected automatically when a suitable device
  is present, with automatic fallback. Every optimization is gated on byte-identical
  known-answer solutions; see [docs/performance.md](docs/performance.md).
- **Self-contained** — no Boost, no Beam runtime. The only dependencies are a
  vendored single-header JSON library and your system OpenSSL.

## Quick start

```sh
# Build
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j

# Run against a pool (replace with your own BEAM wallet address)
./build/mxbm --algo BEAM-III \
             --pool beam.herominers.com:1130 \
             --user <your-beam-address>.<worker-name>
```

See **[docs/building.md](docs/building.md)** for dependencies and platform
notes, and **[docs/usage.md](docs/usage.md)** for the full command-line, config
files, and API.

## Roadmap

| Milestone | What it delivers | Status |
|-----------|------------------|--------|
| Proof-of-work core | BeamHash III primitives + verifier, tested vs Beam | ✅ done |
| Stratum client | Connect, authenticate, receive jobs from a real pool | ✅ done |
| Miner shell | lolMiner-style console, CLI, config files, `/summary` API | ✅ done |
| GPU solver | OpenCL solver finding verified BeamHash III solutions | ✅ done |
| **Solver performance** | **Close the gap to the fastest closed-source miners** | **in progress** |
| Memory efficiency | Run on ≤ 8 GB cards (see [HW_REQUIREMENTS.md](docs/HW_REQUIREMENTS.md)) | next |
| Optimized backends | Tuned CUDA (NVIDIA) and HIP (AMD) kernels | planned |

GPU support targets both NVIDIA and AMD: an OpenCL baseline first (runs on
both), then vendor-tuned CUDA and HIP backends. The OpenCL solver is measured on
NVIDIA; AMD is untested so far.

## Architecture

MXBM is organized as small, independently testable units — a portable
proof-of-work core, a stratum client, a difficulty filter, a solver behind a
single interface (the seam the GPU backends plug into), and the CLI/console/API
shell. See **[docs/architecture.md](docs/architecture.md)** for the full map.

## Performance

The GPU solver's optimization history — every change, its measured effect, and the
experiments that failed — is tracked in **[docs/performance.md](docs/performance.md)**,
along with the measured hardware limits that bound further work. Hardware
requirements, including the current VRAM limitations, are in
**[HW_REQUIREMENTS.md](docs/HW_REQUIREMENTS.md)**.

## Contributing

Issues and pull requests are welcome. Please read
[docs/contributing.md](docs/contributing.md) for the build/test workflow and the
project's clean-room provenance rules (in short: MXBM contains no code copied
from any closed-source miner; Beam-derived code is Apache-2.0 and attributed in
[NOTICE](NOTICE)).

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). MXBM builds on
Beam, which is itself Apache-2.0 licensed.
