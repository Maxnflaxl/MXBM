# MXBM

**An open-source GPU miner for [Beam](https://beam.mw/) (BeamHash III).**

MXBM is a clean-room, from-scratch implementation of the BeamHash III
proof-of-work — the Equihash ⟨150,5⟩ variant Beam uses — built to be a fully
open, auditable alternative to the closed-source miners in the ecosystem. The
proof-of-work core is validated bit-for-bit against Beam's own reference
implementation.

> **Status: pre-GPU.** MXBM today is a complete, working stratum **client**: it
> connects to a real Beam pool over TLS, authenticates with your wallet address,
> receives live jobs, and runs the full job → solve → difficulty → submit
> pipeline on a CPU reference solver. The CPU solver is far too slow to clear
> pool difficulty — it exists to prove the pipeline end-to-end. The GPU solver
> that makes MXBM actually competitive is the next milestone (see
> [Roadmap](#roadmap)). It builds and runs on Linux and macOS.

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
| **GPU solver** | **OpenCL solver — the first accepted share** | **next** |
| Optimized backends | Tuned CUDA (NVIDIA) and HIP (AMD) kernels | planned |

GPU support targets both NVIDIA and AMD: an OpenCL baseline first (runs on
both), then vendor-tuned CUDA and HIP backends.

## Architecture

MXBM is organized as small, independently testable units — a portable
proof-of-work core, a stratum client, a difficulty filter, a solver behind a
single interface (the seam the GPU backends plug into), and the CLI/console/API
shell. See **[docs/architecture.md](docs/architecture.md)** for the full map.

## Contributing

Issues and pull requests are welcome. Please read
[docs/contributing.md](docs/contributing.md) for the build/test workflow and the
project's clean-room provenance rules (in short: MXBM contains no code copied
from any closed-source miner; Beam-derived code is Apache-2.0 and attributed in
[NOTICE](NOTICE)).

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). MXBM builds on
Beam, which is itself Apache-2.0 licensed.
