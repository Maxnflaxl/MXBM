# `src/` — MXBM source

C++17. Each subdirectory is a focused unit with a well-defined interface. For the
full design and the proof-of-work details, see
[`docs/architecture.md`](../docs/architecture.md).

| Directory | Responsibility |
|-----------|----------------|
| `beamhash/` | BeamHash III proof-of-work core (portable, GPU-ready) and host-side verifier |
| `pow/` | Beam-ported difficulty target test + SHA-256 share filter |
| `stratum/` | Pool client — wire messages, TCP+TLS transport, connection state machine |
| `miner/` | `Solver` interface (the GPU seam), reference solver, mining engine, metrics |
| `ui/` | Console output, value formatting, periodic statistics ticker |
| `cli/` | Command-line option parsing |
| `config/` | Configuration-file loaders (JSON profiles + flat) |
| `api/` | The `/summary` HTTP endpoint |
| `version.*` | Build-time git-stamped version string |
| `main.cpp` | Wires the pieces into the `mxbm` binary |

GPU device backends (OpenCL/CUDA/HIP) will live behind the `miner/` `Solver`
interface; see the [roadmap](../README.md#roadmap).
