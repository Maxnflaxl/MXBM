# Architecture

MXBM is built as a set of small, focused units with well-defined interfaces,
each testable on its own. The design goal is a fully auditable miner: every
proof-of-work primitive is validated against Beam's reference, and the GPU
backends (when they land) slot in behind a single solver interface without
touching the networking or UI code.

## Source layout

```
src/
  beamhash/   BeamHash III proof-of-work core (portable, GPU-ready) + verifier
  pow/        Difficulty test (ported from Beam) + SHA-256 share filter
  stratum/    Pool client: wire messages, TCP+TLS transport, connection state machine
  miner/      Solver interface, reference solver, mining engine, metrics
  ui/         Console output, formatting, and the periodic stats ticker
  cli/        Command-line option parsing
  config/     Configuration-file loaders (JSON profiles + flat)
  api/        The /summary HTTP endpoint
  version.*   Build-time git-stamped version string
  main.cpp    Wires it all together into the mxbm binary
third_party/  Vendored: nlohmann/json (MIT), BLAKE2 (CC0), SHA-256 (public domain)
tests/        CTest suite; golden vectors under tests/vectors/
```

## The proof-of-work core

`src/beamhash/` implements BeamHash III — Beam's Equihash-style Wagner search on
the ⟨144,5⟩ parameter shape — from a verified specification. The core primitives
(a non-standard SipHash element generator, the seed step, the mixing and combine
steps, and 25-bit index packing) are written as a portable, dependency-free
header so the same logic can later become a GPU kernel body. A host-side verifier
reproduces Beam's `IsValidSolution`.

The algorithm itself is documented separately, from Beam's published
specifications: **[beamhash-iii.md](beamhash-iii.md)** for the algorithm MXBM
implements, with [beamhash-i.md](beamhash-i.md) and
[beamhash-ii.md](beamhash-ii.md) covering the two predecessors and the Equihash
background they build on.

Correctness is established two ways: against golden known-answer vectors
(generated from Beam's own solver, since Beam ships none), and — when a Beam
checkout is provided at build time — differentially against Beam's reference
verifier over tens of thousands of fuzzed inputs.

## The stratum client

`src/stratum/` speaks Beam's line-delimited stratum protocol:

- **messages** — (de)serialization of `login` / `job` / `solution` / `result`,
  ported from Beam and validated against golden wire vectors.
- **transport** — a blocking, line-framed TCP transport with optional OpenSSL
  TLS.
- **client** — the connection state machine: log in, capture the pool's
  nonce-prefix, dispatch jobs to callbacks, and reconnect with backoff on drop.

## The mining pipeline

`src/miner/` turns jobs into submissions:

- **`Solver`** is a one-method interface: given a header and a nonce, return
  candidate solutions. This is the seam the GPU backends plug into. The current
  reference solver wraps Beam's `OptimisedSolve`; future OpenCL/CUDA/HIP solvers
  implement the same interface.
- **`Engine`** owns the pipeline: it takes the latest job from a mailbox on a
  worker thread, runs the solver, filters each candidate through the difficulty
  test, and submits those that clear.
- **`Stats`** collects windowed hash-rate, share counts, and latency for the
  console and API.

## Threading model

The running miner uses a small number of cooperating threads: the client's
read loop, the engine's solver worker, the stats ticker, and (if enabled) the
HTTP API's accept loop. Shared state is confined to the `Stats` object behind a
single mutex; each thread has a clear owner and a clean shutdown path.

## Difficulty

`src/pow/` ports Beam's packed-difficulty target test (`IsTargetReached`) exactly
— a fixed-width big-integer comparison, no elliptic-curve dependency — over a
vendored SHA-256. The same packed value is also rendered in human-readable units
for the console and API, matching what pools and other miners display.

## Testing

Every unit has a focused test. The suite runs with `ctest` and needs no network
or GPU. Golden vectors (wire messages, proof-of-work KATs, SHA-256 and difficulty
vectors) are checked in under `tests/vectors/` and `tests/`. See
[building.md](building.md) for how to run them, including the optional Beam
differential oracle.

## GPU solver performance

The GPU collision-finding pipeline — its two paths, the per-round schedule, and the
measured hardware limits that shape it — is documented in
[performance.md](performance.md), which also keeps the running log of every
optimization and every failed experiment.
