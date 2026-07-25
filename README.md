# MXBM

**An open-source GPU miner for [Beam](https://beam.mw/) (BeamHash III).**

MXBM is a from-scratch implementation of the BeamHash III proof-of-work —
Beam's Equihash-style Wagner search on the ⟨144,5⟩ parameter shape (see
[docs/beamhash-iii.md](docs/beamhash-iii.md)) — built to be a fully open,
auditable alternative to the closed-source miners in the ecosystem. The
proof-of-work core is validated bit-for-bit against Beam's own reference
implementation.

> **Status: GPU solver working, optimization ongoing.** MXBM mines against a real
> Beam pool over TLS: live jobs in, verified solutions out, shares accepted. On an
> RTX 4070 Ti SUPER the CUDA backend does **56.5 sol/s** and the portable OpenCL
> one 48.2. `--solver auto` prefers CUDA, falls back to OpenCL, then to a CPU
> reference solver. Measured history, hardware limits and the caveats on comparing
> miners: **[docs/performance.md](docs/performance.md)**.
>
> **Cap the board power.** MXBM runs pinned at the card's power limit in every
> kernel, so at stock it looks less efficient than the reference miner purely
> because it uses watts the other cannot reach. Measured at **equal power** it is
> ahead on both axes — 55.5 sol/s at 239 W against 53.3 at 239 W, +3.9 % on
> sol/s/W — and dropping the limit 285 → 240 W costs only 3.5 % of throughput.
> See [Power and efficiency](docs/performance.md#power-and-efficiency).

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
- **Built-in dashboard and `/summary` HTTP API** — `--apiport N` serves a live
  browser dashboard (hashrate, per-share difficulty scatter, power with an
  optional energy-cost axis, clocks, temperature) alongside a JSON endpoint for
  scripts. Entirely self-contained: no CDN, no external assets, so it works on
  an isolated rig.
- **CUDA and OpenCL GPU solvers** — a fused row-bucket Wagner pipeline that finds all
  five rounds' collisions in local memory. The best available backend is selected
  automatically, with fallback all the way down to the CPU reference. Every optimization
  is gated on byte-identical known-answer solutions; see
  [docs/performance.md](docs/performance.md).
- **Self-contained** — no Boost, no Beam runtime. The only dependencies are a
  vendored single-header JSON library and your system OpenSSL.

## Developer fee

MXBM takes a **1.0% developer fee** — one 36-second round per 60 minutes of
mining, about 14.4 minutes a day. That is the same rate the closed-source
BeamHash III miners charge.

It is announced at startup and at each round, reported on its own row of the
statistics table and in `/summary`, and kept in a separate ledger so it never
touches your own share counts, best share, or pool-credited rate. Fee rounds
log in under your own worker name plus the rate (`rig1_1`), so they are
identifiable as yours. `--dev-fee PCT` raises the rate if you want to support
the project with more; it cannot lower it.

The fee is what funds MXBM's development: the solver work, the hardware it is
measured on, and keeping it working as pools and drivers move. It is the same
rate the closed-source miners charge, so an open miner costs you no more.

Full terms, including exactly what is and is not counted, are in
**[docs/devfee.md](docs/devfee.md)**.

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
| Miner shell | lolMiner-style console, CLI, config files, dashboard + `/summary` API | ✅ done |
| GPU solver | OpenCL solver finding verified BeamHash III solutions | ✅ done |
| Solver performance | Close the gap to the fastest closed-source miners | ✅ done (CUDA, +6 %) |
| Memory efficiency | Run on ≤ 8 GB cards (see [HW_REQUIREMENTS.md](docs/HW_REQUIREMENTS.md)) | next |
| Overclocking | NVML clock/power/fan control (see [overclocking.md](docs/overclocking.md)) | designed |
| Optimized backends | Tuned CUDA (NVIDIA) ✅ done; HIP (AMD) | in progress |

GPU support targets both NVIDIA and AMD: an OpenCL baseline (runs on both), then
vendor-tuned backends. The CUDA backend is done and shipping; HIP is not started. Both
solvers are measured on NVIDIA only — AMD is untested so far.

## The algorithm

BeamHash III is documented here in full, written from Beam's published
specifications rather than from folklore:

| Doc | Covers |
|-----|--------|
| **[BeamHash III](docs/beamhash-iii.md)** | The current algorithm, the one MXBM implements — seeding, the per-round mixing step, the combination schedule, and the solution format, cross-referenced to the source |
| [BeamHash II](docs/beamhash-ii.md) | The EquihashR family and the `r` parameter (block 321321 → 777777) |
| [BeamHash I](docs/beamhash-i.md) | Beam's launch PoW, plus the Equihash and Wagner background the other two build on |

## Architecture

MXBM is organized as small, independently testable units — a portable
proof-of-work core, a stratum client, a difficulty filter, a solver behind a
single interface (the seam the GPU backends plug into), and the CLI/console/API
shell. See **[docs/architecture.md](docs/architecture.md)** for the full map.

## Performance

The GPU solver's optimization history — every change, its measured effect, and the
experiments that failed — is tracked in **[docs/performance.md](docs/performance.md)**,
along with the measured hardware limits that bound further work. Hardware
requirements, including the per-backend VRAM thresholds, are in
**[HW_REQUIREMENTS.md](docs/HW_REQUIREMENTS.md)**.

`benchmarks/` holds the energy harnesses: `power_bench.sh` (sol/s and J/sol),
`stage_power.sh` (per-kernel time and power attribution) and `power_sweep.sh`
(the speed/power curve across board power limits; needs root).

Comparing miners is harder than it looks: reported `sol/s` is implementation-defined,
and MXBM measures a 17 % spread between "solutions found" and "solutions that verify"
inside its own pipeline. **[docs/benchmarking.md](docs/benchmarking.md)** documents what
MXBM counts (the conservative number), how to reproduce its figures, and the
accepted-pool-share protocol that settles a comparison without trusting either miner's
counters.

## Contributing

Issues and pull requests are welcome. Please read
[docs/contributing.md](docs/contributing.md) for the build/test workflow and the
code style.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). MXBM builds on
Beam, which is itself Apache-2.0 licensed.
