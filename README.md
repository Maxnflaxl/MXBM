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
> RTX 4070 Ti SUPER the CUDA backend does **58.0 sol/s** and the portable OpenCL
> one 48.2. `--solver auto` prefers CUDA, falls back to OpenCL, then to a CPU
> reference solver. Measured history, hardware limits and the caveats on comparing
> miners: **[docs/performance.md](docs/performance.md)**.
>
> **Where MXBM wins, and where it does not.** Both MXBM and lolMiner can be given a
> board power limit, and both were swept against each other at identical caps
> ([the numbers](docs/performance.md#both-miners-under-the-same-cap)):
>
> | board cap | MXBM | lolMiner 1.98a | |
> |---|---|---|---|
> | 180 W | 46.1 sol/s · 0.256 sol/s/W | **52.7 sol/s · 0.293 sol/s/W** | lolMiner ahead on both |
> | 220 W | **55.4 sol/s · 0.252 sol/s/W** | 54.4 sol/s · 0.248 sol/s/W | MXBM ahead on both |
> | 285 W | **59.2 sol/s** · 0.208 sol/s/W | 53.6 sol/s · **0.227 sol/s/W** | faster vs more efficient |
>
> **MXBM has the higher ceiling — 59.2 sol/s against ~54.4, which lolMiner cannot reach
> at any setting — and it leads on both speed and efficiency between roughly 212 W and
> 257 W.** Outside that window lolMiner is the better choice, and at the efficient end it
> is clearly so: it gives up only 1.8 % of its speed for 21 % less power, so its best
> efficiency beats MXBM's best by 12 %. Closing that is
> [the current priority](docs/performance.md#why-we-lose-the-low-end-watts-buy-us-less-clock).
>
> If you run MXBM, **`--pl 220` is the setting to use** — its own efficiency peaks near
> 200 W and its best speed-per-watt against the alternative is around 220. Needs root;
> restored on exit. See [usage.md](docs/usage.md#power-limit).

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
| Multi-GPU | One solver, thread and nonce lane per card; `--devices` / `--list-devices` | ✅ done, untested on >1 card |
| Memory efficiency | Run on ≤ 8 GB cards (see [HW_REQUIREMENTS.md](docs/HW_REQUIREMENTS.md)) | next |
| Overclocking | `--pl`, `--cclk`, `--mclk`, `--coff`, `--moff`, `--fan` — clamped to the driver's own bands, restored on exit (see [overclocking.md](docs/overclocking.md)) | ✅ done |
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

**[docs/benchmarks.md](docs/benchmarks.md)** publishes the measured results —
MXBM against lolMiner 1.98a on the same card, the power/efficiency curve, and
where each millisecond goes — plus how to reproduce any of it. `benchmarks/`
holds the harnesses: `power_bench.sh` (sol/s and J/sol), `stage_power.sh`
(per-kernel time and power attribution), `power_sweep.sh` (the speed/power
curve; needs root) and `collect_report.sh` (a paste-ready report).

**MXBM has only ever been measured on one GPU.** If you run it on anything else,
`benchmarks/collect_report.sh` produces a report in one command and there is an
issue template waiting for it — results from hardware we do not have are the
single most useful contribution to the project right now.

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
