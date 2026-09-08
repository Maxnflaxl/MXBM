# MXBM

**An open-source GPU miner for [Beam](https://beam.mw/) (BeamHash III).**

A from-scratch implementation of Beam's proof-of-work — an Equihash-style Wagner search
on the ⟨144,5⟩ parameter shape — built to be a fully open, auditable alternative to the
closed-source miners in the ecosystem. It mines against a real pool over TLS: live jobs
in, verified solutions out, shares accepted.

> **Status: working, optimization ongoing.** On an RTX 4070 Ti SUPER the CUDA backend
> does **74.40 sol/s** (27.00 ms/solve) and the portable OpenCL one **63.0**.

## Quick start

```sh
# Build
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j

# Mine (replace with your own BEAM wallet address)
./build/mxbm --algo BEAM-III \
             --pool beam.herominers.com:1130 \
             --user <your-beam-address>.<worker-name>
```

Dependencies and platform notes are in **[docs/building.md](docs/building.md)**; the full
command line, config files and API are in **[docs/usage.md](docs/usage.md)**.

**Cap the board.** MXBM runs pinned at the power limit in every kernel, so the limit
picks the operating point. `--pl 210` is the reference card's efficiency peak;
`sudo mxbm --tune` measures yours. Needs root, restored on exit.

## How it compares

Both miners swept against each other at identical board caps, interleaved in one session
([the full table](docs/performance.md#both-miners-under-the-same-cap)):

| board cap | MXBM | lolMiner 1.98a |
|---|---|---|
| 180 W | **54.0 sol/s** · 0.300 sol/s/W | 53.3 sol/s · 0.296 sol/s/W |
| 210 W | **64.9 sol/s** · 0.309 sol/s/W | 54.2 sol/s · 0.259 sol/s/W |
| 285 W | **70.0 sol/s** · 0.246 sol/s/W | 53.8 sol/s · 0.229 sol/s/W |

MXBM has the higher ceiling — **70.0 sol/s against ~54.4**, which lolMiner cannot reach
at any setting — and it leads on both speed and efficiency from ~177 W to the 285 W
stock limit. Below that lolMiner is ahead on stock memory, at worst by 7.5 % at 160 W.
With each miner at its best memory clock the band splits into two narrow strips: 2–3 %
below 115 W and under 1 % from 155–177 W, with MXBM ahead by 8.5 % at 120 W and 3.5 % at
140 W between them ([the low band](docs/performance.md#the-low-band-on-the-current-kernel)).
Best efficiency is a tie at ~3.22 J/solution, and at its point MXBM does 30 % more work.

## Features

- **BeamHash III from a verified specification** — differentially tested against Beam's
  reference `IsValidSolution` over tens of thousands of fuzzed inputs.
- **Real-pool stratum client** — TLS-first, wallet-address authentication, nonce-prefix
  partitioning, reconnect with backoff, failover across several pools. Verified live
  against [HeroMiners](https://beam.herominers.com/).
- **Multi-GPU, including mixed rigs.** One solver, worker thread and nonce lane per card,
  so no two ever try the same nonce. Cards needing different backends run in the same
  process: the CUDA, OpenCL and NVML device lists are joined on PCI bus id, so each card
  takes its fastest viable backend and one neither can drive is skipped with the reason.
- **Runs unattended** — a watchdog that spots a card which has stopped completing solves
  and exits with a code a supervisor can restart on, plus overclock control (`--pl`,
  `--cclk`, `--mclk`, `--coff`, `--moff`, `--fan`) clamped to the driver's bands and
  restored on exit.
- **Familiar UX** — a lolMiner-shaped console, a compatible command-line surface, and
  both configuration-file formats.
- **Dashboard and `/summary` API** — `--apiport N` serves a live browser dashboard
  (hashrate, share-difficulty scatter, power with an optional energy-cost axis, clocks,
  temperature) alongside a JSON endpoint. Self-contained: no CDN, so it works on an
  isolated rig.
- **CUDA and OpenCL solvers** — a fused row-bucket Wagner pipeline finding all five
  rounds' collisions in local memory, best backend selected automatically, falling back
  to the CPU reference. Every optimization is gated on byte-identical known-answer
  solutions.
- **Apple Silicon native** — a Metal backend at **101 ms/solve (18.8 sol/s)** on an
  M3 Max, ~5× the OpenCL path, which cannot run these kernels at all.
- **Self-contained** — no Boost, no Beam runtime. A vendored single-header JSON library
  and your system OpenSSL.

## Developer fee

MXBM takes a **1.0% developer fee** — one 36-second round per 60 minutes of mining,
about 14.4 minutes a day. It is the same rate the closed-source miners charge, so an
open miner costs you no more, and it is what funds development.

It is announced at startup and at each round, reported on its own row of the statistics
table and in `/summary`, and kept in a separate ledger, so it never touches your own
share counts, best share, or pool-credited rate. Fee rounds log in under your own worker
name plus the rate (`rig1_1`). `--dev-fee PCT` raises it; it cannot be lowered.

Full terms are in **[docs/devfee.md](docs/devfee.md)**.

## Donations

The fee funds MXBM by default. To give more: raise it with `--dev-fee PCT`, send Beam to
the BANS name **`maxnflaxl.beam`**, or point a rig at the developer's address — an
ordinary [HeroMiners](https://beam.herominers.com/) address, so any miner can mine to it:

```sh
./build/mxbm --algo BEAM-III \
             --pool beam.herominers.com:1130 \
             --user 12cafbe121b5f063d2c63152058575479a2826a41fd4176296dae2e8ad3fc9ffc60.<worker-name>
```

## Documentation

| Doc | Covers |
|---|---|
| [building.md](docs/building.md) | Dependencies, platform notes, the tests |
| [usage.md](docs/usage.md) | The command line, config files, console, multi-GPU |
| [overclocking.md](docs/overclocking.md) | The six knobs, what to set, and how to tell instability apart from a failure rate |
| [api.md](docs/api.md) | The dashboard and the `/summary` JSON |
| [devfee.md](docs/devfee.md) | The fee in full |
| [HW_REQUIREMENTS.md](docs/HW_REQUIREMENTS.md) | VRAM thresholds and what each card class gets |
| [benchmarks.md](docs/benchmarks.md) | Measured results: both miners, power curves, where each millisecond goes |
| [benchmarking.md](docs/benchmarking.md) | How to measure a card, and the protocol that settles a cross-miner comparison |
| [architecture.md](docs/architecture.md) | How the code is organised |
| [performance.md](docs/performance.md) | The permanent record — every figure, with its measurement |
| [performance-research.md](docs/performance-research.md) | Every optimization tried, working or not, with its mechanism |
| [beamhash-iii.md](docs/beamhash/beamhash-iii.md) | The algorithm, written from Beam's published specifications ([II](docs/beamhash/beamhash-ii.md), [I](docs/beamhash/beamhash-i.md)) |

Running MXBM on a card that is not yet
[listed](docs/benchmarks.md#benchmarked-devices)? **`mxbm --report`** benchmarks it,
measures its power curve and prints a paste-ready block — one command, no checkout,
nothing uploaded — and there is an issue template waiting for it.

> Comparing miners is harder than it looks: reported `sol/s` is implementation-defined,
> and MXBM measures a 17 % spread between "solutions found" and "solutions that verify"
> inside its own pipeline. It reports the conservative number.
> [benchmarking.md](docs/benchmarking.md) has the method.

## Roadmap

- **More speed, and more of it under a power cap.** Performance work is ongoing. The
  low-power band is where the remaining ground is: MXBM leads from ~177 W up and the gap
  below that is down to two narrow strips ([the low
  band](docs/performance.md#the-low-band-on-the-current-kernel)).
- **AMD, via a HIP backend.** Not started. Both solvers are measured on NVIDIA today, and
  AMD cards run the portable OpenCL path.

## Contributing

Issues and pull requests are welcome. See
[docs/contributing.md](docs/contributing.md) for the build/test workflow and code style.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). MXBM builds on Beam,
which is itself Apache-2.0 licensed.
