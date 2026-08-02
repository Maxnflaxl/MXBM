# MXBM

**An open-source GPU miner for [Beam](https://beam.mw/) (BeamHash III).**

MXBM is a from-scratch implementation of the BeamHash III proof-of-work —
Beam's Equihash-style Wagner search on the ⟨144,5⟩ parameter shape (see
[docs/beamhash/beamhash-iii.md](docs/beamhash/beamhash-iii.md)) — built to be a fully open,
auditable alternative to the closed-source miners in the ecosystem.

> **Status: GPU solver working, optimization ongoing.** MXBM mines against a real
> Beam pool over TLS: live jobs in, verified solutions out, shares accepted. On an
> RTX 4070 Ti SUPER the CUDA backend does **59.8 sol/s** and the portable OpenCL
> one **59.4** — within 1.2 % of each other, measured in the same session.
> `--solver auto` prefers CUDA, falls back to OpenCL, then to a CPU reference
> solver. Measured history, hardware limits and the caveats on comparing miners:
> **[docs/performance.md](docs/performance.md)**.
>
> **Where MXBM wins, and where it does not.** Both MXBM and lolMiner can be given a
> board power limit, and both were swept against each other at identical caps
> ([the numbers](docs/performance.md#both-miners-under-the-same-cap)):
>
> | board cap | MXBM | lolMiner 1.98a | |
> |---|---|---|---|
> | 180 W | 45.2 sol/s · 0.251 sol/s/W | **52.4 sol/s · 0.291 sol/s/W** | lolMiner ahead on both |
> | 220 W | **54.9 sol/s · 0.250 sol/s/W** | 54.4 sol/s · 0.248 sol/s/W | MXBM ahead on both |
> | 285 W | **59.1 sol/s** · 0.208 sol/s/W | 53.7 sol/s · **0.226 sol/s/W** | faster vs more efficient |
>
> Both columns come from one interleaved session (2026-07-28), which is the only way the
> comparison means anything; MXBM has gained ~1 % since, from work that does not change
> its shape.
>
> **MXBM has the higher ceiling — 59.1 sol/s against ~54.0, which lolMiner cannot reach
> at any setting — and it leads on both speed and efficiency between roughly 210 W and
> 256 W.** Outside that window lolMiner is the better choice, and at the efficient end it
> is clearly so: it gives up only 2.4 % of its speed for 24 % less power, so its best
> efficiency beats MXBM's best by 16 %. Closing that is
> [the current priority](docs/performance.md#why-we-lose-the-low-end-watts-buy-us-less-clock).
>
> If you run MXBM, **`--pl 220` is the setting to use** — its own efficiency peaks near
> 200 W and its best speed-per-watt against the alternative is around 220. Needs root;
> restored on exit. See [usage.md](docs/usage.md#power-limit).

---

## Features

- **BeamHash III proof-of-work** implemented from a verified specification,
  differentially tested against Beam's reference `IsValidSolution` over tens of
  thousands of fuzzed inputs.
- **Real-pool stratum client** — TLS-first, wallet-address authentication,
  nonce-prefix partitioning, automatic reconnect with backoff, and failover across
  several pools. Verified live against [HeroMiners](https://beam.herominers.com/).
- **Multi-GPU, including mixed rigs** — one solver, worker thread and nonce lane per
  card, so no two ever try the same nonce; `--devices` and `--list-devices` select by an
  index that means the same card in every flag. Cards that need different backends run
  in the **same process**: the CUDA, OpenCL and NVML device lists are joined on PCI bus
  id, so each card runs on its fastest viable backend (CUDA on Ampere and newer, else
  OpenCL) and one neither can drive is skipped with the reason. Each card gets its own
  row in the statistics table and in `/summary`, with the rig's totals underneath.
- **Runs unattended** — a watchdog that spots a card which has stopped completing
  solves and exits with a code a supervisor can restart on (or runs your script), and
  overclock control (`--pl`, `--cclk`, `--mclk`, `--coff`, `--moff`, `--fan`) clamped
  to the bands your driver reports and restored when MXBM exits.
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
  [docs/performance.md](docs/performance.md) and
  [docs/performance-research.md](docs/performance-research.md).
- **Apple Silicon native** — a Metal backend running the fused row-bucket
  pipeline at **101 ms/solve (18.8 sol/s)** on an M3 Max, ~5× the OpenCL path,
  which cannot run these kernels at all.
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

In rough priority order:

| Next | What it delivers |
|------|------------------|
| **Efficiency at low power** | lolMiner holds 0.2991 sol/s/W at 175 W where MXBM peaks at 0.2575 on stock memory, because [we lose core clock under a cap](docs/performance.md#why-we-lose-the-low-end-watts-buy-us-less-clock). **Half of that gap has since been closed by the memory clock**: below ~173 W the 5001 MHz rung is worth 8.5–14.4 %, which moves MXBM's own record to [0.264 sol/s/W — 3.79 J/solution at 160 W](docs/performance.md#below-stock-the-other-rung-pays-85-to-144--under-caps-below-173-w-and-a-new-efficiency-record) and roughly halves the 120–160 W deficit. The rest is open: [traffic supplies at most 17 % of the clock deficit](docs/performance.md#-closed-2026-07-29-the-low-end-is-not-reachable-by-traffic-and-no-other-mechanism-has-been-found), so a store-everything eco pipeline is the remaining candidate |
| **Smaller footprint** | 7.46 GiB against a 3 GB design target ([HW_REQUIREMENTS.md](docs/HW_REQUIREMENTS.md)). The [24 B quad record](docs/performance-research.md#the-quad-record-29--footprint-and-the-byte-prize-does-not-survive-re-derivation) has taken the floor to **4.66 GiB** and the stated requirement from 8 GB to **6 GB** — on *both* backends since the [record-set split](docs/performance-research.md#the-record-set-split-opencl-reaches-cudas-57-gib-floor-and-gets-faster-doing-it) lifted OpenCL's single-allocation ceiling. What is left is the 3 GB target itself, which needs streaming / in-place layer reuse |
| **HIP backend (AMD)** | Not started. Both solvers are measured on NVIDIA only; AMD is untested |
| **Per-GPU verification** | Multi-GPU and mixed-backend rigs are built and unit-tested, but have never run on a machine with more than one card. The device join has 45 assertions and needs no GPU; what needs hardware is per-card power control, the skip path on a real unsupported card, and two backends live in one process |

GPU support targets both NVIDIA and AMD: an OpenCL baseline that runs on both,
then vendor-tuned backends per vendor.

## The algorithm

BeamHash III is documented here in full, written from Beam's published
specifications rather than from folklore:

| Doc | Covers |
|-----|--------|
| **[BeamHash III](docs/beamhash/beamhash-iii.md)** | The current algorithm, the one MXBM implements — seeding, the per-round mixing step, the combination schedule, and the solution format, cross-referenced to the source |
| [BeamHash II](docs/beamhash/beamhash-ii.md) | The EquihashR family and the `r` parameter (block 321321 → 777777) |
| [BeamHash I](docs/beamhash/beamhash-i.md) | Beam's launch PoW, plus the Equihash and Wagner background the other two build on |

## Architecture

MXBM is organized as small, independently testable units — a portable
proof-of-work core, a stratum client, a difficulty filter, a solver behind a
single interface (the seam the GPU backends plug into), and the CLI/console/API
shell. See **[docs/architecture.md](docs/architecture.md)** for the full map.

## Performance

The GPU solver's measured state — speed, power curves, hardware limits and the open
leads — is **[docs/performance.md](docs/performance.md)**; the experiment log behind it,
every change and every failure with the mechanism that explains it, is
**[docs/performance-research.md](docs/performance-research.md)**. Hardware
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
