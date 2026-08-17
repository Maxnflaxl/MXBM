# MXBM

**An open-source GPU miner for [Beam](https://beam.mw/) (BeamHash III).**

MXBM is a from-scratch implementation of the BeamHash III proof-of-work —
Beam's Equihash-style Wagner search on the ⟨144,5⟩ parameter shape (see
[docs/beamhash/beamhash-iii.md](docs/beamhash/beamhash-iii.md)) — built to be a fully open,
auditable alternative to the closed-source miners in the ecosystem.

> **Status: GPU solver working, optimization ongoing.** MXBM mines against a real
> Beam pool over TLS: live jobs in, verified solutions out, shares accepted. On an
> RTX 4070 Ti SUPER the CUDA backend does **69.20 sol/s** (29.10 ms/solve, the
> locked-clock pin, re-taken 2026-08-16) and the portable OpenCL one **63.0**
> (31.9 ms, 2026-08-17, 1.10× of the same-session CUDA).
>
> **Where MXBM wins, and where it does not.** Both miners were swept against each other
> at identical board power limits, interleaved in one session — the only way the
> comparison means anything ([the numbers](docs/performance.md#both-miners-under-the-same-cap)):
>
> | board cap | MXBM | lolMiner 1.98a | |
> |---|---|---|---|
> | 180 W | 52.1 sol/s · 0.290 sol/s/W | **52.4 sol/s · 0.291 sol/s/W** | level, lolMiner by 0.5 % |
> | 210 W | **63.1 sol/s · 0.301 sol/s/W** | 53.9 sol/s · 0.257 sol/s/W | MXBM ahead on both |
> | 285 W | **69.6 sol/s · 0.245 sol/s/W** | 53.7 sol/s · 0.226 sol/s/W | MXBM ahead on both |
>
> **MXBM has the higher ceiling — 69.6 sol/s against ~54.0, which lolMiner cannot reach
> at any setting — and it leads on both speed and efficiency from roughly 181 W to the
> 285 W stock limit.** (lolMiner's column is the 2026-07-30 sweep; MXBM's was swept end
> to end 2026-08-16.)
> Below that window lolMiner is still ahead, and its best cap is 140 W at +8 %, but the
> gap narrows again toward the floor and is **gone by 100 W**, where the two are level;
> closing the middle of that band is
> [the current priority](docs/performance.md#why-we-lose-the-low-end-watts-buy-us-less-clock).
> The two miners' best efficiency points are now level — 0.3009 sol/s/W at 210 W against
> 0.2991 at 175 — and there MXBM does 21 % more work.
>
> **`--pl 210` is the setting to use.** Needs root; restored on exit. See
> [usage.md](docs/usage.md#power-limit).

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
mining, about 14.4 minutes a day.

It is announced at startup and at each round, reported on its own row of the
statistics table and in `/summary`, and kept in a separate ledger so it never
touches your own share counts, best share, or pool-credited rate. Fee rounds
log in under your own worker name plus the rate (`rig1_1`), so they are
identifiable as yours. `--dev-fee PCT` raises the rate if you want to support
the project with more; it cannot lower it.

The fee is what funds MXBM's development. It is the same rate the closed-source miners charge, so an open miner costs you no more.

Full terms, including exactly what is and is not counted, are in
**[docs/devfee.md](docs/devfee.md)**.

## Donations

The developer fee funds MXBM by default. If you would like to give more, there
are three ways:

- **Point a rig at the developer's address.** It is an ordinary Beam address on
  [HeroMiners](https://beam.herominers.com/), so any miner can mine to it:

  ```sh
  ./build/mxbm --algo BEAM-III \
               --pool beam.herominers.com:1130 \
               --user 12cafbe121b5f063d2c63152058575479a2826a41fd4176296dae2e8ad3fc9ffc60.<worker-name>
  ```

- **Send Beam directly** to that same address, or to the BANS name
  **`maxnflaxl.beam`**.

- **Raise the fee** with `--dev-fee PCT`, if you would rather it came out of
  normal mining than out of a separate run.

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
| **Efficiency at low power** | lolMiner did ~2.3× more useful work per core cycle under a cap, because it stores state where MXBM re-derives it and [under a cap the currency is instructions issued](docs/performance.md#-below-150-w-the-clock-gap-inverts-and-the-clock-explanation-stops-applying). That is now **1.4–1.7×**, and the throughput gap on stock memory is **−2 to −11 % across 100–160 W**, from −20 to −30 %. With each miner on its own best memory clock the band is no longer a deficit at all: MXBM leads at 120 and 140 W, ties at 160, and trails 5.7 % only at the 100 W floor. Two levers did most of it, both by deleting instructions rather than bytes: the [w0-checkpoint record](docs/performance-research.md#the-w0-checkpoint-pair-record-repriced-by-the-address-bits-076-ms-24-) and the [replayed reference rows](docs/performance-research.md#the-back-reference-rows-are-gone-recovery-replays-instead-203-ms-and-688-mib) — each worth more under a cap than at stock. The obvious counter-design was built and measured dead: a [storing round 1 pays back in scattered L2 sectors exactly what the saved rebuild wins](docs/performance-research.md#the-h1-pipeline-built-and-killed-the-caps-currency-is-l2-sectors-not-dram-bytes). What remains is the issue economy itself |
| **Smaller footprint** | **Done for now: a 1.90 GiB floor on both backends against a 3 GB design target** ([HW_REQUIREMENTS.md](docs/HW_REQUIREMENTS.md)) — the target and the card class behind it both cleared, from 2.4× over at the start. The [24 B quad record](docs/performance-research.md#the-quad-record-29--footprint-and-the-byte-prize-does-not-survive-re-derivation), the implicit-bits record, the [dense-cap rungs](docs/performance-research.md#the-reach-composition-built-the-implicit-bits-reclaim-and-the-arena-as-a-rung) and the [32 B octo record](docs/performance-research.md#the-octo-record-the-last-thing-round-3s-output-had-left-to-give) got it there, and every rung below the packed ones then got **4–9 % faster at no footprint cost, on both backends** when its record started carrying [the child's word 0](docs/performance-research.md#the-w0-checkpoint-on-the-quad-record-the-record-holds-word-0-with-no-repacking), which deletes every `apply_mix` in the rebuild that reads it. Above the octo rungs, [recovery replays rounds 3 and 4](docs/performance-research.md#the-back-reference-rows-are-gone-recovery-replays-instead-203-ms-and-688-mib) instead of storing a reference row per child, which takes the default rung to 6.17 GiB. **OpenCL floors at the same 1.90 GiB** — it stores all five rows above the octo rungs, and on them there is only one row to store |
| **HIP backend (AMD)** | Not started. Both solvers are measured on NVIDIA only; AMD is untested |
| **Per-GPU verification** | A two-card rig (3060 Ti + 4070 SUPER) ran [21 minutes on a pool](docs/performance.md#third-party-hardware--a-two-card-rig-2026-08-02), 42 shares, 42 accepted, and found two defects in per-card power control that are now fixed. Both cards chose CUDA, so the case the join exists for — two *different* backends live in one process — is still unexercised, as is the skip path on a card no backend can drive |

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

Running MXBM on a card that is not yet
[listed](docs/benchmarks.md#benchmarked-devices)? `benchmarks/collect_report.sh`
produces a paste-ready report in one command, and there is an issue template waiting
for it.

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
