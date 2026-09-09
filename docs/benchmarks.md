# Benchmark results

Measured numbers for MXBM, and for lolMiner on the same card. Everything here can be
reproduced with the scripts in `benchmarks/`; the command is given under each table.
This page is the results. How to measure a miner without fooling yourself is in
[benchmarking.md](benchmarking.md).

MXBM has been measured on five GPUs: the RTX 4070 Ti SUPER it is developed on, an M3 Max
via Metal, and a contributed RTX 4070 SUPER, RTX 3060 Ti and GTX 1660 Ti. To add yours,
see [send us your numbers](#send-us-your-numbers).

---

## Reference card

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER (Ada, sm_89, 66 SMs) |
| VRAM | 16 GiB GDDR6X, 10251 MHz, 256-bit (656 GB/s peak, ~510 GB/s achievable) |
| Board power limit | 285 W default, 100–366 W permitted by the driver |
| Driver | 610.43.03 · CUDA 13.3 |
| OS | Linux 7.1.4 (Arch) |

Unless a row says otherwise, every figure on this page is this card at stock clocks, with
no overclock or undervolt.

---

## Benchmarked devices

One row per backend per card, each at the card's stock power cap.

| GPU | Memory | Driver / OS | Backend | sol/s | ms/solve | W | sol/s/W | Source |
|---|---|---|---|---|---|---|---|---|
| RTX 4070 Ti SUPER | 16 GiB GDDR6X | 610.43.03 · Linux | CUDA | 74.8 | 26.9 | 285 | 0.263 | ours, `--benchmark` ‡ |
| RTX 4070 Ti SUPER | 16 GiB GDDR6X | 610.43.03 · Linux | OpenCL | 63.2 | 31.8 | 284 | 0.222 | ours, `--benchmark` ‡ |
| RTX 4070 SUPER | 12 GiB GDDR6X | 610.88 · Windows | CUDA | 55.8 | 36.2 | 211 | 0.265 | ZumZum, `--report` 120 s ※ ◇ |
| RTX 3060 Ti | 8 GiB GDDR6 | 610.88 · Windows | CUDA | 26.5 | 75.4 | 192 | 0.138 | ZumZum, `--report` 120 s ※ |
| GTX 1660 Ti | 6 GiB GDDR6 | 595.97 · Windows | CUDA | 9.48 | 210.6 | 106 | 0.089 | ZumZum, `--report` 120 s ◇ ◆ |
| Apple M3 Max (40-core) | 128 GB unified | macOS 26.5 · Metal 3 | Metal | 16.3 | 128.0 | — | — | ours, `--benchmark` |

‡ CUDA is the 285 W point of the 2026-09-09 power sweep below (90 s, released clocks).
OpenCL was taken 2026-08-17 and its kernels are unchanged since. The published headline
is the **27.00 ms / 74.40 sol/s**
[locked-clock pin](performance-research.md#the-named-reference-lgc-2600) of 2026-09-09,
which trades a little boost for reproducibility.

※ 120 s `--report` on MXBM 0.8.408, 2026-08-21. The 3060 Ti runs at its thermal limit
(81 °C) at stock, so its top end is partly a cooling figure.

◇ Taken with other compute processes on the card: 26 on the 4070 SUPER, 13 plus a
display on the 1660 Ti. Indicative only; clean re-runs are wanted. See
[third-party hardware](performance.md#third-party-hardware--contributed-cards).

◆ The first Turing measurement. The same machine on 0.8.412 fell back to OpenCL at
4.28 sol/s / 462.9 ms, so the CUDA path is worth 2.2× there. The card never reached its
board limit and shows no interior efficiency optimum; that may be the contention, so it
is not yet read as a property of Turing.

The M3 Max's pipeline alone runs 101.5 ms/solve (18.8 sol/s); sustained is lower because
the laptop throttles. macOS exposes no GPU power API, so there is no power column.

![Speed and efficiency against the cap, every card measured](tools/cards-curve.svg)

Curves for the four NVIDIA cards, generated from the tables in
[performance.md](performance.md#third-party-hardware--contributed-cards). This is not a
controlled comparison: different machines, operating systems and instruments, so the
distances between curves are unreliable. What holds is each curve's shape and where its
own efficiency peak sits: 220 W on the 4070 Ti SUPER, 148 W on the 4070 SUPER, 130 W on
the 3060 Ti, all well under stock. The 1660 Ti's efficiency was still climbing at the
driver's 70 W floor, so its optimum is a bound, not a measurement.

---

## MXBM vs lolMiner 1.98a

Both mining BeamHash III on the same card, caps set with `nvidia-smi` so one instrument
measures both. At equal power, at MXBM's efficiency peak:

| | MXBM at 220 W | lolMiner 1.98a at 220 W | |
|---|---|---|---|
| Throughput | **69.9 sol/s** | 53.35 sol/s | **+31.0 %** |
| Board power | 219.7 W | 219.6 W | — |
| Efficiency | **0.3182 sol/s/W** | 0.2430 sol/s/W | **+30.9 %** |
| VRAM for a full search | 6.17 GiB[^4g] | ~4 GiB | |

MXBM is ahead on both from roughly 176 W to the 285 W stock limit, and has the higher
ceiling outright: 74.8 sol/s against ~54.4, which lolMiner cannot reach at any setting.
Below ~176 W on stock memory lolMiner is ahead: 0.2 % at 175 W, 6.3 % at 160 W, 9.0 % at
110 W, level again at the 100 W floor. With each miner at its best configuration the gap
is smaller; see [the low band](performance.md#the-low-band-on-the-current-kernel).

lolMiner was swept across the same caps as MXBM rather than sampled once uncapped, so this
is a comparison of two curves, not a curve against a point. That matters, because a
stock-versus-stock comparison rewards whichever miner fails to fill the card: MXBM draws
the full 285 W in every kernel, lolMiner draws 239 W uncapped and leaves 46 W unused.

One caveat the watts do not cover: the two miners' sol/s are separate counters. MXBM
reports CPU-verified solutions; lolMiner's basis is undocumented, and the distinction is
worth ~17 % inside our own pipeline
([why](benchmarking.md#1-why-reported-sols-is-not-comparable)). Accepted pool shares over
a fixed interval are the only arbiter that needs neither counter.

<details>
<summary>The 2026-07 stock-versus-stock session, on the build of 2026-08-13</summary>

| | MXBM (CUDA) | lolMiner 1.98a | |
|---|---|---|---|
| Throughput | **62.3 sol/s** | 53.27 sol/s | **+17 %** |
| Board power | 281.4 W | 238.7 W | +18 % |
| Efficiency | 0.221 sol/s/W | **0.223 sol/s/W** | −1 % |
| Core clock | 2610 MHz | 2745 MHz | |
| Memory clock | 10251 MHz | 10251 MHz | |
| Temperature | 67 °C | 60 °C | |

MXBM's column is 80 minutes against a pool (62.32 sol/s, ±0.11 on the mean); the
controlled benchmark the same day read 62.7 sol/s at 32.0 ms/solve. The kernel work since
has taken that to 74.40 sol/s at 27.00 ms, so every margin here is a floor on the current
one. lolMiner's binary is unchanged.

</details>

[^4g]: at the fastest geometry, which is what both miners run here. lolMiner selects
"BeamHash III **4G** (CUDA)" on this card; MXBM's own floor is 1.90 GiB on a card that
cannot host the fast rung, a class below the 4 GB one. See
[HW_REQUIREMENTS.md](HW_REQUIREMENTS.md#the-design-target-is-met).

---

## MXBM power curve

The board power limit is the most valuable setting on this card, because MXBM is pinned
against it. Measured 2026-09-09 on the shipped build and defaults with `power_sweep.sh`,
90 s per point, one session, headless:

| `--pl` | sol/s | ms/solve | measured | core clock | sol/s/W | J/solution |
|---|---|---|---|---|---|---|
| 100 W | 24.0 | 83.0 | 99.3 W | 780 MHz | 0.242 | 4.14 |
| 110 W | 27.8 | 71.3 | 109.5 W | 885 MHz | 0.254 | 3.93 |
| 120 W | 31.9 | 62.1 | 119.7 W | 960 MHz | 0.267 | 3.75 |
| 140 W | 39.2 | 50.8 | 139.7 W | 1230 MHz | 0.281 | 3.57 |
| 160 W | 47.0 | 42.4 | 160.1 W | 1500 MHz | 0.294 | 3.40 |
| 175 W | 52.8 | 38.2 | 175.0 W | 1725 MHz | 0.302 | 3.32 |
| 180 W | 54.7 | 36.8 | 180.0 W | 1785 MHz | 0.304 | 3.29 |
| 190 W | 58.4 | 34.4 | 189.8 W | 1935 MHz | 0.308 | 3.25 |
| 200 W | 62.3 | 32.3 | 199.9 W | 2100 MHz | 0.312 | 3.21 |
| 210 W | 66.5 | 30.3 | 209.9 W | 2325 MHz | 0.317 | 3.16 |
| **220 W** | **69.9** | 28.9 | 219.7 W | 2355 MHz | **0.318** | **3.14** |
| 240 W | 72.3 | 27.9 | 239.9 W | 2475 MHz | 0.301 | 3.32 |
| 255 W | 73.5 | 27.5 | 254.9 W | 2535 MHz | 0.288 | 3.47 |
| 270 W | 74.3 | 27.2 | 269.8 W | 2580 MHz | 0.275 | 3.63 |
| 285 W *(stock)* | **74.8** | 26.9 | 284.8 W | 2625 MHz | 0.263 | 3.81 |

The efficiency optimum on stock memory is **220 W**: 3.14 J/solution at 93 % of the stock
speed. Energy is read from the card's own energy counter over each 90 s window. At this
sample size the sol/s column reads about 1 % high (a 90 s run sees ~2.02 solutions per
solve where a long run measures 2.006); every point was taken the same way, so the shape,
the peak and the crossings are unaffected, and the figures are left as measured.

Swept against lolMiner at identical caps:

![Speed, efficiency and power drawn, both miners at the same caps](tools/power-curve.svg)

lolMiner is ahead on speed and efficiency between ~101 W and ~176 W on stock memory, by
6–9 % from 110 to 160 W, and level at the 100 W floor. From ~176 W up MXBM is ahead on
both, by 31 % at 220 W and 39 % at 285 W. lolMiner barely responds to a cap: its ceiling
is 54.4 sol/s at 200 W, and above ~235 W the cap does nothing. MXBM's best efficiency
(0.3182 sol/s/W, 3.14 J/solution) is 2 % above lolMiner's (0.3120 at 160 W on its own
counter), and at that point the same energy per solution does 69.9 sol/s against 49.95.

The lolMiner column is the 2026-08-18 session and the MXBM column the 2026-09-09 sweep;
a table from two sessions carries a cap-dependent term that reaches a few percent at the
low caps, and the MXBM rows below 140 W read 2–4 % low because of it. The full table,
the mechanism and how the term was sized are in
[performance.md](performance.md#both-miners-under-the-same-cap). Reproduce with
`benchmarks/compare_power.sh`; the chart is drawn from that table by
`python3 docs/tools/plot_power.py`.

Two things worth knowing before you cap your own card:

- **Efficiency peaks at 220 W and gets worse below it.** At 180 W the core clock has
  fallen to 1785 MHz and the parts of the board that do not scale with it, memory, uncore
  and leakage, are paid for out of less work. Lower is not always better, all the way
  down to the card's 100 W floor.
- **The last watts are the worst value.** 220 → 285 W buys 4.9 sol/s for 65 W; the 20 W
  from 200 to 220 buys 7.6. Where to sit is a question about your power price.

```sh
sudo mxbm --algo BEAM-III --pool ... --user ... --pl 220
```

`--pl` needs root. Without it MXBM says so and mines on at the card's current limit. The
previous limit is restored on exit, including on Ctrl+C. See
[usage.md](usage.md#power-limit).

---

## At the other end: the memory V/F offset

The memory rung below is what a capped rig wants. An uncapped one wants the opposite
knob. Rounds 3 and 4 are bandwidth-bound, so the memory clock is the only knob on the
resource that binds them, and the reference card has headroom above the 10251 MHz it holds
under load.

Take it with `--moff`, not `--mclk`. Locking to the driver's reported 10501 maximum
measures 0.95 % *slower*, because a locked P-state buys its clock with voltage and the
board is already at its power limit. The offset shifts the V/F curve instead. Bracketed
`A B B A`, 12 arms of 40 s, headless, drop counters zero on both arms, on the build of
2026-08-16 (the relative figures are what carries):

| | ms/solve | range | J/sol | sol/s |
|---|---|---|---|---|
| stock memory | 28.733 | 28.711 – 28.760 | 4.090 | 69.8 |
| `--moff 1600` (11051 MHz) | **28.050** | 28.027 – 28.066 | **4.008** | **71.5** |

**−2.37 % time and −2.00 % energy.** Both at once, because the board is pinned at its
limit: power is constant, so the energy win is the speed win. The sweep ran monotone to
+2400 (11451 MHz), but the gain stops tracking the clock past about +1600, which is the
retry wall arriving. [The mechanism](performance-research.md#locking-the-memory-clock-to-its-reported-maximum-costs-095--and-the-offset-is-the-knob-that-does-not);
[how to find your own](overclocking.md#finding-your-own-memory-offset).

This figure is excluded from every comparison on this page. MXBM moves 10.2 GB a solve
where a state-storing design moves 17.7, so the same clock is worth more to the miner
that stores: an overclocked head-to-head measures the overclock.

---

## Capped rigs: the memory rung and the low-power kernels

The curve above is stock memory. Under a cap that is the wrong configuration: the memory
clock does not scale with the core, so the interface burns a fixed slice of a small budget
for bandwidth nothing is using. Dropping it to the 5001 MHz rung returns those watts as
core clock, and below ~130 W MXBM also switches to a different geometry and rebuild
variant. Together that is MXBM's best configuration at each cap.

Swept 2026-09-09 on the shipping build: `-lmc 5001,5001` held across `power_sweep.sh`,
forward then reversed, 120 s points, shipped defaults. "vs stock memory" is against the
same day's stock-memory sweep; the comparison column is lolMiner at *its* own best
configuration per cap from the 2026-08-18 session (cross-session term about 1 % here).

| cap | ms/solve | sol/s | J/sol | sol/s/W | vs stock memory | lolMiner best | gap |
|---|---|---|---|---|---|---|---|
| 100 W | 70.1 | 28.45 | 3.505 | 0.285 | +18.5 % | 30.35 *(rung)* | **−6.3 %** |
| 110 W | 61.5 | 32.65 | 3.375 | 0.297 | +17.4 % | 33.95 *(rung)* | **−3.8 %** |
| 120 W | 55.1 | 36.45 | 3.300 | 0.303 | +14.3 % | 34.10 *(rung)* | **+6.9 %** |
| 140 W | 46.6 | 43.10 | 3.255 | 0.308 | +9.9 % | 41.50 *(stock)* | **+3.9 %** |
| **160 W** | 40.2 | **50.20** | **3.185** | **0.314** | +6.8 % | 49.95 *(stock)* | +0.5 % |
| 175 W | 37.3 | 53.90 | 3.220 | 0.311 | +2.1 % | 52.90 *(stock)* | +1.9 % |
| 180 W | 37.2 | 54.20 | 3.300 | 0.303 | −0.9 % | 53.25 *(stock)* | +2.7 % *(stock vs stock)* |

Take the rung only if you are hard-limited to ~175 W or below. It is worth +7 % at 160 W
rising to +18 % at the 100 W floor, and with each miner at its own best configuration MXBM
leads from 120 W up and trails only below ~114 W, by 4–6 %. It is not an efficiency
setting: its best point, 3.185 J/solution at 160 W, sits 1.4 % above the stock-memory
optimum at 220 W, and stock does 69.9 sol/s against 50.2 there.

The mechanism, the A/Bs behind each step and lolMiner's own rung sweep are in
[performance.md](performance.md#below-stock-the-other-rung-pays-7-to-18--under-caps-below-175-w).

```sh
sudo mxbm --algo BEAM-III --pool ... --user ... --pl 160 --mclk 5001
```

Both settings are restored on exit. The crossover is ~178 W; from ~180 W up the rung is a
wall at ~54 sol/s, so do not use it there.

One more thing for a capped rig: use the CUDA path, not the OpenCL fallback. The
fallback's gap grows as the cap drops: 16 % behind at stock, 15 % at 140 W and 29 % at
100 W ([the curve](performance.md#the-opencl-backend-under-caps)).

---

## Where the time and energy go

Per solve on the shipping build, from `benchmarks/stage_power.sh` (which replays one stage
many times inside a real solve and solves for its own time and power) and from Nsight
Compute for the traffic:

![Board power across one solve, by stage](tools/stages.svg)

The chart is the table below drawn as a timeline: width is time, height is power, so each
block's area is the energy that stage costs. Generated from the table by
`python3 docs/tools/plot_stages.py`.

Measured 2026-09-09 on the shipping build, 8 reps, 45 s per stage, harness solve
28.04 ms, drops 0, KAT 3/3:

| stage | ms | % of solve | power | DRAM traffic | bound by |
|---|---|---|---|---|---|
| `entry_scatter` | 2.69 | 9.6 % | 282.9 W | 0.26 GB | compute (SipHash) |
| round 1 | 4.72 | 16.8 % | 284.4 W | 0.80 GB | compute (SipHash) |
| round 2 | 8.14 | 29.0 % | 284.9 W | 2.67 GB | compute (SipHash) |
| round 3 | 8.09 | 28.9 % | 287.0 W | 4.07 GB | **DRAM** |
| round 4 | 3.94 | 14.1 % | 284.1 W | 2.15 GB | **DRAM** |
| terminal | 0.73 | 2.6 % | 289.2 W | 0.27 GB | DRAM |
| **total** | **28.04** | 101.0 % | 285.2 W | **10.21 GB** | |

The stages sum to 28.31 ms against the harness's 28.04 ms solve, so the attribution
residual is 1 %, and their summed energy (8.07 J) lands within 1.4 % of the counter's
whole-solve figure (7.96 J). Entry is replayed on its own here; in the miner the next
solve's entry pass runs beside rounds 3 and 4 on a higher-priority stream, which is where
the 27.0 ms headline's extra millisecond comes from.

Every stage draws the board limit, within a 6 W spread, so there is no single kernel to
fix for power: the watts follow the cap, and only the time per stage differs. The front
half is arithmetic. Entry and rounds 1 and 2 are 55 % of the solve and are bound by the
integer pipe, almost all of it SipHash. The back half is bandwidth: rounds 3 and 4 and
the terminal round move 6.5 of the 10.21 GB and run at 77–89 % of DRAM peak.

Total traffic is within 1.5 % of the compulsory minimum for these record widths on every
line, so there are no wasted bytes to reclaim, only records to narrow, and narrowing is
priced in memory instructions and sectors rather than in bytes. The round-3 record went
64 B to 56 B on 2026-09-08 and took 0.54 GB out of the total, but the half millisecond it
saved came from keeping the record's four 16 B transactions, not from the bytes.
Full analysis in [performance.md](performance.md#power-and-efficiency).

---

## Reproducing any of this

Build first, see [building.md](building.md).

| What | Command | Needs root |
|---|---|---|
| Everything below in one block: hardware, throughput, telemetry, power curve | `mxbm --report` | for the curve only |
| Throughput, clocks, temperature and J/sol | `mxbm --benchmark BEAM-III --benchmark-seconds 120` | no |
| Power/speed curve for your own card | `mxbm --report` (or `--tune` for the recommendation alone) | yes |
| Per-stage time and power | `benchmarks/stage_power.sh` | no |
| Correctness gate for a CUDA change | `./build/tests/test_cuda_solver`, 3/3 goldens on 22 configurations | no |

The kernel-level instruments (Nsight counters, the pipeline A/B, the VRAM-ladder rungs,
the drop counters and their positive controls) are catalogued in
[performance-research.md](performance-research.md#instruments-diagnostic-and-ablation-flags).

Run on an idle GPU. A benchmark taken while something else is using the card measures
contention, not the miner: a lolMiner run taken while MXBM was mining read 25–27 sol/s
instead of 53.

Quote a median over several thousand solves, not a peak. Individual 15 s windows on this
card reach 61.5 sol/s and that figure is noise; the expected maximum of 304 such samples
is about 67. The median moves by less than 1 % between runs, peaks move by 10 %.

---

## Apple Silicon (Metal)

No cross-vendor comparison is intended. A 40-core laptop GPU at ~400 GB/s is not the same
class of hardware as a 285 W desktop board; the numbers are here because there is nowhere
else to put them.

| | |
|---|---|
| GPU | Apple M3 Max, 40 GPU cores (Apple9 family, Metal 3) |
| Memory | 128 GB unified, 107.5 GB reported as the GPU working set |
| OS | macOS 26.5, Metal toolchain 17F109 |
| Power / thermal control | none available |

### Results

| | ms/solve | sol/s | |
|---|---:|---:|---|
| **MXBM (Metal, SEEDF)** | **101.5** | **18.8** | default on Apple |
| MXBM (Metal, rebuild path) | 117.8 | 16.2 | the configuration CUDA ships |
| MXBM (OpenCL, sort path) | ~505 | ~3.8 | the only OpenCL path that runs here |
| lolMiner | — | — | no Apple Silicon build exists |

Pipeline medians over 6–8 solves, cold, with the baseline bracketed either side. Sustained
through the miner over 60 s: 16.3 sol/s, 128.0 ms/solve median (p5 100.0, p95 135.7)
across 491 solves. The p5 matches the pipeline figure, and the gap above it is thermal.

Apple's OpenCL cannot run this pipeline at all: both fused collision finders fail at
`clCreateKernel` with `CL_INVALID_KERNEL (-48)`, leaving OpenCL the slowest of the three
finders. Metal is the only way to run the fused algorithm on a Mac.

Two things this platform will not tell you:

- **No power figure.** macOS exposes no public API for GPU power, clocks, temperature or
  fan, so MXBM reports utilization and nothing else. Every efficiency claim on this page
  is Ada-only.
- **No overclocking or power limiting.** `--pl`, `--cclk` and friends report "not
  supported on this platform".

### Reproducing

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DOPENSSL_ROOT_DIR="$(brew --prefix openssl@3)"
cmake --build build -j

# pipeline median over 20 solves, gated on the KAT goldens
MXBM_BENCH=20 ./build/tests/test_metal_solver

# per-phase GPU time (GPUStartTime/GPUEndTime, not wall clock)
MXBM_METAL_TIMING=1 ./build/tests/test_metal_solver

# the A/B behind the SEEDF default -- both pairs are compiled in
MXBM_METAL_REBUILD=1 MXBM_BENCH=20 ./build/tests/test_metal_solver

# sustained, through the miner
./build/mxbm --benchmark BEAM-III --benchmark-seconds 60
```

Needs the Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`); without it the
Metal backend is not built and everything else still works.

Benchmark on an otherwise idle machine. This is a laptop: a concurrent build inflated one
measurement by 30 %. Quote medians, bracket a sweep with the baseline first and last, and
discard the run if the two brackets disagree.

---

## Send us your numbers

Everything in the Ada sections above is measured at one memory bandwidth with one
shared-memory budget, and some of it does not transfer. Two cases are already known:
round 2's rebuild costs 0.4 ms on Ada and 21 ms on Apple, and the 5001 MHz memory rung
pays below ~178 W on the 4070 Ti SUPER but only below 121 W on the 4070 SUPER.

Particularly wanted:

- **Anything that is not Ada**: Turing, Blackwell, A-series. Ampere and Turing have one
  report each so far, and Turing is where MXBM is furthest behind.
- **Smaller cards**, 8–12 GB. MXBM refuses to start below its threshold rather than mine
  nothing, so a refusal is itself a useful report; tell us what it said.
- **AMD**, via the OpenCL backend. It is untested there.
- **OpenCL on anything but the reference card.** Every contributed run so far chose CUDA.
- **Anything where MXBM is slower than another miner on your hardware.** That is the most
  useful report of all.

### One command

```sh
mxbm --report
```

It benchmarks the card for two minutes, measures its power/speed curve, and prints a
paste-ready markdown block: hardware, driver, sol/s, telemetry, efficiency and the whole
curve. Nothing is uploaded; it writes to your terminal, or to a file with
`--report-out report.md`.

The curve takes about half an hour, so it is measured once and reused: a later `--report`
on the same binary reads it back in two minutes, and a new build re-measures. Setting the
power limit needs root (`sudo mxbm --report`) or an Administrator terminal on Windows;
without it you still get everything except the curve, which is still a useful report.

Then open an issue with the **Benchmark report** template and paste it in. If MXBM refuses
to start on your card, the refusal message is the report.

Results get added to the [benchmarked-devices table](#benchmarked-devices) with
attribution, and hardware that behaves differently from the reference card becomes a work
item.
