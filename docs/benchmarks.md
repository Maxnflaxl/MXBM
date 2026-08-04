# Benchmark results

Measured numbers for MXBM, and for lolMiner on the same card, so the comparison is
like-for-like. Everything here is reproducible with the scripts in `benchmarks/` — the
commands are given under each table.

**MXBM has been measured on four GPUs**: an RTX 4070 Ti SUPER (the card everything is
developed against), an M3 Max via Metal, and a contributed RTX 4070 SUPER and RTX 3060 Ti.
See [benchmarked devices](#benchmarked-devices), and [send us yours](#send-us-your-numbers).

For *why* comparing miners is harder than reading two numbers off two screens, see
[benchmarking.md](benchmarking.md). This page is the results; that page is the method.

---

## Reference card

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER (Ada, sm_89, 66 SMs) |
| VRAM | 16 GiB GDDR6X, 10251 MHz, 256-bit (**656 GB/s peak at that rung**, ~510 GB/s achievable) |
| Board power limit | 285 W default, 100–366 W permitted by the driver |
| Driver | 610.43.03 · CUDA 13.3 |
| OS | Linux 7.1.4 (Arch) |

All figures below are on this card at **stock clocks**, no overclock or undervolt, unless
the row says otherwise.

---

## MXBM vs lolMiner 1.98a

Both mining BeamHash III against `de.beam.herominers.com:1130` over TLS, stock settings.

| | MXBM (CUDA) | lolMiner 1.98a | |
|---|---|---|---|
| Throughput | **56.2 sol/s** | 53.27 sol/s | **+5.5 %** |
| Board power | 284.0 W | 238.7 W | +19 % |
| Efficiency | 0.198 sol/s/W | **0.223 sol/s/W** | −11 % |
| Core clock | 2700 MHz | 2745 MHz | |
| Memory clock | 10251 MHz | 10251 MHz | |
| Temperature | 65 °C | 60 °C | |
| VRAM for a full search | 7.46 GiB | ~4 GiB[^4g] | |

Both figures are 15 s-window medians from the head-to-head session. MXBM's own
long-run benchmark on the current build measures **58.9 sol/s at 33.8 ms/solve**
over 8,866 solves (p5–p95 33.6–34.7), so the margin above is if anything understated —
the head-to-head predates the 2026-07-26 group-cap change, which was worth −0.9 ms.

**Read that efficiency row carefully — it compares two different operating points.** MXBM
runs pinned at the card's 285 W board limit in every kernel (verified: the driver reports
`sw_power_cap` active continuously, at 63 °C, so it is a power ceiling and not a thermal
one). lolMiner draws 239 W and is *not* capped — it leaves 46 W unused. Comparing sol/s/W
at stock therefore rewards whichever miner fails to fill the card.

At **equal power** the ranking reverses, but only inside a window. Both miners capped
to 220 W:

| | MXBM at 220 W | lolMiner at 220 W | |
|---|---|---|---|
| Throughput | **54.9 sol/s** | 54.4 sol/s | **+0.9 %** |
| Board power | 219.5 W | 219.4 W | — |
| Efficiency | **0.250 sol/s/W** | 0.248 sol/s/W | **+0.9 %** |

MXBM is ahead on both between roughly **210 W and 256 W**. Below that lolMiner is ahead
on both — at 180 W by about 16 % — and above it lolMiner is more efficient while MXBM is
faster.

**Both curves are measured.** lolMiner was swept across the same caps as MXBM rather than
sampled once at its own uncapped draw, which is what makes the window above a comparison
of two curves instead of a curve against a point. Capping lolMiner improves its efficiency
substantially, and that is already priced into the numbers above.

**What this does not show.** The two miners' `sol/s` are separate counters whose
relationship is [an open question](benchmarking.md#1-the-problem-with-comparing-reported-sols)
— MXBM reports CPU-verified solutions and lolMiner's basis is undocumented, a distinction
worth ~17 % inside our own pipeline. The watts are trustworthy across miners because one
instrument measured both; the sol/s columns are each miner against itself. Accepted pool
shares over a fixed interval remain the only arbiter that needs neither counter.

[^4g]: lolMiner selects "BeamHash III **4G** (CUDA)" on this card. MXBM needing 7.46 GiB
is a known gap — see [HW_REQUIREMENTS.md](HW_REQUIREMENTS.md#memory-efficiency-is-24-off-the-algorithms-design-target).

---

## MXBM power curve

The board power limit is the most valuable setting on this card, because MXBM is pinned
against it. Measured with `benchmarks/power_sweep.sh`, 90 s per point:

| `--pl` | sol/s | ms/solve | measured | core clock | sol/s/W | J/solution |
|---|---|---|---|---|---|---|
| 180 W | 45.2 | 44.6 | 180.2 W | 1905 MHz | 0.2508 | 3.99 |
| 200 W | 50.6 | 39.6 | 199.8 W | 2220 MHz | **0.2533** | **3.95** |
| **220 W** | **53.8** | 37.2 | 219.6 W | 2460 MHz | 0.2450 | 4.08 |
| 240 W | 55.2 | 36.3 | 239.5 W | 2550 MHz | 0.2305 | 4.34 |
| 255 W | 56.3 | 35.6 | 254.4 W | 2625 MHz | 0.2213 | 4.52 |
| 270 W | 57.1 | 35.1 | 269.2 W | 2670 MHz | 0.2121 | 4.71 |
| 285 W *(stock)* | **57.5** | **34.8** | 284.1 W | 2700 MHz | 0.2024 | 4.94 |

**Both miners have a `--pl`, and the honest comparison sweeps both.** The table above is
MXBM's curve alone; lolMiner was only ever measured uncapped, which turned out to matter
a great deal. Swept against each other at identical caps:

![Speed, efficiency and power drawn, both miners at the same caps](tools/power-curve.svg)

**Where each one wins.** Below ~210 W lolMiner is ahead on speed *and* efficiency, by
about 16 % at 180 W. Between ~210 W and ~256 W MXBM is ahead on both, by about 1 % at
220 W and 5–6 % at 240 W. Above ~256 W MXBM is faster and lolMiner is more efficient.
lolMiner barely responds to a cap at all — it gives up 2.4 % of its speed for 24 % less
power, and above ~237 W the cap does nothing — so its best efficiency (0.2991 sol/s/W at
175 W) beats MXBM's best anywhere (0.2575 at 210 W). What MXBM has is the ceiling:
59.05 sol/s against ~54.0, which lolMiner cannot reach at any setting.

The sweep was run twice, two days apart, and **reproduces within ~1 %** at every cap —
so these crossings are a property of the two miners, not of the day they were measured.
It now runs down to the card's **100 W floor**, which settles the question of whether
lolMiner had a better efficiency point hiding below the old 120 W left edge: it does not.
Both curves fall away monotonically below their peak.

Full numbers, the mechanism (we lose 570 MHz of core clock to it at 180 W) and what it
implies for the roadmap are in
[performance.md](performance.md#both-miners-under-the-same-cap). Reproduce with
`benchmarks/compare_power.sh`; the chart is generated from that table by
`python3 docs/tools/plot_power.py`.

Each point is 90 s (~2,000–2,500 solves). At that sample size the solutions-per-solve
factor reads 2.01 where an 8,500-solve run measures 1.99, so **the sol/s column is
about 1 % high in absolute terms** — stock read 57.5 here and 56.4 over a long run
(both on the build of 2026-07-25; the current one is 58.9).
Every point was measured the same way, so the curve's shape, its peak and the crossings
against lolMiner are unaffected. Left as measured rather than rescaled to numbers nobody
observed.

Two things worth knowing before you cap your own card:

- **Efficiency peaks around 200–210 W and gets *worse* below it.** At 180 W the core clock
  has fallen to 1905 MHz and the parts of the board that do not scale with it — memory,
  uncore, leakage — are being paid for out of less work. Lower is not always better, and
  the head-to-head sweep now shows this holds all the way down to the card's 100 W floor.
- **The last watts are the worst value.** Going 240 → 285 W buys 2.3 sol/s for 45 W; the
  first 20 W above 180 buys 5.4. Where to sit is an economic choice about your power
  price, not a technical one.

```sh
sudo mxbm --algo BEAM-III --pool ... --user ... --pl 220
```

`--pl` needs root. Without it MXBM says so by name and mines on at the card's current
limit. The previous limit is restored on exit, including on Ctrl+C. See
[usage.md](usage.md#power-limit).

---

## Where the time and energy go

Per solve, from `benchmarks/stage_power.sh` (which replays one stage many times inside a
real solve and solves for its own time and power) and from Nsight Compute:

![Board power across one solve, by stage](tools/stages.svg)

The chart is the table below, drawn as a timeline: width is time, height is power, so
each block's **area** is the energy that stage costs. Generated from the table by
`python3 docs/tools/plot_stages.py`, so the two cannot drift.

| stage | ms | % of solve | power | DRAM traffic | bound by |
|---|---|---|---|---|---|
| `entry_scatter` | 2.68 | 7.7 % | 284.0 W | 0.26 GB | compute (BLAKE2b) |
| round 1 | 5.79 | 16.5 % | 284.5 W | 1.07 GB | latency |
| round 2 | 10.68 | 30.5 % | 283.6 W | 3.35 GB | latency |
| round 3 | 9.54 | 27.2 % | **270.9 W** | 4.83 GB | **DRAM** |
| round 4 | 5.65 | 16.1 % | 282.9 W | 2.95 GB | **DRAM** |
| terminal | 1.05 | 3.0 % | 284.1 W | 0.54 GB | DRAM |
| **total** | **35.0** | | 284 W | **13.0 GB** | |

Every stage draws the board limit, which is why there is no single kernel to "fix" for
power. Total DRAM traffic is within **1 %** of the compulsory minimum for the record
widths — there is no waste to reclaim, only records to narrow, which is the same problem
as the 7.46 GiB footprint. Full analysis in
[performance.md](performance.md#power-and-efficiency).

---

## Reproducing any of this

Build first — see [building.md](building.md).

| What | Command | Needs root |
|---|---|---|
| Throughput | `mxbm --benchmark BEAM-III --benchmark-seconds 120` | no |
| Throughput + power + J/sol | `benchmarks/power_bench.sh 120 myrun -- --solver cuda` | no |
| Power/speed curve | `benchmarks/power_sweep.sh` | yes (`nvidia-smi -pl`) |
| Per-stage time and power | `benchmarks/stage_power.sh` | no |
| Kernel counters (DRAM bytes, stalls) | `sudo ./cuda/profile.sh` | yes (Nsight) |
| Pipeline A/B for a code change | `./cuda/pipeline 700` | no |

Run on an **idle GPU**. A benchmark taken while something else is using the card measures
contention, not the miner — a lolMiner run taken while MXBM was mining read 25–27 sol/s
instead of 53.

Quote a **median over several thousand solves**, not a peak. Under a few thousand, the
solutions-per-solve factor alone moves the headline by more than a sol/s — this project
published 57.5 from a 90 s run and measured 56.4 from a 300 s one, same build (2026-07-25). Individual 15 s
windows on this card reach 61.5 sol/s, and that figure is noise: for the observed spread,
the expected maximum of 304 samples is about 67. The median moves by less than 1 %
between runs; peaks move by 10 %.

---

## Apple Silicon (Metal)

The second card MXBM has been measured on. **No cross-vendor comparison is intended**:
the numbers below sit next to Ada's in this document only because there is nowhere else
to put them, and a 40-core laptop GPU at ~400 GB/s is not the same class of hardware as a
285 W desktop board.

| | |
|---|---|
| GPU | Apple M3 Max, 40 GPU cores (Apple9 family, Metal 3) |
| Memory | 128 GB unified, 107.5 GB reported as the GPU working set |
| OS | macOS 26.5, Metal toolchain 17F109 |
| Power / thermal control | none available — see below |

### Results

| | ms/solve | sol/s | |
|---|---:|---:|---|
| **MXBM (Metal, SEEDF)** | **101.5** | **18.8** | default on Apple |
| MXBM (Metal, rebuild path) | 117.8 | 16.2 | the configuration CUDA ships |
| MXBM (OpenCL, sort path) | ~505 | ~3.8 | the only OpenCL path that runs here |
| lolMiner | — | — | no Apple Silicon build exists |

Pipeline medians over 6–8 solves, cold, with the baseline bracketed either side. The
sustained miner figure over a 60 s run is **16.3 sol/s, 128.0 ms/solve median (p5 100.0,
p95 135.7)** across 491 solves — the p5 matches the pipeline figure exactly, and the gap
above it is thermal, not overhead.

**Apple's OpenCL cannot run this pipeline at all.** Both the LDS and fused row-bucket
collision finders fail at `clCreateKernel` with `CL_INVALID_KERNEL (-48)`, confining the
OpenCL path to the slowest of the three finders. Metal is not a faster dialect here; it
is the only way to run the fused algorithm on a Mac. The six OpenCL gates that cover
those kernels are marked disabled on macOS rather than silently skipped.

### Two things this platform will not tell you

- **No power figure.** macOS exposes no public API for GPU power, clocks, temperature or
  fan. `powermetrics` reads them through a private framework whose layout moves between
  OS releases, so MXBM reports utilization and nothing else rather than a number nobody
  can trust. Every efficiency claim elsewhere in this document is therefore
  Ada-only — there is no sol/s/W row here because there is no honest W.
- **No overclocking or power limiting.** `--pl`, `--cclk` and friends report
  "not supported on this platform".

### Reproducing

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DOPENSSL_ROOT_DIR="$(brew --prefix openssl@3)"
cmake --build build -j

# pipeline median over 20 solves, gated on the KAT goldens
MXBM_BENCH=20 ./build/tests/test_metal_solver

# per-phase GPU time (GPUStartTime/GPUEndTime, not wall clock)
MXBM_METAL_TIMING=1 ./build/tests/test_metal_solver

# the A/B behind the SEEDF default -- no rebuild needed, both pairs are compiled in
MXBM_METAL_REBUILD=1 MXBM_BENCH=20 ./build/tests/test_metal_solver

# sustained, through the miner
./build/mxbm --benchmark BEAM-III --benchmark-seconds 60
```

Needs the Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`); without it
the Metal backend is simply not built and everything else still works.

**Benchmark only on an otherwise-idle machine.** This is a laptop: a concurrent build
inflated one early measurement by 30 % and produced a figure that had to be retracted
from `performance.md`. Quote medians, bracket a sweep with the baseline repeated first
and last, and discard the run if the two brackets disagree.

---

## Send us your numbers

Everything in the Ada sections above — the geometry choices, the power curve, the record
layout, the claim that rounds 3 and 4 are bandwidth-bound — is measured at one memory
bandwidth with one shared-memory budget, and some of it does not transfer. Two cases are
already known: round 2's rebuild costs 0.4 ms on Ada and 21 ms on Apple, and the 5001 MHz
memory rung pays below 173 W on the 4070 Ti SUPER but only below 121 W on the 4070 SUPER.

Particularly wanted:

- **Anything that is not Ada** — Turing, Blackwell, A-series. (Ampere: one report so far.)
- **Smaller cards**: 8–12 GB. MXBM refuses to start below its threshold rather than mine
  nothing, so a refusal is itself a useful report — tell us what it said.
- **AMD**, via the OpenCL backend. It is completely untested there.
- **OpenCL on anything but the reference card.** Every contributed run so far chose CUDA.
- **Anything where MXBM is slower than another miner on your hardware.** That is the most
  useful report of all, and it will not offend anyone.

### One command

```sh
benchmarks/collect_report.sh
```

It runs a two-minute benchmark, samples telemetry while it runs, and prints a
paste-ready markdown block — hardware, driver, sol/s, power, efficiency. Nothing is
uploaded; it only writes to your terminal. Add the power curve with:

```sh
SWEEP=1 benchmarks/collect_report.sh      # needs sudo for nvidia-smi -pl
```

Then open an issue with the **Benchmark report** template and paste it in.

### If you would rather not run a script

Paste whatever you have. The minimum that makes a report usable:

- GPU model, VRAM, driver version, OS
- `mxbm --benchmark BEAM-III --benchmark-seconds 120` output in full
- whether anything else was using the GPU
- any overclock, undervolt or power limit applied

### What happens to it

Results get added to the benchmarked-devices table below with attribution, and hardware
that behaves
differently from the reference card becomes a work item. If MXBM turns out to be slow or
broken on your GPU, that is a bug report we want, not a disappointment to manage.

### Benchmarked devices

Each row is at the card's **stock** power cap.

| GPU | Memory | Driver / OS | Backend | sol/s | ms/solve | W | sol/s/W | Source |
|---|---|---|---|---|---|---|---|---|
| RTX 4070 Ti SUPER | 16 GiB GDDR6X | 610.43.03 · Linux | CUDA | 60.2 | 33.1 | 284 | 0.212 | ours, `--benchmark` |
| RTX 4070 Ti SUPER | 16 GiB GDDR6X | 610.43.03 · Linux | OpenCL | 59.4 | 33.5 | 284 | 0.209 | ours, `--benchmark`, same session |
| RTX 4070 SUPER | 12 GiB GDDR6X | 610.62 · Windows | CUDA | 43.9 | 44.8 | 214 | 0.205 | ZumZum, `--tune` † |
| RTX 3060 Ti | 8 GiB GDDR6 | 610.62 · Windows | CUDA | 22.4 | 88.1 | 195 | 0.115 | ZumZum, `--tune` † |
| Apple M3 Max (40-core) | 128 GB unified | macOS 26.5 · Metal 3 | Metal | 16.3 | 128.0 | — | — | ours, `--benchmark` |

† `--tune` sweep points, not `--benchmark`: 60 s, CPU-verified, warmed up and
drift-gauged. On the 4070 SUPER the miner loop reported ~48 sol/s at the same watts in
the same session, ~10 % above the sweep, and which instrument is right is
[open](performance.md#open-the-tune-harness-and-the-miner-loop-disagree). The
lower number is quoted.

The M3 Max's pipeline alone runs **101.5 ms/solve (18.8 sol/s)**; the sustained figure is
lower because a laptop throttles. It has no power column because macOS exposes no public
GPU power API — blank rather than estimated.

![Speed and efficiency against the cap, every card measured](tools/cards-curve.svg)

Curves for the three NVIDIA cards, generated from the tables in
[performance.md](performance.md#third-party-hardware--a-two-card-rig-2026-08-02). **Not a
controlled comparison** — different machines, operating systems and instruments — so
cross-card distances are unreliable. What holds is each curve's shape and where its own
efficiency peak sits: 210 W, 148 W and 130 W, all well under stock.
