# GPU Performance

The permanent record of MXBM's GPU solver speed and efficiency: where it stands, how
it was measured, and what is still open. Every number here was produced by a committed
benchmark or test in this repository — nothing is estimated or extrapolated unless
explicitly labelled.

**The experiments themselves live in
[performance-research.md](performance-research.md)** — every optimization that worked,
every one that did not, the measured mechanism behind each, and the measurement
methodology. This page links into it throughout; that page is the one to read before
proposing a new lever, because most of them have already been tried.

**Target:** lolMiner does **~53 sol/s** on an RTX 4070 Ti SUPER at **stock clocks, no
special configuration** (user-measured). BeamHash III yields ~1.9 solutions per solve,
so 53 sol/s ÷ 1.9 ≈ **28 solve/s ≈ 36 ms/solve**. That is the bar.

## Apple Silicon (Metal) — M3 Max, 2026-07-30

A second reference platform. Everything else in this document is the 4070 Ti; this
section is the Mac, and the two are not comparable except in shape.

| path | median ms/solve | sol/s | notes |
|---|---|---|---|
| OpenCL, sort | ~505 (494.6 / 516.6) | ~3.8 | the only OpenCL path that runs here |
| OpenCL, LDS | — | — | `clCreateKernel` → `CL_INVALID_KERNEL` |
| OpenCL, row-bucket | — | — | `clCreateKernel` → `CL_INVALID_KERNEL` |
| **Metal, fused row-bucket** | **117** (117.4 / 116.9) | **16.2** | 20/20 clean, spread 0.4 % |

**Apple's OpenCL cannot build the fused kernels at all.** Both the LDS and row-bucket
collision finders fail at `newComputePipelineState`, which is why every macOS-relevant
GPU test in `CMakeLists.txt` is pinned to `MXBM_NO_ROWBUCKET=1`. So on this hardware
OpenCL is confined to the slowest of the three finders, and Metal is not a faster
dialect of the same pipeline — it is the only way to run the fused one.

`bench_rounds`-equivalent figures above come from `MXBM_BENCH=20 test_metal_solver`,
median over 20 solves across two sessions, per this document's own rule that a maximum
of noisy draws reads high.

**The miner delivers less than the pipeline**, and by the same margin this document
records for CUDA: `--benchmark BEAM-III` reports **155.2 ms/solve median (p5 148.7,
p95 310.5), 8.5 sol/s** against the pipeline's 117 ms. Roughly 38 ms per solve is spent
outside the measured region — nonce iteration, stats, the difficulty filter — and the
long p95 tail is not yet attributed. Quote 117 ms for the *pipeline* and 155 ms for the
*miner*; they answer different questions.

Threadgroup memory, the resource the fused rounds are bound by (32 768 B ceiling,
measured):

| kernel | Metal | CUDA | of ceiling |
|---|---|---|---|
| `fused_round_r1` | 18 992 B | 18 980 B | 58.0 % |
| `fused_round_r2` | 23 600 B | — | 72.0 % |
| `fused_round_r3` | 26 160 B | 26 148 B | 79.8 % |
| `fused_round_r4` | 22 320 B | — | 68.1 % |
| `terminal_round` | 9 744 B | — | 29.7 % |

The r1/r3 agreement to within 12 B (the `gcount`/`cnt8` atomics rounding) is the
strongest available evidence that the two ports allocate the same shape.

Not tuned: `kFCap`, `kR1FCap`, `kWG` and the `bb + sm = 17` line are all carried over
from Ada as starting values. Register counts are not observable on Metal (see
`tests/test_metal_resources.mm`), so the occupancy work that produced those numbers on
CUDA has no direct equivalent here.

---

**Reference hardware** for every measurement below: RTX 4070 Ti SUPER (Ada, sm_89,
66 CUs, 16 GB, 48 KB LDS/workgroup, ~510 GB/s achievable copy bandwidth).

**How to reproduce.** Two figures are quoted throughout, and they measure different things:

- `./build/bench_rounds 20` — median over 20 runs of the **solver pipeline** (entry through
  terminal). This is the controlled number used to make optimization decisions.
- `./build/test_gpu_solver` — **end-to-end `GpuSolver::solve()`**, including survivor
  readback, back-reference recovery and CPU verification. Take `solve #2` (steady state,
  persistent buffers); `solve #1` pays one-off allocation. This is what a miner reports,
  so it is the headline.

Both are noisy at the ~1 ms level, so single samples are not meaningful — quote a median
of at least 5. `MXBM_NO_ROWBUCKET=1` forces the fallback sort path.

sol/s ≈ 1900 / ms, since BeamHash III yields ~1.9 solutions per solve.

---

## Progress log

Times are median ms per solve, lower is better. The "Worked" and "Didn't work" columns
link into [performance-research.md](performance-research.md), which explains each result
and what it measured.

Rows above the backend switch are `bench_rounds` **pipeline** medians; rows below are
**end-to-end** `solve()` medians including recovery and CPU verification. The two tracked
each other closely on OpenCL where both were measured (40.3 vs 41.0 ms), so the trend is
continuous across the switch — but the CUDA rows are the stricter measurement, not the
looser one.

The first row is the **real-world** starting point: the miner's own reported speed when
the GPU solver found its first share. Later rows are `bench_rounds` pipeline medians,
the controlled measurement used to make optimization decisions. Early on these disagreed
sharply — the pipeline benched at 245 ms (≈ 7.9 sol/s equivalent) while the miner
actually delivered ~1.8 sol/s, i.e. most of a solve was spent *outside* the measured
pipeline. That overhead is gone; bench and end-to-end now track each other.

![Throughput and solve time, log scale](tools/progress.svg)

![Throughput and solve time, linear scale](tools/progress-linear.svg)

Both charts are generated from the table below by `python3 docs/tools/plot_progress.py`
(no dependencies), so the two cannot drift — add a row, re-run it. Two things it is
deliberately explicit about:

- **The x axis is optimization step, not calendar time.** Only three dates exist and 16
  of the rows fall on one of them, so a literal date axis would collapse into three
  vertical stacks. Dates are drawn as bands instead.
- **The dashed divider is a change of measurement, not just of backend.** Everything left
  of it is a `bench_rounds` pipeline median; everything right of it is an end-to-end
  `solve()` median including recovery and CPU verification. The right-hand points are the
  stricter measurement, so the line is continuous but the two halves are not
  interchangeable.

The two scales answer different questions and neither is sufficient alone. **Log** makes
equal vertical distance mean equal *ratio*, matching the Δ % column, and is the only way
to see the early changes at all next to a 30× range. **Linear** starts both axes at zero,
so equal height means equal *absolute* change — which is what makes it obvious that
almost all of the ms won was won early, while almost all of the sol/s gained came late.
Both are true; the log chart alone would understate how flat the recent ms curve is.

Error bars are ±1σ, drawn only where a spread was actually measured — the parser reads
`56.1 ± 2.3` out of the table, so a row without a measured spread stays a bare point
rather than being given a fabricated one.

**The spread is Poisson on the observed solution count, `1/√N`, and it cannot be got by
repeating the run.** Five repeats of the 2026-07-26 build read 58.3, 58.4, 58.4, 58.4,
58.4 — σ = 0.045, which looks like a superbly precise measurement and is not one. A repeat
at the same duration replays the *same nonce sequence*, so it re-observes the same
solutions and measures timing jitter only. The real uncertainty is in
solutions-per-solve, and only a longer run samples more of it: the 300 s run saw 1.99
where the 120 s runs all saw 2.01, a difference nine times that σ. So `58.0 ± 0.4` comes
from 8,729 solves × 1.99 = 17,371 solutions → `1/√17371` = 0.76 %, the same model that
gives the 2026-07-25 row its ±2.3 (600 solutions → 4.1 %).

| Date | Change | Before | After | **sol/s** | Δ ms | Δ % | Worked | Didn't work |
|---|---|---|---|---|---|---|---|---|
| 2026-07-23 | **First working GPU solver** (first share found) | — | ~1050 | **1.8** | — | — | — | — |
| 2026-07-23 | Sort-based collision finder (P1–P3) | 245.0 | 239.0 | **7.95** | −6.0 | −2.4 % | [Sort path](performance-research.md#sort-based-collision-finder) | — |
| 2026-07-23 | Tiled 4-bit × 6-pass radix sort (P4) | 239.0 | 225.0 | **8.44** | −14.0 | −5.9 % | [Tiled radix](performance-research.md#tiled-radix-sort) | — |
| 2026-07-24 | Fixed-width AoS compaction `[7,7,6,5,1]` | 224.4 | 214.5 | **8.86** | −9.9 | −4.4 % | [Compaction](performance-research.md#fixed-width-compaction) | [SoA planes](performance-research.md#soa-word-planes), [runtime widths](performance-research.md#runtime-loop-bounds) |
| 2026-07-24 | Fused row-bucket pipeline (STEP A–C) | 215.0 | 148.0 | **12.8** | −67.0 | −31.2 % | [Row-bucket](performance-research.md#fused-row-bucket-pipeline) | [Un-fused LDS](performance-research.md#un-fused-lds-path), [lead array](performance-research.md#stride-1-lead-array) |
| 2026-07-24 | L5-thin: drop round-5 leaf payload | 151.9 | 139.1 | **13.7** | −12.8 | −8.4 % | [L5-thin](performance-research.md#l5-thin-emit) | — |
| 2026-07-24 | Per-round work compaction on row-bucket | 139.1 | 136.0 | **14.0** | −3.1 | −2.2 % | [Row-bucket compaction](performance-research.md#per-round-compaction-on-the-row-bucket-path) | — |
| 2026-07-24 | Async enqueue (de-bubble) | 135.0 | 133.0 | **14.3** | −2.0 | −1.5 % | [De-bubble](performance-research.md#async-de-bubble) | [Magazine](performance-research.md#shared-memory-magazine), [gi atomic](performance-research.md#global-gi-atomic) |
| 2026-07-24 | Compact `leftContrib` (fold 8 leaves → 1 u64) | 131.1 | 124.8 | **15.2** | −6.3 | −4.8 % | [leftContrib](performance-research.md#compact-leftcontrib) | [General mix-state](performance-research.md#general-compact-mix-state) |
| 2026-07-24 | Packed element record (1 block, not 4 arrays) | 124.8 | 115.7 | **16.4** | −9.1 | −7.3 % | [Packed record](performance-research.md#packed-element-record) | [SoA LDS staging](performance-research.md#soa-lds-staging) |
| 2026-07-24 | Re-derive round-1 seeds from indices | 114.8 | 102.7 | **18.5** | −12.1 | −10.5 % | [Seed re-derivation](performance-research.md#seed-re-derivation) | [Round-2 re-derivation (first attempt)](performance-research.md#round-2-re-derivation-first-attempt) |
| 2026-07-24 | Defer round-1's seed expand to a non-divergent loop | 103.6 | 95.3 | **19.9** | −8.3 | −8.0 % | [Non-divergent expand](performance-research.md#the-non-divergent-expand) | [Deferring the stage read](performance-research.md#deferring-the-stage-read) |
| 2026-07-24 | Round 2 re-derives from a 16 B pair record | 95.3 | 86.8 | **21.9** | −8.5 | −8.9 % | [Re-derivation chain](performance-research.md#the-re-derivation-chain) | — |
| 2026-07-24 | Round 3 re-derives from a 24 B quad record *(later reverted)* | 86.8 | 83.2 | **22.8** | −3.6 | −4.1 % | [Re-derivation chain](performance-research.md#the-re-derivation-chain) | [Register cap](performance-research.md#register-cap-tuning) |
| 2026-07-24 | Bake per-round constants into the fused kernels | 83.2 | 56.2 | **33.8** | −27.0 | −32.5 % | [Compile-time constants](performance-research.md#compile-time-round-constants) | [`gi` elimination](performance-research.md#eliminating-the-dense-gi) |
| 2026-07-24 | **Retire** the round-3 quad record + per-set stride sizing | 56.2 | 54.0 | **35.2** | −2.2 | −3.9 % | [Quad record retired](performance-research.md#retiring-the-round-3-quad-record) | — |
| 2026-07-24 | Drop the redundant `lead` from the round-3 record | 54.0 | 52.7 | **36.1** | −1.3 | −2.4 % | [Redundant lead](performance-research.md#the-redundant-lead-field) | — |
| 2026-07-24 | Round 5 stores **1** work word, not 5 | 52.7 | 47.5 | **40.0** | −5.2 | −9.9 % | [Terminal work words](performance-research.md#the-terminal-rounds-dead-work-words) | — |
| 2026-07-24 | Retune row-bucket geometry to (15, 2) | 47.5 | 42.7 | **44.5** | −4.8 | −10.1 % | [Geometry](performance-research.md#row-bucket-geometry) | — |
| 2026-07-24 | √-scaled bucket capacity → geometry (16, 1) | 42.7 | 40.4 | **47.0** | −2.3 | −5.4 % | [Capacity](performance-research.md#bucket-capacity), [Geometry](performance-research.md#row-bucket-geometry) | [Occupancy, again](performance-research.md#occupancy-again) |
| | | | | | | | | [Occupancy tuning](performance-research.md#occupancy-tuning), [dense key array](performance-research.md#dense-key-array), [decoupled scatter](performance-research.md#decoupled-scatter), [two-level bucketing](performance-research.md#two-level-bucketing) |
| | ↓ *backend switches to CUDA; rows below are **end-to-end**, incl. recover + CPU verify* | | | | | | | |
| 2026-07-25 | CUDA port, algorithm unchanged | 41.0 | 41.6 | **47.6** | +0.6 | +1.5 % | [CUDA backend](#the-cuda-backend) | — |
| 2026-07-25 | 128-bit access on the round-2/3 records | 41.6 | 38.4 | **51.6** | −3.2 | −7.7 % | [CUDA backend](#the-cuda-backend) | — |
| 2026-07-25 | 128-bit access on the remaining records | 38.4 | 35.2 | **56.1 ± 2.3** | −3.2 | −8.3 % | [CUDA backend](#the-cuda-backend) | [cp.async, block size, pair record](performance-research.md#levers-tried-after-the-mio-fix--all-null) |
| 2026-07-25 | Un-pad round 2's record — 9th word to its own plane | 35.16 | 35.0 | **56.4** | −0.2 | −0.6 % | [Alignment pad](performance-research.md#the-round-2-alignment-pad) *(the win is −0.36 GiB; the speed is noise-level)* | — |
| 2026-07-26 | Perfect chain table + group spill → `kFCap` 320 | 35.0 | 34.1 | **58.0 ± 0.4** | −0.9 | −2.6 % | [Group cap](performance-research.md#shipped-the-group-cap-no-longer-has-to-cover-the-tail-125-ms), [Occupancy](performance-research.md#occupancy-is-worth-real-time-and-shared-memory-is-the-only-gate) | [streaming stores, warp-aggregated gi, (17,0), sub-pass, co-tenant entry](performance-research.md#bytes-are-nearly-free-per-element-work-is-not) |

| | sol/s | ms/solve | |
|---|---|---|---|
| **OpenCL** | 48.2 | 41.0 | fallback / `--solver opencl` |
| **CUDA** | **58.9 ± 0.4**[^drift] | **33.8**[^drift] | **shipping** — default when a CUDA device is present |
| **Target** | 53.0 | 35.8 | lolMiner, stock — user-measured |

The CUDA row is **8,866 solves over 300 s** (`benchmarks/headline.sh`, 2026-07-28), at
1.990 verified solutions per solve and a p5–p95 of 33.6–34.7 ms, measured under stated
conditions rather than opportunistically: stock 285 W, card at thermal equilibrium after a
discarded 240 s warmup, 2685 MHz / 10251 MHz / 284.1 W / 67 °C throughout.

[^drift]: **Carries a ~5 % cross-session band.** Within one session the measurement is
    tight — six repeats spread 0.3 % — but four sessions of the same binaries have landed
    between 33.7 and 35.6 ms and the cause is not yet known. See [how far these figures
    reproduce](#-the-absolute-figures-reproduce-to-03--within-a-session-and-5--between-sessions).
    Every A/B on this page was interleaved, so the deltas are unaffected; only the scale
    moves. OpenCL is the older 300-nonce measurement, ±4.1 % (1σ), and has not been
    re-measured under these conditions.

> **Quote ms/solve, and treat sol/s as derived.** `sol/s = solves/s × solutions/solve`,
> and only the first factor is a property of the solver. The second is a property of
> BeamHash III (~1.98) that a short run estimates badly: across four runs of this same
> build, ms/solve held at 34.9–35.1 while the measured solutions/solve wandered
> 1.99–2.04, moving the headline by more than a full sol/s. Runs under a few thousand
> solves read **high**. Anything quoted here as a speed *change* is an ms/solve
> comparison for that reason.

**Independently confirmed by live mining.** A 2.5-hour session against HeroMiners was
logged and its reported rate tallied — a completely separate measurement from the
benchmark, through the stratum path, on real jobs:

| window | samples | median | mean | σ | min | max |
|---|---|---|---|---|---|---|
| 60 s | 65 | **56.1** | 55.89 | **2.30** | 39.7 | 58.9 |
| 15 s | 304 | **56.1** | 55.94 | 3.27 | 24.9 | 61.5 |

The live median is **56.1 on both windows, matching the 300-nonce benchmark median
exactly**, and the 60 s σ of 2.30 matches the benchmark's ±2.3. Two independent
measurements agreeing to three significant figures is the strongest evidence available
that the figure is real.

That table was tallied by hand from a logged session. It no longer has to be: the
`/summary` API accumulates the same statistics itself, and `Session_Stats.Speed_60s`
reports `N`, `Mean`, `Stddev`, `Min` and `Max` over the whole run for both windows and
for power, clocks and temperature — see
[usage.md](usage.md#dashboard-and-monitoring-api). Only the median still needs the
samples kept; the accumulators are constant-memory and hold none.

It also shows why **the peak must not be quoted**. Individual 15 s windows reached
**61.5 sol/s**, which is tempting and wrong: it is the maximum of 304 draws from a noisy
distribution, and for σ = 3.27 the expected maximum of 304 draws is ≈ 67 — so 61.5 is
unremarkable rather than evidence of a higher true rate. Narrowing the window inflates
the peak (61.5 at 15 s vs 58.9 at 60 s) while leaving the median untouched, which is the
signature of noise rather than throughput. The three samples below 45 sol/s are likewise
artefacts: they are consecutive, and coincide with a lolMiner benchmark and an MXBM
benchmark being run against the same GPU. The CUDA
backend is **~6 % past the target** and the OpenCL path 1.12× short — read
[the caveats](#the-cuda-backend) before treating the target as beaten.

Started at **1.8 sol/s** when the solver first worked → **31× faster**.

VRAM for a full search: **8.36 → 7.46 GiB** (268 → 239 B/element).

*The ~7 ms of non-pipeline overhead seen at 83 ms did not scale with the rounds: OpenCL's
pipeline bench and end-to-end solve now agree to within a millisecond (40.3 vs 41.0).*

*Solutions per second (sol/s) is the number miners and pools report. BeamHash III
yields ~1.9 solutions per solve, so sol/s ≈ 1900 / (ms per solve).*

---

## ⚠ The absolute figures reproduce to 0.3 % within a session and 5 % between sessions

*(Opened 2026-07-26, re-measured under controlled conditions 2026-07-28. Read before
trusting any single-number claim here.)*

**The original observation.** The same binaries that measured **34.15–34.20 ms** early in
one session measured **35.49–35.72 ms** a few hours later, on the same machine,
unchanged. Every absolute figure published before 2026-07-28 was taken in the earlier of
those two regimes.

**The controlled re-measurement.** `benchmarks/headline.sh` was built to answer this: it
refuses to start if another process holds VRAM, runs a discarded 240 s warmup and reports
the residual temperature slope, then repeats the measurement and records NVML telemetry
*per run* — SM clock, memory clock, watts, temperature, and the
`clocks_event_reasons` bitmask that the original measurement never captured.

Six runs of 120 s, stock 285 W board limit, card at equilibrium, host at 0.41 load:

| | ms/solve | sol/s | SM | mem | W | °C | capped |
|---|---|---|---|---|---|---|---|
| run 1 | 33.80 | 59.1 | 2685 | 10251 | 284.1 | 67 | 99 % |
| run 2 | 33.80 | 59.2 | 2685 | 10251 | 284.2 | 67 | 100 % |
| run 3 | 33.70 | 59.3 | 2685 | 10251 | 284.1 | 67 | 99 % |
| run 4 | 33.70 | 59.4 | 2685 | 10251 | 284.1 | 68 | 100 % |
| run 5 | 33.70 | 59.4 | 2685 | 10251 | 284.1 | 67 | 100 % |
| run 6 | 33.70 | 59.4 | 2685 | 10251 | 284.1 | 67 | 99 % |
| **300 s** | **33.80** | **58.9** | 2685 | 10251 | 284.1 | 67 | 100 % |

**What it settles.** Within a session the measurement is *tight*: 0.3 % across six runs,
with the SM clock, memory clock and board power identical to the last digit. So the 4 %
gap in the original observation is not run-to-run noise being mistaken for a regime — a
single run does measure its own session to a few tenths of a percent.

**What it does not settle.** There are now four sessions on record and no two agree:

| session | ms/solve | vs fastest |
|---|---|---|
| 2026-07-26 early | 34.18 | +1.3 % |
| 2026-07-26 later | 35.60 | +5.5 % |
| 2026-07-28 sweep (285 W point) | ~33.7 | — |
| 2026-07-28 controlled | 33.75 | — |

The two 2026-07-28 sessions agree with each other and are the fastest; the slow regime has
not recurred and is still unexplained. **Cross-session variation of up to ~5 % on
identical binaries is therefore a property of this measurement setup, not a one-off**, and
any absolute figure on this page should be read with that band around it. Deltas are
unaffected — every A/B here was interleaved.

**What it is not.** Four explanations checked and rejected in 2026-07-26:

| candidate | evidence against |
|---|---|
| thermal throttling | card is *cooler and clocking higher* in the slow regime — 54 °C / 2760 MHz against 63 °C / 2685 MHz |
| within-run drift | the 15 s windows of the published run show no trend: first five average 58.5, last five 58.1 |
| memory clock | 10251 MHz in both regimes, and 10251 again in the controlled run |
| host CPU contention | the bench does host-side verification and the box has `tte` on 51 % of a core, but the **GPU kernel sum itself rose 34.49 → 35.41 ms** while host overhead is only +0.15 ms |

That last row is the important one: the slowdown is **on the GPU**, not in the host path.
Per-round, measured by replay in both regimes: entry 2.72→2.79, r1 5.33→5.37, r2
10.38→10.58, r3 9.50→9.78, r4 5.49→5.77, terminal 1.07→1.12. Everything is ~2–3 % slower
in step, which points at a global clock/power state the reported SM and memory clocks do
not capture. The controlled run adds one fact to that: the card is at
`sw_power_cap` **99–100 % of the time** at stock, so the SM clock is not a free variable —
it is whatever 285 W happens to buy. The surviving hypothesis is that the same 285 W buys
a different point on the V/f curve on different days.

**The test that would settle it, not yet run:** lock the SM clock
(`sudo -v && LGC=2600 benchmarks/headline.sh`) and repeat across sessions. At a locked
clock the V/f point is pinned, so if ms/solve becomes stable across days the regime is a
clock/voltage effect; if it still moves, the memory subsystem is implicated and the
reported 10251 MHz is not telling the whole story. A locked-clock number is also the right
thing to track *builds* against, since it removes the card's own day-to-day discretion
from a regression test.

**How to quote a number from this page:** use the controlled figure with its conditions
attached — 33.8 ms / 58.9 sol/s at stock 285 W, 2685 MHz, 67 °C — and carry the ~5 %
cross-session band. Do not re-derive a headline from a short run: see the note on
solutions/solve under the progress table.

---

## Power and efficiency

*(User-facing summary of these results, plus how to reproduce them on your own
card: [benchmarks.md](benchmarks.md).)*

Every figure above measures **speed**. A run against lolMiner on the
same card measured **power** for the first time, and it does not point the same
way:

| | MXBM | lolMiner 1.98a | ratio |
|---|---|---|---|
| speed (15 s median) | 56.20 sol/s | 53.27 sol/s | **1.055×** |
| power (median) | 284.0 W | 238.7 W | **1.190×** |
| efficiency | 0.198 sol/s/W | 0.223 sol/s/W | **0.887×** |
| temperature | 65 °C | 60 °C | |
| core clock | 2700 MHz | 2745 MHz | |

**~5 % faster, ~11 % less efficient** — 45 W more for 3 sol/s more. On a
power-limited rig, which is most rigs, that reads as a loss.

**It is not one. That table compares two different operating points, and at equal
power MXBM wins both.** All four leads listed when this was first recorded have
now been measured; the rest of this section is what they said. Reproduce with
`benchmarks/power_bench.sh`, `benchmarks/stage_power.sh` and
`benchmarks/power_sweep.sh`.

### The card is at its power limit, in every kernel

`clocks_throttle_reasons.sw_power_cap` reads **Active** continuously under MXBM,
while `hw_slowdown` and `sw_thermal_slowdown` never fire and the GPU sits at
62–65 °C — far from its throttle point. The 285 W is a **board limit being hit**,
not heat, and the 5 °C gap to lolMiner is the *consequence* of drawing
45 W more, not an independent symptom.

That is not a property of one hot kernel. `benchmarks/stage_power.sh` replays a
single stage N times inside the solve that produced its input — so its input
bytes, bucket occupancies and output addresses are the real ones — and solves for
that stage's own time and power from the shift it causes:

    t_s = (T_N - T_1) / (N-1)        P_s = (P_N*T_N - P_1*T_1) / (T_N - T_1)

| stage | ms | % of solve | power | J/solve | % of energy |
|---|---|---|---|---|---|
| `entry_scatter` | 2.68 | 7.7 | 284.0 W | 0.76 | 7.7 |
| round 1 | 5.79 | 16.5 | 284.5 W | 1.65 | 16.6 |
| round 2 | 10.68 | 30.5 | 283.6 W | 3.03 | 30.5 |
| round 3 | 9.54 | 27.2 | **270.9 W** | 2.58 | 26.1 |
| round 4 | 5.65 | 16.1 | 282.9 W | 1.60 | 16.1 |
| terminal | 1.05 | 3.0 | 284.1 W | 0.30 | 3.0 |
| **sum** | **35.39** | **101 %** | 280.3 W | **9.92** | |

Solve is 35.03 ms, so the stages account for 101 % of it — the 1 % overshoot is the
measurement's own error, and it bounds anything unattributed (memsets, launch
gaps, readback, CPU verify) at essentially zero. 9.92 J per solve ÷ 1.98 verified
solutions = **4.96 J per solution**.

Every stage draws the cap. There is no power-hog kernel to fix: the workload
saturates the board limit from `entry_scatter` through the terminal round, and
the driver pulls the core clock down (2625–2760 MHz) to hold 285 W. Only round 3
sits measurably below it, and round 3 is the DRAM-bound round — moving bytes
costs this board less than working the SM does.

Two consequences, and they matter more than the table:

1. **Energy per solve tracks time per solve.** At a fixed cap, sol/s/W is just
   sol/s ÷ 285. Every speed optimization on record is an efficiency
   optimization of the same size, and no amount of watt-shaving *inside* the
   workload is available independently of it — there is no slack kernel to
   quieten. Matching lolMiner's 0.223 sol/s/W while still drawing
   285 W would need **63.5 sol/s**. Moving the cap itself is the other lever, and
   it turns out to be the cheap one.
2. **The published comparison is between two different operating points.** The
   lolMiner draws 238.7 W with σ 4.05: it is *not* at the cap, and leaves
   46 W unused. Comparing sol/s/W at stock rewards whichever miner fails to fill
   the machine — see [the equal-power comparison](#the-equal-power-comparison),
   which is the one that decides a power-limited rig.

### The equal-power comparison

`benchmarks/power_sweep.sh`, 90 s per point, board limit set with
`nvidia-smi -pl`, restored afterwards.

**On the sol/s column:** each point is a 90 s run, ~2 000–2 500 solves, and at that
sample size the solutions-per-solve factor reads 2.01 against the 1.99 an 8 500-solve
run measures. Every row is therefore about **1 % high in absolute terms** — the 285 W
row says 57.5 where a long run of the same build says 56.4. The ms/solve column and the
*shape* of the curve are unaffected, since every point was measured identically, and the
shape is what the section is about. They are left as measured rather than rescaled:

*Measured on the build of 2026-07-25, i.e. before the group-cap change took stock from
35.1 to 34.1 ms. Every row would shift down by roughly that much; the knee, which is what
this table is for, is a property of the power curve and does not move.*

| board limit | sol/s | ms/solve | measured | SM clock | sol/s/W | J/solution | Δ speed | Δ efficiency |
|---|---|---|---|---|---|---|---|---|
| 180 W | 45.2 | 44.6 | 180.2 W | 1905 MHz | 0.2508 | 3.99 | −15.1 % | +12.4 % |
| 200 W | 50.6 | 39.6 | 199.8 W | 2220 MHz | **0.2533** | **3.95** | −5.0 % | **+13.5 %** |
| 220 W | 53.8 | 37.2 | 219.6 W | 2460 MHz | 0.2450 | 4.08 | **+1.0 %** | **+9.8 %** |
| 240 W | 55.2 | 36.3 | 239.5 W | 2550 MHz | 0.2305 | 4.34 | +3.6 % | +3.3 % |
| 255 W | 56.3 | 35.6 | 254.4 W | 2625 MHz | 0.2213 | 4.52 | +5.7 % | −0.8 % |
| 270 W | 57.1 | 35.1 | 269.2 W | 2670 MHz | 0.2121 | 4.71 | +7.2 % | −5.0 % |
| 285 W *(stock)* | **57.5** | **34.8** | 284.1 W | 2700 MHz | 0.2024 | 4.94 | +7.9 % | −9.3 % |

![Speed and efficiency against board power, with the band where MXBM leads on both](tools/power-curve.svg)

Generated from the table above by `python3 docs/tools/plot_power.py` — including
lolMiner's operating point and both crossing powers, which it reads out of the
paragraphs below, so a number cannot be edited here and left stale in the picture.
Two panels rather than two y axes: sol/s and sol/s/W have no common scale, and an
arbitrary alignment between them would put the crossings wherever the author liked,
which is precisely what this section is arguing about.

> **⚠ The Δ columns and the conclusion that used to follow them are superseded.** They
> compare capped MXBM against **uncapped** lolMiner, on the assumption that lolMiner had
> no choice about its operating point. It does — `--pl` is in its help — and once both
> are swept the comparison changes shape. The MXBM measurements above stand as MXBM's
> own curve; for the head-to-head read
> [Both miners under the same cap](#both-miners-under-the-same-cap) instead.

Δ columns are against lolMiner at *its* uncapped operating point — 53.27 sol/s
at 238.7 W = 0.2232 sol/s/W. Interpolating the crossings against *that* point: MXBM
overtakes it on **speed at ~217 W** and falls behind it on **efficiency at ~252 W**.
Both crossings survive the proper head-to-head almost unchanged (212 W and 257 W), so the
*band* was right — but the margin inside it does not survive, because at 220 W lolMiner
capped to 220 W does 54.35 sol/s at 0.2476, not 53.27 at 0.2232. The advantage there is
**1.9 %**, not the ~10 % this section used to report.

The 11 % efficiency deficit at stock is still an operating-point artefact: MXBM spends
every watt the board allows and converts it into throughput, and lolMiner does not reach
the limit. **But the asymmetry claimed here was wrong in the other direction.** This
document used to say *we* can choose to draw less while *it* cannot choose to draw more.
It can also choose to draw *less* — and it loses almost nothing by it, 1.8 % of its speed
for 21 % less power, which is the single most important thing the head-to-head found.

**Efficiency has an interior optimum at ~200 W**, and the fact that it *falls
again* at 180 W is the informative part: below ~200 W the core clock has dropped
far enough (2220 → 1905 MHz) that the parts of the board which do not scale with
it — memory, uncore, leakage — are being paid for out of less work. There is a
floor, and the sweep found it rather than assuming monotonicity. Pushed to the
driver's own 100 W minimum the collapse is unmistakable: 16.3 sol/s, 0.163
sol/s/W, well under half the peak — too small a sample (48 solves) to be a curve
point, but the direction is not in question.

**Marginal return collapses well before stock.** Extra sol/s bought per extra
watt, walking up the curve:

| step | 180→200 | 200→220 | 220→240 | 240→255 | 255→270 | 270→285 |
|---|---|---|---|---|---|---|
| sol/s per W | 0.276 | 0.162 | 0.070 | 0.074 | 0.054 | 0.027 |

The last 45 W (240 → 285) buys 2.3 sol/s; the first 20 W above 180 buys 5.4. So
the right cap is an economic choice, not a technical one: **220 W for a rig that
pays for electricity, 285 W only where power is free.** Whichever is chosen, the
knob does not exist yet — see [lead 0](#current-focus-and-open-leads).

**What this does not show.** lolMiner was measured only at its own
uncapped draw, so this compares MXBM's *curve* against one *point*. Capping it
would very likely improve its efficiency too, and its curve is unmeasured — the
honest claim is bounded to "against lolMiner as it ships," not
"MXBM's curve dominates." Two further caveats inherited from
`docs-internal/MINER_COMP_RESULTS.md`: the two miners' runs were not
simultaneous, and its sol/s is its own counter, whose definition relative to ours
is the open question the accepted-share protocol exists to settle.

Reproducibility: the 240 W point appears in both sweeps, same build, an hour
apart — 55.5 sol/s / 0.2318 and 55.2 / 0.2305. Run-to-run spread is ~0.6 %,
smaller than every gap called out above. The table quotes the second run.

### Both miners under the same cap

*(Measured 2026-07-28 with `benchmarks/compare_power.sh`.)*

The sweep above compares MXBM's whole curve against lolMiner at one **uncapped** point,
on the reasoning that uncapped is how people run it. That reasoning was wrong. lolMiner
1.98a has `--pl` too — it is in the installed binary's help — so it has a curve of its
own, and nobody had measured it. This section is that measurement, and it revises the
conclusion above rather than supporting it.

Both miners, same session, interleaved, alternating which goes first, caps set externally
with `nvidia-smi -pl` so neither miner's own OC code is a variable. Power is sampled from
NVML for both — never taken from either miner's own statistics block, which averages in
its ramp and reads ~20 W low. Two repeats per cell; the spread within a cell is ±0.1 sol/s.

| cap | MXBM sol/s | MXBM W | MXBM sol/s/W | lolMiner sol/s | lolMiner W | lolMiner sol/s/W |
|---|---|---|---|---|---|---|
| 120 W | 26.30 | 119.8 | 0.2196 | **33.50** | 119.5 | **0.2805** |
| 140 W | 32.65 | 139.9 | 0.2334 | **40.60** | 139.6 | **0.2908** |
| 160 W | 39.35 | 160.1 | 0.2458 | **48.60** | 160.1 | **0.3036** |
| 175 W | 43.95 | 175.2 | 0.2509 | **52.00** | 174.6 | **0.2978** |
| 180 W | 46.05 | 180.0 | 0.2558 | **52.65** | 179.6 | **0.2932** |
| 190 W | 48.75 | 189.9 | 0.2567 | **53.25** | 189.6 | **0.2809** |
| 200 W | 52.15 | 199.8 | **0.2611** | **54.25** | 199.6 | **0.2718** |
| 210 W | 54.15 | 209.6 | 0.2583 | 54.35 | 209.4 | 0.2595 |
| 220 W | **55.40** | 219.5 | **0.2524** | 54.35 | 219.6 | 0.2476 |
| 240 W | **57.20** | 239.4 | **0.2389** | 53.30 | 235.7 | 0.2261 |
| 255 W | **58.00** | 254.2 | **0.2282** | 53.45 | 235.8 | 0.2267 |
| 285 W | **59.15** | 284.2 | 0.2081 | 53.60 | 235.8 | **0.2274** |

![Speed, efficiency and power drawn, both miners at the same caps](tools/power-curve.svg)

**The result is three-part, and only the middle part is the one this document used to
claim.** Crossings are interpolated from the table:

- **Below ~212 W, lolMiner wins on both, and the margin grows as the cap tightens.**
  At 180 W it does 52.65 sol/s to MXBM's 46.05 (**+14.3 %**); at 160 W, 48.60 to 39.35
  (**+23.5 %**); at 120 W, 33.50 to 26.30 (**+27.4 %**).
- **Between ~212 W and ~257 W, MXBM wins on both** — by 1.9 % on each at 220 W, widening
  to 7.3 % speed and 5.7 % efficiency at 240 W. The previously published band of
  217–252 W was right, and slightly conservative.
- **Above ~257 W, MXBM is faster and lolMiner is more efficient**, which is what the
  stock-versus-stock comparison always showed.

Two facts that reframe the whole comparison:

**lolMiner barely responds to the cap at all.** From 285 W down to 180 W it moves 53.6 →
52.65 sol/s — it gives up **1.8 %** of its speed for **21 %** less power. MXBM over the
same range gives up 22 %. And above ~236 W the cap stops doing anything: the 240, 255 and
285 W rows all draw 235.7–235.8 W, which is why it never reaches the board limit.

**Both miners have an interior efficiency optimum; lolMiner's is higher and further
left.** MXBM peaks at **0.2611 sol/s/W at 200 W**, lolMiner at **0.3036 at 160 W** — a
**16.3 %** gap between one miner at its best and the other at its best. Both fall away
below their peak, for the same reason: past a certain point the core clock has dropped
far enough that the parts of the board which do not scale with it are being paid for out
of less work.

Put the other way, and this is the sharpest form of it: **lolMiner at 160 W delivers
48.60 sol/s for 160.1 W, where MXBM needs 199.8 W to deliver 52.15.** Nearly the same
throughput for 40 W less. That is the argument this document used to make in the other
direction at 220 W, and at the efficient end of the curve it now belongs to them.

What MXBM keeps is the top end: **59.15 sol/s against a ceiling of ~54.4**, a **8.8 %**
higher maximum throughput that lolMiner cannot reach at any setting.

### Why we lose the low end: watts buy us less clock

The mechanism is visible in the clocks, and it is the same story at every point:

| cap | MXBM SM clock | lolMiner SM clock | gap |
|---|---|---|---|
| 180 W | 1905 MHz | 2475 MHz | **−570** |
| 190 W | 2048 MHz | 2550 MHz | −502 |
| 200 W | 2242 MHz | 2602 MHz | −360 |
| 210 W | 2385 MHz | 2655 MHz | −270 |
| 220 W | 2460 MHz | 2700 MHz | −240 |
| 240 W | 2542 MHz | 2745 MHz | −203 |
| 285 W | 2685 MHz | 2745 MHz | −60 |

MXBM's kernels cost more power per clock, so a tightening cap takes clock away from us
faster than from them — 60 MHz behind at stock, 570 MHz behind at 180 W. The suspect was
DRAM traffic: MXBM moves [14.23 GB/solve](#current-focus-and-open-leads) at geometry
(16,1), while lolMiner selects a **4G** variant that fits the search in 4 GB and must
therefore move far less.

**That suspicion has since been measured, and it accounts for the shape of this table.**
Narrowing round 2's record so the solve moves 16 % fewer bytes buys **60 MHz at 285 W and
210 MHz at 180 W** — the price of a byte rises 5× as the cap tightens, which is exactly
the asymmetry the gap column shows. One lever worth 210 MHz against a 570 MHz deficit
means the whole gap is the right order of magnitude for a traffic story: it needs about
2.7× our lever, and lolMiner's 4 GB variant against our 7.46 GiB is plausibly that. See
[bytes are not free in watts](performance-research.md#but-bytes-are-not-free-in-watts-and-under-a-cap-watts-are-clock-60-mhz).

**This inverts one of our own conclusions.** [Bytes are nearly
free](performance-research.md#bytes-are-nearly-free-per-element-work-is-not) measured that narrowing records buys
almost no *speed*, because the pipeline is latency- and occupancy-bound rather than
bandwidth-bound, and that is why memory efficiency was filed under reach. Under a power
cap bytes are not free at all: they are watts, watts are clock, and clock is speed. What
that promotes is **narrowing records**, which cuts bytes moved; in-place layer reuse cuts
the footprint while moving the same bytes, so it stays a reach lever.

#### ⚠ Closed 2026-07-29: the low end is not reachable by traffic, and no other mechanism has been found

**The "2.7× our lever" arithmetic above is withdrawn.** It divided a whole-solve clock
deficit by a lever measured on a `MXBM_ROUND_REPS=2:24` replay timeline, and it priced
that lever against a denominator nobody had measured. Both terms were wrong, and both in
the optimistic direction.

**The denominator, measured.** `benchmarks/abl_bytes.sh` profiles the two builds the byte
experiment already used. Ablating round 2's payload removes **1.47 GB**, not the 2.09 GB
the derivation gave — that figure scaled round 2's *compulsory* write and charged the
ablation with 268 MB of back-refs it never touches. Two mechanism questions closed with
it: a 16 B store costs a full **32 B sector** (805 MB predicted at 16 B, 1342 at 32 B,
1330 measured), and there is **no read-for-ownership** (that would have added ~1074 MB of
read; read moved +7.8 MB).

**The numerator is still a replay clock.** The 210 MHz was measured on a timeline that is
~91 % round 2, while round 2 is 30 % of a real solve — and this document's own rows show
what that costs: at 285 W the narrowed ×24 replay clocks 2670 against a real-solve
baseline of 2685, so the whole-solve gain there is ≤ 0, not +60. Until a no-replay A/B
exists the whole-solve rate is a bound, not a number.

| | replay timeline | whole solve |
|---|---|---|
| exchange rate | 210 / 1.47 = **143 MHz/GB** | **≤ 92 MHz/GB** |

**What that leaves.** Closing 570 MHz at ≤ 92 MHz/GB needs **≥ 6.2 GB of a 13.0 GB solve
— 48 % of all traffic**. Every narrowing the [record
audit](performance-research.md#the-record-redundancy-audit) leaves available totals
1.07 GB: **≤ 98 MHz, 17 % of the deficit**. Traffic cannot close the low end, and the
remaining 83 % has no identified mechanism behind it.

**Two other candidates died in the same week**, which is why this is recorded as closed
rather than open:

- **Instruction issue.** Moving siphash's four 64-bit adds off the saturated ALU pipe is
  arithmetically impossible on this ISA: sm_89 has no `IMAD` form that writes a
  carry-out, so `ptxas` lowers `mad.lo.cc.u32` to `IMAD.IADD` **plus** a carry-synthesising
  `LEA`, and the ALU pipe gets *busier* (`entry_scatter` 855 → 876). The high halves were
  already on FMA as `IMAD.X`, 157 of 160. The surviving rotate-only form is −5.1 %
  static, ≈0.18 ms, under this project's 1 %-of-a-solve floor.
- **Undervolting.** A curve-shift offset is the one lever that acts on clock directly, and
  it is **not testable on the reference machine**: the benchmark card also drives the
  display, so an unstable offset corrupts the screen and needs a hard restart, and it does
  so *before* it corrupts a solve — the KAT gate cannot protect against a failure that
  arrives ahead of it. `--coff 300` did exactly this. See
  [overclocking.md](overclocking.md).

**The practical conclusion, which is a real answer and not a placeholder.** MXBM's
advantage is a band, and the band is where it was measured: **lolMiner below ~212 W,
MXBM between ~212 W and ~257 W, and MXBM alone above it** at a ceiling lolMiner cannot
reach at any setting (59.15 sol/s against ~54.4). Recommending MXBM for a rig capped
below 212 W is not supportable on this hardware, and no change in this document's reach
would make it so. **Effort belongs on goals 2 and 3**, where the pipeline runs at 57 % of
its 19.8 ms floor and rounds 1 and 2 hold 65 % of the gap.

*Reopening conditions, since the ledger's own rule is that a closure states them: a
no-replay 180 W A/B that puts the whole-solve rate materially above 92 MHz/GB; a
narrowing worth more than the 1.07 GB the audit allows; a machine where the compute GPU
does not drive the display, which makes undervolting measurable; or a mechanism nobody
has proposed. The first two would move the arithmetic, not the conclusion — 48 % of all
traffic is a long way from 1.07 GB.*

### Above stock: the curve continues to ~311 W, and the memory rung is unreachable

*(Measured 2026-07-29, current build, 60 s arms, bracketed. Every power sweep in this
document stops at the 285 W stock limit; the board's own maximum is 366 W, so the top of
the curve had never been measured.)*

| cap | ms/solve | sol/s | SM clock | **drawn** | sol/s/W | vs stock |
|---|---|---|---|---|---|---|
| 285 W *(stock)* | 33.8 | 58.7 | 2685 MHz | 284.0 W | 0.2067 | — |
| 300 W | 33.5 | 59.2 | 2715 | 299.1 | 0.1979 | −4.2 % |
| **315 W** | **33.3** | **59.6** | **2745** | 309.4 | 0.1926 | −6.8 % |
| 330 W | 33.3 | 59.6 | 2745 | **311.2** | 0.1915 | −7.3 % |
| 366 W | 33.2 | 59.7 | 2745 | **311.2** | 0.1918 | −7.2 % |

**MXBM saturates at ~311 W.** The 330 W and 366 W caps draw the same 311.2 W and return
the same clock, so above ~315 W the cap stops binding — the mirror of the behaviour this
document records for [lolMiner at ~236 W](#both-miners-under-the-same-cap). The 285 W
bracket reproduced exactly (33.8 / 58.7 before and after), and thermals are not the limit
at any rung: 68 °C at 366 W against a 64 °C stock baseline.

**`--pl 315` is the operating point above stock**, worth **+1.5 % speed for +8.9 % power**.
The marginal return over 285 → 315 W is **0.035 sol/s per W**, which is *higher* than the
0.027 this document records for the 270 → 285 step — the curve does not roll off above
stock the way extrapolating the sub-stock sweep would suggest. It is still a goal-2 lever
bought at a goal-3 cost, and 220 W remains the recommendation for a rig that pays for
electricity.

**The 10501 MHz memory rung is not reachable, and this is a mechanism rather than a
null.** The card advertises it (`nvidia-smi -q -d SUPPORTED_CLOCKS` lists
10501/10251/5001/810/405) and every figure ever published here was taken at 10251, so it
looked like free bandwidth for a pipeline at 76–80 % of DRAM peak in rounds 3 and 4.
`nvidia-smi -lmc 10501,10501` **applies at idle and is dropped the moment the workload
runs** — six bracketed arms at stock all reported 10251 under load with
`sw_power_cap` active, and the rung is still not taken at a 366 W cap where the cap does
not bind. `benchmarks/mclk_rung.sh` reproduces this; its `MEM_MHz` column exists precisely
so the null cannot be mistaken for a measurement of the rung. What was measured is that
the lever cannot be applied, not that it does not pay.

### The memory traffic is compulsory

Per solve, against the minimum the round schedule and record widths require —
each round reading its input layer once and writing its output layer once:

| stage | compulsory rd | wr | measured rd | wr | excess |
|---|---|---|---|---|---|
| entry | — | 268 MB | 5 | 258 MB | −1.9 % |
| r1 | 268 | 805 | 271 | 796 | −0.6 % |
| r2 | 537 | 2685 | 543 | 2811 | +4.1 % |
| r3 | 2416 | 2416 | 2427 | 2398 | −0.2 % |
| r4 | 2147 | 805 | 2149 | 802 | −0.1 % |
| terminal | 537 | — | 539 | 1 | +0.4 % |
| **total** | | | **13.00 GB** | | **+1.0 %** |

Measured traffic is within **1 %** of compulsory: no write amplification, no
redundant re-reads, nothing left for the cache to save. The 13.0 GB is not waste
— it is what a 72 B element costs when five rounds each consume a layer and
produce one. The kernel times sum to **101 %** of the solve, which also rules out
the fourth lead: there is no idle spin between rounds to reclaim.

This is the post-fix profile (`sudo ./cuda/profile.sh`, 2026-07-25). Before
[the pad was removed](performance-research.md#the-round-2-alignment-pad) the total was **13.51 GB**:
round 3's read fell by exactly the predicted 268 MB and round 2's write by
204 MB, the shortfall being the sector granularity of splitting one store stream
into two. Round 2 is now the only line materially above compulsory, and that
+4.1 % is the same split — it writes 72 B of record as 64 + 8 to two places.

So the byte count can only fall by making records narrower, and one place was
found where a record was wider than its own contents ([the round-2 alignment
pad](performance-research.md#the-round-2-alignment-pad)). Everything else needs the structural change in
[HW_REQUIREMENTS.md](HW_REQUIREMENTS.md#2-memory-efficiency-is-24-off-the-algorithms-design-target):
streaming / in-place layer reuse.

### What the footprint still costs

The sweep settles the *ranking*, not the *margin*. At its own operating point the
lolMiner spends **4.48 J per solution**; MXBM spends **4.94** at stock and
only undercuts it by capping — 4.08 at 220 W, 3.95 at the ~200 W peak. Winning by
11 % on energy while giving up 12 % of throughput to get there is a real lead but
a bought one, and where the rest of it went is not mysterious: 13.0 GB of
compulsory traffic per solve. **Traffic, specifically — not the 7.46 GiB
footprint it sits in.** Those were treated here as one problem and they are two:
[the byte-power measurement](performance-research.md#but-bytes-are-not-free-in-watts-and-under-a-cap-watts-are-clock-60-mhz)
prices the bytes *moved* at 60 MHz of sustained clock per 16 % of traffic, while
a smaller peak allocation that moves the same bytes to reused addresses buys
none of that. Narrowing records extends this lead; in-place reuse extends the
range of cards that can run at all.

Full data, method and caveats for the head-to-head: `docs-internal/MINER_COMP_RESULTS.md`.
The two runs were not simultaneous, so lolMiner's power figure still
wants a back-to-back rerun before it is quoted outside that document.


---

## Architecture

The solver runs Wagner's algorithm on the ⟨144,5⟩ parameter shape: 2^25 seed elements, five
rounds, each finding pairs that collide on a 24-bit key and combining them. BeamHash III
adds a mandatory per-round `apply_mix` that folds the element's growing index tree into
word 0, re-deriving the collision key every round.

Two collision-finding paths exist. The **fused row-bucket path** is the default wherever
device memory allows; the **sort path** is the fallback for smaller cards.

### Fused row-bucket pipeline

*One kernel per round* does round *r*'s match **and** round *r+1*'s mix **and** round
*r+1*'s scatter. Per workgroup (one per bucket × sub-mask):

1. Coalesced-load the bucket's elements into LDS (work words + gi + lead + leaf payload).
2. Build an LDS hash table over the middle key bits via `atomic_xchg` chains.
3. For each equal-key pair: order by (lead, gi), `bh3_combine` at `Lout(r)`.
4. Materialize the child's leaf prefix, fold `apply_mix(Lmix(r+1))` to set its next key.
5. Emit the child **directly into round r+1's buckets**, keyed by that fresh key.

Fusion is the point: the coalesced-gather win is destroyed if a separate scatter pass
re-reads the elements — see [Un-fused LDS path](performance-research.md#un-fused-lds-path).

Path selection is by device memory (`want_rowbucket` in `src/gpu/round_pipeline.cpp`):
row-bucket needs 7.46 GiB with a 3.27 GiB largest single allocation, so smaller cards fall
back to the sort path automatically. `MXBM_ROWBUCKET=1` / `MXBM_NO_ROWBUCKET=1` override.
(The *budget* still gates on a single sort-path-sized constant, which is stricter than
this — see [open leads](#current-focus-and-open-leads).)

### Round schedule

| Round | Lmix | Lout | padNum | Significant work words | Leaf payload carried |
|---|---|---|---|---|---|
| 1 | 448 | 424 | 1 | 7 | 1 → 2 raw leaves |
| 2 | 424 | 400 | 2 | 7 | 2 → 4 raw leaves |
| 3 | 400 | 376 | 4 | 6 | 4 raw → 1 u64 `leftContrib` |
| 4 | 376 | 288 | 6 | 5 | `leftContrib` → none |
| 5 | 288 | 24 | 9 | 1 | none (terminal) |

### Stored record per round

Rounds 1–2 do not store work state at all: their elements are re-derivable from seed
indices, and those indices are also exactly the leaves the round needs, so the compact
record *replaces* the payload rather than adding to it (see
[the re-derivation chain](performance-research.md#the-re-derivation-chain)). Round 3 tried the same trick and it
was later [measured to be a loss](performance-research.md#retiring-the-round-3-quad-record).

| Read by | Record | u64 | Bytes | Contents |
|---|---|---|---|---|
| Round 1 | seed | 1 | 8 | `key \| index << 32` |
| Round 2 | pair | 2 | 16 | `key \| i0<<24`, `i1 \| gi<<25` |
| Round 3 | packed | 9 | 72 | 7 work words, then 4 leaves + `gi` bit-packed into 2 u64 |
| Round 4 | packed | 8 | 64 | 6 work words, meta, `leftContrib` |
| Round 5 | thin | 2 | 16 | **1** work word, meta — see [why](performance-research.md#the-terminal-rounds-dead-work-words) |

Seed indices are 25-bit (2^25 seeds) and `gi` is 26-bit, so the pair and quad records
pack exactly with bits to spare.

Round 3's 9 u64 are **not** stored as a 9-u64 stride: 8 of them sit at a 16 B-aligned
stride and the 9th lives in a plane behind the records, which is what keeps the 128-bit
accesses legal without paying for a padding word — see
[the round-2 alignment pad](performance-research.md#the-round-2-alignment-pad).


---

## The CUDA backend

Built, gated, measured, and **wired into the miner** (2026-07-25). `--solver auto`, the
default, now prefers CUDA and falls back to OpenCL and then to the CPU reference; the
backend can be pinned with `--solver cuda|opencl`. CUDA is an *optional* build component
— CMake probes for it with `check_language(CUDA)`, so a machine without a CUDA toolchain
still builds and ships the OpenCL path unchanged.

A standalone `nvcc`-only bench still lives under `cuda/` for profiling work (Nsight
Compute cannot profile OpenCL, which is what motivated the port in the first place).

| | end-to-end | verified/solve | sol/s |
|---|---|---|---|
| OpenCL (fallback) | 41.0 ms | 1.98 | 48.2 |
| **CUDA backend (default)** | **34.1 ms** | **2.00** | **58.1** |
| lolMiner (user-measured, stock) | — | — | 53.0 |

Same methodology on both sides: median over **300 distinct nonces**, persistent buffers,
including survivor readback, back-reference recovery and CPU verification, counting only
solutions that pass `bh3::is_valid_solution`. The verified rate coming out at **1.98 on
both** is a useful cross-check that the two implementations agree.

**On sample size.** Solutions per solve is Poisson-ish, so the rate needs ~600 observed
solutions for ±4 %. An earlier revision quoted 55.2 sol/s from 20 nonces — about 39
solutions, ±16 % — which was not enough to state a margin over a 53 sol/s target. At 300
nonces the figure is **56.1 ± 2.3**, so the lower bound only just clears the target. That
is a real margin but a thin one relative to the error, which is precisely why the
share-rate comparison is the thing that settles it.

**Why it is faster, and it is not the language.** The port initially measured 41.6 ms —
*slower* than OpenCL — because it reproduced the OpenCL algorithm exactly. The gain came
from one thing the profiler found: rounds 2 and 3 were stalling 12 % and 22 % of
warp-active cycles on **MIO-queue throttle**, the load/store *instruction* queue backing
up. That is an issue-rate limit, not a bandwidth one — and round 2 was simultaneously at
only 38 % of DRAM peak, so **bytes were cheap and memory instructions were expensive**,
the exact inverse of the direction every previous optimization here took.

Padding the round-3 record 9 → 10 u64 (4 % more traffic) made every record stride an even
number of u64, so `slot*stride*8` is 16 B aligned and 128-bit accesses became legal:

```
r2 emit  11 x ST.64 -> 4 x ST.128 + 3      r3 load  10 x LD.64 -> 4 x LD.128 + 3
r3 emit  10 x ST.64 -> 4 x ST.128 + 2      r4 load   9 x LD.64 -> 4 x LD.128 + 2
```

41.5 → 38.4 → 35.3 ms. **The compiler does not do this for you** — it cannot prove the
base pointer's alignment and emitted zero 128-bit accesses even after the padding made
them legal. `cuobjdump -sass | grep LDG.E.128` is the check.

**Caveats, because "target beaten" is a claim worth being careful with:**

- The 53 sol/s figure is user-measured from lolMiner's own display. The
  [counting question](#the-cuda-backend) is still open — this solver produces 2.29
  survivors per solve but only 1.95 that verify, a 17 % gap between "solutions found" and
  "solutions that are solutions". If lolMiner reports the former, our margin is larger; if
  the latter, it is the ~4 % measured here.
- MXBM counts **verified** solutions, the conservative definition of the two.
- Accepted pool shares over a fixed interval remain the only comparison that does not
  depend on either miner's counters. See `docs-internal/MINER_COMP.md`.
- The backend is standalone. Wiring it into the stratum path, and keeping OpenCL as the
  portable fallback, is remaining work.

*How it got there, what the profiler said afterwards, and the four levers tried and
rejected on top of it are in
[the CUDA backend in detail](performance-research.md#the-cuda-backend-in-detail).*

---

## Established limits

Measured properties of the problem on this hardware. These bound what any further
optimization can achieve.

1. **The element cannot shrink below `[7,7,6,5,1]`.** That schedule is BeamHash III's
   information floor, confirmed independently by Wilke Trei's own `beamhashverify`
   reference. The compact 16 B element in the official `BeamMW/cuda-miner` belongs to
   **BeamHash I** (2018, pre-Fork2: shift-out-key running XOR, no per-round `apply_mix`),
   and does not transfer.
2. **`apply_mix` is genuinely per-round**, folding padNum ∈ {1,2,4,6,9} index-tree entries
   into word 0. It rewrites only word 0; words 1–6 pass through read-only.
3. **A key-random fat scatter runs at ~260 GB/s (51 % of peak)** and is invariant to
   occupancy, bucket count, and atomics. It is random-access-bound.
4. **Coalescing cannot be bought back** — maximum 1.91–2.22×, against a 3× traffic cost
   for any two-pass scheme.
5. **Divergence is only paid by compute.** Work placed in the sub-mask-filtered staging
   loop runs at ~4/32 lanes; the same work in the following all-lanes loop runs at 32/32.
   For *arithmetic* this is worth up to ~58× (round 2's rebuild: 23.5 → 0.4 ms). For
   *loads* the ordering reverses — the staging loop's 8 iterations overlap their loads,
   the all-lanes loop's single iteration cannot
   ([measured](performance-research.md#deferring-the-stage-read)).
6. **The compute-hiding budget is finite — and it SHRINKS as the kernel gets faster.**
   The fused kernel stalls on memory, and arithmetic issued into those stalls is free
   until it exceeds them. This is not register spilling
   ([register cap](performance-research.md#register-cap-tuning) is a wash); it is the stall budget being used
   up. Crucially the budget is not a constant of the algorithm: removing the local-memory
   spill in [compile-time constants](performance-research.md#compile-time-round-constants) cut the stalls, and
   the *same* round-2 rebuild went from +0.4 ms to +4.3 ms without changing a line of it.
   **Any compute-for-memory trade must be re-measured after any change that speeds up the
   round it lives in** — see [retiring the quad record](performance-research.md#retiring-the-round-3-quad-record).

Consequence: element size, coalescing, occupancy and atomics are all settled. The gap is
down to ~1.6×, and the single largest step toward it was not an algorithmic idea at all —
it was a compiler-visibility bug in code that had been read many times. Worth weighing
before assuming lolMiner holds an unknown technique: the last 26.6 ms came from making
existing arithmetic compile down properly, not from moving fewer bytes.


---

## Current focus and open leads

**Where the time goes.** *(CUDA, the shipping path. Re-measured 2026-07-26, after the
`kFCap` 320 change.)*

**The per-round picture is obtainable without a profiler**, which matters because Nsight
needs root and a reboot on a card that drives a display. `MXBM_ROUND_REPS="R:N"` replays
round *R* in place *N* times, so `(t_N − t_1)/(N−1)` is that round's marginal cost; the
traffic is exact from the stride table (`kFbStride`, plus 8 B/element for the sub-mask
rescan and 8 B for back-refs). Over 60 nonces, against the card's true 672 GB/s:

| kernel | ms | GB moved | GB/s | % of peak | floor at peak |
|---|---|---|---|---|---|
| entry | 2.72 | 0.27 | 99 | 15 % | 0.40 |
| r1 | 5.33 | 1.34 | 252 | 37 % | 2.00 |
| r2 | 10.38 | 3.49 | 336 | 50 % | 5.19 |
| r3 | 9.50 | 5.10 | 537 | **80 %** | 7.59 |
| r4 | 5.49 | 3.22 | 587 | **87 %** | 4.79 |
| terminal | 1.07 | 0.81 | 753 | >100 % (L2 absorbs the rescan) | 1.20 |
| **total** | **34.49** | **14.23** | **412** | **61 %** | **21.16** |

*(r1 and r2 carry almost all of the −1.25 ms the `kFCap` 320 change bought, 5.85 → 5.33
and 10.71 → 10.38, which is what their 3 → 4 blocks/SM is worth.)*

This reproduces the profiler table within a few points on every row, from a stopwatch and
arithmetic. Use it before booking a profiling session.

**So the floor is 21.2 ms and the pipeline is at 61 % of peak** — the right-hand column,
priced at the hardware's 672 GB/s rather than at any rate the solver happens to achieve.

> **Correction (2026-07-29): the floor is 19.8 ms and the pipeline is at 57 % of peak.**
> Two independent errors above, pushing the same way.
>
> **(1) The sub-mask rescan is an L2 hit, not DRAM traffic.** The 14.23 GB charges
> 8 B/element of rescan to five stages, 1.34 GB in all, and the profiler says almost none
> of it reaches DRAM: per stage, (compulsory read + 268 MB rescan) − measured read comes
> to 265 / 262 / 257 / 266 / 266 MB for r1 / r2 / r3 / r4 / terminal, so **98 % of the
> 1.34 GB budget is absorbed** ([the compulsory-traffic
> table](#the-memory-traffic-is-compulsory), whose measured total is 13.00 GB). A floor
> has to be priced on bytes the DRAM actually moves.
>
> **(2) The card runs at the 10251 MHz rung, where peak is 656 GB/s.** 672 needs the
> 10501 rung; `nvidia-smi -q -d SUPPORTED_CLOCKS` lists exactly 10501 / 10251 / 5001 /
> 810 / 405, and every controlled run on record reads 10251 under load (the 300 s run
> above, and both miners in the head-to-head). 10251 × 2 × 32 B = 656 GB/s.
>
> | kernel | ms | GB moved | GB/s | % of peak | floor at peak |
> |---|---|---|---|---|---|
> | entry | 2.72 | 0.26 | 97 | 15 % | 0.40 |
> | r1 | 5.33 | 1.07 | 200 | 31 % | 1.63 |
> | r2 | 10.38 | 3.35 | 323 | 49 % | 5.11 |
> | r3 | 9.50 | 4.83 | 508 | **77 %** | 7.35 |
> | r4 | 5.49 | 2.95 | 538 | **82 %** | 4.50 |
> | terminal | 1.07 | 0.54 | 505 | 77 % | 0.82 |
> | **total** | **34.49** | **13.00** | **377** | **57 %** | **19.82** |
>
> The terminal row's impossible ">100 % of peak" was the tell, and it is gone: it was
> 0.81 GB of arithmetic against 0.54 GB of DRAM. On the same measured bytes at 672 GB/s
> the figures are 19.34 ms and 56 %, so (1) does nearly all the work and (2) is worth
> 2.4 %.
>
> **Two figures below inherit the old basis and are not re-derived here.** "Of the 14.5 ms
> above that floor, r1 (3.85) and r2 (5.52)" uses the *pre*-`kFCap`-320 round times
> (5.85 / 10.71) against a current floor; on the row above it is r1 3.70 and r2 5.27,
> **8.97 ms, 61 % of the 14.68 ms above the floor**. "r3 and r4 … are at 79–84 % of peak"
> matches neither table; they are at **77–82 %**.

> **⚠ This supersedes the "39.1 ms memory floor" and the "within ~5 %" claim that stood
> here before.** That floor was priced at the *OpenCL-achieved* 312 GB/s, and CUDA
> already runs at 34.2 — below its own stated floor, which should have been the tell.
> **A floor priced at achieved bandwidth is circular**: it can only ever report that the
> pipeline is near it. The retracted accounting is kept below because it is still a
> correct description of *the OpenCL path*.
>
> | | read + write | GiB | floor at 312 GB/s |
> |---|---|---|---|
> | r1 | 8 + 16 B | 0.75 | 2.6 ms |
> | r2 | 16 + 72 B | 2.75 | 9.5 ms |
> | r3 | 72 + 64 B | 4.25 | 14.6 ms |
> | r4 | 64 + 16 B | 2.50 | 8.6 ms |
> | | | | **35.3 ms** + entry 2.7 + terminal 1.1 = **39.1** |
>
> OpenCL measured **40.3 ms** clean against that (41.2 under ablation instrumentation),
> at entry 2.7, r1 6.3, r2 13.0, r3 12.3, r4 6.4, terminal 1.1. Its non-memory work
> ablated to ~2.2 ms in total: `apply_mix` 0.9, back-refs 1.1, round 2's rebuild 0.2. On
> CUDA that work is smaller again — `apply_mix` measures **0.01–0.06 ms** per round and
> round 2's rebuild 0.97
> ([attribution](performance-research.md#bytes-are-nearly-free-per-element-work-is-not)).

**Which round holds the gap.** Of the 14.5 ms above that floor, r1 (3.85) and r2 (5.52)
hold **9.4 ms, 65 % of it** — and they are exactly the two rounds that re-derive per
element instead of reading stored state (r1 re-seeds, r2 rebuilds from two parent seeds
in 14 siphashes). r3 and r4, which only move bytes, are at 79–84 % of peak. So the split
is not "some rounds are tuned and some are not": **the rounds that compute are slow and
the rounds that only stream are fast**, and both re-derivations have been A/B'd in the
right direction (r1's seed re-derivation 2.5 vs 18.2 ms; r2's pair record 35.0 vs 39.2).
The work is BeamHash III's own hash, which the PoW definition fixes.

**The 1.71× / 96 sol/s ceiling is real arithmetic but it is not one overlap away.**
`max(SM-busy, DRAM-busy)` = 20.6 ms agrees with the 21.2 ms DRAM floor above, so the
number is right. What it assumes is that the six kernels' work can be interleaved
arbitrarily — and within a solve it cannot: entry→r1→r2→r3→r4→terminal is a strict chain
with a global barrier at every step, so at any instant only one round's work exists. All
independent work comes from *another solve*, and [lead 5](#current-focus-and-open-leads)
measured that streams cannot deliver it. Reaching 21 ms therefore needs a
software-pipelined heterogeneous launch, not an overlap. **That co-resident launch has
since been built and measured, and it is null too** — hosting `entry` inside a round costs
7.25 ms against its 2.77 ms standalone, because a memory-stalled warp still holds its slot
and the work lands *after* the stall in the same warp rather than beside it. See
[phase overlap cannot reach the roofline](performance-research.md#phase-overlap-cannot-reach-the-roofline).
The binding resource is **resident warps**, which shared memory caps at 24 of 48 per SM,
and `kFCap` cannot shrink to free any. One objection has lifted, for whenever that
constraint is broken: the 48 MiB shortfall that retired two resident pipelines was
computed at geometry (16,1), and two at (15,2) need 13.8 GiB — they fit.

**The large levers, and where each now stands:**

| lever | status |
|---|---|
| fewer bytes | every record is `ceil(bits/64)` ([audit](performance-research.md#the-record-redundancy-audit)) — except one that was *stored* wider than that, [since fixed](performance-research.md#the-round-2-alignment-pad). And bytes buy almost no **time**: shrinking r2's and r3's records to 16 B, well past what the audit allows, is worth ~4 ms of 34 ([measured](performance-research.md#bytes-are-nearly-free-per-element-work-is-not)) |
| bytes as **watts** | where that lever moved to. Under a cap bytes are clock: 16 % less traffic is worth [60 MHz at 285 W and 210 MHz at 180 W](performance-research.md#but-bytes-are-not-free-in-watts-and-under-a-cap-watts-are-clock-60-mhz) — but that is the prize for a *free* narrowing, and the [one built](performance-research.md#the-quad-record-29--footprint-and-the-byte-prize-does-not-survive-re-derivation) spends the freed watts on the arithmetic replacing the bytes. **CLOSED 2026-07-29.** The denominator is 1.47 GB measured, not 2.09 derived, and the numerator is a replay clock — so the whole-solve rate is ≤ 92 MHz/GB and closing 570 MHz needs 48 % of all traffic against the 1.07 GB the audit allows. [Details](#-closed-2026-07-29-the-low-end-is-not-reachable-by-traffic-and-no-other-mechanism-has-been-found) |
| redundant rescan | removing it entirely buys nothing over halving it ([geometry](performance-research.md#row-bucket-geometry)); re-confirmed on CUDA, where (17,0) is −9 % traffic and +9 % time |
| occupancy | ⚠ **reopened on CUDA, and partly collected.** OpenCL's 48 KB LDS made it structurally unreachable ([details](performance-research.md#occupancy-again)); CUDA exposes 100 KB/SM, where [3 → 4 blocks/SM is worth ~1.6 ms](performance-research.md#occupancy-is-worth-real-time-and-shared-memory-is-the-only-gate). r1/r2 crossed at `kFCap` 320 (**−1.25 ms**) and [r1 took a fifth block](performance-research.md#round-1-takes-a-fifth-block-015-ms-via-a-per-round-group-cap) (−0.15). r3 reaches 4 blocks and [does not care](performance-research.md#round-3-does-not-want-a-fourth-block--occupancy-pays-only-where-a-round-is-latency-bound) — occupancy pays where a round is latency-bound, not where it is bandwidth-bound. **Shared memory per staged element (`B`) is the only term left** |
| coalescing the emit | max 1.9–2.2× against a 3× traffic cost ([two-level](performance-research.md#two-level-bucketing)) |
| the two atomics | both load-bearing; removing either is slower |
| `apply_mix`, back-refs, rebuild | 2.2 ms combined on OpenCL and less on CUDA — nothing left to win |
| phase overlap | closed by two mechanisms — grid depth and warp slots ([details](performance-research.md#phase-overlap-cannot-reach-the-roofline)) |

**Leads.** The CUDA backend is ~6 % past the target and the OpenCL path 1.12× short.
**Goal 1 — the low-power gap — is closed by measurement**, not by exhaustion of effort:
traffic supplies at most 17 % of the 570 MHz deficit, the instruction-issue route is
impossible on this ISA, and undervolting is untestable on a card that drives its own
display. See [the closure](#-closed-2026-07-29-the-low-end-is-not-reachable-by-traffic-and-no-other-mechanism-has-been-found).
What remains is speed and efficiency at and above the band, where the pipeline sits at
57 % of a 19.8 ms floor and r1 and r2 hold 65 % of the gap:

0. ~~**Ship the power cap as a setting.**~~ **Done 2026-07-25** — `--pl W` and
   `--no-oc-reset`, per-GPU list syntax, clamped to the band the driver reports,
   restored on exit including on Ctrl+C. See [usage.md](usage.md#power-limit) and
   [overclocking.md](overclocking.md). **220 W is the operating point to recommend**:
   55.4 sol/s for 219.5 W, and the point where MXBM's lead over a *equally capped*
   lolMiner is at its best value. ~~re-measure lolMiner under a cap of its own~~ —
   **done 2026-07-28, and it changed the answer**: see
   [Both miners under the same cap](#both-miners-under-the-same-cap). Remaining follow-up:
   whether MXBM should
   *default* to a cap rather than only offering one, which is a release decision rather
   than a technical one and is parked with the other pre-release questions. Note the
   general case is not settled by one card: a default derived from an RTX 4070 Ti SUPER
   has no standing on hardware whose curve nobody has swept, so `--pl auto` would need
   MXBM to find the knee itself.
0b. **The footprint is a REACH lever, not a speed lever.** 7.46 GiB against a 3 GB design
   target, and shrinking it is what put the CUDA backend on 8 GB cards (lead 4). What it
   no longer is, is a way to go faster: bytes were measured and they are nearly free —
   shrinking round 2's and round 3's records to 16 B, well past what the record audit
   allows, is worth ~4 ms of 35. See
   [bytes are nearly free](performance-research.md#bytes-are-nearly-free-per-element-work-is-not). This entry
   used to claim footprint "is the only one that moves both axes at once"; it moves one.
1. **Settle the comparison.** Accepted pool shares over a fixed interval, the only metric
   independent of either miner's counters. `docs-internal/MINER_COMP.md` has the protocol
   and the sample-size arithmetic. The margin (56.1 ± 2.3 against 53) is real but thin
   enough that this matters.
2. ~~**Wire the CUDA backend into the miner.**~~ **Done 2026-07-25** — `--solver
   auto|cuda|opencl|gpu|ref`, CUDA preferred, OpenCL kept as the portable fallback.
3. **Overclocking.** Deferred until parity was in sight. BeamHash III is bandwidth-bound,
   so a memory offset scales it close to linearly — and it applies to lolMiner equally,
   so the honest comparison is overclocked against overclocked. Design decisions, the
   verified NVML ranges and the verified lolMiner behaviour are in
   [overclocking.md](overclocking.md). `mxbm --benchmark BEAM-III` exists to A/B the
   settings without a pool.
4. ~~**Per-path VRAM budget on the CUDA side.**~~ **Done 2026-07-26.** The CUDA backend
   now picks its geometry from `rb_geometry_for()` — the same arithmetic the OpenCL path
   uses, moved to `src/gpu/rowbucket_geom.{h,cpp}` so the two cannot drift. It was
   labelled "reach, not speed", and reach is what it turned out to be worth: a large one.

   **The ladder is free to walk on CUDA.** `bb`/`sm`/`cap` were already kernel
   *arguments*, and everything compile-time is invariant along the `bb + sm = 17` line —
   the grid is 2^17 blocks, the staged group is `capacity / 2^17` = 264, and the
   128-entry chain table is a perfect hash for the `24 - bb - sm` = 7 bits left varying.
   So the coarser geometries needed no new code, only permission. Measured, one binary,
   200 distinct nonces, KAT green and drops zero at every rung:

   | geometry | footprint | ms/solve | sol/s |
   |---|---|---|---|
   | (16,1) | 7.46 GiB | 35.1 | 58.0 |
   | (15,2) | 6.88 GiB | 37.8 | 53.8 |
   | (14,3) | 6.50 GiB | 42.8 | 47.5 |

   **What it reaches.** CUDA has no `CL_DEVICE_MAX_MEM_ALLOC_SIZE`, which is the limit
   that binds OpenCL below 12 GB
   ([limitation 1](HW_REQUIREMENTS.md#1-below-11-gb-opencl-is-capped-by-its-single-allocation-limit)),
   so the CUDA ladder is bounded by total VRAM alone:

   | card | OpenCL | CUDA before | CUDA now |
   |---|---|---|---|
   | 16 GB | (16,1) | (16,1) | (16,1) |
   | 12 GB | (14,3), single-alloc bound | (16,1) | (16,1) |
   | 10 GB | sort path, 215 ms | (16,1) | (16,1) |
   | 8 GB | **nothing** — a reduced seed layer mines nothing | **refused** | **(15,2), 37.8 ms** |

   The 8 GB row is the point. `available()` demanded (16,1)'s 7.46 GiB + 1 GiB, so every
   8 GB Ampere/Ada card was turned away — and the OpenCL fallback cannot host a full seed
   layer at 284 B/element either, so those cards mined *nothing at all*. They now run the
   full search one rung down. The floor for the fast path is **7.6 GiB** of total VRAM.

   **The prediction is checked against the allocator, not trusted.** `rb_geometry_for()`
   sizes against *total* VRAM; a compositor can be holding a gigabyte of it. The
   constructor now steps down on real allocation failure, verified by holding VRAM
   hostage from another process: at 7.39 GiB free it walks 65536 → 32768 and mines
   correctly at (15,2).

   **This also fixed a leak.** `CudaSolver`'s constructor throws when a device is too
   small, and a throwing constructor never runs its own destructor — so every `cudaMalloc`
   made before the throw leaked for the life of the process. Any caller catching the
   throw and retrying smaller got *less* memory each attempt. The frees moved into
   `~Impl`, which runs either way. Under the VRAM-hostage test this is the difference
   between three geometries failing and one.
4a. **The 8σ bucket capacity is load-bearing — it is not the slack this document twice
   called it.** `fb_cap_for` reserves `mean + 8σ + 32` per bucket, 215 of 743 slots at
   bb=16, and both the sizing comment and lead 5 below recorded the gap to the measured
   ~4.3σ maximum as available. Measured on CUDA (`MXBM_OCC`, 201 solves × 5 scatter
   stages, unclamped counters): the worst bucket ever seen is **4.6–5.35σ, 85–91 % of
   what is reserved**. That reads like 3σ of slack, and it is not, because the tail must
   be priced per *draw* and a miner draws 2.45e6 solves × 5 stages × 65536 buckets =
   **8.0e11 bucket occupancies per day**:

   | reservation | cap | overflows/day | MTBF |
   |---|---|---|---|
   | 6σ | 697 | 8.1e-01 | ~1 dropped element **per day** |
   | 7σ | 720 | 8.1e-04 | 3.4 years |
   | 8σ | 743 | 4.0e-07 | 6900 years |

   A dropped child can be a valid solution's ancestor, and `bucketDrops != 0` fails the
   gate every run is held to. So the geometry ladder, not a tighter cap, is the lever for
   fitting a smaller card — and the "8σ→6σ reduction" lead 5 records as moot-but-real was
   never real.
4b. **Optimize the sort path.** *(Started 2026-07-28; **214.8 → 190.4 ms** so far — see
   [the round-4 mix anomaly](performance-research.md#the-round-4-mix-anomaly-was-a-runtime-constant-270--82-ms),
   which was the first thing on it and is now closed.)* It had been parked at ~214.6 ms
   since 2026-07-24, when the row-bucket path took over and every optimization since went
   there: it never received compile-time round constants (worth **−32.5 %** on the fused
   path), index-only records, or the packed-record work. The first instalment of the
   constants is in and was worth −9.8 %; the other two are untouched. It is
   not a dead path — it is what runs on any device that cannot host a row-bucket
   geometry, which per
   [limitation 1](HW_REQUIREMENTS.md#1-below-11-gb-opencl-is-capped-by-its-single-allocation-limit)
   means 11 GB cards and anything whose OpenCL `max_alloc` is small, and the CUDA backend
   has no sort path at all. So this is reach and it is also the only path some users will
   ever run. Known starting points: the [record audit](performance-research.md#the-record-redundancy-audit)
   already took it 285 → 264 B/element and left one deliberate 4 B/element (~138 MB) of
   slack in `leaves[2]`; whether the row-bucket path's wins transfer at all is the open
   question, since the two differ in structure and not just in tuning.
5. ~~**Overlap the phases.**~~ **Closed — built, measured, and null by two independent
   mechanisms. Do not retry with streams.** The profile is genuinely complementary: the
   SM is busy 50 % of the wall clock and DRAM 58 %, at different moments, which prices a
   `max(SM-busy, DRAM-busy)` roofline at 20.6 ms — a **1.71× ceiling, ~96 sol/s**.
   Neither mechanism reaches it. Two streams cannot, because every round is **662 waves
   deep** and a second kernel can only start in the last wave's tail. A co-resident
   launch cannot, because a memory-stalled warp still holds its warp slot, so hosting
   `entry` inside round 3 costs **7.25 ms against its 2.77 ms standalone**. The binding
   resource is **resident blocks per SM** — worth ~1.6 ms per extra block — and shared
   memory is what caps it. Full measurements, the positive control, the wave arithmetic
   and why the whole family is closed:
   [phase overlap cannot reach the roofline](performance-research.md#phase-overlap-cannot-reach-the-roofline).

**Ruled out — do not revisit** (all measured, see [What didn't work](performance-research.md#what-didnt-work)):
two-level bucketing, shared-memory magazines, warp-aggregated atomics, decoupling the
scatter for occupancy, SoA layouts in either global or local memory, and shrinking the
element below the `[7,7,6,5,1]` schedule. Added 2026-07-26: streaming stores, geometry
(17,0), a tighter bucket capacity, **and phase overlap by every available mechanism** —
two streams (grid depth) and one co-resident launch (warp slots). The binding resource
behind the last two is **resident blocks per SM**, not bandwidth, not bytes and not
scheduling policy — and that resource is worth ~1.6 ms per extra block, so it is the one
thing on this page still open. Larger blocks are *not* a substitute
([measured](performance-research.md#occupancy-is-worth-real-time-and-shared-memory-is-the-only-gate)).

---

## Benchmarks and gates

| Command | What it measures |
|---|---|
| `./build/bench_rounds 20` | Median ms/solve over 20 solves + drop-free gate |
| `./build/test_gpu_rounds` | Per-round timing, drop counters, survivor count |
| `./build/test_gpu_solver` | End-to-end solve incl. recovery and CPU verification |
| `./build/test_lds_collide` | All isolated kernel benches and the correctness oracles |
| `ctest --test-dir build` | Full suite (37 tests) |

Correctness gates that must stay green for any performance change:
`bucketDrops == 0`, `pairDrops == 0`, all three KAT goldens byte-identical, and the full
suite. The goldens are checked through every collision path
(`gpu_*_rowbucket`, `gpu_*_lds`, `gpu_*_legacy`) so a path-specific regression is caught.

Diagnostic environment variables, ablation build flags and the rules they were designed
around are catalogued with the experiments that use them:
[instruments](performance-research.md#instruments-diagnostic-and-ablation-flags).
