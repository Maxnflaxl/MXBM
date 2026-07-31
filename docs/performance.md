# GPU Performance

The permanent record of MXBM's GPU solver speed and efficiency: the headline figures,
the progress log, the power and efficiency curves, and the measurement caveats that
qualify them. Every number here was produced by a committed benchmark or test in this
repository — nothing is estimated or extrapolated unless explicitly labelled.

**Everything else lives in [performance-research.md](performance-research.md)**: the
solver architecture, every optimization that worked and every one that did not with
the measured mechanism behind each, the established hardware limits, and the open
leads. This page links into it throughout; that page is the one to read before
proposing a new lever, because most of them have already been tried.

**Target:** lolMiner does **~53 sol/s** on an RTX 4070 Ti SUPER at **stock clocks, no
special configuration** (user-measured). BeamHash III yields ~1.9 solutions per solve,
so 53 sol/s ÷ 1.9 ≈ **28 solve/s ≈ 36 ms/solve**. That is the bar.

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
| 2026-07-25 | CUDA port, algorithm unchanged | 41.0 | 41.6 | **47.6** | +0.6 | +1.5 % | [CUDA backend](performance-research.md#the-cuda-backend) | — |
| 2026-07-25 | 128-bit access on the round-2/3 records | 41.6 | 38.4 | **51.6** | −3.2 | −7.7 % | [CUDA backend](performance-research.md#the-cuda-backend) | — |
| 2026-07-25 | 128-bit access on the remaining records | 38.4 | 35.2 | **56.1 ± 2.3** | −3.2 | −8.3 % | [CUDA backend](performance-research.md#the-cuda-backend) | [cp.async, block size, pair record](performance-research.md#levers-tried-after-the-mio-fix--all-null) |
| 2026-07-25 | Un-pad round 2's record — 9th word to its own plane | 35.16 | 35.0 | **56.4** | −0.2 | −0.6 % | [Alignment pad](performance-research.md#the-round-2-alignment-pad) *(the win is −0.36 GiB; the speed is noise-level)* | — |
| 2026-07-26 | Perfect chain table + group spill → `kFCap` 320 | 35.0 | 34.1 | **58.0 ± 0.4** | −0.9 | −2.6 % | [Group cap](performance-research.md#shipped-the-group-cap-no-longer-has-to-cover-the-tail-125-ms), [Occupancy](performance-research.md#occupancy-is-worth-real-time-and-shared-memory-is-the-only-gate) | [streaming stores, warp-aggregated gi, (17,0), sub-pass, co-tenant entry](performance-research.md#bytes-are-nearly-free-per-element-work-is-not) |
| 2026-07-31 | **Speculative entry co-scheduling** — r4's launch hosts the next nonce's entry as interleaved co-blocks | 33.85 | 33.4 | **59.5** | −0.45 | −1.3 % | [Co-blocks](performance-research.md#co-blocks-the-third-overlap-mechanism-works--and-it-is-worth-04-ms-not-14), [ships](performance-research.md#speculative-entry-co-scheduling-ships-in-the-miner-045-ms) | [pipe2 1:1, split host, r2/r3 hosts](performance-research.md#fused_pair-two-solves-rounds-in-one-launch--the-familys-ceiling-is-05-ms), [register-forced occupancy](performance-research.md#occupancy-is-closed-from-both-resources--r2-sits-on-the-whole-register-file) |
| 2026-07-31 | Below-the-floor pair: r2's 16 B record in one `LD.128` + the terminal round joins the perfect table | 33.4 | 33.2 | **59.7 ± 0.8** | −0.2 | −0.7 % | [Below-the-floor levers](performance-research.md#two-below-the-floor-levers-clear-noise-on-cuda-r2s-pair-record-in-one-ld128-and-the-terminal-round-joins-the-perfect-table-022-ms) *(delta pinned by ×9 replay: r2 −0.145, terminal −0.07; the ± is the 60 s-window σ of the live session below)* | — |

| | sol/s | ms/solve | |
|---|---|---|---|
| **OpenCL** | 49.7 | 40.0 | fallback / `--solver opencl` — 2026-07-31, after [the match-win backport](performance-research.md#the-cuda-match-wins-backported-to-opencl-perfect-table--spill--per-round-caps-06-ms) |
| **CUDA** | **59.8**[^drift] | **33.3**[^drift] | **shipping** — default when a CUDA device is present |
| **Target** | 53.0 | 35.8 | lolMiner, stock — user-measured |

The CUDA row is **4 × 120 s, 14,296 solves** (`benchmarks/headline.sh`, 2026-07-31, the
first controlled run with [speculative entry
co-scheduling](performance-research.md#speculative-entry-co-scheduling-ships-in-the-miner-045-ms)
shipped), 0.0 % spread across the four runs, measured under stated conditions rather
than opportunistically: stock 285 W, card at thermal equilibrium after a discarded 240 s
warmup, 2670 MHz / 10251 MHz / 284.1 W / 69 °C throughout. The previous controlled
figure (2026-07-28, same conditions, pre-speculation) was 33.8 ms / 58.9 sol/s.

[^drift]: **Carries a ~2.5 % cross-session band** (narrowed 2026-07-31 from the ~5 %
    it opened at). Within one session the measurement is tight — six repeats spread
    0.3 % — and every session since 2026-07-26 has landed within ~2.5 % of its peers;
    the single 35.6 ms outlier that set the original band never recurred and stays on
    record. See [how far these figures
    reproduce](#-the-absolute-figures-reproduce-to-03--within-a-session-and-5--between-sessions).
    Every A/B on this page was interleaved, so the deltas are unaffected; only the scale
    moves. OpenCL is a single 60 s miner benchmark (1,494 solves, 2026-07-31), measured
    opportunistically rather than under the controlled-conditions protocol.

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

**Re-confirmed on the current build, 2026-07-31.** A 20-minute live HeroMiners
session on the shipping binary (stock 285 W, worker-logged, tallied from the pasted
console):

| window | samples | median | mean | σ | min | max |
|---|---|---|---|---|---|---|
| 60 s | 20 | **59.35** | 59.57 | **0.82** | 58.6 | 61.7 |
| 15 s | 83 | **59.60** | 59.51 | 1.93 | 54.2 | 63.9 |

Both window medians agree with the benchmark figure (59.7) to under 1 %, shares ran
**47 accepted / 0 stale / 0 rejected**, and the card held 284 W / 2595–2685 MHz /
66–68 °C throughout. Per this page's own model the *session-mean* uncertainty is the
Poisson one (~71,000 solutions → ±0.2 sol/s); the ±0.8 carried on the progress row
is the 60 s-window σ, the same convention as the 56.1 ± 2.3 session above.

It also shows why **the peak must not be quoted**. Individual 15 s windows reached
**61.5 sol/s**, which is tempting and wrong: it is the maximum of 304 draws from a noisy
distribution, and for σ = 3.27 the expected maximum of 304 draws is ≈ 67 — so 61.5 is
unremarkable rather than evidence of a higher true rate. Narrowing the window inflates
the peak (61.5 at 15 s vs 58.9 at 60 s) while leaving the median untouched, which is the
signature of noise rather than throughput. The three samples below 45 sol/s are likewise
artefacts: they are consecutive, and coincide with a lolMiner benchmark and an MXBM
benchmark being run against the same GPU. The CUDA
backend is **~6 % past the target** and the OpenCL path 1.07× short — read
[the caveats](performance-research.md#the-cuda-backend) before treating the target as beaten.

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

**Narrowed to ~2.5 %, 2026-07-31.** Written when it was one slow session out of four,
the 5 % band has not been earned again since: across 2026-07-26 → 07-31 the ledger now
holds nine-plus same-binary stock sessions, and apart from the single 35.6 ms event
every session median lands within ~2.5 % of its peers — the strongest cross-day pair
being [the full power sweep repeated two days apart](#the-sweep-reproduces-across-sessions-to-1),
twelve caps all within 2.5 % (mean −1 %). The mechanism side also firmed up: the
locked-clock reference removes the suspected V/f variable by construction and
reproduced to 0.00 % across the interval the anomaly originally appeared on. **The band
this page carries is therefore ~2.5 %** — the worst same-binary cross-session delta
actually observed since — with the 2026-07-26 event kept on record as a one-time,
unexplained, on-GPU slowdown. *Reopening conditions: any same-binary session median
landing more than ~2.5 % from its peers restores the 5 % band; and the cross-day
locked-clock repeat this section owes is still owed — the two locked runs share one
calendar day.*

### The locked-clock reference is stable to the digit — run it 2026-07-30

The test proposed for this has now been run: lock the SM and memory clocks
(`LGC=2600 LMC=10251 benchmarks/headline.sh`) and repeat, so the V/f point is pinned
rather than left to the card's discretion. **Two runs, bracketing 70 minutes of
continuous varied load** — the full 100–285 W head-to-head sweep, both miners, ran
between them.

| | ms/solve | sol/s | SM | mem | W | °C | capped |
|---|---|---|---|---|---|---|---|
| **run A** (21:20) | **34.30** | 58.35 | 2610 | 10251 | 261.8 | 67 | **none** |
| **run B** (22:49) | **34.30** | 58.45 | 2610 | 10251 | 262.1 | 67 | **none** |
| Δ | **0.00 %** | +0.17 % | 0 | 0 | +0.11 % | | |

Six runs of 120 s each, 0.3 % within-run spread on both.

**The pin removes the suspected mechanism by construction, and that is the point.** At
stock the card sits at `sw_power_cap` **99–100 % of the time**, so the SM clock is not a
free variable — it is whatever 285 W happens to buy, and the surviving hypothesis for the
cross-session drift was that the same 285 W buys a different point on the V/f curve on
different days. At a locked 2610 MHz the card draws **262 W against a 285 W limit with
`clocks_event_reasons: none`** — no throttle of any kind, 23 W of headroom unused. The
card is no longer buying clock with watts at all.

**And the timescale matters.** The original anomaly was not a day-to-day effect: it was
34.15–34.20 ms early in a session and 35.49–35.72 ms *a few hours later*, same machine,
unchanged. Runs A and B here are 106 minutes apart with the card driven across its whole
power range in between — the same timescale, and a far more hostile one — and ms/solve
did not move at all. Against up to **5 %** unlocked, that is the difference between a
number that describes the build and a number that describes the afternoon.

**So: use the locked-clock figure to regression-test builds.** `34.30 ms at LGC=2600
LMC=10251` is now the reference, and a build that moves it has moved. Quote the *stock*
figure when the question is what the card actually does for a user, and carry the ~5 %
band with it.

**What is still not settled.** Both runs are the same calendar day. This is strong
evidence that the regime is a clock/voltage effect — pinning the clock pinned the number
across exactly the interval on which the anomaly first appeared — but a genuine
cross-day repeat is still owed, and the slow regime has not recurred since 2026-07-26 to
be caught in the act.

### The memory junction temperature is not observable on this card — dead end

*(Probed 2026-07-30.)* The memory half of that hypothesis had a specific, cheap test
behind it. GDDR6X raises its refresh rate as it heats, which lowers *effective* bandwidth
while the reported memory clock sits still at 10251 MHz — which is exactly the signature
above: a slowdown that is on the GPU, ~2–3 % in step across every round, and invisible in
every clock we log. `nvidia-smi` reports `temperature.memory` as N/A here, but that only
says its own query does not resolve; NVML might still answer `NVML_FI_DEV_MEMORY_TEMP`
(field 82) through `nvmlDeviceGetFieldValues`, which `src/gpu/nvml.cpp` has never called.

**It does not.** Field 82 returns `Not Supported`, idle and under full load alike.

The probe was positive-controlled, because a bare "Not Supported" is indistinguishable
from a wrong struct layout, a failed `dlsym`, or a handle that is not really a device. The
same call, in the same array, also asked for fields this card certainly does answer:

| field | | result |
|---|---|---|
| 82 | `MEMORY_TEMP` | **Not Supported** |
| 186 | `POWER_INSTANT` (control) | Success — 8 852 mW idle, 285 779 mW under load |
| 83 | `TOTAL_ENERGY` (control) | Success — monotonic mJ counter |
| 193–196 | the whole `T.Limit` family | **Not Supported** |

The controls resolve to the right values to three digits — 285.8 W under load against a
285 W board limit — so the mechanism is proven and the field is genuinely absent rather
than mis-called. `nvidia-smi -q -d TEMPERATURE` agrees from the other side: `Memory
Current Temp: N/A`, `Memory Max Operating Temp: N/A`, and `GPU Current T.Limit Temp: N/A`.
This is a GeForce restriction, not a driver-version or an API-surface problem, and no
amount of NVML gets around it.

**Two consequences.** The VRAM-temperature column lolMiner has and we do not is **not
available to us on this class of card** — that item should stop being carried as work.
And the GDDR6X-refresh explanation of the cross-session regime is not *disprovable* with
the instruments this card exposes; it stays a hypothesis, and the locked-clock reference
below is what has to carry the question instead. Testing it directly would need
out-of-band telemetry the card does not offer.

**One thing worth keeping from the probe.** Field 83,
`NVML_FI_DEV_TOTAL_ENERGY_CONSUMPTION`, *is* supported: a monotonic millijoule counter
since driver load. Every efficiency figure on this page is currently integrated from
`power.draw` samples at 5 Hz, which is a reconstruction of exactly what that counter
already holds exactly. Reading it before and after a run would give J/solution directly,
with no sampling error and no dependence on sample rate. Not done here — it would change
the basis of numbers mid-sweep, which is the one thing a reproducibility section must not
do — but it is the right instrument for the next one. *(Wired 2026-07-31: `--benchmark`
now prints J total, mean W and J/solution from the counter, `--tune`'s per-point draw
uses it over the sampler when present, `/summary` exposes the raw counter as `Energy_J`
for rig software to diff, and `benchmarks/lib.sh` gained `bench_energy_mj` for sweeps.
Existing tables keep their sampled basis until each is next re-measured whole.)*

**How to quote a number from this page:** use the controlled figure with its conditions
attached — 33.3 ms / 59.8 sol/s at stock 285 W, 2670 MHz, 69 °C (2026-07-31, with
speculative entry) — and carry the ~2.5 % cross-session band (narrowed 2026-07-31; see
above). Do not re-derive a headline
from a short run: see the note on solutions/solve under the progress table.

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
Both crossings survive the proper head-to-head almost unchanged (210 W and 256 W), so the
*band* was right — but the margin inside it does not survive, because at 220 W lolMiner
capped to 220 W does 54.35 sol/s at 0.2477, not 53.27 at 0.2232. The advantage there is
**0.9 %**, not the ~10 % this section used to report.

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
knob does not exist yet — see [lead 0](performance-research.md#current-focus-and-open-leads).

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

**Re-measured 2026-07-30** on a second session, and extended down to the card's 100 W
floor. The table below is that second session; the reproducibility check against the
first is the subsection that follows.

| cap | MXBM sol/s | MXBM W | MXBM sol/s/W | lolMiner sol/s | lolMiner W | lolMiner sol/s/W |
|---|---|---|---|---|---|---|
| 100 W | 19.10 | 99.5 | 0.1919 | **23.05** | 99.3 | **0.2320** |
| 110 W | 22.30 | 109.5 | 0.2036 | **27.75** | 109.2 | **0.2541** |
| 120 W | 25.65 | 119.8 | 0.2142 | **33.40** | 119.5 | **0.2794** |
| 140 W | 32.45 | 139.9 | 0.2320 | **40.45** | 139.6 | **0.2898** |
| 160 W | 38.80 | 160.1 | 0.2423 | **47.60** | 160.1 | **0.2973** |
| 175 W | 43.75 | 175.2 | 0.2497 | **52.25** | 174.7 | **0.2991** |
| 180 W | 45.15 | 180.1 | 0.2508 | **52.35** | 179.6 | **0.2914** |
| 190 W | 48.15 | 189.9 | 0.2536 | **53.05** | 189.8 | **0.2796** |
| 200 W | 51.25 | 199.8 | 0.2566 | **54.00** | 199.6 | **0.2706** |
| 210 W | **53.95** | 209.6 | **0.2575** | 53.90 | 209.6 | 0.2572 |
| 220 W | **54.85** | 219.5 | **0.2499** | 54.35 | 219.4 | 0.2477 |
| 240 W | **56.95** | 239.5 | **0.2378** | 53.75 | 236.6 | 0.2272 |
| 255 W | **57.85** | 254.3 | **0.2275** | 53.90 | 237.5 | 0.2269 |
| 285 W | **59.05** | 284.0 | 0.2079 | 53.65 | 237.4 | **0.2259** |

![Speed, efficiency and power drawn, both miners at the same caps](tools/power-curve.svg)

**The result is three-part, and only the middle part is the one this document used to
claim.** Crossings are interpolated from the table:

- **Below ~210 W, lolMiner wins on both, and the margin grows as the cap tightens —
  until the very bottom, where it does not.** At 180 W it does 52.35 sol/s to MXBM's
  45.15 (**+15.9 %**); at 160 W, 47.60 to 38.80 (**+22.7 %**); at 120 W, 33.40 to 25.65
  (**+30.2 %**). Then at 100 W the gap *narrows* to **+20.7 %** — see the floor below.
- **Between ~210 W and ~256 W, MXBM wins on both** — by 0.9 % on each at 220 W, widening
  to 6.0 % speed and 4.7 % efficiency at 240 W. (Band computed on a fine grid by
  `docs/tools/plot_power.py`, which reads this same table: 209.9–255.9 W.)
- **Above ~256 W, MXBM is faster and lolMiner is more efficient**, which is what the
  stock-versus-stock comparison always showed.

Three facts that reframe the whole comparison:

**lolMiner barely responds to the cap at all.** From 285 W down to 180 W it moves 53.65 →
52.35 sol/s — it gives up **2.4 %** of its speed for **24 %** less power. MXBM over the
same range gives up 24 %. And above ~237 W the cap stops doing anything: the 240, 255 and
285 W rows all draw 236.6–237.5 W, which is why it never reaches the board limit.

**Both miners have an interior efficiency optimum; lolMiner's is higher and further
left.** MXBM peaks at **0.2575 sol/s/W at 210 W**, lolMiner at **0.2991 at 175 W** — a
**16.2 %** gap between one miner at its best and the other at its best. Read both peaks
as *regions*, not points: MXBM's 200 and 210 W rows are 0.35 % apart and lolMiner's 160
and 175 W rows 0.6 %, which is inside this measurement's own repeatability.

**The left edge is now the card's, not the sweep's — and there is nothing hiding down
there.** This was the open question: lolMiner's efficiency was still climbing at 180 W
when the sweep stopped, so its peak might have been unmeasured and its low-power
advantage larger than recorded. Taking the sweep to the card's 100 W floor settles it.
**Both curves fall away monotonically below their peak**, and lolMiner falls *faster*:
its advantage over MXBM peaks at +30.2 % around 120 W and then shrinks to +20.7 % at
100 W. Its peak is genuinely interior and genuinely measured, and no hidden low-power
regime exists for either miner.

Put the other way, and this is the sharpest form of it: **lolMiner at 175 W delivers
52.25 sol/s for 174.7 W, where MXBM needs 209.6 W to deliver 53.95.** Nearly the same
throughput for 35 W less. That is the argument this document used to make in the other
direction at 220 W, and at the efficient end of the curve it belongs to them.

What MXBM keeps is the top end: **59.05 sol/s against a ceiling of ~54.0**, a **9.4 %**
higher maximum throughput that lolMiner cannot reach at any setting.

### The sweep reproduces across sessions to ~1 %

*(The `cad8fde` question, answered for the comparison rather than for a single number.)*

Everything above was one interleaved session, which is exactly right for the *comparison*
and says nothing about whether the absolute numbers survive a different day. The whole
sweep was therefore re-run on 2026-07-30, two days after the first, on the same binaries.

| cap | MXBM 07-28 → 07-30 | Δ | lolMiner 07-28 → 07-30 | Δ |
|---|---|---|---|---|
| 120 W | 26.30 → 25.65 | −2.5 % | 33.50 → 33.40 | −0.3 % |
| 140 W | 32.65 → 32.45 | −0.6 % | 40.60 → 40.45 | −0.4 % |
| 160 W | 39.35 → 38.80 | −1.4 % | 48.60 → 47.60 | −2.1 % |
| 175 W | 43.95 → 43.75 | −0.5 % | 52.00 → 52.25 | +0.5 % |
| 180 W | 46.05 → 45.15 | −2.0 % | 52.65 → 52.35 | −0.6 % |
| 190 W | 48.75 → 48.15 | −1.2 % | 53.25 → 53.05 | −0.4 % |
| 200 W | 52.15 → 51.25 | −1.7 % | 54.25 → 54.00 | −0.5 % |
| 210 W | 54.15 → 53.95 | −0.4 % | 54.35 → 53.90 | −0.8 % |
| 220 W | 55.40 → 54.85 | −1.0 % | 54.35 → 54.35 | 0.0 % |
| 240 W | 57.20 → 56.95 | −0.4 % | 53.30 → 53.75 | +0.8 % |
| 255 W | 58.00 → 57.85 | −0.3 % | 53.45 → 53.90 | +0.8 % |
| 285 W | 59.15 → 59.05 | −0.2 % | 53.60 → 53.65 | +0.1 % |

**This is much better than the ~5 % band the headline section warned about at the
time** (this sweep pair is, in fact, the evidence that later narrowed it to ~2.5 %).
Every point
of both miners reproduces within 2.5 %, and all but three within ~1 %. The board power
drawn at each cap is identical to a tenth of a watt.

Two things are worth separating in it. **MXBM's deltas are all the same sign** — twelve
of twelve slightly slower on the second day, mean about −1.0 % — which is a small session
offset, not noise; noise would change sign. **lolMiner's are mixed** (+0.8 % to −2.1 %,
mean ≈ −0.2 %), i.e. genuinely noise-like. So the second session was marginally
unfavourable to MXBM, and every conclusion above is drawn from *that* session, which
makes them conservative rather than flattering.

**No conclusion changed.** Both crossings moved by less than one grid spacing (~212 →
~210 W, ~257 → ~256 W), both efficiency peaks moved one row within a flat region, and
lolMiner's saturation ceiling reproduced at 236.6–237.5 W against 235.7–235.8 W. The
band to attach to any single *absolute* number from this page is still the headline
section's ~5 %; what this shows is that the *comparison* is stable well inside it.

### Why we lose the low end: watts buy us less clock

The mechanism is visible in the clocks — **down to 160 W, below which it inverts.**
Mean SM clock per cap, both repeats, 2026-07-30:

| cap | MXBM SM clock | lolMiner SM clock | gap |
|---|---|---|---|
| 100 W | 795 MHz | 435 MHz | **+360** |
| 110 W | 922 MHz | 548 MHz | **+374** |
| 120 W | 1035 MHz | 678 MHz | **+357** |
| 140 W | 1320 MHz | 956 MHz | **+364** |
| 160 W | 1560 MHz | 1556 MHz | +4 |
| 175 W | 1792 MHz | 2392 MHz | **−600** |
| 180 W | 1882 MHz | 2468 MHz | −586 |
| 190 W | 2025 MHz | 2535 MHz | −510 |
| 200 W | 2205 MHz | 2595 MHz | −390 |
| 210 W | 2378 MHz | 2640 MHz | −262 |
| 220 W | 2452 MHz | 2685 MHz | −233 |
| 240 W | 2535 MHz | 2745 MHz | −210 |
| 255 W | 2588 MHz | 2745 MHz | −157 |
| 285 W | 2670 MHz | 2745 MHz | −75 |

**Above 160 W this is the story the section was written to tell.** MXBM's kernels cost
more power per clock, so a tightening cap takes clock away from us faster than from them
— 75 MHz behind at stock, 600 MHz behind at 175 W. The suspect was
DRAM traffic: MXBM moves [14.23 GB/solve](performance-research.md#current-focus-and-open-leads) at geometry
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

#### ⚠ Below 160 W the clock gap INVERTS, and the clock explanation stops applying

*(New 2026-07-30, from extending the sweep to the card's 100 W floor. **Resolved
2026-07-31** — the duty-cycle reading below is refuted and the work-per-clock reading
confirmed; see the addendum after the candidate readings.)*

Everything above is a story about clock: lolMiner clocks higher under a cap, and that is
why it wins. **Below 160 W that is simply not what happens.** At 140 W and under, **MXBM
holds a ~360 MHz HIGHER SM clock than lolMiner and still loses by 20–30 % on sol/s.** The
sign flips cleanly at 160 W, where the two are within 4 MHz of each other, and the
inversion is present in both repeats at every point below it — it is not one bad sample.

| cap | clock gap | sol/s gap |
|---|---|---|
| 100 W | MXBM **+360 MHz** | lolMiner **+20.7 %** |
| 120 W | MXBM **+357 MHz** | lolMiner **+30.2 %** |
| 140 W | MXBM **+364 MHz** | lolMiner **+24.7 %** |
| 160 W | +4 MHz | lolMiner +22.7 % |
| 180 W | MXBM −586 MHz | lolMiner +15.9 % |

So in the bottom third of the range our deficit is **not a clock deficit at all** — it is
work done per clock, and lolMiner is getting roughly twice as much of it while running
its core far slower. Whatever explains 175 W and above does not explain this, and the
"bytes → watts → clock" chain cannot: it predicts the miner moving fewer bytes clocks
*higher* under a cap, and below 160 W lolMiner clocks lower.

**Two candidate readings, neither tested.** Either lolMiner is duty-cycling at these caps
— bursting and idling, so a median `clocks.sm` samples a mixture rather than the clock
during work, which would make the whole column mean something different for it than for
us — or its 4G variant genuinely does far more useful work per cycle at low clock in a
way that only becomes visible once the clock is low enough. The first is testable cheaply
from the 5 Hz telemetry already captured, by looking at the *distribution* of the clock
samples rather than their median: a duty-cycling miner is bimodal.

**What it does not change.** Every conclusion in the sections above is drawn at 175 W and
higher, where the clock story holds and is confirmed by this session. What it removes is
the temptation to extrapolate that story downward — the low end has a different mechanism
and nobody has identified it.

**Addendum 2026-07-31 — the mechanism is identified, and it is the second reading.**
The cheap test was run: clock samples at 10 Hz under both miners at 100 W and 140 W.
lolMiner's distribution is broad and **unimodal** — no burst/idle bimodality, so it is
not duty-cycling; it genuinely executes at a median 480 MHz at 100 W and still outsolves
MXBM at 750 MHz by ~22 %. The deficit is **~2.3× useful work per core cycle**, real and
measured. Below ~190 W everything is issue-bound (103.7 ms at 100 W is 2.94× stock at
3.38× less clock — even the DRAM-bound rounds stop saturating DRAM, because the LSU rate
scales with core clock), so the currency down there is instructions issued, and MXBM's
organization — sub-mask rescans, chain walks, per-element bookkeeping — simply spends
more of them. *(That last clause was tested the same day and is wrong: [the
reorganization probes](performance-research.md#the-solver-reorganization-probes-the-cycle-deficit-is-not-bookkeeping)
measured the bookkeeping at ~0.6 ms of round 1's 4.9 — the deficit is real, its
mechanism is not bookkeeping, and it is unexplained.)* *(Explained later that day by
profiling lolMiner itself: it is a state-storing streaming design — 64 B/element,
17.7 GB/solve, no derive or rebuild arithmetic anywhere — so at low caps it simply
has almost no instructions to issue. See [lolMiner measured under
ncu](performance-research.md#lolminer-measured-under-ncu-the-state-storing-design-confirmed--and-its-54-sols-ceiling-is-a-dram-roofline).)* What our own knobs recover is
measured and small: geometry (17,0), which
deletes the rescan, crosses over at ~190 W and buys 1 % in the 140–180 W band and 2.6 %
at the floor (`MXBM_BB=17`, needs 8.35 GiB) — *a result that failed same-day
reproduction in both the miner loop and the sweep's own binary; auto-selection was
built, verified and then disarmed the same day (see
[docs/overclocking.md](overclocking.md#the-power-limit-geometry-policy--built-verified-and-currently-disarmed)
and the eco-sweep addenda)*; the byte-heavy `MXBM_R2_FULL` **loses at
every cap** — re-derivation is the right trade at all power levels. Full tables:
[the eco sweep](performance-research.md#the-eco-sweep-170-crosses-over-below-190-w-r2_full-never-does)
and [the duty-cycle probe](performance-research.md#lolminer-is-not-duty-cycling--the-low-end-gap-is-real-work-per-clock).

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
advantage is a band, and the band is where it was measured: **lolMiner below ~210 W,
MXBM between ~210 W and ~256 W, and MXBM alone above it** at a ceiling lolMiner cannot
reach at any setting (59.05 sol/s against ~54.0). Recommending MXBM for a rig capped
below 210 W is not supportable on this hardware, and no change in this document's reach
would make it so. **Effort belongs on goals 2 and 3**, where the pipeline runs at 57 % of
its 19.8 ms floor and rounds 1 and 2 hold 65 % of the gap.

*Reopening conditions, since the ledger's own rule is that a closure states them: a
no-replay 180 W A/B that puts the whole-solve rate materially above 92 MHz/GB; a
narrowing worth more than the 1.07 GB the audit allows; a machine where the compute GPU
does not drive the display, which makes undervolting measurable; or a mechanism nobody
has proposed. The first two would move the arithmetic, not the conclusion — 48 % of all
traffic is a long way from 1.07 GB.*

*2026-07-31: the "mechanism nobody has proposed" arrived — instruction issue as the
low-cap currency — was measured, and the closure holds. It is real (geometry (17,0)
crosses over below ~190 W and buys up to 2.6 %) and it is small: every instruction-side
trim this pipeline exposes totals a few per cent against a 20–30 % deficit that the
duty-cycle probe pinned to ~2.3× work-per-clock in the solver's organization itself.*

*Later the same day, "the solver's organization" was itself put to the test and the
closure got stronger. A reorganization of the match — counting-sort runs, keys-only
8 B staging, register pairing; chains, rescans, spill and most barriers deleted — was
built as a standalone probe, produced the exact pair multiset of the shipping round-1
kernel, and was **1.75× slower**. Its ablation carve decomposed round 1's cycles:
derive ~2.5 ms + emit arithmetic ~1.8 + all bookkeeping **~0.6** of the 4.9 ms
standalone kernel. The bookkeeping the reorganization would have removed is 12 % of
the round, not 55 % — so the 2.3× work-per-clock deficit does not come from
bookkeeping, and its mechanism is again unidentified. Details:
[the reorganization probes](performance-research.md#the-solver-reorganization-probes-the-cycle-deficit-is-not-bookkeeping).*

*And later still that day, identified for good — from lolMiner's own kernels rather
than from ours. `ncu` on its closed binary: a state-storing streaming pipeline,
64 B/element layers, 17.7 GB/solve against our 13.0, no re-derivation anywhere, and
a sustained ~494 GB/s that makes its ~54 sol/s ceiling a DRAM roofline. The 2.3× is
the absence of our arithmetic, the closure stands with its mechanism finally named,
and the band structure of this whole comparison is one design trade held from
opposite ends. See [lolMiner measured under
ncu](performance-research.md#lolminer-measured-under-ncu-the-state-storing-design-confirmed--and-its-54-sols-ceiling-is-a-dram-roofline).*

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

## Apple Silicon (Metal) — M3 Max, 2026-07-30

A second reference platform. Everything else in this document is the 4070 Ti; this
section is the Mac, and the two are not comparable except in shape.

| path | median ms/solve | sol/s | notes |
|---|---|---|---|
| OpenCL, sort | ~505 (494.6 / 516.6) | ~3.8 | the only OpenCL path that runs here |
| OpenCL, LDS | — | — | `clCreateKernel` → `CL_INVALID_KERNEL` |
| OpenCL, row-bucket | — | — | `clCreateKernel` → `CL_INVALID_KERNEL` |
| Metal, fused row-bucket (rebuild) | 117 (117.4 / 116.9) | 16.2 | the CUDA-shipping configuration |
| **Metal, fused row-bucket (SEEDF)** | **101.5** (102.3 / 100.7) | **18.8** | **default on Apple**, see below |

**Apple's OpenCL cannot build the fused kernels at all.** Both the LDS and row-bucket
collision finders fail at `newComputePipelineState`, which is why every macOS-relevant
GPU test in `CMakeLists.txt` is pinned to `MXBM_NO_ROWBUCKET=1`. So on this hardware
OpenCL is confined to the slowest of the three finders, and Metal is not a faster
dialect of the same pipeline — it is the only way to run the fused one.

`bench_rounds`-equivalent figures above come from `MXBM_BENCH=20 test_metal_solver`,
median over 20 solves across two sessions, per this document's own rule that a maximum
of noisy draws reads high.

**The miner's per-solve overhead is ~0; what moves is thermal.** `--benchmark BEAM-III`
on an idle machine:

| run length | median | p5 | p95 | sol/s |
|---|---:|---:|---:|---:|
| 12 s | 119.3 ms | 116.6 | 126.3 | 14.3 |
| 45 s | 130.9 ms | 116.8 | 153.2 | 15.3 |

The **p5 is 116.8 ms in both — the pipeline's own 117 ms**, which agrees with the 1.6 ms
the GPU timing measures outside the kernels. Nonce iteration, stats and the difficulty
filter cost essentially nothing. What rises with run length is the median and the p95,
which is a MacBook throttling under sustained load, not overhead.

So quote 117 ms as the pipeline figure and ~15 sol/s as the sustained one, and expect
the gap between them to widen with ambient temperature and run length rather than being
fixable in software.

> An earlier revision of this section reported 155.2 ms/solve and attributed a ~38 ms
> gap to work outside the pipeline. That measurement was taken while builds and tests
> were competing for the same GPU; on an idle machine it does not reproduce. Benchmark
> figures here are only valid from an otherwise-idle machine.

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

### Where the 117 ms goes

Per-phase GPU time (`MXBM_METAL_TIMING=1`, from `MTLCommandBuffer`'s
GPUStartTime/GPUEndTime, so this is GPU execution and not the wall clock around the
submit):

| phase | rebuild | SEEDF (default) | |
|---|---:|---:|---|
| entry | 12.0 | 12.2 | |
| r1 | 23.4 | 26.2 | pays for the wider record |
| **r2** | **44.0** | **24.3** | **38 % of the solve, before** |
| r3 | 17.1 | 17.1 | |
| r4 | 14.0 | 14.2 | |
| terminal | 4.4 | 4.4 | |
| recover | 0.0 | 0.0 | |
| **GPU** | **115.0** | **98.4** | wall +1.6 ms |

Two things fell out immediately. The six per-round command buffers, each with a
`waitUntilCompleted`, cost **1.6 ms total** — batching them is not a lead. And r2 was the
whole story, which is what the rest of this section is about.

### r2's rebuild does not hide on Apple — 21 ms exposed, against 0.4 ms on Ada

`-DMXBM_METAL_ABL_DERIVE=2` replaces `rebuild_r2`'s 14 `siphash24` calls with a cheap
spread. Results are intentionally wrong; the ablation is safe because the rebuild reads
only threadgroup memory and registers, so the global access footprint is untouched and
the number measures compute alone.

```
baseline   entry 12.1  r1 23.4  r2 43.9  r3 17.1  r4 13.9  terminal 4.4  | GPU 115.0
ABL=2      entry 12.2  r1 23.4  r2 22.9  r3 17.1  r4 13.9  terminal 4.4  | GPU  93.9
```

**21.0 ms of exposed compute.** The same rebuild costs **0.4 ms** on Ada
(performance-research.md, "Register cap tuning") because Ada's memory-level parallelism
absorbs it. This is the single largest architectural divergence found between the two
backends, and it is where any further Apple work should start.

### The fix: round 1 stores what round 2 was rebuilding (SEEDF) — 14 % faster

Round 1 already computes the exact element round 2 reconstructs: r1 emits
`apply_mix(combine(a, b, 424), {li, ri}, 2, 424)`, and `rebuild_r2(pp, li, ri)` recomputes
precisely that from the two seed indices. So r1 stores it instead — its record goes from
2 u64 to 9 — and r2 reads it. **This is the CUDA source's `LM_SEEDF` idea, which loses on
Ada and wins here**, because the compute it removes is exposed on Apple and hidden on Ada.

Cold, uncontested, medians over 6–8 solves with the baseline bracketed either side:

| | r1 | r2 | GPU total | pipeline median |
|---|---:|---:|---:|---:|
| rebuild | 23.4 | **44.0** | 115.0 | 117.8 ms |
| **SEEDF** | 26.2 | **24.3** | **98.4** | **101.5 ms** |

**−16.3 ms, 13.8 %** — 16.2 → 18.8 sol/s. The ranges do not overlap (SEEDF 100.1–104.9,
rebuild 116.8–119.3). r1 pays 2.8 ms for the wider write; r2 saves 19.7 ms, which is the
ablation's 21 ms less the cost of reading the wider record. Every other phase is
unchanged to within noise, which is the internal check that the change did what it says.

The win is larger under thermal load, not smaller (r2 55.8 → 25.4 ms when hot), because
the rebuild's compute is itself part of what generates the heat.

SEEDF is the **default on Apple**. `MXBM_METAL_REBUILD=1` selects the CUDA-shipping
configuration; both kernel pairs live in the same metallib and the host picks at runtime,
so the A/B needs no rebuild.

> An earlier revision of this section predicted this change would net only ~4 ms and
> declined to attempt it. That arithmetic assumed ~251 GB/s effective bandwidth, inferred
> by dividing round 3's traffic by its wall time — but round 3 is not purely
> bandwidth-bound, so the real figure is far higher and the extra traffic much cheaper
> than predicted. Measure the thing; do not divide two numbers and call it a bandwidth.

### Three nulls, so they are not re-proposed

- **Threadgroup size.** Bracketed medians over 10 solves: 128 → 144.6, 192 → 126.7,
  256 → 117.0/117.3, 288 → 116.6, 320 → 116.8/116.9, 352 → 118.3, 384 → 136.0,
  512 → 144.8. Flat across 256–320; the 0.3 ms between them is inside the spread. An
  earlier single-solve sweep read 320 as a 1.2 % win — it did not replicate.
- **Occupancy via smaller groups.** Cutting `kFCap` to fit two threadgroups per core's
  32 KB, with `sm` raised to keep groups full, makes r2 monotonically *worse*:
  (FCAP, sm) = (320,1) → 118.5, (192,2) → 125.3, (160,2) → 129.1, (128,3) → 178.0.
  The extra groups pay table-init, barriers and sub-mask passes that outweigh anything
  the occupancy buys. Apple is not occupancy-limited here in the way Ada is.
- **128-bit vector loads and stores.** CUDA documents `4 x ST.128 + 1 x ST.64` beating
  `9 x ST.64`, because round 3's top stall there is MIO-queue *instruction* issue.
  Ported to `ulong2` on Metal it measures **neutral to slightly worse** (117.9 vs 117.1
  median), so the scalar form is kept and the complexity is not. Apple's compiler
  appears to coalesce these already.

### A measurement hazard worth knowing

Single-solve timings on this machine are thermally confounded. An identical
configuration read 115 ms early in a sweep and 133 ms late, and one FCAP sweep drifted
16.7 ms end to end — enough to invent a 14 % "win" out of nothing. Every figure above
is a median over >= 10 solves with the baseline configuration repeated first *and last*;
if the two brackets disagree by more than a millisecond or two, the run is discarded.

Not tuned beyond the above: the `bb + sm = 17` line is carried over from Ada. Register
counts are not observable on Metal (see `tests/test_metal_resources.mm`), so the
occupancy work that produced Ada's numbers has no direct equivalent here.

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
