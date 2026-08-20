# GPU Performance

The permanent record of MXBM's GPU solver speed and efficiency: the headline figures,
the progress log, the power and efficiency curves, and the measurement caveats that
qualify them. Every number here was produced by a committed benchmark or test in this
repository — nothing is estimated or extrapolated unless explicitly labelled.

**Everything else lives in [performance-research.md](performance-research.md)**: the
solver architecture, every optimization that worked and every one that did not with
the measured mechanism behind each, the established hardware limits, and the open
leads. Read it before proposing a new lever — most have already been tried.

**Target:** lolMiner does **~53 sol/s** on an RTX 4070 Ti SUPER at **stock clocks, no
special configuration** (user-measured). BeamHash III yields **2.006** solutions per
solve, so 53 sol/s ÷ 2.006 ≈ **26.4 solve/s ≈ 37.9 ms/solve**. That is the bar.

## Progress log

Median ms per solve, lower is better. "Worked" and "Didn't work" link into
[performance-research.md](performance-research.md).

Rows above the backend switch are `bench_rounds` **pipeline** medians; rows below are
**end-to-end** `solve()` medians including recovery and CPU verification — the stricter
measurement. The two agreed to within a millisecond on OpenCL where both were measured
(40.3 vs 41.0 ms), so the trend is continuous across the switch.

![Throughput and solve time, log scale](tools/progress.svg)

![Throughput and solve time, linear scale](tools/progress-linear.svg)

<details>
<summary>Why two scales, and what the error bars mean</summary>

Both charts are generated from the table below by `python3 docs/tools/plot_progress.py`,
so they cannot drift from it.

**Log** makes equal vertical distance mean equal *ratio*, matching the Δ % column, and is
the only way to see the early changes next to a 30× range. **Linear** starts both axes at
zero, so equal height means equal *absolute* change — which shows that almost all of the
ms was won early while almost all of the sol/s came late.

Two y axes are safe here because sol/s and ms/solve are one quantity inverted, so no
correlation between two things is implied. On the log chart the lines are near-mirror
images; on the linear chart the reciprocal's convexity makes them different shapes.
Relabelling one axis in the other's units is *not* available — `sol/s × ms` is ~1900 over
the OpenCL rows and ~1980 over the CUDA ones, so one converted ruler would misstate half
the chart by 4 %.

The x axis is optimization step, not calendar time: only three dates exist and 16 rows
fall on one of them, so dates are drawn as bands. The dashed divider is the change of
measurement described above, not just of backend.

**Error bars are Poisson on the observed solution count, `1/√N`, and cannot be got by
repeating the run.** Five repeats of the 2026-07-26 build read 58.3/58.4/58.4/58.4/58.4,
σ = 0.045 — which looks precise and is not. A repeat at the same duration replays the
*same nonce sequence*, so it re-observes the same solutions and measures timing jitter
only. The real uncertainty is in solutions-per-solve, and only a longer run samples it:
the 300 s run saw 1.99 where the 120 s runs saw 2.01, nine times that σ. So `58.0 ± 0.4`
is 8,729 solves × 1.99 = 17,371 solutions → `1/√17371` = 0.76 %; the same model gives the
2026-07-25 row its ±2.3 (600 solutions → 4.1 %) and the 2026-08-13 row its ±0.3 (6 × 120 s
at 32.00 ms = 22,500 solves × 2.006 = 45,135 solutions → 0.47 %). Rows whose run left no
recorded solve count stay bare points rather than being given a fabricated bar.

**It is an uncertainty on the figure, never a window σ.** A per-window standard deviation
measures how far a one-minute reading bounces and does not shrink as the run lengthens, so
it would make a well-sampled build look as uncertain as a short one. The two differ by
6–8× on identical data: the 80-minute 2026-08-13 pool session carries a 60 s-window σ of
0.90 against a `1/√N` of 0.11, and the 31-minute 2026-08-16 one 1.09 against 0.19. Window
spreads are reported with the live-mining sessions below, as spreads.

**The bar sits on the sol/s column, and it measures BeamHash III rather than the solver.**
Across a live session's 60 s windows the sol/s spread runs 1.4–1.6 %, nearly all of it the
solutions-per-solve sampling above; the solves/s factor moves by less than the miner's own
display step, so a log cannot resolve the solver's timing spread at all. Same asymmetry
that makes ms/solve the quoted quantity.

</details>

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
| 2026-07-31 | Below-the-floor pair: r2's 16 B record in one `LD.128` + the terminal round joins the perfect table | 33.4 | 33.2 | **59.7** | −0.2 | −0.7 % | [Below-the-floor levers](performance-research.md#two-below-the-floor-levers-clear-noise-on-cuda-r2s-pair-record-in-one-ld128-and-the-terminal-round-joins-the-perfect-table-022-ms) *(delta pinned by ×9 replay: r2 −0.145, terminal −0.07; a bare point — this run's solve count was not recorded, so `1/√N` cannot be formed for it)* | — |
| 2026-08-13 | **Implicit-bits record** — the packed r2 record stops storing the key bits its bucket address encodes; the side plane loses its writer and reader | 33.2 | 32.0 | **62.7 ± 0.3** | −1.2 | −3.6 % | [address-redundant bits](performance-research.md#populations-are-pinned-at-225-and-the-occupancy-tail-prices-a-spill-arena) *(± is `1/√N` over the headline run's 45,135 solutions; the live-pool session's window spreads are quoted with the validation below)* | — |
| 2026-08-15 | **The w0-checkpoint pair record** — round 1 stores the child's post-mix work word 0, so round 2 derives only the linear lane (12 siphashes, no mixes); the record stays 16 B because the address-implied key bits pay for word 0 | 32.0 | 31.3 | **64.2 ± 0.3** | −0.7 | −2.2 % | [w0 checkpoint](performance-research.md#the-w0-checkpoint-pair-record-repriced-by-the-address-bits-076-ms-24-) *(−0.76 ms measured ABBA-interleaved; the pin prints to 0.1 ms. ± is `1/√N` over the pin's 46,290 solutions)* | [the 24 B form of the same record, +3.8 %](performance-research.md#measured-results-2026-08-12) |
| 2026-08-15 | **Recovery replays instead of storing reference rows** — round 4's row, then its record at 8 B instead of 16, then round 3's row and rows 1–3 with it; recovery re-runs two rounds over one bucket each and reads round 2's four leaves | 31.3 | 29.3 | **68.6 ± 0.3** | −2.0 | −6.4 % | [the replay](performance-research.md#the-back-reference-rows-are-gone-recovery-replays-instead-203-ms-and-688-mib) *(−2.029 ms measured ABBA-interleaved in three increments; the pin prints to 0.1 ms. ± is `1/√N` over the pin's 49,392 solutions. Also −688 MiB)* | — |
| 2026-08-16 | **Two barriers' worth of work that nothing needed** — the block-exit `__syncthreads()`, which at stock guards a pass that never runs; then **singleton-free staging**, which never stages the 12.8 % of a group alone in its chain slot, since a perfect table makes that element provably partnerless | 29.3 | 29.1 | **69.0 ± 0.3** | −0.2 | −0.7 % | [the barrier](performance-research.md#the-block-exit-barrier-is-removable-and-the-barrier-family-is-over-priced-10x), [singleton-free staging](performance-research.md#singleton-free-staging-the-prize-is-085-ms-and-the-prepass-that-finds-it-costs-076) *(−0.048 and −0.089 ms measured ABBA-interleaved; the pin resolves the pair, not the halves, because the barrier shipped without a re-pin. The second is a −0.852 ms skip against a +0.763 ms census — see the ledger, which is where the value of this row is. ± is `1/√N` over the pin's ~49,700 solutions)* | — |
| 2026-08-16 | **The census and the chain share one `tab` word** — the singleton filter's word-0 count goes in the high 16 bits and the chain head in the low 16, so the table is initialised once per group instead of twice and the barrier between the two clears goes with it | 29.1 | 29.1 | **69.2 ± 0.3** | −0.09 | −0.3 % | [the packed tab](performance-research.md#the-census-and-the-chain-share-one-tab-word-and-a-barrier-goes-with-it) *(ms/solve is 29.10 on both pins because −0.060 ms is under the print resolution; the finer `solves/s` line reads 34.33 → 34.44, i.e. −0.094 ms, and −0.060 was measured over eight interleaved arms a side with the ranges not overlapping. The two agree inside the cross-session band. With the block-exit barrier's −0.048 this is the second independent measurement of what a `__syncthreads()` costs here. ± is `1/√N` over the pin's 49,856 solutions)* | — |
| 2026-08-16 | **The reach floor's round 4 stops re-deriving what it can be handed** — round 3's 32 B octo record spends a `gi` nothing indexes and 14 key bits the bucket address already carries on the child's work word 0, which deletes every `apply_mix` in `rebuild_r4` and 8 of its 56 siphashes | — | — | — | −4.39 | **−7.9 %** | [the octo checkpoint](performance-research.md#the-w0-checkpoint-on-the-octo-record-the-mixes-go-and-the-record-does-not-grow) *(a REACH-RUNG row: no shipping kernel changed, so the headline pin is untouched and there is nothing to re-pin. The octo (16,1) rung goes 55.31 → 50.92 ms and the 1.90 GiB floor rung 61.14 → 56.97, both ABBA-interleaved with non-overlapping ranges, at an unchanged 32 B record)* | — |
| 2026-08-16 | **The same checkpoint on the reach rungs above it** — round 2's 24 B quad record stores the child's work word 0 in the u64 that held its key, since word 0's own low 24 bits *are* that key; the four leaves and `gi` move down into the pack the packed record already uses. `rebuild_r3` loses all 7 `apply_mix` calls and 4 of its 28 siphashes | — | — | — | −1.84 | **−5.1 %** | [the quad checkpoint](performance-research.md#the-w0-checkpoint-on-the-quad-record-the-record-holds-word-0-with-no-repacking) *(a REACH-RUNG row, like the one above: no shipping kernel changed, so the headline pin and the cap table are untouched. Quad (16,1) goes 36.02 → 34.18 ms and the 3.78 GiB quad floor 44.75 → 42.80, both ABBA-interleaved with non-overlapping ranges, at an unchanged 24 B record. Needs neither the implicit-bits pack nor the perfect table, because nothing is dropped to make room)* | — |
| 2026-08-16 | **OpenCL reaches the same 1.90 GiB floor, and gains the implicit-bits record** — the octo rung's round 4 wrote its reference row three rows past the end of a one-row allocation, and its references named a HALF-LOCAL slot, which means two different records in a split set — and a split set is every card the rung exists for. The implicit-bits record ported with it | — | — | — | −1.12 | **−3.3 %** | [the OpenCL floor](performance-research.md#opencl-reaches-the-same-190-gib-floor-two-bugs-in-the-octo-rung-both-about-addresses) *(an OPENCL row: no CUDA kernel changed, so the headline pin and the cap table are untouched. The ms figure is OpenCL's own (16,1) rung, 33.400 → 32.283 over twelve interleaved arms with non-overlapping ranges; the reach half takes that backend's floor from 4.04 GiB to CUDA's 1.90)* | — |
| 2026-08-17 | **The w0 checkpoint reaches OpenCL** — round 1 stores the child's post-mix work word 0 in the same 16 B pair record, so round 2 derives the linear lane alone: 12 siphashes and no `apply_mix`, against 14 and three. A port rather than a new lever, and one the implicit-bits pack unlocked the week before by freeing the address bits that pay for word 0 | — | — | — | −0.745 | **−2.3 %** | [the OpenCL checkpoint](performance-research.md#the-w0-checkpoint-reaches-opencl-0745-ms-229--the-same-lever-at-the-same-size) *(an OPENCL row: no CUDA kernel changed, so the headline pin and the cap table are untouched. 32.600 → 31.855 over twelve interleaved arms with non-overlapping ranges; CUDA measured −0.76 ms for the identical change. Confirmed again by a one-sitting ladder re-measurement, where the eleven byte-identical rungs size the session offset at +1.26 % and correct the two changed rungs to −2.19 / −2.46 %)* | — |
| 2026-08-17 | **And its quad and octo records, which are the bigger half** — round 2's 24 B quad record holds word 0 with no repacking, since word 0's own low 24 bits *are* the key it stored there; round 3's 32 B octo record buys word 0's 40 bits out of a `gi` nothing indexes and 14 key bits the bucket address carries. Rounds 3 and 4 then run the lane alone — 24 and 48 siphashes, and not one of the 22 `apply_mix` calls between them | — | — | — | −2.43 / −5.28 | **−6.3 / −8.6 %** | [the quad and octo ports](performance-research.md#the-w0-checkpoint-reaches-opencls-quad-and-octo-records) *(an OPENCL REACH-RUNG row: no CUDA kernel changed and no shipping OpenCL rung did either, so the headline pin and the cap table are untouched. Quad (16,1) goes 38.933 → 36.500 and octo (16,1) 61.733 → 56.450, six interleaved arms a side each with non-overlapping ranges. CUDA measured −1.84 and −4.39 ms for the identical changes. Confirmed again off the four ladder rungs that did not change, session offset −0.73 %, corrected −6.63 % and −8.72 %. The OpenCL floor goes 68.5 → 63.0 ms)* | — |
| 2026-08-18 | **Speculative entry's gate moves to its measured crossovers** — the co-blocks ride round 4 having idle issue capacity, which a cap removes, so spec turns off below 220 W on stock memory and below 150 W on a held 5001 memory rung, where the freed watts un-starve the core and the sign flips back. Match-first decouples onto the 130 W band it was measured in | 29.1 | 29.1 | **69.20 ± 0.3** | 0 at stock | **−1.75 % at 140/160 W, −0.99 % at 180, −0.22 % at 210; −0.71 % at 140 W on the rung** | [the spec gate](performance-research.md#speculative-entry-under-a-cap-both-crossovers-measured-and-the-gate-moves-to-them) *(a CAP-BAND row: the stock path is unchanged — spec stays on at 285 W — and the re-taken pin reproduces 29.10 / 69.20 with 0.0 % spread on both columns over six runs. The band deltas are position-balanced paired A/Bs, |t| ≥ 3.7 everywhere, with the 285 W arm as the applied-knob control (+1.03 %, spec's stock win). The head-to-head and low-band tables were re-measured the same session on the shipped defaults)* | — |

| | sol/s | ms/solve | |
|---|---|---|---|
| **OpenCL** | 63.0 | 31.9 | fallback / `--solver opencl` |
| **CUDA** | **69.20**[^drift] | **29.10**[^drift] | **shipping** — default when a CUDA device is present |
| **Target** | 53.0 | 35.8 | lolMiner, stock — user-measured |

The CUDA row is six 120 s runs at stock 285 W, headless, at **0.0 % spread on both
columns** — [the pin and its lineage](performance-research.md#the-named-reference-lgc-2600).

[^drift]: Absolute figures carry a ~2.5 % cross-session band ([how far they
    reproduce](#-the-absolute-figures-reproduce-to-03--within-a-session-and-5--between-sessions));
    every A/B on this page was interleaved, so deltas are unaffected. The OpenCL row
    is a single opportunistic 120 s run; its 1.012× gap is a same-session pair
    (2026-08-02), not this row against that one.

> **Quote ms/solve, and treat sol/s as derived.** `sol/s = solves/s × solutions/solve`,
> and only the first factor is a property of the solver — the second belongs to BeamHash
> III (**2.006**, pinned over 28,305 solves) and a short run estimates it noisily enough
> to move the headline by a full sol/s. Every speed *change* here is an ms/solve figure.

**Live mining reproduced the pin**, on the shipping kernel. A 31-minute HeroMiners
session medianed **28.74 ms/solve** — 1.2 % under the pin, at stock boost (2655 MHz
median) against the pin's locked 2600, which is the direction and roughly the size the
clock difference predicts. **69 shares, 0 stale, 0 rejected** over 38 jobs, on a path
sharing no code with the harness.

<details>
<summary>That session's windows, and why its σ is not an error bar</summary>

| window | samples | median | mean | σ | min | max |
|---|---|---|---|---|---|---|
| 60 s | 31 | **69.67** | 69.50 | **1.15** | 67.35 | 72.33 |
| 15 s | 124 | 69.60 | 69.50 | 2.33 | 63.93 | 75.27 |

Iteration rate held **34.8 it/s at σ 0.025** — the ms/solve figure, and the one quotable
column, since the sol/s above carry the run's own 2.002 solutions/solve. Stock 285 W,
clocks unlocked: 284–285 W / 2640–2670 MHz / 64–66 °C, memory pinned at 10251 by the
P-state.

The 1.15 is a spread of the reading, not uncertainty on the mean — that is the Poisson
±0.19 over ~129,600 solutions. The progress rows keep their own `1/√N`, because a bar
belongs to the run behind its row and this is a different measurement rather than more of
the same one.

The pool's own sol/s column is **not** a second measurement of the same thing: it
converges on share count, reading 51.0 after one minute and 72.6 by the last, at a ~12 %
standard error over 69 shares. It agrees; it cannot corroborate.

</details>

**Standing:** CUDA is **~31 % past the target**, OpenCL 1.07× short — read
[the caveats](performance-research.md#the-cuda-backend) before treating it as beaten.
From **1.8 sol/s** at the start, **31× faster**.

**VRAM for a full search:** 8.36 → **6.17 GiB** at the fastest geometry (268 → 197
B/element), and **1.90 GiB** at the floor on *both* backends — a 3 GB card, clearing
BeamHash III's stated 3 GB minimum. The octo rungs that reach it cost 2.00× their own top
rung on OpenCL against 1.98× on CUDA, now that
[both carry the w0 checkpoint](performance-research.md#the-w0-checkpoint-reaches-opencls-quad-and-octo-records).

<details>
<summary>The earlier sessions, and why a peak must never be quoted</summary>

The same shape on the two builds before it. 2026-07-31, twenty minutes on the 59.7 ms
build, medians agreeing to under 1 % and shares 47/0/0:

| window | samples | median | mean | σ | min | max |
|---|---|---|---|---|---|---|
| 60 s | 20 | **59.35** | 59.57 | **0.82** | 58.6 | 61.7 |
| 15 s | 83 | **59.60** | 59.51 | 1.93 | 54.2 | 63.9 |

And a 2.5-hour session on the 2026-07-25 build:

| window | samples | median | mean | σ | min | max |
|---|---|---|---|---|---|---|
| 60 s | 65 | **56.1** | 55.89 | **2.30** | 39.7 | 58.9 |
| 15 s | 304 | **56.1** | 55.94 | 3.27 | 24.9 | 61.5 |

The median was 56.1 on both windows, matching that build's benchmark median exactly, and
the 60 s σ of 2.30 matched its ±2.3.

Individual 15 s windows in it reached **61.5 sol/s**, which is tempting and wrong: for
σ = 3.27 the expected maximum of 304 draws is ≈ 67, so 61.5 is unremarkable rather than
evidence of a higher true rate. Narrowing the window inflates the peak (61.5 at 15 s vs
58.9 at 60 s) while leaving the median untouched — the signature of noise, not
throughput. The three samples below 45 sol/s are consecutive and coincide with two other
benchmarks being run against the same GPU.

That table was tallied by hand. It no longer has to be: `/summary`'s
`Session_Stats.Speed_60s` accumulates `N`, `Mean`, `Stddev`, `Min` and `Max` over the
whole run for both windows and for power, clocks and temperature — see
[usage.md](usage.md#dashboard-and-monitoring-api).

</details>

*sol/s is the number miners and pools report. BeamHash III yields 2.006 solutions per
solve, so sol/s ≈ 2006 / (ms per solve).*

---

## ⚠ The absolute figures reproduce to 0.3 % within a session and 5 % between sessions

*(Read before trusting any single-number claim here.)*

**Within a session the measurement is tight**: six 120 s runs at stock spread 0.3 %, with
SM clock, memory clock and board power identical to the last digit.

**Across sessions it is not.** The band this page carries is **~2.5 %** — the worst
same-binary cross-session delta observed since 2026-07-26, across nine-plus stock
sessions. One 2026-07-26 event at +5.5 % remains on record, unexplained and never
recurred. Deltas are unaffected: every A/B here was interleaved.

Under a locked clock the rig reproduces **to the digit across days** (0.0 % spread over
six runs on each of the five pins that measured an unchanged build, and 0.3 % on each of
the two that measured a changed one), so use `LGC=2600 LMC=10251 benchmarks/headline.sh`
to regression-test builds and the stock figure to describe what a user gets. The pin
stands at **29.10 ms** as of 2026-08-16, taken with the compute GPU headless — a condition of
the number, since a compositor on the card costs a measured 0.20 ms and ~6 W, and one
that follows the HDMI cable per login, so it is verified from the miner's own banner
each session.

*Reopening: any same-binary session median landing more than ~2.5 % from its peers
restores the 5 % band.*

<details>
<summary>The original observation, the controlled re-measurement, and four rejected explanations</summary>

The same binaries that measured **34.15–34.20 ms** early in one session measured
**35.49–35.72 ms** a few hours later, same machine, unchanged.

`benchmarks/headline.sh` was built to answer it: refuses to start if another process
holds VRAM, runs a discarded 240 s warmup, reports the residual temperature slope, and
records NVML telemetry per run including the `clocks_event_reasons` bitmask the original
measurement never captured. Six runs of 120 s, stock 285 W, card at equilibrium:

| | ms/solve | sol/s | SM | mem | W | °C | capped |
|---|---|---|---|---|---|---|---|
| run 1 | 33.80 | 59.1 | 2685 | 10251 | 284.1 | 67 | 99 % |
| run 2 | 33.80 | 59.2 | 2685 | 10251 | 284.2 | 67 | 100 % |
| run 3 | 33.70 | 59.3 | 2685 | 10251 | 284.1 | 67 | 99 % |
| run 4 | 33.70 | 59.4 | 2685 | 10251 | 284.1 | 68 | 100 % |
| run 5 | 33.70 | 59.4 | 2685 | 10251 | 284.1 | 67 | 100 % |
| run 6 | 33.70 | 59.4 | 2685 | 10251 | 284.1 | 67 | 99 % |
| **300 s** | **33.80** | **58.9** | 2685 | 10251 | 284.1 | 67 | 100 % |

So the 4 % gap was not run-to-run noise mistaken for a regime. The four sessions on
record at the time:

| session | ms/solve | vs fastest |
|---|---|---|
| 2026-07-26 early | 34.18 | +1.3 % |
| 2026-07-26 later | 35.60 | +5.5 % |
| 2026-07-28 sweep (285 W point) | ~33.7 | — |
| 2026-07-28 controlled | 33.75 | — |

Four explanations checked and rejected:

| candidate | evidence against |
|---|---|
| thermal throttling | card is *cooler and clocking higher* in the slow regime — 54 °C / 2760 MHz against 63 °C / 2685 MHz |
| within-run drift | the 15 s windows show no trend: first five average 58.5, last five 58.1 |
| memory clock | 10251 MHz in both regimes, and in the controlled run |
| host CPU contention | the **GPU kernel sum itself rose 34.49 → 35.41 ms** while host overhead moved +0.15 ms |

The slowdown is **on the GPU**. Per-round by replay in both regimes: entry 2.72→2.79,
r1 5.33→5.37, r2 10.38→10.58, r3 9.50→9.78, r4 5.49→5.77, terminal 1.07→1.12 — ~2–3 %
slower in step, pointing at a global clock/power state the logged clocks do not capture.
At stock the card is at `sw_power_cap` 99–100 % of the time, so the SM clock is whatever
285 W happens to buy; the surviving hypothesis is that the same 285 W buys a different
point on the V/f curve on different days.

The band was narrowed from 5 % to 2.5 % on 2026-07-31: the strongest cross-day evidence
is [the full power sweep repeated two days apart](#the-sweep-reproduces-across-sessions),
twelve caps all within 2.5 % (mean −1 %).

</details>

### The locked-clock reference is stable to the digit — run it 2026-07-30

Lock the SM and memory clocks (`LGC=2600 LMC=10251 benchmarks/headline.sh`) and the V/f
point is pinned rather than left to the card's discretion. Two runs bracketing 70 minutes
of continuous varied load — the full 100–285 W head-to-head sweep, both miners, ran
between them:

| | ms/solve | sol/s | SM | mem | W | °C | capped |
|---|---|---|---|---|---|---|---|
| **run A** (21:20) | **34.30** | 58.35 | 2610 | 10251 | 261.8 | 67 | **none** |
| **run B** (22:49) | **34.30** | 58.45 | 2610 | 10251 | 262.1 | 67 | **none** |
| Δ | **0.00 %** | +0.17 % | 0 | 0 | +0.11 % | | |

**The pin removes the suspected mechanism by construction.** At a locked 2610 MHz the
card draws 262 W against a 285 W limit with `clocks_event_reasons: none` — 23 W of
headroom unused, no longer buying clock with watts at all. The original anomaly appeared
*within* a session a few hours apart; runs A and B are 106 minutes apart across the
card's whole power range, a far more hostile interval, and ms/solve did not move.

The cross-day repeat this owed was paid on 2026-08-01: 33.50 ms, six runs, 0.0 % spread
— see [the sweep reproduction section](#the-sweep-reproduces-across-sessions).

**Re-pinned at 33.30 on 2026-08-04, with the compute GPU headless — and the move is
attributed: it is the rig, not the build.** Six runs, 0.0 % spread, 2610/10251, no
throttle flags, 271.1 W. The monitor moved to the motherboard iGPU that day, so the
compositor no longer holds a graphics context on the card; rebuilding the 08-01 pin's
own build and running it headless the same day reads **33.30 to the
digit at 270.4 W** — so the compositor was costing **0.20 ms and ~6 W**, and the
intervening commits are time-neutral. The geometry is unchanged across the move —
still `(16,1)`, the miner reporting `reserving 64 MB (headless)`. Full lineage:
[the pin](performance-research.md#the-named-reference-lgc-2600).

**Re-pinned at 32.00 on 2026-08-13** (the implicit-bits record):
six runs, 0.0 % spread, 2610 / 10251, 281.4 W, 67 °C, headless. The draw now sits
within 4 W of the board limit and `sw_power_cap` is intermittently active — a
consequence of the busier binary, recorded as a standing condition of stock pins
from here on.

### The memory junction temperature is not observable on this card — dead end

GDDR6X raises its refresh rate as it heats, lowering *effective* bandwidth while the
reported memory clock sits at 10251 MHz — exactly the signature above. **NVML cannot see
it on this card**: field 82 (`MEMORY_TEMP`) returns `Not Supported` idle and under load,
as does the whole `T.Limit` family.

So the VRAM-temperature column lolMiner has is **not available on this class of card** and
should stop being carried as work, and the GDDR6X-refresh hypothesis is not disprovable
with the instruments here — the locked-clock reference carries the question instead.

The probe did find one usable thing: field 83
(`NVML_FI_DEV_TOTAL_ENERGY_CONSUMPTION`) *is* supported, a monotonic millijoule counter.
Wired 2026-07-31 — `--benchmark` prints J total, mean W and J/solution from it, `--tune`
prefers it over the sampler, `/summary` exposes it as `Energy_J`, and `benchmarks/lib.sh`
gained `bench_energy_mj`. Existing tables keep their sampled basis until re-measured
whole.

<details>
<summary>The positive control, because "Not Supported" alone proves nothing</summary>

A bare `Not Supported` is indistinguishable from a wrong struct layout, a failed `dlsym`,
or a handle that is not really a device. The same call, in the same array, asked for
fields this card certainly does answer:

| field | | result |
|---|---|---|
| 82 | `MEMORY_TEMP` | **Not Supported** |
| 186 | `POWER_INSTANT` (control) | Success — 8 852 mW idle, 285 779 mW under load |
| 83 | `TOTAL_ENERGY` (control) | Success — monotonic mJ counter |
| 193–196 | the whole `T.Limit` family | **Not Supported** |

The controls resolve to three digits — 285.8 W under load against a 285 W board limit —
so the field is genuinely absent rather than mis-called. `nvidia-smi -q -d TEMPERATURE`
agrees: `Memory Current Temp: N/A`, `Memory Max Operating Temp: N/A`, `GPU Current
T.Limit Temp: N/A`. A GeForce restriction, not a driver or API-surface problem.

</details>

**How to quote a number from this page:** use the controlled figure with its conditions
attached — **29.10 ms / 69.20 sol/s** at stock 285 W, headless, locked LGC=2600
LMC=10251, six runs at 0.0 % spread (most recently re-pinned 2026-08-16; see
the lineage table in [benchmarking.md](benchmarking.md)) — and carry the ~2.5 %
cross-session band (narrowed 2026-07-31; see above). Do not re-derive a headline
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
power MXBM wins both.** Reproduce with `mxbm --report`,
`benchmarks/stage_power.sh` and `benchmarks/power_sweep.sh`.

### The card is at its power limit, in every kernel

`clocks_throttle_reasons.sw_power_cap` reads **Active** continuously under MXBM,
while `hw_slowdown` and `sw_thermal_slowdown` never fire and the GPU sits at
62–65 °C — far from its throttle point. The 285 W is a **board limit being hit**,
not heat, and the 5 °C gap to lolMiner is the *consequence* of drawing
45 W more, not an independent symptom.

That is not a property of one hot kernel. `benchmarks/stage_power.sh` replays a
single stage N times inside the solve that produced its input — so its input
bytes, bucket occupancies and output addresses are the real ones — and solves for
that stage's own time and energy from the shift it causes (since 2026-07-31 the
joules come from the card's energy counter, bracketed around the timed loop
inside the binary; the sampled-watts identity remains the fallback):

    t_s = (T_N - T_1) / (N-1)        E_s = (J_N - J_1) / (N-1)        P_s = E_s / t_s

Re-measured 2026-08-16 on the current kernels, 8 reps, 45 s per stage:

| stage | ms | % of solve | power | J/solve | % of energy |
|---|---|---|---|---|---|
| `entry_scatter` | 2.66 | 9.1 | 282.9 W | 0.75 | 9.1 |
| round 1 | 5.02 | 17.2 | 285.1 W | 1.43 | 17.3 |
| round 2 | 8.21 | 28.1 | 282.8 W | 2.32 | 28.0 |
| round 3 | 8.42 | 28.8 | **281.0 W** | 2.37 | 28.6 |
| round 4 | 4.20 | 14.4 | 283.5 W | 1.19 | 14.4 |
| terminal | 0.78 | 2.7 | 285.3 W | 0.22 | 2.7 |
| **sum** | **29.30** | **100.2 %** | 282.8 W | **8.29** | |

**Scope: this table predates the day's two kernel changes** (the block-exit barrier and
singleton-free staging, −0.20 ms together on the pin, both landing in rounds 1 and 2).
It is the 29.30 ms build's breakdown, not the 29.10 one's; the per-stage split has not
been re-taken and `benchmarks/stage_power.sh` is what re-takes it.

Solve is 29.23 ms in the harness, so the stages account for 100.2 % of it, and
their summed energy lands within 0.2 % of the counter's own whole-solve figure
(8.30 J) — both closures bound anything unattributed (memsets, launch gaps,
readback, CPU verify) at essentially zero. 8.30 J per solve ÷ 2.01 verified
solutions = **4.13 J per solution** at stock, from 4.44 on the previous profile:
energy per solve fell 8.92 → 8.30 J at an unchanged multiplier.

Each of the last three record changes is visible here as its own mechanism and
nothing else's. **Deleting the reference rows** is round 4 **−1.21 ms** — the only
round that lost both a row and half its output record — with round 3 −0.54, round 2
−0.29, round 1 −0.14, the terminal round −0.20 and entry unmoved, which is the four
rows plus the 8 B record exactly where they were written and read.
The **w0-checkpoint record** was round 2 **−0.80 ms** against round
1 **+0.08** for the pack, with rounds 3, 4 and terminal identical to the digit —
the round that stopped deriving, and the round that pays to store. Before it, the
**implicit-bits record** was round 2 −0.94 and round 3 −0.52, the round that
writes the narrowed record and the round that reads it, with no other stage moving
past 0.06 ms.

Every stage draws the cap. There is no power-hog kernel to fix: the workload
saturates the board limit from `entry_scatter` through the terminal round, and
the driver pulls the core clock down (2595–2745 MHz) to hold 285 W. Only round 3
sits measurably below it — and it runs the highest clock of any stage — because
round 3 is the DRAM-bound round: moving bytes costs this board less than working
the SM does, so the freed power comes back as core clock.

Two consequences:

1. **Energy per solve tracks time per solve.** At a fixed cap, sol/s/W is just
   sol/s ÷ 285. Every speed optimization on record is an efficiency
   optimization of the same size, and no amount of watt-shaving *inside* the
   workload is available independently of it — there is no slack kernel to
   quieten. Matching lolMiner's 0.223 sol/s/W while still drawing
   285 W would need **63.5 sol/s**, against 62.4 measured — 1.8 % away, and only
   reachable through ms/solve. Moving the cap itself is the other lever, and
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
35.1 to 34.1 ms. Every row would shift down by roughly that much; the bend, which is what
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

> **⚠ The Δ columns are superseded.** They compare capped MXBM against **uncapped**
> lolMiner, assuming lolMiner had no choice about its operating point. It does — `--pl` is
> in its help — and once both are swept the comparison changes shape. The MXBM rows stand
> as MXBM's own curve; for the head-to-head read
> [Both miners under the same cap](#both-miners-under-the-same-cap).

**Efficiency has an interior optimum, and it *falls again* below it** — that is the
informative part, and it is a property of the board rather than of a build. Below the
peak the core clock has dropped far enough that the parts of the board which do not scale
with it — memory, uncore, leakage — are paid for out of less work. On the current kernels
the peak is **210 W at 0.3094 sol/s/W**, with 200 W a full 1.0 % below it and 220 W
2.5 % below; at the driver's 100 W minimum the collapse is unmistakable, 24.5 sol/s and
0.2462 sol/s/W. The peak has moved right to left across builds — 240 W, then 220, now
210 with the speculative-entry gate — because a lever that pays more under a cap than at stock
lifts the left half of the curve more than the right; it is read from
[the live table](#both-miners-under-the-same-cap), never from the historical one above.

**Marginal return collapses well before stock.** Extra sol/s per extra watt, from the
live column:

| step | 180→190 | 190→200 | 200→210 | 210→220 | 220→240 | 240→255 | 255→285 |
|---|---|---|---|---|---|---|---|
| sol/s per W | 0.380 | 0.340 | 0.370 | 0.140 | 0.080 | 0.063 | 0.037 |

(The 190→200 and 200→210 split is inside this rig's resolution; the 20 W span they
cover, **0.355 per W**, is the figure to quote.) The last 45 W
(240 → 285) buys 2.05 sol/s; the 20 W from 180 to 200 buys 7.20. The bend is sharp and
it is at **210–220 W** — immediately past the peak, with 220 W already 2.5 % below it.
The right cap is an economic choice: **210 W for a rig that pays for electricity,
285 W only where power is free.**

### Both miners under the same cap

*(Measured **2026-08-18, one session end to end**. The 100–120, 200–255 and 285 W rows
are `compare_power.sh` interleaved pairs — 60 s runs, two repeats, the miners
alternating order at every point; the 140–210 W MXBM rows are the same day's 120 s
`power_sweep.sh` on the shipped defaults, each inside 0.5 % of that session's paired
A/B arms. Caps are set externally with `nvidia-smi -pl` for both miners; power is
sampled from NVML for both, never from a miner's own report. One session because a
head-to-head cap table assembled from two inherits the cap-dependent cross-session
term, which reaches ~5 % at the low caps
([method](performance-research.md#how-small-a-difference-the-rig-can-resolve)).)*

*(The MXBM rows carry the shipped speculative-entry gate — off below 220 W on stock
memory, the measured crossover
([ledger](performance-research.md#speculative-entry-under-a-cap-both-crossovers-measured-and-the-gate-moves-to-them)).
lolMiner 1.98a is the unchanged reference binary.)*

| cap | MXBM sol/s | MXBM W | MXBM sol/s/W | lolMiner sol/s | lolMiner W | lolMiner sol/s/W |
|---|---|---|---|---|---|---|
| 100 W | **24.50** | 99.5 | **0.2462** | 23.75 | 99.5 | 0.2387 |
| 110 W | 28.85 | 109.7 | 0.2630 | **30.30** | 109.1 | **0.2777** |
| 120 W | 32.75 | 119.8 | 0.2734 | **33.95** | 119.2 | **0.2848** |
| 140 W | 38.80 | 139.8 | 0.2775 | **41.50** | 139.7 | **0.2971** |
| 160 W | 46.20 | 160.0 | 0.2887 | **49.95** | 160.1 | **0.3120** |
| 175 W | 52.70 | 175.1 | 0.3010 | **52.90** | 174.6 | **0.3030** |
| 180 W | **54.00** | 180.1 | **0.2999** | 53.25 | 179.7 | 0.2963 |
| 190 W | **57.80** | 190.0 | **0.3043** | 54.25 | 189.6 | 0.2861 |
| 200 W | 61.20 | 199.9 | 0.3062 | 54.40 | 199.6 | 0.2725 |
| 210 W | **64.90** | 209.8 | **0.3094** | 54.20 | 209.5 | 0.2587 |
| 220 W | **66.30** | 219.6 | **0.3018** | 53.35 | 219.6 | 0.2430 |
| 240 W | **67.90** | 239.6 | **0.2834** | 53.80 | 234.7 | 0.2292 |
| 255 W | **68.85** | 254.5 | **0.2705** | 53.70 | 235.1 | 0.2284 |
| 285 W | **69.95** | 284.2 | **0.2461** | 53.75 | 235.1 | 0.2286 |

![Speed, efficiency and power drawn, both miners at the same caps](tools/power-curve.svg)

**The result is three-part.** Crossings are interpolated from the table:

- **lolMiner's stock-memory window is ~104 W to ~177 W, and both of its edges have
  moved in.** Its best cap is **160 W at +7.5 %** (49.95 against 46.20); 140 W is
  +6.5 %, and the lead is down to **0.4 % by 175 W**. Below ~104 W it inverts: at the
  100 W floor MXBM is **ahead by 3.2 %**, outside the repeat spread. With each miner
  at its best configuration the window shrinks to two strips — the ≤115 W floor at
  2–3 % and 155–177 W at under 1 %, with MXBM ahead 8.5 % at 120 W and 3.5 % at 140
  between them ([the low band](#the-low-band-on-the-current-kernel)).
- **From ~177 W to the 285 W stock limit, MXBM wins on both** — +1.4 % on speed at
  180 W, +6.5 % at 190, +19.7 % at 210, widening to **+30.1 %** at 285.
- **Both crossings land together at ~177 W** — `docs/tools/plot_power.py` computes
  **176.8 W** from a fine grid over this table, for speed and efficiency alike. The
  175/180 W rows differ by less than the repeat spread on either side, so read the
  crossing as a band around 177.

Three facts behind it:

**lolMiner barely responds to the cap at all — and it never reaches the board limit.**
Its ceiling is **54.40 sol/s at a 200 W cap**; from there down to 180 W it gives up
2.1 %, and above ~235 W the cap stops doing anything — its 240, 255 and 285 W rows all
draw 234.7–235.1 W. MXBM at the 285 W board limit does **69.95, a 28.6 % higher
ceiling** that lolMiner cannot reach at any setting.

**Both miners have an interior efficiency optimum, and today they are level.** MXBM's
stock-memory peak is a flat **0.309–0.310 sol/s/W across 200–210 W**; its rung point
(160 W + 5001 MHz) reads 3.219 J/sol, and lolMiner's best is **0.3120 at 160 W**
(3.205 J/sol on its own sol/s definition) — the three sit within a percent, which is a
tie under the own-definition caveat and the repeat spread. The difference is what the
energy buys: at MXBM's peak the same joules-per-solution do **64.9 sol/s against
49.95 — 30 % more work**.

**Nothing is hiding below the sweep's left edge.** Both curves fall away monotonically
below their efficiency peaks, and lolMiner's advantage vanishes at both ends of its own
window — gone by 175 W above and inverted by ~104 W below.

### The sweep reproduces across sessions

The whole sweep was re-run on 2026-07-30, two days after the first, on the same binaries.

<details>
<summary>Point-by-point, both miners</summary>

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

**MXBM's deltas are all the same sign** — twelve of twelve slightly slower on the second
day, mean −1.0 % — which is a session offset, not noise; noise would change sign.
**lolMiner's are mixed** (+0.8 % to −2.1 %, mean ≈ −0.2 %), i.e. noise-like. The second
session was marginally unfavourable to MXBM, and every conclusion above is drawn from it,
which makes them conservative.

</details>

Every point of both miners reproduces within 2.5 %, all but three within ~1 %, and the
board power drawn at each cap is identical to a tenth of a watt. This pair is the evidence
that narrowed the headline band from ~5 % to ~2.5 %.

**No conclusion changed.** Both crossings moved by less than one grid spacing (~212 →
~210 W, ~257 → ~256 W), both efficiency peaks moved one row within a flat region, and
lolMiner's saturation ceiling reproduced at 236.6–237.5 W against 235.7–235.8 W.

**The LGC-2600 cross-day repeat, 2026-08-01.** The locked-clock reference ran on a
different day for the first time: six 120 s runs, ms/solve spread **0.0 %** (33.50 to the
digit, all six), SM/mem pinned at 2610/10251, 68 °C, power spread 0.6 %. It moved from
34.30 to **33.50 (−2.3 %)**, and the lineage accounts for it exactly: two kernel changes
shipped 2026-07-31 totalling 0.80 ms, and the draw rose 262 → 277 W at identical clocks —
the signature of a busier binary, not a different rig-day. **Under the pin the rig
reproduces to the digit across days; the number moves only when something about the
machine does** — the build, or the card's other tenants. The 2026-08-04 re-take is
the second kind: −0.6 % on the day the monitor left the compute GPU, attributed by
rebuilding the 08-01 commit and re-running it headless (33.30 to the digit — the
compositor was the whole 0.20 ms and ~6 W). The pin stands at **32.00** (2026-08-13) and carries
"headless" as a stated condition; repeat it after any kernel-shipping day or any
change to what else the card is doing.

### Why we lose the low end: watts buy us less clock

The mechanism is visible in the clocks — **down to ~150 W, below which it inverts.**
Mean SM clock per cap:

| cap | MXBM SM clock | lolMiner SM clock | gap |
|---|---|---|---|
| 100 W | 802 MHz | 470 MHz | **+332** |
| 110 W | 934 MHz | 605 MHz | **+329** |
| 120 W | 1066 MHz | 737 MHz | **+329** |
| 140 W | 1288 MHz | 992 MHz | **+296** |
| 160 W | 1555 MHz | 1660 MHz | −105 |
| 175 W | 1813 MHz | 2428 MHz | **−615** |
| 180 W | 1874 MHz | 2477 MHz | −603 |
| 190 W | 2055 MHz | 2545 MHz | −490 |
| 200 W | 2239 MHz | 2601 MHz | −362 |
| 210 W | 2396 MHz | 2650 MHz | −254 |
| 220 W | 2428 MHz | 2693 MHz | −265 |
| 240 W | 2520 MHz | 2745 MHz | −225 |
| 255 W | 2578 MHz | 2749 MHz | −171 |
| 285 W | 2664 MHz | 2751 MHz | −87 |

*(Both columns are the 2026-08-18 session that produced the head-to-head table, sample
means over each point's steady state, MXBM on the shipped defaults. The same session
also checked the memory clocks: both miners hold the full 10251 MHz throughout the
band, so the split below is core-clock only.)*

**Above 160 W the mechanism is traffic.** MXBM's kernels cost more power per clock, so a
tightening cap takes clock away from us faster than from them — 87 MHz behind at stock,
615 MHz behind at 175 W. MXBM moves
[10.73 GB/solve](benchmarks.md#where-the-time-and-energy-go) at geometry (16,1), while
lolMiner selects a **4G** variant that fits the search in 4 GB and moves far less.
Narrowing round 2's record so the solve moves 16 % fewer bytes buys **60 MHz at 285 W and
210 MHz at 180 W** — the price of a byte rises 5× as the cap tightens, which is the
asymmetry the gap column shows. One lever worth 210 MHz against a 570 MHz deficit puts the
whole gap in range of a traffic story: it needs about 2.7× our lever, and lolMiner's 4 GB
variant against our 6.17 GiB is plausibly that. See
[bytes are not free in watts](performance-research.md#but-bytes-are-not-free-in-watts-and-under-a-cap-watts-are-clock-60-mhz).

**Traffic is not the whole of it.** Two levers that deleted *instructions* rather than
bytes — the w0-checkpoint record and the replayed reference rows — each paid more under a
cap than at stock, and between them took the 180 W deficit from −10.4 % to −1.8 % without
narrowing a single record. The clock gap they were measured against widened rather than
closed. So traffic bounds the gap but does not exhaust it, and the issue economy is the
other half.

**This inverts one of our own conclusions.** [Bytes are nearly
free](performance-research.md#bytes-are-nearly-free-per-element-work-is-not) measured that narrowing records buys
almost no *speed*, because the pipeline is latency- and occupancy-bound rather than
bandwidth-bound, and that is why memory efficiency was filed under reach. Under a power
cap bytes are not free at all: they are watts, watts are clock, and clock is speed. What
that promotes is **narrowing records**, which cuts bytes moved; in-place layer reuse cuts
the footprint while moving the same bytes, so it stays a reach lever.

#### ⚠ Below ~150 W the clock gap INVERTS, and the clock explanation stops applying

Everything above is a story about clock: lolMiner clocks higher under a cap, and that is
why it wins. **Below ~150 W that is not what happens.** At 110–140 W, **MXBM holds a
~300 MHz HIGHER SM clock than lolMiner and still loses on sol/s** — by 7.0 % at 140 W
and 3.7 % at 120, where the same inversion cost 20–30 % two kernels ago. The sign flips
between 140 and 160 W — and at the 100 W floor the inversion has now closed entirely:
the same clock advantage finally buys the lead.

| cap | clock gap | sol/s gap |
|---|---|---|
| 100 W | MXBM **+332 MHz** | **MXBM +3.2 %** |
| 120 W | MXBM **+329 MHz** | lolMiner **+3.7 %** |
| 140 W | MXBM **+296 MHz** | lolMiner **+7.0 %** |
| 160 W | −105 MHz | lolMiner +8.1 % |
| 180 W | MXBM −603 MHz | **MXBM +1.4 %** |

So in the bottom third of the range the deficit is **not a clock deficit** — it is work
done per clock. The "bytes → watts → clock" chain cannot explain it: that predicts the
miner moving fewer bytes clocks *higher* under a cap, and below ~150 W lolMiner clocks
lower. **The per-cycle deficit is what the shipped levers have been closing**: at 100 W
the same clock advantage read −21 % two kernels ago, −2 % one kernel ago, and is **+3.2 %
now** — every lever that got it there deleted instructions rather than bytes, the
speculative-entry gate most recently
([ledger](performance-research.md#speculative-entry-under-a-cap-both-crossovers-measured-and-the-gate-moves-to-them)).

**Mechanism, settled 2026-07-31.** lolMiner is not duty-cycling — 10 Hz clock sampling
gives a broad but **unimodal** distribution, so it genuinely executes near 470 MHz
at 100 W — where it outsolved MXBM-at-733-MHz by ~22 % when this was measured, and is
outsolved by MXBM-at-802-MHz by 3.2 % now. The deficit was **~2.3× useful work per core
cycle** and is **1.4–1.65×** on the current kernels (sol/s per MHz, 100–140 W). Below ~190 W everything is issue-bound (103.7 ms at 100 W is 2.94× stock
at 3.38× less clock — even the DRAM-bound rounds stop saturating DRAM, because the LSU
rate scales with core clock), so the currency is instructions issued. Profiling lolMiner
under `ncu` named the reason: a state-storing streaming design, 64 B/element,
17.7 GB/solve, **no derive or rebuild arithmetic anywhere**, so at low caps it has almost
no instructions to issue. See [lolMiner under
ncu](performance-research.md#lolminer-measured-under-ncu-the-state-storing-design-confirmed--and-its-54-sols-ceiling-is-a-dram-roofline).

Our own knobs recover little. Geometry (17,0) deletes the rescan, crosses over at ~190 W
and buys 1 % in the 140–180 W band and 2.6 % at the floor — but it failed same-day
reproduction and auto-selection was built, verified and
[disarmed](performance-research.md#the-eco-sweep-170-crosses-over-below-190-w-r2_full-never-does).
The byte-heavy `MXBM_R2_FULL` **loses at every cap**: re-derivation is the right trade at
all power levels. Tables: [the eco
sweep](performance-research.md#the-eco-sweep-170-crosses-over-below-190-w-r2_full-never-does).

An intermediate reading — that MXBM's own bookkeeping (sub-mask rescans, chain walks) was
the cost — was tested and is **wrong**: [the reorganization
probes](performance-research.md#the-solver-reorganization-probes-the-cycle-deficit-is-not-bookkeeping)
measured bookkeeping at ~0.6 ms of round 1's 4.9.

#### ⚠ Closed 2026-07-29: the low end is not reachable by traffic, and no other mechanism has been found

**Traffic cannot close the low end.** Closing the 570 MHz clock deficit at the measured
exchange rate needs **≥ 6.2 GB of a 13.0 GB solve — 48 % of all traffic**. Every
narrowing the [record audit](performance-research.md#the-record-redundancy-audit) leaves
available totals 1.07 GB: **≤ 98 MHz, 17 % of the deficit.**

Two other candidates died in the same week:

- **Instruction issue.** Moving siphash's four 64-bit adds off the saturated ALU pipe is
  arithmetically impossible on this ISA: sm_89 has no `IMAD` form that writes a carry-out,
  so `ptxas` lowers `mad.lo.cc.u32` to `IMAD.IADD` **plus** a carry-synthesising `LEA`,
  and the ALU pipe gets *busier* (`entry_scatter` 855 → 876). The surviving rotate-only
  form is ≈0.18 ms, under the 1 %-of-a-solve floor.
- **Undervolting.** Untestable at the time, because the benchmark card drove the display.
  It became testable when the monitor moved to the motherboard iGPU and was then
  [measured and closed](performance-research.md#undervolting-buys-nothing-under-a-power-cap--the-cap-outranks-both-knobs):
  null under every cap, because the power governor outranks the clock lock.

**The practical conclusion, and what has since moved.** MXBM's advantage is a band:
lolMiner below the crossing, MXBM between the crossings, and MXBM alone above the upper
one at a ceiling lolMiner cannot reach at any setting. That shape still holds; the figures
in it do not. On the current kernels the band is **~177 W to the 285 W stock limit** and
the ceiling **69.95 sol/s against ~54.4** — see
[both miners under the same cap](#both-miners-under-the-same-cap), which is the live
table. The low-end verdict has moved furthest: "not supportable below 210 W" was true of
the 07-29 build, and on stock memory the low band is now **a 3.2 % lead at the 100 W
floor** with a worst case of **−7.5 % at 160 W**, against the −19 to −32 % of that
build — and at best configuration the residual is the ≤115 W floor at 2–3 % and the
155–177 W strip at under 1 % ([the low band](#the-low-band-on-the-current-kernel)).

*Reopening: a no-replay 180 W A/B putting the whole-solve rate materially above
92 MHz/GB; a narrowing worth more than 1.07 GB; or a machine where the compute GPU does
not drive the display. The first two would move the arithmetic, not the conclusion —
48 % of all traffic is a long way from 1.07 GB.*

<details>
<summary>The withdrawn arithmetic, and how the closure survived three later challenges</summary>

The original "2.7× our lever" estimate divided a whole-solve clock deficit by a lever
measured on a `MXBM_ROUND_REPS=2:24` replay timeline, priced against a denominator nobody
had measured. Both terms were wrong, both optimistically.

**The denominator.** Ablating round 2's payload removes
**1.47 GB**, not the 2.09 GB the derivation gave — that figure scaled round 2's
*compulsory* write and charged the ablation with 268 MB of back-refs it never touches. Two
mechanism questions closed with it: a 16 B store costs a full **32 B sector** (805 MB
predicted at 16 B, 1342 at 32 B, 1330 measured), and there is **no read-for-ownership**
(that would have added ~1074 MB of read; read moved +7.8 MB).

**The numerator is still a replay clock**, measured on a timeline that is ~91 % round 2
while round 2 is 30 % of a real solve. At 285 W the narrowed ×24 replay clocks 2670
against a real-solve baseline of 2685, so the whole-solve gain there is ≤ 0, not +60.

| | replay timeline | whole solve |
|---|---|---|
| exchange rate | 210 / 1.47 = **143 MHz/GB** | **≤ 92 MHz/GB** |

Three later results tested the closure and it held each time. Instruction issue as the
low-cap currency is real but small — geometry (17,0) buys up to 2.6 % against a 20–30 %
deficit. A full reorganization of the match (counting-sort runs, keys-only 8 B staging,
register pairing; chains, rescans, spill and most barriers deleted) produced the exact
pair multiset of the shipping round-1 kernel and ran **1.75× slower**, and its ablation
carve put all bookkeeping at ~0.6 ms of round 1's 4.9 — 12 % of the round, not the 55 %
that reading needed. Finally `ncu` on lolMiner's own binary named the mechanism from the
other side: a state-storing streaming pipeline, 64 B/element, 17.7 GB/solve against our
13.0, no re-derivation anywhere, sustaining ~494 GB/s — which makes its ~54 sol/s ceiling
a DRAM roofline and the 2.3× simply the absence of our arithmetic. The band structure of
this whole comparison is one design trade held from opposite ends. See [lolMiner under
ncu](performance-research.md#lolminer-measured-under-ncu-the-state-storing-design-confirmed--and-its-54-sols-ceiling-is-a-dram-roofline).

</details>

### Above stock: the curve continues to ~321 W, and the memory rung is unreachable

*(Re-measured **2026-08-20**, shipping build, headless, 90 s arms, the whole sweep run
ascending and then descending so thermal drift cancels. Arms agree to ≤0.14 % at every
rung and the 285 W bracket reproduced 70.5 / 70.6 either side. Every power sweep
elsewhere in this document stops at the 285 W stock limit; the board's own maximum is
366 W.)*

| cap | ms/solve | sol/s | SM clock | **drawn** | sol/s/W | vs stock |
|---|---|---|---|---|---|---|
| 285 W *(stock)* | 28.7 | 70.55 | 2660 MHz | 284.4 W | 0.2481 | — |
| 300 W | 28.5 | 70.85 | 2700 | 299.3 | 0.2367 | −4.6 % |
| **315 W** | **28.3** | **71.35** | **2720** | 314.3 | 0.2270 | −8.5 % |
| 330 W | 28.25 | 71.55 | 2740 | **319.9** | 0.2237 | −9.8 % |
| 366 W | 28.25 | 71.55 | 2740 | **321.9** | 0.2223 | −10.4 % |

**MXBM saturates at ~321 W.** The 330 W and 366 W caps draw the same ~320 W, hold the
same 2740 MHz and return the same 71.55 sol/s, so above ~315 W the cap stops binding —
the mirror of the behaviour this document records for
[lolMiner at ~236 W](#both-miners-under-the-same-cap). Thermals are not the limit at any
rung: 68 °C at 366 W against 65 °C at stock.

**`--pl 315` is the operating point above stock**, worth **+1.1 % speed for +10.5 %
power**. The marginal return over 285 → 315 W is **0.027 sol/s per W**, which is *below*
the 0.037 this document records for the 255 → 285 step — the curve rolls off above stock
rather than continuing straight, so extrapolating the sub-stock sweep overstates what is
there. It is a goal-2 lever bought at a goal-3 cost, and 210 W remains the recommendation
for a rig that pays for electricity.

**This sweep is a later session than [the head-to-head table](#both-miners-under-the-same-cap)**
and carries the cross-session term the method note prices: its 285 W point reads 70.55
against that table's 69.95, **+0.9 %**, on the same binary. Read the rungs against each
other, not against the caps below 285 W in the tables above.

**The 10501 MHz memory rung is not reachable, and this is a mechanism rather than a
null.** The card advertises it (`nvidia-smi -q -d SUPPORTED_CLOCKS` lists
10501/10251/5001/810/405) and every figure ever published here was taken at 10251, so it
looked like free bandwidth for a pipeline at 76–80 % of DRAM peak in rounds 3 and 4.
`nvidia-smi -lmc 10501,10501` **applies at idle and is dropped the moment the workload
runs**, and the rung is still not taken at a 366 W cap where the cap does not bind — all
ten arms of this sweep reported 10251 under load. The sweep records the achieved memory
clock beside every point, precisely so the null cannot be mistaken for a measurement of
the rung. What is measured is that the lever cannot be applied, not that it does not pay.

### Below stock the OTHER rung pays: +7 to +18 % under caps below ~165 W

Under a core power cap the memory clock does not scale, so the interface burns a fixed
slice of a small budget on bandwidth nothing is using; at the 5001 MHz rung those watts
come back as core clock. On the current kernel the rung is worth **+7 % at 160 W rising
to +18 % at 100 W** with its crossover at ~167 W ([the low
band](#the-low-band-on-the-current-kernel) has the full table); above the crossover it
is a wall — the solver's own DRAM roofline. lolMiner rides the same rung to its own
roofline, which arrives at ~110 W because its design moves ~17.7 GB/solve against our
10.69. Guidance for capped rigs: below ~165 W, pair the cap with the rung —
`sudo mxbm ... --pl <cap> --mclk 5001` (both restored on exit), or
`sudo nvidia-smi -pl <cap> -lmc 5001,5001` once at boot on rigs that mine unprivileged.
At or above ~180 W, never.

<details>
<summary>The 2026-07-31 sweep that established it — 18 ABBA arms, both miners</summary>

The down-rung is the up-rung's mirror, and it is honored: `-lmc 5001,5001` held in
every arm of an 18-arm ABBA-bracketed sweep (2026-07-31, miner loop, energy-counter
draw; full table and mechanism in
[the research doc](performance-research.md#the-5001-memory-rung-85-to-144--below-its-173-w-crossover--the-largest-low-band-lever-ever-measured-here)).
Under a core power cap the memory clock does not scale, so the interface burns a fixed
slice of a small budget for bandwidth nothing is using; at 5001 MHz those watts come
back as core clock. Measured: **−14.3 % at 100 W, −14.4 % at 120, −10.8 % at 140,
−8.5 % at 160; crossover ~173 W; above it the rung is a wall** (46.5 ms flat from
200 W up — the solver's own DRAM roofline at 280 GB/s sustained, 87 % of the rung's
peak, with draw saturating at ~230 W under a 285 W cap).

**The efficiency-optimal operating point moved onto the rung: 160 W + 5001 MHz gives
3.70 J/solution** (2026-08-14, 120 s pairs in mirrored order — the 2026-07-31 sweep read
3.79 there). The gain at this cap is the implicit-bits record shipped 2026-08-13, not the
low-power pair: that gate fires only below 130 W. Guidance for capped rigs: below ~170 W, always
pair the cap with the rung — `sudo mxbm ... --pl <cap> --mclk 5001` (both restored on
exit), or `sudo nvidia-smi -pl <cap> -lmc 5001,5001` once at boot on rigs that mine
unprivileged. At or above ~180 W, never.

**lolMiner was swept at the same rung the same evening** (same ABBA discipline, its
sol/s from its own steady 15 s windows — its definition, not comparable to ours
row-for-row; the watts are one instrument for both). It rides the rung too, but only
to ~126 W: its store-everything design needs ~17.7 GB/solve, so on the rung it
plateaus at **~34 sol/s flat from 120 W up** — its own DRAM roofline at the reduced
interface (263–301 GB/s sustained depending on its per-solve counting; at or near the
rung's ceiling either way), against our plateau of 43.6. Below ~126 W its interface
refund is *larger* than ours (+24 % at 100 W against our +14 %) — it was burning even
more of a tiny budget on bandwidth. Net, with each miner at its best memory clock per
cap, the low-band gap roughly halves in the 120–160 W band (from ~25–29 % to ~11–16 %
on reported figures) and *widens* at the 100 W floor.

| cap | MXBM sol/s @10251 | @5001 | MXBM J/sol @10251 | @5001 | lolMiner sol/s @10251 | @5001 |
|---|---|---|---|---|---|---|
| 100 W | 18.6 | 21.85 | 5.36 | 4.57 | 23.25 | 28.85 |
| 120 W | 25.0 | 29.3 | 4.79 | 4.09 | 32.2 | 34.05 |
| 140 W | 31.6 | 35.65 | 4.43 | 3.94 | 39.75 | 34.0 |
| 160 W | 38.3 | 41.9 | 4.18 | 3.79 | 47.85 | 34.1 |
| 180 W | 45.25 | 43.2 | 3.98 | 4.12 | — | — |
| 200 W | 50.75 | 43.3 | 3.94 | 4.54 | — | — |
| 220 W | 56.05 | 43.55 | 3.92 | 4.93 | — | — |
| 250 W | 58.2 | 43.65 | 4.29 | 5.25 | — | — |
| 285 W | 59.5 | 43.65 | 4.78 | 5.27 | — | — |

*The lolMiner dashes above 160 W are "not measured", and deliberately: its
stock-memory curve at 180–285 W is already in the head-to-head table above, and its
rung column cannot move — it is bandwidth-bound there, proven inside the sweep itself
(140 → 160 W added 20 W of draw and 0.1 sol/s). Re-measuring a plateau three points
already pin would be arms spent on a foregone answer.*

![Both miners at both memory rungs: speed and MXBM's J/solution against the cap](tools/mclk-curve.svg)

</details>

### The low band on the current kernel

Re-swept **2026-08-18, one session end to end with the head-to-head table above**:
`-lmc 5001,5001` held across `power_sweep.sh`, forward then reversed (the arms agree to
0.5 % at 120 W and above, 3.1 % at the 100 W floor), 120 s points, shipped defaults —
speculative entry off below 150 W on the rung, its measured crossover
([ledger](performance-research.md#speculative-entry-under-a-cap-both-crossovers-measured-and-the-gate-moves-to-them)).
The comparison column is **lolMiner at its own best configuration per cap**, measured
in the same session: it rides the 5001 rung too, and below ~115 W rides it harder —
its interface refund at the floor is larger than ours — but its store-everything design
needs ~17.7 GB/solve, so on the rung it meets its own DRAM roofline and **plateaus at
~34 sol/s from 110 W up**, where its best reverts to stock memory. The gap column is
the interleaved pair where both miners run the rung (100–120 W) and column arithmetic
above that.

| cap | ms/solve | sol/s | J/sol | sol/s/W | vs stock memory | lolMiner best | gap |
|---|---|---|---|---|---|---|---|
| 100 W | 69.4 | 28.85 | 3.465 | 0.289 | +17.8 % | 30.35 *(rung)* | **−3.0 %** |
| 110 W | 61.2 | 32.85 | 3.347 | 0.299 | +13.9 % | 33.95 *(rung)* | **−2.1 %** |
| 120 W | 54.6 | 36.80 | 3.258 | 0.307 | +12.4 % | 34.10 *(rung)* | **+8.5 %** |
| 140 W | 46.7 | 42.95 | 3.257 | 0.307 | +10.7 % | 41.50 *(stock)* | **+3.5 %** |
| **160 W** | 40.4 | **49.55** | **3.219** | **0.311** | +7.3 % | 49.95 *(stock)* | −0.8 % |
| 175 W | 39.9 | 50.60 | 3.416 | 0.293 | −4.0 % | 52.90 *(stock)* | −0.4 % *(stock vs stock)* |

**With each miner at its best configuration the band splits at ~115 W.** Below it
lolMiner leads by 2–3 % — the one place its bandwidth-light band advantage and its
larger rung refund stack. From 120 W up the plateau caps it: MXBM leads by **8.5 % at
120 W** and **3.5 % at 140**, is level at 160 and 175 (−0.8 and −0.4 %), and leads
outright from ~177 W ([the head-to-head table](#both-miners-under-the-same-cap)). The
residual deficits are the ≤115 W floor at 2–3 % and the 155–177 W strip at under 1 %.

**The rung plateaus near 50.7 sol/s from 175 W up** — above ~172 W it stops drawing
its cap (178 W observed at a 180 W limit): the memory watts the rung frees exceed what
the core can absorb. The plateau was ~47 sol/s one kernel ago; levers that delete
instructions pay more on the rung, where the core is the constraint, than at stock. The
rung-versus-stock crossover sits at ~167 W.

**The efficiency optimum is a tie between the rung and stock memory.** This sweep puts
the rung at 160 W at **3.219 J/sol** against stock memory's 3.232 at 210 W — inside
the arm spread — with stock delivering 31 % more throughput at its point (64.9 against
49.55 sol/s). The recommendation is unchanged: 210 W on stock memory where the rig can
supply it, the 160 W rung where it cannot. lolMiner's best point in the same session
is 3.205 J/sol on its own sol/s definition, a tie with both.

The geometry below the ~130 W gate is unchanged and its A/B stands, at its recorded
scope:

| 5001 rung | (16,1), shipping | (17,0), forced | |
|---|---|---|---|
| 140 W | 37.9 / 37.6 → **37.75** | 37.7 / 37.6 → **37.65** | −0.3 %, null |
| 160 W | 43.8 / 43.7 → **43.75** | 40.5 / 40.5 → **40.50** | **−7.4 %** |

**The gate is in the right place.** The two geometries tie at 140 W and (16,1) wins
clearly by 160 W, so the crossover sits just above the gate rather than well above it.
The 160 W arm is also that experiment's positive control: the same `MXBM_BB=17` that
changes nothing at 140 W moves 160 W by 7.4 % with both arms identical to 0.1 sol/s.
*(The 2026-08-15 kernel; the geometries were not re-raced on the current one.)*

### The OpenCL backend under caps

The fallback's first cap curve (2026-08-18): `--solver opencl`, one session, 60 s per
point after a two-minute warmup, stock memory, coarse grid — single-session absolute
figures, so they carry the cross-session band, not the paired harness's.

| cap | sol/s | ms/solve | J/solution |
|---|---|---|---|
| 100 W | 20.8 | 96.7 | 4.79 |
| 110 W | 24.3 | 82.5 | 4.52 |
| 120 W | 27.6 | 73.6 | 4.34 |
| 140 W | 34.0 | 58.7 | 4.12 |
| 160 W | 40.6 | 49.1 | 3.95 |
| 175 W | 45.7 | 43.6 | 3.83 |
| 190 W | 50.7 | 39.4 | 3.74 |
| 210 W | 57.4 | 34.8 | **3.65** |
| 240 W | 61.1 | 32.9 | 3.92 |
| 285 W | 63.6 | 31.6 | 4.46 |

**The fallback's deficit against CUDA widens as the cap drops**: ~1.09× at stock,
~1.15× at 140 W, ~1.29× at 100 W — the cap multiplier pricing the backend's larger
dynamic instruction count. Its efficiency floor (3.65 J/solution at 210 W, against
CUDA's 3.23) never crosses under CUDA's, so a capped rig should run the CUDA path
wherever the card allows it. Speculative entry and its gate are CUDA-only; nothing on
this curve depends on them.

### The memory traffic is compulsory

Per solve, against the minimum the round schedule and record widths require —
each round reading its input layer once and writing its output layer once:

| stage | compulsory rd | wr | measured rd | wr | excess |
|---|---|---|---|---|---|
| entry | — | 268 MB | 0 | 257 MB | −4.1 % |
| r1 | 268 | 537 | 271 | 525 | −1.4 % |
| r2 | 537 | 2147 | 539 | 2130 | −0.5 % |
| r3 | 2147 | 2147 | 2148 | 2131 | −0.4 % |
| r4 | 2147 | 268 | 2149 | 267 | −0.1 % |
| terminal | 268 | — | 270 | 2 | +0.7 % |
| **total** | | | **10.69 GB** | | **−0.6 %** |

Every line is within **1.4 %** of compulsory: no write amplification, no
redundant re-reads, nothing left for the cache to save. Nsight re-taken 2026-08-16
on the shipping record, so these are measured rather than derived — the previous
edition carried the reference rows' and the round-4 record's deletions as arithmetic
on an older run, and the re-take lands within 1 MB of it on every line. Entry's
−4.1 % is not a saving — it is ~11 MB of its output still sitting dirty in a 48 MB
L2 when the kernel ends, billed to the next one. The kernel times sum to **101 %**
of the solve, which also rules out the fourth lead: there is no idle spin between
rounds to reclaim.

`./cuda/profile.sh`, 2026-08-14. The total was 13.51 GB before
[the pad was removed](performance-research.md#the-round-2-alignment-pad) and 13.00 GB
after; the [implicit-bits record](performance-research.md#populations-are-pinned-at-225-and-the-occupancy-tail-prices-a-spill-arena)
took it to 12.30 by deleting round 2's side plane, and
[the replay](performance-research.md#the-back-reference-rows-are-gone-recovery-replays-instead-203-ms-and-688-mib)
to **10.69** — 1.11 GB of reference rows and 0.54 GB from round 4's record at 8 B instead
of 16. That plane was the last line
materially above compulsory — round 2 used to write 72 B of record as 64 + 8 to
two places, and the +4.1 % it cost was the sector granularity of the split.

So the byte count can only fall by making records narrower, and one place was
found where a record was wider than its own contents ([the round-2 alignment
pad](performance-research.md#the-round-2-alignment-pad)), and one where the record
carried bits its own address already encoded ([the implicit-bits
record](performance-research.md#populations-are-pinned-at-225-and-the-occupancy-tail-prices-a-spill-arena)).
Everything else needs the structural change in
[HW_REQUIREMENTS.md](HW_REQUIREMENTS.md#the-design-target-is-met):
streaming / in-place layer reuse.

### What the footprint still costs

The sweep settles the *ranking*, not the *margin*. At its own operating point the
lolMiner spends **4.48 J per solution**; MXBM spends **4.11** at stock and **3.37 at
220 W**, its optimum. Undercutting it at stock rather than only by capping is new, and
where the remaining gap to the ideal went is not mysterious: 10.69 GB of
compulsory traffic per solve, the total measured in the table above. **Traffic,
specifically — not the footprint it sits in.** Those were treated here as one problem and they are two:
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

> **Benchmark figures here are only valid from an otherwise-idle machine.** A 155.2 ms
> reading, with a ~38 ms gap that looked like work outside the pipeline, turned out to be
> builds and tests competing for the same GPU. It does not reproduce on an idle one.

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

The six per-round command buffers, each with a `waitUntilCompleted`, cost **1.6 ms
total** — batching them is not a lead. r2 is the whole story.

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

> **Measure the thing; do not divide two numbers and call it a bandwidth.** This change
> was predicted to net only ~4 ms, and declined on that basis. The arithmetic assumed
> ~251 GB/s effective bandwidth, inferred by dividing round 3's traffic by its wall time —
> but round 3 is not purely bandwidth-bound, so the real figure is far higher and the
> extra traffic much cheaper than predicted.

### Three nulls, so they are not re-proposed

<details>
<summary>Threadgroup size, occupancy via smaller groups, 128-bit vector access</summary>

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
  median), so the scalar form is kept and the complexity is not.

</details>

### A measurement hazard worth knowing

Single-solve timings on this machine are thermally confounded: an identical configuration
read 115 ms early in a sweep and 133 ms late, and one FCAP sweep drifted 16.7 ms end to
end — enough to invent a 14 % "win" out of nothing. Every figure above is a median over
≥ 10 solves with the baseline repeated first *and last*; if the two brackets disagree by
more than a millisecond or two the run is discarded.

The `bb + sm = 17` line is carried over from Ada and not re-tuned here. Register counts
are not observable on Metal (`tests/test_metal_resources.mm`), so Ada's occupancy work has
no direct equivalent.

---

**Reference hardware** for every measurement below: RTX 4070 Ti SUPER (Ada, sm_89,
66 CUs, 16 GB, 48 KB LDS/workgroup, ~510 GB/s achievable copy bandwidth).

**How to reproduce.** Two figures are quoted throughout, and they measure different things:

- `./build/bench_rounds 20` — median over 20 runs of the **solver pipeline** (entry through
  terminal). This is the controlled number used to make optimization decisions.
- `./build/tests/test_gpu_solver` — **end-to-end `GpuSolver::solve()`**, including survivor
  readback, back-reference recovery and CPU verification. Take `solve #2` (steady state,
  persistent buffers); `solve #1` pays one-off allocation. This is what a miner reports,
  so it is the headline.

Both are noisy at the ~1 ms level, so single samples are not meaningful — quote a median
of at least 5. `MXBM_NO_ROWBUCKET=1` forces the fallback sort path.

sol/s ≈ 2006 / ms, since BeamHash III yields 2.006 solutions per solve.

---

## Third-party hardware — a two-card rig, 2026-08-02

Contributed: `--tune` sweeps plus a 21-minute pool session. MXBM 0.7.253, driver
610.62, Windows, both cards on CUDA. Different silicon and cooling from the
reference card, so only the shape of each curve transfers.

**RTX 4070 SUPER** (Ada, 12 GB, 192-bit), stock memory:

| cap W | draw W | sol/s | ms/solve | sol/s/W |
|---|---|---|---|---|
| 220 | 213.8 | 43.88 | 44.8 | 0.2053 |
| 212 | 205.3 | 44.83 | 43.5 | 0.2183 |
| 202 | 195.8 | 44.28 | 44.4 | 0.2262 |
| 196 | 191.1 | 44.13 | 44.9 | 0.2309 |
| 192 | 187.5 | 42.87 | 45.7 | 0.2287 |
| 182 | 176.9 | 41.77 | 46.9 | 0.2361 |
| 172 | 167.6 | 42.30 | 46.2 | 0.2524 |
| 148 | 144.3 | 36.93 | 53.7 | 0.2559 |
| 124 | 121.1 | 30.25 | 66.4 | 0.2498 |
| 100 | 97.2 | 22.41 | 87.6 | 0.2305 |

**RTX 3060 Ti** (Ampere, 8 GB), stock memory:

| cap W | draw W | sol/s | ms/solve | sol/s/W |
|---|---|---|---|---|
| 200 | 195.2 | 22.41 | 88.1 | 0.1148 |
| 180 | 175.9 | 21.92 | 90.5 | 0.1246 |
| 160 | 155.9 | 21.04 | 94.4 | 0.1350 |
| 150 | 146.5 | 20.79 | 95.9 | 0.1419 |
| 140 | 136.7 | 19.70 | 100.5 | 0.1442 |
| 130 | 126.8 | 19.11 | 104.7 | 0.1506 |
| 120 | 116.9 | 17.14 | 115.8 | 0.1467 |
| 100 | 98.0 | 11.71 | 171.9 | 0.1196 |

![Every measured card's speed and efficiency against the cap it was given](tools/cards-curve.svg)

Generated from the tables above and the head-to-head table by
`docs/tools/plot_cards.py`. **Not a controlled comparison** — different machines,
operating systems and instruments — so cross-card distances are unreliable. Each
curve's shape and its own efficiency peak are what transfer.

### The memory rung's crossover is a property of the card, not of the algorithm

The 5001 MHz down-rung pays below ~173 W on the 4070 Ti SUPER
([the record](#below-stock-the-other-rung-pays-7-to-18--under-caps-below-165-w)).
On the 4070 SUPER it pays only below **~121 W**, and above that it is expensive:

| cap W | draw W | sol/s | ms/solve | sol/s/W |
|---|---|---|---|---|
| 172 | 156.9 | 30.20 | 66.5 | 0.1925 |
| 148 | 139.0 | 30.15 | 66.5 | 0.2169 |
| 124 | 116.8 | 29.84 | 67.3 | 0.2555 |
| 120 | 113.4 | 29.60 | 66.6 | 0.2609 |
| 110 | 107.0 | 28.50 | 70.3 | **0.2663** |
| 100 | 97.9 | 25.62 | 78.0 | 0.2616 |

At 148–172 W the rung costs ~29 %. The mechanism is in the column: from 172 W down
to 120 W the time is **pinned at ~66.5 ms regardless of cap**, so the memory system
binds and the power cap does not. A 192-bit bus against the reference card's
256-bit is three quarters of the bandwidth at the same clock, which moves the
crossover 50 W. `--tune` found it unaided and recommended the rung only below
121 W.

The 3060 Ti got no rung pass: its GDDR6 runs at 6801 MHz stock and offers no 5001
rung. Correct, but the sweep skipped it silently and should name the skip.

### Open: the tune harness and the miner loop disagree

`--tune` measured the 4070 SUPER at **43.88 sol/s @ 220 W**; the live miner reported
**48.12 @ 219 W** in the same session, with the pool-side rate favouring the higher
figure. The discrepancy runs in the harder direction — the other card was idle
during the sweep and at 82 °C / 96 % fan during mining. The reported rate also
climbed 41.7 → 48.1 over twenty minutes while the core clock *fell* 2700 →
2550 MHz, which should not happen for a compute-bound kernel. The same session has
the 4070 SUPER outrunning the larger reference card below ~155 W, which may be the
same disagreement seen a second way. Unreproduced; not usable for a cross-miner
comparison until it is.

### The multi-GPU path itself

21 minutes, both cards in one process: **42 shares found, 42 accepted, 0 stale, 0
rejected**, no errors. The pool-side rate converged onto the miner's own total to
within ~4 % over the last five statistics blocks (0.6 % at the closest).

Two defects fell out, both fixed: `--pl auto` resolved a per-card wattage and passed
it to the apply path as a bare number, which is GPU 0's list entry, so every later
card announced a cap and kept its stock limit; and the startup device block reported
"Device 0" whichever card was primary. Still unexercised on hardware: two *different*
backends in one process (both cards chose CUDA), and the skip path on a card no
backend can drive.

---

## Benchmarks and gates

| Command | What it measures |
|---|---|
| `./build/bench_rounds 20` | Median ms/solve over 20 solves + drop-free gate |
| `./build/tests/test_gpu_rounds` | Per-round timing, drop counters, survivor count |
| `./build/tests/test_gpu_solver` | End-to-end solve incl. recovery and CPU verification |
| `./build/tests/test_lds_collide` | All isolated kernel benches and the correctness oracles |
| `ctest --test-dir build` | Full suite (37 tests) |

Correctness gates that must stay green for any performance change:
`bucketDrops == 0`, `pairDrops == 0`, all three KAT goldens byte-identical, and the full
suite. The goldens are checked through every collision path
(`gpu_*_rowbucket`, `gpu_*_lds`, `gpu_*_legacy`) so a path-specific regression is caught.

Diagnostic environment variables, ablation build flags and the rules they were designed
around are catalogued with the experiments that use them:
[instruments](performance-research.md#instruments-diagnostic-and-ablation-flags).
