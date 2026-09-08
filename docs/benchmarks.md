# Benchmark results

Measured numbers for MXBM, and for lolMiner on the same card, so the comparison is
like-for-like. Everything here is reproducible with the scripts in `benchmarks/` — the
commands are given under each table.

**MXBM has been measured on five GPUs**: an RTX 4070 Ti SUPER (the card everything is
developed against), an M3 Max via Metal, and a contributed RTX 4070 SUPER, RTX 3060 Ti
and GTX 1660 Ti.
They are tabulated under [benchmarked devices](#benchmarked-devices) below; to add yours,
see [send us your numbers](#send-us-your-numbers).

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

Unless a row or section says otherwise, every figure on this page is this card at
**stock clocks**, no overclock or undervolt.

---

## Benchmarked devices

One row per backend per card, each at the card's **stock** power cap — the reference
card appears twice because both of its backends are measured.

| GPU | Memory | Driver / OS | Backend | sol/s | ms/solve | W | sol/s/W | Source |
|---|---|---|---|---|---|---|---|---|
| RTX 4070 Ti SUPER | 16 GiB GDDR6X | 610.43.03 · Linux | CUDA | 74.8 | 26.9 | 285 | 0.263 | ours, `--benchmark` ‡ |
| RTX 4070 Ti SUPER | 16 GiB GDDR6X | 610.43.03 · Linux | OpenCL | 63.2 | 31.8 | 284 | 0.222 | ours, `--benchmark` ‡ |
| RTX 4070 SUPER | 12 GiB GDDR6X | 610.88 · Windows | CUDA | 55.8 | 36.2 | 211 | 0.265 | ZumZum, `--report` 120 s ※ ◇ |
| RTX 3060 Ti | 8 GiB GDDR6 | 610.88 · Windows | CUDA | 26.5 | 75.4 | 192 | 0.138 | ZumZum, `--report` 120 s ※ |
| GTX 1660 Ti | 6 GiB GDDR6 | 595.97 · Windows | CUDA | 9.48 | 210.6 | 106 | 0.089 | ZumZum, `--report` 120 s ◇ ◆ |
| Apple M3 Max (40-core) | 128 GB unified | macOS 26.5 · Metal 3 | Metal | 16.3 | 128.0 | — | — | ours, `--benchmark` |

‡ CUDA: the 285 W point of the 2026-09-09 power sweep below (90 s, released clocks).
OpenCL: 2026-08-17 at released clocks; the OpenCL kernels are unchanged since. The
published headline is the **27.00 ms / 74.40 sol/s**
[locked-clock pin](performance-research.md#the-named-reference-lgc-2600) of 2026-09-09,
which trades boost for reproducibility.

※ 120 s `--report` on MXBM 0.8.408, 2026-08-21. The 3060 Ti is thermally limited at
stock (81 °C), so its top end is partly a cooling figure.

◇ Measured with other compute processes on the card — 26 on the 4070 SUPER, 13 and a
display attached on the 1660 Ti. Read both as indicative, not as benchmarks; clean
re-runs are wanted. See
[third-party hardware](performance.md#third-party-hardware--contributed-cards).

◆ **The first Turing measurement**, and the one that prices the sm_75 cubin: the same
machine on 0.8.412 fell to the OpenCL path at 4.28 sol/s / 462.9 ms, so CUDA is worth
**2.2×** here. The 1660 Ti also never reached its board limit and shows no interior
efficiency optimum, which no other card does — possibly the contention, so it is not yet
read as a property of Turing.

The M3 Max's pipeline alone runs **101.5 ms/solve (18.8 sol/s)**; sustained is lower
because the laptop throttles. No power column — macOS exposes no GPU power API.

![Speed and efficiency against the cap, every card measured](tools/cards-curve.svg)

Curves for the four NVIDIA cards, generated from the tables in
[performance.md](performance.md#third-party-hardware--contributed-cards). **Not a
controlled comparison** — different machines, operating systems and instruments — so
cross-card distances are unreliable. What holds is each curve's shape and where its own
efficiency peak sits: 220 W on the 4070 Ti SUPER, 148 W on the 4070 SUPER and 130 W on
the 3060 Ti, all well under stock. The 1660 Ti is the
exception and the reason its figure carries a `≥`: its efficiency was still climbing at
the driver's 70 W floor, so that curve never turned over and its optimum is a bound.

---

## MXBM vs lolMiner 1.98a

Both mining BeamHash III on the same card, caps set externally with `nvidia-smi` so one
instrument measures both. **At equal power, inside the window where MXBM leads** — both
miners at 220 W, 2026-08-18, one interleaved session (the MXBM column is that day's
build; the current one is bounded −1.8 % ms at 220 W by paired arms, so every margin
here is a floor):

| | MXBM at 220 W | lolMiner 1.98a at 220 W | |
|---|---|---|---|
| Throughput | **66.3 sol/s** | 53.35 sol/s | **+24.3 %** |
| Board power | 219.6 W | 219.6 W | — |
| Efficiency | **0.3018 sol/s/W** | 0.2430 sol/s/W | **+24.2 %** |
| VRAM for a full search | 6.17 GiB[^4g] | ~4 GiB | |

**MXBM is ahead on both from roughly 177 W to the 285 W stock limit**, and has the higher
ceiling outright: 74.8 sol/s against ~54.4, which lolMiner cannot reach at any setting.
Below ~177 W lolMiner is ahead on stock memory — 0.4 % at 175 W, peaking at 7.5 % at
160 W — until ~104 W, below which MXBM leads again (+3.2 % at the 100 W floor); those
crossings are the 2026-08-18 head-to-head, whose MXBM column the current build is bounded
within +0.5 % (140 W) and 0.0 % (100 W) of. With each
miner at its best configuration the low band splits at ~115 W; see
[the low band](performance.md#the-low-band-on-the-current-kernel).

**Both curves are measured.** lolMiner was swept across the same caps as MXBM rather than
sampled once at its own uncapped draw, which is what makes that a comparison of two curves
instead of a curve against a point. Capping lolMiner improves its efficiency
substantially, and that is already priced in above.

**A stock-versus-stock comparison rewards whichever miner fails to fill the card.** MXBM
runs against the 285 W board limit in every kernel (284.8 W drawn unlocked; the driver
reports `sw_power_cap` active, so it is a power ceiling and not a thermal one); lolMiner draws 239 W
uncapped and leaves 46 W unused. That is why the equal-power table is the one above.

**What this does not show.** The two miners' `sol/s` are separate counters whose
relationship is [an open question](benchmarking.md#1-why-reported-sols-is-not-comparable)
— MXBM reports CPU-verified solutions and lolMiner's basis is undocumented, a distinction
worth ~17 % inside our own pipeline. The watts are trustworthy across miners because one
instrument measured both; the sol/s columns are each miner against itself. Accepted pool
shares over a fixed interval remain the only arbiter that needs neither counter.

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

MXBM's column: 80 minutes against the pool reads **62.32 sol/s**, its 15 s windows
spread σ = 1.99 (the uncertainty on that mean is ±0.11), and the controlled benchmark the
same day reads **62.7 sol/s at 32.0 ms/solve** (six 120 s runs, 0.0 % spread). The
kernel work since has taken the controlled figure to **74.40 sol/s at 27.00 ms**
(+18.7 %), so every margin in it is a floor on the current one. lolMiner's column is the 2026-07 session; its binary is unchanged.

</details>

[^4g]: at the fastest geometry, which is what both miners run here. lolMiner selects
"BeamHash III **4G** (CUDA)" on this card; MXBM's own floor is 1.90 GiB on a card that
cannot host the fast rung, which is a class below the 4 GB one — see
[HW_REQUIREMENTS.md](HW_REQUIREMENTS.md#the-design-target-is-met).

---

## MXBM power curve

The board power limit is the most valuable setting on this card, because MXBM is pinned
against it. Measured 2026-09-09 on the shipped build and defaults — `power_sweep.sh`,
90 s points, one session, headless:

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
speed. Above it each watt buys less clock; from 240 W up the curve flattens toward the
stock figure. Energy is read from the card's own energy counter over each 90 s window;
sol/s at this sample size reads about 1 % high (see the note under the head-to-head).

**Both miners have a `--pl`, and the honest comparison sweeps both.** The table above is
MXBM's curve alone; lolMiner was only ever measured uncapped, which turned out to matter
a great deal. Swept against each other at identical caps:

![Speed, efficiency and power drawn, both miners at the same caps](tools/power-curve.svg)

**Where each one wins.** In the head-to-head session (2026-08-18, both miners
interleaved at each cap, caps set externally, MXBM on the shipped defaults) lolMiner is
ahead on speed *and* efficiency between ~104 W and ~177 W on stock memory, at worst by
7.5 % at 160 W and down to 0.4 % by 175 W; below ~104 W it inverts and MXBM leads the
floor by 3.2 %. From ~177 W to the 285 W stock limit MXBM is ahead on both, by 24 % at
220 W and 26 % at 240 W. lolMiner barely responds to a cap at all — its ceiling is
54.4 sol/s at a 200 W cap, and above ~235 W the cap does nothing. Its best efficiency
(0.3120 sol/s/W at 160 W, 3.21 J/solution on its own counter) and MXBM's best in that
session (0.3094 at 210 W, 3.23 J/solution) are level within the repeat spread and the
own-definition caveat — and at MXBM's point the same energy per solution does **64.9 sol/s
against 49.95, 30 % more work**. What MXBM has is the ceiling: ~70 sol/s then and 74.8
now against ~54.4, which lolMiner cannot reach at any setting.

A head-to-head table is taken in one session because one assembled from two inherits a
cap-dependent term that reaches ~5 % at the low caps
([method](performance-research.md#how-small-a-difference-the-rig-can-resolve)). The MXBM
column in it therefore predates the kernel changes of 2026-09-08/09 and is a floor: the
changes are bounded by paired old-against-new arms at +0.5 % (140 W), 0.0 % (100 W),
−1.8 % ms (220 W) and −2.5 % (255 W), inside the row spacing of every crossing. The sweep
runs down to the card's **100 W floor**, which settles whether lolMiner had a better
efficiency point hiding below 120 W: it does not. Both curves fall away monotonically
below their peak.

Full numbers, the mechanism (we lose 570 MHz of core clock to it at 180 W) and what it
implies for the roadmap are in
[performance.md](performance.md#both-miners-under-the-same-cap). Reproduce with
`benchmarks/compare_power.sh`; the chart is generated from that table by
`python3 docs/tools/plot_power.py`.

Each point is 90 s (~2,000–2,500 solves). At that sample size the solutions-per-solve
factor reads ~2.02 where an 8,500-solve run measures 2.006, so **the sol/s columns read
about 1 % high in absolute terms**. Every point was measured the same way, so each
curve's shape, its peak and the crossings are unaffected. The figures are left as
measured; rescaling them would publish numbers no run produced.

Two things worth knowing before you cap your own card:

- **Efficiency peaks at 220 W and gets *worse* below it.** At 180 W the core clock
  has fallen to 1785 MHz and the parts of the board that do not scale with it — memory,
  uncore, leakage — are being paid for out of less work. Lower is not always better, and
  the curve shows this all the way down to the card's 100 W floor.
- **The last watts are the worst value.** Going 220 → 285 W buys 4.9 sol/s for 65 W; the
  20 W from 200 to 220 buys 7.6. Where to sit is an economic choice about your power
  price, not a technical one.

```sh
sudo mxbm --algo BEAM-III --pool ... --user ... --pl 220
```

`--pl` needs root. Without it MXBM says so by name and mines on at the card's current
limit. The previous limit is restored on exit, including on Ctrl+C. See
[usage.md](usage.md#power-limit).

---

## At the other end: the memory V/F offset

The low rung below is what a *capped* rig wants. An uncapped one wants the opposite knob.
Rounds 3 and 4 are 45 % of the solve at 77.4 % and 88.6 % of DRAM peak, so the memory
clock is the only knob on the resource that binds them — and on the reference card there
is headroom above the 10251 MHz it holds under load.

**Take it with `--moff`, not `--mclk`.** Locking to the driver's reported 10501 maximum
measures **+0.95 % slower**, because a locked P-state buys its clock with voltage and the
board is already at its power limit, so the watts come out of a solve that is 54 %
core-bound. The offset shifts the V/F curve instead. Bracketed `A B B A`, 12 arms of 40 s,
headless, drop counters zero and verified solutions/solve in band on both arms, on the
build of 2026-08-16 (the relative figures are what carries; the current stock median is
26.9 ms):

| | ms/solve | range | J/sol | sol/s |
|---|---|---|---|---|
| stock memory | 28.733 | 28.711 – 28.760 | 4.090 | 69.8 |
| `--moff 1600` (11051 MHz) | **28.050** | 28.027 – 28.066 | **4.008** | **71.5** |

**−2.37 % time and −2.00 % energy**, ranges separated by 0.645 ms. Both figures at once,
because the board is pinned at its limit: power is constant, so the energy win *is* the
speed win.

The sweep behind that point ran monotone to +2400 (11451 MHz) without reversing, but the
gain stops tracking the clock: a two-clock model predicts the measurement to 101 % and
97 % through +1400, then 80 % at +1600 and 71 % at +2200. That is the retry wall arriving
gradually, so +1600 is where the returns bend rather than where they stop.
[The mechanism](performance-research.md#locking-the-memory-clock-to-its-reported-maximum-costs-095--and-the-offset-is-the-knob-that-does-not);
[how to find your own](overclocking.md#finding-your-own-memory-offset).

**This figure is excluded from every comparison in this document,** and not only as
methodology: MXBM moves 10.69 GB a solve where a state-storing design moves 17.67, so the
same clock is worth more to the miner that stores. An overclocked head-to-head measures
the overclock.

---

## Capped rigs: the memory rung and the low-power kernels

The curve above is stock memory. Under a cap that is the wrong configuration: the memory
clock does not scale with the core, so the interface burns a fixed slice of a small budget
for bandwidth nothing is using. Dropping it to the **5001 MHz rung** returns those watts as
core clock, and below ~130 W MXBM also switches to a different geometry and rebuild
variant. Together that is MXBM's best configuration at each cap.

Measured **2026-08-18, one session end to end** with the head-to-head table above:
`-lmc 5001,5001` held across `power_sweep.sh`, forward then reversed (the arms agree to
0.5 % at 120 W and above, 3.1 % at the 100 W floor), 120 s points, shipped defaults. The
"vs stock memory" column is against that session's own stock-memory sweep, and the
comparison column is lolMiner at *its* own best configuration per cap, measured in the
same session. The rung has not been re-swept on the current build; below 220 W the
kernels it runs are unchanged except for the four 2026-09-08 changes, bounded at
+0.5 % (140 W) and 0.0 % (100 W) by paired arms:

| cap | ms/solve | sol/s | J/solution | sol/s/W | vs stock memory | lolMiner at its own best | gap |
|---|---|---|---|---|---|---|---|
| 100 W | 69.4 | 28.85 | 3.465 | 0.289 | +17.8 % | 30.35 *(rung)* | **−3.0 %** |
| 110 W | 61.2 | 32.85 | 3.347 | 0.299 | +13.9 % | 33.95 *(rung)* | **−2.1 %** |
| 120 W | 54.6 | 36.80 | 3.258 | 0.307 | +12.4 % | 34.10 *(rung)* | **+8.5 %** |
| 140 W | 46.7 | 42.95 | 3.257 | 0.307 | +10.7 % | 41.50 *(stock)* | **+3.5 %** |
| **160 W** | 40.4 | **49.55** | **3.219** | **0.311** | +7.3 % | 49.95 *(stock)* | −0.8 % |
| 175 W | 39.9 | 50.60 | 3.416 | 0.293 | −4.0 % | 52.90 *(stock)* | −0.4 % |

**Take the rung only if you are hard-limited to ~167 W or below.** It is worth +7 % at
160 W rising to +18 % at the 100 W floor — and with each miner at its own best
configuration MXBM *leads* the band at 120 and 140 W, is level at 160 and 175, and
trails only below ~115 W. What it is not is an efficiency setting: its 3.219 J/solution
at 160 W was a tie with that session's stock-memory optimum, and the current build's
stock optimum reads 3.14 J/solution at 220 W doing **69.9 sol/s against 49.55** — 41 %
more work for no more energy per solution.

The mechanism, the A/Bs behind each step and lolMiner's own rung sweep are in
[performance.md](performance.md#below-stock-the-other-rung-pays-7-to-18--under-caps-below-165-w).

```sh
sudo mxbm --algo BEAM-III --pool ... --user ... --pl 160 --mclk 5001
```

Both settings are restored on exit. The crossover is ~167 W; above ~180 W the rung is a
wall — do not use it there.

One more thing for a capped rig: **use the CUDA path, not the OpenCL fallback.** The
fallback's gap grows as the cap drops — 16 % behind at stock on the current build, and
15 % at 140 W and 29 % at 100 W when the two were swept together
([the curve](performance.md#the-opencl-backend-under-caps)).

---

## Where the time and energy go

Per solve, from `benchmarks/stage_power.sh` (which replays one stage many times inside a
real solve and solves for its own time and power) and from Nsight Compute:

![Board power across one solve, by stage](tools/stages.svg)

The chart is the table below, drawn as a timeline: width is time, height is power, so
each block's **area** is the energy that stage costs. Generated from the table by
`python3 docs/tools/plot_stages.py`, so the two cannot drift.

Measured 2026-08-16 on that day's kernels, 8 reps, 45 s per stage, baseline
29.23 ms/solve, drops 0:

| stage | ms | % of solve | power | DRAM traffic | bound by |
|---|---|---|---|---|---|
| `entry_scatter` | 2.66 | 9.1 % | 282.9 W | 0.27 GB | compute (SipHash) |
| round 1 | 5.02 | 17.2 % | 285.1 W | 0.81 GB | latency |
| round 2 | 8.21 | 28.1 % | 282.8 W | 2.68 GB | latency |
| round 3 | 8.42 | 28.8 % | **281.0 W** | 4.29 GB | **DRAM** |
| round 4 | 4.20 | 14.4 % | 283.5 W | 2.42 GB | **DRAM** |
| terminal | 0.78 | 2.7 % | 285.3 W | 0.27 GB | DRAM |
| **total** | **29.30** | 100.2 % | 282.8 W | **10.73 GB** | |

**Scope: this is the 29.30 ms build's breakdown, not the 27.00 ms one's.** The per-stage
split has not been re-taken with power; `benchmarks/stage_power.sh` is what re-takes it.
The *time* split on the current build is in the per-stage table `MXBM_ROUND_STATS=1`
prints after a benchmark (round 1 4.73, round 2 8.00, round 3 9.68 with the next solve's
entry pass running beside it, round 4 3.92, terminal 0.51, recovery 0.03 ms). That is
also why `entry_scatter` has no dispatch of its own at stock: speculative entry runs the
next solve's pass beside rounds 3 and 4 on a higher-priority stream (inside round 4's
launch when this table was taken, where the pair cost 6.39 ms against the 6.86 these two
rows sum to when replayed apart).

Deleting the reference rows is visible here as its own mechanism: **round 4 −1.21 ms**
(its row and half its output record), **round 3 −0.54**, **round 2 −0.29**, **round 1
−0.14**, **terminal −0.20**, and entry unmoved. Round 4 carries more than half the win
because it lost the most: a row for every child *and* a record that went 16 B to 8. The
DRAM column falls 12.30 → 10.73 GB, all of it the four rows (0.27 GB each) plus round
4's narrowed record on both the write and the terminal's read; it is the compulsory figure
from the stride table, which the previous kernel's Nsight measurement matched to 0.7 %.

The stage sum runs 0.2 % over the baseline it is measured against, so read the per-stage
deltas as attribution and the interleaved **−2.029 ms** as the number.

Every stage draws the board limit, which is why there is no single kernel to "fix" for
power. Total DRAM traffic is within **0.7 %** of the compulsory minimum for the record
widths — there is no waste to reclaim, only records to narrow, which is the same problem
as the footprint. Full analysis in
[performance.md](performance.md#power-and-efficiency).

Note what "compulsory" does and does not claim: it fixes the **byte count**, not the rate
those bytes move at. Narrowing this path is priced in memory instructions and sectors,
never in bytes — halving round 3's write sectors buys 4 % of the round, an implied
~3000 GB/s against a 656 GB/s bus. See
[the dead word](performance-research.md#round-3-stores-a-work-word-round-4-never-reads--worth-268-mb-and-nothing-in-time).

---

## Reproducing any of this

Build first — see [building.md](building.md).

| What | Command | Needs root |
|---|---|---|
| **Everything below in one block** — hardware, throughput, telemetry, power curve | `mxbm --report` | for the curve only |
| Throughput, clocks, temperature and J/sol | `mxbm --benchmark BEAM-III --benchmark-seconds 120` | no |
| Power/speed curve for your own card | `mxbm --report` (or `--tune` for the recommendation alone) | yes |
| Per-stage time and power | `benchmarks/stage_power.sh` | no |
| **Correctness gate for a CUDA change** | `./build/tests/test_cuda_solver` — 3/3 goldens on 22 configurations | no |

The kernel-level instruments — Nsight counters, the pipeline A/B, the VRAM-ladder
rungs, the drop counters and their positive controls — are catalogued in
[performance-research.md](performance-research.md#instruments-diagnostic-and-ablation-flags).

Run on an **idle GPU**. A benchmark taken while something else is using the card measures
contention, not the miner — a lolMiner run taken while MXBM was mining read 25–27 sol/s
instead of 53.

Quote a **median over several thousand solves**, not a peak. Under a few thousand, the
solutions-per-solve factor alone moves the headline by more than a sol/s — a 90 s run
reads ~2.02 solutions per solve where a long run measures 2.006. Individual 15 s
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
  OS releases, so MXBM reports utilization and nothing else. Every efficiency claim
  elsewhere in this document is therefore Ada-only, and there is no sol/s/W row here.
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
inflated one measurement by 30 %. Quote medians, bracket a sweep with the baseline repeated first
and last, and discard the run if the two brackets disagree.

---

## Send us your numbers

Everything in the Ada sections above — the geometry choices, the power curve, the record
layout, the claim that rounds 3 and 4 are bandwidth-bound — is measured at one memory
bandwidth with one shared-memory budget, and some of it does not transfer. Two cases are
already known: round 2's rebuild costs 0.4 ms on Ada and 21 ms on Apple, and the 5001 MHz
memory rung pays below ~167 W on the 4070 Ti SUPER but only below 121 W on the 4070 SUPER.

Particularly wanted:

- **Anything that is not Ada** — Turing, Blackwell, A-series. (Ampere and Turing: one
  report each so far.) Turing is where MXBM is furthest behind, and every report from an
  RTX 20 or GTX 16 card carries the per-stage table that says why.
- **Smaller cards**: 8–12 GB. MXBM refuses to start below its threshold rather than mine
  nothing, so a refusal is itself a useful report — tell us what it said.
- **AMD**, via the OpenCL backend. It is completely untested there.
- **OpenCL on anything but the reference card.** Every contributed run so far chose CUDA.
- **Anything where MXBM is slower than another miner on your hardware.** That is the most
  useful report of all, and it will not offend anyone.

### One command

```sh
mxbm --report
```

It benchmarks the card for two minutes, then measures its power/speed curve, and
prints a paste-ready markdown block — hardware, driver, sol/s, telemetry, efficiency
and the whole curve. Nothing is uploaded; it only writes to your terminal, and to a
file with `--report-out report.md`.

The curve is the valuable half and it takes about half an hour, so it is measured
once and reused: a later `--report` on the same binary reads it back in two minutes.
A new build re-measures, because a curve taken on different kernels describes a
different program. Setting the power limit needs root (`sudo mxbm --report`), or an
Administrator terminal on Windows — without it you still get everything except the
curve, and that is still a useful report.

Then open an issue with the **Benchmark report** template and paste it in. If MXBM
refuses to start on your card, the refusal message is the report — send that.

### What happens to it

Results get added to the [benchmarked-devices table](#benchmarked-devices) with
attribution, and hardware that behaves differently from the reference card becomes a
work item. If MXBM turns out to be slow or broken on your GPU, please open a bug report.
