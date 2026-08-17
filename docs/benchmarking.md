# Benchmarking and comparing miners

> Looking for the **numbers**? They are in **[benchmarks.md](benchmarks.md)**. This page is
> the *method* behind them.

Reported `sol/s` is **implementation-defined** — two miners doing identical work can differ
by 17 % legitimately, because they count different things. This page covers how MXBM
measures itself, how small a difference the rig can resolve, and the protocol that settles
a cross-miner comparison without trusting either miner's counters.

---

## 1. Why reported sol/s is not comparable

A BeamHash III run over `2^25` elements yields **about two solutions on average** — a
property of the Wagner parameterization, not of any implementation ([bh2] measured
1.92–2.03 across the EquihashR family). But "a solution" admits two readings, and MXBM has
measured both on its own pipeline:

| What is counted | per solve |
|---|---|
| Survivors the GPU search produces | **2.29** |
| Survivors that pass CPU verification | **1.95** |

A **17 % spread inside one implementation.** MXBM reports the conservative number:
`GpuSolver::solve()` runs `bh3::is_valid_solution` on every candidate first, and the stats
hook counts exactly what `solve()` returns. Which definition a closed-source miner uses is
unknown, so **a gap smaller than that ambiguity cannot be called real** from reported
figures alone. Use §3.

---

## 2. What MXBM measures

```sh
./build/bench_rounds 20            # solver pipeline only, median of 20
./build/tests/test_gpu_solver      # end-to-end solve(), incl. recovery + verify (take #2)
MXBM_SOLRATE=20 ./build/tests/test_gpu_solver     # verified solutions per solve
./build/mxbm --benchmark BEAM-III --benchmark-seconds 120    # offline, no pool
benchmarks/headline.sh             # the only way to publish an absolute figure
```

A single sample is not meaningful — quote a median of at least 5 and prefer a spread.
`MXBM_NO_ROWBUCKET=1` forces the fallback sort path.

### The named reference: `LGC-2600`

"Benchmark it" means exactly this, so two people saying it run the same thing:

```sh
sudo -v && LGC=2600 LMC=10251 benchmarks/headline.sh
```

`headline.sh` refuses to run if another process holds >256 MiB of VRAM — **checked between
every arm, not just before the first** — discards a 240 s warmup, repeats the run, and
records NVML telemetry per run. The telemetry is what makes a failure diagnosable: the
original measurement recorded a number and nothing else, so when it failed to reproduce
there was nothing to diff.

Reference values (RTX 4070 Ti SUPER, driver 610.43.03, re-measured 2026-08-16, six 120 s
runs, **compute GPU headless**):

| quantity | reference | gate |
|---|---|---|
| ms/solve median | **29.10** (spread 0.0 %) | a build change is real past ±0.5 % |
| sol/s | **69.20** | derived; quoted because the pin cites it |
| SM / mem clock | 2610 / 10251 MHz | must match, or it is a different V/f point |
| board draw | **275.5 W** | context, not a gate — 271–281 W across six sessions at identical clocks and work |
| `clocks_event_reasons` | none, or `sw_power_cap` at any fraction | any **other** flag voids the run. The fraction is not predictive and is not a gate |
| display on the compute GPU | **no** | a compositor costs **0.20 ms and ~6 W** |

That last row is a condition, not decoration — it is worth 0.6 %, more than the ±0.5 %
gate, and it is not a standing rig property: the compositor pin (`~/.config/uwsm/env`)
follows the HDMI cable at every login. Verify per session — the miner prints `reserving
64 MB (headless)` or `reserving 256 MB (display attached)`.

**Three rules.** Builds are compared at this pin, never on the stock number (stock carries
the ~2.5 % cross-session band and gates nothing). It is only a pin while the clocks read
2610/10251 — a run that drifted off is discarded, not averaged. Record J/solution
alongside ms/solve. Re-pin after any kernel-shipping day, or any change to what else the
card is doing; a repeat that moves past ±0.5 % *without* a shipped change reopens the
V/f-state investigation.

**Rows name what shipped and link the ledger, never a commit** — rewriting a message
re-hashes the commit and the citation dangles, which has already happened here.

<details>
<summary>Pin lineage</summary>

| pin | ms/solve | draw | what moved |
|---|---|---|---|
| 2026-08-16 | **29.10** | 275.5 W | the packed `tab` word ([ledger](performance-research.md#the-census-and-the-chain-share-one-tab-word-and-a-barrier-goes-with-it)) — **69.00 → 69.20 sol/s**. −0.060 ms is under the print resolution; the finer `solves/s` line reads 34.33 → 34.44, i.e. −0.094 ms, agreeing with the interleaved −0.060. First pin with 0.0 % spread on both columns |
| 2026-08-16 | **29.10** | 277.0 W | the block-exit barrier, then singleton-free staging ([barrier](performance-research.md#the-block-exit-barrier-is-removable-and-the-barrier-family-is-over-priced-10x), [singletons](performance-research.md#singleton-free-staging-the-prize-is-085-ms-and-the-prepass-that-finds-it-costs-076)) — **−0.20 ms, 68.60 → 69.00**. Covers two changes; separable only in the interleaved A/Bs (−0.048 and −0.089, summing to −0.137) |
| 2026-08-15 | **29.30** | 271.1 W | the reference rows came out ([ledger](performance-research.md#the-back-reference-rows-are-gone-recovery-replays-instead-203-ms-and-688-mib)) — **−2.00 ms, 64.20 → 68.60**, against −2.029 measured ABBA. Verified solutions/solve 2.01, unchanged |
| 2026-08-15 | **31.30** | 271.9 W | the w0-checkpoint pair record ([ledger](performance-research.md#the-w0-checkpoint-pair-record-repriced-by-the-address-bits-076-ms-24-)) — **−0.70 ms, 62.70 → 64.20**, against −0.76 measured ABBA |
| 2026-08-15 | **32.00** | 272.1 W | nothing — a campaign session baseline. Fifth consecutive 32.00 / 62.70, and the first spanning two calendar days, so the pin reproduces *across* sessions |
| 2026-08-14 | **32.00** | 272.9 W | nothing at stock (three back-ref rows retired and `gi` dropped from r2, both octo-rung only). Draw reproduces the previous pin to the decimal — the first cross-session agreement on it |
| 2026-08-14 | **32.00** | 272.9 W | nothing at stock (octo record instantiations, bottom three rungs only) |
| 2026-08-14 | **32.00** | 279.9 W | nothing at stock (reach work: implicit-bits reclaim, overflow-arena rung, availability allowance). `sw_power_cap` 14–18 % of samples against the morning's 0–3 % at identical time — the fraction is a rig-day reading, not a build property |
| 2026-08-14 | **32.00** | 274.1 W | nothing at stock; a re-take under the kernel-shipping-day rule. Draw 7.3 W below the 08-13 pin at identical clocks and work, unexplained, so recorded as a rig-day difference and **not** attributed to the change |
| 2026-08-13 | **32.00** | 281.4 W | the implicit-bits record (−3.2 % in-miner A/B, [campaign](performance-research.md#populations-are-pinned-at-225-and-the-occupancy-tail-prices-a-spill-arena)); draw rose to the board limit |
| 2026-08-04 | 33.30 | 271.1 W | the monitor moved to the iGPU — **attributed to the rig**: the 08-01 build re-run headless reads 33.30 to the digit |
| 2026-08-01 | 33.50 | ~277 W | entry co-scheduling; r2 LD.128 + terminal perfect table. −2.3 % matches ship-time |
| 2026-07-28 | 34.30 | ~262 W | first pin |

The draw column is the cross-check: a busier binary *raises* draw at identical clocks
(08-01, +15 W), a quieter rig *lowers* it (08-04, −6 W).

</details>

---

## 3. How small a difference the rig can resolve

One number was doing three jobs. The ~2.5 % cross-session band is right for a *published
absolute figure* and wrong as the resolution of a *paired* A/B — used as the latter it
retires levers this rig can see. Measured directly:

| protocol | band |
|---|---|
| cross-session absolute | ~2.5 % — still the right caution for a published figure |
| one 30 s run against another, post-warmup | **0.112 %** (1σ, n = 30) |
| paired ABBA, 12 arms a side | **0.040 %** |
| both arm orderings (48 runs, ~30 min) | resolves **~0.05 %** at t ≈ 4 |

`benchmarks/paired_ab.sh` is the bottom row:

```sh
benchmarks/paired_ab.sh old new                          # ~25 min, ~0.05 %
SECS=90 BLOCKS=4 benchmarks/paired_ab.sh a b             # finer, slower
ENV="MXBM_BB=17 MXBM_SM=0" benchmarks/paired_ab.sh a b   # another rung
sudo -v && CAP=120 benchmarks/paired_ab.sh a b           # under a cap
```

- **Count solves over a fixed window**, not the printed median — that prints to 0.1 ms
  (0.35 %), while ~1045 solves resolves 0.1 %. The count, not `sol/s`: the
  solutions-per-solve multiplier is the algorithm's and only adds variance. The harness is
  *quantization*-limited (sd ~0.45 solves against a 1-solve grid), so resolution improves
  linearly with window length.
- **Warm up ~2 minutes.** Drift over 15 minutes is **+0.205 %** (`corr(ms, temp) = +0.575`),
  nearly all in the first four runs: the count falls 1050 → 1045, then holds 1045 ± 1.
- **Run both arm orderings.** ABBA cancels linear drift but not the A slot itself, which
  reads **+0.024 %** high. `effect = (d1 − d2)/2`, `bias = (d1 + d2)/2`; a single ABBA
  measures their sum.
- **Do not clock-normalise.** Tried, and worse: the band goes 0.112 % → 0.298 %. Once warm
  the clock spans **0.5 %**, not the ±10 % such a correction assumes.

**Two builds carry no layout noise.** Rebuilding identical source is not byte-identical —
48 bytes differ across 23.7 MB — but those are the ELF build-id note and the per-cubin
UUIDs; `cuobjdump --dump-sass` matches to the digit over 265,242 lines. So the whole
difference between two builds is the change.

---

## 4. Comparing against another miner

**Accepted pool shares are not implementation-defined.** The pool verifies every share, so
accepted shares over the same wall-clock time on the same hardware settles the question
without trusting either miner. Run both on the same GPU, pool, wallet and worker-name
convention, back to back — never concurrently.

1. **Fix the variables.** No other GPU load; same power limit and clocks, recorded. Use
   stock clocks — see [overclocking.md](overclocking.md). This is a rule, not a
   preference: **a memory overclock does not move two miners equally.** MXBM re-derives
   where a state-storing design writes 17.67 GB a solve against 10.69, so a memory OC
   flatters whichever design sits closer to its DRAM roofline and a core OC flatters the
   other.
2. **Same duration, 30 minutes minimum**, 60 better.
3. **Record per run:** accepted shares (from the *pool dashboard*), rejected and stale,
   duration, the miner's reported sol/s (the number under test), and end temperature and
   clocks.
4. **Compare accepted shares per hour**, trusting the pool over either miner.

**How long is long enough.** Shares are Poisson, so relative standard error is `1/√N`:
25 shares → 20 %, 100 → 10 %, 400 → 5 %, 1000 → 3 %. Resolving a 10 % difference at 2σ
needs roughly **400 accepted shares per miner** — hours, not minutes. If a full run is
impractical, report a confidence interval: a 30-minute run yielding 40 shares each cannot
distinguish 48 from 53 sol/s.

### At the same board settings

`benchmarks/compare_power.sh` runs both at identical settings. Four decisions make it mean
anything: settings are applied **externally via `nvidia-smi`**, identically to both (each
miner's own `--pl` would put their OC implementation into a kernel measurement); power is
**sampled from NVML**, never read from a miner's report (lolMiner's block averages in its
ramp-up — 215 W reported against 235 sampled); the two **alternate order at every point**;
and lolMiner's `--benchmark` is a fixed ~61 s run (established, not assumed), so both run
60 s and points repeat.

```sh
LIMITS="200 220 240 285" REPEATS=2 benchmarks/compare_power.sh
LIMITS="220 240" CCLKS="* 2100 2200" benchmarks/compare_power.sh   # a grid
```

`LIMITS` × `CCLKS` × `MCLKS` sweeps as a grid; `*` leaves a clock to the driver.
Everything is restored on exit including Ctrl+C — a card left locked outlives the script.
This does not make the sol/s columns comparable between miners (§1); the watts are, because
one instrument measured both.

**A swept column is provisional until one of its points is bracketed.** `power_sweep.sh`
gives one unrepeated run per cap — the right instrument for a curve's *shape*, the wrong
one for any single number in it. One column read 4–6 % low below 220 W and did not
reproduce on the same binary and script hours later; five explanations were priced and sum
to ~0.8 % of a 5.5 % gap ([write-up](performance-research.md#a-swept-cap-column-that-did-not-reproduce-and-the-five-explanations-that-were-not-it)).
So: reproduce at least one cap with an interleaved A/B before publishing, and **never treat
a difference between two sweeps as evidence about a kernel change.**

---

## 5. Gotchas

- **Check nothing else is on the card, before *and* after.** A benchmark left running by
  accident once sat on the card for 24 minutes and **halved both miners' throughput** at
  one power point — 175 W read 19.8 and 24.5 sol/s where a clean re-run gave 44.0 and 52.0.
  The signature is a point contradicting its neighbours *while the core clock rises*, and
  it looks like a finding until you check `nvidia-smi --query-compute-apps`.
- **A re-measurement REPLACES what it re-measures, never averages with it.** The same
  episode produced a median of the contaminated run and its clean replacement — 31.75
  sol/s, a number neither run measured.
- **Do not compare across pools**; difficulty and share accounting differ.
- **Count stale and rejected separately.** Submitting faster but staler is not better.
- **Restart between runs** — solve #1 pays one-off allocation.
- **If end clocks differ from start clocks, the comparison is invalid.**
- **State which definition of "solution" a figure uses**, or say it is unknown.

---

## 6. Reading the outcome

| Result | What it means |
|---|---|
| Their shares/hour clearly above ours | The gap is real; the rest is engineering |
| Within noise | The reported-sol/s gap was a **counting artefact**; parity |
| Ours above theirs | We are ahead, and the reported figures understate it |

The middle outcome is entirely plausible given the 2.29-vs-1.95 spread. Run the protocol
before drawing conclusions from headline sol/s in either direction, including our own
favour.

---

## References

- [performance.md](performance.md) — the measured state, power curves, hardware limits.
- [performance-research.md](performance-research.md) — every optimization tried, and why.
- [overclocking.md](overclocking.md) — why OC is excluded from a like-for-like run.
- [beamhash-iii.md](beamhash/beamhash-iii.md) — why a run yields ~2 solutions, and what a
  "survivor" is.
- [\[bh2\]][bh2] Wilke Trei, *BeamHash II Specification*, 17 June 2019 — §2.4 tabulates
  solutions per iteration across the EquihashR family.

[bh2]: https://docs.beam.mw/BeamHashII.pdf
