# Benchmarking and comparing miners

> Looking for the **numbers**? They are in
> **[benchmarks.md](benchmarks.md)** — MXBM against lolMiner on the reference card,
> the power curve, and how to submit results from your own GPU. This page is the
> *method* behind them.

Reported `sol/s` is **implementation-defined**. Two miners doing identical work can
display figures 17 % apart, legitimately, because they count different things. This page
documents how MXBM measures itself, why the obvious comparison is unreliable, and the
protocol that settles a comparison with no trust in either miner's counters.

---

## 1. The problem with comparing reported sol/s

A BeamHash III run over `2^25` elements yields **about two solutions on average** — a
property of the Wagner parameterization, not of any implementation. Wilke Trei's
[BeamHash II specification][bh2] measured 1.92–2.03 solutions per iteration across the
whole EquihashR family over 10,000 nonces each, and MXBM sees the same order on
BeamHash III.

But "a solution" admits two readings, and MXBM has measured both on its own GPU pipeline:

| What is counted | per solve |
|---|---|
| Survivors the GPU search produces | **2.29** |
| Survivors that pass CPU verification | **1.95** |

That is a **17 % spread inside a single implementation.** A miner reporting the first
number displays 17 % higher sol/s than one reporting the second, for identical work.

**MXBM reports the conservative number.** `GpuSolver::solve()` runs
`bh3::is_valid_solution` on every candidate before returning it, and the engine's stats
hook counts exactly what `solve()` returns. So MXBM's sol/s is verified solutions — the
lower of the two definitions.

We do not know which definition any closed-source miner uses. When a measured gap is
smaller than the ambiguity in the metric, **the gap cannot be called real** on reported
figures alone.

---

## 2. What MXBM measures, and how to reproduce it

Two figures are quoted throughout the docs and they measure different things:

```sh
# Solver pipeline only (entry through terminal), median of 20 runs.
# The controlled number used to make optimization decisions.
./build/bench_rounds 20

# End-to-end GpuSolver::solve(), including survivor readback, back-reference
# recovery and CPU verification. This is what a miner reports, so it is the headline.
# Take solve #2 -- solve #1 pays one-off allocation.
./build/test_gpu_solver

# Verified solutions per solve, over distinct prePows.
MXBM_SOLRATE=20 ./build/test_gpu_solver

# Offline solver benchmark, no pool and no wallet.
./build/mxbm --benchmark BEAM-III --benchmark-seconds 120
```

Both timings are noisy at the ~1 ms level, so **a single sample is not meaningful** —
quote a median of at least 5, and prefer a spread. `MXBM_NO_ROWBUCKET=1` forces the
fallback sort path.

The full measured history, including every failed experiment, is in
[performance.md](performance.md).

---

## 3. The protocol: accepted pool shares

**Accepted pool shares are not implementation-defined.** The pool verifies every share
independently, so comparing accepted shares over the same wall-clock time, on the same
hardware, against the same pool, settles the question without trusting either miner.

Run both miners on the **same GPU, same pool, same wallet, same worker-name convention**,
back to back — never concurrently, they would contend for the GPU.

1. **Fix the variables.** No other GPU load. Same power limit and clocks for both runs
   (`nvidia-smi -q -d POWER,CLOCK` before each; record it). Use stock clocks —
   overclocking is a separate exercise and must not be mixed in. See
   [overclocking.md](overclocking.md).
2. **Run each miner for the same duration.** 30 minutes minimum; 60 is better. Share
   arrival is Poisson, so short runs are dominated by variance — see §4.
3. **Record, per run:** accepted shares (from the *pool dashboard*, not the miner),
   rejected and stale shares, wall-clock duration, the miner's own reported sol/s (this
   is the number under test), and GPU temperature and clocks at the end.
4. **Compare accepted shares per hour.** Trust the pool's count over either miner's.

---

## 4. How long is long enough

Accepted shares are Poisson, so the relative standard error on the rate is `1/√N`:

| Shares observed | ±1σ on the rate |
|---|---|
| 25 | 20 % |
| 100 | 10 % |
| 400 | 5 % |
| 1000 | 3 % |

To resolve a **10 % difference at 2σ you need roughly 400 accepted shares per miner.** At
typical pool difficulty that is hours, not minutes.

If a full run is impractical, **the honest reporting is a confidence interval, not a point
estimate.** A 30-minute run yielding 40 shares each cannot distinguish 48 from 53 sol/s
and must not be presented as if it can.

---

## 4b. Comparing two miners at the same board settings

`benchmarks/compare_power.sh` runs MXBM and lolMiner against each other at identical
settings. Four decisions in it are what make the comparison mean anything, and each is a
way it could have been wrong instead:

- **Settings are applied externally, with `nvidia-smi`, identically to both.** Using each
  miner's own `--pl`/`--cclk` would put their overclock implementations into a
  measurement that is supposed to be about their kernels.
- **Power is sampled from NVML for both, never read from a miner's own report.**
  lolMiner's statistics block averages in its ramp-up: it reads ~215 W on a one-minute
  run where sampling that same run gives 235.
- **The two miners alternate order at every point**, so a thermal trend across a long
  sweep cannot land on one of them systematically.
- **lolMiner's `--benchmark` is a fixed ~61 s run** — established by giving it a 240 s
  ceiling and watching it stop at 61 with exit 0, not assumed. So both miners run 60 s
  and each point is repeated, rather than one long run each.

```sh
LIMITS="200 220 240 285" REPEATS=2 benchmarks/compare_power.sh
LIMITS="220 240" CCLKS="* 2100 2200" benchmarks/compare_power.sh   # a grid
```

`LIMITS` × `CCLKS` × `MCLKS` sweeps as a full grid; `*` in a clock list leaves that clock
to the driver. Everything is restored on exit including on Ctrl+C, and the clocks are
unlocked even when the run is interrupted — a card left locked outlives the script and
silently caps whatever runs next.

This still does not make the sol/s columns comparable *between* miners, for the reason in
§1. The watts are comparable, because one instrument measured both.

---

## 5. Gotchas

- **Do not compare across pools.** Difficulty and share accounting differ.
- **Do not run the miners concurrently, and check that nothing ELSE is on the card.**
  They contend for GPU and memory bandwidth. This is not a theoretical caution: during
  this project's own head-to-head sweep, a benchmark process left running by accident
  (started without `--benchmark-seconds`, so it never exited) sat on the card for 24
  minutes and **halved both miners' throughput** at one power point — 175 W read 19.8 and
  24.5 sol/s where a clean re-run measured 44.0 and 52.0. The signature is a point that
  contradicts its neighbours *while the core clock rises*, and it looks like a real
  finding until you check `nvidia-smi --query-compute-apps`. Do that before, and after.
- **A re-measurement must REPLACE what it re-measures, never average with it.** The same
  episode produced a summary that took the median of the contaminated run and its clean
  replacement together, reporting 31.75 sol/s — a number neither run measured and which
  sat exactly between a right answer and a wrong one.
- **Count stale and rejected shares separately.** A miner that submits faster but staler
  is not better; net accepted is what pays.
- **Restart between runs.** Persistent buffer reuse makes solve #1 slower than steady
  state; a short run is penalised for it.
- **Watch for thermal throttling.** If clocks at the end differ from clocks at the start,
  the comparison is invalid.
- **State which definition of "solution" the reported figure uses.** If the other miner's
  definition is unknown, say so rather than implying the numbers are commensurable.

---

## 6. Reading the outcome

| Result | What it means |
|---|---|
| Their shares/hour clearly above ours | The gap is real; the remaining work is engineering. |
| Within noise of each other | The reported-sol/s gap was a **counting artefact** and the two miners are at parity. |
| Ours above theirs | We are ahead, and the reported figures understate it. |

The middle outcome is entirely plausible given the 2.29-vs-1.95 finding above, which is
why this protocol is worth running before drawing conclusions from headline sol/s — in
either direction, including our own favour.

---

## References

- [performance.md](performance.md) — the measured optimization history and the hardware
  limits that bound it.
- [overclocking.md](overclocking.md) — why OC must be excluded from a like-for-like run.
- [\[bh2\]][bh2] Wilke Trei, *BeamHash II Specification*, 17 June 2019 — §2.4 tabulates
  solutions per iteration across the EquihashR family.
- [beamhash-iii.md](beamhash-iii.md) — why a run yields ~2 solutions rather than exactly
  one, and what a "survivor" is.

[bh2]: https://docs.beam.mw/BeamHashII.pdf
