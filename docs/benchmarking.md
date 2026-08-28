# Benchmarking and comparing miners

> Looking for the **numbers**? They are in **[benchmarks.md](benchmarks.md)**. This page is
> the *method* behind them.

Reported `sol/s` is **implementation-defined** — two miners doing identical work can differ
by 17 % legitimately, because they count different things. This page covers how MXBM
measures itself and the protocol that settles a cross-miner comparison without
trusting either miner's counters.

---

## 1. Why reported sol/s is not comparable

A BeamHash III run over `2^25` elements yields **about two solutions on average** — a
property of the Wagner parameterization, not of any implementation ([bh2] measured
1.92–2.03 across the EquihashR family). But "a solution" admits two readings, and MXBM
has measured both on its own pipeline:

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

## 2. Measuring your own card

```sh
mxbm --report                       # everything below, as one paste-ready block
mxbm --benchmark BEAM-III --benchmark-seconds 120     # throughput alone
sudo mxbm --tune                    # the power/speed curve and a --pl recommendation
```

Run `--report`. It benchmarks the card through the same solve path mining drives, samples telemetry, measures the power/speed curve (or reuses the one this binary
already measured), and prints the lot as markdown. It also records the three conditions
that decide whether a figure means anything: whether a display was on the card, whether
another process was, and what the clocks were limited by.

A single sample is not meaningful — quote a median of at least 5, and prefer a spread.

**The reference rig's own protocol** — the locked-clock pin every published MXBM figure
is taken at, and how small a difference this rig can resolve — is in
[performance-research.md](performance-research.md#measuring-the-locked-clock-pin-and-what-this-rig-can-resolve).
That is the method behind [performance.md](performance.md), not something a contributor
needs to reproduce.

---

## 3. Comparing against another miner

**Accepted pool shares are not implementation-defined.** The pool verifies every share, so
accepted shares over the same wall-clock time on the same hardware settles the question
without trusting either miner. Run both on the same GPU, pool, wallet and worker-name
convention, back to back — never concurrently.

1. **Fix the variables.** No other GPU load; same power limit and clocks, recorded. Use
   stock clocks — see [overclocking.md](overclocking.md). This is a rule, not a
   preference: **a memory overclock does not move two miners equally.** A memory OC
   flatters whichever design sits closer to its DRAM roofline, and a core OC flatters
   the other.
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
and lolMiner's `--benchmark` is a fixed ~61 s run, measured, so both run 60 s and points
repeat.

```sh
LIMITS="200 220 240 285" REPEATS=2 benchmarks/compare_power.sh
LIMITS="220 240" CCLKS="* 2100 2200" benchmarks/compare_power.sh   # a grid
```

`LIMITS` × `CCLKS` × `MCLKS` sweeps as a grid; `*` leaves a clock to the driver.
Everything is restored on exit including Ctrl+C — a card left locked outlives the script.
This does not make the sol/s columns comparable between miners (§1); the watts are, because
one instrument measured both.

**A swept column is provisional until one of its points is bracketed.** `power_sweep.sh`
gives one unrepeated run per cap: the right instrument for a curve's *shape*, the wrong
one for any single number in it. Two sweeps of the same binary have disagreed by up to
~5 %, in both directions, and the disagreement grows toward the low caps
([the write-ups](performance-research.md#a-swept-cap-column-that-did-not-reproduce-and-the-five-explanations-that-were-not-it)).
So: reproduce at least one cap with an interleaved A/B before publishing, **never treat a
difference between two sweeps as evidence about a kernel change**, and take a head-to-head
cap table **from one session** — columns measured weeks apart inherit the whole term at
exactly the caps where the margins are a few percent.

---

## 4. Gotchas

- **Check nothing else is on the card, before *and* after.** A stray process halves both
  miners' throughput. The signature is a point contradicting its neighbours *while the
  core clock rises*, and it looks like a finding until you check
  `nvidia-smi --query-compute-apps`.
- **A re-measurement REPLACES what it re-measures, never averages with it.**
- **Do not compare across pools**; difficulty and share accounting differ.
- **Count stale and rejected separately.** Submitting faster but staler is not better.
- **Restart between runs** — solve #1 pays one-off allocation.
- **If end clocks differ from start clocks, the comparison is invalid.**
- **State which definition of "solution" a figure uses**, or say it is unknown.

---

## 5. Reading the outcome

| Result | What it means |
|---|---|
| Their shares/hour clearly above ours | The gap is real; the rest is engineering |
| Within noise | The reported-sol/s gap was a **counting artefact**; parity |
| Ours above theirs | We are ahead, and the reported figures understate it |

The middle outcome is entirely plausible given the 2.29-vs-1.95 spread. Run the protocol
before drawing conclusions from headline sol/s in either direction, including ours.

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
