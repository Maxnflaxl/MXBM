# Overclocking — design decisions

Status: **all knobs implemented — `--pl` 2026-07-25, `--cclk`/`--mclk`/`--coff`/`--moff`/`--fan`
2026-07-28.** Everything below is the design they follow. The "To determine" section is
what remains *measured-unknown*, and the memory-offset unit in particular is still open:
the code reports the offset in the units the user typed and logs the resulting clock, so
the ambiguity is visible rather than hidden, but it is not resolved.

`--pl` was taken first because it is the only knob whose value is
[measured](performance.md#both-miners-under-the-same-cap) rather than assumed: the card
runs pinned at its limit in every kernel, so the limit picks the operating point outright.
The clock offsets remain hypotheses — see "Whether the community's recommended OC is right
*for MXBM*".

The head-to-head sweep also gave `--pl` a second job it was not designed for: it is the
instrument that found MXBM losing **570 MHz of core clock to lolMiner at a 180 W cap**,
against 60 MHz at stock. Whatever the clock knobs eventually do, that gap is the thing
they would be working against.

```
--pl W               board power limit in watts, per GPU ("240", "240,*,260"; * skips)
--cclk MHz           lock the core clock        --coff MHz  shift its V/F curve (signed)
--mclk MHz           lock the memory clock      --moff MHz  shift its V/F curve (signed)
--fan PCT            fan target, in percent
--no-oc-reset [0|1]  leave them applied at exit instead of restoring (default: off)
```

Config-file keys `PL`, `CCLK`, `MCLK`, `COFF`, `MOFF`, `FAN` and `NO_OC_RESET` set the
same things, and each also accepts a JSON array (`"PL": [220, "*", 260]`). Applied after
device enumeration and **before the solver is constructed** (2026-07-31; previously after),
so a benchmark measures the same operating point mining will use — and so anything that
selects on the observed limit reads the value `--pl` just set.

## The power-limit geometry policy — built, verified, and currently disarmed

The machinery for selecting the solver's bucket geometry from the board power limit
exists and is verified end to end: the order is **apply, then observe, then size** —
`--pl` lands before the solver is constructed and the selection reads the limit off the
card afterwards, so a cap applied outside MXBM (`nvidia-smi -pl` before launch — the
only route on a rig that doesn't run MXBM as root) counts exactly the same as one
`--pl` applied, and a `--pl` that fails without root cannot mis-select. Selection,
when armed, happens once per run (a geometry switch is a multi-GiB reallocation); a
cap that later crosses the threshold gets a one-line restart notice, never a mid-run
re-select. All of this was verified live on 2026-07-31, including the notice.

**The policy is disarmed** (`kRbLowPowerW == 0` in `rowbucket_geom.h`): the eco sweep's
measured prize for (17,0) below ~190 W failed same-day reproduction — a wash in the
miner loop at 160 W and 100 W, and on a cooled card the sweep's own binary read (17,0)
**+1.8 % worse** at the floor while its base arm reproduced to 0.2 ms (the
[eco-sweep addenda](performance-research.md#the-eco-sweep-170-crosses-over-below-190-w-r2_full-never-does)).
Holding (17,0) at stock costs ~23 % in the miner loop, so arming a selection with no
reproducible win was the wrong trade. Whatever next claims the low band — a re-measured
crossover, or the Tier-2 eco pipeline — re-arms it by setting that one constant.
`MXBM_BB=17` still forces the geometry by hand.

## `--tune`: measure this card's power curve, then `--pl auto`

The head-to-head sweeps that produced the tables below were bash scripts around
`nvidia-smi`; `--tune` is that measurement as a first-class mode. Three properties it
holds:

- **The right loop.** Every point runs `run_benchmark()` — the exact `Engine` path mining
  runs, CPU-verified sol/s — not a pipeline replay. A curve measured in any other loop is
  a curve of a different program.
- **The current build.** The sweep lives inside the shipping binary; there is no separate
  bench binary to go stale.
- **A drift gauge.** After both passes the first point is measured again, and past ±1.5 %
  the table is flagged as carrying that uncertainty. Un-gauged same-day sweeps on this
  card have disagreed by 2 %/arm from heat-soak alone.

**Four passes**, ~25 min by default:

1. **Coarse** — 6 points across the band the *driver* reports, 60 s each, high to low,
   discarded warmup first. Locates the knee's neighbourhood.
2. **Fine** — ~10 W steps filling the two coarse intervals touching the knee. The step
   widens only if the bracket would exceed 8 points; refinement bounds the tail, it never
   restarts the sweep.
3. **Rung** — re-measures the coarse caps at or below the knee at the card's low memory
   rung.
4. **Efficiency-refining** — brackets the best sol/s-per-watt point found so far, on
   whichever memory clock that point sits. Without it the coarse spacing can hide the true
   optimum between two arms; that is how the reference card's first full run missed its
   160 W + rung record.

The rung candidate comes from the driver's own supported-clock list — the largest clock
below 70 % of the card's maximum, rejected under 20 % of it. Nothing is inherited from the
reference card's 5001. Every rung arm verifies from telemetry that the clock actually
*held*: a refused rung prints REFUSED and is dropped, never counted as "does not pay".

The verdict adds "below ~X W, add `--mclk <rung>`", X interpolated from this card's own
sign change, and `--pl auto` reminds about it when the resolved cap is inside the paying
band — recommended, never auto-applied. `--tune-caps` runs the coarse pass only, and a
given `--mclk` disables the rung pass: chosen settings stand.

**Capped below ~170 W: pair the cap with `--mclk 5001`** (measured 2026-07-31, the
largest low-band lever on record: −8.5 % ms/solve at 160 W growing to −14.4 % at 120,
and the card's efficiency record at 160 W + rung — see
[performance.md](performance.md#below-stock-the-other-rung-pays-85-to-144--under-caps-below-173-w-and-a-new-efficiency-record)).
The mirror warning holds: at ~180 W and above the rung loses, +39 % at stock, because
the solver hits its own DRAM roofline at the rung's bandwidth. Strictly a low-cap
pairing, applied and restored like every other knob here.

The recommendation has two numbers: the **knee** — climbing from the lowest cap, the
highest one where each extra watt still returns at least `--tune-knee` sol/s (default
0.07/W, the one number that is a preference rather than a measurement) — and the
**best-efficiency point** (max sol/s per *measured* watt of draw, not per cap-watt,
because a core-limited card draws under its limit at stock). Both are printed;
the knee is stored per card (`~/.config/mxbm/tune.json`, keyed by name@PCI) and
`--pl auto` applies it on any later launch.

Root is needed exactly as for `--pl` — every NVML write is. The intended flow is `sudo
mxbm --tune` **once** (~8 min; the store lands in the *invoking* user's config dir, not
root's, and is chowned back), then daily runs with `--pl auto`. That still needs root to
*apply* the limit, so on a rig that mines unprivileged: read the recommendation once, then
put `sudo nvidia-smi -pl <knee>` in the boot sequence.

`--tune` refuses a simultaneous `--pl` — it drives the limit itself — but allows the other
OC knobs, so an undervolted card's curve can be tuned. Ctrl+C aborts and restores the
limit that was on the card, as does every completed sweep. Re-run after driver updates,
cooling changes, or a season; `--pl auto` prints the measurement date so a stale tune is
visible.

**Apply order is fixed**: power limit, core offset, memory offset, locked core clock,
locked memory clock, fan. The V/F curve is shaped before anything is pinned onto it, and
the fan is set last, against the thermal load the rest implies. **Restore is the exact
reverse**, so the fan returns to the driver's curve while the clocks are still coming
down, and the power limit is put back last — nothing is ever unlocked into a clock the
old limit would not have permitted.

Everything in the "Verified" sections below was measured on this machine on 2026-07-25,
not taken from documentation or recollection. Everything in "To determine" is explicitly
*not* known yet and must be measured before it is relied on.

Reference card: RTX 4070 Ti SUPER (Ada, sm_89), driver 610.43.03.

---

## Verified: what the hardware allows

Probed by dlopening `libnvidia-ml.so.1` and reading NVML directly.

| Knob | NVML call | Current | Driver-permitted range |
|---|---|---|---|
| Core clock offset | `nvmlDeviceSetGpcClkVfOffset` | +0 MHz | −1000 … +1000 |
| Memory clock offset | `nvmlDeviceSetMemClkVfOffset` | +0 MHz | −2000 … +6000 |
| Locked core clock | `nvmlDeviceSetGpuLockedClocks` | unlocked | up to 3150 MHz max |
| Locked memory clock | `nvmlDeviceSetMemoryLockedClocks` | unlocked | up to 10501 MHz max |
| Power limit | `nvmlDeviceSetPowerManagementLimit` | 285 W | 100 … 366 (default 285) |
| Fan (2 fans) | `nvmlDeviceSetFanSpeed_v2` | auto | 30 … 100 % |

The permitted range is **register width, not a recommendation**. `+6000` on memory is
not a stable setting on any card; the driver simply does not police it.

Observed under MXBM load (from the stats table): core 2670 MHz, memory 10251 MHz,
284 W, 63 °C, fan 50 %.

## Verified: every NVML write needs root

`nvmlDeviceSetPowerManagementLimit`, called as uid 1000 with the value the card already
had (a true no-op), returns `NVML_ERROR_NO_PERMISSION` (rc=4). Reads are unprivileged;
writes are not.

## Verified: `--pl` end to end, under sudo

Run 2026-07-25 on the reference card. Each case is one the implementation could plausibly
get wrong, and the card was read back with `nvidia-smi` afterwards:

| case | console | result |
|---|---|---|
| `--pl 220` | `Power limit: 220 W (was 285 W, device allows 100-366 W)` | 53.8 sol/s at 220 W — matches the sweep's 53.8 exactly |
| `--pl 50` | `50 W requested, applied 100 W (device allows 100-366 W)` | clamped to the driver's floor and said so; 16.3 sol/s |
| `--pl 220`, then Ctrl+C | — | card read back at **285 W**: restore fires on signal |
| unprivileged `--pl 240` | `not applied: insufficient permission - re-run under sudo` | mined on at 285 W; card untouched |

The live statistics block reports the applied limit and its efficiency directly
(`Power 220`, `Eff. 0.243 sol/s/W`), so the setting is visible while mining rather than
only at startup.

At the clamped floor of 100 W throughput collapses to 16.3 sol/s — 0.163 sol/s/W, far
below the 0.253 peak at 200 W. That is only a 48-solve sample and not quotable as a curve
point, but it is the same story the sweep tells: there is a floor, and undervolting past
it costs efficiency rather than buying it.

## Verified: what lolMiner actually does

Checked against the installed lolMiner 1.98a binary, not from documentation.

Its `--help` section is titled **"Overclock (Experimental)"** and offers:

```
--cclk   core clock            --coff  core clock offset
--mclk   memory clock          --moff  memory clock offset
--pl     power limit           --fan   fan target
--no-oc-reset [=arg(=1)] (=0)  disable reset of GPU settings at miner exit
```

All are per-GPU, comma-separated, with `*` to skip a GPU.

Run as a normal user with `--pl 285 --moff 0`, it prints:

```
Applying overclock settings...
GPU 0: memory clock offset 0 not applied (insufficient user rights).
GPU 0: power limit 285 not applied.
Start Benchmark...
```

So lolMiner: applies OC **in-process**, **requires root**, applies it **after device
enumeration and before mining starts**, and **warns and continues at stock** rather than
refusing to start. `--no-oc-reset` defaulting to `0` means it **restores stock settings
at exit by default**.

---

## Decisions

### 1. Privilege model: the whole miner runs under sudo

Matching lolMiner. Chosen 2026-07-25.

The reason it wins is **automatic restore-on-exit**. A card left at +1200 memory after a
crash corrupts whatever runs next, and only an in-process owner of the settings can put
them back on a signal. The alternatives were considered and rejected *for now*:

- *Privileged one-shot* (`sudo mxbm --oc-apply`, then mine unprivileged) — keeps root
  away from the network code, but gives up restore-on-exit, which is the property that
  matters most.
- *setuid helper* — permanent attack surface for a feature used interactively.
- **Privileged child process** — MXBM starts as root, forks a small OC-owner process,
  then drops privileges in the parent *before it opens a socket*. Keeps restore-on-exit
  **and** keeps the TLS/JSON/stratum code unprivileged. This is strictly better on both
  axes and is **deferred, not rejected** — revisit once the basic feature works.

The cost being accepted: the stratum TLS client and JSON parser run as root.

### 2. Flags: lolMiner-compatible names

`--cclk`, `--mclk`, `--coff`, `--moff`, `--pl`, `--fan`, and `--no-oc-reset`. MXBM
already mirrors lolMiner's CLI surface elsewhere; someone moving over should not have to
relearn these. Single-GPU for now, but parse the comma/`*` syntax from the start so
multi-GPU does not become a breaking change.

**Implemented as designed.** Two details the implementation had to settle that this
document did not:

- **A lock and an offset are undone differently.** An offset is restored by writing the
  previous value back; a lock has no previous value to write, because it took clock
  management away from the driver, so it is undone by an explicit reset call. Getting
  that backwards leaves a card pinned after exit, which is the failure restore exists to
  prevent, and it is pinned in `test_overclock`.
- **The fan is restored to AUTO, not to the percentage it read at startup.** Writing back
  the observed percentage would leave the fan fixed at a speed chosen for a cold idle
  card, which on a card that later gets hot is a way to cook it.

### 3. Behaviour when OC is requested but cannot be applied

**Warn loudly and continue at stock**, like lolMiner — but the message must name the
cause (`insufficient permission — re-run under sudo`), not just report failure.

Note the tension with `HW_REQUIREMENTS.md` "Known limitation #2", which argues a
silently-degraded configuration is a bug. The distinction: a partial *search* mines
nothing while looking healthy, whereas a miner that failed to apply OC still mines
correctly, just slower. Degrading is acceptable; being quiet about it is not.

### 4. Restore stock settings on exit, by default

Including on `SIGINT`/`SIGTERM`. `--no-oc-reset` opts out, matching lolMiner's default
of `0`. Restore must be idempotent and must not itself require the miner to have exited
cleanly.

### 5. Clamps — settled for `--pl`: use the band the driver reports

The tension was that any constant we pick is a guess about someone else's silicon. For
the power limit there is no need to guess: `nvmlDeviceGetPowerManagementLimitConstraints`
reports the card's own permitted band (**100–366 W** on the reference card, default 285),
so MXBM clamps to that and *says* it clamped. A value outside the band is reported at the
value actually applied, never silently accepted — a clamped setting that looked applied
would misattribute every measurement taken after it.

The CLI checks only the *syntax* of `--pl`, since the legal range is a property of the
installed card and the parser cannot see it. Watts are validated at apply time.

This does not settle clamps for the clock offsets, where the driver's `+6000` memory
range genuinely is register width rather than a recommendation.

---

## To determine — must be measured, do not assume

### The memory offset unit

`nvidia-settings` exposes memory offsets as `GPUMemoryTransferRateOffset`, in MHz of
**transfer rate**, which is 2× the memory clock. It is **not confirmed** that NVML's
`nvmlDeviceSetMemClkVfOffset` uses the same convention. So `--moff 2000` may mean
+2000 MHz or +1000 MHz of actual clock.

**Test:** apply a known offset, then read `nvmlDeviceGetClockInfo(NVML_CLOCK_MEM)` and
compare against the 10251 MHz baseline (10501 max). One measurement settles it. Report
the offset in the units the user typed, but log the resulting clock so it is unambiguous.

### Whether the community's recommended OC is right *for MXBM*

hashrate.no lists `--coff 300 --cclk 2205 --moff 2000 --pl 300` for this card.

The shape is coherent: `--cclk` plus `--coff` is the standard Linux undervolt idiom —
lock the core clock and shift the V/F curve up, so 2205 MHz runs at the voltage that
would otherwise deliver 1905 MHz.

**But those numbers are tuned for lolMiner's kernels, not ours**, and one of them looks
actively wrong for us:

- `--moff 2000` — highest expected value. MXBM is memory-bandwidth-bound; this is the
  lever on the bound resource.
- `--pl 300` — mild raise over the 285 W default (max 366). Cheap, low risk. Worth
  testing *downward* too: the `Eff. sol/s/W` column makes efficiency measurable, and a
  memory-bound kernel often loses little from a lower limit.
- `--cclk 2205` — **suspect for MXBM.** The card runs **2670 MHz** under our load today,
  so this is a ~465 MHz reduction. That is free for a fully memory-bound miner, but ncu
  measured entry/r1/r2 at *low* DRAM %peak — parts of our pipeline are not
  bandwidth-saturated, so core clock plausibly costs us throughput where it costs
  lolMiner nothing. Treat as a hypothesis to test, not a setting to adopt.

**Apply one knob at a time**, against the established methodology: median over 300
distinct nonces, ±4.1 % (1σ). A combined change cannot be attributed.

---

## Stability canaries

Two knobs fail in two different ways, so watch two different signals.

**Core instability → CPU-verify failure rate.** Every candidate goes through
`bh3::is_valid_solution` before submission. Measured baseline: 2.29 survivors/solve →
1.98 verified, so ~17 % fail. That 17 % is *algorithmic* — Wagner produces candidates
that do not satisfy the full constraint — and it is stable, which is what makes it a
usable reference. A rise means the GPU is computing wrong answers, and MXBM sees it
**before the pool does**, because bad solutions are filtered locally rather than
submitted. Worth surfacing as a stats column.

**Memory instability → sol/s falling as the offset rises.** This canary will *not* fire
on verify failures, because Ada's GDDR6X uses link-level ECC with retry: pushing memory
too far causes retries that eat bandwidth rather than corruption that changes results.
The signature is the classic hashrate cliff — throughput drops while everything still
looks correct. Find the offset where sol/s peaks and back off from it.

---

## Also worth knowing

`lolMiner --benchmark BEAM-III` runs the algorithm with **no pool and no wallet** and
prints `Average speed (15s): N sol/s`, in minutes.

**It does not settle MXBM vs lolMiner, and must not be used for that.** An earlier
revision of this document claimed it replaced the share-rate protocol in
`docs-internal/MINER_COMP.md`. That was wrong, and wrong for the exact reason that
protocol was written: reported sol/s is *implementation-defined*. MXBM counts solutions
that pass CPU verification (1.98/solve) and not raw survivors (2.29/solve) — a 17 %
difference — and we do not know which of the two lolMiner reports. Comparing its
self-reported benchmark number against our self-reported number is precisely the
untrustworthy comparison. Accepted pool shares remain the only arbiter, because the pool
verifies independently of either miner's counters.

Where the benchmark *is* useful is **A/B-ing OC settings within lolMiner**, the same way
`bench_rounds` A/Bs settings within MXBM. A metric only has to be self-consistent to
measure a delta; it has to be shared to measure a gap. So use it to find lolMiner's best
OC quickly, then run the share-rate protocol once, between two already-tuned miners.

Either way it must run on an **idle GPU**. A run taken while MXBM was mining reported
25.5–27.2 sol/s, which measures contention, not lolMiner.

lolMiner selects "BeamHash III **4G** (CUDA)" on this card — it fits the search in 4 GB
where MXBM needs 7.46 GiB. That is independent evidence for the streaming / in-place
layer-reuse route described in `HW_REQUIREMENTS.md` § "Memory efficiency".
