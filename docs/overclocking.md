# Overclocking

The six knobs, and what to set.

Start with `--pl`. The card runs pinned at its limit in every kernel, so the limit picks
the operating point outright, and its value is
[measured](performance.md#both-miners-under-the-same-cap).

```
--pl W               board power limit in watts, per GPU ("240", "240,*,260"; * skips)
--cclk MHz           lock the core clock        --coff MHz  shift its V/F curve (signed)
--mclk MHz           lock the memory clock      --moff MHz  shift its V/F curve (signed)
--fan PCT            fan target, in percent
--no-oc-reset [0|1]  leave them applied at exit instead of restoring (default: off)
```

Config-file keys `PL`, `CCLK`, `MCLK`, `COFF`, `MOFF`, `FAN` and `NO_OC_RESET` set the
same things, and each also accepts a JSON array (`"PL": [220, "*", 260]`). Applied after
device enumeration and **before the solver is constructed**, so a benchmark measures the
same operating point mining will use.

Four measured facts decide what to set:

- **A rig that pays for electricity should cap.** `--pl 220` is the reference card's
  efficiency peak; `--tune` finds your card's own
  ([the curve](benchmarks.md#mxbm-power-curve)).
- **Below ~167 W, pair the cap with the card's low memory rung** (`--mclk 5001` on
  GDDR6X): worth +7 % sol/s at 160 W rising to +18 % at the 100 W floor. The crossover
  is ~167 W and above ~180 W the rung is a wall — strictly a low-cap pairing.
- **Undervolting (`--cclk` + `--coff`) does nothing under a power cap** — the governor
  outranks the lock, on either knob, at any offset
  ([measured](performance-research.md#undervolting-buys-nothing-under-a-power-cap--the-cap-outranks-both-knobs)).
  At stock it is worth ~1 % and sits close to the hang threshold; not recommended.
- **To raise memory, reach for `--moff` before `--mclk`.** A locked P-state buys its clock
  with voltage, and on a board pinned at its power limit those watts come out of the core
  clock; an offset shifts the V/F curve instead — same voltage, higher frequency. On the
  reference card that inverts the knob: locking to the driver's reported 10501 MHz maximum
  measures **0.95 % slower**, while `--moff 1600` measures **2.37 % faster and 2.00 %
  better J/sol**
  ([both](performance-research.md#locking-the-memory-clock-to-its-reported-maximum-costs-095--and-the-offset-is-the-knob-that-does-not)).
  **Both the sign and the size belong to your card and your power limit.** Note this is the
  opposite direction from the low-rung advice above: locking memory *down* under a low cap
  returns watts to the core, which is why it pays there.

## Finding your own memory offset

On the reference card this is the largest single knob after the power limit. It can also
hang a card, which is why nothing here applies it for you.

**The wall does not announce itself.** A core offset that is too high fails loudly:
solutions stop verifying. A memory offset that is too high fails *quietly* — GDDR6X
answers timing it cannot hold with link-level retries rather than wrong data, so the
symptom is sol/s falling while everything still verifies. On the reference card the gain
tracked the clock almost exactly to +1400 and then fell to 80 % and then 71 % of it, never
going backwards. **Stop where the returns bend, not where they reverse**, and back off.

`benchmarks/mem_offset_sweep.sh` walks the offsets with that gate built in, and
`benchmarks/mem_offset_ab.sh` brackets whichever one you settle on — a swept point is a
single sample, never a figure. Both refuse to run with another process on the card.

Two things a benchmark cannot tell you. **40 s is not a soak**: retry degradation grows
with heat, so run a candidate for hours before trusting it. And on many cards — the
reference one included — `temperature.memory` reads `N/A`, so there is no junction
temperature to watch. Leave headroom.

## `--tune`: measure this card's power curve, then `--pl auto`

`--tune` is the power sweep as a first-class mode, inside the shipping binary. Every
point runs the exact `Engine` path mining runs, CPU-verified sol/s, with the solver
rebuilt at each cap — so a cap is measured as a daily run at that cap would behave. After
the passes the first point is re-measured as a drift gauge; past ±1.5 % the table is
flagged, because un-gauged sweeps here have disagreed by 2 % per arm from heat-soak alone.

**Four passes**, ~25 min by default: coarse (6 points across the driver's band, high to
low, warmup discarded), fine (~10 W steps around the recommendation), rung (the capped
points re-measured at the card's low memory rung), and efficiency-refining (brackets the
best sol/s-per-watt point, on whichever memory clock it sits).

The rung candidate comes from the driver's own supported-clock list — the largest clock
below 70 % of the card's maximum, rejected under 20 % of it; nothing is inherited from
the reference card's 5001. Every rung arm verifies from telemetry that the clock actually
*held*: a refused rung prints REFUSED and is dropped, never counted as "does not pay".

The verdict has two numbers: the **recommended `--pl`** — climbing from the lowest cap,
the highest one where each extra watt still returns at least `--tune-min-gain` sol/s
(default 0.07/W, the one number that is a preference rather than a measurement) — and the
**best-efficiency point** (max sol/s per *measured* watt of draw, not per cap-watt). When
the rung paid it adds "below ~X W, add `--mclk <rung>`", X interpolated from this card's
own sign change — recommended, never auto-applied. The recommendation is stored per card
(`~/.config/mxbm/tune.json`, keyed by name@PCI) and `--pl auto` applies it later.

Root is needed exactly as for `--pl`. The intended flow is `sudo mxbm --tune` **once**
(the store lands in the *invoking* user's config dir), then daily runs with `--pl auto` —
which still needs root to apply the limit, so on a rig that mines unprivileged, read the
recommendation once and put `sudo nvidia-smi -pl <watts>` in the boot sequence.

`--tune` refuses a simultaneous `--pl` — it drives the limit itself — but allows the
other OC knobs. Ctrl+C aborts and restores the limit that was on the card, as does every
completed sweep. Re-run after driver updates, cooling changes, or a season; `--pl auto`
prints the measurement date so a stale tune is visible. `--tune-caps` runs the coarse
pass only, and a given `--mclk` disables the rung pass: chosen settings stand.

## Apply order, and restore

**Apply order is fixed**: power limit, core offset, memory offset, locked core clock,
locked memory clock, fan. The V/F curve is shaped before anything is pinned onto it, and
the fan is set last, against the thermal load the rest implies. **Restore is the exact
reverse**, so the fan returns to the driver's curve while the clocks are still coming
down, and the power limit is put back last — nothing is ever unlocked into a clock the
old limit would not have permitted.

## The memory offset is in transfer-rate MHz, so `--moff` is 2× the clock

NVML's memory offset follows the `GPUMemoryTransferRateOffset` convention: MHz of
transfer rate, twice the memory clock — `--moff 200` moves the clock +100 MHz (measured).
The unit is *not* silently halved on input, so a value copied from a community guide does
the same thing here as everywhere else. The statistics block reports the resulting clock,
so the doubling is visible while mining.

## Community OC settings

hashrate.no lists `--coff 300 --cclk 2205 --moff 2000 --pl 300` for this card — tuned
for lolMiner's kernels. Do not adopt it wholesale: the undervolt half is a measured null
under any cap, `--moff 2000` is +1000 MHz of *actual clock* (past the driver's ceiling),
and a locked 2600 MHz measures 0.5 % under the unlocked stock figure (27.00 against
26.9 ms), so 2205 needs measuring, not trusting. One knob at a time.

## Stability canaries

Two knobs fail in two different ways, so watch two different signals.

**Core instability → CPU-verify rejections.** Every candidate goes through
`bh3::is_valid_solution` before submission, and the healthy baseline is **zero rejections**
(0 of 3,935 across both backends), so *any* rejection means the GPU is computing wrong
answers — caught locally, before the pool sees a bad share. `MXBM_VERIFY_STATS=1` prints
the tally at exit.

**Memory instability → sol/s falling as the offset rises.** It will *not* show up as a
verify failure: GDDR6X uses link-level ECC with retry, so pushing memory too far eats
bandwidth rather than corrupting results. Find the offset where sol/s peaks, and back off.
