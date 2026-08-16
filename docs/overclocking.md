# Overclocking — design decisions

The six knobs, the design they follow, and what to set. The measurements behind every
claim here — NVML probe tables, competitor verification, the sweep evidence — are in
`docs-internal/OC_RESEARCH.md` (in-repo research notes, not shipped).

`--pl` is the knob whose value is [measured](performance.md#both-miners-under-the-same-cap)
rather than assumed: the card runs pinned at its limit in every kernel, so the limit picks
the operating point outright.

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

Three measured facts that decide what to set:

- **A rig that pays for electricity should cap.** ~220 W is the reference card's knee;
  `--tune` finds your card's own ([the curves](performance.md#power-and-efficiency)).
- **Below ~170 W, pair the cap with the card's low memory rung** (`--mclk 5001` on
  GDDR6X): worth −8.5 to −14.4 % ms/solve and the efficiency record. At ~180 W and above
  the rung loses badly — strictly a low-cap pairing.
- **Undervolting (`--cclk` + `--coff`) does nothing under a power cap** — the governor
  outranks the lock, on either knob, at any offset
  ([measured](performance-research.md#undervolting-buys-nothing-under-a-power-cap--the-cap-outranks-both-knobs)).
  At stock it is worth ~1 % and sits close to the hang threshold; not recommended.
- **To raise memory, reach for `--moff` before `--mclk`** — they are different
  instruments, and only the second one competes with the core for watts. A locked P-state
  buys its clock with voltage; on a board already pinned at its power limit those watts
  come out of the core clock, which is a loss whenever more of the solve is core-bound
  than memory-bound. An offset shifts the V/F curve instead: same voltage, higher
  frequency. Note this is the opposite direction from the low-rung advice above —
  locking memory *down* under a low cap returns watts to the core, which is why it pays
  there.

  On the reference card the effect is large enough to invert the knob: it holds 10251 MHz
  under load against a reported maximum of 10501, so the headroom is real, and *locking*
  to that maximum still measures **+0.95 % slower** — while `--moff 1600` on the same card
  measures **−2.37 % and −2.00 % J/sol**, bracketed over 12 interleaved arms
  ([both](performance-research.md#locking-the-memory-clock-to-its-reported-maximum-costs-095--and-the-offset-is-the-knob-that-does-not)).
  **Both the sign and the size are properties of your card and your power limit** — a card
  not sitting at its limit has no such trade to make. `--tune` measures the cap and the low
  rung for you; the offset has no auto-sweep, so treat it as a manual experiment and change
  one knob at a time.

## Finding your own memory offset

Worth doing — on the reference card it is the largest single knob after the power limit —
but it is the one knob that can hang a card, so it is never applied for you.

**The wall does not announce itself.** A core offset that is too high fails loudly:
solutions stop verifying. A memory offset that is too high fails *quietly*, because GDDR6X
answers timing it cannot hold with link-level retries rather than with wrong data — so the
symptom is sol/s falling while everything still verifies. On the reference card the gain
tracked the clock almost exactly to +1400 and then fell to 80 % and then 71 % of it, with
no step ever going backwards. **Stop where the returns bend, not where they reverse**, and
back off from there.

`benchmarks/mem_offset_sweep.sh` walks the offsets with that gate built in, and
`benchmarks/mem_offset_ab.sh` brackets whichever one you settle on — a swept point is a
single sample and never a figure. Both refuse to run with another process on the card,
because a clock knob is device-global and a co-tenant would run at your clocks while
taking your solves.

Two things the benchmark cannot tell you. **40 s is not a soak**: retry degradation grows
with heat, so run a candidate for hours before trusting it. And on many cards
`temperature.memory` reads `N/A` — the reference card included — so there is no junction
temperature to watch, which is its own argument for leaving headroom.

## `--tune`: measure this card's power curve, then `--pl auto`

`--tune` is the power sweep as a first-class mode. Three properties it holds:

- **The right loop.** Every point runs the exact `Engine` path mining runs, CPU-verified
  sol/s — not a pipeline replay.
- **The current build.** The sweep lives inside the shipping binary.
- **A drift gauge.** After both passes the first point is re-measured; past ±1.5 % the
  table is flagged. Un-gauged sweeps on the reference card have disagreed by 2 %/arm
  from heat-soak alone.

**Four passes**, ~25 min by default: coarse (6 points across the driver's band, high to
low, discarded warmup first), fine (~10 W steps around the knee), rung (the capped
points re-measured at the card's low memory rung), and efficiency-refining (brackets the
best sol/s-per-watt point found, on whichever memory clock it sits).

The rung candidate comes from the driver's own supported-clock list — the largest clock
below 70 % of the card's maximum, rejected under 20 % of it; nothing is inherited from
the reference card's 5001. Every rung arm verifies from telemetry that the clock
actually *held*: a refused rung prints REFUSED and is dropped, never counted as "does
not pay". The verdict adds "below ~X W, add `--mclk <rung>`", X interpolated from this
card's own sign change — recommended, never auto-applied.

The recommendation has two numbers: the **knee** — climbing from the lowest cap, the
highest one where each extra watt still returns at least `--tune-knee` sol/s (default
0.07/W, the one number that is a preference rather than a measurement) — and the
**best-efficiency point** (max sol/s per *measured* watt of draw, not per cap-watt).
Both are printed; the knee is stored per card (`~/.config/mxbm/tune.json`, keyed by
name@PCI) and `--pl auto` applies it on any later launch.

Root is needed exactly as for `--pl` — every NVML write is. The intended flow is `sudo
mxbm --tune` **once** (the store lands in the *invoking* user's config dir and is
chowned back), then daily runs with `--pl auto`. That still needs root to *apply* the
limit, so on a rig that mines unprivileged: read the recommendation once, then put
`sudo nvidia-smi -pl <knee>` in the boot sequence.

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
transfer rate, twice the memory clock — `--moff 200` moves the clock +100 MHz
(measured; details in `docs-internal/OC_RESEARCH.md`). The unit is
*not* silently halved on input: `--moff` means what it means to every other tool that
sets this knob, so a value copied from a community guide does the same thing here. The
statistics block reports the resulting clock, so the doubling is visible while mining.

## Decisions

### 1. Privilege model: the whole miner runs under sudo

Matching lolMiner. Chosen 2026-07-25. The reason it wins is **automatic
restore-on-exit**: a card left at +1200 memory after a crash corrupts whatever runs
next, and only an in-process owner of the settings can put them back on a signal. A
**privileged child process** (fork an OC owner, drop privileges before opening a socket)
keeps restore *and* keeps the TLS/JSON/stratum code unprivileged — strictly better, and
**deferred, not rejected**. The cost being accepted meanwhile: the stratum TLS client
and JSON parser run as root.

### 2. Flags: lolMiner-compatible names

Someone moving over should not have to relearn these. Multi-GPU comma/`*` syntax parsed
from the start so it never becomes a breaking change. Two details settled in the
implementation:

- **A lock and an offset are undone differently.** An offset is restored by writing the
  previous value back; a lock is undone by an explicit reset call, because it took clock
  management away from the driver. Getting that backwards leaves a card pinned after
  exit; `test_overclock` pins it.
- **The fan is restored to AUTO**, not to the percentage read at startup — that
  percentage was chosen for a cold idle card.

### 3. When OC is requested but cannot be applied

**Warn loudly and continue at stock**, and the message names the cause (`insufficient
permission — re-run under sudo`). A miner that failed to apply OC still mines correctly,
just slower: degrading is acceptable; being quiet about it is not.

### 4. Restore stock settings on exit, by default

Including on `SIGINT`/`SIGTERM`. `--no-oc-reset` opts out. Restore is idempotent and
does not require a clean exit.

### 5. Clamps: use the band the driver reports

`nvmlDeviceGetPowerManagementLimitConstraints` reports the card's own permitted band, so
MXBM clamps `--pl` to that and *says* it clamped — a clamped setting that looked applied
would misattribute every measurement taken after it. The CLI checks only syntax; watts
are validated at apply time, where the installed card is visible. The clock offsets have
no such clamp: the driver's ranges there are register width, not recommendations.

## Community OC settings

hashrate.no lists `--coff 300 --cclk 2205 --moff 2000 --pl 300` for this card — tuned
for lolMiner's kernels. Do not adopt it wholesale: the undervolt half is a measured null
under any cap, `--moff 2000` is +1000 MHz of *actual clock* (past the driver's ceiling),
and a locked 2600 MHz already costs MXBM 2.6 % of throughput, so 2205 needs measuring,
not trusting. One knob at a time; the knob-by-knob analysis is in
`docs-internal/OC_RESEARCH.md`.

## Stability canaries

Two knobs fail in two different ways, so watch two different signals.

**Core instability → CPU-verify rejections.** Every candidate goes through
`bh3::is_valid_solution` before submission, and the healthy baseline is **zero
rejections** (0 of 3,935 across both backends), so *any* rejection means the GPU is
computing wrong answers — caught locally, before the pool sees a bad share.
`MXBM_VERIFY_STATS=1` prints the tally at exit; this is what gates every offset sweep.

**Memory instability → sol/s falling as the offset rises.** This canary will *not* fire
on verify failures: Ada's GDDR6X uses link-level ECC with retry, so pushing memory too
far eats bandwidth rather than corrupting results. The signature is the classic hashrate
cliff — throughput drops while everything still looks correct. Find the offset where
sol/s peaks and back off from it.
