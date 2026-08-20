# Hardware Requirements

MXBM mines BeamHash III — an Equihash-style Wagner search on the ⟨144,5⟩ parameter shape;
see [beamhash-iii.md](beamhash/beamhash-iii.md). This document states what hardware is needed, why,
and where the current implementation falls short.

All figures are computed from the allocator in `src/gpu/round_pipeline.cpp` and
`src/gpu/budget.cpp`, and cross-checked against measurements on the reference card.
Throughput is quoted in **sol/s** (solutions per second), the figure miners and pools
report; BeamHash III yields 2.006 solutions per solve.

---

## Summary

| | Requirement |
|---|---|
| **GPU** | OpenCL 1.2+ device. A CUDA device (Ampere or newer) additionally unlocks the faster CUDA backend, which is the default when present. Developed and measured on NVIDIA (Ada, sm_89). |
| **VRAM** | **3 GB on either backend** — BeamHash III's own stated minimum (needs > 2.6 GiB *reported*). 8 GB and up get the fastest geometry; below that [the ladder](#the-vram-ladder) steps down, 3–110 % slower |
| **VRAM — what a full search occupies** | CUDA **6.17 GiB** at the fastest geometry, down to **1.90 GiB** at the coarsest; OpenCL **6.84** down to the same **1.90**. Both carry the [quad record](performance-research.md#the-quad-record-29--footprint-and-the-byte-prize-does-not-survive-re-derivation), the [dense-cap rungs](performance-research.md#the-overflow-arena-ported-36--for-a-404-gib-floor), the [implicit-bits record](performance-research.md#populations-are-pinned-at-225-and-the-occupancy-tail-prices-a-spill-arena) and the octo record; the 0.67 GiB between them is the three reference rows CUDA [replays instead of storing](performance-research.md#the-back-reference-rows-are-gone-recovery-replays-instead-203-ms-and-688-mib) |
| **VRAM — what BeamHash III is designed to need** | **3 GB** ([Beam docs](https://beam.mw/docs/mining)) — met: MXBM's floor is **1.90 GiB**, so a 3 GB card lands on the octo rungs with room |
| **Host RAM** | Modest; only survivor candidates (≤ 1024 × 128 B) are read back per solve. |
| **CPU** | Any; the CPU verifies candidates only (a few per solve). |

A card below the threshold does not mine slowly — it **refuses to start**, by design. See
[the partial-search caveat](#the-partial-search-caveat).

---

## Why BeamHash III needs a lot of memory

A BeamHash III solution is **32 indices drawn from the full `[0, 2^25)` space**. Wagner's
algorithm therefore has to hold a layer of 2^25 (≈ 33.5 M) elements, and each element
carries 448 bits (56 B) of work state — BeamHash III's element is roughly 3× fatter than
a classic Equihash element because its `apply_mix` re-derives the collision key from the
whole element every round.

The solver keeps two layers live at once (round *r*'s input and round *r+1*'s output), so
the work words alone are:

```
2 layers × 33.5 M elements × 56 B  =  3.61 GiB
```

On top of that come the leaf payloads needed to reconstruct a solution's 32 indices, and
per-bucket capacity slack so no bucket overflows. CUDA stores no back-reference rows at
all: recovery re-runs rounds 3 and 4 on the buckets along a survivor's path and reads the
leaves out of round 2's record. OpenCL and Metal still store all five rows, which is where
the two backends' figures separate.

### Current footprint (full 2^25 search)

| Path | Total | Per element | Largest single allocation |
|---|---|---|---|
| Row-bucket, CUDA default | **6.17 GiB** | 197 B | 2.90 GiB |
| Row-bucket, OpenCL default | **6.84 GiB** | 219 B | 3.27 GiB |
| Row-bucket, quad record — CUDA / OpenCL | **4.76 / 5.02 GiB** | 152 / 161 B | 2.90 GiB |
| Row-bucket, quad + dense caps — CUDA / OpenCL | **4.03 / 4.29 GiB** | 129 / 137 B | 2.42 GiB |
| Row-bucket, quad + dense caps + octo — both | **1.90 GiB** | 61 B | 1.35 GiB |
| Sort (fallback) | 8.25 GiB | 264 B | ~1.8 GiB |

The largest single allocation is listed because OpenCL caps it: NVIDIA reports
`CL_DEVICE_MAX_MEM_ALLOC_SIZE` as **¼ of VRAM**, against record arrays the ladder cannot
shrink below 2.45 GiB. A set that would bust the cap is allocated as **two bucket-halves**
instead (the kernels select by child bucket; measured free — pipeline and 60 s miner
medians identical to unsplit), so what decides a card's rung is total VRAM, on both
backends. CUDA has no equivalent ceiling.

---

## The VRAM ladder

Both backends pick their geometry from the same `rb_geometry_for()`
(`src/gpu/rowbucket_geom.cpp`): the first rung that fits, with headroom held back for what
the driver and this process's own context occupy — 640 MiB on CUDA, measured, and 1 GiB on
OpenCL — and, on OpenCL, the single-allocation cap. That is only the *offer*; construction
re-asks against free memory and steps down on a real allocation failure. Every rung is
KAT-gated with drops zero.

**The ladder has two axes, and it is ordered by measured time rather than by footprint** —
because the two disagree. Times are CUDA, one binary, 200 distinct nonces, 2026-07-28; the
build has since gained ~1 ms overall and the footprints predate the implicit-bits pack,
so the ratios are the point rather than the absolutes:

| rung | footprint | ms/solve | reached when |
|---|---|---|---|
| packed (16,1) | 7.20 GiB | **33.7** | ≥ 8.2 GiB free |
| packed (15,2) | 6.62 GiB | 36.0 | ≥ 7.7 |
| **quad (16,1)** | **5.02 GiB** | 38.4 | ≥ 6.1 |
| packed (14,3) | 6.24 GiB | 40.0 | only when a single-allocation ceiling binds |
| **quad (15,2)** | **4.65 GiB** | 40.7 | ≥ 5.7 |
| **quad (14,3)** | **4.40 GiB** | 44.8 | ≥ 5.4 |

The row that matters is the third. **quad (16,1) is both smaller and faster than packed
(14,3)** — 5.02 GiB at 38.4 ms against 6.24 at 40.0 — because a coarser geometry pays in
scatter locality what the quad record pays in re-derivation arithmetic, and the arithmetic
is cheaper. A ladder sorted by footprint would hand those cards the slower rung. packed
(14,3) sits below it on a *single*-allocation argument — 2.76 GiB against 2.90 — but that
argument does not survive the next row down: **quad (16,1) + dense caps is smaller on both
axes** (4.29 GiB total, 2.37 single) and faster, so no card that fails it can pass packed
(14,3). The row is unreachable wherever quad rungs are allowed, and exists for a backend
where they are not.

**Three mechanisms narrow every row of that table and add a row between each pair**, and
all three are on both backends:

- The **implicit-bits record** drops the key bits the bucket address already encodes, so
  round 2's 9th-word plane has no writer and set 0 is 8 u64 per slot instead of 9. It fits
  (16,1) and (17,0) only. Worth −0.36 GiB, and it is also ~3 % *faster* there. Deleting
  the plane is what lets a packed rung take dense caps at all — the pool had nowhere to
  live behind it — so (15,2) + dense caps is refused and the ladder steps past it.
- **Dense caps** (both backends) replace the per-bucket tail reserve with one overflow
  pool per record set: a bucket that fills appends to the pool instead of dropping, so the cap falls from
  mean + 8σ to mean + 2σ and ~22 % of the record slots go away. Zero drops becomes a
  pool-full check against 65,536 slots — measured spill is ~56 elements per solve. The
  mechanism costs a fixed few per cent, so it is a rung, never a default.
- The **octo record** does to round 3's output what the quad record does to
  round 2's:
  eight seed indices determine the element, so its six work words, its lead and its
  leftContrib are all derivable and only the leaves are stored. 32 B instead of 64, which
  halves the second record set. It also **deletes three of the five back-reference rows**,
  because a record that is its own eight leaves is what recovery was walking those rows to
  reach — and with those rows gone nothing reads a round-2 element's `gi` either, so that
  record drops from 3 u64 to 2. Together −2.14 GiB, for +35 %, which is why the octo rungs
  sit below everything.

| CUDA rung | footprint | ms/solve | runs when free VRAM is |
|---|---|---|---|
| packed (16,1) | **6.17 GiB** | 28.8 | ≥ 6.2 GiB |
| packed (16,1) + dense caps | **5.03 GiB** | 29.8 | ≥ 5.1 |
| quad (16,1) | 4.76 GiB | 34.1 | ≥ 4.8 |
| quad (16,1) + dense caps | **4.03 GiB** | 35.0 | ≥ 4.1 |
| quad (15,2) + dense caps | **3.87 GiB** | 37.6 | ≥ 3.9 |
| quad (14,3) + dense caps | **3.78 GiB** | 42.9 | ≥ 3.9 |
| quad (16,1) + dense caps + octo | **2.04 GiB** | 50.9 | ≥ 2.1 |
| quad (15,2) + dense caps + octo | **1.95 GiB** | 52.9 | ≥ 2.0 |
| **quad (14,3) + dense caps + octo** | **1.90 GiB** | 57.0 | ≥ 2.0 — the CUDA floor |

The whole column was re-measured in one sitting on 2026-08-16, at 40 s a rung with the
clocks released, so the rows are comparable with each other and not with the locked-clock
headline. Every row below the packed ones carries a
[w0 checkpoint](performance-research.md#the-w0-checkpoint-on-the-quad-record-the-record-holds-word-0-with-no-repacking):
the record a re-deriving round reads hands it the child's work word 0, so the rebuild
derives only the linear lane below it and runs no `apply_mix` at all. The quad rows carry
it on round 3's input; the octo rows on round 4's, since the 16 B round-2 record they read
instead has no slack to put word 0 in.

(The (15,2) and packed-(14,3) rows are still in the list for a backend without the quad
record; with it, a card that fits them fits a smaller and faster row first.)

**OpenCL now runs every record on the ladder**, and so reaches the same floor. It stores
all five reference rows where CUDA replays three of them away, which is the whole of the
difference above the octo rungs — and none of it on them, where only one row exists
either way. Its own times, measured on the reference card at a forced geometry, so they
are not comparable with the CUDA column above. The whole column was re-taken in one
sitting on 2026-08-17, after the [w0
checkpoint](performance-research.md#the-w0-checkpoint-reaches-opencls-quad-and-octo-records)
reached the two record boundaries it had not: **nine of the fourteen rungs moved** —
the six quad rows and the three octo ones — and the four byte-identical rows that did not
size the sitting's own offset at −0.73 %, which is what makes the column comparable with
the one before it.

| OpenCL rung | footprint | ms/solve | runs when usable VRAM is |
|---|---|---|---|
| packed (16,1) | **6.84 GiB** | 31.5 | ≥ 6.9 GiB |
| packed (16,1) + dense caps | **5.77 GiB** | 32.5 | ≥ 5.8 |
| packed (15,2) | 6.62 GiB | 34.9 | ≥ 6.7 |
| packed (15,2) + dense caps | — | — | refused: no pack at bb 15, so the pool has nowhere to live |
| quad (16,1) | 5.02 GiB | 36.5 | ≥ 5.1 |
| quad (16,1) + dense caps | **4.29 GiB** | 37.7 | ≥ 4.3 |
| packed (14,3) | 6.24 GiB | 38.5 | only when a single-allocation ceiling binds |
| quad (15,2) | 4.65 GiB | 37.8 | ≥ 4.7 |
| quad (15,2) + dense caps | **4.13 GiB** | 39.5 | ≥ 4.2 |
| quad (14,3) | 4.40 GiB | 41.4 | ≥ 4.5 |
| quad (14,3) + dense caps | **4.04 GiB** | 43.7 | ≥ 4.1 |
| quad (16,1) + dense caps + octo | **2.04 GiB** | 56.5 | ≥ 2.1 |
| quad (15,2) + dense caps + octo | **1.95 GiB** | 58.7 | ≥ 2.0 |
| **quad (14,3) + dense caps + octo** | **1.90 GiB** | 63.0 | ≥ 2.0 — the OpenCL floor |

The dense-cap rung is where the ladder's ordering earns itself: at 4.45 GiB a card takes
quad (16,1) + dense caps at 37.7 ms instead of quad (14,3) at 41.4, so it uses **less**
memory and runs **8.9 % faster**. An arena rung also turns speculative entry off — that
set has no pool region — so the delivered cost of the mechanism there is nearer 5 % than
the 3.6 % an interleaved A/B measures for the caps alone.

packed (14,3) at 38.5 ms sits above quad (15,2) at 37.8 and is **unreachable on this
backend**: quad (16,1) + dense caps, the row above it, is smaller on total *and* on the
single allocation *and* faster, so nothing can fail that row and reach this one. It is
there for a backend without the quad record.

The octo rungs now cost this backend what they cost CUDA — **2.00× the top rung against
CUDA's 1.98×** — because round 4's rebuild here carries the same w0 checkpoint, and with
it none of the fifteen `apply_mix` calls the eight-leaf tree used to run. That is worth
**−7.3 to −8.7 %** across the three rungs, and it is what closed a gap that was 1.9×
against 1.8× while only CUDA had the record.

**Two stages, two different numbers, and it matters which one is quoted.** A device is
*offered* when its TOTAL VRAM clears the floor plus the 640 MiB allowance — 2.53 GiB
reported, so a 3 GB card makes the list and a 2 GB one does not. Which rung it then
*runs* is decided against FREE memory at construction, which is usually a better answer
than the offer implied. The column above is the second number. Which card classes this
actually moves is [the table below](#what-each-card-class-gets) — not every row of a
ladder corresponds to a card that exists.

The quad record buys no speed and no watts at any power limit — that is
[measured, not assumed](performance-research.md#under-a-cap-the-effect-appears--and-the-trade-still-never-pays) —
so the ladder reaches for it only when no packed rung fits. It is purely what lets a
smaller card run at all. `MXBM_QUAD=0|1` forces the choice.

### What each card class gets

Both columns are the rung the ladder picks against the *reported* figure, with the whole
card idle. A desktop holding half a gigabyte steps a card down one row, which is why the
8 GB and 6 GB cases are called out below.

| Card | reported gmem / max_alloc | CUDA rung | OpenCL rung | OpenCL allocation |
|---|---|---|---|---|
| 16 GB (4070 Ti S) | 15.59 / 3.90 GiB | (16,1) | (16,1) | one buffer per set |
| 12 GB | 11.60 / 2.90 GiB | (16,1) | (16,1) | split into bucket-halves |
| 11 GB | 10.60 / 2.65 GiB | (16,1) | (16,1) | split |
| 10 GB | 9.70 / 2.42 GiB | (16,1) | (16,1) | split |
| 8 GB | 8.00 / 2.00 GiB | (16,1) | (16,1) | split |
| 6 GB | 5.70 / 1.43 GiB | (16,1) + dense caps | quad (16,1) + dense caps | split |
| 5 GB | 4.70 / 1.18 GiB | quad (16,1) + dense caps | quad (15,2) + dense caps | split |
| 4 GB | 3.90 / 0.98 GiB | quad (15,2) + dense caps | **quad (16,1) + d.c. + octo** | split |
| 3 GB | 2.93 / 0.73 GiB | **quad (16,1) + d.c. + octo** | **quad (16,1) + d.c. + octo** | split |
| 2 GB | 1.95 / 0.49 GiB | — not offered | — not offered | — |

The general effect is that every rung's requirement fell, so a card climbs one whenever it
was within 0.36 GiB of the next rung up (implicit bits) or 1.07 GiB (dense caps). Which
cards that is depends on how much of the card the desktop is holding, since construction
sizes against *free* memory — so the concrete cases are worth naming:

- **3 and 4 GB gain reach, on both backends.** 2.93 GiB clears the 1.90 GiB octo floor;
  before, nothing on the ladder fit them and neither device was offered at all. The
  reported figure is ~97.5 % of physical VRAM on this driver — `totalGlobalMem` reads
  15.598 GiB where nvidia-smi says 15.99 — and availability holds back 640 MiB of it.
- **8 GB gains speed when it is also driving a desktop.** Idle and headless it already
  cleared 7.20 GiB on OpenCL and ran (16,1); with a compositor holding ~0.5 GiB it did
  not, and stepped to (15,2) at 34.9 ms. 6.84 GiB clears that case too, at 31.5.
- **6 GB is unchanged at rest** on either backend: it was already on a dense-cap rung
  through the free-memory path. What it gains is the same desktop fallback the 8 GB row
  gains — one row down rather than two.

2 GB is out of reach, and is not a target — see [what is left](#what-is-left-below-it).

#### Simulated on the reference card, end to end

`--keepfree` hands the solver the memory budget a smaller card would have, so each class
can be run through the real `Engine` path — CPU-verified solutions and all — on hardware
we have. 20 s per point, headless, nothing else on the card, 2026-08-16:

| simulated card | usable | CUDA | OpenCL |
|---|---|---|---|
| 8 GB | 7.50 GiB | 28.7 ms / 69.7 sol/s | 32.3 / 61.8 |
| 6 GB | 5.60 GiB | 29.6 / 67.1 | 39.0 / 51.2 |
| 4 GB | 3.75 GiB | 50.4 / 40.4 | 61.7 / 32.9 |
| **3 GB** | **2.79 GiB** | **50.3 / 40.5** | **61.6 / 32.9** |
| 2 GB | 1.80 GiB | refuses, names the cause | refuses, names the cause |

Verified solutions per solve stayed in band at every point (1.94–2.04), so these are
solves that find solutions, not a hashrate. The 3 GB and 4 GB rows are the same rung —
(16,1) quad + dense caps + octo at 2.04 GiB — which is why they read the same.

What this does **not** simulate is the single-allocation ceiling: `--keepfree` shrinks the
budget, not `CL_DEVICE_MAX_MEM_ALLOC_SIZE`, and a real 3 GB card reports ~0.73 GiB there
and would split every record set into bucket-halves. `MXBM_SPLIT=1` forces that path, and
the 3 GB point measures identically with it on. Nor does it simulate a smaller SM count,
so read the times as *which rung runs*, not as what a 3 GB card would score.

The back-ref rows never split. On CUDA they no longer exist off the octo rungs; on OpenCL
they are 0.64 GiB above those rungs. On the octo rungs themselves both backends keep one
gi-indexed row pair and a survivor-sized one, 0.13 GiB, which is why the two floors are
the same number.

**The prediction is checked against the allocator, not trusted.** `rb_geometry_for()`
sizes against *total* VRAM and a desktop compositor can be holding a gigabyte of it, so
both backends step down on real allocation failure — and they walk the rung list rather
than the geometry, so a card that loses a packed rung to a busy desktop falls through onto
the quad ones instead of being refused.

Only the sort path still carries a flat per-element bound (284 B/element, ~10.4 GiB
reported), and no card class needs it: it is the fallback for a device the row-bucket
ladder cannot fit at all.

---

## The partial-search caveat

When VRAM is tight, `compute_budget` reduces `elems_per_round`, so the miner only
generates seed indices `[0, epr)` instead of the full `[0, 2^25)`. A solution is then
findable only if **all 32** of its indices happen to fall inside that range:

```
P(solution findable) = (epr / 2^25)^32

  epr = 0.83 × 2^25   →  0.3 %
  epr = 2^24 (half)   →  0.00000002 %
```

**A reduced search does not mine slower — it mines essentially nothing.** The miner would
appear to run normally while finding no shares, which is the worst possible failure mode,
so **both backends refuse to start instead**:

- OpenCL: `GpuSolver`'s constructor throws with the required and available GiB once
  `budget_can_find_solutions()` fails (`src/gpu/gpu_solver.cpp`).
- CUDA: `CudaSolver::available()` returns false unless *some* geometry on the ladder fits
  in total memory plus 1 GiB of headroom, so the backend is never selected
  (`src/gpu/cuda_solver.cu`). The seed layer is never reduced to make one fit — the ladder
  trades speed for footprint and nothing else.

`MXBM_ALLOW_PARTIAL_SEARCH=1` overrides the OpenCL refusal, for experiments only.

---

## Reference measurements

Reference card: **RTX 4070 Ti SUPER** (Ada, sm_89, 66 CUs, 16 GB, 48 KB LDS/workgroup).
Reported by OpenCL: `gmem = 15.59 GiB`, `max_alloc = 3.90 GiB`.

| | CUDA (default) | OpenCL (fallback) |
|---|---|---|
| Throughput | **69.20 sol/s** | 63.2 sol/s |
| End-to-end solve | **29.10 ms** | 31.8 ms |
| Board power | 275.5 W at the pin, near the card's 285 W limit; `sw_power_cap` intermittently active | — |
| Efficiency | **0.228 sol/s/W** stock, **0.275 at 220 W** (its optimum on the stock memory clock), matched but not beaten by **0.276** at 160 W with `--mclk 5001` | — |

*(CUDA column: the locked-clock pin, re-taken 2026-08-16 after the block-exit barrier and
singleton-free staging shipped, and the 2026-08-14 rung measurements. Stock efficiency is that pin's own
throughput over its own board draw. **The capped figures were not re-measured against
the new kernel** — under a cap the binding currency is different, so their sign there is
unknown rather than assumed.
The OpenCL column is its own 2026-08-17 measurement, after the implicit-bits record and
the w0-checkpoint pair record shipped there.)*

The two backends are within **1.10×** of each other, measured in the same session at
released clocks (CUDA 28.9 ms, OpenCL 31.8 ms, 2026-08-17, two 30 s arms each); the same
pair read 1.012× on 2026-08-02, before CUDA took several records OpenCL still lacks.
Cross-session figures carry a ~2.5 % band, so the same-session pair is the only comparison
that means anything; the column figures above are each backend's own controlled headline,
and the CUDA one is a locked-clock pin while the OpenCL one is not.

The card is power-limited, not thermally limited, in every kernel, so the board power
limit is the most valuable knob on it: **at 210 W the solver does 64.9 sol/s for 209.8 W,
against 69.95 at 284 W** — 93 % of the speed for 74 % of the power, and 210 W is
also the efficiency peak (**3.23 J/solution**). Below ~167 W the memory clock is the
second knob: `--mclk 5001` is worth +7 % at 160 W rising to +18 % at 100 W, and its
160 W point matches stock memory's 210 W on energy per solution (3.22 against 3.23 J)
at a third less throughput. Against an *equally capped* lolMiner, MXBM wins on
both speed and efficiency from ~177 W to the 285 W stock limit and again below ~104 W
on stock memory; in between it trails — worst at 160 W (−7.5 %) — and at each miner's
best configuration the residual is 2–3 % at 100–110 W and under 1 % at 155–177 W, with
MXBM ahead at 120–140 W. The full curves are in
[Both miners under the same cap](performance.md#both-miners-under-the-same-cap) and
[the 5001 memory rung](performance.md#below-stock-the-other-rung-pays-7-to-18--under-caps-below-165-w).

End-to-end is the headline figure — `solve()` including survivor readback, back-reference
recovery and CPU verification — as a median over 300 distinct nonces. BeamHash III yields
2.006 *verified* solutions per solve, pinned over 28,305 of them.
`./build/bench_rounds 20` reports the pipeline-only median instead, which is the controlled
number used to make optimization decisions.

See [performance.md](performance.md) for the measured state, and
[performance-research.md](performance-research.md) for the full optimization history.

---

## Known limitations

These are open issues in MXBM, not properties of BeamHash III.

### The design target is met

Beam's own mining documentation states:

> "Beam Hash III requires a graphics card with a minimum 3GB of memory."
> — <https://beam.mw/docs/mining>

That is the **algorithm's design target**, not a third-party miner's quirk: BeamHash III
was designed by Wilke Trei, who also writes lolMiner. MXBM's floor is **1.90 GiB on both
backends**, which clears the stated figure and the card class behind it with room — down from 2.4×
over, which is what the quad record, the survivor-sized back-ref row, the implicit-bits
record, the dense-cap rungs, the octo record, the reference rows it made dead and the
`gi` field those rows were the last reader of together bought. A 3 GB card reports
~2.93 GiB and the ladder lands it on the octo rungs. The replay that deleted the
reference rows is worth another 0.26 GiB on every rung above those and 0.45 more where the
implicit-bits record admits it, but it does not move the floor: the octo rungs had already
deleted three of the five rows by carrying eight leaves in the record.

#### What is left below it

At the floor the 1.90 GiB is two things and nothing else:

| | GiB | what it is |
|---|---|---|
| record sets | 1.64 | 36.6 M slots × (16 B round-2 record + 32 B round-3 record) |
| back-ref rows | 0.26 | one gi-indexed row pair plus a survivor-sized one — the octo rungs' round 4 is the one round still storing references |

**Both are now at what the design can carry.** Every record is its own seed indices plus
the key, with no field nothing reads: 6 u64 per slot, and the only two records that could
narrow further are round 4's output (which the terminal round needs whole) and the pair
record (which is already below the width set 1 is sized by). Three of the five reference
rows are gone. What is left needs a structural change rather than a narrowing:

- **The last reference rows off the card.** ~0.26 GiB, written near-sequentially by `gi`
  and read ~160 times per solve, which is the shape pinned host memory suits. The gate is
  the PCIe write rate against the solve time at the rung that would use it.
- **Streaming / in-place layer reuse**, below — the only lever that attacks the
  two-live-layers structure itself, and the only route under ~1.6 GiB.

**Streaming / in-place layer reuse** — writing round *r+1* into space freed by consumed
round-*r* buckets — remains open and is the only lever that attacks the two-live-layers
structure itself. It is a *reach* lever and only a reach lever: it writes
the same bytes to reused addresses, so the peak allocation falls and the traffic does not
move at all.

**Index-only storage with re-derivation now runs the whole pipeline.** Every round from
1 to 4 can take its input as seed indices and rebuild — 8 B, 16 B, 24 B and 32 B records
against 8, 72, 24 and 64 — and what stops that being the default is time, not information:
rebuild cost doubles per round while the record it replaces shrinks, so the quad record
costs +14 % and the octo record +38 %. Both are therefore rungs.

**Traffic is the separate quantity, and it is the efficiency lever.** A solve moves
**10.73 GB** of DRAM traffic, the compulsory minimum for these record widths to within the
1 % the last Nsight run measured — there is no waste left to reclaim, only records to
narrow. Round 5's input is the most recent to narrow, 16 B → 8. That the narrowing
pays is measured, not assumed: cutting round 2's record 72 B → 16 B, so the solve moves
16 % fewer bytes, raises the clock the card sustains at a fixed 285 W by **60 MHz**, with a
positive control at ±0 MHz ([details](performance-research.md#but-bytes-are-not-free-in-watts-and-under-a-cap-watts-are-clock-60-mhz)).
The prize is bounded, though — ≤ 92 MHz/GB across the whole solve — and every record is
already at `ceil(bits/64)`, so it is not a lever anything currently reaches.

Meanwhile the card runs pinned at its 285 W board limit in every kernel, so joules per
solution are set by how long a solve takes. See
["Power and efficiency"](performance.md#power-and-efficiency).
