# Hardware Requirements

MXBM mines BeamHash III — an Equihash-style Wagner search on the ⟨144,5⟩ parameter shape;
see [beamhash-iii.md](beamhash/beamhash-iii.md). This document states what hardware is needed, why,
and where the current implementation falls short.

All figures are computed from the allocator in `src/gpu/round_pipeline.cpp` and
`src/gpu/budget.cpp`, and cross-checked against measurements on the reference card.
Throughput is quoted in **sol/s** (solutions per second), the figure miners and pools
report; BeamHash III yields ~1.9 solutions per solve.

---

## Summary

| | Requirement |
|---|---|
| **GPU** | OpenCL 1.2+ device. A CUDA device (Ampere or newer) additionally unlocks the faster CUDA backend, which is the default when present. Developed and measured on NVIDIA (Ada, sm_89). |
| **VRAM — either backend** | **6 GB** — needs > 5.7 GiB *reported*. 10 GB and up get the fastest geometry; below that [the ladder](#the-vram-ladder) steps down, 7–33 % slower |
| **VRAM — what a full search occupies** | **7.46 GiB** at the fastest geometry, down to **4.66 GiB** at the coarsest (either backend, with the [quad record](performance-research.md#the-quad-record-29--footprint-and-the-byte-prize-does-not-survive-re-derivation)) |
| **VRAM — what BeamHash III is designed to need** | **3 GB** ([Beam docs](https://beam.mw/docs/mining)) — MXBM is ~2.4× over |
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

On top of that come the leaf/back-reference payloads needed to reconstruct a solution's
32 indices, and per-bucket capacity slack so no bucket overflows.

### Current footprint (full 2^25 search)

| Path | Total | Per element | Largest single allocation |
|---|---|---|---|
| Row-bucket (default) | **7.46 GiB** | 239 B | 3.27 GiB |
| Row-bucket, quad record (both backends) | **5.28 GiB** | 169 B | 2.90 GiB |
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
(`src/gpu/rowbucket_geom.cpp`): the first rung that fits total VRAM with 1 GiB of
headroom — and, on OpenCL, the single-allocation cap. Every rung is KAT-gated with drops
zero.

**The ladder has two axes, and it is ordered by measured time rather than by footprint** —
because the two disagree. Times are CUDA, one binary, 200 distinct nonces, 2026-07-28; the
build has since gained ~1 ms overall, so the ratios are the point rather than the
absolutes:

| rung | footprint | ms/solve | reached when |
|---|---|---|---|
| packed (16,1) | 7.46 GiB | **33.7** | ≥ 8.5 GiB free |
| packed (15,2) | 6.88 GiB | 36.0 | ≥ 7.9 |
| **quad (16,1)** | **5.28 GiB** | 38.4 | ≥ 6.3 |
| packed (14,3) | 6.50 GiB | 40.0 | only when a single-allocation ceiling binds |
| **quad (15,2)** | **4.91 GiB** | 40.7 | ≥ 5.9 |
| **quad (14,3)** | **4.66 GiB** | 44.8 | ≥ 5.7 — the floor |

The row that matters is the third. **quad (16,1) is both smaller and faster than packed
(14,3)** — 5.28 GiB at 38.4 ms against 6.50 at 40.0 — because a coarser geometry pays in
scatter locality what the quad record pays in re-derivation arithmetic, and the arithmetic
is cheaper. A ladder sorted by footprint would hand those cards the slower rung. packed
(14,3) is kept below it only because its *single* allocation is smaller (2.76 GiB against
2.90), which a `max_alloc`-bound backend can still need.

The quad record buys no speed and no watts at any power limit — that is
[measured, not assumed](performance-research.md#under-a-cap-the-effect-appears--and-the-trade-still-never-pays) —
so the ladder reaches for it only when no packed rung fits. It is purely what lets a
smaller card run at all. `MXBM_QUAD=0|1` forces the choice.

### What each card class gets

| Card | reported gmem / max_alloc | rung (both backends) | OpenCL allocation |
|---|---|---|---|
| 16 GB (4070 Ti S) | 15.59 / 3.90 GiB | (16,1) | one buffer per set |
| 12 GB | 11.60 / 2.90 GiB | (16,1) | split into bucket-halves |
| 11 GB | 10.60 / 2.65 GiB | (16,1) | split |
| 10 GB | 9.70 / 2.42 GiB | (16,1) | split |
| 8 GB | 8.00 / 2.00 GiB | (15,2) | split |
| 6 GB | 5.70 / 1.43 GiB | quad (14,3) | split |

The back-ref rows (0.64 GiB) never split; they bind only below ~2.6 GiB of VRAM, which is
under the floor anyway.

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
| Throughput | **59.8 sol/s** | 59.4 sol/s |
| End-to-end solve | **33.3 ms** | 33.5 ms |
| Board power | 284 W — the card's 285 W limit, `sw_power_cap` active 99–100 % of the time | — |
| Efficiency | 0.208 sol/s/W stock, **0.258 at 210 W** (its optimum on the stock memory clock), best measured **0.264** at 160 W with `--mclk 5001` | — |

The two backends are within **1.012×** of each other, measured in the same session
(CUDA 33.1 ms, OpenCL 33.5 ms, 2026-08-02). Cross-session figures carry a ~2.5 % band, so
that is the only comparison that means anything; the column figures above are each
backend's own controlled headline.

The card is power-limited, not thermally limited, in every kernel, so the board power
limit is the most valuable knob on it: **at 220 W the solver does 54.85 sol/s for 219.5 W,
against 59.05 at 284 W** — 7 % of the speed for 23 % of the power. Below ~170 W the memory
clock is the second knob: `--mclk 5001` is worth 8.5–14.4 % there and puts the efficiency
optimum at 160 W (**3.79 J/solution**, against ~4.8 at stock). Against an *equally capped*
lolMiner, MXBM wins on both speed and efficiency between ~210 W and ~256 W, and loses
below that band. The full curves are in
[Both miners under the same cap](performance.md#both-miners-under-the-same-cap) and
[the 5001 memory rung](performance.md#below-stock-the-other-rung-pays-85-to-144--under-caps-below-173-w-and-a-new-efficiency-record).

End-to-end is the headline figure — `solve()` including survivor readback, back-reference
recovery and CPU verification — as a median over 300 distinct nonces. BeamHash III yields
~1.98 *verified* solutions per solve, measured independently on both backends.
`./build/bench_rounds 20` reports the pipeline-only median instead, which is the controlled
number used to make optimization decisions.

See [performance.md](performance.md) for the measured state, and
[performance-research.md](performance-research.md) for the full optimization history.

---

## Known limitations

These are open issues in MXBM, not properties of BeamHash III.

### Memory efficiency is ~2.4× off the algorithm's design target

Beam's own mining documentation states:

> "Beam Hash III requires a graphics card with a minimum 3GB of memory."
> — <https://beam.mw/docs/mining>

That is the **algorithm's design target**, not a third-party miner's quirk: BeamHash III
was designed by Wilke Trei, who also writes lolMiner. MXBM currently needs **7.46 GiB**,
roughly **2.4× more** than the algorithm is meant to require.

At 2^25 elements, a 3 GB budget implies **≈ 96 B/element in total** — less than the
**3.61 GiB** that two live layers of raw 56 B work words occupy in MXBM's design. So the
target is not reachable by trimming MXBM's current structure; it rules out holding two
full layers of work words at once. A conforming implementation must do something
structurally different, such as:

- **streaming / in-place layer reuse** — writing round *r+1* into space freed by consumed
  round-*r* buckets, exploiting the fact that the stored element narrows every round
  (53 → 50 → 47 → 36 B), or
- **index-only storage with re-derivation** — keeping 4 B indices and recomputing work
  state on demand, trading arithmetic for memory.

**The second route is implemented as far as it goes.** Rounds 1–2 store no work state at
all — an element is rebuilt from the seed indices it already had to carry as leaves, so
their records are 8 B and 16 B instead of 8 and 72. Round 3 stores work state: it had the
same treatment and it was
[taken back out](performance-research.md#retiring-the-round-3-quad-record), because once
compile-time round constants sped the kernel up, its 4-seed rebuild no longer hid in memory
stalls and cost more than the bytes it saved. It does **not** extend to rounds 4–5 either:
rebuild cost doubles per round while the record it replaces shrinks, and round 3 is already
past the point where the recompute hides inside the kernel's stalls (see "the
compute-hiding budget" in
[performance-research.md](performance-research.md#established-limits)).

**Reaching 3 GB therefore needs the first route — streaming / in-place reuse.** That is a
*reach* lever and only a reach lever: it writes the same bytes to reused addresses, so the
peak allocation falls and the traffic does not move at all.

**Traffic is the separate quantity, and it is the efficiency lever.** A solve moves
**13.0 GB** of DRAM traffic, within **1 %** of the compulsory minimum for these record
widths — there is no waste left to reclaim, only records to narrow. That the narrowing
pays is measured, not assumed: cutting round 2's record 72 B → 16 B, so the solve moves
16 % fewer bytes, raises the clock the card sustains at a fixed 285 W by **60 MHz**, with a
positive control at ±0 MHz ([details](performance-research.md#but-bytes-are-not-free-in-watts-and-under-a-cap-watts-are-clock-60-mhz)).
The prize is bounded, though — ≤ 92 MHz/GB across the whole solve — and every record is
already at `ceil(bits/64)`, so it is not a lever anything currently reaches.

Meanwhile the card runs pinned at its 285 W board limit in every kernel, so joules per
solution are set by how long a solve takes. See
["Power and efficiency"](performance.md#power-and-efficiency).
