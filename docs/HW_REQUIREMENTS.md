# Hardware Requirements

MXBM mines BeamHash III — an Equihash-style Wagner search on the ⟨144,5⟩ parameter shape;
see [beamhash-iii.md](beamhash-iii.md). This document states what hardware is needed, why,
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
| **VRAM — CUDA backend (the default)** | **6 GB** — needs > 5.7 GiB *reported*. 10 GB and up get the fastest geometry; below that the ladder steps down, 7–33 % slower |
| **VRAM — OpenCL backend (fallback)** | **11 GB** — the single-allocation ceiling binds first, and the [quad record](performance.md#the-quad-record-on-opencl-takes-11-gb-cards-off-the-sort-path-190--45-ms) is what clears it at 11 GB; see [limitation 1](#1-below-11-gb-opencl-is-capped-by-its-single-allocation-limit) |
| **VRAM — what a full search occupies** | **7.46 GiB** at the fastest geometry, down to **4.66 GiB** at the coarsest (CUDA, with the [quad record](performance.md#the-quad-record-29--footprint-and-the-byte-prize-does-not-survive-re-derivation)); 6.50 GiB is the OpenCL floor |
| **VRAM — what BeamHash III is designed to need** | **3 GB** ([Beam docs](https://beam.mw/docs/mining)) — MXBM is ~2.4× over |
| **Host RAM** | Modest; only survivor candidates (≤ 1024 × 128 B) are read back per solve. |
| **CPU** | Any; the CPU verifies candidates only (a few per solve). |

A card below its backend's threshold does not mine slowly — it **refuses to start**, by
design. See [the partial-search caveat](#the-partial-search-caveat).

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

The largest single allocation matters independently of total VRAM: OpenCL reports
`CL_DEVICE_MAX_MEM_ALLOC_SIZE`, commonly **¼ of VRAM** on NVIDIA. The row-bucket path's
3.27 GiB record array therefore needs a card reporting ≥ ~3.3 GiB max-alloc; below that
`rb_pick_geometry()` steps the bucket geometry down one notch at a time (16,1) → (15,2) →
(14,3), which needs only 2.76 GiB and costs ~5 % speed, and only if even that will not fit
does the path fall back to the sort path (~5× slower).

**CUDA is not subject to that ceiling** — there is no `CL_DEVICE_MAX_MEM_ALLOC_SIZE`
equivalent, so `CudaSolver` allocates its 3.27 GiB record array at the fastest (16,1)
geometry on any card with the total memory for it. This is why the CUDA threshold is a
whole size class lower than the OpenCL one.

CUDA walks the *same* ladder, from the same `rb_geometry_for()`, bounded by total VRAM
alone — and then checks the prediction against the allocator, stepping down again if the
memory is not actually free (a desktop compositor can be holding a gigabyte). It also
walks **three further rungs OpenCL does not have**, using the 24 B quad record: the full
six-rung table, with its measured times, is under
[limitation 1](#1-below-11-gb-opencl-is-capped-by-its-single-allocation-limit). Every rung
is KAT-gated with drops zero, and the allocator retry walks the rung list rather than the
geometry, so a card that loses a packed rung to a busy desktop falls through onto the quad
ones instead of being refused.

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
| Throughput | **58.9 sol/s** | 48.2 sol/s |
| End-to-end solve | **33.8 ms** | 41.0 ms |
| Board power | 284 W — the card's 285 W limit, `sw_power_cap` active 99–100 % of the time | — |
| Efficiency | 0.207 sol/s/W stock, **0.245 at 220 W**, peak 0.253 at 200 W | — |

The card is power-limited, not thermally limited, in every kernel, so the board
power limit is the most valuable knob on it: **at 220 W the solver does 55.4 sol/s for
219.5 W, against 59.2 at 284 W** — 6 % of the speed for 23 % of the power.
The full curve, its interior efficiency optimum at ~200 W, and how it compares against
lolMiner capped to the same limits are in
[Both miners under the same cap](performance.md#both-miners-under-the-same-cap).

End-to-end is the headline figure — `solve()` including survivor readback, back-reference
recovery and CPU verification — as a median over 300 distinct nonces (±4.1 %, 1σ).
BeamHash III yields ~1.98 *verified* solutions per solve, measured independently on both
backends. `./build/bench_rounds 20` reports the pipeline-only median instead (40.3 ms on
OpenCL), which is the controlled number used to make optimization decisions.

See [performance.md](performance.md) for the full optimization history.

---

## Known limitations

These are open issues in MXBM, not properties of BeamHash III.

### 1. Below 11 GB, OpenCL is capped by its single-allocation limit

Not by total VRAM. Worked through from `compute_budget()` and `rb_pick_geometry()`:

| Card | reported gmem / max_alloc | OpenCL | CUDA |
|---|---|---|---|
| 16 GB (4070 Ti S) | 15.59 / 3.90 GiB | full, geometry (16,1) | full, (16,1), 35.1 ms |
| 12 GB | 11.60 / 2.90 GiB | full, geometry (14,3), ~5 % slower | full, (16,1), 35.1 ms |
| 11 GB | 10.60 / 2.65 GiB | full, but on the **sort path** (~5× slower) | full, (16,1), 35.1 ms |
| 10 GB | 9.70 / 2.42 GiB | **refuses** — sort path fits only 0.93 × 2^25 | full, (16,1), 35.1 ms |
| 8 GB | 7.70 / 1.93 GiB | **refuses** — 0.74 × 2^25 | full, (15,2), 36.0 ms |
| 6 GB | 5.70 / 1.43 GiB | **refuses** | full, **quad (14,3)**, 44.8 ms |

**The CUDA ladder has two axes, and it is ordered by measured time rather than by
footprint** — because the two disagree. All six rungs are KAT-gated and drop-free:

| rung | footprint | ms/solve | reached when |
|---|---|---|---|
| packed (16,1) | 7.46 GiB | **33.7** | ≥ 8.5 GiB free |
| packed (15,2) | 6.88 GiB | 36.0 | ≥ 7.9 |
| **quad (16,1)** | **5.28 GiB** | 38.4 | ≥ 6.3 |
| packed (14,3) | 6.50 GiB | 40.0 | only when a single-allocation ceiling binds |
| **quad (15,2)** | 4.91 GiB | 40.7 | ≥ 5.9 |
| **quad (14,3)** | 4.66 GiB | 44.8 | ≥ 5.7 — the floor |

The row that matters is the third. **quad (16,1) is both smaller and faster than packed
(14,3)** — 5.28 GiB at 38.4 ms against 6.50 at 40.0 — because a coarser geometry pays in
scatter locality what the quad record pays in re-derivation arithmetic, and the arithmetic
is cheaper. A ladder sorted by footprint would hand those cards the slower rung. packed
(14,3) is kept below it only because its *single* allocation is smaller (2.76 GiB against
2.90), which a `max_alloc`-bound backend can still need.

The quad record buys no speed and no watts at any power limit — that is
[measured, not assumed](performance.md#under-a-cap-the-effect-appears--and-the-trade-still-never-pays) —
so the ladder reaches for it only when no packed rung fits. It is purely what lets a
smaller card run at all. `MXBM_QUAD=0|1` forces the choice.

The 8 GB row changed on 2026-07-26. `CudaSolver::available()` used to demand the (16,1)
footprint specifically, so those cards were refused by both backends and mined nothing;
they now run the full search one rung down the ladder. The CUDA floor moved again on
2026-07-28, to 5.7 GiB reported, when the quad record joined the ladder as a second axis.

The seed layer itself is not the problem at any of these sizes: at 256 B/element a full
2^25 layer needs only 8.59 GiB, so `elems_per_round` stays at 2^25 down to ~10.1 GiB of
reported VRAM. What binds first is that NVIDIA's OpenCL reports `max_alloc = ¼ VRAM`,
and the coarsest row-bucket geometry still wants a 2.76 GiB record array — i.e. an
11 GB card or smaller cannot host *any* row-bucket geometry and drops to the sort path,
whose 284 B/element no longer fits a full layer at 10 GB.

**This was the "~1.65× too conservative budget" (fixed 2026-07-25):** `compute_budget`
used a single 396 B/element figure left over from a six-buffer design, which granted a
full search only above 14.6 GiB — 16 GB cards only. It is now sized per path (256 B
row-bucket, 284 B sort; measured 239 and 264) and the geometry adapts to the device.

Remaining work, in order of value: splitting the row-bucket record array into two
buffers would put 10–11 GB cards on the fast path under OpenCL too, since the constraint
is the size of the *largest single* allocation and not the total.

### 2. Memory efficiency is ~2.4× off the algorithm's design target

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

> **Update (2026-07-24): the second route is now partly implemented, and the prediction
> below held.** Rounds 1–3 store no work state at all — an element is rebuilt from the
> seed indices that it already had to carry as leaves, so the records are 8 B / 16 B /
> 24 B instead of 8 / 72 / 80. Footprint fell **8.36 → 6.95 GiB** and solve time fell
> **103.6 → 83.2 ms** *in the same changes*.
>
> **Caveat added later:** the round-3 part of that was subsequently *reverted* — once
> compile-time round constants sped the kernel up, its 4-seed rebuild no longer hid in
> memory stalls and cost more than the bytes it saved. Rounds 1–2 keep index-only
> storage; round 3 stores work state again. Solve time is now 40.4 ms and the footprint
> 7.46 GiB. See "retiring the round-3 quad record" in [performance.md](performance.md).
>
> It does **not** extend to rounds 4–5. Rebuild cost doubles per round while the record
> it replaces shrinks, and round 3 already sits at the point where the recompute stops
> hiding inside the kernel's memory stalls (see "the compute-hiding budget" in
> [docs/performance.md](performance.md)). Reaching 3 GB needs the *first* route —
> streaming / in-place reuse — not more re-derivation.

This is correctness-neutral, but it is **also an energy cost**, and that half is now
measured rather than suspected. A solve moves **13.0 GB** of DRAM traffic, which is
within **1 %** of the compulsory minimum for these record widths — there is no waste
left to reclaim, only records to narrow. Meanwhile the card runs pinned at its 285 W
board limit in every kernel, so joules per solution are set by how long a solve takes:
4.94 J/solution at stock, 4.08 capped to 220 W and 3.95 at the ~200 W efficiency peak,
against lolMiner's 4.48 at its own uncapped 239 W. MXBM is therefore ahead on energy
when capped and behind when not, and the margin either way is set by the bytes. See
["Power and efficiency"](performance.md#power-and-efficiency).

> **Correction (2026-07-28): "3 GB instead of 7.46 would move proportionally fewer bytes"
> is wrong, and the two halves of this section are two different projects.** Footprint and
> traffic are not the same quantity. Streaming / in-place layer reuse — the route named
> above for reaching 3 GB — writes *the same bytes to reused addresses*: the peak
> allocation falls, the traffic does not move at all. Only **narrower records** reduce
> traffic, and traffic is what the energy argument rests on.
>
> The energy half is now measured rather than inferred. Cutting round 2's record 72 B →
> 16 B, so the solve moves 16 % fewer bytes, raises the clock the card sustains at a fixed
> 285 W by **60 MHz** — with a positive control at ±0 MHz, since rounds 1 and 4 already
> store 16 B and ablating them changes nothing. See ["bytes are not free in
> watts"](performance.md#but-bytes-are-not-free-in-watts-and-under-a-cap-watts-are-clock-60-mhz).
>
> So: **narrowing records is the efficiency lever** (measured, and it is the one that
> widens the lead against lolMiner under a cap). **In-place reuse is the reach lever** —
> it is what puts a full search on an 8 GB card, which is worth doing on its own terms and
> is not an energy improvement.
