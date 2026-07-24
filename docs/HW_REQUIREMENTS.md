# Hardware Requirements

MXBM mines BeamHash III (Equihash⟨150,5⟩). This document states what hardware is
needed, why, and where the current implementation falls short.

All figures are computed from the allocator in `src/gpu/round_pipeline.cpp` and
`src/gpu/budget.cpp`, and cross-checked against measurements on the reference card.
Throughput is quoted in **sol/s** (solutions per second), the figure miners and pools
report; BeamHash III yields ~1.9 solutions per solve.

---

## Summary

| | Requirement |
|---|---|
| **GPU** | OpenCL 1.2+ device. Developed and measured on NVIDIA (Ada, sm_89). |
| **VRAM — to run at all** | ~6 GB |
| **VRAM — for a search that actually finds solutions** | **~16 GB today** (see the caveat below) |
| **VRAM — what a full search genuinely needs** | **~6.95 GiB** (row-bucket path) |
| **VRAM — what BeamHash III is designed to need** | **3 GB** ([Beam docs](https://beam.mw/docs/mining)) — MXBM is ~2.3× over |
| **Host RAM** | Modest; only survivor candidates (≤ 1024 × 128 B) are read back per solve. |
| **CPU** | Any; the CPU verifies candidates only (a few per solve). |

> **Known limitation.** The 16 GB figure is *not* a property of the algorithm — it is a
> stale heuristic in `compute_budget`. The real full-search footprint is ~6.95 GiB, so
> 12 GB cards should be usable. See [Known limitations](#known-limitations).

---

## Why BeamHash III needs a lot of memory

A BeamHash III solution is **32 indices drawn from the full `[0, 2^25)` space**. Wagner's
algorithm therefore has to hold a layer of 2^25 (≈ 33.5 M) elements, and each element
carries 448 bits (56 B) of work state — BeamHash III's element is roughly 3× fatter than
classic Equihash⟨150,5⟩ because its `apply_mix` re-derives the collision key from the
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
| Row-bucket (default) | **6.95 GiB** | 222 B | 2.83 GiB |
| Sort (fallback) | 8.89 GiB | 285 B | ~1.8 GiB |

The largest single allocation matters independently of total VRAM: OpenCL reports
`CL_DEVICE_MAX_MEM_ALLOC_SIZE`, commonly **¼ of VRAM** on NVIDIA. The row-bucket path's
2.83 GiB record array therefore needs a card reporting ≥ ~2.9 GiB max-alloc (i.e. ≥ ~12 GB
VRAM); smaller cards fall back to the sort path automatically (`want_rowbucket`).

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

**A reduced search does not mine slower — it mines essentially nothing.** The miner will
appear to run normally while finding no shares. Until this is fixed (see below), treat
any configuration that cannot host the full 2^25 seed layer as non-functional.

---

## Reference measurements

Reference card: **RTX 4070 Ti SUPER** (Ada, sm_89, 66 CUs, 16 GB, 48 KB LDS/workgroup).
Reported by OpenCL: `gmem = 15.59 GiB`, `max_alloc = 3.90 GiB`.

| | |
|---|---|
| Throughput | **33.8 sol/s** (BeamHash III yields ~1.9 solutions per solve) |
| End-to-end solve | 56–62 ms (`GpuSolver::solve()`, incl. recovery + CPU verification) |
| Solve time | 56.2 ms median (`./build/bench_rounds 20`) |

See [performance.md](performance.md) for the full optimization history.

---

## Known limitations

These are open issues in MXBM, not properties of BeamHash III.

### 1. The VRAM budget is ~1.65× too conservative

`compute_budget` sizes the seed layer at **396 B/element**
(`5 rounds × 68 B resident + 56 B seed`), a formula left over from an earlier
six-buffer design. The current pipeline needs **222 B/element** (row-bucket).

The budget also uses a *single* `kBytesPerElement = 304` for both paths, sized for the
sort path — so row-bucket cards are assessed against the sort path's appetite. Splitting
the constant per path is the remaining work here.

Consequences:

- A full 2^25 search is only granted when `VRAM × 0.85 ≥ 2^25 × 396`, i.e. **≥ 14.6 GiB
  reported VRAM** — effectively 16 GB cards only.
- **12 GB cards are downgraded to a partial (non-functional) search** even though the
  real 6.95 GiB footprint would fit.
- 16 GB cards with high driver/display reservation can fall below the threshold and
  silently degrade.

### 2. A partial search fails silently

There is no warning or error when `elems_per_round < 2^25`, despite that configuration
being unable to find solutions in practice. It should refuse to start, or at minimum warn
loudly, rather than mine nothing while appearing healthy.

### 3. Memory efficiency is ~2.3× off the algorithm's design target

Beam's own mining documentation states:

> "Beam Hash III requires a graphics card with a minimum 3GB of memory."
> — <https://beam.mw/docs/mining>

That is the **algorithm's design target**, not a third-party miner's quirk: BeamHash III
was designed by Wilke Trei, who also writes lolMiner. MXBM currently needs **6.95 GiB**,
roughly **2.3× more** than the algorithm is meant to require.

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
> **103.6 → 83.2 ms** *in the same changes*, which is the "same problem" hypothesis
> confirming itself. (Solve time has since fallen further, to 56.6 ms, for an unrelated
> reason — see "compile-time round constants" in [performance.md](performance.md).)
>
> It does **not** extend to rounds 4–5. Rebuild cost doubles per round while the record
> it replaces shrinks, and round 3 already sits at the point where the recompute stops
> hiding inside the kernel's memory stalls (see "the compute-hiding budget" in
> [docs/performance.md](performance.md)). Reaching 3 GB needs the *first* route —
> streaming / in-place reuse — not more re-derivation.

This is correctness-neutral, but it is very likely **also a performance gap**. MXBM's
solver is memory-bandwidth-bound — every optimization that has worked so far won by
moving fewer bytes (see [docs/performance.md](performance.md)). An implementation
using less memory would be expected to move proportionally fewer bytes per round,
which is the leading hypothesis for the remaining throughput difference. Memory
efficiency and throughput are therefore probably the *same* problem, and reducing the
per-element footprint is the highest-value open work item.
