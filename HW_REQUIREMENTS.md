# Hardware Requirements

MXBM mines BeamHash III (Equihash⟨150,5⟩). This document states what hardware is
needed, why, and where the current implementation falls short.

All figures are computed from the allocator in `src/gpu/round_pipeline.cpp` and
`src/gpu/budget.cpp`, and cross-checked against measurements on the reference card.

---

## Summary

| | Requirement |
|---|---|
| **GPU** | OpenCL 1.2+ device. Developed and measured on NVIDIA (Ada, sm_89). |
| **VRAM — to run at all** | ~6 GB |
| **VRAM — for a search that actually finds solutions** | **~16 GB today** (see the caveat below) |
| **VRAM — what a full search genuinely needs** | **~8.4 GiB** (row-bucket path) |
| **Host RAM** | Modest; only survivor candidates (≤ 1024 × 128 B) are read back per solve. |
| **CPU** | Any; the CPU verifies candidates only (a few per solve). |

> **Known limitation.** The 16 GB figure is *not* a property of the algorithm — it is a
> stale heuristic in `compute_budget`. The real full-search footprint is ~8.4 GiB, so
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
| Row-bucket (default) | **8.36 GiB** | 268 B | 3.54 GiB |
| Sort (fallback) | 8.89 GiB | 285 B | ~1.8 GiB |

The largest single allocation matters independently of total VRAM: OpenCL reports
`CL_DEVICE_MAX_MEM_ALLOC_SIZE`, commonly **¼ of VRAM** on NVIDIA. The row-bucket path's
3.54 GiB record array therefore needs a card reporting ≥ ~3.6 GiB max-alloc (i.e. ≥ ~14 GB
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
| Solve time | 115.7 ms median (`./build/bench_rounds 20`) |
| Throughput | 8.64 solve/s ≈ 16.4 sol/s (BeamHash III yields ~1.9 solutions/solve) |

See [docs/performance.md](docs/performance.md) for the full optimization history.

---

## Known limitations

These are open issues in MXBM, not properties of BeamHash III.

### 1. The VRAM budget is ~1.65× too conservative

`compute_budget` sizes the seed layer at **396 B/element**
(`5 rounds × 68 B resident + 56 B seed`), a formula left over from an earlier
six-buffer design. The current pipeline needs **268 B/element** (row-bucket).

Consequences:

- A full 2^25 search is only granted when `VRAM × 0.85 ≥ 2^25 × 396`, i.e. **≥ 14.6 GiB
  reported VRAM** — effectively 16 GB cards only.
- **12 GB cards are downgraded to a partial (non-functional) search** even though the
  real 8.4 GiB footprint would fit.
- 16 GB cards with high driver/display reservation can fall below the threshold and
  silently degrade.

### 2. A partial search fails silently

There is no warning or error when `elems_per_round < 2^25`, despite that configuration
being unable to find solutions in practice. It should refuse to start, or at minimum warn
loudly, rather than mine nothing while appearing healthy.

### 3. Memory efficiency is well short of the state of the art

For comparison, other BeamHash III miners are reported to run in far less VRAM (lolMiner
is understood to need ~3 GB). At 2^25 elements that implies **≈ 96 B/element in total** —
less than the 3.61 GiB that two live layers of raw work words occupy here. Achieving it
would require a fundamentally leaner strategy than MXBM's current two-live-layer design:
streaming or in-place layer reuse, or storing indices and re-deriving work state on
demand.

This is a correctness-neutral efficiency gap, but likely also a **performance** gap: the
solver is memory-bandwidth-bound, so an implementation moving ~2.5× fewer bytes per round
would be expected to run substantially faster. It is the leading hypothesis for the
remaining throughput difference against lolMiner.
