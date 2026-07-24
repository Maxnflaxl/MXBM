# GPU Performance

The permanent record of MXBM's GPU solver optimization: what we tried, what it
measured, and why. Every number here was produced by a committed benchmark or test
in this repository — nothing is estimated or extrapolated unless explicitly labelled.

**Target:** lolMiner does **~53 sol/s** on an RTX 4070 Ti SUPER at **stock clocks, no
special configuration** (user-measured). BeamHash III yields ~1.9 solutions per solve,
so 53 sol/s ÷ 1.9 ≈ **28 solve/s ≈ 36 ms/solve**. That is the bar.

**Reference hardware** for every measurement below: RTX 4070 Ti SUPER (Ada, sm_89,
66 CUs, 16 GB, 48 KB LDS/workgroup, ~510 GB/s achievable copy bandwidth).

**How to reproduce:** `./build/bench_rounds 20` reports the median over 20 full solves
plus a correctness gate. `MXBM_NO_ROWBUCKET=1` forces the fallback sort path.

---

## Progress log

Times are median ms per solve (`bench_rounds`), lower is better. "Worked" and
"Didn't work" link to the sections explaining each result.

| Date | Change | Before | After | Δ ms | Δ % | Worked | Didn't work |
|---|---|---|---|---|---|---|---|
| 2026-07-23 | Sort-based collision finder (P1–P3) | 245.0 | 239.0 | −6.0 | −2.4 % | [Sort path](#sort-based-collision-finder) | — |
| 2026-07-23 | Tiled 4-bit × 6-pass radix sort (P4) | 239.0 | 225.0 | −14.0 | −5.9 % | [Tiled radix](#tiled-radix-sort) | — |
| 2026-07-24 | Fixed-width AoS compaction `[7,7,6,5,1]` | 224.4 | 214.5 | −9.9 | −4.4 % | [Compaction](#fixed-width-compaction) | [SoA planes](#soa-word-planes), [runtime widths](#runtime-loop-bounds) |
| 2026-07-24 | Fused row-bucket pipeline (STEP A–C) | 215.0 | 148.0 | −67.0 | −31.2 % | [Row-bucket](#fused-row-bucket-pipeline) | [Un-fused LDS](#un-fused-lds-path), [lead array](#stride-1-lead-array) |
| 2026-07-24 | L5-thin: drop round-5 leaf payload | 151.9 | 139.1 | −12.8 | −8.4 % | [L5-thin](#l5-thin-emit) | — |
| 2026-07-24 | Per-round work compaction on row-bucket | 139.1 | 136.0 | −3.1 | −2.2 % | [Row-bucket compaction](#per-round-compaction-on-the-row-bucket-path) | — |
| 2026-07-24 | Async enqueue (de-bubble) | 135.0 | 133.0 | −2.0 | −1.5 % | [De-bubble](#async-de-bubble) | [Magazine](#shared-memory-magazine), [gi atomic](#global-gi-atomic) |
| 2026-07-24 | Compact `leftContrib` (fold 8 leaves → 1 u64) | 131.1 | 124.8 | −6.3 | −4.8 % | [leftContrib](#compact-leftcontrib) | [General mix-state](#general-compact-mix-state) |
| 2026-07-24 | Packed element record (1 block, not 4 arrays) | 124.8 | 115.7 | −9.1 | −7.3 % | [Packed record](#packed-element-record) | [SoA LDS staging](#soa-lds-staging) |
| | | | | | | | [Occupancy tuning](#occupancy-tuning), [dense key array](#dense-key-array), [decoupled scatter](#decoupled-scatter), [two-level bucketing](#two-level-bucketing) |

**Current: 115.7 ms/solve (8.64 solve/s).** Started at 245 ms → **−53 %**.
Remaining gap to lolMiner: **~3.2×**.

---

## Architecture

The solver runs Wagner's algorithm for Equihash⟨150,5⟩: 2^25 seed elements, five
rounds, each finding pairs that collide on a 24-bit key and combining them. BeamHash III
adds a mandatory per-round `apply_mix` that folds the element's growing index tree into
word 0, re-deriving the collision key every round.

Two collision-finding paths exist. The **fused row-bucket path** is the default wherever
device memory allows; the **sort path** is the fallback for smaller cards.

### Fused row-bucket pipeline

*One kernel per round* does round *r*'s match **and** round *r+1*'s mix **and** round
*r+1*'s scatter. Per workgroup (one per bucket × sub-mask):

1. Coalesced-load the bucket's elements into LDS (work words + gi + lead + leaf payload).
2. Build an LDS hash table over the middle key bits via `atomic_xchg` chains.
3. For each equal-key pair: order by (lead, gi), `bh3_combine` at `Lout(r)`.
4. Materialize the child's leaf prefix, fold `apply_mix(Lmix(r+1))` to set its next key.
5. Emit the child **directly into round r+1's buckets**, keyed by that fresh key.

Fusion is the point: the coalesced-gather win is destroyed if a separate scatter pass
re-reads the elements — see [Un-fused LDS path](#un-fused-lds-path).

Path selection is by device memory (`want_rowbucket` in `src/gpu/round_pipeline.cpp`):
row-bucket needs ~11 GB with a 3.3 GB largest single allocation, so 8–12 GB cards fall
back to the sort path automatically. `MXBM_ROWBUCKET=1` / `MXBM_NO_ROWBUCKET=1` override.

### Round schedule

| Round | Lmix | Lout | padNum | Significant work words | Leaf payload carried |
|---|---|---|---|---|---|
| 1 | 448 | 424 | 1 | 7 | 1 → 2 raw leaves |
| 2 | 424 | 400 | 2 | 7 | 2 → 4 raw leaves |
| 3 | 400 | 376 | 4 | 6 | 4 raw → 1 u64 `leftContrib` |
| 4 | 376 | 288 | 6 | 5 | `leftContrib` → none |
| 5 | 288 | 24 | 9 | 1 | none (terminal) |

---

## What worked

### Sort-based collision finder
Replaced the atomic-bucket scatter + all-pairs match with a stable radix sort of
(key, index) pairs and a coalesced run-scan emit. Also fused the sort-pair emission into
the mix, deleting a whole strided pass over the work array. 245 → 225 ms across P1–P4.

### Tiled radix sort
CUB-style 4-bit × 6-pass sort with per-thread private histograms and an in-place
cross-thread prefix; 8 items/thread measured optimal (16 overflows 48 KB LDS).
Sort cost per round 18 → 9.5 ms.

### Fixed-width compaction
`bh3_combine` shifts out the 24 matched bits and masks to `Lout` each round, so the
significant work words shrink on the schedule **[7, 7, 6, 5, 1]**. Storing only those
words is bit-exact (upper words are provably zero). The win requires **compile-time**
widths — see [runtime loop bounds](#runtime-loop-bounds). Measured 224.4 → 214.5 ms on
the sort path, and the same schedule was later applied to the row-bucket path.

### Fused row-bucket pipeline
The single largest win: **215 → 148 ms (−31 %)**. Built in three measured steps.

**STEP A — bucket layout.** Carrying the leaf prefix *inside* the bucket element makes
the leaf read coalesced instead of scattered, at the cost of a wider emit. Measured per
level: read win +41.2 ms, write penalty −24.1 ms → **net +17.1 ms for "fat" buckets**.
Level 5 was the exception (−1.2 ms), which later became [L5-thin](#l5-thin-emit).

**STEP B — the kernel.** `round_fused_lds`, validated byte-exact against the CPU oracle
(`ref::cpu_round`) for every leaf width, and drop-free at 2^24 scale.

**STEP C — the pipeline.** Entry kernel (seed+mix+scatter into round-1 buckets), four
fused rounds ping-ponging bucket sets, and a terminal round-5 kernel that detects
all-zero survivors. Back-refs keep the same consolidated row layout, so `recover` was
unchanged.

*Note on launch geometry:* the kernel is one **workgroup** per (bucket, sub-mask), so it
must be launched as `run1d(groups * WG, WG)` — launching `groups` work-items silently
finds 0.4 % of collisions.

### L5-thin emit
Round 4 was the most expensive round because it emitted round-5 children carrying a full
9-leaf payload. Round 5 is terminal — it never mixes, and `recover` walks back-refs — so
those leaves are dead weight. The kernel now decouples *build* from *store*: it builds
`max(padNum_next, sOut)` leaves for the child's own mix but stores only `sOut`.
Round 4: 37.2 → 29.3 ms; pipeline 151.9 → 139.1 ms.

### Per-round compaction on the row-bucket path
The `[7,7,6,5,1]` schedule applied to the fused kernel as compile-time width variants
(r3 = 7→6, r4 = 6→5), with round 5 reading at stride 5. Round 4: 29.3 → 26.8 ms;
survivor scan 6.1 → 4.6 ms.

### Async de-bubble
`run1d` blocks on `clFinish` after every kernel. In production (non-verbose) the
row-bucket pipeline now enqueues the whole chain asynchronously on the in-order queue and
drains once at the survivor readback. Only ~2 ms: the pipeline is genuinely GPU-bound, so
those waits were mostly waiting on real work.

### Compact leftContrib
Round 4's mix needs `padNum(5) = 9` leaves: **all 8** of the left parent's, but only the
right parent's **first** — which is already carried as `lead`. `apply_mix` decomposes
exactly:

```
mix = rotl24( workPart + indexPart ),   indexPart additive over leaves
```

because index bits sit at positions ≥ `Lmix` while the element's own bits sit below it
(`Lmix(r) = Lout(r-1)`) — disjoint, and `rotl` distributes over a disjoint OR. So a
round-4 element carries **one u64** instead of 8 raw leaves:

```
leftContrib = rotl40( apply_mix(0, itsOwn8Leaves, 8, 288) )        // round 3 emits this
key         = rotl24( rotl40( apply_mix(child, [0×8, rightLead], 9, 288) ) + leftContrib )
```

Both sides use the pinned `bh3_apply_mix`, so no rotation constants are duplicated. Round
3's emit drops 88 → 64 B/child and round 4's stage read by the same; the leaf buffer
shrinks 9 → 4 uints/slot (~1.1 GB). **131.1 → 124.8 ms.** The algebra is pinned by
`test_contrib_identity` (200 000 random trials) so a regression fails loudly.

### Packed element record
Elements were stored across **four parallel arrays** (work, gi, lead, leaves) indexed by
the same random slot. Global memory is serviced in **32 B sectors**, so a 4 B `gi` and a
4 B `lead` each burned a whole sector — roughly **5 sectors of traffic for a 72 B child**,
on both the scattered emit and the staging read. An isolated probe writing the same 72 B
three ways measured the cost directly:

| Layout | ms | useful GB/s |
|---|---|---|
| 4 split arrays | 21.78 | 111 |
| 1 packed record | **18.20** | **133** |
| packed, padded to 96 B (3 aligned sectors) | 18.25 | 132 |

Padding to sector alignment buys nothing beyond contiguity, so the record wastes no
bytes. The element is now one contiguous block:

```
[work: inwords u64][meta: (gi << 32) | lead][leaf payload: two u32 per u64]
```

Strides are per-round runtime arguments while every **loop bound stays compile-time**, so
the word loops still fully unroll. Bucket capacity was retuned to `mean + mean/4 + 256`
to keep the widest packed array under the 3.9 GB single-allocation limit — still ~17σ of
headroom against a distribution whose max sits near mean + 4.5σ, and drop counters gate
every run. **124.8 → 115.7 ms**, every round faster, and ~4 GB less memory.

---

## What didn't work

Each of these was measured and reverted. They are recorded so they are not retried.

### SoA word planes
One buffer per element word, skipping zero planes. Measured **only 1 % better** than
fixed-width AoS (14.22 vs 14.42 ms) — not worth a full-pipeline rewrite. An AoS element
write touches ~1 cache line; the same element in SoA fans across W buffers.

### Runtime loop bounds
The first compaction attempt used runtime-variable word-loop bounds (`w < inW`). This
defeats compiler unrolling and cost **+24 to +27 ms**. Compile-time widths are mandatory
everywhere in the hot path; this failure mode recurred twice.

### Un-fused LDS path
Bucket-scatter as a *separate* pass, then an LDS collide pass: **244 ms, slower than the
215 ms sort path**. The separate fat scatter consumes exactly the coalesced-gather win it
enables. This is why the row-bucket rewrite had to fuse.

### Stride-1 lead array
Publishing leaf 0 into a dedicated dense array so the pair-ordering read would be
sequential. Goldens stayed correct but match **regressed 158.9 → 176.3 ms**. Leaf 0 is
cache-adjacent to the leaf-prefix data the same kernel already reads, so splitting it out
added a second memory stream. The lead read and prefix build are cache-coupled; neither
can be peeled off alone.

### Occupancy tuning
The fused kernel uses ~41 KB of the 48 KB LDS → exactly **1 workgroup/SM (~17 %
occupancy)**. Two attempts to buy occupancy both lost:
- Halving the group cap and adding a sub-mask bit (LDS 41 → 22 KB, 2 wg/SM): **slower**,
  because a finer sub-mask re-scans each bucket 16× instead of 8×.
- De-fusing to free the 12 KB leaf array only reaches 29 KB — still >24 KB, so still
  1 wg/SM — while re-introducing the +17 ms scattered leaf gather.

Decisively: [decoupled scatter](#decoupled-scatter) proved occupancy is irrelevant to the
dominant cost anyway.

### Dense key array
A dense per-slot key array so the sub-mask scan reads 4 B coalesced instead of striding
into the 56 B work records. **Slower (133 → 146 ms)**: the redundant strided key scan is
already L2-served and was never DRAM-bound, so the extra array was pure added traffic.

### Decoupled scatter
Hypothesis: the emit is slow because the fused kernel's low occupancy can't hide its
latency, so a standalone high-occupancy scatter kernel would be faster. **Refuted.**
`test_scatter_occupancy` runs the identical scatter at three LDS footprints:

| LDS/workgroup | ms | GB/s |
|---|---|---|
| tiny (max occupancy) | 14.44 | 260 |
| 24 KB (~2 wg/SM) | 14.21 | 265 |
| 40 KB (fused-kernel occupancy) | 14.21 | 264 |

Identical (lo/hi = 0.98×). The scatter is **random-access-bound, not latency-bound**, at
~51 % of the 510 GB/s coalesced peak. Bucket count is equally irrelevant: 264 GB/s at
1024 buckets vs 260 GB/s at 16384 (256 buckets is *slower*, 221 GB/s, from counter
contention).

### Global gi atomic
Suspected the per-child `atomic_inc` on a single global counter was serializing the emit.
Stubbed it out: **no speedup** — NVIDIA's atomic aggregation already handles it. The
per-bucket counters were tested too (`scatter_noatomic`): removing them made the scatter
**slower** (235 vs 260 GB/s), because `atomic_inc` hands out sequential in-bucket slots,
which is *better* locality than a computed slot.

### Shared-memory magazine
Staging children in LDS per destination bucket and flushing coalesced. Infeasible at our
fan-out: a workgroup produces ~256 children spread over 16384 buckets, so magazines hold
~1 element and never fill. Confirmed by the literature (local-reorder needs ≤256 bins) and
by Wilke Trei's own FishHashMiner, which contains no magazine.

### SoA LDS staging
The LDS staging array used AoS indexing `lwork[pos*7 + w]` — a stride of 7 ulongs = 14
uints, and `gcd(14, 32 banks) = 2`, so every access was nominally a 2-way bank conflict.
Switching to plane-major SoA (`lwork[w*CAP + pos]`) measured **125.7–125.9 vs 124.8 ms —
reproducibly ~1 ms slower**. The dominant LDS accesses are in the collision-walk, where
`leftPos`/`rightPos` come from hash-chain traversal and are scattered under either
layout; only the staging loop has consecutive indices, and it is the minority of
accesses. (Note this is LDS layout only — the *global* packing win above is a separate,
real effect.)

### Two-level bucketing
The last structural idea: partition coarsely (256 bins, long runs, coalesced flush) then
refine. Settled by measuring the **coalescing benefit curve** directly (`scatter_runs`)
— run length is the only thing any reorder scheme changes:

| Emit width | Real (random dest) | Perfect (R=32) | Max reorder gain |
|---|---|---|---|
| 56 B (r1/r2) | 260 GB/s | 497 GB/s | 1.91× |
| 48 B (r3) | 260 GB/s | 547 GB/s | 2.10× |
| 40 B (r4) | 261 GB/s | 578 GB/s | 2.22× |

Our emit is **write-only** — children are produced in registers, there is no source read
to amortize — so two-level costs 1 write + 1 read + 1 write against today's single write:

```
break even needs   1/260 > 1/BW_A + 2/BW_B
at the card's peak  3/578 = 0.00519  ≫  1/260 = 0.00385
```

Two-level would need coalesced bandwidth **> 780 GB/s**, above the card's 510 GB/s peak.
**Impossible on this hardware**, at every width — including the late-round "thinner
element" hybrid, which clears a naive 2× bar but still fails this one.

### General compact mix-state
The `leftContrib` decomposition ([above](#compact-leftcontrib)) raises the question of
carrying mix state instead of leaves everywhere. It does not generalize: `Lmix` changes
every round, so a contribution precomputed at one round's base is useless at the next, and
computing any round's contribution requires the raw leaves anyway. Serving all future
rounds would cost more than the leaves it replaces. Round 4 is the sole exception because
it needs *all* of one parent's leaves and only *one* of the other's.

---

## Established limits

Measured properties of the problem on this hardware. These bound what any further
optimization can achieve.

1. **The element cannot shrink below `[7,7,6,5,1]`.** That schedule is BeamHash III's
   information floor, confirmed independently by Wilke Trei's own `beamhashverify`
   reference. The compact 16 B element in the official `BeamMW/cuda-miner` belongs to
   **BeamHash I** (2018, pre-Fork2: shift-out-key running XOR, no per-round `apply_mix`),
   and does not transfer.
2. **`apply_mix` is genuinely per-round**, folding padNum ∈ {1,2,4,6,9} index-tree entries
   into word 0. It rewrites only word 0; words 1–6 pass through read-only.
3. **A key-random fat scatter runs at ~260 GB/s (51 % of peak)** and is invariant to
   occupancy, bucket count, and atomics. It is random-access-bound.
4. **Coalescing cannot be bought back** — maximum 1.91–2.22×, against a 3× traffic cost
   for any two-pass scheme.

Consequence: lolMiner carries the same ~56 B element and hits the same scatter ceiling,
so its remaining ~3.5× advantage is not explained by element size, coalescing, occupancy,
or atomics. It is most likely a technique in its collision-finding or overall pipeline
structure that is not derivable from the public references.

---

## Benchmarks and gates

| Command | What it measures |
|---|---|
| `./build/bench_rounds 20` | Median ms/solve over 20 solves + drop-free gate |
| `./build/test_gpu_rounds` | Per-round timing, drop counters, survivor count |
| `./build/test_gpu_solver` | End-to-end solve incl. recovery and CPU verification |
| `./build/test_lds_collide` | All isolated kernel benches and the correctness oracles |
| `ctest --test-dir build` | Full suite (37 tests) |

Correctness gates that must stay green for any performance change:
`bucketDrops == 0`, `pairDrops == 0`, all three KAT goldens byte-identical, and the full
suite. The goldens are checked through every collision path
(`gpu_*_rowbucket`, `gpu_*_lds`, `gpu_*_legacy`) so a path-specific regression is caught.

Diagnostic environment variables: `MXBM_ROWBUCKET` / `MXBM_NO_ROWBUCKET`,
`MXBM_LEGACY_MATCH`, `MXBM_LDS_MATCH`, `MXBM_NO_COMPACT`, `MXBM_SORT_PROFILE`,
`MXBM_ABLATE` (bit 0 skips `apply_mix`, bit 1 skips the fat emit — for phase attribution).
