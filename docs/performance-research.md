# GPU Performance Research

The experiment log behind [performance.md](performance.md): every lever tried on the
BeamHash III GPU solver, what it measured, and the mechanism that explains the result.
Negatives are kept deliberately and in full — a null with a mechanism is what stops an
idea being re-proposed, and roughly half of what is recorded here is a null.

Read this before proposing an optimization. The headline figures, the progress log and
the power curves are on [performance.md](performance.md); this page holds everything
those numbers rest on — the solver architecture, the experiments, the established
limits, and the open leads.

**Reference hardware** for every measurement: RTX 4070 Ti SUPER (Ada, sm_89, 66 CUs,
16 GB, 48 KB LDS/workgroup under OpenCL and 100 KB/SM under CUDA, ~510 GB/s achievable
copy bandwidth, ~672 GB/s theoretical). Absolute figures carry a
[~2.5 % cross-session band](performance.md#-the-absolute-figures-reproduce-to-03--within-a-session-and-5--between-sessions)
(narrowed 2026-07-31 from ~5 %); every A/B here was interleaved, so the deltas do not.

| | |
|---|---|
| [Architecture](#architecture) | the fused row-bucket pipeline, the round schedule, the stored records |
| [A note on method](#a-note-on-method) | how to measure here without fooling yourself |
| [Instruments](#instruments-diagnostic-and-ablation-flags) | the diagnostic and ablation flags |
| [What worked](#what-worked) | the 20 shipped optimizations, with their mechanisms |
| [What didn't work](#what-didnt-work) | the 18 measured and reverted |
| [Measured results, 2026-07-26 to 2026-07-28](#measured-results-2026-07-26-to-2026-07-28) | the recent deep write-ups |
| [Measured results, 2026-07-31](#measured-results-2026-07-31) | co-blocks, the third overlap mechanism; speculative entry ships; the solver reorganization — the proposal, condensed, and the probes that killed it; the CUDA match wins backported to OpenCL (−0.6 ms); two below-the-floor levers ship (−0.22 ms); the found-vs-verified gap is gone; **lolMiner measured under ncu — state-storing confirmed, its ceiling is a DRAM roofline**; the sort path's k1/k2 regression is half occupancy, half unexplained — generic stays |
| [Established limits](#established-limits) | measured properties that bound any further optimization |
| [Current focus and open leads](#current-focus-and-open-leads) | where the time goes, the lever table, the numbered leads |
| [The CUDA backend](#the-cuda-backend) | what it is, its headline, and why it is faster |
| [The CUDA backend in detail](#the-cuda-backend-in-detail) | the profiler findings |
| [What a CUDA backend was predicted to buy](#what-a-cuda-backend-was-predicted-to-buy-retained-for-calibration) | a forecast, kept to score it |

*Every section body below is collapsed behind a `Details` toggle so the document reads
as an outline; the headings stay plain markdown, so every anchor link into this page
keeps working.*

---

## Architecture
<details>
<summary>Details</summary>

The solver runs Wagner's algorithm on the ⟨144,5⟩ parameter shape: 2^25 seed elements, five
rounds, each finding pairs that collide on a 24-bit key and combining them. BeamHash III
adds a mandatory per-round `apply_mix` that folds the element's growing index tree into
word 0, re-deriving the collision key every round.

Two collision-finding paths exist. The **fused row-bucket path** is the default wherever
device memory allows; the **sort path** is the fallback for smaller cards.

</details>

### Fused row-bucket pipeline
<details>
<summary>Details</summary>

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
row-bucket needs 7.46 GiB with a 3.27 GiB largest single allocation, so smaller cards fall
back to the sort path automatically. `MXBM_ROWBUCKET=1` / `MXBM_NO_ROWBUCKET=1` override.
(The *budget* still gates on a single sort-path-sized constant, which is stricter than
this — see [open leads](#current-focus-and-open-leads).)

</details>

### Round schedule
<details>
<summary>Details</summary>

| Round | Lmix | Lout | padNum | Significant work words | Leaf payload carried |
|---|---|---|---|---|---|
| 1 | 448 | 424 | 1 | 7 | 1 → 2 raw leaves |
| 2 | 424 | 400 | 2 | 7 | 2 → 4 raw leaves |
| 3 | 400 | 376 | 4 | 6 | 4 raw → 1 u64 `leftContrib` |
| 4 | 376 | 288 | 6 | 5 | `leftContrib` → none |
| 5 | 288 | 24 | 9 | 1 | none (terminal) |

</details>

### Stored record per round
<details>
<summary>Details</summary>

Rounds 1–2 do not store work state at all: their elements are re-derivable from seed
indices, and those indices are also exactly the leaves the round needs, so the compact
record *replaces* the payload rather than adding to it (see
[the re-derivation chain](#the-re-derivation-chain)). Round 3 tried the same trick and it
was later [measured to be a loss](#retiring-the-round-3-quad-record).

| Read by | Record | u64 | Bytes | Contents |
|---|---|---|---|---|
| Round 1 | seed | 1 | 8 | `key \| index << 32` |
| Round 2 | pair | 2 | 16 | `key \| i0<<24`, `i1 \| gi<<25` |
| Round 3 | packed | 9 | 72 | 7 work words, then 4 leaves + `gi` bit-packed into 2 u64 |
| Round 4 | packed | 8 | 64 | 6 work words, meta, `leftContrib` |
| Round 5 | thin | 2 | 16 | **1** work word, meta — see [why](#the-terminal-rounds-dead-work-words) |

Seed indices are 25-bit (2^25 seeds) and `gi` is 26-bit, so the pair and quad records
pack exactly with bits to spare.

Round 3's 9 u64 are **not** stored as a 9-u64 stride: 8 of them sit at a 16 B-aligned
stride and the 9th lives in a plane behind the records, which is what keeps the 128-bit
accesses legal without paying for a padding word — see
[the round-2 alignment pad](#the-round-2-alignment-pad).

</details>

---

## A note on method
<details>
<summary>Details</summary>

Three results on this page were initially wrong because of the *measurement*, not the
code, and the failure modes generalise. They are the reason several sections here carry a
positive control:

- **An ablation that replaces a value must preserve its access footprint.** Substituting a
  tiny LDS index for a global counter priced "atomic removed **and** locality improved" as
  one number, and predicted a −2.5 ms win that was really +4.5 ms
  ([details](#eliminating-the-dense-gi)).
- **Put a positive control on every counter.** A GPU probe reading back 0 may mean "no
  fault" or "never ran". Add a counter whose value is already known — "count everything
  staged" against the known input count — and check it in the same run.
- **Assert that a patch applied.** The round-3 A/B was chased across three sessions as a
  suspected race, nondeterministic across runs with zero drop counters. It was a
  non-asserted string replace in the patch script that silently failed to match, so the
  variant's rebuild never ran and it read uninitialised LDS as work state. Every
  hypothesis reasoned from that symptom — strides, buffer sizing, out-of-bounds writes,
  the tie-break, compiler miscompilation — was wrong. Two "facts" it produced were
  artefacts and are retracted: equal-lead pairs at round 2 are **0**, not 17, and the
  round-2 → round-3 record round-trip was always byte-perfect.
- **A faster variant can be the broken one.** The bad A/B measured 3.8 ms *faster* while
  silently losing golden solutions. Gate an A/B on output counts and goldens, never on
  timing alone.

</details>

---

## Instruments: diagnostic and ablation flags
<details>
<summary>Details</summary>

Diagnostic environment variables: `MXBM_ROWBUCKET` / `MXBM_NO_ROWBUCKET`,
`MXBM_LEGACY_MATCH`, `MXBM_LDS_MATCH`, `MXBM_NO_COMPACT`, `MXBM_SORT_PROFILE`,
`MXBM_MATCH_K12` (sort path: 1 = the constant k1/k2 for r1/r2, 2 = the
occupancy-pinned k1p/k2p; ballast size via `MXBM_CL_OPTS="-DKPIN_UINTS=N"` — both
measured losses, kept as instruments),
`MXBM_ABLATE` (bit 0 skips `apply_mix`, bit 1 skips the fat emit — for phase attribution),
`LEADTIE_PROBE` (a `-D` build flag counting equal-lead pairs into the chain-drop
counter; 17 per solve out of ~134 M),
`MXBM_CL_OPTS` (extra OpenCL build options for the fused program, e.g.
`-cl-nv-maxrregcount=128`; see [register cap](#register-cap-tuning)).
Power-curve sweeps that used to be bash around `nvidia-smi` now also exist as
`--tune` (docs/usage.md): miner-loop arms inside the shipping binary, coarse pass +
knee-refining pass, end-of-sweep drift gauge, result stored per card for `--pl auto` —
usable as a research instrument wherever a drift-gauged power curve is the question.
The first full three-pass run (2026-07-31, reference card) reproduced the dedicated
sweeps: knee 231 W (refined down from the coarse 248), rung crossover interpolated at
~171 W vs the sweeps' ~173, drift +0.5 % where un-gauged sessions had shown ~2 %/arm.
One grid miss surfaced and was closed the same day: pass 3 reuses the coarse caps,
whose 37 W spacing skips ~160 W — so the verdict named 211 W stock as best efficiency
(0.256 sol/s/W) and could not see the true record between its rung points (160 W +
rung, 0.264). The knee and rung-band verdicts, the ones `--pl auto` acts on, were
unaffected. Fix: a fourth pass that aims `tune_refine_caps` (budget 4, so ~15 W
steps) at the best-efficiency point on its own memory clock; on this card's series
that grid is 115/130/145/160 — the record cap lands on it exactly
(test_tune pins the grid). Re-running `--tune` regenerates the store with it.
Energy is measured, not integrated, since 2026-07-31: the card's millijoule counter
(NVML `nvmlDeviceGetTotalEnergyConsumption`, no root to read) backs `--benchmark`'s
J/solution line, `--tune`'s per-point draw, `/summary`'s `Energy_J`, and
`bench_energy_mj` in `benchmarks/lib.sh` — bracket a window, diff the counter, divide;
the 5 Hz `power.draw` sampler remains only as the fallback for cards without it.
The composition of the two replay instruments with the counter is
`benchmarks/stage_power.sh` since 2026-07-31: `cuda/pipeline` brackets the counter
around its own timed loop (init excluded — a bash-side bracket would bias the small
stages 5–50 %), so per-stage joules are exact, `E_s = (J_N − J_1)/(N−1)`. The stock
re-measurement on the current kernels tightened both closures (stage times sum to
100.1 % of the solve, stage energies to within 0.8 % of the counter's whole-solve
figure) and re-confirmed the stock verdict with exact joules: no watt-hog — every
stage draws 283–285 W except round 3 at 275.4 W, and r3 runs the highest clock of
any stage (2745 MHz), the DRAM-bound signature. 9.63 J/solve ÷ 1.96 verified =
4.91 J/solution at stock; the refreshed stage table is in performance.md. The same
attribution at the capped/rung operating points is
[the eco attribution](#per-stage-attribution-at-the-eco-points-the-floors-time-is-round-2s),
which named round 2 as the floor's diet target.
The row-bucket
kernels' own knobs ride the same option string: `-DLDS_FCAP` / `-DLDS_FCAP_R1`
(group caps, default 320/288), `-DLDS_FTAB` (chain-table entries, 128),
`-DLDS_PERFECT_TAB=0` and `-DLDS_SPILL=0` — together those five restore the
pre-2026-07-31 configuration for A/Bs, see
[the backport](#the-cuda-match-wins-backported-to-opencl-perfect-table--spill--per-round-caps-06-ms).

`-DMXBM_SUBPASS=1` builds the sub-pass kernel (block = whole bucket, sub-masks swept out
of shared, keys read once); `-DMXBM_FCAP=N`, `-DMXBM_TAB=N`, `-DMXBM_SKEY=N` size it.
`MXBM_FORCE_TIMING=1` times a configuration that fails the gate — only ever valid when the
drop count is quoted with it, since drops remove downstream work and flatter the result.

CUDA-side: `MXBM_BB` / `MXBM_SM` pin the row-bucket geometry (both backends; an explicit
choice is never stepped down under the user), `MXBM_OCC` prints the worst bucket occupancy
in units of the reservation's sigma, `MXBM_ROUND_REPS="R:N"` and `MXBM_ENTRY_REPS=N` replay
one stage in place for per-round attribution. Build flags: `-DMXBM_STCS=1` (streaming
stores), `-DMXBM_WARPAGG=1` (warp-aggregated `gi`), `-DMXBM_CPASYNC=1`, `-DMXBM_WG=N`,
`-DMXBM_FCAP=N` (group cap → occupancy), `-DMXBM_R2_FULL=1`.

`./cuda/pipeline N --fuse` runs the co-tenant entry experiment (`MXBM_CO_ROUND` picks the
host round, default 3); `--overlap` runs the two-stream one. Both are null — kept because
a null with a mechanism is what stops the idea being re-proposed.

Overlap instruments added 2026-07-31 (see [co-blocks](#co-blocks-the-third-overlap-mechanism-works--and-it-is-worth-04-ms-not-14)):
`MXBM_COB_S=S` turns `--fuse` into the co-BLOCKS form (every S-th block of the host
round's grid seeds; S−1 must divide nb≪sm); `MXBM_CO_ROUND=34` splits the co-work across
r3 and r4; `MXBM_COB_DUMMY=1` gives the co-blocks zero work (prices displacement alone,
real entry launched separately). `--pipe2` runs the two-solve software pipeline with
`fused_pair` (r4(i) ‖ r1(i+1) in one launch); `MXBM_P2_VAR=1` gives r1 two thirds of the
residency. On the shipped miner, `MXBM_NO_SPEC=1` disables speculative entry
co-scheduling for A/Bs. `-DMXBM_MB_SEED=N` / `-DMXBM_MB_RD2=N` force a register budget
(`__launch_bounds__` minBlocks) on r1 / r2 — measured null, shared memory caps first.
`-DMXBM_PAIR128=0` restores r2's two scalar pair-record loads
([shipped on 2026-07-31](#two-below-the-floor-levers-clear-noise-on-cuda-r2s-pair-record-in-one-ld128-and-the-terminal-round-joins-the-perfect-table-022-ms)).
`MXBM_VERIFY_STATS=1` classifies every CPU-verify outcome by reject reason on either
backend and prints tallies at exit
([the found-vs-verified section](#the-found-vs-verified-gap-is-gone-0-of-3935-candidates-rejected)).

**CUDA attribution builds** — `-DMXBM_ABL_EMIT=R` (narrow round *R*'s payload to 16 B),
`-DMXBM_ABL_DERIVE=1|2|3` (round 1's seed / round 2's rebuild), `-DMXBM_ABL_MIX=R` (skip
`apply_mix`), `-DMXBM_CO_NOSCATTER=1` (co-tenant computes but stores nothing, and
`--fuse` then keeps launching the real entry so the co-tenant is purely additive).
**Results are intentionally wrong** and the KAT gate is bypassed for these builds only. Two rules they were designed around, both learned the hard way here:
substituted work must keep the *global* access footprint and the bucket distribution
(check `drops == 0` and `MXBM_OCC`), and removing a store must not let DCE remove the
compute feeding it — hence `ABL_EMIT` narrows rather than skips, with rounds 1 and 4 as
zero-delta positive controls.

**Phase ablation (`ABL`).** Built through `MXBM_CL_OPTS`, e.g.
`MXBM_CL_OPTS="-DABL=1 -DABL_MODE=2"`. Results are *intentionally wrong* when enabled —
it attributes time, it does not compute. Bits: 1 = skip the emit payload write (counters
and back-refs kept, so element counts stay representative), 2 = skip the re-derivation
rebuild, 4 = skip the whole match body, 8 = skip back-refs, 16 = skip the mix, 32/64 =
skip the per-bucket / `gi` atomic, 128 = skip the combine.

`ABL_MODE` selects **one round** by its `LMODE` (1 = r1 seed, 4 = r2, 5 = r3, 2 = r4).
This matters: ablating a round corrupts the bucket distribution of every round *after*
it, so only the ablated round's own number is meaningful. Two further traps found in
use — ablating the match body also makes the rebuild dead code (so it under-reads), and
ablating the combine leaves `c[0] = a[0]`, which sends every child of a group to one
bucket and measures pathological contention rather than the removed work.

This replaces the `MXBM_ABLATE` that the row-bucket rewrite removed while this document
still listed it.

</details>

---

## What worked

### Sort-based collision finder
<details>
<summary>Details</summary>

Replaced the atomic-bucket scatter + all-pairs match with a stable radix sort of
(key, index) pairs and a coalesced run-scan emit. Also fused the sort-pair emission into
the mix, deleting a whole strided pass over the work array. 245 → 225 ms across P1–P4.

</details>

### Tiled radix sort
<details>
<summary>Details</summary>

CUB-style 4-bit × 6-pass sort with per-thread private histograms and an in-place
cross-thread prefix; 8 items/thread measured optimal (16 overflows 48 KB LDS).
Sort cost per round 18 → 9.5 ms.

</details>

### Fixed-width compaction
<details>
<summary>Details</summary>

`bh3_combine` shifts out the 24 matched bits and masks to `Lout` each round, so the
significant work words shrink on the schedule **[7, 7, 6, 5, 1]**. Storing only those
words is bit-exact (upper words are provably zero). The win requires **compile-time**
widths — see [runtime loop bounds](#runtime-loop-bounds). Measured 224.4 → 214.5 ms on
the sort path, and the same schedule was later applied to the row-bucket path.

</details>

### Fused row-bucket pipeline
<details>
<summary>Details</summary>

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

</details>

### L5-thin emit
<details>
<summary>Details</summary>

Round 4 was the most expensive round because it emitted round-5 children carrying a full
9-leaf payload. Round 5 is terminal — it never mixes, and `recover` walks back-refs — so
those leaves are dead weight. The kernel now decouples *build* from *store*: it builds
`max(padNum_next, sOut)` leaves for the child's own mix but stores only `sOut`.
Round 4: 37.2 → 29.3 ms; pipeline 151.9 → 139.1 ms.

</details>

### Per-round compaction on the row-bucket path
<details>
<summary>Details</summary>

The `[7,7,6,5,1]` schedule applied to the fused kernel as compile-time width variants
(r3 = 7→6, r4 = 6→5), with round 5 reading at stride 5. Round 4: 29.3 → 26.8 ms;
survivor scan 6.1 → 4.6 ms.

</details>

### Async de-bubble
<details>
<summary>Details</summary>

`run1d` blocks on `clFinish` after every kernel. In production (non-verbose) the
row-bucket pipeline now enqueues the whole chain asynchronously on the in-order queue and
drains once at the survivor readback. Only ~2 ms: the pipeline is genuinely GPU-bound, so
those waits were mostly waiting on real work.

</details>

### Compact leftContrib
<details>
<summary>Details</summary>

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

</details>

### Packed element record
<details>
<summary>Details</summary>

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

</details>

### Seed re-derivation
<details>
<summary>Details</summary>

Round-1 elements are **seeds**: each is fully determined by its 25-bit index via
7 × `siphash24` + `apply_mix`. We were storing the resulting 72 B record and reading it
straight back. GPUs are compute-rich and memory-poor, and the isolated measurement is
lopsided:

| | ms |
|---|---|
| Recompute only (7 × siphash24 + apply_mix) | 2.47 |
| Entry storing 8 B `(index, key)` | 2.50 |
| Entry storing the 72 B packed record | 18.16 |

Recompute costs **14 %** of what storing the record costs. The round-1 bucket record is
now a bare 8 B `(index << 32) | key`, and a fused variant (`LMODE_SEED`) re-derives the
56 B work state while staging into LDS. The key is still stored so the sub-mask scan
stays cheap; only the ~1/8 of a bucket each pass owns is expanded, so every element is
derived exactly once across the 8 passes.

**114.8 → 102.7 ms.** The entry pass drops 19.2 → 2.7 ms, and round 1 pays +3.6 ms of
recompute for a 15.7 ms saving.

This is the first confirmation of the compute-for-memory trade that BeamHash III's 3 GB
design target implies (see [HW_REQUIREMENTS.md](HW_REQUIREMENTS.md)): the solver is
bandwidth-bound, so paying arithmetic to avoid moving bytes wins.

</details>

### The non-divergent expand
<details>
<summary>Details</summary>

The single most useful structural finding so far, and the key that unlocked
[the re-derivation chain](#the-re-derivation-chain).

The fused kernel stages a bucket in a **sub-mask filtered** loop:

```c
for (uint p = lId; p < cnt; p += LDS_WG) {      // cnt ~ 2048, 8 iterations
    uint key = ...;
    if ((key & (submaskCount - 1u)) != mask) continue;   // ~7/8 of lanes leave
    uint pos = atomic_inc(&gcount);
    /* ...work here runs at ~4 of 32 lanes per warp... */
}
```

With `submaskBits = 3` only ~1/8 of lanes survive the filter, so **anything expensive
placed there executes at ~4/32 lane utilisation**. Immediately afterwards the kernel runs
a second loop over the ~256 staged elements — `for (pos = lId; pos < total; pos += 256)` —
which is a single iteration with **every lane active**.

Moving the seed derivation from the first loop to the second is therefore free of any
structural cost: same seeds, same LDS, no extra barrier (the index was already parked in
`lgi`), 8× the SIMD utilisation.

**Round 1: 26.4 → 18.5 ms; pipeline 103.6 → 95.3 ms.** Round 1 became the *fastest* round,
which is what it should always have been — its input record is 8 B against every other
round's 64–72 B.

This also explains, retrospectively, why
[the first round-2 attempt](#round-2-re-derivation-first-attempt) failed so badly: it was
never an arithmetic problem, only a placement one.

</details>

### The re-derivation chain
<details>
<summary>Details</summary>

With the divergence removed, "store indices, rebuild the element" extends up the pipeline.
Each round's element is a combine of two of the previous round's, so an element at round
*r* is determined by 2^(r-1) seed indices — **and those indices are exactly the leaves that
round already has to carry**. The compact record therefore replaces the leaf payload and
the work words together, rather than adding to them:

| | Record | Emit (round *r*−1) | Stage read (round *r*) |
|---|---|---|---|
| Round 2 | 16 B pair (2 indices) | 72 → 16 B | 72 → 16 B |
| Round 3 | 24 B quad (4 indices) | 80 → 24 B | 80 → 24 B |

The rebuild cost was measured **in situ** before either was implemented, using a probe
that re-derives the element and overwrites the staged work state *while still reading the
old record* — so the delta is the recompute alone, and it self-verifies against the
goldens. Round 2's rebuild measured **+0.4 ms**, against **+23.5 ms** for the identical
arithmetic in the divergent slot.

> A probe like this must be **liveness-checked**. A golden pass alone proves nothing: if
> the compiler had eliminated the recompute, `lwork` would simply retain the correct
> loaded value and the goldens would pass anyway. Writing garbage through the probe and
> confirming the goldens *break* is what establishes the measurement is real. (A first
> attempt at this control — perturbing `Lout` 424 → 425 — was a no-op, because
> `bh3_combine` already leaves bit 424 zero.)

Measured results:

| | Round *r*−1 emit | Round *r* stage+rebuild | Pipeline |
|---|---|---|---|
| Round 2 pair record | 18.2 → 12.1 ms | 24.2 → 21.3 ms | 95.3 → 86.8 ms |
| Round 3 quad record | 21.3 → 15.0 ms | 24.2 → 26.5 ms | 86.8 → 83.2 ms |

**The chain stops at round 3.** Round 2's 14 siphashes/element hid almost entirely inside
the kernel's existing memory stalls (+0.4 ms); round 3's 28 exceeded what is hideable and
cost ~9 ms of exposed compute, more than the traffic it removed locally — round 3 is only
net-positive because of what it saves on *round 2's emit*. Round 4 would need 56
siphashes for a 64 → 32 B record: twice the compute for half the saving, on the wrong side
of a boundary round 3 already crosses. It was not implemented. See
[the compute-hiding budget](#established-limits).

As a side effect the widest bucket stride (10 u64) retired, so both ping-pong buffers
shrank 20 % and a full search now fits in **6.95 GiB** (222 B/element), down from 8.36 GiB.
[HW_REQUIREMENTS.md](HW_REQUIREMENTS.md) had predicted exactly this — it listed
"index-only storage with re-derivation" as a route to the 3 GB target and argued memory
efficiency and throughput were probably the same problem. Both moved together.

</details>

### Compile-time round constants
<details>
<summary>Details</summary>

The largest single win of the session, and a **third recurrence** of the failure mode
already recorded twice under [runtime loop bounds](#runtime-loop-bounds).

`Lout`, `Lmix`, `padNum` and the leaf counts `sIn`/`sOut`/`sBuild` were **runtime kernel
arguments**. Every one is a per-round constant. Passing them at runtime reached into
`bh3_apply_mix`, which runs once per emitted child:

```c
padNum = ((512 - Lmix) + 24) / 25;      // runtime
for (uint i = 0; i < padNum; ++i) {     // runtime bound -> no unroll
    pos = Lmix + i*25; word = pos >> 6; // runtime
    t[word] |= v << sh;                 // DYNAMIC INDEX into t[8]
```

A dynamically indexed private array cannot live in registers; it goes to **local memory**,
which on NVIDIA is global-memory-backed. So every child was paying a spill round-trip,
~33.5 M times per round. Making the constants compile-time macro parameters of
`FUSED_LDS` unrolls the tree loop, constant-folds `word`/`sh`, and keeps `t[8]` in
registers:

| | r1 | r2 | r3 | r4 | pipeline |
|---|---|---|---|---|---|
| runtime args | 11.9 | 15.0 | 26.5 | 24.9 | 83.2 |
| compile-time | **7.5** | **10.9** | **21.5** | **10.9** | **56.2** |

The terminal round-5 kernel had the same fault and was fixed in the same way
(`LDS_R5LOUT`): at `Lout(5) = 24`, `bh3_combine`'s masking loop folds to "mask word 0,
zero words 1–6". Survivor pass 4.4 → 4.1 ms. The entry kernel was already passing
literals. That is the whole default path audited.

Round 4 more than halved — it has `padNum(5) = 9`, the longest tree loop.

`bh3.cl` is **not modified**: the primitives are simply called with constants. The cost
is that the shipped kernels now *ignore* the matching runtime arguments, which is a
silent-drift hazard, so `fused_consts_for()` centralises the table and
`test_gpu_rounds` pins it against both the `lds.cl` literals and the independent `ref::`
round table (verified to fail when perturbed). `round_fused_lds` keeps runtime
parameters so `test_fused_round` still validates the mechanism generically for r=1..4.

**The generalised lesson:** "compile-time widths in the hot path" had been read as being
about the *word* loops. It is about **every loop bound the hot path can reach, including
through a callee**. This one hid inside a primitive that looked untouchable.

Found with the `ABL` phase ablation (below), which attributed round 4 as: emit payload
5.0 ms, per-bucket atomic 1.9, `gi` atomic 0.8, back-refs 0.5 — and **17.3 ms in
`apply_mix` alone**, against the "~10–15 % mix" this document had previously assumed.

</details>

### Retiring the round-3 quad record
<details>
<summary>Details</summary>

**A shipped optimization that a later change turned into a loss.** The
[quad record](#the-re-derivation-chain) won −3.6 ms at 86.8 ms. After
[compile-time constants](#compile-time-round-constants) took the pipeline to 56 ms it was
costing +2.2 ms, because that change removed the memory stalls its 4-seed rebuild had been
hiding inside. Measured, with both configurations byte-identical on the goldens:

| round 2 + round 3 | ms |
|---|---|
| quad record (rebuild in round 3) | 10.7 + 21.1 = **31.8** |
| full record (no rebuild) | 14.7 + 14.3 = **29.0** |

So round 2 emits the full packed record again and round 3 reads it. `LMODE_RD3`,
`round_fused_rd3` and the `rd3_*` packing helpers are gone.

**The round-2 pair record was re-tested at the same time and survives** — 5.1 ms ahead
(r1+r2 = 22.7 vs 27.8). Worth noting *why* the two differ: round 2 costs the **same**
15.2 ms with or without its rebuild, so its 2-seed recompute is entirely hidden, and the
whole win is round 1's narrower emit. Round 3's 4-seed rebuild was never hidden.

Per-set bucket-array sizing landed here too (an old open lead). Round *r* writes set
`r&1`, so set 0 carries the round-2/round-4 outputs (10 u64 at the time) and set 1 the
round-1/round-3 outputs (8). Sizing them separately gives back most of what the wider
record costs: 6.95 → 8.36 → **7.65 GiB**. `fb_set_stride()` derives both widths from
`kFbStride` so they cannot drift.

</details>

### The redundant `lead` field
<details>
<summary>Details</summary>

The generic packed record stores `meta = (gi << 32) | lead`. For the round-2 → round-3
record `lead` **is** `ctree[0]`, which is already leaf 0 of the payload — it was being
written and read twice. Removing it leaves 4 leaves (25 bits each) + `gi` (26) = 126 bits,
which fits two u64 with the meta word deleted entirely:

```
p0 = l0 | l1<<25 | (l2 low 14)<<50        p1 = l2>>14 | l3<<11 | gi<<36
```

Record 80 → 72 B. Round 2 15.2 → 14.0 ms (emit), round 3 15.5 → 14.5 ms (stage read);
pipeline 54.0 → 52.7 ms and VRAM 7.65 → 7.30 GiB.

*The same redundancy does not exist at the other boundaries: round 4's payload is the
`leftContrib` rather than leaves, so its `lead` is not duplicated anywhere.*

</details>

### The terminal round's dead work words
<details>
<summary>Details</summary>

Round 4 was emitting all five significant work words of its children; round 5 reads four
of them and uses none.

The terminal test is `z = OR(c[0..6])` after `bh3_combine` at `Lout(5) = 24`. That `Lout`
forces `c[1..6]` to zero and masks `c[0]` to 24 bits, and

```
c[0] = ((x[0] >> 24) | (x[1] << 40)) & 0xFFFFFF
```

where the `x[1]` term shifts zeros into bits 0..23. So the whole test reduces to **"bits
24..47 of `a[0]^b[0]` are zero"**, and the collision key is word 0's low 24 bits — from
the same word. Words 1..4 are dead weight. Checked over 200 000 random pairs before
changing anything, then gated on the survivor count staying at exactly 3.

Round 4's record: 48 → **16 B**. Round 4 10.9 → 8.4 ms (emit), survivor pass 4.1 → 2.0 ms
(stage read), pipeline 52.7 → 47.5 ms.

This is the work-word analogue of [L5-thin](#l5-thin-emit), which did the same for round
5's *leaf* payload. **And the generalisation says it was the only instance:** every round
except the terminal one feeds its combine result through `apply_mix`, which sums rotations
of all seven words, so no earlier round can drop any. Round 5 is the only round with no
mix.

</details>

### The record-redundancy audit
<details>
<summary>Details</summary>

Prompted by [the redundant `lead`](#the-redundant-lead-field) hiding in plain sight for the
whole project. Every stored field, against what actually reads it. Besides the terminal
round above, it came up **empty** — the records are at their information floors:

| Record | Bits needed | u64 | Slack |
|---|---|---|---|
| round-1 seed | 25 idx + 24 key = 49 | 1 | 15 |
| round-2 pair | 25+25 idx + 24 key + 26 gi = 100 | 2 | 28 |
| round-3 packed | 400 work + 100 leaves + 26 gi = 526 | 9 | 50 |
| round-4 packed | 376 work + 64 contrib + 25 lead + 26 gi = 491 | 8 | 21 |
| round-5 thin | 64 work + 25 lead + 26 gi = 115 | 2 | 13 |

Every one of these is `ceil(bits/64)` — no record can lose a u64 without losing a field.
Two fields were checked specifically and are **not** removable: `gi` is needed for the
back-reference rows even where the `(lead, gi)` tie-break never fires (and replacing it
with the bucket slot [costs 4.5 ms](#eliminating-the-dense-gi)); and round 3 needs *all*
eight of its leaves, six for the mix and all eight for `contribOut`, so the
[leftContrib](#compact-leftcontrib) fold cannot be applied a level earlier.

**The sort fallback was audited too**, and had one real finding: a **692 MB back-ref
array that nothing read or wrote**. An element's lead is leaf 0 of its own prefix, so
`round_match` stopped writing `all_lead` some time ago — `round.cl` says so in two places
— but the `5 × capacity` array outlived the write by several rounds of cleanup. Still
allocated, still bound, never touched; only tests wrote it, as a fixture feeding an input
the kernel had stopped reading. Sort path **285 → 264 B/element (8.89 → 8.25 GiB)**. The
row-bucket path never allocated it.

That is the *same redundancy class* as [the round-3 `lead` field](#the-redundant-lead-field)
— an element's lead duplicating its own leaf 0 — which is what prompted looking there at
all. Two independent instances of one mistake suggests the pattern, not the instances, is
the thing to remember: **a field derived from another field in the same record will not
announce itself; it has to be looked for deliberately.**

The rest of the sort path is clean, with one deliberate exception: `leaves[2]` is a fixed
9 uints where per-buffer sizing would save 4 B/element (~138 MB), because `leaves[1]`
holds round 5's 9-leaf prefix and `leaves[0]` round 4's 8. Left alone — it complicates the
ping-pong for little gain on a path that measures 214.6 ms against row-bucket's 47.5.

The audit was of *contents*, though, and one record was wider than its contents for a
reason the audit could not see — see [the round-2 alignment pad](#the-round-2-alignment-pad).

</details>

### The round-2 alignment pad
<details>
<summary>Details</summary>

The audit above put round 3's input record at its 9-u64 information floor. The CUDA port
then stored it in a **10**-u64 stride, so that `od*8` is always 16 B aligned and the
[128-bit accesses](#the-cuda-backend) are legal. That pad is not free. Round 2 writes it
and round 3 reads it back, and because records are contiguous the pad's bytes fall inside
sectors that are fetched anyway: **8 B × 2^25 × 2 = 537 MB per solve, 4.0 % of all DRAM
traffic, carrying nothing.** It was invisible to a contents audit and invisible to a
per-kernel profile — only the [compulsory-traffic table](performance.md#the-memory-traffic-is-compulsory)
made it show up, as the one line where measured exceeded compulsory.

Both properties are obtainable at once. The record is stored as 8 u64 — a stride that is
16 B aligned for *every* slot, not every other one — and the 9th word moves to its own
plane laid out after all the records. The kernel derives that plane's offset from the
geometry it already has (`nb = 1 << bucket_bits`, times the bucket capacity), so no
parameter changes, and the instruction count is identical either way: 4 × ST.128 plus one
ST.64, and the matching loads.

| | before | after | |
|---|---|---|---|
| stored record | 80 B | **72 B** | its actual contents |
| CUDA footprint | 7.82 GiB | **7.46 GiB** | now identical to the OpenCL path |
| round 3 | 9.99 ms | **9.55 ms** | **−4.4 %** |
| round 2 | 10.56 ms | 10.68 ms | +1.2 % |
| end-to-end | 35.16 ms | **34.96 ms** | −0.20 ms, −0.6 % |

**Do not read this as a throughput win.** −0.6 % is at the edge of what the
whole-solve harness resolves: `mxbm --benchmark` measured 35.1 ms/solve both
before and after. The A/B above is believable only because it was run as four
alternating pairs of 700 solves and the new build won all four, and because the
round-level figures explain it. **The reason to keep the change is the 0.36 GiB**,
which is not marginal and is what lowers the CUDA backend's VRAM threshold.

Round-level figures are from `MXBM_ROUND_REPS=R:9` (`benchmarks/stage_power.sh`), which
replays one round nine times inside the solve that produced its input and so measures a
0.4 ms change as a 3.5 ms one; both A/Bs reproduced to ±0.05 ms. End-to-end is the median
of 700 solves, and the new build won all four alternating pairs.

**Why round 2 loses what round 3 gains — most of it.** Round 2's children land in any of
65 536 output buckets, so its stores are already scattered across the whole array; the
split makes each thread write two distant destinations instead of one, doubling the
distinct sectors in flight. The first version cost round 2 **+3.5 %**, and hoisting the
duplicated `cb * out_bucket_cap + cpos` multiply out of the two address computations
recovered two thirds of that.

The asymmetry is the part worth keeping. Round 3 moves 5.1 GB in 10 ms = **511 GB/s**,
which is the card's ~510 GB/s achievable ceiling, so it converts bytes saved into time
saved nearly 1:1. Round 2 runs at 53 % of peak bandwidth and is limited by something else,
so it pays for the extra stream and is not repaid in bytes. **Removing bytes only buys
time in the rounds that are actually bandwidth-bound** — which, per the
[stage table](performance.md#the-card-is-at-its-power-limit-in-every-kernel), is rounds 3 and 4 and not
rounds 1 and 2.

</details>

### Row-bucket geometry
<details>
<summary>Details</summary>

`bucketBits = 14`, `submaskBits = 3` was chosen when the fused path was built and never
revisited, though almost everything it was balanced against has since changed.

The staging loop **rescans each bucket 2^submaskBits times**, keeping ~1/8 of the elements
per pass, so `submaskBits` multiplies redundant scan work directly. It cannot simply be
lowered: the group size is `mean_bucket / 2^submaskBits`, and `LDS_FCAP` must cover its
**tail**, not its mean. Dropping to `(14, 2)` puts the group mean at 512 and needs
`FCAP ≈ 693`, which does not fit in 48 KB — measured as **773 655 group drops**.

But the group size depends only on the *sum* `bucketBits + submaskBits`. Halving the
bucket and the sub-mask together holds the group fixed — same `FCAP`, same LDS — while
halving the rescan. Swept twice, before and after [capacity](#bucket-capacity) was fixed:

| (bb, sm) | rescan | ms | |
|---|---|---|---|
| (14, 3) | 8× | 47.5 | original |
| (15, 2) | 4× | 42.6 | |
| **(16, 1)** | **2×** | **40.4** | **now**, 7.46 GiB |
| (17, 0) | 1× | 40.4 | no gain over 2×, and 8.35 GiB |
| (16, 2), (17, 1) | | 46.6, 45.4 | group falls to 132, half the workgroup idle |
| (18, 0) | | — | single allocation exceeds the device limit → sort-path fallback |

Every round gained twice over: r1 7.5 → 6.4, r2 14.0 → 13.0, r3 14.5 → 12.3, r4 8.4 → 6.5.

**(16, 1) only became reachable after the capacity fix** — at the old capacity its single
allocation was 4.3 GB against the device's 3.9 GB limit, so `want_rowbucket` silently fell
back to the sort path. That is the 214.9 ms that appeared in an earlier version of this
table: not a slow geometry, a fallback.

**(17, 0) is the informative negative.** Removing the redundant rescan *entirely* buys
nothing over halving it, for 0.9 GiB more — so this lever is spent, and `submaskBits = 1`
is where it stops paying. Exposed as `MXBM_BB` / `MXBM_SM`, because the right balance
depends on the record widths and those moved four times in one session.

</details>

### Bucket capacity
<details>
<summary>Details</summary>

`fb_cap_for` was dimensionally wrong. Occupancy is Poisson(mean) so σ = **√mean**, but the
headroom term was `mean/4`, which scales with the mean — over-reserving, and worsening as
buckets coarsen. Measured with a new `MXBM_OCC` probe over 52 round-solve samples:

```
mean 1024   sd 32   max = mean + 4.07..4.35 sd   (tightly concentrated)
old cap = mean + 17.3 sd
```

The concentration is expected: the max of *n* Poisson draws sits near
`mean + σ√(2 ln n)`. Now `mean + 8√mean + 32` — still an enormous margin (a Poisson tail
at 8σ is ~1e-15 per bucket) at 14 % less memory. On its own it is memory-only
(7.83 → 6.88 GiB, speed unchanged); its value is that it unlocked the geometry above.

</details>

---

## What didn't work
<details>
<summary>Details</summary>

Each of these was measured and reverted. They are recorded so they are not retried.

</details>

### SoA word planes
<details>
<summary>Details</summary>

One buffer per element word, skipping zero planes. Measured **only 1 % better** than
fixed-width AoS (14.22 vs 14.42 ms) — not worth a full-pipeline rewrite. An AoS element
write touches ~1 cache line; the same element in SoA fans across W buffers.

</details>

### Runtime loop bounds
<details>
<summary>Details</summary>

The first compaction attempt used runtime-variable word-loop bounds (`w < inW`). This
defeats compiler unrolling and cost **+24 to +27 ms**. Compile-time widths are mandatory
everywhere in the hot path; this failure mode recurred twice.

</details>

### Un-fused LDS path
<details>
<summary>Details</summary>

Bucket-scatter as a *separate* pass, then an LDS collide pass: **244 ms, slower than the
215 ms sort path**. The separate fat scatter consumes exactly the coalesced-gather win it
enables. This is why the row-bucket rewrite had to fuse.

</details>

### Stride-1 lead array
<details>
<summary>Details</summary>

Publishing leaf 0 into a dedicated dense array so the pair-ordering read would be
sequential. Goldens stayed correct but match **regressed 158.9 → 176.3 ms**. Leaf 0 is
cache-adjacent to the leaf-prefix data the same kernel already reads, so splitting it out
added a second memory stream. The lead read and prefix build are cache-coupled; neither
can be peeled off alone.

</details>

### Occupancy tuning
<details>
<summary>Details</summary>

The fused kernel uses ~41 KB of the 48 KB LDS → exactly **1 workgroup/SM (~17 %
occupancy)**. Two attempts to buy occupancy both lost:
- Halving the group cap and adding a sub-mask bit (LDS 41 → 22 KB, 2 wg/SM): **slower**,
  because a finer sub-mask re-scans each bucket 16× instead of 8×.
- De-fusing to free the 12 KB leaf array only reaches 29 KB — still >24 KB, so still
  1 wg/SM — while re-introducing the +17 ms scattered leaf gather.

Decisively: [decoupled scatter](#decoupled-scatter) proved occupancy is irrelevant to the
dominant cost anyway.

</details>

### Dense key array
<details>
<summary>Details</summary>

A dense per-slot key array so the sub-mask scan reads 4 B coalesced instead of striding
into the 56 B work records. **Slower (133 → 146 ms)**: the redundant strided key scan is
already L2-served and was never DRAM-bound, so the extra array was pure added traffic.

</details>

### Decoupled scatter
<details>
<summary>Details</summary>

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

</details>

### Global gi atomic
<details>
<summary>Details</summary>

Suspected the per-child `atomic_inc` on a single global counter was serializing the emit.
Stubbed it out: **no speedup** — NVIDIA's atomic aggregation already handles it. The
per-bucket counters were tested too (`scatter_noatomic`): removing them made the scatter
**slower** (235 vs 260 GB/s), because `atomic_inc` hands out sequential in-bucket slots,
which is *better* locality than a computed slot.

</details>

### Shared-memory magazine
<details>
<summary>Details</summary>

Staging children in LDS per destination bucket and flushing coalesced. Infeasible at our
fan-out: a workgroup produces ~256 children spread over 16384 buckets, so magazines hold
~1 element and never fill. Confirmed by the literature (local-reorder needs ≤256 bins) and
by Wilke Trei's own FishHashMiner, which contains no magazine.

</details>

### SoA LDS staging
<details>
<summary>Details</summary>

The LDS staging array used AoS indexing `lwork[pos*7 + w]` — a stride of 7 ulongs = 14
uints, and `gcd(14, 32 banks) = 2`, so every access was nominally a 2-way bank conflict.
Switching to plane-major SoA (`lwork[w*CAP + pos]`) measured **125.7–125.9 vs 124.8 ms —
reproducibly ~1 ms slower**. The dominant LDS accesses are in the collision-walk, where
`leftPos`/`rightPos` come from hash-chain traversal and are scattered under either
layout; only the staging loop has consecutive indices, and it is the minority of
accesses. (Note this is LDS layout only — the *global* packing win above is a separate,
real effect.)

</details>

### Round-2 re-derivation (first attempt)
<details>
<summary>Details</summary>

> **Superseded — this was diagnosed and is now shipping.** The conclusion below ("round 2
> is past the threshold") was **wrong about the cause**. The arithmetic was never the
> problem; its *placement* was. Re-measured in the
> [non-divergent expand](#the-non-divergent-expand), the same recompute costs **+0.4 ms
> instead of +23.5 ms**, and round 2 now ships a 16 B pair record
> ([the re-derivation chain](#the-re-derivation-chain)). The record is kept because the
> reasoning error is the instructive part: an in-situ measurement was taken, it was
> genuinely reproducible, and it still supported the wrong conclusion — because the
> kernel has *two* candidate sites and only one had been tried.

The natural extension of [seed re-derivation](#seed-re-derivation): a round-2 element is
one combine away from two seeds, so it could be stored as a 16 B
`(leftIdx, rightIdx, key, gi)` record instead of the 72 B packed one — and its two leaves
*are* those indices. Isolated, the arithmetic looked clearly profitable:

| | ms |
|---|---|
| Re-derive a round-2 element (2 seeds + 2 mixes + combine + mix) | 5.27 |
| Traffic it removes (emit 72→16 B, plus round 2's stage read) | ~11.4 |

**Implemented and reverted: round 2 went 23.2 → 46.7 ms** (total 107 → 131). The
re-derivation cost **4.5× its standalone measurement** once inside the fused kernel.
Aliasing the combine output to cut 7 registers changed nothing (48.3 ms), so register
count alone is not the explanation; the staging loop's sub-mask filter leaves only ~1/8
of a warp's lanes active during the recompute, and at round 2's working-set size that
divergence plus the added occupancy pressure swamps the traffic saved.

**The lesson that survives:** a standalone measurement of recomputation cost is not
transferable into the fused kernel — measure **in situ**. The lesson that did *not*
survive is the threshold claim; "in situ" turned out to mean *in the right loop*, and
"measured in situ" is not the same as "measured in the best available placement".

</details>

### Deferring the stage read
<details>
<summary>Details</summary>

[The non-divergent expand](#the-non-divergent-expand) won by moving compute out of the
sub-mask-filtered loop, so the obvious next step was to move the **64–72 B record load**
out too — park the slot index, read the record in the all-lanes loop. **Slower** (r2
24.3 → 26.2, r3 24.5 → 25.9, r4 25.1 → 27.5 ms).

The loads were never the problem. The staging loop runs **8 iterations**, so its loads
already overlap across iterations; the all-lanes loop runs **one**, so deferring collapses
that memory-level parallelism and adds an LDS round-trip on top. Divergence costs
*compute*, not *bandwidth* — the two halves of the staging loop want opposite treatment,
and that is why only the recompute moved.

</details>

### Register cap tuning
<details>
<summary>Details</summary>

Round 3's rebuild costs ~9 ms of exposed compute where round 2's cost 0.4 ms — superlinear
enough to suspect the compiler was spilling, since it budgets registers for an occupancy
this LDS-bound kernel (1 workgroup/SM) can never reach. Swept
`-cl-nv-maxrregcount` over {96, 128, 168, 200}:

| Cap | r1 | r2 | r3 | r4 | Σ |
|---|---|---|---|---|---|
| default | 12.1 | 15.1 | 26.5 | 25.6 | 79.3 |
| 128 | 12.2 | 17.1 | 26.1 | 24.1 | 79.5 |
| 168 | 13.2 | 17.1 | 25.2 | 24.1 | 79.6 |

A wash — r3/r4 gain what r2 loses. **Spilling is not the explanation**; the rebuild is
genuinely more arithmetic than the kernel's memory stalls can hide. Retained as the
`MXBM_CL_OPTS` diagnostic knob, defaulting to empty.

</details>

### Two-level bucketing
<details>
<summary>Details</summary>

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

</details>

### General compact mix-state
<details>
<summary>Details</summary>

The `leftContrib` decomposition ([above](#compact-leftcontrib)) raises the question of
carrying mix state instead of leaves everywhere. It does not generalize: `Lmix` changes
every round, so a contribution precomputed at one round's base is useless at the next, and
computing any round's contribution requires the raw leaves anyway. Serving all future
rounds would cost more than the leaves it replaces. Round 4 is the sole exception because
it needs *all* of one parent's leaves and only *one* of the other's.

</details>

### Eliminating the dense `gi`
<details>
<summary>Details</summary>

Long-standing lead: give each element its **bucket slot** (`cb*cap + cpos`) as identity
instead of a 4 B counter value from `atomic_inc(gi_counter)`, removing a global atomic per
emitted child. Ablating that atomic measured **-2.5 ms**, and the `(lead, gi)` ->
`(lead, slot)` tie-break change was verified safe first — such a pair shares a seed index
so it can never be part of a valid solution.

**Implemented and reverted: 56.2 -> 60.7 ms (+4.5 ms), every round slower.** Correct
throughout — just slower.

The dense `gi` is **not a pure cost — it buys write locality**. `atomic_inc` hands out
*consecutive* values across a warp, so the two back-ref writes coalesce. Bucket slots
scatter the same writes over a 190 MB row. This is the *same* effect already recorded
under [global gi atomic](#global-gi-atomic) for the per-bucket counter, and it was not
connected to this lead until the measurement forced it.

**The ablation that suggested this was itself misleading.** It substitutes `cgi = pos`, an
LDS index in 0..383, so back-ref writes land in a ~1.5 KB window — far *more* local than
either real alternative. It therefore measured "atomic removed **and** locality improved".
An ablation that replaces a value must substitute something with the same access
footprint, or it prices two changes as one.

</details>

### Occupancy, again
<details>
<summary>Details</summary>

[Occupancy tuning](#occupancy-tuning) was measured irrelevant early on, when the pipeline
was purely scatter-bound. That is **no longer true** — the geometry work moved the solver
off the pure-scatter rate, and occupancy now helps. It is just not reachable.

`(17,1)` with `LDS_FCAP=192` fits **2 workgroups/SM** and measured **39.3 ms** against
`(16,1)`'s 40.4. But `FCAP=192` is only `mean + 5.2σ` on the group tail, against the
`8σ` standard [capacity](#bucket-capacity) now uses. At a matched margin
(`FCAP=224`, 8σ) the same configuration measures **40.6 ms — no better than (16,1)**. The
gain was bought with drop headroom, not for free.

And it cannot be bought back by trimming LDS. For 2 wg/SM at group 264 the budget is
`(24576 − 2052) / 384 = 58.7 B/element`; `lwork` alone is **56 B** (`INW = 7` u64), so
everything else — `lgi`, `llead`, `lkey`, `lchain`, `lleaf` — would have to fit in 2.6 B.
Not staging `lwork` is the [un-fused path](#un-fused-lds-path), which doubles the work
traffic and measured far worse. **Occupancy is structurally blocked by the element width.**

*Addendum 2026-07-31: partially reopened, and harvested. The CUDA-era cuts — the
perfect table, r1's gi/leaf fold, and a spill-backed near-mean cap — take r1 to
64 B/element ≈ 19.0 KB/workgroup, and its second workgroup DOES fit and DOES pay:
see [the backport](#the-cuda-match-wins-backported-to-opencl-perfect-table--spill--per-round-caps-06-ms)
(−0.6 ms). r2 and r3 remain blocked exactly as argued here.*

</details>

### Overlapping the entry pass
<details>
<summary>Details</summary>

The last structural lead, and the reasoning behind it was sound: the entry is ~2.7 ms of
which only ~1.0 is its own writes, the rest being 235 M siphashes. It is the one
compute-bound phase in an otherwise bandwidth-bound pipeline, so it is the one phase that
*should* hide behind the rounds. The fused kernels also use only 34 KB of the SM's 48 KB
LDS, leaving room for a low-LDS kernel to co-reside.

**Refuted by a one-hour experiment instead of a six-file refactor.** A *redundant* entry
pass was enqueued on a second command queue, concurrent with the rounds, writing to
scratch — results unaffected, so the only question was what it cost:

| | ms |
|---|---|
| baseline | 40.2 – 40.7 |
| + a redundant 2.7 ms entry pass, concurrent on queue 2 | **42.8 – 43.2** |

It adds **~2.6 ms — its entire serial cost.** There is no overlap to be had. The round
kernels launch **131 072 workgroups**; the scheduler keeps every SM full for their whole
duration, so a second-queue kernel simply queues behind them. Spare *LDS* is not spare
*capacity* when the launch backlog is that deep.

The full change would have needed a second seed buffer (~390 MB), dedicated seed counters,
speculative next-nonce prefetch in `GpuSolver`, an addition to the `Solver` interface, and
cross-queue event synchronisation — for zero. **Test the premise before building the
mechanism**: the cost here was one throwaway `MXBM_CONC_TEST` block.

</details>

---

## Measured results, 2026-07-26 to 2026-07-28
<details>
<summary>Details</summary>

Everything below post-dates the CUDA backend shipping. These are longer write-ups than
the entries above because most of them are nulls whose *mechanism* is the finding, and
because several correct one another.

</details>

### Phase overlap cannot reach the roofline
<details>
<summary>Details</summary>

*(Two streams and one co-resident launch, both built and both null. Summarised as
[lead 5](#current-focus-and-open-leads).)*

The profile said this should work. Profiled 2026-07-25 (`sudo ./cuda/profile.sh`):

| kernel | ms | SM %peak | DRAM %peak | SM-busy | DRAM-busy | bottleneck |
|---|---|---|---|---|---|---|
| entry | 2.57 | **98.0** | 15.9 | 2.52 | 0.41 | compute-saturated |
| r1 | 5.80 | 65.0 | 28.2 | 3.77 | 1.63 | neither — latency-bound |
| r2 | 10.21 | 65.8 | 53.0 | 6.72 | 5.41 | neither — latency-bound |
| r3 | 9.94 | 21.1 | 78.1 | 2.10 | 7.77 | DRAM-saturated |
| r4 | 5.63 | 28.8 | 79.9 | 1.62 | 4.50 | DRAM-saturated |
| terminal | 1.04 | 70.1 | 80.3 | 0.73 | 0.84 | DRAM-saturated |
| **total** | **35.20** | | | **17.46** | **20.56** | |

*(Profiled 2026-07-25, before the `kFCap` 320 change. Per-round times have since moved
— entry 2.72, r1 5.33, r2 10.38, r3 9.50, r4 5.49, terminal 1.07, total 34.49 — but the
utilisation columns, which are what this table is for, keep their shape.)*

The **SM is busy 50 % of the wall clock and DRAM 58 %** — both idle about half the
time, at different moments. No round saturates both. `entry` is the extreme (98 % SM
against 16 % DRAM) and r3/r4 are its mirror (21–29 % SM, 78–80 % DRAM). Taking
`max(SM-busy, DRAM-busy)` = 20.6 ms suggests a **1.71× ceiling, ~96 sol/s**.

**That ceiling is not reachable by running kernels concurrently, and this was measured,
not argued.** `cuda/pipeline.cu --overlap` implements it: `entry` gets its own dense
stride-1 buffer (0.363 GiB, double-buffered), a second stream, and events ordering the
write-after-read hazard, so `entry(i+1)` is issued right after `r1(i)` and runs against
r2–r4 of nonce *i*. Both paths live in one binary, so they share a compilation, the
allocations and the kernels — only the scheduling differs. Over 100 distinct nonces:

| | ms/solve | sol/s | KAT | drops |
|---|---|---|---|---|
| sequential | 35.22 | 56.7 | PASS | 0 |
| `--overlap` | 35.30 | 56.6 | PASS | 0 |

**Positive control, because a null result must be shown to be capable of moving.**
`MXBM_ENTRY_REPS=4` makes `entry` four times as expensive (re-zeroing between reps so
each is a correct pass). If the overlap were hiding `entry`, the extra work would cost
far less under `--overlap`:

| | entry ×1 | entry ×4 | cost of 3 extra passes |
|---|---|---|---|
| sequential | 35.07 | 43.01 | **+7.94 ms** |
| `--overlap` | 35.15 | 43.11 | **+7.96 ms** |

3 × 2.57 = 7.71 ms predicted, ~7.95 measured, and **identical in both modes**. The
harness is live and the overlap hides exactly nothing.

**Why: the grids saturate the machine.** Concurrent kernels only run when the first
kernel has no blocks left to schedule — the work distributor fills every freed slot
from the running kernel first. Measured on this card (66 SMs):

| kernel | blocks/SM | resident | grid | **waves** |
|---|---|---|---|---|
| entry | 6 | 396 | 131 072 | 331 |
| r1–r4 | 3 | 198 | 131 072 | **662** |

Every round is **662 waves deep**. A second stream's kernel can only start in the last
wave's tail, so the reachable overlap is ~1/662 of a kernel — indistinguishable from
zero, which is exactly what was measured. Utilization percentages describe *how busy a
unit was*, not whether *scheduling slots were free*; the roofline above conflates the
two, which is why it was unreachable.

**This also retires the two-full-pipelines idea, and the capacity work it needed.**
Two concurrent solves would launch the same grid-saturating kernels on two streams and
hit the same wall, so the 48 MiB shortfall (7.820 GiB per pipeline; two plus a ~0.4 GiB
context is 16.04 GiB against a 15.99 GiB card) and the 8σ→6σ bucket-capacity reduction
that would have closed it are both moot. **And the reduction was never available
anyway** — see [4a](#current-focus-and-open-leads): ~636 is the Poisson maximum over
*one* draw of 65 536 buckets, but mining draws 8.0e11 of them a day, at which rate 6σ
drops an element roughly daily. The slack recorded here does not exist.

**The co-resident single launch was then built too, and it is also null — for a
different and more final reason.** *(2026-07-26.)* The previous paragraph proposed it
and it has now been measured, so the proposal is retracted below rather than left
standing.

`fused_round` gained a `COTENANT` template parameter that runs the **next** nonce's
entry pass inside the round's own blocks, sharing `entry_body` with `entry_scatter` so
the two cannot drift. The grids make it exact: a round launches
`(nb << sm) × kWG = 2^17 × 256 = 2^25` threads, precisely one per entry element.
Registers allowed it — r3 goes 48 → 77, still 3 blocks/SM, shared-memory-bound as
before. `--fuse` drops the separate `entry_scatter` entirely, and the 2.04
verified/solve it still reports is the positive control: had the co-tenant not run,
every solve after the first would have started from an empty buffer.

| host round for `entry(N+1)` | end-to-end |
|---|---|
| none (sequential) | **35.3** |
| r1 | 40.8 |
| r2 | 40.2 |
| r3 | 36.9 |
| r4 | 38.4 |

Removing a 2.77 ms kernel made the pipeline **slower everywhere**. The isolating
measurement (`-DMXBM_CO_NOSCATTER=1`, which computes entry but stores nothing, with
the real `entry_scatter` still launched so the pipeline stays correct and the co-tenant
is purely additive): hosting entry's **arithmetic alone** inside r3 costs **7.25 ms**,
against the 2.77 ms the whole pass costs standalone.

**Why, and this is the part that generalises: "the SM is 79 % idle" does not mean 79 %
of its issue slots are available to someone else's work.** r3's warps are idle because
they are *stalled on memory*, and a stalled warp still holds its warp slot. Putting
entry's work in those same warps does not interleave with the stall — it runs *after*
it, serially, in the same warp. The only way to fill a memory stall is **more resident
warps**, and r3 has 3 blocks × 8 warps = 24 of the SM's 48 because
[shared memory caps it at 3 blocks](#bytes-are-nearly-free-per-element-work-is-not).
Entry standalone gets 6 blocks/SM; hosting it halves its occupancy, which is exactly
the ~2.6× it slows down by.

That closes the whole family, not just this instance. Independent work added to a round
either goes **in the existing warps** (serialises, measured above) or **needs new
warps** — and a separate co-tenant block in the same launch must still reserve the
round's 32.8 KB of shared memory, so it displaces a round block one for one. There is
no arrangement of the same launch that adds warps without taking them away.

**So the 1.71× roofline is unreachable by any mechanism now tried:** not streams
(grid depth), not one fused launch (warp slots). What it would actually take is a
round kernel that needs materially less shared memory per block, which is the same
constraint that blocks occupancy, and `kFCap` cannot shrink — 320 already drops 180
elements. Worth noting the *memory* objection to two resident pipelines has lifted
(13.8 GiB at geometry (15,2), against the 48 MiB shortfall at (16,1) that retired it),
so if the shared-memory constraint is ever broken, that idea is affordable again.

</details>

### Streaming stores: implemented, measured, null
<details>
<summary>Details</summary>

`-DMXBM_STCS=1` routes every emit — records *and* back-refs — through `__stcs`, which is right on the face of it: a record is
read exactly once by the next round, and back-refs are read only for the ≤1024 survivors
out of 33 M, yet both allocate L2 normally and evict the rescan reads that do have reuse.
Verified applied at the SASS level (`cuobjdump -sass | grep STG.E.EF` → 32 stores, zero in
the baseline). **35.34/35.51 baseline vs 35.53/35.55 streaming**, KAT green and drops zero
both ways. The mechanism it needed does not exist here: the stores are already full-width
and coalesced within a bucket row, so there is no read-for-ownership to avoid, and the
"hot set" being protected is 0.27 GB against a 48 MB L2 — nothing was fitting either way.
This retires the item ranked **first** in [what a CUDA backend was predicted to
buy](#what-a-cuda-backend-was-predicted-to-buy-retained-for-calibration), the one called
"the only item that could move the achieved bandwidth".

</details>

### Bytes are nearly free; per-element work is not
<details>
<summary>Details</summary>

*(Measured 2026-07-26. This is the most consequential result in this document and it
retires a strategy, so the method matters as much as the number.)*

**Attribution needs the compute kept alive.** The obvious ablation — skip the emit's
payload store and diff — is wrong, and wrong in a way that announces itself: `combine`,
`apply_mix` and the whole `ctree` build feed *only* that store, so the compiler deletes
them with it, and round 1 comes out at **−3.46 ms**. `MXBM_ABL_EMIT=R` therefore
**narrows** the payload to 16 B instead of dropping it, XOR-folding every word the real
store would have written into what it does store. Rounds 1 and 4 already store 16 B, so
they are the built-in positive control: their delta must be zero, and it is (0.00, 0.05).

With that in place, `MXBM_ABL_EMIT` / `MXBM_ABL_DERIVE` / `MXBM_ABL_MIX` decompose a
round (marginal ms via `MXBM_ROUND_REPS`, 60 nonces):

| | round 2 | round 3 |
|---|---|---|
| full | **10.73** | **9.61** |
| `apply_mix` | 0.01 | 0.06 |
| re-derivation (seed / 14-siphash rebuild) | 0.97 | — |
| payload bytes (72→16 B, 64→16 B) | 3.47 | 0.34 |
| **remainder: staging read + chain walk + scatter transactions** | **6.28** | **9.20** |

**Reducing bytes moved is not a speed lever.** Cutting 56 B/element out of round 2 buys
1.6–3.5 ms and 48 B out of round 3 buys 0.3 ms; those marginal bytes move at **1194 and
4095 GB/s**, several times the card's 672 GB/s peak, which is only possible if they were
never what the round was waiting on. Shrinking *both* records to 16 B — far past anything
the [record audit](#the-record-redundancy-audit) leaves available — would take 34.2 ms to
about 30. **The footprint work remains the right lever for *reach*** (it is what put the
CUDA backend on 8 GB cards, lead 4) **but it is close to spent as a lever for speed**, and
the claim these docs used to make — that footprint "is the only one that moves both axes
at once" — is wrong on the speed axis.

**`apply_mix` is free on CUDA** — 0.01–0.06 ms, against the 0.9 ms the OpenCL ablation
recorded. Anything that trades bytes to avoid mixing is trading against nothing.

**What the remainder is not.** Four candidate explanations for it were built and measured,
and all four are null:

| lever | result | why |
|---|---|---|
| streaming stores (`__stcs`) | 35.3 → 35.5 | 32 `STG.E.EF` in SASS, nothing to protect |
| warp-aggregated `gi` | 35.3 → 35.6 | ptxas already emits the aggregation — see the correction below |
| geometry (17,0), no sub-mask rescan | 35.3 → 38.5 | −9 % traffic, but 2× the bucket rows costs more in scatter locality |
| occupancy 3 → 4 blocks/SM | impossible | below |

> **Correction (2026-07-29): the `gi` row's stated reason was wrong.**
> `kernels/cuda/fused_round.cuh:170-171` claimed *"the SASS shows a plain `ATOMG.E.ADD`
> per lane, with no aggregation prologue."* It does not.
> `cuobjdump -sass build/libmxbm_cuda.a`, round-2 packed instantiation, `0x98f0`–`0x9a90`:
> `VOTEU.ANY` → `FLO.U32` (leader) → `POPC` (group size) → `@P0 ATOMG.E.ADD` (**one**
> atomic, adding the popcount) → `SR_LTMASK` + `POPC` (rank) → `SHFL.IDX` (broadcast).
> The address is a warp-uniform kernel-parameter pair — `gi_counter`. The kernel holds
> `VOTEU.ANY` ×5, FLO-family ×5, POPC-family ×8, `ATOMG.E.ADD` ×2, `SHFL.IDX` ×2.
>
> **ptxas inserts it, not NVVM.** The same construct compiled with `nvcc -ptx` emits a
> bare `atom.global.add.u32 %r, [%rd], 1`; compiled with `nvcc -cubin` it emits the full
> idiom. The control is in the same kernel: `atomicAdd(&out_counts[cb], 1u)` at `0x9810`
> has a lane-varying address and gets a bare `ATOMG.E.ADD` with no prologue — which is
> what "no aggregation" actually looks like in this disassembly.
>
> So `MXBM_WARPAGG`'s +0.3 ms is **a redundant second aggregation stacked on one ptxas
> had already done**, not evidence that the coalesced group is too small to be worth
> aggregating. The "1–4 lanes" figure is unmeasured — nothing in the tree measures the
> coalesced group size — and is withdrawn with it.

**Occupancy is structurally capped at 3 blocks/SM, and now there is a number for it.**
`kFCap` sets shared memory per block; the group tail will not fit a smaller one — 352 is
clean, **320 drops 180 elements and 288 drops 77 099**. The cap is the group-size
distribution, not a tuning choice. See
[occupancy is worth ~1.6 ms](#occupancy-is-worth-real-time-and-shared-memory-is-the-only-gate)
for what that cap is costing, which is not nothing.

So the remaining 6–9 ms per round is the staging read, the chain walk and the *transaction
count* of the scatter — none of which is byte-count-driven, and none of which any
single-kernel lever has moved. That is the same conclusion the overlap work reached from
the other direction, and it points at the same place: co-residency.

</details>

### …but bytes are not free in WATTS, and under a cap watts are clock (−60 MHz)
<details>
<summary>Details</summary>

*(Measured 2026-07-28, `benchmarks/byte_power.sh`. This is the counterpart to the section
above and it does not contradict it: the same bytes are cheap in time and expensive in
power.)*

The [head-to-head sweep](performance.md#both-miners-under-the-same-cap) found MXBM clocking 60–570 MHz
below lolMiner at every equal power cap and named DRAM traffic as **the suspect**. That
was an inference from the fact that lolMiner fits the search in 4 GB. This measures it.

**The card is power-capped essentially all the time**, which is what makes the experiment
possible: the controlled headline run recorded `sw_power_cap` active **99–100 %** of the
time at stock. So board watts are pinned by definition and the *clock* is the free
variable — if work costs less power, the card spends the saving on frequency, and the SM
clock is a direct readout of the power cost of the work.

Method: the same `MXBM_ABL_EMIT` ablation the speed attribution used — round 2's scattered
payload narrowed 72 B → 16 B with `combine`, `apply_mix` and the ctree build all still
live, so bytes are the only variable — replayed with `MXBM_ROUND_REPS` so the round
dominates the timeline. Each comparison is bracketed by two unablated runs, so drift has
to be beaten rather than assumed; it came out at 0–0.5 ms and **0 MHz** every time.

| | ms/solve | SM | W | |
|---|---|---|---|---|
| baseline, no replay | 33.79 | 2685 | 284.2 | |
| r1 full ×8 | 69.84 | 2655 | 284.1 | **control** — r1 already stores 16 B |
| r1 narrowed ×8 | 70.00 | 2655 | 284.1 | **±0 MHz, ±0.0 W**, as it must be |
| r2 full ×24 | 270.89 | **2610** | 284.3 | |
| r2 narrowed ×24 | 233.57 | **2670** | 283.9 | **+60 MHz** at the same cap |

**Removing 2.09 GB/solve — 16.1 % of the solve's 13.0 GB — buys 60 MHz, 2.3 % of clock.**
The control is what makes that quotable: rounds 1 and 4 already store 16 B, so ablating r1
changes no traffic, and it moves the clock by exactly nothing.

> **⚠ Correction (2026-07-29): the denominator is 1.47 GB, not 2.09, and it is now
> measured rather than derived.** The 2.09 GB was never reproducible from the
> [traffic table](#current-focus-and-open-leads) — it scaled round 2's
> *compulsory* write, which includes 268 MB of back-refs the ablation does not touch.
> Profiled directly on the two binaries this section already uses
> (`ncu --metrics dram__bytes_write.sum,dram__bytes_read.sum`, round 2 selected
> positionally as the 2nd `fused_round` launch):
>
> | | write | read | |
> |---|---|---|---|
> | baseline | **2.81 GB** | 538.8 MB | matches the published 2811 MB r2 write exactly |
> | `MXBM_ABL_EMIT=2` | **1.33 GB** | 546.6 MB | |
> | removed | 1.48 GB | −7.8 MB | **net 1.47 GB** |
>
> Two open questions close with it, both by arithmetic that only holds one way:
>
> - **A 16 B store costs a full 32 B sector at DRAM.** 16 B/element plus back-refs
>   predicts 805 MB; a 32 B sector predicts 1342 MB; measured 1330 MB.
> - **There is no read-for-ownership.** A partial-sector store pulling its line in would
>   have added ~1074 MB of read. Read moved **+7.8 MB**, 1.4 %.
>
> So the exchange rate *on this replay timeline* is **210 / 1.47 = 143 MHz/GB**, and both
> of its terms are now measured on the same timeline, which the published figure's were
> not. **The whole-solve rate is still only bounded**, because the 210 MHz numerator is
> itself a replay clock — see the correction below — giving **≤ 92 MHz/GB**.
>
> **What that costs the byte programme.** Closing the 570 MHz deficit at 180 W needs
> **≥ 6.2 GB of the solve's 13.0 — 48 % of all traffic**. Every record narrowing the
> [audit](#the-record-redundancy-audit) leaves available totals 1.07 GB, i.e. **≤ 98 MHz,
> 17 % of the deficit**. Traffic alone does not close the low end, and this is the first
> statement of that with a measured denominator rather than an inferred one.

**The obvious confound is excluded by construction.** The narrowed build feeds round 3 a
deliberately garbled record, which costs 180–197 k bucket drops (0.5 % of elements) and
therefore slightly less downstream work. Sweeping the replay count separates the two,
because more replays mean round 2 occupies more of the timeline while the drop count per
solve stays fixed:

| replays | non-round-2 share of the solve | Δ clock |
|---|---|---|
| 4 | 36.6 % | −75 MHz |
| 8 | 22.2 % | −75 MHz |
| 16 | 12.5 % | −60 MHz |
| 24 | 8.7 % | −60 MHz |

The drop-affected share of the timeline falls **4.2×** and the effect does not follow it
down — it converges on 60 MHz, which is the clock the card holds while actually running
round 2. Had the drops been driving it, 24 replays would have shown roughly 18 MHz.

**What the claim is, precisely.** The ablation removes stores, so it removes DRAM traffic,
L2 traffic *and* store-instruction issue together, and this experiment cannot separate
them. The honest statement is that **round 2's scattered store path costs 60 MHz of
sustained clock at a fixed 285 W** — not that DRAM specifically does. For the decision it
informs that distinction does not matter, because narrowing a record reduces all three at
once; it matters for anyone tempted to price the effect per byte and extrapolate.

**And it sharpens rather than reverses "bytes are nearly free".** The marginal cost of
round 2's payload here is **1.69 ms** per pass (10.32 full against 8.63 narrowed, taken
across 4→24 replays), of which about 0.2 ms is itself the clock the freed watts bought. So
in *time* the bytes are even cheaper than the 3.47 ms the 2026-07-26 attribution recorded
on the pre-group-cap build. Bytes are cheap in time and expensive in power, and on a card
that is power-capped 99 % of the time the second one is what reaches the headline.

**What this changes about the roadmap** — and it splits an item the docs had as one:

- **Narrowing records** cuts bytes moved → cuts watts → buys clock. This is the
  **efficiency** lever, and it is now measured rather than suspected. ⚠ **But see [the
  quad record](#the-quad-record-29--footprint-and-the-byte-prize-does-not-survive-re-derivation):
  the one implemented narrowing buys 0 MHz, because re-derivation spends the freed watts
  on the arithmetic that replaces the bytes.** The prize below is an upper bound reachable
  only by a narrowing that costs nothing, and the record audit says none is left.
- **Streaming / in-place layer reuse** cuts the 7.46 GiB *footprint* by writing the same
  bytes to reused addresses. Traffic is unchanged, so by this measurement it buys no
  clock at all. It is a **reach** lever — 8 GB cards — and nothing else.

[HW_REQUIREMENTS.md](HW_REQUIREMENTS.md#2-memory-efficiency-is-24-off-the-algorithms-design-target)
names in-place reuse as the route to the 3 GB target and treats memory efficiency and
energy efficiency as one problem. It is the route to the *footprint* target; the energy
half has to come from narrower records. Since [measured traffic is within 1 % of
compulsory](performance.md#the-memory-traffic-is-compulsory) for the current widths, narrower records
means a structural change and not tuning.

</details>

#### Under a low cap the same bytes cost 5× as much clock
<details>
<summary>Details</summary>

*(Measured 2026-07-28, `sudo PL=180 benchmarks/byte_power.sh`. This is the result that
matters, because 180 W is where MXBM actually loses.)*

At stock the clock has almost no room to move and 16 % of the traffic was worth 2.3 % of
it. Capped to 180 W the card is far below its ceiling, and the same ablation moves it a
great deal further:

| cap | round 2 full | round 2 narrowed | Δ | as % of clock | drift |
|---|---|---|---|---|---|
| 285 W | 2610 MHz | 2670 MHz | **+60** | 2.3 % | 0 MHz |
| 180 W | 1815 MHz | 2025 MHz | **+210** | **11.6 %** | 30 MHz |

**The exchange rate is 5.0× steeper at 180 W**, and the signal clears the drift by 7×.
Board power is pinned at 179.9 W against 179.5 in the two arms, so this is a clean clock
readout and not a power one. In time, round 2's payload costs **3.66 ms per pass** here
against 1.69 ms at stock.

The single most legible number in the run needs no arithmetic at all: the **unablated
baseline holds 1890 MHz**, round 2 with its full record drags the card down to **1815**,
and the same round with a 16 B record runs at **2025** — above the baseline. Round 2's
stores are what set the clock of the whole solve.

**Round 1 corroborates it for free.** Replaying round 1 — which is byte-light and which
the ablation cannot change, since it already stores 16 B — lets the card clock *up*, to
1935–1950 MHz against the 1890 baseline, with the ablated and unablated arms agreeing to
within the drift. Same mechanism, opposite direction, no ablation involved.

**This is the explanation for [the low-end deficit](performance.md#why-we-lose-the-low-end-watts-buy-us-less-clock).**
That table records MXBM 570 MHz behind lolMiner at 180 W and only 60 MHz behind at stock,
and the asymmetry was unexplained. It is the same asymmetry measured here from the other
side: bytes are nearly free at 285 W and expensive at 180 W. One lever worth 210 MHz
against a 570 MHz gap puts the whole deficit within reach of a traffic explanation —
it needs ~2.7× the traffic reduction this ablation makes, and a 4 GB variant against our
7.46 GiB is plausibly that. Plausibly, not demonstrably: nobody has measured lolMiner's
traffic.

**What this measures is the BENEFIT side only, and it is an upper bound.** The ablation
removes 56 B/element and pays nothing for them: it stores garbage. Every real way of
moving fewer bytes pays something — re-derivation arithmetic, an extra pass, or a worse
access pattern — so 60 MHz and 210 MHz are the size of the prize, not a forecast of any
change that could ship. **Nothing measured today priced the cost side**, and until
something does, "narrow the records" is a lead with a known ceiling and an unknown floor.

**What it makes worth re-testing: the round-3 quad record.** `b360695` had round 3
re-derive from a **24 B** record instead of storing 72 B of work state; `2584518` retired
it at *"a genuine −3.6 ms at 86.8 ms is a +2.2 ms loss at 56."* The record it removes is
written by round 2 and read by round 3, ~5.1 GB of the solve's 13.0, so at 24 B it takes
out roughly **26 % of all traffic** — about 1.6× the lever measured above. It is the only
existing implementation of the byte/arithmetic trade, which makes sweeping it the cheapest
way to learn what the cost side is worth.

> **Correction (2026-07-28).** An earlier version of this section, and the commit message
> of `5960fd4`, said that trade was made "on a metric that did not price watts". **That is
> wrong.** The +2.2 ms was end-to-end wall time on a card that is `sw_power_cap`-limited
> ~100 % of the time, so it already included whatever clock the freed watts bought — a
> wall-clock A/B on a capped card prices the power effect implicitly, without anyone
> having to know it is there. The +2.2 ms is the *net*, meaning the gross arithmetic cost
> was larger still, offset by a clock gain nobody had identified.
>
> What was never measured is the same trade **at a low cap**, and that is the whole
> argument for re-testing it. The benefit scales 5× between 285 W and 180 W while the
> arithmetic cost is roughly fixed in cycles. Taking the stock numbers at face value —
> ~1.3 ms of clock gain implied by 26 % of traffic, so ~3.5 ms gross arithmetic — and
> rescaling both to 1815 MHz gives ~6.5 ms of gain against ~5.0 ms of cost: a **~1.5 ms
> net win at 180 W**, which is thin enough that it could land either side of zero. That is
> a real prediction and it is the point of running the sweep; it is not a reason to expect
> much.

</details>

### The occupancy optimum does move under a cap — by 0.08 % of a solve
<details>
<summary>Details</summary>

*(`benchmarks/occupancy_cap.sh`, 2026-07-28. Direction predicted correctly, magnitude
overestimated 2.5×, and the honest conclusion is to drop it.)*

Every occupancy decision here was tuned at stock, where more resident warps hide more
latency. Under a cap they also cost power, and power spent on occupancy is not spent on
clock. Round 1 has the cleanest knob for this: `MXBM_R1_FCAP=288` fits five blocks/SM,
320 fits four, and the stock answer was already recorded as 0.15 ms in favour of 288.

Round 1 replayed 9× to lift a sub-ms difference clear of run-to-run spread:

| cap | 288 (5 blocks) | 320 (4 blocks) | Δ per r1 pass | Δ clock | winner |
|---|---|---|---|---|---|
| 285 W | 74.86 / 74.97 ms @ 2670/2655 MHz | 76.36 / 75.50 @ 2670/2685 | **+0.113 ms** | +15 MHz | 288 |
| 180 W | 97.40 / 97.14 @ 1995/1965 | 96.91 / 96.99 @ 2010/2010 | **−0.036 ms** | +30 MHz | **320** |

**The ordering flips**, and the clock story is the cleaner half: dropping one block buys
15 MHz at stock and 30 MHz at 180 W, with every 320 sample clocking at or above every 288
sample. The mechanism is real and it does steepen under a cap.

**And it is far too small to matter.** −0.036 ms per r1 pass is **0.08 % of a 43.6 ms
solve**; even assuming all four fused rounds gained the same, the ceiling is **0.33 %**.
The time signal at 180 W (0.32 ms across 9 replays) also barely clears the 288 arm's own
0.26 ms spread, so it is directionally credible and quantitatively marginal.

**Where the prediction went wrong, since it was written down first:** the activity→clock
steepening was taken from the DRAM-traffic result (5×, 60 → 210 MHz) and applied to
occupancy. Occupancy's own steepening is **2×** (15 → 30 MHz). Scaling one lever's
exchange rate by another lever's is what produced a 5×-too-large forecast.

**And the pre-registered decision rule was the wrong rule.** It said "if 288 still wins at
180 W, the idea is dead" — a test on *sign*. The sign flipped and the lever is still dead,
because 0.33 % is not worth a runtime-selected kernel variant. A stopping rule should have
named a *magnitude*: below ~1 % of a solve, drop it regardless of direction. Recorded
because it is the second time today a correctly-predicted direction came with a
uselessly-small size.

</details>

### The quad record on OpenCL takes 11 GB cards off the sort path (~190 → 45 ms)
<details>
<summary>Details</summary>

*(2026-07-28. The reason to port it: on OpenCL the quad record is not a footprint
nicety, it is what clears a hard ceiling.)*

OpenCL is bound by `CL_DEVICE_MAX_MEM_ALLOC_SIZE`, VRAM/4 on NVIDIA, and that is what
decides whether a card runs the row-bucket path at all. The quad record takes the
**largest single allocation from 2.76 GiB to 2.45** — and an 11 GB card reports 2.65.

| card | max_alloc | packed | with the quad record |
|---|---|---|---|
| 16 GB | 3.90 GiB | row-bucket (16,1) | (unchanged — packed wins) |
| 12 GB | 2.90 | row-bucket (14,3) | (unchanged) |
| **11 GB** | **2.65** | **sort path, ~190 ms** | **quad (15,2), 45.1 ms** |
| 10 GB | 2.42 | sort path | sort path — 2.42 misses quad (14,3)'s 2.45 by 30 MB |

**That is 4.2× for the 11 GB class**, and it is the largest single improvement available
to those cards: the sort path is structurally ~4.7× the row-bucket path and
[three separate attempts to close that by transplanting row-bucket wins](#index-only-round-1-does-not-transfer-to-the-sort-path-27-ms)
have now returned −21 ms, −3.7 ms and +27 ms. Moving the card off the path beats tuning it.

`LMODE_RD3` and `round_fused_rd2q`/`round_fused_rd3` mirror the CUDA implementation
exactly — the `rd3_*` accessors and `rd_elem2` were still in `lds.cl`, having survived
the 2026-07-24 revert of the original OpenCL attempt. Gate green on the first run at
every rung:

| geometry | packed | quad |
|---|---|---|
| (16,1) | 40.3 ms | 42.5 |
| (15,2) | 43.6 | 45.1 |
| (14,3) | — | 50.8 |

**The record costs +4.4 % here against +14 % on CUDA**, which is the "idle issue slots"
rule again: this path is more memory-stalled, so the two round-2 rebuilds hide in the
deferred expand rather than landing on the critical path.

One thing fixed on the way, unrelated but sharp: `MXBM_BB=15` alone left `sm` at its
default of 1, putting the geometry off the `bb + sm == 17` line every compile-time kernel
constant assumes — and the OpenCL path, unlike CUDA's, did not check, so it **silently
produced wrong results** (0/3 goldens) rather than refusing. `sm` is now derived from `bb`.

</details>

### The round-4 mix anomaly was a runtime constant (27.0 → 8.2 ms)
<details>
<summary>Details</summary>

*(Closed 2026-07-28. Open since before the CUDA backend existed, and the fix is a
transformation this document already recommended twice.)*

The sort path's `round_mix` is **one kernel**; only `padNum`, `Lmix` and the buffers
differ per round. Round 4's mix measured **27.0 ms against 7.6 / 7.6 / 7.8** for rounds
2 / 3 / 5, and three explanations had already been ruled out: not the leaf read (a fixed
contiguous 9-leaf read left it at 30 ms), not work volume (**round 5 does *more* — the
internal loop runs 2/4/6/9 times for r2–r5 — and runs 4× faster**), not a branch, since
the data is round-independent.

**It was `Lmix` arriving as a kernel argument.** Inside `bh3_apply_mix`:

```c
uint word = (Lmix + i*25u) >> 6;
t[word] |= v << sh;                  // t is a private ulong[8]
```

A private array indexed by a value the compiler cannot resolve **cannot be kept in
registers**. `t[8]` went to scratch, and every `|=` became a scratch load-modify-store.
Round 4 is where the cliff landed: six iterations, three of which also trigger the
`t[word+1]` carry write — one more live scratch slot than the allocator had. Rounds 2 and
3 trigger the carry once; round 5 triggers it four times but reads a fully contiguous
36 B leaf stride.

The fix is `ROUND_MIX_K`, which bakes `padNum` and `Lmix` in alongside the `INW` that was
*already* compile-time — and the reason that one was is recorded three lines above it in
the same file: *"a runtime stride here cost +24 ms (mix regressed 50→74), the same
unrolling-killer that sank runtime-width match."* The lesson had been written down and
applied to one of the three constants.

| | runtime `Lmix` | constants | |
|---|---|---|---|
| r2 mix | 7.6 ms | 7.3 | |
| r3 mix | 7.6 | 7.3 | |
| **r4 mix** | **27.0** | **8.2** | −18.8 ms |
| r5 mix | 7.8 | 6.3 | |
| total mix | 50.6 / 50.8 | 29.2 / 29.8 | |
| **solve** | **214.8 / 215.3** | **193.5 / 194.1** | **−9.8 %** |

Interleaved, 15 solves each, goldens byte-identical and drops zero on every run.
`bh3_apply_mix` itself is untouched — it is on the never-modify list with `bh3_combine`
and `bh3_siphash24`; being `inline`, it simply receives literals now. `MXBM_MIX_RUNTIME`
restores the old kernels so the A/B stays re-measurable.

**The generalisable part:** a constant that is *known per round* but *passed at runtime*
is not a small inefficiency here — it can be a 3× cliff, because it decides whether a
private array lives in registers or in scratch. The row-bucket path learned this as
"−32.5 % from compile-time round constants"; the sort path had the same three constants
and only one of them had been fixed.

</details>

### The same trick on `round_match` mostly does not work (−3.7 ms, and two rounds lose)
<details>
<summary>Details</summary>

*(2026-07-28. Ran because `Lout`, `lead_identity` and the leaf widths `s_in`/`s_out` were
equally runtime in the match kernel — 158 ms of the then-193. The result corrects the
lesson above rather than extending it.)*

`MATCH_SORTED_K` bakes all three in alongside the work widths that were already constant.
Per round, interleaved, reproduced on every run:

| | r1 | r2 | r3 | r4 | r5 | total |
|---|---|---|---|---|---|---|
| runtime | 31.6 | 33.0 | 34.4 | 33.6 | 25.0 | 158.8 |
| constant | 34.4 | 35.3 | **32.8** | **32.7** | **23.8** | 160.0 |
| | +2.8 | +2.3 | −1.6 | −0.9 | −1.2 | **+1.4** |

**It wins for r3/r4/r5 and loses for r1/r2**, so only those three are selected and rounds
1–2 stay on the generic kernel: **155.3 ms, −3.7 ms on the solve.** `k1`/`k2` are kept,
compiled and unused, so the null stays re-measurable; their presence costs nothing.

The split tracks `s_out`, the leaf-copy loop — 8 / 9 / 0 for r3 / r4 / r5, where unrolling
it (or deleting it outright at 0) pays, against 2 / 4 for r1 / r2 where it does not. **What
costs r1/r2 those 2–3 ms is not explained.** It is not the redundant zero-fill (guarding it
changed nothing), and `lead_identity` folding to 1 should if anything have *helped* r1,
since it removes two global leaf loads per candidate pair.

**This is the correction to the lesson, and it matters more than the 3.7 ms.** Baking a
constant pays where it lets a **private array escape scratch** — `t[8]` above, 27 → 8 ms —
and is roughly free-to-negative where it only removes comparisons. `round_match` has no
dynamically-indexed private array: `ea`/`eb`/`ec` are indexed by unrolled loop counters.
There was never a cliff here to find, and "compile-time round constants are worth −32.5 %"
generalised from a case that had one. Look for the dynamic index first; the constants are
the fix, not the diagnosis.

</details>

#### Index-only round 1 does NOT transfer to the sort path (+27 ms)
<details>
<summary>Details</summary>

*(Built, gated, measured, reverted to opt-in 2026-07-28. `MXBM_IDXONLY`.)*

The row-bucket path's largest single win of its kind was
[re-deriving round-1 seeds from indices](#seed-re-derivation): **−12.1 ms of 114.8**. The
sort path has the identical opportunity — a round-1 element is `seed_element(pp, idx)`
mixed at 448 over the tree `{idx}`, fully determined by an index the sort pair already
carries — and storing it costs 56 B/element of work plus a 4 B leaf write that lands in its
own 32 B sector, then 2 × 56 B of scattered reads per candidate pair.

`round1_mix_seeds_idx` stores only the sort pair; `round_match_seed` derives both parents,
hoisting the left one out of the inner loop since it is invariant there. **Correct on the
first run** — goldens byte-identical, drops zero, 10/10 clean.

| | stored | index-only |
|---|---|---|
| r1 match | 31.6 / 31.7 ms | **60.2 / 61.2** |
| solve | 190.5 / 190.8 | **215.8 / 218.1** |

**+27 ms.** The seed kernel got ~3 ms cheaper from dropping ~3 GB of writes, and the match
paid ~29 for it.

**Why it wins there and loses here, which is the transferable part.** On the row-bucket
path the derivation sits in the *deferred all-lanes loop*, where the kernel is already
memory-stalled and has idle issue slots — the same document records that arithmetic as
**"+0.4 ms here versus +23.5 ms for the same arithmetic in the sub-mask-filtered staging
loop"**. The sort path's match has no deferred phase; every derivation is on the critical
path. And the counts differ: the seed kernel derives **once per element** (33.5 M), the
match derives **once per pair-member** — at Poisson(2) multiplicity that is
`E[m] + E[m(m−1)/2] = 4` per run over 16.7 M runs, so **≈2× the derivations, all in the
worst place.**

So the rule is not "re-derivation beats storage". It is **re-derivation pays only where the
kernel has idle issue slots to hide it in** — which is the same finding as
[the quad record](#the-quad-record-29--footprint-and-the-byte-prize-does-not-survive-re-derivation)
(the arithmetic ate the watts the bytes freed) and
[the match constants](#the-same-trick-on-round_match-mostly-does-not-work-37-ms-and-two-rounds-lose)
(no cliff to find) from two more directions.

</details>

#### The OpenCL path finally has a register/spill instrument
<details>
<summary>Details</summary>

`MXBM_CL_VERBOSE=1` appends `-cl-nv-verbose` and prints NVIDIA's `ptxas` report. This is
the only register-level instrument available here — Nsight cannot profile OpenCL, which is
why every figure on this path had come from ablation and arithmetic. Pair it with
`CUDA_CACHE_DISABLE=1`, or the driver's on-disk cache returns the binary without
recompiling and the report is empty.

It settles both questions above from the compiler rather than from timings:

| kernel | stack frame | spill | registers |
|---|---|---|---|
| `round_mix` / `_c5` / `_c6` *(runtime `Lmix`)* | **112 B** | 0 | 39 |
| `round_mix_k2…k5` *(constant)* | **0 B** | 0 | 28–32 |
| `round_match` | 0 B | 0 | 40 |
| `round_match_sorted` *(r1/r2)* | 0 B | 0 | **48** |
| `round_match_sorted_7_6` / `_6_5`, `round_match_k3`/`k4` | 0 B | 0 | 40 |

**112 B is exactly `t[8]` (64 B) + `tree[9]` (36 B)** — the diagnosis confirmed at source,
not inferred. And **no match kernel spills at all**, so the mechanism worth 3× in the mix
is simply absent there; that line of enquiry is closed rather than open.

The register column also offers a mechanism for the unexplained r1/r2 regression: the
generic kernel they use costs **48** registers and the K variants **40**, so
constant-folding *raised* occupancy. Forcing the generic down to 40 with
`MXBM_CL_MAXREG=40` costs **r1 +4.9 ms and r2 +9.5** — the same direction as the K
regression and larger. On a gather-bound kernel more resident warps can cost more in cache
locality than they buy in latency hiding, which is the same shape as
[the (17,0) geometry result](#bytes-are-nearly-free-per-element-work-is-not) (−9 % traffic,
+9 % time, lost to scatter locality). Stated as supported rather than proven: the capped
build's own spill numbers were not in the driver's (truncated) log, so "the cap did not
simply induce a spill" is not directly verified.

</details>

### The quad record: −29 % footprint, and the byte prize does NOT survive re-derivation
<details>
<summary>Details</summary>

*(Built and measured 2026-07-28. `MXBM_R3_QUAD`, off by default. This is the experiment
the two sections above called for, and it answers them in the negative.)*

Round 2 emits a **24 B quad record** — key, four leaves, `gi` — instead of the 72 B packed
record, and round 3 rebuilds the seven work words from those leaves (`LM_RD3`,
`rebuild_r3`: two `rebuild_r2` calls, combined at Lout(2)=400 and mixed at Lmix(3)=400).
The information is identical; only the bytes differ. It is the same trick `LM_RD2` already
plays one round higher, and it existed once on the OpenCL path (`b360695`, retired in
`2584518`).

Gate green at every geometry — KAT 3/3 survivors, goldens byte-identical, `bucketDrops`
and `pairDrops` zero:

| geometry | footprint | ms/solve | | footprint | ms/solve |
|---|---|---|---|---|---|
| | **packed record** | | | **quad record** | |
| (16,1) | 7.46 GiB | 33.67 | | **5.28 GiB** | 38.39 |
| (15,2) | 6.88 GiB | 36.01 | | **4.91 GiB** | 40.65 |
| (14,3) | 6.50 GiB | 40.02 | | **4.66 GiB** | 44.75 |

**−29 % of the footprint for +14 % of the time**, interleaved A/B, three repeats each,
spread under 0.2 ms.

</details>

#### The result that matters: the clock does not move
<details>
<summary>Details</summary>

Sampling NVML underneath both variants at stock, alternated:

| | ms/solve | SM clock | W |
|---|---|---|---|
| packed | 33.72 / 33.56 | 2700 / 2715 | 283.8 / 283.5 |
| quad | 38.53 / 38.61 | 2700 / 2692 | 284.1 / 284.8 |

**Identical, at the same 284 W.** The quad record moves **26 % less traffic** than the
packed one — more than the 16 % the ablation removed for +60 MHz — and buys **zero clock**.

The reason is the thing the ablation could not tell us, and it retires the lead:
**re-derivation spends the freed watts on the arithmetic that replaces the bytes.** The
ablation removed 56 B/element and paid *nothing* for them, because it stored garbage; that
is what made its 60 MHz an upper bound rather than a forecast. A real narrowing has to put
something in the bytes' place, and fourteen siphash rounds cost about what the DRAM
traffic they replace costs. Net power: unchanged. Net clock: unchanged. Net time: worse by
the arithmetic.

So **the byte→watt prize is real and is not reachable by re-derivation.** Harvesting it
needs a narrowing that costs nothing — and [the record audit](#the-record-redundancy-audit)
says every record is already at `ceil(bits/64)`, with the one free win
([the alignment pad](#the-round-2-alignment-pad)) already taken. That is a much harder
place to stand than "footprint is the main efficiency lever available", which is what this
document said before the experiment.

*(At stock. Under a cap the freed watts do start to show as clock — +128 MHz at 180 W —
without ever covering the arithmetic. The cap sweep is below.)*

</details>

#### Under a cap the effect appears — and the trade still never pays
<details>
<summary>Details</summary>

*(`benchmarks/quad_cap.sh`, 2026-07-28. Packed / quad / packed at each cap, so the two
packed runs bracket the quad one and their spread is the drift the delta must beat.)*

| cap | packed | quad | Δ time | Δ clock | drift |
|---|---|---|---|---|---|
| 285 W | 33.62 ms @ 2692 MHz | 38.47 @ 2670 | +4.85 ms | −22 MHz | 15 MHz |
| 220 W | 35.73 ms @ 2460 MHz | 41.37 @ 2460 | +5.64 ms | ±0 MHz | 0 MHz |
| 180 W | 43.68 ms @ 1882 MHz | 50.01 @ 2010 | +6.34 ms | **+128 MHz** | 15 MHz |

**The byte→watt mechanism is real, and this is the first time it has been shown in a
correctness-preserving change rather than an ablation.** At 180 W the quad record holds
**128 MHz more clock** on the same board power — 6.8 % — for no reason other than moving
26 % fewer bytes. The clock delta marches −22 → 0 → +128 as the cap tightens, exactly the
direction the [5× steepening](#under-a-low-cap-the-same-bytes-cost-5-as-much-clock)
predicts, and for the reason that predicts it: the memory clock does *not* scale with a
core power cap (10251 MHz at 180 W and at 285), so DRAM is a far larger share of a tight
budget than a loose one.

**And the trade still loses at every cap — by MORE as the cap tightens.** +4.85, +5.64,
+6.34 ms. That is the part worth internalising: the clock gain grows, and the deficit
grows faster, because the arithmetic's *wall-time* cost rises as the clock falls. At 180 W
the rebuild costs 9.7 ms gross (quad at packed's clock would be 53.4 ms) and the 128 MHz
hands back only 3.4 of it. The bar set before the run was +358 MHz to break even; it
managed 128.

Efficiency follows time, so it loses there too: at 180 W packed does 0.248 sol/s/W against
quad's 0.217.

**The question is closed.** The quad record is a **footprint lever and nothing else** —
there is no cap at which it becomes a speed or efficiency win, and the trend runs away
from one rather than toward it.

</details>

#### What it IS: a new bottom rung on the geometry ladder
<details>
<summary>Details</summary>

The footprint half stands on its own, and the ladder is where it pays. Comparing on
**memory budget** rather than on geometry:

| a card that can host… | best packed option | best quad option |
|---|---|---|
| 6.9 GiB | (15,2), 36.01 ms | — *(packed wins)* |
| 6.5 GiB | (14,3), 40.02 ms | **(16,1), 38.39 ms** |
| 5.3 GiB | *refuses* | **(16,1), 38.39 ms** |
| 4.7 GiB | *refuses* | **(14,3), 44.75 ms** |

Below ~6.5 GiB the quad record is both **smaller and faster** than stepping the geometry
down, because a coarser geometry pays in scatter locality what the quad record pays in
arithmetic — and the arithmetic is cheaper. Above it, the packed record wins and should
stay the default.

That extends the CUDA path from a 7.46 GiB floor to **4.66 GiB**, which is the 6 GB card
class (an RTX 3050 6 GB reports ~5.7 GiB).

**Shipped 2026-07-28.** `rb_geometry_for` takes an `allow_quad` flag — true only from
`CudaSolver`, since the record has no OpenCL counterpart — and walks a six-rung ladder
ordered by measured time. Both record formats are instantiated and the choice is made at
runtime from the card's VRAM, so one binary serves every rung; `MXBM_QUAD=0|1` forces it.
The allocator-retry path walks the same rung list rather than decrementing `bb`, or it
would stop at the bottom of the packed half and refuse a card the quad rungs would host.
Goldens match `[1,1,1]` under `MXBM_QUAD=1` and the miner reports 1.99 verified
solutions/solve on that path, so the mining route is gated and not just the bench.

**Stated CUDA requirement drops from 8 GB to 6 GB**, and the cards between 6.3 and 7.9 GiB
free get a *faster* rung than before — quad (16,1) at 38.4 ms where the packed ladder gave
them (14,3) at 40.0. Full table in
[HW_REQUIREMENTS.md](HW_REQUIREMENTS.md#1-below-11-gb-opencl-is-capped-by-its-single-allocation-limit).

</details>

### Shipped: the group cap no longer has to cover the tail (−1.25 ms)
<details>
<summary>Details</summary>

*(2026-07-26. **35.4 → 34.1 ms/solve, 55.9 → 58.1 sol/s**, goldens byte-exact, drops 0.)*

Two changes, and the second is what makes the first safe.

**1. `MXBM_PERFECT_TAB` — drop `lkey` and the comparison it exists for.** Inside a group
the bucket fixes the key's top `bb` bits and the sub-mask its bottom `sm`, so exactly
`24 - bb - sm` = 7 bits vary, and `hk = (key >> sm) & 127` selects precisely those. The
table is therefore a **perfect hash**: two elements share a chain if and only if they
share the full key, which makes `lkey[oth] == key` in the walk a tautology. Removing it
takes 4 B/element of shared memory *and* a shared load plus branch out of the innermost
loop of the chain walk. `lwork[pos*INW]` already carries the key in its low 24 bits in
every mode, so nothing else needed it. Guarded in `cuda_solver.cu`: every geometry on the
`bb + sm = 17` line leaves exactly 7 bits, but a hand-set `MXBM_BB`/`MXBM_SM` off that
line would silently combine unequal keys, so it throws instead.

**2. `MXBM_SPILL` — split an overflowing group instead of dropping it.** `kFCap` had to
cover the group's *tail* (352 at a mean of 264) while only the mean is ever staged. When a
group overflows, splitting it on one more key bit is exactly safe — two elements whose
keys differ in that bit can never collide, so no pair is lost or doubled — and the split
halves the group, so it is self-limiting. `kFCap` can then sit near the mean.

**The split level must be chosen before any part is walked.** The obvious version
escalates when a part overflows, which re-emits everything the earlier, coarser parts
already emitted. That does not look like a failure: bucket counts inflate, real elements
get evicted, and it presents as **lost goldens with a zero drop counter**. The shipped
version pays one word-0 pass to count all 8 possible sub-parts at once and picks the level
from real sizes — and only in the rare overflow case, so the common path is untouched.

**Where the time actually came from, which was not where I expected.** `kFCap` 320 beats
`kFCap` 300 even though 300 buys r3 a 4th block, so it is not r3's occupancy. It is
**rounds 1 and 2 crossing to 4 blocks/SM** — at 320 both their shared memory *and* their
register count fall below the threshold (80 → 64 registers, which at 256 threads is 3
blocks → 4):

| | entry | r1 | r2 | r3 | r4 |
|---|---|---|---|---|---|
| `kFCap` 336 | 6 | 3 | 3 | 3 | 4 |
| `kFCap` 320 | 6 | **4** | **4** | 3 | 4 |

This is the "partial win before a total one" the budget below predicts: `B` differs per
round (r3 84 B/element, r1/r2 76, r4 72), `blocks/SM` is per kernel, so the lighter rounds
cross first. **Round 3 is still at 3 blocks and still needs its own cut.**

> **Correction (2026-07-29): those four `B` values are all pre-`MXBM_PERFECT_TAB`, and
> r1's is two further changes stale.** Read out of the shipping archive today —
> `cuobjdump -res-usage build/libmxbm_cuda.a`, whose eleven dependencies are all older
> than the archive, so it describes current sources. Shared-per-block fits
> `S = B × FCAP + 548` **exactly** on all four rounds, and the constant is exact too:
> 548 = 4×128 (`tab`) + 4 (`gcount`) + 4×8 (`cnt8`).
>
> | | SHARED | FCAP | B today | B above |
> |---|---|---|---|---|
> | r1 | 18 980 | 288 | **64** | 76 |
> | r2 | 23 588 | 320 | **72** | 76 |
> | r3 | 26 148 | 320 | **80** | 84 |
> | r4 | 22 308 | 320 | **68** | 72 |
> | terminal | 9 732 | 384 | **24** | — |
>
> Every row is exactly 4 B high because it predates `MXBM_PERFECT_TAB` dropping `lkey`.
> r1 is 12 B high because it also predates `kGiIsLead` (LM_SEED reads `gi` through
> `lleaf[0]`, −4 B) and `LEAFW` 2 → 1 (−4 B). Terminal is `24 × 384 + 516` — no `cnt8`.
> The kernel source has carried the right numbers since the per-round cap landed:
> `fused_round.cuh:284-285` reads *"r3 80, r2 72, r1 64"*.

Isolated, because the two changes arrived together: at `kFCap` 336 spill is worth nothing
(35.16 vs 35.24 — it never triggers); at `kFCap` 320 the no-spill build is the same speed
but **drops 142 elements**. So the speed is entirely the smaller allocation, and spill's
whole contribution is making that allocation *correct*.

| | ms | notes |
|---|---|---|
| baseline | 35.57 | mean of 3 × 300 nonces |
| + perfect table | 35.45 | −0.12, and 4 B/element |
| + spill, `kFCap` 320 | **34.32** | **−1.25** |
| + spill, `kFCap` 300 | 34.40 | r3 gains a block, and it is *slower* |
| + spill, `kFCap` 288 | 34.54 | |

</details>

### Occupancy is worth real time, and shared memory is the only gate
<details>
<summary>Details</summary>

*(Measured 2026-07-26. This reverses "occupancy, measured properly, still does not help",
which was measured at 2 → 3 blocks and is not what 3 → 5 does.)*

The geometry ladder is `bb + sm = 17`, but the two knobs do different jobs: **`bb` alone
sets the bucket count and therefore scatter locality; `sm` alone sets the group size and
therefore every shared array.** Nothing forced them to move together — the ladder walked
the diagonal because rescan traffic was assumed expensive, and
[bytes are nearly free](#bytes-are-nearly-free-per-element-work-is-not) says it is not.
Raising `sm` at fixed `bb` shrinks shared memory, leaves the footprint identical, and
costs only rescan. The 2×2 that separates the two effects:

| | rescan | blocks/SM | ms |
|---|---|---|---|
| (16,1) `kFCap` 384 | 2× | 3 (24 warps) | **35.29** |
| (16,2) `kFCap` 384 | 4× | 3 (24 warps) | 42.94 |
| (16,2) `kFCap` 224 | 4× | 5 (40 warps) | 37.72 |

**Occupancy 3 → 5 blocks is worth 5.22 ms.** The extra rescan costs 7.65 ms, so this
particular way of buying it loses — but the prize is real and had been hidden inside a
confound every previous geometry measurement carried.

Confirmed at *fixed* geometry, by shrinking `kFCap` past the safe point. Drops remove
downstream work and flatter the result, so they are reported with it; at `kFCap` 280 the
0.56 % dropped accounts for ~0.2 ms of the 1.8:

| `kFCap` | blocks/SM | ms | drops | verified/solve |
|---|---|---|---|---|
| 384 | 3 | 35.23 | 0 | 2.04 |
| 280 | 4 | **33.42** | 0.56 % | 1.79 |
| 264 | 4 | 31.32 | 2.00 % | 1.16 |
| 240 | 4 | 24.52 | 7.08 % | 0.21 |

So **3 → 4 blocks is worth ~1.6 ms (4.5 %) at the shipping geometry**, and the lower two
rows are too distorted to use for anything but confirming the direction.

**It is blocks that pay, not warps.** Raising `kWG` also raises warps/SM at no shared-memory
cost, and it is slower — 24 warps at `kWG` 256 beats 36 warps at 384:

| `kWG` | blocks/SM | warps/SM | ms |
|---|---|---|---|
| 256 | 3 | 24 | **35.23** |
| 384 | 3 | 36 | 36.81 |
| 512 | 2 | 32 | 37.31 |
| 768 | 1 | 24 | 42.85 |

A bigger block spreads one group of 264 across more lanes that then idle through the walk
and wait at the same barriers. What hides a memory stall is another *independent group*,
not more lanes on the same one.

**The sub-mask and the rescan were welded together, and unwelding them did not help.**
A block re-reads its whole bucket only *because* it owns one sub-mask of it. `SUBPASS`
(`-DMXBM_SUBPASS=1`) breaks that: one block owns a whole **bucket**, reads every element's
sub-mask bits once into a byte array in shared, then sweeps the sub-masks out of shared.
Finer sub-masks then shrink the group — and therefore shared memory — at **no extra global
traffic at all**, which is strictly better than the geometry ladder could ever offer.
It is correct (goldens 3/3, drops 0 at every setting) and it is **still monotonically
worse**:

| | blocks/SM | group | `kFCap` | fill | blocks × group | ms |
|---|---|---|---|---|---|---|
| baseline, no sub-pass | 3 | 264 | 384 | 0.69 | 792 | **35.24** |
| sub-pass sm=1 | 3 | 264 | 384 | 0.69 | 792 | 35.70 |
| sub-pass sm=2 | 4 | 132 | 256 | 0.52 | 528 | 38.08 |
| sub-pass sm=3 | 5 | 66 | 176 | 0.38 | 330 | 45.34 |
| sub-pass sm=4 | 5 | 33 | 128 | 0.26 | 165 | 65.59 |

**Time is monotonic in `blocks × group`, not in blocks and not in warps.** That is the
quantity to optimise, and it has a closed form. `kFCap` must cover the group's *tail*
while only its *mean* is ever staged, and for a Poisson group the ratio is
`1 + 8/√mean` — so shrinking the group wastes a steadily larger fraction of shared memory
on a reservation that is almost never used:

```
useful elements resident per SM  =  100 KB / ( (1 + 8/sqrt(mean)) x B )
                                          B = shared bytes per staged element
```

| mean | tail/mean | useful (at B = 84) |
|---|---|---|
| 528 | 1.35 | 904 — but `kFCap` 743 fits only **1** block, so 528 |
| **264** | **1.49** | **817 → 3 × 264 = 792, the shipping point** |
| 132 | 1.70 | 719 |
| 66 | 1.98 | 614 |

Coarser is better on fill and loses to integer block flooring; finer is worse on fill.
**sm=1 is a genuine optimum, and geometry is now closed from both directions.** `B` is the
only term left.

</details>

### Round 1 takes a fifth block (−0.15 ms), via a per-round group cap
<details>
<summary>Details</summary>

*(2026-07-26.)* r1 had two redundancies the other rounds do not:

- **its `gi` IS its leaf.** Both are the seed index, written from the same `rec0 >> 32`.
  LM_SEED now keeps one copy and reads `gi` through `lleaf[0]`, dropping a whole
  4 B/element array — and, unexpectedly, its register count with it, **64 → 48**.
- **`LEAFW` was 2 but only slot 0 is ever touched** (`SIN` = 1, and the ctree build reads
  index 0 of both parents), so it is 1.

That takes r1 to 64 B/element. It is then held back by a *global* `kFCap`: at 320 it fits
4 blocks/SM, at 288 it fits 5, but lowering `kFCap` globally makes every other round spill
more and the total does not move. **`FCAP` is therefore a per-round template parameter**,
which is the natural completion of "B differs per round" — r1 runs at 288 and pays its own
slightly higher spill rate, r2/r3/r4 stay at 320.

r1 marginal **5.28 → 5.13 ms**, end-to-end 34.30 → 34.17 across three interleaved runs.
No launch bound is needed to hold 5 blocks: 48 registers and 18 980 B against the 48 and
19 456 that 5 blocks requires — i.e. exactly zero register headroom, because registers are
allocated per warp in units of 256, so 49 rounds to 1 792/warp = 9 warps/sub-partition = 4
blocks. (An earlier version of this line said 51, which is 65 536/(5×256) with that
granularity dropped; at 51 registers r1 gets 4 blocks.) `tests/test_cuda_resources.cpp`
now asserts the whole cliff table against `cuobjdump -res-usage` at build and test time.

**A latent bug surfaced while measuring this.** Each round's template arguments were
written out separately at the launch, at the carveout call and at the bench's occupancy
probe, and had drifted in three of the four: the carveout was being applied to `LM_RD2`
with `OUTSTR` 10 and `LM_EMIT` with `INSTR` 10, **instantiations that no longer run**, so
the kernels that do run never received the shared-memory preference — and the bench's
blocks/SM column was reporting a different kernel than it timed, which is why r1 appeared
stuck at 4. The arguments are now named once as `MXBM_Rn_ARGS` and used by all three.

</details>

### Round 3 does not want a fourth block — occupancy pays only where a round is latency-bound
<details>
<summary>Details</summary>

*(2026-07-26. This closes the "r3 still needs its own cut" lead rather than delivering it.)*

r3 is the one round still at 3 blocks/SM, so it looked like the obvious next 1.6 ms.
`MXBM_NARROW6=1` gets it there: r3's input is r2's output masked to `LOUT` = 400 bits, so
word 6 carries **16 significant bits in a u64**, and moving it to its own `uint16_t` plane
takes r3 from 80 to 74 B per staged element — under the 4-block line at `kFCap` 320, with
no extra spill. The goldens passing is itself the check that word 6 really is that narrow.

It buys nothing. r3's marginal cost is **9.50 ms at 3 blocks and 9.51 at 4**, and the
end-to-end figure does not move (34.31 against 34.20–34.37 baseline).

**And it is not the bank conflicts cancelling a gain**, which is what the stride change
predicted: the conflict degree of a u64 stride *S* is `32/gcd(2S,32)`, so odd strides give
2-way and even ones 4-way, and dropping `lwork` from 7 u64 to 6 doubles the conflicts on
the walk's dominant shared access. Forcing both builds to 3 blocks isolates it — r3
marginal **9.54 with stride 7, 9.51 with stride 6**. The penalty is zero, which agrees
with the earlier finding that [bank conflicts here are a red herring](#where-the-cuda-backend-stands-after-the-fix).

So two independent routes to r3-at-4-blocks — lowering `kFCap` to 300, and narrowing word
6 — are both null, and the reason is visible in the bandwidth column:

| round | GB/s | % of peak | what a 4th block bought |
|---|---|---|---|
| r1 | 252 | 37 % | 5.85 → 5.33 (−9 %) |
| r2 | 336 | 50 % | 10.71 → 10.38 (−3 %) |
| r3 | 537 | **80 %** | 9.50 → 9.51 (**null**) |
| r4 | 587 | 87 % | already at 4 |

**Occupancy pays where a round is latency-bound and not where it is bandwidth-bound** —
more warps cannot hide a stall in a memory system that is already saturated. That refines
the equation below: it predicts where occupancy is *available*, not where it is *worth
having*. r1 and r2 were the two rounds with DRAM headroom and they are exactly the two
that paid; r3 and r4 sit at 80–87 % of peak and are done.

`MXBM_NARROW6` is retained default-off. It is null on speed but real on footprint (−6
B/element of shared for r3), so it is worth having if anything ever makes r3
latency-bound again.

**The open lead, with its budget.** 4 blocks/SM needs ≤ 24 576 B of shared per block
(25 600 minus ~1 KB the driver reserves). At the safe `kFCap` of 352 that is **68.3 B per
staged element against the 84 B used today — a 15.7 B cut.** What is identifiable:

| field | B/elem | why it might go |
|---|---|---|
| `lkey` | 4 | redundant: it is `lwork[pos*INW] & 0xFFFFFF` in every mode |
| `lgi` + `lleaf` | 4 of 20 | the global record already packs them into 2 u64; unpack in the walk |
| `lwork` word 6 | 6 | r3's input is masked to 400 bits, so word 6 carries 16 significant bits of 64 |

That is ~14 B of the 15.7 needed, and **every one of them adds work to the chain walk**,
which is the hottest loop in the kernel (`lkey[oth] == key` runs ~92 M times per round and
would become a u64 load). So this is a real lead with a measured prize, not a free win.

The other route attacks the `1 + 8/√mean` term instead of `B`: **spill group overflow to a
second pass rather than dropping it**, letting `kFCap` fall from the tail (352) to near the
mean (264) — a 25 % cut, enough on its own, and it never touches the walk. It is also the
only one of the two that gets *better* as the group shrinks, since the tail ratio it
removes is exactly what makes finer sub-masks lose.

Note the budget differs per round, because `B` does: r3 is 80 B/element (INW 7, LEAFW 4),
r1/r2 are 72, r4 is 68. `blocks/SM` is per kernel, so the lighter rounds reach 4 blocks on
a smaller cut — which is what shipped. **r3 is the exception and is now closed above: it
reaches 4 blocks and does not care.**

> **Correction (2026-07-29): r1 is 64 B/element, not 72** — `kGiIsLead` drops `lgi` and
> `LEAFW` is 1. r3 80, r2 72 and r4 68 are current. `cuobjdump -res-usage
> build/libmxbm_cuda.a` gives r1 SHARED 18 980 at `FCAP` 288, and (18 980 − 548)/288 = 64.
> The **84 B used today** four paragraphs above is r3's pre-`MXBM_PERFECT_TAB` figure and
> is now 80, which is why that budget closes more easily than it reads.

</details>

---

## Measured results, 2026-07-31

### Co-blocks: the third overlap mechanism works — and it is worth 0.4 ms, not 14
<details>
<summary>Details</summary>

*(Built and measured 2026-07-31. `MXBM_COB_S` / `MXBM_CO_ROUND` on `cuda/pipeline
--fuse`; shipped as [speculative entry](#speculative-entry-co-scheduling-ships-in-the-miner-045-ms).)*

Two overlap mechanisms were measured null and closed: a second **stream** (662 waves of
backlog; the second kernel starts in the last wave's tail) and **same-warp hosting**
(COTENANT: a memory-stalled warp still holds its slot, so entry's arithmetic inside r3's
warps cost 7.25 ms against 2.77 standalone). This is the third: the next nonce's entry
pass as **separate blocks interleaved through a round's own grid** — every S-th block of
the launch seeds instead of matching. Interleaved by block index, not appended: appended
blocks all dispatch in the tail, which is the stream null all over again. Entry's
instructions then issue from their own warps, resident *beside* the round's stalled ones.

Host and stride swept, 60 solves each, sequential bracket 33.9 ms:

| host | S=2 | S=3 | S=5 | S=9 | S=17 |
|---|---|---|---|---|---|
| r2 | | 35.4 | 35.5 | 35.5 | 35.5 |
| r3 | 33.9 | 33.9 | 34.1 | 34.1 | 34.2 |
| **r4** | 33.5 | **33.4–33.5** | 33.5 | 33.7 | 34.0 |

**r4 is the host, S=3 the plateau, −0.4 to −0.5 ms** (KAT green, drops 0 throughout).
Splitting the entry across r3 *and* r4 (`MXBM_CO_ROUND=34`) is worse than r4 alone
(34.6), and r2 — the round with the least idle of either resource — pays outright.

**The attribution, which is the reusable part.** Three builds separate the hosting cost
(sequential bracket 33.9; entry standalone is ~2.6 ms):

| co-blocks do | end-to-end | what it prices |
|---|---|---|
| nothing (`MXBM_COB_DUMMY`, real entry separate) | 33.87 | **displacement: ~0** |
| compute only (`MXBM_CO_NOSCATTER`, real entry separate) | 34.74 | **entry's compute hosted: +0.83** |
| the full entry pass (no separate entry) | 33.41 | **hosting cost 2.1 of 2.6** |

Three facts fall out. **Displacing a third of r4's blocks is free** — a DRAM-bound round
at 82 % of peak genuinely does not need them, which the 3→4-blocks null for r3 already
suggested. **Co-scheduled compute hides at ~2/3** — 2.5 ms of dense siphash costs 0.83
hosted, which is the overlap the same-warp mechanism could never reach. And **the
scatter side does not hide** (~1.3 ms of the 2.1): 33.5 M bucket atomics plus 8 B writes
land on a memory system already at 82 % of peak.

</details>

### fused_pair: two solves' rounds in one launch — the family's ceiling is ~0.5 ms
<details>
<summary>Details</summary>

The general form: `fused_round`'s body became a device function over a `RoundShared`
struct, and `fused_pair` runs **round 4 of solve i and round 1 of solve i+1 as
alternating blocks of one launch**, the two rounds' shared layouts in a union so a block
pays max() rather than sum() (22.3 KB, still 4 blocks/SM; registers take the max of both
paths and land on exactly 64). `cuda/pipeline --pipe2` orchestrates the two-solve
pipeline: r1's output moves to its own double-buffered pair-record set, back-refs are
double-buffered, and the main elem ping-pong hands over between solves. Correct on the
first run (2.03 verified/solve over 60 distinct nonces, drops 0), and:

| | ms/solve |
|---|---|
| sequential bracket | 33.87 |
| `--pipe2`, 1:1 residency | 33.77–33.84 |
| `--pipe2`, r1-major (r4 blocks doubled, 2/3 of residency to r1) | **33.48** |

**The same −0.4 ms the entry co-blocks buy, by a much heavier route.** r1 at a 1:1
residency split gets 16 warps against its standalone 40 and overlaps nothing; giving it
2/3 of the blocks recovers exactly the gain the simpler mechanism already had. The
conclusion to carry: **r4's exploitable idle is ~0.5 ms/solve, whatever co-work is
offered** — entry's compute, or a whole round. The 1.71× roofline is now unreachable by
three independent mechanisms, and the family is closed with its 0.5 ms harvested.

</details>

### Occupancy is closed from BOTH resources — r2 sits on the whole register file
<details>
<summary>Details</summary>

*(The `MXBM_MB_SEED` / `MXBM_MB_RD2` probe, 2026-07-31.)* The occupancy budget in
[shared memory](#occupancy-is-worth-real-time-and-shared-memory-is-the-only-gate) said
`B` is the only term left. It is not the only gate: **r2 runs 4 blocks × 256 threads ×
64 registers = 65,536 — the entire register file.** Forcing a 5-block register budget
with `__launch_bounds__` (the MINBLOCKS template parameter) changes nothing, because the
shared side caps first: 5 × 23.6 KB > 100 KB at `kFCap` 320, and an FCAP that fits five
blocks (276) sits below mean + 1σ of the group tail, where the spill path stops being
rare. r1 is the same story one block up (5 × 19.0 KB, 6 needs FCAP under the mean).
**Every further occupancy route for r1/r2 now needs BOTH fewer shared bytes per element
and fewer registers per thread simultaneously**; neither alone moves blocks/SM. Closed.

</details>

### Speculative entry co-scheduling SHIPS in the miner (−0.45 ms)
<details>
<summary>Details</summary>

The co-blocks result needs the *next* nonce's prePow during the current solve, and the
`Solver` interface only sees one nonce at a time. No interface change: the engine walks
nonces at a fixed stride within a job, so `CudaSolver` **learns the delta between
consecutive calls** and seeds (nonce + delta) inside round 4's launch; the next call
skips its entry pass when the prediction was right, and a job change simply misses once
(one solve at the plain cost, out of the hundreds a job lasts). Entry output moves to a
dedicated dense buffer (+0.39 GiB) because round 2 overwrites `elem[0]` mid-solve;
if that allocation fails, or under `MXBM_NO_SPEC=1`, everything runs exactly as before.

Bracketed on the shipped miner (`--benchmark BEAM-III`, 60 s arms, same session):

| | ms/solve | sol/s |
|---|---|---|
| `MXBM_NO_SPEC=1` | 33.8 | 58.7 |
| **speculation on** | **33.4** | **59.5** |
| `MXBM_NO_SPEC=1` | 33.9 | 58.7 |

Verified solutions/solve identical (2.00), drops 0, all 45 tests green. The hosted r4
lands on **exactly 64 registers** — the 4-blocks/SM cliff — and
`tests/test_cuda_resources.cpp` pins it there with its own contract row.

*Bookkeeping note:* moving the shared arrays into `RoundShared` costs each round
+12–20 B of shared for the struct's placeholder members and padding. blocks/SM held on
every kernel and registers held or fell (r4 47 → 46); the resource contract was
re-baselined per its own RESOURCE DRIFT rule.

</details>

### The eco sweep: (17,0) crosses over below ~190 W; R2_FULL never does
<details>
<summary>Details</summary>

*(`benchmarks/eco_sweep.sh`, 2026-07-31. The below-160 W inversion said the low-power
currency is core cycles, not bytes — this reprices the two instruction-vs-bytes trades
at every cap. KAT green and drops 0 at all 28 points.)*

The premise held in general and failed in the specific. At 100 W the solve is 103.7 ms
against 33.8 at stock — 2.94× slower at 3.38× less clock — so essentially *everything*,
DRAM-bound rounds included, is issue-limited at the bottom: the LSU feeds DRAM at a rate
that scales with core clock. Two settled stock results were therefore re-run under caps:

| cap | base (16,1) | (17,0) | Δ | `MXBM_R2_FULL` | Δ |
|---|---|---|---|---|---|
| 100 W | 103.7 ms | **100.9** | **−2.6 %** | 112.7 | +8.7 % |
| 120 W | 76.8 | 76.9 | +0.2 % | 82.8 | +7.8 % |
| 140 W | 62.3 | **61.8** | **−0.7 %** | 66.2 | +6.3 % |
| 160 W | 51.5 | **50.9** | **−1.1 %** | 54.4 | +5.6 % |
| 180 W | 44.1 | **43.5** | **−1.2 %** | 46.3 | +5.0 % |
| 210 W | 36.9 | 39.2 | +6.3 % | 40.9 | +10.9 % |
| 285 W | 34.1 | 37.8 | +10.9 % | 39.3 | +15.4 % |

**Geometry (17,0) — sub-mask rescan removed — crosses over at ~190 W** and wins ~1 % in
the 140–180 W band, 2.6 % at the card's floor. Its stock loss is scatter locality, a
DRAM-side price that a capped card has spare bandwidth to pay; its win is the removed
staging instructions, a core-side saving that a capped card values. `MXBM_BB=17` selects
it at runtime; it needs the (17,0) footprint (8.35 GiB).

**Shipped as behavior 2026-07-31.** The CUDA solver now selects (17,0) itself when the
board power limit observed at startup is below 190 W and the 8.35 GiB fits — applied
first (`--pl`), then observed (NVML), then sized, so a cap set outside MXBM
(`nvidia-smi -pl` before launch) counts the same as one MXBM applied. Chosen once per
run: a geometry switch is a multi-GiB realloc, so a cap changed mid-run gets a one-line
restart notice instead of a re-selection. `MXBM_BB` still overrides both ways.

**Live at 160 W in the MINER loop, same day: a wash, not the sweep's −1.1 %.** First
capped measurement in the loop users run (`--benchmark`, verify and speculative
co-tenant entry included) rather than the pipeline replay the sweep used: interleaved
brackets gave (17,0) 51.7/51.7/52.0 ms vs (16,1) 51.8/51.8 — 0 ± 0.3 %. Not a
contradiction, a loop difference: the spec entry rides round 4, exactly where (17,0)
changes the block population, and plausibly absorbs the mid-band saving. The selection
stays (harmless at worst mid-band, and the floor's +2.6 % was measured at 3× the
mid-band margin) — but **the floor claim is still pipeline-loop only**: a 100 W
miner-loop bracket is the cheap test that would settle whether the auto-selection pays
where it claims to, and it is owed.

**The floor bracket ran the same afternoon, and its positive control FAILED.** At
100 W the miner loop read a wash across 10 arms (detrended (17,0) −0 to +0.4 % *worse*),
so the sweep's own binary was re-run as the control: base's first-run absolute
reproduced (104.3 vs the sweep's 103.7) but **bb17 did not** — 106.5/111.4 against the
sweep's 100.9, i.e. the −2.6 % floor prize failed same-day reproduction in the loop
that produced it. Both loops also showed a ~2 %-per-arm monotonic drift (up when
arms heat-soak back-to-back, down after a warmup — core at 41–45 °C throughout, no
throttle flags, no locked clocks; GDDR temperature is the unexposed suspect) that sets
today's measurement ceiling well above the effect size. Standing: **the auto-selection
currently has no reproducible prize in either loop**; it keeps its default only because
nothing is released and the deciding measurement is cheap — re-run the pipeline control
at 100 W from a clean cold boot; if the −2.6 % does not come back, the selection
defaults off until something reproducible claims the band.

**The control ran post-reboot on a cool card the same evening, and the prize did not
come back: the selection is disarmed.** Base 103.90 ms — reproducing the sweep's 103.7
to 0.2 ms, which validates the conditions — and (17,0) **105.72, +1.8 % worse**,
against the sweep's 100.9. Same binary, same cap, same arm order as the sweep
(`cuda/pipeline`'s mtime is 2026-07-26: both measurements ran it unmodified), so the
sweep's bb17 row is irreproducible *on its own instrument* — and that instrument is
additionally five days stale against the shipping kernels, so even at face value it
described a build the miner no longer runs. `kRbLowPowerW` is now 0: every wattage
hint is inert, the restart notice can
never fire, and (17,0) is reachable only by `MXBM_BB=17`. Everything else stays live —
the apply-then-observe startup order, the observed-limit hint through the CUDA ctor,
the notice machinery — because Tier 2's eco pipeline (or a crossover that reproduces)
re-arms the policy by setting that one constant. The restart notice itself was
verified live before the disarm (start capped at 100 W, `nvidia-smi -pl 285` mid-run:
fired once, immediately, correct direction, then silent for the rest of the run), and
the same runs priced the guard it provides: (17,0) held at a 285 W stock limit costs
**~23 % in the miner loop** (40.9 ms vs 33.1) — worth knowing for whenever the policy
is re-armed.

**`MXBM_R2_FULL` loses at EVERY cap, monotonically.** This kills the clean form of the
instruction-currency theory: round 2's 14-siphash rebuild is ~500 ALU ops per element
against the ~10 extra memory instructions the full record costs, and the ALU side still
wins at 765 MHz. The rebuild's arithmetic hides in stall shadows at low clock exactly as
it does at stock — what a low cap makes expensive is *memory-instruction* work, not
hidden ALU work. **Re-derivation is the right trade at every power level this card can
run**, which closes the question the 2026-07-28 quad-record cap sweep opened from the
other side.

</details>

### The 5001 memory rung: −8.5 to −14.4 % below its ~173 W crossover — the largest low-band lever ever measured here
<details>
<summary>Details</summary>

*(2026-07-31, `docs-internal/rootruns/run_mclk_eco.sh`, raw logs in
`rootruns/mclk-eco/`. The lead's mechanism was the ledger's own: below ~190 W the
solver is issue-bound while the memory clock does NOT scale with a core cap — 10251 MHz
at 100 W and at 285 — so the GDDR6X interface draws a fixed slice of a tiny budget for
bandwidth nothing is using. `-lmc 5001` frees interface watts the capped core re-spends
as clock. Method carried all three of the same day's lessons: miner-loop arms through
the shipping `--benchmark` binary, ABBA brackets per cap so drift cancels in the pair
means, the memory clock sampled during every arm — no arm refused the rung — and draw
and J/solution read from the energy counter.)*

Nine caps × two rungs, drift-cancelled pair means:

| cap | 10251 MHz | 5001 MHz | Δ speed | J/solution 10251 → 5001 |
|---|---|---|---|---|
| 100 W | 108.3 ms | **92.9** | **−14.3 %** | 5.36 → **4.57** |
| 120 W | 79.7 | **68.3** | **−14.4 %** | 4.79 → **4.09** |
| 140 W | 63.3 | **56.5** | **−10.8 %** | 4.43 → **3.94** |
| 160 W | 52.7 | **48.2** | **−8.5 %** | 4.18 → **3.79** |
| 180 W | 44.8 | 47.0 | +4.9 % | 3.98 → 4.12 |
| 200 W | 39.4 | 46.8 | +18.8 % | 3.94 → 4.54 |
| 220 W | 35.4 | 46.6 | +31.6 % | 3.92 → 4.93 |
| 250 W | 34.3 | 46.6 | +35.9 % | 4.29 → 5.25 |
| 285 W | 33.5 | 46.5 | +38.8 % | 4.78 → 5.27 |

**The crossover is ~173 W** (linear between −8.5 % at 160 and +4.9 % at 180) — above
the lead's own demand-arithmetic prediction of 140–160 W: the freed interface watts
buy more than the arithmetic priced.

**Above ~200 W the rung column is a wall, and the wall is a measurement.** 46.5–46.8 ms
flat from 200 W to 285, with draw saturating at ~230 W under a 285 W cap — the card
cannot even spend its budget. That is this solver's own DRAM roofline at the rung:
13.0 GB/solve / 46.5 ms = **280 GB/s sustained, 87 % of the rung's 320 GB/s peak** —
the same saturation behavior lolMiner shows at stock, now reproduced on our own solver
by shrinking the interface instead of growing the demand. It also explains the shape of
the low-band wins: at 160 W the rung arm (48.2 ms) sits 3.7 % off that roofline, so the
gain is already bandwidth-clipped; by 120 W the core is the only constraint at either
rung and the win is the full interface-power refund (−14 %).

**160 W + 5001 is the new global efficiency record: 3.79 J/solution** (0.264 sol/s/W,
41.9 sol/s at 159 W measured draw), beating the previous peak of ~3.83 at 200 W on
stock memory. The efficiency-optimal operating point moved, and it moved onto the rung.

What this does to the standing conclusions:

- **Capped-rig guidance changes**: below ~170 W the cap should always be paired with
  `-lmc 5001` (`nvidia-smi -lmc 5001,5001`, or MXBM's `--mclk 5001` under root). The
  head-to-head low band narrows by roughly a third — at 160 W MXBM moves 38.3 →
  41.9 sol/s against lolMiner's ~50 — without closing.
- **A strategy note for the eco-pipeline question**: a store-everything design needs
  ~17.7 GB/solve, which floors at ≥63 ms on the rung's 280 GB/s — the rung is a lever
  only a re-derivation design can pull this hard. The owed lolMiner arm at the rung
  would measure exactly this; if its low-cap numbers *also* jump, part of its 2.3×
  work-per-clock dissolves into memory-interface power instead.
- The 10501 null stands unchanged: that was an up-rung being refused; this down-rung
  was honored in every arm and the `MEM` column proves it.

**The lolMiner arm ran the same evening** (`rootruns/run_mclk_lol.sh`, same ABBA
discipline, MEM sampled — no refusals; its sol/s from its own steady 15 s windows with
the ramping first window dropped, its definition throughout). Drift-cancelled pair
means, reported sol/s:

| cap | 10251 | 5001 | Δ |
|---|---|---|---|
| 100 W | 23.25 | **28.85** | **+24.1 %** |
| 120 W | 32.2 | **34.05** | **+5.7 %** |
| 140 W | 39.75 | 34.0 | −14.5 % |
| 160 W | 47.85 | 34.1 | −28.7 % |

Three readings, one per hypothesis the arm existed to test:

- **The rung is NOT exclusively ours — but the 130–173 W window is.** lolMiner's
  crossover is ~126 W against our ~173: between them we gain 8–11 % while it loses
  14–29 %. That window is exactly where eco-minded rigs run.
- **Its rung plateau is the store design's signature**: ~34 sol/s flat from 120 W up
  = 17.7 GB/solve against the reduced interface (263–301 GB/s sustained depending on
  its per-solve counting — at or near the rung's ceiling under either basis), the
  mirror of our 43.6 sol/s plateau at 13.0 GB/solve. Both miners now exhibit the
  same roofline behavior on the same rung, each at the height its bytes-per-solve
  dictates — re-derivation's smaller appetite is worth +28 % of plateau.
- **Part of the "mysterious 2.3× work-per-clock" dissolves at the floor**: +24 % at
  100 W from the interface refund alone says a large slice of lolMiner's 100 W budget
  was memory-interface power, not core work. Below ~126 W its refund exceeds ours
  (+24 vs +14 %) because it was burning more interface watts to begin with.

Net head-to-head, each at its best memory clock per cap (reported bases, not
strictly comparable): the low-band gap roughly **halves in 120–160 W** (~25–29 % →
+16/+11.5/+14 % at 120/140/160) and **widens at the 100 W floor** (+32 %). For the
Tier-2 eco-pipeline question this re-prices the prize: a store design cannot hold its
speed on the rung in the 130–173 W window, so the pipeline's projected +20–30 %
shrinks to ~+11–16 % there — the deep floor (≤120 W) remains its strongest case.

</details>

### The rung regime does not reopen the closed geometry trades — quad and (17,0) both null
<details>
<summary>Details</summary>

*(2026-07-31, same evening as the rung sweeps. The rung created a new operating
regime, and every closed geometry trade had been priced only in the old ones —
the one honest doubt left on two of the ledger's closures. Both re-priced in one
ABBA sitting: rung locked and sampled every arm, miner loop, 40 s arms, pair
means so drift cancels.)*

**Quad record × rung, 160/140 W** — the best-case prediction was here: at
160 W + rung the solver sits 3.7 % off its 280 GB/s roofline, where bytes
convert to time ~1:1, so if quad's byte saving could ever beat its
re-derivation price, this was the spot. It does not: **+13.8 % at 160 W
(48.0 → 54.6 ms pair means), +15.1 % at 140 W (56.7 → 65.2)** — the same
~+14 % it costs at stock, unmoved by the regime. The reading: the arithmetic
price is paid in core cycles, and a cap starves exactly those — the extra
work inflates in ms terms with the same slowed clock everything else runs on,
while the byte refund stays under the noise at every measured point. The quad
record remains footprint-only in every regime measured.

**(17,0) × rung, 120/100 W** — the deep floor, where the core is the only
constraint and the disarm verdict had no coverage: **−0.4 % at 120 W
(68.5 → 68.2 ms), +0.7 % at 100 W (88.9 → 89.5)** — opposite signs, both
under the 1 %-of-a-solve floor. A wash, matching the stock-memory
reproduction failure. The disarm verdict now covers the rung regime too.

The base arms doubled as a same-day reproduction of the rung sweep: 47.8–48.2 ms
and 3.76–3.79 J/sol at 160 W + rung, against the sweep's 47.1 ms / 3.79 record.
Artifacts: `docs-internal/rootruns/rung-compose/`.

</details>

### The round-2 instruction census: 86.5 % hash arithmetic, and one named lever

<details>
<summary>Details</summary>

*(2026-07-31, ncu `--set full` on the shipping r2 instantiation — the fused
round at (16,1), 64 regs, 23.6 KB shared, 131072×256 — profiled at stock,
priced for the floor. The eco attribution had named r2 as the floor's largest
and growing item; this is the census it asked for. Retired-instruction counts
are clock-invariant, so a stock profile transfers: r2's 26.19 M elapsed cycles
÷ the floor's 0.885 GHz = 29.6 ms, against the attribution's measured 29.4 —
the floor's r2 time is fully explained by cycles × clock, and every cycle cut
at stock converts 1:1 to floor milliseconds.)*

**The mix.** 3.30 G executed instructions per kernel; the ALU quartet is
86.5 % of them — LOP3 28.8 %, SHF 27.8 %, IMAD 16.9 %, IADD3 13.0 %. The
stream IS the hash arithmetic (rotates and xors), which is why ALU is the top
pipe (69.9 %) and why the compiler axis had nothing: there is no fat here,
only work. LDS is 2.8 %, stores 0.8 %. IPC 1.91; 16.5 warp-cycles per issued
instruction; 25.07/32 average active threads (the rescan filter's designed
divergence). Zero spills; occupancy exactly the contract's 4 blocks, registers
AND shared both binding.

**Known-structural, re-confirmed with numbers**: the scatter's uncoalesced
stores (45 % excessive L2 sectors; the claim atomic runs at 81 % excess) are
the all-to-all bucket layout the ledger closed; nothing new there.

**The one named lever: shared-memory bank conflicts at the staged-record
read.** 39 % of shared-load wavefronts are conflict replays (74 M of 189 M,
2.1-way average), and the source page localizes nearly all of it to two
clusters of seven consecutive `LDS.64` — the 7-u64 (56 B, 14-word) staged
record. Stride 14 words over 32 banks → gcd(14, 32) = 2 → exactly the 2-way
measured. The design KNEW this: fused_round.cuh's staging comment prices
conflict degree by stride and chose an odd u64 stride as the stride-family
floor (odd = 2-way, even = 4+). What was never priced is an XOR swizzle to
1-way — and the census shows it is cheap to try: the seven loads share one
base register computed once per element, so a swizzle is ~one ALU op per
element (~0.5 % of the ALU budget), not per access. Constraints for the probe
(now lead 2.9): shared memory is a binding occupancy resource (24.6 of
25.6 KB at 4 blocks), so any layout that grows the arrays loses the fourth
resident block and with it more than conflicts pay — swizzle in place, never
pad. Honest pricing: ncu's 16.9 % estimate is an upper bound; the kernel is
"well-balanced" at stock so replay removal may convert weakly there, but the
floor is core-bound, where LSU replay cycles are the scarce currency, and r2
is 33 % of the floor's time. Sign unknown; a KAT-gated ABBA at stock and
100 W closes it either way. Report: `docs-internal/rootruns/r2-census.ncu-rep`.

</details>

### The lwork SoA probe: conflict-free costs more than the conflicts — a measured loss

<details>
<summary>Details</summary>

*(2026-08-01, closing the census's one named lever. `MXBM_LWORK_SOA=1` lays the
staged work array out column-major — word w of element p at `w*FCAP + p` — so
consecutive lanes read consecutive u64: bank-conflict-free by construction,
same bytes, same shared budget. KAT green, r1 even dropped 46→43 registers.
The measurement said no, twice.)*

| operating point | base (pair means) | SoA | verdict |
|---|---|---|---|
| stock 285 W | 33.4 ms | 33.45 ms | wash |
| 120 W + rung | 70.7 ms | 71.3 ms | **+0.85 %** |
| 100 W + rung | 94.25 ms | 98.3 ms | **+4.3 %** (4.65 → 4.80 J/sol) |

Worse exactly where it was supposed to win, growing toward the floor. The
SASS explains it — the AoS record layout was quietly load-bearing for
vectorization on BOTH shared paths:

- **Stores**: 6 STS.128 in the base become 0 in the SoA build (50 STS.64 vs
  38 — +12, exactly the un-fused pairs). Adjacent record words could fuse to
  128-bit; column-major puts them `FCAP` apart, unfusable.
- **Loads**: 27 LDS.128 become 19 (88 LDS.64 vs 72, +16) — part of the record
  reads were 128-bit too, and the transpose broke those as well.

Two flaws in the design premise, named so they stay named: (1) the
consecutive-`p` assumption holds only in the linear rescan loops — the chain
walk reads records at pointer-scattered positions where no stride is
conflict-free; (2) conflict replays and memory instructions are different
currencies, and on this kernel the replays are the cheap one — they hide
under the ALU pipe (top at 69.9 %), while every added instruction bills the
starved core directly at a cap. The same mechanism as the NARROW6-at-the-floor
loss: instructions are the one thing a capped card cannot afford.

The flag stays in the tree as the closure's instrument (off by default,
flag-off codegen unchanged). With this, the census's non-structural list for
r2 is empty: what remains in the round is hash arithmetic (86.5 %), designed
divergence, and the structural scatter. Artifacts:
`docs-internal/rootruns/soa-floor/`.

</details>

### The compiler axis, re-swept on the July kernels — a clean null

<details>
<summary>Details</summary>

*(2026-07-31. The per-kernel register budgets, ptxas settings and toolkit were
last tuned before the perfect table, spill, LD.128, co-blocks and the quad
record landed — the one stock-speed axis not re-visited since. Six variants,
each rebuilt in an isolated probe dir with the resource guard off, KAT-gated,
register-dumped to prove the flag applied, ABBA-bracketed against the shipping
binary in the miner loop, 40 s arms. Null control read 0.0 %.)*

| variant | what it does | result |
|---|---|---|
| null control | probe dir, no flag change | 33.1 → 33.1 ms (instrument clean) |
| `-Xptxas=-O2` | one -O level down | 0.0 % — codegen-neutral |
| `MB_SEED=6` | r1 squeezed 48 → 40 regs (6 blocks/SM) | +0.15 % (noise) |
| `MB_SEED=4` | r1 relaxed to 64 regs (4 blocks) | **+0.6 %** — the fifth block still pays |
| `MB_RD2=5` | r2 squeezed 64 → 48 regs (5 blocks) | **+0.5 %** |
| `MB_RD2=3` | r2 relaxed to 80 regs (3 blocks) | **+1.2 %** |

The current configuration is the local optimum on the current kernels: r2's
64-register cliff binds from BOTH directions, r1's fifth resident block is
still worth its register price, and ptxas has nothing left on the -O axis.
The installed toolkit (13.3.73) has only a patch step available (13.3.1),
priced as not worth a slot. Registers moved exactly as commanded in every
variant (cuobjdump before/after in the artifacts), so these are real nulls,
not unapplied flags. Artifacts: `docs-internal/rootruns/compiler-sweep/`.

</details>

### Per-stage attribution at the eco points: the floor's time is round 2's
<details>
<summary>Details</summary>

*(2026-07-31, `stage_power.sh` at the two operating points that matter to capped
rigs — 160 W + 5001 (the efficiency record) and 100 W + 5001 (the Tier 2 floor) —
rung held and verified over each whole point, energy from the counter, 8 reps,
45 s-sized runs. The stock table is in performance.md; this is the first look at
where the time goes when the cap actually bites.)*

| stage | stock 285 W, %time | 160 W + rung | 100 W + rung |
|---|---|---|---|
| `entry_scatter` | 7.9 | 8.0 | 8.3 |
| round 1 | 15.2 | 14.9 | 15.1 |
| **round 2** | **30.2** | **31.4** | **33.1** |
| round 3 | 28.0 | 32.5 | 23.7 |
| round 4 | 16.1 | 19.4 | 14.8 |
| terminal | 2.8 | 3.4 | 2.5 |

Two regimes, opposite signatures:

- **160 W + rung is bytes-bound.** r3 and r4 — the DRAM-heavy rounds — inflate to
  52 % of the solve between them (r3 9.5 → 16.0 ms, +69 %; r4 5.5 → 9.6, +75 %):
  the rung roofline biting, exactly where the rung stops paying above ~173 W. Not
  actionable: the byte lever is closed in this regime too (quad's refund never
  surfaces against its arithmetic price — see the composition nulls above).
- **The floor is round 2's.** r2's share grows monotonically toward the floor —
  30.2 → 31.4 → 33.1 % of time, 34.0 % of energy at 100 W, the largest single
  item — while the DRAM rounds *shrink* (bandwidth stops binding when the core
  runs at 885 MHz). r2 is the compute round; at the floor, per-element
  instructions are the currency, and the floor is Tier 2's strongest band. An
  instruction census on r2 priced at 885 MHz is the follow-on lead.

Method caveat, visible in the closures: at 160 W + rung the stage sums overshoot
(109.7 % of solve time, −12.6 % energy) where stock closed at 100.1 %/0.8 % and
the floor at 97.4 %/+2.7 %. Under a cap, replaying one stage shifts the
governor's sustained clock (the r2-heavy mix ran at 1695 MHz, the r3-heavy at
2610, baseline 2310), so the non-replayed stages no longer cost what baseline
charged them — single-stage amplification over-attributes by the shift. The
%-shares are the readable quantity at that point; absolute per-stage ms there
carry ~±10 %. At the floor the clocks barely move between mixes (885 baseline,
750–960 across arms), which is why its closure is tight and the r2 verdict is
solid. Efficiency cross-check: baselines read 7.71 J/solve at 160 W + rung vs
8.88 at 100 W + rung — the same ~15 % efficiency drop toward the floor the miner
sweeps measured (3.78 → 4.43 J/solution).

Artifacts: `docs-internal/rootruns/stage-eco/`.

</details>

### lolMiner is NOT duty-cycling — the low-end gap is real work-per-clock
<details>
<summary>Details</summary>

*(2026-07-31, the test [the inversion
section](performance.md#-below-160-w-the-clock-gap-inverts-and-the-clock-explanation-stops-applying)
named and nobody had run: the clock-sample DISTRIBUTION, at 10 Hz, under both miners'
benchmarks at the same caps.)*

| | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|
| lolMiner, 100 W | 360 | 405 | 480 | 600 | 705 |
| MXBM, 100 W | 525 | 660 | 750 | 825 | 855 |
| lolMiner, 140 W | 810 | 885 | 975 | 1020 | 1065 |
| MXBM, 140 W | 1020 | 1185 | 1230 | 1275 | 1320 |

A duty-cycling miner would be bimodal — a burst mode and an idle mode with little in
between. lolMiner's distribution is **broad and unimodal** (100 W histogram in 150 MHz
bins: 300:199, 450:132, 600:125 — no idle peak, no burst peak). It genuinely executes at
a median 480 MHz and still outsolves MXBM at 750 MHz by ~22 %. **The below-160 W deficit
is confirmed as ~2.3× work per core cycle, measured, with the sampling artefact
excluded.** Together with the eco sweep — our own instruction trims recover 1–2.6 % —
the conclusion is that the low-end gap is a property of the solver's *organization*
(rescans, chain walks, per-element bookkeeping), not of any toggle this pipeline
exposes. The 2026-07-29 closure of the low-power goal stands, now with its mechanism
named.

*Amended same day: the parenthesis was tested and is wrong. [The reorganization
probes below](#the-solver-reorganization-probes-the-cycle-deficit-is-not-bookkeeping)
measured the bookkeeping at ~0.6 ms of r1's 5.2 ms of core cycles — the deficit is
real but its mechanism is NOT rescans/chains/bookkeeping, and it remains unexplained.*

*Resolved later the same day, by measurement rather than inference:
[lolMiner's own kernels](#lolminer-measured-under-ncu-the-state-storing-design-confirmed--and-its-54-sols-ceiling-is-a-dram-roofline)
show a state-storing streaming design with no derive or rebuild arithmetic at all —
the work-per-clock gap is the absence of our arithmetic, not a leaner version of it.*

</details>

### The solver reorganization: the proposal behind the probes, condensed
<details>
<summary>Details</summary>

*(2026-07-31. The full proposal — its design options, risk register and de-risking
ladder — is `docs-internal/SOLVER_REORG.md`; this is the short form, kept here because
the C/D measurement it stood on and the closure it ended in belong to this ledger. It
was closed the same day it was proposed, by its own probes, before the prototype was
paid for. The probe write-up is
[the section below](#the-solver-reorganization-probes-the-cycle-deficit-is-not-bookkeeping).)*

**The measurement it stood on.** Per-round marginal cost at two locked core clocks
(`nvidia-smi -lgc`, 2600 vs 800 MHz, clock ratio 3.25×), `MXBM_ROUND_REPS` replay,
split into a core-cycle part `C` (scales with clock) and a clock-invariant part `D`
(DRAM/fixed), `C = (m800 − m2600)/2.25`:

| stage | @2600 | @800 | scaling | **C (core cycles)** | D (invariant) |
|---|---|---|---|---|---|
| entry | 2.58 | 8.34 | 3.23× | 2.56 | ~0 |
| **r1** | 5.29 | 16.92 | **3.20×** | **5.17** | 0.12 |
| **r2** | 10.55 | 33.57 | **3.18×** | **10.23** | 0.32 |
| r3 | 9.62 | 23.58 | 2.45× | 6.20 | 3.42 |
| r4 | 5.45 | 10.57 | 1.94× | 2.28 | 3.17 |
| terminal | 0.95 | 2.67 | 2.81× | 0.76 | 0.19 |
| **whole solve** | 34.95 | 97.24 | 2.78× | **~27** | ~7 |

**r1 and r2 are ~98 % core-cycle-bound at stock** — they scale with the clock to
within measurement error, so 15.4 of the 35 ms solve is pure core cycles in those two
rounds alone, ~27 of 35 overall. That is the same currency as the low-power gap
(lolMiner's measured ~2.3× work per core cycle), which is what made the proposal look
like one lever moving stock speed and the whole power curve at once.

**The premise.** Order-of-magnitude, ~130–160 B move through shared memory per element
per round (~35–45 wavefronts) against ~10 global memory instructions and ~100–200 ALU
ops — so the cycles were read as *organization*: staging stores, chain build and walk,
the sub-mask rescan, barriers. Not hashing, not DRAM.

**The design.** Option B, the real candidate: emit into 2^21 fine buckets (mean
occupancy 16), one **warp** per bucket, `__match_any_sync` on the residual key bits,
partner state fetched by `__shfl_sync` from registers — the chain table, the walk, the
rescan and every barrier deleted, and at ~4.8 GiB it would have been the reach lever
too. Option A (keys-only staging + a counting sort making equal-key runs contiguous +
L2 gather, geometry unchanged) and Option C (match-on-load) were the fallbacks.
Predicted, conservatively: r1 5.3 → ~3.0–3.5 ms, r2 10.6 → ~6.5–7.5, solve ~28–30 ms
/ 63–68 sol/s — and a 2× r1+r2 cycle cut would have flipped the 140–180 W
head-to-head, the band where the comparison is currently lost.

**The probes killed it in one day, in order:**

- **P2** (`cuda/emit_shape_probe.cu`): thin records cannot scatter into fine buckets.
  8 B records at 2^21 buckets run at **0.22×** their 2^16 rate — once 2M bucket-tail
  sectors exceed the 48 MB L2, sibling-write merging fails and every 8 B store pays a
  read-modify-write sector. The cliff starts at bb = 18. Fine bucketing cannot live in
  the global layout, so the design was re-scoped: fine buckets created at match time,
  in shared memory, inside a coarse bucket — Option A's shape with Option B's core.
- **P3** (`cuda/sorted_r1_probe.cu`): the re-scoped sorted-runs r1 is *correct* — it
  emits the exact pair multiset of the shipping kernel — and **1.75× slower** (8.56
  vs 4.89 ms). Its ablation carve decomposes the chain kernel itself: **derive ~2.5 +
  emit arithmetic ~1.8 + all bookkeeping ~0.6 ms** of 4.9. The bookkeeping the
  reorganization would have removed is 12 % of r1, not 55 %.

**Closure.** There is no 2× in the match for any organization of it; any match
rewrite's ceiling is ~0.6 ms/round. The prototype and the r2 conversion were
cancelled, and Options A and C died with the wavefront model they were priced
against. What survives: the C/D decomposition method, the compile-time carve + sink
ablation, the 0.44 ms counting sort (if a future record format ever makes staging the
wall), three CUDA codegen traps documented in the probe header — and the honest
position that lolMiner's work-per-clock advantage is back to unexplained: the one
hypothesis left standing is a state-storing layout that skips the derives, and even a
free derive is ~5–6 ms of ~27 ms of C, short of 2.3×. *(That hypothesis was
[confirmed by measurement the same day](#lolminer-measured-under-ncu-the-state-storing-design-confirmed--and-its-54-sols-ceiling-is-a-dram-roofline).)*

</details>

### The solver-reorganization probes: the cycle deficit is NOT bookkeeping
<details>
<summary>Details</summary>

*(2026-07-31. The clock-scaling C/D split above says r1/r2 are ~98 % core-cycle-bound
at stock — 15.4 of 35 ms. The standing model said those cycles were staging, chain
walks, rescans and barriers, and a reorganization was drafted to remove them:
fine-grained buckets matched in registers. Two probes were designed to kill it cheaply
before a prototype was paid for. Both fired.)*

**P2 — thin records cannot scatter into fine buckets.** `emit_shape_probe sweep`
(committed, results in its header): scatter rate vs bucket count at the three real
record widths. 8 B records at 2^21 buckets run at **0.22×** their 2^16 rate
(131 → 28.6 GB/s), 16 B at 0.42×, 72 B flat at 1.02×; the cliff starts at bb = 18 and
only bb = 17 is free. Mechanism: in-bucket sibling writes only merge while the bucket-
tail sectors stay L2-resident; 2M tails × 32 B = 64 MB > 48 MB L2, and once merging
fails every 8 B write pays a read-modify-write sector. At `-lgc 800` the penalty
mostly hides (0.61× / 0.95×) because the probe is issue-bound there — the memory
system's slack absorbs the amplification — but 140–180 W runs at 1300–1700 MHz, where
it does not. Bonus: the sweep's unclamped counters measured bucket overflow at mean 16
/ cap 32 as 0.0016 %, matching Poisson. **Consequence: fine bucketing cannot live in
the global layout.**

**P3 — a sorted-runs match core is correct and 1.75× slower.** `sorted_r1_probe`
(committed): one block per bucket, 8 B keys-only staging, a 256-bin counting sort
making equal-key runs contiguous, each lane deriving its element into registers and
pairing via statically-unrolled constant-distance `__shfl_up_sync` — the chain table,
the walk, the rescan, the spill machinery and 5 of 8 barriers all gone. It emits the
**exact pair multiset** of the shipping kernel (per-bucket counts, gi-masked record
folds, back-ref folds all equal, drops 0, five nonces) at 8.56 ms against the chain
kernel's 4.89. The compile-time ablation carve attributes every millisecond:

| piece | ms |
|---|---|
| counting sort: count + scan + place + staged read | 0.44 |
| seed derivation into registers | +2.5 |
| match: bare shuffles + the leftover walk | +3.6 (walk ~3.3) |
| emit: combine + mix + scatter | +2.0 |

Even the design's floor — sort + derive + shuffles + perfectly converged emit — is
~3.9 ms, 0.80× the chain kernel, against a ≥ 2× target. **The decomposition is the
finding: the chain kernel's 4.89 ms is derive ~2.5 + emit arithmetic ~1.8 +
bookkeeping ~0.6.** The wavefront model — 130–160 B through shared per element as the
wall — is refuted at stock: that traffic overlaps ALU issue almost completely. What
fills r1's cycles is the algorithm (hashing, combine, mix), not the organization, and
r2's rebuild share is larger still. No match-side rewrite can reach 2×; the
reorganization is closed, prototype unbuilt.

Three CUDA codegen traps found on the way, documented in the probe header: a
runtime-selected pointer (`cond ? &a : &b`) demotes both operands to local memory
(168 B stack, 3× the kernel — order the mix tree instead, combine is a symmetric
XOR); warp collectives inside a data-dependent loop each compile to a WARPSYNC
trampoline call (constant-distance shuffles in straight-line code compile bare); and
a divergent re-derive costs ~32× its converged price (the leftover walk served 6.3 %
of pairs for 3.3 ms).

What survives: the compile-time-carve + sink ablation method, the 0.44 ms counting
sort (if a future record format ever makes staging the wall), a hard ceiling of
~0.6 ms/round on any future match rewrite — and an honest problem statement: with
bookkeeping at 12 % of r1, lolMiner's 2.3× work-per-clock cannot come from leaner
bookkeeping, and its actual mechanism is back to unexplained. The one hypothesis
left standing is that its ~4 GB footprint stores derived state and skips the
derive/rebuild arithmetic entirely — our R2_FULL evidence against that trade priced
a variant that stored records *on top of* the pair machinery, not instead of it —
but even a free derive is ~5–6 ms of ~27 ms of C, short of 2.3×.

*Confirmed 2026-07-31, by profiling lolMiner itself: the state-storing hypothesis
is the measured truth — 64 B/element layers, 17.7 GB/solve, no re-derivation
anywhere, footprint = 2 × 33.5 M × 64 B exactly. See
[lolMiner measured under ncu](#lolminer-measured-under-ncu-the-state-storing-design-confirmed--and-its-54-sols-ceiling-is-a-dram-roofline).*

</details>

### The CUDA match wins, backported to OpenCL: perfect table + spill + per-round caps (−0.6 ms)
<details>
<summary>Details</summary>

*(2026-07-31, the first lead executed from `docs-internal/PERF_LEADS.md`. OpenCL is
the only fast path non-CUDA devices get, and `lds.cl` predated everything the match
learned on 2026-07-26 — it still carried `lkey`, a tail-sized group cap and no
spill.)*

Three transplants, all as `FUSED_LDS` **macro parameters** (`FCAP, TAB, PERFECT,
SPILL`) so the generic `round_fused_lds` — whose tests run it at geometries like
(11,0) where the perfect-hash argument does not hold — keeps the legacy behavior:

- **Perfect chain table** ([CUDA original](#shipped-the-group-cap-no-longer-has-to-cover-the-tail-125-ms)):
  on the `bb + sm = 17` line only 7 key bits vary inside a group, so a 128-entry
  table is a perfect hash, and `lkey` (4 B/element of LDS) plus the walk's
  innermost-loop compare are deleted. `run_pipeline_rowbucket` refuses an off-line
  geometry unless `-DLDS_PERFECT_TAB=0` is passed, mirroring `CudaSolver`'s check.
- **Group spill**: an overflowing group splits on one more key bit, the level chosen
  from an 8-bin word-0 count *before* any part is walked. `LDS_FCAP` then drops from
  the tail (384) to near the mean: **320, and 288 for r1** (per-round caps, the
  CUDA values).
- **r1's folds**: gi == lead == leaf 0 == the seed index, so one `lleaf` slot
  carries all three and `LEAFW` goes 2 → 1.

That puts r1 at 64 B/element ≈ **19.0 KB per workgroup — two workgroups fit** the
48 KB NVIDIA's OpenCL exposes per SM; r2 (23.6 KB) and r3 (26.2 KB) stay at one.

| | pipeline median (`bench_rounds 20`) | miner end-to-end (60 s, ~1 470 solves) |
|---|---|---|
| legacy config | 40.3–40.5 ms | 40.6 ms / 48.7 sol/s |
| **backport** | **39.6–39.8 ms** | **40.0 ms / 49.7 sol/s** |

**−0.6 ms (−1.5 %), 48.2 → 49.7 sol/s on the miner's own 60 s benchmark.** Every
backport run beat every legacy run across two interleaved brackets, and the
same-session miner A/B agrees with the pipeline delta. Gates: goldens [1,1,1] on the
packed *and* the quad rungs, drops 0, 45/45 tests. The verbose marginals put the win
in **r1** (6.3 → ~5.5 ms), the latency-bound round that gained the workgroup — the
same shape as CUDA's r1-takes-a-fifth-block result, at OpenCL's coarser granularity.

Two nulls, so they are not re-proposed:

- **`LDS_FCAP` 304 and 288 buy nothing** (39.6 / 39.9 ms against 39.6–39.7 at 320).
  The arithmetic put r2 under the two-workgroup line at 304 (22.4 KB against the
  `(24 576 − 2 052)` per-workgroup budget); either the OpenCL runtime does not grant
  the second workgroup or r2 does not pay for it, and no instrument resolves which —
  the driver's verbose log truncates before the fused kernels. 320 stays: it matches
  CUDA and has the lowest spill rate.
- The r4 child count wobbles ±1 across runs (…633/…634) — the pre-existing
  gi-tie-break nondeterminism (~17 equal-lead pairs per solve whose left/right
  choice follows emission order), confirmed present on identical binaries. The KAT
  is the gate and holds.

A/B recipe, no rebuild needed (OpenCL compiles kernels at run time):
`MXBM_CL_OPTS='-DLDS_PERFECT_TAB=0 -DLDS_SPILL=0 -DLDS_FCAP=384 -DLDS_FCAP_R1=384
-DLDS_FTAB=512'` restores the legacy configuration exactly.

**Metal, checked the same day: nothing to backport.** The Metal fused rounds postdate
the 2026-07-26 CUDA wins and already carry all of them — no `lkey` anywhere, the walk
compare omitted as a stated tautology, group spill, per-round FCAP (320 / r1 288), and
the gi-is-lead fold. The one candidate left is `terminal_round`
(`kernels/metal/pipeline_kernels.metal`), which keeps its `lkey` deliberately: its
comment notes the table "is not guaranteed perfect at every geometry" because the
Metal host has no bb+sm ≥ 17 guard. The remaining work is a Mac-session item — add the
guard (mirroring `run_pipeline_rowbucket`/`CudaSolver`), drop terminal's `lkey` the
way CUDA's `MXBM_PERFECT_TAB` did, and measure there. The CUDA analogue was worth
≤ 0.1 ms of a 1.07 ms kernel; Metal's terminal is 4.4 ms of ~101, so expect small but
nonzero. Not attempted from this rig: no Metal toolchain, no KAT gate.

</details>

### Two below-the-floor levers clear noise on CUDA: r2's pair record in one LD.128, and the terminal round joins the perfect table (−0.22 ms)
<details>
<summary>Details</summary>

*(2026-07-31, the "below-the-floor basket" from `docs-internal/PERF_LEADS.md`: items
individually priced at or under the 1 %-of-a-solve floor, batched so one day's
measurement decides all of them. Two of three ran — the third,
`MXBM_NARROW6` × (17,0) under caps, needs root for the power limits and is queued.)*

- **`MXBM_PAIR128`** (default on): round 2 stages its 16 B pair record with one
  `LD.128` instead of two `LD.64` — the stride is 2 u64, so the address is always
  16 B aligned, the same argument the wide records already use. The rescan lanes
  that fail the sub-mask filter now fetch 16 B where they fetched 8, but those
  bytes are ~98 % L2-absorbed; what every staged element saves is a memory
  *instruction*.
- **Perfect table in `terminal_round`**: the tautology holds there identically
  (same `bb + sm = 17` line, 7 varying bits, 128-entry table), but the kernel still
  carried `lkey` and its walk compare. Both gone; `cuda_solver.cu`'s existing
  geometry guard covers this kernel too.

Measured by replay amplification (`MXBM_ROUND_REPS`, ×9, `cuda/pipeline`, 30 nonces
per arm, two interleaved pairs each), then confirmed end-to-end:

| | base | new | per-solve Δ |
|---|---|---|---|
| r2 ×9 (`2:9`) | 115.53 / 116.13 ms | 114.38 / 114.66 | **−0.145 ms** |
| terminal ×9 (`5:9`) | 42.02 / 41.95 | 41.31 / 41.36 | **−0.072 ms** |
| end-to-end, 60 nonces × 3 pairs | 33.72 / 33.63 / 33.61 | **33.41 / 33.45 / 33.44** | **−0.22 ms** |
| miner `--benchmark`, 60 s arms | 33.4 / 33.4 / 33.4 | 33.2 / 33.2 *(one 34.4 outlier)* | ≈ −0.2 ms |

The end-to-end delta equals the sum of the two attributions, every standalone pair
ordered the same way, KAT green and drops 0 throughout. **Both ship, on by
default.** The known risk did not fire: r2 sits on the 64-of-64 register cliff and
the vector load held it at 64 (`test_cuda_resources` passes unchanged). The terminal
round re-baselined under the contract's RESOURCE DRIFT rule — REG 22 → 24, SHARED
9 732 → 8 196 B (the 4 B × 384 `lkey`), blocks/SM held at its warp-capped 6.

</details>

### The found-vs-verified gap is GONE: 0 of 3,935 candidates rejected

<details>
<summary>Details</summary>

*(2026-07-31, `MXBM_VERIFY_STATS`. This closes the MXBM half of the counting
question and retires a caveat this document had carried since the CUDA port.)*

The CUDA-backend caveats recorded *"this solver produces 2.29 survivors per solve
but only 1.95 that verify, a 17 % gap"* (2026-07-25), and nobody had ever looked at
*why* candidates fail. `bh3::classify_solution` now mirrors `is_valid_solution`
check for check but returns which one failed and at which round — duplicate index,
tree-order violation, collision-bits mismatch, or non-zero final XOR — and
`MXBM_VERIFY_STATS=1` tallies every candidate on either backend, printing at exit.

| backend | solves | candidates | verified | rejected |
|---|---|---|---|---|
| CUDA (45 s) | 1 351 | 2 690 (1.991/solve) | 2 690 | **0** |
| OpenCL (25 s) | 622 | 1 245 (2.002/solve) | 1 245 | **0** |

**Every survivor the pipeline produces today is a valid solution.** Not
mostly-duplicates-plus-a-few-bugs: zero rejections of any class, on both backends.
The 2026-07-25 figure does not reproduce on the current build; whether it was real
then (and closed by one of the intervening changes) or an artefact of that era's
measurement cannot be reconstructed and does not matter — the current truth is that
"found" and "verified" are the same number.

Two consequences. The conservative counting MXBM ships (verified only) costs
nothing — there is nothing looser to report. And the counting question in the
lolMiner comparison is now one-sided: if its counter reports raw survivors, any
definitional gap is bounded by *its* false-survivor rate, not ours. The
accepted-share protocol (`docs-internal/MINER_COMP.md`) remains the only
counter-independent settle, but the MXBM side of the ambiguity is measured away.

The instrument stays: it is free when the env var is unset, and a future regression
that starts shedding candidates will be attributable the day it appears.

</details>

### MXBM_NARROW6 × (17,0) under caps: loses at every cap — the basket's third item is a null

<details>
<summary>Details</summary>

*(2026-07-31, `docs-internal/rootruns/run_n6_caps.sh`. The premise: at low caps the
currency is instructions, `MXBM_NARROW6` trims staged bytes, and the two knobs had
never been composed. Standalone `cuda/pipeline` vs a `-DMXBM_NARROW6=1` build, both
at `MXBM_BB=17`, 60 nonces per arm, two interleaved pairs per cap, KAT and drops
green on every arm.)*

| cap | base (17,0) | + NARROW6 | Δ |
|---|---|---|---|
| 100 W | 100.69 / 100.56 ms | 102.82 / 104.36 | **+3.0 %** |
| 140 W | 60.02 / 60.05 | 61.91 / 61.77 | **+3.0 %** |
| 180 W | 42.57 / 42.97 | 43.17 / 43.22 | **+0.9 %** |

NARROW6 loses every pair at every cap, and by *more* as the cap tightens — the
opposite of the premise. The staged-byte saving buys nothing (shared traffic
overlaps issue, per the reorganization probes), while the split `lw6` plane costs
real instructions on both the staging store and every `ldw` in the walk — and
instructions are exactly what a capped card cannot afford. The knob stays
default-off, now with a measured reason at both ends: null at stock (16,1), a loss
under caps at (17,0). Do not re-propose.

</details>

### lolMiner measured under ncu: the state-storing design, confirmed — and its ~54 sol/s ceiling is a DRAM roofline

<details>
<summary>Details</summary>

*(2026-07-31, `docs-internal/rootruns/run_lol_ncu.sh` → `lol_ncu.ncu-rep`. The
measurement three sections of this ledger called for: lolMiner 1.98a's own kernels,
profiled over a 120-launch window of its fixed 61 s BEAM-III benchmark —
`ncu` profiles closed binaries, no source needed. 15 complete solve cycles, byte
counts stable to ±1 % across all 15. The kernel names are unobfuscated.)*

**Its pipeline is our pipeline's shape** — a seed pass and five rounds, one kernel
each, in a strict chain: `cleanUp → seed → R1 → R2 → R3 → R4 → R5A → R5B`. What
differs is everything the bytes say:

| kernel | grid × block | ms | rd MB | wr MB | GB/s |
|---|---|---|---|---|---|
| `cleanUp` | 640 × 256 | 0.005 | 0 | 0 | — |
| `beamHashIII_seed` | 524288 × 64 | 5.43 | 2.5 | **2131** | 393 |
| `beamHashIII_R1` | 131072 × 512 | 9.10 | 2155 | 2133 | 471 |
| `beamHashIII_R2` | 131072 × 512 | 9.11 | 2156 | 2131 | 471 |
| `beamHashIII_R3` | 131072 × 512 | 8.37 | 2154 | 2128 | 512 |
| `beamHashIII_R4` | 65536 × 1024 | 4.34 | 2151 | **263** | 556 |
| `beamHashIII_R5A` | 16384 × 1024 | 1.08 | 269 | 0.6 | 250 |
| `beamHashIII_R5B` | 20 × 64 | 0.005 | 0.1 | 0 | — |
| **per solve** | | **37.4** | **8.89 GB** | **8.79 GB** | |

**17.67 GB per solve, against MXBM's 13.00.** The layers are ~64 B/element
(2 131 MB ÷ 33.5 M — 8 u64), written by the seed pass and both read and written in
full by R1–R3. That settles the hypothesis in one line: **the seed kernel writes the
complete derived element to DRAM, and no round ever re-derives anything.** Its
famous ~4 GB footprint is exactly two ping-pong sets × 33.5 M × 64 B = 4.3 GiB —
the arithmetic `SOLVER_REORG.md` §6b guessed and could not check. (Our records are
*narrower* where we re-derive — r1 8 B, r2 16 B — and *wider* where we store, 72 B
with leaf payloads; theirs is a flat 64 B everywhere with, presumably, parent refs
inline. R4→R5 it out-thins us: ~8 B/element against our 16.)

Three standing questions close at once:

- **The ~2.3× work-per-clock at low caps is not a mystery technique — it is the
  absence of our arithmetic.** No 235 M-siphash seed re-derivation in R1, no
  14-siphash rebuild in R2; every round is a streaming pass at 393–556 GB/s. Under
  a core-power cap the memory clock does not scale down, so DRAM — their currency —
  survives the cap while instruction issue — ours — collapses with it. The
  [duty-cycle probe](#lolminer-is-not-duty-cycling--the-low-end-gap-is-real-work-per-clock)'s
  "real work per clock" is real *absence* of work per element.
- **Its ~54.0 sol/s ceiling is its DRAM roofline.** 17.67 GB/solve at its real
  ~35.8 ms/solve is **~494 GB/s sustained** — within a few per cent of the card's
  ~510 GB/s achievable — which is also why its draw saturates at ~237 W and caps
  above that buy it nothing: it cannot spend core watts it has no instructions for.
- **The whole band structure of the head-to-head is one design choice, held from
  opposite ends.** MXBM pays arithmetic to move 4.7 GB/solve less; lolMiner pays
  bytes to issue far fewer instructions. Above ~210 W the arithmetic hides in
  memory stalls and the byte bill is the binding one — we win, and our ceiling
  (59.8) sits above their roofline (54). Below ~190 W the arithmetic is priced at
  full clock-starved cost — they win. Nobody is doing anything the other could not
  in principle do; the two designs are the two ends of the same trade.

**What it does and does not re-open.** It does not un-close the low-power goal by
itself: our own full-record variants (`MXBM_R2_FULL`, the quad record) lose at
every cap *inside this organization*, because they bolt stored bytes onto machinery
that still stages, chains and walks per element — the measured difference is their
ground-up streaming design (flat 64 B records, no leaf payloads, inline ancestry,
512–1024-thread blocks, no sub-mask rescan). A from-scratch "eco pipeline" in that
style is now a *specified* project rather than a mystery — with its prize honestly
bounded by the band: ~20–30 % below 160 W, zero above ~210 W, on a card class
where the band is already ceded. Parked as a design note, not a lead.

*Caveats: byte counts are per-kernel under ncu replay (cache-flushed between
launches, so cross-kernel L2 reuse — small for streams this size — reads as DRAM);
durations under serialized replay, though their sum (37.4 ms) agrees with the
uninstrumented ~35.8. lolMiner still reported 49–55 sol/s during the run, so the
window barely distorted it. What its sol/s counter counts remains unmeasured — the
accepted-share protocol stays the settle for that.*

</details>

### The sort path's constant-kernel regression: half of it is occupancy, the other half is still the constants — generic stays

<details>
<summary>Details</summary>

*(2026-07-31, the last open end of lead 4b. The ledger's named mechanism for k1/k2
losing r1/r2 (+2.8/+2.3 ms) was that constant-folding dropped them from the generic
kernel's 48 registers to 40, raising residency from 5 to 6 blocks/SM on a
gather-bound kernel — supported by the converse probe (capping the GENERIC to 40
cost r1 +4.9 / r2 +9.5) but never tested in the profitable direction: keep the
constants, give the occupancy back.)*

`ptxas` confirms the register facts exactly (generic 48, every K variant 40), and
the test is `round_match_k1p`/`k2p`: the constant kernels plus a `__local` BALLAST
array sized by `-DKPIN_UINTS` (kept live by a touch under a runtime condition that
never holds, so it costs nothing at run time), selected by `MXBM_MATCH_K12=2`
(`=1` re-measures the plain k1/k2 regression in the same binary). Whole-solve
medians, `bench_rounds 10`, sort path, bracketed:

| arm | ms |
|---|---|
| generic (shipping) | **193.5 / 190.6** |
| k1/k2 (constants, 6 blocks/SM) | 202.5 |
| k1p/k2p, 9.6 KB ballast | 197.7 |
| k1p/k2p, 17.4 KB ballast | 197.9 |

Per-round attribution (`MXBM_SORT_PROFILE`, emit means over 6 solves): generic
r1 **21.0** / r2 **22.3** ms; pinned constants r1 **24.1** / r2 **25.1**.

**Verdict: the pin recovers roughly half of the regression (202.5 → 197.7) — the
occupancy mechanism is real — and the constant kernels still lose r1 +3.1 / r2
+2.8 ms to the generic even at pinned occupancy.** The residual travels with the
constant-folding itself and stays unexplained, now with occupancy excluded as its
cause. The generic kernel remains selected for r1/r2; both instruments stay in the
tree so neither half has to be re-derived. That the two ballast sizes tie (197.7 vs
197.9) also says the pin point saturated — the residual is not a partially-applied
pin. Two caveats: the sweep ran while the card cooled from a mining session (the
brackets moved 193.5 → 190.6, smaller than every gap read), and today's whole-solve
regression (~+10 ms) reads larger than the ledger's original per-round +5.1 — same
sign, different session.

With this, **lead 4b's open ends are exhausted on the tuning side**: the K-variant
question is answered, `leaves[2]`'s 138 MB stays declined for the audit's original
reason (memory-only, on a path bound by other allocations), and what remains for
the sort path is what always remained — it is the fallback for cards the row-bucket
geometries cannot host, at ~190 ms.

</details>

---

## Established limits
<details>
<summary>Details</summary>

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
5. **Divergence is only paid by compute.** Work placed in the sub-mask-filtered staging
   loop runs at ~4/32 lanes; the same work in the following all-lanes loop runs at 32/32.
   For *arithmetic* this is worth up to ~58× (round 2's rebuild: 23.5 → 0.4 ms). For
   *loads* the ordering reverses — the staging loop's 8 iterations overlap their loads,
   the all-lanes loop's single iteration cannot
   ([measured](#deferring-the-stage-read)).
6. **The compute-hiding budget is finite — and it SHRINKS as the kernel gets faster.**
   The fused kernel stalls on memory, and arithmetic issued into those stalls is free
   until it exceeds them. This is not register spilling
   ([register cap](#register-cap-tuning) is a wash); it is the stall budget being used
   up. Crucially the budget is not a constant of the algorithm: removing the local-memory
   spill in [compile-time constants](#compile-time-round-constants) cut the stalls, and
   the *same* round-2 rebuild went from +0.4 ms to +4.3 ms without changing a line of it.
   **Any compute-for-memory trade must be re-measured after any change that speeds up the
   round it lives in** — see [retiring the quad record](#retiring-the-round-3-quad-record).

Consequence: element size, coalescing, occupancy and atomics are all settled. The gap is
down to ~1.6×, and the single largest step toward it was not an algorithmic idea at all —
it was a compiler-visibility bug in code that had been read many times. Worth weighing
before assuming lolMiner holds an unknown technique: the last 26.6 ms came from making
existing arithmetic compile down properly, not from moving fewer bytes.

</details>

---

## Current focus and open leads
<details>
<summary>Details</summary>

**Where the time goes.** *(CUDA, the shipping path. Re-measured 2026-07-26, after the
`kFCap` 320 change.)*

**The per-round picture is obtainable without a profiler**, which matters because Nsight
needs root and a reboot on a card that drives a display. `MXBM_ROUND_REPS="R:N"` replays
round *R* in place *N* times, so `(t_N − t_1)/(N−1)` is that round's marginal cost; the
traffic is exact from the stride table (`kFbStride`, plus 8 B/element for the sub-mask
rescan and 8 B for back-refs). Over 60 nonces, against the card's true 672 GB/s:

| kernel | ms | GB moved | GB/s | % of peak | floor at peak |
|---|---|---|---|---|---|
| entry | 2.72 | 0.27 | 99 | 15 % | 0.40 |
| r1 | 5.33 | 1.34 | 252 | 37 % | 2.00 |
| r2 | 10.38 | 3.49 | 336 | 50 % | 5.19 |
| r3 | 9.50 | 5.10 | 537 | **80 %** | 7.59 |
| r4 | 5.49 | 3.22 | 587 | **87 %** | 4.79 |
| terminal | 1.07 | 0.81 | 753 | >100 % (L2 absorbs the rescan) | 1.20 |
| **total** | **34.49** | **14.23** | **412** | **61 %** | **21.16** |

*(r1 and r2 carry almost all of the −1.25 ms the `kFCap` 320 change bought, 5.85 → 5.33
and 10.71 → 10.38, which is what their 3 → 4 blocks/SM is worth.)*

This reproduces the profiler table within a few points on every row, from a stopwatch and
arithmetic. Use it before booking a profiling session.

**So the floor is 21.2 ms and the pipeline is at 61 % of peak** — the right-hand column,
priced at the hardware's 672 GB/s rather than at any rate the solver happens to achieve.

> **Correction (2026-07-29): the floor is 19.8 ms and the pipeline is at 57 % of peak.**
> Two independent errors above, pushing the same way.
>
> **(1) The sub-mask rescan is an L2 hit, not DRAM traffic.** The 14.23 GB charges
> 8 B/element of rescan to five stages, 1.34 GB in all, and the profiler says almost none
> of it reaches DRAM: per stage, (compulsory read + 268 MB rescan) − measured read comes
> to 265 / 262 / 257 / 266 / 266 MB for r1 / r2 / r3 / r4 / terminal, so **98 % of the
> 1.34 GB budget is absorbed** ([the compulsory-traffic
> table](performance.md#the-memory-traffic-is-compulsory), whose measured total is 13.00 GB). A floor
> has to be priced on bytes the DRAM actually moves.
>
> **(2) The card runs at the 10251 MHz rung, where peak is 656 GB/s.** 672 needs the
> 10501 rung; `nvidia-smi -q -d SUPPORTED_CLOCKS` lists exactly 10501 / 10251 / 5001 /
> 810 / 405, and every controlled run on record reads 10251 under load (the 300 s run
> above, and both miners in the head-to-head). 10251 × 2 × 32 B = 656 GB/s.
>
> | kernel | ms | GB moved | GB/s | % of peak | floor at peak |
> |---|---|---|---|---|---|
> | entry | 2.72 | 0.26 | 97 | 15 % | 0.40 |
> | r1 | 5.33 | 1.07 | 200 | 31 % | 1.63 |
> | r2 | 10.38 | 3.35 | 323 | 49 % | 5.11 |
> | r3 | 9.50 | 4.83 | 508 | **77 %** | 7.35 |
> | r4 | 5.49 | 2.95 | 538 | **82 %** | 4.50 |
> | terminal | 1.07 | 0.54 | 505 | 77 % | 0.82 |
> | **total** | **34.49** | **13.00** | **377** | **57 %** | **19.82** |
>
> The terminal row's impossible ">100 % of peak" was the tell, and it is gone: it was
> 0.81 GB of arithmetic against 0.54 GB of DRAM. On the same measured bytes at 672 GB/s
> the figures are 19.34 ms and 56 %, so (1) does nearly all the work and (2) is worth
> 2.4 %.
>
> **Two figures below inherit the old basis and are not re-derived here.** "Of the 14.5 ms
> above that floor, r1 (3.85) and r2 (5.52)" uses the *pre*-`kFCap`-320 round times
> (5.85 / 10.71) against a current floor; on the row above it is r1 3.70 and r2 5.27,
> **8.97 ms, 61 % of the 14.68 ms above the floor**. "r3 and r4 … are at 79–84 % of peak"
> matches neither table; they are at **77–82 %**.

> **⚠ This supersedes the "39.1 ms memory floor" and the "within ~5 %" claim that stood
> here before.** That floor was priced at the *OpenCL-achieved* 312 GB/s, and CUDA
> already runs at 34.2 — below its own stated floor, which should have been the tell.
> **A floor priced at achieved bandwidth is circular**: it can only ever report that the
> pipeline is near it. The retracted accounting is kept below because it is still a
> correct description of *the OpenCL path*.
>
> | | read + write | GiB | floor at 312 GB/s |
> |---|---|---|---|
> | r1 | 8 + 16 B | 0.75 | 2.6 ms |
> | r2 | 16 + 72 B | 2.75 | 9.5 ms |
> | r3 | 72 + 64 B | 4.25 | 14.6 ms |
> | r4 | 64 + 16 B | 2.50 | 8.6 ms |
> | | | | **35.3 ms** + entry 2.7 + terminal 1.1 = **39.1** |
>
> OpenCL measured **40.3 ms** clean against that (41.2 under ablation instrumentation),
> at entry 2.7, r1 6.3, r2 13.0, r3 12.3, r4 6.4, terminal 1.1. Its non-memory work
> ablated to ~2.2 ms in total: `apply_mix` 0.9, back-refs 1.1, round 2's rebuild 0.2. On
> CUDA that work is smaller again — `apply_mix` measures **0.01–0.06 ms** per round and
> round 2's rebuild 0.97
> ([attribution](#bytes-are-nearly-free-per-element-work-is-not)).

**Which round holds the gap.** Of the 14.5 ms above that floor, r1 (3.85) and r2 (5.52)
hold **9.4 ms, 65 % of it** — and they are exactly the two rounds that re-derive per
element instead of reading stored state (r1 re-seeds, r2 rebuilds from two parent seeds
in 14 siphashes). r3 and r4, which only move bytes, are at 79–84 % of peak. So the split
is not "some rounds are tuned and some are not": **the rounds that compute are slow and
the rounds that only stream are fast**, and both re-derivations have been A/B'd in the
right direction (r1's seed re-derivation 2.5 vs 18.2 ms; r2's pair record 35.0 vs 39.2).
The work is BeamHash III's own hash, which the PoW definition fixes.

**The 1.71× / 96 sol/s ceiling is real arithmetic but it is not one overlap away.**
`max(SM-busy, DRAM-busy)` = 20.6 ms agrees with the 21.2 ms DRAM floor above, so the
number is right. What it assumes is that the six kernels' work can be interleaved
arbitrarily — and within a solve it cannot: entry→r1→r2→r3→r4→terminal is a strict chain
with a global barrier at every step, so at any instant only one round's work exists. All
independent work comes from *another solve*, and [lead 5](#current-focus-and-open-leads)
measured that streams cannot deliver it. Reaching 21 ms therefore needs a
software-pipelined heterogeneous launch, not an overlap. **That co-resident launch has
since been built and measured, and it is null too** — hosting `entry` inside a round costs
7.25 ms against its 2.77 ms standalone, because a memory-stalled warp still holds its slot
and the work lands *after* the stall in the same warp rather than beside it. See
[phase overlap cannot reach the roofline](#phase-overlap-cannot-reach-the-roofline).
The binding resource is **resident warps**, which shared memory caps at 24 of 48 per SM,
and `kFCap` cannot shrink to free any. One objection has lifted, for whenever that
constraint is broken: the 48 MiB shortfall that retired two resident pipelines was
computed at geometry (16,1), and two at (15,2) need 13.8 GiB — they fit.

**The large levers, and where each now stands:**

| lever | status |
|---|---|
| fewer bytes | every record is `ceil(bits/64)` ([audit](#the-record-redundancy-audit)) — except one that was *stored* wider than that, [since fixed](#the-round-2-alignment-pad). And bytes buy almost no **time**: shrinking r2's and r3's records to 16 B, well past what the audit allows, is worth ~4 ms of 34 ([measured](#bytes-are-nearly-free-per-element-work-is-not)) |
| bytes as **watts** | where that lever moved to. Under a cap bytes are clock: 16 % less traffic is worth [60 MHz at 285 W and 210 MHz at 180 W](#but-bytes-are-not-free-in-watts-and-under-a-cap-watts-are-clock-60-mhz) — but that is the prize for a *free* narrowing, and the [one built](#the-quad-record-29--footprint-and-the-byte-prize-does-not-survive-re-derivation) spends the freed watts on the arithmetic replacing the bytes. **CLOSED 2026-07-29.** The denominator is 1.47 GB measured, not 2.09 derived, and the numerator is a replay clock — so the whole-solve rate is ≤ 92 MHz/GB and closing 570 MHz needs 48 % of all traffic against the 1.07 GB the audit allows. [Details](performance.md#-closed-2026-07-29-the-low-end-is-not-reachable-by-traffic-and-no-other-mechanism-has-been-found) |
| redundant rescan | removing it entirely buys nothing over halving it ([geometry](#row-bucket-geometry)); re-confirmed on CUDA, where (17,0) is −9 % traffic and +9 % time |
| occupancy | **closed 2026-07-31, from both resources.** OpenCL's 48 KB LDS made it structurally unreachable ([details](#occupancy-again)); CUDA exposes 100 KB/SM, where [3 → 4 blocks/SM is worth ~1.6 ms](#occupancy-is-worth-real-time-and-shared-memory-is-the-only-gate). r1/r2 crossed at `kFCap` 320 (**−1.25 ms**) and [r1 took a fifth block](#round-1-takes-a-fifth-block-015-ms-via-a-per-round-group-cap) (−0.15). r3 reaches 4 blocks and [does not care](#round-3-does-not-want-a-fourth-block--occupancy-pays-only-where-a-round-is-latency-bound). This entry used to end "`B` is the only term left"; it is not — **r2's 4 × 256 × 64 registers are the ENTIRE register file**, so the next block needs fewer shared bytes AND fewer registers at once, and [forcing the register side is null](#occupancy-is-closed-from-both-resources--r2-sits-on-the-whole-register-file) |
| coalescing the emit | max 1.9–2.2× against a 3× traffic cost ([two-level](#two-level-bucketing)) |
| the two atomics | both load-bearing; removing either is slower |
| `apply_mix`, back-refs, rebuild | 2.2 ms combined on OpenCL and less on CUDA — nothing left to win |
| phase overlap | **closed by three mechanisms and HARVESTED 2026-07-31.** Grid depth (streams) and warp slots (same-warp hosting) were null; the third — [co-blocks](#co-blocks-the-third-overlap-mechanism-works--and-it-is-worth-04-ms-not-14), separate interleaved blocks in one launch — works, and its whole yield is **~0.5 ms**: r4's exploitable idle, whatever co-work is offered (entry, or [a whole round of the next solve](#fused_pair-two-solves-rounds-in-one-launch--the-familys-ceiling-is-05-ms)). Shipped as [speculative entry](#speculative-entry-co-scheduling-ships-in-the-miner-045-ms); the 1.71× roofline stays out of reach |
| match organization | **closed 2026-07-31, by two probes, prototype unbuilt.** The plan was fine-grained buckets matched in registers (chains, rescans, staging and barriers deleted). P2: [thin records lose 4.5× scattering into 2^21 buckets](#the-solver-reorganization-probes-the-cycle-deficit-is-not-bookkeeping) (the L2 bucket-tail cliff starts at bb = 18), so fine buckets cannot live in the global layout. P3: a counting-sort/register-pair r1 built in shared instead emits the exact pair multiset and is **1.75× slower** — its carve prices r1's cycles as derive ~2.5 + emit arithmetic ~1.8 + **all bookkeeping ~0.6** of 4.9 ms. There is no 2× in the match, for any organization of it; any rewrite's ceiling is ~0.6 ms/round |

**Leads.** The CUDA backend is ~6 % past the target and the OpenCL path 1.07× short
(1.12× before [the 2026-07-31 backport](#the-cuda-match-wins-backported-to-opencl-perfect-table--spill--per-round-caps-06-ms)).
**Goal 1 — the low-power gap — is closed by measurement**, not by exhaustion of effort:
traffic supplies at most 17 % of the 570 MHz deficit, the instruction-issue route is
impossible on this ISA, and undervolting is untestable on a card that drives its own
display. See [the closure](performance.md#-closed-2026-07-29-the-low-end-is-not-reachable-by-traffic-and-no-other-mechanism-has-been-found).
What remains is speed and efficiency at and above the band, where the pipeline sits at
57 % of a 19.8 ms floor and r1 and r2 hold 65 % of the gap:

0. ~~**Ship the power cap as a setting.**~~ **Done 2026-07-25** — `--pl W` and
   `--no-oc-reset`, per-GPU list syntax, clamped to the band the driver reports,
   restored on exit including on Ctrl+C. See [usage.md](usage.md#power-limit) and
   [overclocking.md](overclocking.md). **220 W is the operating point to recommend**:
   55.4 sol/s for 219.5 W, and the point where MXBM's lead over a *equally capped*
   lolMiner is at its best value. ~~re-measure lolMiner under a cap of its own~~ —
   **done 2026-07-28, and it changed the answer**: see
   [Both miners under the same cap](performance.md#both-miners-under-the-same-cap). Remaining follow-up:
   whether MXBM should
   *default* to a cap rather than only offering one, which is a release decision rather
   than a technical one and is parked with the other pre-release questions. Note the
   general case is not settled by one card: a default derived from an RTX 4070 Ti SUPER
   has no standing on hardware whose curve nobody has swept, so `--pl auto` would need
   MXBM to find the knee itself.
0b. **The footprint is a REACH lever, not a speed lever.** 7.46 GiB against a 3 GB design
   target, and shrinking it is what put the CUDA backend on 8 GB cards (lead 4). What it
   no longer is, is a way to go faster: bytes were measured and they are nearly free —
   shrinking round 2's and round 3's records to 16 B, well past what the record audit
   allows, is worth ~4 ms of 35. See
   [bytes are nearly free](#bytes-are-nearly-free-per-element-work-is-not). This entry
   used to claim footprint "is the only one that moves both axes at once"; it moves one.
1. **Settle the comparison.** Accepted pool shares over a fixed interval, the only metric
   independent of either miner's counters. `docs-internal/MINER_COMP.md` has the protocol
   and the sample-size arithmetic. The margin (56.1 ± 2.3 against 53) is real but thin
   enough that this matters.
2. ~~**Wire the CUDA backend into the miner.**~~ **Done 2026-07-25** — `--solver
   auto|cuda|opencl|gpu|ref`, CUDA preferred, OpenCL kept as the portable fallback.
3. **Overclocking.** Deferred until parity was in sight. BeamHash III is bandwidth-bound,
   so a memory offset scales it close to linearly — and it applies to lolMiner equally,
   so the honest comparison is overclocked against overclocked. Design decisions, the
   verified NVML ranges and the verified lolMiner behaviour are in
   [overclocking.md](overclocking.md). `mxbm --benchmark BEAM-III` exists to A/B the
   settings without a pool.
4. ~~**Per-path VRAM budget on the CUDA side.**~~ **Done 2026-07-26.** The CUDA backend
   now picks its geometry from `rb_geometry_for()` — the same arithmetic the OpenCL path
   uses, moved to `src/gpu/rowbucket_geom.{h,cpp}` so the two cannot drift. It was
   labelled "reach, not speed", and reach is what it turned out to be worth: a large one.

   **The ladder is free to walk on CUDA.** `bb`/`sm`/`cap` were already kernel
   *arguments*, and everything compile-time is invariant along the `bb + sm = 17` line —
   the grid is 2^17 blocks, the staged group is `capacity / 2^17` = 264, and the
   128-entry chain table is a perfect hash for the `24 - bb - sm` = 7 bits left varying.
   So the coarser geometries needed no new code, only permission. Measured, one binary,
   200 distinct nonces, KAT green and drops zero at every rung:

   | geometry | footprint | ms/solve | sol/s |
   |---|---|---|---|
   | (16,1) | 7.46 GiB | 35.1 | 58.0 |
   | (15,2) | 6.88 GiB | 37.8 | 53.8 |
   | (14,3) | 6.50 GiB | 42.8 | 47.5 |

   **What it reaches.** CUDA has no `CL_DEVICE_MAX_MEM_ALLOC_SIZE`, which is the limit
   that binds OpenCL below 12 GB
   ([limitation 1](HW_REQUIREMENTS.md#1-below-11-gb-opencl-is-capped-by-its-single-allocation-limit)),
   so the CUDA ladder is bounded by total VRAM alone:

   | card | OpenCL | CUDA before | CUDA now |
   |---|---|---|---|
   | 16 GB | (16,1) | (16,1) | (16,1) |
   | 12 GB | (14,3), single-alloc bound | (16,1) | (16,1) |
   | 10 GB | sort path, 215 ms | (16,1) | (16,1) |
   | 8 GB | **nothing** — a reduced seed layer mines nothing | **refused** | **(15,2), 37.8 ms** |

   The 8 GB row is the point. `available()` demanded (16,1)'s 7.46 GiB + 1 GiB, so every
   8 GB Ampere/Ada card was turned away — and the OpenCL fallback cannot host a full seed
   layer at 284 B/element either, so those cards mined *nothing at all*. They now run the
   full search one rung down. The floor for the fast path is **7.6 GiB** of total VRAM.

   **The prediction is checked against the allocator, not trusted.** `rb_geometry_for()`
   sizes against *total* VRAM; a compositor can be holding a gigabyte of it. The
   constructor now steps down on real allocation failure, verified by holding VRAM
   hostage from another process: at 7.39 GiB free it walks 65536 → 32768 and mines
   correctly at (15,2).

   **This also fixed a leak.** `CudaSolver`'s constructor throws when a device is too
   small, and a throwing constructor never runs its own destructor — so every `cudaMalloc`
   made before the throw leaked for the life of the process. Any caller catching the
   throw and retrying smaller got *less* memory each attempt. The frees moved into
   `~Impl`, which runs either way. Under the VRAM-hostage test this is the difference
   between three geometries failing and one.
4a. **The 8σ bucket capacity is load-bearing — it is not the slack this document twice
   called it.** `fb_cap_for` reserves `mean + 8σ + 32` per bucket, 215 of 743 slots at
   bb=16, and both the sizing comment and lead 5 below recorded the gap to the measured
   ~4.3σ maximum as available. Measured on CUDA (`MXBM_OCC`, 201 solves × 5 scatter
   stages, unclamped counters): the worst bucket ever seen is **4.6–5.35σ, 85–91 % of
   what is reserved**. That reads like 3σ of slack, and it is not, because the tail must
   be priced per *draw* and a miner draws 2.45e6 solves × 5 stages × 65536 buckets =
   **8.0e11 bucket occupancies per day**:

   | reservation | cap | overflows/day | MTBF |
   |---|---|---|---|
   | 6σ | 697 | 8.1e-01 | ~1 dropped element **per day** |
   | 7σ | 720 | 8.1e-04 | 3.4 years |
   | 8σ | 743 | 4.0e-07 | 6900 years |

   A dropped child can be a valid solution's ancestor, and `bucketDrops != 0` fails the
   gate every run is held to. So the geometry ladder, not a tighter cap, is the lever for
   fitting a smaller card — and the "8σ→6σ reduction" lead 5 records as moot-but-real was
   never real.
4b. **Optimize the sort path.** *(Started 2026-07-28; **214.8 → 190.4 ms** so far — see
   [the round-4 mix anomaly](#the-round-4-mix-anomaly-was-a-runtime-constant-270--82-ms),
   which was the first thing on it and is now closed.)* It had been parked at ~214.6 ms
   since 2026-07-24, when the row-bucket path took over and every optimization since went
   there: it never received compile-time round constants (worth **−32.5 %** on the fused
   path), index-only records, or the packed-record work. The first instalment of the
   constants is in and was worth −9.8 %; the other two are untouched. It is
   not a dead path — it is what runs on any device that cannot host a row-bucket
   geometry, which per
   [limitation 1](HW_REQUIREMENTS.md#1-below-11-gb-opencl-is-capped-by-its-single-allocation-limit)
   means 11 GB cards and anything whose OpenCL `max_alloc` is small, and the CUDA backend
   has no sort path at all. So this is reach and it is also the only path some users will
   ever run. Known starting points: the [record audit](#the-record-redundancy-audit)
   already took it 285 → 264 B/element and left one deliberate 4 B/element (~138 MB) of
   slack in `leaves[2]`; whether the row-bucket path's wins transfer at all is the open
   question, since the two differ in structure and not just in tuning.
5. ~~**Overlap the phases.**~~ **Closed — built, measured, and null by two independent
   mechanisms. Do not retry with streams.** The profile is genuinely complementary: the
   SM is busy 50 % of the wall clock and DRAM 58 %, at different moments, which prices a
   `max(SM-busy, DRAM-busy)` roofline at 20.6 ms — a **1.71× ceiling, ~96 sol/s**.
   Neither mechanism reaches it. Two streams cannot, because every round is **662 waves
   deep** and a second kernel can only start in the last wave's tail. A co-resident
   launch cannot, because a memory-stalled warp still holds its warp slot, so hosting
   `entry` inside round 3 costs **7.25 ms against its 2.77 ms standalone**. The binding
   resource is **resident blocks per SM** — worth ~1.6 ms per extra block — and shared
   memory is what caps it. Full measurements, the positive control, the wave arithmetic
   and why the whole family is closed:
   [phase overlap cannot reach the roofline](#phase-overlap-cannot-reach-the-roofline).

**Ruled out — do not revisit** (all measured, see [What didn't work](#what-didnt-work)):
two-level bucketing, shared-memory magazines, warp-aggregated atomics, decoupling the
scatter for occupancy, SoA layouts in either global or local memory, and shrinking the
element below the `[7,7,6,5,1]` schedule. Added 2026-07-26: streaming stores, geometry
(17,0) *at stock — see the eco results for what it does under a cap*, a tighter bucket
capacity, and phase overlap by streams (grid depth) and same-warp hosting (warp slots).
Amended 2026-07-31: the third overlap mechanism — separate co-scheduled blocks — **works
and is now shipped**, and with it the family is finished rather than merely closed: its
entire yield is r4's ~0.5 ms of exploitable idle. The resident-blocks lever behind
occupancy is closed from both resources at once (shared memory AND the register file,
which r2 fills exactly). Larger blocks are *not* a substitute
([measured](#occupancy-is-worth-real-time-and-shared-memory-is-the-only-gate)).

</details>

---

## The CUDA backend
<details>
<summary>Details</summary>

Built, gated, measured, and **wired into the miner** (2026-07-25). `--solver auto`, the
default, now prefers CUDA and falls back to OpenCL and then to the CPU reference; the
backend can be pinned with `--solver cuda|opencl`. CUDA is an *optional* build component
— CMake probes for it with `check_language(CUDA)`, so a machine without a CUDA toolchain
still builds and ships the OpenCL path unchanged.

A standalone `nvcc`-only bench still lives under `cuda/` for profiling work (Nsight
Compute cannot profile OpenCL, which is what motivated the port in the first place).

| | end-to-end | verified/solve | sol/s |
|---|---|---|---|
| OpenCL (fallback) | 41.0 ms | 1.98 | 48.2 |
| **CUDA backend (default)** | **34.1 ms** | **2.00** | **58.1** |
| lolMiner (user-measured, stock) | — | — | 53.0 |

*(The OpenCL row was re-measured 2026-07-31 after
[the match-win backport](#the-cuda-match-wins-backported-to-opencl-perfect-table--spill--per-round-caps-06-ms):
**40.0 ms / 49.7 sol/s**, 60 s miner benchmark, 1 494 solves. The table above is kept
as the port-era comparison its surrounding text describes.)*

Same methodology on both sides: median over **300 distinct nonces**, persistent buffers,
including survivor readback, back-reference recovery and CPU verification, counting only
solutions that pass `bh3::is_valid_solution`. The verified rate coming out at **1.98 on
both** is a useful cross-check that the two implementations agree.

**On sample size.** Solutions per solve is Poisson-ish, so the rate needs ~600 observed
solutions for ±4 %. An earlier revision quoted 55.2 sol/s from 20 nonces — about 39
solutions, ±16 % — which was not enough to state a margin over a 53 sol/s target. At 300
nonces the figure is **56.1 ± 2.3**, so the lower bound only just clears the target. That
is a real margin but a thin one relative to the error, which is precisely why the
share-rate comparison is the thing that settles it.

**Why it is faster, and it is not the language.** The port initially measured 41.6 ms —
*slower* than OpenCL — because it reproduced the OpenCL algorithm exactly. The gain came
from one thing the profiler found: rounds 2 and 3 were stalling 12 % and 22 % of
warp-active cycles on **MIO-queue throttle**, the load/store *instruction* queue backing
up. That is an issue-rate limit, not a bandwidth one — and round 2 was simultaneously at
only 38 % of DRAM peak, so **bytes were cheap and memory instructions were expensive**,
the exact inverse of the direction every previous optimization here took.

Padding the round-3 record 9 → 10 u64 (4 % more traffic) made every record stride an even
number of u64, so `slot*stride*8` is 16 B aligned and 128-bit accesses became legal:

```
r2 emit  11 x ST.64 -> 4 x ST.128 + 3      r3 load  10 x LD.64 -> 4 x LD.128 + 3
r3 emit  10 x ST.64 -> 4 x ST.128 + 2      r4 load   9 x LD.64 -> 4 x LD.128 + 2
```

41.5 → 38.4 → 35.3 ms. **The compiler does not do this for you** — it cannot prove the
base pointer's alignment and emitted zero 128-bit accesses even after the padding made
them legal. `cuobjdump -sass | grep LDG.E.128` is the check.

**Caveats, because "target beaten" is a claim worth being careful with:**

- The 53 sol/s figure is user-measured from lolMiner's own display. The
  [counting question](#the-cuda-backend) is still open — this solver produces 2.29
  survivors per solve but only 1.95 that verify, a 17 % gap between "solutions found" and
  "solutions that are solutions". If lolMiner reports the former, our margin is larger; if
  the latter, it is the ~4 % measured here. *(Superseded 2026-07-31: that gap
  [no longer exists](#the-found-vs-verified-gap-is-gone-0-of-3935-candidates-rejected)
  — 0 rejected candidates in 3,935 across both backends, so found == verified here
  and only lolMiner's own counting basis remains open.)*
- MXBM counts **verified** solutions, the conservative definition of the two.
- Accepted pool shares over a fixed interval remain the only comparison that does not
  depend on either miner's counters. See `docs-internal/MINER_COMP.md`.
- The backend is standalone. Wiring it into the stratum path, and keeping OpenCL as the
  portable fallback, is remaining work.

*How it got there, what the profiler said afterwards, and the four levers tried and
rejected on top of it are in
[the CUDA backend in detail](#the-cuda-backend-in-detail).*

</details>

---

## The CUDA backend in detail
<details>
<summary>Details</summary>

*(What the backend is and where it stands today is in
[the section above](#the-cuda-backend). This is what the profiler found and
what was tried on top of it.)*

</details>

### Where the CUDA backend stands after the fix
<details>
<summary>Details</summary>

Second profiler run, comparing against the first:

| kernel | ms | MIO throttle | global stall | DRAM %peak |
|---|---|---|---|---|
| entry | 2.58 → 2.58 | 0.0 → 0.0 % | 2.4 % | 16.2 % |
| r1 | 5.93 → 5.82 | 1.8 → 1.5 % | 20.2 % | **28.1 %** |
| r2 | 13.29 → **10.22** | **12.0 → 2.6 %** | 20.3 % | **53.0 %** |
| r3 | 12.23 → **9.95** | **22.0 → 9.3 %** | **52.8 %** | **78.1 %** |
| r4 | 6.39 → 5.65 | 3.0 → 2.4 % | **58.7 %** | **79.8 %** |
| terminal | 1.05 → 1.04 | 2.4 → 2.5 % | 34.8 % | **79.5 %** |

**The MIO lever is spent** and `long_scoreboard` (global memory latency) has replaced it,
which is the correct state for a memory-bound kernel. Three consequences:

- **r3, r4 and terminal are effectively done** — all three at 78–80 % of theoretical DRAM
  peak, which is very high for a scattered-access mix. 16.6 of the 35.2 ms with perhaps
  20 % of headroom even under perfect latency hiding.
- **The remaining headroom is r1 + r2**, 16.0 ms at only 28 % and 53 % DRAM.
- **Shared-memory bank conflicts are a red herring.** 65–81 M per round looks alarming but
  is ~0.9 per shared load, which is the *inherent* cost of 64-bit shared access: a 32-lane
  `u64` load needs 64 banks' worth and always takes two wavefronts. No stride fixes it —
  7 u64 gives `gcd(14,32) = 2` (2-way), 8 u64 gives 16-way, 10 u64 gives 4-way while
  enabling 128-bit loads, i.e. 4 instructions × 4-way beats 7 × 2-way by nothing. This is
  also why the OpenCL [SoA LDS staging](#soa-lds-staging) experiment measured slower.

**The pair record was re-tested here and survives.** Rounds 1 and 2 are the two kernels
with DRAM headroom, so trading bytes for the removal of round 2's 14-siphash rebuild looked
plausible — and this trade has reversed on every previous regime change. Measured on CUDA
with 128-bit access in place: **35.0 ms with the pair record against 39.2 ms without**.
Keep it.

</details>

### Levers tried after the MIO fix — all null
<details>
<summary>Details</summary>

Each was indicated by the profile, implemented, measured, and kept out. Recorded with the
mechanism, because "we tried it" without a reason is not reusable.

| lever | result | why |
|---|---|---|
| `cp.async` staging | 35.0 → 35.3 ms | alignment, below |
| block size 288 / 320 / 384 | 35.1 → 35.5 / 35.6 / 36.9 | below |
| pair record removed | 35.0 → 39.2 ms | keep the pair record |
| streaming stores (`__stcs`) | 35.3 → 35.5 ms | nothing to protect; [details](#streaming-stores-implemented-measured-null) |

**`cp.async`.** The indicated lever: global-memory latency is now the top stall
everywhere (r3 52.8 %, r4 58.7 %) and this is the feature built for it. It does not pay,
and the reason is structural. `cp.async` is only on its fast path at **16 B granularity**,
where it bypasses L1 and the register file — and 16 B requires *both* source and
destination to be 16 B aligned. Our shared destination is `lwork[pos*INW]` with `INW = 7`,
i.e. byte offset `pos*56`, aligned only for even `pos`. Padding `lwork` to an even u64
stride fixes alignment and destroys banking:

```
stride 7 u64 -> 14 u32 banks, gcd(14,32) = 2  ->  2-way conflict   (today)
stride 8 u64 -> 16 u32 banks, gcd(16,32) = 16 -> 16-way conflict   (16 B aligned)
```

The chain walk reads `a[0..6]` and `b[0..6]` per matched pair — **92 M shared loads per
round**, the dominant shared traffic. Trading 2-way for 16-way conflicts there to enable a
4-instruction async copy in staging is not close. That leaves the 8 B path, which is not
the fast path, and it measured accordingly. Retained behind `-DMXBM_CPASYNC=1`.

**Block size.** The staged group is `mean_bucket / 2^submaskBits` = **264** elements
against a 256-thread block, so one warp of eight runs a second loop iteration with 8 of
256 lanes busy while the other seven wait — which is exactly the 15–16 % barrier stall the
first profile showed. Sizing the block at or above 264 removes the imbalance and is still
slower: bigger blocks cost more in shared memory per block and scheduling flexibility than
the imbalance costs. 256 stays.

**Pair record.** Re-tested because rounds 1 and 2 are the two kernels with DRAM headroom
(28 % and 53 %), so trading bytes to delete round 2's 14-siphash rebuild looked plausible,
and this trade has reversed on every previous regime change. It did not reverse here.

> Finding these needed a working error check. A launch that fails — `terminal_round` still
> carried `__launch_bounds__(256)` while being launched with 288 threads — returns *zero
> survivors*, which reads exactly like a correctness bug in the kernels. `solve()` now
> checks `cudaGetLastError()` and says so.

</details>

### What CUDA offered that OpenCL could not
<details>
<summary>Details</summary>

Ranked by what actually paid, now that it has been measured rather than guessed:

1. **A profiler.** Nsight cannot profile OpenCL at all. Every optimization in this
   document before this point came from ablation and arithmetic; the MIO-throttle finding
   was invisible to that method and was worth 6 ms.
2. **128-bit memory access** — 6 ms, above.
3. **Real shared-memory limits.** Ada has 100 KB/SM; OpenCL only ever exposes 48 KB. This
   is why [occupancy](#occupancy-again) was recorded as structurally blocked — the
   arithmetic was against the wrong budget. In CUDA it is reachable (2 → 3 blocks/SM) and,
   measured properly, **still does not help**.
4. `cp.async`, `__match_any_sync` — untried; the profile suggests neither addresses the
   current limiters.

</details>

---

## What a CUDA backend was predicted to buy (retained for calibration)
<details>
<summary>Details</summary>

Scoped because it is the largest remaining roadmap item, and the temptation is to assume it
closes the gap. It probably does not.

**Port surface.** Small, which is the good news:

| | lines | difficulty |
|---|---|---|
| `bh3.cl` primitives (siphash, `apply_mix`, `combine`) | 79 | mechanical, and the KAT gates it exactly |
| `FUSED_LDS` (one macro → 5 round kernels) | ~200 | the macro becomes a template — arguably cleaner |
| entry, terminal, recover, survivor scan | ~200 | mechanical |
| host layer (`cl_runtime`, `round_pipeline`) | ~1200 | mostly OpenCL plumbing CUDA does not need |

**Estimate: ~2 weeks** to a bit-exact port that *matches* current OpenCL performance — half
a day for the primitives, 2–3 days for the kernels, 2–3 for the host layer, 2–3 getting it
byte-identical on the goldens. That last item is where ports of this kind actually sink
time. Optimisation is on top of that.

**What CUDA offers that OpenCL cannot express, ranked by plausible value *here*:**

1. **Non-temporal / streaming stores** (`st.global.cs`). Emitted records are read exactly
   once, by the next round. Today they pass through L2 and evict staging reads that *do*
   have reuse. This is the only item that could move the **achieved bandwidth**, which is
   the only thing that moves the floor.
2. **Better register allocation and scheduling** than the OpenCL compiler — historically
   worth a few per cent.
3. **`cp.async`** (Ampere+) for the staging loop, which is exactly our global→shared
   pattern. But it hides *latency*, and we are bandwidth-bound. Expect little.
4. **`__match_any_sync`** to replace the LDS hash table in the collision find. Elegant, and
   worth ~nothing: all non-memory work totals 2.2 ms.

**Honest ceiling.** The pipeline is within ~3 % of its own memory floor. Nothing above
reduces bytes moved, so items 2–4 are bounded by that ~1.2 ms of headroom. Only item 1 can
move the floor, by an unknown amount. A realistic range is **3–8 % (1.2–3.2 ms)** →
~37–39 ms / 49–51 sol/s. **Short of 53 unless possibility 3 above is also true.**

**Recommendation: settle the counting question first.** A CUDA rewrite is two weeks against
a target that may be 11 % away or may be zero. The share-rate comparison costs an evening.

> **How this estimate held up.** The range (3–8 %, landing 49–51 sol/s) was too
> conservative: the measured result is **16 % over OpenCL, at 56.1 sol/s**. The ranking was
> also wrong. Streaming stores were called the only item that could move achieved
> bandwidth, and they were never needed; the win came from **instruction issue**, which
> this section did not consider at all — because without a profiler there was no way to
> see MIO-queue throttle. The two-week figure was human-calibrated and wrong too. What the
> section got right is that occupancy and `cp.async` would not matter.
>
> **Item 1 has since been measured too, and it is null.** Streaming stores were built
> (`-DMXBM_STCS=1`), verified applied in the SASS, and moved nothing — see "where the
> remaining time is". So the ranking was wrong at both ends: the item called "the only
> one that could move the achieved bandwidth" moves none of it, and the item that
> actually paid (instruction issue) is not in the list at all.

</details>
