# BeamHash II

**Beam's second proof-of-work.** Activated at block **321321** (≈ 15 August 2019, node
release *Clear Cathode 3.0*) and retired at block **777777** (28 June 2020) in favour of
[BeamHash III](beamhash-iii.md).

BeamHash II is **EquihashR ⟨150,5,3⟩**. It keeps [BeamHash I](beamhash-i.md)'s
parameters, memory profile and wire format, and adds one new parameter — `r` — that cuts
the BLAKE2b work by a factor of `2^r`.

The direction is worth stating up front, because it inverts BeamHash I's. BeamHash I made
BLAKE2b recomputation *more* expensive, to punish an ASIC time-memory trade. BeamHash II
concluded that the better lever was to make BLAKE2b *matter less* — shrinking the one
phase where special-purpose silicon has a real advantage over a GPU.

---

## 1. The EquihashR family

The [BeamHash II specification][bh2] defines a family, not a single algorithm.

> **Definition.** Given integer parameters `n`, `k` and `r`, the **EquihashR ⟨n,k,r⟩**
> proof-of-work scheme equals Equihash ⟨n,k⟩ with the following modifications:
>
> **(a)** BeamHash I's [hash data-path change](beamhash-i.md#22-the-change) is applied to
> the hash calculation.
>
> **(b)** Instead of `2^(m+1)` step rows, the BLAKE2b phase creates `2^(m+1−r)` of them.
> The generated step rows equal those of the original algorithm, restricted to the first
> indices until the cap is reached.
>
> **(c)** The first `2·r` bits of each `n`-bit step row are reset to 0.
>
> The rest of the algorithm is unchanged, so by (c) the first round matches on `m − 2r`
> bits instead of `m`.

*Paraphrased from [BeamHash II Specification][bh2], §2.2.*

Note that (a) makes BeamHash I itself a member of the family: **EquihashR ⟨150,5,0⟩**.

### 1.1 Why the round arithmetic still works

The three modifications are balanced so that the algorithm's shape is preserved. Fewer
step rows enter round 1, but they are matched on fewer bits, so more of them pair up. The
expected number of round-1 outputs is

```
(2^(m+1−r))² / (2 · 2^(m−2r)) = 2^(2m+2−2r) / 2^(m−2r+1) = 2^(m+1)
```

— **identical to stock Equihash.** From round 2 onward the two are indistinguishable in
population, so rounds 2…k are untouched, and so is peak memory. The only structural
saving is a smaller round-0 index tree.

This is the crux of the design: `r` buys a `2^r` reduction in the compute-heavy phase
while leaving the memory-hard phase exactly as it was.

### 1.2 What `r` costs

The saving is not free. Because only `2^(m+1−r)` step rows are generated but round 1 still
emits `2^(m+1)`, each index appears in more round-1 step rows — on average `2 · 2^r`
instead of 2.

That erodes the independence assumption behind the round-population estimate, and it
increases the rate of [degenerate all-zero step rows](beamhash-i.md#14-degenerate-solutions).
The specification is explicit that `r` must therefore be used **moderately**, and reports
that from `r = 9` upward the number of candidates that survive round 5 only to be
discarded for duplicate indices grows exponentially.

---

## 2. BeamHash II concretely

For the August 2019 fork Beam selected:

| Parameter | Value |
|---|---|
| `n` | 150 |
| `k` | 5 |
| `r` | **3** |
| Collision length `m` | 25 |
| BLAKE2b reduction | **8×** |
| Step rows generated | `2^23` (was `2^26`) |
| Bits zeroed per step row | `2r` = 6 |
| Round-1 match width | `m − 2r` = **19 bits** |
| Rounds | 5 |
| Peak step rows | `2^26`, unchanged |
| Memory | ~3.2 GB, unchanged |

The step row pattern gains a dead 6-bit prefix; round 1 matches only the 19 bits that
remain of the first sub-segment:

```
      zeroed
bit 0 ┌─6─┐    25      50      75     100                 150
      |///|  19  |   m   |   m   |   m   |        2m        |
```

### 2.1 Solution format — deliberately unchanged

Solutions remain **32 × 26 bits = 832 bits = 104 bytes**, with an 8-byte nonce.

In principle `r = 3` allows a smaller encoding: with 6 bits of every step row forced to
zero, indices could be packed in 23 bits, giving `32 × 23` bits. The specification records
that this reduction was **consciously discarded** — keeping the format identical spared
miner developers and pools an implementation change and removed a source of error. The
stratum protocol was therefore unaffected by the fork.

---

## 3. Empirical results

The specification validated `r` by simulating EquihashR ⟨150,5,r⟩ over at least 10,000
nonces per value of `r`.

**Step rows surviving each round** (averages over 10,000 simulations):

| Variant | BLAKE2b round | Round 1 | Round 2 | Round 3 | Round 4 | Solutions |
|---|---|---|---|---|---|---|
| BeamHash I (`r=0`) | 67,108,608 | 67,108,440.8 | 67,108,161.7 | 67,107,131.0 | 67,106,132.9 | **1.989** |
| `r=1` | 33,553,920 | 67,108,892.9 | 67,105,068.0 | 67,101,323.7 | 67,093,913.1 | **1.971** |
| `r=2` | 16,776,960 | 67,106,527.9 | 67,104,287.7 | 67,099,851.3 | 67,090,942.7 | **2.026** |
| **BeamHash II (`r=3`)** | 8,388,096 | 67,098,069.8 | 67,087,248.5 | 67,065,681.5 | 67,022,639.6 | **2.015** |
| `r=8` | 512,776 | 66,977,746.5 | 66,846,936.1 | 66,584,134.8 | 66,063,507.7 | **1.924** |

*Source: [BeamHash II Specification][bh2], Table 2.*

Two things to read off it. First, the round-1 output snaps back to ≈ `2^26` for every
value of `r` — the theory in §1.1 holds empirically. Second, **solutions per iteration
stays at ≈ 2.0** across the whole range, so `r` does not change the yield of the puzzle.

(The BLAKE2b-round counts are slightly under the exact `2^(26−r)`; the specification notes
the count was rounded down to keep the number of BLAKE2b calls divisible by 256, which
suited their GPU implementation.)

**Performance and power**, reference OpenCL miner, stock clocks:

| GPU | BeamHash I | | BeamHash II | |
|---|---|---|---|---|
| | sol/s | Watts | sol/s | Watts |
| AMD Radeon VII | 17.5 | 206 W | **23.6** | **175 W** |
| Nvidia GTX 1080 | 8.5 | 132 W | **11.3** | **118 W** |

*Source: [BeamHash II Specification][bh2], Table 3.*

About **30 % more solutions per second at 10–20 % lower power**, consistently across both
vendors. Since the extra throughput comes from deleting compute rather than from moving
more data, both figures move the right way at once — which is what the design predicted.

---

## 4. Why it was replaced

BeamHash II was a parameter change, not a redesign: an Equihash ⟨150,5⟩ search with a
cheaper fill phase. Its memory profile, element width and round structure were all still
stock Equihash, and the elements themselves remained static strings — an element's
collision key for every round is a fixed slice of a value computed once, so a solver can
carry a compact "running XOR" and never revisit the hash.

[BeamHash III](beamhash-iii.md) broke that property deliberately. It replaced the BLAKE2b
generator with SipHash-2-4 and introduced a per-round mixing step that folds the
accumulated index tree back into the element, so the collision key must be re-derived
every round from data that grows as the search proceeds.

---

## References

- [\[bh2\]][bh2] Wilke Trei, *BeamHash II Specification*, 17 June 2019 — the primary
  source for everything on this page.
- [Beam Hard Fork Announcement][fork1] — the August 2019 fork at block 321321.

[bh2]: https://docs.beam.mw/BeamHashII.pdf
[fork1]: https://medium.com/beam-mw/mimblewimble-hardfork-announcement-c956909f87d3

---

*MXBM implements [BeamHash III](beamhash-iii.md) only. BeamHash I and II are documented
here for context; no MXBM code implements them.*
