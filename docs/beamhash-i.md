# BeamHash I

**Beam's launch proof-of-work.** Active from the mainnet's first block on
3 January 2019 until block 321321 (≈ 15 August 2019), when
[BeamHash II](beamhash-ii.md) replaced it.

BeamHash I is **Equihash ⟨150,5⟩ with one change to the hash data path**. Everything
that makes Equihash what it is — the generalized birthday problem, Wagner's algorithm,
the round structure, the solution encoding — is unmodified. The change is confined to
how the initial elements are derived from BLAKE2b, and it exists to make one specific
ASIC optimization expensive.

Retrospectively the [BeamHash II specification][bh2] classifies it as
**EquihashR ⟨150,5,0⟩** — the `r = 0` member of the family it introduces.

---

## 1. Equihash background

This section is the foundation for all three BeamHash variants. If you already know
Equihash, skip to [§2](#2-the-beamhash-i-data-path-change).

### 1.1 The problem

Equihash, introduced by Biryukov and Khovratovich in 2017 [\[1\]][equihash], is not a
hash function. It is a *problem*: an instance of the generalized birthday problem, whose
best known solution is [Wagner's algorithm][wagner]. The point is asymmetry — finding a
solution is memory-hard and slow, verifying one is cheap.

Equihash is parameterized by two integers `n` and `k`. Define the **collision length**

```
m := n / (k + 1)
```

Given a block header `work` and a `nonce`, elements come from BLAKE2b:

```
B(l) := Blake2b(work ‖ nonce ‖ l)          for l = 0 … 2^(m+1) / ⌊512/n⌋
```

Each 512-bit BLAKE2b output is cut into `s := ⌊512/n⌋` disjoint sections of `n` bits.
Sections are indexed first by `l`, then by position within `B(l)`. Section start points
are byte-aligned, which does not change `s` but does shift where the sections sit when
`n` is not a multiple of 8.

For Beam's ⟨150,5⟩: `m = 150/6 = 25`, and `s = ⌊512/150⌋ = 3`. The three sections of a
512-bit output begin at bits 0, 152 and 304 — byte-aligned, not at 0/150/300 — leaving
bits 456…512 as padding.

### 1.2 What a solution is

A valid solution is a set of `2^k` indices such that:

1. all indices are pairwise distinct;
2. for any `1 ≤ i < k`, the XOR of all elements referenced by any `2^(i+1)`-element
   index block is zero on the first `i·m` bits;
3. the XOR of all elements indexed by the solution is zero;
4. the indices are canonically ordered — for two index blocks of `2^i` indices, the one
   with the lowest leading element comes first.

For ⟨150,5⟩ that is `2^5 = 32` indices.

Rule 4 matters more than it looks: without it the same solution could be submitted in
`2^k`-many orderings, so it is what makes a solution's encoding unique.

### 1.3 Wagner's algorithm and step rows

The search operates on **step rows**. A step row is an `n`-bit segment carved into `k−1`
sub-segments of `m` bits plus a final one of `2m` bits, with its index list stored
alongside. For ⟨150,5⟩ that is four 25-bit sub-segments and one 50-bit tail:

```
bit 0      25      50      75     100                 150
   |   m   |   m   |   m   |   m   |        2m         |
```

Wagner's algorithm runs `k` rounds. Each round sorts the step rows by their most
significant remaining sub-segment and combines every pair whose sub-segments are equal.
The output step row is the XOR of the inputs, one sub-segment shorter, carrying the
merged index list. Rule 4 above decides which input's list goes first.

The round arithmetic is what makes the parameters hang together. Starting from `2^(m+1)`
step rows matched on `m` bits, the expected number of outputs is

```
(2^(m+1))² / (2 · 2^m) = 2^(m+1)
```

— the population is *stable* across rounds. That is exactly why `m = n/(k+1)` is the
right collision length. The final round matches on `2m` bits instead of `m`, so it yields

```
(2^(m+1))² / (2 · 2^(2m)) = 2
```

**about two solutions per run, on average — not a guarantee.** An Equihash instance may
have no solution at all. This is a real difference from a conventional hash-based PoW,
where each iteration produces exactly one candidate.

### 1.4 Degenerate solutions

If a step row enters a round twice by different paths, the round can produce an
all-zero step row, which then trivially "collides" with everything. The
[specification][bh2] works the example: if four step rows `A,B,C,D` land in the same
bucket in round 1, the round emits `AB, AC, AD, BC, BD, CD`; if any two of those match
again in round 2, `ABCD` arises by multiple paths and round 3 sees fully-zero match bits.

This is common enough that it cannot be ignored, so from round 3 onward every practical
implementation tests whether an output step row is already entirely zero and discards it.

### 1.5 Cost

Memory is driven by the number of step rows, which is `2^(m+1)` — complexity class
`O((k+n) · 2^(m+1))`. The specification tabulates the deployed Equihash instances:

| n | 96 | 200 | 210 | 144 | 192 | 125 | **150** |
|---|---|---|---|---|---|---|---|
| k | 5 | 9 | 9 | 5 | 7 | 4 | **5** |
| First coin | MinexCoin | ZCash | AION | BitcoinZ | Zero | ZelCash | **BEAM** |
| m | 16 | 20 | 21 | 24 | 24 | 25 | **25** |
| Step rows | 2^17 | 2^21 | 2^22 | 2^25 | 2^25 | 2^26 | **2^26** |
| Memory (MB) | ~7 | ~200 | ~600 | ~1500 | ~2800 | ~3000 | **~3200** |

*Source: [BeamHash II Specification][bh2], Table 1. Memory is MB per instance for
GPU-efficient implementations of the day.*

Beam's ⟨150,5⟩ is the most memory-hungry entry in that table at roughly **3.2 GB**. That
figure is the origin of the widely-quoted "Beam needs at least 3 GB" requirement.

Time is dominated by two very different phases. The BLAKE2b fill is compute-intensive;
the matching rounds are XOR plus a bucket sort, computationally light and dominated by
moving elements in and out of memory. For instances with 3 step rows per BLAKE2b output
(144/5 and 150/5) the specification measured the BLAKE2b phase at roughly the duration
of the first two matching rounds, or **~20 % of total execution time**.

---

## 2. The BeamHash I data path change

### 2.1 The attack being priced

Equihash's memory hardness can be traded away for compute. In the early rounds a chip can
store *fewer* bits per step row than the algorithm needs, and when the discarded bits are
required again, recompute them from the indices the step row still carries. This is
cheapest exactly where it hurts most — in the early rounds, where the elements are wide
and few distinct BLAKE2b passes are needed to recover them.

For a GPU this trade is never worth taking: extra BLAKE2b work adds time it cannot hide,
and raises power draw. For an ASIC, which can throw silicon at BLAKE2b, it is very
attractive. BLAKE2b is thus the component offering an ASIC the most leverage, and
Equihash ⟨150,5⟩'s low **blocking rate of 3** — three step rows per BLAKE2b call — makes
recomputation cheap.

### 2.2 The change

BeamHash I raises that blocking factor. Let `B(l)` be the 512-bit BLAKE2b output for
index `l`, and `B(l)_j` its `j`-th 32-bit component read as a 32-bit integer. The
element actually used is `B'(l)`, a running componentwise sum over the enclosing
aligned group of 16 hashes:

```
c ← 16 · ⌊l / 16⌋
B'(l) ← 0
while c ≤ l do
    B'(l)_j ← B'(l)_j + B(c)_j     for all 0 ≤ j < 16
    c ← c + 1
end while
```

*Source: [BeamHash II Specification][bh2], §2.1. Sums are over 32-bit components.*

So `B'(l)` depends not just on `B(l)` but on every `B(c)` from the start of its group of
16 up to `l`. The blocking factor rises from a flat 3 to **between 3 and 45**, depending
on the index's position within its group.

### 2.3 Why this is nearly free for a GPU and expensive for an ASIC

Computing `B'` the intended way — sweeping the whole group once, in order — costs almost
nothing extra, because you need all 16 hashes anyway and each is used as you go. On a GPU
this is a local-memory reduction in the initial fill phase, and on modern hardware
neighbouring threads whose indices differ only in the low four bits run in lockstep, so
sharing through local memory adds no synchronization barrier.

Recomputing a *single* `B'(l)` later, from an index, is the expensive case: you must
recompute up to 15 other hashes in the same group. That is up to **16× more costly** than
before, and up to **48× in the initial round** — an average extra factor of about **24**
against a time-memory trade that was previously cheap.

The intent is *algorithm binding*: make the GPU-shaped implementation the only efficient
one, so divergent designs must either follow it or pay drastically more power and area.

### 2.4 Cost to verifiers

Verification gets more expensive by an average factor of about eight. This is acceptable
because the asymmetry is enormous to begin with: verifying a solution takes 32 BLAKE2b
runs, while *generating* one averages over 11,184,810 ≈ ⅓·2^26. Even with the change, the
specification notes that verifying an Equihash ⟨150,5⟩ solution costs less on average
than verifying an Equihash ⟨200,9⟩ one, with equal worst cases.

---

## 3. Solution encoding

Unchanged from stock Equihash ⟨150,5⟩:

| Property | Value |
|---|---|
| Indices per solution | 32 (`2^k`, k = 5) |
| Bits per index | 26 (`m + 1`, m = 25) |
| Solution size | 32 × 26 = 832 bits = **104 bytes** |
| Nonce | 8 bytes |

This 104-byte wire format survived every later algorithm change — see
[BeamHash III](beamhash-iii.md#5-solution-format), which keeps the size while
repurposing the last four bytes.

---

## 4. Measured performance

From the reference OpenCL miner at stock clocks, as published in the
[BeamHash II specification][bh2] (Table 3):

| GPU | Solutions/s | Power |
|---|---|---|
| AMD Radeon VII | 17.5 sol/s | 206 W |
| Nvidia GTX 1080 | 8.5 sol/s | 132 W |

Useful as a historical baseline only — these are 2019 cards running a reference
implementation, not tuned miners.

---

## 5. Why it was replaced

BeamHash I met its binding goal but left the BLAKE2b phase carrying ~20 % of runtime and
a disproportionate share of peak power. Beam's stated policy was periodic PoW review, and
the first review concluded that *reducing* the BLAKE2b share — rather than making it more
expensive — better served the goal of favouring commodity GPUs. That produced the
EquihashR family and [BeamHash II](beamhash-ii.md), which cut the BLAKE2b component by 8×
while leaving the memory profile intact.

---

## References

- [\[bh2\]][bh2] Wilke Trei, *BeamHash II Specification*, 17 June 2019 — the primary
  source for BeamHash I's definition, the data-path change, and the Equihash background
  above.
- [\[equihash\]][equihash] Biryukov & Khovratovich, *Equihash: Asymmetric Proof-of-Work
  Based on the Generalized Birthday Problem*, 2017.
- [\[wagner\]][wagner] David Wagner, *A Generalized Birthday Problem*, LNCS 2442 (2002).

[bh2]: https://docs.beam.mw/BeamHashII.pdf
[equihash]: https://ledger.pitt.edu/ojs/index.php/ledger/article/view/48
[wagner]: https://link.springer.com/chapter/10.1007/3-540-45708-9_19

---

*MXBM implements [BeamHash III](beamhash-iii.md) only, which is the algorithm Beam has
used since June 2020. BeamHash I and II are documented here for context; no MXBM code
implements them.*
