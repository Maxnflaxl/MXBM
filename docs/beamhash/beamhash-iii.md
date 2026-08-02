# BeamHash III

**Beam's current proof-of-work**, active since block **777777** (28 June 2020). This is
the algorithm MXBM implements.

BeamHash III is a redesign rather than a re-parameterization. Where
[BeamHash I](beamhash-i.md) and [II](beamhash-ii.md) were Equihash ⟨150,5⟩ with the hash
data path adjusted, BeamHash III changes what an element *is*:

- **BLAKE2b is demoted to a seed.** It is called once per nonce to derive a 32-byte
  key; the `2^25` elements themselves come from a non-standard **SipHash-2-4**.
- **A mixing step runs before every round**, folding the element's accumulated index
  tree back into its own work bits. The collision key is therefore **re-derived each
  round from data that grows as the search proceeds** — it is not a static slice of a
  value computed once.
- **The parameter shape moves to Equihash ⟨144,5⟩** — 24-bit collisions, 25-bit
  indices, `2^25` step rows — from ⟨150,5⟩'s 25-bit collisions and `2^26` step rows.

The second point is the one that matters to a solver. In BeamHash I/II an element can be
carried as a compact running XOR and its key for round `r` read straight off it. In
BeamHash III that trick does not exist: every round must recompute the mix, and the mix
input includes the index tree, which doubles in size each round.

> **Primary source.** Wilke Trei, [*BeamHash III Short-Specification*][bh3], 28 March
> 2020. Every constant and formula on this page is from that document unless marked
> otherwise; the cross-references to MXBM source are our implementation of it.

---

## 1. Naming: ⟨150,5⟩ or ⟨144,5⟩?

Beam's own website describes BeamHash III as "a modified version of Equihash (150,5)".
That label is inherited from BeamHash I and II, and **it does not describe BeamHash III's
parameters.** The specification instead says the index tree "has similar properties to the
ones known from the **Equihash 144/5** algorithm", and the constants agree:

| | Equihash ⟨150,5⟩ (BeamHash I/II) | Equihash ⟨144,5⟩ shape (BeamHash III) |
|---|---|---|
| Collision length `m = n/(k+1)` | 25 bits | **24 bits** |
| Bits per index (`m+1`) | 26 | **25** |
| Step rows `2^(m+1)` | 2^26 | **2^25** |
| Indices per solution `2^k` | 32 | 32 |
| Packed index bytes | 104 | **100** |

Both land on a 104-byte wire format — ⟨150,5⟩ as `32 × 26` bits, BeamHash III as 100
packed bytes plus a 4-byte extra nonce — which is probably why the older label stuck.

Strictly, BeamHash III is not literally Equihash ⟨144,5⟩ either: its elements are 448
bits wide, not 144, and the collision key comes out of the mixing step rather than being
a slice of the element. The accurate description is **an Equihash-style Wagner search
using the ⟨144,5⟩ parameter shape**, with a different element generator and a per-round
mix. This document uses that phrasing.

---

## 2. Structure

A solution is computed in **six rounds**: round 0 is the *seeding phase*, rounds 1–5 are
*combination* rounds. The final solution is the index tree produced by round 5.

Every element — a **step row** — has two parts:

| Part | Round 0 | after R1 | after R2 | after R3 | after R4 | after R5 |
|---|---|---|---|---|---|---|
| **work bits** | 448 | 424 | 400 | 376 | **288** | 0 |
| **index tree** (bits) | 25 | 50 | 100 | 200 | 400 | 800 |
| index tree (entries) | 1 | 2 | 4 | 8 | 16 | 32 |

*Source: [BeamHash III Short-Specification][bh3], §3 table.*

The index tree obeys three rules:

1. Each entry in the tree must be **unique**.
2. Two trees output in round `i` may only combine in round `i+1` if the **24 least
   significant work bits** of their step rows are equal — i.e. XOR to zero. **In round 5
   this applies to the lowest 48 work bits.**
3. When two trees combine, the one with the **lowest entry is placed first** in memory.

Rule 3 is what makes a solution's encoding canonical, and rule 1 is what rejects the
[degenerate solutions](beamhash-i.md#14-degenerate-solutions) that Wagner searches
otherwise produce.

Note round 4's work-bit width: 288, not the 352 that dropping 24 bits would give. See
[the round-4 anomaly](#the-round-4-anomaly).

---

## 3. Seeding phase (round 0)

### 3.1 Deriving the key

A pool or solo node supplies a **32-byte pre-work** — a hash of the block header. The
miner appends an **8-byte nonce** and a **4-byte extra nonce**:

```
IndividualWork = Blake2B( pre-work ‖ nonce ‖ extraNonce )      → 32 bytes
```

The BLAKE2b input is therefore **44 bytes** in that order, and the 32-byte output is read
as four little-endian `uint64` values — MXBM calls them `prePow[0..3]`.

> **Specification vs. deployed implementation.** The specification says BLAKE2b is
> initialized with `"BeamHash"` as the personalization string. Beam's deployed
> implementation instead uses a 16-byte personalization of `"Beam-PoW"` followed by
> `LE32(448)` and `LE32(5)` — the work-bit width and round count, following the usual
> Equihash convention of binding the parameters into the personalization. MXBM uses the
> latter, which is what reproduces Beam's reference verifier on the checked-in
> known-answer vectors. Implement from the code, not from the string in the spec.
>
> — `src/beamhash/bh3_blake2b.cpp:10`

### 3.2 Generating elements

Round 0 generates **`2^25` initial step rows**. Each one's index tree is just its own
25-bit index. Its 448 work bits are produced as seven 64-bit chunks:

```
WorkBits[ i·64 … (i+1)·64 − 1 ]  =  SipHash24( IndividualWork, (index << 3) + i )
```

for `i = 0 … 6`. Because the counter is `(index << 3) + i` with only seven values of `i`
used, one input in every eight (`i = 7`) is never consumed.

### 3.3 The SipHash-2-4 is non-standard

Two deviations from RFC SipHash-2-4, both visible in `src/beamhash/bh3_primitives.h:8`:

- **No constant XOR at initialization.** The state is set *directly* to
  `v0..v3 = prePow[0..3]`. Standard SipHash XORs the key against
  `"somepseudorandomlygeneratedbytes"`.
- **No length/padding block.** The input is a bare 64-bit counter, so the usual final
  block encoding the message length is absent.

The compression structure itself is standard: `v3 ^= nonce`, two SipRounds, then
`v0 ^= nonce`, `v2 ^= 0xff`, four SipRounds, returning `v0 ^ v1 ^ v2 ^ v3`.

This makes the seeding phase pure integer ALU work with no memory traffic — which is
exactly how it profiles in practice. In MXBM's CUDA backend the entry kernel runs at
**98 % of SM throughput and 16 % of DRAM throughput**; see
[performance-research.md](../performance-research.md).

---

## 4. Combination phase (rounds 1–5)

Each round applies a **mixing step** to every step row, then combines matching pairs.

### 4.1 The mixing step

Before each combination, the lowest 64 work bits of every step row are *replaced*.

**Input.** Serialize the step row: its current index tree is concatenated to its
remaining work bits, and the result is zero-padded or truncated to **512 bits**, read as
eight 64-bit unsigned integers `s_0 … s_7`.

**Formula.**

```
                ⎛  7                              ⎞
WorkBits[0..63] = ⎜  Σ   s_i ⋘ (29·(i+1) mod 64)  ⎟ ⋘ 24
                ⎝ i=0                             ⎠
```

where `⋘` is 64-bit rotate-left and the sum is **64-bit modular addition, not XOR**.

The rotation amounts `29·(i+1) mod 64` evaluate to:

| i | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| rotate | 29 | 58 | 23 | 52 | 17 | 46 | 11 | 40 |

— which is the `ROT[8]` table in `src/beamhash/bh3_primitives.h:50`.

Three consequences worth being explicit about:

- **Only word 0 is overwritten.** Words 1–6 pass through the mix untouched; they only
  lose 24 bits later, at the combine. The collision key is read from word 0
  (`w[0] & 0xFFFFFF`), so the mix is precisely the step that determines what collides.
- **The index tree is an input, not an output.** It is folded into a scratch buffer to
  compute word 0 and then discarded; it does not need to travel inside the work words.
- **How much of the tree actually reaches the mix is capped.** The tree is written into
  the 512-bit buffer at bit offset `Lmix + i·25`, so at most `⌈(512 − Lmix)/25⌉` entries
  fit — 3, 4, 5, 6, 9 for rounds 1–5. The tree itself holds 1, 2, 4, 8, 16, so the
  effective count is

  ```
  padNum = min( ⌈(512 − Lmix)/25⌉ , treeLen )  =  1, 2, 4, 6, 9
  ```

  In rounds 1–3 the tree is the binding constraint and every entry is folded in; from
  round 4 the 512-bit buffer binds instead, and the mix reads only a *prefix* of the tree.

Because the mix consumes the index tree, **it cannot be hoisted out of the round loop**.
Every round genuinely re-derives its own collision key. This is the single most important
structural fact for anyone writing a solver: the compact running-XOR element that works
for [BeamHash I/II](beamhash-ii.md#4-why-it-was-replaced) has no analogue here.

### 4.2 The combination step

Let `A` and `B` be two step rows whose mixing step is complete, satisfying

```
WorkBits_A[0..23] == WorkBits_B[0..23]
```

Then a new step row `C` is created for the next round with

```
WorkBits_C[0..n] = WorkBits_A[24..n+23] ⊗ WorkBits_B[24..n+23]
```

— XOR the pair and shift out the 24 bits just resolved. `C`'s index tree is the
concatenation of `A`'s and `B`'s, ordered by rule 3.

The width `n` of the result per round is the table in [§2](#2-structure). Every round
drops the 24 matched bits, **except round 4, which drops 88.**

#### The round-4 anomaly

Round 4 leaves **288** work bits where the pattern predicts 352. The specification is
explicit that this is intentional:

> Note that intentionally in round 4 we drop the most significant left work bits and
> reduce to a size of 288 bits […] This encourages implementations that do the mixing
> step of the next round before writing out the results of the current round.

With 288 work bits and a 400-bit index tree, a round-4 output is 688 bits — but an
implementation that performs round 5's mix *before* storing sees only what the mix needs,
and the extra 64-bit drop lets the retained state fit a 512-bit (`ulong8`) vector. The
parameter choice is a deliberate nudge toward fusing the mix into the emit, which is
exactly what MXBM's fused row-bucket kernel does.

### 4.3 Acceptance

After round 5 the work bits must be **entirely zero**. Combined with rule 2's 48-bit
match in round 5, this is the standard Equihash terminal condition: the final XOR
resolves the last two 24-bit blocks at once.

MXBM implements this as a 24-bit match to pair the elements, a combine at `Lout = 24`,
then an all-words-zero test — equivalent, and it is where `is_valid_solution` returns.

---

## 5. Solution format

Beam's wire format predates BeamHash III and was kept unchanged, which produces a small
piece of protocol archaeology.

| | BeamHash I / II | BeamHash III |
|---|---|---|
| Indices | 32 | 32 |
| Bits per index | 26 | **25** |
| Packed indices | 832 bits = 104 bytes | 800 bits = **100 bytes** |
| Trailing bytes | — | **4-byte extra nonce** |
| **Total solution** | **104 bytes** | **104 bytes** |
| Nonce on the wire | 8 bytes | 8 bytes |
| Effective nonce | 8 bytes | **12 bytes** |

BeamHash III's ideal packing is 100 bytes, so the **4 most significant bytes of the
solution are defined as an extra nonce**, freely chosen by the miner to extend the nonce
range. Nodes and pools still send an 8-byte nonce and expect a 104-byte solution exactly
as before; in reality the miner is working with a 12-byte nonce and a 100-byte solution.

Two properties follow, and both matter:

- The extra nonce is **part of the difficulty calculation**, because it is part of the
  104 solution bytes that get hashed.
- The extra nonce is **also appended to the BLAKE2b seeding input** ([§3.1](#31-deriving-the-key)),
  so changing it changes every element. It is a genuine extra nonce, not padding.

---

## 6. Verifying a solution

A verifier does not search. Given the header, nonce and a 104-byte solution it:

1. Reads the extra nonce from `soln[100..103]` and recomputes `prePow` from
   header ‖ nonce ‖ extraNonce.
2. Unpacks 32 × 25-bit indices from `soln[0..99]`.
3. Seeds one element per index via SipHash-2-4.
4. Walks rounds 1…5. At each round, for each pair: apply the mix at that round's `Lmix`,
   check the 24 collision bits are equal, check the two index sets are **disjoint**,
   check the ordering rule `tree[0] < tree'[0]`, then combine at that round's `Lout`.
5. Accepts iff the single surviving element is all-zero.

Any failed check rejects immediately. This is `bh3::is_valid_solution` in
`src/beamhash/bh3_verify.cpp`, which mirrors Beam's `BeamHash_III::IsValidSolution`.

Cost is 32 SipHash-2-4 element generations plus 31 mix/combine pairs — microseconds,
against a search that averages ~2 solutions per `2^25`-element run.

---

## 7. Round schedule, concretely

Everything above collapses into two per-round constants. `Lmix` is the bit offset at
which the index tree is written into the mix buffer (equivalently, the surviving work
width entering the round); `Lout` is the work width after the combine.

| Round | `Lmix` | `padNum` | `Lout` | index tree in | index tree out | significant words |
|---|---|---|---|---|---|---|
| 1 | 448 | 1 | 424 | 1 | 2 | 7 |
| 2 | 424 | 2 | 400 | 2 | 4 | 7 |
| 3 | 400 | 4 | 376 | 4 | 8 | 6 |
| 4 | 376 | 6 | **288** | 8 | 16 | 5 |
| 5 | 288 | 9 | 24 → 0 | 16 | 32 | 1 |

Derived in `src/beamhash/bh3_verify.cpp:35`:
`Lmix = 448 − (r−1)·24`, less 64 at round 5; `Lout = 448 − r·24`, less 64 at round 4, and
24 at round 5.

The **significant words** column is the practical consequence: because `Lout` shrinks,
the number of nonzero 64-bit work words falls `[7,7,6,5,1]`. A solver that stores full
7-word elements every round moves bytes it can prove are zero. MXBM exploits this — see
[fixed-width compaction](../performance-research.md#fixed-width-compaction).

---

## 8. Where this lives in MXBM

The proof-of-work core is portable, dependency-free and shared by every backend: the
same header compiles as host C++, as an OpenCL kernel body, and as CUDA device code.

| Concept | File |
|---|---|
| Constants (rounds, collision bits, index width) | `src/beamhash/bh3_types.h` |
| `prePow` derivation (BLAKE2b) | `src/beamhash/bh3_blake2b.cpp` |
| SipHash-2-4, `seed_element` | `src/beamhash/bh3_primitives.h:8` |
| `apply_mix` (the mixing step) | `src/beamhash/bh3_primitives.h:35` |
| `combine` (the combination step) | `src/beamhash/bh3_primitives.h:57` |
| Index packing / unpacking | `src/beamhash/bh3_primitives.h:69` |
| Solution validation | `src/beamhash/bh3_verify.cpp` |
| OpenCL kernels | `kernels/opencl/` |
| CUDA kernels | `kernels/cuda/` |

Correctness is established two ways: against golden known-answer vectors generated from
Beam's own solver (`tests/vectors/beamhash3-kat.md`), and — when a Beam checkout is
supplied at build time — differentially against Beam's reference `IsValidSolution` over
tens of thousands of fuzzed inputs. Every GPU optimization is gated on reproducing the
known-answer solutions byte-for-byte.

For how the search itself is organized on a GPU — bucketing, the fused row-bucket
pipeline, and the measured hardware limits that shape it — see
[performance.md](../performance.md) and [architecture.md](../architecture.md).

---

## References

- [\[bh3\]][bh3] Wilke Trei, *BeamHash III Short-Specification*, 28 March 2020 —
  the primary source for this page.
- [Introduction to Beam Hash III][slides] — Wilke Trei's companion slide deck.
- [BeamHashIII: Beam forks at block 777777][fork2] — the June 2020 activation.
- Predecessors: [BeamHash I](beamhash-i.md), [BeamHash II](beamhash-ii.md).

[bh3]: https://docs.beam.mw/beamHash_III_spec.pdf
[slides]: https://docs.beam.mw/Beam_Hash_III_Slides.pdf
[fork2]: https://medium.com/minerstat/beamhashiii-beam-forks-to-a-new-algorithm-at-block-777777-dd2aeacc9e5
