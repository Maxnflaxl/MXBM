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
| [Measured results, 2026-08-12](#measured-results-2026-08-12) | the mix's tree truncation at r4/r5 and the linear-lane decomposition (both pinned by identity tests); the w0-checkpoint record — a measured loss with its mechanism; **instruction placement is not a lever on sm_89 (single-issue), only count is**; MATCH_FIRST wins at the floor and composes with (17,0) for −2.4/−2.7 % (the tail closure was stock-scoped); (17,0)'s floor prize reproduces on the current build — re-arm condition met; **the switching-height re-pricing: a store design with a re-derived round 1 bounds −13/−26/−11 % at 140/120/100 W — and the h=1 build-out (2026-08-13) kills it: +0.9–1.8 % measured at all three points, because the cap's currency is L2 sectors × core clock, not the DRAM bytes the mock priced**; **co-residency is closed for same-mix tenants** — two real pipelines in green-context partitions gain nothing at any operating point; under a cap the card is power-bound and SMs are fungible with clock (34 of 66 SMs costs 4 % at 120 W); back-ref row 5 was capacity-sized for survivor-indexed data, −0.26 GiB every rung; **the stock free list inverts at the floor** — r2's marginal store bytes move at the rung's own 282 GB/s at 100 W (pure bytes) and the mix bills ~1.5 ms/solve; **the first hardware census** — r1's stock wall is its own barrier (31.5 % of stalls), r3 idles 61 % of its lanes, and the census reconciles with the marginal-replay instrument across methods; **populations are pinned at 2^25 and the occupancy tail is thin** — a mean+2σ dense cap + ~9 MB spill arena buys −1.50 GiB at (16,1), and the record's address-redundant bits buy another −0.36; **merging the two back-ref stores into one u64 is a wash everywhere** — gi-sequential store streams already merge in L2, so a store stream is priced by its scatter pattern, not its store count |
| [Measured results, 2026-08-14](#measured-results-2026-08-14) | **the implicit-bits pack generalized to (17,0)** and shipped under the low-power gate — −3.9/−4.7 % at 120/100 W + rung on top of match-first, SASS-identical at stock; the floor curve moves to 3.70 J/sol at 160 W and 3.76 at 120 W; a carveout drift fixed; **duty-cycled average power closed with mechanism** — the arbitrage is real but a resident context idles at 41–47 W in P2/P3 and the managed floor (31.7 W) sits above the 27.6 W break-even; kWG 288, PRMT rotates and uniform-datapath offload all null; **the stall structure holds no lever the shipped knobs do not** — r3's idle lanes ARE the sub-mask filter, half-used store sectors are the 16 B-store floor, and warp specialization needs a second ~19 KB staging area against r1's 448 B of headroom; **the reach composition built** — the implicit-bits allocation reclaimed (−0.36 GiB) and the overflow arena ported to a fit-ladder rung (+1.7 % for −1.07 GiB, no round losing a block), taking the CUDA floor 4.40 → **4.03 GiB**, and with the availability allowance re-derived (1 GiB → 640 MiB) that is a 5 GB card; **the octo record takes it to 1.90 GiB for +35 %** — round 3's output as its eight leaves, which kills three back-reference rows and then round 2's `gi` with them, six u64 per slot with no field nothing reads, clearing BeamHash III's stated 3 GB minimum; **the stock per-stage profile re-taken** — 32.39 ms attributed, the implicit-bits record's 1.5 ms landing where its mechanism says (r2 −0.94, r3 −0.52, nothing else past 0.06); **the octo record's two halves priced apart** — round 4's rebuild is a flat **+16 ms** on either input, so a reach lever that can never be a speed lever; **the solutions-per-solve multiplier is the algorithm's, not the solver's** — nothing dropped (four counters zero over 1,892 solves), nothing invalid produced (56,781 of 56,783 candidates verify over 28,305 solves), and `bb + sm + 7 = 24` makes the partition incapable of separating a colliding pair. Pinned at **2.006 ± 0.008**, which corrects the 1.98 this ledger quoted and moves the reference miner's solve time to **37.9 ms**; and **the stock census re-taken** — `sm__throughput` IS the integer pipe, so the currency is ALU-pipe slots on a half-width INT32 unit: entry at 98.7 %, r1+r2 at 76–78 % with issue slots half busy, r3+r4 at 76.5/82.7 % of DRAM peak with traffic exactly compulsory (12.30 GB). The barrier is top stall in round 2 as well as round 1, which falsifies the warp-specialization closure's premise without changing its verdict. Both leads the census pointed at are **zero** — **wider LOP3 LUT fusion** (SipHash's XORs are irreducibly 2-input, which empties the single-issue closure's exemption list) and **lane density in r3/r4 at stock** (memory-bound with an 18 %-idle ALU pipe; the (17,0) that would delete the filter costs ~23 % at stock) |
| [Measured results, 2026-07-31](#measured-results-2026-07-31) | co-blocks, the third overlap mechanism; speculative entry ships; the solver reorganization — the proposal, condensed, and the probes that killed it; the CUDA match wins backported to OpenCL (−0.6 ms); two below-the-floor levers ship (−0.22 ms); the found-vs-verified gap is gone; **lolMiner measured under ncu — state-storing confirmed, its ceiling is a DRAM roofline**; the sort path's k1/k2 regression is half occupancy, half unexplained — generic stays; **the OpenCL small-card push (2026-08-01/02): the record-set split takes the floor from 11 GB to CUDA's 5.7 GiB, the 128-bit family −5.4 ms, speculative entry −0.35 ms — the fallback ends at 1.012× of CUDA** |
| [Measured results, 2026-08-15](#measured-results-2026-08-15) | **every back-reference row deleted, −2.03 ms and −688 MiB** — recovery replays rounds 3 and 4 over the one bucket each child's record names, identifies the pair by content, and reads round 2's four leaves; both bucket hints ride in bits that provably reach nothing, and round 4's record drops to 8 B. The closure that held this back priced a third record set at +2.5 GB from the width of its neighbours; round 4's own plane costs 0.32 GiB. Gated leaf-for-leaf over 1426 solves; the pipe census — r1/r2 are 22 points under the ALU roofline and it is warp supply; store-versus-derive collapses to one exchange rate with h=2 a structural optimum; **round 3 stores a work word round 4 provably never reads** (identity test + whole-pipeline poisoning, both positive-controlled) — worth 268 MB and **nothing in time**, because halving r3's write sectors buys 4 % of the round, an implied ~3000 GB/s against a 656 GB/s bus: *price narrowings in memory instructions and sectors, never in bytes*; three corrections — the warp-supply target is **unreachable** (8 blocks × 256 threads is 64 warps, sm_89 allows 48) and worth ≈ −0.66 ms not −3.07; a fifth r2 block needs **≤48 registers as well as ≤19456 B shared**, and the register half is now measured solved (64 → 48, zero spill, +0.023 ms), so **shared is the sole gate** at a 4168 B deficit no *pair* of the available cuts reaches; **`drops[0]` has no writer**, so every four-counter gate has been three; **round 3's `apply_mix` costs 0.024 ms and round 1's 0.736**, bounding ALU restructuring to r1; **the w0-checkpoint pair record ships at −0.76 ms (−2.4 %)** — round 1 stores the child's post-mix word 0 and round 2 derives only the linear lane (12 siphashes, no mixes) in the SAME 16 B, because the address-implied key bits pay for word 0's extra 24: a re-derivation is priced against what its checkpoint costs to CARRY, so one that fits a packed record's slack is worth re-pricing even after losing in a form that had to grow the record — and it is the first lever here that pays MORE under a cap than at stock, **+11.9 % at 100 W against +3.3 % at 285**, moving both crossings to ~200 W and the efficiency peak to 220 W (3.642 J/sol); **two instruments were returning false greens** — `bench_rounds`/`test_gpu_solver` drive the OpenCL backend and cannot see a CUDA change, and a CONSTANT poison is invisible through `combine`'s pairwise XOR, so a control here must be element-dependent; **the block-exit barrier is gone for −0.048 ms, a tenth of the prediction, and that bounds its whole family** — a barrier's stall percentage is not its time cost when the SM holds other blocks; **singleton-free staging ships at −0.089 ms** — 12.8 % of a group is alone in its chain slot and therefore provably nobody's ancestor, worth **−0.85 ms** to skip against **+0.76** for the word-0 prepass that finds it, registers and shared byte-identical: a large prize and a larger question, which is only what it costs to learn which slots are singletons; **the packed `tab` word ships at −0.060 ms** — the census count and the chain head share one word, so the table is initialised once per group and a second barrier goes with the second clear, which with the block-exit barrier is two independent measurements of ~0.05 ms per `__syncthreads()`; **the 1.68 ms of bank conflicts is a ceiling on a pipe that is never the critical path** — a calibration against time shows the counter does not charge for a 64-bit access's two wavefronts, the solver's `l1tex` runs at 32–43 % of peak, and the strongest available layout change moves the count under 2 %, because the population is the chain walk's scattered reads; and **compile-time geometry (+0.26 % stock, +1.12 % at 120 W) and the FMA-pipe re-encoding are both dead** — a kernel parameter that reaches the SASS only as an instruction operand is already free, and ptxas has already moved 95.9 % of the carry-consuming high halves off the ALU pipe; and **the w0 checkpoint reaches the octo record for −4.39 ms (−7.93 %) at no footprint cost** — storing the child's word 0 deletes all 15 `apply_mix` calls and 8 of the 56 siphashes in `rebuild_r4`, paid for out of a `gi` nothing indexes any more and 14 key bits the bucket address already carries, so the record stays 32 B and the 1.90 GiB floor rung goes 61.14 → 56.97 ms; **and it reaches the quad record for −1.84 ms (−5.11 %), where word 0 needs no repacking at all** — the quad record's first u64 held the key, and word 0's own low 24 bits *are* that key, so it goes in whole and the leaves move down into the pack the packed record already uses, 190 bits of 192, with no implicit-bits pack and no perfect table required; every quad rung drops ~2 ms; **the walk's shared reads are free** — `combine` is an XOR, so the lane's own element can be read once per chain instead of once per step, which takes the inner loop from 12 `LDS` to 9 at unchanged occupancy and measures **+0.10 %**, because each round emits about as many children as it consumes and the walk therefore averages **one chain step per element**: there is nothing to amortise, which bounds that whole family; and **the GPU computes 99.3 % of a solve** — a steady-state solve is 25 null-stream dispatches (8 kernels, 14 memsets, 3 copies) and the entire non-kernel budget is **0.19 ms**, so there is no idle-GPU family; **and OpenCL reaches the same 1.90 GiB floor** — the octo rung's two remaining bugs were both about addresses, round 4 writing its reference row three rows past the end of a one-row allocation and a half-local slot number meaning two different records in a split set (which is *every* card this rung exists for), and the implicit-bits record ported with it for **−3.34 % and −0.36 GiB**, taking the two backends' floors from 4.04 / 1.90 GiB to the same number and unlocking a packed dense-cap rung that used to crash rather than step down; and **the instruction census: 68.8 % of every instruction the solver executes is SipHash**, 80.2 % inside the three ALU-pipe-bound rounds, at an ALU-pipe floor of **8.84 ms of 28.37** -- so the ALU-bound half accounts completely as hash 8.84, every other ALU instruction **1.98**, warp supply and non-ALU issue 4.18, and the whole instruction-count family is bounded by that 1.98 ms with no named region reaching 0.3. Taken by joining `nvdisasm -g` line info to `ncu` per-instruction counts at 100 % opcode agreement, positive-controlled at both ends (entry predicted 2.41 ms against 2.48; `apply_mix` at 23.3 % of round 3's instructions costs +0.024 ms to delete entirely). With it, **round 2's warp ceiling is shown unreachable by geometry** -- sm_89 permits 66.7 B of shared per thread at 48 warps and round 2 runs at 92.3, an invariant under scaling the group and the block together, so shared bytes per staged element and warp supply are one currency and not two; and **abandoning a dud solve early is bounded at 0.10 ms** -- the population is pinned at 2^25 in every round with drops zero, so no round before the terminal one carries a statistic correlated with yield, and the ceiling is the terminal round's own 0.77 ms on the 13.5 % of solves that yield nothing, which closes axis 2 on all three of the correlation theorem's channels; and **the w0 checkpoint reaches OpenCL for -0.745 ms (-2.29 %)** -- a revival rather than a new idea, since the pack that pays for it shipped there the same week, and it lands at CUDA's own -0.76 ms / -2.4 % on the identical change. Confirmed twice: twelve interleaved arms with non-overlapping ranges, and a one-sitting re-measurement of the ladder where the eleven byte-identical rungs size the session offset at +1.26 % and correct the two changed ones to -2.19 / -2.46 %; **and it reaches that backend's quad and octo records for -6.25 % and -8.56 %**, the two biggest instances of the lever anywhere -- CUDA got -1.84 and -4.39 ms from the identical changes, and a deeper re-derivation on a slower backend is where a checkpoint that deletes every `apply_mix` in it should pay most. Confirmed again off the four rungs that did not change (session offset -0.73 %, corrected -6.63 % and -8.72 %), controlled at both boundaries, and it moves nine of the fourteen ladder rungs -- including the OpenCL floor, 68.5 -> 63.0 ms -- so **the octo rungs now cost this backend what they cost CUDA, 2.00x the top rung against 1.98x**, where the gap was 1.9x against 1.8x while only CUDA had the record |
| [Measured results, 2026-08-18](#speculative-entry-under-a-cap-both-crossovers-measured-and-the-gate-moves-to-them) | **the speculative-entry gate moves to its measured crossovers and ships** — nospec −1.75 % at 140/160 W, −0.99 % at 180 and −0.22 % at 210, against spec +0.79 % at 240 and +1.03 % at stock, so the stock-memory gate goes 130 → 220 W; **the crossover is a function of the memory clock, not the cap alone** — on a held 5001 rung the freed watts un-starve the core and spec wins again at 160 W (+1.81 %) while losing at 140 (−0.71 %), so a down-rung observed at startup carries its own 150 W crossover, and match-first decouples onto the 130 W band it was measured in; **a third swept-column session offset, the first reading HIGH** — a 15-cap sweep read +0.3 % (285 W) rising to +5.2 % (100 W) over the published column on binaries a three-cap bracket shows performance-identical (±0.2 %, |t| < 1.5), so the cap-dependent cross-session term reaches ~5 % at the low caps, a head-to-head column assembled from two sessions inherits it, and the chain-walk unroll is null under caps too — the 2–4.7× cap multiplier prices *dynamic* instructions, never static footprint; **the head-to-head re-baselined in one session, both configurations** — stock-config: the worst point is 160 W at −9.1 % pre-gate (−7.5 % once the gate shipped), 100 W is a +3.2 % lead, both miners hold 10251 MHz memory while the reference runs its core at roughly half our clock and still wins the band, which names the deficit a power-split/issue-count mismatch; best-config (both miners on the 5001 rung where it pays): the reference leads 100–110 W by 2–3 %, MXBM +8.5 % at 120 W and +3.5 % at 140, level at 160–175, the rung plateau lifted ~47 → 50.7 sol/s on the current kernel; **the w0 checkpoint transfers to the rung** — −2.1 % / t = 80 at 160 W with the rung held, the deleted rebuild returned as +66 MHz of core, no cap amplification (the rung's clock is already near stock) — and its `MXBM_PAIR_W0=0` arm had ROTTED since the round-3 replay change (r1 at IMPB=0 wrote reference rows the impb rungs never allocate; fixed with r1's IMPB unconditional, =1 SASS-identical); **the 140 W census** — r1+r2 = 47.9 % of a 51.2 ms solve and entry stretches most under the cap (×1.86 vs stock, r2 ×1.54, whole solve ×1.45), so the band levers are the compute-heavy front half; **the harness floor under a cap** — a 140 W null reads +0.000 % (no slot bias) but per-arm sd is 0.14–0.21 % against 0.045 % stock (0.04–0.08 % on the rung), so a 24-run cap A/B resolves ~0.07 % se; **the OpenCL backend's first cap curve** — its CUDA deficit widens as the cap drops (~1.09× stock, ~1.15× at 140 W, ~1.29× at 100 W: the cap multiplier pricing its larger instruction count), efficiency floor 3.65 J/sol at 210 W against CUDA's 3.23; **and `--tune` measured every cap at the startup construction policy** — the spec gate and the geometry preference are ctor-time, so the sweep now rebuilds the solver at each point (`TuneConfig::remake`), and a tune stored before the gate is stale |
| [Measured results, 2026-08-19](#round-3s-write-sectors-re-priced-under-the-cap) | **round 3's write sectors re-priced under the cap — 4 % of the round at stock is 25 % at 140 W** — the dead-word closure's own ablation (emit 64 → 16 B, footprint and occupancy preserved, 4 → 1 STG.E.128 in SASS) re-run at 285/140/100 W in one session: −3.6 % of r3's replay marginal at stock (reproducing the closure), **−24.8 % at 140 W and −23.3 % at 100** — a **×6.8 cap multiplier**, the largest measured on any lever, with the implied rate moving from an impossible ~3000 GB/s at stock to ~950 GB/s under the cap: capped, the scattered stores are priced near their real sector traffic; **the emit-sector census behind it** — ~11.3 GB of L1→L2 store sectors per solve for ~5.4 GB of payload, r2 and r3 each 4 sectors/element (four scattered ST.128 at the 16 B-store floor) carrying 73 % of the total, L2 merging 5/14 %; what it does NOT reopen — the quad record's bytes-for-rebuilds trade stays closed under caps (the rebuild's cycles starve too) and chunk-staged emits are geometrically infeasible at 2^25 (a ~20 KB staged tile wants ≤ ~400 buckets, an in-shared match ≥ ~25000, and the bridging pass costs 4 sector-units against the scatter's 3); what it does not reopen either — the **48 B r3→r4 record stays dead by bits**: with `cgi` retired (its three tie-break readers are slot-substitutable, the quad16/ow0 precedent, worth only the `gi_alloc` atomic) the minimum content is 296 work + 25 lead + 64 contrib + 17 replay bucket hint = **401 bits in 384**, the overage exactly the hint `replay_r3` reads from word 5 — the only r3-level parent pointer since the reference rows went, non-derivable past the combine's `>> 24`, and every relocation re-adds a scattered store or unaligns the record; and what it prices UP — the **warp-cooperative sector-paired emit** for r2 + r3: active emitter lanes pair via ballot + shuffle so each ST.128 writes a 16 B half of one record's 32 B sector and the per-instruction coalescer merges the pair, 4 → 2 transactions/element on the rounds carrying 73 % of the store bill with records, recovery and output bytes untouched — gross ≈ −2.2 ms at 140 W for r3 alone against a ~16-instruction/element shuffle tax, not covered by the read-side cooperative-staging null (its own scope line: the write-side currency correction does not transfer to reads); **and the paired emit then measured NULL at the solve level** — built and fully gated (KAT 3/3 × 15, drops 0, occupancy held, store sectors 4.00 → 2.11 per element on both rounds), it moves the 140 W solve only **+0.50 %** and costs −1.41 % at stock, because **DRAM bytes written are identical in both arms (10.605 GB)**: a lane's two halves of one sector arrive a few cycles apart and the L2 was already merging them before writeback, so the halved L1→L2 transaction count never reaches the memory system's bill — **the write-side currency under a cap is DRAM-reaching sectors, not store transactions**, which re-prices P1a's own exchange rate (its ablation narrows the payload and so halves r3's DRAM-dirtied sectors; the −3.367 ms rides that); **the DRAM-side census then closes sector efficiency at every record width** — pivoted from the same profile, every emit reaches DRAM at payload (entry 7.6 B/element, r1 15.6, r2/r3 63.4; 10.68 GB moved/solve) because the ~8 MB write frontier merges adjacent slots in L2 before writeback; **the access-pattern census under caps** — streams are cap-immune (596–639 GB/s down to 120 W; a full-rate stream draws only ~139 W) while the per-element-atomic scatter floor is ~250–265 GB/s at every power and every bucket count 2^6–2^16, so the 2.3× stream-vs-scatter asymmetry is stock-real and only cap-widened — with the census this makes r2 and r3 **scatter-rate-bound at stock** (round times 7.94/8.51 ms against 8.4 ms of floor-priced scattered writes each), bounding the pattern tax at ~8 ms of a stock solve and ~12.6 ms of a 140 W solve; **the reference's band lead decomposed** — both currencies at once, our stretched rebuild hash (~12 ms at 140 W) plus our scatter-floor tax (~12.6) against its +11.7 ms streamed byte surplus reproduces the measured −6.5 % to model precision, the h-boundary trade stays closed even at stream rates, and the surviving axis-4 shape is the **coarse-chunked emit + consumer-side fine split in persistent L2** (fine buckets exist only in L2, never in DRAM — the streamstore geometric kill's second constraint dissolves; the reference's 471 GB/s one-pass rounds are the existence proof); and **the sort path is not a band alternative** — its ×1.36 cap stretch against the row-bucket's ×1.99 confirms the stream mechanism from 5.9× too far behind; **the escape mechanism itself is then confirmed** — chunked appends (one atomic per chunk, warp-contiguous writes) measure **604–620 GB/s at 285 W and 140 W alike**, full stream rate, down to 256 B chunks and at 2^12 buckets, so the floor is a property of per-element allocation alone and the open cost is the classification that fills the chunks; **the round-pair probe then split the axis-4 verdict** — the chunk-staged emit is confirmed in a live kernel with classification priced (462–511 GB/s at stock against the 242–256 floor, 301–435 at 140 W against 178–197, staging tax ~0.6–1 ms per 2.15 GB layer, the win growing under the cap) **but the consumer-side fine split re-pays everything the emit saves and more** (the mock pair regresses 16–49 % at stock, 16–46 % at 140 W, fingerprint-gated): the scatter floor **follows the access shape into L2** — per-element scattered 64 B stores into an L2-resident tile run ~620 GB/s independent and ~360 GB/s under the allocation dependence, the persisting-L2 window buys nothing while its device-wide carve-out taxes everything else ~5 %, and the split-phase barrier forfeits the fused consumer's read/compute/child overlap (the tile-fed match phase alone ≥ the whole shipping consumer at 140 W) — the serial-phase family is bounded out with reorganization free (+9 % stock / +1 % at 140 W), and **the two-level chunk-staged residue then measured strictly worse than the one-level it was built to beat** (13.3 vs 9.0 ms — the L2-resident tile forces 256 serialized coarse-phase steps too shallow to hide latency, and three attacks on the 9.0 all failed), **closing the whole axis at every probed shape**: partner adjacency costs one scattered touch of the full state per round, the shipping scatter emit at the floor is the cheapest measured way to pay it, and it comes back only with a sub-9.0-ms split per layer or a match that consumes coarse buckets directly; **the coarse-direct match then measured out the same day** — an in-L2 chain match per coarse bucket (cooperative per-bucket window, `atomicExch` chain table, snapshot walks, partner gathers at the census L2 rate) never beats the shipping pair (best a dead tie at 140 W / work 0, −7 to −29 % everywhere else, fingerprint-gated), its child emit floored per-element at every destination geometry (2^16 bins: +1.78 GB of partial-sector RMW; ≤2^8: counter serialization; the chunk-staged escape loses more to lockstep than it saves) and no DRAM/L2 overlap materializing (latency-bound at 0.09 eligible warps, occupancy-immune) — **consuming the second reopening condition: axis 4 is closed with none open**, and the streaming pipeline does not get built |
| [Measured results, 2026-09-08](#the-terminal-rounds-time-was-its-block-count-not-its-bytes) | **The terminal round was wave-bound, not byte-bound: 0.80 → 0.52 ms, −0.7 % of the solve** — 131k blocks each paying two or three DRAM latencies in series (one load per pass behind the sub-mask filter and the shared atomic); hoisting a block's loads ahead of the filter is 0.80 → 0.73, and one block per bucket on the (16,1) rungs (staging 832, a 256-entry perfect table, half the grid) is **0.52**, ABBA non-overlapping, 28.8 → 28.6 ms; the octo rung's terminal goes 1.10 → 1.03 at four blocks per SM instead of six; KAT 3/3 × 15, drops 0. The "GPU idle time" closure counted the gaps between kernels and not the latency chains inside a short one |
| [Measured results, 2026-09-08](#the-replays-were-warp-serialised-on-their-hits-not-latency-bound) | **Recovery's replays were warp-serialised on their hits: 0.19 → 0.03 ms, +0.4 % on solves** — each replay ran its pair search as a per-lane loop with the child rebuild inline, so every same-key hit executed hundreds of instructions with the warp masked to one lane (`replay_r3` keeps 8 key bits in a bucket, ~500 candidates each; 9.4 of 32 lanes active); a chain table, a shared candidate list and lockstep processing take `replay_r4` 60 → 10 µs and `replay_r3` 115 → 10.5; ABBA both orderings non-overlapping, KAT 3/3 × 15, drops 0. **The lever stops at round 3**: its staging shows the same 60 % long-scoreboard profile, but hoisting the word-0 loads (with or without an L2 prefetch of the record) measures **0.00 / +0.17 ms** — at 508 GB/s the stalls are bus queueing, not exposed latency |
| [Measured results, 2026-09-08](#the-round-3-record-is-56-b-the-dead-word-goes-and-the-four-16-b-transactions-stay) | **The r3 → r4 record is 56 B, +1.7 % at stock** — the dead sixth work word goes and the replay hint moves into word 4's free top byte and the meta word; the 8 B-off slots take one `ST.64` and three `ST.128` chosen by slot parity, so transactions stay at four per element (seven scalar stores measured +23 % on the round). r3 8.54 → 8.23, r4 6.43 → 6.20, KAT 3/3 × 15 both arms, drops 0. **Under a 140 W cap the rounds gain 1.5 ms and the solve 0.1**: entry/r1/r2 give the time back as clock — a capped solve is energy-bound, so round marginals under a cap overstate and the ×6.8 write-sector multiplier is a round figure, not a solve one |
| [Measured results, 2026-09-08](#the-gi-allocator-goes-where-nothing-indexes-a-gi-09) | **The gi allocator is not issued on the replayed rungs, +0.9 %** — its only reader left was the walk's tiebreak, which now orders by the parent record's slot on both the round and its replay; round 1 4.88 → 4.73 ms (issue-bound: the aggregated atomic's ballot/popc/shuffle were issue slots), round 2 8.06 → 8.00; KAT 3/3 × 15, drops 0. **The entry → r1 word-0 checkpoint in a 16 B record is a null**: r1 −0.37 ms, the hosting round 4 +0.53 from 0.27 GB more scattered writes, net −0.6 % |
| [Measured results, 2026-09-08](#turings-shared-memory-budget-and-the-per-card-staging-cap) | **Turing's 64 KB of shared memory caps every round at two resident blocks per SM, and a per-card staging cap of 280 lifts rounds 1, 2 and 4 to three** — the sm_75 cubin read with `cuobjdump`: at kFCap 320 every round kernel sits at 22.3–26.9 KB, over the 21,845 B a third block needs, and ptxas fills the registers to 111–128 because nothing better is reachable; at 280, rounds 2 and 4 fall to 20.7–21.4 KB and ptxas re-fits them at 80 registers, round 1 (already 288) fits, round 3 (leaf staging) does not at any cap the population allows — **it takes its 16-bit word 6 in a u16 plane (the NARROW6 flag, now per card) at a cap of 272**, 20.7–21.4 KB and three blocks on sm_75, worth **−8.3 % of round 3** where the round is capped at two (pad emulation on the reference card, ABBA non-overlapping); the trade priced on the reference card with `MXBM_SMEM_PAD`: two blocks per SM instead of three or four costs **+16 % on every round** (28.7 → 33.3 ms), so the lever is worth up to ~that on a Turing card; the caps are a per-card choice because on the reference card, at unchanged occupancy, they measure **+0.5 %** (round 4 +0.08 ms, round 3 +0.05); gated 3/3 × 15 goldens in both arms, 0 drops over 8,364 solves. **The per-stage table shipped with it** (`MXBM_ROUND_STATS`, and always in `--report`) — device timestamps at every stage boundary, free at this rig's resolution, r2 7.93 / r3 8.51 ms reproducing the census to the hundredth |
| [Measured results, 2026-09-08](#the-entry-pass-beside-rounds-3-and-4-on-a-stream-the-block-scheduler-prefers) | **The next solve's entry pass runs beside rounds 3 and 4 on a stream at the device's greatest priority, +2.0 % at stock** — the block scheduler dispatches its blocks into the warp and register room round 3's shared-memory cap leaves idle, displacing no round block (the two closures of the overlap family were dispatch order and displacement, not SM room). The co-runner's cost to round 3 goes as the square of its issue duty (compute-only: +2.0 ms packed into 3 ms, +0.65 spread over 8, 0 over 16), so the pass ships throttled: one dependent SipHash chain per lane, four warps per SM, 1 µs of sleep per element. Store hints move ≤ 0.1 ms; `evict_first` on every store costs 4.7. Loses under every cap (energy-bound; 140 W 51.7 vs 50.5 standalone), crossover at the existing 220 W gate. Paired ABBA 27.59 → 27.05 ms |
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

Path selection is by device memory (`rowbucket_viable` in `src/gpu/gpu_solver.cpp`): on
OPENCL the row-bucket path needs 6.84 GiB with a 3.27 GiB largest single allocation at its
finest rung and **1.90 at its coarsest**, so only a card under a 3 GB one falls back to the
sort path. CUDA runs the same ladder from 6.17 GiB down to the same 1.90, and has no sort
path to fall back to. `MXBM_ROWBUCKET=1` / `MXBM_NO_ROWBUCKET=1` override.
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
record *replaces* the payload instead of adding to it (see
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

## Measuring: the locked-clock pin and what this rig can resolve
<details>
<summary>Details</summary>

The protocol behind every published MXBM figure. It belongs to the reference rig,
not to a contributor's card — [benchmarking.md](benchmarking.md) is what a
contributor runs.

### The named reference: `LGC-2600`

"Benchmark it" means exactly this, so two people saying it run the same thing:

```sh
sudo -v && LGC=2600 LMC=10251 benchmarks/headline.sh
```

`headline.sh` refuses to run if another process holds >256 MiB of VRAM — **checked between
every arm, not just before the first** — discards a 240 s warmup, repeats the run, and
records NVML telemetry per run. The telemetry is what makes a failure diagnosable: the
original measurement recorded a number and nothing else, so when it failed to reproduce
there was nothing to diff.

Reference values (RTX 4070 Ti SUPER, driver 610.43.03, re-measured 2026-09-09, six 120 s
runs, **compute GPU headless**):

| quantity | reference | gate |
|---|---|---|
| ms/solve median | **27.00** (spread 0.0 %) | a build change is real past ±0.5 % |
| sol/s | **74.40** | derived; quoted because the pin cites it |
| SM / mem clock | 2610 / 10251 MHz | must match, or it is a different V/f point |
| board draw | **276.4 W** | context, not a gate — 265–281 W across ten sessions; near the limit `sw_power_cap` is intermittently active (1–10 % of samples) at identical clocks and work |
| `clocks_event_reasons` | none, or `sw_power_cap` at any fraction | any **other** flag voids the run. The fraction is not predictive and is not a gate |
| display on the compute GPU | **no** | a compositor costs **0.20 ms and ~6 W** |

That last row is a condition of the number, and no kind of decoration — it is worth 0.6 %, more than the ±0.5 %
gate, and it is not a standing rig property: the compositor pin (`~/.config/uwsm/env`)
follows the HDMI cable at every login. Verify per session — the miner prints `reserving
64 MB (headless)` or `reserving 256 MB (display attached)`.

**Three rules.** Builds are compared at this pin, never on the stock number (stock carries
the ~2.5 % cross-session band and gates nothing). It is only a pin while the clocks read
2610/10251 — a run that drifted off is discarded; averaging it in would be worse. Record J/solution
alongside ms/solve. Re-pin after any kernel-shipping day, or any change to what else the
card is doing; a repeat that moves past ±0.5 % *without* a shipped change reopens the
V/f-state investigation.

**Rows name what shipped and link the ledger, never a commit** — rewriting a message
re-hashes the commit and the citation dangles, which has already happened here.

<details>
<summary>Pin lineage</summary>

| pin | ms/solve | draw | what moved |
|---|---|---|---|
| 2026-09-09 | **27.00** | 276.4 W | the next solve's entry pass beside rounds 3 and 4 on a priority stream ([ledger](#the-entry-pass-beside-rounds-3-and-4-on-a-stream-the-block-scheduler-prefers)) — **−0.80 ms, 72.30 → 74.40 sol/s**, against −0.55 on the 60 s ABBA and +3.0 % on the pin's solves (4314 → 4445 per 120 s). Six runs at 0.0 % spread, p95 27.0 |
| 2026-09-08 | **27.80** | 276.8 W | the gi allocator not issued on the replayed rungs ([ledger](#the-gi-allocator-goes-where-nothing-indexes-a-gi-09)) — **−0.20 ms, 71.90 → 72.30 sol/s**, against −0.25 on the rounds' timers and +0.9 % on solves ABBA. Six runs at 0.0 % spread |
| 2026-09-08 | **28.00** | 280.9 W | the round-3 record at 56 B ([ledger](#the-round-3-record-is-56-b-the-dead-word-goes-and-the-four-16-b-transactions-stay)) — **−0.60 ms, 70.30 → 71.90 sol/s**, against −0.55 on the rounds' timers and +1.7 % on solves ABBA. Six runs at 0.0 % spread; the draw is up 13 W and the board limit is touched in 2–10 % of samples |
| 2026-09-08 | **28.60** | 268.1 W | recovery's replays by chain table, candidates in lockstep ([ledger](#the-replays-were-warp-serialised-on-their-hits-not-latency-bound)) — **−0.20 ms, 69.90 → 70.30 sol/s**, against −0.16 measured on the stage's own timer and +0.4 % on solves ABBA. Six runs at 0.0 % spread |
| 2026-09-08 | **28.80** | 273.1 W | the terminal round at one block per bucket with its loads in flight together ([ledger](#the-terminal-rounds-time-was-its-block-count-not-its-bytes)) — **−0.30 ms, 69.20 → 69.90 sol/s**, against −0.28 measured ABBA on the round's own timer. The same day's Turing staging caps change no kernel the reference card runs. Six runs at 0.0 % spread |
| 2026-08-18 | **29.10** | 265.0 W | nothing at stock — the [speculative-entry gate](#speculative-entry-under-a-cap-both-crossovers-measured-and-the-gate-moves-to-them) changes behavior only below 220 W, and the devfee work touches no kernel. Six runs, **0.0 % spread on both columns**, reproducing the 2026-08-16 pin to the digit; draw 10.5 W below it at identical clocks and work, a rig-day difference per the 08-14 precedent |
| 2026-08-16 | **29.10** | 275.5 W | the packed `tab` word ([ledger](#the-census-and-the-chain-share-one-tab-word-and-a-barrier-goes-with-it)) — **69.00 → 69.20 sol/s**. −0.060 ms is under the print resolution; the finer `solves/s` line reads 34.33 → 34.44, i.e. −0.094 ms, agreeing with the interleaved −0.060. First pin with 0.0 % spread on both columns |
| 2026-08-16 | **29.10** | 277.0 W | the block-exit barrier, then singleton-free staging ([barrier](#the-block-exit-barrier-is-removable-and-the-barrier-family-is-over-priced-10x), [singletons](#singleton-free-staging-the-prize-is-085-ms-and-the-prepass-that-finds-it-costs-076)) — **−0.20 ms, 68.60 → 69.00**. Covers two changes; separable only in the interleaved A/Bs (−0.048 and −0.089, summing to −0.137) |
| 2026-08-15 | **29.30** | 271.1 W | the reference rows came out ([ledger](#the-back-reference-rows-are-gone-recovery-replays-instead-203-ms-and-688-mib)) — **−2.00 ms, 64.20 → 68.60**, against −2.029 measured ABBA. Verified solutions/solve 2.01, unchanged |
| 2026-08-15 | **31.30** | 271.9 W | the w0-checkpoint pair record ([ledger](#the-w0-checkpoint-pair-record-repriced-by-the-address-bits-076-ms-24-)) — **−0.70 ms, 62.70 → 64.20**, against −0.76 measured ABBA |
| 2026-08-15 | **32.00** | 272.1 W | nothing — a campaign session baseline. Fifth consecutive 32.00 / 62.70, and the first spanning two calendar days, so the pin reproduces *across* sessions |
| 2026-08-14 | **32.00** | 272.9 W | nothing at stock (three back-ref rows retired and `gi` dropped from r2, both octo-rung only). Draw reproduces the previous pin to the decimal — the first cross-session agreement on it |
| 2026-08-14 | **32.00** | 272.9 W | nothing at stock (octo record instantiations, bottom three rungs only) |
| 2026-08-14 | **32.00** | 279.9 W | nothing at stock (reach work: implicit-bits reclaim, overflow-arena rung, availability allowance). `sw_power_cap` 14–18 % of samples against the morning's 0–3 % at identical time — the fraction is a rig-day reading, not a build property |
| 2026-08-14 | **32.00** | 274.1 W | nothing at stock; a re-take under the kernel-shipping-day rule. Draw 7.3 W below the 08-13 pin at identical clocks and work, unexplained, so recorded as a rig-day difference and **not** attributed to the change |
| 2026-08-13 | **32.00** | 281.4 W | the implicit-bits record (−3.2 % in-miner A/B, [campaign](#populations-are-pinned-at-225-and-the-occupancy-tail-prices-a-spill-arena)); draw rose to the board limit |
| 2026-08-04 | 33.30 | 271.1 W | the monitor moved to the iGPU — **attributed to the rig**: the 08-01 build re-run headless reads 33.30 to the digit |
| 2026-08-01 | 33.50 | ~277 W | entry co-scheduling; r2 LD.128 + terminal perfect table. −2.3 % matches ship-time |
| 2026-07-28 | 34.30 | ~262 W | first pin |

The draw column is the cross-check: a busier binary *raises* draw at identical clocks
(08-01, +15 W), a quieter rig *lowers* it (08-04, −6 W).

</details>

---

### How small a difference the rig can resolve

One number was doing three jobs. The ~2.5 % cross-session band is right for a *published
absolute figure* and wrong as the resolution of a *paired* A/B — used as the latter it
retires levers this rig can see. Measured directly:

| protocol | band |
|---|---|
| cross-session absolute | ~2.5 % — still the right caution for a published figure |
| cross-session absolute, under a cap | **cap-dependent, up to ~5 % at 100–120 W** — three instances, both signs; shrinks toward the stock band above ~220 W |
| one 30 s run against another, post-warmup | **0.112 %** (1σ, n = 30) |
| paired ABBA, 12 arms a side | **0.040 %** |
| both arm orderings (48 runs, ~30 min) | resolves **~0.05 %** at t ≈ 4 |
| paired, under a power cap | **~0.07 %** (24 × 60 s at 140 W) — no bias, but per-run scatter runs ~4× stock, so keep the full 24 runs for any sub-0.3 % claim; on the 5001 memory rung the scatter returns to stock levels |

`benchmarks/paired_ab.sh` is the bottom row:

```sh
benchmarks/paired_ab.sh old new                          # ~25 min, ~0.05 %
SECS=90 BLOCKS=4 benchmarks/paired_ab.sh a b             # finer, slower
ENV="MXBM_BB=17 MXBM_SM=0" benchmarks/paired_ab.sh a b   # another rung
sudo -v && CAP=120 benchmarks/paired_ab.sh a b           # under a cap
```

- **Count solves over a fixed window**, not the printed median — that prints to 0.1 ms
  (0.35 %), while ~1045 solves resolves 0.1 %. The count, not `sol/s`: the
  solutions-per-solve multiplier is the algorithm's and only adds variance. The harness is
  *quantization*-limited (sd ~0.45 solves against a 1-solve grid), so resolution improves
  linearly with window length.
- **Warm up ~2 minutes.** Drift over 15 minutes is **+0.205 %** (`corr(ms, temp) = +0.575`),
  nearly all in the first four runs: the count falls 1050 → 1045, then holds 1045 ± 1.
- **Run both arm orderings.** ABBA cancels linear drift but not the A slot itself, which
  reads **+0.024 %** high. `effect = (d1 − d2)/2`, `bias = (d1 + d2)/2`; a single ABBA
  measures their sum.
- **Do not clock-normalise.** Tried, and worse: the band goes 0.112 % → 0.298 %. Once warm
  the clock spans **0.5 %**, where such a correction assumes ±10 %.

**Two builds carry no layout noise.** Rebuilding identical source is not byte-identical —
48 bytes differ across 23.7 MB — but those are the ELF build-id note and the per-cubin
UUIDs; `cuobjdump --dump-sass` matches to the digit over 265,242 lines. So the whole
difference between two builds is the change.

---

</details>

## Instruments: diagnostic and ablation flags
<details>
<summary>Details</summary>

Diagnostic environment variables: `MXBM_ROWBUCKET` / `MXBM_NO_ROWBUCKET`,
`MXBM_SORT_PROFILE`,
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
refining passes, end-of-sweep drift gauge, result stored per card for `--pl auto` —
usable as a research instrument wherever a drift-gauged power curve is the question.
The first full three-pass run (2026-07-31, reference card) reproduced the dedicated
sweeps: recommendation 231 W (refined down from the coarse 248), rung crossover interpolated at
~171 W vs the sweeps' ~173, drift +0.5 % where un-gauged sessions had shown ~2 %/arm.
One grid miss surfaced and was closed the same day: pass 3 reuses the coarse caps,
whose 37 W spacing skips ~160 W — so the verdict named 211 W stock as best efficiency
(0.256 sol/s/W) and could not see the true record between its rung points (160 W +
rung, 0.264). The cap and rung-band verdicts, the ones `--pl auto` acts on, were
unaffected. Fix: a fourth pass that aims `tune_refine_caps` (budget 4, so ~15 W
steps) at the best-efficiency point on its own memory clock; on this card's series
that grid is 115/130/145/160 — the record cap lands on it exactly
(test_tune pins the grid). Re-running `--tune` regenerates the store with it.
Energy has been measured directly since 2026-07-31, with nothing integrated: the card's millijoule counter
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
choice is never stepped down under the user), `MXBM_QUAD=0|1` and `MXBM_ARENA=0|1` pin the
other two axes of the CUDA ladder — the quad record and the dense-cap/overflow-pool rung —
and `MXBM_NO_IMPB=1` turns the implicit-bits record off in the kernels *and* in the
allocation, so an A/B with it never sizes against a record it is not using.
`MXBM_DROP_STATS=1` prints the four drop counters and the arena's run-long spill total
after every solve — one counter per capacity, **entry / stage / out / walk**: on an arena
arm the gate is `drops == 0` **and** `spills != 0`, since a pool that never filled means
the overflow path never ran and the arm measured nothing.
`MXBM_ROUND_STATS=1` prints per-stage GPU medians (entry, rounds 1–4, terminal, recovery)
after a `--benchmark`, from device timestamps at every stage boundary; the same table is
always in a `--report`, so a contributed card names the stage it stretches. Off, the
timestamps are not recorded; on, they cost nothing this rig can resolve (28.7 ms both
arms), but a timing event is also a dispatch fence at its boundary, so the instrument can
hide a stream-ordering tail the plain run has ([the entry pass beside rounds 3 and
4](#the-entry-pass-beside-rounds-3-and-4-on-a-stream-the-block-scheduler-prefers)) —
check a change's solves/s with it off as well. `MXBM_SMEM_PAD=<bytes>` adds that much unused dynamic shared memory to every
round launch: resident blocks per SM fall and nothing else moves, so it prices occupancy
on this card at another card's shared-memory budget (12288 reproduces Turing's two
blocks per SM here; above 48 KB per block the launch fails). `MXBM_FCAP_SMALL=0|1`
forces the 64 KB-card staging arm (280 on rounds 2 and 4, 272 with the narrow word-6
plane on round 3) off or on regardless of the device's budget.
`MXBM_CAP_SIGMA=S` forces the bucket reservation's tail bound for every caller at once
(footprint arithmetic and allocation alike, so the two cannot disagree); negative values
undershoot the mean and are the positive control that makes each drop channel fire.
`MXBM_OCC` prints the worst bucket occupancy
in units of the reservation's sigma, `MXBM_ROUND_REPS="R:N"` and `MXBM_ENTRY_REPS=N` replay
one stage in place for per-round attribution. Build flags: `-DMXBM_STCS=1` (streaming
stores), `-DMXBM_WG=N`,
`-DMXBM_FCAP=N` (group cap → occupancy), `-DMXBM_R2_FULL=1`.
`-DMXBM_SOLO=0|1|2` splits the singleton filter into its two halves — 0 off, 1 the prepass
alone, 2 the shipping pair — so the cost and the prize are separable in one ABBA;
`-DMXBM_SOLO_R` is the round mask it runs on (1 = r1, 2 = r2, 4 = r3, 8 = r4 and the reach
rungs, default 3). Its positive control is a `-DMXBM_SPILL=0 -DMXBM_FCAP=64` build, where
`drops[1]` becomes the staged count less a constant and the two arms differ by exactly the
elements refused. `-DMXBM_COOP=1` puts four consecutive lanes on one of round 3's 64 B
records; it is a measured loss, kept because it is the one build that provably halves that
round's read sectors (6.03 → 4.07 per element) and so bounds any claim made about them.
`-DMXBM_PAIRED_EMIT=1` is its write-side counterpart: r2/r3's active emitter lanes pair
over ballot + shuffle so each ST.128 fills half of one record's 32 B sector — store
transactions 4.00 → 2.11 per element on both rounds with DRAM bytes identical, a
[measured null at the solve level](#the-sector-paired-emit-the-transactions-halve-the-dram-bytes-do-not-and-the-solve-does-not-move),
kept as the arm that prices a transaction count apart from the DRAM traffic behind it.

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
co-scheduling for A/Bs — on **either** backend since the OpenCL port
(2026-08-02), where `MXBM_CO_STRIDE=N` also sweeps the displacement ratio
(default 3; measured null across 2–6 on Ada, kept as a per-card lever); on CUDA
`MXBM_SPEC_COBLOCKS=1` selects the co-blocks form inside round 4's launch instead of
the default [priority-stream pass beside rounds 3 and 4](#the-entry-pass-beside-rounds-3-and-4-on-a-stream-the-block-scheduler-prefers). `-DMXBM_MB_SEED=N` / `-DMXBM_MB_RD2=N` force a register budget
(`__launch_bounds__` minBlocks) on r1 / r2 — measured null, shared memory caps first.
`-DMXBM_PAIR128=0` restores r2's two scalar pair-record loads
([shipped on 2026-07-31](#two-below-the-floor-levers-clear-noise-on-cuda-r2s-pair-record-in-one-ld128-and-the-terminal-round-joins-the-perfect-table-022-ms)).
`MXBM_VERIFY_STATS=1` classifies every CPU-verify outcome by reject reason on either
backend and prints tallies at exit
([the found-vs-verified section](#the-found-vs-verified-gap-is-gone-0-of-3935-candidates-rejected)).

**OpenCL row-bucket build flags**, through `MXBM_CL_OPTS`: `-DLDS_PW0=0` turns the
[w0-checkpoint pair record](#the-w0-checkpoint-reaches-opencl-0745-ms-229--the-same-lever-at-the-same-size)
off on **both** sides of the r1/r2 boundary at once — one condition gates round 1's emit and
round 2's staging and expand, so a writer and a reader cannot end up on different layouts —
and `gpu_solver_nopw0` gates that arm. `-DLDS_PW0_BREAK=1` is its positive control: it
poisons the stored checkpoint with an **element-dependent** value at the emit, taking the
goldens 3/3 → 0/3. A constant would be invisible, because it cancels through `combine`'s
pairwise XOR.

`-DLDS_QW0=0` and `-DLDS_OW0=0` are the same pair of switches for the
[quad and octo records](#the-w0-checkpoint-reaches-opencls-quad-and-octo-records), with
`-DLDS_QW0_BREAK=1` / `-DLDS_OW0_BREAK=1` as their controls and `gpu_solver_noqw0` /
`gpu_solver_noow0` as the gates. `LDS_OW0` reaches `recover` too, which reads round 3's
record directly on an octo build; that is why the recovery program is built with
`MXBM_CL_OPTS` and not with `-DLDS_OCTO=1` alone.

**CUDA attribution builds** — `-DMXBM_POISON_W=N` poisons work word *N−1* of the r3→r4
record at round 3's emit and reads the answer off the KAT, which is how
[the dead word](#round-3-stores-a-work-word-round-4-never-reads--worth-268-mb-and-nothing-in-time)
was established; N is 1-based, so the dead word is N=6 and N=4/5 are its positive controls.
`-DMXBM_ABL_EMIT=R` (narrow round *R*'s payload to 16 B),
`-DMXBM_ABL_DERIVE=1|2|3` (round 1's seed / round 2's rebuild), `-DMXBM_ABL_MIX=R` (skip
`apply_mix`), `-DMXBM_CO_NOSCATTER=1` (co-tenant computes but stores nothing, and
`--fuse` then keeps launching the real entry so the co-tenant is purely additive).
`MXBM_POP=dir` (cuda/pipeline.cu) dumps every round's unclamped `counts[]` as raw
u32[nb] files per solve — per-bucket occupancy histograms and true per-round
populations; synchronous copies, so never in a timing run.

The two harnesses those flags are usually driven through: `./cuda/pipeline N` times the
solver pipeline over N solves, which is the fast A/B for a kernel change and is blind to
everything above the solver; `CLOCKS=none TARGET=miner ./cuda/profile.sh` collects the
Nsight counters (DRAM bytes, stall reasons, sectors). Nsight needs GPU counter access,
admin-restricted by default and lifted permanently on this rig by
`NVreg_RestrictProfilingToAdminUsers=0` in `/etc/modprobe.d/nvidia-profiling.conf` —
check with `grep RmProfilingAdminOnly /proc/driver/nvidia/params` (0 = unrestricted).
Installing that file takes one reboot; running the profiler afterwards takes neither
root nor a reboot.

`-DMXBM_PAIR_W0=0|1` ships as a build knob, and is not an attribution instrument — the
[w0-checkpoint pair record](#the-w0-checkpoint-pair-record-repriced-by-the-address-bits-076-ms-24-),
on by default. Setting it on the CUDA flags alone is a valid A/B build: the kernel set
and the occupancy contract are identical under both values, and the macro keys only
the record format
([the transfer entry](#the-w0-checkpoint-transfers-to-the-5001-rung--and-its-off-arm-had-rotted)
has the mechanism).

**Which binary is the CUDA gate.** `bench_rounds` and `tests/test_gpu_solver` drive the
**OpenCL** backend — `GpuSolver` holds a `cl_runtime.h` `Runtime` — so neither can see a
CUDA-only change, and both return a passing gate and an unmoved number regardless. Use
`tests/test_cuda_solver` (3/3 goldens across 15 geometries) and `mxbm --benchmark
BEAM-III`; derive ms/solve from the `solves/s` figure, which carries two more digits than
the median line.

**Results are intentionally wrong** and the KAT gate is bypassed for these builds only. Two rules they were designed around, both learned the hard way here:
substituted work must keep the *global* access footprint and the bucket distribution
(check `drops == 0` and `MXBM_OCC`), and removing a store must not let DCE remove the
compute feeding it — hence `ABL_EMIT` narrows the emit instead of skipping it, with rounds 1 and 4 as
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
bucket, so what it measures is pathological contention and not the removed work.

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
the work words together instead of adding to them:

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
literals; that is the whole default path.

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
(r1+r2 = 22.7 vs 27.8). The two differ because round 2 costs the **same** 15.2 ms with or
without its rebuild, so its 2-seed recompute is entirely hidden and the whole win is round
1's narrower emit. Round 3's 4-seed rebuild was never hidden.

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
`leftContrib` and not leaves, so its `lead` is not duplicated anywhere.*

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

Same redundancy class as [the round-3 `lead` field](#the-redundant-lead-field) — an
element's lead duplicating its own leaf 0. Two independent instances of one mistake:
**a field derived from another field in the same record will not announce itself.**

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
16 B aligned for *every* slot, where before it was every other one — and the 9th word moves to its own
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

**Do not read this as a throughput win.** −0.6 % is at the edge of what the whole-solve
harness resolves: `mxbm --benchmark` measured 35.1 ms/solve both before and after. The A/B
is believable only because it ran as four alternating pairs of 700 solves and the new
build won all four, and because the round-level figures explain it. **The reason to keep
the change is the 0.36 GiB**, which is what lowers the CUDA backend's VRAM threshold.

Round-level figures are from `MXBM_ROUND_REPS=R:9` (`benchmarks/stage_power.sh`), which
replays one round nine times inside the solve that produced its input and so measures a
0.4 ms change as a 3.5 ms one; both A/Bs reproduced to ±0.05 ms.

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

### The terminal round's sm = 0 fast path costs 0.022 % on the rung that ships
<details>
<summary>Details</summary>

At sm = 0 the terminal round's sub-mask filter admits every element, so the compaction
atomic has nothing to compact and the staging index is already dense. Taking that path
costs nothing to write and it is correct — but `submask_bits` is a **kernel argument**,
not a template parameter, so the test is a runtime predicate, and the shipping rung is
non-arena **(16,1)** where it is false. The effect is to put the filter's `continue` under
a branch whose taken side never executes, making a uniform loop divergent for a path the
card never uses. Confirmed against the ladder: default 28.6 ms, `MXBM_ARENA=1` 29.6 ms.

Measured **+0.0223 % (t = 2.05)** in a position-balanced three-way A/B/C — block order
`A B C C B A` so each arm averages position 3.5 — at 90 s per window, n = 10 per arm. The
same run prices the chain-walk unroll beside it at **−0.0064 % (t = −0.67)**, i.e. free, so
the whole of the pair's regression is this one hunk. Reverted; the unroll stays.

**And it buys nothing on the rung it was written for.** Measured directly at
`MXBM_BB=17 MXBM_SM=0`, where the fast path *is* taken: **−0.0090 % (t = −0.61)**, n = 6/7
at 60 s, both arms at 32.6 ms and 2739–2741 MHz. Deleting 33.5 M single-address shared
atomics is not worth a tenth of the instrument floor. So this was never a stock-versus-(17,0)
trade to be recovered by templating `noMask` — it is a cost at stock and a null everywhere
else, and the 20 lines of template boilerplate would protect nothing.

Two things worth keeping. The **static instruction count predicted the wrong sign** — it
sees one extra uniform branch over ~33.5 M elements and misses the divergence — which is
the general caution for pricing a control-flow change by counting. And **the atomic was
never the cost it looked like**: a single-address shared `atomicAdd` taken by every lane
of a block is aggregated by the hardware, so 33.5 M of them price at zero here. An
instruction count is not a serialization count.

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

Identical (lo/hi = 0.98×). The scatter is **random-access-bound**, and latency is not what binds it, at
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
gain was bought with drop headroom; nothing here was free.

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
different and more final reason.** *(2026-07-26.)*

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

**Why it generalises: "the SM is 79 % idle" does not mean 79 % of its issue slots are
available to someone else's work.** r3's warps are idle because
they are *stalled on memory*, and a stalled warp still holds its warp slot. Putting
entry's work in those same warps does not interleave with the stall — it runs *after*
it, serially, in the same warp. The only way to fill a memory stall is **more resident
warps**, and r3 has 3 blocks × 8 warps = 24 of the SM's 48 because
[shared memory caps it at 3 blocks](#bytes-are-nearly-free-per-element-work-is-not).
Entry standalone gets 6 blocks/SM; hosting it halves its occupancy, which is exactly
the ~2.6× it slows down by.

That closes the whole family, and not merely this instance. Independent work added to a round
either goes **in the existing warps** (serialises, measured above) or **needs new
warps** — and a separate co-tenant block in the same launch must still reserve the
round's 32.8 KB of shared memory, so it displaces a round block one for one. There is
no arrangement of the same launch that adds warps without taking them away.

**So the 1.71× roofline is unreachable by any mechanism now tried:** not streams
(grid depth), not one fused launch (warp slots). What it would actually take is a
round kernel that needs materially less shared memory per block, which is the same
constraint that blocks occupancy, and `kFCap` cannot shrink — 320 already drops 180
elements. The *memory* objection to two resident pipelines has since lifted (13.8 GiB at
geometry (15,2), against the 48 MiB shortfall at (16,1) that retired it), so if the
shared-memory constraint is ever broken that idea is affordable again.

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

*(Measured 2026-07-26, at stock. This is the most consequential result in this
document and it retires a strategy — at stock. Under a power cap it inverts: the
same marginal bytes bill at up to the full memory-rung bandwidth; see
[the floor re-pricing](#the-free-list-re-priced-at-the-floor-bytes-and-the-mix-bill-once-the-core-slows).)*

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
distribution, and no kind of tuning choice. See
[occupancy is worth ~1.6 ms](#occupancy-is-worth-real-time-and-shared-memory-is-the-only-gate)
for what that cap is costing, which is not nothing.

So the remaining 6–9 ms per round is the staging read, the chain walk and the *transaction
count* of the scatter — none of which is byte-count-driven, and none of which any
single-kernel lever has moved. That is the same conclusion the overlap work reached from
the other direction. It pointed at co-residency, which has since been closed for
same-mix tenants at every operating point (see the co-residency section below), and
at the switching-height redesign, which was built and killed 2026-08-13 (see the
h=1 section below). No surviving route to this time is known.

</details>

### …but bytes are not free in WATTS, and under a cap watts are clock (−60 MHz)
<details>
<summary>Details</summary>

*(Measured 2026-07-28. This is the counterpart to the section
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
to be beaten, not assumed; it came out at 0–0.5 ms and **0 MHz** every time.

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
> statement of that, with a measured denominator in place of an inferred one.

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

**And it sharpens "bytes are nearly free" instead of reversing it.** The marginal cost of
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

[HW_REQUIREMENTS.md](HW_REQUIREMENTS.md#the-design-target-is-met)
names in-place reuse as the route to the 3 GB target and treats memory efficiency and
energy efficiency as one problem. It is the route to the *footprint* target; the energy
half has to come from narrower records. Since [measured traffic is within 1 % of
compulsory](performance.md#the-memory-traffic-is-compulsory) for the current widths, narrower records
means a structural change and not tuning.

</details>

#### Under a low cap the same bytes cost 5× as much clock
<details>
<summary>Details</summary>

*(Measured 2026-07-28 at a 180 W cap. This is the result that
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
access pattern — so 60 MHz and 210 MHz size the prize; they forecast no
change that could ship. **Nothing measured today priced the cost side**, and until
something does, "narrow the records" is a lead with a known ceiling and an unknown floor.

**What it makes worth re-testing: the round-3 quad record.** An earlier OpenCL build had
round 3 re-derive from a **24 B** record instead of storing 72 B of work state, and it was
retired at *"a genuine −3.6 ms at 86.8 ms is a +2.2 ms loss at 56."* The record it removes is
written by round 2 and read by round 3, ~5.1 GB of the solve's 13.0, so at 24 B it takes
out roughly **26 % of all traffic** — about 1.6× the lever measured above. It is the only
existing implementation of the byte/arithmetic trade, which makes sweeping it the cheapest
way to learn what the cost side is worth.

> **Correction (2026-07-28).** An earlier version of this section, and the commit message
> that shipped it, said that trade was made "on a metric that did not price watts". **That is
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

*(Measured 2026-07-28 under a cap. Direction predicted correctly, magnitude
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
deferred expand instead of landing on the critical path.

One thing fixed on the way, unrelated but sharp: `MXBM_BB=15` alone left `sm` at its
default of 1, putting the geometry off the `bb + sm == 17` line every compile-time kernel
constant assumes — and the OpenCL path, unlike CUDA's, did not check, so it **silently
produced wrong results** (0/3 goldens) instead of refusing. `sm` is now derived from `bb`.

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
and `bh3_siphash24`; being `inline`, it simply receives literals now. The old
partially-constant kernels have since been removed from the tree; git holds them.

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
lesson above instead of extending it.)*

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
the fix; the diagnosis came first.

</details>

#### Index-only round 1 does NOT transfer to the sort path (+27 ms)
<details>
<summary>Details</summary>

*(Built, gated, measured, reverted 2026-07-28; the opt-in has since been removed from the tree.)*

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

It settles both questions above from the compiler, with no timings involved:

| kernel | stack frame | spill | registers |
|---|---|---|---|
| `round_mix` / `_c5` / `_c6` *(runtime `Lmix`)* | **112 B** | 0 | 39 |
| `round_mix_k2…k5` *(constant)* | **0 B** | 0 | 28–32 |
| `round_match` | 0 B | 0 | 40 |
| `round_match_sorted` *(r1/r2)* | 0 B | 0 | **48** |
| `round_match_sorted_7_6` / `_6_5`, `round_match_k3`/`k4` | 0 B | 0 | 40 |

**112 B is exactly `t[8]` (64 B) + `tree[9]` (36 B)** — the diagnosis confirmed at source,
not inferred. And **no match kernel spills at all**, so the mechanism worth 3× in the mix
is simply absent there, which closes that line of enquiry.

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
plays one round higher, and it existed once on the OpenCL path before being retired.

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
is what made its 60 MHz an upper bound and not a forecast. A real narrowing has to put
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

*(Measured 2026-07-28. Packed / quad / packed at each cap, so the two
packed runs bracket the quad one and their spread is the drift the delta must beat.)*

| cap | packed | quad | Δ time | Δ clock | drift |
|---|---|---|---|---|---|
| 285 W | 33.62 ms @ 2692 MHz | 38.47 @ 2670 | +4.85 ms | −22 MHz | 15 MHz |
| 220 W | 35.73 ms @ 2460 MHz | 41.37 @ 2460 | +5.64 ms | ±0 MHz | 0 MHz |
| 180 W | 43.68 ms @ 1882 MHz | 50.01 @ 2010 | +6.34 ms | **+128 MHz** | 15 MHz |

**The byte→watt mechanism is real, and this is the first time it has been shown in a
correctness-preserving change, and not an ablation.** At 180 W the quad record holds
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
away from one instead of toward it.

</details>

#### What it IS: a new bottom rung on the geometry ladder
<details>
<summary>Details</summary>

The footprint half stands on its own, and the ladder is where it pays. Comparing on
**memory budget** instead of on geometry:

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
The allocator-retry path walks the same rung list instead of decrementing `bb`, or it
would stop at the bottom of the packed half and refuse a card the quad rungs would host.
Goldens match `[1,1,1]` under `MXBM_QUAD=1` and the miner reports 1.99 verified
solutions/solve on that path, so the mining route is gated and not just the bench.

**Stated CUDA requirement drops from 8 GB to 6 GB**, and the cards between 6.3 and 7.9 GiB
free get a *faster* rung than before — quad (16,1) at 38.4 ms where the packed ladder gave
them (14,3) at 40.0. Full table in
[HW_REQUIREMENTS.md](HW_REQUIREMENTS.md#the-vram-ladder).

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

**Blocks are what pay here, and warps are not.** Raising `kWG` also raises warps/SM at no shared-memory
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

*(2026-07-26. This closes the "r3 still needs its own cut" lead; it does not deliver it.)*

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
the equation below: it predicts where occupancy is *available*, and says nothing about where it is *worth
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
would become a u64 load). So this is a real lead with a measured prize, and no free win.

The other route attacks the `1 + 8/√mean` term instead of `B`: **spill group overflow to a
second pass instead of dropping it**, letting `kFCap` fall from the tail (352) to near the
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
the launch seeds instead of matching. Interleaved by block index — appended
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
pays max() instead of sum() (22.3 KB, still 4 blocks/SM; registers land on exactly 64,
which is *above* either path's own 47/48 — ptxas spends up to the block limit it is given,
see [co-tenanting rounds 3 and 2](#co-tenanting-rounds-3-and-2-is-gated-by-shared-memory-and-narrow6-opens-the-gate)). `cuda/pipeline --pipe2` orchestrates the two-solve
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

*(Measured 2026-07-31. The below-160 W inversion said the low-power
currency is core cycles and not bytes — this reprices the two instruction-vs-bytes trades
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

**Live at 160 W in the MINER loop, same day: a wash, against the sweep's −1.1 %.** First
capped measurement in the loop users run (`--benchmark`, verify and speculative
co-tenant entry included), where the sweep used the pipeline replay: interleaved
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

**160 W + 5001 is the global efficiency record: 3.79 J/solution** (0.264 sol/s/W,
41.9 sol/s at 159 W measured draw), beating the previous peak of ~3.83 at 200 W on
stock memory. The efficiency-optimal operating point moved, and it moved onto the rung.
(The same operating point reads **3.70 J/solution** on the 2026-08-13/14 kernels — the
point did not move again, the miner got faster at it.)

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
  was memory-interface power; core work was not. Below ~126 W its refund exceeds ours
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
clusters of seven consecutive `LDS.64`. Constraints for a probe: shared memory
is a binding occupancy resource (24.6 of 25.6 KB at 4 blocks), so any layout
that grows the arrays loses the fourth resident block and with it more than
conflicts pay — swizzle in place, never pad. ncu's 16.9 % estimate is an upper
bound. Report: `docs-internal/rootruns/r2-census.ncu-rep`.

> **⚠ Correction: the stride was not the cause, and the localization was.** This
> section attributed the replays to the 7-u64 staged record — "stride 14 words over
> 32 banks → gcd(14, 32) = 2 → exactly the 2-way measured" — and proposed an XOR
> swizzle against it. A calibration against time says a u64 access at that stride
> reads **zero** conflicts; the arithmetic agreed with the measurement by
> coincidence. What the two clusters of seven `LDS.64` actually are is the **walk**
> reading its two chain-selected parents, seven words each, at scattered positions —
> which no stride and no swizzle reaches. See
> [the bank conflicts](#the-shared-bank-conflicts-are-the-chain-walks-and-no-layout-reaches-them).

</details>

### The lwork SoA probe: conflict-free costs more than the conflicts — a measured loss

<details>
<summary>Details</summary>

*(2026-08-01, closing the census's one named lever. `MXBM_LWORK_SOA=1` lays the
staged work array out column-major — word w of element p at `w*FCAP + p` — so
consecutive lanes read consecutive u64, same bytes, same shared budget. KAT green,
r1 even dropped 46→43 registers. The measurement said no, twice — and the
[2026-08-16 conflict count](#the-shared-bank-conflicts-are-the-chain-walks-and-no-layout-reaches-them)
says the premise was wrong as well: the layout it replaces was never the conflicts'
source, and swapping it moves the counter by under 2 %.)*

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

**Addendum 2026-08-02, the flags the sweep did not cover — and this time the
answer needed no stopwatch.** `--extra-device-vectorization`, `-Xptxas
-allow-expensive-optimizations=true` and an explicit `-Xptxas -O3` produce
**byte-identical SASS**: 39,136 instructions across the module either way, and
`diff` over the full disassembly reports one changed line, the `ptxasOptions`
string ptxas records in its own metadata. ptxas already runs at -O3 with
expensive optimizations enabled, and the vectorizer finds nothing the record
loads have not already fused. Codegen equality is a stronger closure than a
timing A/B, because no noise enters it.

The **host** level is not device codegen and was measured: `-DCMAKE_BUILD_TYPE=Release`
(-O3) against the shipping `RelWithDebInfo` (-O2), 8 interleaved 25 s miner
benchmarks, **−0.050 ms, −0.15 %** — inside noise and far under the 1 %-of-a-solve
floor, which is what the kernel-time accounting already predicted: the stages sum
to ~100 % of the solve, so there is no host time to optimize. The documented build
type stays.

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
section](performance.md#-below-150-w-the-clock-gap-inverts-and-the-clock-explanation-stops-applying)
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

*Resolved later the same day, by measurement and not by inference:
[lolMiner's own kernels](#lolminer-measured-under-ncu-the-state-storing-design-confirmed--and-its-54-sols-ceiling-is-a-dram-roofline)
show a state-storing streaming design with no derive or rebuild arithmetic at all —
the work-per-clock gap is the absence of our arithmetic, and never a leaner version of it.*

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
fills r1's cycles is the algorithm (hashing, combine, mix) and never the organization, and
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
definitional gap is bounded by *its* false-survivor rate, where ours does not enter. The
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
  bytes to issue far fewer instructions. Above ~200 W the arithmetic hides in
  memory stalls and the byte bill is the binding one — we win, and our ceiling
  (64.9) sits above their roofline (54). Below ~190 W the arithmetic is priced at
  full clock-starved cost — they win. Nobody is doing anything the other could not
  in principle do; the two designs are the two ends of the same trade.
  **The w0-checkpoint record is this model's first confirmation from the other
  direction**: deleting arithmetic bought 11–14 % at 100–120 W against 3.3 % at
  285 W, which is what "priced at full clock-starved cost" predicts.

**What it does and does not re-open.** It does not un-close the low-power goal by
itself: our own full-record variants (`MXBM_R2_FULL`, the quad record) lose at
every cap *inside this organization*, because they bolt stored bytes onto machinery
that still stages, chains and walks per element — the measured difference is their
ground-up streaming design (flat 64 B records, no leaf payloads, inline ancestry,
512–1024-thread blocks, no sub-mask rescan). A from-scratch "eco pipeline" in that
style is now a *specified* project instead of a mystery — with its prize honestly
bounded by the band: ~20–30 % below 160 W, zero above ~200 W, on a card class
where the band is already ceded. Parked as a design note; it is not a lead.

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

### The record-set split: OpenCL reaches CUDA's 5.7 GiB floor and gets faster doing it
<details>
<summary>Details</summary>

*(2026-08-01. Found while planning the OpenCL small-card work: the first tester
round was about to send a 1660 Ti and a GTX 1080 against a build whose OpenCL
path refuses below ~10 GiB reported — the flat 256 B/element budget fires before
the geometry ladder is consulted, and NVIDIA's `max_alloc = ¼ VRAM` blocks every
rung's 2.45–3.27 GiB record array anyway. lolMiner's floor is 3–4 GB.)*

Two host-side fixes and one kernel change:

- **The ladder is the authority.** `budget_full_rowbucket()` builds the full-2^25
  budget and `rowbucket_viable()` decides; the flat divisor now gates only the
  sort path. (The same disease as the 2026-07-25 "396 B/element" fix, one layer
  up: a conservative flat figure refusing cards the real allocation would fit.)
- **Bucket-half split.** A record set that busts `max_alloc` is allocated as two
  buffers, buckets `[0, nb/2)` / `[nb/2, nb)`. The fused kernels take lo/hi + a
  half count: the input select is once per workgroup (bucket is uniform), the
  emit select once per child. Unsplit binds one buffer twice with `half == nb`,
  so the hi pointer is never dereferenced. `MXBM_SPLIT=1` forces the split on a
  card that does not need it — the 16 GB rig's only way to run the code path.
- **Allocator step-down + touch.** OpenCL now walks the rung list on allocation
  failure exactly as CUDA does. `clCreateBuffer` is lazy on NVIDIA, so each big
  buffer is touched (`clEnqueueFillBuffer`) at alloc time — the commit fails
  where the ladder can still step, and never at first launch.

Measured (interleaved ABBA, `bench_rounds 20` pipeline medians, KAT 3/3 and
drops 0 in every arm; 49/49 suite green in both modes):

| arm | ms |
|---|---|
| baseline (pre-change binary) | 39.8 / 39.7 |
| new kernels, unsplit | **38.6 / 38.6** |
| new kernels, `MXBM_SPLIT=1` | 38.6 / 38.7 (second bracket 38.5/38.6 unsplit) |

**The split itself is free** — pipeline and 60 s miner medians identical to the
decimal (38.9 ms/solve, 1,536 vs 1,535 solves, 50.9 sol/s both arms, energy
counter within 0.2 %).

**Standing unexplained: the new kernels are −1.1 ms (−2.9 %) with the split OFF.**
Both brackets agree (39.8/39.7 vs 38.6/38.6). The change on that path is the
per-workgroup base-pointer select, a per-child compare+select, and a 22→26 arg
list — nothing that should pay. Candidate mechanism: hoisting `in_belem` into an
explicit uniform local pointer changed what the compiler proves loop-invariant in
the staging loops. Not yet chased through `MXBM_CL_VERBOSE` register deltas; the
gain is banked but its mechanism is unnamed, so do not build on it.

Reach outcome (arithmetic pinned by `test_rowbucket_geom`, table in
[HW_REQUIREMENTS](HW_REQUIREMENTS.md#the-vram-ladder)):
11–12 GB cards climb to packed (16,1) — they were on quad (15,2) at 45.1 ms or
(14,3) — 10 GB joins at (16,1), 8 GB at split (15,2), 6 GB at quad (14,3). The
OpenCL floor moves 11 GB → ~5.7 GiB reported, equal to CUDA's. **No sub-11-GB
card exists on the rig**: `MXBM_SPLIT=1` is the local proxy, and the first real
validation is the testers' 1080/1660 Ti (TESTERS.md).

</details>

### The 128-bit family, ported to OpenCL: side plane + vector access + PAIR128 (−5.4 ms)
<details>
<summary>Details</summary>

*(2026-08-01, same day as the split. The largest CUDA win never ported: 128-bit
record access was worth 41.5 → 35.3 ms there in 2026-07-25, via the MIO
instruction-queue mechanism, and `lds.cl` had no vector types at all. Blocked on
the same precondition CUDA hit: r2's 9-u64 record stride leaves nothing 16 B
aligned.)*

Ported as three pieces, mirroring `fused_round.cuh` shape-for-shape:

- **Side plane** (structural, not toggleable): the packed r2→r3 record drops to
  stride 8 `[work0..6 | p0]`; the 9th word (`r3_p1`) lives in `fb_side`, its own
  never-split buffer indexed by GLOBAL slot. Slot cost stays 9 u64, so
  `rowbucket_bytes`' totals are untouched; `single` is now ~12 % conservative for
  packed rungs, which the split makes moot.
- **Vector access** (`-DLDS_V2=0` restores scalar): staging 4 × 16 B loads for r3
  (was 9 scalars), 3 × for r4's work words; emit 4 × 16 B stores for r2-packed
  and r3 (both formats), 1 × for r1's pair record and r4's 16 B record; the
  terminal round's 16 B record in one load. Quad's 24 B record stays scalar
  (CUDA measured the padding trade against it — "fewer instructions AND fewer
  bytes").
- **PAIR128**: r2 stages its 16 B pair record in one aligned load, before the
  sub-mask filter, exactly where CUDA's `MXBM_PAIR128` sits.

Measured (interleaved ABBA, `bench_rounds 20` medians; KAT 3/3 + drops 0 in
every arm and every mode combination — quad, split, quad+split, `LDS_V2=0`;
49/49 suite green):

| arm | ms |
|---|---|
| baseline (record-set-split build) | 39.1 / 38.7 |
| **new, vectors on** | **33.4 / 33.5** |
| new, `-DLDS_V2=0` (plane only, scalar) | 39.5 / 39.3 |
| quad: baseline → new | 40.0/40.1 → 39.1/38.9 (−1.05) |

The decomposition reproduces CUDA's history exactly: **the plane alone is a
wash** (what "un-pad round 2" measured there) and **the vectors are the whole
−5.7 ms**. Zero spill stores/loads in every `-cl-nv-verbose` report.

Miner loop, 120 s opportunistic: **34.0 ms/solve median, 58.6 sol/s, 4.84
J/solution** (3,506 solves, 2.01 solutions/solve — short-run high side; quote
the ms). `MXBM_SPLIT=1` identical to the decimal. Against CUDA's 33.3–33.5 the
fallback's gap is now **~1.02×** (was 1.16× a week ago, 1.22× at the port), and
the OpenCL path passes the 53 sol/s lolMiner target on its own.

Standing note: the profiler-free mechanism attribution here is inherited from
CUDA (MIO instruction-queue pressure) and never measured on this path —
NVIDIA's OpenCL still cannot be profiled. The Ada result transfers; whether the
same shapes pay on Pascal/Turing is round-two tester data.

**The terminal round joined the perfect table too — and it is a TIME NULL here.**
`round5_fused_lds` was the last kernel still running the 512-entry keyed walk
after the 07-31 backport did the rounds. On the `bb+sm=17` line the same
perfect-hash argument applies, so `lkey` and the walk's compare are dead: the
kernel's LDS falls **11,268 → 8,200 B (−27 %)**, which on paper takes it from 4
resident workgroups to 5 inside OpenCL's 48 KB. Measured 33.7/33.6 → 33.6/33.5
ms — **+0.1 ms, under the 0.34 ms floor, i.e. nothing.** Kept anyway, on the
same grounds CUDA shipped its −0.07 ms version: it deletes code instead of
adding a knob, the redundancy is *proved* (not tuned), and it removes a
divergence between the two backends. `-DLDS_PERFECT_TAB=0` restores the keyed
walk in the terminal round as it already did in the rounds. The reason a −27 %
LDS cut buys no time is that the terminal round is ~0.9 ms of a 34 ms solve;
it was never occupancy-bound, exactly as its `LDS_TCAP` comment says.

**The cheap-flag batch: three nulls, one of them instructive.**

- **`-cl-nv-opt-level`.** The build options string had been empty since the
  path was written, so this was free to try: 2, 3 (default) and 4 all read
  33.5–33.6 ms. **Positive-controlled** — `-cl-nv-opt-level=0` reads
  **179.7 ms**, 5.3×, so the flag demonstrably reaches ptxas and the null is
  the compiler's answer, and no dropped argument. Same shape as the CUDA
  compiler axis: the default is already the local optimum.
- **`restrict` on `round5_fused_lds`.** Every pointer, matching the fused
  rounds. Null on time; kept, being free information to the compiler.
- **`reqd_work_group_size(256)` on the entry kernel — measured, then
  REVERTED.** The first bracketed round read −0.25 ms and the second, with the
  arm order reversed, read +0.3; pooled, 33.625 vs 33.65 ms. The apparent win
  was session drift, and the reversed bracket is the only reason it did not get
  written down as one. CUDA pins its `entry_scatter` at 256 because CUDA has no
  driver-choice mechanism; OpenCL does, this kernel is a flat 1-D map with no
  LDS, and pinning it would take that freedom away on every card we cannot
  measure. Left at the driver's choice, with the null recorded in the kernel so
  it is not re-proposed.
- **Baking the residual geometry (`bb`, `sm`, the bucket caps) — a properly
  powered null, and the instrument is the story.** These were the last runtime
  kernel args on the fast path, and the host picks all four *before* it builds
  the program, so they can be literals: the per-child bucket shift and the two
  64-bit slot multiplies stop carrying runtime operands. Built as `GEO_BAKED`,
  with the geometry appended to the options string so `cached_program` keys on
  it and **both arms come from one binary** (`MXBM_NO_GEOBAKE` selects).

  First reading was four `bench_rounds` arms each: 33.60 against 33.65, called
  a wash. **That call was underpowered** — the effect under test (0.15 %) is
  half the documented within-session band, so it was not evidence of absence.
  Re-run properly, 20 paired 60 s miner arms, order alternated:

  | instrument | baked | runtime | diff |
  |---|---|---|---|
  | ms/solve median, n=10 each | 33.900 | 33.920 | +0.020 ms, t=0.80 |
  | **solves per 60 s, n=10 each** | **1760.6** | **1760.7** | **−0.14 solves, −0.008 %, t=0.26** |

  **The ms median cannot answer this question**: it prints to 0.1 ms and 17 of
  20 runs read exactly 33.9, so the quantisation is five times the effect. The
  solve *count* over a fixed window is not quantised that way — the runtime arm
  reproduced to **sd 0.7 solves in 1760, 0.038 %** — and at that resolution the
  difference is 0.008 %, sign-inconsistent with the first pass. This is now a
  null with the power to mean it.

  **Kept anyway, enabled, for a reason the reference card cannot test.** On
  sm_61 a 32-bit integer multiply is *multiple instructions* (XMAD sequences;
  CUDA C Programming Guide Table 4), where Ada has a native IMAD — so a runtime
  operand in the per-child slot arithmetic costs materially more on the Pascal
  and Turing cards the record-set split just made reachable than it does here.
  It is measured non-negative on Ada, correct in both modes, and
  `MXBM_NO_GEOBAKE` makes the small-card measurement a one-command experiment
  for whoever first has the hardware. Deleting it would mean rewriting it to
  ask that question.

  **What this does close is the compile-time-constants family on Ada.** The
  −26 ms that started it was never about constants as such; it was about a
  dynamically-indexed private array escaping scratch (`t[8]` in `apply_mix`).
  These four index nothing, so folding them buys nothing here. *Look for the
  dynamic index first; the constant is the fix, and the diagnosis came before it.*

  Method note: one baked arm in twenty read 1705 solves / 35.2 ms — 3 % slow
  with the clock unchanged at 2760 MHz — an external transient, and no property of an arm
  effect. It moved the naive mean by 5.6 solves and would have inverted the
  verdict on its own. Use medians and an explicit outlier check; means alone will not do.

</details>

### Speculative entry, ported to OpenCL: the entry pass hides inside round 4 (−0.35 ms)
<details>
<summary>Details</summary>

*(2026-08-02. The last structural CUDA win not on this path. There it is worth
−0.45 ms; the mechanism is plain grid-index dispatch, so nothing about it
needed OpenCL to grow a feature.)*

**The mechanism.** Round 4's grid gains one *entry* workgroup per `co_stride - 1`
round workgroups, interleaved by index instead of appended — appending would
schedule them all in the tail, which is the second-stream null over again. Entry
groups return before the first barrier, and barriers are per-workgroup, so the
round groups never see them. The entry output moves to its own set (`spec_elem`,
+314–372 MiB by rung) because `fb_elem[0]` is overwritten by round 2 and could
carry nothing across a solve. `GpuSolver` learns the engine's nonce stride, so
the next solve finds its entry pass already done.

Measured, 20 arms, paired and order-alternated, one binary (`MXBM_NO_SPEC`
switches the arms), solve count over a fixed 60 s window as the instrument:

| | no-spec | spec |
|---|---|---|
| solves / 60 s | 1764.9 | **1783.2** |
| ms/solve | 33.996 | **33.647** |
| printed median ms | 33.85 | 33.54 |

**+18.3 solves paired (+1.04 %), sd 7.0, t = 8.29 on 9 df.** 2.00 verified
solutions/solve in all 20 arms. Miner loop, 120 s: **33.5 ms/solve median, 59.4
sol/s, 4.78 J/solution**. Same-session CUDA on the same card: 33.1 ms, 60.2
sol/s, 4.72 J — the fallback's gap is **1.012×**.

**The size is the evidence.** The entry pass costs ~2.4 ms standing alone, 7 %
of a solve. If speculation merely *skipped* it the gain would be ~7 %; recovering
1 % means about a fifth of it hid in round 4's stalls and the rest displaced real
round work — the same fraction CUDA measured. A result near 7 % would have meant
a correctness bug wearing a triumph's numbers.

**`co_stride` is a null across a 3× range.** Strides 2 / 3 / 4 / 6 — dedicating
50 % / 33 % / 25 % / 17 % of the grid to entry groups — measured 1783.0 / 1784.8 /
1783.8 / 1785.0 solves, a 0.11 % spread against a per-arm sd of ~5. The work hides
whether it is given a sixth of the groups or half of them, so the mechanism is
robust instead of tuned. Default stays CUDA's 3; `MXBM_CO_STRIDE` remains the
per-card lever.

**The control that mattered.** Correctness here is not visible in timing: a
mis-speculated entry set still produces survivors, just useless ones. Seeding the
speculation one nonce off collapsed verified solutions from 2.00/solve to **0.01**
— so the 2.00 in the real build can only come from round 4 having produced *this*
nonce's entry set. Nothing in the suite reached the path before (`run_pipeline`'s
tests pass no `SpecEntry`, and it takes three fixed-stride solves before a
prediction lands), so `test_gpu_solver` now walks n₀−2, n₀−1, n₀ onto the KAT
nonce and asserts both the goldens and that the hit *happened*.

**A silent-failure mode found while wiring it.** The solver first marked the
speculation valid whenever it *asked* for one. But `run_pipeline` returns early on
abort, possibly before round 4 ran — and the next matching solve would then have
read the previous nonce's entry set and mined nothing for that nonce, with no
error anywhere. `SpecEntry::seeded`, set only by the actual co-block launch, is
the fix. Today's caller happens to be safe (aborts arrive with a job change, which
fails the input compare); that accident belongs to the caller and not to the contract.

**Reach is not traded for it.** The buffers are allocated only after the rung
ladder has settled, best-effort: a card that cannot spare 0.31–0.36 GiB simply
runs the entry standalone. Sizing them into the rung choice would let a card lose
a whole rung — a measured ~1 ms of geometry — to buy a third of one.

</details>

## Measured results, 2026-08-02

### The top half of the pipeline is a siphash machine — and three ways of making it cheaper are already taken
<details>
<summary>Details</summary>

*(CUDA, `cuda/pipeline` on the current kernels, `MXBM_ROUND_REPS=R:9`, 40 nonces,
each replay bracketed by a baseline run so drift is visible. KAT green throughout
except where an ablation is named — those are deliberately wrong.)*

**The profile has not moved.** Marginal cost per round: r1 **5.12**, r2 **10.17**,
r3 **9.49**, r4 **5.48**, terminal **0.98** ms — within 0.1 ms of the 2026-07-31
stage-power table on every row. The 08-01/08-02 work was OpenCL, and the CUDA
profile says so.

**r1's scattered payload store is FREE; r2's costs 1.41 ms.** `MXBM_ABL_EMIT=R`
removes only the scattered record write, keeping the atomics, back-refs, combine
and apply_mix, so the element counts and every other access stay representative:

| round | full | store ablated | the store costs |
|---|---|---|---|
| r1 | 5.117 | 5.301 | **0.00** (the ablation measured *slower*, inside drift) |
| r2 | 10.170 | 8.758 | **1.41** |

**What the two rounds are, then.** r1 and r2 stand in the ratio of their siphash
counts — 235 M against 470 M predicts 2.00, measured 1.99 — and ~11 ms of the
33.5 ms solve is re-derivation. That one number sits behind two standing results:
why every byte-side lever in these two rounds has been null (their bytes are free
or nearly so), and why *both* directions of the store-instead-of-derive trade lose
([R2_FULL](#the-eco-sweep-170-crosses-over-below-190-w-r2_full-never-does), the
[quad record](#the-quad-record-29--footprint-and-the-byte-prize-does-not-survive-re-derivation)).
The way to speed up the top half is fewer siphashes, and the schedule fixes how
many: 7 per seed element, 14 to rebuild a pair, every word load-bearing (the
child's top word reads the parent's, which reads the seed's).

> **Correction, same day: the ratio is real, the arithmetic behind it was not.**
> This section first solved `M + D = 5.12` and `M + 2D + 1.41 = 10.17` for a
> derivation of 3.64 ms and 1.5 ms of match machinery per round. That assumes r1
> and r2 have the SAME machinery, and they do not — r1 reads 8 B and writes 16,
> r2 reads 16 and writes 72, at 5 blocks/SM against 4. Measured directly instead,
> by removing one of the nine derivation passes a group needs
> ([below](#the-tail-and-the-elements-with-no-partner-a-family-priced-and-closed)):
> **the derivation is ~2.8 ms in r1 and ~3.5 ms in r2**, not 3.6 and 7.3. r2's is
> well under twice r1's, which says the extra 7 siphashes partly hide in the
> traffic r2 has and r1 does not. The 1.99 : 2.00 agreement is then a coincidence
> of two rounds whose machinery differs by about as much as their derivation does.
> `MXBM_ABL_DERIVE` cannot be used to price this — it replaces the work state with
> a spread, which moves every key and blows the walk up: both rounds read ~10.0 ms
> under it, r1 *slower* by 4.7 ms than with the derivation left in.

**Three ways of making the siphash itself cheaper — all already taken by the
compiler.** Each was worth checking precisely because it would otherwise stay
plausible forever:

| lever | verdict |
|---|---|
| `rotl64` → explicit `__funnelshift_l` pair | **Already optimal.** The shift-or form compiles to `SHF.L.W.U32.HI` — 44 of them for the six siphash rounds — so NVPTX recognises the rotate. Writing the funnel shift by hand is *worse*: 176 SASS instructions against 168 |
| hoisting the nonce-independent siphash prefix out of the 7 seed calls | **Already shared.** `seed_element` is 1016 SASS instructions against ~1136 for seven independent copies: nvcc lifts ~17 per call, which is the whole prefix (`v0+=v1`, `rotl(v1,13)`, `v1^=v0`, `rotl(v0,32)`, `rotl(v1,17)`) |
| the prePow key in `__constant__` instead of registers (`MXBM_PP_CONST=1`) | **A wash, and the premise was wrong.** The key is read by every siphash of every element, so it looked like 8 pinned registers of the 64 r2 is capped at — but register counts are byte-identical in every instantiation with and without it, so ptxas was already re-materialising rather than pinning. 12 ABBA arms: **−0.023 ms, −0.07 %**, against the 1 %-of-a-solve floor |

`MXBM_PP_CONST` stays in the tree as the closure's instrument, off by default and
codegen-neutral when off, the same way `MXBM_LWORK_SOA` does.

</details>

### The tail, and the elements with no partner: a family priced and closed
<details>
<summary>Details</summary>

*(2026-08-02. Two inefficiencies that are visible from the source and turn out to
be worth much less than they look. Both probes are in the tree, off by default and
byte-identical SASS when off, so the family can be re-priced instead of
re-argued.)*

**The shapes.** A staged group holds `capacity / 2^17` = **264** elements and a
block is **256** threads, so the expand loop needs `ceil(264/32)` = **9 warp
passes** for 8.25 passes of work — the last one runs 8 lanes of 32. Separately,
the walk only ever reads an element that shares its chain slot with another, and
at 264 elements over a 128-entry perfect table `e^-2.06` = **12.8 %** of a group
is alone in its slot: every one of those pays a full 7- or 14-siphash rebuild for
a work state nothing reads.

**Priced** (`MXBM_ABL_TAIL`, `MXBM_ROUND_REPS=R:9`, 40 nonces, results wrong by
construction so `MXBM_FORCE_TIMING`; the derive-fraction arm runs at
`MXBM_PERFECT_TAB=0` so the chain comes from the staged key and the walk and emit
stay whole):

| probe | r1 | r2 |
|---|---|---|
| drop the 8-element tail entirely | −0.25 | −1.03 |
| derive 7/8 of the group — one derive pass of nine | **−0.315** | **−0.393** |
| skip 1 record load in 8, r3 (bandwidth arm) | — | −0.163 (r3) |

The middle row is the honest one, and it is what prices the derivation itself at
**~2.8 ms in r1 and ~3.5 ms in r2** (nine times the saving). It also caps the
prize: skipping every unmatched element is worth one pass of nine, **~0.7 ms
across r1 and r2**, because 12.8 % fewer elements is exactly one fewer pass.

**Built, correct, and a net loss** (`MXBM_MATCH_FIRST`). The chain moves into the
staging loop, where the key is already in a register; one thread per slot then
walks its members and compacts those with a partner into `mlist`; the rebuild runs
over that. KAT 3/3 byte-identical, drops 0 — and per round: **r1 +0.131, r2
−0.055, r3 +0.015, r4 +0.051**. Three costs eat the 0.7:

- `mlist` is 640 B of shared, which takes **r1 from 5 resident blocks to 4** — the
  fifth block is worth [−0.15 ms](#round-1-takes-a-fifth-block-015-ms-via-a-per-round-group-cap) on its own.
- the compaction pass is a pointer walk per chain slot, and it needs its own barrier.
- compacted order is not slot order, so the rebuild's `lwork` writes stop being a
  regular stride — straight into the [2-way bank conflicts](#the-round-2-instruction-census-865--hash-arithmetic-and-one-named-lever) the census already measured there.

**And the bandwidth half does not pay either.** In a DRAM-bound round the same
idea needs no compaction — skipping a load saves sectors even when the lane
diverges — but r3 gives only **0.163 ms** for one load in eight, because r3's top
stall is the MIO queue: memory-INSTRUCTION issue, with bandwidth uninvolved, and a divergent
skip issues anyway.

So the family is closed at the source: **the tail is 9 passes for 8.25 passes of
work, the unmatched elements are 12.8 % of a group, and together they are worth
~0.7 ms gross and less than that net.** Anything that claims more from this
direction is claiming more than the passes exist to give.

</details>

---

### Three standing candidates, measured and closed: the entry pass, the host loop, the group cap
<details>
<summary>Details</summary>

*(2026-08-02, the three directions left open after the siphash and tail families.
All null; the value is in the mechanisms, which are sharper than "measured null".)*

**1. The entry pass cannot be hosted in round 3, and the register file was not
what stopped it.** ~2.2 ms of entry's 2.68 is still exposed after speculative
entry recovers 0.45, and r3 runs at 21 % SM utilisation for 9.5 ms, so it looks
like the obvious host. The stated obstacle was residency: r3's three blocks at
**80 registers** hold 61,440 of the SM's 65,536, leaving 4,096 — short of the
11,776 a 256-thread entry block needs. `MXBM_MB_EMIT=4` removes that obstacle
exactly: r3 drops to **64 registers with zero spill**, freeing 16,384, and it
costs nothing (34.63 / 34.64 against 34.59 / 34.69 sequential).

It changes nothing. Entry as co-blocks, stride 3, 40 nonces:

| host | stock r3 (80 reg) | r3 capped to 64 reg |
|---|---|---|
| **r4** (shipping) | **34.26** | **34.24** |
| r3 | 34.80 | 34.76 |

Hosting in r3 is *worse than not overlapping at all* (sequential 34.64), with or
without the registers. The mechanism is visible in the traffic table: **no round
has both an idle SM and an idle memory system.** r1 and r2 run at 200 and
323 GB/s — a third and a half of peak — and are saturated on issue; r3 and r4 run
at 508 and 538 and are saturated on memory. Entry needs both at once, so wherever
it is put it displaces the thing that round is short of. That, and not any
particular mechanism's failure, is why the whole overlap family tops out at
~0.5 ms.

**2. The host loop has no gap to reclaim.** The kernel-time accounting already
summed to ~100 % of a solve, but that was arithmetic over replays and not an
observation of the card. `utilization.gpu` sampled through a 35 s miner benchmark
reads **100 % on every sample**, against a 2 % idle baseline. Whatever the
readback, the recovery kernel and the CPU verify cost, they are not costing GPU
time.

**3. The group cap is still 320, and round 3's fourth block is still worth
nothing.** `MXBM_FCAP` re-swept on the current kernels — the last sweep predates
the perfect table, the spill, LD.128 and co-blocks:

| FCAP | r1 | r2 | r3 | r4 | solve |
|---|---|---|---|---|---|
| 288 | 5 | 4 | **4** | 4 | +0.37 % |
| **320** | 5 | 4 | 3 | 4 | — |
| 352 | 5 | 3 | 3 | 4 | worse |
| 384 | 5 | 3 | 3 | 3 | worse |

288 buys r3 the fourth resident block the ledger has twice said it does not want,
and this time it is priced per round instead of in aggregate: **r3's marginal
time moves +0.007 ms**, 9.890 → 9.897. The block is free and useless. The overall
+0.37 % is the spill: at 288 the cap sits at mean + 1.5σ where 320 is mean + 3.4σ.
A per-round cap cannot rescue it, because there is nothing to rescue.

**And the finding that came out of the three.** Pricing each round's scattered
payload store with `MXBM_ABL_EMIT`:

| | r1 | r2 | r3 | r4 |
|---|---|---|---|---|
| the store costs | **0.00** | 1.41 | 0.54 | **0.00** |

**Writes are free; reads are not.** Three of the four rounds pay nothing at all
for the bytes they scatter, and that sharpens "bytes are nearly free" into
something directional: any trade that buys compute with *written* bytes has
already been given its bytes for free and still has to win on the read side,
which is where [R2_FULL](#the-eco-sweep-170-crosses-over-below-190-w-r2_full-never-does)
and the [quad record](#the-quad-record-29--footprint-and-the-byte-prize-does-not-survive-re-derivation)
both actually lost.

</details>

---

### The algorithm itself: what the PoW fixes, what is free, and the yield we actually get
<details>
<summary>Details</summary>

*(2026-08-02. Asked directly: forget the implementation, is the SEARCH better?
The answer is bounded from both ends — measured yield on one side, the PoW
definition on the other.)*

**We already find every solution there is.** 200 distinct nonces, sequential
path, KAT-gated:

| | |
|---|---|
| verified solutions / solve | **2.04** |
| survivors / solve | **2.04 mean, 7 max**, against a 1024 cap |
| drops (bucket, pair, chain-walk) | **0** |

Two things fall out. **Survivors equal verified solutions exactly**, so the
terminal round emits no candidate that fails the CPU's distinct-leaf and
`indexAfter` checks — there is no filtering loss to reclaim. And the survivor cap
— the one place a candidate could vanish with no counter behind it, since
`hs > survCap` is a silent clamp on both backends — has **146× headroom**. It is
now printed by `cuda/pipeline`, so the day it stops having headroom, somebody
sees it.

Against theory: Wagner on ⟨144,5⟩ with N = 2^25 is parameterised to yield ~2
solutions per nonce, and 2.04 is that. **There is no yield to win** — only time.

**What the PoW fixes, and therefore what cannot be chosen:**

- **N = 2^25.** The index is 25 bits, so that is the whole space — and it is also
  Wagner's fixed point, since N²/2^25 = N is what keeps a layer the same size
  across rounds. Fewer seeds is the (epr/2^25)^32 collapse
  ([the partial-search caveat](HW_REQUIREMENTS.md#the-partial-search-caveat)); more do not exist.
- **Five rounds, 24 bits each, and the width schedule** 448 → 424 → 400 → 376 →
  … The verifier recomputes it, so a solution found any other way is not a
  solution.
- **Seven siphashes per seed element**, and `apply_mix` every round.
- **`apply_mix` sums rotations instead of XOR-ing them.** This is the load-bearing
  one, and it is deliberate. Under XOR the mix would be linear, a child's key would
  be a function of its parents' keys, and the whole match could run on keys without
  ever materialising a child — which is exactly what BeamHash **I** allowed with its
  16 B element, and exactly what Fork2 removed
  ([limit 1](#established-limits)). The `+` is why the element is 56 B and why every
  record-narrowing idea in this document runs into re-derivation instead of algebra.

**What is actually free** is the search's *organisation* — bucket versus sort,
store versus re-derive, record widths, occupancy, overlap, geometry — which is
what every entry above is about, and it is the part that has been optimised to
the point where the closures are structural.

**Where algorithmic headroom does still exist, and it is not stock speed:**
the memory–time trade (streaming / in-place layer reuse, the route to the 3 GB
target) and the store-everything eco pipeline for the low-power band. Both are
[leads](#current-focus-and-open-leads) already, and both trade time for something
other than time.

</details>

---

## Measured results, 2026-08-04

### The memory VF-offset dial cannot replace or extend the rung
<details>
<summary>Details</summary>

The hard ~173 W rung crossover invited a continuum: shift the memory clock by a
VF offset (`nvmlDeviceSetMemClkVfOffset`, root) instead of the two-point P-state
choice. Swept with zero-offset brackets at 160/140 W, read-back asserted per set,
yield gated (2.02–2.04 verified/solve in every arm, no corruption to −750 real
MHz):

- **Down-dial from stock memory applies** (reported clock moves at offset/2, the
  GDDR6X convention) and is **time-neutral**: −750 real MHz bought ~8 MHz of SM
  clock at 160 W. A VF offset moves clock at constant voltage, so its power
  refund is linear in frequency — small by construction.
- **Up-dial on the locked rung does not apply**: the clock stays 5001 under
  `-lmc` in every arm. The lock outranks the offset — the same precedence family
  as the undervolt result below.

The rung's value is the P-state **voltage step** (worth +450 MHz of SM clock and
−7 % at 160 W, reconfirmed in this sweep's own arms); no offset magnitude
replicates a V² saving. The crossover stays a two-point choice on this card;
cards with finer memory P-state grids remain the open generalization.

</details>

### Undervolting buys nothing under a power cap — the cap outranks both knobs
<details>
<summary>Details</summary>

*(`docs-internal/rootruns/coff_sweep.sh` and `uv_sweep.sh`, on the shipping miner loop.
The lever the ledger had carried as "the one unmeasured axis this card exposes",
untestable until the monitor moved to the motherboard iGPU and the compute GPU stopped
holding a graphics context.)*

**The V/F offset alone.** ABBA per (cap, offset), miner-loop arms, gated on
`rejected == 0`:

| cap | +0 → +100 | Δ | SM clock |
|---|---|---|---|
| 120 W | 68.05 → 68.05 ms | +0.00 % | **1155 → 1155** |
| 160 W | 47.80 → 47.80 | +0.00 % | 2152 → 2152 |
| 285 W | 33.00 → **32.60** | **−1.21 %** | 2648 → **2730** |

Under a cap the achieved clock does not move, so neither can the time. At stock it
does. 160 W resolves nothing either way — its two base arms drifted 1.05 %, wider than
any effect on the row.

**The full idiom, `--cclk` plus `--coff`.** Locking the clock above what the cap
delivers is the half that should work — the offset supplies that clock at a lower
voltage, and the cap fits it. It does not: **every target was void**, the requested
clock never approached at any offset.

| cap | asked | achieved, all arms | with no lock |
|---|---|---|---|
| 120 W | 1320 / 1440 / 1560 | 1140–1170 | 1200 |
| 160 W | 2340 / 2550 / 2775 | 2085–2205 | 2130 |

**The mechanism: the lock reaches the hardware and the power governor outranks it.**
The hung arms are the evidence — their stats line reads `CCLK 2550` at 36 W with 0.00
sol/s, so the register took the locked value while idle, and only under load is it
overridden. The offset reaches the silicon too: it destabilises the card at +300. It
is applied, it can hang the card, and it still buys zero clock under a cap.

Stability inverts between the two sweeps, which is worth keeping straight. With a free
clock, +300 hangs at 120 W; with the clock pinned it does not — that hang was a boost
excursion during ramp, and no sign of steady-state instability. At 160 W, +300 hangs either way.

**So the lead closes for the low band.** The stock −1.21 % is left as an open item, not
a result: one ABBA pair, and +100 sits one rung under a hang, which a 45 s arm cannot
qualify for a miner that runs for days.

*Reopening: a driver that credits the offset in the governor's power model, or a card
whose cap does not override a locked clock.*

</details>

### A byte's cost under a cap is set by SHAPE, and shape survives the cap — R2_FULL is withdrawn as evidence against the eco pipeline
<details>
<summary>Details</summary>

*(`cuda/emit_shape_probe shape` via `docs-internal/rootruns/byte_shape_caps.sh`,
2026-08-04. The
eco-pipeline question turned on one uncontrolled variable: the evidence against a
store-everything design is `MXBM_R2_FULL`, whose extra bytes are SCATTERED 72 B stores
bolted onto the pair machinery, while a streaming design's traffic is half coalesced.
If shape drove that loss, the evidence does not transfer. Sustained ~2.5 s loops per
shape so the governor's steady state is inside the timed window — best-of-N minimums
under a cap select the least-throttled rep — stock memory clock, energy from the
counter. Full table: `docs-internal/rootruns/byte-shape/`.)*

| cap | seq GB/s @ SM | 64 B scatter GB/s @ SM | J/GB seq | J/GB 64 B |
|---|---|---|---|---|
| 285 W | 639.5 @ 2775 | 132.9 @ 2775 | 0.26 | 1.09 |
| 160 W | 639.4 @ 2730 | 132.9 @ 2775 | 0.26 | 1.11 |
| 120 W | **633.6 @ 1005** | 127.2 @ 2670 | 0.20 | 1.02 |
| 100 W | 435.3 @ 465 | **118.2 @ 2505** | 0.24 | **0.93** |

Three findings:

- **DRAM work survives a core cap.** Sequential holds 633 GB/s at 120 W on a **1005 MHz
  core**, and 435 at the 100 W floor on 465 MHz; scattered wide records lose only 12 %
  floor-vs-stock. The cap's currency is core cycles, not bytes — the lolMiner mechanism,
  now confirmed from a third direction on our own instrument.
- **The clock column is a work-per-cycle readout.** Dense stores cost so much power per
  cycle that the governor crushes seq to 465 MHz inside 100 W, while the stall-heavy
  scatter keeps 2505 MHz in the same budget. Same watts, opposite clocks.
- **J/GB is shape-dominated, and the cap barely enters**: a scattered byte costs ~4.3× a
  sequential one at every cap, and the floor makes bytes *cheaper* per GB rather than dearer.

**What this closes: R2_FULL no longer prices the eco design's byte bill.** Its loss at
every cap cannot have been the shape of its bytes — scattered bytes are nearly
cap-immune — so it priced "stored records on top of the pair machinery", not "a
streaming design's traffic". That was the one measured objection to the Tier-2 eco
pipeline; it is withdrawn.

**What this does NOT change: the prize.** The rung sweeps already bound it from the
existence-proof side: **~+11–16 % in the 120–160 W band and ~+32 % at the 100 W floor**
(each design at its best memory clock per cap — a store design cannot follow
re-derivation down the rung, which is what shrank the mid-band from the naive +20–30 %).
The probe makes the design feasible in mechanism, not larger in prize: still weeks of
work, strongest at the deep floor.

*Caveat: this probe's scattered absolutes are pessimistic — pure write bursts with no
reads to share the bus; lolMiner's real interleaved kernels sustain 393–556 GB/s on the
same class of traffic. The ratios and their trend across caps are the measurement.*

</details>

### The eco kill-probe: a streaming round holds 343 GB/s at 120 W — the design survives its cheapest kill-test
<details>
<summary>Details</summary>

*(`cuda/emit_shape_probe shape round`, 2026-08-04, same harness and discipline. One
round of a store-everything pipeline stripped to its traffic: coalesced 64 B layer
read, key from word 0, atomic slot, scattered 64 B write into bb=16 buckets — no
derive, no rebuild, no shared memory. The gate was pre-registered before the run:
17.7 GB/solve must beat MXBM+rung at the same cap, so ≥259 GB/s combined at 120 W and
≥190 at 100 W break even; ≥350 was set as "alive with margin".)*

| cap | combined GB/s | SM | 17.7 GB floor | MXBM+rung today |
|---|---|---|---|---|
| 285 W | 512.2 | 2775 | — | *(calibration: lolMiner's measured 471–512 class, reproduced)* |
| 120 W | **343.2** | **1740** | 51.6 ms | 68.3 ms |
| 100 W | **250.0** | 1200 | 70.8 ms | 92.9 ms |

**1.3× break-even at both capped points** — the traffic floor sits ~24 % below today's
MXBM+rung, and that margin is the budget for everything a real pipeline adds (terminal
pass, launch gaps, the residual arithmetic). The mechanism is the design thesis in one
contrast: this instruction-light kernel keeps **1740 MHz inside 120 W** where MXBM's
real rounds hold ~1155 — fewer instructions per byte, so the same watts buy more of
everything.

Two caveats. The probe is one round, not five chained with a seed pass — it bounds the
traffic, not the pipeline. And its W / J-per-GB columns are biased high for this mode
(the energy bracket spans the whole process including a 2.1 GB host copy, divided by
the timed wall only); the rate and clock columns are clean and are the result.

**Standing: superseded the same day by the full-pipeline mock below, which measures
the whole solve instead of extrapolating from one round.**

</details>

### The floor program: spec entry off below 130 W ships; the rebuild is worth 11 ms there and cannot be harvested in place
<details>
<summary>Details</summary>

*(2026-08-04, `docs-internal/rootruns/floor_gaps.sh`, `floor_derive.sh`,
`floor_r2full_carve.sh`, `floor_r2full_ncu.sh`. A pass over the low-band closures
looking for untested assumptions; three A/Bs and a profile, all at 120/100 W + rung.)*

**Shipped: speculative entry is off below `kSpecMinPowerW` (130 W).** Its co-blocks
ride on r4 having idle issue capacity — true while r4 is DRAM-bound, false once a cap
makes every round issue-bound. Measured in the miner loop, ABBA: **nospec −1.76 % at
120 W, −1.87 % at 100 W**, spec keeping its win at stock and at the 285 W rung point.
The gate reuses the observed-limit plumbing next to `kRbLowPowerW`. CUDA only. The
mid-band was measured on 2026-08-18 and the gate moved to the measured crossovers —
see [speculative entry under a
cap](#speculative-entry-under-a-cap-both-crossovers-measured-and-the-gate-moves-to-them).

**Closed: `kWG 288`** (removes the 9-passes-for-8.25 expand tail) is a wash under caps
(+0.0/+0.3/+0.4 %) and +0.24 % at stock, re-measured on current kernels.

**The rebuild's floor price, and why it stays.** `MXBM_ABL_DERIVE=2` by replay
amplification — the spread corrupts r2's output keys, so whole-solve A/Bs are invalid
(83 M drops on the first attempt) and the corruption must cancel inside each binary's
own (t9−t1)/8: **r2's rebuild costs 8.52 ms/solve at 120 W and 11.15 at 100 W**
(12.5 % of the solve). The ALU pipe is the critical path at the floor (800 MHz census:
r1 76 %, r2 70 %), so the prize is real — and R2_FULL, which deletes exactly this
rebuild, still loses. The carve locates its overhead (r1's storing side +8.6 ms, r2's
read side eating 6.7 of the 11.2) and the profile names it: **the hash arithmetic is
the kernel's latency sponge.** r2 minus its rebuild drops to 974 M instructions
(−70 %) and 12 % issue-active, warps parked in long-scoreboard (2.6 → 24.9) and
barrier (4.6 → 20.6) stalls; r1 with the fat emit loses its fifth block and backs up
the store queue (mio 12×). More occupancy is the textbook sponge and is closed from
both resources. Best-case cleanup (side-plane record, FCAP 288, register trim) nets
~+1–2 ms — below the bar.

**"Re-derivation is the right trade" now carries its mechanism**: the arithmetic
doubles as latency hiding, and this organization has no other sponge to offer.
Harvesting the 11 ms needs an organization whose sponge is occupancy — the parked eco
pipeline — which is why it cannot be grafted in piecemeal.

</details>

### Speculative entry under a cap: both crossovers measured, and the gate moves to them

*(2026-08-18. `paired_ab.sh` with wrapper arms toggling `MXBM_NO_SPEC` at runtime — the
spec buffers only, match-first untouched — SECS=60, one position-balanced block per
point, caps set externally, KAT 3/3 and drops 0 on both arms, every point in one
session.)*

The floor-program entry above turned spec off below 130 W and said 140–160 W was
unmeasured. Measured, the stock-memory crossover is nowhere near 130 — and the memory
clock moves it:

| point | spec → nospec | t | core clock, spec → nospec |
|---|---|---|---|
| 140 W | **−1.75 %** | −12.7 | 1248 → 1296 MHz |
| 160 W | **−1.75 %** | −14.8 | 1495 → 1555 |
| 180 W | **−0.99 %** | −9.7 | 1818 → 1871 |
| 210 W | **−0.22 %** | −3.7 | 2330 → 2400 |
| 240 W | +0.79 % | every arm identical (2023 vs 2007 solves, sd 0 both sides) | 2518 → 2542 |
| 285 W stock | +1.03 % | +39.8 | positive control: spec's shipped win reproduced, so the knob demonstrably applied |
| 5001 rung, 140 W | **−0.71 %** | −15.6 | 1572 → 1707 |
| 5001 rung, 160 W | **+1.81 %** | +24.1 | 2242 → 2330 |

The mechanism the floor measurement named, now confirmed from both sides. Spec's
co-blocks ride round 4 having idle issue capacity; under a cap they displace work the
card can no longer spare, and the clock column prices the displacement directly —
dropping them returns 48–135 MHz of core. On a held 5001 rung the freed memory watts
un-starve the core (2242 MHz at 160 W against stock memory's 1495) and bandwidth is
halved, so r4 is DRAM-bound again and spec's win comes back **below** the stock
crossover: the crossover is a function of the memory clock, not of the cap alone.

**Shipped**: the gate becomes two measured crossovers — `kSpecMinPowerW` 130 → **220**
on stock memory, and `kSpecMinPowerRungW` **150** when the memory clock observed at
startup is ≤ 5100 MHz (a held rung reports its lock even on an idle card; the > 600 MHz
guard excludes the idle P-state, so an unlocked card reads as stock). Match-first is
decoupled onto its own `kMatchFirstMaxPowerW` at the 130 W band it was measured in —
these arms toggled only the spec buffers, so match-first above 130 W stays unmeasured.
The policy line printed in all three regimes asserts the patch applied.

What it buys at the caps a capped rig runs: **−1.75 % ms/solve at 140 and 160 W,
−1 % at 180, −0.2 % at 210**, nothing at or above 220, and −0.7 % at 140 W on the
rung. The 175–200 W rows in the head-to-head table are re-measured with the shipped
default, with nothing interpolated.

### The full-pipeline eco mock: −29 % at the 100 W floor, −13 % at 120 W — alive at the deep floor only
<details>
<summary>Details</summary>

*(`cuda/emit_shape_probe shape pipe`, 2026-08-04. The whole store-everything pipeline
as a traffic-faithful mock: a REAL 235 M-siphash seed pass — the one derive the design
keeps, and compute is the expensive currency at a cap — then four chained streaming
rounds and a thin terminal read, ping-ponging real bucket layers, six launches and the
counter resets included. 17.45 GB/solve on the competitor's measured schedule. Stock
calibration: 37.4–37.9 ms against the real thing's 35.8 on the same schedule, within
6 %. Gates pre-registered per arm before each run.)*

| cap | mock, stock memory | mock + 5001 rung | MXBM + rung today | best vs MXBM |
|---|---|---|---|---|
| 285 W | 37.4 ms | 59.0 ms *(the rung roofline — lock-held sanity arm)* | — | — |
| 120 W | 60.4 | 59.6 | 68.3 | **−12.7 %** — thin (gate < 55) |
| 100 W | 82.1 | **66.1** | 92.9 | **−28.9 %** — alive (gate < 70) |

Three findings:

- **The one-round margin halves once the pipeline is whole.** The kill-probe's −24 %
  became −11.6 % on stock memory: the seed's real arithmetic bills at the starved
  clock, and the thin legs run below the fat rounds' efficiency. Extrapolating a
  pipeline from its best kernel overstates it — measured, not assumed.
- **The rung crossover transfers to a store design, and sits between 120 and 100 W** —
  the competitor's own ~126 W, reproduced. The 120 W rung arm shows the balance
  exactly: the interface refund lifts the clock 1620 → 2520 MHz and the rate does not
  move, because it lands on the rung's bandwidth ceiling. At 100 W the refund wins
  (−19.5 % over stock memory).
- **These floors are ceilings for a real implementation.** The mock has no match
  arithmetic — no compare/pair work beyond a key update — so a real design eats into
  the −13 % / −29 % from above.

**Verdict for Tier 2, now measured end-to-end on this rig:** the eco pipeline is
**alive only at the deep floor** (~−29 % at 100 W, before match arithmetic), **thin in
the 120 W band** (−13 % before match arithmetic, against a weeks-sized build), and
dead at 140 W and above where the rung plateau falls behind re-derivation. The
decision input is complete; what remains is only whether the ≤~110 W rig class is
worth weeks of work.

*Superseded 2026-08-12 by [the switching-height re-pricing](#the-switching-height-re-pricing-a-store-everything-design-with-a-re-derived-round-1-wins-the-whole-100150-w-band): a store design that re-derives only round 1 moves the viable band to ~100–150 W, strongest in the middle.*

</details>

---

## Measured results, 2026-08-12

### The mix provably ignores parts of the index tree at rounds 4 and 5
<details>
<summary>Details</summary>

`apply_mix` writes tree entry *i* at bit offset `Lmix + 25i` of the 512-bit buffer and
never carries past `t[7]`, so the buffer boundary truncates the fold. Two consequences,
pinned by a 200 000-trial randomized identity test with live-bit positive controls
(campaign probe `lane_test.cpp`; the perturbed-input liveness control fails as
required):

- **Round 4** (`Lmix` 376, tree 8): entries 6–7 never enter the mix, and entry 5
  contributes only its **low 11 bits** (offset 501; the carry branch requires
  `word < 7`). The round-4 key consumes 136 of the tree's 200 bits.
- **Round 5** (`Lmix` 288, tree 16): entries 9–15 never enter, and entry 8 drops its
  bit 24. The round-5 key consumes 224 of 400 bits.

The `padNum` prefix rule was documented; the partial truncation of the last folded
entry was not. Anything that carries tree bits solely to feed a future round's key
needs only the consumed prefix — relevant to any store-design record and to
emit-side key computation at rounds 3–4.

</details>

### The linear lane: words 1–6 are XOR-shift-linear in seed words 1–6 — and the record that exploits it still loses
<details>
<summary>Details</summary>

The mix rewrites only word 0 and `combine` is XOR+shift, so **words 1–6 of every
element at every round are GF(2)-linear in its leaves' seed words 1–6** — no seed word
0 and no mixed word ever reaches them. All the nonlinearity (the additive mix) lives in
one sequential 64-bit word-0 lane. Same 200 k-trial identity test as above, elements
entering rounds 2–5.

Consequence: any rebuild can split into a stored 8 B word-0 checkpoint plus a linear
lane of 6 siphashes per leaf with **zero mixes**. Built as `MXBM_PAIR_W0` (pair record
16 → 24 B carrying the child's full post-mix word 0; round 2's rebuild drops from 14
hashes + 3 mixes to 12 hashes + shifts; registers and shared byte-identical to the
shipping build; KAT green incl. the speculative-entry goldens; yield identical).
**Measured: a loss at every operating point** — +3.8 % at stock, +1.6 % at
120 W + rung, +0.9 % at 100 W + rung, interleaved ABBA, miner loop.

The mechanism is the single-issue result below: the trade swaps ~300 ALU instructions
per element for ~4 memory instructions (the staging loses its single LD.128 to three
scalar loads, the emit gains two stores) plus 537 MB/solve — and instructions plus
bytes outbill hidden-at-stock/starved-at-floor arithmetic everywhere. The loss
shrinking monotonically toward the floor shows the arithmetic saving is real and
growing; it never crosses the overhead. A 2-u64 + side-plane packing that keeps the
LD.128 would halve the overhead and might flip the deep floor by under a percent —
open, low value. The implementation is removed from the tree (a measured loss earns no residency); it is preserved as a patch in the maintainer's campaign records and can be recreated from this section. The linear-lane fact itself stands independent of this implementation and is
load-bearing for store-design records.

</details>

### sm_89 issue is single-slot: instruction PLACEMENT is not a lever, only count is
<details>
<summary>Details</summary>

The r2 census shows the ALU pipe (LOP3/SHF/IADD3) at 69.9 % busy with the FMA pipe
(IMAD) mostly idle, which reads like a two-pipe balancing opportunity: constant-amount
rotates re-encode as IMAD-by-2^k pairs on the idle pipe. Built as a standalone siphash
throughput microbench, three encodings, SASS-asserted (reference 88 SHF / 57 IMAD;
full conversion 1 SHF / 192 IMAD; half conversion between), encodings
correctness-checked against each other, internal ABBA:

| encoding | stock | locked 800 MHz |
|---|---|---|
| funnel-shift (shipping) | 90.5 Ghash/s | 27.5 |
| all rotates as IMAD | −14 % | −14 % |
| half converted | −2 % | −2 % |

**The loss is identical at both clock regimes and tracks total instruction count, not
per-pipe pressure: the schedulers issue one instruction per cycle regardless of
destination pipe.** Pipe utilization percentages are occupancy of the pipe, not of the
issue slot. Closes the rotate/add re-encoding family onto the FMA pipe in either
direction; count-*reducing* encodings (PRMT-fused shift-OR folds, wider LOP3 LUT
fusion) are untouched by this closure. Corollary applied elsewhere the same day: the
`MXBM_PAIR_W0` loss above, and the floor pricing of every instruction-adding trade.

The three encodings measured here were all count-*increasing* (145 instructions became
193), which left a count-neutral move to the FMA pipe formally untested. Census closed it
for the adds — ptxas has already made that move on 95.9 % of the population, see
[the carry-consuming IMAD](#the-carry-consuming-imad-is-already-on-the-fma-pipe-96-percent-of-it)
— and on 2026-09-08 the rotates were measured too, in the shipping kernels rather than a
microbench. `rotl64` as two `mul.wide.u32` by 2^b, whose OR folds into the XOR that
follows every rotate in SipRound: round 2's SASS goes 24,566 → 1,922 `SHF`, +22,336
`IMAD`, +2,924 `LOP3` where the OR met an add instead of an XOR — ALU-pipe instructions
**−30 %** at **+3.8 %** total count. KAT 3/3 × 15, eight 30 s arms: **round 1 +5.5 %,
round 2 +1.6 %**, ranges non-overlapping. Taking 19,000 instructions off a pipe reported
70–75 % busy bought nothing; the loss tracks the count and the wide multiply's latency.
The pipe-utilisation figure is not the binding constraint at 8 warps per scheduler; the
issue slot and the dependency chain are.

</details>

### The tail-family closure is stock-scoped: MATCH_FIRST wins at the floor
<details>
<summary>Details</summary>

`MXBM_MATCH_FIRST` (build the chain from the staged key, rebuild only elements the
walk will read — skipping the 12.8 % of a group that is alone in its chain slot) was
measured a net loss at stock (+0.4 %: `mlist` costs r1 its fifth block, the compaction
needs a barrier). Re-priced where the rebuild does not hide in stalls — flag
SASS-asserted (+~640 B shared per round, r1 48 → 64 registers), KAT green, miner loop,
ABBA, rung held:

| | base | MATCH_FIRST | Δ |
|---|---|---|---|
| 120 W + 5001 | 66.60 ms | 65.95 | **−1.0 %** |
| 100 W + 5001 | 86.85 | 85.50 | **−1.6 %** |

At the floor the skipped rebuilds bill at full issue rate (~1.4–1.8 ms of the 11 ms
rebuild item) and outbuy the occupancy cost. Ship shape: dual instantiation behind the
same observed-power gate speculative entry already uses (`kSpecMinPowerW` pattern).
The stock closure stands unchanged; its scope was the operating point.

**Composed with (17,0), same day: ≈ additive.** MATCH_FIRST + `MXBM_BB=17` together,
same protocol: **−2.7 % at 120 W + rung (67.45 → 65.65 ms pair means), −2.4 % at
100 W (87.55 → 85.45)** — the two mechanisms (rebuild skip, rescan removal) do not
overlap. The pair is one gate away from shipping.

</details>

### (17,0) at the floor reproduces on the current build — the disarm's re-arm condition is met
<details>
<summary>Details</summary>

The (17,0) auto-selection was disarmed when its floor prize failed same-day
reproduction (2026-07-31). Those measurements ran with speculative entry riding
round 4; since 2026-08-04 spec entry is off below 130 W, which changes exactly the
block population the geometry alters. Re-measured on the shipping binary
(`MXBM_BB=17` vs default, miner loop, ABBA, rung held), **two independent sessions**:

| | session 1 | session 2 |
|---|---|---|
| 120 W + 5001 | −1.5 % (67.10 → 66.10 ms) | −1.2 % (67.85 → 67.05) |
| 100 W + 5001 | *(discarded — co-tenant)* | −1.0 % (87.40 → 86.50) |

Every bracket ordered the same way; one 100 W block was discarded outright for a
co-tenant process during one arm. This is the "reproduced crossover on a CURRENT
build" the disarm text names as the re-arm condition: setting `kRbLowPowerW` to
~130 W (matching the spec-entry gate) re-arms the selection for cards that fit the
(17,0) footprint (8.35 GiB).

</details>

### The switching-height re-pricing: a store-everything design with a re-derived round 1 wins the whole 100–150 W band

*Superseded 2026-08-13 by [the h=1 build-out](#the-h1-pipeline-built-and-killed-the-caps-currency-is-l2-sectors-not-dram-bytes):
the bounds below did not survive the real build.*

<details>
<summary>Details</summary>

The full-pipeline eco mock (2026-08-04) priced a store-everything design at the flat
64 B/element schedule and found it alive only at the deep floor. Two additions
re-price it; both ran on an extended mock (`k3_pipe_h`, campaign probes; calibration
gate: the unmodified arm reproduces the 08-04 mock within 2–3 % and its first,
scalar-load build was rejected by that gate — the vectorization trap recurring).

**1. Switching height h = 1** (store the seed layer as 8 B `(key, idx)` records,
re-derive in round 1, store everything downstream — the point between the shipping
design's h = 2 and the store design's h = 0): −21 % of the design's traffic for one
extra derive pass. Measured against the flat-schedule arm: **−15 % at stock, −19 % at
140 W + rung, −17 % at 120, −7 % at 100** — h = 1 dominates h = 0 at every operating
point, sitting on the 5001 rung's roofline at 120–140 W (278 vs ~280 GB/s).

**2. One real streaming match round** (sorted-runs match, one block per bucket, 512
threads, ~5.6 KB shared, full-key pair enumeration, real `combine` + `apply_mix` per
child, partner gathers L2-served; child population ≈ n, the Wagner design point).
Its cost over the mock round: **hidden at stock and at 140/120 W + rung (±0.1–0.35 ms,
within bracket spread); +4.9 ms per round at 100 W + rung**, where issue is scarcest
and the naive per-pair gather bills.

Composed bound for a real h = 1 streaming pipeline (h1 floor + 4 × match cost), against
the shipping solver at its best rung per cap: **−13 % at 140 W, −26 % at 120 W, −11 %
at 100 W** — and ~parity at stock on the bound (no stock claim; the bound has no
refined match). This replaces the 08-04 verdict: the design's strongest band is the
**middle** of the capped range, not the deep floor, and 140 W is alive. The 100 W
figure is pessimistic-side (the probe's match is a first cut). Band-by-band
head-to-head numbers and the build recommendation live in
`docs-internal/PERF_LEADS.md`; remaining design unknowns before a build: the
back-ref/recovery scheme, spill policy, and the seed-to-round-1 handoff.

</details>

### The h=1 pipeline, built and killed: the cap's currency is L2 sectors, not DRAM bytes
<details>
<summary>Details</summary>

The h=1 design (previous section) was built to its own kill gates: a real round 1
(one block per bucket, derive-once into shared, exact low-byte classes, real
`combine` + `apply_mix`, full packed children + back-ref row 0), byte-for-byte
correct against the shipping pipeline — order-independent multiset fingerprints of
each round's output match exactly, KAT 3/3, drops 0 — and then measured
whole-pipeline against the same-tree h=2 probe, ABBA at each point.

**Verdict: killed.** With its best variant, h=1 is **+0.9 % at 120 W + rung,
+1.6 % at 140 W, +1.8 % at 100 W** against bounds of −26/−13/−11 %. Zero of the
bound survives. Three measured reasons the bound model was wrong:

1. **The mock priced traffic in DRAM bytes; under a cap the scarce currency is L2
   sectors × core clock.** h=1's round 1 materialises 72 B records where h=2
   writes 16 B pairs — +56 B/child of *scattered stores*, and the profiler shows
   the round pinned on L2 (70 %) with DRAM at 40 % and SM at 37 %. L2 runs in the
   core-clock domain, so exactly when the cap pulls the clock down, those sectors
   stretch: the round's marginal cost is 16.4 ms at 120 W + rung against 11.0 for
   the h=2 round 1.
2. **Bytes-for-hashes trades ~1:1 at the rung on the current baseline.** Round 2
   with no rebuild saves −4.2 ms/pass at 120 W; round 1's materialisation costs
   +5.4. The information moved one round earlier, and the round that writes it
   pays what the round that reads it saves — the ledger's compute-for-memory rule
   seen from the other side. Rounds 3–4 were already streaming in the shipping
   design, so there was nothing left downstream for h=1 to save.
3. **The baseline moved between the bound and the build** — the implicit-bits
   record (−3.2 %) shipped into the h=2 comparison the same week.

Two levers were priced on the way, one a keeper: **packing gi's high bits into the
record's spare payload bits deletes the side-plane word** — one scattered
read-modify-write sector per child gone on emit, one gather gone in the next
round's staging — worth **−2.4 ms whole-solve** on the h=1 arm (the shipping
packed record cannot take it: its plane carries 62 bits against 48 spare,
checked). And the keys-only-shared match shape (stage 14 B/element, gather full
records from L2 at pair time) **loses to full-record shared staging at every
point** (+5.5 % on the round's marginal at 120 W + rung) despite 1.5× the resident
threads: three scattered 8 B staging loads cost more sectors than one contiguous
staged read, before the gathers re-bill the records through L2. A wider-workgroup
variant (1536 threads/SM at 40 registers, zero spills) changed nothing — an
L2-bandwidth-bound round does not want more warps.

The arc cost two sessions of the ~three weeks budgeted; the staged kill gates did
their job. Probe code, diffs and per-point logs:
`docs-internal/campaign/2026-08-12-lowpower/` (M1, M2).

</details>

### Merging the two back-ref stores into one u64 is a wash: a store stream is priced by its scatter pattern
<details>
<summary>Details</summary>

The plane-deletion keeper above (one scattered RMW sector per child gone, −2.4 ms)
suggested a family: the shipping pipeline writes two 4 B back-refs per child
(`all_left`, `all_right`) in every fused round, the terminal and the recovery walk.
`MXBM_BRLR` merges them into one u64 store, `(right << 32) | left`. KAT 3/3 on both
arms (the recovery walk is exactly what the KAT gates), drops 0, warmed and
interleaved; a first 120 W set that drifted +1.5 % bracket-to-bracket was discarded
and retaken.

Wash at every operating point: stock 33.46 → 33.39 ms, 120 W + rung 70.55 → 70.57,
100 W + rung 88.45 → 88.32 — all inside the bracket spread (an early unwarmed
−0.8 % at 100 W did not reproduce). Display-attached session; deltas only.

Mechanism, which also bounds the plane-deletion win: back-ref writes index by
`cgi`, which a warp allocates near-sequentially — 32 lanes' stores to each plane
span ~4 L2 sectors and merge, while the u64 form spans ~8 sectors for its one
store. Same sector traffic, one instruction fewer, and instruction issue is not
the binding resource in these rounds. The h=1 plane paid because it was
bucket-scattered: slot-indexed stores with no neighbours to merge with. **A store
stream is priced by its scatter pattern, not its store count.** Patch:
`docs-internal/campaign/2026-08-12-lowpower/patches/brlr.patch`.

</details>

### Co-residency is closed for same-mix tenants: the binding budget is joules, not SM-seconds
<details>
<summary>Details</summary>

Every closed overlap mechanism shares one assumption — that co-tenants compete for the
*same* SMs. Streams fail on grid depth (662 waves), same-warp hosting fails because a
memory-stalled warp still holds its slot, co-blocks displace round blocks one for one.
**Disjoint SM partitions break the assumption**, and the driver supports them:
`cuDevSmResourceSplitByCount` + `cuGreenCtxCreate` (driver API 13.3 on this rig). A
feasibility probe (66 SMs split 24 + 42, register-only ALU spinner beside a 512 MB
streamer) overlaps at **99 % efficiency** — the partition mechanism itself is real.

**The solver gains nothing from it.** Two complete pipelines — real `fused_round`
kernels, KAT-gated per instance, drops zero throughout — resident together (packed
(15,2) ≈ 14.9 GiB), one per green-context partition (34 + 32 SMs), against one
pipeline on the whole device. ABBA, N = 100 solves per arm:

| packed (15,2), pair means | full (66 SMs) | half (34 SMs, solo) | dual (34+32) | dual, plain streams |
|---|---|---|---|---|
| 120 W | 81.2 ms | 84.6 (×1.04) | 84.2 (+3.6 %) | 82.3 (+1.3 %) |
| 100 W | 109.8 | 112.0 (×1.02) | 111.2 (+1.3 %) | 109.3 (−0.4 %) |
| stock | 35.4 | 56.4 (×1.59) | 35.2 (−0.7 %) | 35.4 (0.0 %) |

The quad record at 120 W reads the same (+3.3 % dual); energy agrees (+4.2 % J/solve
dual at 120 W). The saturation control that voided the earlier synthetic-tenant
attempt passes here: at stock, half the SMs cost ×1.59, so the kernels genuinely
contend for issue and the partition binds.

**The half column is the mechanism.** At 120 W a 34-SM partition does 96 % of the
full card's work: under a cap the card is power-bound, and removing SMs re-spends
the same watts as clock on the SMs that remain. A spatial partition therefore
divides SMs without dividing the scarce resource — both tenants draw one power
pool, each runs at ~half speed, and the aggregate cannot beat one pipeline. The
1.71× / 20.6 ms `max(SM-busy, DRAM-busy)` roofline prices SM-seconds and
DRAM-seconds; on this card the binding budget is joules at every operating point
(stock draw sits against the 285 W board limit), so the roofline is unreachable for
*any* co-residency of same-mix tenants — temporal or spatial. The partition itself
is pure rigidity against plain streams (+1–3 % everywhere).

What survives of the family is only a complementary-mix design — co-scheduled
halves that are deliberately ALU-heavy and DRAM-heavy, which is a different solver,
not a scheduling change to this one. The switching-height direction (previous
section) attacks the same watts by the honest route: fewer instructions and fewer
bytes per solve.

</details>

### The free list re-priced at the floor: bytes and the mix bill once the core slows
<details>
<summary>Details</summary>

Marginal-replay attribution (replay one round 9× in place, diff against the same
binary's base run; ablation arms bracketed inside full-binary brackets), 120/100 W
with the 5001 memory rung. The zero-delta positive controls (rounds 1 and 4 already
store 16 B) hold to +0.14/+0.10 ms at 120 W and +0.01/+0.02 at 100 W.

| marginal cost per solve | stock | 120 W + rung | 100 W + rung |
|---|---|---|---|
| `apply_mix` in r2 | 0.01 ms | 0.14 (≤ control band) | **0.49** |
| `apply_mix` in r3 | 0.06 | **0.71** | **1.02** |
| r2 payload store, 72→16 B | 3.47 | **5.01** | **6.67** |
| r3 payload store, 64→16 B | 0.34 | **2.75** | **4.15** |

Round marginals at the floor: r1 10.57/13.05, r2 22.51/27.74, r3 16.25/20.49,
r4 9.96/13.15 ms at 120/100 W (stock 5.12/10.17/9.49/5.48).

**The byte rate is the finding.** r2's 56 marginal bytes/element (1.88 GB/solve)
move at 1194 GB/s at stock — several times the card's peak, i.e. never what the
round waited on — but at 375 GB/s at 120 W and **282 GB/s at 100 W, which is the
5001 rung's own ~280 GB/s ceiling: at the deep floor the r2 store is pure bytes.**
r3's rate runs 4095 → 586 → 388 GB/s (an instruction-count component remains, per
the single-issue result above). The mix goes from noise to ~1.5 ms/solve across
r2+r3 at 100 W.

Two scope fences. These are attribution prices, not a shippable narrowing — the
ablation folds information away, and any real narrowing pays an
information-preserving mechanism (re-derivation or a redesigned schedule), which is
what the switching-height arms price. And the stock closure above stands at stock:
this section is its floor complement, and together they say the byte lever's value
is a function of the operating point — near zero at 285 W, roofline-priced at
100 W. That is the ablation-side confirmation of the switching-height premise.

</details>

### The first hardware census: r1 stalls on its own barrier, r3 idles 61 % of its lanes
<details>
<summary>Details</summary>

Nsight Compute over the four fused rounds (base clocks locked, card headless) —
the observability the CUDA port was built for, now actually exercised. The census
agrees with the marginal-replay instrument across methods: compute-bound r1/r2
scale by clock ratio onto the stock marginals (5.73 → 5.14 vs 5.12 measured;
11.17 → 10.02 vs 10.17), DRAM-bound r3/r4 are clock-invariant (9.72 vs 9.49,
5.74 vs 5.48).

| | r1 | r2 | r3 | r4 |
|---|---|---|---|---|
| SM throughput | **76.3 %** | 70.0 | 24.3 | 29.5 |
| DRAM throughput | 28.4 % (186 GB/s) | 45.8 (300) | **75.6 (496)** | **78.4 (514)** |
| issue slots busy | 55.1 % | 48.3 | 18.5 | 21.3 |
| eligible warps/scheduler | 2.55 | 1.79 | 0.28 | 0.31 |
| active threads/warp | 23.3 | 25.1 | **12.6** | 14.1 |
| executed instructions | 1.91 G | **3.29 G** | 1.07 G | 0.72 G |
| achieved occupancy | 82.7 % | 66.3 | 49.7 | 66.1 |

**At stock, r1's wall is eligibility, not issue saturation**: issue slots are 55 %
busy, and the largest stall is the round's own CTA barrier — 31.5 % of the 18.0
warp-cycles between issues (r4's barrier is worse still, 11.8 cycles). Lane waste
is now a number: 27 % in r1, 61 % in r3. Shared loads average a 2.0-way bank
conflict on a third of wavefronts; the scatter carries 43 % excessive sectors.
This closes the loop with the floor re-ablation above: arithmetic that hides in
eligibility gaps at stock bills at issue rate under a cap, because the cap shrinks
memory latency in cycles and eligibility rises. It also arms the deferred
warp-specialization lead with a measured target: the barrier share of the stall
budget. Full report in the maintainer's campaign tree.

</details>

### Populations are pinned at 2^25, and the occupancy tail prices a spill arena
<details>
<summary>Details</summary>

Per-round populations and full bucket-occupancy histograms (`MXBM_POP`), 21 solves,
bb=16 and bb=14, KAT-gated, zero drops.

**Populations do not run light.** Every round's output population is 2^25 within
+0.02 % on the mean and +0.47 % on the worst single solve. Set sizing cannot shrink
from population; the global elems/32 headroom (+3.125 %) is ~6× the observed worst
drift.

**The occupancy tail is thin, and that is worth 1.5 GiB.** The per-bucket cap ships
at mean+8σ+32 (1.407× mean at bb16) because a cap that drops loses solutions; the
worst bucket observed across 105 round-instances is mean+4.4σ. A dense cap of
mean+2σ (1.085×) would overflow only 0.01 % of elements (worst solve: 2,774) into a
global spill pool — a 64 K-slot arena (>20× the observed worst) costs ~9 MB against
−1.50 GiB of slot slack at (16,1), −0.5 GiB at (14,3). Zero drops becomes a
pool-full check instead of a tail bound. Unpriced, and the build's gate: the spill
path's time cost in the scatter and the match's arena pass.

**The record's address-redundant bits check out.** The packed r2→r3 record stores
400 work bits in 7 u64; the bucket address pins bb of them. 384 remaining bits at
(16,1) fit 6 u64 exactly — record 9 → 8 u64, side plane deleted, −0.36 GiB, with
(15,2) fitting via the leaf words' spare bits and (14,3) one bit short without
canonical-order leaf packing. Cost: a 17-bit cross-word repack on both sides of the
store, priced below — the expected sign was wrong, in the lever's favor.

Both levers are now BUILT and priced in the probe (KAT-gated byte-for-byte, zero
drops, spill counts deterministic across repeats):

| point (pair means) | base | implicit bits (−0.37 GiB) | + arena (−1.44 GiB) |
|---|---|---|---|
| stock | 33.33 ms | **32.23 (−3.3 %)** | 32.79 (−1.6 %) |
| 120 W + rung | 70.20 | **68.00 (−3.1 %)** | 68.83 (−2.0 %) |
| 100 W + rung | 88.92 | **85.84 (−3.5 %)** | 85.62 (−3.7 %) |

The arena's price is a fixed mechanism cost (+1.9 % stock, +0.7 % at 120 W — its
empty-pool control costs the same as its loaded pool), so it ships as a fit-ladder
rung where the alternative is the quad record's +14–15 %, never as a default.
Implicit bits is a default-candidate: the deleted plane word was one scattered
load plus one scattered store per element (~67 M memory instructions, 537 MB per
solve), which is exactly r3's MIO-issue stall currency. The "record narrowing
never pays" closure gains its precise scope here: narrowing that pays
re-derivation loses; narrowing that deletes memory instructions while keeping the
information wins at every operating point. Implicit bits is SHIPPED: both record
formats are instantiated and the solver picks at runtime (bucket_bits 16, packed);
in the miner loop it reads **33.05 → 32.0 ms/solve at stock (−3.2 %, 60.0 → 62.5
sol/s, yield identical)**, with kernel resources identical to the base record.
The set keeps its 9-u64 allocation for now — the footprint reclaim ships with the
small-card ladder work, together with the arena rung and the row-5 fix below.

</details>

### Back-ref row 5 was survivor-indexed but capacity-sized: −0.26 GiB on every row-bucket rung
<details>
<summary>Details</summary>

Rows 1–4 of the consolidated back-ref arrays are gi-indexed and need full capacity;
**row 5 is written only by the terminal round, at survivor indices bounded by the
1024 survivor cap** — and `recover` reads level 5 at those same slots. The allocation
and the footprint arithmetic sized it at capacity anyway: ~264 MB reserved for at
most 8 KB of use, on the CUDA and OpenCL row-bucket paths alike. Same redundancy
class as the sort path's dead 692 MB `all_lead` array and the round-3 `lead` field:
a row indexed by one thing and sized by another does not announce itself.

Fixed in the allocations and in `rowbucket_bytes` (the sort path keeps 5 × capacity —
its round-5 match writes per-child at gi); the drift-guard tests re-baselined to the
intended change. **Every row-bucket rung drops 0.26 GiB — (16,1) 7.46 → 7.20 GiB,
quad (14,3) 4.66 → 4.40 — the fast-path VRAM floor moves to ~7.25 GiB total, and two
card classes climb a rung: ~6.2 GiB free now hosts quad (16,1) and the 6 GB class
quad (15,2), each the faster geometry.** Gates: full suite green, GPU goldens on
both paths (recover is on the golden path). Metal keeps 5 × capacity until a Mac
session can gate the same change.

</details>

## Measured results, 2026-08-14

### The implicit-bits record at (17,0): the low-power gate gets the pack too
<details>
<summary>Details</summary>

The packed record's dropped bit count is the bucket-address width, and nothing in
the pack is specific to 16: with IMPB carrying the count as a template value,
(17,0) packs 400−17 = 383 bits into the same 6 u64 with one spare bit, and at
sm=0 the kept 7 low key bits are exactly the tab-hash bits — the sub-mask rescan
does not exist there. Gates: SASS byte-identical at bb=16 (the generalization is
a no-op on the stock config), KAT 3/3 with zero drops at (17,0), occupancy
contract extended (all four (17,0) variants identical to their 16-bit twins; r2
stays exactly on the 64-register cliff).

ABBA at (17,0), N=150, pair means:

| point | plain | packed | Δ | match-first | MF + packed | Δ |
|---|---|---|---|---|---|---|
| 120 W + rung | 65.96 | 63.42 | **−3.9 %** | 65.68 | 63.04 | **−4.0 %** |
| 100 W + rung | 82.56 | 78.92 | **−4.4 %** | 82.07 | 78.24 | **−4.7 %** |

Composes additively with match-first, like the (16,1) pair did. Shipped: the
low-power gate now selects match-first × packed-17, and the live miner reads
(45 s runs, headless, 5001 rung):

| cap | ms/solve | sol/s | J/sol |
|---|---|---|---|
| 100 W | 78.8 | 25.5 | 3.94 |
| 120 W | 63.0 | 31.9 | 3.76 |
| 140 W | 54.3 | 37.0 | 3.79 |
| 160 W | 46.6 | 43.2 | **3.70** |

(120 s pairs in mirrored cap order; arms agree within 1.5 %.) Stock is untouched
(31.7 vs 31.6 ms same-session). 3.70 J/sol at 160 W + rung is the new efficiency
record (was 3.79), 120 W sits at 3.76, and the efficiency curve is nearly flat
from 120 to 160 W — it flattened leftward.

The same change fixed a carveout drift: since the (16,1) pack shipped, the CARVE
list had named the unpacked non-match-first pair — instantiations that no longer
launch. Every launched variant now receives the max-shared preference; no
measurable stock delta from the fix on this driver.

</details>

### Duty-cycled average power: the concave-hull arbitrage is real and the idle floor eats it
<details>
<summary>Details</summary>

Time-sharing an efficient burst point with idle traces the concave hull of the
sol/s-vs-W curve, which beats every continuous point left of the efficiency peak
— IF idle is cheap. Measured with SIGSTOP/SIGCONT cycling around the benchmark
(the energy counter and wall clock span the stops): solve throughput tracks duty
exactly (12.58 vs 0.577 × 21.78 solves/s at 57 % duty), so the mechanism is real.
The efficiency is not: 4.42 J/sol at a 110 W average, against 4.16 for the
continuous 100 W + rung point (pre-17-pack figures, same session).

The killer is the idle intercept, decomposed: a resident CUDA context holds the
card at P2/P3 with the SM at full idle clock — 41–47 W, flat over 83 s, and
releasing the memory lock makes it WORSE (P2, mem 10251). The best managed gap
state (SM locked to the 210 MHz floor) is 31.7 W; break-even at a 100 W average
needs < 27.6 W. Ceiling with full context teardown (the no-context 14.2 W floor):
+7.5 % at 100 W, minus ~1–3 % per-burst realloc of the 7.2 GiB working set — thin,
and it shrinks to nothing by 120 W. Closed. Scope: this driver's resident-context
P-state behavior; a driver that parks a resident context at P8 reopens the
arithmetic, as does a card whose sub-peak curve sags more.

Residual fact: duty-cycling reaches average draws BELOW the driver's 100 W cap
floor (~80 W at ~4.8 J/sol) where no continuous configuration exists — a niche
capability, not an efficiency win.

</details>

### The stall structure offers no lever the shipped knobs do not already hold
<details>
<summary>Details</summary>

Two leads the hardware census armed, both closed by measuring the thing that would
decide them before building either.

**r3's idle lanes are the sub-mask filter, and its removal is already a shipped knob.**
The census reported r3 running 12.6 of 32 threads per warp. The source is the staging
loop's sub-mask filter: at sm=1 each bucket is swept by two blocks and each discards
about half of what it examines. A fresh profile of the packed build shows the matching
memory signature — r3 and r4 loads use **12.0 of 32 bytes per sector**, which is the
8 B word-0 scan striding over 64 B records. Deleting that filter and its scan is
precisely what geometry (17,0) does, and that is measured at **−1.0 to −1.5 % under a
cap**, shipping under the ≤130 W gate. A partitioned-emit variant at (16,1) — one slot
counter per sub-mask value, so each sweep reads a dense partition — would buy the same
package without (17,0)'s doubled bucket count, so its prize is bounded by that same
one-and-a-half percent, and not by the 61 % the lane figure suggests.

**Scattered stores using half of each sector is the hardware floor, not waste.** Every
round writes 16.2–16.7 of the 32 bytes per store sector, which the profiler bills as a
20–24 % opportunity. It is not one: 16 B is the widest store the ISA offers, so a
scattered 64 B record necessarily touches each 32 B sector twice. Merging across lanes
is separately closed — the emit is write-only, so a two-level reorder needs >780 GB/s
coalesced to break even. The one sector figure that did move is r2's, from 117 M
excessive to 94 M, which is the implicit-bits record deleting the side plane and an
independent confirmation of that mechanism.

**Warp specialization is blocked by r1's shared budget.** At `-lgc 800`, the regime
that matters, the CTA barrier is the top stall in r1 (5.6 cycles, 31.3 %, unchanged
from stock) and in *no other round* — r2, r3 and r4 all sit on long-scoreboard memory
dependencies (14.7–18.1 cycles, ~45 %), which producer/consumer warps do not address.
In r1 the barrier is required by the algorithm, not kept for convenience: a chain
cannot be walked until every element hashing to its slot has been inserted, and
insertion order across a bucket is arbitrary, so consumers cannot trail producers
within a bucket. Overlap therefore needs double buffering across buckets — a second
staging area of ~19 KB — against r1's **448 B** of shared headroom at five blocks per
SM, on a round where 640 B of match-first bookkeeping already costs the fifth block.
And the imbalance the barrier exposes is what match-first's compaction already
addresses, at a measured −1 % under a cap.

Scope for both: this round shape and this card's 100 KB of shared memory per SM. A
part with a larger shared budget, or a round that stages less, reopens the
double-buffer arm.

</details>

### Three small closures: kWG 288, PRMT rotates, uniform-datapath offload
<details>
<summary>Details</summary>

**kWG 288** (one 92 %-full pass over the 264-element group instead of 256 + a
thin second pass): occupancy survives — r1 keeps 5 blocks/SM, every round
unchanged — and it still loses +0.4 % at stock, +0.2 % at 120 W + rung, same sign
in every bracket. The thin second passes were nearly free: predicated-off lanes
and one extra barrier cost less than the wider group's slower barrier. With the
census showing r1's stalls 31.5 % on its own CTA barrier, this closes the cheap
end of the barrier family: the bill is the WAIT, not the pass count, and only the
producer/consumer warp redesign attacks the wait. (The chain build already rides
the staging loop — the stage→build barrier was fused out long ago.)

**PRMT for byte-aligned rotates**: the emitted SASS has zero PRMT and all 64-bit
rotates are already 2-SHF funnel pairs — the shift+OR fusion PRMT would provide
is already fused, so PRMT is a same-count re-encoding, and instruction placement
is not a currency on sm_89 (single-issue). Null by audit.

**Uniform-datapath offload for r2**: freed vector registers buy nothing — r2 is
shared-capped at 4 blocks/SM (23616 B × 5 > the carveout), so the 64-register
cliff is not the binding resource at the shipped shape. Null by arithmetic.

Emit store width audited on the way: every fused round already emits 4 ×
STG.E.128 — the minimum for the record.

</details>

### The reach composition, built: the implicit-bits reclaim and the arena as a rung
<details>
<summary>Details</summary>

The two footprint components measured on 08-13 are now in the solver, and the ladder
carries the result. Both are CUDA-only; OpenCL and Metal pass `allow_impb` and
`allow_arena` false and every figure they see is what it was.

**The implicit-bits allocation was still reserving a plane nothing writes.** The pack
shipped on 08-13 deletes round 2's 9th word, but `alloc_geometry` kept sizing set 0 at 9
u64 per slot because `impb` was decided *after* the geometry. It is now decided *with* it
(`rb_impb_ok`, shared by the footprint arithmetic and the allocator), and the stride comes
from `fb_set_stride`, so the two cannot disagree. (16,1) goes 7.20 → **6.84 GiB** and
(17,0) 8.09 → **7.67**; the miner's own occupancy on the card reads 7962 → **7590 MiB**,
which is `nslots × 8 B` to the megabyte. Nothing else moves — the ladder's other rungs are
quad, and the quad record has no plane to delete.

**The overflow arena is now a rung, and no longer a probe flag.** `ARENA` joined `MFIRST` and
`IMPB` as a template parameter with both variants instantiated, so the plain kernels are
byte-for-byte what they were and the occupancy contract says so on every non-arena row.
Dense per-bucket caps (mean + 2σ + 32, so 605 slots at (16,1) against the tail bound's
743) with one 65,536-slot overflow pool per record set, laid out behind the bucket records
in the same buffer so a single slot index addresses both regions and no store changed.
`arena_link` threads the pool onto per-bucket chains between each producer and its
consumer; the consumer walks its bucket's chain once into a 128 B shared list.

*No round loses a block to it*, which is the fact the rung stands on: the shared list is
128–136 B and round 1 had 448 B of headroom, so its fifth block survives (19144 B of
19456); r2 holds 4 blocks on exactly 64 registers, r3 3, r4 4. Registers move only on the
plain record (r3 56 → 60), which is shared-bound anyway.

Speculative entry is off on these rungs, in the other direction from the low-power gate: a
card takes an arena rung because it is short of memory, and speculation's dedicated entry
buffer is 0.36 GiB of exactly that.

Gates: KAT 3/3 and `drops == 0` on every arm, **and a non-zero spill count** — on an arena
arm those two readings mean opposite things, since a pool that never filled means the
overflow path never ran (`MXBM_DROP_STATS=1`; ~56 spills per solve at (16,1), reproducing
to the element across repeats, which doubles as a determinism check).

| (16,1), 3 interleaved ABBA blocks, 45 s arms, speculation off in both | ms/solve |
|---|---|
| tail-bound cap | 32.37 |
| dense caps + pool | 32.92 |
| | **+1.70 %** |

−1.07 GiB of records for +1.7 %, and −1.43 GiB of card occupancy once speculation's buffer
goes with it (7590 → **6126 MiB** in the miner). That price is a fixed mechanism cost, not
a spill cost — the 08-13 probe's empty-pool control cost the same as its loaded pool — so
it ships as a rung and never as a default.

**What the ladder does with them.** Eleven rungs now, still ordered by measured time:

| CUDA rung | footprint | was |
|---|---|---|
| packed (16,1) | 6.84 GiB | 7.20 |
| packed (16,1) + dense caps | **5.77** | — |
| quad (16,1) | 5.02 | 5.02 |
| quad (16,1) + dense caps | **4.29** | — |
| quad (15,2) + dense caps | **4.13** | — |
| quad (14,3) + dense caps | **4.03** | 4.40, the floor |

**And one number stood between that floor and a card.** Availability cannot call
`cudaMemGetInfo` without creating a context on every enumerated device, so it sizes
against TOTAL VRAM less a fixed allowance — which was a flat 1 GiB. Measured on this card,
the driver holds 408 MB of 16376 with nothing running and this process's context 214 MB,
both roughly fixed: ~0.4 GiB of that allowance was pure margin, and on a 5 GB card it was
the whole difference between an offered device and a refused one. It is 640 MiB now, from
the measurement. Being optimistic there is safe by construction — the constructor re-asks
against *free* memory, steps down the ladder on a real allocation failure, and a throw is
reported per-GPU with the OpenCL path as fallback.

So a 5 GB card runs at all, where before nothing on the ladder fit. The other classes
climb a rung only where a desktop was holding enough of the card to push them off one:
8 GB back onto (16,1), 6 GB back onto quad (16,1). Idle and headless they were already
there. 4 GB is still out of reach, and what it would take is on the record in
[HW_REQUIREMENTS](HW_REQUIREMENTS.md#what-is-left-below-it): the record sets are
3.00 GiB at the floor and the back-ref rows 1.03, so the two named unbuilt levers are an
octo (8-leaf, 32 B) round-3 record and moving or narrowing the references.

</details>

### The octo record: the last thing round 3's output had left to give
<details>
<summary>Details</summary>

**Round 3's output is 96 % information-dense as stored**, which is what made the next step
obvious and the step after it impossible. 376 work bits + 25 of lead + 64 of leftContrib
+ 26 of `gi` is 491 of the 512 the 8 u64 provide, so there is no packing lever left on it
at all — only re-derivation.

Eight seed indices determine a round-3 output element: it combines two round-3 inputs,
each determined by four. So the work words, the lead (which IS leaf 0) and the leftContrib
(a mix of the eight leaves over a zero element, needing no work state) are all derivable,
and the record becomes key + 8 × 25 + `gi` = **250 bits in the 256 that 4 u64 give**. Three
leaves straddle a word boundary; there is no layout that avoids it.

`rebuild_r4` is two `rebuild_r3` calls combined at Lout(3) = 376 and mixed at Lmix(4) =
376 — twenty-eight siphash rounds, against fourteen for the quad record one round up. It
sits in the all-lanes loop for the same reason both other rebuilds do.

Measured at quad (14,3) + dense caps, 25 s arms, KAT-gated with zero drops and the CPU
verifier reporting 2.05 verified solutions per solve:

| | ms/solve | on the card |
|---|---|---|
| quad + dense caps | 45.0 | 4354 MiB |
| + octo | **62.1** | **3236 MiB** |
| | +38 % | −1.09 GiB |

Two things pay part of that back, and both are worth stating because both invert a
finding from a different operating point:

- **Match-first now defaults ON here.** What it skips for an element alone in its chain
  slot is the eight-leaf rebuild, which is most of the round: −6.5 % (67.8 → 63.4 ms),
  against a loss at stock and a −2.7 % win under a deep cap. Three operating points,
  three different reasons, same variant.
- **Asking ptxas for a third resident block pays, unusually.** Left alone the rebuild
  takes 128 registers, which is exactly 2 blocks/SM. `__launch_bounds__(kWG, 3)` takes it
  to 80 registers and 88–96 B of spill per thread — and measures **−2 %** rather than the
  loss a launch bound usually is. The occupancy contract is what makes that visible: it
  printed the spill instead of letting the third block look free.

Round 3's own side is free — same kernel, a narrower store, and the 80-register cliff it
already sat on.

**And it makes three of the five back-reference rows dead.** Recovery walks references
from level 5 down to level 1 to reach a survivor's 32 seed indices — but a level-3 element
IS its eight leaves under this record, and set 1 is never written after round 4 reads it,
so those leaves are still resident when `recover` runs. Round 4 names its parents by SLOT
rather than by `gi`, the walk stops two levels deep, and rows 1–3 are neither written nor
allocated. **No `gi`-to-slot map is needed** — the idea's stated blocker turned out to be
avoidable by moving what the reference names one level up.

The leaf order is right by construction, with no luck involved: round 3 builds `ctree` and
writes `all_left`/`all_right` from the same `leftPos`/`rightPos`, so the record's eight
leaves are exactly the left-to-right order the walk would have produced.

| quad (14,3) + dense caps + octo | ms/solve | on the card |
|---|---|---|
| references in five rows | 62.1 | 3236 MiB |
| references in two | **61.1** | **2444 MiB** |

−0.77 GiB *and* −1.6 %, the time coming from the 0.8 GB/solve of reference stores that
rounds 1–3 no longer issue. `recover`'s octo form also loses its explicit DFS stack — 64 B
of local memory down to none — because a two-level walk needs no stack at all.

**Where the ladder ends up.** Three octo rungs at the bottom, offered only with quad and
dense caps, because a card that can host the packed record has no use for them:

| rung | footprint | ms/solve |
|---|---|---|
| quad (16,1) + dense caps + octo | 2.04 GiB | 54.9 |
| quad (15,2) + dense caps + octo | 1.95 GiB | 57.0 |
| quad (14,3) + dense caps + octo | **1.90 GiB** | 60.6 |

**1.90 GiB clears BeamHash III's stated 3 GB minimum and the card class behind it**, from
2.4× over where this started. Verified on this card by simulation and not by
arithmetic alone: `--keepfree 13100` leaves 2868 MB usable, which is what a 3 GB card
offers, and the ladder picks quad (16,1) + dense caps + octo unaided and mines at
36.3 sol/s with 2.01 verified solutions per solve.

One correction falls out of it. The availability allowance was set at 640 MiB on the
reasoning that the driver holds ~408 MB and our context 214 MB. `totalGlobalMem` reads
15.598 GiB against nvidia-smi's 15.99 — it equals `cudaMemGetInfo`'s total exactly — so
the driver's carve-out is ALREADY excluded from the figure availability subtracts from,
and the only real cost is the 0.214 GiB context. The 640 MiB stands as margin; the
arithmetic behind it did not.

**And the reference rows were the last reader of `gi`.** With rows 1–3 dead, `gi_of` is
called from exactly two places — the next round's left/right tiebreak, and `ref_of`, which
round 4 no longer uses — so every record's `gi` is a tiebreak token and nothing more. The
slot orders the staged elements just as well, and the consumer computes it anyway to load
the record. Only one of the three records shrinks by dropping it (the pair record is 2 u64
either way, the octo record 4), but the quad record's 24 + 4 × 25 = 124 bits fit 2 u64
instead of 3 — and round 2's `gi_alloc` then has no consumer at all and is not issued,
which is 33.5 M atomics a solve.

| quad (14,3) + dense caps + octo | ms/solve | on the card |
|---|---|---|
| references in two rows | 61.1 | 2444 MiB |
| + `gi` retired from round 2 | **60.6** | **2164 MiB** |

−280 MiB is `nslots × 8 B` exactly. The tiebreak change is sound, though not proven by
construction: it only fires when two elements share their first leaf, and their child then
has a duplicate index and cannot be part of a solution — so valid solutions never see it,
but the goldens are what confirm that.

**1.90 GiB is this design's floor.** Six u64 per slot, and no record carries a field
nothing reads: set 0 is round 4's output, which the terminal round needs whole, and set 1
is round 3's eight leaves. Below this needs a structural change — the last reference rows
off the card (0.26 GiB), or streaming / in-place layer reuse, which is the only lever that
attacks holding two layers at once.

</details>

### The OpenCL ladder was answering against the wrong number
<details>
<summary>Details</summary>

**A ladder is only as good as the figure it is asked about.** `rb_pick_geometry` passed
the card's TOTAL VRAM with a flat 1 GiB slack; `rowbucket_viable` then checked the rung it
returned against the budget's USABLE figure — driver free, less the reserve. On anything
but a large card the two disagree by construction: the ladder replied packed (16,1),
viability refused it at 7.20 GiB, and the budget fell through to the sort-path divisor,
which cannot host a full seed layer. The device was then dropped with "too little memory".

So the practical floor was **~8.9 GiB usable — an 11–12 GB card** — while the arithmetic,
the tests and this document all said 4.40 GiB. The reach work of 2026-08-01 (ladder-
authoritative budget, bucket-half split, allocator step-down) was correct and had been
correct for a fortnight; nothing fed it the right number. The lesson is the one this
document keeps relearning one level up: the geometry test pinned `rb_geometry_for` and
passed, because it called the pure function directly with figures the caller never used.

Passing `b.usable` with slack 0 — `rowbucket_geom.h`'s own stated contract for a caller
that has already taken the reserve off — is the whole fix. Measured by simulating each
class with `--keepfree`, ~2.0 CPU-verified solutions per solve at every rung:

| usable VRAM | before | after |
|---|---|---|
| 7.09 GiB (8 GB class) | refused | packed (15,2), 34.9 ms |
| 5.18 GiB (6 GB class) | refused | quad (16,1), 39.0 ms |
| 4.45 GiB | refused | quad (14,3), 43.5 ms |
| 4.30 GiB | refused | refused |

**The 8 GB class sits ON the boundary**: (16,1) needs 7.20 GiB of the 7.20 such a card
reports free, so anything else resident takes it a rung down. That is what the step-down
is for, not a regression — but it is why the class is quoted as packed instead of as a
particular rung.

</details>

### The overflow arena, ported: +3.6 % for a 4.04 GiB floor
<details>
<summary>Details</summary>

The mechanism is CUDA's (see *the reach composition* above): per-bucket capacities fall
from a mean + 8σ tail bound to mean + 2σ, and a bucket that fills appends to a pool behind
the bucket records instead of dropping. What differs here is the split. A record set over
`CL_DEVICE_MAX_MEM_ALLOC_SIZE` arrives as two bucket-halves, so each half carries its own
pool and its own cursor, the two summing to the pool the footprint arithmetic budgets —
CUDA, with no allocation ceiling, needs none of that.

It is a **build option, not a kernel parameter**. The row-bucket program is already
compiled after the ladder settles (the same reason `GEO_BAKED` works), so an arena rung
compiles the pool paths in and every other rung compiles the kernels it always had. The
earlier note here — that OpenCL's lack of templates would force a second program build or
a runtime flag paid in local memory — priced a problem that the existing compile ordering
had already solved.

Two pairings are **refused instead of mis-indexed**, and both are about a buffer with no
pool region behind it: the packed r2→r3 side plane (so the arena requires the quad record)
and the speculative-entry set (so speculation is off on an arena rung). The first throws;
the second silently declines, since speculation is already best-effort.

Cost, interleaved ABBA at quad (14,3) with speculation off in **both** arms — the first
attempt had it on in one, which priced the arena and the lost speculation as one number:

| | ms/solve |
|---|---|
| arena off | 44.2, 44.3 |
| arena on | 45.8, 45.9 |
| | **+3.6 %** |

against CUDA's +1.70 % at (16,1). A card that takes an arena rung also gives up
speculative entry, so the delivered cost there is nearer 5 %.

The gate was 3/3 goldens and zero drops on all three counters, plus the **positive
control that matters**: the pool spilled 92 elements per solve at (16,1). An arena that
never spills reads exactly like one that works, and would have measured the plain rung
under another name.

| usable VRAM | without the arena | with it |
|---|---|---|
| 4.45 GiB | quad (14,3), 43.5 ms | **quad (16,1) + arena, 39.9 ms** |
| 4.11 GiB | refused | **quad (14,3) + arena, 45.9 ms** |

So the floor goes **4.40 → 4.04 GiB**, and — because the freed slots buy a *finer*
geometry instead of a coarser one — a card at 4.45 GiB gets **8.3 % faster** while using
less memory. That inverts the usual direction of a reach lever and is the reason the
ladder is ordered by measured time, and not by footprint.

</details>

### The octo record on OpenCL: the record is proven, round 4 is not
<details>
<summary>Details</summary>

**Not shipping.** `rb_pick_geometry` passes `allow_octo=false`, so the ladder cannot
reach these rungs; `MXBM_OCTO=1` is the only way in. Recorded here because the parts that
are settled are the expensive ones to rediscover, and because the failure is localised
far enough to be worth stating.

Settled, by positive controls that ran over a full solve and reported zero mismatches
(`-DLDS_OCTO_CHECK=1`, compiled out by default):

- `rd_elem4` applied to a record's eight leaves reproduces **every work word** round 3
  stored for that child — checked against round 3's own `c[0..5]` in the packed build,
  where both sides are available.
- `oc_contrib` reproduces the leftContrib round 3 stored, from the same eight leaves.
- The octo record round-trips: the leaves it carries re-derive the key it carries.
- Rounds 1–3 emit **key multisets identical to the packed build's**, bucket for bucket,
  so the 16 B quad record and the octo record are both being written and read correctly.

Not settled: round 4 places **~21 % of its children in a bucket whose index does not
match the child's own key** — 111,146 of 525,283 in a 1024-bucket sample — and the
terminal round then reads ~18,000 spurious survivors against the true 3. The stored
record and the address it was stored at disagree, which makes it a store-address fault
rather than an arithmetic one, and the arithmetic checks above say the same.

Three explanations are already excluded, each by measurement:

| candidate | why not |
|---|---|
| the lead tiebreak resolving differently (gi order is not build-stable) | `LEADTIE_PROBE` counts **17** ties in an entire solve |
| the overflow pool | 92 elements spill per solve, and the affected slots are ordinary bucket slots |
| nondeterminism | the packed build reproduces its own round-4 output **exactly** across runs |

The one structural difference left unexplored: under octo, set 0's stride is 2, so round 2
and round 4 write the same addresses, where in every shipping configuration they do not
(3 against 2). Nothing has yet shown that to be the mechanism.

</details>

### The stock census on the shipping kernels: every round is against a roofline
<details>
<summary>Details</summary>

The previous hardware census was taken at base clock on the pre-implicit-bits build, and
its conclusions are quoted throughout this ledger. Re-taken at stock on the shipping
kernels (`./cuda/profile.sh`, two passes, 2026-08-14). Cross-instrument gate: the ncu
durations sum to 32.06 ms against the marginal-replay attribution's 32.39, agreeing per
stage within 1–3 %.

| | entry | r1 | r2 | r3 | r4 | term |
|---|---|---|---|---|---|---|
| duration (ms) | 2.48 | 4.82 | 7.70 | 8.44 | 4.16 | 0.77 |
| achieved clock (MHz) | 2762 | 2768 | 2758 | 2759 | 2758 | 2753 |
| **pipe ALU %** | **98.8** | **76.9** | **74.9** | 17.9 | 21.8 | 33.0 |
| `sm__throughput` % | 98.7 | 76.8 | 74.8 | 21.5 | 25.8 | 51.4 |
| DRAM rd / wr (MB) | 0 / 257 | 271 / 525 | 539 / 2130 | 2148 / 2131 | 2149 / 267 | 270 / 2 |
| DRAM % of peak | 15.8 | 25.2 | 52.9 | **77.4** | **88.6** | 54.2 |
| ld sectors/request | 1.00 | 3.30 | 5.39 | **17.10** | **17.10** | 6.06 |
| bank conflicts (M) | 0.0 | 50.3 | 76.9 | 93.9 | 54.4 | 9.8 |
| `no_instruction` % | 0.10 | 0.66 | 0.50 | 0.52 | 0.69 | 2.26 |
| barrier stall % | 0.0 | **31.2** | **28.9** | 28.7 | 27.7 | 15.0 |
| long scoreboard % | 2.3 | 10.7 | 15.0 | **46.4** | **43.3** | 35.6 |

Re-taken 2026-08-16 on the shipping record at `--clock-control none`. **Three things
about the previous edition were wrong, and all three came from how it was collected
rather than from what it measured.**

**It ran at base clock.** `cuda/profile.sh` passed no `--clock-control`, and ncu defaults
to `base` — 2337 MHz against the 2758 the card actually holds. The correction is not
uniform, which is the part that matters: at the higher clock entry, r1 and r2 shrink
**~15 %** while r3 and r4 shrink **~3 %**, because a DRAM-bound round does not scale with
core clock. So the old table systematically overstated the ALU-bound half — 57 % of the
solve at base clock against **54 %** at the real one.

**`sm__throughput` is the integer pipe only where the integer pipe is busy.** It agrees
with `sm__pipe_alu_cycles_active` to a tenth on entry, r1 and r2, and diverges by 4, 4 and
**18 points** on r3, r4 and the terminal round — where the composite is reporting the LSU
sub-unit instead. Every ALU figure quoted for a DRAM-bound round came from the composite.

**The profiled binary was not the shipping one.** `MXBM_IMPBITS` defaults to 0, so the
standalone bench ran the 9-u64 set-0 stride with its side plane and wrote reference rows
for rounds 1–3. Built with `-DMXBM_IMPBITS=1` its per-round DRAM traffic lands within 1 MB
of the compulsory table on every line, which is the direct confirmation that deleting the
rows and narrowing round 4's record reached DRAM and not just L2.

Two claims the re-take settles against their proposers. **The instruction cache is not a
currency**: the collected `imc_miss` metric is the *immediate constant* cache, and the
instruction-fetch stall `no_instruction` — never collected before — reads **0.10–0.69 %**
on every shipping round against a predicted 2–6 %, so there is no −0.7 ms behind a
code-size reduction and the dead `SOUT` template parameter has no speed case. And
**round 4 is at 88.6 % of DRAM peak**, four points past the 83.9 % that was being quoted
as this card's achievable ceiling; that figure was the *old* terminal round's, and the
terminal round now sits at 54.2 % because an 8 B record left it 272 MB to move.

<details>
<summary>The previous edition, at base clock on the pre-implicit-bits bench</summary>

| | entry | r1 | r2 | r3 | r4 | term |
|---|---|---|---|---|---|---|
| duration (ms) | 2.57 | 5.01 | 8.98 | 9.07 | 5.45 | 0.98 |
| instructions (M) | 1083 | 1901 | **3330** | 1120 | 716 | 272 |
| issue slots busy % | 59.9 | 55.2 | 53.4 | 18.2 | 19.7 | 43.1 |
| pipe ALU % | **98.7** | **76.4** | **78.3** | 17.9 | 17.7 | 28.0 |
| pipe FMA % | 9.6 | 8.3 | 8.8 | 3.8 | 2.9 | 4.0 |
| pipe LSU % | 3.3 | 30.2 | 21.2 | 22.6 | 27.3 | 60.4 |
| DRAM % of peak | 15.3 | 32.5 | 49.9 | **76.5** | **82.7** | 84.0 |
| threads/warp | 32.0 | 23.4 | 24.8 | 12.7 | 14.1 | 19.2 |
| eligible warps/sched | 5.45 | 2.53 | 2.18 | 0.27 | 0.29 | 0.66 |
| top stall | math 43 % | barrier 31 % | barrier 28 % | long sb 45 % | long sb 45 % | long sb 39 % |

</details>

**`sm__throughput` IS the integer pipe.** For every round the SM-throughput figure and the
ALU-pipe figure are the same number to two decimals (78.29 for round 2). On Ada a
sub-partition has 32 FP32 lanes and 16 INT32, so integer ops issue at half rate — and this
solver is integer end to end. That is the sharper form of the ledger's "only count is a
currency": the currency is specifically **ALU-pipe slots**, and they are the top SM
sub-unit in every round that is not DRAM-bound.

The solve splits three ways, and each part is against something:

- **entry (8 %) is at the ALU roofline**, 98.7 %. Pure siphash, 32 threads/warp, nothing
  else running. There is no headroom here at all.
- **r1 + r2 (45 %) are ALU-pipe-bound** at 76–78 %, with issue slots only half busy.
  Round 2 alone executes 3.33 G of the solve's 8.42 G instructions, roughly two thirds of
  which is the re-derivation of its two seed indices.
- **r3 + r4 (45 %) are at the memory roofline**, 76.5 % and 82.7 % of DRAM peak, with the
  ALU pipe idle at 18 % — and their traffic is exactly compulsory (see
  [the traffic table](performance.md#the-memory-traffic-is-compulsory)), so there are no
  bytes left to remove short of the octo record, which costs a flat 16 ms.

**A correction to the earlier census's stall reading, which does not change its verdict.**
That census recorded the CTA barrier as the top stall *only* in round 1, with r2/r3/r4
long-scoreboard-bound, and that sentence is the stated premise of the warp-specialization
closure. On the current build the barrier is also the top stall in round 2 (27.9 %, above
math at 19.8 % and long scoreboard at 11.3 %). The premise is therefore wrong for round 2
— but the closure survives on better evidence: with four resident blocks the SM issues
from the other three while one waits, which is why round 2 reaches 78 % ALU-pipe occupancy
*despite* a 28 % barrier stall. The barrier is a warp-level wait, not a lost SM cycle,
which is exactly what the kWG-288 kill test measured from the other side (+0.4 % at stock).
Only three barriers per pass are unconditional; the spill block's are inside a
group-uniform condition the common path skips.

</details>

### Both count-reducing encodings the single-issue closure left open are zero
<details>
<summary>Details</summary>

The [single-issue closure](#sm_89-issue-is-single-slot-instruction-placement-is-not-a-lever-only-count-is)
ends by exempting two things from itself: *"count-reducing encodings (PRMT-fused shift-OR
folds, wider LOP3 LUT fusion) are untouched by this closure."* PRMT was audited to zero
the same week — every 64-bit rotate is already a 2-SHF funnel pair. LOP3 fusion is the
last one, and the stock census makes it worth asking, because it identifies the ALU pipe
as the binding SM resource in exactly the rounds this would touch.

The prima facie case is strong. In round 2, **680 of 794 LOP3s carry LUT `0x3c`** — plain
two-input XOR — against only 28 of the three-input `0x96`. A LOP3 computes any three-input
boolean function in one instruction, so every `(a^b)^c` written as two `0x3c` ops is an
instruction ptxas left on the table.

It left none. A dataflow pass over the shipped SASS — per basic block, looking for a
`0x3c` LOP3 whose destination has exactly one consumer, itself a `0x3c` LOP3 — finds
**zero fusable pairs** in `entry_scatter`, round 1 and round 2 alike. Positive control:
the same detector finds the pair in a synthetic block written to contain one, so the
zeros are measurements.

The mechanism is the algorithm, not the compiler. SipHash's round is `a += b; b = rotl(b);
b ^= a` — every XOR has exactly two inputs and its result is consumed by an add or a
rotate, never by another XOR. There is no three-input XOR anywhere in the hot loop to
fold, and the 680 two-input LOP3s are 340 64-bit XORs that are irreducibly two-input.
With this the exemption list is empty and the instruction-count family is closed on the
encoding side; only a different *schedule* could change the count, and that is the h=1
question, measured dead.

Opcode mix for reference (shipping kernels, `cuobjdump -sass`):

| | LOP3 | SHF | IMAD | IADD3 | ISETP |
|---|---|---|---|---|---|
| round 1 | 409 | 374 | 227 | 214 | 79 |
| round 2 | 794 | 750 | 411 | 382 | 82 |

</details>

### Lane density in rounds 3 and 4 cannot pay at stock: the rounds are not issue-limited
<details>
<summary>Details</summary>

Rounds 3 and 4 run at **12.7 and 14.1 active threads per warp** — the sub-mask filter,
which the earlier census quantified as 61 % idle lanes in round 3 and which has stood as
the largest-looking inefficiency in the profile.

The stock census prices it at zero. Both rounds are **at the memory roofline** — 76.5 %
and 82.7 % of DRAM peak — with traffic already exactly compulsory to within 0.3 %, and
their **ALU pipe idling at 17.9 % and 17.7 %** with issue slots 18–20 % busy. Lane density
is an issue-side property: filling those lanes produces more work per instruction, and
there is no instruction-side deficit to recover. The bytes would not change, and the bytes
are what these rounds are waiting on.

That is a sharper bound than the earlier one, which put the prize at ~1–1.5 % by reference
to (17,0)'s floor measurement. At stock it is zero — and the only mechanism that removes
the filter, the (17,0) geometry, is separately measured at **~23 % worse at stock**
(40.9 ms against 33.1), which is why it ships behind the ≤130 W gate.

</details>

### The solutions-per-solve multiplier is the algorithm's, not the solver's
<details>
<summary>Details</summary>

`sol/s = solves/s × solutions/solve`, and the ledger has only ever optimised the first
factor. The second multiplies every published figure, so it is worth knowing whether any
of it is being lost — a solution recovered costs no time at all.

Three independent lines say none of it is, all on the shipping build at stock:

**Nothing is dropped.** `MXBM_DROP_STATS` over 1,892 consecutive solves: all four
counters zero on every one of them. That includes the two that could lose a solution
silently — the chain walk's 64-step cap (`drops[3]`), and the output-bucket overflow
(`drops[2]`). Group-cap overflow cannot lose anything by construction: `MXBM_SPILL`
splits the group on one more key bit and redoes it instead of dropping the tail.

**Nothing invalid is produced.** `MXBM_VERIFY_STATS` over 28,305 solves: 56,783
candidates, **56,781 verified**. The two rejects are duplicate-index solutions (one at
round 4, one at round 5) — the inherent Equihash case where a leaf appears in both
subtrees, at a rate of 3.5e-5. No ordering violation, no collision failure, no non-zero
final. There is no found-versus-verified gap to reclaim.

**The partition cannot separate a colliding pair.** The bucket fixes the key's top `bb`
bits, the sub-mask its bottom `sm`, and the 128-entry perfect table hashes the middle 7:
`bb + sm + 7 = 24`, the whole key. Two elements with equal keys therefore land in the same
bucket, the same sub-pass and the same chain — which is exactly why the walk's key
comparison is a tautology under `MXBM_PERFECT_TAB`. Every colliding pair is emitted once.

Positive control, because a counter reading zero may mean "no fault" or "never ran": with
`kRbDenseSigma` forced to −6 the same counters report **5,767,168** bucket drops and the
yield collapses from 2.01 to 0.00, while the single surviving candidate still verifies.
Both instruments respond; the zeros above are measurements, not silence.

**The measured multiplier is 2.006 over 28,305 solves**, standard error 0.008 — 0.7 sigma
from the theoretical C(2^25, 2) / 2^48 = 2, and 3.1 sigma away from the 1.98 this ledger
used to quote. The same run reads 31.8 ms/solve, 63.1 sol/s, 284.4 W, 4.51 J/solution.

Two consequences beyond closing the lead. The earlier claim that runs under a few thousand
solves read **high** is not supported: today's short runs scatter 1.98–2.02 around 2.006
with no sign of bias, and the standard error alone (0.033 at 1,900 solves) accounts for
the spread. And the bar moves — 53 sol/s ÷ 2.006 is 26.4 solve/s, so the reference miner's
solve time is **37.9 ms, not the 36 the old 1.9 multiplier gave**.

</details>

### The octo record's two halves, priced apart: round 4's rebuild is a flat 16 ms
<details>
<summary>Details</summary>

The octo record had only ever been measured as a bundle — round 2 emitting the 16 B quad
form, round 3 rebuilding from it *and* emitting the octo one, round 4 rebuilding again —
and booked as a percentage (+38 % at quad (14,3) + dense caps). The two halves are
independent in the arithmetic: `fb_round_stride` takes round 2's width from `quad` and
round 3's from `octo` and never crosses them. Only the kernel dispatch tied them together,
and `fused_round`'s octo emit branch already accepted `LM_EMIT`, so round 3 could read the
packed record and emit the octo one with no rebuild of its own.

The reason to expect that to pay at stock: round 4 runs at **78 % of DRAM peak with 70 %
of its SM throughput idle**, and the octo record halves what it reads, 64 B → 32. A
compute-for-memory trade aimed at the one round with a memory roofline under it and an
empty ALU above it.

ABBA-bracketed, 40 s arms, the miner's own loop, headless, KAT-gated with the CPU
verifier reporting 1.99–2.02 verified solutions per solve in every arm:

| what round 3 reads | without octo | with octo | delta |
|---|---|---|---|
| the packed record, (16,1) + dense caps | 32.7 ms | 48.8 | **+16.1 ms** |
| the quad record, (16,1) + dense caps | 39.0 | 55.0 | **+16.0 ms** |

*(Scoped: the +16 ms below is the PLAIN octo record. The
[w0 checkpoint](#the-w0-checkpoint-on-the-octo-record-the-mixes-go-and-the-record-does-not-grow)
later took it to +13.7 ms in the same 32 B. The verdict is unchanged -- a reach lever, never
a speed one -- and everything below still holds at its own date.)*

**The cost is a constant, not a fraction, and it does not depend on round 3 at all.**
Twenty-eight siphash rounds per element in round 4's all-lanes loop bill ~16 ms wherever
they are run — which also re-reads the bundled figure: the +38 % at the floor rung and
the +49 % here are the same 16 ms over two different baselines, not two different trades.

So the record halving buys round 4 at most ~2 ms of DRAM time against 16 ms of issue, and
the idle ALU is not the currency: this round is single-issue-bound like every other, and
70 % idle throughput is not 70 % of a free rebuild. **The octo record is a reach lever and
can never be a speed lever, at any operating point on this card.** Its ~1.1 GiB (plus the
0.8 of back-reference rows that die with it) is bought at a fixed 16 ms, and a packed-input
octo rung would sit at 4.9 GiB / 48.8 ms — dominated on both axes by quad (16,1) + dense
caps at 4.29 GiB / 39.0 ms, so nothing on the ladder wants it.

Reverted; the twelve kernels it instantiated are not in the shipping binary. What it left
behind is a fixed bug: `MXBM_OCTO` set the octo flag alone, so on a card whose ladder picks
a packed rung the forced geometry wrote 16 B records into a set sized for 64 and handed the
chain walk null pointers. The flag now brings the quad record and the arena with it.

</details>

## Measured results, 2026-08-15

### The pipe census: r1 and r2 are 22 points under the ALU roofline, and it is warp supply
<details>
<summary>Details</summary>

*(`ncu` executed-instruction counts per pipe and per-SASS stall attribution over
`cuda/pipeline 1`, headless, shipping flags. The harness runs a KAT solve plus N nonces,
so every figure is halved. Cross-instrument gate: the totals reproduce the stock census
at 8423 M warp instructions against 8422 M.)*

| per solve | entry | r1 | r2 | r3 | r4 |
|---|---|---|---|---|---|
| warp instructions (M) | 1083 | 1901 | 3331 | 1120 | 716 |
| ALU pipe (M) | 892 | 1345 | 2476 | 570 | 340 |
| FMA pipe (M) | 174 | 293 | 556 | 240 | 110 |
| **ALU-pipe util** | **98.7 %** | 76.4 | 78.3 | 17.9 | 17.7 |
| **issue util** | 57.7 % | 55.2 | 53.4 | 17.6 | 18.7 |

**The census's ALU column is derivable from instruction counts alone.** An sm_89
sub-partition has 16 INT32 lanes, so a warp ALU instruction holds the pipe two cycles, and
`2 × alu / (sub-partitions × elapsed cycles)` returns r3's 17.9 % and r4's 17.7 % exactly.

**Entry is ALU-pipe-bound, not issue-bound** — 98.7 % of the INT pipe with 42 % of its
issue slots empty. This *re-scopes* the single-issue closure instead of contradicting it:
on the ALU-bound stages the currency is ALU-pipe instructions specifically, and LSU and
branch instructions issue in slots the INT pipe cannot use. Trading an XOR for a shared
load pays there; the reverse does not. What is left of entry's own work cannot leave the
INT pipe: every SipRound operation is a LOP3, an SHF, or the **carry-producing** low half
of a 64-bit add, and IMAD can consume a carry but not produce one. The high halves have
already gone — see
[the carry-consuming IMAD](#the-carry-consuming-imad-is-already-on-the-fma-pipe-96-percent-of-it).
Entry is also at its
instruction floor: 1074 SASS instructions for 7 siphash calls, 153 per call against a
hand-derived minimum of 155 (48 IADD3 + 48 LOP3 + 48 SHF, the two `rotl64(·,32)` free as
register swaps), with `IADD3 165 + IMAD 173 = 338` matching the 336 predicted 64-bit adds.

**r1 and r2 are bound by neither pipe nor issue.** Entry proves 98.7 % is reachable on
this card; r1/r2 sit at 76.4 / 78.3 with issue only half full, so the gap is stall.
Closing it is r1 5.08 → 3.93 ms and r2 9.30 → 7.38 ms, **−3.07 ms, −9.7 % of the solve.**

> **That is the distance, not a target.** The mechanism this section goes on to name —
> resident blocks — tops out at **≈ −0.66 ms** on this card: it would need ~8 blocks and
> sm_89 allows 48 warps/SM, i.e. 6 at 256 threads, with entry already there. See
> [the corrections](#three-corrections-to-figures-this-ledger-quotes).

The stated basis of the warp-specialization closure — *four resident blocks keep the SM
issuing while one waits* — does not survive the warp-supply metric:

| | entry | r1 | r2 | r3 | r4 |
|---|---|---|---|---|---|
| `warps_eligible.avg.per_cycle_active` | **5.45** | 2.53 | 2.18 | 0.27 | 0.29 |
| resident warps / sub-partition | 12 | 10 | 8 | 6 | 8 |

Entry's stalls are the healthy pair, math-pipe throttle 43.1 % and not-selected 43.4 % —
surplus warps queued behind a saturated pipe. r1/r2 hold under half entry's eligible
supply, so the four blocks cover the wait only partially and the wait does cost SM cycles.
Per-SASS attribution over the shipped r2 gives barrier **35.9 %** of not-issued cycles,
against math-pipe throttle 31.2 (the floor), long scoreboard 13.6 and wait 10.0.
**Barrier stalls are booked against the instruction after `BAR.SYNC`**, so searching SASS
for barrier opcodes finds 0.6 % and misses the effect; the cost lands on four sites, each
executed once per warp.

**The deficit is warp SUPPLY, and one resident block is worth 5.3 points of ALU pipe.**
`MXBM_FCAP=352` drops r2 from four resident blocks to three and changes nothing else —
the executed ALU count is 2475 M against the shipping 2476 M, so the perturbation is pure
latency-hiding:

| r2 | blocks/SM | eligible warps/cycle | ALU pipe | issue | ms |
|---|---|---|---|---|---|
| FCAP 320 *(ships)* | 4 | 2.18 | 78.3 % | 53.4 | 9.30 |
| FCAP 352 | 3 | **1.55** | **73.0 %** | 50.1 | **9.56** |

KAT 3/3 and drops 0 in both arms; the solve reads 32.24 → 32.79 ms. Removing a quarter of
the warps costs 5.3 points of ALU pipe and 2.8 % of the round, and the barrier stall
barely moves (27.9 → 28.8 %) — so the barrier is a *symptom* of thin warp supply, not the
cause. Extrapolating the same slope, entry's 98.7 % would need roughly eight resident
blocks — **which this card cannot provide**: eight 256-thread blocks is 64 warps/SM
against sm_89's 48. Six is the ceiling, entry already sits there, and the reachable move is
+2 blocks in r2 and +1 in r1. Note also that this perturbation's cycle cost (7.2 %,
implied by the ALU-utilisation change at constant executed ALU) and its time cost (2.8 %)
differ, because the 285 W cap hands back clock when occupancy falls: a block is worth
**2.8 % of the round in time**, and the ALU-points slope overstates what is cashable.

**That closes the warp-specialization family for a better reason than the ledger had.**
Producer-consumer warps redistribute work among the warps a block already has; they do not
add warps, so they cannot repair a warp-supply deficit. The same argument disposes of the
barrier-skew mechanisms independently of their own nulls (kWG 288 at +0.4 %, MATCH_FIRST).
What is left is occupancy, and occupancy is closed twice over — shared memory caps r2 at
four blocks and the register file caps it again if kWG grows.

**What the measurement leaves behind is an exchange rate for shared memory.** A fifth
resident block in r2 needs its footprint under 20 KB against ~24.8 KB today — **4.8 KB,
worth ~0.26 ms (0.8 % of the solve)**, and a sixth about the same again. **Shared memory is
necessary and not sufficient**: r2 also uses 64 registers, exactly the four-block cliff, so
a fifth block needs **≤ 48 registers as well as ≤ 19456 B shared** and a sixth ≤ 40 and
≤ 16000 B. Freeing the 4.8 KB alone leaves the register file returning four. That prices every
future shared-memory narrowing in r1/r2, which previously had no exchange rate at all.
It also says why the near misses do not pay: `lwork` is 320 × 7 × 8 B = 17.9 KB of the
24.8, and even narrowing the staged element by a whole u64 — the shared-memory analogue of
the implicit-bits record — frees only 2.5 KB and still lands on four blocks.

</details>

### Store-versus-derive is one exchange rate, and h=2 is a structural optimum
<details>
<summary>Details</summary>

The h=1, quad, `R2_FULL` and w0-checkpoint nulls are four measurements of a single
constant. From h=1's own attribution — r1's materialisation +5.4 ms against r2's rebuild
−4.2 ms over 2^25 elements — **a siphash call costs 8.9 ps of solve time and a byte
written and read back through a bucket scatter costs 4.35 ps.** A call is therefore worth
about two bytes while producing eight: **re-deriving beats storing by ~3.9× on this card**,
in whichever round the trade is made.

One structural fact then fixes the switching height without further measurement: **an
element is its own minimal checkpoint.** Rebuilding a level-k element requires its two
level-(k−1) parents, which together carry more bits than the element does, so there is no
cheaper intermediate to store. Deriving earlier (h=1) buys arithmetic that is 3.9× cheaper
with scattered stores; deriving later (quad at r3, octo at r4) doubles the rebuild —
7 → 14 → 28 siphash calls — against a byte saving that only falls linearly, which is why
round 4's eight-leaf rebuild prices at a flat +16 ms. **h=2 is a true optimum rather than
an empirical one**, and that is why both neighbours measured worse.

</details>

### Round 3 stores a work word round 4 never reads — worth 268 MB, and nothing in time
<details>
<summary>Details</summary>

`Lout(4)` is **288**, not 352 — round 4 is the spec's deliberate 88-bit drop. In
`combine`, `out.w[4]` is masked to its low 32 bits, which deletes the only term `x[5]`
contributes (`x[5] << 40` lands in bits 40–63), and `out.w[5]` is zeroed outright. **Work
word 5 of the 64 B r3→r4 record cannot reach round 4's output.**

Two independent tests, each with a positive control.

**The identity test** (`campaign/2026-08-15-structural/probes/deadword.cpp`): 200 000
randomized trials per word, perturbing one word in *both* parents so the collision
structure is preserved, at every round's real `(Lmix, Lout)`. r1, r2 and r3 have all seven
input words live; **r4's word 5 is dead**; r5's words 1–4 are dead. The r5 row is the
control — it independently rediscovers the terminal round's dead work words, which the
2-u64 thin record already ships, so the zeros are measurements rather than silence.

**The pipeline test**, `MXBM_POISON_W=N`, which replaces work word `N−1` of the r3→r4
record with `0xDEADBEEFDEADBEEF` at round 3's emit — poison rather than zero, so a reader
must corrupt rather than coincide:

| poisons | KAT |
|---|---|
| work word 3 | survivors 2, verified 0, **goldens 0/3** |
| work word 4 | survivors 2, verified 0, **goldens 0/3** |
| **work word 5** | survivors 3, verified 3, **goldens 3/3**, drops 0 |

So the word is unread by the whole downstream pipeline, not merely by `combine`.

**It is not a speed lever.** The prize was priced by assuming time scales with bytes on a
round at 76.5 % of DRAM peak. That assumption is false on this stream. `MXBM_ABL_EMIT=3`
narrows round 3's emit 64 B → 16 B — halving its write *sectors*, ~1.07 GB/solve — with
the stride, the bucket distribution and the read side untouched, and r3's own marginal read
by in-place replay (`MXBM_ROUND_REPS="3:9"`) so the corrupted downstream cannot contaminate
it. ABBA, 10 solves/point, patch asserted by `STG.E.128` falling 45 → 33:

| arm | r3 marginal |
|---|---|
| baseline | 8.968 ms |
| emit → 16 B | 8.613 ms |
| | **−0.355 ms, −3.96 % of the round** (6.8× the bracket spread) |

Removing ~1.07 GB of sector traffic bought 0.355 ms — an implied **~3000 GB/s, about 4.6×
the card's 656 GB/s peak.** A stream cannot beat the bus, so r3's writes are not on the
critical path at anything like the roofline rate: **half a round's write sectors are worth
4 % of its time.**

Dropping the dead word narrows the stride 64 → 56 B, which is −12.5 % of write sectors,
not −50 %. Scaling the measured slope: **≈ −0.089 ms**, a quarter of the instrument floor,
and a 7-u64 stride would also forfeit the 128-bit accesses (`slot × 7 × 8` is 8 B-aligned).
The word is worth **268 MB of record set** and is a reach lever only.

This is the same result as [bytes are nearly free](#bytes-are-nearly-free-per-element-work-is-not),
arrived at from the write side, and it explains why the implicit-bits record won for
deleting a memory *instruction* rather than for deleting bytes. **Price any narrowing on
this path in memory instructions and sectors, never in bytes.** On that basis a 48 B
r3→r4 record is still worth pricing — 6 u64 is 16 B-aligned and needs 3×LD.128 against
today's 4 — but its case is the instruction, not the byte. It needs 384 bits for 435 bits
of live content; w0's 17 address-implied key bits and w4's dead top 8 bits (live only to
bit 55) give 25, leaving it **26 bits short** unless `cgi` is retired from the record.

</details>

### Three corrections to figures this ledger quotes
<details>
<summary>Details</summary>

**The warp-supply target is not reachable, and it is not −3.07 ms.** The
[pipe census](#the-pipe-census-r1-and-r2-are-22-points-under-the-alu-roofline-and-it-is-warp-supply)
extrapolates that entry's 98.7 % ALU utilisation "would need roughly eight resident
blocks". Eight 256-thread blocks is 64 warps per SM; **sm_89 allows 48** —
`cudaGetDeviceProperties` on this card reports `maxThreadsPerMultiProcessor` 1536. Six
blocks is the ceiling at this block size and `entry_scatter` already sits there, so the
reachable move is **+2 blocks for r2 and +1 for r1**.

The census's own two figures for the `MXBM_FCAP=352` perturbation are also inconsistent
under `time ∝ 1/ALU%`: with executed ALU held constant it reports **7.2 % more cycles but
only 2.8 % more time**. Cycles and time can only diverge if the clock moved, and every
stage draws the 285 W board cap, so three blocks draw less power and the governor returns
~4 % of clock. Priced on the directly measured *time* slope, the whole route is
r2 +2 blocks × 2.8 % of 9.30 ms and r1 +1 × 2.8 % of 5.08 ms = **≈ −0.66 ms, 2.1 % of the
solve** — an upper bound, since the slope is sublinear near the top and entry's eligible
warps come from having no barrier and no memory dependency rather than from warp count
alone.

**A fifth block in r2 needs registers as well as shared memory.** A fifth block needs
**≤ 48 registers and ≤ 19456 B shared**; a sixth needs ≤ 40 and ≤ 16000 B. r2 uses **64
registers**, which by the occupancy model in `tests/test_cuda_resources.cpp` is exactly the
cliff for four blocks, so freeing shared memory alone leaves it at four. The two knobs have
never been moved together — `MXBM_MB_RD2` measured null *because shared capped it at four
anyway*, and the `MXBM_FCAP` sweep never went low enough to clear 19456 B. They have since
been moved together, and [the block is a loss](#r2s-fifth-resident-block-is-reachable-and-costs-010-ms).

**`drops[0]` had no writer, and the entry pass was reporting into the wrong counter.**
Every device increment targeted `drops[1..3]`, so the allocated, zeroed, `pair=`-printed
first counter was structurally always zero and **every `drops == 0` gate stated as four
counters was three**. The cause was not a missing increment but a misdirected one: entry's
scatter overflow incremented `drops[1]`, where it was indistinguishable from a group-cap
or terminal-staging overflow — three different capacities, one counter.

Fixed: one counter per capacity, **entry / stage / out / walk**, with the print relabelled
to match (it read `pair / bucket / gi / walk`, and two of those four names were wrong).
`cuda/pipeline`'s drop gate was ORing three of the four and now ORs all four.

**The positive control is `MXBM_CAP_SIGMA`**, which forces the tail bound in `fb_cap_for`
for every caller at once — it is read there rather than at the call sites because the
footprint arithmetic and the allocation both come through that function, and reserving one
cap while indexing another would corrupt rather than drop. A negative value undershoots the
mean, which is what makes each channel reachable:

| `MXBM_CAP_SIGMA` | entry | stage | out | walk |
|---|---|---|---|---|
| unset (8.0) | 0 | 0 | 0 | 0 |
| 0 | 9 289 | 0 | 30 074 | 0 |
| −6 | 5 898 240 | 0 | 2 | 0 |

The graded response is the point: entry and out move independently while stage stays zero,
which is the demonstration that the three capacities are now separately observable. KAT 3/3
on all fifteen geometries; the shipping path is byte-identical because the counter it
increments is zero either way.

This does not disturb the 2.006 multiplier, which is measured by verification rather than
by counters. What it unblocks is the yield axis, whose every arm is a deliberate loss that
could not previously be told from a bug.

**The knob also settles a standing question about the tail.** `cap = mean + 8√mean + 32`
budgets 8σ against a Poisson assumption, and the objection is that a round's input is not a
fresh Poisson draw but a sum of pair counts over parent buckets, so its variance should
scale with the fourth moment of the parent count — putting the real margin at 5–6.5σ.
`MXBM_OCC` over 41 solves × 65536 buckets (2.69 M draws per stage) says otherwise:

| stage | max occupancy | in Poisson σ |
|---|---|---|
| entry | 622 | 4.09 |
| r1 | 628 | 4.35 |
| r2 | 636 | 4.70 |
| r3 | 629 | 4.40 |
| r4 | 641 | **4.92** |

A Gaussian tail over 2.69 M draws predicts a maximum at **4.95σ** and Poisson's right tail
at λ = 528 is lighter still, so every stage sits at or below the prediction. Over-dispersion
of 1.2× would read 5.9 and 1.6× would read 7.9 — excluded at every round, and least of all
present in round 4 where the pair-count argument says it should be worst. **The 8σ margin
is a genuine 8σ**, and a Gaussian estimate of the spill rate at a tighter cap is
trustworthy, which is what the singleton-eviction family needs to price itself.

### The block-exit barrier is removable, and the barrier family is over-priced 10x

`fused_round_body` carries ten `__syncthreads()`, four of them live at stock. The one at
the bottom of `while (xp < nparts)` guards the *next* pass's `tab` clear and `gcount`
reset against this pass's still-running walk — and at stock there is no next pass:
`nparts` is 1 on every non-spilling bucket and `nsweep` is 1 because SUBPASS is off at
every shipping launch site. So it executes once, at block exit, with nothing after it.
`nparts` is a runtime value, so the compiler cannot see this.

```c
if (xp + 1u < nparts || sweep + 1u < nsweep) __syncthreads();
```

The guard is block-uniform — `nparts` is computed from `gcount` and `cnt8[]` after a
barrier, so every thread derives the same value — which is what makes skipping a barrier
legal rather than undefined.

**−0.048 ms, −0.165 %** (six arms ABBA-interleaved at 2600/10251, 45 s each, 0.06 % spread
per arm, `base` and `bar` ranges non-overlapping: 34.19–34.21 against 34.24–34.26
solves/s). KAT 3/3 on fifteen geometries, and separately on a `MXBM_FCAP=192` build where
the split path is live — positive-controlled, because at that cap the solve is 15 % slower
from the double staging, so the multi-part path provably ran and the guarded barrier
provably fired. Occupancy holds at three blocks/SM on every affected kernel; r3's registers
rise 54→56 and 56→60 with 24 to spare, and the octo r4 rebuild's stack falls 96→88 B. The
octo rung measures 17.68 solves/s on every arm of its own ABBA, so nothing is paid there.

**The number is the finding, and it is a tenth of what was predicted.** The proposal
reasoned from the stall census: barrier is 31.2 % of warp-active cycles in r1 and 28.9 %
in r2, above every other reason, so removing one of four live barriers should be worth
about a quarter of that — ~0.4 ms conservatively, up to 0.95. It is worth 0.048.

**A barrier's stall percentage is not its time cost when the SM holds other blocks.** The
census counts cycles a warp spends waiting at a barrier; those cycles are only wall clock
if no other resident warp can issue into them. r1 runs five blocks per SM and r2 four, so
the wait is largely covered — and the barrier this deletes is the one *least* able to cost
anything, because after it the block only retires. That is screen 9 read from the other
side: a resident block's marginal value decays because the machine is already covered, and
the same coverage is what makes a barrier cheap.

The consequence is a bound on a family. Deleting the one deletable barrier of four buys
0.048 ms, so the whole deletion route is ~0.2 ms at its most generous, and any proposal
that quotes the 28–31 % barrier share as headroom — narrower named barriers, a
work-stealing staging loop to equalise the warps each barrier waits on, generation tags to
retire the `tab` clear — is quoting a number that is already covered by residency. Price
them against 0.048 ms per barrier, not against the census percentage.

*(Kept despite being under the ~1 %-of-a-solve floor: it is three tokens, it is pinned by
six non-overlapping arms rather than inferred, it costs nothing anywhere, and the barrier
genuinely is not needed where it stood.)*

### Singleton-free staging: the prize is 0.85 ms, and the prepass that finds it costs 0.76

Inside a group the 128-entry table is a perfect hash — the bucket fixes the key's top `bb`
bits and the sub-mask its bottom `sm`, leaving exactly the seven that `hk = (key >> sm) &
127` selects — so a chain slot holds exactly the elements sharing a key. An element alone
in its slot has no equal-key partner: it emits nothing, no walk step reads it, and it is
nobody's ancestor. Dropping it is lossless **by construction**, which is what separates it
from every other element-skipping rule on the board. At 264 staged over 128 slots that is
`e^-2.06` = **12.8 %** of every group.

Slot occupancy is not known until every key has been seen, so `MXBM_SOLO` counts it in a
prepass over word 0 — whose low 24 bits are the key in every record format — and staging
skips the slots that come back with a single member. The counts live in `tab`, which the
chain does not need until the expand loop, so the pass costs **no shared memory**; `tab`
is reset to `kEmpty` after staging and before the chain claims it.

| arm | solves/s | ms/solve | Δ vs base |
|---|---|---|---|
| base (`MXBM_SOLO=0`) | 34.255 | 29.193 | — |
| prepass only (`=1`) | 33.383 | 29.956 | **+0.763** (+2.61 %) |
| prepass + filter (`=2`) | 34.360 | 29.104 | **−0.089** (−0.31 %) |

Twelve runs, ABBA-interleaved, four per arm, 30 s each at 2600/10251 headless. Arm spreads
are 0.02, 0.01 and 0.00 solves/s, so the three means are separated by 5–90× their own
noise. Rounds 1 and 2 only (`MXBM_SOLO_R=3`).

**The prize is −0.852 ms and the prepass eats 0.763 of it.** Both halves are worth having
as numbers. The prize sits *above* the tail family's own ceiling of ~0.7 ms across r1 and
r2, because skipping at stage time does more than skip 12.8 % of the rebuilds: it takes
`gcount` from ~264 to ~230, under `kWG`, which removes the ragged second block-pass of the
expand loop — the pass that runs 8 lanes of 256 while the rest wait at the barrier, priced
separately at −0.25 (r1) and −1.03 (r2). The cost is one scalar global load and one shared
`atomicAdd` per input element, plus two barriers at the measured 0.048 ms each.

**The positive control is the group size itself.** Building at `MXBM_SPILL=0` and
`MXBM_FCAP=64` turns `drops[1]` into `Σ max(0, n − 64)` — the staged count less a
constant — so the two arms differ by exactly the elements the filter refused: **25.14 M
against 20.63 M, a gap of 4.51 M**, against 4.30 M predicted for round 2 alone from
`12.8 % × 2^25`, the balance being round 1's own over-cap tail. The filter fires at the
predicted rate, and the KAT stays 3/3 over 22 configurations while it does — the
exponent-0 claim measured rather than asserted.

Registers, stack and shared come back **byte-identical on every kernel in the library**,
so blocks/SM is untouched and the contract in `test_cuda_resources` needed no re-baseline.
`BAR.SYNC` goes 474 → 506, which is the applied-assert: two barriers × the sixteen r1/r2
instantiations.

**It dodges all three of the costs that made [MATCH_FIRST](#the-tail-family-closure-is-stock-scoped-match_first-wins-at-the-floor)
a net loss at stock.** No `mlist`, because 128 counters fit in an array that already
exists, so no shared memory and no lost resident block. No compaction, because the filter
applies before `pos` is claimed, so the group stays in slot order and the rebuild keeps
its regular `lwork` stride. And it deletes the staging stores and the record load as well
as the rebuild, which skipping at derive time cannot.

**Round 3 is out of scope by arithmetic, not by omission.** Its record is 64 B against
r1's 8 and r2's 16, so a word-0 prepass there is a second sector stream through the round
at 88 % of DRAM peak rather than a re-touch of a sector the staging loop reads anyway —
and its own prize is the smaller one: skipping one record load in eight measured −0.163 ms,
because r3's binding stall is memory-instruction issue and a divergent skip issues anyway.
The prepass would have to be five times cheaper there than it is here to break even.

**What this arms is a global key census.**
The lever is now split into a −0.85 ms prize and a +0.76 ms question, and the question is
only ever "what does it cost to learn which slots are singletons". The producing round
already knows: equal keys are always co-resident, so round *r*'s emit sees the class
structure round *r+1* will walk. Anything that carries that across for less than 0.76 ms
collects the difference.

### The key census cannot move to the producer: 4.36 ms against the 0.76 it would replace

The singleton filter's census is [its expensive half](#singleton-free-staging-the-prize-is-085-ms-and-the-prepass-that-finds-it-costs-076) at +0.763 ms, and the
producing round already holds the child's key, having just computed the bucket from it.
Key multiplicity is global — the bucket is the key's top `bb` bits and the sub-mask its
low `sm`, so every element sharing a key lands in the same group, and "alone in its chain
slot" and "this key occurs once in the round" are the same statement. So round *r*'s emit
can record it: two bits per 24-bit key, sixteen keys to a word, 4 MB against a 48 MB L2.

**The census is exactly right and it costs 4.36 ms (+15.0 %)** — 5.7× the prepass it would
replace, four ABBA arms per side, every arm identical to the digit (34.36 against 29.88
solves/s). `MXBM_GCEN=1` writes the table and reads nothing, so that figure is the write
alone.

Correctness is not the problem, and the readback proves it. `MXBM_GCEN_STATS=1` counts the
table against Poisson(2), which is what 2^25 elements over 2^24 keys must give:

| | measured | `2^24 · P` |
|---|---|---|
| keys seen | 14 510 484 | 14 510 000 |
| keys with a partner | 9 966 786 | 9 966 000 |
| **keys seen once** | **4 543 698** | **4 542 000** |

Those 4.54 M keys *are* the partnerless elements, **13.5 % of the round** — and the
shared-memory prepass counted 4.51 M of them through an unrelated instrument. Two
independent routes to the same population.

**Moving a census from the consumer to the producer moves it from shared memory to
global, and that is the whole cost.** The lead argued from contention: 2^24 addresses at
~2 hits each against the emit's existing 2^16 at ~512, so the new atomic is 8× *less*
contended than one already being issued. Contention was the wrong axis. Lower contention
on a bigger table is worse, because 128 shared counters per block live in the SM and 4 M
global ones all funnel through L2, where consecutive lanes of a warp hit as many distinct
sectors as there are lanes. Per producer stage the census costs 0.87 ms against the
prepass's 0.38 ms per round, and it has to run on five stages rather than the two whose
consumers filter.

That kills the family rather than this encoding. The 8-bit-per-key variant trades 1.57
atomics per element for exactly 1 — 2.8 ms, still 3.7× the prepass. Four bits is not
available at all: a wrap at 16 reads as "singleton", the one direction that loses
solutions.

### The singleton filter pays 4.7x more under a cap than at stock

The [filter](#singleton-free-staging-the-prize-is-085-ms-and-the-prepass-that-finds-it-costs-076) shipped on a stock number, and the cap sweep taken straight after read
**slower at every cap below 220 W** than the previous sweep — a tilt, in the direction
NARROW6's null predicts for a lever that adds instructions. Capped rigs are a target, so
that had to be settled before anything else was built on top.

**It is not a capped-rig loss. It is a bigger win capped than at stock**, and the tilt is
not the kernel at all.

Three arms interleaved at 180 W in one session, `A B C C B A` twice, 45 s each: the
shipping build, a `-DMXBM_SOLO=0` build of the same tree, and a build of the tree as it
stood before either of the day's two kernel changes. All twelve runs drew 179.7–180.1 W,
and no two arms' ranges overlap.

| arm | ms/solve | sol/s | Δ vs shipping |
|---|---|---|---|
| shipping | **38.775** | 51.48 | — |
| `MXBM_SOLO=0` | 39.350 | 50.63 | the filter alone is **−1.46 %** |
| neither lever | 39.325 | 50.68 | both levers are **−1.40 %** |

So the filter is worth **−1.46 % at 180 W against −0.31 % at stock**, ~4.7× more under the
cap, which is what the instruction-deleting family has done twice before. Its net is
negative *instructions*: the prepass adds one scalar load per element and the filter
removes a record load, the staging stores, and the ragged second block-pass. The
block-exit barrier is separately **null at this cap** (+0.06 %, inside the arm spread),
against −0.165 % at stock.

**Below 130 W the filter is not in the binary path at all.** `kSpecMinPowerW` turns
match-first on there, and `SOLO` is `!MFIRST` — match-first claims `tab` for the chain in
the staging loop and reaches the same elements later. So the sweep's three *largest*
deficits, at 100/110/120 W, are in a band where the suspected lever does not run. A 120 W
ABBA confirms it: 63.4/63.5 ms shipping against 63.3/63.4 with the filter compiled out,
ranges overlapping. What the filter does below 130 W is nothing, so whatever the low band
needs, it is a match-first question and not this lever's.

### Compile-time geometry costs 0.26 percent at stock and 1.12 under a cap

`bucket_bits`, `submask_bits` and the two bucket caps are runtime kernel arguments, and
the innermost walk loop shifts by one of them and multiplies by another. That is the same
shape as [compile-time round constants](#compile-time-round-constants), the largest single
win in this project's history, so the geometry looked like the same lever one layer down.

It is not, and the reason is worth more than the lever: **a runtime scalar that reaches
the SASS only as an instruction operand is free.** The −27 ms win was never about
constants — it was about a *runtime loop bound* producing a dynamically indexed private
array, which cannot live in registers and went to local memory. Nothing about the geometry
does that.

Probed at its maximum strength: four assignments at the top of `fused_round_body`
hardwiring one rung, built twice — (16,1) with `cap` 743 and (17,0) with `cap` 425. Such a
binary is correct on its own rung and silently wrong on every other, so it cannot run the
ladder's KAT; it is gated instead on all four drop counters zero and on 2.00 / 2.02
verified solutions per solve, which the record layout could not produce if the hardwired
cap were off by one. That doubles as the applied-assert, alongside every
`c[0x0][0x160..0x16c]` reference disappearing from all four kernels.

| | shipping | compile-time geometry |
|---|---|---|
| stock, 284 W, headless | **28.948 ms** | 29.023 (**+0.26 %**) |
| 120 W cap, the (17,0) rung | **65.435 ms** | 66.171 (**+1.12 %**) |

Four interleaved arms a side at each point, no arm overlapping the other build's range.

**Why there was nothing to win.** Three separate reasons, each visible in the SASS:

- A kernel parameter is a **constant-bank operand**, encoded in the instruction. The walk
  loop's shift is `SHF.R.U32.HI R9, RZ, c[0x0][0x164], R9` — one instruction, exactly what
  the immediate form costs. The output slot is `IMAD.WIDE.U32 R4, R47, c[0x0][0x16c], R4`,
  the entire `cb * out_bucket_cap + cpos` as one 32×32→64 multiply-add, which is the
  cheapest encoding available whether the cap is a constant or not.
- The divisions are already gone. `bid / (1u << sm)` and `bid % (1u << sm)` are recognised
  as a shift and a mask without being told `sm` is a shift count.
- What geometry math is left, the compiler hoists into the **uniform datapath** —
  `ULDC.64 UR8, c[0x0][0x160]`, `USHF.L.U64.HI`, `UIMAD.WIDE.U32` — one copy per warp, in
  registers the vector datapath never pays for.

**And folding it costs.** Static instruction counts rise in every kernel: at (16,1) r1
1832 → 1912, r2 2560 → 2632, r3 840 → 896, r4 704 → 760; at (17,0) by 24 to 40 each.
Round 3's `BSSY`/`BSYNC` pairs go 13 → 19 — once the shift amounts are immediates the
scheduler un-if-converts and re-branches. The occupancy contract fails in 13 places, all
upward: r3 arena 58 → 62 registers, r4 47 → 52, and the octo r4 spilling 88 → 96 B of
stack. The 4.3× ratio between the capped and stock losses is the rule that killed NARROW6
read in the same direction: the cap prices instructions, and this adds them.

*Scope: sm_89, CUDA 13.3, driver 610.43.03, the two rungs measured. The probe was removed
— reproducing it is four assignments.*

### The carry-consuming IMAD is already on the FMA pipe, 96 percent of it

The [single-issue closure](#sm_89-issue-is-single-slot-instruction-placement-is-not-a-lever-only-count-is)
rests in part on the claim that a SipRound's operations are "none encodable as IMAD". The
shipped object holds 933 `IMAD.WIDE.U32`, so that phrasing is wrong, and the closure only
ever measured **count-increasing** conversions — 88 SHF becoming 192 IMAD. A count-neutral
one was never tested. Both objections hold. The conclusion still does not, and one
`cuobjdump -sass` pass is the whole answer.

| whole object | count |
|---|---|
| `IMAD.X` — carry-consuming high half on the FMA pipe | **14,160** |
| `IADD3.X` — the same job left on the ALU pipe | **607** |
| `IMAD.SHL.U32` — shifts moved off the ALU pipe | 1,737 |
| `IMAD.MOV.U32` — register moves moved off the ALU pipe | 12,118 |

**ptxas has already done the count-neutral conversion, to 95.9 % of the population.** Per
shipping kernel the remainder is: entry `IADD3.X 3` against `IMAD.X 157`, r1 6/161,
r2 3/270, r3 4/5, r4 3/3.

Entry is the only stage where it could pay — 98.7 % ALU-pipe utilization with the FMA pipe
idle — and its unconverted population is **3 instructions out of 1,080**, 0.28 % of one
stage, two orders of magnitude under this instrument's floor. The correct statement of the
carry claim is that **IMAD cannot *produce* a carry on sm_89**, only consume one, which is
why the low half of every 64-bit add stays `IADD3` and caps the conversion near half the
add instructions in the first place. And a count-neutral move would be worth nothing even
if a population existed: the schedulers issue one instruction per cycle regardless of
destination pipe.

### The shared bank conflicts are the chain walk's, and no layout reaches them

The standing claim: `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` reads 50.7/79.1/
97.8/60.2 M on r1–r4, and 287.8 M conflict-cycles ÷ 66 SM ÷ 2.6 GHz is **1.68 ms/solve**,
of which 0.76 ms sits in r1/r2 where no DRAM stall hides it. The lever proposed against it
is an XOR swizzle of the staged record. Re-taken on the shipping kernels at
`--clock-control none` the count reproduces — **58.8 / 79.4 / 94.0 / 54.3 M, 286.4 M
total** — so the arithmetic is not the error.

**The counter is honest, and it does not charge for the width of a 64-bit access.** A
four-kernel calibration (`lds_cal.cu`, 256 threads, 264 blocks, one shared store and one
rotated load per iteration):

| shared access | conflicts | wavefronts | time |
|---|---|---|---|
| 32-bit, stride 1 | **0** | 17.3 M | 191–199 µs |
| 64-bit, stride 1 | **0** | 34.6 M | 192.5 µs |
| 64-bit, **stride 7** — `lwork`'s own | **0** | 34.6 M | 192.7 µs |
| 64-bit, stride 8 | **242.2 M** | 276.8 M | **1512.9 µs** |

The two-wavefront split every 8 B access pays is not counted, and where conflicts *are*
counted they convert to time 1:1 — stride 8 takes 7.85× the time for 7.9× the wavefronts.
So conflicts→ms is the right conversion, **at a saturated pipe**: those four kernels run
`l1tex` at 99.7–100.0 % of peak.

**The solver's pipe is at a third of peak.** `l1tex__throughput` is **31.9 / 41.7 / 36.5 /
42.9 %** on r1–r4, and the stall a saturated shared pipe produces is not there either —
`mio_throttle` reads **0.14 and 0.33** instructions per issue-active on r1 and r2, against
a barrier stall of 3.61 and 3.15 in the same rounds.

**And the population is not the record layout.** `MXBM_LWORK_SOA=1` takes the staged
element from a 7-u64 stride to stride 1 — the strongest layout change available, and
strictly better than any swizzle — and the conflict count moves **0.2–1.7 %**: 58.79 →
58.82, 79.39 → 79.28, 93.97 → 92.96, 54.25 → 55.16 M. Applied-asserted by 15 kernels'
instruction counts changing. Both layouts already reach the 16 distinct slots a 64-bit
access can address, which the calibration shows is the floor.

What is left is the **data-dependent addressing**: the walk's two chain-selected element
reads, and the `tab` atomics in the census and the chain build. A swizzle is a bijection,
and a bijection maps a uniform random index to a uniform random index. The OpenCL-era
[SoA staging](#soa-lds-staging) reached the same place by timing; this is the same result
with the counter attached. **The 1.68 ms is a correctly computed ceiling on a pipe that is
never the critical path.**

*(Scope: sm_89, CUDA 13.3, driver 610.43.03, `(16,1)` at stock.)*

### The census and the chain share one tab word, and a barrier goes with it

Since [the singleton filter](#singleton-free-staging-the-prize-is-085-ms-and-the-prepass-that-finds-it-costs-076)
shipped, `tab` is initialised **twice** per group — to 0 for the word-0 census, then to
`kEmpty` for the chain the expand loop builds — with a `__syncthreads()` between the
second clear and the first `atomicExch`.

`MXBM_TABPACK` (default on) puts both in one word: the count in the high 16 bits, the
chain head in the low 16. The census adds `1 << 16`, the singleton test compares against
`2 << 16`, and the chain's `atomicExch` masks the word it gets back — the count above the
head is simply overwritten by the first insert, and nothing reads it after staging. The
sentinel becomes `0xFFFF`, which a `static_assert` keeps above `FCAP`.

Applied-asserted in the SASS: **every SOLO kernel loses exactly one `BAR.SYNC`** and 64–80
instructions (r1 1832 → 1760, r2 2560 → 2488), and no other kernel changes. Registers,
shared bytes and blocks/SM are unmoved, so the occupancy contract passes unedited.

| | mean | range |
|---|---|---|
| `MXBM_TABPACK=0` | 28.9551 | 28.927 – 28.969 |
| shipping | **28.8955** | 28.885 – 28.910 |

Eight interleaved arms a side (`A B B A` × 4), 45 s each, stock 285 W headless, ms derived
from `solves/s`. The two ranges do not overlap. Drops zero, 1.99 verified solutions/solve
and 284.0–284.5 W on every one of the sixteen arms; KAT 3/3 over 22 configurations.

**Null under a cap, and compiled out below the low-power gate.** `PACKTAB` is
`MXBM_TABPACK && SOLO` and `SOLO` is `!MFIRST`, so under the `kSpecMinPowerW` gate — where
match-first is on — the change is not in the kernels that run, and all 28 match-first
instantiations are byte-identical to the control's. Above the gate it is live and measures
**−0.14 % at 180 W** (39.597 → 39.541 ms, the shipping range inside the control's) over
four interleaved arms a side. At 120 W, where the two binaries execute the same code, the
same harness reads **+0.12 %** with the ranges overlapping — a positive control on the
method. *(A cap sweep taken the same evening disagreed by 2–4 % below 240 W; it is the
[swept-column offset](#a-swept-cap-column-that-did-not-reproduce-and-the-five-explanations-that-were-not-it),
recurring, and the byte-identical kernels are what prove it.)*

**−0.060 ms (−0.21 %), and it bounds the barrier family a second time.** The
[block-exit barrier](#the-block-exit-barrier-is-removable-and-the-barrier-family-is-over-priced-10x)
was −0.048 ms; this one is −0.060 with a 128-word clear pass thrown in. Two independent
removals agree that a barrier in these kernels costs ~0.05 ms whatever share of the stall
histogram it holds. With this one gone, every remaining `__syncthreads()` in
`fused_round_body` separates a write to `tab`, `gcount`, `cnt8` or `lchain` from another
lane's read of it — there is no third.

### Co-tenanting rounds 3 and 2 is gated by shared memory, and NARROW6 opens the gate

The one co-residency pairing left open: r3 is DRAM-bound and r2 is ALU-bound, so hosting
them as alternating blocks of one launch should overlap disjoint resources. Gated with
`ptxas -v` before any timing, as the lead required:

| kernel | registers | shared | blocks/SM |
|---|---|---|---|
| `fused_round<r2>` | 64 | 23,624 | 4 |
| `fused_round<r3>` | 56 | 26,184 | 3 |
| `fused_pair<r4,r1>` — shipping | 64 | 22,336 | 4 |
| `fused_pair<r4,r2>` | **64** | 23,624 | **4** |
| `fused_pair<r3,r1>` | 79 | 26,184 | 3 |
| **`fused_pair<r3,r2>`** | **80** | **26,184** | **3** |

**The registers are not the binding resource, despite reading like it.** Adding
`__launch_bounds__(kWG, 4)` compiles the same pair to **64 registers with zero spill** —
ptxas simply spends what it is not asked to save. Shared memory is the gate: the union
pays `max(26,184; 23,624)`, which is r3's own figure, and 4 × 26,184 > 102,400. So the
merged kernel inherits r3's 3 blocks/SM, and **r2 gives up its fourth block — a measured
−0.33 ms** — before any overlap is earned.

`MXBM_NARROW6=1` opens it. Round 3's seventh work word carries 16 significant bits in a
u64; moving it to its own `uint16_t` plane takes r3 from 80 to 74 B per staged element,
and the pair to **64 registers, 24,256 B, 4 blocks/SM, zero spill** with no launch bound
at all. NARROW6 is retained default-off as
[null on r3 standalone](#round-3-does-not-want-a-fourth-block--occupancy-pays-only-where-a-round-is-latency-bound)
— r3 is bandwidth-bound, so its own fourth block bought nothing — and that entry's own
condition was "worth having if anything ever makes r3 want a fourth block". This is it.

What remains is the pipeline, not the kernel: r2(i+1) writes the bucket set r3(i) is
reading, so the pairing needs a **third set-0 (~2.5 GiB at (16,1))** on top of `--pipe2`'s
existing double buffers, and a residency-ratio sweep — the `<r4,r1>` build needed 2:1 just
to match what the simpler mechanism already had. Against
[the family's ~0.5 ms ceiling](#fused_pair-two-solves-rounds-in-one-launch--the-familys-ceiling-is-05-ms),
already harvested by speculative entry, that is the trade to weigh. `fused_pair<r4,r2>`
needs neither NARROW6 nor a third set — but it competes for r4's idle, which is the idle
speculative entry already fills.

### Cooperative record staging: the sectors halve exactly, and it costs 0.20 ms

Built, and it does precisely what it was designed to do. `MXBM_COOP=1` moves round 3's
record read out of the sub-mask-filtered staging loop — which parks the element's index in
`lchain` — into a loop where every lane is live and **four consecutive lanes take one 16 B
chunk each of one 64 B record**, so the two sectors a record occupies are fetched once
instead of twice.

**The mechanism assert is exact.** Round 3's read sectors per element go **6.034 → 4.066**
against a predicted 4.03, DRAM reads are unchanged at 2.15 GB, and r1, r2 and r4 do not
move at all. The occupancy contract passes with **zero drift** on every kernel. KAT 3/3
over 22 configurations, drops 0.

It is still slower. Two forms, each 12 runs ABBA at 2600/10251, zero within-arm spread:

| | shipping | v1, four-way branch | v2, computed index |
|---|---|---|---|
| **ms/solve** | **29.100** | 29.500 (**+1.37 %**) | 29.300 (**+0.69 %**) |
| r3 read sectors/element | 6.034 | **4.066** | **4.066** |
| r3 instructions/element | 32.20 | 39.54 | 36.79 |
| r3 shared-store inst/element | 0.787 | 1.538 | 0.905 |
| r3 MIO throttle | 5.78 % | 2.89 % | 2.79 % |
| r3 long scoreboard | 56.21 % | 56.49 % | — |

v1 gave each sub-lane its own branch, which made every shared store a four-way divergent
instruction and nearly doubled the store count. v2 takes the straddling word from the
**left** neighbour instead of the right, which puts every lane's two outputs at `2*sub` and
`2*sub+1` — one store instruction with a computed index — and recovers most of it. The
residual **+4.6 instructions per element** is the second loop, its index arithmetic, the
shuffle and the one branch that remains.

**The prize was never there to collect.** Everything the lead promised arrived — a third of
the read sectors gone, a third of the load instructions gone, MIO throttle halved — and the
round did not get faster, because **`long_scoreboard` does not move**: 56 % of r3's
warp-active cycles are waiting on DRAM, and DRAM bytes are unchanged by construction. MIO
throttle was only 5.78 % to begin with, so halving it cannot pay for 4.6 instructions.
Extrapolating the two forms, a zero-overhead version lands at break-even, not at a win.

**Scope the currency correction to writes.** Halving round 3's *write* sectors bought 4 % of
the round; halving its *read* sectors bought nothing. The asymmetry has a reason — a read
that misses L1 is usually served by L2, while a write has to reach DRAM eventually — so
"sectors are the currency" is a statement about the write path and does not transfer.

`MXBM_COOP` stays default-off. It is kept because it is the only measurement that bounds
the lane-assignment family, and because its assert is reusable: any future claim about
round 3's read sectors can be checked against a build that provably halves them.

### Where round 3's read sectors go, and why deleting one of them costs 2 percent

Two profiles decompose the read side exactly. `CLOCKS=none TARGET=miner cuda/profile.sh`
on the shipping (16,1) geometry, then the same run forced to (17,0), which has no sub-mask
and therefore no second block per bucket. Read sectors **per input element**:

| round | (16,1) | (17,0) | Δ |
|---|---|---|---|
| r1 | 1.24 | 0.70 | −0.54 |
| r2 | 2.21 | 1.18 | −1.03 |
| r3 | **6.03** | 5.03 | **−1.00** |
| r4 | 6.04 | 5.03 | −1.01 |

The −1.00 is exact and it names the two terms. A block owns one sub-mask of a bucket and
scans the **whole** bucket's word 0 to find its share, so at sm=1 every element's word 0 is
read by two blocks and kept by one: **2 sectors of scan**. Then the block that keeps it
reads the 64 B record as four `LD.128`, each lane 16 B at a 64 B stride, so the two sectors
the record occupies are touched twice each: **4 sector-touches**. 2 + 4 = 6.03 measured;
drop the sub-mask and it is 1 + 4 = 5.03. That is also the whole of the 6.48 GB of L1←L2
traffic behind 2.15 GB of DRAM — **193 B of sector traffic per 64 B record**.

**So the record read is over-counted 2×, and the scan is duplicated.** Both look like free
money and only one of them is.

**The duplicated scan is not.** `SUBPASS` already implements its removal — one block owns a
whole bucket, reads each element's sub-mask bits once into shared, and sweeps the masks out
of shared — and on the current kernel it costs **+0.623 ms (+2.11 %)**: 29.490 against
30.113 ms, six arms a side ABBA in `cuda/pipeline`, ranges non-overlapping, KAT 3/3 and
drops 0 on both. It lost 1.3 % on the 35.24 ms build and it loses *more* now.

The mechanism is the one the geometry closure already found — time is monotonic in
**blocks × group**, and SUBPASS halves the blocks while doubling each one's sweeps, so the
product is unchanged and the sub-masks that used to run concurrently on two SMs now run in
sequence inside one. **A sector removed by a structural change is not a sector removed.**

That is the calibration to carry: the [currency
correction](#bytes-are-nearly-free-per-element-work-is-not) was measured on round 3's
*write* sectors, where halving them bought 4 % of the round. On the read side, deleting
17 % of r3's sectors and 45 % of r2's **cost** 2.11 %, because the deletion came bundled
with less parallelism. Sectors are a currency, and nothing conserves them — price the
structure that removes them, never the sectors alone.

**What survives is the 4 → 2 on the record read**, which is structure-neutral: same
instruction count, same blocks, same group, same bytes, only which lane holds which 16 B.
Four consecutive lanes covering one record's 64 B issue two fully-used sector requests
where one lane issuing four `LD.128` issues four half-used ones. Worth **2 of r3's 6.03
read sectors** at (16,1) and 2 of 5.03 at (17,0). Its obstacle is not the one the lead
assumed: the staging loop is sub-mask divergent, so the surviving lanes do not hold
consecutive records and the quad's record index has to arrive by `__shfl`, with the IMPB
unpack straddling lanes because each output word draws bits from two adjacent input words.

Two instrument notes. **Wavefronts per element are 7.9 / 12.4 / 16.8 / 11.0** across r1–r4,
so the "already ≤ 1, nothing to halve" kill test does not fire. And **`-DMXBM_SUBPASS=1`
is a no-op on the miner** — only `cuda/pipeline.cu` reads it, the solver passes the
template parameter as a literal — which a byte-identical `cuobjdump -sass` caught before it
could be reported as a perfect null.

### r2's fifth resident block is reachable and costs 0.10 ms

The singleton filter takes the staged group from ~264 to ~230, which was supposed to make
`kFCap` cuttable: 320 → 260 removes 60 slots at 72 B and takes round 2's shared from
23624 B to the 19456 B line a fifth block needs. **The arithmetic was exactly right** —
the built kernel reports **19304 B** — and every part of the conclusion drawn from it was
wrong.

Four arms, each 12 runs ABBA at 2600/10251 and 285 W, headless, KAT 3/3, drops 0. Every
arm has **zero within-arm spread** — all six runs of each identical to 0.1 ms:

| arm | ms/solve | r2 registers | r2 shared | r2 blocks/SM | board |
|---|---|---|---|---|---|
| shipping | **29.100** | 64 | 23624 | 4 | 267.6 W |
| `kFCap` 260 | 29.300 (**+0.69 %**) | 60 | 19304 | 4 | 284.0 W |
| `kFCap` 260 + `MXBM_MB_RD2=5` | 29.400 (**+1.03 %**) | 48 | 19304 | **5** | 283.9 W |
| `MXBM_MB_RD2=5` alone | 29.100 (**0.00 %**) | 48 | 23624 | 4 | 268.4 W |

Read down the last two rows: asking ptxas for five blocks costs **nothing** on its own,
so the **+0.100 ms** between the middle two arms is the resident block itself. **The fifth
block is a loss**, where the decay of r2's own fourth (−0.33 ms) predicted a −0.15 gain.

**The cut's own cost is the split, and it is measured with a control.** `MXBM_SPILL=0`
turns `drops[1]` into the count of elements past the cap, so the spill volume reads
directly and a zero cannot be mistaken for a dead probe:

| `kFCap` | elements over cap, per solve |
|---|---|
| 320 (shipping) | **184** |
| 260 | **813 900** |

4400×. The lead's premise was a Gaussian tail — 260 sits 2σ above a mean of 230, so ~3 %
of groups split — and the real tail is far heavier than that: 813 900 over-cap elements is
6.2 per group averaged over *every* group, where a normal tail of that width predicts
under 0.1. The split re-runs the whole group pass, which is why the cut arms draw **16 W
more at a locked clock**: same time budget, more work done in it.

The register half [held](#round-2s-register-wall-is-not-a-wall-48-registers-zero-spill-0023-ms) through two record changes: the occupancy contract reports 48 registers with no
spill on the implicit-bits rows, which is what `bb + sm = 17` runs, and 8 B of stack on
the plain-packed variant it does not — the same split that entry recorded, unchanged.

**What this closes.** The `kFCap` route is the only one that reaches the fifth block
today, and it pays +0.200 ms in splits before the block is even reached. The other route —
three structural shared cuts that would free 4168 B without touching the cap — now has a
measured prize with the wrong sign, so it is no longer worth its tiebreak risk. The one
thing it could still argue is that the block only looked bad *because* it was measured
alongside heavy splitting, and that is a real caveat: no configuration exists yet that
gives r2 five blocks without them.

### A swept cap column that did not reproduce, and the five explanations that were not it

A cap sweep is fifteen unrepeated runs. That is fine for a curve's *shape* and it is what
`power_sweep.sh` was written for, but it makes a swept column the one number in this
project that never gets bracketed — and one of them was wrong.

The column swept immediately after the singleton filter shipped read **67.2 ms at 120 W**.
Re-run later the same day, same binary, same script, same cap: **63.9 ms**, agreeing with
an interleaved ABBA taken between the two at 63.4–64.6. A 5.5 % gap, and every candidate
was measured at 120 W as its own interleaved pair rather than argued about:

| suspected | measured | |
|---|---|---|
| no warmup before the timed run | **0.2 %** | a discarded run first changes nothing |
| the NVML sampler running alongside | **0.0 %** | 5 Hz of `nvidia-smi` is free |
| run length | **0.4 %** across 45 / 90 / 180 s | and not monotonic |
| the cap transition the sweep measures through | **0.6 %** | settle at 110 W, raise, start at once, against a settled cap |
| leftover clock locks from the pin | **0.0 %** | both arms settle to 1035 MHz |

Sum of everything found: ~0.8 % of a 5.5 % gap. No co-tenant either — that session's own
logs report 15968 MB free, and a second miner would hold gigabytes. **The cause is
unknown**, which is the honest state, and the useful part is what it cost: the bad column
was read as a tilt, the tilt had a mechanism ready to explain it, and a day's shipped work
was suspected of a capped-rig regression it did not have.

Two things follow, and the second is the one that generalises. **A swept column is
provisional until at least one of its points is reproduced by a bracketed A/B** — the
current column carries 120 W and 180 W agreeing with interleaved arms to 0.2 %, and that
is now the bar for adopting one. And **a cross-session sweep difference is not evidence
about a kernel.** It cannot be: the sweep's own repeatability is the thing under test.

**It recurred the same day, and the second instance is airtight.** The column swept after
the packed `tab` word read 2–4 % slower than the previous kernel's at every cap below
240 W and +0.7 % at stock — a smooth monotonic ramp, and a mechanism was available to
explain it (the change adds one instruction per staged element and removes a barrier,
which is the wrong trade for a starved core). It is not the kernel, and no measurement of
the kernel was needed to know that: **`PACKTAB` is `MXBM_TABPACK && SOLO`, `SOLO` is
`!MFIRST`, and below the 130 W gate match-first is on** — so on the three points at 100,
110 and 120 W the two builds run byte-identical SASS. All 28 match-first kernels compare
equal instruction for instruction. The sweep read −2.2 %, −1.9 % and **−4.4 %** between
two binaries that execute the same code there.

Bracketed pairs then measured the two caps that matter, `A B B A`, 45 s arms:

| cap | control | shipping | |
|---|---|---|---|
| 120 W | 65.013 ms (64.935–65.104) | 65.090 (64.893–65.147) | +0.12 %, ranges overlapping — and **null by construction**, which makes it a positive control on the harness |
| 180 W | 39.597 ms (39.324–39.730) | 39.541 (39.448–39.604) | −0.14 %, the shipping range inside the control's |

So the swept difference is a **per-session offset on the sweep**, twice now, in the same
band. The bar above stands and is worth restating in its strong form: a swept column may
be compared with *itself*, never with another session's — and where a per-cap conclusion
matters, the cap gets a bracketed pair.

**A third instance (2026-08-18) shows the offset runs in both directions and grows as
the cap falls.** A full 15-cap sweep of the shipping binary read *above* the published
column at every point — +0.3 % at 285 W rising monotonically to +5.2 % at 100 W — a
shape with a mechanism ready to explain it (two instruction-deleting kernel changes had
shipped since the column, and instructions are the capped card's currency). Bracketed
pairs at three caps, the published column's binary against the shipping one, killed it:
**−0.18 % at 140 W, +0.15 % at 120, −0.06 % at 190**, every |t| < 1.5 against a 0.040 %
null floor. The binaries are performance-identical and the entire column difference is
the session. Three consequences:

- **The flat ~2.5 % cross-session band understates the low caps.** Cap-column absolutes
  carry a *cap-dependent* session term that has now been observed at −4.4 % and +5.2 %
  at the 100–120 W points. A candidate mechanism — a fixed power limit makes the
  operating point power-bound, so leakage (silicon temperature, ambient) moves the MHz
  each watt buys, while at stock the V/F ceiling absorbs it — fits the monotonic shape
  but is unproven; what is established is the size and the sign varying by session.
- **The chain-walk `#pragma unroll 1` is null under caps too.** The same brackets close
  the only stock-rung kernel change in the span (with the tab word — already null under
  a cap in-session — and the quad/octo w0 records, which are off the stock rung, along
  for the ride) at **±0.2 % at 120/140/190 W**. The cap multiplier on instruction
  deletion — 2× to 4.7× on this record — is a multiplier on *dynamic* work, and the
  unroll deleted static footprint whose dynamic chase was already length one. **Price
  cap candidates in dynamic instructions and sectors, never in footprint.**
- **A head-to-head column assembled from two sessions inherits the term.** The cap
  table's MXBM column (08-16) and its reference column (07-30) sit a session apart, so
  the per-cap margins between the crossings are known only to within this band. Where a
  margin drives a decision, both miners get measured in one session, interleaved.

**A fourth instance (2026-09-08) reproduced the shape with the sign reversed again.** The
15-cap sweep after the terminal round shipped read *below* the 08-18 column at every cap
under 220 W — −2.8 % at 140 W, −5.3 % at 100, −8.1 % at 110 — and above it from 240 W
up (+0.7 to +1.1 %). The pre-stint binary, built from its commit and run in the same
session, sits at the swept figure, not the published one: old against new is **+0.5 % at
140 W** for the new build (eight 60 s arms, ranges non-overlapping) and **0.0 % at 100 W**
(second ordering; the first ordering's two arms after the first hand-off read 3–6 % low
in both a 60 s and a 90 s repeat, a settling transient specific to the floor cap). The
sweep's own ~4 % at 140 W is the session, and the column was left as swept on 08-18 with
the change bounded by the pairs. One method note came out of it: the sweep script's
output filter matched the miner's periodic status header, which also says `sol/s`; the
summary lines were intact and the filter is now anchored to them.

The clock-lock row is worth keeping separately. It is the third independent confirmation
that [the governor outranks the lock](#undervolting-buys-nothing-under-a-power-cap--the-cap-outranks-both-knobs) — under a 120 W cap, `-lgc 2600 -lmc 10251` and released
clocks both settle to the same 1035 MHz and the same time per solve.

### The w0-checkpoint pair record, repriced by the address bits: −0.76 ms (−2.4 %)

**Round 2 no longer derives work word 0. Round 1 stores it, in the same 16 bytes.**

`apply_mix` writes word 0 and nothing else, and `combine` is XOR-plus-shift, so words 1..6
of a round-2 element are linear in the parents' *seed* words 1..6 alone. Storing word 0
therefore lets round 2 skip both parent mixes, the child mix and the two `k = 0` siphashes:
`rebuild_r2`'s 14 siphashes + 3 mixes + a combine become `rebuild_r2_lane`'s **12 siphashes
and nothing else**.

This is the same idea as `MXBM_PAIR_W0` (K5), which measured **+3.8 % at stock** and was
closed on its merits. What changed is not the algebra — unchanged, and identity-confirmed
then — but the **packing budget**. K5 grew the record 16 → 24 B, which cost ~4 memory
instructions and 537 MB per solve, and its 24 B form used only 140 of its 192 bits. The
address-implied key bits, which shipped 08-13/14 for the r2 → r3 record, pay for word 0's
extra 24 bits outright:

| field | bits |
|---|---|
| word 0, less the `IMPB` bits the bucket index already encodes | 48 (47 at `IMPB` 17) |
| `li`, `ri` | 25 + 25 |
| `gi` | 26 |
| **total** | **124 of 128** |

So the record stays 16 B: **no added bytes, no added memory instructions, staging keeps its
single LD.128, and every resource is unchanged** — r1 holds both of its cliffs (48 registers
of 48, 19016 B of 19456, five blocks/SM) and r2 its four blocks on exactly 64 registers.
The kept low `24 − IMPB` key bits stay at the bottom of word 0, so the staging loop's
sub-mask filter and tab hash read them where they always were; that is why this rides
`IMPB` and is off wherever the packed r2 → r3 record is off.

ABBA-interleaved, clocks locked, headless, six arms each:

| | mean | spread |
|---|---|---|
| baseline | 32.024 ms | 0.041 |
| w0-checkpoint | **31.266 ms** | 0.029 |
| **delta** | **−0.758 ms (−2.37 %)** | ~19× the within-arm spread; the arms do not overlap |

Predicted −0.8 ms from two independent accountings — 2 siphashes + 3 mixes per element, and
K5's measured +3.8 % less the byte cost the repack removes. Both land on the measurement.
The two builds' CUDA flags differ **only** by the define, checked by diffing `flags.make`.

**The general form.** A re-derivation is priced against what the record costs to *carry*,
and the address bits are free carrying capacity. Any checkpoint that fits in the slack a
packed record already has is worth re-pricing, even when the same checkpoint was measured a
loss in a form that had to grow the record to hold it.

**It pays about twice as much under a cap as it does at stock** — the first lever on this
record that does. A full 14-point power sweep on the shipping build against the same sweep
on the previous kernel:

| cap | 100 W | 110 W | 120 W | 140–220 W | 240 W | 285 W |
|---|---|---|---|---|---|---|
| gain | **+11.9 %** | **+13.8 %** | **+11.1 %** | +6 to +8 % | +4.0 % | +3.3 % |

The mechanism is the [NARROW6 floor loss](#round-3-does-not-want-a-fourth-block--occupancy-pays-only-where-a-round-is-latency-bound)
run backwards. That change *added* instructions to save shared bytes and lost at the
floor; this one deletes ~300 ALU instructions per element and wins there. At stock, round
2 is stall-bound and much of that arithmetic was hiding in stalls; at 100 W the core is at
765 MHz and every deleted instruction is deleted work. **Instructions are the currency a
clock-starved card cannot afford, in both directions.**

Two consequences for the head-to-head: the speed and efficiency crossings against lolMiner
both moved from ~210 W to **~200 W** and now coincide, and MXBM's own efficiency peak moved
**left, 240 W → 220 W** (0.2746 sol/s/W, 3.642 J/solution). The three top caps reproduce a
partial sweep taken an hour earlier to within 0.5 %; the low-cap gains are cross-session and
clear the ±2.5 % band several times over. Full table in
[both miners under the same cap](performance.md#both-miners-under-the-same-cap).

### The w0 checkpoint transfers to the 5001 rung — and its off arm had rotted

**−2.1 % / −0.89 ms at 160 W with the rung held (t = 80, 24 × 60 s paired), against
−2.4 % at stock: the checkpoint keeps nearly full value at the rung's own operating
point.** The mechanism is in the clock column — the checkpoint arm runs 2268 MHz core
against the off arm's 2202, the deleted rebuild instructions returned as clock. There
is no cap amplification here: the 2–4.7× multiplier lives on stock-memory caps where
the core clock is halved, while on the rung the clock is already near stock and the
halved bandwidth hides part of the deleted compute — the two roughly cancel. Both arms
spec-on, (16,1), 2.00 verified solutions/solve, KAT 3/3. The checkpoint stays on by
default; nothing ships from this measurement.

<details>
<summary>The off arm's rot, and the fix</summary>

Building `-DMXBM_PAIR_W0=0` produced a KAT-failing binary: an illegal write from round
1's IMPB=0 instantiation storing back-reference rows through a null pointer. On the
implicit-bits rungs the host allocates no rows for rounds 1–3 (`refRows = impb ? 0 : 3`
— recovery replays them), and the kernel keys that elision on template `IMPB != 0`;
round 1 only received a nonzero IMPB through the checkpoint instantiations, which sat
under `#if MXBM_PAIR_W0`. The knob was healthy when the checkpoint shipped and broke
when the round-3 replay landed on top of it. The fix makes round 1's IMPB
instantiations and launch arm unconditional — IMPB carries the row-elision fact
regardless of the checkpoint, and `PW0 = MXBM_PAIR_W0 && IMPB` alone keys the record
format (the emit's else-branch writes the plain pair record). Under `=1` the
preprocessed source is unchanged, so the shipping binary is SASS-identical; the
occupancy-contract rows for these kernels are unconditional too (the `=0` resources
match the `=1` rows exactly), which is what retired the set-the-flag-on-both-compiler-
halves rule in *Instruments*.

</details>

### The 140 W census: the band solve is the front half

`stage_power.sh 8 45` under an external 140 W cap, stock memory, spec off (the
shipping policy there). Baseline 51.19 ms/solve, drops 0, 7.17 J/solve; the replay sum
attributes 102.1 % (±2 % linearity band):

| stage | ms | % time | % energy | stretch vs stock |
|---|---|---|---|---|
| entry | 5.01 | 9.8 | 9.6 | ×1.86 |
| r1 | 8.31 | 16.2 | 15.9 | ×1.43 |
| r2 | 16.20 | 31.7 | 31.0 | ×1.54 |
| r3 | 13.60 | 26.6 | 26.1 | ×1.36 |
| r4 | 7.91 | 15.5 | 15.1 | ×1.39 |
| term | 1.23 | 2.4 | 2.3 | ×1.17 |

**r1 + r2 carry 47.9 % of a 140 W solve** (the per-round desk model predicted ≈49),
and entry + r1 + r2 together 57.7 %. The stretch column ranks cap-sensitivity: entry
(issue-bound seed + scatter) stretches most at ×1.86, r2 (the 14-siphash rebuild)
second at ×1.54, the DRAM-heavy r3/r4 least among the big rounds, the whole solve
×1.45. A band lever is priced against this table — the compute-heavy front half is
where a cap point pays.

### Round 3's write sectors re-priced under the cap

The [dead-word closure](#round-3-stores-a-work-word-round-4-never-reads--worth-268-mb-and-nothing-in-time)
priced half of r3's write sectors at 4 % of the round — at stock, where the implied
~3000 GB/s said the stores were off the critical path. The emit-sector census behind
the re-pricing (ncu, one full solve, clock-independent): the five scatters pay ~11.3 GB
of L1→L2 store-sector traffic for ~5.4 GB of payload, r2 and r3 each 4 sectors/element
(a 64 B record as four scattered ST.128, each half-filling its sector — the 16 B-store
floor), r2+r3 together 73 % of all store sectors; the L2 write port merges 5 % of r2's
and 14 % of r3's before DRAM.

Re-measured with the closure's own instrument (`MXBM_ABL_EMIT=3` narrows r3's emit
64 → 16 B with compute, stride, bucket distribution and occupancy all preserved;
4 → 1 STG.E.128 asserted in SASS; marginal by in-place replay, `MXBM_ROUND_REPS`
3:1 against 3:9, eight position-balanced blocks per point, one session):

| cap | r3 marginal, ship | Δ for −3 store sectors/element | % of round | vs stock |
|---|---|---|---|---|
| 285 W | 8.441 ± 0.013 ms | −0.308 ms | −3.6 % | (reproduces the closure) |
| 140 W | 13.554 ± 0.323 ms | **−3.367 ms** | **−24.8 %** | **×6.8** |
| 100 W | 23.127 ± 0.176 ms | −5.392 ms | −23.3 % | ×6.4 |

The 140 W ship marginal agrees with the census's 13.60 ms from a different instrument.
The implied rate closes the mechanism: −3.37 ms for 3.2 GB of removed sector traffic
is ~950 GB/s — under a cap the scattered stores are priced near their real sector
traffic, where stock priced them at an impossible 3000 GB/s. **The largest cap
multiplier measured on any lever** (the w0 family peaked at ×4.7), landing exactly
where the store-vs-derive desk pricing said the exchange rate moves.

What it does not reopen: the
[quad record under caps](#the-rung-regime-does-not-reopen-the-closed-geometry-trades--quad-and-170-both-null)
stays closed — its saving is bytes-with-rebuilds, and the rebuild's core cycles starve
under the same cap (+13.8/+15.1 % at 160/140 W, unmoved from stock; the QW0 −5.11 %
does not close that); chunk-staged emits are geometrically infeasible at 2^25 (a
staged tile bounds chunks and match from opposite ends). Nor does it reopen the
[48 B r3→r4 record](#round-3-stores-a-work-word-round-4-never-reads--worth-268-mb-and-nothing-in-time),
which the new prize (≈ −1.1 ms at 140 W by this table, against −0.089 at stock)
prompted re-checking against the record's consumers: it stays dead, by bits. With
`cgi` retired — its three remaining readers are the equal-lead orientation
tie-breaks in round 4's match and both replays, each substitutable by the staging
slot exactly as the quad16 and ow0 records already do — the minimum content is
296 work bits (312 consumed, 16 address-implied) + 25 lead + 64 leftContrib + 17
replay bucket hint = **401 bits in 384**. The overage is exactly the hint:
`replay_r3` reads word 5 as the parents' bucket, its only round-3-level parent
pointer since the reference rows were deleted, and the combine's `>> 24` destroys
the parents' collision key, so it cannot be derived, and truncating k bits
multiplies the replay's search by 2^k. Every relocation prices at net zero or
worse in the sector currency itself: a hint side-plane or revived reference rows
re-add ≥ 1 scattered store per element, a 7-u64 record is four transactions again
and unaligns odd slots, and encoding hint bits in sub-bucketed slot ranges buys
≤ 2 bits for arena pressure. Retiring `cgi` alone deletes only the `gi_alloc`
atomic — no sector moves. *Comes back if a recovery scheme can name an
r3-ancestor's parent bucket without a per-element stored field.*

*Both dismissals above were measured on 2026-09-08 and both were wrong in the same way —
priced by category rather than built. The 7-u64 record keeps four transactions by choosing
the 8 B access by slot parity and shipped at
[+1.7 %](#the-round-3-record-is-56-b-the-dead-word-goes-and-the-four-16-b-transactions-stay);
retiring `cgi` on the replayed rungs shipped at
[+0.9 %](#the-gi-allocator-goes-where-nothing-indexes-a-gi-09). The 48 B form stays dead by
bits.*

What the table prices up instead is the format-preserving route: a
**warp-cooperative sector-paired emit** for rounds 2 and 3. Both scatters store a
64 B record as four ST.128, and each 16 B store from a lane half-fills its own
32 B sector — but the coalescer merges same-instruction, same-sector lanes, so if
active emitter lanes pair up (ballot + shuffle: partners exchange record halves
and the destination address) and each instruction writes both halves of ONE
record's sector from two lanes, the pair costs 2 transactions per element instead
of 4 — on the two rounds carrying 73 % of the store bill, with records, recovery
and output bytes untouched. The
[cooperative-staging null](#cooperative-record-staging-the-sectors-halve-exactly-and-it-costs-020-ms)
does not cover it — that closure's own scope line is that the write-side currency
correction does not transfer to reads, and the 16 B-store floor is a per-lane
statement that lane pairing is built to evade. By this table the gross would be
≈ −2.2 ms at 140 W for round 3 alone, against a shuffle tax of roughly 16 extra
instructions per element at the cap's ×2–4.7 instruction pricing.
[Measured: it is a null](#the-sector-paired-emit-the-transactions-halve-the-dram-bytes-do-not-and-the-solve-does-not-move),
and the miss re-prices this table's own exchange rate — the currency is
DRAM-reaching sectors, which the ablation's payload narrowing halves and lane
pairing leaves untouched.

### The sector-paired emit: the transactions halve, the DRAM bytes do not, and the solve does not move

*(2026-08-19. The successor experiment above, built and measured. Closes the
sector-efficiency axis on the write side and re-prices the cap table above it.)*

`MXBM_PAIRED_EMIT` builds exactly the design: on rounds 2 and 3's 64 B emits,
active emitter lanes pair over `__ballot_sync` + `__fns` + `__shfl_sync` (five
u64 shuffled per lane — the partner's record half and its *slot*, never its
pointer, which round-tripped as an integer compiles the stores generic), a solo
lane takes both roles, and each ST.128 writes a 16 B half of one record's 32 B
sector. Every gate green: KAT 3/3 on all 15 geometries, `drops == 0`, blocks/SM
identical per round in both arms (r2 = 4, r3 = 3), and the mechanism confirmed
on two counters — **store sectors per element 4.00 → 2.11 on r2 and 4.00 → 2.12
on r3** (residue = solo lanes), L2 write port r2 127.4 → 70.6 M, r3
114.9 → 70.8 M.

The A/B (40 ABBA arms of 30 s, position-balanced, one session per cap):

| cap | base | paired | paired vs base, solves/window |
|---|---|---|---|
| 140 W | 53.23 ms/solve | 52.98 ms | **+0.50 % (≈3 se)** |
| 285 W | 28.70 ms/solve | 29.12 ms | **−1.41 %** (the shuffle tax) |

Roughly +0.5 % at the cap where the sector arithmetic promised ~5 %, and a
stock loss. The resolving instrument is a DRAM-side ncu pass over one full solve
per arm (the metric P0 could not name is `dram__bytes_write.sum`): across the
fused rounds the paired arm deletes **−37.7 % of L1→L2 store sectors (−253
M/solve)** and **−32.6 % of L2 write-port sectors**, and **DRAM bytes written do
not move — identical in both arms to four digits (5.30 GB/solve; the profile's
two launches per kernel sum to the 10.605 GB the raw verdict quotes), reads
identical**.

The mechanism: a lane's four ST.128s are consecutive instructions, so the two
16 B halves it contributes to each 32 B sector arrive at the L2 a few cycles
apart and the L2 merges them before writeback — **the shipping scatter was
already sector-efficient at DRAM**. Lane pairing moves the merge from the L2
into the per-instruction coalescer and changes nothing the memory system bills.
What a halved transaction count is worth on its own is the +0.50 %.

Two closures re-price with it:

- **The write-side currency under a cap is DRAM-reaching sectors, not store
  transactions.** P0's "2.1× store-sector waste" is real only at the L1→L2
  interface, where it costs ~0.5 % of a 140 W solve — not a currency, a
  bookkeeping level. The [cap re-pricing above](#round-3s-write-sectors-re-priced-under-the-cap)
  stands, but its unit moves: `MXBM_ABL_EMIT=3` narrows the payload 64 → 16 B,
  which halves r3's DRAM-dirtied sectors (2 → 1 per element by sector
  granularity), and the −3.367 ms at 140 W rides that. A lever must change what
  reaches DRAM to collect the ×6.8.
- **Sector efficiency without byte reduction is now dead on both sides**: reads
  by the [cooperative-staging null](#cooperative-record-staging-the-sectors-halve-exactly-and-it-costs-020-ms)
  (a read that misses L1 is usually served by L2), writes by this one (a write's
  half-sectors merge in L2 before DRAM). What pays on the emit path is storing
  fewer bytes per element, and both narrowing routes are dead by bits — so the
  band thread moves to deriving less; scattering better is not the way through.

*Comes back if an emit's same-sector halves stop arriving together — a layout
whose record halves are written by distant instructions or different lanes
without pairing — or on a part whose L2 write path saturates in the core domain.*
The flag ships off-by-default as the decomposition's transaction-only arm; the
flag-off build is SASS-identical to the prior kernel (instruction streams diff
clean), so no published figure re-baselines.

### Every emit is already payload-compact at DRAM: the sector-waste table was an L1-level artefact

*(2026-08-19, desk census — pivoted from the paired-A/B's archived ncu CSVs, no
new GPU time. Extends the closure above from the 64 B records to every width.)*

Per solve, per kernel, at stock clocks:

| kernel | ms | DRAM rd GB | DRAM wr GB | wr B/element | record |
|---|---|---|---|---|---|
| entry | 2.57 | 0.000 | 0.255 | 7.6 | 8 B seed ✓ |
| r1 | 4.78 | 0.271 | 0.523 | 15.6 | 16 B pair ✓ |
| r2 | 7.94 | 0.539 | 2.128 | 63.4 | 72 B packed (the 9th-word plane part-rides L2) |
| r3 | 8.51 | 2.148 | 2.129 | 63.4 | 64 B packed ✓ |
| r4 | 4.17 | 2.149 | 0.265 | 7.9 | thin stride (meta plane L2-resident to terminal) |
| terminal | 0.78 | 0.271 | 0.000 | — | — |
| **total** | 28.9 | **5.38** | **5.30** | | **10.68 GB moved/solve** |

The small records merge too: entry's scattered 8 B stores and r1's 16 B pairs
reach DRAM at **payload**, where the L1→L2 counters bill a 32 B sector —
adjacent slots of a bucket are written closely enough in time that the ~8 MB
write frontier (2^16 buckets × one line) merges them in L2 before writeback.
With the paired-emit result this closes **sector efficiency at DRAM at every
record width**: there are no wasted DRAM bytes anywhere on the write side, and
nothing for any sector-efficiency lever to harvest at any operating point.
*Comes back if the write frontier stops fitting L2 — a bucket count or an L2
carve-out that pushes frontier residency below the slot-pairing interval.*

### Bandwidth by access pattern under power caps: streams are cap-immune, the per-element scatter floor is ~250–265 GB/s everywhere

*(2026-08-19, standalone microbench (campaign `2026-08-19-instrread`), 2 GiB
streams and the emit-shaped scatter — per-element atomic slot allocation into
2^bb buckets, 16 B or 64 B records — at 285→100 W on stock memory, sm clock
sampled per point. Positive controls: stock streams 596–639 GB/s ≈ the card's
achievable; stock scatter 253–263 ≈ the ledger's ~260 GB/s emit floor.)*

| GB/s payload | 285 W | 210 | 175 | 160 | 140 | 120 | 100 |
|---|---|---|---|---|---|---|---|
| stream read | 639 | 639 | 639 | 639 | 639 | 639 | 532 |
| stream copy (rd+wr) | 596 | 596 | 596 | 596 | 595 | 590 | 506 |
| scatter 64 B, 2^16 buckets | 263 | 263 | 263 | 258 | 247 | 229 | 180 |
| scatter 16 B, 2^16 buckets | 260 | 261 | 256 | 247 | 233 | 205 | 154 |

- **Streams do not feel the cap**: full pace to 120 W, because a full-rate
  stream draws only ~139 W — a band cap cannot touch the memory domain.
- **The scatter floor is nearly power-flat and exists at stock**: the 2.3×
  stream-vs-scatter asymmetry is a property of the access pattern (activation
  granularity) and never of the cap; the cap widens it only mildly (2.4× at 140 W,
  2.8× at 100 W).
- **Coarser destinations do not lift it**: 2^12/2^8/2^6 buckets measure equal
  or worse (167–255 GB/s; per-element atomics serialize before any DRAM
  row-batching helps). Escaping the floor needs **chunked appends** — ≥ ~1 KB
  contiguous per allocation — which per-element scatters cannot express.

What it re-prices, with the DRAM census above: **r2 and r3 are scatter-rate-bound
at stock** — each writes 2.13 GB scattered, which at the floor is 8.4 ms, and
their round times are 7.94 and 8.51 ms: the whole back half of the solve runs at
the scatter floor with its reads and compute hidden beneath. Priced against a
streamed layout at equal bytes (596 GB/s → 3.6 ms per round), the pattern tax is
bounded by **~8 ms of a 29.1 ms stock solve and ~12.6 ms of a 51.2 ms 140 W
solve** — the uncovered compute beneath (r2's 3.95 ms hash floor and the walk)
sets what is realizable, which is the round-pair probe's job to price.

**This decomposes the reference miner's band lead** (with its 2026-07-31 ncu
profile: 17.67 GB/solve all-streamed at 393–556 GB/s, no rebuild arithmetic).
At 140 W its two absences — our ~12 ms of stretched rebuild SipHash and our
~12.6 ms scatter-floor tax — against its ~+11.7 ms surplus byte bill (+7.0 GB
streamed) and its own starved match instructions reproduce the measured −6.5 %
stock-config gap to model precision (±15 %). The lead is **both currencies at
once**, and the pattern tax is the larger and the only one with a capture path
that keeps our records: the h-boundary trade stays closed even at stream rates
(deleting r2's rebuild for +3.76 GB streamed reads ≈ wash at 140 W and a clear
stock loss — the desk arithmetic that killed h=1 survives the rate correction).

**What survives as the axis-4 shape** — and what dissolves the streamstore
campaign's geometric kill: that kill priced fine buckets *in DRAM* (a ~20 KB
staged tile supports ≤ ~400 buckets; an in-shared match needs ≥ ~25000). A
consumer-side fine partition — emit into **2^6–2^8 coarse buckets as staged,
chunked, streamed appends**, and let the *next* round's kernel split its coarse
bucket transiently inside a persistent-L2 tile (8 MB at 2^8 against Ada's
48 MB carve-out) — needs the fine layout to exist only in L2, never in DRAM,
so both constraints hold at once. The reference's 471 GB/s one-pass rounds are
the existence proof that some such composition works at 2^25.

**The floor's escape mechanism is confirmed on this card.** The same harness's
chunked-append kernel — one atomic per *chunk*, the chunk written contiguous by
the warp, same total payload — measures **604–620 GB/s at 285 W and at 140 W**,
full stream rate, with the per-element scatter interleaved in the same session
reading its usual 260/235. It holds down to **256 B chunks (16 records)** and
at **2^12 buckets** (554–570 GB/s), so the DRAM side tolerates far finer
bucketing than the coarse 2^6–2^8 the desk assumed — the binding constraint is
only the *staging* side (bins per block-tile in shared). What the microbench
deliberately does not price is the classification that fills the chunks; the
round-pair probe below prices it, and splits the verdict: the emit half is
real, the consumer-side fine split is not.

### The sort path under a cap is not a band alternative

*(2026-08-19, same session. The in-tree store-everything design priced under
caps for the first time — OpenCL, `MXBM_NO_ROWBUCKET=1`, goldens matched, 60 s
benchmark arms, interleaved with the row-bucket path at each cap.)*

| OpenCL, ms/solve | 285 W | 160 W | 140 W | stretch 285→140 |
|---|---|---|---|---|
| row-bucket path | 31.8 | 52.6 | 63.3 | ×1.99 |
| sort path | 187.5 | 209.5 | 255.1 | ×1.36 |

The direction confirms the model — the sort path stretches far less under the
cap (its radix passes are streams, and it runs 1479 MHz at 140 W against the
row-bucket's 1156) — but it starts 5.9× behind at stock and is still 4.0×
behind at 140 W. Storing everything only pays inside a ground-up streaming
design; a sort bolted onto full records is not one. *Comes back if a redesign
brings the stock multiple under ~1.6×, the band stretch ratio measured here —
which is the same probe as the entry above, and no revival of the sort path.*

### The consumer-side fine split re-pays the scatter it saves: the floor follows the access shape into L2

*(2026-08-19, same day, follow-up session. The round-pair probe: a mock r3-emit
→ r4-match pair, 2^25 × 64 B records, uniform 24-bit keys, the shipping (16,1)
row partition. Arm A = the shipping shape — per-element scatter emit,
per-bucket two-sweep staged match, per-element 16 B child scatter. Arm B =
chunk-staged emit into 2^7 coarse bins (64 KB shared staging, 256 B chunks, one
atomic per chunk) + a cooperative consumer that re-splits each coarse bucket
per-element into a 24 MB fine-row tile under a persisting-L2 window, then
matches per row. Both arms gated on multiset-fingerprint equality of the stored
state and of the emitted children, zero drops, and the shipping arm reproducing
the scatter floor — 242–256 GB/s, pass. Child emit identical in both arms.
A WORK knob (mix64 rounds per record) stands in for combine/apply_mix.)*

| pair, ms | A emit | A cons | A pair | B emit | B cons | B pair | B vs A |
|---|---|---|---|---|---|---|---|
| 285 W, work 0 | 8.88 | 5.41 | 14.29 | 4.20 | 12.33 | 16.54 | −16 % |
| 285 W, work 16 | 8.42 | 5.43 | 13.85 | 4.65 | 13.75 | 18.40 | −33 % |
| 140 W, work 0 | 10.91 | 5.51 | 16.42 | 4.94 | 14.15 | 19.10 | −16 % |
| 140 W, work 16 | 10.95 | 5.80 | 16.75 | 5.93 | 16.66 | 22.59 | −35 % |

**The emit half is confirmed in a live kernel.** With real classification —
bin counters, staged chunks, flush logic, retries — the chunk-staged emit runs
**462–511 GB/s at stock against the 242–256 floor, and 301–435 GB/s at 140 W
against 178–197**; the staging tax is ~0.6–1 ms per 2.15 GB layer. The win
grows under the cap. One caveat travels with it: emit-side ALU does not hide
under the fast stream the way it hides under the slow scatter (arm A's rate is
flat in WORK; arm B's falls).

**The consumer-side fine split loses all of it and more**, and its
decomposition says why, in three measured mechanisms:

1. **The scatter floor follows the access shape. The destination memory does not set it.**
   Scattered per-element 64 B stores into the L2-resident tile move at ~620
   GB/s when the addresses are independent and **~360 GB/s under the
   allocation-atomic dependence** — stream rate at best, nowhere near L2's
   nominal bandwidth. Splitting in L2 rather than DRAM lifts the floor only
   2.4×, and the allocation dependence gives a third of that back. The
   streamed read + the allocation atomics alone are read-bound (3.3–3.6 ms):
   atomics are nearly free; the scattered *stores* are the tax.
2. **The persisting-L2 window is not a free lever.** It changed nothing for
   the arm it served — the 24 MB tile stays resident by capacity — and cost
   the *other* arm ~5 % through the device-wide carve-out the limit imposes.
3. **The split/match phase barrier forfeits the fused consumer's overlap.**
   The shipping-shape consumer hides its match compute and child scatter
   under its streamed read (5.4 ms total); the tile-fed match phase *alone*
   costs 4.6 ms at stock and 7.0 ms at 140 W. Grid syncs measured free
   (0.17 ms for 256).

The bound this puts on the whole serial-phase family: with the reorganization
*free*, arm B = emit + read + match phase = **+9 % of the mock pair at stock
and +1 % at 140 W** — under any useful bar before a real reorganization cost
lands. A one-level per-element split pays read + scattered touch ≥ 6.6 ms per
2.13 GB layer, more than the entire shipping consumer.

**The last shape then measured out in the same session.** The two-level
chunk-staged split — both levels binned through shared and flushed as 256 B
chunks, so no store anywhere leaves a warp uncoalesced — is strictly *worse*
than the one-level split it was built to beat: 13.3 ms against 9.0 at stock,
worse again at 140 W. Granularity is the mechanism here, with bandwidth uninvolved: an
L2-resident tile only holds one coarse bucket, which forces 256 serialized
coarse-by-phase steps of ~2.6 records per thread each — too shallow to hide
latency — and a second pass doubles that overhead. Three targeted attacks on
the one-level's 9.0 ms all failed (4-wide MLP batching cost occupancy and ran
slower; a stride-9 staging layout against shared bank conflicts moved nothing;
row work-stealing moved nothing), so 9.0 / 9.5 ms (stock / 140 W) stands as
the measured floor of any split lane — and that closes the overlapped variant
by bound: even a *perfect* overlap of split and match floors the pair at
emit + 9.0 = a stock regression and ≤ +8 % at 140 W.

**The closing statement for the axis**: the fine partition the match needs —
partner adjacency — costs one scattered touch of the full state per round, and
at this record width and geometry every alternative to paying it at emit time
(one-level L2 split, two-level chunked split, their overlapped forms, the sort
path, a DRAM radix) pays more somewhere else. The shipping per-element scatter
emit at the ~250–265 GB/s floor is the cheapest measured way to buy it; the
chunk-staged emit's 2× rate win is real but sits behind a consumer that cannot
be built at this geometry. *Both reopening conditions have since measured out —
the sub-9.0 ms split above, the coarse-direct match in the entry below — so the
axis is closed with no open reopening condition.*

### Round 2's register wall is not a wall: 48 registers, zero spill, +0.023 ms
<details>
<summary>Details</summary>

*(2026-08-15. Closes the "the two knobs have never been moved together" open point.)*

A fifth resident block in round 2 needs **≤ 48 registers and ≤ 19456 B of shared**. Round 2
sits at exactly 64 registers — the four-block cliff — and the standing reading was that
this is a hard wall: `MXBM_MB_RD2` had measured null, and the `kFCap` sweep never reached
the shared line, so each knob was null *for the other's reason*.

Asking `ptxas` for five blocks settles the register half. **Every implicit-bits round-2
variant goes 64 → 48 registers with zero spill**, and the cost at equal occupancy is
**+0.023 ms (+0.073 %)** — 7 % of this instrument's floor.

| r2 variant | REG | spill |
|---|---|---|
| implicit-bits, IMPB 16 and 17, ±match-first, ±arena — **8 rows, the shipping path** | 64 → **48** | **none** |
| plain packed record, quad, quad16 | 64 → 48 | 8 B |
| match-first (plain, quad, arena) | 64 → 48 | 16 B |

It is the address-implied-bits pack that makes it free: the 6-u64 record leaves fewer
values live across the staging read, and every record that is not packed spills.

Method: `-DMXBM_MB_RD2=5` on both the CUDA and CXX flags, shared untouched, so **both
builds run at four blocks/SM** and the delta is pure register pressure. ABBA, three pairs,
60 s arms, clocks locked, headless; quoted on `solves/s`, which prints four significant
figures where the ms/solve median prints three. A arms 31.97–32.01, B arms 31.96 ×6 — they
do not overlap, so the +0.023 ms is real rather than noise. KAT 3/3 on all 15 geometries.

**The applied-assert came free.** The occupancy contract runs POST_BUILD and failed the
build on eighteen round-2 rows, naming each drift — which is exactly the proof that the
define reached the template, and is why the contract fails the *build* rather than a test.

**So shared is the sole gate, and the deficit is bigger than previously stated.** The
`cuobjdump` SHARED figure does not include the 1024 B driver reserve — verified against the
contract's own headroom line, `23624 + 952 = 24576 = 102400/4 − 1024`. The five-block line
is therefore `102400/5 − 1024 = 19456 B` and the deficit is **4168 B, or 13.03 B per staged
element** at `kFCap` 320. Round 2's shared is `lwork` 17920 + `lleaf` 2560 + `lgi` 1280 +
`lchain` 1280 + `tab` 512 + 72. No *pair* of the available cuts reaches the line: staging
the A-form (`lwork` 7 → 6 u64, −2560 B) plus either folding `gi` into `lleaf`'s spare bits
or deleting `lchain` (−1280 B) lands at 19784, **328 B short**. All three together give
18504 and the block.

Against a prize of **≈ −0.10 ms** — r1's own measured block decay is −0.52 for its fourth
and −0.15 for its fifth, and round 2's fourth was −0.33 — that is three structural changes
to the shared layout for something near the instrument floor. Recorded as bounded rather
than pursued.

</details>

### The coarse-direct match measures out: the in-L2 chain match's own floor sits above the bar

*(2026-08-19, same day, third session. The kill probe for the closure's second
reopening condition — "a match structure that consumes coarse buckets
directly". Same mock pair and gates as the round-pair probe; arm A unwindowed
(its best config). Arm B = the chunk-staged coarse emit (unchanged, 2^7 bins)
+ an in-L2 chain match per coarse bucket: the grid loops the 128 buckets with
one sync each (window ≤ 2 buckets ≈ 34 MB, L2-resident by capacity), each
thread interleaves inserting bucket b — one `atomicExch` into a 24-bit-keyed
chain-head table, one link store — with walking bucket b−1's snapshot chains
from registers, gathering each equal-key partner from the L2-resident bucket
and emitting one child per unordered pair. No fine-row form ever exists.)*

| pair, ms | A emit | A cons | A pair | B emit | B cons | B pair | B vs A |
|---|---|---|---|---|---|---|---|
| 285 W, work 0 | 8.90 | 4.67 | 13.57 | 4.14 | 10.37 | 14.51 | −7 % |
| 285 W, work 16 | 8.46 | 4.80 | 13.25 | 4.56 | 11.44 | 16.00 | −21 % |
| 140 W, work 0 | 11.72 | 4.78 | 16.50 | 5.04 | 11.47 | 16.50 | ±0 % |
| 140 W, work 16 | 12.50 | 5.62 | 18.11 | 6.04 | 13.61 | 19.65 | −8 % |

**Dead at both operating points** — the pair never wins (best case a dead tie
at 140 W / work 0) and regresses stock everywhere, against a bar of "≥10 % at
140 W with stock clean". Six implementation variants converged on the same
decomposition, each mechanism measured:

1. **The bucket window must be enforced.** Free-running grid-stride lets
   thread positions spread until the partner gathers miss L2 (6.2 GB DRAM,
   48 % sector hit, 35 ms); the cooperative per-bucket sync restores residency
   for ~0.2 ms per layer, and the gathers then run at the census's L2-scatter
   rate (~620 GB/s independent) — no headroom left in the walk itself.
2. **The child emit cannot escape the per-element allocation floor.** Emitted
   per-element it costs +4.0 ms at stock / +5.6 at 140 W, and the destination
   geometry only trades failure modes: a 2^16 fine layout's 2 MB write
   frontier is evicted mid-merge under the 34 MB window and every 16 B store
   pays a partial-sector RMW (+1.78 GB DRAM measured); ≤ 2^8 bins serialize
   on the allocation counters (+2 ms); 2^12–2^14 is the flat optimum. The
   chunk-staged escape — the same protocol the coarse emit itself uses — was
   built twice for the consumer side (block-lockstep rounds, then concurrent
   per-thread chains) and **lost more to its own block coordination than the
   RMW it saved** (14.5–14.7 vs 11.4): inside a divergent walk, coalescing
   requires lockstep that caps memory-level parallelism at 0.27 eligible
   warps.
3. **No DRAM/L2 overlap materializes.** The match half prices at exactly the
   sum of its parts (stream read 3.7 + push 0.2 + walk 3.5), the walk is
   latency-bound at 0.09 eligible warps with the LSU at 2–7 %, and raising
   occupancy 4 → 6 blocks/SM moves nothing — in-flight requests are capped
   below occupancy, so the extra warps queue.

Component floors put even a perfect implementation at ~9.2–9.5 ms per 2.15 GB
layer against bars of ~8.7 (stock parity) and ~10.6 (the 140 W win): the stock
bar is structurally out of reach, and the 140 W margin is smaller than the
overlap no variant could buy. **This consumes the closure's second reopening
condition: axis 4 is closed with none open**, and the streaming pipeline that
a surviving probe would have green-lit does not get built. Harness and raw
data: `docs-internal` campaign `2026-08-19-coarsematch` (internal checkouts).

### What `apply_mix` actually costs: everything in round 3, nothing at all

<details>
<summary>Details</summary>

*(2026-08-15. Bounds every mix-restructuring idea, in both directions.)*

`MXBM_ABL_MIX=R` deletes round R's `apply_mix` while keeping the combine and still storing
the child, so the global access footprint and the bucket distribution are preserved. Two
builds, ABBA against the shipping binary, 45 s arms:

| round ablated | full | ablated | what **every** mix in that round costs |
|---|---|---|---|
| **3** | 30.979 ms | 30.955 ms | **+0.024 ms — nothing** |
| **1** | 30.993 ms | 30.257 ms | **+0.736 ms** |

Round 3 runs at 76.5 % of DRAM peak with its ALU pipe near 18 %, so its arithmetic is
already hiding entirely inside memory stalls — deleting all of it recovers nothing. Round 4
is at 82.7 % and the same holds a fortiori. Round 1 is stall-bound at 76 % ALU, and there
the mix is real work.

**So the whole mix-restructuring family has a ceiling of 0.74 ms and it lives only in
round 1.** Round 2 has no mixes left at all — the
[w0-checkpoint record](#the-w0-checkpoint-pair-record-repriced-by-the-address-bits-076-ms-24-)
removed them — and rounds 3 and 4's are free. Carry-save trees, butterfly reductions and
mix-partial inheritance are all bounded by that number wherever they target r1, and are
dead outright wherever they target r3 or r4.

Stated generally, and it is the companion to
[round 3 not wanting a fourth block](#round-3-does-not-want-a-fourth-block--occupancy-pays-only-where-a-round-is-latency-bound):
**on a DRAM-bound round neither occupancy nor arithmetic buys anything, because the pipe is
already idle.** Occupancy pays where a round is latency-bound; so does instruction count.

</details>

### The `bb + sm = 17` line is not free to leave: a finer sub-mask costs 24 %

<details>
<summary>Details</summary>

*(2026-08-15.)*

A recurring proposal is to take the DRAM-bound rounds off the `bb + sm = 17` line to a
finer geometry, halving the staged group and with it every shared array. `MXBM_BB` and
`MXBM_SM` are runtime overrides, so the cost side is free to measure — and two facts make
that measurement the honest one rather than a partial one: **`kFCap` is a compile-time
constant**, so a finer group does not shrink the shared arrays, and **SUBPASS is off** in
every shipping instantiation, so the sub-mask rescan is still live.

| geometry | ms/solve | verified solutions/solve |
|---|---|---|
| (16,1), shipping, two arms | 30.9 / 31.0 | 1.99 |
| **(16,2)**, i.e. `bb + sm = 18` | **38.4** | 1.99 |

**+7.43 ms, +24.0 %**, with the multiplier unchanged — real work, and no dropped elements.
The dominant term is not the rescan: at a mean group of 132 against `kWG` 256, **half the
lanes idle in every all-lanes loop**.

So the finer geometry cannot be evaluated without `kWG` and `kFCap` moved with it, and
that programme starts 7.43 ms behind — chasing a residency prize on exactly the two rounds
measured to be occupancy-insensitive.

</details>

### Splitting the terminal record into two planes costs 11 %: sectors, for the third time

<details>
<summary>Details</summary>

*(2026-08-15. A built and measured null. Patch archived; it is not in the tree.)*

Round 5 consults **32 bits** of the 128-bit record round 4 writes it: `combine` at
Lout = 24 yields `((a ^ b) >> 24) & 0xFFFFFF`, so only w0's bits 24..47 decide acceptance,
the low 24 are the key of which the bucket address supplies all but 8, and bits 48..63 are
dead. `gi` and `lead` are needed only inside the accept branch — **`combine` is symmetric,
so the left/right ordering never reaches the collision test** — and that branch fires
about twice per solve.

So the record was split into a **4 B hot plane** streamed by every terminal lane and an
**8 B cold plane** read only for a pair that collides, both inside the record set already
allocated. Round 4 writes 12 B instead of 16; the terminal streams 4 B per element instead
of 8 B for all plus 8 B for the half clearing the sub-mask.

**It loses 3.51 ms — +11.2 %.** ABBA, clocks locked, arms non-overlapping (shipping
31.984 solves/s across five arms, split-plane 28.760 across five, no spread at all on the
B side).

| | shipping | split-plane |
|---|---|---|
| ms/solve | 31.266 | **34.771** |

The mechanism is the one this project keeps rediscovering. Round 4's emit scatters to a
bucket chosen by the child's key, and consecutive slots within a bucket are filled by
different blocks at different times, so nothing write-combines. **Each store dirties its
own 32 B sector regardless of how few bytes it carries**, so splitting one 16 B store into
a 4 B store and an 8 B store does not cut write traffic — it *doubles the sector count*:

| r4 emit | sector traffic |
|---|---|
| one 16 B store per element | 1.07 GB |
| 4 B + 8 B, two stores | **2.15 GB** |

That is the same currency that killed [the h=1 pipeline](#the-h1-pipeline-built-and-killed-the-caps-currency-is-l2-sectors-not-dram-bytes)
and M2a's keys-only staging. Third instance, and the first where the amplification is on
the *write* side.

**What it corrects about the idea it came from.** The narrowing is real and the bit budget
is right, but the prize is much smaller than a byte count suggests, and it has a hard
shape requirement: **the record must stay one store.** An 8 B single-plane record touches
exactly the same 32 B sector as today's 16 B one, so the write side gains nothing at all;
the entire prize is the terminal's *read*, which becomes a contiguous 8 B/element stream
instead of a 16 B-strided one — worth about **−0.5 ms**, where a byte-rate
estimate gives −1.4 ms. And 8 B is only reachable if `gi` leaves the record, because
key 8 + sig 24 + gi 26 + lead 25 = **83 bits**; without `gi` it is 57. So slot-indexed
reference rows are a hard prerequisite; nothing about them is optional.

Gates on the measured variant, for the record: CUDA KAT **3/3 on all 15 geometries**, the
occupancy contract holding blocks/SM at 6 while shared fell 8196 → 5124 B, and an
**element-dependent poison of the hot plane collapsing the KAT to 92 failures**, which is
what proves the new path was live rather than silently bypassed.

</details>

### The back-reference rows are gone: recovery replays instead, −2.03 ms and −688 MiB

`all_left`/`all_right` were written for **every** emitted child of rounds 1–4 — 134 M of
them, 8 B each, 1.07 GB and 8.7 % of the solve's traffic — and read for the **two**
survivors. None of the five rows is written now. Recovery reconstructs a survivor's
ancestry by re-running the rounds on the buckets along its path, identifying each child by
**content** rather than by slot, which is what makes it sound despite slot assignment being
nondeterministic run to run.

**What the rows were worth.** `MXBM_ABL_REFS` as a per-round bitmask skipped the two stores
while keeping every instantiation, the `gi` atomic and the row allocations:

| rows off | ms/solve | delta |
|---|---|---|
| none | 31.277 | — |
| 1 + 2, written by r1 and r2 | 30.921 | **−0.355 ms** |
| 3 + 4, written by r3 and r4 | 30.609 | **−0.668 ms** |
| all four | 30.266 | **−1.011 ms** |

Additive to 1.3 %. Controls: verified solutions/solve went **1.99 → 0.00**, so the stores
really went, and shared memory was byte-identical in every kernel, so no round gained a
block.

**Why a row is worth four times what the same bytes of record are.** `MXBM_ABL_EMIT=3`
removed a similar 1.07 GB of round 3's *record* writes for −0.355 ms. The difference is the
access pattern: the record scatter is bucket-random, so each store dirties its own 32 B
sector, while `gi` is warp-contiguous — ptxas aggregates the counter's atomic — so the
reference rows were written **coalesced**. **Bytes are the right currency for a coalesced
stream and sectors for a scattered one**, and that single distinction explains both this
result and why [splitting the terminal record into two planes](#splitting-the-terminal-record-into-two-planes-costs-11--sectors-for-the-third-time)
lost.

**Every hint the replay needs was already being stored as dead bits.** A replay has to know
which bucket to re-run, and the child does not carry its parents' key — `combine` shifts
the collided bits away. Both hints ride in fields that provably reach nothing:

- **Round 4's child** carries its parents' bucket where the terminal round cannot look. The
  terminal combines at Lout = 24, so only word 0's bits 0..47 reach the acceptance test.
- **Round 3's child** carries its parents' bucket in **work word 5**, which round 4 reads
  and cannot use: at Lout(4) = 288 the shift drops everything above bit 311, so word 5
  never reaches round 4's output. That word was already known dead — see
  [round 3 stores a work word round 4 never reads](#round-3-stores-a-work-word-round-4-never-reads--worth-268-mb-and-nothing-in-time),
  which measured deleting it worth 268 MB and nothing in time. It is worth 0.35 ms as a
  place to put 17 bits.

Only the **bucket** is needed, and never the (bucket, sub-mask) group: elements in different
sub-masks have different keys and can never pair, so a replay stages the whole bucket and
finds the same pairs.

**The identification is by content and it is exact.** `replay_r4` matches a candidate child
on the 47 bits of word 0 the record keeps; `replay_r3` matches on work words 0..4, 320
bits. Against ~600 candidate pairs in a bucket, neither can collide at a rate this
instrument could see. Left from right is decided by the same `(lead, gi)` rule the round
applied, so the leaves come out in tree order without the walk.

**Rounds 1 and 2 need no replay at all** — but they did need a buffer. A round-2 output
record already carries the element's four seed indices **in tree order** (`ctree[0..1]` from
the left parent, `[2..3]` from the right), so once `replay_r3` has named the survivor's
eight round-2 ancestors by slot, reading those records *is* the 32 leaves. The obstacle was
that round 2's output did not survive: the pipeline ping-pongs two record sets and round 4
overwrote `elem[0]`, which is exactly why the octo record is the one carrying eight leaves.
Round 4 now writes a plane of its own — and because that record is down to 8 B, the plane
costs 0.32 GiB against the 0.77 GiB of rows it releases.

**The 8 B round-4 record.** 48 bits of word 0 can reach the terminal round's test, and of
the 24-bit key only the 24 − bb bits the bucket address does not already carry are ever
consulted — 32 bits of information in what was a 16 B record. `gi` and `lead` went with the
row that was the last thing indexing them: `combine` is symmetric, so left from right can be
settled in recovery off the lead the replay reads out of the left parent's record. The
leftover bits pay for the bucket hint. Round 4's `gi` atomic goes too.

**Measured, ABBA at locked clocks, 45 s arms, zero spread on every arm:**

| shipped | ms/solve | delta | footprint |
|---|---|---|---|
| baseline | 31.277 | — | 7590 MiB |
| round 4's row replayed | 30.955 | **−0.322** | −264 |
| + the 8 B round-4 record | 30.008 | **−0.947** | 0 |
| + round 3's row replayed, rows 1–3 with it | 29.248 | **−0.760** | −424 |
| | | **−2.029 ms** | **−688 MiB** |

63.6 → 68.0 sol/s. The 8 B record is worth three times what halving the terminal round's
read can explain — that kernel is only 0.98 ms — so most of it is round 4's scatter: at
8 B a slot four elements share a 32 B sector where two did.

The footprint arithmetic differs by rung. Rows 4 and 5 go on every CUDA rung but octo
(−0.26 GiB); rows 1–3 go only where the implicit-bits record admits the replay, at
−0.77 GiB against the plane's +0.32. The octo rungs are unchanged and the CUDA floor stays
at **1.90 GiB**.

**It pays half again as much under a cap as it does at stock**, and for the reason the
w0-checkpoint record established: what came out is **stores and an atomic**, which are
instructions, and under a cap the currency is instructions issued rather than bytes moved.
Against the same sweep on the previous kernel: **+10.3 % at 120 W, +8.7 % at 140, +8.1 %
at 160, +8.4 % at 180, +9.3 % at 190**, against **+6.6 % at 285 W** — and only +4.6 % at
the 100 W floor, where the card is slow enough that nothing is the constraint. Both
crossings against the reference miner move from ~200 W to **~183 W**, and the efficiency
peak improves from 0.2746 to **0.2968 sol/s/W at 220 W** (3.642 → **3.369 J/solution**),
which is within 0.8 % of that miner's own best point and does 25 % more work at the tie.
(The singleton filter has since taken the crossings to ~181 W and the peak to 0.3009 at
210 W; the live figures are always [the cap table](performance.md#both-miners-under-the-same-cap).)
On the 5001 memory rung, where each miner is at its own best configuration, the low band
stops being a deficit: MXBM leads at 120 and 140 W and ties at 160.

**The gate.** `-DMXBM_R4_ROWS=1` restores every row and the 16 B record, runs both
recoveries side by side and requires the 32 leaves to match index for index. **1426 solves
across (16,1), (17,0) and the dense-cap rung, zero differences**, before a single store was
deleted — which is what separates "the replay is correct" from "the deletion pays". KAT
3/3 on all 15 geometries, drops 0.

### Two instruments that returned false greens, and why

Both were caught this session, both had already produced a passing gate on a change they
could not see.

**`bench_rounds` and `test_gpu_solver` drive the OpenCL backend.** `GpuSolver` holds a
`cl_runtime.h` `Runtime`; neither command touches CUDA. The first ABBA on `bench_rounds`
returned **twelve identical readings** across both arms, which is what a CUDA-only change
looks like through an OpenCL instrument. The CUDA gates are **`tests/test_cuda_solver`**
(3/3 goldens on 15 geometries) and **`mxbm --benchmark BEAM-III`**.

**A constant XOR is not a positive control on this pipeline.** Poisoning one fixed bit of
every stored word 0 left the KAT at 3/3 — which reads as "the code never ran", and is not:
`combine` XORs the two parents, so a constant flip cancels exactly and is invisible to
everything downstream. A poison here must be **element-dependent**. `w0 ^ ((gi & 0xFF) << 40)`
destroyed the goldens on every geometry, which is the control that makes the clean run mean
something. The general rule: on a pipeline whose next step is a pairwise XOR, a perturbation
must vary *between* the elements that will be paired, or it proves nothing.

</details>

### The w0 checkpoint on the octo record: the mixes go, and the record does not grow

<details>
<summary>Details</summary>

`rebuild_r4` is the deepest re-derivation on the ladder — two `rebuild_r3` calls, a combine
and a mix, which is **56 siphashes and 15 `apply_mix` calls per staged element**. Storing
the child's post-mix work word 0 deletes **every one of the mixes and 8 of the siphashes**,
because `apply_mix` writes work word 0 and nothing else and `combine`'s word *i* reads only
`x[i]` and `x[i+1]`: words 1..6 of every node in the tree are a function of the eight
leaves' *seed* words 1..6 alone, and the whole mix chain reaches word 0 and stops there.
The same argument the [pair record](#the-w0-checkpoint-pair-record-repriced-by-the-address-bits-076-ms-24-)
uses, three levels deep instead of one.

**The record does not grow.** The octo record was key 24 + 8 × 25 + gi 26 = 250 of its 256
bits, and word 0 is 64 more — but two of the fields it already carried were worth less than
they cost:

- **`gi`, all 26 bits.** No reference row indexes a round-3 element by one any more, and
  its last consumer is round 4's left/right tiebreak, which orders staged elements by input
  slot just as well. That is the substitution the `quad16` record already makes one round
  up, and the goldens gate it.
- **14 of the key's 24 bits.** Only the low `24 − bb` reach the sub-mask filter, the chain
  hash and the in-group key compare, and `bb + sm = 17` makes that exactly `7 + sm ≤ 10` on
  every rung the octo record runs on. The rest are the bucket address. Word 0's own low 24
  bits *are* the key, and round 4's combine reads bits 24..63 alone, so the dropped 14 are
  read by nothing.

10 + 40 + 8 × 25 = **250 bits of 256** again, in a fixed layout — no template parameter and
no address arithmetic, unlike the r2 → r3 implicit-bits pack. The staged word 0 comes back
with bits 10..23 zero, which is the trade `t5_rec` already makes in the terminal record.

**Measured**, `A B B A` interleaved, 40 s arms, KAT 3/3 on all four octo rungs and every
drop counter zero with the arena's spill counter nonzero as the positive control:

| rung | plain octo record | w0 checkpoint | delta |
|---|---|---|---|
| (16,1) + dense caps + octo | 55.305 ms | **50.917** | **−4.388 (−7.93 %)**, 12 arms |
| (14,3) + dense caps + octo, the 1.90 GiB floor | 61.143 | **56.972** | **−4.171 (−6.82 %)**, 8 arms |

No range overlaps, and both arms report 1.99–2.02 verified solutions/solve throughout.

**Applied-asserted in the SASS**: of 67 kernels exactly five differ — the two round-4 octo
ones at **−1816 instructions each** (10680 → 8864 and 9352 → 7536, −17.0 % and −19.4 %),
the two round-3 octo emit ones at −8, and `recover` at +8 for the new unpack. Registers
hold at 80, shared is unmoved and **blocks/SM stays 3**: the prize is deleted instructions,
not residency. The 88 B of spill those kernels carried is gone — a mix-free lane's live
state fits the same 80 registers with nothing left over.

The algebra was settled on the host before any of this: the lane is bit-identical to
`rebuild_r4`'s words 1..6 over 200 k random trials, with word 0 never coincidentally equal
(0 of 200 000) and words 1..5 covering their full range while word 6 stays zero, which is
`Lout(3) = 376` doing what it should. The pack round-trips exactly over 500 k trials.

**The gi atomic goes with the field**, and it is worth **−0.045 ms** on its own — below the
floor, but a strict deletion (exactly two `ATOM` instructions leave the object) so it ships
with the record. It also cuts across
[the census that could not move to the producer](#the-key-census-cannot-move-to-the-producer-436-ms-against-the-076-it-would-replace)
from the other end: a **single-address** global atomic per emitted element costs 0.045 ms,
where that lead's **4 M-address** one cost 2.8–4.36. Contention was never the axis; the
funnel through L2 was, and this is the second measurement that says so.

**What it does not change**: no shipping kernel. The stock ladder never instantiates a
round-4 octo kernel, so the locked-clock headline and the cap table are untouched. And it
does not make the octo record a speed lever —
[that closure](#the-octo-records-two-halves-priced-apart-round-4s-rebuild-is-a-flat-16-ms)
stands, only cheaper: on this build the rebuild costs **+15.9 ms** over quad (16,1) + dense
caps at 35.0, where it was +18.1 before. A reach rung is still a reach rung.

</details>


---

### The w0 checkpoint on the quad record: the record holds word 0 with no repacking

<details>
<summary>Details</summary>

The same lever one round up from the octo record, and the cheapest of the three to carry.
`rebuild_r3` is two `rebuild_r2` calls, a combine and a mix — **28 siphashes and 7
`apply_mix` calls per staged element**. Storing the child's post-mix work word 0 deletes
**every mix and 4 of the siphashes**, by the argument the
[octo checkpoint](#the-w0-checkpoint-on-the-octo-record-the-mixes-go-and-the-record-does-not-grow)
spells out: `apply_mix` writes work word 0 alone and `combine`'s word *i* reads only `x[i]`
and `x[i+1]`, so words 1..6 are a function of the four leaves' *seed* words 1..6 and the
whole mix chain reaches word 0 and stops.

**Word 0 costs nothing to carry here, and needs no repacking at all.** The quad record was
key 24 + 4 × 25 + gi 26 = 150 of its 192 bits, and word 0's own low 24 bits *are* that key
— so the record simply stores word 0 whole in its first u64 and moves the leaves and `gi`
down into the other two, where the packed record's `r3_p0`/`r3_p1` already put them:

```
w0 = the child's work word 0        w1 = r3_p0(l0,l1,l2)     w2 = r3_p1(l2,l3,gi)
```

64 + 4 × 25 + 26 = **190 bits of 192**. Nothing is dropped, so unlike the other two forms
this one needs neither the implicit-bits pack nor the perfect table, and the staging loop's
`rec0 & 0xFFFFFF` key extraction reads the key exactly where it always was.

The 16 B `quad16` record the **octo** rungs read instead cannot take it — 24 + 4 × 25 = 124
of 128 leaves no slack, and keeping only the address-implied key bits still needs 150 — so
this is a quad-rung lever and the octo rungs keep paying `rebuild_r3` in full.

**Measured**, `A B B A` interleaved, 40 s arms, KAT 3/3 on all four quad rungs and every
drop counter zero with the arena's spill counter nonzero as the positive control:

| rung | plain quad record | w0 checkpoint | delta |
|---|---|---|---|
| (16,1) quad | 36.021 ms | **34.180** | **−1.841 (−5.11 %)**, 12 arms |
| (14,3) quad + dense caps, the 3.78 GiB floor | 44.748 | **42.804** | **−1.944 (−4.34 %)**, 8 arms |

No range overlaps, and both arms report 1.99–2.02 verified solutions/solve throughout. The
control's 36.021 and 44.748 reproduce the ladder's own 36.05 and 44.78 to the digit.

**Applied-asserted in the SASS**: of 67 kernels exactly five differ — the four round-3 quad
ones at **−872 instructions each**, and round 2's quad emit at +8 for the new pack. Nothing
else in the object moves, which is what says the checkpoint is off on the octo rungs' 16 B
path. Registers, shared memory and blocks/SM are all unchanged; the prize is deleted
instructions.

The algebra needed no new proof: `rebuild_r3_lane` is the first three lines of the already
shipping `rebuild_r4_lane`, so the octo record's 200 k-trial identity test covers it, and
the pack is `r3_p0`/`r3_p1` unchanged. `rebuild_r4_lane` now calls it rather than repeating
it.

</details>


---

### Deleting a quarter of the walk's shared reads buys nothing, because the chains are length one

<details>
<summary>Details</summary>

The chain walk reads `INW` words for **both** sides of every pair, and one of the two slots
is always the lane's own `pos` — re-read from shared at every chain step. `combine` is
`a.w[i] ^ b.w[i]`, so the pair's order never reaches it and the lane's own element can be
read once for the whole chain instead, with the left/right ordering left to decide only
which side supplies the leaves.

It works, and it is worth nothing: **+0.028 ms (+0.10 %)** over 12 interleaved arms with
the ranges overlapping. In the SASS the inner walk loop goes from **12 `LDS` to 9** and the
three hoisted loads move out to the per-element loop; registers rise (r3 56 → 68, r4 47 →
50) with **every kernel holding its blocks/SM** and no spill, so the residency gate the
idea was expected to fail is not what stopped it.

**The mechanism is the chain length.** Each round emits about as many children as it
consumes, and emits are exactly walk steps, so the walk averages **one chain step per
element**. Hoisting a per-step load to a per-element one is therefore break-even by
construction; the only saving is the variance term, the elements whose chain is longer
than one. That bounds the whole "amortise across the chain" family, and not merely this member
— and it closes the walk's shared traffic from the opposite end to
[the layout closure](#the-shared-bank-conflicts-are-the-chain-walks-and-no-layout-reaches-them),
which found that no rearrangement reaches that loop. Rearranging does nothing and deleting
does nothing.

</details>


---

### Locking the memory clock to its reported maximum costs 0.95 %, and the offset is the knob that does not

<details>
<summary>Details</summary>

Rounds 3 and 4 are 45 % of the solve at 77.4 % and 88.6 % of DRAM peak, so the memory
clock is the one knob on the resource that binds them. The card's load clock is **10251
MHz** — 25 of 25 samples through a 40 s run, from a single `nvidia-smi -lms` sampler that
costs 0.0 % — against a driver-reported maximum of 10501. The headroom is real.

Taking it makes the solver **slower**. `nvidia-smi -lmc 10501` against the stock clock,
`A B B A` interleaved, 12 arms of 40 s:

| arm | ms/solve | min | max |
|---|---|---|---|
| stock memory clock | **28.733** | 28.711 | 28.744 |
| locked to 10501 | 29.005 | 28.994 | 29.011 |

**+0.272 ms, +0.95 %**, and the ranges do not come within 0.25 ms of touching.

**The board is at its power cap in every kernel**, so the memory clock is not bought with
nothing — a locked P-state raises memory voltage with it, and those watts come out of the
core. The solve is **54 % core-bound** (entry, r1 and r2 at 77–99 % of the ALU pipe)
against 45 % memory-bound, so trading core clock for memory clock loses on the majority of
the solve. Third instance of the same currency after the SM-count and duty-cycle results:
under the cap the resources are fungible with watts, and the split already sits where the
governor put it.

**The V/F offset is a different knob and moves the other way.** `--moff` shifts the curve
— the same voltage at a higher frequency — where `--mclk`/`-lmc` selects a P-state. At
`--moff 1600` (11051 MHz, +7.8 %), bracketed `A B B A` over 12 arms:

| arm | ms/solve | range | J/sol |
|---|---|---|---|
| stock | 28.733 | 28.711 – 28.760 | 4.090 |
| `--moff 1600` | **28.050** | 28.027 – 28.066 | **4.008** |

**−0.682 ms (−2.37 %) and −0.082 J/sol (−2.00 %)**, the ranges separated by 0.645 ms,
verified solutions/solve in band on both arms and every drop counter zero. Faster *and*
more efficient, which follows from the board being pinned at its limit: power is constant,
so the energy win is the speed win.

**The offset pays the same tax, about five times smaller.** The core clock still falls
under it — 2670 → 2640 — so the trade is the same one the lock makes and simply at a much
better rate. A two-clock model over the whole sweep (45 % of the solve tracks the memory
clock, 54 % the core) predicts the measurement to **101 % and 97 %** through +1400, then
decays to **80 %** at +1600 and **71 %** at +2200 while the clock keeps climbing. That
decay is the retry wall arriving gradually rather than as a cliff: GDDR6X answers timing
it cannot hold with link-level retries, which cost bandwidth without ever producing a
wrong answer. It is 28 points of decay against the ±5 the 0.1 ms print resolution can
manufacture, so it is real, and it is why the bracketed figure sits at +1600 rather than
at the +2400 that measured 1 % better.

**It is worth less here than to a miner that stores.** MXBM moves 10.69 GB a solve where
[a state-storing design moves 17.67](#lolminer-measured-under-ncu-the-state-storing-design-confirmed--and-its-54-sols-ceiling-is-a-dram-roofline)
and is against a DRAM roofline in nearly every kernel; 45 % of this solve is memory-bound
against nearly all of that one. The same clock therefore buys a competitor more than it
buys us, which is the measured reason — not merely a methodological one — that
[a comparison run must be taken at stock](benchmarking.md).

What is settled is the instrument:

> Where a board sits at its power limit and the work is majority core-bound, the two
> knobs are not interchangeable: a locked memory P-state is bought from the core clock
> and an offset is not.

The sign and the size both belong to the card and its limit, with the solver uninvolved —
a card with headroom under its limit has no such trade to make, and the same reasoning run
backwards is why locking memory *down* to the low rung pays under a low cap. What
generalises is that the two knobs are not interchangeable; the 0.95 % does not.

</details>


---

### The fourth w0 checkpoint: entry to round 1 needs 73 bits, and would lose if it had them

<details>
<summary>Details</summary>

Three records carry the child's post-mix work word 0 — the
[pair](#the-w0-checkpoint-pair-record-repriced-by-the-address-bits-076-ms-24-),
[quad](#the-w0-checkpoint-on-the-quad-record-the-record-holds-word-0-with-no-repacking)
and [octo](#the-w0-checkpoint-on-the-octo-record-the-mixes-go-and-the-record-does-not-grow)
forms. The fourth boundary is **entry → round 1**, and it is the natural next question:
`entry_body` computes a seed element, applies the mix, and then keeps 24 bits of word 0,
while round 1 recomputes the whole thing from the index. Storing word 0 would delete one
of round 1's seven siphashes and its `apply_mix` outright.

It is closed three ways, all at the desk.

**It does not fit the 8 B record.** Round 1's `combine` reads `x[0]` only through
`(x[0] >> 24)`, so word 0's low 24 bits are read by nothing but the key — the same fact
the octo record spends. The record would then need the seed index (25), word 0's bits
24..63 (40) and the `24 − bb = 7 + sm` key bits the bucket address does not carry (8 at
(16,1)): **73 bits.** Nine over, and there is no field left to squeeze — the index is
2^25 exactly and every one of those 40 bits reaches the combine.

**At 16 B it loses by 3×.** This ledger's own exchange rate — a siphash call costs 8.9 ps
of solve time, a byte written and read back through a bucket scatter costs 4.35 ps — puts
8 extra bytes over 2^25 elements at **1.17 ms** against a saving of ~0.4 ms. The
[terminal record's 16 B → 8 B](#splitting-the-terminal-record-into-two-planes-costs-11--sectors-for-the-third-time)
agrees independently at −0.947 ms over the same element count and the same scatter shape.
A call is worth about two bytes and this one would buy 8; that is the 3.9× store-versus-
derive constant, unchanged.

**As a side plane it is the sector doubling**, +11.2 % measured.

So the checkpoint family is complete, and it is complete by arithmetic rather than by a
null: of the four boundaries, three had a field worth less than word 0 and the fourth is
nine bits short of being able to ask.

</details>


---

### The GPU computes 99.3 % of a solve

<details>
<summary>Details</summary>

Every census here had measured *kernels* — pipes, sectors, stalls, clocks. None had
measured the gaps between them, which left an unbounded family: memset traffic, launch
overhead, and the host round-trip the solve makes before recovery.

Every dispatch goes to the **null stream**, so the GPU timeline is fully serial and the
census is a subtraction. A steady-state solve at the shipping rung is **25 dispatches — 8
kernel launches, 14 `cudaMemset`s and 3 copies**, two of the copies blocking pageable D2H.
Per-kernel durations from `ncu --metrics gpu__time_duration.sum`:

| kernel | ms |
|---|---|
| round 1 | 4.790 |
| round 2 | 7.936 |
| round 3 | 8.523 |
| round 4, carrying the next solve's entry scatter as extra blocks | 6.391 |
| terminal | 0.791 |
| `replay_r4` + `replay_r3` + `recover_from_l2` | 0.176 |
| **total** | **28.608** |

Against **28.8 ms/solve** measured in the same process: **99.3 % of a solve is kernel
execution**, and the whole non-kernel budget — 14 memsets, 3 copies and 25 launch gaps
together — is **0.19 ms, 0.66 %**, under the ~1 % floor this instrument can resolve. The
memsets move ~1.25 MB against the solve's 10.69 GB of compulsory traffic, so what they cost
is launch overhead, with bandwidth uninvolved.

`ncu`'s defaults must be turned off for this to mean anything: with the standard
`--cache-control all --clock-control base` the same sum reads **31.12 ms**, which is more
than the wall time and therefore impossible. That is the positive control that
`--cache-control none --clock-control none` is measuring the real thing.

There is no idle-GPU family. The entry pass no longer appears as its own dispatch at all —
speculative entry carries it inside round 4's launch, at a grid 1.5× round 4's own.

</details>


### OpenCL reaches the same 1.90 GiB floor: two bugs in the octo rung, both about addresses

<details>
<summary>Details</summary>

The OpenCL octo kernels had been built, committed and gated off (`allow_octo=false`)
because round 4 was placing children where their keys did not name. Neither cause was in
the record, the rebuild or the tiebreak — all three had been proved out — and both were
about *where a reference is written and read*.

**Round 4 wrote its reference row three rows past the end.** The five-row layout puts
round *r*'s row at `(r-1) * capacity`, and an octo build allocates **one** row plus the
survivor tail, because a record that carries its own eight leaves is what recovery was
walking rows 1–3 to reach. So round 4's row belongs at 0. Writing it at `3 * capacity`
ran off a buffer of `capacity + 1024`, into whatever the driver had put next — which is
why the symptom read as misplaced records rather than as a bad walk. CUDA had had the
same conditional (`r4Off = octo ? 0 : 3*kCapacity`) since the record shipped.

**And a reference names a round-3 parent by SLOT, which is half-local.** A record set over
`CL_DEVICE_MAX_MEM_ALLOC_SIZE` is allocated as two bucket-halves, and each half numbers
its slots from zero — so the same slot number means two different records and the walk
picks the wrong one about half the time. The slot fits in 31 bits at every rung
(`nb/2 × cap` plus the pool), so the reference carries its half in the top bit and
`recover` selects the buffer from it. This is not a corner case: NVIDIA reports the cap as
a quarter of VRAM, so **every card this rung exists for splits**, and the unsplit path the
16 GB rig runs by default is the one that never happens in the field. It has its own gate
now (`gpu_solver_octo_split`).

3/3 goldens on all three octo rungs, split and unsplit, `drops == 0`, and the arena pool
carrying 92 records a solve as the positive control that the dense caps were live. The
record itself round-trips under `LDS_OCTO_CHECK=1`, which re-derives every staged element
from its eight leaves and compares keys: zero mismatches, at 4× the solve time — the
control that the check compiled in rather than reading zero because it never ran.

| OpenCL octo rung | footprint | ms/solve |
|---|---|---|
| quad (16,1) + dense caps + octo | 2.04 GiB | 61.5 |
| quad (15,2) + dense caps + octo | 1.95 GiB | 63.5 |
| quad (14,3) + dense caps + octo | **1.90 GiB** | 67.6 |

The footprints are CUDA's *exactly*, and not merely close: above the octo rungs OpenCL stores
five reference rows where CUDA replays three away, but on them there is only one row to
store either way. The times are not — 1.9× the top rung against CUDA's 1.8× — because
OpenCL's round 4 rebuilds from eight leaves without the
[w0 checkpoint](#the-w0-checkpoint-on-the-octo-record-the-mixes-go-and-the-record-does-not-grow)
that deletes CUDA's `apply_mix` calls. Against the ~215 ms sort path those cards used to
fall to, it is ~3× faster.

</details>


### The implicit-bits record on OpenCL, and the arena rung it unlocks

<details>
<summary>Details</summary>

A packed round-2 → round-3 record sits in the bucket its key names, so the top `bb` of
those 24 bits are the bucket ADDRESS and need not be stored; the consumer reinserts them
from its own bucket index. What is left — 400 − `bb` work bits — fits **6 u64 instead of
7**, and the word that frees carries the ninth word the side plane held. Set 0 goes from
9 u64 a slot to 8 and the plane loses its writer entirely. `rb_impb_ok` admits it at
`bb` 16 and 17, which on the `bb + sm = 17` line is (16,1) and (17,0).

CUDA has had this since 2026-08-13. Porting it needed the pack and its inverse, both keyed
on a build option rather than on the record width — the width does not change, only the
bit layout, so the two kernels cannot be told apart by `OUTSTR` the way the quad and octo
variants are.

| arm | ms/solve | footprint, (16,1) |
|---|---|---|
| packed record | 33.400 [33.4, 33.4] | 7.20 GiB |
| implicit bits | **32.283** [32.2, 32.3] | **6.84 GiB** |

Twelve interleaved arms, ABBA, ranges separated by 1.1 ms: **−1.117 ms, −3.34 %**, at
2.00 verified solutions/solve either way. The size and the sign both match CUDA's −3.2 %,
which is what a record-width change should do on either backend.

**Deleting the plane also makes a packed arena rung legal for the first time.** The
overflow pool lives *behind* the bucket records in the same buffer, so one slot index
addresses both regions; the side plane has no room behind it, so packed + dense caps used
to be refused. It threw at first *use*, after the ladder had already chosen it and
allocated — so a 6 GB OpenCL card, which the ladder hands exactly that pairing, crashed
rather than stepping down. The refusal now happens in the allocator, where a step-down is
still possible, and with the pack it does not fire at all at (16,1): that rung is
**5.77 GiB at 33.1 ms**, where the ladder's next stop was (15,2) at 6.62 GiB and 34.8.

(15,2) + dense caps stays unreachable on any backend that has no pack at `bb` 15, and is
stepped past.

</details>

### The w0 checkpoint reaches OpenCL: −0.745 ms (−2.29 %), the same lever at the same size
<details>
<summary>Details</summary>

The [pair-record checkpoint](#the-w0-checkpoint-pair-record-repriced-by-the-address-bits-076-ms-24-)
was CUDA-only for one reason, and that reason expired: the address bits pay for it, and
OpenCL had no implicit-bits pack to take them from. It got one the same week. The port is
therefore a revival rather than a new idea — the currency *slack bits in a packed record*
had its stated re-arm condition met, which is a record whose width changes for another
reason.

Round 1 now stores the child's post-mix work word 0 in the same 16 B, and round 2 derives
**only the linear lane** — 12 siphashes and no `apply_mix` at all, against 14 and three.
Words 1..6 pass through both parent mixes and the child mix untouched, so `rd_lane2`
produces exactly what `rd_elem2` did for them; word 0 is read instead of rebuilt, with the
block's own bucket put back into the key bits the address already carries.

**−0.745 ms, 32.600 → 31.855, −2.29 %.** Twelve interleaved arms a side (ABBA), ranges
**non-overlapping** — 31.726–31.898 against 32.541–32.626 — multiplier 1.99–2.00 in both,
`bench_rounds` 6/6 clean on both (drop counters zero, three survivors, all goldens matched).
CUDA measured **−0.76 ms, −2.4 %** for the identical change: same lever, same size, two
backends.

**Confirmed a second way, off eleven rungs that did not change.** The whole OpenCL ladder
was re-measured in one sitting afterwards. The checkpoint exists only where the pack does,
so **only the two packed (16,1) rungs run different kernels** and the other eleven are
byte-identical to the previous column — which turns them into an 11-point measurement of
this session's offset: **+1.26 %** (range +0.78 to +1.81). Correcting the two changed rungs
by it gives **−2.19 %** and **−2.46 %**, bracketing the interleaved figure.

| rung, that sitting | before | after |
|---|---|---|
| packed (16,1) | 32.1 | **31.8** |
| packed (16,1) + dense caps | 33.1 | **32.7** |

**Applied-assert.** `LDS_PW0_BREAK=1` poisons the stored checkpoint with an
**element-dependent** value at the emit — a constant would cancel through `combine`'s
pairwise XOR — and takes the goldens from 3/3 to **0/3** on every geometry. With it off,
3/3. So the path is live and the record round-trips.

One condition, `PW0_ON`, gates round 1's emit and round 2's staging and expand together, so
a writer and a reader cannot end up on different layouts; `-DLDS_PW0=0` turns off both
sides at once and is gated by `gpu_solver_nopw0`. It carries the perfect table because
bits 23..8 of a checkpointed word 0 are work bits, and OpenCL's chain hash already reads
the *rebuilt* word 0 rather than the record's key field, so nothing else had to move.

**What was left on this backend** — the quad and octo records — followed;
see [the next entry](#the-w0-checkpoint-reaches-opencls-quad-and-octo-records).

</details>

### The w0 checkpoint reaches OpenCL's quad and octo records
<details>
<summary>Details</summary>

The pair record was the boundary that moved the shipping rung. The other two are reach
rungs — a card with room for the packed record never sees them — so they were left for
last, and they turn out to be **the two biggest instances of the lever on this backend**.

Both ports carry the CUDA layout over unchanged, and neither record grows:

- **The quad record (r2 → r3), 24 B.** Word 0's own low 24 bits *are* the key the plain
  form stored there, so it goes in whole and needs no repacking; the four leaves and the
  `gi` move down into the two words the packed record already packs them into — 190 bits
  of 192. Round 3 then runs `rd_lane3`: **24 siphashes and no `apply_mix` at all**, against
  28 and seven. Unlike the pair record this needs neither the implicit-bits pack nor the
  perfect table, because nothing is dropped.
- **The octo record (r3 → r4), 32 B.** Here word 0's 40 bits are bought, out of a `gi`
  nothing indexes on an octo build and 14 of the 24 key bits the bucket address already
  carries. Round 4 runs `rd_lane4`: **48 siphashes and none of the fifteen mixes**, against
  56 and all of them. Only 10 key bits survive, so this one *does* carry the perfect table
  — a full-key compare in the walk would reject true partners — and the 10 are enough
  because `bb + sm = 17` puts the highest bit the walk consults at `sm + 6 ≤ 9` on all
  three octo rungs.

**Measured, interleaved (ABBA), on the (16,1) rung of each:**

| boundary | off | on | delta | ranges |
|---|---|---|---|---|
| quad, r2 → r3 | 38.933 | **36.500** | **−2.433 ms, −6.25 %** | 38.8–39.0 against 36.2–36.6 |
| octo, r3 → r4 | 61.733 | **56.450** | **−5.283 ms, −8.56 %** | 61.6–61.8 against 56.3–56.5 |

Six arms a side each, ranges **non-overlapping** in both, multiplier 1.99–2.03 throughout.
CUDA measured −1.84 and −4.39 ms for the same two changes; OpenCL gets **more** from both,
which is what a deeper re-derivation on a slower backend should do.

**Confirmed a second way, off the rungs that did not change.** The ladder was re-taken in
one sitting afterwards. Nine of its fourteen rungs changed kernels — six quad rows and
three octo — and the four byte-identical ones size the sitting's offset at **−0.73 %**
(range −0.52 to −0.94). Correcting by it:

| rung | before | after | corrected delta |
|---|---|---|---|
| quad (16,1) | 39.4 | **36.5** | −6.63 % |
| quad (16,1) + dense caps | 40.2 | **37.7** | −5.49 % |
| quad (15,2) | 40.6 | **37.8** | −6.17 % |
| quad (15,2) + dense caps | 42.1 | **39.5** | −5.44 % |
| quad (14,3) | 44.0 | **41.4** | −5.18 % |
| quad (14,3) + dense caps | 46.2 | **43.7** | −4.68 % |
| quad (16,1) + dense caps + octo | 62.4 | **56.5** | −8.72 % |
| quad (15,2) + dense caps + octo | 64.2 | **58.7** | −7.84 % |
| quad (14,3) + dense caps + octo | 68.5 | **63.0** | −7.30 % |

The two rungs that carry both measurements agree: −6.25 % interleaved against −6.63 %
corrected on quad (16,1), −8.56 % against −8.72 % on the octo one.

**Applied-assert, both boundaries.** `LDS_QW0_BREAK=1` and `LDS_OW0_BREAK=1` each poison
the stored checkpoint with an **element-dependent** value at the emit — a constant cancels
through `combine`'s pairwise XOR — and each takes the goldens from 3/3 to **0/3**. Off,
3/3. The poison sits in word 0's bits 40..47, which `combine` carries into the child's key
while leaving the key the *writer's* own bucketing used intact, so what fails is the
checkpoint and not the addressing.

**One `-D` per boundary, and it has to reach three places for the octo one.** `recover`
walks two levels on an octo build and reads round 3's record directly, so its program is
now built with `MXBM_CL_OPTS` as well — a walk decoding one layout while round 3 writes the
other would return noise. `-DLDS_QW0=0` and `-DLDS_OW0=0` move every side together and are
gated by `gpu_solver_noqw0` and `gpu_solver_noow0`; `gpu_solver_quad` was added with them,
because no gate had ever reached the 24 B quad record at all.

**A consequence for the ladder's order.** OpenCL's quad (15,2) is now **faster** than
packed (14,3) — 37.8 against 38.5, where it was 40.6 against 38.7 — so the one row that
sits out of time order keeps its place on its *allocation* ceiling alone (2.76 GiB against
2.90), where speed is untouched. Both backends now call the row below it faster.

</details>

### The instruction census: 69 % of every instruction is SipHash, and that bounds the rest
<details>
<summary>Details</summary>

*Instruction count on the ALU pipe* is one of the two currencies still open, and only
**entry** had ever been counted to its floor — 153 SASS instructions per SipHash call
against a hand-derived 155. Rounds 1 and 2 are 44 % of the solve, are ALU-pipe-bound, and
had never been decomposed at all.

**The instrument.** A `-lineinfo` build (non-perturbing: 29.11 ms, KAT 3/3, drops 0, 2.00
verified/solve), `nvdisasm -g -c` on the cubin for the SASS → source-line map, joined
**positionally** to `ncu --section SourceCounters` per-instruction executed counts. The
join is verified rather than assumed: opcode agreement is **100 % on all five kernels**
(1080 / 1664 / 2392 / 760 / 624 instructions), and the script refuses to report below 99 %.

Two positive controls, at opposite ends. **The pure case:** entry is seven SipHash calls
and nothing else, and the census's own counts run through a pipe model — an sm_89
sub-partition has 16 INT32 lanes, so a warp ALU instruction holds the pipe two cycles, over
264 sub-partitions at 2758 MHz — predict **2.41 ms against a measured 2.48**, 97.2 %, and
independently reproduce its 98.8 % ALU-pipe utilisation. **The null case:** `apply_mix` is
**23.3 % of round 3's instructions**, and `MXBM_ABL_MIX=3` deletes every one of them for
**+0.024 ms**. Instruction share buys nothing on a DRAM-bound round, measured from both
ends at once.

One attribution hazard, handled: `rotl64` is one source line and serves both SipHash (36
calls) and `apply_mix` (9), so it is split out. Rounds 3 and 4 execute **zero**
SipHash-body instructions, so their whole share of it is the mix.

**The result.** Share of executed thread instructions:

| region | entry | r1 | r2 | r3 | r4 |
|---|---|---|---|---|---|
| **SipHash** | **94.7 %** | **71.4 %** | **78.0 %** | 0.0 % | 0.0 % |
| `apply_mix` | 1.9 | 3.3 | 1.4 | 23.3 | 11.7 |
| `combine` | 0.0 | 2.8 | 1.8 | 8.0 | 8.0 |
| walk + emit | 0.0 | 2.8 | 3.7 | 10.3 | 16.6 |
| stage | 0.0 | 3.1 | 2.2 | 14.1 | 15.0 |
| singleton census + pool | 0.0 | 4.3 | 2.8 | 8.2 | 12.6 |
| setup + address arithmetic | 0.0 | 3.9 | 2.5 | 10.7 | 9.2 |
| expand + chain build | 0.0 | 2.8 | 1.7 | 7.7 | 11.2 |
| atomics | 0.2 | 2.7 | 1.7 | 7.3 | 6.2 |
| record pack / unpack | 0.0 | 1.1 | 2.4 | 2.4 | 2.6 |
| spill / mlist | 0.0 | 1.2 | 0.8 | 3.7 | 5.5 |
| **total (G thread-inst)** | **34.66** | **40.98** | **62.85** | **13.79** | **9.19** |

**161.5 G thread instructions per solve, and 111.1 G — 68.8 % — are SipHash**; inside the
three ALU-pipe-bound rounds, **80.2 %**. In the currency that binds, SipHash's ALU-pipe
floor is **8.84 ms of the 28.37 ms** kernel sum: 2.41 in entry, 2.48 in r1, 3.95 in r2,
zero in r3 and r4.

**What it closes.** The ALU-bound half now accounts completely:

| | ms | of the 15.00 ms |
|---|---|---|
| SipHash, at the ALU-pipe floor | 8.84 | 58.9 % |
| every other ALU-pipe instruction | 1.98 | 13.2 % |
| non-ALU issue + the warp-supply loss | 4.18 | 27.9 % |

So **the whole instruction-count family in entry, r1 and r2 is bounded by 1.98 ms** — and
that is the bound if every non-hash ALU instruction vanished. No named region reaches
0.3 ms of it; the largest are r2's walk+emit (≈0.28 ms of pipe floor), r1's singleton
census (≈0.21) and the two rounds' address arithmetic (≈0.38 together). The mix's `⋘24`
cancelling the combine's `>>24` would remove one rotate of nine from a region worth 1.4 %
of its round — **under 0.05 ms**, a twentieth of the instrument floor — so that identity is
not worth settling whether or not it holds. And `IMAD` is **16.7 %** of round 2's stream and
sits on the idle **FMA** pipe, so it is not a tax.

That retires the currency. It is alive only in that SipHash rides on it, and SipHash is
closed on **count** — `h=2` is a structural optimum and all four checkpoint boundaries are
priced — and on **cost per call**, at 153 SASS against 155 with three cheapening routes
measured and both count-reducing encodings audited to zero.

**Where the solve is, whole:** SipHash at its pipe floor **8.84 ms (31 %)**, compulsory
DRAM in r3/r4/terminal **13.37 (47 %)**, warp supply in r1/r2 **4.18 (15 %)**, every other
ALU instruction **1.98 (7 %)**.

**Falsifier.** Anything that moves SipHash's *count* — a new switching height, or silicon
where a byte through a bucket scatter costs more than ~2.2× a call, which would flip the
store-versus-derive rate — or a round whose non-hash ALU share passes ~25 % of its own
time, which needs the hash to leave rather than the overhead to grow. Scoped to sm_89, the
shipping h=2 record set, `bb + sm = 17`, and stock.

</details>

### Round 2 cannot reach the warp ceiling, because shared bytes per thread do not scale
<details>
<summary>Details</summary>

Occupancy has been closed on round 2 three times — from shared memory, from the register
file, and by building the fifth block and measuring it at **+0.10 ms**. What the closures
did not have is the reason the geometry cannot be used to escape them, which matters
because *"a smaller block"* is the standing re-arm condition on the warp-supply currency.

sm_89 allows **1536 threads and 102,400 B of shared memory** per SM, so the warp ceiling is
reachable only below **66.7 B of shared per thread**. Round 2 runs at **23,624 B / 256
threads = 92.3 B/thread**, 38 % over — which is exactly why it holds 32 of the 48 warps the
card allows.

**Scaling the geometry does not move that ratio.** A finer geometry halves the group and the
block together, and shared-bytes-per-thread is invariant under it. That is the mechanism
behind every geometry probe on record losing by a lot rather than a little: `MXBM_BB=16
MXBM_SM=2` measured **+24.0 %**, and its stated cause — a mean group of 132 against `kWG`
256, half the lanes idle in every all-lanes loop — is the same invariant seen from the lane
side.

So reaching 48 warps needs a **28 % cut in shared per thread**, of which `lwork` is 76 %
(320 × 7 × 8 B = 17,920 of 23,624) — i.e. **staging 4.4 of the element's 7 work words**,
when the round needs all 7 to combine. BeamHash III's `[7,7,6,5,1]` schedule forbids it.

**Two currencies collapse into one.** *Shared bytes per staged element* and *warp supply*
are the same currency, related by 92.3 B/thread against a 66.7 B requirement, and the
exchange rate is already measured: **+0.10 ms per block**. It dies once; it cannot die twice.

</details>

### Abandoning a dud solve early is worth 0.10 ms, because the yield is not knowable sooner
<details>
<summary>Details</summary>

The [correlation theorem](#the-solutions-per-solve-multiplier-is-the-algorithms-not-the-solvers)
left three channels of yield-for-time open. Ancestry-zero shipped and its whole 12.8 %
budget is harvested; correlated-class is worth 4.1× and still lands at ~1.9:1 against a 1:1
break-even. The third — abandoning a whole solve once it is clear it will produce nothing —
had never been analysed, and it looks like the largest of the three: a solve yields Poisson(2)
solutions, so **13.5 % of them yield none**, and dropping those after round 2 would save
13.4 ms of 28.4 on one solve in seven.

**It cannot be known after round 2, or after any round but the last.** The population is
pinned at 2^25 in every round with all four drop counters zero, so no earlier round's
observable state varies with the solve's eventual yield. All of the variance sits in the
terminal round's 48-bit collision count — and computing that count is what the terminal
round *is*.

So the ceiling is the terminal round's own **0.77 ms** on the 13.5 % of solves that yield
nothing, and only if the knowledge arrived free, which it cannot: **≤ 0.10 ms**, a third of
the instrument floor. The Poisson model checks out on the shipping build — 200 consecutive
solves read 2.04 verified/solve with a maximum of 7 survivors, against P(≥7) ≈ 0.45 %, about
one solve in 200 (28.90 ms/solve, KAT 3/3, drops 0, card idle).

**Axis 2 is now closed on all three channels**, and the yield currency is dead outright
rather than dead only as a uniform trade. It comes back only under a round schedule where an
earlier population varies with eventual yield — which requires nonzero drops, i.e. a design
that is already losing solutions.

</details>

---

### Turing's shared-memory budget, and the per-card staging cap

**The 64 KB question, read off the cubin.** The release fatbin's sm_75 image was never
measured on a card until the GTX 1660 Ti reports (docs/performance.md → *Backend reach*),
which land ~2× above what core and DRAM scaling from the reference card predict. The
first thing that scaling ignores is what `cuobjdump --dump-resource-usage` on the sm_75
cubin says: every round kernel's static shared memory is 22.3–26.9 KB, and Turing's
64 KB per SM (Ada: 100 KB) holds two such blocks, not three — a third needs ≤ 21,845 B.
ptxas knows it, and that is where the "112 registers at the heaviest" came from: with
occupancy capped by shared memory at two blocks, it spends the register file to match
(111–128 registers per thread on rounds 2–4, against 64/80 for the same kernels on
sm_89, where the same heuristic targets three or four blocks).

**What two blocks per SM cost, measured on the reference card.** `MXBM_SMEM_PAD=12288`
adds 12 KB of unused dynamic shared memory to every round launch, which drops this card
to two resident blocks per SM and changes nothing else. Every round stretches by about
the same factor:

| | pad 0 (3–4 blocks) | pad 12288 (2 blocks) | |
|---|---|---|---|
| round 1 | 4.80 | 5.54 | +15 % |
| round 2 | 7.94 | 9.44 | +19 % |
| round 3 | 8.51 | 9.52 | +12 % |
| round 4 | 6.40 | 7.67 | +20 % |
| terminal | 0.79 | 0.91 | +15 % |
| **solve** | **28.7** | **33.3** | **+16 %** |

So the third block is a prize on a 64 KB card, and on the reference card it is the
regime the ledger's "occupancy is a price, not a prize" closure never covered — that
closure is scoped to the fourth and fifth blocks at 100 KB.

**The staging cap is the knob, and 280 is where rounds 1, 2 and 4 cross the line.** The
per-block shared budget is `RoundShared`, dominated by `lwork[INW × FCAP]`; each unit of
FCAP is 56 B on the seven-word rounds. At 288 the plain round-2 kernel fits (21,320 B) but
its dense-cap variants do not (21,888–22,016); at 280 every round-1, round-2 and round-4
instantiation lands at 19.0–21.4 KB and ptxas re-fits them at 80 registers — three
blocks per SM by both resources. Round 3 does not fit at 280: its four-word leaf staging
keeps it at 23.0–23.7 KB and it would need 256, which is the group mean and drops
elements. Shipped as `kFCap64K = 280` for rounds 2 and 4, selected at runtime by the
device's `sharedMemPerMultiprocessor` (≤ 64 KB), with `MXBM_FCAP_SMALL=0|1` to force
either arm anywhere; round 1 already ran at 288 and fits as it is.

**Round 3's third block takes the narrow word-6 plane.** Round 3 stages 80 B per
element and the one field with slack is work word 6, which carries 16 significant bits
in a u64 — `MXBM_NARROW6`, [null on the reference card](#round-3-does-not-want-a-fourth-block--occupancy-pays-only-where-a-round-is-latency-bound)
because round 3 wanted no fourth block there. As a template flag rather than a build
switch it can be chosen per card: with word 6 in a `uint16_t` plane the element is 74 B,
and at a cap of 272 every round-3 instantiation lands at 20.7–21.4 KB on sm_75, three
blocks by shared memory and by registers (54–57 on the packed record, 80 on the quad).
Priced on the reference card by emulation — `MXBM_SMEM_PAD=8192` holds rounds 1, 2 and 4
at three blocks in both arms and puts round 3 at two blocks at 320 and three at 272 with
the plane, ABBA at stock:

| | pad 8192, cap 320 (2 blocks) | pad 8192, 272 + narrow plane (3 blocks) | |
|---|---|---|---|
| round 3 | 9.55 / 9.58 | 8.76 / 8.77 | **−8.3 %** |
| solve | 31.1 / 31.2 | 30.7 / 30.7 | −1.4 %, with rounds 2 and 4 paying their 280 tax (+0.1 ms each) in the same arm |

The block is worth about 8 % of round 3 where the round is shared-capped at two — less
than the 12 % the pad table charged for losing it, the difference being the plane's
instructions and the smaller cap's ragged tail. On the reference card at its own budget
the same arm is +0.5 % (28.8 → 29.0 ms; round 3 +0.05 ms), so 272 and the plane are
per-card choices like 280. The octo rung's round 3 rebuild had written its seven words
straight into `lwork`, which the plane turns into a corruption of the next slot; it now
goes through the same store as every other mode, and the goldens are what caught it.

**Why it is per-card and not universal.** On the reference card 280 changes no
occupancy (the resource contract reports drift, blocks/SM held), and measures **+0.5 %**
in an ABBA at stock (28.75 → 28.9 ms; round 4 +0.08 ms, rounds 2 and 3 unchanged) —
a staging loop that no longer ends on a whole warp. A card that gains nothing from the
smaller cap should not pay that, so the choice keys on the budget itself rather than on
the architecture name.

**Gates.** 3/3 goldens × 15 geometries with `MXBM_FCAP_SMALL=1` and without; 0 drops on
every counter over 4,340 solves in the 280 arm and 4,024 in the arm with round 3's
plane; `test_cuda_resources` sets the 280 and 272 instantiations aside (the reference
card never launches them) and holds every existing row. **What is not measured is the
card**: the expected figure on a GTX 1660 Ti is bounded by the two tables above — up to
about −14 % on rounds 1, 2 and 4 and ~−8 % on round 3 — and the next `--report` from one
carries the per-stage table that says how much of it arrived.

**The per-stage table** (`MXBM_ROUND_STATS=1` on `--benchmark`; always in `--report`)
records a device timestamp at every stage boundary and reports medians. It costs nothing
this rig can resolve (28.7 ms in both arms, 30 s each), and its first reading reproduced
the census: r2 7.93 / r3 8.51 ms against the profiler's 7.94 / 8.51. Its timing events
are also dispatch fences at each boundary, which a plain solve does not have
([the entry pass beside rounds 3 and 4](#the-entry-pass-beside-rounds-3-and-4-on-a-stream-the-block-scheduler-prefers)).

### The terminal round's time was its block count, not its bytes

**What it did.** The terminal round reads round 4's 8 B records — 268 MB at stock — and
took 0.80 ms, 53 % of DRAM peak, which the ledger had filed under "a launch too short to
amortize". The other reading of that number is the right one: the kernel's blocks are
tiny (one bucket half, ~2 KB of records, two barriers, a 128-word table clear), and each
thread issued its two or three record loads one at a time behind the sub-mask `continue`
and the shared `atomicAdd`, so a block's lifetime was two or three DRAM latencies in
series. 131,072 such blocks over 66 SMs × 6 resident is ~330 waves, and ~2.5 µs a wave
is 0.8 ms — the time was the block count times a latency chain, and the bytes never
reached the critical path.

**Two changes, priced apart.** First, every record load of a block's chunk is issued
before the first is filtered (`kPre = 4` loads in flight per thread): 0.80 → 0.73 ms,
+12 registers, blocks per SM held at six. Second, one block per *bucket* instead of one
per sub-mask half on the (16,1) rungs: the staging grows to `kTermCap` 832 with a
256-entry perfect table (8 varying key bits at `bb + sm = 16`), the grid halves, and
each block's loads are all in flight at once. ABBA at stock, three arms a side:

| | old | loads hoisted | + one block per bucket |
|---|---|---|---|
| terminal | 0.80 / 0.80 / 0.80 | 0.74 / 0.74 / 0.74 | **0.52 / 0.52** |
| solve | 28.8 / 28.8 / 28.8 | 28.8 / 28.8 / 28.8 | **28.6 / 28.6** |
| solves in 30 s | 1044 / 1042 / 1041 | 1043 / 1043 | 1051 / 1051 |

**−0.28 ms, −35 % of the round, −0.7 % of the solve**, against a bandwidth floor of
~0.42 ms. Shared memory goes 6.7 → 14.3 KB (octo arm 21.1 KB, six → four blocks per
SM) and the octo (16,1) rung's terminal still reads 1.10 → 1.03 ms, so the block count
outweighs the residency there too. The launcher takes the one-block form wherever
`bb ≥ 16` and the bucket cap plus the pool's `kAMax` fit `kTermCap`; the quad (15,2) and
lower rungs keep the sub-mask split with the same table. Gates: 3/3 goldens × 15
geometries, all four drop counters zero on the shipping and the octo rungs.

**Why the "GPU idle time" closure did not cover it.** That census counted the gaps
*between* kernels (0.19 ms) and found none inside them; this is time inside a kernel
whose blocks are too short-lived to overlap their own loads. The same shape is worth
checking on any launch under ~1 ms: recovery's replay kernels were the next candidates
([below](#the-replays-were-warp-serialised-on-their-hits-not-latency-bound)), and the
fused rounds are not — their staging loops already keep eight iterations of loads in
flight.

### The replays were warp-serialised on their hits, not latency-bound

<details>
<summary>Details</summary>

Recovery's three kernels summed to 0.176 ms in the dispatch census, and the terminal
round had just shown that a short kernel's time is its block's serial chain. The
per-kernel split said which chain:

| kernel | grid | before | after |
|---|---|---|---|
| `replay_r4` | 2 blocks per survivor | 60 µs | 10 µs |
| `replay_r3` | 4 per survivor | 115 µs | 10.5 µs |
| `recover_from_l2` | 1 | 1.8 µs | 1.8 µs |

Neither replay was waiting on memory. Each staged its bucket's keys in shared and ran the
pair search as a per-lane loop — lane *p* scanning *q* > *p* and, on a key match,
rebuilding the child in place: `combine`, `apply_mix`, the word-0 identity. That rebuild
is hundreds of instructions, and inside a per-lane loop every hit runs it with the rest
of the warp masked off. `replay_r3` keeps only 8 key bits inside a bucket at the shipping
rung (24 less the 16 the address carries), so a 512-element bucket has ~500 same-key
candidates, ~64 per warp, each serialised: ncu read 9.4 of 32 lanes active per issued
instruction and 1.64 M warp instructions for eight blocks. `replay_r4`'s 24-bit key has
almost no hits, and its 60 µs was the O(n²) scan itself at 46 % lane utilisation.

Both now search the way the rounds do. The staged keys go into a 1024-entry chain table;
each lane walks its own chain for *q* < *p* and appends the (*p*, *q*) pairs to a shared
list; the block then processes the list in lockstep, one candidate per lane, resuming
the walk if the list fills. Instructions drop to what the rebuilds cost with full warps:
recovery's stage median reads **0.03 ms from 0.19**, and the solve **+0.4 %** on solves
in eight 30 s arms, both orderings, ranges non-overlapping (1059–1060 against
1055–1056). KAT 3/3 × 15 — the leaves are exactly what the goldens gate — and every drop
counter zero. Shared per block goes 16.6 → 44.3 KB, which halves the residency of
kernels that launch four to eight blocks in total.

**What the terminal round's lesson looked like here.** Same symptom, a sub-millisecond
kernel far from any roofline; a different chain. The terminal's was DRAM latency paid
one pass at a time; the replays' was ALU work paid one lane at a time. Both are found by
asking what one block does serially, and neither is visible in a bytes-or-instructions
budget for the kernel as a whole.

**Where the lesson stops: round 3's staging.** The same per-instruction stall sampling on
round 3 looks like the terminal round before the fix — 60 % long scoreboard, split between
the word-0 filter load (18 %), the four record loads behind it (21 %), the bucket count
load (3 %) and the allocation atomic's return in the walk (8 %) — and the staging loop is
not unrolled, so a warp's two iterations run their two-load chains back to back. Two
forms of the terminal round's fix were built and gated (KAT 3/3 × 15): hoisting every
iteration's word-0 load and prefetching the survivors' second sector into L2 (**+0.17 ms
on round 3**, +48 % L2 read sectors), and hoisting the word-0 loads into registers with
no reload and no prefetch (**0.00 ms on round 3**, +0.04 on round 4; eight 30 s arms,
identical to the digit). The stall is real and the fix did nothing because round 3 moves
4.30 GB in 8.47 ms — **508 GB/s, 77 % of the bus** — so its long-scoreboard cycles are
queueing on a saturated memory system, not exposed round trips, and re-ordering a warp's
loads changes when it waits, not how long. The short-kernel lever applies where the bus
is idle while blocks wait; it does not transfer to a round near its byte floor. A
side-note from the same session: an ablation that frees the store from the allocation
atomic by hashing the slot is not a valid price for the dependence — it also scatters
consecutive slots and the round slows by half — so the atomic's 8 % has no cheap probe
and would need the pipelined emit built to be priced.

</details>

### The round-3 record is 56 B: the dead word goes, and the four 16 B transactions stay

<details>
<summary>Details</summary>

The r3 → r4 record was 8 u64 for 401 bits of content: five work words that round 4
consumes (312 bits), a sixth that it provably never reads and that carried only
`replay_r3`'s 17-bit parent hint, a meta word (lead, gi) and the leftContrib. The 48 B
form is dead by bits; 56 B holds it with 47 to spare. What kept 56 B off the table was
alignment — 56 B slots alternate between 16 B-aligned and 8 B-off, so a record cannot be
four `ST.128` — and the stock closure that bytes on this path buy nothing.

**Both were priced before building.** A timing-only probe dropped the word on both sides:

| form | r3 at 285 W | r4 | r3 at 140 W | r4 |
|---|---|---|---|---|
| shipping 64 B | 8.53 | 6.43 | 13.30 | 6.01 |
| 56 B as seven `ST.64` / `LD.64` | **10.47** | 6.14 | **20.69** | 5.42 |
| 56 B as one 8 B + three 16 B, placed by slot parity | **8.23** | **6.19** | **11.93** | **5.88** |

Seven scalar stores are seven L2 transactions per element and the round loses a quarter
at stock and half under the cap — the ledger's "memory instructions and sectors" currency
exactly. Choosing the 8 B store by the slot's parity keeps four transactions per element
on both sides; the branch is divergent inside a warp, so eight store instructions issue
with half the lanes off, which round 3 does not notice.

**Shipped form.** Words 0–4, then a meta word `lead | gi << 25 | hint[8..16] << 51`, then
the leftContrib; the hint's low byte rides in word 4's top byte, which is above bit 311
and therefore dropped by `combine`'s Lout mask in round 4 and by the replay's compare
(masked). `replay_r4`, `replay_r3` and `recover_from_l3` read the new offsets; the
set-1 allocation stays sized at 8 words, so every footprint figure stands. Gates: 3/3
goldens × 15 geometries on both staging arms, drops 0, 1.98 verified solutions/solve.
Eight 30 s arms, both orderings, stock: **1054 → 1073 solves, +1.7 %**, round 3
8.54 → 8.23 and round 4 6.43 → 6.20, ranges non-overlapping. Registers move by ±4
and no round changes its blocks per SM.

**Under the cap the rounds win and the solve does not, and that is the finding.** At
140 W round 3 drops 1.37 ms and round 4 0.13 — and entry, round 1 and round 2 give back
0.7 of it: 5.44 → 5.57, 10.43 → 10.83, 17.45 → 17.56. The solve moves 53.6 → 53.5 ms.
A capped card is *energy*-bound: the controller spends a fixed 140 W, a memory-bound
round that finishes sooner leaves the core-bound rounds a larger share of the time and
therefore a lower clock, and what survives is the energy the bytes cost — ~0.5 GB of
DRAM traffic at a few pJ per bit is ~0.5 % of a 7.5 J solve, which is what the solve
gained. Two corrections follow. The ×6.8 "cap multiplier" on round 3's write sectors was
a *round* marginal by in-place replay and overstates the solve by the clock hand-back;
a cap-band lever must be priced on the solve, or in joules. And the stock closure on
this path ("bytes buy nothing") was scoped to a narrowing that kept the 64 B stride —
the sectors it saved were the ones L2 was already merging. Narrowing the *stride* at
equal transaction count is a different lever, and it is the one that paid at stock.

</details>

### The gi allocator goes where nothing indexes a gi: +0.9 %

<details>
<summary>Details</summary>

Every emitted child on rounds 1–3 took a global `gi` from one counter — warp-aggregated,
so one `ATOMG` per warp-batch plus a shuffle to hand each lane its number — and stored it
in the record. Its readers were the reference rows, gone since the replays, and the
walk's left/right tiebreak on equal leads. The ledger had priced retiring it as "only the
`gi_alloc` atomic" and left it. On the replayed rungs the tiebreak now orders by the
parent record's slot, which the staging loop already has and which `replay_r3` and
`replay_r4` compute from the same address, so the rule is identical on both sides and
deterministic run to run. The allocator is not issued on those rungs; rungs that still
write rows keep it, because there the gi is the row index.

KAT 3/3 × 15 on both staging arms, drops 0, 2.04 verified solutions/solve over the
check. Eight 30 s arms at stock, both orderings: **1073 → 1083 solves, +0.9 %** — round 1
4.88 → 4.73 ms, round 2 8.06 → 8.00, rounds 3 and 4 within 0.02. Round 1 pays most: it is
issue-bound and the aggregated atomic's ballot, popcount, shuffle and the wait for the
counter's return were issue slots. Round 3's 5.5 % of stall samples on that shuffle did
not turn into time, which is the bus-queueing story again. 40 fewer `ATOMG` in the
module.

**The same day's null on the boundary above it.** Entry computes all seven SipHashes and
the round-1 mix to get a leaf's key, and round 1 recomputes all of them. A stored word-0
checkpoint needs 73 bits and the entry record has 64, so the 16 B form was built: entry
writes `idx | key` and the mixed word 0 as one `ST.128`, round 1 stages word 0 and
derives words 1..6 without the mix. KAT 3/3 × 15 both arms. Round 1 **4.73 → 4.36 ms**,
the predicted seventh — and round 4, which hosts the next solve's entry, **6.19 → 6.72**:
the extra 0.27 GB of scattered writes lands in the one round that is already moving
2.2 GB. Net **−0.6 % on solves** (1083 → 1077), reverted. The checkpoint pays where it
fits the record's slack and loses where it grows the record, as the ledger's exchange rate
says; the entry record has no slack for it.

</details>

### The entry pass beside rounds 3 and 4, on a stream the block scheduler prefers: +2.0 %

<details>
<summary>Details</summary>

The overlap family was closed on two mechanisms. A second stream never overlaps, because
a round enqueues 131k blocks against ~200 resident and the second kernel's blocks are
dispatched only in the last wave's tail; and co-blocks inside the round's own launch
displace a round block one for one, which round 4 could spare and round 3 could not
(hosting there measured worse than sequential). Neither mechanism was about the SM's
resources: during round 3 every SM has 24 warp slots, ~20k registers and the whole L1
idle, because shared memory caps the round at three blocks. **A stream created with the
highest priority puts the entry's blocks into that room straight away** — the CUDA block
scheduler dispatches a higher-priority kernel's pending blocks ahead of the lower one's,
into any SM with space — and, launched after round 2, they run beside rounds 3 and 4 and
the terminal round without taking a block slot from any of them.

What was measured, all at stock, per-round medians from 10 s runs (the round-3 baseline is
8.18 ms; entry standalone 2.57; the shipping co-blocks form 27.4–27.6 ms/solve, sequential
27.7):

| form of the entry kernel beside round 3 | entry kernel | round 3 | ms/solve |
|---|---|---|---|
| the standalone kernel's grid (131k blocks of 256) | 2.61 | 10.79 (+2.6) | 27.8 |
| one 256-thread block per SM, grid-strided | 2.86 | 10.70 (+2.5) | 27.7 |
| one 128-thread block per SM, hashes made one dependent chain | 4.17 | 9.66 (+1.5) | 26.8 |
| the same, 1 µs sleep after each element | 8.1 | 9.44 (+1.3) | **26.5–26.6** |
| the same at 64 threads | 8.4 | 10.35 (+2.2) | 27.4 |
| launched after round 1 instead (beside round 2) | 8–19 | r2 +1.3–2.6 | 27.7–28.4 |

**A full grid at high priority is not overlap, it is a turn**: every slot a round block
vacates goes to a pending entry block, and the round stops until the entry is through —
its 2.6 ms land on round 3 to the digit. So did one 256-thread block per SM: eight warps
of seven independent SipHash chains saturate the SM's ALU pipe on their own, and the
issue scheduler is greedy — a warp that is always ready keeps issuing. What decides the
round's loss is **how densely the co-resident warps issue**, measured with the compute
alone (stores and atomics removed, the real pass launched separately so the pipeline
stays right):

| compute-only co-runner, 4 warps/SM | co-run | round 3 |
|---|---|---|
| dependent chain, no sleep | 2.98 ms | +2.00 |
| + 0.4 µs sleep per element | 4.11 | +1.27 |
| + 1.2 µs | 8.15 | +0.65 |
| + 4 µs | 16.3 | +0.00 (r4 +0.28, terminal +0.11) |

The same work costs round 3 three times less spread over 8 ms than packed into 3, and
nothing at all spread over 16 — the loss goes as the square of the co-runner's duty, the
shape of queueing at a shared pipe rather than of a conserved budget. Round 3's own
issue is 19 % busy and its ALU pipe 18 %; a co-runner at low duty fits in the gaps, one at
high duty makes round 3's latency-bound chains wait. So the shipped form throttles
itself: the seven hashes of an element are threaded into one dependent chain through a
runtime-zero mask (so a warp issues one instruction per pipe latency instead of seven
deep), four warps per SM, one microsecond of `__nanosleep` per element. Longer sleeps
lower round 3's loss further but spill the pass past round 4 into the next solve's
round 1, which then waits.

**What the memory side costs, and what it does not.** With the compute-only co-runner at
the shipped duty costing 0.65, the real pass costs 1.3: the other half is its 33.5 M
atomics and 8 B stores. Two things were tried against it. Store cache hints —
`evict_last` on every store, `evict_last` while a sector or a 128 B line fills and
`evict_first` on the record that completes it — move it by ≤ 0.1 ms and one form ships
(line-granular release), while `evict_first` on every store costs 4.7 ms: the write
frontier flushed sector by sector under round 3's stream is the disaster case, and the
default policy already avoids most of it. Giving the entry kernel the rounds' shared
carveout preference changes nothing. The remaining ~0.6 ms is the entry's scatter
arriving at an L2 crossbar that round 3 has at 62 %.

**Under a cap it loses, like every hosting form**: at 140 W round 3 takes the whole pass
(10.6 → 15.45 ms) and the solve reads 51.7 against 50.5 with the entry standalone; the
solve is energy-bound there and the hosted joules come back as clock. The crossover is
where the co-blocks' was: 200 W −1.3 %, 220 W +2.4 %, 240 W +3.5 %, 255 W +3.9 % against
standalone, so the existing 220 W gate stands and the form is stock-band only.

**The event between round 2 and the launch must keep its timestamp.** Recorded with
`cudaEventDisableTiming`, the same plumbing runs a two-mode solve: half the solves at 26.5
ms and half at 30.0, mean 28.6, verified count untouched. Round 3's grid is already queued
behind round 2 when round 2 ends, and in about half the solves it wins the SMs before the
priority stream's 66 blocks are eligible; the pass then runs where a plain second stream
would have put it, after the round, and the next solve waits ~3.5 ms for it. A
timing-enabled event at that boundary is a fence — the front end takes the timestamp
before it submits round 3 — and the priority blocks dispatch first every time (224 solves
in 6 s with 2 over 29 ms, against 210 with ~95). A timing event at any other boundary, a
timing-less event at this one, or a host-side pause change nothing. Stage timing
(`MXBM_ROUND_STATS=1`) records timing events at every boundary, which is why the tail is
invisible under the instrument that would have shown it.

**Shipped**: `entry_beside`, one 128-thread block per SM on a `cudaStreamNonBlocking`
stream at the device's greatest priority, launched after round 2 behind a timing-enabled
event, waited on at the start of the next solve; `MXBM_SPEC_COBLOCKS=1` keeps the co-blocks form for
A/Bs. Verified solutions per solve hold at 2.02 with drops 0 throughout; the KAT gate
does not exercise speculation (its inputs are distinct jobs) and the verify count is the
positive control, as for the co-blocks. Paired ABBA at stock, both orderings, 16 × 60 s:
**27.59 → 27.05 ms, −2.02 % on solves (2174.5 → 2218.4 per 60 s, sd 0.02 / 0.13 %, t = −42), against a 0.04 % null floor**; the priority-stream arm ran the card 18 MHz lower for the same limit, so the paired figure is a floor on the kernel-time gain.

*Scope: sm_89, driver 610.43.03, the (16,1) rung. The block-priority mechanism needs an
SM with room beside the round's resident blocks — on a card whose rounds are
register-bound rather than shared-bound there is no room and the entry blocks would wait
for a slot, which is the co-blocks case again.*

</details>

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
existing arithmetic compile down properly, and never from moving fewer bytes.

</details>

---

## Current focus and open leads
<details>
<summary>Details</summary>

**Where the time goes.** *(CUDA, the shipping path. Re-measured 2026-07-26, after the
`kFCap` 320 change.)*

**The per-round picture is obtainable without a profiler**, which matters because Nsight
needs no sudo on this rig (NVreg_RestrictProfilingToAdminUsers=0 is installed; check
`RmProfilingAdminOnly` in /proc/driver/nvidia/params), and takes `CLOCKS=none|base` and
`TARGET=pipeline|miner`. `MXBM_ROUND_REPS="R:N"` replays
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

**Quantified 2026-08-02**: the two rounds stand in exactly the ratio of their siphash
counts (2.00 predicted, 1.99 measured), which prices the derivation at 3.64 ms per
235 M siphashes and the match machinery at ~1.5 ms per round — and r1's scattered
payload store at *zero*. See [the siphash
decomposition](#the-top-half-of-the-pipeline-is-a-siphash-machine--and-three-ways-of-making-it-cheaper-are-already-taken),
which also closes three ways of making the siphash itself cheaper: the compiler
already takes all three.

**The 1.71× / 96 sol/s ceiling is real arithmetic but it is not one overlap away.**
`max(SM-busy, DRAM-busy)` = 20.6 ms agrees with the 21.2 ms DRAM floor above, so the
number is right. What it assumes is that the six kernels' work can be interleaved
arbitrarily — and within a solve it cannot: entry→r1→r2→r3→r4→terminal is a strict chain
with a global barrier at every step, so at any instant only one round's work exists. All
independent work comes from *another solve*, and [lead 5](#current-focus-and-open-leads)
measured that streams cannot deliver it. Reaching 21 ms therefore needs a
software-pipelined heterogeneous launch, and no kind of overlap. **That co-resident launch has
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
| occupancy | **closed 2026-07-31, from both resources.** OpenCL's 48 KB LDS made it structurally unreachable ([details](#occupancy-again)); CUDA exposes 100 KB/SM, where [3 → 4 blocks/SM is worth ~1.6 ms](#occupancy-is-worth-real-time-and-shared-memory-is-the-only-gate). r1/r2 crossed at `kFCap` 320 (**−1.25 ms**) and [r1 took a fifth block](#round-1-takes-a-fifth-block-015-ms-via-a-per-round-group-cap) (−0.15). r3 reaches 4 blocks and [does not care](#round-3-does-not-want-a-fourth-block--occupancy-pays-only-where-a-round-is-latency-bound). This entry used to end "`B` is the only term left"; it is not — **r2's 4 × 256 × 64 registers are the ENTIRE register file**, so the next block needs fewer shared bytes AND fewer registers at once. Both were moved together on 2026-08-16 and **r2's fifth block arrived and cost +0.10 ms** -- occupancy on that round is a price, not a prize ([the fifth block](#r2s-fifth-resident-block-is-reachable-and-costs-010-ms)) |
| coalescing the emit | max 1.9–2.2× against a 3× traffic cost ([two-level](#two-level-bucketing)) |
| the two atomics | both load-bearing; removing either is slower |
| `apply_mix`, back-refs, rebuild | 2.2 ms combined on OpenCL and less on CUDA — nothing left to win |
| phase overlap | **REOPENED 2026-09-08 by a fourth mechanism**: a stream at the device's greatest priority puts the entry's blocks into the warp and register room round 3's shared-memory cap leaves idle, without displacing a round block — [the entry pass beside rounds 3 and 4](#the-entry-pass-beside-rounds-3-and-4-on-a-stream-the-block-scheduler-prefers), +2.0 % at stock, gated off under caps like the co-blocks. Before it: closed by three mechanisms and harvested 2026-07-31. Grid depth (plain streams) and warp slots (same-warp hosting) were null; the third — [co-blocks](#co-blocks-the-third-overlap-mechanism-works--and-it-is-worth-04-ms-not-14), separate interleaved blocks in one launch — works, and its whole yield is **~0.5 ms**: r4's exploitable idle, whatever co-work is offered (entry, or [a whole round of the next solve](#fused_pair-two-solves-rounds-in-one-launch--the-familys-ceiling-is-05-ms)). Shipped as [speculative entry](#speculative-entry-co-scheduling-ships-in-the-miner-045-ms); the 1.71× roofline stays out of reach |
| match organization | **closed 2026-07-31, by two probes, prototype unbuilt.** The plan was fine-grained buckets matched in registers (chains, rescans, staging and barriers deleted). P2: [thin records lose 4.5× scattering into 2^21 buckets](#the-solver-reorganization-probes-the-cycle-deficit-is-not-bookkeeping) (the L2 bucket-tail cliff starts at bb = 18), so fine buckets cannot live in the global layout. P3: a counting-sort/register-pair r1 built in shared instead emits the exact pair multiset and is **1.75× slower** — its carve prices r1's cycles as derive ~2.5 + emit arithmetic ~1.8 + **all bookkeeping ~0.6** of 4.9 ms. There is no 2× in the match, for any organization of it; any rewrite's ceiling is ~0.6 ms/round |

**Leads.** The CUDA backend is ~6 % past the target and the OpenCL path 1.07× short
(1.12× before [the 2026-07-31 backport](#the-cuda-match-wins-backported-to-opencl-perfect-table--spill--per-round-caps-06-ms)).
**Goal 1 — the low-power gap — is closed by measurement**, and never by exhausted effort:
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
   MXBM to find the best cap itself.
0b. **The footprint buys reach. It was never a speed lever.** 7.46 GiB against a 3 GB design
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

   **What it reaches.** CUDA has no `CL_DEVICE_MAX_MEM_ALLOC_SIZE`, which was then the
   limit that bound OpenCL below 12 GB (lifted 2026-08-01 by [the record-set
   split](#the-record-set-split-opencl-reaches-cudas-57-gib-floor-and-gets-faster-doing-it);
   current ladder in [HW_REQUIREMENTS.md](HW_REQUIREMENTS.md#the-vram-ladder)),
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

   **The prediction is checked against the allocator, never trusted.** `rb_geometry_for()`
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
   gate every run is held to. So the geometry ladder is the lever here, and a tighter cap is not, for
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
   geometry at all, and the CUDA backend has no sort path. **Its constituency shrank
   on 2026-08-01**: the record-set split put every NVIDIA card down to ~5.7 GiB
   reported on the row-bucket ladder ([the
   split](#the-record-set-split-opencl-reaches-cudas-57-gib-floor-and-gets-faster-doing-it),
   [ladder](HW_REQUIREMENTS.md#the-vram-ladder)), where it used to catch 11 GB cards and
   anything with a small OpenCL `max_alloc`. What is left for it is devices no rung fits —
   so this is reach, at the far edge. Known starting points: the [record audit](#the-record-redundancy-audit)
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
up. That is an issue-rate limit, with bandwidth uninvolved — and round 2 was simultaneously at
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
the fast path, and it measured accordingly. Since removed from the tree; git holds it.

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
