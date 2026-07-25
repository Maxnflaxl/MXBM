# Overclocking — design decisions

Status: **design agreed, not implemented.** This is the handoff for the implementation.

Everything in the "Verified" sections below was measured on this machine on 2026-07-25,
not taken from documentation or recollection. Everything in "To determine" is explicitly
*not* known yet and must be measured before it is relied on.

Reference card: RTX 4070 Ti SUPER (Ada, sm_89), driver 610.43.03.

---

## Verified: what the hardware allows

Probed by dlopening `libnvidia-ml.so.1` and reading NVML directly.

| Knob | NVML call | Current | Driver-permitted range |
|---|---|---|---|
| Core clock offset | `nvmlDeviceSetGpcClkVfOffset` | +0 MHz | −1000 … +1000 |
| Memory clock offset | `nvmlDeviceSetMemClkVfOffset` | +0 MHz | −2000 … +6000 |
| Locked core clock | `nvmlDeviceSetGpuLockedClocks` | unlocked | up to 3150 MHz max |
| Locked memory clock | `nvmlDeviceSetMemoryLockedClocks` | unlocked | up to 10501 MHz max |
| Power limit | `nvmlDeviceSetPowerManagementLimit` | 285 W | 100 … 366 (default 285) |
| Fan (2 fans) | `nvmlDeviceSetFanSpeed_v2` | auto | 30 … 100 % |

The permitted range is **register width, not a recommendation**. `+6000` on memory is
not a stable setting on any card; the driver simply does not police it.

Observed under MXBM load (from the stats table): core 2670 MHz, memory 10251 MHz,
284 W, 63 °C, fan 50 %.

## Verified: every NVML write needs root

`nvmlDeviceSetPowerManagementLimit`, called as uid 1000 with the value the card already
had (a true no-op), returns `NVML_ERROR_NO_PERMISSION` (rc=4). Reads are unprivileged;
writes are not.

## Verified: what lolMiner actually does

Checked against the installed lolMiner 1.98a binary, not from documentation.

Its `--help` section is titled **"Overclock (Experimental)"** and offers:

```
--cclk   core clock            --coff  core clock offset
--mclk   memory clock          --moff  memory clock offset
--pl     power limit           --fan   fan target
--no-oc-reset [=arg(=1)] (=0)  disable reset of GPU settings at miner exit
```

All are per-GPU, comma-separated, with `*` to skip a GPU.

Run as a normal user with `--pl 285 --moff 0`, it prints:

```
Applying overclock settings...
GPU 0: memory clock offset 0 not applied (insufficient user rights).
GPU 0: power limit 285 not applied.
Start Benchmark...
```

So lolMiner: applies OC **in-process**, **requires root**, applies it **after device
enumeration and before mining starts**, and **warns and continues at stock** rather than
refusing to start. `--no-oc-reset` defaulting to `0` means it **restores stock settings
at exit by default**.

---

## Decisions

### 1. Privilege model: the whole miner runs under sudo

Matching lolMiner. Chosen 2026-07-25.

The reason it wins is **automatic restore-on-exit**. A card left at +1200 memory after a
crash corrupts whatever runs next, and only an in-process owner of the settings can put
them back on a signal. The alternatives were considered and rejected *for now*:

- *Privileged one-shot* (`sudo mxbm --oc-apply`, then mine unprivileged) — keeps root
  away from the network code, but gives up restore-on-exit, which is the property that
  matters most.
- *setuid helper* — permanent attack surface for a feature used interactively.
- **Privileged child process** — MXBM starts as root, forks a small OC-owner process,
  then drops privileges in the parent *before it opens a socket*. Keeps restore-on-exit
  **and** keeps the TLS/JSON/stratum code unprivileged. This is strictly better on both
  axes and is **deferred, not rejected** — revisit once the basic feature works.

The cost being accepted: the stratum TLS client and JSON parser run as root.

### 2. Flags: lolMiner-compatible names

`--cclk`, `--mclk`, `--coff`, `--moff`, `--pl`, `--fan`, and `--no-oc-reset`. MXBM
already mirrors lolMiner's CLI surface elsewhere; someone moving over should not have to
relearn these. Single-GPU for now, but parse the comma/`*` syntax from the start so
multi-GPU does not become a breaking change.

### 3. Behaviour when OC is requested but cannot be applied

**Warn loudly and continue at stock**, like lolMiner — but the message must name the
cause (`insufficient permission — re-run under sudo`), not just report failure.

Note the tension with `HW_REQUIREMENTS.md` "Known limitation #2", which argues a
silently-degraded configuration is a bug. The distinction: a partial *search* mines
nothing while looking healthy, whereas a miner that failed to apply OC still mines
correctly, just slower. Degrading is acceptable; being quiet about it is not.

### 4. Restore stock settings on exit, by default

Including on `SIGINT`/`SIGTERM`. `--no-oc-reset` opts out, matching lolMiner's default
of `0`. Restore must be idempotent and must not itself require the miner to have exited
cleanly.

### 5. Clamps — **open, decide at implementation time**

Not settled. The argument for clamping is that the driver reports `+6000` for memory,
and a typo'd `--moff 12000` should not reach the card. The argument against is that any
clamp we pick is a guess about someone else's silicon. Leaning toward: clamp to a
conservative band, with an explicit override flag to exceed it.

---

## To determine — must be measured, do not assume

### The memory offset unit

`nvidia-settings` exposes memory offsets as `GPUMemoryTransferRateOffset`, in MHz of
**transfer rate**, which is 2× the memory clock. It is **not confirmed** that NVML's
`nvmlDeviceSetMemClkVfOffset` uses the same convention. So `--moff 2000` may mean
+2000 MHz or +1000 MHz of actual clock.

**Test:** apply a known offset, then read `nvmlDeviceGetClockInfo(NVML_CLOCK_MEM)` and
compare against the 10251 MHz baseline (10501 max). One measurement settles it. Report
the offset in the units the user typed, but log the resulting clock so it is unambiguous.

### Whether the community's recommended OC is right *for MXBM*

hashrate.no lists `--coff 300 --cclk 2205 --moff 2000 --pl 300` for this card.

The shape is coherent: `--cclk` plus `--coff` is the standard Linux undervolt idiom —
lock the core clock and shift the V/F curve up, so 2205 MHz runs at the voltage that
would otherwise deliver 1905 MHz.

**But those numbers are tuned for lolMiner's kernels, not ours**, and one of them looks
actively wrong for us:

- `--moff 2000` — highest expected value. MXBM is memory-bandwidth-bound; this is the
  lever on the bound resource.
- `--pl 300` — mild raise over the 285 W default (max 366). Cheap, low risk. Worth
  testing *downward* too: the `Eff. sol/s/W` column makes efficiency measurable, and a
  memory-bound kernel often loses little from a lower limit.
- `--cclk 2205` — **suspect for MXBM.** The card runs **2670 MHz** under our load today,
  so this is a ~465 MHz reduction. That is free for a fully memory-bound miner, but ncu
  measured entry/r1/r2 at *low* DRAM %peak — parts of our pipeline are not
  bandwidth-saturated, so core clock plausibly costs us throughput where it costs
  lolMiner nothing. Treat as a hypothesis to test, not a setting to adopt.

**Apply one knob at a time**, against the established methodology: median over 300
distinct nonces, ±4.1 % (1σ). A combined change cannot be attributed.

---

## Stability canaries

Two knobs fail in two different ways, so watch two different signals.

**Core instability → CPU-verify failure rate.** Every candidate goes through
`bh3::is_valid_solution` before submission. Measured baseline: 2.29 survivors/solve →
1.98 verified, so ~17 % fail. That 17 % is *algorithmic* — Wagner produces candidates
that do not satisfy the full constraint — and it is stable, which is what makes it a
usable reference. A rise means the GPU is computing wrong answers, and MXBM sees it
**before the pool does**, because bad solutions are filtered locally rather than
submitted. Worth surfacing as a stats column.

**Memory instability → sol/s falling as the offset rises.** This canary will *not* fire
on verify failures, because Ada's GDDR6X uses link-level ECC with retry: pushing memory
too far causes retries that eat bandwidth rather than corruption that changes results.
The signature is the classic hashrate cliff — throughput drops while everything still
looks correct. Find the offset where sol/s peaks and back off from it.

---

## Also worth knowing

`lolMiner --benchmark BEAM-III` runs the algorithm with **no pool and no wallet** and
prints `Average speed (15s): N sol/s` — the same metric MXBM reports. This is a much
better head-to-head than the ~400-share protocol in `docs-internal/MINER_COMP.md`:
minutes instead of days, and no pool-side variance.

It must be run on an **idle GPU**. A run taken while MXBM was mining reported 25.5–27.2
sol/s, which measures contention, not lolMiner.

lolMiner selects "BeamHash III **4G** (CUDA)" on this card — it fits the search in 4 GB
where MXBM needs 7.46 GiB. That is independent evidence for the streaming / in-place
layer-reuse route described in `HW_REQUIREMENTS.md` § "Memory efficiency".
