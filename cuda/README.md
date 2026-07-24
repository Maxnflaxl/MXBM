# CUDA backend — working notes

Standalone for now: built with `nvcc` directly, not wired into CMake, so it cannot
destabilise the shipping OpenCL path while it is incomplete.

```sh
# full solver: KAT gate, then N distinct nonces end-to-end
nvcc -O3 -arch=sm_89 -std=c++17 -diag-suppress 186 -I src -I kernels/cuda -I tests \
     -I third_party/blake2b cuda/pipeline.cu src/beamhash/bh3_blake2b.cpp \
     src/beamhash/bh3_verify.cpp third_party/blake2b/blake2b-ref.c -o cuda/pipeline
./cuda/pipeline 20

nvcc -O3 -arch=sm_89 -I src cuda/test_primitives.cu -o cuda/test_primitives && ./cuda/test_primitives
nvcc -O3 -arch=sm_89       cuda/emit_shape_probe.cu -o cuda/emit_shape_probe && ./cuda/emit_shape_probe

sudo ./cuda/profile.sh          # -> cuda/ncu_report.txt
```

## Why CUDA at all

Not primarily for the language features. **Nsight Compute cannot profile OpenCL** — that
capability died with nvprof and is not coming back. Every optimization decision in
`docs/performance.md` has been made from ablation and arithmetic because we have no
instrument. Porting makes the machine observable; the language features (streaming stores,
`cp.async`, warp intrinsics) are a secondary benefit and are scoped in
`docs/performance.md` → "What a CUDA backend would (and would not) buy".

Nsight also needs counter access, which is admin-restricted by default:

```sh
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
# then reboot, or: sudo modprobe -r nvidia && sudo modprobe nvidia
```

## Status

| step | state |
|---|---|
| primitives on device | **done** — 0 mismatches vs host over 1 M elements |
| emit-shape probe | **done** — see `emit_shape_probe.cu` for results and caveats |
| fused round kernel | **done** — `kernels/cuda/fused_round.cuh`, templated |
| entry / terminal kernels | **done** — in `pipeline.cu` |
| host layer (persistent buffers, solve loop, verify) | **done** — mirrors `GpuSolver` |
| **full KAT solve** | **PASS** — 3/3 survivors, drop-free, 41.5–43.1 ms |
| recover + golden byte-compare | **PASS** — 3/3 goldens byte-identical |
| profile under Nsight | `sudo ./cuda/profile.sh` → `cuda/ncu_report.txt` |
| wire into CMake behind a flag | not started |

## Current standing vs OpenCL

| | ms | sol/s |
|---|---|---|
| OpenCL (shipping) | 40.3 | 47.1 |
| CUDA, first working port | 41.6 | 45.7 |
| **CUDA + 128-bit record access** | **35.3** | **55.2** |

Now measured **like-for-like**: median over 20 distinct nonces, persistent buffers,
including survivor readback, recovery and CPU verification, counting only solutions that
pass `bh3::is_valid_solution`. Both paths report **1.95 verified solutions/solve**, which
cross-checks that the two implementations agree.

## What the profiler actually bought

The 6 ms did not come from a guess. Nsight's stall breakdown named the limiter:

| kernel | ms | global | shared | **MIO queue** | barrier | math |
|---|---|---|---|---|---|---|
| entry | 2.58 | 2.4% | 0.1% | 0.0% | 0.0% | **43.1%** |
| r2 | 13.29 | 16.3% | 11.3% | **12.0%** | 16.4% | 11.2% |
| r3 | 12.23 | 31.2% | 15.1% | **22.0%** | 11.2% | 0.8% |
| r4 | 6.39 | **61.6%** | 8.2% | 3.0% | 10.3% | 0.5% |

MIO-queue throttle is the load/store **instruction** queue backing up — an issue-rate
limit, not a bandwidth one. With r2 at only 38 % of DRAM peak, bytes were the cheap
currency and instructions were not. Padding the round-3 record 9 → 10 u64 (costing 4 %
more traffic) made every record stride an even number of u64, so `slot*stride*8` is 16 B
aligned and 128-bit accesses became legal:

```
r2 emit :  11 x ST.64  ->  4 x ST.128 + 3      r3 load : 10 x LD.64 -> 4 x LD.128 + 3
r3 emit :  10 x ST.64  ->  4 x ST.128 + 2      r4 load :  9 x LD.64 -> 4 x LD.128 + 2
```

41.5 → 38.4 → 35.3 ms. **The compiler will not do this for you**: it cannot prove the base
pointer's alignment, and emitted zero 128-bit accesses even after the padding made them
valid. Check with `cuobjdump -sass | grep LDG.E.128` rather than assuming.

Two traps met on the way: a plain `if (LMODE == ...)` still *compiles* discarded branches,
so the alignment `static_assert`s fired in unrelated instantiations until the dispatch
became `if constexpr`; and the `cudaOccupancy` calls instantiate the templates too, so a
stale stride there fails the build in a way that looks like a kernel bug.

## The one thing that made this cheap

`src/beamhash/bh3_primitives.h` is annotated `MXBM_HD`, which expands to
`__host__ __device__` under `__CUDACC__`. The CUDA kernels therefore call **the same
source** as the CPU verifier — siphash24, apply_mix, combine and pack_indices are not
reimplemented, so bit-exactness is structural rather than something the tests have to keep
proving. That removes the part of a port like this that normally consumes the schedule.

Corollary: **do not write CUDA copies of these primitives.** Include the header.

## Porting notes for the round kernel

- `__local` → `__shared__`; `atomic_xchg` → `atomicExch`; `atomic_inc(p)` → `atomicAdd(p,1)`
  (note OpenCL's `atomic_inc` returns the OLD value, as `atomicAdd` does).
- `FUSED_LDS`'s macro parameters become template parameters — the per-round constants
  (`LOUT`, `PADN`, `SIN`, `SOUT`, `SBUILD`, strides) must stay compile-time. Making them
  runtime arguments cost **27 ms** on the OpenCL side; see "compile-time round constants"
  in `docs/performance.md`. This is the single most important thing not to regress.
- `get_local_id(0)` → `threadIdx.x`, `get_group_id(0)` → `blockIdx.x`,
  `barrier(CLK_LOCAL_MEM_FENCE)` → `__syncthreads()`.
- Launch geometry is one block per (bucket, sub-mask): `blocks = nb << submaskBits`,
  `threads = 256`. Launching one thread per group instead finds 0.4 % of collisions and
  looks almost right — that bug has already been made once.
- Keep `restrict` (`__restrict__`): worth 0.8 ms on the OpenCL side.

## Gate for the port

Same as the OpenCL path, and non-negotiable: 3 KAT goldens byte-identical,
`bucketDrops == 0`, `pairDrops == 0`, and survivors == 3 on the KAT input. **All four are
met.** Survivor count alone is not enough — it would not catch a wrong DFS order in
`recover` or a mis-packed index, both of which produce a valid-looking permutation that
fails verification, so the golden byte-compare is the real gate.
