# CUDA backend — working notes

Standalone for now: built with `nvcc` directly, not wired into CMake, so it cannot
destabilise the shipping OpenCL path while it is incomplete.

```sh
nvcc -O3 -arch=sm_89 -I src cuda/test_primitives.cu -o cuda/test_primitives && ./cuda/test_primitives
nvcc -O3 -arch=sm_89       cuda/emit_shape_probe.cu -o cuda/emit_shape_probe && ./cuda/emit_shape_probe
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
| host layer (buffers, launches) | **done** — enough to run and gate a full solve |
| **full KAT solve** | **PASS** — 3/3 survivors, drop-free, 41.6–42.4 ms |
| recover + golden byte-compare | not started (survivor count is the gate for now) |
| profile under Nsight | blocked on counter permission (reboot) |
| wire into CMake behind a flag | not started |

## Current standing vs OpenCL

| | ms | note |
|---|---|---|
| OpenCL (shipping) | **40.2** | after this session's tuning |
| CUDA (first working port) | **41.6** | untuned; no CUDA-specific feature used yet |

Within ~4 % on the first run, which is the expected starting point — the port reproduces
the OpenCL algorithm exactly, including every tuning decision baked into it (compile-time
per-round constants, the (16,1) geometry, `restrict`, the non-divergent expand). Nothing
CUDA-only has been applied yet; that is the next phase, and the profiler is the point.

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
`bucketDrops == 0`, `pairDrops == 0`, and survivors == 3 on the KAT input.
