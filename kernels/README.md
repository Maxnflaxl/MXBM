# kernels/ — GPU BeamHash III solvers

Device code, one dir per backend (see [`../docs/architecture.md`](../docs/architecture.md)):

- `opencl/` — cross-vendor baseline, built first per the [roadmap](../README.md#roadmap).
- `cuda/` — NVIDIA, primary optimized backend (`sm_89` / Ada), planned after the baseline.
- `hip/` — AMD (ROCm), CUDA-alike, planned after the baseline.

All three implement the same BeamHash III pipeline (see
[`../docs/architecture.md`](../docs/architecture.md)): SipHash-2-4 seed → 5
Wagner rounds (bucket + collide + mix) → isZero() candidates → on-device
SHA-256 difficulty filter.

> Reminder: port from `beamHashIII_impl.cpp` (SipHash-based III), **not** the
> in-repo `equihash_150_5.cl` (Blake2b-based I/II). Empty until milestone M3.
