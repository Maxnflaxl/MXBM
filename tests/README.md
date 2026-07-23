# tests/ — correctness harness

- `vectors/beamhash3-kat.md` — golden known-answer vectors (source of truth).
- `kat_vectors.h` — those vectors as C arrays.
- `check.h` — zero-dependency assertion runner.
- `smoke` — build/plumbing sanity.
- `oracle_spike` — confirms the Beam reference island compiles/links.
- `primitives` — SipHash, prePow, seed, apply_mix, combine, pack KATs.
- `verify` — golden solutions accepted, bit-flips rejected.
- `differential` — our verifier vs Beam's reference (fuzz); not built
  unless `BEAM_SOURCE_DIR` is set.

Build & run: `cmake -S . -B build && cmake --build build -j && ctest --test-dir build`.
