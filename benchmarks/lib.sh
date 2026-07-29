#!/usr/bin/env bash
# Shared helpers for the benchmark scripts. Source after ROOT is set:
#
#     ROOT=$(cd "$(dirname "$0")/.." && pwd)
#     . "$ROOT/benchmarks/lib.sh"
#
# WHY THIS FILE EXISTS. Three scripts had independently written the same cache
# check -- `[ -x "$out" ] && return 0` -- which reuses a cached binary on
# filename existence alone, with no source hash and no mtime. Across a source
# change that serves the PREVIOUS build: the run succeeds, the gate passes, the
# numbers look plausible, and they belong to code that is no longer in the tree.
# It is the same silent-no-op class as the unasserted string replace in
# docs/performance-research.md "A note on method", which was chased across three
# sessions as a suspected race before the patch was found never to have applied.
# Patching three copies would have left the fourth to be written next time, so
# the check lives here once.

# bench_stale <binary> [extra-source...]
#
# True (0) when <binary> must be rebuilt: missing, or older than any source that
# feeds it. Sources are the CUDA pipeline translation unit and every tree it
# includes; pass extra paths for anything a particular script also compiles.
#
#     bench_stale "$out" || return 0      # inside a build function
#     bench_stale "$out" || continue      # inside a build loop
bench_stale() {
    [ -x "$1" ] || return 0
    local bin=$1; shift
    [ -n "$(find "$ROOT/cuda/pipeline.cu" \
                 "$ROOT/kernels/cuda" \
                 "$ROOT/src/beamhash" \
                 "$ROOT/third_party/blake2b" \
                 "$@" -newer "$bin" -print -quit 2>/dev/null)" ]
}

# bench_build_pipeline
#
# Build cuda/pipeline if it is missing or stale. stage_power.sh ran the
# checked-in binary directly and never built it, so a stage attribution could be
# taken against a pipeline predating the change under test -- the binary in the
# tree on 2026-07-29 was three days older than fused_round.cuh, i.e. older than
# both the quad record and the per-round group cap.
bench_build_pipeline() {
    local out="$ROOT/cuda/pipeline"
    bench_stale "$out" || return 0
    echo "  building cuda/pipeline (stale or missing) ..."
    "${NVCC:-nvcc}" -O3 -arch="${ARCH:-sm_89}" -std=c++17 -diag-suppress 186 \
        -I "$ROOT/src" -I "$ROOT/kernels/cuda" -I "$ROOT/tests" -I "$ROOT/third_party/blake2b" \
        "$ROOT/cuda/pipeline.cu" "$ROOT/src/beamhash/bh3_blake2b.cpp" \
        "$ROOT/src/beamhash/bh3_verify.cpp" "$ROOT/third_party/blake2b/blake2b-ref.c" \
        -o "$out" 2> "$ROOT/cuda/build-pipeline.log" \
      || { echo "build failed, see cuda/build-pipeline.log"; tail -5 "$ROOT/cuda/build-pipeline.log"; exit 1; }
}
