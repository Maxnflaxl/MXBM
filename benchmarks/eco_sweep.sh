#!/usr/bin/env bash
# Do the compute-for-bytes trades stay right under a power cap?
#
# WHY. Below ~160 W the whole pipeline is core-clock-bound: 99.5 ms/solve at 100 W is
# 2.94x stock at 3.38x less clock, i.e. even the "DRAM-bound" rounds stop saturating
# DRAM because the LSU issue rate scales with core clock. The currency at a low cap is
# therefore issued instructions, not bytes -- which reprices two settled trades:
#
#   * geometry (17,0) removes the sub-mask rescan (fewer memory instructions) and pays
#     in scatter locality (a DRAM-side cost, idle at low caps). At stock it is +9 %;
#     the prediction is that it crosses over somewhere below ~160 W.
#   * MXBM_R2_FULL removes round 2's 14-siphash rebuild (fewer ALU ops) and pays in
#     scattered bytes. Same argument, opposite resource.
#
# Both are measured against the shipping (16,1) re-derivation build at each cap, KAT
# and drops gated at every point. Run as a user with NOPASSWD nvidia-smi (see
# bench_gpu_root in lib.sh); the stock cap is restored on every exit path.
#
#   benchmarks/eco_sweep.sh
#   CAPS="100 140 180" benchmarks/eco_sweep.sh
#
# MEASURED 2026-07-31 (RTX 4070 Ti SUPER, table in docs/performance.md): the (17,0)
# half of the prediction is right -- it crosses at ~160 W and wins 2.6 % at the 100 W
# floor -- and the R2_FULL half is WRONG: it loses at every cap, because the rebuild's
# ALU work hides in stall shadows at low clock just as it does at stock, while its
# extra memory instructions do not. Re-derivation is the right trade at every power.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
. "$ROOT/benchmarks/lib.sh"

if ! bench_gpu_root; then
    echo "needs passwordless 'sudo nvidia-smi' to set power limits" >&2
    exit 1
fi

CAPS=${CAPS:-"100 120 140 160 180 210 285"}
OUT=${OUT_DIR:-/tmp/mxbm-eco}
mkdir -p "$OUT"

DEFAULT=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | cut -d. -f1)
restore() { sudo -n nvidia-smi -pl "$DEFAULT" >/dev/null 2>&1; }
trap restore EXIT INT TERM

bench_build_pipeline
R2F="$ROOT/cuda/pipeline-r2full"
if bench_stale "$R2F"; then
    echo "  building cuda/pipeline-r2full ..."
    "${NVCC:-nvcc}" -O3 -arch="${ARCH:-sm_89}" -std=c++17 -diag-suppress 186 \
        -DMXBM_R2_FULL=1 \
        -I "$ROOT/src" -I "$ROOT/kernels/cuda" -I "$ROOT/tests" -I "$ROOT/third_party/blake2b" \
        "$ROOT/cuda/pipeline.cu" "$ROOT/src/beamhash/bh3_blake2b.cpp" \
        "$ROOT/src/beamhash/bh3_verify.cpp" "$ROOT/third_party/blake2b/blake2b-ref.c" \
        -o "$R2F" 2> "$OUT/build-r2full.log" \
      || { echo "build failed, see $OUT/build-r2full.log"; exit 1; }
fi

run_one() {  # run_one <label> <cap> <nsolves> <env> <binary>
    local label=$1 cap=$2 n=$3 envs=$4 bin=$5
    nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader,nounits -lms 500 \
        > "$OUT/$label-$cap.csv" &
    local sampler=$!
    local out; out=$(env $envs "$bin" "$n" 2>&1)
    kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
    echo "$out" > "$OUT/$label-$cap.log"
    local kat ms sols drops
    kat=$(echo "$out" | grep -o "KAT.*-> [A-Z]*" | grep -o "[A-Z]*$")
    ms=$(echo "$out" | grep "end-to-end" | grep -oP '[\d.]+(?= ms/solve)')
    sols=$(echo "$out" | grep "solutions" | grep -oP '[\d.]+(?= sol/s)')
    drops=$(echo "$out" | grep "drops     :" | awk '{print $3}')
    local clk; clk=$(python3 - "$OUT/$label-$cap.csv" <<'PY'
import sys
rows=[l.split(',') for l in open(sys.argv[1]) if ',' in l]
rows=rows[len(rows)//5 : -len(rows)//5 or None]
print(sorted(int(r[0]) for r in rows)[len(rows)//2] if rows else 0)
PY
)
    printf "%-4s %-7s ms=%-8s sol/s=%-6s SM=%-5s KAT=%s drops=%s\n" \
        "$cap" "$label" "$ms" "$sols" "$clk" "$kat" "$drops" | tee -a "$OUT/rows.txt"
}

: > "$OUT/rows.txt"
for cap in $CAPS; do
    sudo -n nvidia-smi -pl "$cap" >/dev/null || { echo "cap $cap refused"; exit 1; }
    sleep 3
    case $cap in                       # ~40-50 s of solves per run
        100) n=350;; 110|120) n=420;; 130|140) n=520;; 150|160) n=650;;
        170|180) n=750;; 190|200|210) n=950;; *) n=1200;;
    esac
    run_one base   "$cap" "$n" ""           "$ROOT/cuda/pipeline"
    run_one bb17   "$cap" "$n" "MXBM_BB=17" "$ROOT/cuda/pipeline"
    run_one r2full "$cap" "$n" ""           "$R2F"
done
restore
echo "stock cap restored; rows in $OUT/rows.txt"
