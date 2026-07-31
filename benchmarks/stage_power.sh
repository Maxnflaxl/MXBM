#!/usr/bin/env bash
# Per-stage board-power attribution.
#
# The card runs pinned at its 285 W limit, so "which kernel costs the watts" cannot be
# read off a whole-solve figure. This replays ONE stage N times inside the solve that
# produced its input (MXBM_ROUND_REPS / MXBM_ENTRY_REPS -- see cuda/pipeline.cu) and
# solves for that stage's own time and power from the shift it causes:
#
#     t_s = (T_N - T_1) / (N-1)
#     E_s = (J_N - J_1) / (N-1)        P_s = E_s / t_s
#
# T is ms/solve and J the joules/solve from the card's energy counter, which the
# pipeline binary brackets around its own timed loop (so init is excluded). On cards
# without the counter, P falls back to the sampled median via the old identity
# P_s = (P_N*T_N - P_1*T_1)/(T_N - T_1). Every rep is a complete, correct pass over
# the real input, so per-rep memory traffic matches production exactly.
#
# Every run must be long enough that the sampled median is steady-state and not ramp:
# the solve count is derived per stage from its known cost so each run lasts SECONDS.
#
# Usage: benchmarks/stage_power.sh [REPS] [SECONDS]
set -u
REPS=${1:-8}
SECONDS_PER_RUN=${2:-45}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
. "$ROOT/benchmarks/lib.sh"
OUT=${OUT_DIR:-/tmp/mxbm-stage}
mkdir -p "$OUT"

# ms/solve for the baseline, and each stage's own ms (measured; only used to size runs).
# BASE_MS is overridable so a capped/rung wrapper can size its runs for the slower
# solve; the per-stage estimates scale with it.
BASE_MS=${BASE_MS:-35.2}
SCALE=$(python3 -c "print($BASE_MS/35.2)")
declare -A STAGE_MS=( [entry]=2.7 [r1]=5.8 [r2]=10.5 [r3]=10.0 [r4]=5.7 [term]=1.05 )
for k in "${!STAGE_MS[@]}"; do STAGE_MS[$k]=$(python3 -c "print(${STAGE_MS[$k]}*$SCALE)"); done

run() {   # run <label> <solves> <env-assignment...>
    local label=$1 solves=$2; shift 2
    nvidia-smi --query-gpu=power.draw,clocks.sm,temperature.gpu \
               --format=csv,noheader,nounits -lms 200 > "$OUT/$label.csv" &
    local sampler=$!
    env "$@" "$ROOT/cuda/pipeline" "$solves" > "$OUT/$label.log" 2>&1
    sleep 0.3; kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
}

solves_for() { python3 -c "print(max(20,int($SECONDS_PER_RUN*1000/$1)))"; }

# Before anything is measured, and never while a run is in flight -- this script
# used to run whatever cuda/pipeline happened to be in the tree.
bench_build_pipeline

echo "baseline ($(solves_for $BASE_MS) solves) ..."
run baseline "$(solves_for $BASE_MS)" MXBM_NONE=1
for s in entry r1 r2 r3 r4 term; do
    case $s in
        entry) e="MXBM_ENTRY_REPS=$REPS" ;;
        term)  e="MXBM_ROUND_REPS=5:$REPS" ;;
        *)     e="MXBM_ROUND_REPS=${s#r}:$REPS" ;;
    esac
    ms=$(python3 -c "print($BASE_MS + ($REPS-1)*${STAGE_MS[$s]})")
    n=$(solves_for "$ms")
    echo "measuring $s ($e, $n solves ~${ms}ms each) ..."
    run "$s" "$n" "$e"
done

python3 - "$OUT" "$REPS" <<'PY'
import sys, statistics as st, os
out, reps = sys.argv[1], int(sys.argv[2])
def read(label):
    ms = None; drops = None; jps = None
    for line in open(f"{out}/{label}.log"):
        if line.startswith("end-to-end"): ms = float(line.split(":")[1].split("ms")[0])
        if line.startswith("energy"): jps = float(line.split(":")[1].split("J/solve")[0])
        if line.startswith("drops"): drops = line.split(":")[1].strip()
    rows = []
    for line in open(f"{out}/{label}.csv"):
        p = [x.strip() for x in line.split(",")]
        if len(p) < 3: continue
        try: rows.append(tuple(float(x) for x in p))
        except ValueError: pass
    body = rows[50:-3] if len(rows) > 120 else rows       # drop 10 s of ramp and the tail
    if len(body) < 60: print(f"WARNING: {label} has only {len(body)} steady samples")
    return ms, st.median([r[0] for r in body]), st.median([r[1] for r in body]), drops, jps
T1, P1, C1, d1, J1 = read("baseline")
counter = J1 is not None
print(f"\nbaseline: {T1:.2f} ms/solve   {P1:.1f} W   {C1:.0f} MHz   drops {d1}"
      + (f"   {J1:.2f} J/solve (counter)" if counter else "   (no energy counter: sampled watts)"))
print(f"\n{'stage':6} {'T_N ms':>8} {'P_N W':>7} {'clk':>6} | {'t_stage ms':>10} {'%time':>6} "
      f"{'P_stage W':>10} {'J/solve':>8} {'%energy':>8}")
rows = []
for s in ("entry","r1","r2","r3","r4","term"):
    TN, PN, CN, dn, JN = read(s)
    dt = TN - T1
    ts = dt/(reps-1)
    if counter and JN is not None:
        Es = (JN - J1)/(reps-1)          # exact joules per stage execution
        Ps = 1000*Es/ts
    else:
        Ps = (PN*TN - P1*T1)/dt
        Es = ts*Ps/1000
    rows.append((s, TN, PN, CN, ts, Ps, Es, dn))
tot_t = sum(r[4] for r in rows)
tot_e = sum(r[6] for r in rows)
for s, TN, PN, CN, ts, Ps, Es, dn in rows:
    print(f"{s:6} {TN:8.2f} {PN:7.1f} {CN:6.0f} | {ts:10.2f} {100*ts/T1:6.1f} "
          f"{Ps:10.1f} {Es:8.2f} {100*Es/tot_e:8.1f}"
          + ("   DROPS!" if dn and dn != "0  (clean)" else ""))
print(f"{'sum':6} {'':8} {'':7} {'':6} | {tot_t:10.2f} {100*tot_t/T1:6.1f} "
      f"{1000*tot_e/tot_t:10.1f} {tot_e:8.2f}")
print(f"\nunattributed: {T1-tot_t:.2f} ms/solve ({100*(T1-tot_t)/T1:.1f} %) -- memsets, "
      f"launch gaps, readback, CPU verify"
      + (f"; {J1-tot_e:+.2f} J/solve vs the counter's whole-solve figure" if counter else ""))
PY
