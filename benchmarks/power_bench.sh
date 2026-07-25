#!/usr/bin/env bash
# Energy benchmark: run the offline benchmark while sampling NVML, and report
# joules per solution alongside sol/s. Watts alone conflate "faster" with
# "hungrier"; J/sol is the figure that decides a power-limited rig.
#
# Usage: benchmarks/power_bench.sh [SECONDS] [LABEL] [-- extra mxbm args...]
set -u
SECS=${1:-90}
LABEL=${2:-run}
shift 2 2>/dev/null || shift $#
[ "${1:-}" = "--" ] && shift

ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT_DIR:-/tmp/mxbm-power}
mkdir -p "$OUT"
CSV="$OUT/$LABEL.csv"
LOG="$OUT/$LABEL.log"

nvidia-smi --query-gpu=power.draw,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,utilization.memory \
           --format=csv,noheader,nounits -lms 200 > "$CSV" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

"$ROOT/build/mxbm" --benchmark BEAM-III --benchmark-seconds "$SECS" --nocolor "$@" 2>&1 | tee "$LOG"

sleep 0.5
kill $SAMPLER 2>/dev/null
wait $SAMPLER 2>/dev/null

python3 - "$CSV" "$LOG" "$LABEL" "$SECS" <<'PY'
import sys, statistics as st
csv, log, label, secs = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
rows = []
for line in open(csv):
    p = [x.strip() for x in line.split(',')]
    if len(p) < 6: continue
    try: rows.append(tuple(float(x) for x in p))
    except ValueError: continue
# Drop the ramp: the first 15 s of samples (5/s) and the tail after the run ends.
warm = int(15 * 5)
body = rows[warm:] if len(rows) > warm + 20 else rows
if not body:
    print("no samples"); sys.exit(0)
def col(i): return [r[i] for r in body]
sol = None
for line in open(log):
    if 'sol/s' in line and 'verified' in line:
        sol = float(line.split('sol/s')[0].split()[-1])
p = col(0)
print(f"\n==== POWER [{label}] n={len(body)} samples over {len(body)*0.2:.0f}s (warmup dropped) ====")
print(f"  power      median {st.median(p):7.1f} W   mean {st.mean(p):7.1f}   sd {st.pstdev(p):5.2f}   "
      f"min {min(p):.1f}   max {max(p):.1f}")
for name, i, unit in (("sm clock",1,"MHz"), ("mem clock",2,"MHz"), ("temp",3,"C"),
                      ("util gpu",4,"%"), ("util mem",5,"%")):
    c = col(i)
    print(f"  {name:10s} median {st.median(c):7.1f} {unit:3s}  mean {st.mean(c):7.1f}   "
          f"min {min(c):.0f}   max {max(c):.0f}")
if sol:
    print(f"  sol/s      {sol:.2f}")
    print(f"  EFFICIENCY {sol/st.median(p):.4f} sol/s/W     {st.median(p)/sol:.3f} J/sol")
PY
