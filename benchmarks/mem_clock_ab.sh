#!/usr/bin/env bash
# Lead 7.21: does the memory clock have headroom above its running 10251 MHz?
# Rounds 3 and 4 sit at 76.5 % and 82.7 % of DRAM peak and are ~55 % of the solve,
# so a higher rung raises their roofline rather than chasing the gap to it.
#
# Needs root for `nvidia-smi -lmc`. Run with the card otherwise idle:
#     sudo benchmarks/mem_clock_ab.sh
set -u
cd "$(dirname "$0")/.."
BIN=${BIN:-./build/mxbm}
SECS=${SECS:-40}
BLOCKS=${BLOCKS:-3}
OUT=$(mktemp -d)

echo "== supported memory clocks"
nvidia-smi --query-supported-clocks=mem --format=csv,noheader | sort -un | tail -5
BASE=$(nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader,nounits)
echo "== reported max ${BASE} MHz"

cleanup() { nvidia-smi -rmc > /dev/null 2>&1; nvidia-smi -rgc > /dev/null 2>&1; }
trap cleanup EXIT

run() {  # run <arm> <mem-mhz>
  local arm=$1 mc=$2
  if [ "$mc" = "auto" ]; then nvidia-smi -rmc > /dev/null
  else nvidia-smi -lmc "$mc" > /dev/null || { echo "  -lmc $mc REFUSED"; return 1; }
  fi
  local n; n=$(ls "$OUT"/${arm}_*.log 2>/dev/null | wc -l)
  "$BIN" --benchmark BEAM-III --benchmark-seconds "$SECS" > "$OUT/${arm}_$n.log" 2>&1
  printf "%s (%s MHz, achieved %s)  %s\n" "$arm" "$mc" \
    "$(nvidia-smi --query-gpu=clocks.mem --format=csv,noheader,nounits)" \
    "$(grep -o '[0-9.]* solves/s' "$OUT/${arm}_$n.log")"
}

"$BIN" --benchmark BEAM-III --benchmark-seconds 45 > "$OUT/warm.log" 2>&1
for i in $(seq 1 "$BLOCKS"); do
  for arm in A B B A; do
    [ "$arm" = A ] && run A auto || run B "$BASE"
  done
done

python3 - "$OUT" <<'PY'
import sys, re, glob, statistics
out=sys.argv[1]; res={}
for arm in "AB":
    v=[]
    for f in sorted(glob.glob(f"{out}/{arm}_*.log")):
        m=re.search(r"([\d.]+) solves/s", open(f).read())
        if m: v.append(1000/float(m.group(1)))
    if not v: continue
    res[arm]=v
    print(f"{arm}  n={len(v)}  mean {statistics.mean(v):.3f} ms  min {min(v):.3f}  max {max(v):.3f}")
if len(res)==2:
    a,b=statistics.mean(res['A']),statistics.mean(res['B'])
    ov = max(res['B'])>min(res['A']) and max(res['A'])>min(res['B'])
    print(f"A = stock memory clock, B = locked to the reported max")
    print(f"delta {b-a:+.3f} ms  ({100*(b-a)/a:+.2f} %)   ranges overlap={'YES' if ov else 'no'}")
PY
echo "logs in $OUT"
