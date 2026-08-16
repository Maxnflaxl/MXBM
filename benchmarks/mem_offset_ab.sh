#!/usr/bin/env bash
# Bracket one memory V/F offset against stock. The sweep locates a candidate with one
# unbracketed sample per point; this is what turns it into a figure.
#
# Both arms pass --moff so the OC apply/restore path is identical in each, and only the
# value differs. Watch J/sol as well as ms: the board is at its power limit, so a memory
# clock is bought with watts and a speed win can still be an efficiency loss.
#
#     sudo MOFF=1200 benchmarks/mem_offset_ab.sh
set -u
cd "$(dirname "$0")/.."
. benchmarks/oc_guard.sh
BIN=${BIN:-./build/mxbm}
MOFF=${MOFF:?set MOFF to the offset under test}
SECS=${SECS:-40}
BLOCKS=${BLOCKS:-3}
OUT=$(mktemp -d)

oc_require_idle
echo "warming up"
"$BIN" --benchmark BEAM-III --benchmark-seconds 45 --moff 0 > "$OUT/warm.log" 2>&1

run() {  # run <arm> <offset>
  oc_require_idle
  local arm=$1 off=$2 n
  n=$(ls "$OUT"/${arm}_*.log 2>/dev/null | wc -l)
  oc_sample_start "$OUT/clk_${arm}_$n.csv"
  MXBM_DROP_STATS=1 "$BIN" --benchmark BEAM-III --benchmark-seconds "$SECS" \
      --moff "$off" > "$OUT/${arm}_$n.log" 2>&1
  printf "%s (+%s, held %s)  %s  %s\n" "$arm" "$off" "$(oc_sample_stop)" \
    "$(grep -o '[0-9.]* solves/s' "$OUT/${arm}_$n.log")" \
    "$(grep -o '[0-9.]* J/solution' "$OUT/${arm}_$n.log")"
}

for i in $(seq 1 "$BLOCKS"); do
  for arm in A B B A; do
    [ "$arm" = A ] && run A 0 || run B "$MOFF"
  done
done

python3 - "$OUT" "$MOFF" <<'PY'
import sys, re, glob, statistics
out, moff = sys.argv[1], sys.argv[2]
res, en, ver = {}, {}, {}
for arm in "AB":
    v, e, s = [], [], []
    for f in sorted(glob.glob(f"{out}/{arm}_*.log")):
        t = open(f).read()
        m = re.search(r"([\d.]+) solves/s", t)
        j = re.search(r"([\d.]+) J/solution", t)
        k = re.search(r"([\d.]+) verified solutions/solve", t)
        if m: v.append(1000/float(m.group(1)))
        if j: e.append(float(j.group(1)))
        if k: s.append(float(k.group(1)))
        if "entry=0 stage=0 out=0 walk=0" not in t:
            print(f"  !! drop counter fired in {f}")
    res[arm], en[arm], ver[arm] = v, e, s
    print(f"{arm}  n={len(v)}  {statistics.mean(v):.3f} ms  "
          f"[{min(v):.3f}, {max(v):.3f}]   {statistics.mean(e):.3f} J/sol   "
          f"verified/solve {min(s):.2f}-{max(s):.2f}")
a, b = statistics.mean(res['A']), statistics.mean(res['B'])
ja, jb = statistics.mean(en['A']), statistics.mean(en['B'])
ov = max(res['B']) > min(res['A']) and max(res['A']) > min(res['B'])
print(f"\nA = stock, B = --moff {moff}")
print(f"time   {b-a:+.3f} ms  ({100*(b-a)/a:+.2f} %)   ranges overlap={'YES' if ov else 'no'}")
print(f"energy {jb-ja:+.3f} J/sol ({100*(jb-ja)/ja:+.2f} %)")
if min(ver['B']) < 1.94:
    print("WARNING: verified solutions/solve below band in the offset arm")
PY
echo "logs in $OUT"
