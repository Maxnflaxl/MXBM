#!/usr/bin/env bash
# Interleaved A/B of the packed tab word (MXBM_TABPACK) under a power cap.
#
# WHY. The change is -0.060 ms at stock, measured over eight interleaved arms a
# side. The 2026-08-16 cap sweep that followed it reads 2-4 % SLOWER than the
# previous kernel's sweep at every cap below 240 W, which is the wrong direction
# for an instruction-deleting change -- and the right direction for what this
# change actually is: it removes a per-group barrier and a 128-word clear, and
# adds one mask per staged element. A barrier is a wait, so it is worth less as
# the core slows; an added instruction bills the starved core directly.
#
# A swept column is one unbracketed run per cap, and this instrument already has
# a documented non-reproduction (lead 7.17, ~5.5 % at 120 W on an unchanged
# binary). So the sweep cannot decide this. An interleaved pair can.
#
# Sets the cap, runs A B B A per block, restores the default on exit.
#   sudo -v && benchmarks/tabpack_cap_ab.sh            # 120 W, 3 blocks, 45 s arms
#   sudo -v && CAP=180 BLOCKS=2 benchmarks/tabpack_cap_ab.sh
set -u
CAP=${CAP:-120}
SECS=${SECS:-45}
BLOCKS=${BLOCKS:-3}
A=${A:-./build-tp0/mxbm}      # control: MXBM_TABPACK=0
B=${B:-./build/mxbm}          # shipping: packed
OUT=${OUT:-/tmp/mxbm-tabpack-cap}

for f in "$A" "$B"; do
    [ -x "$f" ] || { echo "missing $f"; exit 1; }
done
sudo -n true 2>/dev/null || { echo "needs a warm sudo ticket: run 'sudo -v' first"; exit 1; }

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$used" -gt 256 ] && { echo "another process holds ${used} MiB on the card"; exit 1; }

DEFAULT=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | cut -d. -f1)
restore() { sudo -n nvidia-smi -pl "$DEFAULT" >/dev/null 2>&1; echo "restored ${DEFAULT} W"; }
trap restore EXIT INT TERM
sudo -n nvidia-smi -pl "$CAP" >/dev/null 2>&1 || { echo "could not set -pl $CAP"; exit 1; }

mkdir -p "$OUT"; rm -f "$OUT"/*.log "$OUT/seq.txt"
declare -A BIN=([A]=$A [B]=$B)
echo "cap ${CAP} W, ${BLOCKS} x (A B B A) of ${SECS}s, warmup 90s"
"$A" --benchmark BEAM-III --benchmark-seconds 90 > "$OUT/warmup.log" 2>&1

printf "%-3s %10s %8s %8s %7s\n" arm ms/solve sol/s verif W
for i in $(seq 1 "$BLOCKS"); do
  for arm in A B B A; do
    n=$(ls "$OUT"/${arm}_*.log 2>/dev/null | wc -l)
    f="$OUT/${arm}_$n.log"
    "${BIN[$arm]}" --benchmark BEAM-III --benchmark-seconds "$SECS" > "$f" 2>&1
    sps=$(sed -n 's/.*(\([0-9.]*\) verified solutions\/solve, \([0-9.]*\) solves\/s).*/\2/p' "$f")
    ver=$(sed -n 's/.*(\([0-9.]*\) verified solutions\/solve.*/\1/p' "$f")
    sol=$(sed -n 's/^ *\([0-9.]*\) sol\/s.*/\1/p' "$f")
    w=$(sed -n 's/.*(\([0-9.]*\) W mean.*/\1/p' "$f")
    ms=$(python3 -c "print(f'{1000/$sps:.3f}')" 2>/dev/null)
    printf "%-3s %10s %8s %8s %7s\n" "$arm" "$ms" "$sol" "$ver" "$w" | tee -a "$OUT/seq.txt"
  done
done

python3 - "$OUT" <<'PY'
import sys, re, glob, statistics
out = sys.argv[1]
for arm in "AB":
    v = []
    for f in sorted(glob.glob(f"{out}/{arm}_*.log")):
        m = re.search(r"([\d.]+) solves/s", open(f).read())
        if m: v.append(1000 / float(m.group(1)))
    if v:
        print(f"{arm}  n={len(v)}  mean {statistics.mean(v):.3f}  "
              f"min {min(v):.3f}  max {max(v):.3f}")
PY
echo "A = MXBM_TABPACK=0 (control), B = shipping. Ranges must not overlap to read a sign."
echo "logs: $OUT"
