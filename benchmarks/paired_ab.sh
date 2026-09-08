#!/usr/bin/env bash
# Paired A/B of two binaries, resolved to ~0.05 %.
#
#   benchmarks/paired_ab.sh /path/to/a /path/to/b
#   SECS=90 BLOCKS=4 benchmarks/paired_ab.sh a b             # finer, slower
#   ENV="MXBM_BB=17 MXBM_SM=0" benchmarks/paired_ab.sh a b   # measure another rung
#   sudo -v && CAP=120 benchmarks/paired_ab.sh a b           # under a power cap (root)
#
# CAP exists because every null in this ledger was measured at stock, and under a cap the
# binding currency changes -- the clock sweep puts the core-clock-scaling fraction at
# 0.83-0.90 down there against 0.62 at stock, so a change that deletes DYNAMIC work can
# pay under a cap while measuring null at 285 W.
#
# WHY THIS EXISTS. The ~2.5 % figure quoted for absolute runs is a CROSS-SESSION band and
# was being used as the resolution of a paired A/B, which retired levers this rig can see.
# Measured with this protocol: a null A/B on byte-identical binaries reads 0.040 %, and a
# real difference of 0.05 % is a four-sigma result in half an hour.
#
# The four things that make it work, each measured rather than assumed:
#
#  * SOLVE COUNT over a fixed window, not the printed median. The median prints to 0.1 ms
#    -- 0.35 %, coarser than the effects worth chasing -- while ~1045 solves in 30 s
#    resolves 0.1 %. Use the count, not sol/s: the solutions-per-solve multiplier is the
#    algorithm's and only adds variance. The harness is QUANTIZATION-limited (sd ~0.45
#    solves against a 1-solve grid), so resolution improves linearly with SECS.
#  * A WARMUP. Drift over the first fifteen minutes is +0.205 %, nearly all of it in the
#    first four runs: the count falls 1050 -> 1045 and then holds 1045 +- 1.
#  * POSITION BALANCE. ABBA cancels linear drift but not the slot itself -- the A slot
#    reads +0.024 % high on identical binaries. The block here is A B B A B A A B, so
#    both arms occupy positions summing to 18 and the bias cancels inside one run.
#  * NO CLOCK NORMALISATION. Dividing by the clock-scaling model makes the band 2.7x
#    worse: once warm the clock spans 0.5 %, not the +-10 % such a correction assumes.
#
# Two builds of one source are SASS-identical here (only the build-id note and cubin
# UUIDs differ), so a build A/B carries no code-layout term and the whole difference is
# the change. Gate the verdict on KAT 3/3 and drops == 0 for BOTH arms regardless.
set -u

BIN_A=${1:-${BIN_A:?usage: paired_ab.sh <binA> <binB>}}
BIN_B=${2:-${BIN_B:?usage: paired_ab.sh <binA> <binB>}}
NAME_A=${NAME_A:-$(basename "$BIN_A")}
NAME_B=${NAME_B:-$(basename "$BIN_B")}
SECS=${SECS:-60}
BLOCKS=${BLOCKS:-3}          # BLOCKS x 8 runs
WARMUP=${WARMUP:-150}
ENV=${ENV:-}                 # extra MXBM_* passed to both arms, e.g. "MXBM_BB=17 MXBM_SM=0"
CAP=${CAP:-}                 # board power limit in W; needs root. Empty = leave it alone
PL_DEFAULT=${PL_DEFAULT:-285}
OUT=${OUT:-paired_ab.csv}

for b in "$BIN_A" "$BIN_B"; do [ -x "$b" ] || { echo "not executable: $b"; exit 1; }; done
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$busy" -eq 0 ] || { echo "ABORT: $busy co-tenant process(es) on the card"; exit 1; }
cmp -s "$BIN_A" "$BIN_B" && echo "NOTE: the arms are byte-identical -- this is a null test"

if [ -n "$CAP" ]; then
    [ "$(id -u)" -eq 0 ] || { echo "CAP needs root (nvidia-smi -pl)"; exit 1; }
    nvidia-smi -pl "$CAP" >/dev/null || { echo "ABORT: -pl $CAP rejected"; exit 1; }
    trap 'echo "restoring ${PL_DEFAULT} W"; nvidia-smi -pl "$PL_DEFAULT" >/dev/null 2>&1' EXIT
    trap 'exit 130' INT TERM
    sleep 3
    echo "cap pinned at ${CAP} W (restored to ${PL_DEFAULT} W on exit)"
fi

echo "A=$NAME_A  B=$NAME_B   ${SECS}s x $((BLOCKS*8)) runs${ENV:+   env: $ENV}"
echo "warmup ${WARMUP}s (discarded)"
env $ENV "$BIN_A" --benchmark BEAM-III --benchmark-seconds "$WARMUP" >/dev/null 2>&1

echo "arm,pos,solves,ms_median,sm_mean,temp_mean" > "$OUT"
run() {
    local tag="$1" bin="$2" pos="$3" o solves ms sm temp tf
    tf=$(mktemp)
    ( for _ in $(seq 1 "$SECS"); do
        nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader,nounits \
            | tr -d ' ' >> "$tf"; sleep 1; done ) &
    local tel=$!
    o=$(env $ENV "$bin" --benchmark BEAM-III --benchmark-seconds "$SECS" 2>&1)
    kill $tel 2>/dev/null; wait $tel 2>/dev/null
    # Skip the ramp samples: the covariate should be the run's clock, not one instant of
    # a wandering one. Recorded for diagnosis only -- nothing is corrected by it.
    read -r sm temp < <(awk -F, 'NR>5 && $1>500 {n++; s+=$1; t+=$2}
                        END {if(n) printf "%.1f %.1f\n", s/n, t/n; else print "NA NA"}' "$tf")
    rm -f "$tf"
    solves=$(printf '%s\n' "$o" | grep -oP 'Benchmark: \K[0-9]+(?= solves)')
    ms=$(printf '%s\n' "$o"     | grep -oP '^\s+\K[0-9.]+(?= ms/solve median)')
    [ -n "$solves" ] || { echo "ABORT: no solve count parsed -- did the run fail?"; exit 1; }
    echo "$tag,$pos,$solves,$ms,$sm,$temp" | tee -a "$OUT"
}

i=0
while [ "$i" -lt "$BLOCKS" ]; do
    run "$NAME_A" "$BIN_A" 1; run "$NAME_B" "$BIN_B" 2
    run "$NAME_B" "$BIN_B" 3; run "$NAME_A" "$BIN_A" 4
    run "$NAME_B" "$BIN_B" 5; run "$NAME_A" "$BIN_A" 6
    run "$NAME_A" "$BIN_A" 7; run "$NAME_B" "$BIN_B" 8
    i=$((i + 1))
done

awk -F, -v s="$SECS" -v a="$NAME_A" -v b="$NAME_B" -v cap="$CAP" 'NR>1 {
    n[$1]++; v[$1]+=$3; q[$1]+=$3*$3; c[$1]+=$5 }
END {
  for (k in n) { m[k]=v[k]/n[k]; sd[k]=sqrt(q[k]/n[k]-m[k]*m[k])*sqrt(n[k]/(n[k]-1))
      printf "  %-14s n=%2d  %9.2f solves  sd %.3f (%.4f %%)   %8.4f ms   %.0f MHz\n",
             k, n[k], m[k], sd[k], 100*sd[k]/m[k], s*1000/m[k], c[k]/n[k] }
  se = sqrt(sd[a]^2/n[a] + sd[b]^2/n[b]); d = m[a]-m[b]
  printf "\n  %s -> %s : %+.3f solves   %+.4f %%   %+.4f ms   t=%+.2f\n",
         a, b, d, 100*d/m[a], s*1000/m[b] - s*1000/m[a], (se ? d/se : 0)
  print  "  (positive = B is slower. The null floor for this protocol is 0.040 %.)"
  if (cap != "")
      print "  Check the MHz column: if the cap did not bind, both arms are stock runs."
}' "$OUT"
