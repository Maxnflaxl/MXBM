#!/usr/bin/env bash
# Where does the memory V/F offset stop paying?
#
# Rounds 3 and 4 are 45 % of the solve at 77-89 % of DRAM peak, so this is the one knob
# on the bound resource. The unit is transfer rate, so an offset of N is N/2 MHz of
# actual clock, on a card whose load clock is 10251.
#
# Use the OFFSET, not `--mclk`: locking the memory clock to the driver's reported maximum
# is a measured LOSS on this card (+0.95 %), because a locked P-state buys its clock with
# voltage and the board is already at its power cap. An offset shifts the V/F curve
# instead. See docs/performance-research.md.
#
# GDDR6X answers a clock it cannot hold by RETRYING rather than by corrupting, so the
# failure signature is a slowdown, not a wrong answer. A step is rejected if the clock did
# not move, if verified solutions/solve leaves its band, if any drop counter fires, or if
# ms/solve regresses materially.
#
# This is a sweep: one unbracketed sample per point, so it LOCATES a candidate and is
# never itself evidence about one. Bracket the winner with an interleaved A/B before
# quoting it.
#
# Needs root for nvmlDeviceSetMemClkVfOffset. Run with the card otherwise idle:
#     sudo benchmarks/mem_offset_sweep.sh
set -u
cd "$(dirname "$0")/.."
. benchmarks/oc_guard.sh
BIN=${BIN:-./build/mxbm}
SECS=${SECS:-40}
STEPS=${STEPS:-"0 200 400 600 800 1000 1200"}
OUT=$(mktemp -d)

oc_require_idle
echo "warming up"
"$BIN" --benchmark BEAM-III --benchmark-seconds 45 > "$OUT/warm.log" 2>&1

printf '%8s %14s %10s %10s %9s  %s\n' offset "mem,core MHz" "ms/solve" "J/sol" verif note

prev_ms=""
for off in $STEPS; do
  oc_require_idle                       # the knob is device-global; re-check every step
  log="$OUT/off$off.log"
  oc_sample_start "$OUT/clk$off.csv"
  MXBM_DROP_STATS=1 "$BIN" --benchmark BEAM-III --benchmark-seconds "$SECS" \
      --moff "$off" > "$log" 2>&1
  clk=$(oc_sample_stop)
  ms=$(grep -oE '[0-9.]+ ms/solve median' "$log" | grep -oE '^[0-9.]+')
  jps=$(grep -oE '[0-9.]+ J/solution' "$log" | grep -oE '^[0-9.]+')
  ver=$(grep -oE '[0-9.]+ verified solutions/solve' "$log" | grep -oE '^[0-9.]+')
  bad=$(grep -oE 'drops entry=[0-9]+ stage=[0-9]+ out=[0-9]+ walk=[0-9]+' "$log" \
        | grep -vc 'entry=0 stage=0 out=0 walk=0' || true)

  note=""
  if   grep -qiE 'not applied|insufficient|REFUSED' "$log"; then note="REFUSED"
  elif [ -z "${ms:-}" ];        then note="NO RESULT"
  elif [ "${bad:-0}" != "0" ];  then note="DROPS -- reject"
  elif [ -n "${ver:-}" ] && awk "BEGIN{exit !($ver < 1.94 || $ver > 2.07)}"; then
       note="verified/solve out of band -- reject"
  elif [ -n "${prev_ms:-}" ] && awk "BEGIN{exit !($ms > $prev_ms * 1.02)}"; then
       note="2 % slower than previous -- the retry wall"
  fi

  printf '%8s %14s %10s %10s %9s  %s\n' \
         "+$off" "${clk:-?}" "${ms:-?}" "${jps:-?}" "${ver:-?}" "$note"
  [ -n "$note" ] && { echo "stopping at +$off"; break; }
  prev_ms=$ms
done

echo
echo "The offset unit is transfer rate: actual clock is 10251 + offset/2 MHz."
echo "This located a candidate. Bracket it before quoting it."
echo "logs in $OUT"
