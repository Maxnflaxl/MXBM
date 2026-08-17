#!/usr/bin/env bash
# M1 -- does solve time scale with CORE clock harder than the DRAM share allows?
#
# The solve accounting says 15.00 ms of 28.37 is core-clock-bound (SipHash 8.84 + other
# ALU 1.98 + warp supply 4.18) and 13.37 ms is compulsory DRAM, which at a fixed memory
# clock should NOT scale. That predicts
#
#     ms(f) = 15.00 * (f_ref / f) + 13.37          f_ref = the top locked rung
#
# If the measured curve is steeper -- closer to ms(f) = 28.37 * (f_ref / f) -- then the
# DRAM block is latency-bound rather than bandwidth-bound, and the ledger's largest
# "closed" entry is mispriced. This script only produces the curve; the fit is the answer.
#
# Memory clock is deliberately NOT touched: the P-state pins it at 10251 and holding it
# there is what makes the core-clock term identifiable.
#
# Needs root for -lgc/-rgc. Nothing else on the card while it runs.
#   sudo benchmarks/clock_scaling.sh
set -u

BIN="${BIN:-./build/mxbm}"
SECS="${SECS:-30}"
CLOCKS="${CLOCKS:-2600 2400 2200 2000 1800 1600}"
OUT="${OUT:-clock_scaling.csv}"

[ "$(id -u)" -eq 0 ] || { echo "run me as root (needs nvidia-smi -lgc)"; exit 1; }
[ -x "$BIN" ] || { echo "no binary at $BIN"; exit 1; }

busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$busy" -eq 0 ] || { echo "ABORT: $busy co-tenant process(es) on the card"; exit 1; }

cleanup() { echo "resetting clocks"; nvidia-smi -rgc >/dev/null 2>&1; }
trap cleanup EXIT INT TERM

echo "clock_mhz,pass,ms_per_solve,sol_per_s,solves,sm_mhz_observed,mem_mhz_observed" > "$OUT"

run_one() {
    local f="$1" pass="$2" o ms sol solves smobs memobs
    nvidia-smi -lgc "$f","$f" >/dev/null 2>&1 || { echo "  -lgc $f rejected"; return; }
    sleep 2
    o=$("$BIN" --benchmark BEAM-III --benchmark-seconds "$SECS" 2>&1)
    read -r smobs memobs < <(nvidia-smi --query-gpu=clocks.sm,clocks.mem \
                             --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
    ms=$(printf '%s\n' "$o"    | grep -oP '^\s+\K[0-9.]+(?= ms/solve median)')
    sol=$(printf '%s\n' "$o"   | grep -oP '^\s+\K[0-9.]+(?= sol/s)')
    solves=$(printf '%s\n' "$o"| grep -oP 'Benchmark: \K[0-9]+(?= solves)')
    echo "$f,$pass,$ms,$sol,$solves,$smobs,$memobs" | tee -a "$OUT"
}

# Two passes, the second in reverse order, so any thermal or session drift shows up as a
# disagreement between passes at the same clock rather than as curvature.
for f in $CLOCKS;                     do run_one "$f" 1; done
for f in $(echo $CLOCKS | tr ' ' '\n' | tac | tr '\n' ' '); do run_one "$f" 2; done

echo
echo "wrote $OUT -- check that mem_mhz_observed is constant across every row, and that"
echo "sm_mhz_observed tracks the requested clock; a row where it does not is not a datum."
