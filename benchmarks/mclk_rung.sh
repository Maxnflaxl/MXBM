#!/usr/bin/env bash
# Is the 10501 MHz memory rung worth taking?
#
# WHY THIS EXISTS. Every figure ever published for this project was measured with the
# memory clock at 10251 MHz -- the six controlled runs in headline.sh, all 34 rows of the
# lolMiner comparison, every power sweep. But the card advertises a higher rung:
#
#   nvidia-smi -q -d SUPPORTED_CLOCKS  ->  10501 / 10251 / 5001 / 810 / 405 MHz
#
# and BeamHash III is bandwidth-bound at 76-80 % of DRAM peak in rounds 3 and 4. Nobody
# has measured the top rung, so the question is open by omission rather than by argument.
#
# THIS IS NOT AN OVERCLOCK. -lmc selects a rung the card itself reports as supported, and
# it reverts on exit and on reboot. It is not --moff, which is a real GDDR6X overclock
# whose failure mode is wrong results rather than a crash (docs/overclocking.md:250-254),
# and it is not --coff, which on this box corrupts the display and needs a hard restart.
#
# Needs root for nvidia-smi -lmc/-rmc only. Nothing else is touched; the board power limit
# stays at stock.
#
#   sudo benchmarks/mclk_rung.sh          # 90 s arms, 2 reps
#   sudo SECS=45 REPS=1 benchmarks/mclk_rung.sh
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

SECS=${SECS:-90}
REPS=${REPS:-2}
OUT=${OUT_DIR:-$ROOT/.mclk}          # NOT /tmp: tmpfs here, and a reboot wipes it
RUNGS=${RUNGS:-"10251 10501 10251"}  # bracketed, so drift has to be beaten not assumed
mkdir -p "$OUT"

cleanup() { sudo -n nvidia-smi -rmc >/dev/null 2>&1; echo "[cleanup] memory clock unlocked"; }
trap cleanup EXIT INT TERM

busy=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '$1+0>256')
[ -n "$busy" ] && { echo "ABORT: another process holds >256 MiB on the card"; exit 1; }

arm() {   # arm <tag> -> "<ms> <sol/s> <verified/solve> <SM MHz> <MEM MHz> <W>"
    local tag=$1; local log="$OUT/$tag.log"
    ( while :; do nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw \
                             --format=csv,noheader,nounits | tr -d ' '; sleep 2; done ) \
        > "$OUT/$tag.telem" 2>/dev/null & local s=$!
    timeout $((SECS + 120)) ./build/mxbm --benchmark BEAM-III --benchmark-seconds "$SECS" \
        --solver cuda > "$log" 2>&1
    kill $s 2>/dev/null; wait $s 2>/dev/null
    # anchored: a greedy regex here once captured "1" out of "77.1"
    local ms sol ver
    ms=$(grep -oP '^\s*\K[0-9.]+(?= ms/solve median)' "$log" | head -1)
    sol=$(grep -oP '^\s*\K[0-9.]+(?= sol/s)' "$log" | head -1)
    ver=$(grep -oP '\(\K[0-9.]+(?= verified solutions/solve)' "$log" | head -1)
    # drop the first 3 samples (ramp), then median each column
    local t; t=$(awk -F, 'NR>3{a[n]=$1;b[n]=$2;c[n]=$3;n++} END{
        if(!n){print "? ? ?";exit} asort(a);asort(b);asort(c)
        printf "%s %s %s", a[int(n/2)+1], b[int(n/2)+1], c[int(n/2)+1] }' "$OUT/$tag.telem")
    echo "${ms:-?} ${sol:-?} ${ver:-?} $t"
}

echo "warmup (${SECS}s, discarded) ..."
arm warmup > /dev/null

printf "\n%-16s %9s %8s %9s %8s %8s %8s\n" arm ms/solve sol/s verif/slv SM_MHz MEM_MHz W
for r in $(seq 1 "$REPS"); do
  for rung in $RUNGS; do
    sudo -n nvidia-smi -lmc "$rung,$rung" >/dev/null 2>&1 \
      || { echo "ABORT: could not lock the memory clock to $rung"; exit 1; }
    sleep 3
    got=$(nvidia-smi --query-gpu=clocks.mem --format=csv,noheader,nounits | tr -d ' ')
    [ "$got" = "$rung" ] || echo "  WARNING: asked for $rung, card reports $got"
    printf "%-16s %9s %8s %9s %8s %8s %8s\n" "$rung (rep$r)" $(arm "m$rung.r$r")
  done
done

echo
echo "Read verif/slv as the correctness column: GDDR6X link ECC retries on a marginal"
echo "rung, which costs throughput rather than corrupting results -- but a fall here"
echo "would mean the rung is not actually clean. Logs under $OUT."
