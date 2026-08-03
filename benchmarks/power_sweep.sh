#!/usr/bin/env bash
# Sweep the board power limit and measure sol/s and J/sol at each point.
#
# WHY: at stock the card sits pinned at its 285 W cap in every kernel (verified:
# clocks_throttle_reasons.sw_power_cap = Active throughout), while the reference draws
# ~239 W and is NOT capped. So "MXBM is 11 % less efficient" compares two different
# operating points. This measures MXBM's own speed/power curve, which is the only way
# to compare the two miners at equal power.
#
#   sudo -v && benchmarks/power_sweep.sh            # sudo is used ONLY for nvidia-smi -pl
#
# The limit is restored to the card's default on exit, including on Ctrl+C.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SECS=${SECS:-90}
LIMITS=${LIMITS:-"240 255 270 285"}

DEFAULT=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | cut -d. -f1)
echo "default power limit: ${DEFAULT} W"
restore() { echo "restoring ${DEFAULT} W"; sudo nvidia-smi -pl "$DEFAULT" >/dev/null 2>&1; }
trap restore EXIT INT TERM

for pl in $LIMITS; do
    echo "=================== power limit ${pl} W ==================="
    if ! sudo nvidia-smi -pl "$pl" >/dev/null; then
        echo "could not set ${pl} W -- skipping"; continue
    fi
    OUT_DIR=/tmp/mxbm-power "$ROOT/benchmarks/power_bench.sh" "$SECS" "pl$pl" -- --solver cuda \
        | grep -E "POWER|power |sm clock|EFFICIENCY|sol/s   \(|ms/solve"
done

echo
echo "reference on this card: 53.27 sol/s at 238.7 W = 0.223 sol/s/W"
