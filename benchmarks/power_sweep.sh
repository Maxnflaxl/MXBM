#!/usr/bin/env bash
# Sweep the board power limit and measure sol/s and J/sol at each point.
#
# WHY: at stock the card sits pinned at its cap in every kernel, while a miner that
# does not fill the card draws less and is NOT capped. So an efficiency comparison at
# stock compares two different operating points. This measures MXBM's own speed/power
# curve, which is what makes an equal-power comparison possible.
#
#   sudo -v && benchmarks/power_sweep.sh            # sudo is used ONLY for nvidia-smi -pl
#
# The default LIMITS is the top of the band only. A head-to-head table wants the
# whole thing:  LIMITS="100 110 120 140 160 175 180 190 200 210 220 240 255 270 285"
#
# For a single card's curve prefer `mxbm --report`, which sweeps in the miner loop,
# rebuilds the solver per point, verifies held clocks and gauges drift. This script
# exists for the cross-miner table, where both miners must be driven from outside.
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
    # The binary samples its own telemetry and reads the card's energy counter, so
    # the sol/s, clocks and J/solution below come from one instrument inside the
    # measured window rather than from a sampler bolted on outside it.
    "$ROOT/build/mxbm" --benchmark BEAM-III --benchmark-seconds "$SECS" --nocolor \
        --solver cuda 2>&1 \
        | grep -E "sol/s|ms/solve|MHz core|J total"
done
