#!/usr/bin/env bash
# Produce a paste-ready benchmark report for docs/benchmarks.md.
#
# One command, no arguments needed. Prints a markdown block you can paste
# straight into a GitHub issue -- hardware, driver, MXBM's own figures, and the
# telemetry sampled while it ran. Nothing is uploaded; it only writes to stdout.
#
#   benchmarks/collect_report.sh [SECONDS]        (default 120)
#
# Add a power sweep with:  SWEEP=1 benchmarks/collect_report.sh
# (needs sudo for nvidia-smi -pl; the limit is restored afterwards)
set -u
SECS=${1:-120}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT_DIR:-/tmp/mxbm-report}
mkdir -p "$OUT"

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found — this report wants an NVIDIA card" >&2; exit 1; }
[ -x "$ROOT/build/mxbm" ] || { echo "build/mxbm not found — build first (see docs/building.md)" >&2; exit 1; }

# One query per field: a GPU name contains spaces, so splitting a combined line
# on whitespace mangles every field after it.
q() { nvidia-smi --query-gpu="$1" --format=csv,noheader,nounits | head -1 | sed 's/^ *//;s/ *$//'; }
GPU=$(q name); DRIVER=$(q driver_version); VRAM=$(q memory.total); PLDEF=$(q power.default_limit)

echo "collecting — this takes about $((SECS + 20))s" >&2

"$ROOT/benchmarks/power_bench.sh" "$SECS" report -- --solver cuda > "$OUT/bench.txt" 2>&1
SOL=$(grep -oE "^  [0-9.]+ sol/s" "$OUT/bench.txt" | head -1 | awk '{print $1}')
PER=$(grep -oE "\([0-9.]+ verified" "$OUT/bench.txt" | head -1 | tr -d '(' | awk '{print $1}')
MS=$(grep -oE "^  [0-9.]+ ms/solve" "$OUT/bench.txt" | head -1 | awk '{print $1}')
SOLVES=$(grep -oE "Benchmark: [0-9]+ solves" "$OUT/bench.txt" | awk '{print $2}')
PW=$(grep "^  power " "$OUT/bench.txt" | sed -E 's/.*median +([0-9.]+).*/\1/')
CLK=$(grep "^  sm clock " "$OUT/bench.txt" | sed -E 's/.*median +([0-9.]+).*/\1/')
TMP=$(grep "^  temp " "$OUT/bench.txt" | sed -E 's/.*median +([0-9.]+).*/\1/')
EFF=$(grep "EFFICIENCY" "$OUT/bench.txt" | awk '{print $2}')
JPS=$(grep "EFFICIENCY" "$OUT/bench.txt" | awk '{print $4}')
BACKEND=$(grep -oE "BeamHash III \((Cuda|OpenCL)\)" "$OUT/bench.txt" | head -1 | sed 's/.*(\(.*\))/\1/')

cat <<REPORT

--------- paste everything below into the issue ---------

### Hardware
| | |
|---|---|
| GPU | $GPU |
| VRAM | ${VRAM} MiB |
| Driver | $DRIVER |
| Default power limit | ${PLDEF} W |
| OS | $(uname -sr) |
| MXBM version | $("$ROOT/build/mxbm" --version 2>/dev/null | head -1) |
| Backend used | ${BACKEND:-unknown} |

### Result (\`mxbm --benchmark BEAM-III\`, ${SECS}s, $SOLVES solves)
| | |
|---|---|
| Throughput | **${SOL:-?} sol/s** |
| Solve time | ${MS:-?} ms median |
| Verified solutions/solve | ${PER:-?} |
| Board power | ${PW:-?} W |
| Core clock | ${CLK:-?} MHz |
| Temperature | ${TMP:-?} °C |
| Efficiency | ${EFF:-?} sol/s/W (${JPS:-?} J/sol) |
REPORT

if [ "${SWEEP:-0}" = "1" ]; then
    echo
    echo "### Power curve (\`benchmarks/power_sweep.sh\`)"
    echo "| limit | sol/s | measured W | sol/s/W |"
    echo "|---|---|---|---|"
    SECS=60 LIMITS="${LIMITS:-180 200 220 240 285}" "$ROOT/benchmarks/power_sweep.sh" 2>/dev/null \
      | awk '/power limit/{lim=$4} /sol\/s   \(/{s=$1} /EFFICIENCY/{printf "| %s W | %s | - | %s |\n", lim, s, $2}'
fi

cat <<'TAIL'

### Anything unusual?
<!-- overclock/undervolt applied, other GPU load, laptop, risers, etc.
     Leave blank if it was a clean stock run. -->

--------- end ---------
TAIL
