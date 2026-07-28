#!/usr/bin/env bash
# Packed record against the 24 B QUAD record, at several power caps.
#
# WHY THIS EXISTS. The quad record (MXBM_R3_QUAD) moves 26 % less DRAM traffic and, at
# stock, buys exactly ZERO clock: 2700/2692 MHz against 2700/2715 at the same 284 W. The
# arithmetic that rebuilds the work words costs what the traffic it replaces costs. So at
# 285 W it is a pure footprint trade -- -29 % memory for +14 % time -- and nothing else.
#
# The open question is whether that survives a tight cap, and there is one specific reason
# it might not: THE MEMORY CLOCK DOES NOT SCALE WITH A CORE POWER CAP. At 180 W the card
# runs 1815 MHz core against 2610 at stock, while the memory clock sits at 10251 MHz in
# both -- so DRAM is a much larger share of a tight budget than a loose one. That is the
# mechanism behind the measured 5x steepening of the byte->clock exchange rate
# (docs/performance.md, "under a low cap the same bytes cost 5x as much clock").
#
# THE BAR IS HIGH AND IS STATED BEFORE THE RUN, so this cannot be read as a success
# afterwards. At 180 W the quad record's arithmetic costs ~7.2 ms (its +4.85 ms at stock,
# rescaled by 2700/1815). To break even it needs the same in clock, which is about
# +358 MHz -- starting from the +0 MHz it manages at stock. The honest prior is that it
# LOSES, and the point of running it is to close the question, not to win it.
#
# METHOD, and each of these is a way it could have been wrong:
#  * Every point is packed / quad / packed, so the two packed runs BRACKET the quad one
#    in time and their spread is the drift the delta has to beat.
#  * Solve counts are calibrated per cap from a probe, not from stock-clock constants --
#    a solve takes 43.6 ms at 180 W against 33.8 at stock, and a fixed count would make
#    the low-cap runs a third shorter and their medians noisier.
#  * The first third of every telemetry stream is dropped: a solve run begins with a
#    7.5 GiB allocation that is not representative of anything.
#  * Both binaries come from ONE source tree and differ only in -DMXBM_R3_QUAD.
#  * Drops are printed for both. A nonzero count voids that row; the quad record is a
#    correctness-preserving change and is gated on the KAT, unlike the ablation builds.
#
#   benchmarks/quad_cap.sh                        # stock only, no root
#   sudo -v && CAPS="285 220 180" benchmarks/quad_cap.sh
#   CAPS=180 REPEATS=2 benchmarks/quad_cap.sh
set -uf

ROOT=$(cd "$(dirname "$0")/.." && pwd)
CAPS=${CAPS:-stock}
SECS=${SECS:-30}            # target seconds of SOLVING per run, after calibration
REPEATS=${REPEATS:-1}
SETTLE=${SETTLE:-8}
OUT=${OUT_DIR:-/tmp/mxbm-quadcap}
NVCC=${NVCC:-nvcc}
ARCH=${ARCH:-sm_89}
mkdir -p "$OUT"
: > "$OUT/rows.txt"

build() {   # build <suffix> <extra-defines...>
    local sfx=$1 out="$OUT/pipeline$1"; shift
    [ -x "$out" ] && return 0
    echo "  building $(basename "$out") ..."
    "$NVCC" -O3 -arch="$ARCH" -std=c++17 -diag-suppress 186 \
        -I "$ROOT/src" -I "$ROOT/kernels/cuda" -I "$ROOT/tests" -I "$ROOT/third_party/blake2b" \
        "$@" \
        "$ROOT/cuda/pipeline.cu" "$ROOT/src/beamhash/bh3_blake2b.cpp" \
        "$ROOT/src/beamhash/bh3_verify.cpp" "$ROOT/third_party/blake2b/blake2b-ref.c" \
        -o "$out" 2> "$OUT/build$sfx.log" \
      || { echo "build failed, see $OUT/build$sfx.log"; tail -5 "$OUT/build$sfx.log"; exit 1; }
}
echo "building (cached in $OUT) ..."
build ".packed"
build ".quad" -DMXBM_R3_QUAD=1

BIG=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null \
      | awk '{ if ($1+0 > 256) print }')
[ -n "$BIG" ] && { echo "ABORT: another process holds >256 MiB on the card"; exit 1; }

DEFAULT=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | cut -d. -f1)
NEEDS_ROOT=no
for c in $CAPS; do [ "$c" != "stock" ] && [ "$c" != "$DEFAULT" ] && NEEDS_ROOT=yes; done

KEEPALIVE=""
if [ "$NEEDS_ROOT" = yes ]; then
    sudo -v || { echo "need root to set a power limit -- aborting before measuring"; exit 1; }
    ( while kill -0 "$$" 2>/dev/null; do sudo -n true 2>/dev/null; sleep 45; done ) &
    KEEPALIVE=$!
fi
restore() {
    [ -n "$KEEPALIVE" ] && kill "$KEEPALIVE" 2>/dev/null
    [ "$NEEDS_ROOT" = yes ] && sudo -n nvidia-smi -pl "$DEFAULT" >/dev/null 2>&1
    return 0
}
trap restore EXIT INT TERM

# One run. Prints "ms sm_clock watts drops".
run_one() {   # run_one <variant> <label> <solves>
    local v=$1 label=$2 n=$3
    local csv="$OUT/$label.csv" log="$OUT/$label.log"
    nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader,nounits -lms 200 \
        > "$csv" 2>/dev/null &
    local sampler=$!
    "$OUT/pipeline.$v" "$n" > "$log" 2>&1
    local rc=$?
    sleep 0.3; kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
    [ $rc -eq 0 ] || { echo "  $label FAILED (exit $rc)"; tail -3 "$log"; return 1; }
    python3 - "$csv" "$log" <<'PY'
import statistics as st, sys
ms = drops = None
for line in open(sys.argv[2]):
    if "per-solve median" in line: ms = float(line.split("per-solve median")[1].split("ms")[0])
    if line.startswith("drops"):   drops = line.split(":")[1].strip().replace(" ", "")
sm, pw = [], []
for line in open(sys.argv[1]):
    p = [x.strip() for x in line.split(",")]
    if len(p) < 2: continue
    try: s, w = float(p[0]), float(p[1])
    except ValueError: continue
    if s < 500: continue
    sm.append(s); pw.append(w)
if ms is None or not sm:
    sys.stderr.write("parse failed\n"); sys.exit(2)
b = slice(len(sm)//3, None)          # the first third is the 7.5 GiB allocation
print("%.2f %.0f %.1f %s" % (ms, st.median(sm[b]), st.median(pw[b]), drops or "?"))
PY
}

printf "%-7s %-8s %9s %8s %8s  %s\n" cap variant ms/solve sm_MHz watts drops
printf -- "------------------------------------------------------------\n"

for cap in $CAPS; do
    if [ "$cap" != "stock" ]; then
        sudo -n nvidia-smi -pl "$cap" >/dev/null 2>&1 || { echo "could not set -pl $cap"; continue; }
        sleep 3
    fi
    # Calibrate: 25 solves of the SLOWER variant sizes both runs for this cap.
    C=$(run_one quad "cal$cap" 25) || exit 1
    CAL=$(echo "$C" | awk '{print $1}')
    N=$(python3 -c "print(max(40, int($SECS*1000/$CAL)))")
    echo "  cap $cap: calibrated at ${CAL} ms/solve -> $N solves per run"

    r=0
    while [ "$r" -lt "$REPEATS" ]; do
        r=$((r + 1))
        sleep "$SETTLE"
        A=$(run_one packed "p$cap.$r.a" "$N") || exit 1
        Q=$(run_one quad   "q$cap.$r"   "$N") || exit 1
        B=$(run_one packed "p$cap.$r.b" "$N") || exit 1
        echo "$cap packed $A" >> "$OUT/rows.txt"
        echo "$cap quad   $Q" >> "$OUT/rows.txt"
        echo "$A" | awk -v c="$cap" '{ printf "%-7s %-8s %9s %8s %8s  %s\n", c, "packed",  $1,$2,$3,$4 }'
        echo "$Q" | awk -v c="$cap" '{ printf "%-7s %-8s %9s %8s %8s  %s\n", c, "quad",    $1,$2,$3,$4 }'
        echo "$B" | awk -v c="$cap" '{ printf "%-7s %-8s %9s %8s %8s  %s\n", c, "packed²", $1,$2,$3,$4 }'
        python3 - "$A" "$Q" "$B" <<'PY'
import sys
def col(s): f = s.split(); return float(f[0]), float(f[1]), float(f[2])
(am, ash, aw), (qm, qsh, qw), (bm, bsh, bw) = (col(sys.argv[i]) for i in (1, 2, 3))
pm, psh = (am + bm) / 2, (ash + bsh) / 2
dms, dsm = qm - pm, qsh - psh
drift_ms, drift_sm = abs(bm - am), abs(bsh - ash)
verdict = "quad WINS" if dms < -drift_ms else ("quad loses" if dms > drift_ms else "tie")
print("    quad %+.2f ms  %+.0f MHz   (drift %.2f ms, %.0f MHz)  -> %s"
      % (dms, dsm, drift_ms, drift_sm, verdict))
if abs(dsm) <= drift_sm:
    print("    clock delta is inside the drift -- read it as zero, which is what stock gave")
PY
    done
done

echo
echo "logs: $OUT"
