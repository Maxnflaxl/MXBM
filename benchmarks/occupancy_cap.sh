#!/usr/bin/env bash
# Does the OCCUPANCY optimum move under a power cap?
#
# WHY. Every occupancy decision in this project was tuned at stock, where the reasoning
# is "more resident warps hide more latency". MXBM_R1_FCAP=288 exists because it fits
# round 1 five blocks/SM where 320 fits four, and docs/performance.md records it as worth
# 0.15 ms of r1's 5.28.
#
# Under a power cap that reasoning is incomplete: resident warps are not free. More
# concurrent work is more instantaneous power, and power spent on occupancy is power not
# spent on clock. Measured at stock, one block/SM on round 1 is worth 23 MHz -- so the
# mechanism is real; it simply loses to the latency it costs. At 180 W the exchange rate
# between activity and frequency is ~5x steeper (measured: the same DRAM traffic cut buys
# 60 MHz at 285 W and 210 at 180), which is where the ordering could flip.
#
# THE PREDICTION, WRITTEN DOWN BEFORE THE RUN so this cannot be read as a success
# afterwards. At stock, 288 (5 blocks) beats 320 (4 blocks) by 0.17 ms per r1 pass while
# clocking 23 MHz lower. Scaling that 23 MHz by the measured 5x gives ~115 MHz at 180 W,
# 6.3 % of 1815, against a latency cost that grows only as the clock falls (x1.44). That
# is roughly -0.23 ms per pass in favour of 320 -- a FLIP. If 288 still wins at 180 W by
# about the 0.24 ms the scaling predicts for it, the idea is dead and should be dropped
# rather than swept further.
#
# METHOD. Round 1 is replayed 9x (MXBM_ROUND_REPS=1:9) so a 0.17 ms per-pass difference
# becomes ~1.5 ms of solve -- well clear of the 0.1 ms run-to-run spread. Variants
# alternate, and the whole A/B is repeated, so a thermal trend cannot land on one of them.
# Both binaries come from one tree and differ only in -DMXBM_R1_FCAP.
#
#   benchmarks/occupancy_cap.sh                       # stock, no root
#   sudo -v && CAPS="285 180" benchmarks/occupancy_cap.sh
set -uf

ROOT=$(cd "$(dirname "$0")/.." && pwd)
. "$ROOT/benchmarks/lib.sh"
CAPS=${CAPS:-stock}
FCAPS=${FCAPS:-"288 320"}
REPS=${REPS:-9}          # round-1 replays, to amplify a sub-ms difference
SOLVES=${SOLVES:-400}
REPEATS=${REPEATS:-2}
SETTLE=${SETTLE:-8}
OUT=${OUT_DIR:-/tmp/mxbm-occ}
NVCC=${NVCC:-nvcc}
ARCH=${ARCH:-sm_89}
mkdir -p "$OUT"

echo "building (cached in $OUT) ..."
for f in $FCAPS; do
    out="$OUT/pipeline.f$f"
    bench_stale "$out" || continue
    echo "  building r1 FCAP=$f ..."
    "$NVCC" -O3 -arch="$ARCH" -std=c++17 -diag-suppress 186 -DMXBM_R1_FCAP="$f" \
        -I "$ROOT/src" -I "$ROOT/kernels/cuda" -I "$ROOT/tests" -I "$ROOT/third_party/blake2b" \
        "$ROOT/cuda/pipeline.cu" "$ROOT/src/beamhash/bh3_blake2b.cpp" \
        "$ROOT/src/beamhash/bh3_verify.cpp" "$ROOT/third_party/blake2b/blake2b-ref.c" \
        -o "$out" 2> "$OUT/build.f$f.log" \
      || { echo "build failed, see $OUT/build.f$f.log"; tail -5 "$OUT/build.f$f.log"; exit 1; }
done

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

run_one() {   # run_one <fcap> <label>
    local f=$1 label=$2
    nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader,nounits -lms 200 \
        > "$OUT/$label.csv" 2>/dev/null &
    local s=$!
    MXBM_ROUND_REPS="1:$REPS" "$OUT/pipeline.f$f" "$SOLVES" > "$OUT/$label.log" 2>&1
    local rc=$?
    sleep 0.3; kill $s 2>/dev/null; wait $s 2>/dev/null
    [ $rc -eq 0 ] || { echo "  $label FAILED"; tail -3 "$OUT/$label.log"; return 1; }
    python3 - "$OUT/$label.csv" "$OUT/$label.log" <<'PY'
import statistics as st, sys
ms = drops = None; blocks = "?"
for line in open(sys.argv[2]):
    if "per-solve median" in line: ms = float(line.split("per-solve median")[1].split("ms")[0])
    if line.strip().startswith("r1 "):  blocks = line.split("blocks/SM=")[1].split()[0]
    if line.startswith("drops"):        drops = line.split(":")[1].strip().replace(" ", "")
sm, pw = [], []
for line in open(sys.argv[1]):
    p = [x.strip() for x in line.split(",")]
    if len(p) < 2: continue
    try: s, w = float(p[0]), float(p[1])
    except ValueError: continue
    if s < 500: continue
    sm.append(s); pw.append(w)
b = slice(len(sm)//3, None)
print("%.2f %s %.0f %.1f %s" % (ms, blocks, st.median(sm[b]), st.median(pw[b]), drops or "?"))
PY
}

printf "%-7s %-7s %9s %8s %8s %8s  %s\n" cap FCAP ms/solve blocks/SM sm_MHz watts drops
printf -- "----------------------------------------------------------------------\n"
for cap in $CAPS; do
    if [ "$cap" != "stock" ]; then
        sudo -n nvidia-smi -pl "$cap" >/dev/null 2>&1 \
            || { echo "ABORT: could not set -pl $cap (no cached sudo credentials?)."; \
                 echo "  Skipping it would drop the point silently and leave the"; \
                 echo "  remaining rows looking like a complete sweep."; exit 1; }
        sleep 3
    fi
    r=0
    while [ "$r" -lt "$REPEATS" ]; do
        r=$((r + 1))
        for f in $FCAPS; do
            sleep "$SETTLE"
            R=$(run_one "$f" "c$cap.f$f.$r") || exit 1
            echo "$cap $f $R" >> "$OUT/rows.txt"
            echo "$R" | awk -v c="$cap" -v f="$f" \
                '{ printf "%-7s %-7s %9s %8s %8s %8s  %s\n", c, f, $1, $2, $3, $4, $5 }'
        done
    done
    python3 - "$OUT/rows.txt" "$cap" "$REPS" <<'PY'
import sys, statistics as st
rows = [l.split() for l in open(sys.argv[1]) if l.split()[0] == sys.argv[2]]
reps = int(sys.argv[3])
by = {}
for r in rows: by.setdefault(r[1], []).append((float(r[2]), float(r[4])))
ks = sorted(by, key=int)
if len(ks) == 2:
    a, b = ks
    dm = st.median([x[0] for x in by[b]]) - st.median([x[0] for x in by[a]])
    ds = st.median([x[1] for x in by[b]]) - st.median([x[1] for x in by[a]])
    print("    FCAP %s vs %s at %s: %+.2f ms over %d replays (%+.3f ms/pass), %+.0f MHz"
          % (b, a, sys.argv[2], dm, reps, dm / reps, ds))
    print("    -> %s wins" % (b if dm < 0 else a))
PY
done
echo
echo "logs: $OUT"
