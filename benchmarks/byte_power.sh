#!/usr/bin/env bash
# What does a byte of DRAM traffic cost in WATTS, and therefore in CLOCK?
#
# WHY THIS EXISTS. docs/performance.md currently holds two findings that point
# opposite ways, and the second is labelled a suspicion:
#
#   "Bytes are nearly free"  -- measured. Narrowing round 2's record from 72 B to
#   16 B buys 3.47 ms of 34, and those marginal bytes move at 1194 GB/s against a
#   672 GB/s peak, which is only possible if the round was never waiting on them.
#   Footprint was filed under REACH on the strength of this.
#
#   "The suspect is DRAM traffic"  -- NOT measured. The head-to-head sweep found
#   MXBM clocking 60-570 MHz below the reference miner at every equal power cap,
#   and blamed the 13.0 GB/solve we move against the 4 GB variant it selects.
#
# Both can be true: bytes are free when the limit is latency and expensive when
# the limit is watts, because a power-capped card pays for every DRAM access out
# of the same budget that buys clock. But "can be true" is not a measurement, and
# the difference decides whether streaming / in-place layer reuse is the top
# performance item or a card-compatibility item.
#
# THE MEASUREMENT. Exactly the ablation the speed attribution used --
# MXBM_ABL_EMIT=R narrows round R's scattered payload store to 16 B while keeping
# combine, apply_mix and the ctree build alive, so the only variable is bytes --
# replayed with MXBM_ROUND_REPS=R:N so that round dominates the timeline. Then
# read the card, not the clock in the miner:
#
#   * At a power CAP the board watts are pinned by definition, so the free
#     variable is the SM CLOCK. If dropping 56 B/element lets the card hold a
#     higher clock at the same watts, bytes are watts.
#   * Run it at a LOW cap as well as at stock. At 285 W the clock has little room
#     to move; the sweep's clock gap widened from 60 MHz at stock to 570 MHz at
#     180 W, so that is where the effect, if real, is legible.
#
# THE POSITIVE CONTROL IS NOT OPTIONAL. Rounds 1 and 4 already store 16 B, so
# ablating them changes nothing about the traffic and their delta must come out
# at zero. If r1 shows a watt or clock delta, this script is measuring its own
# machinery and every other row is void. r1 is therefore run by default.
#
#   benchmarks/byte_power.sh                       # stock, rounds 1 (control) + 2
#   ROUNDS="1 2 3" REPS=8 benchmarks/byte_power.sh
#   sudo -v && PL=180 benchmarks/byte_power.sh      # where the effect should be largest
#
# THE ABLATED BUILDS PRODUCE WRONG RESULTS BY DESIGN. They are not miners and
# must never be used to quote sol/s. Only the full build's drop counters mean
# anything, and this script checks those.
set -uf

ROOT=$(cd "$(dirname "$0")/.." && pwd)
. "$ROOT/benchmarks/lib.sh"
ROUNDS=${ROUNDS:-"1 2"}     # 1 is the control; 2 is the round with the bytes
REPS=${REPS:-8}             # replays of the round inside one solve
SECS=${SECS:-40}            # per run, after which the sampled median is steady state
PL=${PL:-stock}
OUT=${OUT_DIR:-/tmp/mxbm-bytepower}
mkdir -p "$OUT"
: > "$OUT/rows.txt"

NVCC=${NVCC:-nvcc}
ARCH=${ARCH:-sm_89}

# ---- build, BEFORE anything is measured ------------------------------------------------
# nvcc on this file is minutes of full-core compile. Compiling while a run is in
# flight would contaminate it through the host verify, which is the same class of
# mistake as leaving a second miner on the card.
build() {   # build <suffix> <extra-defines...>
    # The suffix is needed again after the shift, for the log path, so it is
    # captured rather than read back out of $1 -- which under `set -u` is an
    # unbound variable once the last argument has been shifted away.
    local sfx=$1 out="$OUT/pipeline$1"; shift
    bench_stale "$out" || return 0
    echo "  building $(basename "$out") ..."
    "$NVCC" -O3 -arch="$ARCH" -std=c++17 -diag-suppress 186 \
        -I "$ROOT/src" -I "$ROOT/kernels/cuda" -I "$ROOT/tests" -I "$ROOT/third_party/blake2b" \
        "$@" \
        "$ROOT/cuda/pipeline.cu" "$ROOT/src/beamhash/bh3_blake2b.cpp" \
        "$ROOT/src/beamhash/bh3_verify.cpp" "$ROOT/third_party/blake2b/blake2b-ref.c" \
        -o "$out" 2> "$OUT/build$sfx.log" \
      || { echo "build failed, see $OUT/build$sfx.log"; tail -5 "$OUT/build$sfx.log"; exit 1; }
}

echo "building (once; cached in $OUT) ..."
build ""
for r in $ROUNDS; do build ".abl$r" -DMXBM_ABL_EMIT="$r"; done

# ---- conditions ------------------------------------------------------------------------
APPS=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null \
       | awk '{ if ($1+0 > 256) print }')
[ -n "$APPS" ] && { echo "ABORT: another process holds >256 MiB on the card"; exit 1; }

DEFAULT=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | cut -d. -f1)
KEEPALIVE=""
if [ "$PL" != "stock" ]; then
    sudo -v || { echo "need root for -pl"; exit 1; }
    ( while kill -0 "$$" 2>/dev/null; do sudo -n true 2>/dev/null; sleep 45; done ) &
    KEEPALIVE=$!
    sudo -n nvidia-smi -pl "$PL" >/dev/null 2>&1 || { echo "could not set -pl $PL"; exit 1; }
fi
restore() {
    [ -n "$KEEPALIVE" ] && kill "$KEEPALIVE" 2>/dev/null
    [ "$PL" != "stock" ] && sudo -n nvidia-smi -pl "$DEFAULT" >/dev/null 2>&1
}
trap restore EXIT INT TERM

echo "cap: ${PL}   reps: ${REPS}   ${SECS}s per run"
echo

# ---- one run ---------------------------------------------------------------------------
# Prints "ms sm_clock power temp drops"
run_one() {   # run_one <binary> <label> <ms-estimate> <env...>
    local bin=$1 label=$2; shift 2
    local csv="$OUT/$label.csv" log="$OUT/$label.log"

    # Sized from the measured per-solve cost so every run lasts SECS regardless of
    # how many replays it carries -- a short run reports its own ramp.
    local ms=$1; shift
    local solves
    solves=$(python3 -c "print(max(12,int($SECS*1000/$ms)))")

    nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu \
               --format=csv,noheader,nounits -lms 200 > "$csv" 2>/dev/null &
    local sampler=$!
    env "$@" "$bin" "$solves" > "$log" 2>&1
    local rc=$?
    sleep 0.3; kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
    [ $rc -eq 0 ] || { echo "  $label FAILED (exit $rc)"; tail -3 "$log"; return 1; }

    python3 - "$csv" "$log" <<'PY'
import statistics as st, sys
csv, log = sys.argv[1], sys.argv[2]
ms = drops = None
for line in open(log):
    if line.startswith("end-to-end"): ms = float(line.split(":")[1].split("ms")[0])
    if line.startswith("drops"):      drops = line.split(":")[1].strip().replace(" ", "")
sm, pw, tp = [], [], []
for line in open(csv):
    p = [x.strip() for x in line.split(",")]
    if len(p) < 3: continue
    try: s, w, t = float(p[0]), float(p[1]), float(p[2])
    except ValueError: continue
    if s < 500: continue          # idle samples: the card before and after the run
    sm.append(s); pw.append(w); tp.append(t)
if ms is None or not sm:
    sys.stderr.write("could not parse %s / no active samples\n" % log); sys.exit(2)
body = slice(len(sm)//5, None)    # drop the ramp
print("%.2f %.0f %.1f %.0f %s" % (ms, st.median(sm[body]), st.median(pw[body]),
                                  max(tp), drops or "?"))
PY
}

printf "%-12s %9s %8s %8s %6s  %s\n" variant ms/solve sm_MHz watts temp drops
printf -- "---------------------------------------------------------------\n"

# Baseline once: the unablated build with no replay. Its measured ms is also what
# sizes every later run, because the per-round costs below were taken at stock and
# a capped card is slower in proportion -- at 180 W a solve is 43.6 ms against
# 33.8, so sizing from the stock constants overshoots every run by a third.
B=$(run_one "$OUT/pipeline" base 35 MXBM_NONE=1) || exit 1
echo "$B" | awk '{ printf "%-12s %9s %8s %8s %6s  %s\n", "baseline", $1, $2, $3, $4, $5 }'
BASE_MS=$(echo "$B" | awk '{ print $1 }')
SCALE=$(python3 -c "print(max(0.5, $BASE_MS / 33.8))")

# Per round: full-with-replay against narrowed-with-replay, ALTERNATED, so a
# thermal trend across the pair cannot land on one of them.
for r in $ROUNDS; do
    echo
    # A replayed round dominates the solve, so the sampled clock and watts are
    # very nearly "the card running that round".
    case $r in 1) rms=5.8 ;; 2) rms=10.5 ;; 3) rms=10.0 ;; 4) rms=5.7 ;; *) rms=8 ;; esac
    est=$(python3 -c "print($BASE_MS + ($REPS-1)*$rms*$SCALE)")

    F=$(run_one "$OUT/pipeline"      "r$r.full"   "$est" MXBM_ROUND_REPS="$r:$REPS") || exit 1
    N=$(run_one "$OUT/pipeline.abl$r" "r$r.narrow" "$est" MXBM_ROUND_REPS="$r:$REPS") || exit 1
    F2=$(run_one "$OUT/pipeline"     "r$r.full2"  "$est" MXBM_ROUND_REPS="$r:$REPS") || exit 1

    echo "r$r $F"  >> "$OUT/rows.txt"
    echo "r$r $N"  >> "$OUT/rows.txt"
    echo "$F"  | awk -v r="$r" '{ printf "%-12s %9s %8s %8s %6s  %s\n", "r"r" full",   $1,$2,$3,$4,$5 }'
    echo "$N"  | awk -v r="$r" '{ printf "%-12s %9s %8s %8s %6s  %s\n", "r"r" 16 B",   $1,$2,$3,$4,$5 }'
    echo "$F2" | awk -v r="$r" '{ printf "%-12s %9s %8s %8s %6s  %s\n", "r"r" full²",  $1,$2,$3,$4,$5 }'

    python3 - "$r" "$F" "$N" "$F2" "$REPS" "$rms" <<'PY'
import sys
r, reps, rms = sys.argv[1], int(sys.argv[5]), float(sys.argv[6])
def col(s):
    f = s.split()
    return float(f[0]), float(f[1]), float(f[2])
(fm, fs, fw), (nm, ns, nw), (f2m, f2s, f2w) = col(sys.argv[2]), col(sys.argv[3]), col(sys.argv[4])
# The two full runs bracket the narrow one in time, so their spread IS the drift
# this comparison has to beat. A delta smaller than it is not a result.
drift_ms, drift_sm = abs(f2m - fm), abs(f2s - fs)
fm_, fs_, fw_ = (fm + f2m) / 2, (fs + f2s) / 2, (fw + f2w) / 2
print("  round %s, %d replays: full %.2f ms / %.0f MHz / %.1f W   narrow %.2f / %.0f / %.1f"
      % (r, reps, fm_, fs_, fw_, nm, ns, nw))
print("  bytes cost:  %+.2f ms   %+.0f MHz   %+.1f W        (drift between the two full "
      "runs: %.2f ms, %.0f MHz)" % (fm_ - nm, fs_ - ns, fw_ - nw, drift_ms, drift_sm))
if abs(fs_ - ns) <= drift_sm:
    print("  ^ the clock delta does not clear the drift between the two full runs. "
          "Read it as zero,")
    print("    or lengthen SECS and repeat. For r1 -- the control -- zero is the "
          "expected answer.")
PY
done

echo
echo "logs: $OUT"
echo "Reminder: the .abl builds are intentionally WRONG. Never quote sol/s from them."
