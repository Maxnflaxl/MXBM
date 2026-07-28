#!/usr/bin/env bash
# Measure MXBM and lolMiner on the SAME board power limits, interleaved.
#
# WHY THIS EXISTS. docs/performance.md compares MXBM's whole power curve against
# lolMiner at ONE operating point -- its uncapped draw -- because that is "how
# people actually run it". But lolMiner 1.98a has --pl too (verified in the
# installed binary's help), so it has a curve of its own that nobody has
# measured. Until it is measured, "MXBM is ahead on efficiency below 252 W" is a
# claim about a miner we only ever watched at full tilt.
#
#   benchmarks/compare_power.sh                     # stock only, no root needed
#   sudo -v && LIMITS="200 220 240 285" benchmarks/compare_power.sh
#
# MEASURED CONSTRAINT ON THE METHOD: lolMiner's --benchmark is a FIXED ~61 s run.
# It self-terminates with exit 0 -- it is not being cut short by a timeout, which
# was checked by letting it run under a 240 s ceiling and watching it stop at 61.
# So a single invocation yields ~35 s of steady state and four 15 s speed
# windows, the first of which is ramp. Both miners are therefore run for the same
# 60 s and REPEATED, rather than run once for longer.
#
# FAIRNESS, and each of these is a decision someone could get wrong:
#
#  * The cap is set EXTERNALLY, with nvidia-smi -pl, identically for both. Using
#    each miner's own --pl would make their OC implementations a variable in a
#    measurement that is supposed to be about their kernels.
#  * Power is sampled by nvidia-smi for both, never taken from either miner's
#    self-report. lolMiner's own statistics block averages in its ramp-up, which
#    reads ~214 W on a one-minute run against a steady-state figure well above
#    that -- a number that flatters whichever miner you read it from.
#  * The two miners ALTERNATE order at each point, so a thermal trend across the
#    sweep cannot land on one of them systematically.
#  * WARMUP seconds are discarded from both the power samples and the speed
#    windows, by the same rule.
#
# WHAT THIS STILL CANNOT SETTLE. Each miner's sol/s is its OWN definition --
# MXBM counts CPU-verified solutions (1.98/solve) where lolMiner's basis is
# unknown, a spread of ~17 % inside our own pipeline. So the sol/s COLUMNS are
# not directly comparable between miners, and the honest reading is each
# miner's curve against ITSELF (how gracefully does it degrade under a cap) plus
# the watts, which are measured by one instrument for both. Accepted pool shares
# remain the only cross-miner arbiter; see docs-internal/MINER_COMP.md.
# -f disables pathname expansion. Load-bearing, not tidiness: CCLKS defaults to
# "*" meaning "leave this clock alone", and `for cclk in $CCLKS` would otherwise
# glob it into the directory listing and try to lock the core clock to a value
# of "benchmarks". Every list this script iterates is unquoted by necessity, so
# globbing is switched off for all of them at once.
set -uf

ROOT=$(cd "$(dirname "$0")/.." && pwd)
LOL=${LOL:-/home/maxnflaxl/Documents/lolMiner/1.98a/lolMiner}
SECS=${SECS:-60}         # matched to lolMiner's fixed benchmark length
WARMUP=${WARMUP:-25}     # both miners are flat by ~10 s; 25 is deliberately generous
TAIL=${TAIL:-4}          # drop the wind-down, which is real on lolMiner (235 -> 227 W)
REPEATS=${REPEATS:-2}
LIMITS=${LIMITS:-stock}
CCLKS=${CCLKS:-"*"}      # locked core clock, MHz; "*" leaves it to the driver
MCLKS=${MCLKS:-"*"}      # locked memory clock, MHz
SETTLE=${SETTLE:-15}
OUT=${OUT_DIR:-/tmp/mxbm-compare}
mkdir -p "$OUT"

[ -x "$LOL" ] || { echo "lolMiner not found at $LOL (set LOL=...)"; exit 1; }
[ -x "$ROOT/build/mxbm" ] || { echo "build/mxbm not found -- build first"; exit 1; }

DEFAULT=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | cut -d. -f1)
echo "card default power limit: ${DEFAULT} W   |   ${SECS}s per run, first ${WARMUP}s discarded"

# A full sweep runs far longer than sudo's credential lifetime -- 5 points is
# ~25 minutes against a default timestamp_timeout of 5-15. Authenticate once,
# then refresh in the background for as long as this script lives, so the cap
# for point 4 is as privileged as the cap for point 1. Without this the sweep
# sets the first limit, silently loses the right to set the rest, and (before
# the guard below) skipped them one by one.
KEEPALIVE=""
NEEDS_ROOT=no
[ "$LIMITS" != "stock" ] && NEEDS_ROOT=yes
[ "$CCLKS" != "*" ] && NEEDS_ROOT=yes
[ "$MCLKS" != "*" ] && NEEDS_ROOT=yes
if [ "$NEEDS_ROOT" = yes ]; then
    if ! sudo -v; then
        echo "need root to set the power limit -- aborting before measuring anything"
        exit 1
    fi
    ( while kill -0 "$$" 2>/dev/null; do sudo -n true 2>/dev/null; sleep 45; done ) &
    KEEPALIVE=$!
fi

restore() {
    [ -n "$KEEPALIVE" ] && kill "$KEEPALIVE" 2>/dev/null
    # Always try to unlock the clocks, even on Ctrl+C and even if no clock list
    # was given: a card left locked outlives this script and silently caps
    # whatever runs next.
    sudo -n nvidia-smi -rgc >/dev/null 2>&1
    sudo -n nvidia-smi -rmc >/dev/null 2>&1
    if [ "$LIMITS" != "stock" ]; then
        echo "restoring ${DEFAULT} W"
        sudo -n nvidia-smi -pl "$DEFAULT" >/dev/null 2>&1
    fi
}
trap restore EXIT INT TERM

# One run: sample NVML underneath, return "sol/s watts sm_clock mem_clock temp".
run_one() {   # $1 = miner (mxbm|lol), $2 = label
    local who=$1 label=$2
    local csv="$OUT/$label.csv" log="$OUT/$label.log"

    nvidia-smi --query-gpu=power.draw,clocks.sm,clocks.mem,temperature.gpu \
               --format=csv,noheader,nounits -lms 200 > "$csv" &
    local sampler=$!

    if [ "$who" = "mxbm" ]; then
        "$ROOT/build/mxbm" --benchmark BEAM-III --benchmark-seconds "$SECS" \
                           --nocolor --solver cuda > "$log" 2>&1
    else
        # lolMiner's benchmark stops itself at ~61 s; the timeout is only a
        # backstop in case a future version does not.
        ( cd "$(dirname "$LOL")" && timeout $((SECS + 60)) ./"$(basename "$LOL")" \
              --benchmark BEAM-III --shortstats 15 ) > "$log" 2>&1
    fi

    sleep 0.5
    kill $sampler 2>/dev/null; wait $sampler 2>/dev/null

    python3 - "$csv" "$log" "$who" "$WARMUP" "$TAIL" <<'PY'
import re, statistics as st, sys
csv, log, who = sys.argv[1], sys.argv[2], sys.argv[3]
warm, tail = float(sys.argv[4]), float(sys.argv[5])

rows = []
for line in open(csv):
    p = [x.strip() for x in line.split(',')]
    if len(p) < 4:
        continue
    try:
        rows.append(tuple(float(x) for x in p))
    except ValueError:
        continue
body = rows[int(warm * 5):len(rows) - int(tail * 5)] or rows      # 5 samples/s
if not body:
    print("ERR no power samples"); sys.exit(0)

text = re.sub(r'\x1b\[[0-9;]*m', '', open(log, errors='replace').read())
if who == 'mxbm':
    # The benchmark's own summary line, the CPU-verified figure.
    m = [float(l.split('sol/s')[0].split()[-1])
         for l in text.splitlines() if 'sol/s' in l and 'verified' in l]
    sol = m[-1] if m else None
else:
    # Every "Average speed (15s): N sol/s" window past the warmup, medianed.
    w = [float(x) for x in re.findall(r'Average speed \(\d+s\): ([\d.]+) sol/s', text)]
    keep = w[int(warm // 15):] or w
    sol = st.median(keep) if keep else None

col = lambda i: [r[i] for r in body]
print("%s %.1f %.0f %.0f %.0f %d" % (
    ("%.2f" % sol) if sol else "ERR",
    st.median(col(0)), st.median(col(1)), st.median(col(2)), st.median(col(3)), len(body)))
PY
}

printf '\n%-14s %-9s %8s %8s %9s %8s %7s %s\n' \
       "POINT" MINER "sol/s" "watts" "sol/s/W" "SMclk" "temp" "samples"
printf '%s\n' "-----------------------------------------------------------------------------------"

flip=0
for pl in $LIMITS; do
 for cclk in $CCLKS; do
  for mclk in $MCLKS; do
    # Clock locks are applied with nvidia-smi for the same reason the cap is:
    # one instrument, applied identically to both miners, so neither miner's own
    # overclock code is a variable in a measurement about their kernels.
    # A clock that could not be locked SKIPS the point. Measuring at the
    # driver's own clock while labelling the row with a lock that never
    # applied is the same mislabelling the power cap is guarded against, and
    # it is worse here because the number would look plausible.
    lock_ok=yes
    if [ "$cclk" != "*" ]; then
        sudo -n nvidia-smi -lgc "$cclk,$cclk" >/dev/null 2>&1 \
            || { echo "!! could not lock core clock to ${cclk} MHz - skipping this point"
                 lock_ok=no; }
    fi
    if [ "$lock_ok" = yes ] && [ "$mclk" != "*" ]; then
        sudo -n nvidia-smi -lmc "$mclk,$mclk" >/dev/null 2>&1 \
            || { echo "!! could not lock memory clock to ${mclk} MHz - skipping this point"
                 lock_ok=no; }
    fi
    if [ "$lock_ok" = no ]; then
        sudo -n nvidia-smi -rgc >/dev/null 2>&1
        sudo -n nvidia-smi -rmc >/dev/null 2>&1
        continue
    fi
    point="$pl"
    [ "$cclk" != "*" ] && point="$point/c$cclk"
    [ "$mclk" != "*" ] && point="$point/m$mclk"

    if [ "$pl" != "stock" ]; then
        # Distinguish "lost root" from "the card rejected this value": the first
        # dooms every remaining point and must stop the sweep, the second is
        # local to one value. Reporting a run at a cap that was not applied would
        # be worse than either -- it would label stock numbers as capped ones.
        if ! sudo -n true 2>/dev/null; then
            echo "!! lost root before ${pl} W -- the keepalive died. Aborting rather than"
            echo "   measuring at a limit that was never applied. Points done so far stand."
            break
        fi
        if ! sudo -n nvidia-smi -pl "$pl" >/dev/null 2>&1; then
            echo "!! the card rejected ${pl} W (outside its permitted band?) -- skipping."
            continue
        fi
        applied=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | cut -d. -f1)
        [ "$applied" = "$pl" ] || echo "   note: asked for ${pl} W, card reports ${applied} W"
        sleep 3
    fi

    # Alternate which miner goes first, so a drifting machine does not bias one.
    if [ $((flip % 2)) -eq 0 ]; then order="mxbm lol"; else order="lol mxbm"; fi
    flip=$((flip + 1))

    for rep in $(seq 1 "$REPEATS"); do
        for who in $order; do
            sleep "$SETTLE"                 # let the card fall back to idle first
            read -r sol watt smclk memclk temp n <<<"$(run_one "$who" "${point//\//_}_${who}_r${rep}")"
            if [ "$sol" = "ERR" ]; then
                printf '%-14s %-9s %8s %8s %9s %8s %7s %s\n' \
                       "$point" "$who#$rep" ERR "$watt" - "$smclk" "$temp" "$n"
            else
                eff=$(python3 -c "print('%.4f' % ($sol/$watt))" 2>/dev/null || echo -)
                printf '%-14s %-9s %8s %8s %9s %8s %7s %s\n' \
                       "$point" "$who#$rep" "$sol" "$watt" "$eff" "$smclk" "$temp" "$n"
                echo "$point $who $sol $watt" >> "$OUT/rows.txt"
            fi
        done
        # Reverse the order for the next repeat as well, so within a point the
        # two miners see the same distribution of "went first" and "went second".
        if [ "$order" = "mxbm lol" ]; then order="lol mxbm"; else order="mxbm lol"; fi
    done
    # Unlock between grid points, so the next one starts from the driver's own
    # management rather than inheriting the last lock.
    [ "$cclk" != "*" ] && sudo -n nvidia-smi -rgc >/dev/null 2>&1
    [ "$mclk" != "*" ] && sudo -n nvidia-smi -rmc >/dev/null 2>&1
  done
 done
done

# Medians across repeats -- the figure to quote, since a single 60 s run of
# either miner moves by more than the difference being measured.
if [ -s "$OUT/rows.txt" ]; then
    echo
    printf '%-14s %-9s %8s %8s %9s   %s\n' "POINT" MINER "sol/s" "watts" "sol/s/W" "(median of repeats)"
    printf '%s\n' "-----------------------------------------------------------------------------------"
    python3 - "$OUT/rows.txt" <<'PY'
import statistics as st, sys
from collections import OrderedDict
g = OrderedDict()
for line in open(sys.argv[1]):
    pl, who, sol, watt = line.split()
    g.setdefault((pl, who), []).append((float(sol), float(watt)))
for (pl, who), v in g.items():
    s, w = st.median([x[0] for x in v]), st.median([x[1] for x in v])
    print("%-14s %-9s %8.2f %8.1f %9.4f   n=%d" % (pl, who, s, w, s / w, len(v)))
PY
fi

echo
echo "sol/s is each miner's OWN definition and the two are not directly comparable;"
echo "watts are one instrument for both. Raw logs and samples in $OUT."
