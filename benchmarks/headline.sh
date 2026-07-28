#!/usr/bin/env bash
# Re-measure the headline (ms/solve, sol/s) under CONTROLLED conditions.
#
# WHY THIS EXISTS. Commit cad8fde recorded that the same binaries measured
# 34.15-34.20 ms early in a session and 35.49-35.72 ms hours later, same
# machine, nothing changed -- a 4 % swing, larger than most of the wins in the
# progress table. Four explanations were checked and rejected: thermal (the card
# is COOLER and clocking HIGHER in the slow regime, 54 C / 2760 MHz against
# 63 C / 2685 MHz), within-run drift, the memory clock (10251 MHz in both), and
# host CPU contention (the GPU kernel sum itself moved, 34.49 -> 35.41 ms). So
# every absolute figure in docs/performance.md is a number from one regime,
# quoted as if it were a property of the build.
#
# The fix is not a longer run. A 300 s run inside a regime is precise and still
# wrong about the scale. The fix is to state the conditions, hold them, record
# them alongside the number, and repeat -- so a future re-measure can be
# compared against this one instead of against a number with no conditions
# attached.
#
# WHAT IS HELD, and what each one is for:
#
#  * NOTHING ELSE ON THE CARD. Hard-checked, not assumed. This project has
#    already published a contaminated point: a benchmark left running by
#    accident halved both miners' throughput at 175 W and looked like a finding.
#  * THERMAL EQUILIBRIUM. A discarded warmup run first, and the temperature
#    slope over its last minute is printed -- a run that begins on a cold card
#    measures the ramp, not the card.
#  * A PINNED POWER LIMIT (root only). The card's default is already 285 W here,
#    so the stock measurement needs no root; pinning it removes any doubt about
#    what the driver had it set to.
#  * OPTIONALLY LOCKED CLOCKS (root only). This is the reproducible reference:
#    if ms/solve at a locked clock is stable across days while the stock number
#    is not, the regime is a clock/power-state effect and the locked number is
#    the one to track builds against. LGC=2600 LMC=10501 is a reasonable pin.
#  * REPEATS, SPACED. One number is a sample, not a measurement. Each run is
#    reported separately and the summary quotes the median AND the full range,
#    because the range is the quantity actually in question here.
#
# WHAT IS RECORDED ALONGSIDE. Every run carries its own NVML telemetry: SM and
# memory clock, power, temperature, and the clocks_event_reasons bitmask, which
# the original measurement never captured and which is the most likely place for
# an unexplained global slowdown to be visible.
#
#   benchmarks/headline.sh                        # stock, no root
#   RUNS=8 SECS=180 benchmarks/headline.sh
#   sudo -v && LGC=2600 LMC=10501 benchmarks/headline.sh   # locked reference
#   LONG=300 benchmarks/headline.sh               # + one long run for the sol/s basis
#
# -f disables pathname expansion: LGC/LMC default to "*" meaning "leave it to
# the driver", and an unquoted "*" would otherwise glob into the directory
# listing. Same bug this project already shipped once in compare_power.sh.
set -uf

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SECS=${SECS:-120}          # per measured run
RUNS=${RUNS:-6}
WARMUP_S=${WARMUP_S:-240}  # discarded run, to reach thermal equilibrium
SETTLE=${SETTLE:-10}       # between runs; the card falls back to idle clocks
LONG=${LONG:-0}            # optional final long run, 0 = skip
PL=${PL:-stock}            # board power limit in W, or "stock"
LGC=${LGC:-"*"}            # locked SM clock in MHz, "*" = leave to the driver
LMC=${LMC:-"*"}            # locked memory clock in MHz
OUT=${OUT_DIR:-/tmp/mxbm-headline}
mkdir -p "$OUT"

# THIS RUN's rows only. Accumulating across invocations is how a re-measurement
# ends up averaged with the thing it was meant to replace.
: > "$OUT/rows.txt"

[ -x "$ROOT/build/mxbm" ] || { echo "build/mxbm not found -- build first"; exit 1; }

# ---- preconditions ------------------------------------------------------------------
# Anything else on the card invalidates the whole run, so this is a hard stop
# rather than a warning. A few hundred MiB of desktop compositing is tolerated;
# a co-tenant with a real allocation is not.
APPS=$(nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>/dev/null)
if [ -n "$APPS" ]; then
    echo "processes currently on the GPU:"
    echo "$APPS" | sed 's/^/    /'
    BIG=$(echo "$APPS" | awk -F, '{gsub(/[^0-9]/,"",$2); if ($2+0 > 256) print}')
    if [ -n "$BIG" ]; then
        echo "ABORT: a process is holding more than 256 MiB of VRAM. Stop it first --"
        echo "a co-tenant on the card does not add noise, it moves the number."
        exit 1
    fi
fi
LOAD=$(cut -d' ' -f1 /proc/loadavg)
echo "host load average (1 min): $LOAD"
awk -v l="$LOAD" 'BEGIN{ if (l+0 > 1.0) print "  WARNING: the host is busy; solve() verifies on the CPU" }'

DEFAULT=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | cut -d. -f1)
NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)

NEEDS_ROOT=no
[ "$PL"  != "stock" ] && NEEDS_ROOT=yes
[ "$LGC" != "*" ]     && NEEDS_ROOT=yes
[ "$LMC" != "*" ]     && NEEDS_ROOT=yes

KEEPALIVE=""
if [ "$NEEDS_ROOT" = yes ]; then
    if ! sudo -v; then
        echo "need root to pin the power limit or lock clocks -- aborting before measuring"
        exit 1
    fi
    ( while kill -0 "$$" 2>/dev/null; do sudo -n true 2>/dev/null; sleep 45; done ) &
    KEEPALIVE=$!
fi

restore() {
    [ -n "$KEEPALIVE" ] && kill "$KEEPALIVE" 2>/dev/null
    # Unconditionally, and on Ctrl+C: a card left locked outlives this script and
    # silently caps whatever runs next.
    if [ "$NEEDS_ROOT" = yes ]; then
        sudo -n nvidia-smi -rgc >/dev/null 2>&1
        sudo -n nvidia-smi -rmc >/dev/null 2>&1
        [ "$PL" != "stock" ] && sudo -n nvidia-smi -pl "$DEFAULT" >/dev/null 2>&1
    fi
}
trap restore EXIT INT TERM

if [ "$PL" != "stock" ]; then
    sudo -n nvidia-smi -pl "$PL" >/dev/null 2>&1 || { echo "could not set -pl $PL"; exit 1; }
fi
if [ "$LGC" != "*" ]; then
    sudo -n nvidia-smi -lgc "$LGC,$LGC" >/dev/null 2>&1 || { echo "could not lock SM clock to $LGC"; exit 1; }
fi
if [ "$LMC" != "*" ]; then
    sudo -n nvidia-smi -lmc "$LMC,$LMC" >/dev/null 2>&1 || { echo "could not lock memory clock to $LMC"; exit 1; }
fi

echo "$NAME   driver $DRIVER   default limit ${DEFAULT} W"
echo "conditions: pl=${PL}  lgc=${LGC}  lmc=${LMC}   ${RUNS} x ${SECS}s, warmup ${WARMUP_S}s"
echo

# ---- one run ------------------------------------------------------------------------
# Prints "median_ms p5 p95 sol_s solves per_solve  sm mem power temp  reasons"
run_one() {   # $1 = label, $2 = seconds
    local label=$1 secs=$2
    local csv="$OUT/$label.csv" log="$OUT/$label.log"

    nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,clocks_event_reasons.active \
               --format=csv,noheader,nounits -lms 250 > "$csv" 2>/dev/null &
    local sampler=$!

    "$ROOT/build/mxbm" --benchmark BEAM-III --benchmark-seconds "$secs" \
                       --nocolor --solver cuda > "$log" 2>&1
    local rc=$?

    sleep 0.4
    kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
    [ $rc -eq 0 ] || { echo "  run $label FAILED (exit $rc), see $log"; return 1; }

    python3 - "$csv" "$log" "$label" <<'PY'
import re, statistics as st, sys

csv, log, label = sys.argv[1], sys.argv[2], sys.argv[3]

# The bits nvidia-smi packs into clocks_event_reasons.active. GpuIdle is dropped
# from the report: it is set whenever the card is between runs and says nothing
# about the measurement.
BITS = [(0x0004, "sw_power_cap"), (0x0008, "hw_slowdown"), (0x0010, "sync_boost"),
        (0x0020, "sw_thermal"),   (0x0040, "hw_thermal"),  (0x0080, "hw_power_brake"),
        (0x0002, "app_clocks"),   (0x0100, "display_clk")]

temp, sm, mem, pwr, reasons = [], [], [], [], {}
for line in open(csv):
    p = [x.strip() for x in line.split(',')]
    if len(p) < 5:
        continue
    try:
        t, s, m, w = float(p[0]), float(p[1]), float(p[2]), float(p[3])
        r = int(p[4], 16)
    except ValueError:
        continue
    # Idle samples: the card between runs, or the sampler outliving the miner.
    # Including them drags every median toward the idle state.
    if s < 500:
        continue
    temp.append(t); sm.append(s); mem.append(m); pwr.append(w)
    for bit, nm in BITS:
        if r & bit:
            reasons[nm] = reasons.get(nm, 0) + 1

txt = open(log).read()
def grab(pat, what):
    m = re.search(pat, txt)
    if not m:
        sys.stderr.write("could not parse %s out of %s\n" % (what, log))
        sys.exit(2)
    return m

g = grab(r'([\d.]+) sol/s\s+\(([\d.]+) verified solutions/solve, ([\d.]+) solves/s\)', 'sol/s')
sol_s, per_solve = float(g.group(1)), float(g.group(2))
g2 = grab(r'([\d.]+) ms/solve median\s+\(p5 ([\d.]+), p95 ([\d.]+)\)', 'ms/solve')
med, p5, p95 = float(g2.group(1)), float(g2.group(2)), float(g2.group(3))
g3 = grab(r'Benchmark: (\d+) solves in ([\d.]+)s', 'solve count')
solves = int(g3.group(1))

if not sm:
    sys.stderr.write("no active telemetry samples for %s -- sampler died?\n" % label)
    sys.exit(2)

n = len(sm)
# Temperature slope over the run, C/min: a run that is still heating is a run
# measured on a moving card.
slope = (st.fmean(temp[-max(4, n//5):]) - st.fmean(temp[:max(4, n//5)])) / (n * 0.25 / 60.0)
rs = ",".join("%s:%d%%" % (k, round(100.0 * v / n)) for k, v in sorted(reasons.items())) or "none"

print("%s %.2f %.2f %.2f %.2f %d %.3f %.0f %.0f %.1f %.0f %+.1f %s" % (
    label, med, p5, p95, sol_s, solves, per_solve,
    st.median(sm), st.median(mem), st.median(pwr), max(temp), slope, rs))
PY
}

# ---- warmup ---------------------------------------------------------------------------
# Discarded. Its only outputs are a hot card and the slope, which says whether
# the card was still climbing when the measured runs began.
if [ "$WARMUP_S" -gt 0 ]; then
    echo "warmup ${WARMUP_S}s (discarded) ..."
    W=$(run_one warmup "$WARMUP_S") || exit 1
    echo "$W" | awk '{ printf "  end of warmup: %s C peak, %s MHz sm, %s W, slope %s C/min\n", $11, $8, $10, $12 }'
    echo
fi

printf "%-9s %8s %8s %8s %8s %7s %6s %6s %6s %5s\n" \
       run ms/solve p5 p95 sol/s solves sm mem W temp
printf -- "----------------------------------------------------------------------------------\n"

i=0
while [ "$i" -lt "$RUNS" ]; do
    i=$((i + 1))
    sleep "$SETTLE"
    R=$(run_one "run$i" "$SECS") || exit 1
    echo "$R" >> "$OUT/rows.txt"
    echo "$R" | awk '{ printf "%-9s %8s %8s %8s %8s %7s %6s %6s %6s %5s   %s\n", \
                        $1, $2, $3, $4, $5, $6, $8, $9, $10, $11, $13 }'
done

if [ "$LONG" -gt 0 ]; then
    sleep "$SETTLE"
    echo
    echo "long run (${LONG}s) -- the sol/s basis; solutions/solve needs thousands of solves"
    R=$(run_one long "$LONG") || exit 1
    echo "$R" >> "$OUT/rows.txt"
    echo "$R" | awk '{ printf "%-9s %8s %8s %8s %8s %7s %6s %6s %6s %5s   %s\n", \
                        $1, $2, $3, $4, $5, $6, $8, $9, $10, $11, $13 }'
fi

# ---- summary --------------------------------------------------------------------------
echo
python3 - "$OUT/rows.txt" "$SECS" <<'PY'
import statistics as st, sys

rows = []
bad = 0
for line in open(sys.argv[1]):
    f = line.split()
    if len(f) < 12:
        bad += 1
        continue
    rows.append(f)

# The long run is a different duration, so it is reported but kept out of the
# spread: mixing durations into one median measures the mix.
short = [r for r in rows if r[0].startswith("run")]
if not short:
    print("no rows"); sys.exit(1)

ms   = [float(r[1]) for r in short]
sols = [float(r[4]) for r in short]
smc  = [float(r[7]) for r in short]
pw   = [float(r[9]) for r in short]

def line(name, v, unit, fmt="%.2f"):
    lo, hi = min(v), max(v)
    spread = 100.0 * (hi - lo) / st.median(v)
    print(("  %-14s " + fmt + " %s   (min " + fmt + ", max " + fmt + ", spread %.1f %%)")
          % (name, st.median(v), unit, lo, hi, spread))

print("%d runs of %ss" % (len(short), sys.argv[2]))
line("ms/solve", ms, "ms")
line("sol/s", sols, "sol/s")
line("SM clock", smc, "MHz", "%.0f")
line("power", pw, "W", "%.1f")
if bad:
    print("  %d malformed rows skipped" % bad)

longr = [r for r in rows if r[0] == "long"]
if longr:
    r = longr[0]
    print("  long run: %s ms/solve, %s sol/s over %s solves at %s solutions/solve"
          % (r[1], r[4], r[5], r[6]))

# The whole point of the exercise: is the spread small enough that an absolute
# number can be quoted at all?
sp = 100.0 * (max(ms) - min(ms)) / st.median(ms)
print()
if len(short) < 2:
    print("  One run measures no spread at all. RUNS=6 or more before quoting anything.")
elif sp < 1.0:
    print("  Within-session spread is %.1f %% -- tight enough to quote an absolute number" % sp)
    print("  FOR THIS SESSION. Cross-session reproducibility needs this run repeated on")
    print("  another day; that is what cad8fde was about.")
else:
    print("  Within-session spread is %.1f %% -- an absolute number quoted from a single" % sp)
    print("  run of this build carries at least that much uncertainty.")
PY

echo
echo "logs and telemetry: $OUT"
