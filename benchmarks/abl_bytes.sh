#!/usr/bin/env bash
# How many DRAM bytes does an MXBM_ABL_EMIT ablation actually remove?
#
# WHY THIS EXISTS. docs/performance-research.md priced the byte->watt->clock exchange rate
# against "2.09 GB/solve, 16.1 % of 13.0 GB". That figure was DERIVED, by scaling round 2's
# compulsory write, and it is wrong two ways: it charges the ablation with 268 MB of
# back-refs the ablation never touches, and it assumes a 16 B store costs 16 B at DRAM when
# it costs a full 32 B sector. Measured, the removal is 1.47 GB -- the published rate was
# 42 % low, and every "worth X MHz at 180 W" forecast divides by it.
#
# It also settles two mechanism questions that no amount of arithmetic could:
#   * a 16 B store costs a 32 B sector      (805 MB predicted at 16 B, 1342 at 32, 1330 measured)
#   * there is no read-for-ownership        (RFO would add ~1074 MB of read; measured +7.8)
#
# NEEDS ROOT, for ncu only. Nothing here sets a clock or a power limit.
#
#   sudo benchmarks/abl_bytes.sh              # round 2, the documented case
#   sudo ROUND=3 benchmarks/abl_bytes.sh      # any other ablatable round
#
# Round R is the Rth fused_round launch of four (cuda/pipeline.cu), so it is selected
# POSITIONALLY with --launch-skip. It cannot be selected by name: the kernel is plain
# `fused_round` and the round is a template argument, so a name regex matches all four or
# none. Verify the selection in the output -- the printed template arguments carry the mode
# (4 = LM_RD2 = round 2) and the run asserts it.
#
# ncu --csv does NOT suppress the target's own stdout, so the CSV stream begins with the
# pipeline's geometry banner. Parse the human-readable form instead; that is what the
# first attempt at this got wrong, and it silently reported zeroes.
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
. "$ROOT/benchmarks/lib.sh"

ROUND=${ROUND:-2}
SOLVES=${SOLVES:-3}
OUT=${OUT_DIR:-/tmp/mxbm-ablbytes}
NVCC=${NVCC:-nvcc}
ARCH=${ARCH:-sm_89}
mkdir -p "$OUT"

command -v ncu >/dev/null || { echo "ncu not on PATH"; exit 1; }
[ "$(id -u)" -eq 0 ] || echo "note: ncu usually needs root; continuing anyway"

build() {   # build <suffix> <extra-defines...>
    local out="$OUT/pipeline$1"; shift
    bench_stale "$out" || return 0
    echo "  building $(basename "$out") ..."
    "$NVCC" -O3 -arch="$ARCH" -std=c++17 -diag-suppress 186 "$@" \
        -I "$ROOT/src" -I "$ROOT/kernels/cuda" -I "$ROOT/tests" -I "$ROOT/third_party/blake2b" \
        "$ROOT/cuda/pipeline.cu" "$ROOT/src/beamhash/bh3_blake2b.cpp" \
        "$ROOT/src/beamhash/bh3_verify.cpp" "$ROOT/third_party/blake2b/blake2b-ref.c" \
        -o "$out" 2> "$out.log" \
      || { echo "build failed, see $out.log"; tail -5 "$out.log"; exit 1; }
}

echo "building (cached in $OUT) ..."
build ".base"
build ".abl$ROUND" -DMXBM_ABL_EMIT="$ROUND"

# The ablation must be shown to have applied before its bytes mean anything. An ablated
# build feeds the next round a garbled record and drops ~0.5 % of elements; a build where
# the define silently did nothing is clean, and would read as "the bytes were already
# gone". docs/performance-research.md "A note on method" is this failure, twice.
drops_of() { "$1" "$SOLVES" 2>&1 | sed -n 's/^drops *: *\([0-9]*\).*/\1/p' | head -1; }
BD=$(drops_of "$OUT/pipeline.base"); AD=$(drops_of "$OUT/pipeline.abl$ROUND")
echo "  control: baseline drops=$BD, ablated drops=$AD"
[ "${BD:-x}" = "0" ] || { echo "ABORT: baseline is not clean (drops=$BD)"; exit 1; }
[ "${AD:-0}" -gt 1000 ] 2>/dev/null \
  || { echo "ABORT: ablation did not apply -- ablated build drops $AD, expected ~180000."; exit 1; }

# ncu prints "<metric>  <unit>  <value>"; normalise the unit to bytes.
bytes_of() {   # bytes_of <binary> <metric>
    ncu --metrics "$2" --kernel-name-base function --kernel-name fused_round \
        --launch-skip "$((ROUND - 1))" --launch-count 1 "$1" "$SOLVES" 2>&1 \
      | awk -v m="$2" '
          $1 == m { u=$2; v=$3+0
                    if (u ~ /^Kbyte/) v *= 1e3; else if (u ~ /^Mbyte/) v *= 1e6
                    else if (u ~ /^Gbyte/) v *= 1e9
                    printf "%.0f", v; exit }'
}

echo "profiling round $ROUND (launch $((ROUND - 1)) of 4) ..."
BW=$(bytes_of "$OUT/pipeline.base"        dram__bytes_write.sum)
BR=$(bytes_of "$OUT/pipeline.base"        dram__bytes_read.sum)
AW=$(bytes_of "$OUT/pipeline.abl$ROUND"   dram__bytes_write.sum)
AR=$(bytes_of "$OUT/pipeline.abl$ROUND"   dram__bytes_read.sum)

for v in BW BR AW AR; do
    [ -n "${!v}" ] || { echo "ABORT: could not parse $v from ncu. Run one arm by hand:"
                        echo "  ncu --metrics dram__bytes_write.sum --kernel-name-base function \\"
                        echo "      --kernel-name fused_round --launch-skip $((ROUND-1)) --launch-count 1 \\"
                        echo "      $OUT/pipeline.base $SOLVES"; exit 1; }
done

python3 - "$BW" "$BR" "$AW" "$AR" "$ROUND" <<'PY'
import sys
bw, br, aw, ar = (float(x) for x in sys.argv[1:5]); rnd = sys.argv[5]
G, M, N = 1e9, 1e6, 2**25
net = (bw + br) - (aw + ar)
print()
print(f"  round {rnd}            write GB     read MB")
print(f"  baseline           {bw/G:9.3f}   {br/M:9.1f}")
print(f"  MXBM_ABL_EMIT={rnd}    {aw/G:9.3f}   {ar/M:9.1f}")
print(f"  removed            {(bw-aw)/G:9.3f}   {(br-ar)/M:+9.1f}")
print(f"\n  NET REMOVED = {net/G:.3f} GB")
if net <= 0:
    print("  ABORT: non-positive removal -- the ablation moved no bytes."); sys.exit(1)
back = 2*4*N
print(f"\n  sector question: 16 B/elem + back-refs = {(16*N+back)/M:.0f} MB,"
      f" 32 B sector = {(32*N+back)/M:.0f} MB, measured {aw/M:.0f} MB")
print(f"  RFO question   : would add ~{32*N/M:.0f} MB of read; measured {(ar-br)/M:+.1f} MB")
print(f"\n  replay-timeline rate : 210 MHz / {net/G:.2f} GB = {210/(net/G):.0f} MHz/GB")
print(f"  whole-solve bound    : <= 135 / {net/G:.2f}     = <= {135/(net/G):.0f} MHz/GB")
print(f"  to close 570 MHz     : >= {570/(135/(net/G)):.1f} GB of 13.0"
      f" = {570/(135/(net/G))/13.0*100:.0f} % of ALL traffic")
PY
