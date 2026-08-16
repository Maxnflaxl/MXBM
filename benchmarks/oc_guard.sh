# Sourced by the OC benchmark scripts. Two things every clock experiment needs and
# that a hand-rolled loop keeps getting wrong.
#
# oc_require_idle -- a co-tenant does not add noise to a benchmark, it moves the number,
# and an OC sweep is especially exposed because the knob is DEVICE-global: another
# process on the card runs at this script's clocks and steals its solves at the same
# time. headline.sh hard-aborts on one; so does this.
#
# oc_clock_under_load -- `nvidia-smi --query-gpu=clocks.mem` after a run samples the
# card settling, not the clock the run held. Sample throughout instead, from ONE
# process: repeatedly launching nvidia-smi initialises a driver context each time and
# is not free, where a single `-lms` sampler measures at 0.0 %.

oc_require_idle() {
    local n
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
    if [ "${n:-0}" -ne 0 ]; then
        echo "ABORT: $n other compute process(es) on the card." >&2
        nvidia-smi --query-compute-apps=pid,process_name --format=csv >&2
        exit 1
    fi
}

# oc_sample_start <file>  /  oc_sample_stop -> prints "<modal mem>,<modal core>"
oc_sample_start() {
    OC_SAMPLE_FILE=$1
    nvidia-smi --query-gpu=clocks.mem,clocks.gr --format=csv,noheader,nounits \
               -lms 1500 > "$OC_SAMPLE_FILE" 2>&1 &
    OC_SAMPLE_PID=$!
}
oc_sample_stop() {
    kill "${OC_SAMPLE_PID:-0}" 2>/dev/null
    wait "${OC_SAMPLE_PID:-0}" 2>/dev/null
    # Modal sample, dropping the idle rows either side of the run.
    awk -F', *' '$1 > 1000 {print $1","$2}' "$OC_SAMPLE_FILE" \
        | sort | uniq -c | sort -rn | head -1 | awk '{print $2}'
}
