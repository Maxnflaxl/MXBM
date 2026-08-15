#!/usr/bin/env bash
# Profile the CUDA pipeline and write the report to cuda/ncu_report.txt.
#
# Nsight needs GPU counter access, admin-restricted by default. Lifted permanently on this
# rig by NVreg_RestrictProfilingToAdminUsers=0 in /etc/modprobe.d/nvidia-profiling.conf, so
# no sudo is needed -- check with `grep RmProfilingAdminOnly /proc/driver/nvidia/params`
# (0 = unrestricted). Installing that file needs one reboot, because the nvidia module
# cannot be reloaded while the GPU drives a display; running the profiler does not.
#
# CLOCKS=none|base|... -> ncu's --clock-control. ncu defaults to `base`, which locks the
# card below the clock the miner actually runs at, so every duration in a default report
# is at a clock the product never sees. Both are worth taking: `base` is reproducible
# across sessions, `none` is what the solve costs.
#
# TARGET=pipeline|miner. The miner folds the entry pass into round 4 as co-blocks, so
# profiling ./cuda/pipeline reports entry and r4 as two kernels that the product has as
# one. `pipeline` is the historical baseline; `miner` is what ships.
#
# One solve only: ncu replays every launch to collect counters.
set -u
CLOCKS=${CLOCKS:-base}
TARGET=${TARGET:-pipeline}
cd "$(dirname "$0")/.."
OUT=cuda/ncu_report.txt

METRICS=$(tr -d ' \n' <<'M'
gpu__time_duration.sum,
dram__bytes_read.sum,
dram__bytes_write.sum,
dram__throughput.avg.pct_of_peak_sustained_elapsed,
lts__t_sector_hit_rate.pct,
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,
l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,
smsp__warps_launched.sum,
sm__throughput.avg.pct_of_peak_sustained_elapsed,
launch__occupancy_limit_shared_mem,
smsp__average_warp_latency_per_inst_issued.ratio,
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,
smsp__warp_issue_stalled_barrier_per_warp_active.pct,
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,
smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct,
smsp__warp_issue_stalled_wait_per_warp_active.pct,
smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,
smsp__warp_issue_stalled_imc_miss_per_warp_active.pct,
smsp__warp_issue_stalled_no_instruction_per_warp_active.pct,
smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct,
sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,
l1tex__cycles_active.avg.pct_of_peak_sustained_active,
smsp__inst_issued.avg.per_cycle_active,
gpc__cycles_elapsed.avg.per_second,
smsp__inst_executed_op_global_ld.sum,
smsp__inst_executed_op_global_st.sum,
smsp__inst_executed_op_shared_ld.sum,
smsp__inst_executed_op_shared_st.sum,
l1tex__data_pipe_lsu_wavefronts.sum,
l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio
M
)

# `1` = KAT gate + one timed solve. ncu replays every launch to collect counters, so
# the default 20-nonce loop would take an age for no extra signal.
if [ "$TARGET" = miner ]; then
    CMD="./build/mxbm --benchmark BEAM-III --benchmark-seconds 1"
else
    CMD="./cuda/pipeline 1"
fi
echo "target=$TARGET  clocks=$CLOCKS  -> $OUT"
ncu --target-processes all --metrics "$METRICS" --clock-control "$CLOCKS" \
    --csv --log-file "$OUT" $CMD
rc=$?
echo "exit=$rc  ->  $OUT  ($(wc -l < "$OUT" 2>/dev/null || echo 0) lines)"
[ $rc -ne 0 ] && echo "if this says ERR_NVGPUCTRPERM, check RmProfilingAdminOnly (see top)"
exit 0
