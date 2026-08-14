#!/usr/bin/env bash
# Profile the CUDA pipeline and write the report to cuda/ncu_report.txt.
#
# Nsight needs GPU counter access, which is admin-restricted by default. Run this with
# sudo, or lift the restriction permanently with NVreg_RestrictProfilingToAdminUsers=0 in
# /etc/modprobe.d and reboot -- the module cannot be reloaded while the GPU drives a
# display. /proc/driver/nvidia/params reports the state as RmProfilingAdminOnly.
#
# One solve only (MXBM_CUDA_ITERS=1): ncu replays each launch to collect counters.
set -u
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
ncu --target-processes all --metrics "$METRICS" \
    --csv --log-file "$OUT" ./cuda/pipeline 1
rc=$?
echo "exit=$rc  ->  $OUT  ($(wc -l < "$OUT" 2>/dev/null || echo 0) lines)"
[ $rc -ne 0 ] && echo "if this says ERR_NVGPUCTRPERM, rerun with sudo"
exit 0
