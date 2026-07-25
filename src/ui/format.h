#pragma once
#include <string>

#include "miner/stats.h"

namespace mxbm { namespace ui {

// Pure the reference miner-style formatting: no I/O, no clock reads, no globals -- every
// function here is a straight function of its arguments.

// the reference miner-style unit formatting: one decimal always, "k" below 1e6, "M"
// at/above 1e6. The threshold is compared against the raw value, not the
// rounded one, so 999999 prints "1000.0k".
std::string format_units(double value);

// The "--shortstats" one-liner: "Average speed (15s): X.XX sol/s" from
// snapshot.sol15. The "(15s)" is a fixed literal matching the default
// --shortstats interval, NOT the caller's actual ticker interval.
std::string format_speed_line(const miner::Stats::Snapshot& snapshot, int digits = 2);

// The "--longstats" multi-line table: session header (clock and uptime,
// version, algorithm, pool+latency), a two-line column header, one device row
// (Stats has no per-device breakdown, so it echoes the aggregate Snapshot), a
// Total row (minus the clock/temp/fan columns the reference miner's own Total omits) and
// dashed rules. Returns the block WITHOUT a trailing newline.
//
// `clock_hhmmss` is the caller-supplied "HH:MM:SS" string; see ui::Ticker.
//
// Column-value notes (not obvious from Snapshot's field names alone):
//   Pool column  = pool_sol_session (the reference miner's "poolHr"): the rate the POOL
//                  credited, from the summed target difficulty of accepted
//                  shares. Same quantity as Speed by a different route -- see
//                  docs/usage.md, "Speed vs pool rate", for why they diverge.
//   Iter. column = iter60, the raw solver attempt rate -- distinct from
//                  Speed/Pool's *candidate*-rate "sol/s" units.
//
// `api_port` is the port the /summary API is serving on, or 0 when it is off;
// it and the NVIDIA driver version are omitted when unknown rather than faked.
//
// `digits` (--digits) sets the decimals on the Speed and Pool columns, whose
// field widens to match so nothing to their right shifts; the two header lines
// are padded by the same amount.
std::string format_stats_block(const miner::Stats::Snapshot& snapshot,
                                const char* version,
                                const char* clock_hhmmss,
                                int digits = 2,
                                int api_port = 0);

} } // namespace mxbm::ui
