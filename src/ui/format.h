#pragma once
#include <string>

#include "miner/stats.h"

namespace mxbm { namespace ui {

// Pure the reference miner-style formatting: no I/O, no clock reads, no globals -- every
// function here is a straight function of its arguments, which is what makes
// them golden-testable (see tests/test_format.cpp). Console/ticker do the
// printing; these just produce the strings.

// the reference miner-style difficulty/hashrate unit formatting: one decimal always,
// "k" (thousands) below 1e6, "M" (millions) at/above 1e6 -- e.g. 512 ->
// "0.5k", 8012 -> "8.0k", 239500 -> "239.5k", 1300000 -> "1.3M". The 1e6
// threshold is compared against the raw value, not the rounded display
// value, so 999999 prints "1000.0k" (still below the M cutoff) rather than
// rounding up into "M". Inputs in this codebase (achieved-share units,
// candidate rates) are always >= 0.
std::string format_units(double value);

// The the reference miner-style "--shortstats" one-liner: "Average speed (15s): X.XX
// sol/s", the value being snapshot.sol15 to `digits` decimals (--digits; 2 by
// default, matching the stats table's Speed column). The "(15s)" text is a
// fixed literal matching the default --shortstats interval -- it does not read
// back the caller's actual ticker interval, per this function's
// single-Snapshot-argument interface (Ticker may run on a different
// interval; see ticker.h).
std::string format_speed_line(const miner::Stats::Snapshot& snapshot, int digits = 2);

// The the reference miner-style "--longstats" multi-line table: opening dashed rule,
// session header (clock/uptime, version, algorithm, pool+latency), a blank
// line, a two-line column header, one device row (Phase B always has
// exactly one device: "CPU 0 reference", the CPU reference solver -- there
// is no per-device breakdown in Stats yet, so the row echoes the aggregate
// Snapshot), a short dashed rule, a Total row (same aggregate values, minus
// the clock/temp/fan columns the reference miner's own Total omits), and a closing
// dashed rule matching the opening one. Returns the block WITHOUT a
// trailing newline -- callers that want one add it themselves (e.g.
// console::info(), via its own print_line()).
//
// `version` is the fully-formed "<maj>.<min>.<count> [<hash>]" string from
// mxbm::version(); `clock_hhmmss` is the caller-supplied wall-clock string
// ("HH:MM:SS") -- format.cpp is pure and never reads the clock itself (see
// ui::Ticker, which supplies both from the live version()/localtime()).
//
// Column-value notes (not obvious from Snapshot's field names alone):
//   Speed column  = snapshot.sol60 (the 60s-windowed candidate rate -- this
//                   block is itself generated on the long-stats, ~60s-by-
//                   default cadence, so the 60s window is the "since last
//                   refresh" figure; the short, more volatile 15s window is
//                   what the separate format_speed_line() ticker line uses
//                   instead, matching the reference miner's own two-cadence split).
//   Pool column   = snapshot.pool_sol_session (the reference miner's "poolHr"): the rate
//                   the POOL credited, from the summed target difficulty of
//                   accepted shares. Measures the same quantity as Speed by a
//                   different route and agrees with it in expectation -- see
//                   docs/usage.md, "Speed vs pool rate", for why they diverge.
//   Iter. column  = snapshot.iter60 (raw solver attempt rate, i.e. solve()
//                   calls/sec -- distinct from Speed/Pool's *candidate*-rate
//                   "sol/s" units).
//   Eff./Power/CCLK/MCLK/Core/Fan = per-field telemetry, each "--" when the
//                   platform could not supply that particular reading.
//
// `api_port` is the port the /summary API is serving on, or 0 when it is off --
// it and the NVIDIA driver version (Snapshot::driver_version) join the version
// on the block's identity line, both omitted when unknown rather than faked.
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
