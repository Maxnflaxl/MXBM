#pragma once
#include <string>
#include <vector>

#include "miner/stats.h"

namespace mxbm { namespace ui {

// Pure formatting: no I/O, no clock reads, no globals -- every function here
// is a straight function of its arguments.

// Unit formatting: one decimal always, "k" below 1e6, "M"
// at/above 1e6. The threshold is compared against the raw value, not the
// rounded one, so 999999 prints "1000.0k".
std::string format_units(double value);

// The "--shortstats" one-liner: "Average speed (15s): X.XX sol/s" from
// snapshot.sol15. The "(15s)" is a fixed literal matching the default
// --shortstats interval, NOT the caller's actual ticker interval.
std::string format_speed_line(const miner::Stats::Snapshot& snapshot, int digits = 2);

// One column of the statistics table. `total` false means the Total row leaves
// the cell blank. `vlabel` is the spelled-out row label --vstats uses.
struct StatColumn {
    const char* name;      // the --statsformat token
    const char* head1;
    const char* head2;
    int         width;
    bool        total;
    const char* vlabel;
};

// Every column that exists, in the order the default table uses. The first
// twelve are the default set; the rest are selectable with --statsformat.
const std::vector<StatColumn>& stat_columns();
const StatColumn* stat_column(const std::string& name);   // null if unknown
std::vector<std::string> default_stat_format();

// Parses a --statsformat list ("gpuName,speed,power") into column names. There
// are NO presets: a bare word that is not a field name is an error naming the
// fields, never a silently different table.
bool parse_stats_format(const std::string& spec, std::vector<std::string>& out,
                        std::string& err);

// How the table is laid out. `columns` empty means the default set.
struct StatsLayout {
    std::vector<std::string> columns;
    bool vertical = false;      // --vstats: one column per device, fields as rows
    int  wrap_width = 0;        // --hstats N: wrap columns into groups of N chars
    int  digits = 2;
    int  api_port = 0;
};

// The "--longstats" multi-line table: session header (clock and uptime,
// version, algorithm, pool+latency), a two-line column header, one device row
// (Stats has no per-device breakdown, so it echoes the aggregate Snapshot), a
// Total row (minus the clock/temp/fan columns, which do not sum) and
// dashed rules. Returns the block WITHOUT a trailing newline.
//
// `clock_hhmmss` is the caller-supplied "HH:MM:SS" string; see ui::Ticker.
//
// Column-value notes (not obvious from Snapshot's field names alone):
//   Pool column  = pool_sol_session ("poolHr"): the rate the POOL
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
                               const StatsLayout& layout);

std::string format_stats_block(const miner::Stats::Snapshot& snapshot,
                                const char* version,
                                const char* clock_hhmmss,
                                int digits = 2,
                                int api_port = 0);

} } // namespace mxbm::ui
