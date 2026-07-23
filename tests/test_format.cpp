// Console v2 pure-format tests: format_units, format_speed_line, and the
// full format_stats_block golden. Golden strings below are LITERALS --
// never built by calling the functions under test -- so a regression in
// format.cpp's field widths/wording shows up as a byte-for-byte mismatch
// against a fixed expectation, not a self-consistent-but-wrong pass.
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "check.h"
#include "miner/stats.h"
#include "ui/format.h"

using namespace mxbm;
using namespace mxbm::ui;

namespace {

std::vector<std::string> split_lines(const std::string& s) {
    std::vector<std::string> out;
    size_t start = 0;
    while (true) {
        size_t nl = s.find('\n', start);
        if (nl == std::string::npos) { out.push_back(s.substr(start)); break; }
        out.push_back(s.substr(start, nl - start));
        start = nl + 1;
    }
    return out;
}

// Wraps check() for the multi-line golden so a mismatch prints exactly
// which line(s) differ -- eyeballing a 12-line blob diff otherwise is
// impractical.
void check_golden_block(const std::string& got, const std::string& want, const char* msg) {
    check(got == want, msg);
    if (got == want) return;
    std::vector<std::string> gl = split_lines(got), wl = split_lines(want);
    size_t n = gl.size() > wl.size() ? gl.size() : wl.size();
    for (size_t i = 0; i < n; ++i) {
        std::string g = i < gl.size() ? gl[i] : "<missing line>";
        std::string w = i < wl.size() ? wl[i] : "<missing line>";
        if (g != w) {
            std::printf("    line %zu:\n      got : %s\n      want: %s\n", i, g.c_str(), w.c_str());
        }
    }
}

} // namespace

int main() {
    // -- format_units: k below 1e6 (always one decimal), M at/above 1e6 --
    check(format_units(512.0) == "0.5k", "format_units(512) == 0.5k");
    check(format_units(8012.0) == "8.0k", "format_units(8012) == 8.0k");
    check(format_units(239500.0) == "239.5k", "format_units(239500) == 239.5k");
    check(format_units(1300000.0) == "1.3M", "format_units(1300000) == 1.3M");
    check(format_units(999999.0) == "1000.0k",
          "format_units(999999) == 1000.0k (still below the 1e6 M-threshold)");

    // -- format_speed_line: the --shortstats one-liner, sol15-driven --
    {
        miner::Stats::Snapshot s{};
        s.sol15 = 53.5;
        check(format_speed_line(s) == "Average speed (15s): 53.5 sol/s",
              "format_speed_line renders sol15 to one decimal");
    }

    // -- format_stats_block: full --longstats table, golden byte-for-byte --
    // Fixture per the task brief: sol15=sol60=sol_session=0.01, iter60=0.3,
    // 1 accepted/0 stale/0 rejected, best share 1234.0 units (-> "1.2k"),
    // last_latency_ms=12 (so connect_ms below must be ignored), pool
    // "pool.example.com:1130", uptime 8130s (-> "2h 15m 30s").
    {
        miner::Stats::Snapshot s{};
        s.sol15 = 0.01;
        s.sol60 = 0.01;
        s.sol_session = 0.01;
        s.iter60 = 0.3;
        s.accepted = 1;
        s.stale = 0;
        s.rejected = 0;
        s.best_share_units = 1234.0;
        s.last_latency_ms = 12;
        s.pool = "pool.example.com:1130";
        s.device_label = "CPU 0 reference";   // the stats row's worker name is now snapshot-driven
        s.connect_ms = 999;   // must be ignored: last_latency_ms >= 0 takes priority
        s.uptime = std::chrono::seconds(8130);
        s.last_job_id = "";
        s.last_job_units = 0.0;
        s.reconnects = 0;

        const std::string golden =
R"GOLDEN(-----------------------------------------------
Statistics (19:47:26); Uptime: 2h 15m 30s
MXBM 0.2.31 [abc1234]
Mining: BeamHash III
Connected to: pool.example.com:1130 (12ms latency)

      Name        Speed   Pool  Iter.   Shares   Best     Eff.  Power  CCLK   MCLK  Core  Fan
                  sol/s  sol/s   it/s    A/S/R  Share  sol/s/W      W   MHz    MHz  Temp  Pct
CPU 0 reference    0.01   0.00    0.3    1/0/0   1.2k       --     --    --     --    --   --
---------------------------
Total               0.01   0.00    0.3    1/0/0   1.2k       --     --
-----------------------------------------------)GOLDEN";

        std::string got = format_stats_block(s, "0.2.31 [abc1234]", "19:47:26");
        check_golden_block(got, golden, "format_stats_block matches the the reference miner-style golden exactly");
    }

    return summary("format");
}
