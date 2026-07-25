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
    //
    // Two decimals by default, matching the stats table's Speed column: the
    // line and the table report the same quantity, and printing it to
    // different precisions in the two places was an inconsistency, not a
    // feature. --digits moves both together.
    {
        miner::Stats::Snapshot s{};
        s.sol15 = 53.4712;
        check(format_speed_line(s) == "Average speed (15s): 53.47 sol/s",
              "format_speed_line renders sol15 to two decimals by default");
        check(format_speed_line(s, 0) == "Average speed (15s): 53 sol/s",
              "--digits 0 drops the decimal point entirely");
        check(format_speed_line(s, 4) == "Average speed (15s): 53.4712 sol/s",
              "--digits widens the value");
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

    // -- the Name column must not shift the other columns --
    // A full NVIDIA device string is 32 chars and used to overflow the 17-wide Name
    // field, pushing every following column right and breaking the header alignment
    // (reported from a live run). Assert on COLUMN POSITIONS against the header rather
    // than a golden, so this keeps testing the actual invariant if the table changes.
    {
        miner::Stats::Snapshot s{};
        s.sol60 = 57.08;
        s.pool_sol_session = 54.30;
        s.iter60 = 28.1;
        s.accepted = 5;
        s.best_share_units = 2700.0;
        s.last_latency_ms = 17;
        s.pool = "de.beam.herominers.com:1130";
        s.device_label = "NVIDIA GeForce RTX 4070 Ti SUPER";
        s.uptime = std::chrono::seconds(60);

        std::string got = format_stats_block(s, "0.4", "19:47:26");
        std::vector<std::string> lines;
        for (size_t i = 0, j; i <= got.size(); i = j + 1) {
            j = got.find('\n', i);
            if (j == std::string::npos) j = got.size();
            lines.push_back(got.substr(i, j - i));
        }
        // lines: 0 rule, 1 stats, 2 MXBM, 3 mining, 4 connected, 5 blank,
        //        6 header1, 7 header2, 8 device
        const std::string& header = lines.at(6);
        const std::string& device = lines.at(8);

        check(device.compare(0, 17, "RTX 4070 Ti SUPER") == 0,
              "device row strips the vendor prefix to fit the 17-wide Name column");
        check(device.find("NVIDIA") == std::string::npos,
              "the redundant vendor prefix is gone from the table");
        // Right-aligned numeric columns must end where their header ends.
        check(header.find("Speed") + 5 == device.find("57.08") + 5,
              "Speed column value ends flush with its header");
        check(header.find("Pool") + 4 == device.find("54.30") + 5,
              "Pool column value ends flush with its header");
        check(header.find("Iter.") + 5 == device.find("28.1") + 4,
              "Iter. column value ends flush with its header");
        check(header.find("Shares") + 6 == device.find("5/0/0") + 5,
              "Shares column value ends flush with its header");
    }

    // -- the dev-fee row: present when a fee is charged, absent when none is --
    //
    // The golden block above is a no-fee snapshot (devfee_rate 0), which is
    // what pins the "absent" half: adding an unconditional row would have
    // broken it. This pins the other half -- that a build which DOES charge
    // says so on the table, with the numbers a user needs to check the rate
    // against their own uptime rather than take it on trust.
    {
        miner::Stats::Snapshot s{};
        s.pool = "de.beam.herominers.com:1130";
        s.device_label = "GPU 0";
        s.uptime = std::chrono::seconds(7200);
        s.accepted = 40;
        s.devfee_rate = 0.01;
        s.devfee_slices = 2;
        s.devfee_seconds = 72.0;
        s.devfee_accepted = 1;

        std::string got = format_stats_block(s, "0.4", "19:47:26");
        check(got.find("Dev fee 1%") != std::string::npos,
              "the stats table states the fee rate when one is charged");
        check(format_stats_block([&]{ auto t = s; t.devfee_rate = 0.025; return t; }(), "0.4", "19:47:26")
                  .find("Dev fee 2.5%") != std::string::npos,
              "a fractional rate is shown exactly, not rounded");
        check(got.find("2 rounds, 72s total") != std::string::npos,
              "...alongside what has actually been spent this session");
        check(got.find("1/0/0 A/S/R") != std::string::npos,
              "...and the fee pool's own share verdicts, kept out of the user's row");
        check(got.find("(active now)") == std::string::npos,
              "no round is marked active when none is running");

        // The user's own Shares column must still read 40/0/0 -- the fee's
        // single accepted share must not have leaked into their count.
        check(got.find("40/0/0") != std::string::npos,
              "the user's share counters exclude dev-fee shares");

        s.devfee_active = true;
        check(format_stats_block(s, "0.4", "19:47:26").find("(active now)") != std::string::npos,
              "a round in progress is marked, so the pool switch is explained");

        s.devfee_rate = 0.0;
        check(format_stats_block(s, "0.4", "19:47:26").find("Dev fee") == std::string::npos,
              "a build with no fee shows no dev-fee row at all");
    }

    // -- the identity line: version, driver, API port --
    //
    // Both extras are omitted rather than faked when unknown, which is what
    // keeps the golden above (a fixture with neither) valid.
    {
        miner::Stats::Snapshot s{};
        s.pool = "pool.example.com:1130";
        s.uptime = std::chrono::seconds(60);

        std::string bare = split_lines(format_stats_block(s, "0.5.1", "12:00:00"))[2];
        check(bare == "MXBM 0.5.1",
              "no driver and no API port leaves the version standing alone");

        s.driver_version = "610.43.03";
        std::string full = split_lines(format_stats_block(s, "0.5.1", "12:00:00", 2, 8080))[2];
        check(full == "MXBM 0.5.1, Nvidia 610.43.03, API port 8080",
              "the driver version and API port join the version line");

        std::string no_api = split_lines(format_stats_block(s, "0.5.1", "12:00:00", 2, 0))[2];
        check(no_api == "MXBM 0.5.1, Nvidia 610.43.03",
              "API port 0 means the API is off, so it is left off the line -- not printed as port 0");
    }

    // -- --digits keeps the whole table in lockstep --
    //
    // Speed and Pool each widen by (digits - 2), so the two header lines and
    // the two value rows must ALL grow by exactly 2*(digits-2) characters. If
    // the headers ever drift from the values, every column to the right of
    // Pool is silently misaligned -- which a test that only checks the numbers
    // would never notice.
    {
        miner::Stats::Snapshot s{};
        s.sol60 = 53.4712;
        s.pool_sol_session = 52.1035;
        s.iter60 = 28.1;
        s.accepted = 10;
        s.best_share_units = 8012.0;
        s.pool = "pool.example.com:1130";
        s.device_label = "NVIDIA GeForce RTX 4070 Ti SUPER";
        s.uptime = std::chrono::seconds(8130);

        std::vector<std::string> a = split_lines(format_stats_block(s, "1.0.0", "12:00:00", 2));
        std::vector<std::string> b = split_lines(format_stats_block(s, "1.0.0", "12:00:00", 4));
        check(a.size() == b.size(), "--digits does not add or drop table lines");

        int widened = 0;
        bool only_expected_growth = true;
        if (a.size() == b.size()) {
            for (size_t i = 0; i < a.size(); ++i) {
                const long grew = (long)b[i].size() - (long)a[i].size();
                if (grew == 4) ++widened;               // 2 columns x 2 extra decimals
                else if (grew != 0) only_expected_growth = false;
            }
        }
        check(only_expected_growth,
              "every line either keeps its width or grows by exactly the two columns' worth");
        check(widened == 4,
              "the two header lines and the two value rows all widen together");

        // And the values really are at the new precision.
        check(format_stats_block(s, "1.0.0", "12:00:00", 4).find("53.4712") != std::string::npos,
              "--digits 4 renders the Speed column to four decimals");
    }

    return summary("format");
}
