#include "ui/format.h"

#include <chrono>
#include <cstdio>
#include <cstring>

namespace mxbm { namespace ui {

namespace {

// "Xh Ym Zs" from a whole-second duration (Beam/the reference miner-style uptime).
std::string format_uptime(std::chrono::seconds uptime) {
    long long total = uptime.count();
    long long h = total / 3600;
    long long m = (total % 3600) / 60;
    long long s = total % 60;
    char buf[32];
    std::snprintf(buf, sizeof buf, "%lldh %lldm %llds", h, m, s);
    return buf;
}

// "A/S/R" packed accepted/stale/rejected triple, shared by both table rows.
std::string format_shares(const miner::Stats::Snapshot& s) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%llu/%llu/%llu",
        (unsigned long long)s.accepted, (unsigned long long)s.stale,
        (unsigned long long)s.rejected);
    return buf;
}

// Column headers -- literal text, not derived from any format string (see
// format_stats_block's doc comment for why the row format strings below
// omit the separator space right after the Name field: matching this exact
// header alignment is what pins that down).
const char* const kRule47 =
    "-----------------------------------------------";
const char* const kRule27 = "---------------------------";
const char* const kHeader1 =
    "      Name        Speed   Pool  Iter.   Shares   Best     Eff.  Power  CCLK   MCLK  Core  Fan";
const char* const kHeader2 =
    "                  sol/s  sol/s   it/s    A/S/R  Share  sol/s/W      W   MHz    MHz  Temp  Pct";

} // namespace

std::string format_units(double value) {
    char buf[32];
    if (value >= 1e6) {
        std::snprintf(buf, sizeof buf, "%.1fM", value / 1e6);
    } else {
        std::snprintf(buf, sizeof buf, "%.1fk", value / 1e3);
    }
    return buf;
}

std::string format_speed_line(const miner::Stats::Snapshot& s) {
    char buf[128];
    std::snprintf(buf, sizeof buf, "Average speed (15s): %.1f sol/s", s.sol15);
    return buf;
}

// The Name column is 17 wide. A full device string ("NVIDIA GeForce RTX 4070 Ti
// SUPER", 32 chars) overflows it and pushes every following column out of line, so
// drop the vendor prefix -- which carries no information in a per-device row -- and
// truncate only if what remains still does not fit.
static std::string short_device_name(const std::string& full) {
    static const char* kPrefixes[] = {
        "NVIDIA GeForce ", "NVIDIA ", "AMD Radeon ", "AMD ",
        "Advanced Micro Devices, Inc. ", "Intel(R) ", "Intel ",
    };
    std::string n = full;
    for (const char* p : kPrefixes) {
        const size_t len = std::strlen(p);
        if (n.compare(0, len, p) == 0) { n.erase(0, len); break; }
    }
    constexpr size_t kNameWidth = 17;
    // Plain truncation, not an ellipsis: %-17s pads by BYTES and a multi-byte glyph
    // would occupy one column while counting as three, re-breaking the alignment.
    if (n.size() > kNameWidth) n.resize(kNameWidth);
    return n;
}

std::string format_stats_block(const miner::Stats::Snapshot& s,
                                const char* version,
                                const char* clock_hhmmss) {
    // Latency shows the connect-handshake duration until the first share
    // result comes back (last_latency_ms < 0), then switches to the
    // measured submit-to-result latency.
    long long latency_ms = (s.last_latency_ms >= 0) ? s.last_latency_ms : s.connect_ms;

    std::string shares = format_shares(s);
    std::string best = format_units(s.best_share_units);

    char line2[128];
    std::snprintf(line2, sizeof line2, "Statistics (%s); Uptime: %s",
        clock_hhmmss, format_uptime(s.uptime).c_str());

    char line5[256];
    std::snprintf(line5, sizeof line5, "Connected to: %s (%lldms latency)",
        s.pool.c_str(), latency_ms);

    // Device row: %-17s deliberately abuts the first %6.2f with NO literal
    // separator space between them (every OTHER field is single-space
    // separated) -- this is a reconciliation against the task brief's
    // golden table: the brief's prose format string put a space there too,
    // but the golden's own column alignment has one fewer space in that one
    // gap, so the golden (what ships) wins; see task-3-report.md.
    // Telemetry columns show a value when the platform supplied one and "--" when it
    // did not, per field: a card that reports power but not fan shows the power.
    char eff[16], pw[16], cclk[16], mclk[16], tmp[16], fan[16];
    auto dashf = [](char* b, size_t n, const char* fmt, bool have, double v) {
        if (have) std::snprintf(b, n, fmt, v); else std::snprintf(b, n, "--");
    };
    dashf(eff,  sizeof eff,  "%.3f", s.has_power && s.power_w > 0.0, s.sol60 / (s.power_w > 0.0 ? s.power_w : 1.0));
    dashf(pw,   sizeof pw,   "%.0f", s.has_power,     s.power_w);
    dashf(cclk, sizeof cclk, "%.0f", s.has_sm_clock,  (double)s.sm_clock_mhz);
    dashf(mclk, sizeof mclk, "%.0f", s.has_mem_clock, (double)s.mem_clock_mhz);
    dashf(tmp,  sizeof tmp,  "%.0f", s.has_temp,      (double)s.temp_c);
    dashf(fan,  sizeof fan,  "%.0f", s.has_fan,       (double)s.fan_pct);

    char device_row[256];
    std::snprintf(device_row, sizeof device_row,
        "%-17s%6.2f %6.2f %6.1f %8s %6s %8s %6s %5s %6s %5s %4s",
        short_device_name(s.device_label).c_str(), s.sol60, s.pool_sol_session, s.iter60,
        shares.c_str(), best.c_str(),
        eff, pw, cclk, mclk, tmp, fan);

    // Total row: same reconciliation (no separator after %-18s); the reference miner's
    // own Total omits the clock/temp/fan columns entirely (not just blanks
    // them), so this format string is shorter, not just filled with "--".
    char total_row[256];
    std::snprintf(total_row, sizeof total_row,
        "%-18s%6.2f %6.2f %6.1f %8s %6s %8s %6s",
        "Total", s.sol60, s.pool_sol_session, s.iter60,
        shares.c_str(), best.c_str(), "--", "--");

    std::string out;
    out.reserve(640);
    out += kRule47;   out += '\n';
    out += line2;     out += '\n';
    out += "MXBM ";   out += version; out += '\n';
    out += "Mining: BeamHash III\n";
    out += line5;     out += '\n';
    out += '\n';
    out += kHeader1;  out += '\n';
    out += kHeader2;  out += '\n';
    out += device_row; out += '\n';
    out += kRule27;   out += '\n';
    out += total_row; out += '\n';
    out += kRule47;
    return out;
}

} } // namespace mxbm::ui
