#include "ui/format.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

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

// "A/S/R" packed accepted/stale/rejected triple, shared by every table row.
std::string format_shares(uint64_t a, uint64_t st, uint64_t r) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%llu/%llu/%llu",
        (unsigned long long)a, (unsigned long long)st, (unsigned long long)r);
    return buf;
}

// Column headers -- literal text. Matching this exact alignment is what pins
// down the missing separator space after the Name field in the row formats.
const char* const kRule47 =
    "-----------------------------------------------";
const char* const kRule27 = "---------------------------";
const char* const kHeader1 =
    "      Name        Speed   Pool  Iter.   Shares   Best     Eff.  Power  CCLK   MCLK  Core  Fan";
const char* const kHeader2 =
    "                  sol/s  sol/s   it/s    A/S/R  Share  sol/s/W      W   MHz    MHz  Temp  Pct";

// Layout constants the --digits widening needs: the Name field is 17 wide, the
// Speed value follows it with NO separator, and Speed and Pool are 6 wide at
// the default 2 decimals, one space apart.
constexpr int kNameW = 17;
constexpr int kSpeedW = 6;          // at kDefaultDigits
constexpr int kDefaultDigits = 2;

// Speed and Pool are the only --digits-sensitive columns, and both grow by
// (digits - 2) characters. The two header lines have to gain the same padding
// at the same two boundaries, or every column to the right of Pool goes out of
// line with the values beneath it.
std::string widen_header(const char* header, int digits) {
    const int k = digits - kDefaultDigits;
    if (k <= 0) return header;
    std::string h = header;
    const std::string pad((size_t)k, ' ');
    h.insert(kNameW, pad);                       // Speed's field grows...
    h.insert(kNameW + k + kSpeedW + 1, pad);     // ...then Pool's, one space later
    return h;
}

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

std::string format_speed_line(const miner::Stats::Snapshot& s, int digits) {
    char buf[128];
    std::snprintf(buf, sizeof buf, "Average speed (15s): %.*f sol/s", digits, s.sol15);
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
                                const char* clock_hhmmss,
                                int digits,
                                int api_port) {
    // Widens with --digits; widen_header pads the headers by the same amount.
    const int speed_w = kSpeedW + (digits - kDefaultDigits);
    // Latency shows the connect-handshake duration until the first share result
    // comes back, then the measured submit-to-result latency.
    long long latency_ms = (s.last_latency_ms >= 0) ? s.last_latency_ms : s.connect_ms;

    std::string shares = format_shares(s.accepted, s.stale, s.rejected);
    std::string best = format_units(s.best_share_units);

    char line2[128];
    std::snprintf(line2, sizeof line2, "Statistics (%s); Uptime: %s",
        clock_hhmmss, format_uptime(s.uptime).c_str());

    char line5[256];
    std::snprintf(line5, sizeof line5, "Connected to: %s (%lldms latency)",
        s.pool.c_str(), latency_ms);

    // One row per device: %-17s deliberately abuts the first %*.*f with NO
    // literal separator space (every OTHER field is single-space separated), as
    // the header alignment requires. Telemetry columns show "--" per missing
    // field.
    //
    // The pool-rate column is a property of the CONNECTION, not of a card --
    // the pool credits shares without saying which GPU found them -- so it is
    // shown only on the Total row and dashed out per device. Attributing a
    // share of it per card would be a number nobody measured.
    auto dashf = [](char* b, size_t n, const char* fmt, bool have, double v) {
        if (have) std::snprintf(b, n, fmt, v); else std::snprintf(b, n, "--");
    };

    // A Snapshot built by hand -- a test, or any consumer predating the device
    // vector -- carries the legacy single-device fields and no rows. Synthesise
    // the one row they describe rather than printing a table with no devices in
    // it; Stats::snapshot() always fills the vector, so this is the fallback
    // path only.
    std::vector<miner::Stats::Device> rows = s.devices;
    if (rows.empty()) {
        miner::Stats::Device d;
        d.label = s.device_label;
        d.sol15 = s.sol15;  d.sol60 = s.sol60;  d.iter60 = s.iter60;
        d.accepted = s.accepted; d.stale = s.stale; d.rejected = s.rejected;
        d.best_share_units = s.best_share_units;
        d.has_power = s.has_power;         d.power_w = s.power_w;
        d.has_sm_clock = s.has_sm_clock;   d.sm_clock_mhz = s.sm_clock_mhz;
        d.has_mem_clock = s.has_mem_clock; d.mem_clock_mhz = s.mem_clock_mhz;
        d.has_temp = s.has_temp;           d.temp_c = s.temp_c;
        d.has_fan = s.has_fan;             d.fan_pct = s.fan_pct;
        rows.push_back(std::move(d));
    }

    std::string device_rows;
    const bool multi = rows.size() > 1;
    for (size_t i = 0; i < rows.size(); ++i) {
        const miner::Stats::Device& d = rows[i];
        char eff[16], pw[16], cclk[16], mclk[16], tmp[16], fan[16];
        dashf(eff,  sizeof eff,  "%.3f", d.has_power && d.power_w > 0.0,
              d.sol60 / (d.power_w > 0.0 ? d.power_w : 1.0));
        dashf(pw,   sizeof pw,   "%.0f", d.has_power,     d.power_w);
        dashf(cclk, sizeof cclk, "%.0f", d.has_sm_clock,  (double)d.sm_clock_mhz);
        dashf(mclk, sizeof mclk, "%.0f", d.has_mem_clock, (double)d.mem_clock_mhz);
        dashf(tmp,  sizeof tmp,  "%.0f", d.has_temp,      (double)d.temp_c);
        dashf(fan,  sizeof fan,  "%.0f", d.has_fan,       (double)d.fan_pct);

        // "GPU 0 " prefixes the name once there is more than one card, so the
        // index in --devices and --pl is readable straight off the table.
        std::string name = short_device_name(d.label);
        if (multi) {
            char pfx[24];
            std::snprintf(pfx, sizeof pfx, "GPU %zu ", i);
            name = std::string(pfx) + name;
            if (name.size() > (size_t)kNameW) name.resize((size_t)kNameW);
        }
        const std::string dshares = format_shares(d.accepted, d.stale, d.rejected);
        const std::string dbest = format_units(d.best_share_units);

        char row[256];
        std::snprintf(row, sizeof row,
            "%-17s%*.*f %*s %6.1f %8s %6s %8s %6s %5s %6s %5s %4s",
            name.c_str(),
            speed_w, digits, d.sol60, speed_w, multi ? "--" : "",
            d.iter60, dshares.c_str(), dbest.c_str(),
            eff, pw, cclk, mclk, tmp, fan);
        // A single-device rig keeps the pool column on its one row: there is
        // no Total line under it to carry the figure.
        if (!multi) {
            std::snprintf(row, sizeof row,
                "%-17s%*.*f %*.*f %6.1f %8s %6s %8s %6s %5s %6s %5s %4s",
                name.c_str(),
                speed_w, digits, d.sol60, speed_w, digits, s.pool_sol_session,
                d.iter60, dshares.c_str(), dbest.c_str(),
                eff, pw, cclk, mclk, tmp, fan);
        }
        device_rows += row;
        device_rows += '\n';
    }

    // Total row: same missing separator after %-18s. The clock/temp/fan columns
    // stay blank, as the reference miner's own Total does -- summing a temperature is
    // meaningless and averaging one reports a figure no card measured.
    //
    // POWER IS DIFFERENT: watts add, and the rig's draw at the wall is the
    // number that decides whether a PSU is big enough and what the electricity
    // costs. So is the efficiency built from it -- total sol/s over total watts
    // is the rig's real sol/s/W, not an average of per-card ratios.
    //
    // Summed only when EVERY device reports power. A partial sum labelled
    // "Total" would understate the draw of a rig with one card NVML cannot see,
    // and understating a power total is the direction that trips a breaker.
    double total_w = 0.0;
    bool all_have_power = !rows.empty();
    for (const auto& d : rows) {
        if (d.has_power) total_w += d.power_w;
        else all_have_power = false;
    }
    char total_eff[16], total_pw[16];
    dashf(total_pw,  sizeof total_pw,  "%.0f", all_have_power, total_w);
    dashf(total_eff, sizeof total_eff, "%.3f", all_have_power && total_w > 0.0,
          s.sol60 / (total_w > 0.0 ? total_w : 1.0));

    char total_row[256];
    std::snprintf(total_row, sizeof total_row,
        "%-18s%*.*f %*.*f %6.1f %8s %6s %8s %6s",
        "Total", speed_w, digits, s.sol60, speed_w, digits, s.pool_sol_session, s.iter60,
        shares.c_str(), best.c_str(), total_eff, total_pw);

    // Dev-fee row: only when a fee is configured, so a build without one has an
    // unchanged table. What was really spent this session -- rounds, seconds, share
    // verdicts -- next to the configured rate, so a user can check it themselves.
    char devfee_row[192];
    if (s.devfee_rate > 0.0) {
        // %.4g, matching console::devfee_notice: never rounds a rate the user
        // chose into one they did not (%.1f would show 2.55% as "2.5%").
        std::snprintf(devfee_row, sizeof devfee_row,
            "Dev fee %.4g%%: %llu round%s, %.0fs total, %llu/%llu/%llu A/S/R%s",
            s.devfee_rate * 100.0,
            (unsigned long long)s.devfee_slices, s.devfee_slices == 1 ? "" : "s",
            s.devfee_seconds,
            (unsigned long long)s.devfee_accepted, (unsigned long long)s.devfee_stale,
            (unsigned long long)s.devfee_rejected,
            s.devfee_active ? "  (active now)" : "");
    } else {
        devfee_row[0] = '\0';
    }

    // Identity line: what is running, what driver it runs on, and where the API
    // is. The driver version is the component most likely to explain a hashrate
    // that moved on its own. Both halves are omitted rather than faked.
    std::string ident = "MXBM " + std::string(version);
    if (!s.driver_version.empty()) ident += ", Nvidia " + s.driver_version;
    if (api_port > 0) ident += ", API port " + std::to_string(api_port);

    std::string out;
    out.reserve(704);
    out += kRule47;   out += '\n';
    out += line2;     out += '\n';
    out += ident;     out += '\n';
    out += "Mining: BeamHash III\n";
    out += line5;     out += '\n';
    out += '\n';
    out += widen_header(kHeader1, digits);  out += '\n';
    out += widen_header(kHeader2, digits);  out += '\n';
    out += device_rows;               // already newline-terminated per row
    out += kRule27;   out += '\n';
    out += total_row; out += '\n';
    if (devfee_row[0]) { out += devfee_row; out += '\n'; }
    out += kRule47;
    return out;
}

} } // namespace mxbm::ui
