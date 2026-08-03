#include "ui/format.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <vector>

namespace mxbm { namespace ui {

namespace {

// "Xh Ym Zs" from a whole-second duration.
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

const char* const kRule47 =
    "-----------------------------------------------";
const char* const kRule27 = "---------------------------";

constexpr int kNameW = 17;
constexpr int kDefaultDigits = 2;

std::string pad_left(const std::string& s, int w) {
    if ((int)s.size() >= w) return s;
    return std::string((size_t)(w - (int)s.size()), ' ') + s;
}

std::string pad_right(const std::string& s, int w) {
    if ((int)s.size() >= w) return s;
    return s + std::string((size_t)(w - (int)s.size()), ' ');
}

std::string num(const char* fmt, double v) {
    char b[32];
    std::snprintf(b, sizeof b, fmt, v);
    return b;
}

// A value the platform did not report. Never 0: a fan that cannot be read and a
// fan that is stopped are different facts.
const char* const kMissing = "--";

// The Name column is 17 wide. A full device string ("NVIDIA GeForce RTX 4070 Ti
// SUPER", 32 chars) overflows it, so drop the vendor prefix -- which carries no
// information in a per-device row -- and truncate only if what remains still
// does not fit.
std::string short_device_name(const std::string& full) {
    static const char* kPrefixes[] = {
        "NVIDIA GeForce ", "NVIDIA ", "AMD Radeon ", "AMD ",
        "Advanced Micro Devices, Inc. ", "Intel(R) ", "Intel ",
    };
    std::string n = full;
    for (const char* p : kPrefixes) {
        const size_t len = std::strlen(p);
        if (n.compare(0, len, p) == 0) { n.erase(0, len); break; }
    }
    // Plain truncation, not an ellipsis: padding counts BYTES and a multi-byte
    // glyph would occupy one column while counting as three.
    if (n.size() > (size_t)kNameW) n.resize((size_t)kNameW);
    return n;
}

// The per-device figures the columns read, gathered once so the aggregate
// (a hand-built snapshot with no device vector) and the per-device case share
// one code path.
struct RowData {
    std::string label;
    double sol60 = 0.0, iter60 = 0.0;
    uint64_t accepted = 0, stale = 0, rejected = 0;
    double best_share_units = 0.0;
    bool has_power = false;     double power_w = 0.0;
    bool has_sm_clock = false;  unsigned sm_clock_mhz = 0;
    bool has_mem_clock = false; unsigned mem_clock_mhz = 0;
    bool has_temp = false;      unsigned temp_c = 0;
    bool has_fan = false;       unsigned fan_pct = 0;
    bool has_util = false;      unsigned util_pct = 0;
    bool paused = false;
};

RowData row_from(const miner::Stats::Device& d) {
    RowData r;
    r.label = d.label;
    r.sol60 = d.sol60;  r.iter60 = d.iter60;
    r.accepted = d.accepted; r.stale = d.stale; r.rejected = d.rejected;
    r.best_share_units = d.best_share_units;
    r.has_power = d.has_power;         r.power_w = d.power_w;
    r.has_sm_clock = d.has_sm_clock;   r.sm_clock_mhz = d.sm_clock_mhz;
    r.has_mem_clock = d.has_mem_clock; r.mem_clock_mhz = d.mem_clock_mhz;
    r.has_temp = d.has_temp;           r.temp_c = d.temp_c;
    r.has_fan = d.has_fan;             r.fan_pct = d.fan_pct;
    r.has_util = d.has_util;           r.util_pct = d.util_pct;
    r.paused = d.paused;
    return r;
}

// Everything the Total row needs that is not a per-device sum.
struct TotalData {
    double sol60 = 0.0, pool_sol = 0.0, iter60 = 0.0;
    uint64_t accepted = 0, stale = 0, rejected = 0;
    double best_share_units = 0.0;
    bool has_power = false;  double power_w = 0.0;
    double uptime_s = 0.0;
};

} // namespace

const std::vector<StatColumn>& stat_columns() {
    // Widths are the table's contract: both header lines and every row pad to
    // these, so a column added here lines up without touching anything else.
    // `total` false leaves the Total cell blank -- summing a temperature is
    // meaningless, and averaging one reports a figure no card measured.
    static const std::vector<StatColumn> kCols = {
        {"gpuName",      "Name",   "",        kNameW, true,  "Name"},
        {"speed",        "Speed",  "sol/s",   6,      true,  "Speed (sol/s)"},
        {"poolHr",       "Pool",   "sol/s",   6,      true,  "Pool (sol/s)"},
        {"iter",         "Iter.",  "it/s",    6,      true,  "Iter. (it/s)"},
        {"shares",       "Shares", "A/S/R",   8,      true,  "Shares (A/S/R)"},
        {"bestShare",    "Best",   "Share",   6,      true,  "Best Share"},
        {"hrPerWatt",    "Eff.",   "sol/s/W", 8,      true,  "Efficiency (sol/s/W)"},
        {"power",        "Power",  "W",       6,      true,  "Power (W)"},
        {"coreClk",      "CCLK",   "MHz",     5,      false, "Core Clock (MHz)"},
        {"memClk",       "MCLK",   "MHz",     6,      false, "Memory Clock (MHz)"},
        {"coreT",        "Core",   "Temp",    5,      false, "Temp (deg C)"},
        {"fanPct",       "Fan",    "Pct",     4,      false, "Fan Speed (%)"},
        // Past the default set: selectable with --statsformat, absent unless asked
        // for. `state` and `util` have no counterpart in other miners' field
        // lists; they are ours because we have the data.
        {"sharesPerMin", "Shr/m",  "min",     6,      true,  "Shares per minute"},
        {"wattPerHr",    "W per",  "sol/s",   7,      true,  "Watt per sol/s"},
        {"util",         "Util",   "Pct",     5,      false, "Utilisation (%)"},
        {"state",        "State",  "",        7,      false, "State"},
    };
    return kCols;
}

const StatColumn* stat_column(const std::string& name) {
    for (const StatColumn& c : stat_columns()) {
        if (name.size() != std::strlen(c.name)) continue;
        bool same = true;
        for (size_t i = 0; i < name.size(); ++i) {
            if (std::tolower((unsigned char)name[i]) != std::tolower((unsigned char)c.name[i])) {
                same = false;
                break;
            }
        }
        if (same) return &c;
    }
    return nullptr;
}

std::vector<std::string> default_stat_format() {
    std::vector<std::string> out;
    for (const StatColumn& c : stat_columns()) {
        out.push_back(c.name);
        if (std::strcmp(c.name, "fanPct") == 0) break;   // the default set ends here
    }
    return out;
}

bool parse_stats_format(const std::string& spec, std::vector<std::string>& out,
                        std::string& err) {
    out.clear();
    err.clear();
    size_t pos = 0;
    for (;;) {
        const size_t comma = spec.find(',', pos);
        std::string tok = spec.substr(pos, comma == std::string::npos ? std::string::npos
                                                                      : comma - pos);
        while (!tok.empty() && (tok.front() == ' ' || tok.front() == '\t')) tok.erase(0, 1);
        while (!tok.empty() && (tok.back()  == ' ' || tok.back()  == '\t')) tok.pop_back();
        if (tok.empty()) { err = "empty field in the --statsformat list"; return false; }
        const StatColumn* c = stat_column(tok);
        if (!c) {
            // There are no presets, so a bare word is simply an unknown field --
            // and naming the fields beats leaving the operator to guess.
            err = "'" + tok + "' is not a statistics field. MXBM has no presets; list "
                  "fields, e.g. gpuName,speed,power,coreT. Available: ";
            const std::vector<StatColumn>& all = stat_columns();
            for (size_t i = 0; i < all.size(); ++i) {
                err += all[i].name;
                if (i + 1 < all.size()) err += ", ";
            }
            return false;
        }
        out.push_back(c->name);
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    if (out.empty()) { err = "the --statsformat list selected nothing"; return false; }
    return true;
}

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

namespace {

bool is_name_col(const StatColumn& c) { return std::strcmp(c.name, "gpuName") == 0; }

// One cell, unpadded. `index` is the device's own; `multi` says whether the rig
// has more than one card, which decides whether the pool rate can be attributed
// to a row at all.
std::string cell_device(const StatColumn& c, const RowData& d, const TotalData& t,
                        size_t index, bool multi, int digits) {
    const std::string n = c.name;
    const std::string rate_fmt = "%." + std::to_string(digits) + "f";
    if (n == "gpuName") {
        std::string name = short_device_name(d.label);
        if (multi) {
            char pfx[24];
            std::snprintf(pfx, sizeof pfx, "GPU %zu ", index);
            name = std::string(pfx) + name;
            if (name.size() > (size_t)kNameW) name.resize((size_t)kNameW);
        }
        return name;
    }
    if (n == "speed")  return num(rate_fmt.c_str(), d.sol60);
    // The pool rate is a property of the CONNECTION, not of a card -- the pool
    // credits shares without saying which GPU found them -- so a multi-card rig
    // carries it on the Total row only.
    if (n == "poolHr") return multi ? kMissing : num(rate_fmt.c_str(), t.pool_sol);
    if (n == "iter")   return num("%.1f", d.iter60);
    if (n == "shares") return format_shares(d.accepted, d.stale, d.rejected);
    if (n == "bestShare") return format_units(d.best_share_units);
    if (n == "hrPerWatt")
        return (d.has_power && d.power_w > 0.0) ? num("%.3f", d.sol60 / d.power_w) : kMissing;
    if (n == "wattPerHr")
        return (d.has_power && d.sol60 > 0.0) ? num("%.2f", d.power_w / d.sol60) : kMissing;
    if (n == "power")   return d.has_power ? num("%.0f", d.power_w) : kMissing;
    if (n == "coreClk") return d.has_sm_clock ? num("%.0f", (double)d.sm_clock_mhz) : kMissing;
    if (n == "memClk")  return d.has_mem_clock ? num("%.0f", (double)d.mem_clock_mhz) : kMissing;
    if (n == "coreT")   return d.has_temp ? num("%.0f", (double)d.temp_c) : kMissing;
    if (n == "fanPct")  return d.has_fan ? num("%.0f", (double)d.fan_pct) : kMissing;
    if (n == "util")    return d.has_util ? num("%.0f", (double)d.util_pct) : kMissing;
    if (n == "sharesPerMin")
        return t.uptime_s > 0.0 ? num("%.2f", (double)d.accepted * 60.0 / t.uptime_s) : kMissing;
    if (n == "state")   return d.paused ? "paused" : "mining";
    return kMissing;
}

std::string cell_total(const StatColumn& c, const TotalData& t, int digits) {
    if (!c.total) return std::string();
    const std::string n = c.name;
    const std::string rate_fmt = "%." + std::to_string(digits) + "f";
    if (n == "gpuName") return "Total";
    if (n == "speed")   return num(rate_fmt.c_str(), t.sol60);
    if (n == "poolHr")  return num(rate_fmt.c_str(), t.pool_sol);
    if (n == "iter")    return num("%.1f", t.iter60);
    if (n == "shares")  return format_shares(t.accepted, t.stale, t.rejected);
    if (n == "bestShare") return format_units(t.best_share_units);
    // Watts add, and the rig's draw is what decides whether a PSU is big enough.
    // Summed only when EVERY device reports power: a partial sum labelled "Total"
    // understates the draw, and understating a power total trips breakers.
    if (n == "hrPerWatt")
        return (t.has_power && t.power_w > 0.0) ? num("%.3f", t.sol60 / t.power_w) : kMissing;
    if (n == "wattPerHr")
        return (t.has_power && t.sol60 > 0.0) ? num("%.2f", t.power_w / t.sol60) : kMissing;
    if (n == "power")   return t.has_power ? num("%.0f", t.power_w) : kMissing;
    if (n == "sharesPerMin")
        return t.uptime_s > 0.0 ? num("%.2f", (double)t.accepted * 60.0 / t.uptime_s) : kMissing;
    return std::string();
}

// --digits widens the two rate columns and nothing else, so the headers over them
// grow by the same amount or the table shears.
int column_width(const StatColumn& c, int digits) {
    const std::string n = c.name;
    const int extra = digits > kDefaultDigits ? digits - kDefaultDigits : 0;
    if (n == "speed" || n == "poolHr") return c.width + extra;
    return c.width;
}

// One horizontal group: two header lines, a row per device, a rule, the Total row.
void emit_group(std::string& out, const std::vector<const StatColumn*>& cols,
                const std::vector<RowData>& rows, const TotalData& total,
                bool multi, int digits, bool label_narrow) {
    const bool leads_with_name = !cols.empty() && is_name_col(*cols[0]);
    auto width_of = [&](size_t i) {
        return is_name_col(*cols[i]) ? (label_narrow ? 6 : kNameW)
                                     : column_width(*cols[i], digits);
    };
    // The name column is left-aligned and abuts the first value with no separator;
    // every other pair is one space apart. That single missing space is what the
    // header alignment has always encoded.
    auto compose = [&](const std::vector<std::string>& cells, bool name_left) {
        std::string line;
        for (size_t i = 0; i < cols.size(); ++i) {
            if (i > 0 && !(i == 1 && leads_with_name)) line += ' ';
            const bool nc = is_name_col(*cols[i]);
            line += (nc && name_left) ? pad_right(cells[i], width_of(i))
                                      : pad_left(cells[i], width_of(i));
        }
        while (!line.empty() && line.back() == ' ') line.pop_back();
        return line;
    };

    std::vector<std::string> h1, h2;
    for (const StatColumn* c : cols) {
        // "      Name" sits inside its own field rather than being padded like a
        // value, which is why the header is emitted left-aligned.
        h1.push_back(is_name_col(*c) ? (label_narrow ? std::string() : std::string("      Name"))
                                     : std::string(c->head1));
        h2.push_back(is_name_col(*c) ? std::string() : std::string(c->head2));
    }
    out += compose(h1, /*name_left=*/true); out += '\n';
    out += compose(h2, /*name_left=*/true); out += '\n';

    for (size_t r = 0; r < rows.size(); ++r) {
        std::vector<std::string> cells;
        for (const StatColumn* c : cols) {
            if (label_narrow && is_name_col(*c)) {
                char pfx[16];
                std::snprintf(pfx, sizeof pfx, "GPU %zu", r);
                cells.push_back(pfx);
            } else {
                cells.push_back(cell_device(*c, rows[r], total, r, multi, digits));
            }
        }
        out += compose(cells, /*name_left=*/true); out += '\n';
    }

    out += kRule27; out += '\n';
    std::vector<std::string> tcells;
    for (const StatColumn* c : cols) tcells.push_back(cell_total(*c, total, digits));
    out += compose(tcells, /*name_left=*/true); out += '\n';
}

// Vertical table: one column per device plus a Total, one row per field.
void emit_vertical(std::string& out, const std::vector<const StatColumn*>& cols,
                   const std::vector<RowData>& rows, const TotalData& total,
                   bool multi, int digits) {
    size_t label_w = 0;
    for (const StatColumn* c : cols) label_w = std::max(label_w, std::strlen(c->vlabel) + 2);
    constexpr int kCellW = 12;

    std::string head = std::string(label_w, ' ');
    for (size_t r = 0; r < rows.size(); ++r) {
        char b[16];
        std::snprintf(b, sizeof b, "GPU %zu", r);
        head += pad_left(b, kCellW);
    }
    head += pad_left("Total", kCellW);
    out += head; out += '\n';

    for (const StatColumn* c : cols) {
        std::string line = pad_right(std::string(c->vlabel) + ":", (int)label_w);
        for (size_t r = 0; r < rows.size(); ++r) {
            std::string v = is_name_col(*c) ? short_device_name(rows[r].label)
                                            : cell_device(*c, rows[r], total, r, multi, digits);
            if ((int)v.size() > kCellW) v.resize((size_t)kCellW);
            line += pad_left(v, kCellW);
        }
        std::string tv = is_name_col(*c) ? std::string("Rig") : cell_total(*c, total, digits);
        if ((int)tv.size() > kCellW) tv.resize((size_t)kCellW);
        line += pad_left(tv, kCellW);
        while (!line.empty() && line.back() == ' ') line.pop_back();
        out += line; out += '\n';
    }
}

} // namespace

std::string format_stats_block(const miner::Stats::Snapshot& s,
                                const char* version,
                                const char* clock_hhmmss,
                                int digits,
                                int api_port) {
    StatsLayout layout;
    layout.digits = digits;
    layout.api_port = api_port;
    return format_stats_block(s, version, clock_hhmmss, layout);
}

std::string format_stats_block(const miner::Stats::Snapshot& s,
                                const char* version,
                                const char* clock_hhmmss,
                                const StatsLayout& layout) {
    const int digits = layout.digits;

    // Latency shows the connect-handshake duration until the first share result
    // comes back, then the measured submit-to-result latency.
    const long long latency_ms = (s.last_latency_ms >= 0) ? s.last_latency_ms : s.connect_ms;

    char line2[128];
    std::snprintf(line2, sizeof line2, "Statistics (%s); Uptime: %s",
        clock_hhmmss, format_uptime(s.uptime).c_str());

    char line5[256];
    std::snprintf(line5, sizeof line5, "Connected to: %s (%lldms latency)",
        s.pool.c_str(), latency_ms);

    // A Snapshot built by hand -- a test, or any consumer predating the device
    // vector -- carries the legacy single-device fields and no rows. Synthesise
    // the one row they describe rather than printing a table with no devices.
    std::vector<RowData> rows;
    for (const auto& d : s.devices) rows.push_back(row_from(d));
    if (rows.empty()) {
        RowData d;
        d.label = s.device_label;
        d.sol60 = s.sol60;  d.iter60 = s.iter60;
        d.accepted = s.accepted; d.stale = s.stale; d.rejected = s.rejected;
        d.best_share_units = s.best_share_units;
        d.has_power = s.has_power;         d.power_w = s.power_w;
        d.has_sm_clock = s.has_sm_clock;   d.sm_clock_mhz = s.sm_clock_mhz;
        d.has_mem_clock = s.has_mem_clock; d.mem_clock_mhz = s.mem_clock_mhz;
        d.has_temp = s.has_temp;           d.temp_c = s.temp_c;
        d.has_fan = s.has_fan;             d.fan_pct = s.fan_pct;
        rows.push_back(std::move(d));
    }
    const bool multi = rows.size() > 1;

    TotalData total;
    total.sol60 = s.sol60;
    total.pool_sol = s.pool_sol_session;
    total.iter60 = s.iter60;
    total.accepted = s.accepted; total.stale = s.stale; total.rejected = s.rejected;
    total.best_share_units = s.best_share_units;
    total.uptime_s = (double)s.uptime.count();
    total.has_power = !rows.empty();
    for (const RowData& d : rows) {
        if (d.has_power) total.power_w += d.power_w;
        else total.has_power = false;
    }

    const std::vector<std::string> names = layout.columns.empty() ? default_stat_format()
                                                                  : layout.columns;
    std::vector<const StatColumn*> cols;
    for (const std::string& n : names) {
        if (const StatColumn* c = stat_column(n)) cols.push_back(c);
    }
    if (cols.empty()) return std::string();

    std::string table;
    if (layout.vertical) {
        emit_vertical(table, cols, rows, total, multi, digits);
    } else if (layout.wrap_width > 0) {
        // Column groups that fit the width, each repeating the label column and
        // the Total row so a group stands on its own.
        const bool has_name = is_name_col(*cols[0]);
        size_t i = has_name ? 1 : 0;
        bool first_group = true;
        while (i < cols.size()) {
            std::vector<const StatColumn*> group;
            int width = 0;
            if (has_name) {
                group.push_back(cols[0]);
                width = first_group ? kNameW : 6;
            }
            while (i < cols.size()) {
                if (is_name_col(*cols[i])) { ++i; continue; }
                const int w = column_width(*cols[i], digits) + 1;
                if (group.size() > (has_name ? 1u : 0u) && width + w > layout.wrap_width) break;
                group.push_back(cols[i]);
                width += w;
                ++i;
            }
            if (group.empty() || (has_name && group.size() == 1)) break;
            if (!first_group) table += '\n';
            emit_group(table, group, rows, total, multi, digits,
                       /*label_narrow=*/!first_group && has_name);
            first_group = false;
        }
    } else {
        emit_group(table, cols, rows, total, multi, digits, /*label_narrow=*/false);
    }

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
    if (layout.api_port > 0) ident += ", API port " + std::to_string(layout.api_port);

    std::string out;
    out.reserve(704);
    out += kRule47;   out += '\n';
    out += line2;     out += '\n';
    out += ident;     out += '\n';
    out += "Mining: BeamHash III\n";
    out += line5;     out += '\n';
    out += '\n';
    out += table;                     // already newline-terminated
    if (devfee_row[0]) { out += devfee_row; out += '\n'; }
    out += kRule47;
    return out;
}

} } // namespace mxbm::ui
