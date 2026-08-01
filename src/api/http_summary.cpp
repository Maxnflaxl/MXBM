#include "api/http_summary.h"

#include <cerrno>
#include <chrono>
#include <cstdio>

#include "net/compat.h"

#include "nlohmann/json.hpp"
#include "gpu/nvml.h"

#include "api/dashboard_html.h"
using nlohmann::json;

namespace mxbm { namespace api {

namespace {

constexpr size_t kMaxRequestBytes = 4096;   // request-line+headers cap
constexpr int kListenBacklog = 16;
constexpr int kRecvTimeoutSec = 2;          // per-connection SO_RCVTIMEO

// "Xh Ym Zs" from a whole-second duration. Not shared with ui::format.cpp's
// identical helper: mxbm_api does not link mxbm_ui (see CMakeLists.txt).
std::string format_uptime_human(std::chrono::seconds uptime) {
    long long total = uptime.count();
    long long h = total / 3600;
    long long m = (total % 3600) / 60;
    long long s = total % 60;
    char buf[32];
    std::snprintf(buf, sizeof buf, "%lldh %lldm %llds", h, m, s);
    return buf;
}

// One miner::Stats::Series as JSON. An unsampled series reports N: 0 with the
// rest null -- not 0.0, which for a power or clock figure would read as
// "measured, and it was zero".
json series_json(const miner::Stats::Series& s) {
    if (s.n == 0) {
        return json{{"N", 0}, {"Mean", nullptr}, {"Stddev", nullptr},
                    {"Min", nullptr}, {"Max", nullptr}};
    }
    return json{{"N", s.n}, {"Mean", s.mean}, {"Stddev", s.stddev()},
                {"Min", s.min}, {"Max", s.max}};
}

std::string http_response(int code, const char* reason,
                           const char* content_type, const std::string& body) {
    std::string out;
    out += "HTTP/1.1 " + std::to_string(code) + " " + reason + "\r\n";
    if (content_type) out += std::string("Content-Type: ") + content_type + "\r\n";
    out += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    out += "Connection: close\r\n\r\n";
    out += body;
    return out;
}

// Writes `data` in full, looping over short writes. SIGPIPE is ignored
// process-wide (see the header), so writing into a dropped connection just
// returns -1/EPIPE -- no MSG_NOSIGNAL guard needed.
void send_all(int fd, const std::string& data) {
    size_t sent = 0;
    while (sent < data.size()) {
        ssize_t n = send(net::from_fd(fd), data.data() + sent,
                         static_cast<int>(data.size() - sent), 0);
        if (n <= 0) return;   // dead socket
        sent += static_cast<size_t>(n);
    }
}

// Splits an HTTP request line ("METHOD SP TARGET SP VERSION") into method +
// target; the version token is skipped. Deliberately shallow -- any line
// splitting into two tokens is accepted, and the dispatcher's GET-/summary
// check (else 404) is what actually validates them.
bool parse_request_line(const std::string& line, std::string& method, std::string& target) {
    size_t sp1 = line.find(' ');
    if (sp1 == std::string::npos || sp1 == 0) return false;
    size_t sp2 = line.find(' ', sp1 + 1);
    std::string t = (sp2 == std::string::npos) ? line.substr(sp1 + 1)
                                                : line.substr(sp1 + 1, sp2 - sp1 - 1);
    if (t.empty()) return false;
    method = line.substr(0, sp1);
    target = t;
    return true;
}

} // namespace

HttpSummary::~HttpSummary() { stop(); }

bool HttpSummary::start(uint16_t port, const miner::Stats& stats, const char* version) {
    if (started_) return true;   // idempotent

    if (!net::startup()) return false;

    int fd = net::to_fd(socket(AF_INET, SOCK_STREAM, 0));
    if (fd < 0) return false;

#ifndef _WIN32
    // POSIX-only: on Windows SO_REUSEADDR means "allow hijacking the port",
    // and rebinding after a close needs no flag there anyway.
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#endif

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);   // 0.0.0.0 -- see class comment
    addr.sin_port = htons(port);                // port 0 -> kernel assigns an ephemeral port

    if (bind(net::from_fd(fd), reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) { net::close_fd(fd); return false; }
    if (listen(net::from_fd(fd), kListenBacklog) < 0) { net::close_fd(fd); return false; }

    sockaddr_in bound{};
    socklen_t boundlen = sizeof(bound);
    if (getsockname(net::from_fd(fd), reinterpret_cast<sockaddr*>(&bound), &boundlen) < 0) { net::close_fd(fd); return false; }

    // Must exist before the thread is spawned: accept_loop polls stop_pipe_[0]
    // from its very first iteration.
    if (!net::signal_pair(stop_pipe_)) { net::close_fd(fd); return false; }

    stats_ = &stats;
    version_ = version ? version : "";
    listen_fd_ = fd;
    bound_port_ = ntohs(bound.sin_port);

    // The fds are passed by value so accept_loop never reads
    // listen_fd_/stop_pipe_; stop()'s close-then-clear is their only access.
    accept_thread_ = std::thread([this, fd] { try { accept_loop(fd, stop_pipe_[0]); } catch (...) {} });
    started_ = true;
    return true;
}

void HttpSummary::stop() {
    if (!started_) return;

    // Wake accept_loop() out of poll(). Must precede the join: the loop cannot
    // see the stop request until this byte lands.
    if (stop_pipe_[1] >= 0) net::signal_send(stop_pipe_[1]);

    if (accept_thread_.joinable()) accept_thread_.join();

    // Safe only after the join: the wakeup above, not this close, ends the loop.
    if (listen_fd_ >= 0) { net::close_fd(listen_fd_); listen_fd_ = -1; }
    if (stop_pipe_[0] >= 0) { net::close_fd(stop_pipe_[0]); stop_pipe_[0] = -1; }
    if (stop_pipe_[1] >= 0) { net::close_fd(stop_pipe_[1]); stop_pipe_[1] = -1; }

    started_ = false;
}

void HttpSummary::accept_loop(int fd, int stop_read_fd) {
    // poll() on the listen socket AND the self-pipe's read end rather than
    // blocking in accept(), which has no portable wakeup (see class comment).
    net::pollfd_t fds[2] = {};
    fds[0].fd = net::from_fd(fd);
    fds[0].events = POLLIN;
    fds[1].fd = net::from_fd(stop_read_fd);
    fds[1].events = POLLIN;

    while (true) {
        int pr = net::poll_fds(fds, 2, -1);
        if (pr < 0) {
            if (net::transient(net::last_error())) continue;   // signal: retry, not fatal
            break;                                             // genuine poll() failure
        }

        if (fds[1].revents & POLLIN) break;   // stop() wrote the wakeup byte

        if (fds[0].revents & POLLIN) {
            int conn = net::to_fd(accept(net::from_fd(fd), nullptr, nullptr));
            if (conn < 0) {
                // Signal, peer reset, or a momentarily full fd table: all
                // transient, so drop this attempt, not the whole server.
                if (net::transient(net::last_error())) continue;
                break;   // non-transient
            }
            handle_connection(conn);   // closes conn itself when done
        }
    }
}

void HttpSummary::handle_connection(int conn_fd) const {
    net::set_recv_timeout(conn_fd, kRecvTimeoutSec);

    // Read until the end of the header block, bounded by the 4 KiB cap and the
    // SO_RCVTIMEO above. Both bail out the same way: header_end stays npos and
    // nothing is written back -- there is no complete request to answer.
    std::string buf;
    size_t header_end = std::string::npos;
    while (header_end == std::string::npos && buf.size() < kMaxRequestBytes) {
        char chunk[1024];
        ssize_t n = recv(net::from_fd(conn_fd), chunk, (int)sizeof(chunk), 0);
        if (n <= 0) break;   // timeout, error, or peer closed early
        buf.append(chunk, static_cast<size_t>(n));
        header_end = buf.find("\r\n\r\n");
    }

    if (header_end != std::string::npos) {
        size_t line_end = buf.find("\r\n");   // always found: header_end implies >=1 "\r\n"
        std::string line = buf.substr(0, line_end);

        std::string method, target;
        std::string response;
        if (!parse_request_line(line, method, target)) {
            response = http_response(400, "Bad Request", nullptr, "");
        } else {
            // Route on the path alone: a browser may append a query or fragment
            // this server does not use, and the raw target would 404 those.
            const std::string path = target.substr(0, target.find_first_of("?#"));
            if (method == "GET" && path == "/summary") {
                response = http_response(200, "OK", "application/json", build_body());
            } else if (method == "GET" && (path == "/" || path == "/index.html")) {
                response = http_response(200, "OK", "text/html; charset=utf-8",
                                         dashboard_html());
            } else {
                response = http_response(404, "Not Found", nullptr, "");
            }
        }
        send_all(conn_fd, response);
    }

    net::close_fd(conn_fd);
}

// ---------------------------------------------------------------------
// JSON schema. MXBM's own design -- it borrows the reference miner's /summary path but
// not its flat GPU0/GPU1 shape, grouping fields the way the console groups
// them (session totals, per-device numbers, pool/stratum state).
//
//   {
//     "Software": "MXBM <version>",
//     "Mining": { "Algorithm": "BeamHash III" },
//     "Session": {
//       "Uptime_Human": "0h 4m 5s",        // uptime, human-formatted (see note)
//       "Uptime_s": 245,                   // uptime, raw seconds
//       "Speed_15s": 0.0, "Speed_60s": 0.0, "Speed_Session": 0.0,
//       "Pool_Speed_Session": 0.0,
//       "Accepted": 0, "Stale": 0, "Rejected": 0,
//       "Best_Share": 0.0,
//       "Job_Difficulty": 0.0, "Job_Id": ""
//     },
//     "Workers": [
//       { "Index": 0, "Name": "GPU 0", "Performance": 0.0, "Iterations_s": 0.0,
//         "Power_W": null, "Core_Clock_MHz": null, "Mem_Clock_MHz": null,
//         "Temp_C": null, "Fan_Pct": null }
//     ],
//     "Stratum": { "Current_Pool": "", "Latency_ms": -1, "Reconnects": 0 },
//     "DevFee": {                        // MXBM-specific; no the reference miner counterpart
//       "Rate": 0.01, "Active": false, "Rounds": 0, "Seconds": 0.0,
//       "Accepted": 0, "Stale": 0, "Rejected": 0
//     },
//     "Session_Stats": {                 // session-long, one per sampled field
//       "Speed_60s": { "N": 65, "Mean": 55.89, "Stddev": 2.30,
//                      "Min": 39.7, "Max": 58.9 }, ...
//     },
//     "Recent_Shares": [                 // oldest first, at most kRecentShares
//       { "Age_s": 3.4, "Difficulty": 0.0, "Target": 0.0, "Dev": false }
//     ]
//   }
//
// Field notes:
// - "Uptime_Human": a formatted twin of Uptime_s, not a session-start
//   timestamp -- miner::Stats never reads wall-clock time (stats.h).
// - "Best_Share": the raw achieved-difficulty number, in the units of
//   pow::to_display_units, not a formatted "28.9G" string.
// - "Latency_ms": passed through as-is, including Stats' -1 sentinel ("no
//   share round-trip measured yet"), so a consumer can tell "never measured"
//   from "measured, 0ms".
// ---------------------------------------------------------------------
std::string HttpSummary::build_body() const {
    miner::Stats::Snapshot s = stats_->snapshot();

    json j;
    j["Software"] = "MXBM " + version_;
    j["Mining"] = { {"Algorithm", "BeamHash III"} };
    j["Session"] = {
        {"Uptime_Human", format_uptime_human(s.uptime)},
        {"Uptime_s", static_cast<int64_t>(s.uptime.count())},
        {"Speed_15s", s.sol15},
        {"Speed_60s", s.sol60},
        {"Speed_Session", s.sol_session},
        {"Pool_Speed_Session", s.pool_sol_session},
        {"Accepted", s.accepted},
        {"Stale", s.stale},
        {"Rejected", s.rejected},
        {"Best_Share", s.best_share_units},
        // The pool's CURRENT target difficulty, in the same display units as
        // Best_Share. Vardiff moves it around a lot, and a share count means
        // little without it.
        {"Job_Difficulty", s.last_job_units},
        {"Job_Id", s.last_job_id},
    };
    // Device 0's cumulative energy counter in joules, monotonic since driver load
    // (NVML; Volta+). Deliberately the RAW counter rather than a session delta:
    // rig software polls /summary and diffs consecutive readings itself, which
    // gives exact joules over any window with no sampling error. null when the
    // card or driver has no counter -- same rule as the other telemetry fields.
    {
        unsigned long long mj = 0;
        j["Energy_J"] = gpu::nvml_total_energy_mj(mj) ? json(mj / 1000.0) : json(nullptr);
    }

    // One entry per mining device. A snapshot built by hand carries no device
    // vector, so fall back to the legacy single-device fields rather than
    // serving an empty Workers array -- see the same fallback in ui/format.cpp.
    j["Workers"] = json::array();
    if (s.devices.empty()) {
        json worker;
        worker["Index"] = 0;
        worker["Name"] = s.device_label;
        worker["Performance"] = s.sol60;
        worker["Iterations_s"] = s.iter60;
        // null -- not 0, not omitted -- for a field the platform cannot supply:
        // 0 would misreport "fan unknown" as "fan stopped", and an absent key
        // would be indistinguishable from an older MXBM.
        worker["Power_W"]        = s.has_power     ? json(s.power_w)      : json(nullptr);
        worker["Core_Clock_MHz"] = s.has_sm_clock  ? json(s.sm_clock_mhz) : json(nullptr);
        worker["Mem_Clock_MHz"]  = s.has_mem_clock ? json(s.mem_clock_mhz): json(nullptr);
        worker["Temp_C"]         = s.has_temp      ? json(s.temp_c)       : json(nullptr);
        worker["Fan_Pct"]        = s.has_fan       ? json(s.fan_pct)      : json(nullptr);
        j["Workers"].push_back(worker);
    } else {
        for (size_t i = 0; i < s.devices.size(); ++i) {
            const miner::Stats::Device& d = s.devices[i];
            json worker;
            worker["Index"] = (unsigned)i;
            worker["Name"] = d.label;
            // THIS DEVICE's 60 s rate: the per-device breakdown, not the miner
            // total Session.Speed_60s carries. They coincide only with one card.
            worker["Performance"] = d.sol60;
            worker["Iterations_s"] = d.iter60;
            worker["Accepted"] = d.accepted;
            worker["Stale"]    = d.stale;
            worker["Rejected"] = d.rejected;
            worker["Best_Share"] = d.best_share_units;
            worker["Power_W"]        = d.has_power     ? json(d.power_w)      : json(nullptr);
            worker["Core_Clock_MHz"] = d.has_sm_clock  ? json(d.sm_clock_mhz) : json(nullptr);
            worker["Mem_Clock_MHz"]  = d.has_mem_clock ? json(d.mem_clock_mhz): json(nullptr);
            worker["Temp_C"]         = d.has_temp      ? json(d.temp_c)       : json(nullptr);
            worker["Fan_Pct"]        = d.has_fan       ? json(d.fan_pct)      : json(nullptr);
            j["Workers"].push_back(worker);
        }
    }

    j["Stratum"] = {
        {"Current_Pool", s.pool},
        {"Latency_ms", s.last_latency_ms},
        {"Reconnects", s.reconnects},
    };

    // Always present, all-zero in a build with no fee configured: an absent key
    // would be indistinguishable from an older MXBM. Session.Accepted/Stale/
    // Rejected above exclude these -- those are the user's own shares.
    j["DevFee"] = {
        {"Rate", s.devfee_rate},                 // fraction: 0.01 == 1.0%
        {"Active", s.devfee_active},             // a fee round is running right now
        {"Rounds", s.devfee_slices},
        {"Seconds", s.devfee_seconds},
        {"Accepted", s.devfee_accepted},
        {"Stale", s.devfee_stale},
        {"Rejected", s.devfee_rejected},
    };

    // Session-long summary of every sampled quantity. The rest of this response
    // is instantaneous and so cannot answer "how steady has it been".
    //
    // Keys mirror the field each one summarises (Session.Speed_60s ->
    // Session_Stats.Speed_60s). Sampling rules -- when a window starts being
    // sampled, and what N actually counts -- are in miner::Stats::snapshot().
    // No median: that needs the samples kept, and these are constant-memory
    // accumulators (see miner::Stats::Series).
    j["Session_Stats"] = {
        {"Speed_15s", series_json(s.series.sol15)},
        {"Speed_60s", series_json(s.series.sol60)},
        {"Iterations_s", series_json(s.series.iter60)},
        {"Power_W", series_json(s.series.power_w)},
        {"Core_Clock_MHz", series_json(s.series.sm_clock_mhz)},
        {"Mem_Clock_MHz", series_json(s.series.mem_clock_mhz)},
        {"Temp_C", series_json(s.series.temp_c)},
        {"Fan_Pct", series_json(s.series.fan_pct)},
    };

    // Per-share log, oldest first: each recent share's achieved difficulty and
    // its age in seconds as of this snapshot. Age rather than a timestamp
    // because miner::Stats never reads wall-clock time.
    j["Recent_Shares"] = json::array();
    for (const auto& sh : s.recent_shares) {
        j["Recent_Shares"].push_back(json{
            {"Age_s", sh.age_s},
            {"Difficulty", sh.units},
            // The target THIS share cleared, captured when it was found (0 when
            // unknown) -- not Session.Job_Difficulty, which is only the current
            // one and may have moved since.
            {"Target", sh.target},
            {"Dev", sh.dev},
        });
    }

    return j.dump();
}

} } // namespace mxbm::api
