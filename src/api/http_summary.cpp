#include "api/http_summary.h"

#include <cerrno>
#include <chrono>
#include <cstdio>

#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include "nlohmann/json.hpp"
using nlohmann::json;

namespace mxbm { namespace api {

namespace {

constexpr size_t kMaxRequestBytes = 4096;   // request-line+headers cap, per task contract
constexpr int kListenBacklog = 16;
constexpr int kRecvTimeoutSec = 2;          // per-connection SO_RCVTIMEO, per task contract

// "Xh Ym Zs" from a whole-second duration -- deliberately re-implemented
// here rather than reusing ui::format.cpp's private helper of the same
// shape: that one is file-local (not exported via ui/format.h), and
// mxbm_api intentionally does not link mxbm_ui (see CMakeLists.txt) -- the
// /summary API is meant to stay usable without pulling in console-
// formatting code.
std::string format_uptime_human(std::chrono::seconds uptime) {
    long long total = uptime.count();
    long long h = total / 3600;
    long long m = (total % 3600) / 60;
    long long s = total % 60;
    char buf[32];
    std::snprintf(buf, sizeof buf, "%lldh %lldm %llds", h, m, s);
    return buf;
}

// One-line rollup of the three windowed candidate rates Stats tracks --
// this API's analog of the reference miner's periodic "Average speed (Ns): X sol/s"
// short-stats line, just carrying all three windows at once instead of
// picking one.
std::string format_performance_summary(const miner::Stats::Snapshot& s) {
    char buf[160];
    std::snprintf(buf, sizeof buf,
        "15s: %.2f sol/s | 60s: %.2f sol/s | session: %.2f sol/s",
        s.sol15, s.sol60, s.sol_session);
    return buf;
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

// Writes `data` in full, looping over short writes (mirrors
// stratum::Transport::send_line's own partial-write loop). SIGPIPE is
// already ignored process-wide by the time this ever runs for real (see
// main.cpp, Phase A, and this class's header comment) so a write into a
// connection the peer already dropped just returns -1/EPIPE here -- no
// MSG_NOSIGNAL/SO_NOSIGPIPE guard needed.
void send_all(int fd, const std::string& data) {
    size_t sent = 0;
    while (sent < data.size()) {
        ssize_t n = send(fd, data.data() + sent, data.size() - sent, 0);
        if (n <= 0) return;   // dead socket: nothing more we can do, drop it
        sent += static_cast<size_t>(n);
    }
}

// Splits an HTTP request line ("METHOD SP TARGET SP VERSION") into method
// + target. The HTTP-version token is skipped past but not validated or
// kept -- nothing downstream needs it. Returns false (method/target left
// untouched) if the line doesn't even contain a method and a target
// separated by a space -- e.g. a line with no space at all. This is a
// deliberately shallow parse: a line that DOES split into two tokens is
// accepted as well-formed even if `method` isn't a real HTTP verb or
// `target` isn't a real path -- the dispatcher's GET-/summary check (or
// lack of a match, landing on 404) is what actually validates those.
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
    if (started_) return true;   // no-op if already started (mirrors Engine/Ticker)

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return false;

    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);   // 0.0.0.0 -- see class comment
    addr.sin_port = htons(port);                // port 0 -> kernel assigns an ephemeral port

    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) { ::close(fd); return false; }
    if (listen(fd, kListenBacklog) < 0) { ::close(fd); return false; }

    sockaddr_in bound{};
    socklen_t boundlen = sizeof(bound);
    if (getsockname(fd, reinterpret_cast<sockaddr*>(&bound), &boundlen) < 0) { ::close(fd); return false; }

    // Self-pipe for the accept loop's poll()-based wakeup -- see class
    // comment. Created before the thread is spawned so accept_loop always
    // has a valid stop_pipe_[0] to poll on from its very first iteration.
    if (::pipe(stop_pipe_) < 0) { ::close(fd); return false; }

    stats_ = &stats;
    version_ = version ? version : "";
    listen_fd_ = fd;
    bound_port_ = ntohs(bound.sin_port);

    // Pass fd and the pipe's read end by value into the thread rather than
    // having accept_loop re-read listen_fd_/stop_pipe_ every iteration:
    // std::thread's constructor copies both arguments before the new
    // thread starts running, so accept_loop never touches those members at
    // all -- the only writes to them after this point are stop()'s
    // close-then-clear, on the caller's thread, with nothing else reading
    // those members concurrently.
    accept_thread_ = std::thread(&HttpSummary::accept_loop, this, fd, stop_pipe_[0]);
    started_ = true;
    return true;
}

void HttpSummary::stop() {
    if (!started_) return;

    // Wake accept_loop() out of poll() -- see class comment. Write before
    // join (the loop can't see the stop request until this lands), and
    // ignore the return: this is a best-effort single-byte wakeup into a
    // pipe with plenty of buffer headroom, and there is nothing more
    // useful to do with a failed write here than let join() below still
    // wait for the thread to actually exit.
    if (stop_pipe_[1] >= 0) { (void)::write(stop_pipe_[1], "x", 1); }

    if (accept_thread_.joinable()) accept_thread_.join();

    // Only close fds once the thread is confirmed gone -- accept_loop no
    // longer needs listen_fd_/stop_pipe_[0] to be closed to wake up (the
    // pipe write above already did that), so closing here can't race the
    // loop's own poll()/accept() calls.
    if (listen_fd_ >= 0) { ::close(listen_fd_); listen_fd_ = -1; }
    if (stop_pipe_[0] >= 0) { ::close(stop_pipe_[0]); stop_pipe_[0] = -1; }
    if (stop_pipe_[1] >= 0) { ::close(stop_pipe_[1]); stop_pipe_[1] = -1; }

    started_ = false;
}

void HttpSummary::accept_loop(int fd, int stop_read_fd) {
    // poll() on the listen socket AND the self-pipe's read end, rather
    // than blocking directly in accept() -- see class comment for why:
    // closing listen_fd_ alone (the old design) is not a portable wakeup
    // for a thread parked in accept() on Linux, but a pipe write always
    // wakes a poll() blocked on that pipe's read end.
    struct pollfd fds[2] = {};
    fds[0].fd = fd;
    fds[0].events = POLLIN;
    fds[1].fd = stop_read_fd;
    fds[1].events = POLLIN;

    while (true) {
        int pr = ::poll(fds, 2, -1);
        if (pr < 0) {
            if (errno == EINTR) continue;   // interrupted by a signal: retry, not fatal
            break;                          // genuine poll() failure: nothing left to serve
        }

        if (fds[1].revents & POLLIN) break;   // stop() wrote the wakeup byte: exit cleanly

        if (fds[0].revents & POLLIN) {
            int conn = accept(fd, nullptr, nullptr);
            if (conn < 0) {
                // EINTR: a signal landed between poll() and accept() --
                // retry. ECONNABORTED: the peer reset the connection
                // before we could finish accepting it -- nothing to hand
                // off. EMFILE/ENFILE: the process/system fd table is
                // momentarily full. All three are transient: skip this
                // accept attempt and keep serving rather than killing the
                // whole server over one bad connection attempt.
                if (errno == EINTR || errno == ECONNABORTED ||
                    errno == EMFILE || errno == ENFILE) {
                    continue;
                }
                break;   // anything else: a genuine, non-transient error
            }
            handle_connection(conn);   // closes conn itself when done
        }
    }
}

void HttpSummary::handle_connection(int conn_fd) const {
    timeval tv{};
    tv.tv_sec = kRecvTimeoutSec;
    tv.tv_usec = 0;
    setsockopt(conn_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    // Read until "\r\n\r\n" (the end of the header block) or the request
    // gives up first -- 4 KiB cap so a client that never sends a blank
    // line can't grow this buffer without bound, 2s SO_RCVTIMEO (set
    // above) so a slow/silent client can't block this thread forever
    // either. Both bail out the SAME way: header_end is left npos, and no
    // response is written at all (see below) -- there is no complete
    // request to answer.
    std::string buf;
    size_t header_end = std::string::npos;
    while (header_end == std::string::npos && buf.size() < kMaxRequestBytes) {
        char chunk[1024];
        ssize_t n = recv(conn_fd, chunk, sizeof(chunk), 0);
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
        } else if (method == "GET" && target == "/summary") {
            response = http_response(200, "OK", "application/json", build_body());
        } else {
            response = http_response(404, "Not Found", nullptr, "");
        }
        send_all(conn_fd, response);
    }

    ::close(conn_fd);
}

// ---------------------------------------------------------------------
// JSON schema -- BEST-EFFORT, UNVERIFIED against a real the reference miner /summary
// response. the reference miner's own --apiport docs confirm the *path* (`/summary`)
// and turned up exactly one real example, from a 2018/v0.4 the reference miner (flat
// top-level metadata + one object per GPU keyed "GPU0"/"GPU1"/...) --
// explicitly flagged as stale (temp/fan/power/clocks, junction/memory
// temps, and worker name were all added to the API in later versions per
// the README's own changelog) and not attempted to be reproduced
// byte-for-byte here. What follows is MXBM's own schema design, grouping
// fields the way the reference miner's *console* output groups them (session totals
// vs. per-device numbers vs. pool/stratum state) rather
// than the reference miner's flat GPU0/GPU1 shape. Open item: capture a real
// the reference miner --apiport response (`curl localhost:PORT/summary` against a
// live binary) and reconcile field names/nesting against it.
//
//   {
//     "Software": "MXBM <version>",
//     "Mining": { "Algorithm": "BeamHash III" },
//     "Session": {
//       "Uptime_Human": "0h 4m 5s",        // uptime, human-formatted (see note)
//       "Uptime_s": 245,                   // uptime, raw seconds
//       "Performance_Summary": "15s: .. | 60s: .. | session: ..",
//       "Accepted": 0, "Stale": 0, "Rejected": 0,
//       "Best_Share": 0.0
//     },
//     "Workers": [
//       { "Index": 0, "Name": "CPU 0 reference", "Performance": 0.0, "Iterations_s": 0.0 }
//     ],
//     "Stratum": { "Current_Pool": "", "Latency_ms": -1, "Reconnects": 0 }
//   }
//
// Notes on specific fields (why they're not more the reference miner-literal):
// - "Uptime_Human": the reference miner's real (2018) schema has a same-shaped
//   *"Startup"* field holding an absolute wall-clock timestamp
//   ("2018-07-22 22:08:37"). miner::Stats deliberately never touches
//   wall-clock time -- every time read inside it goes through its
//   steady_clock-based now_fn seam so tests stay deterministic (see
//   stats.h's class comment) -- and this class's only inputs are
//   `stats.snapshot()` + `version`, so there is no session-start
//   wall-clock value available here to report honestly. Rather than
//   fabricate one (e.g. `system_clock::now() - uptime`, which would LOOK
//   like a real captured timestamp but is really just a round-tripped
//   re-derivation of the same uptime, off by request latency and exposed
//   to system-clock skew), this reports the same duration Uptime_s
//   carries, just human-formatted. Keeping the reference miner's "Startup" key name
//   on a value that isn't a timestamp would be actively misleading --
//   it would read as session-start-time to any consumer going by the name
//   alone, while actually just duplicating Uptime_s in another format --
//   so the key itself is named for what it actually holds: "Uptime_Human",
//   a formatted twin of the numeric Uptime_s right next to it. Less
//   the reference miner-literal, but not misleadingly named about something this
//   endpoint doesn't actually know.
// - "Best_Share": the raw achieved-difficulty number
//   (miner::Stats::Snapshot::best_share_units, same units as
//   pow::to_display_units/achieved_units), not a the reference miner-style formatted
//   "28.9G" magnitude string -- ui::format_units() isn't reachable here
//   (mxbm_api doesn't link mxbm_ui; see class comment), and a raw number
//   is arguably more useful to a JSON API consumer than a pre-formatted
//   suffix anyway.
// - "Latency_ms": passed through as-is, including the -1 sentinel Stats
//   itself documents ("no share round-trip measured yet", stats.h) --
//   not remapped to null/0, so a consumer can distinguish "never
//   measured" from "measured, 0ms".
// - Single worker row: Phase B has exactly one "device", the CPU
//   reference solver (same "CPU 0 reference" label ui::format_stats_block
//   uses for its console table row) -- no GPU fields (power/clock/temp)
//   are included at all, rather than populated with fake numbers. M3's
//   GPU backend is what will eventually turn Workers into a real
//   per-device array.
// ---------------------------------------------------------------------
std::string HttpSummary::build_body() const {
    miner::Stats::Snapshot s = stats_->snapshot();

    json j;
    j["Software"] = "MXBM " + version_;
    j["Mining"] = { {"Algorithm", "BeamHash III"} };
    j["Session"] = {
        {"Uptime_Human", format_uptime_human(s.uptime)},
        {"Uptime_s", static_cast<int64_t>(s.uptime.count())},
        {"Performance_Summary", format_performance_summary(s)},
        {"Accepted", s.accepted},
        {"Stale", s.stale},
        {"Rejected", s.rejected},
        {"Best_Share", s.best_share_units},
    };

    json worker;
    worker["Index"] = 0;
    worker["Name"] = "CPU 0 reference";
    worker["Performance"] = s.sol60;
    worker["Iterations_s"] = s.iter60;
    j["Workers"] = json::array();
    j["Workers"].push_back(worker);

    j["Stratum"] = {
        {"Current_Pool", s.pool},
        {"Latency_ms", s.last_latency_ms},
        {"Reconnects", s.reconnects},
    };

    return j.dump();
}

} } // namespace mxbm::api
