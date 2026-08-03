// Loopback test for api::HttpSummary. HttpSummary is the server under test, so
// this file plays the client against the ephemeral port it bound. Reads are
// SO_RCVTIMEO-bounded throughout, so a server-side bug (hang, never-close,
// busy-spin) fails the test rather than wedging the suite.
#include "check.h"
#include "api/http_summary.h"
#include "miner/stats.h"
#include "nlohmann/json.hpp"

#include <chrono>
#include <string>
#include "net/compat.h"

using namespace mxbm;
using namespace mxbm::api;
using nlohmann::json;

namespace {

// Sends `request` verbatim, then reads until the peer closes (HttpSummary always
// answers Connection: close, so EOF ends the response) or a 5s SO_RCVTIMEO
// fires. Returns "" if the connect fails, e.g. after stop() closed the listener.
std::string http_get(uint16_t port, const std::string& request) {
    net::startup();
    int fd = net::to_fd(socket(AF_INET, SOCK_STREAM, 0));
    if (fd < 0) return "";

    net::set_recv_timeout(fd, 5);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port);

    if (connect(net::from_fd(fd), reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        net::close_fd(fd);
        return "";
    }

    send(net::from_fd(fd), request.data(), (int)request.size(), 0);

    std::string resp;
    char buf[4096];
    while (true) {
        ssize_t n = recv(net::from_fd(fd), buf, (int)sizeof(buf), 0);
        if (n <= 0) break;   // EOF (Connection: close), timeout, or error
        resp.append(buf, static_cast<size_t>(n));
    }
    net::close_fd(fd);
    return resp;
}

// True if `s` starts with `prefix` -- no std::string::starts_with pre-C++20.
bool starts_with(const std::string& s, const char* prefix) {
    return s.rfind(prefix, 0) == 0;
}

} // namespace

int main() {
    miner::Stats stats;
    stats.record_attempt(4);

    HttpSummary srv;
    check(srv.start(0, stats, "0.2.0-test"), "start() succeeds on port 0");
    uint16_t port = srv.bound_port();
    check(port != 0, "bound_port() is nonzero after starting on port 0");

    // -- GET /summary: 200 + parseable JSON with the expected top-level keys --
    {
        std::string resp = http_get(port, "GET /summary HTTP/1.1\r\nHost: x\r\n\r\n");
        check(starts_with(resp, "HTTP/1.1 200"), "GET /summary returns 200");

        size_t body_start = resp.find("\r\n\r\n");
        check(body_start != std::string::npos, "response has a header/body separator");
        std::string body = (body_start != std::string::npos) ? resp.substr(body_start + 4) : "";

        bool parsed = false;
        json j;
        try { j = json::parse(body); parsed = true; } catch (...) { parsed = false; }
        check(parsed, "response body parses as JSON");

        if (parsed) {
            check(j.contains("Software"), "body has Software key");
            check(j.contains("Session"), "body has Session key");
            check(j.contains("Workers"), "body has Workers key");
            check(j.contains("Stratum"), "body has Stratum key");
            check(j["Workers"].is_array() && j["Workers"].size() == 1,
                  "Workers is a one-element array");

            // The dashboard plots these, so every speed is served as a number,
            // never as prose a consumer would have to scrape apart.
            const json& sess = j["Session"];
            check(sess.contains("Speed_15s") && sess["Speed_15s"].is_number(),
                  "Session.Speed_15s is a number the chart can plot");
            check(sess.contains("Speed_60s") && sess["Speed_60s"].is_number(),
                  "Session.Speed_60s is a number");
            check(sess.contains("Speed_Session") && sess["Speed_Session"].is_number(),
                  "Session.Speed_Session is a number");
            check(sess.contains("Pool_Speed_Session") && sess["Pool_Speed_Session"].is_number(),
                  "Session.Pool_Speed_Session is a number");
            check(j.contains("DevFee") && j["DevFee"].contains("Rate"),
                  "DevFee is always present, so 'no fee' is distinguishable from too-old");

            // This fixture installs no telemetry source. A real 0 would render
            // as "fan stopped" on the dashboard, so unknown must read as null.
            const json& w0 = j["Workers"][0];
            check(w0.contains("Temp_C") && w0["Temp_C"].is_null(),
                  "absent telemetry reports null, not a misleading 0");
            check(w0.contains("Power_W") && w0["Power_W"].is_null(),
                  "absent power reports null");
        }
    }

    // -- Recent_Shares: the per-share log the difficulty scatter plots --
    // Recorded AFTER the server is up, which also pins that the endpoint
    // reflects live state rather than a snapshot taken at start().
    {
        // The difficulty steps between shares, as pool vardiff does, so a share
        // paired with the merely-current target would be caught.
        stats.record_job("j1", 2048.0, miner::Origin::Main);
        stats.record_share_found(4096.0, miner::Origin::Main);
        stats.record_job("j2", 8192.0, miner::Origin::Main);   // vardiff steps up
        stats.record_share_found(70000.0, miner::Origin::Main);
        stats.record_job("d1", 512.0, miner::Origin::Dev);
        stats.record_share_found(1234.0, miner::Origin::Dev);

        std::string resp = http_get(port, "GET /summary HTTP/1.1\r\nHost: x\r\n\r\n");
        size_t body_start = resp.find("\r\n\r\n");
        json j = json::parse(resp.substr(body_start + 4), nullptr, false);
        check(!j.is_discarded(), "summary parses for the share-log check");
        if (!j.is_discarded()) {
            check(j.contains("Recent_Shares") && j["Recent_Shares"].is_array(),
                  "Recent_Shares is an array");
            const json& rs = j["Recent_Shares"];
            check(rs.size() == 3, "every found share is logged, dev rounds included");
            if (rs.size() == 3) {
                // Oldest first, so the scatter can be drawn without sorting.
                check(rs[0]["Difficulty"] == 4096.0 && rs[1]["Difficulty"] == 70000.0,
                      "shares are logged oldest-first at their achieved difficulty");
                check(rs[0]["Dev"] == false && rs[2]["Dev"] == true,
                      "the dev-fee tag survives into the share log");
                check(rs[0]["Age_s"].is_number() && rs[0]["Age_s"] >= 0.0,
                      "each share carries a non-negative age in seconds");
                // Pairing share 0 with the current 8192 would report 0.5x, not 2.0x.
                check(rs[0]["Target"] == 2048.0,
                      "a share keeps the target it cleared, not the current one");
                check(rs[1]["Target"] == 8192.0, "the next share carries the stepped-up target");
                check(rs[2]["Target"] == 512.0,
                      "a fee-round share carries the FEE pool's target, not the user's");
            }
            // 70000 is the max of the two Main shares, not of the Dev one.
            check(j["Session"]["Best_Share"] == 70000.0,
                  "best share stays the user's own, excluding fee-round shares");
        }
    }

    // -- GET /: the dashboard page --
    {
        std::string resp = http_get(port, "GET / HTTP/1.1\r\nHost: x\r\n\r\n");
        check(starts_with(resp, "HTTP/1.1 200"), "GET / returns 200");
        check(resp.find("Content-Type: text/html") != std::string::npos,
              "GET / is served as HTML");
        check(resp.find("<!doctype html") != std::string::npos, "GET / returns a document");

        // Rigs sit on isolated networks: a dashboard quietly needing a CDN
        // would render blank exactly where it is most needed.
        size_t body_start = resp.find("\r\n\r\n");
        std::string html = (body_start != std::string::npos) ? resp.substr(body_start + 4) : "";
        check(html.find("http://") == std::string::npos &&
              html.find("https://") == std::string::npos,
              "the dashboard references no external resources");
    }
    {
        std::string resp = http_get(port, "GET /index.html HTTP/1.1\r\nHost: x\r\n\r\n");
        check(starts_with(resp, "HTTP/1.1 200"), "GET /index.html returns 200");
    }
    {
        // A bookmark can carry a query; routing on the raw target would 404 it.
        std::string resp = http_get(port, "GET /?v=1 HTTP/1.1\r\nHost: x\r\n\r\n");
        check(starts_with(resp, "HTTP/1.1 200"), "GET / with a query string still returns 200");
        std::string r2 = http_get(port, "GET /summary?x=1 HTTP/1.1\r\nHost: x\r\n\r\n");
        check(starts_with(r2, "HTTP/1.1 200"), "GET /summary with a query string still returns 200");
    }

    // -- Session_Stats: the session-long spread, over HTTP --
    // Driven through a fake clock because the windows only start being sampled
    // once 15 s / 60 s of uptime have passed, and a test cannot wait a minute.
    {
        auto base = std::chrono::steady_clock::now();
        long long fake_ms = 0;
        stats.now_fn = [&fake_ms, base] { return base + std::chrono::milliseconds(fake_ms); };

        // Two folds at differing rates, so mean/min/max and a nonzero stddev are
        // all distinguishable from a stuck value.
        fake_ms = 61000; stats.record_attempt(60);
        stats.snapshot();
        fake_ms = 62000; stats.record_attempt(120);
        stats.snapshot();

        std::string resp = http_get(port, "GET /summary HTTP/1.1\r\nHost: x\r\n\r\n");
        size_t body_start = resp.find("\r\n\r\n");
        json j = json::parse(resp.substr(body_start + 4), nullptr, false);
        check(!j.is_discarded(), "summary parses for the Session_Stats check");
        if (!j.is_discarded()) {
            check(j.contains("Session_Stats"), "body has Session_Stats");
            const json& st = j["Session_Stats"];
            check(st.contains("Speed_60s") && st["Speed_60s"]["N"] >= 2,
                  "Speed_60s accumulates a sample per snapshot once its window is full");
            check(st["Speed_60s"]["Min"] <= st["Speed_60s"]["Mean"] &&
                  st["Speed_60s"]["Mean"] <= st["Speed_60s"]["Max"],
                  "min <= mean <= max, as any real distribution must be");
            check(st["Speed_60s"]["Stddev"] > 0.0,
                  "a spread of differing samples reports a nonzero stddev");
            check(st["Power_W"]["N"] == 0 && st["Power_W"]["Mean"].is_null(),
                  "an unsampled series reports N 0 and null, not a misleading 0.0");
        }
    }

    // -- GET /nope: 404 --
    {
        std::string resp = http_get(port, "GET /nope HTTP/1.1\r\nHost: x\r\n\r\n");
        check(starts_with(resp, "HTTP/1.1 404"), "GET /nope returns 404");
    }

    // -- unparseable request line (no space at all), not just an unknown target --
    {
        std::string resp = http_get(port, "GARBAGE\r\n\r\n");
        check(starts_with(resp, "HTTP/1.1 400"), "malformed request line returns 400");
    }

    srv.stop();

    // -- stop() closes the port --
    {
        int fd = net::to_fd(socket(AF_INET, SOCK_STREAM, 0));
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = htons(port);
        bool connected = (connect(net::from_fd(fd), reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
        net::close_fd(fd);
        check(!connected, "connect() fails after stop()");
    }

    // -- --apihost: a loopback bind still serves loopback --
    {
        HttpSummary local;
        check(local.start(0, stats, "0.2.0-test", "127.0.0.1"),
              "start() succeeds binding 127.0.0.1");
        std::string resp = http_get(local.bound_port(), "GET /summary HTTP/1.1\r\nHost: x\r\n\r\n");
        check(starts_with(resp, "HTTP/1.1 200"), "loopback-bound server answers on loopback");
        local.stop();
    }

    // -- an unparseable host is refused, not silently replaced with 0.0.0.0 --
    {
        HttpSummary bad;
        check(!bad.start(0, stats, "0.2.0-test", "not-an-address"),
              "start() fails on a host that is not dotted-quad IPv4");
        check(bad.bound_port() == 0, "a refused bind leaves bound_port() zero");
    }

    return summary("api");
}
