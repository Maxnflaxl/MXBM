// Loopback test for api::HttpSummary's hand-rolled GET /summary endpoint.
// Mirrors tests/test_transport_loopback.cpp's raw-socket pattern, just
// with the roles swapped: there the test drives a tiny hand-rolled server
// against the Transport client under test; here HttpSummary itself IS the
// server under test, so this file plays the client -- connect to the
// ephemeral port HttpSummary bound, send raw request bytes, read the raw
// response. Same SO_RCVTIMEO-bounded-read discipline either way, so a
// server-side bug (hang, never-close, busy-spin) fails the test instead of
// wedging the suite.
#include "check.h"
#include "api/http_summary.h"
#include "miner/stats.h"
#include "nlohmann/json.hpp"

#include <chrono>
#include <string>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

using namespace mxbm;
using namespace mxbm::api;
using nlohmann::json;

namespace {

// Connects to 127.0.0.1:port, sends `request` verbatim, then reads until
// the peer closes the connection (HttpSummary always answers with
// Connection: close, so EOF is the normal end-of-response signal) or a 5s
// SO_RCVTIMEO fires. Returns "" if the connect itself fails (e.g. after
// stop() has closed the listener).
std::string http_get(uint16_t port, const std::string& request) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return "";

    timeval tv{}; tv.tv_sec = 5; tv.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port);

    if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        close(fd);
        return "";
    }

    send(fd, request.data(), request.size(), 0);

    std::string resp;
    char buf[4096];
    while (true) {
        ssize_t n = recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) break;   // EOF (Connection: close), timeout, or error
        resp.append(buf, static_cast<size_t>(n));
    }
    close(fd);
    return resp;
}

// True if `s` starts with `prefix` -- no std::string::starts_with pre-C++20.
bool starts_with(const std::string& s, const char* prefix) {
    return s.rfind(prefix, 0) == 0;
}

} // namespace

int main() {
    miner::Stats stats;
    stats.record_attempt(4);   // one recorded attempt, per task brief

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

            // The fields the built-in dashboard plots. The chart needs
            // numbers, so every speed is served as one -- never as prose a
            // consumer would have to scrape back apart.
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

            // Telemetry: null -- not 0, not absent -- when the platform
            // supplies nothing. This fixture installs no telemetry source, so
            // every field must read as unknown rather than as a real zero,
            // which would render as "fan stopped" on the dashboard.
            const json& w0 = j["Workers"][0];
            check(w0.contains("Temp_C") && w0["Temp_C"].is_null(),
                  "absent telemetry reports null, not a misleading 0");
            check(w0.contains("Power_W") && w0["Power_W"].is_null(),
                  "absent power reports null");
        }
    }

    // -- Recent_Shares: the per-share log the difficulty scatter plots --
    //
    // Every other share field is an aggregate, so this is the only thing that
    // can answer "what did the last few shares look like". Recorded here
    // AFTER the server is already up, which also pins that the endpoint
    // reflects live state rather than a snapshot taken at start().
    {
        // Each share must capture the target in force AT THAT MOMENT. Step the
        // job difficulty between shares, exactly as pool vardiff does, so a
        // share paired with the merely-current target would be caught.
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
                // Origin must survive: a fee round's share is the developer's,
                // and the chart colours them apart.
                check(rs[0]["Dev"] == false && rs[2]["Dev"] == true,
                      "the dev-fee tag survives into the share log");
                check(rs[0]["Age_s"].is_number() && rs[0]["Age_s"] >= 0.0,
                      "each share carries a non-negative age in seconds");
                // The whole point: share 0 kept 2048 even though the target has
                // since stepped to 8192. Pairing it with the current target
                // would have reported it as 0.5x rather than 2.0x.
                check(rs[0]["Target"] == 2048.0,
                      "a share keeps the target it cleared, not the current one");
                check(rs[1]["Target"] == 8192.0, "the next share carries the stepped-up target");
                check(rs[2]["Target"] == 512.0,
                      "a fee-round share carries the FEE pool's target, not the user's");
            }
            // The user's best-share stat must still ignore the dev share --
            // 70000 is the max of the two Main shares, not the Dev one.
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

        // The page must depend on nothing but this miner. Rigs sit on
        // isolated networks and MXBM's whole dependency story is "a vendored
        // JSON header and system OpenSSL" -- a dashboard quietly needing a
        // CDN would render blank exactly where it is most needed.
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
        // A browser arriving from a bookmark can carry a query or fragment;
        // routing on the raw target rather than the path would 404 those.
        std::string resp = http_get(port, "GET /?v=1 HTTP/1.1\r\nHost: x\r\n\r\n");
        check(starts_with(resp, "HTTP/1.1 200"), "GET / with a query string still returns 200");
        std::string r2 = http_get(port, "GET /summary?x=1 HTTP/1.1\r\nHost: x\r\n\r\n");
        check(starts_with(r2, "HTTP/1.1 200"), "GET /summary with a query string still returns 200");
    }

    // -- Session_Stats: the session-long spread, over HTTP --
    //
    // The point of this block is that the numbers outlive any consumer's own
    // history, so it is worth pinning that they arrive populated rather than
    // as a shape full of nulls. Driven through a fake clock (the same seam
    // miner/stats.h documents) because the windows only start being sampled
    // once 15 s / 60 s of uptime have passed, and a test cannot wait a minute.
    {
        auto base = std::chrono::steady_clock::now();
        long long fake_ms = 0;
        stats.now_fn = [&fake_ms, base] { return base + std::chrono::milliseconds(fake_ms); };

        // Two folds a second apart, at rates that differ, so mean/min/max and a
        // nonzero stddev are all distinguishable from a stuck value.
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
            // Same null-not-zero rule the telemetry fields follow: this fixture
            // installs no telemetry source, so "never measured" must not read
            // as a real 0 W.
            check(st["Power_W"]["N"] == 0 && st["Power_W"]["Mean"].is_null(),
                  "an unsampled series reports N 0 and null, not a misleading 0.0");
        }
    }

    // -- GET /nope: 404 --
    {
        std::string resp = http_get(port, "GET /nope HTTP/1.1\r\nHost: x\r\n\r\n");
        check(starts_with(resp, "HTTP/1.1 404"), "GET /nope returns 404");
    }

    // -- malformed request line (no space at all -- unparseable, not just
    // an unknown target): 400 --
    {
        std::string resp = http_get(port, "GARBAGE\r\n\r\n");
        check(starts_with(resp, "HTTP/1.1 400"), "malformed request line returns 400");
    }

    srv.stop();

    // -- stop() closes the port: a fresh connect attempt must fail --
    {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = htons(port);
        bool connected = (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
        close(fd);
        check(!connected, "connect() fails after stop()");
    }

    return summary("api");
}
