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
