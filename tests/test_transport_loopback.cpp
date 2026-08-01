// Loopback test for stratum::Transport over plain TCP (no TLS): a minimal
// echo server on 127.0.0.1 exercises the connect/send_line/recv_line
// path, including the rxbuf_ split when two lines arrive in one segment.
// TLS path is exercised live in Task 8 (against the pool), not here.
#include "check.h"
#include "stratum/transport.h"
#include <string>
#include <cstring>
#include <thread>
#include "net/compat.h"

using namespace mxbm;
using namespace mxbm::stratum;

namespace {

// Byte-at-a-time read until '\n' (inclusive) or EOF/error. Deliberately not
// the transport under test — this is the server's own tiny line reader so
// the test doesn't validate itself.
std::string recv_line_raw(int fd) {
    std::string line;
    char c;
    while (true) {
        ssize_t n = recv(net::from_fd(fd), &c, 1, 0);
        if (n <= 0) break;
        line.push_back(c);
        if (c == '\n') break;
    }
    return line;
}

// Accepts exactly one connection, echoes back the first line it receives,
// then pushes two more lines ("alpha\nbeta\n") in a single send() call to
// exercise the client's multi-line rxbuf_ split. Bounded by SO_RCVTIMEO so a
// broken client can never hang the test.
void server_thread(int listen_fd) {
    int conn = net::to_fd(accept(net::from_fd(listen_fd), nullptr, nullptr));
    net::close_fd(listen_fd);
    if (conn < 0) return;   // timeout/error: let the client-side checks fail instead of hanging

    net::set_recv_timeout(conn, 5);

    std::string line = recv_line_raw(conn);
    if (!line.empty()) send(net::from_fd(conn), line.data(), (int)line.size(), 0);

    const char two[] = "alpha\nbeta\n";
    send(net::from_fd(conn), two, (int)(sizeof(two) - 1), 0);

    net::close_fd(conn);
}

// Binds an ephemeral listening socket on 127.0.0.1, returns its fd and
// writes the assigned port to *port_out via getsockname.
int make_listener(uint16_t* port_out) {
    net::startup();
    int fd = net::to_fd(socket(AF_INET, SOCK_STREAM, 0));
#ifndef _WIN32
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#endif

    net::set_recv_timeout(fd, 5);   // bounds accept() (POSIX; a no-op for accept on Winsock)

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;   // ephemeral: kernel picks a free port

    bind(net::from_fd(fd), reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    listen(net::from_fd(fd), 1);

    sockaddr_in bound{};
    socklen_t boundlen = sizeof(bound);
    getsockname(net::from_fd(fd), reinterpret_cast<sockaddr*>(&bound), &boundlen);
    *port_out = ntohs(bound.sin_port);
    return fd;
}

} // namespace

int main() {
    uint16_t port = 0;
    int listen_fd = make_listener(&port);
    std::thread server(server_thread, listen_fd);

    Transport t;
    check(t.connect("127.0.0.1", port, /*tls=*/false, /*verify=*/false), "connect to loopback echo server");
    check(t.connected(), "connected() true after connect");

    check(t.send_line("hello"), "send_line hello");
    std::string got;
    check(t.recv_line(got), "recv_line returns true");
    check(got == "hello", "loopback echo line");

    // Framing: the server pushed "alpha\nbeta\n" in one send() call. Two
    // recv_line() calls must split it into "alpha" then "beta", proving the
    // rxbuf_ buffering/split path (not just a lucky one-line-per-read case).
    std::string first, second;
    check(t.recv_line(first) && first == "alpha", "recv_line splits first line of combined segment");
    check(t.recv_line(second) && second == "beta", "recv_line splits second line of combined segment");

    t.close();
    check(!t.connected(), "connected() false after close");

    server.join();
    return summary("transport_loopback");
}
