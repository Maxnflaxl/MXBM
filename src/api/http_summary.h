// GET /summary hand-rolled HTTP API: a minimal HTTP/1.1 server reporting a
// live miner::Stats snapshot as JSON, MXBM's analog of the reference miner's --apiport.
// The schema is documented on build_body() in http_summary.cpp.
//
// Lifecycle mirrors miner::Engine/ui::Ticker (start() spawns one worker
// thread, stop() signals it and joins, the destructor calls stop()). The
// accept-loop thread blocks in poll() on TWO file descriptors -- the listen
// socket and the read end of a wakeup channel (stop_pipe_: a self-pipe on
// POSIX, a loopback socket pair on Windows; net/compat.h) -- rather than in
// accept(). stop() writes one byte to it, which wakes poll()
// deterministically, THEN joins, THEN closes the listen fd and both ends.
// The wakeup channel is what makes this portable: closing the listen fd does
// not reliably wake a concurrent accept() on Linux (it does on macOS/BSD), so
// stop() could otherwise hang in join() until the next inbound connection.
//
// Binds 0.0.0.0 by design, matching the reference miner's --apihost default: an
// UNAUTHENTICATED, LAN-visible read-only endpoint. An operator who wants it
// local-only can firewall the port.
//
// Every response is `Connection: close` -- one request per connection, no
// keep-alive, no pipelining, no chunked bodies.
//
// SIGPIPE is ignored process-wide before this class is ever used (main.cpp;
// Windows has no SIGPIPE at all), so a write into a dropped connection just
// returns an error -- no per-write MSG_NOSIGNAL guard is needed (contrast
// stratum::Transport::raw_write, which can be used before that point, e.g. in
// unit tests).
#pragma once
#include <cstdint>
#include <string>
#include <thread>

#include "miner/stats.h"

namespace mxbm { namespace api {

class HttpSummary {
public:
    HttpSummary() = default;
    ~HttpSummary();

    HttpSummary(const HttpSummary&) = delete;
    HttpSummary& operator=(const HttpSummary&) = delete;

    // Binds 0.0.0.0:port (port 0 -> kernel picks an ephemeral port; read it
    // back via bound_port()) and spawns the accept-loop thread. `stats` and
    // `version` must outlive the HttpSummary, or at least outlive stop(): the
    // stats are held by pointer and snapshotted fresh on every request. On any
    // setup failure returns false, leaving the object as if start() had never
    // been called. No-op (returns true) if already started.
    bool start(uint16_t port, const miner::Stats& stats, const char* version);

    // Signals the accept-loop thread via the self-pipe, joins it, then closes
    // the listen socket and both pipe ends. No-op if not started.
    void stop();

    // The actual bound port -- only meaningful after a successful start()
    // (0 otherwise). Needed by callers that passed port 0.
    uint16_t bound_port() const { return bound_port_; }

private:
    void accept_loop(int fd, int stop_read_fd);
    void handle_connection(int conn_fd) const;
    std::string build_body() const;

    const miner::Stats* stats_ = nullptr;
    std::string version_;

    int listen_fd_ = -1;
    uint16_t bound_port_ = 0;
    std::thread accept_thread_;
    bool started_ = false;

    // [0] = read end (given to accept_loop), [1] = write end (stop() writes one
    // byte here). Both -1 when not started.
    int stop_pipe_[2] = {-1, -1};
};

} } // namespace mxbm::api
