// GET /summary hand-rolled HTTP API: a minimal, single-purpose HTTP/1.1
// server exposing one endpoint (per-request, no keep-alive) that reports a
// live miner::Stats snapshot as JSON -- MXBM's analog of the reference miner's own
// --apiport/--apihost TCP+JSON API.
//
// Lifecycle mirrors miner::Engine/ui::Ticker (start() spawns one worker
// thread, stop() signals it and joins, the destructor calls stop()). The
// accept-loop thread blocks in poll() on TWO file descriptors -- the
// listen socket and the read end of a self-pipe (stop_pipe_) -- rather
// than blocking directly in accept(). stop() writes one byte to the
// pipe's write end, which wakes poll() deterministically, THEN joins the
// thread (now guaranteed to be on its way out), THEN closes the listen fd
// and both pipe ends.
//
// Portability note: an earlier version of this class relied on stop()
// closing the listen fd to make a thread blocked in accept() return --
// that works on macOS/BSD but is NOT guaranteed on Linux, where close()
// on a listening socket does not reliably wake a concurrent accept() call;
// stop() could hang in join() there until the next inbound connection
// happened to arrive. A self-pipe sidesteps the platform difference
// entirely: writing to a pipe always wakes a poll() blocked on that
// pipe's read end, on every POSIX platform.
//
// Binds 0.0.0.0 (all interfaces) by design, matching the reference miner's own
// --apihost default: this is an UNAUTHENTICATED, LAN-visible read-only
// endpoint -- the same exposure
// the reference miner itself has out of the box. Acceptable for this tool; an
// operator who wants it local-only can firewall the port (the reference miner offers
// no better story here either -- --apihost 127.0.0.1 is opt-in, not the
// default).
//
// Every response is `Connection: close` -- one request per accepted
// connection, then the socket is closed. No persistent/keep-alive
// connections, no pipelining, no chunked bodies: this is a hand-rolled,
// deliberately minimal server, not a general-purpose one.
//
// SIGPIPE is already ignored process-wide by the time this class is ever
// used for real (see main.cpp, Phase A: std::signal(SIGPIPE, SIG_IGN) runs
// before anything else in main()), so a write into a connection the peer
// already dropped just returns -1/EPIPE here -- no per-write MSG_NOSIGNAL/
// SO_NOSIGPIPE guard is needed (contrast stratum::Transport::raw_write,
// which guards every write because a Transport can be constructed and used
// well before that point, e.g. in isolated unit tests).
//
// JSON schema: see http_summary.cpp's build_body() doc comment -- field
// names are a best-effort MXBM design, explicitly UNVERIFIED against a
// real the reference miner /summary response. Capturing one from a live the reference miner
// --apiport run and reconciling remains an open item.
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
    // `version` must outlive the HttpSummary (or at least until
    // stop()/the destructor returns): stats_ is stored as a pointer and
    // snapshotted fresh on every request (no polling/caching), so live
    // updates show up immediately. Returns false on any socket setup
    // failure (socket/bind/listen/getsockname/pipe), leaving the object
    // exactly as if start() had never been called -- no thread spawned, no
    // partial state. No-op (returns true) if already started.
    bool start(uint16_t port, const miner::Stats& stats, const char* version);

    // Signals the accept-loop thread to stop (via the self-pipe wakeup --
    // see class comment), joins it, then closes the listen socket and
    // both pipe ends. No-op if not started.
    void stop();

    // The actual bound port -- only meaningful after a successful start()
    // (0 otherwise). Needed by callers/tests that passed port 0.
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

    // Self-pipe used to wake accept_loop()'s poll() from stop() -- see
    // class comment. [0] = read end (given to accept_loop), [1] = write
    // end (stop() writes one byte here). Both -1 when not started.
    int stop_pipe_[2] = {-1, -1};
};

} } // namespace mxbm::api
