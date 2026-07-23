#pragma once
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

#include "stratum/messages.h"
#include "stratum/transport.h"

namespace mxbm { namespace stratum {

// Stratum client: connect -> login -> nonceprefix capture -> job dispatch,
// with automatic reconnect-on-drop. run() is a blocking read loop that never
// returns (the reference miner-style: process exit is Ctrl+C, not a return path).
//
// Threading (Phase A): login() and submit() both write to the transport via
// send_line(); write_mutex_ serializes those writes against each other so
// M3's GPU solver can call submit() from its own thread while run() blocks
// in recv_line() on the same Transport. It does NOT make a concurrent
// read-while-write safe when TLS is on: a single OpenSSL SSL* accessed
// simultaneously from a reader thread and a writer thread needs its own
// locking (e.g. BIO-level or split read/write objects), which this
// transport does not yet provide. Known M3 work: every worker-thread
// submit() races the read loop parked in SSL_read on the same SSL* (not
// thread-safe), and reconnect (transport_.connect from run()'s thread) is
// likewise unsynchronized against cross-thread submit(). At the reference
// solver's rate this fires rarely (~daily expectation at the observed
// 512-unit vardiff floor) but it is NOT unreachable; M3's transport rework
// must fix it before the GPU solver makes submits frequent.
class Client {
public:
    std::function<void(const Job&)> on_job;
    std::function<void(const Result&)> on_result;

    // Stats/console hook: fired (if set -- empty by default, guarded at the
    // call site) from run()'s own thread, once when an ESTABLISHED
    // connection's read fails (the recv_line-failure branch, right BEFORE
    // the 1s backoff sleep) -- i.e. once per established-connection drop,
    // NOT once per reconnect/redial attempt: while a pool stays down, the
    // reconnect loop's own repeated redial attempts do not re-fire it.
    // Wired to console::disconnected() + Stats::record_disconnect() in
    // main.cpp.
    std::function<void()> on_disconnect;

    // Opens the connection; stores host/port/tls so run() can reconnect with
    // the same parameters later. Transport verify is always false in Phase A
    // (pool mode; plan-mandated default — see transport.h).
    bool connect(const std::string& host, uint16_t port, bool tls);

    // Stores api_key for re-login on reconnect, then sends Login on the wire.
    void login(const std::string& api_key);

    // Sends a Solution on the wire.
    void submit(const Solution& solution);

    // Blocking read loop: recv_line -> handle_line, forever. On a dropped
    // connection: close, wait 1s, reconnect + re-login with the stored
    // parameters, keep looping. Never returns; process exit is Ctrl+C.
    void run();

    // Parses one inbound line and dispatches it (Job -> on_job; Result ->
    // capture nonceprefix if non-empty, then on_result). Public so tests can
    // drive the state machine without a real socket. Unparseable lines are
    // ignored silently; Cancel/Unknown/Login/Solution inbound methods are
    // no-ops — no cancel handling and no job queue in the client (stock node
    // never sends cancel; YAGNI).
    void handle_line(const std::string& line);

    std::string current_nonceprefix() const;

private:
    // Reconnects with the stored host_/port_/tls_ and, if that succeeds,
    // resends Login with the stored api_key_. False if either step fails.
    bool reconnect_and_login();

    Transport transport_;
    std::mutex write_mutex_;   // serializes send_line() from login()/submit()/reconnect

    std::string host_;
    uint16_t port_ = 0;
    bool tls_ = false;
    std::string api_key_;
    std::string nonceprefix_;
};

} } // namespace mxbm::stratum
