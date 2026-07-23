// Stratum client state machine: login -> nonceprefix capture -> job dispatch
// -> reconnect. See client.h for the write_mutex_ threading contract and its
// known TLS cross-thread limitation.
#include "stratum/client.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace mxbm { namespace stratum {

namespace {
// MXBM_TRACE=1 dumps raw wire lines to stderr ("< " inbound, "> " outbound) —
// used to capture pool dialects. Off by default so console output stays
// clean.
bool trace_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("MXBM_TRACE");
        return v && *v && *v != '0';
    }();
    return on;
}
void trace(char dir, const std::string& line) {
    if (trace_enabled()) std::fprintf(stderr, "[trace] %c %s\n", dir, line.c_str());
}
} // namespace

bool Client::connect(const std::string& host, uint16_t port, bool tls) {
    host_ = host;
    port_ = port;
    tls_ = tls;
    return transport_.connect(host_, port_, tls_, /*verify=*/false);
}

void Client::login(const std::string& api_key) {
    api_key_ = api_key;
    const std::string wire = serialize(Login{api_key_});
    trace('>', wire);
    std::lock_guard<std::mutex> lock(write_mutex_);
    transport_.send_line(wire);
}

void Client::submit(const Solution& solution) {
    const std::string wire = serialize(solution);
    trace('>', wire);
    std::lock_guard<std::mutex> lock(write_mutex_);
    transport_.send_line(wire);
}

void Client::handle_line(const std::string& line) {
    trace('<', line);
    Msg msg;
    if (!parse(line, msg)) return;   // malformed/unparseable: ignored silently

    switch (msg.method) {
        case Method::Job:
            if (on_job) on_job(msg.job);
            break;
        case Method::Result:
            // Pools may resend nonceprefix on reconnect-login; only overwrite
            // when a new one is actually present, otherwise keep the last one.
            if (!msg.result.nonceprefix.empty()) nonceprefix_ = msg.result.nonceprefix;
            if (on_result) on_result(msg.result);
            break;
        default:
            break;   // Cancel/Unknown/Login/Solution inbound: no-op (YAGNI; stock node never cancels)
    }
}

std::string Client::current_nonceprefix() const {
    return nonceprefix_;
}

bool Client::reconnect_and_login() {
    if (!transport_.connect(host_, port_, tls_, /*verify=*/false)) return false;
    login(api_key_);
    return transport_.connected();
}

void Client::run() {
    while (true) {
        if (!transport_.connected()) {
            if (!reconnect_and_login()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }
            continue;
        }

        std::string line;
        if (transport_.recv_line(line)) {
            handle_line(line);
        } else {
            transport_.close();
            if (on_disconnect) on_disconnect();
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
    }
}

} } // namespace mxbm::stratum
