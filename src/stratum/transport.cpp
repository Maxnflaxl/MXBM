// TCP + TLS line transport: blocking POSIX sockets, optional OpenSSL client
// TLS. Line-delimited protocol ('\n'); recv_line buffers partial reads in
// rxbuf_ and enforces a 64 KiB line cap. Reconnect is driven by the caller
// (stratum::Client, Task 5) — connect() tears down any prior state first, so
// it doubles as a reset.
#include "stratum/transport.h"

#include <cstdint>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <openssl/ssl.h>
#include <openssl/x509v3.h>

namespace mxbm { namespace stratum {

namespace {
constexpr size_t kMaxLine = 64 * 1024;   // 64 KiB line cap (recv_line only)

// >0: bytes transferred. 0: peer closed. <0: error. Routes through OpenSSL
// when ssl is non-null, otherwise plain recv()/send(). MSG_NOSIGNAL (where
// defined) keeps a write into a dying socket from raising SIGPIPE.
ssize_t raw_write(int fd, SSL* ssl, const char* buf, size_t len) {
    if (ssl) return SSL_write(ssl, buf, static_cast<int>(len));
#ifdef MSG_NOSIGNAL
    return send(fd, buf, len, MSG_NOSIGNAL);
#else
    return send(fd, buf, len, 0);
#endif
}

ssize_t raw_read(int fd, SSL* ssl, char* buf, size_t len) {
    if (ssl) return SSL_read(ssl, buf, static_cast<int>(len));
    return recv(fd, buf, len, 0);
}
} // namespace

Transport::~Transport() { close(); }

bool Transport::connect(const std::string& host, uint16_t port, bool tls, bool verify) {
    close();   // idempotent: discard any prior connection/TLS state first

    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo* results = nullptr;
    const std::string port_str = std::to_string(port);
    if (getaddrinfo(host.c_str(), port_str.c_str(), &hints, &results) != 0 || !results) {
        return false;
    }

    int fd = -1;
    for (addrinfo* ai = results; ai != nullptr; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        if (::connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) break;
        ::close(fd);
        fd = -1;
    }
    freeaddrinfo(results);

    if (fd < 0) return false;

#ifdef SO_NOSIGPIPE
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif

    fd_ = fd;

    if (!tls) return true;

    SSL_CTX* ctx = SSL_CTX_new(TLS_client_method());
    if (!ctx) { close(); return false; }
    ctx_ = ctx;

    if (verify) {
        SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, nullptr);
        SSL_CTX_set_default_verify_paths(ctx);
    } else {
        SSL_CTX_set_verify(ctx, SSL_VERIFY_NONE, nullptr);
    }

    SSL* ssl = SSL_new(ctx);
    if (!ssl) { close(); return false; }
    ssl_ = ssl;

    if (verify) {
        SSL_set_hostflags(ssl, X509_CHECK_FLAG_NO_PARTIAL_WILDCARDS);
        if (SSL_set1_host(ssl, host.c_str()) != 1) { close(); return false; }
    }

    SSL_set_tlsext_host_name(ssl, host.c_str());   // SNI — many pools route on it
    SSL_set_fd(ssl, fd_);

    if (SSL_connect(ssl) != 1) { close(); return false; }

    return true;
}

bool Transport::send_line(const std::string& line) {
    if (fd_ < 0) return false;

    std::string out = line;
    out.push_back('\n');

    size_t sent = 0;
    while (sent < out.size()) {
        ssize_t n = raw_write(fd_, static_cast<SSL*>(ssl_), out.data() + sent, out.size() - sent);
        if (n <= 0) { close(); return false; }   // dead socket: drop and let the caller reconnect
        sent += static_cast<size_t>(n);
    }
    return true;
}

bool Transport::recv_line(std::string& out) {
    if (fd_ < 0) return false;

    while (true) {
        size_t nl = rxbuf_.find('\n');
        if (nl != std::string::npos) {
            out = rxbuf_.substr(0, nl);
            if (!out.empty() && out.back() == '\r') out.pop_back();   // defensive CRLF tolerance
            rxbuf_.erase(0, nl + 1);
            return true;
        }

        if (rxbuf_.size() > kMaxLine) { close(); return false; }   // runaway line: bail, never grow unbounded

        char buf[4096];
        ssize_t n = raw_read(fd_, static_cast<SSL*>(ssl_), buf, sizeof(buf));
        if (n <= 0) { close(); return false; }   // 0 = peer closed, <0 = error
        rxbuf_.append(buf, static_cast<size_t>(n));
    }
}

void Transport::close() {
    if (ssl_) {
        SSL* ssl = static_cast<SSL*>(ssl_);
        SSL_shutdown(ssl);   // best-effort close_notify; ignore result, never block on the peer's reply
        SSL_free(ssl);
        ssl_ = nullptr;
    }
    if (ctx_) {
        SSL_CTX_free(static_cast<SSL_CTX*>(ctx_));
        ctx_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    rxbuf_.clear();
}

} } // namespace mxbm::stratum
