#pragma once
#include <cstdint>
#include <string>
namespace mxbm { namespace stratum {
class Transport {
public:
    ~Transport();
    bool connect(const std::string& host, uint16_t port, bool tls, bool verify);
    bool send_line(const std::string& line);   // appends '\n'; false + close() on write error
    bool recv_line(std::string& out);           // blocks; strips '\n'; false on close/error
    void close();
    bool connected() const { return fd_ >= 0; }

    // The IP the hostname actually resolved to for the CURRENT connection,
    // "" before the first successful connect. Worth surfacing because the
    // large pools round-robin a single hostname across regions and machines:
    // without the address, two runs that behaved differently are
    // indistinguishable in a log.
    const std::string& peer_ip() const { return peer_ip_; }
private:
    int fd_ = -1;
    std::string peer_ip_;
    void* ssl_ = nullptr;   // SSL* (opaque)
    void* ctx_ = nullptr;   // SSL_CTX*
    std::string rxbuf_;
};
} } // namespace mxbm::stratum
