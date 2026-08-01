// The POSIX/Winsock seam, in one place, for everything that opens a socket
// (stratum/transport, api/http_summary, and their loopback tests). Sockets
// stay plain `int` on both sides: a win64 SOCKET is a wider handle, but real
// handles fit, and INVALID_SOCKET truncates to exactly -1 -- so the `fd < 0`
// idiom keeps meaning "no socket". to_fd()/from_fd() are the only casts.
#pragma once

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <basetsd.h>
using ssize_t = SSIZE_T;
#else
#include <arpa/inet.h>
#include <cerrno>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace mxbm { namespace net {

// Idempotent, process-wide; every socket entry point calls it first.
inline bool startup() {
#ifdef _WIN32
    static const bool ok = [] { WSADATA w; return WSAStartup(MAKEWORD(2, 2), &w) == 0; }();
    return ok;
#else
    return true;
#endif
}

#ifdef _WIN32
inline int    to_fd(SOCKET s) { return (int)(INT_PTR)s; }
inline SOCKET from_fd(int fd) { return (SOCKET)(INT_PTR)fd; }
#else
inline int to_fd(int s) { return s; }
inline int from_fd(int fd) { return fd; }
#endif

// Closes sockets on both platforms, and the POSIX pipe fds signal_pair hands
// out (on Windows those are sockets too).
inline void close_fd(int fd) {
#ifdef _WIN32
    ::closesocket(from_fd(fd));
#else
    ::close(fd);
#endif
}

inline int last_error() {
#ifdef _WIN32
    return WSAGetLastError();
#else
    return errno;
#endif
}

// poll()/accept() failures that mean "try again", not "tear the server down".
inline bool transient(int e) {
#ifdef _WIN32
    return e == WSAEINTR || e == WSAECONNABORTED || e == WSAEMFILE;
#else
    return e == EINTR || e == ECONNABORTED || e == EMFILE || e == ENFILE;
#endif
}

#ifdef _WIN32
using pollfd_t = WSAPOLLFD;
#else
using pollfd_t = struct pollfd;
#endif

inline int poll_fds(pollfd_t* fds, unsigned n, int timeout_ms) {
#ifdef _WIN32
    return ::WSAPoll(fds, (ULONG)n, timeout_ms);
#else
    return ::poll(fds, (nfds_t)n, timeout_ms);
#endif
}

// SO_RCVTIMEO wants a timeval on POSIX and DWORD milliseconds on Windows.
inline void set_recv_timeout(int fd, int seconds) {
#ifdef _WIN32
    DWORD ms = (DWORD)seconds * 1000;
    ::setsockopt(from_fd(fd), SOL_SOCKET, SO_RCVTIMEO,
                 reinterpret_cast<const char*>(&ms), sizeof ms);
#else
    timeval tv{};
    tv.tv_sec = seconds;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
#endif
}

// A poll()-able wakeup channel: pipe() on POSIX, a connected loopback socket
// pair on Windows (WSAPoll only knows sockets). fds[0] is the read/poll end,
// fds[1] the write end; close both with close_fd().
inline bool signal_pair(int fds[2]) {
#ifdef _WIN32
    if (!startup()) return false;
    fds[0] = fds[1] = -1;
    SOCKET l = ::socket(AF_INET, SOCK_STREAM, 0);
    if (l == INVALID_SOCKET) return false;
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = 0;
    int alen = sizeof a;
    SOCKET w = INVALID_SOCKET, r = INVALID_SOCKET;
    if (::bind(l, reinterpret_cast<sockaddr*>(&a), sizeof a) == 0 && ::listen(l, 1) == 0
        && ::getsockname(l, reinterpret_cast<sockaddr*>(&a), &alen) == 0
        && (w = ::socket(AF_INET, SOCK_STREAM, 0)) != INVALID_SOCKET
        && ::connect(w, reinterpret_cast<sockaddr*>(&a), sizeof a) == 0
        && (r = ::accept(l, nullptr, nullptr)) != INVALID_SOCKET) {
        ::closesocket(l);
        fds[0] = to_fd(r);
        fds[1] = to_fd(w);
        return true;
    }
    if (w != INVALID_SOCKET) ::closesocket(w);
    ::closesocket(l);
    return false;
#else
    return ::pipe(fds) == 0;
#endif
}

inline void signal_send(int fd) {
#ifdef _WIN32
    (void)::send(from_fd(fd), "x", 1, 0);
#else
    (void)::write(fd, "x", 1);
#endif
}

} } // namespace mxbm::net
