#include "ui/keys.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>

#ifdef _WIN32
#include <conio.h>
#include <io.h>
#include <windows.h>
#else
#include <poll.h>
#include <termios.h>
#include <unistd.h>
#endif

namespace mxbm { namespace ui {

Key key_command(char c) {
    switch (c) {
    case 'h': case 'H': return Key::Speed;
    case 's': case 'S': return Key::StatsBlock;
    case 'c': case 'C': return Key::Connection;
    case 'p': case 'P': return Key::Pause;
    case 'r': case 'R': return Key::Resume;
    case '?':           return Key::Help;
    default:            return Key::None;
    }
}

const char* key_help_line() {
    return "Console keys: h speed, s stats, c connection, p pause, r resume";
}

namespace {

std::atomic<bool> g_stop{false};

#ifndef _WIN32

// The terminal state start() replaced, for every restore path below. tcsetattr
// is async-signal-safe, so the signal handler may call restore() directly.
termios g_saved_termios;
std::atomic<bool> g_raw_active{false};

void restore_terminal() {
    if (g_raw_active.exchange(false))
        (void)::tcsetattr(STDIN_FILENO, TCSANOW, &g_saved_termios);
}

// A fatal signal must put the terminal back BEFORE whatever handler was
// installed first gets to exit the process -- the overclock restore hook
// (gpu/overclock.cpp) re-raises with the default disposition, which skips
// atexit. Chaining preserves that hook instead of replacing it.
using SignalHandler = void (*)(int);
SignalHandler g_prev_int = SIG_DFL, g_prev_term = SIG_DFL;

void keys_signal_handler(int sig) {
    restore_terminal();
    const SignalHandler prev = (sig == SIGTERM) ? g_prev_term : g_prev_int;
    if (prev != SIG_DFL && prev != SIG_IGN) {
        prev(sig);
    } else {
        std::signal(sig, SIG_DFL);
        std::raise(sig);
    }
}

std::atomic<bool> g_atexit_installed{false};

// Puts stdin into single-key mode. False when stdin is not a terminal or the
// switch fails; nothing is changed on that path.
bool enter_raw_mode() {
    if (::isatty(STDIN_FILENO) != 1) return false;
    if (::tcgetattr(STDIN_FILENO, &g_saved_termios) != 0) return false;
    termios raw = g_saved_termios;
    // No line buffering, no echo. ISIG stays: Ctrl+C must still interrupt.
    raw.c_lflag &= ~(tcflag_t)(ICANON | ECHO);
    if (::tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0) return false;
    g_raw_active.store(true);

    // A miner backgrounded under job control would otherwise be STOPPED by its
    // own stdin read (SIGTTIN). Ignored, the read fails with EIO instead and
    // the loop just waits; keys work again after fg.
    std::signal(SIGTTIN, SIG_IGN);
    std::signal(SIGTTOU, SIG_IGN);

    g_prev_int = std::signal(SIGINT, keys_signal_handler);
    g_prev_term = std::signal(SIGTERM, keys_signal_handler);
    if (!g_atexit_installed.exchange(true)) std::atexit(restore_terminal);
    return true;
}

// One key, or 0 after a ~250 ms wait with none available (also on a read
// error, so a background read cannot spin).
char poll_key() {
    pollfd pfd{STDIN_FILENO, POLLIN, 0};
    if (::poll(&pfd, 1, 250) <= 0 || !(pfd.revents & POLLIN)) return 0;
    char c = 0;
    if (::read(STDIN_FILENO, &c, 1) != 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        return 0;
    }
    return c;
}

#else // _WIN32

// The Windows console needs no mode switch: _getch() is already unbuffered
// and unechoed, and _kbhit() makes it non-blocking.
void restore_terminal() {}
bool enter_raw_mode() { return _isatty(_fileno(stdin)) != 0; }

char poll_key() {
    if (!_kbhit()) { Sleep(250); return 0; }
    const int c = _getch();
    return (c > 0 && c < 256) ? (char)c : 0;
}

#endif

} // namespace

KeyReader::~KeyReader() { stop(); }

bool KeyReader::start(std::function<void(Key)> on_key) {
    if (started_ || !on_key) return false;
    if (!enter_raw_mode()) return false;
    on_key_ = std::move(on_key);
    g_stop.store(false);
    started_ = true;
    reader_ = std::thread([this] { try { reader_main(); } catch (...) {} });
    return true;
}

void KeyReader::stop() {
    if (!started_) return;
    g_stop.store(true);
    if (reader_.joinable()) reader_.join();
    restore_terminal();
    started_ = false;
}

void KeyReader::reader_main() {
    while (!g_stop.load(std::memory_order_relaxed)) {
        const char c = poll_key();
        if (c == 0) continue;
        const Key k = key_command(c);
        if (k != Key::None) on_key_(k);
    }
}

} } // namespace mxbm::ui
