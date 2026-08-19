#include "ui/console.h"
#include "pow/difficulty.h"
#include "ui/format.h"
#include "version.h"

#include <atomic>
#include <cstdio>
#include <functional>
#include <ctime>
#include <mutex>

#ifdef _WIN32
#include <direct.h>
#include <windows.h>
// Absent from pre-10.0.10586 SDK headers, where SetConsoleMode then rejects it.
#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif
#else
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace mxbm { namespace ui { namespace console {

namespace {

bool g_nocolor = false;

// --silence 0..3 and --compactaccept. Read by job()/share_found()/share_result()
// to decide whether a line is printed at all, and by take_accept_marks(), which
// carries the accepts a silenced console did not print to the speed line.
int  g_silence = 0;
bool g_compact_accept = false;
std::atomic<unsigned> g_accept_marks{0};

// The transcript, when --log is on. Guarded by g_mutex along with the stdout
// writes themselves: console functions are called from four threads (the
// ticker, the stratum client, the mining worker, main), and without the lock a
// multi-line stats block could be split down the middle by a share line
// arriving from another thread -- on screen and, worse, in the file.
std::mutex g_mutex;
std::FILE* g_log = nullptr;

const char* const kGreen = "\033[1;32m";
const char* const kRed   = "\033[1;31m";
const char* const kBlue  = "\033[1;34m";
// Dark yellow: SGR 33 WITHOUT the bold/bright attribute the others carry --
// bright yellow on a light terminal is close to unreadable.
const char* const kYellow = "\033[33m";
const char* const kReset = "\033[0m";

// localtime_r/localtime_s, not std::localtime: the latter returns a shared
// static buffer.
std::tm local_now() {
    std::time_t t = std::time(nullptr);
    std::tm out{};
#ifdef _WIN32
    localtime_s(&out, &t);
#else
    localtime_r(&t, &out);
#endif
    return out;
}

// One "[YYYY-MM-DD HH:MM:SS] text" line into the transcript. Caller holds
// g_mutex. Flushed per line so a rig that loses power mid-session still has
// everything up to the last second.
void log_line(const char* text, size_t len) {
    if (!g_log) return;
    std::tm tm = local_now();
    char stamp[32];
    std::snprintf(stamp, sizeof stamp, "[%04d-%02d-%02d %02d:%02d:%02d] ",
                  tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec);
    std::fputs(stamp, g_log);
    std::fwrite(text, 1, len, g_log);
    std::fputc('\n', g_log);
    std::fflush(g_log);
}

// Every line of a block gets its own timestamp -- same reason stats_block
// colours per line: each line has to stand on its own.
void log_block(const std::string& block) {
    if (!g_log) return;
    size_t i = 0;
    while (i <= block.size()) {
        size_t j = block.find('\n', i);
        const size_t end = (j == std::string::npos) ? block.size() : j;
        log_line(block.data() + i, end - i);
        if (j == std::string::npos) break;
        i = j + 1;
    }
}

void print_line(const std::string& text) {
    std::lock_guard<std::mutex> lock(g_mutex);
    std::fputs(text.c_str(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
    log_line(text.data(), text.size());
}

void print_colored(const char* color, const std::string& text) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_nocolor) {
        std::fputs(text.c_str(), stdout);
    } else {
        std::fputs(color, stdout);
        std::fputs(text.c_str(), stdout);
        std::fputs(kReset, stdout);
    }
    std::fputc('\n', stdout);
    std::fflush(stdout);
    log_line(text.data(), text.size());   // never coloured in the file
}

} // namespace

void init(bool nocolor) { g_nocolor = nocolor; }

void set_verbosity(int silence, bool compact_accept) {
    g_silence = silence < 0 ? 0 : (silence > 3 ? 3 : silence);
    g_compact_accept = compact_accept || g_silence >= 2;
}

unsigned take_accept_marks() { return g_accept_marks.exchange(0); }

int terminal_width() {
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO info{};
    const HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    if (h == INVALID_HANDLE_VALUE || h == nullptr) return 0;
    if (!GetConsoleScreenBufferInfo(h, &info)) return 0;
    const int w = info.srWindow.Right - info.srWindow.Left + 1;
    return w > 0 ? w : 0;
#else
    struct winsize ws {};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) != 0) return 0;
    return ws.ws_col;
#endif
}

bool enable_terminal_color() {
#ifdef _WIN32
    const HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    // The queries fail on a non-console handle, the write on a Windows too old
    // for the flag; either way the escapes would print literally.
    return h != INVALID_HANDLE_VALUE && h != nullptr && GetConsoleMode(h, &mode)
        && SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#else
    return ::isatty(::fileno(stdout)) == 1;
#endif
}

bool open_log(const std::string& path, std::string* resolved_path) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_log) { std::fclose(g_log); g_log = nullptr; }

    std::string target = path;
    if (target.empty()) {
        // mkdir failing is not checked separately: if the directory is neither
        // created nor already there, the fopen below fails and reports it.
#ifdef _WIN32
        ::_mkdir("logs");
#else
        ::mkdir("logs", 0755);
#endif
        std::tm tm = local_now();
        char name[64];
        std::snprintf(name, sizeof name, "logs/mxbm_%04d-%02d-%02d_%02d-%02d-%02d.log",
                      tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                      tm.tm_hour, tm.tm_min, tm.tm_sec);
        target = name;
    }

    g_log = std::fopen(target.c_str(), "a");
    if (!g_log) return false;
    if (resolved_path) *resolved_path = target;
    return true;
}

void close_log() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_log) { std::fclose(g_log); g_log = nullptr; }
}

void banner(const std::string& backends, double devfee_rate) {
    if (g_silence >= 3) return;
    // ASCII only, and a fixed 57-column interior. Box-drawing and half-block glyphs
    // depend on the terminal font having them at full cell height; this renders the
    // same everywhere, including in a log file someone pastes into an issue. The
    // padding below counts bytes, so a multi-byte character in `backends` would short
    // the row -- another reason callers keep it ASCII.
    constexpr int kInner = 57;
    static const char* const kArt[] = {
        " __  __ __  __  ____   __  __ ",
        "|  \\/  |\\ \\/ / | __ ) |  \\/  |",
        "| |\\/| | \\  /  |  _ \\ | |\\/| |",
        "| |  | | /  \\  | |_) || |  | |",
        "|_|  |_|/_/\\_\\ |____/ |_|  |_|",
    };
    const std::string rule(kInner, '-');
    // One row, filled out to the interior.
    auto raw = [&](std::string body) {
        if ((int)body.size() < kInner) body.append(kInner - body.size(), ' ');
        print_line("|" + body + "|");
    };
    // One centred row. Content wider than the interior wraps at a space rather than
    // being cut -- the version and the backend list are both build-dependent, and a
    // truncated row would read as complete.
    std::function<void(std::string)> emit = [&](std::string body) {
        while ((int)body.size() > kInner) {
            size_t cut = body.rfind(' ', kInner);
            if (cut == std::string::npos || cut < 2) cut = kInner;
            raw(std::string((kInner - cut) / 2, ' ') + body.substr(0, cut));
            size_t next = body.find_first_not_of(' ', cut);
            if (next == std::string::npos) return;
            body = body.substr(next);
        }
        raw(std::string((kInner - body.size()) / 2, ' ') + body);
    };
    // The wordmark centres as a block, on the width of its widest line: centring each
    // line on its own would shear the glyphs apart.
    size_t art_w = 0;
    for (const char* line : kArt) {
        std::string t(line);
        t.erase(t.find_last_not_of(' ') + 1);
        if (t.size() > art_w) art_w = t.size();
    }
    const std::string art_pad((kInner - (int)art_w) / 2, ' ');
    print_line("+" + rule + "+");
    raw("");
    for (const char* line : kArt) raw(art_pad + line);
    emit(mxbm::version());
    raw("");
    emit("BeamHash III  (Beam)");
    raw("");
    emit("Backends in this build:");
    emit(backends);
    raw("");
    emit("Open source, Apache-2.0.");
    emit("github.com/maxnflaxl/MXBM");
    if (devfee_rate > 0.0) {
        char fee[64];
        std::snprintf(fee, sizeof fee, "Developer fee: %g%%  (--dev-fee to raise)",
                      devfee_rate * 100.0);
        emit(fee);
    }
    raw("");
    print_line("+" + rule + "+");
    print_line("");
}

void setup_miner() { print_line("Setup Miner..."); }

void driver_detected(const char* api, int count) {
    print_line(std::string(api) + " driver detected.");
    print_line("Number of " + std::string(api) + " supported GPUs: " + std::to_string(count));
}

void device_block(int index, const std::string& name, const std::string& address,
                  const std::string& vendor, const std::string& driver,
                  unsigned long long memory_bytes, const std::string& active_detail) {
    print_line("Device " + std::to_string(index) + ":");
    auto field = [](const char* label, const std::string& value) {
        if (value.empty()) return;   // omitted, never guessed -- see console.h
        print_line(std::string("    ") + label + value);
    };
    field("Name:    ", name);
    field("Address: ", address);
    field("Vendor:  ", vendor);
    field("Drivers: ", driver);
    if (memory_bytes) {
        // MByte on the 1024 scale, matching how every GPU tool reports VRAM.
        field("Memory:  ", std::to_string(memory_bytes / (1024ULL * 1024ULL)) + " MByte");
    }
    field("Active:  ", active_detail.empty() ? "false"
                                             : "true (" + active_detail + ")");
    print_line("");
}

void connecting_to_pool() { print_line("Connecting to pool..."); }

void connected_to(const std::string& host, const std::string& ip, uint16_t port, bool tls) {
    std::string text = "Connected to " + host;
    if (!ip.empty()) text += "(" + ip + ")";
    text += ":" + std::to_string(port);
    if (tls) text += "  (TLS enabled)";
    print_colored(kGreen, text);
}

void tls_handshake_ok() { print_colored(kGreen, "TLS Handshake success"); }

void connecting(const std::string& host, uint16_t port, bool tls) {
    std::string text = "Connecting to pool " + host + ":" + std::to_string(port);
    if (tls) text += " (TLS)";
    text += "...";
    print_line(text);
}

void connected(bool tls) {
    std::string text = "Connected";
    if (tls) text += " (TLS)";
    print_colored(kGreen, text);
}

void authorized(const std::string& user) {
    print_colored(kGreen, "Authorized worker: " + user);
}

void start_mining() {
    print_line("Start Mining...");
}

void job(const std::string& id, uint32_t difficulty, uint64_t height) {
    if (g_silence >= 1) return;
    // Beam display units, not the raw packed uint32, and unabbreviated --
    // unlike format_units()'s k/M share notation.
    char units[32];
    std::snprintf(units, sizeof units, "%.0f", pow::to_display_units(difficulty));

    // The job id stays, parenthesised behind the height: it is what a share, a
    // cancel and a /summary entry are correlated by.
    //
    // A pool that omits "height" leaves it 0 (messages.cpp defaults it), and
    // "blockheight 0" would be a lie about the chain rather than a missing field,
    // so that case keeps the original wording instead.
    std::string text = height != 0
        ? "New job received for blockheight " + std::to_string(height) + " (job " + id + ")"
        : "New job received: " + id;
    print_colored(kYellow, text + " Difficulty: " + units);
}

void share_found(const std::string& device, double units, double target_units) {
    if (g_silence >= 2 || g_compact_accept) return;
    std::string text = device + ": Found a share of difficulty " + format_units(units);
    if (target_units > 0.0) {
        // "(3.9x target of 2048)": the multiple AND the bar it cleared, so the
        // line stands alone in a log, with the target unabbreviated. Plain ASCII
        // "x", not the "×" the dashboard uses -- this line gets grepped.
        char mult[64];
        std::snprintf(mult, sizeof mult, " (%.1fx target of %.0f)",
                      units / target_units, target_units);
        text += mult;
    }
    print_colored(kGreen, text);
}

void share_result(int code, const std::string& description, long long ms) {
    if (code == 1 && g_compact_accept) {
        g_accept_marks.fetch_add(1);
        return;
    }
    // A rejection survives every silence level: it is the one share line an
    // operator needs to see, and it is rare enough not to flood anything.
    if (code == 1 && g_silence >= 2) return;
    std::string suffix = (ms >= 0) ? (" (" + std::to_string(ms) + " ms)") : "";
    if (code == 1) {
        print_colored(kGreen, "Share accepted" + suffix);
    } else if (code < 0) {
        print_colored(kRed, "Pool error " + std::to_string(code) + ": " + description);
    } else {
        // 2 = rejected, 3 = expired, or any other non-accept verdict.
        print_colored(kRed, "Share rejected (" + std::to_string(code) + "): " + description + suffix);
    }
}

void disconnected() {
    print_colored(kRed, "Pool connection lost - reconnecting...");
}

void devfee_notice(double rate, std::chrono::seconds slice,
                   std::chrono::seconds cycle, const std::string& pool) {
    // The cadence reads in minutes for the hour-scale cycle that ships, but
    // falls back to seconds rather than rounding a compressed test cycle down
    // to "per 0min".
    char cadence[32];
    const long long secs = (long long)cycle.count();
    if (secs >= 60 && secs % 60 == 0) std::snprintf(cadence, sizeof cadence, "%lldmin", secs / 60);
    else                              std::snprintf(cadence, sizeof cadence, "%llds", secs);

    char buf[192];
    std::snprintf(buf, sizeof buf,
        "Dev fee: %.4g%% - one %llds round per %s of mining, to %s",
        rate * 100.0, (long long)slice.count(), cadence, pool.c_str());
    print_line(buf);
}

void devfee_start(std::chrono::seconds slice) {
    char buf[128];
    std::snprintf(buf, sizeof buf,
        "Dev fee round started (%llds) - mining to the developer's address",
        (long long)slice.count());
    print_line(buf);
}

void devfee_end(std::chrono::seconds slice) {
    char buf[128];
    std::snprintf(buf, sizeof buf,
        "Dev fee round finished (%llds) - back on your pool",
        (long long)slice.count());
    print_line(buf);
}

void info(const std::string& msg) {
    print_line(msg);
}

void stats_block(const std::string& block) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_nocolor) {
        std::fputs(block.c_str(), stdout);
        std::fputc('\n', stdout);
        std::fflush(stdout);
        log_block(block);
        return;
    }
    // Colour each line separately rather than wrapping the whole block in one
    // SGR pair. The block is multi-line, and a single unterminated colour would
    // stay in effect across every newline -- so anything another thread printed
    // mid-block (a share line, a new job) would come out blue too, and a log
    // truncated inside the block would leave the colour set forever. Per-line
    // pairs keep every line self-contained.
    size_t i = 0;
    while (i <= block.size()) {
        size_t j = block.find('\n', i);
        const size_t end = (j == std::string::npos) ? block.size() : j;
        std::fputs(kBlue, stdout);
        std::fwrite(block.data() + i, 1, end - i, stdout);
        std::fputs(kReset, stdout);
        std::fputc('\n', stdout);
        if (j == std::string::npos) break;
        i = j + 1;
    }
    std::fflush(stdout);
    log_block(block);   // uncoloured, on both paths through this function
}

void error(const std::string& msg) {
    print_colored(kRed, msg);
}

} } } // namespace mxbm::ui::console
