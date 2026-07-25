#include "ui/console.h"
#include "pow/difficulty.h"
#include "ui/format.h"
#include "version.h"

#include <cstdio>
#include <ctime>
#include <mutex>

#include <sys/stat.h>

namespace mxbm { namespace ui { namespace console {

namespace {

bool g_nocolor = false;

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

// localtime_r, not std::localtime: the latter returns a shared static buffer.
std::tm local_now() {
    std::time_t t = std::time(nullptr);
    std::tm out{};
    localtime_r(&t, &out);
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

bool open_log(const std::string& path, std::string* resolved_path) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_log) { std::fclose(g_log); g_log = nullptr; }

    std::string target = path;
    if (target.empty()) {
        // mkdir failing is not checked separately: if the directory is neither
        // created nor already there, the fopen below fails and reports it.
        ::mkdir("logs", 0755);
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

void banner() {
    std::string text = std::string("MXBM ") + mxbm::version() + " — open BeamHash III miner";
    print_line(text);
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
    (void)height;   // signature stability only -- see doc comment in console.h
    // Beam display units, not the raw packed uint32, and unabbreviated --
    // unlike format_units()'s k/M share notation.
    char units[32];
    std::snprintf(units, sizeof units, "%.0f", pow::to_display_units(difficulty));
    print_colored(kYellow, "New job received: " + id + " Difficulty: " + units);
}

void share_found(const std::string& device, double units, double target_units) {
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
