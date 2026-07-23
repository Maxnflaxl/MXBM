#include "ui/console.h"
#include "pow/difficulty.h"
#include "ui/format.h"
#include "version.h"

#include <cstdio>

namespace mxbm { namespace ui { namespace console {

namespace {

bool g_nocolor = false;

const char* const kGreen = "\033[1;32m";
const char* const kRed   = "\033[1;31m";
const char* const kReset = "\033[0m";

void print_line(const std::string& text) {
    std::fputs(text.c_str(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

void print_colored(const char* color, const std::string& text) {
    if (g_nocolor) { print_line(text); return; }
    std::fputs(color, stdout);
    std::fputs(text.c_str(), stdout);
    std::fputs(kReset, stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

} // namespace

void init(bool nocolor) { g_nocolor = nocolor; }

void banner() {
    std::string text = std::string("MXBM ") + mxbm::version() + " — open BeamHash III miner";
    print_line(text);
    print_line("");
}

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
    // Difficulty in Beam display units (the reference miner parity), not the raw packed
    // uint32; unabbreviated (plain integer), unlike format_units()'s k/M
    // share/best-share notation.
    char units[32];
    std::snprintf(units, sizeof units, "%.0f", pow::to_display_units(difficulty));
    print_line("New job received: " + id + " Difficulty: " + units);
}

void share_found(const std::string& device, double units) {
    print_colored(kGreen, device + ": Found a share of difficulty " + format_units(units));
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

void info(const std::string& msg) {
    print_line(msg);
}

void error(const std::string& msg) {
    print_colored(kRed, msg);
}

} } } // namespace mxbm::ui::console
