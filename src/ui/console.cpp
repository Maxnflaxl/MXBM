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
const char* const kBlue  = "\033[1;34m";
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

void share_found(const std::string& device, double units, double target_units) {
    std::string text = device + ": Found a share of difficulty " + format_units(units);
    if (target_units > 0.0) {
        // Plain ASCII "x", not the "×" the dashboard uses: this line is
        // routinely piped through tee, grep and log shippers, and an ASCII
        // multiplier stays greppable everywhere.
        // "(3.9x target of 2048)": the multiple AND the bar it cleared, on the
        // one line. The multiple alone leaves you hunting back through the log
        // for whatever the job difficulty was at the time, and vardiff moves it
        // all session. Target unabbreviated -- it is the number you compare
        // against, so throwing away digits to save four columns is a bad trade.
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
    // Plain white, deliberately: this is neither good news nor an error, and
    // colouring it either way would editorialise. Everything a user needs to
    // check the arithmetic themselves is on the line -- rate, round length,
    // cadence and destination.
    // The cadence reads in minutes for the hour-scale cycle that ships, but
    // falls back to seconds rather than rounding a short cycle down to
    // "per 0min" -- which is what a test build with a compressed cycle would
    // otherwise print.
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
    if (g_nocolor) { print_line(block); return; }
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
}

void error(const std::string& msg) {
    print_colored(kRed, msg);
}

} } } // namespace mxbm::ui::console
