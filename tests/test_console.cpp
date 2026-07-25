// Console colour tests. console.cpp is a thin print layer, but the --nocolor
// contract is a real one: miners are commonly watched through `tee` into a log,
// and escape codes leaking into a log the user asked to be plain is a silent
// regression that no other test would catch. stdout is redirected to a temp
// file so the emitted bytes -- escapes included -- can be asserted exactly.
#include <cstdio>
#include <string>
#include <vector>
#include <unistd.h>   // dup/dup2 for the stdout capture

#include "check.h"
#include "ui/console.h"
#include "ui/format.h"

using namespace mxbm;

namespace {

const char* const kBlue  = "\033[1;34m";
const char* const kReset = "\033[0m";

// Run `fn` with stdout redirected, and return everything it printed.
std::string capture(void (*fn)()) {
    std::string path = std::string(std::tmpnam(nullptr));
    fflush(stdout);
    FILE* saved = fdopen(dup(fileno(stdout)), "w");
    FILE* redir = freopen(path.c_str(), "w+", stdout);
    (void)redir;
    fn();
    fflush(stdout);
    // Restore stdout before reading, so check() output still reaches the runner.
    dup2(fileno(saved), fileno(stdout));
    fclose(saved);

    std::string out;
    if (FILE* f = fopen(path.c_str(), "rb")) {
        char buf[4096];
        size_t n;
        while ((n = fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
        fclose(f);
    }
    std::remove(path.c_str());
    return out;
}

miner::Stats::Snapshot fixture() {
    miner::Stats::Snapshot s{};
    s.sol60 = 57.08;
    s.pool_sol_session = 54.30;
    s.iter60 = 28.1;
    s.accepted = 5;
    s.best_share_units = 2700.0;
    s.last_latency_ms = 17;
    s.pool = "pool.example.com:1130";
    s.device_label = "NVIDIA GeForce RTX 4070 Ti SUPER";
    s.uptime = std::chrono::seconds(3600);
    return s;
}

void emit_colored() {
    ui::console::init(false);
    ui::console::stats_block(ui::format_stats_block(fixture(), "0.4", "02:33:14"));
}

void emit_plain() {
    ui::console::init(true);
    ui::console::stats_block(ui::format_stats_block(fixture(), "0.4", "02:33:14"));
}

std::vector<std::string> split_lines(const std::string& s) {
    std::vector<std::string> v;
    for (size_t i = 0, j; i < s.size(); i = j + 1) {
        j = s.find('\n', i);
        if (j == std::string::npos) j = s.size();
        v.push_back(s.substr(i, j - i));
    }
    return v;
}

} // namespace

int main() {
    const std::string colored = capture(&emit_colored);
    const std::string plain   = capture(&emit_plain);

    check(colored.find(kBlue) != std::string::npos,
          "stats_block emits blue when colour is enabled");

    // Every line carries its own colour/reset pair. A single pair wrapping the
    // whole block would leave the colour in effect across newlines, so anything
    // printed mid-block by another thread would come out blue too.
    std::vector<std::string> lines = split_lines(colored);
    check(!lines.empty(), "stats_block produced output");
    bool every_line_paired = true;
    for (const std::string& ln : lines) {
        if (ln.compare(0, 7, kBlue) != 0) every_line_paired = false;
        if (ln.size() < 11 || ln.compare(ln.size() - 4, 4, kReset) != 0) every_line_paired = false;
    }
    check(every_line_paired,
          "every line of the block is independently colour-wrapped and reset");

    // --nocolor must produce bytes a log can hold verbatim.
    check(plain.find('\033') == std::string::npos,
          "init(true) suppresses every escape code");
    check(plain == ui::format_stats_block(fixture(), "0.4", "02:33:14") + "\n",
          "the --nocolor block is exactly the formatted table plus a newline");

    // Stripping the escapes from the coloured form must give back the plain form:
    // colour must not alter content, spacing or the column alignment.
    std::string stripped;
    for (size_t i = 0; i < colored.size(); ++i) {
        if (colored[i] == '\033') { i = colored.find('m', i); continue; }
        stripped += colored[i];
    }
    check(stripped == plain, "colour changes only the escapes, never the table content");

    return summary("console");
}
