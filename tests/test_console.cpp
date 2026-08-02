// Console colour tests. The --nocolor contract matters: miners are watched
// through `tee` into a log, and leaked escape codes are a silent regression no
// other test would catch. stdout is redirected so the bytes can be asserted.
#include <cstdio>
#include <string>
#include <vector>
#ifdef _WIN32
#include <io.h>
#define dup    _dup
#define dup2   _dup2
#define fileno _fileno
#else
#include <unistd.h>
#endif

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

// capture() redirects stdout, which is the state that must answer no.
void emit_terminal_probe() {
    std::printf("%d", ui::console::enable_terminal_color() ? 1 : 0);
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

    // A single pair wrapping the whole block would leave the colour in effect
    // across newlines, so a mid-block print from another thread would be blue.
    std::vector<std::string> lines = split_lines(colored);
    check(!lines.empty(), "stats_block produced output");
    bool every_line_paired = true;
    for (const std::string& ln : lines) {
        if (ln.compare(0, 7, kBlue) != 0) every_line_paired = false;
        if (ln.size() < 11 || ln.compare(ln.size() - 4, 4, kReset) != 0) every_line_paired = false;
    }
    check(every_line_paired,
          "every line of the block is independently colour-wrapped and reset");

    check(plain.find('\033') == std::string::npos,
          "init(true) suppresses every escape code");

    // What main passes to init(): a redirected run must write a clean file.
    check(capture(&emit_terminal_probe) == "0",
          "enable_terminal_color() is false when stdout is not a terminal");
    check(plain == ui::format_stats_block(fixture(), "0.4", "02:33:14") + "\n",
          "the --nocolor block is exactly the formatted table plus a newline");

    std::string stripped;
    for (size_t i = 0; i < colored.size(); ++i) {
        if (colored[i] == '\033') { i = colored.find('m', i); continue; }
        stripped += colored[i];
    }
    check(stripped == plain, "colour changes only the escapes, never the table content");

    // -- share_found: the achieved/target multiple --
    // Pinned as literal text: this is the line users grep their logs for.
    {
        std::string out = capture([] {
            ui::console::init(true);
            ui::console::share_found("GPU 0", 8000.0, 2048.0);
            ui::console::share_found("GPU 0", 8000.0);       // no target known
            ui::console::share_found("GPU 0", 8000.0, 0.0);  // target not yet received
        });
        check(out.find("GPU 0: Found a share of difficulty 8.0k (3.9x target of 2048)") != std::string::npos,
              "share line reports the multiple AND the target it cleared, on one line");
        size_t plain_lines = 0, from = 0;
        while ((from = out.find("difficulty 8.0k\n", from)) != std::string::npos) { ++plain_lines; ++from; }
        check(plain_lines == 2,
              "an unknown or zero target omits the multiple rather than dividing by it");
        check(out.find("inf") == std::string::npos && out.find("nan") == std::string::npos,
              "no infinity or NaN ever reaches the console");
    }

    // -- the job line is dark yellow --
    // Bright yellow is close to unreadable on a light terminal, and this line
    // arrives often enough that it should not compete with the share lines.
    {
        std::string out = capture([] {
            ui::console::init(false);
            ui::console::job("362264", 0x09000000u, 0);
        });
        check(out.compare(0, 5, "\033[33m") == 0,
              "the job line opens with dark yellow (33), not bright (1;33)");
        check(out.find("New job received: 362264") != std::string::npos,
              "...around the job line itself");
        check(out.find(kReset) != std::string::npos, "and resets afterwards");
    }

    // -- the job line carries the block height --
    // Beam's stratum job carries "height" and the console used to discard it.
    // The prefix is the reference miner's verbatim so one grep spans both miners' logs; the
    // job id survives in parentheses because shares and cancels key off it.
    {
        std::string out = capture([] {
            ui::console::init(true);
            ui::console::job("5417", 0x09000000u, 2500000);
        });
        check(out.find("New job received for blockheight 2500000 (job 5417) "
                       "Difficulty: 512") != std::string::npos,
              "the job line names the block height, keeps the id, and keeps the difficulty");
    }

    // A pool that omits "height" leaves it 0. "blockheight 0" would assert
    // something false about the chain, so the line reverts to its older wording
    // rather than printing a placeholder.
    {
        std::string out = capture([] {
            ui::console::init(true);
            ui::console::job("5417", 0x09000000u, 0);
        });
        check(out.find("New job received: 5417 Difficulty: 512") != std::string::npos,
              "a missing height falls back to the id-only wording");
        check(out.find("blockheight") == std::string::npos,
              "...and never prints 'blockheight 0'");
    }

    // -- the startup block, in the reference miner's order and shape --
    {
        std::string out = capture([] {
            ui::console::init(true);
            ui::console::setup_miner();
            ui::console::driver_detected("Cuda", 1);
            ui::console::device_block(0, "NVIDIA GeForce RTX 4070 Ti SUPER", "1:0",
                                      "NVIDIA Corporation", "Cuda", 16736055296ULL,
                                      "Selected Algorithm: BeamHash III (Cuda)");
            ui::console::connecting_to_pool();
            ui::console::connected_to("de.beam.herominers.com", "141.95.126.31", 1130, true);
            ui::console::tls_handshake_ok();
        });
        check(out.find("Setup Miner...\n") == 0, "the hardware section opens the run");
        check(out.find("Number of Cuda supported GPUs: 1") != std::string::npos,
              "each detected runtime reports its device count");
        check(out.find("Device 0:\n    Name:    NVIDIA GeForce RTX 4070 Ti SUPER\n") != std::string::npos,
              "the device block is indented under its index");
        check(out.find("    Address: 1:0\n") != std::string::npos, "the PCI address is reported");
        check(out.find("    Memory:  15960 MByte\n") != std::string::npos,
              "memory is reported in MByte on the 1024 scale");
        check(out.find("    Active:  true (Selected Algorithm: BeamHash III (Cuda))") != std::string::npos,
              "the active line names the algorithm and the backend that claimed it");
        check(out.find("Connected to de.beam.herominers.com(141.95.126.31):1130  (TLS enabled)")
                  != std::string::npos,
              "the connected line carries the address the hostname resolved to");
        check(out.find("TLS Handshake success") != std::string::npos, "the handshake is confirmed");
    }

    // -- a field with nothing behind it is omitted, never guessed --
    {
        std::string out = capture([] {
            ui::console::init(true);
            // A non-NVIDIA OpenCL device: no NVML, so no address and no vendor.
            ui::console::device_block(0, "gfx1100", "", "", "OpenCL", 0,
                                      "Selected Algorithm: BeamHash III (OpenCL)");
        });
        check(out.find("Address:") == std::string::npos &&
              out.find("Vendor:") == std::string::npos,
              "unknown address/vendor lines are dropped rather than printed blank");
        check(out.find("Memory:") == std::string::npos,
              "an unknown memory size is dropped too, not printed as 0 MByte");
        check(out.find("Name:    gfx1100") != std::string::npos, "what IS known is still shown");
    }

    // -- the transcript log (--log / --logfile) --
    // The append rule matters most: a rig that watchdog-restarts overnight must
    // not wake up having erased the evidence.
    {
        std::string path = std::string(std::tmpnam(nullptr));

        // open/close outside capture(): they print nothing, and check() called
        // while stdout is redirected would have its own output swallowed.
        check(ui::console::open_log(path), "open_log succeeds on a writable path");
        std::string screen = capture([] {
            ui::console::init(false);                  // colour ON: the file must not get it
            ui::console::info("first line");
            ui::console::share_found("GPU 0", 8000.0, 2048.0);
            ui::console::stats_block(ui::format_stats_block(fixture(), "0.4", "02:33:14"));
        });
        ui::console::close_log();
        check(screen.find('\033') != std::string::npos, "the screen still got its colour");

        std::string logged;
        if (FILE* f = fopen(path.c_str(), "rb")) {
            char buf[4096];
            size_t n;
            while ((n = fread(buf, 1, sizeof buf, f)) > 0) logged.append(buf, n);
            fclose(f);
        }
        check(!logged.empty(), "the transcript captured something");
        check(logged.find('\033') == std::string::npos,
              "no escape code ever reaches the file, even with colour on");
        check(logged.find("first line") != std::string::npos &&
              logged.find("Found a share of difficulty 8.0k (3.9x target of 2048)") != std::string::npos,
              "event lines land in the transcript verbatim");

        std::vector<std::string> log_lines = split_lines(logged);
        bool all_stamped = !log_lines.empty();
        for (const std::string& ln : log_lines) {
            // "[YYYY-MM-DD HH:MM:SS] " -- 22 characters before the text.
            if (ln.size() < 22 || ln[0] != '[' || ln[11] != ' ' || ln[20] != ']') all_stamped = false;
        }
        check(all_stamped, "every transcript line is timestamped, block lines included");
        check(log_lines.size() > 3,
              "the multi-line stats block is split into stamped lines, not written as one");

        ui::console::open_log(path);
        capture([] {
            ui::console::init(true);
            ui::console::info("second session");
        });
        ui::console::close_log();
        std::string after;
        if (FILE* f = fopen(path.c_str(), "rb")) {
            char buf[4096];
            size_t n;
            while ((n = fread(buf, 1, sizeof buf, f)) > 0) after.append(buf, n);
            fclose(f);
        }
        check(after.find("first line") != std::string::npos &&
              after.find("second session") != std::string::npos,
              "reopening appends -- a restart continues the record, never truncates it");

        std::remove(path.c_str());

        check(!ui::console::open_log("/nonexistent-dir-mxbm/deep/er/x.log"),
              "an unopenable path reports failure rather than throwing or aborting");
        ui::console::close_log();
    }

    return summary("console");
}
