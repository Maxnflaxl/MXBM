#pragma once
#include <functional>
#include <thread>

namespace mxbm { namespace ui {

// Single-key console commands, read raw from stdin while mining.
enum class Key {
    None,        // unmapped -- ignored, so escape sequences stay silent
    Speed,       // h: the short speed line, immediately
    StatsBlock,  // s: the full statistics table, immediately
    Connection,  // c: pool, uptime, last job, share latency
    Pause,       // p: hold every device out of mining
    Resume,      // r: resume after p
    Help,        // ?: the key list
};

// Pure key -> command mapping, case-insensitive. Everything unmapped is None:
// a terminal sends multi-byte escape sequences for arrows and function keys,
// and reacting to their tail bytes would fire commands nobody typed.
Key key_command(char c);

// One line naming every key; printed at startup and on '?'.
const char* key_help_line();

// Owns the reader thread and the terminal mode. start() switches stdin to
// unbuffered single-key input and undoes it on stop(), at exit, and on a fatal
// signal (chaining any handler already installed, e.g. the overclock restore).
// On a non-interactive stdin -- a pipe, a service, a redirect -- start()
// refuses and changes nothing, so scripted runs never see a mode change.
class KeyReader {
public:
    ~KeyReader();

    // Spawns the reader; `on_key` runs on the reader's own thread and only for
    // mapped keys. False (and fully inert) when stdin is not a terminal or a
    // reader is already running.
    bool start(std::function<void(Key)> on_key);

    // Signals the reader to stop, joins it, restores the terminal.
    void stop();

private:
    void reader_main();

    std::function<void(Key)> on_key_;
    std::thread reader_;
    bool started_ = false;
};

} } // namespace mxbm::ui
