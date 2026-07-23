#pragma once
#include <cstdint>
#include <string>

namespace mxbm { namespace ui { namespace console {

// the reference miner-flavored console output, matching the reference miner's documented
// console/CLI behavior. Every function below prints one line (or,
// for the ticker's multi-line stats block, one already-assembled multi-line
// string) to stdout and fflushes immediately: miners are commonly watched
// through `tee`/log pipes, which would otherwise sit behind stdio's own
// buffering until the process exits. Positive events (connected, authorized,
// share found/accepted) render bold green; errors/disconnects render red;
// both are suppressed by init(true) ("--nocolor"). The pure formatting logic
// (unit abbreviation, the speed line, the stats block) lives in ui::format,
// consumed here and by ui::Ticker -- console.cpp itself stays a thin,
// mostly-mechanical print layer.
void init(bool nocolor);

void banner();
void connecting(const std::string& host, uint16_t port, bool tls);
void connected(bool tls);
void authorized(const std::string& user);
void start_mining();

// "New job received: <id> Difficulty: <N>" -- difficulty is printed as a
// plain rounded integer (pow::to_display_units, %.0f), NOT the k/M
// format_units() notation used for share/best-share lines (matches
// the reference miner's own job-line style, which is unabbreviated). `height` is
// accepted for signature stability (main.cpp's on_job call site passes
// j.height unconditionally) but is currently unused -- Phase A's
// height-suffix was dropped to match this wording exactly; a future task
// may reintroduce it.
void job(const std::string& id, uint32_t difficulty, uint64_t height);

// "<device>: Found a share of difficulty <format_units(units)>", e.g.
// "CPU 0: Found a share of difficulty 8.0k" (green). `units` is the
// achieved share difficulty in the same display units as
// pow::achieved_units/pow::to_display_units.
void share_found(const std::string& device, double units);

// ms < 0 (default) means "latency unknown" -- Phase A does not measure
// submit-to-result round-trip -- and is omitted from the printed line.
void share_result(int code, const std::string& description, long long ms = -1);

// Red "Pool connection lost - reconnecting..." -- wire to
// stratum::Client::on_disconnect.
void disconnected();

void info(const std::string& msg);
void error(const std::string& msg);

} } } // namespace mxbm::ui::console
