#pragma once
#include <chrono>
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
//
// `target_units` is the job's target in the same units. When positive, the
// line gains the achieved/target multiple -- "8.0k (3.9x target)" -- which is
// what turns a bare number into a judgement: difficulty alone means nothing
// without the bar it had to clear, and pool vardiff moves that bar around all
// session. Pass <= 0 (the default) when no target is known yet, and the
// suffix is omitted rather than printing a meaningless ratio.
void share_found(const std::string& device, double units, double target_units = -1.0);

// ms < 0 (default) means "latency unknown" -- Phase A does not measure
// submit-to-result round-trip -- and is omitted from the printed line.
void share_result(int code, const std::string& description, long long ms = -1);

// Red "Pool connection lost - reconnecting..." -- wire to
// stratum::Client::on_disconnect.
void disconnected();

// -- developer fee ----------------------------------------------------------
//
// The fee is announced, not concealed: one disclosure line at startup
// stating the rate, the cadence and the destination, and a line at each end
// of every round so the pool change is never a surprise. See
// miner/devfee.h for why an open-source miner gains nothing from hiding it.

// "Dev fee: 1.0% - one 36s round per 60min of mining, to <host>:<port>."
void devfee_notice(double rate, std::chrono::seconds slice,
                   std::chrono::seconds cycle, const std::string& pool);

// "Dev fee round started (36s) - mining to the developer's address" and
// "Dev fee round finished (36s) - back on your pool".
void devfee_start(std::chrono::seconds slice);
void devfee_end(std::chrono::seconds slice);

void info(const std::string& msg);

// The periodic multi-line statistics table, rendered blue to separate it from
// the surrounding event lines. Takes an already-assembled block from
// ui::format_stats_block; suppressed to plain text by init(true).
void stats_block(const std::string& block);
void error(const std::string& msg);

} } } // namespace mxbm::ui::console
