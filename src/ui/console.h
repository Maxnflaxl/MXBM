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

// -- transcript log (--log / --logfile) --------------------------------------
//
// Tees everything printed above into a file: the same lines, in the same
// order, with two deliberate differences.
//
//   1. Never coloured. A log is read later, by grep and by eye, and ANSI
//      escapes in a file are noise at best and confusing at worst.
//   2. ALWAYS timestamped, one "[YYYY-MM-DD HH:MM:SS] " prefix per line,
//      regardless of --timeprint. That flag is about the console, which is
//      watched live and where the time is usually redundant; a file is read
//      hours later, where a line with no time on it is nearly useless.
//
// Opens `path` for APPEND -- a restart continues the record instead of
// truncating it, which is the one behaviour that matters on a rig that
// watchdog-restarts overnight. An empty `path` means the default,
// "logs/mxbm_<YYYY-MM-DD_HH-MM-SS>.log" relative to the working directory,
// creating logs/ if it does not exist; the per-session filename is why
// appending rarely matters in practice but still costs nothing.
//
// Returns false if the file (or the directory) could not be opened, leaving
// logging off -- a miner must not refuse to mine because it could not write a
// log. On success, `resolved_path` (when non-null) receives the path actually
// opened, which the caller is expected to print so the user knows where it
// went.
bool open_log(const std::string& path, std::string* resolved_path = nullptr);

// Flushes and closes the transcript. Safe to call when no log is open.
void close_log();

void banner();

// -- startup sequence, in the reference miner's order and wording ----------------------
//
// Hardware first, pool second: what the miner found and chose, then where it
// is sending the work. That order is the reference miner's, and it is the more useful one
// -- a device that failed to initialise is the reason a pool connection is
// pointless, so seeing it first saves reading further.

// "Setup Miner..." -- opens the hardware section.
void setup_miner();

// "<API> driver detected." / "Number of <API> supported GPUs: N", the pair
// the reference miner prints per runtime. `count` is what that runtime enumerated.
void driver_detected(const char* api, int count);

// The indented per-device block:
//
//   Device 0:
//       Name:    NVIDIA GeForce RTX 4070 Ti SUPER
//       Address: 1:0
//       Vendor:  NVIDIA Corporation
//       Drivers: Cuda
//       Memory:  15963 MByte
//       Active:  true (Selected Algorithm: BeamHash III (CUDA))
//
// Every field except Name and Active is dropped when empty rather than printed
// as a blank or a guess: MXBM reads the address from NVML, which is absent on
// a non-NVIDIA card, and inventing a PCI address would be worse than omitting
// the line. `active_detail` is the parenthetical after "true"; pass "" for an
// inactive device.
void device_block(int index, const std::string& name, const std::string& address,
                  const std::string& vendor, const std::string& driver,
                  unsigned long long memory_bytes, const std::string& active_detail);

// "Connecting to pool..." -- no host on this line; the host appears on the
// connected() line below, with the address it actually resolved to.
void connecting_to_pool();

void connecting(const std::string& host, uint16_t port, bool tls);

// "Connected to de.beam.herominers.com(141.95.126.31):1130  (TLS enabled)" --
// the resolved IP is on the line because a pool that round-robins across
// regions is otherwise impossible to tell apart in a log. `ip` may be empty
// (then only the hostname is shown).
void connected_to(const std::string& host, const std::string& ip, uint16_t port, bool tls);

// "TLS Handshake success" -- only printed for a TLS connection.
void tls_handshake_ok();

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
