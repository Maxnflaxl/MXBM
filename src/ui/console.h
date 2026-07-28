#pragma once
#include <chrono>
#include <cstdint>
#include <string>

namespace mxbm { namespace ui { namespace console {

// the reference miner-flavored console output. Every function below prints one line (or,
// for the ticker's stats block, one assembled multi-line string) to stdout and
// fflushes immediately: miners are commonly watched through `tee`/log pipes,
// which would otherwise sit behind stdio's own buffering until the process
// exits. init(true) ("--nocolor") suppresses the colouring.
void init(bool nocolor);

// -- transcript log (--log / --logfile) --------------------------------------
//
// Tees everything printed above into a file, uncoloured and ALWAYS timestamped
// ("[YYYY-MM-DD HH:MM:SS] " per line) regardless of --timeprint: that flag is
// about the console, watched live, where the time is usually redundant; a file
// is read hours later, where a line with no time on it is nearly useless.
//
// Opens `path` for APPEND -- a restart continues the record instead of
// truncating it, which is what matters on a rig that watchdog-restarts
// overnight. An empty `path` means "logs/mxbm_<YYYY-MM-DD_HH-MM-SS>.log",
// creating logs/ if needed. Returns false if it could not be opened, leaving
// logging off; `resolved_path` (when non-null) receives the path opened.
bool open_log(const std::string& path, std::string* resolved_path = nullptr);

// Flushes and closes the transcript. Safe to call when no log is open.
void close_log();

void banner();

// -- startup sequence, in the reference miner's order and wording: hardware, then pool ---

void setup_miner();

// "<API> driver detected." / "Number of <API> supported GPUs: N".
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
// as a blank or a guess: the address comes from NVML, absent on a non-NVIDIA
// card, and a guessed PCI address is worse than no line at all. `active_detail`
// is the parenthetical after "true"; "" means inactive.
void device_block(int index, const std::string& name, const std::string& address,
                  const std::string& vendor, const std::string& driver,
                  unsigned long long memory_bytes, const std::string& active_detail);

void connecting_to_pool();

void connecting(const std::string& host, uint16_t port, bool tls);

// The resolved IP goes on this line because a pool that round-robins across
// regions is otherwise impossible to tell apart in a log. `ip` may be empty.
void connected_to(const std::string& host, const std::string& ip, uint16_t port, bool tls);

void tls_handshake_ok();

void connected(bool tls);
void authorized(const std::string& user);
void start_mining();

// "New job received for blockheight <H> (job <id>) Difficulty: <N>", matching
// the reference miner's wording for the height so one grep works across both miners' logs,
// but keeping the job id it drops. A `height` of 0 means the pool sent none, and
// the line falls back to "New job received: <id> Difficulty: <N>".
//
// Difficulty is a plain rounded integer (pow::to_display_units), NOT the k/M
// format_units() notation the share lines use.
void job(const std::string& id, uint32_t difficulty, uint64_t height);

// "CPU 0: Found a share of difficulty 8.0k" (green). `units` and `target_units`
// are both in pow::to_display_units units; when `target_units` is positive the
// line gains the achieved/target multiple, since a difficulty means nothing
// without the bar it cleared. Pass <= 0 and the suffix is omitted.
void share_found(const std::string& device, double units, double target_units = -1.0);

// ms < 0 (default) means "latency unknown" and is omitted from the line.
void share_result(int code, const std::string& description, long long ms = -1);

void disconnected();

// -- developer fee ----------------------------------------------------------
// The fee is announced, not concealed: one disclosure line at startup stating
// rate, cadence and destination, and a line at each end of every round. See
// miner/devfee.h for why an open-source miner gains nothing from hiding it.

void devfee_notice(double rate, std::chrono::seconds slice,
                   std::chrono::seconds cycle, const std::string& pool);

void devfee_start(std::chrono::seconds slice);
void devfee_end(std::chrono::seconds slice);

void info(const std::string& msg);

// The periodic multi-line statistics table, rendered blue to separate it from
// the surrounding event lines. Block comes assembled from format_stats_block().
void stats_block(const std::string& block);
void error(const std::string& msg);

} } } // namespace mxbm::ui::console
