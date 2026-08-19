#pragma once
// --report: one command that produces a paste-ready benchmark report for this card.
//
// The measurement is not new: the throughput block is run_benchmark(), the same
// Engine path mining drives, and the power curve is --tune's, read back from its
// store or measured on the spot. What this adds is the framing -- the hardware and
// driver block, the telemetry taken over the run, and the three conditions that
// decide whether a figure means anything (a display on the card, another process on
// the card, and what the clocks were limited by).
//
// A stored curve is republished only when the binary that measured it is the binary
// running now. Any other build re-measures: a curve taken on different kernels is a
// curve of a different program.
#include <atomic>
#include <memory>
#include <string>

#include "miner/solver.h"
#include "miner/stats.h"
#include "miner/tune.h"

namespace mxbm { namespace miner {

// Whether a stored curve may be republished as this build's measurement, as a pure
// function so a test can pin it without a GPU. Only an exact match counts: a curve
// taken on different kernels describes a different program. An unstamped build never
// matches -- without a git count two different local builds carry the same string,
// so "same version" would stop meaning "same code".
inline bool report_curve_fresh(const std::string& stored, const std::string& running) {
    if (stored.empty() || running.empty()) return false;
    if (running.find("[nogit]") != std::string::npos) return false;
    return stored == running;
}

struct ReportConfig {
    unsigned    device = 0;          // bus order: the card measured and reported on
    int         seconds = 120;       // --report-seconds
    std::string device_key;          // the key --tune stores under
    std::string device_name;         // fallback when NVML reports no name
    std::string backend;             // "CUDA" / "OpenCL" / "Metal" / "CPU reference"
    bool        have_nvml = false;
    // Where this card sits on the rig. A measuring mode drives ONE card, so a
    // multi-GPU report has to name which, and say what covers the others -- a block
    // that silently describes card 0 while the device table above it lists four is
    // a report about a rig it did not measure.
    unsigned    device_position = 0;   // as --devices numbers it
    unsigned    device_count = 1;      // cards detected on this rig
    // How to sweep, when a sweep is needed. Carries the solver-rebuild hook and the
    // live-solver publication that --tune needs; run_report fills in nothing else.
    TuneConfig  tune;
};

// The file a report lands in when the user named no path: next to the tune store, in
// the invoking user's config dir, as `report-<gpu>-<date>.md`. `gpu` is the card's
// marketing name slugged (a PCI address is nothing to a reader); `date` is local
// YYYY-MM-DD, so a second run the same day replaces the first rather than littering.
// A run covering several cards writes one file for the rig instead, since its blocks
// are collected into one paste.
std::string report_default_path(const std::string& gpu_name, bool whole_rig);

// Where a report will actually be written, given what the user asked for.
//
//   empty      -> report_default_path()
//   a file     -> that file; a relative path is relative to the working directory,
//                 and a leading ~/ expands (a quoted argument reaches us unexpanded)
//   a directory-> the default FILENAME inside it, rather than a file named after the
//                 directory the user meant to put it in
//
// Missing parent directories are created. `err` is set and "" returned when the path
// cannot be made writable, which is a reason to say so, never to lose the report.
std::string report_resolve_path(const std::string& requested, const std::string& gpu_name,
                                bool whole_rig, std::string& err);

// Writes `text` there and hands the file back to the invoking user when running under
// sudo. False with `err` set on failure; the caller still has the block on screen.
bool report_write(const std::string& path, const std::string& text, std::string& err);

// Runs the report for ONE card and prints its markdown block. When `collected` is
// given the block is appended to it as well, so a multi-card run can write every
// card's block into one file -- one paste, one issue.
//
// Returns a process exit code; 0 means a report was produced, which includes the
// reduced one a card with no privilege (no power curve) or no NVML (no telemetry)
// can still support.
int run_report(std::unique_ptr<Solver>& solver, Stats& stats, const ReportConfig& cfg,
               std::atomic<bool>& stop, std::string* collected = nullptr);

}} // namespace mxbm::miner
