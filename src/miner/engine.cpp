#include "miner/engine.h"

#include <cstring>

#include "pow/difficulty.h"

namespace mxbm { namespace miner {

namespace {

// Lowercase hex encode.
std::string to_hex(const uint8_t* data, size_t len) {
    static const char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(len * 2);
    for (size_t i = 0; i < len; ++i) {
        out.push_back(digits[data[i] >> 4]);
        out.push_back(digits[data[i] & 0xF]);
    }
    return out;
}

// Strict hex decode: `hex` must be exactly out_len*2 lowercase-hex-digit
// characters. Any other character (including uppercase) or a length
// mismatch fails without touching `out`.
bool from_hex_strict(const std::string& hex, uint8_t* out, size_t out_len) {
    if (hex.size() != out_len * 2) return false;
    for (size_t i = 0; i < out_len; ++i) {
        int hi = -1, lo = -1;
        char ch = hex[i * 2];
        if (ch >= '0' && ch <= '9') hi = ch - '0';
        else if (ch >= 'a' && ch <= 'f') hi = ch - 'a' + 10;
        char cl = hex[i * 2 + 1];
        if (cl >= '0' && cl <= '9') lo = cl - '0';
        else if (cl >= 'a' && cl <= 'f') lo = cl - 'a' + 10;
        if (hi < 0 || lo < 0) return false;
        out[i] = (uint8_t)((hi << 4) | lo);
    }
    return true;
}

// Builds the 8-byte nonce: `prefix` (lowercase hex, 0-6 chars / <=3 bytes,
// per stratum::Client::current_nonceprefix()) is written into the leading
// hex positions of the nonce's 16-hex-char memory-order representation --
// an odd-length prefix's last nibble becomes the HIGH nibble of its byte,
// leaving that byte's low nibble at 0. `counter` fills the remaining WHOLE
// trailing bytes, little-endian, starting at the first byte the prefix
// doesn't touch at all.
void build_nonce(const std::string& prefix, uint64_t counter, uint8_t out[8]) {
    std::memset(out, 0, 8);
    size_t nibbles = prefix.size();
    if (nibbles > 16) nibbles = 16;   // defensive; contract caps this at 6
    for (size_t i = 0; i < nibbles; ++i) {
        char c = prefix[i];
        int v = (c >= '0' && c <= '9') ? c - '0'
              : (c >= 'a' && c <= 'f') ? c - 'a' + 10
              : 0;   // defensive: current_nonceprefix() is documented lowercase hex
        size_t byte_idx = i / 2;
        if (i % 2 == 0) out[byte_idx] = (uint8_t)(out[byte_idx] | (v << 4));
        else            out[byte_idx] = (uint8_t)(out[byte_idx] | v);
    }
    size_t first_counter_byte = (nibbles + 1) / 2;   // ceil(nibbles/2): skip a partial prefix byte
    for (size_t b = first_counter_byte; b < 8 && counter != 0; ++b) {
        out[b] = (uint8_t)(counter & 0xFF);
        counter >>= 8;
    }
}

} // namespace

Engine::Engine(stratum::Client& client, Solver& solver)
    : client_(client), solver_(solver) {
    submit_fn = [this](const stratum::Solution& s) { client_.submit(s); };
}

Engine::~Engine() {
    stop();
}

void Engine::process_job(const stratum::Job& job, const std::string& nonceprefix) {
    uint8_t input32[32];
    if (!from_hex_strict(job.input, input32, 32)) return;   // malformed: no crash, no submit

    uint8_t nonce8[8];
    build_nonce(nonceprefix, nonce_counter_++, nonce8);

    std::vector<std::array<uint8_t, 104>> candidates = solver_.solve(input32, nonce8);

    if (on_attempt) on_attempt((uint32_t)candidates.size());   // stats hook, before any filtering

    if (superseded_by_newer_job(job.id)) return;   // a newer, different job pre-empted this one

    std::string nonce_hex = to_hex(nonce8, 8);
    for (const auto& cand : candidates) {
        if (!pow::clears_difficulty(cand.data(), job.difficulty)) continue;
        submit_fn(stratum::Solution{job.id, nonce_hex, to_hex(cand.data(), cand.size())});
    }
}

void Engine::on_job(const stratum::Job& job) {
    // Runs on Client::run()'s thread -- the same thread that writes
    // Client::nonceprefix_ -- so this current_nonceprefix() read is
    // race-free (see class comment in engine.h). Snapshot it into the
    // mailbox alongside the job so the worker thread never has to touch
    // client_ itself.
    std::string prefix = client_.current_nonceprefix();
    {
        std::lock_guard<std::mutex> lock(mailbox_mutex_);
        latest_job_ = job;
        latest_nonceprefix_ = prefix;
        has_job_ = true;
    }
    mailbox_cv_.notify_one();
    // Ask any in-flight solve() to return promptly so the worker abandons
    // the now-superseded job's current nonce attempt instead of finishing
    // it first. Default no-op (solver.h) on solvers, like SolverRef, that
    // can't interrupt mid-call -- cheap and safe to call unconditionally
    // either way.
    solver_.request_abort();
}

void Engine::start() {
    if (started_) return;
    started_ = true;
    stop_requested_ = false;
    worker_ = std::thread(&Engine::worker_main, this);
}

void Engine::stop() {
    if (!started_) return;
    // Ask any in-flight solve() to return promptly, before signalling
    // stop_requested_, so a multi-second-or-longer solve() doesn't make
    // this join() block for its full remaining duration. Default no-op
    // (solver.h) on solvers, like SolverRef, that can't interrupt mid-call.
    solver_.request_abort();
    {
        std::lock_guard<std::mutex> lock(mailbox_mutex_);
        stop_requested_ = true;
    }
    mailbox_cv_.notify_one();
    if (worker_.joinable()) worker_.join();
    started_ = false;
}

void Engine::worker_main() {
    while (true) {
        stratum::Job job;
        std::string prefix;
        {
            std::unique_lock<std::mutex> lock(mailbox_mutex_);
            mailbox_cv_.wait(lock, [this] { return has_job_ || stop_requested_; });
            if (stop_requested_) return;   // stop wins over starting another solve
            job = latest_job_;
            prefix = latest_nonceprefix_;
            has_job_ = false;
        }
        // A malformed job.input never becomes solvable, and process_job()
        // treats it as an instant no-op early-return (from_hex_strict fails
        // before any solve()). Mining it in the continuous inner loop below
        // would busy-spin one core until the next job arrives -- job/prefix
        // are fixed for the whole inner loop, so the malformed condition can
        // never clear. Validate once here and, if bad, drop straight back to
        // waiting for a newer job.
        {
            uint8_t input32[32];
            if (!from_hex_strict(job.input, input32, 32)) continue;
        }
        // Mine this job continuously: each process_job() call tries the
        // next nonce (nonce_counter_ advances every call -- see its own
        // comment). Keep calling it for this SAME (job, prefix) until a
        // newer job lands in the mailbox or a stop is requested --
        // re-checked under mailbox_mutex_ between iterations so a
        // concurrent on_job()/stop() is never missed. When this breaks for
        // a new job, the outer loop above picks it up immediately (its
        // cv.wait()'s predicate is already satisfied, so it returns without
        // actually blocking), and solve()'s own abort-flag reset (e.g.
        // GpuSolver, src/gpu/gpu_solver.cpp) makes that next call start
        // clean.
        while (true) {
            process_job(job, prefix);   // may block here for a while with a real solver
            std::lock_guard<std::mutex> lock(mailbox_mutex_);
            if (stop_requested_ || has_job_) break;
        }
    }
}

bool Engine::superseded_by_newer_job(const std::string& job_id) const {
    std::lock_guard<std::mutex> lock(mailbox_mutex_);
    return has_job_ && latest_job_.id != job_id;
}

} } // namespace mxbm::miner
