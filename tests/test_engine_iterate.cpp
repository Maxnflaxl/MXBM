// Engine continuous-iteration + abort-wiring test (M3 Phase C, Task 3).
//
// Uses a CountingSolver fake (host-only, no real BeamHash search) whose
// solve() BLOCKS on a condition_variable until the test explicitly releases
// it -- or its abort flag gets set -- so the test can drive Engine's real
// worker thread (start()/on_job()/stop(), actual std::thread) through each
// step deterministically via a CV handshake. No bare sleeps and no
// wall-clock races are used as synchronization anywhere below; the only
// timeouts present are bounded backstops that turn an unexpected hang into
// a clean test FAILURE instead of blocking the test process forever (see
// CountingSolver::wait_parked_at_least and the stop() section at the
// bottom) -- never the primary synchronization mechanism.
//
// Covers all three Task 3 changes:
//   (a) worker_main() mines one posted job CONTINUOUSLY -- many solve()
//       calls, not one -- until a newer job or stop preempts it.
//   (b) on_job() of a new job calls Solver::request_abort(), and the
//       worker's very next solve() call carries the new job's input.
//   (c) stop() calls Solver::request_abort() and joins without hanging.
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "check.h"
#include "miner/engine.h"
#include "miner/solver.h"
#include "stratum/client.h"
#include "vectors/stratum_wire.h"

using namespace mxbm;
using namespace mxbm::stratum;
using namespace mxbm::miner;

namespace {

// Independent hex decoder (deliberately not shared with engine.cpp), like
// test_engine.cpp's, so the byte comparisons below pin what process_job()
// actually decoded against a second implementation.
void from_hex(const std::string& hex, uint8_t* out) {
    auto nibble = [](char c) { return (c <= '9') ? c - '0' : c - 'a' + 10; };
    for (size_t i = 0; i < hex.size() / 2; ++i)
        out[i] = (uint8_t)((nibble(hex[i * 2]) << 4) | nibble(hex[i * 2 + 1]));
}

// A Solver whose solve() parks on a condition_variable until the test hands
// it a release token (release_one()/flood_release()) or Engine calls
// request_abort() -- so the test controls exactly how far the worker thread
// advances, one solve() call at a time. All mutable state is guarded by
// `mtx`; `waiting_cv` announces "a call just started and is parked" to the
// test thread, `release_cv` is how the test (or request_abort()) tells a
// parked call "return now".
struct CountingSolver : Solver {
    std::mutex mtx;
    std::condition_variable waiting_cv;
    std::condition_variable release_cv;

    int call_count = 0;
    bool currently_waiting = false;
    uint64_t release_budget = 0;
    int abort_call_count = 0;
    std::atomic<bool> abort_requested{false};
    std::array<uint8_t, 32> last_input{};

    std::vector<std::array<uint8_t, 104>> solve(const uint8_t input[32], const uint8_t[8]) override {
        std::unique_lock<std::mutex> lock(mtx);
        // Mirrors GpuSolver::solve()'s contract (src/gpu/gpu_solver.cpp):
        // reset the abort flag at ENTRY, so Engine never has to reset it
        // itself and a freshly-switched-to job's first solve() always runs
        // clean.
        abort_requested.store(false, std::memory_order_relaxed);
        std::memcpy(last_input.data(), input, 32);
        ++call_count;
        currently_waiting = true;
        waiting_cv.notify_all();
        release_cv.wait(lock, [this] {
            return release_budget > 0 || abort_requested.load(std::memory_order_relaxed);
        });
        currently_waiting = false;
        if (release_budget > 0) --release_budget;
        return {};   // never any candidates -- keeps process_job()'s difficulty filter a no-op
    }

    void request_abort() override {
        std::lock_guard<std::mutex> lock(mtx);
        ++abort_call_count;
        abort_requested.store(true, std::memory_order_relaxed);
        release_cv.notify_all();
    }

    // -- Test-thread helpers: every access goes through mtx, never races
    //    solve()'s own access to the same fields. --

    // Waits until a solve() call is currently parked with call_count >= n.
    // The timeout is a backstop only (see file header): the real
    // synchronization is waiting_cv, notified exactly when solve() parks.
    bool wait_parked_at_least(int n, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mtx);
        return waiting_cv.wait_for(lock, timeout,
            [this, n] { return currently_waiting && call_count >= n; });
    }

    void release_one() {
        std::lock_guard<std::mutex> lock(mtx);
        ++release_budget;
        release_cv.notify_all();
    }

    // Used only right before the stop() section below: makes every current
    // or future solve() call return immediately, so stop()'s join() can
    // never block on THIS FAKE regardless of scheduling. (The real
    // GpuSolver is self-bounding via its own pipeline runtime instead of an
    // external token -- see gpu_solver.cpp -- so it doesn't need this; this
    // fake does, since nothing else ever hands out a release token once
    // we've stopped tracking call-by-call state.)
    void flood_release() {
        std::lock_guard<std::mutex> lock(mtx);
        release_budget += 1000000;
        release_cv.notify_all();
    }

    int snapshot_call_count() {
        std::lock_guard<std::mutex> lock(mtx);
        return call_count;
    }
    int snapshot_abort_call_count() {
        std::lock_guard<std::mutex> lock(mtx);
        return abort_call_count;
    }
    std::array<uint8_t, 32> snapshot_last_input() {
        std::lock_guard<std::mutex> lock(mtx);
        return last_input;
    }
};

} // namespace

int main() {
    using namespace std::chrono_literals;
    const auto backstop = 2000ms;   // failure-only backstop, see file header

    // job1's input: the same 64-lowercase-hex-char (32-byte) fixture bytes
    // as test_engine.cpp's valid_input.
    const std::string job1_input =
        "636b90cc38bc7a347f074d9ca97c3a2158330f6844f8f52075a38a15ab483223";
    check(job1_input.size() == 64, "job1 fixture input is 64 hex chars");

    // job2's input: built programmatically (not hand-typed) so its length
    // can't be miscounted -- 64 lowercase hex chars, distinct from job1's
    // from the very first byte.
    std::string job2_input(64, 'a');
    job2_input[0] = 'b';
    check(job2_input.size() == 64 && job2_input != job1_input,
          "job2 fixture input is 64 hex chars and differs from job1's");

    uint8_t job1_bytes[32], job2_bytes[32];
    from_hex(job1_input, job1_bytes);
    from_hex(job2_input, job2_bytes);

    CountingSolver solver;
    Client client;
    client.handle_line(wire::kResultLoginOk);   // seeds nonceprefix "a1f", as in test_engine.cpp

    Engine engine(client, solver);
    engine.submit_fn = [](const Solution&) {};   // CountingSolver never returns candidates; defensive no-op

    Job job1{"job-1", job1_input, 0u, 2500000};
    Job job2{"job-2", job2_input, 0u, 2500001};

    engine.start();

    // ---- (a) continuous iteration: one job posted -> MANY solve() calls ----
    engine.on_job(job1);

    const int kIterations = 5;   // clean, released calls -- demonstrates "many", not "once"
    for (int i = 1; i <= kIterations; ++i) {
        bool parked = solver.wait_parked_at_least(i, backstop);
        char msg[128];
        std::snprintf(msg, sizeof msg, "job1 solve() call reached count>=%d promptly (no hang)", i);
        check(parked, msg);
        if (!parked) { engine.stop(); return summary("engine_iterate"); }
        solver.release_one();
    }

    // One further job1 call, deliberately left blocked (not released) so
    // its exact call_count can be pinned right before triggering the switch
    // below.
    bool parked_last_job1 = solver.wait_parked_at_least(kIterations + 1, backstop);
    check(parked_last_job1, "one further job1 solve() call parked, ready for the abort/switch test");
    if (!parked_last_job1) { engine.stop(); return summary("engine_iterate"); }

    int calls_before_switch = solver.snapshot_call_count();
    check(calls_before_switch >= kIterations + 1,
          "worker called solve() many times for job1 with no second job ever posted (continuous iteration)");
    {
        auto got = solver.snapshot_last_input();
        check_eq_bytes(got.data(), job1_bytes, 32,
                        "the parked call is still for job1's input (no premature switch)");
    }

    // ---- (b) on_job() of a new job triggers request_abort(); the worker's
    //          very next solve() call carries the new job's input ----
    int aborts_before_switch = solver.snapshot_abort_call_count();
    engine.on_job(job2);

    // Strictly greater than calls_before_switch: only a solve() call that
    // STARTED after the snapshot above can satisfy this, so whatever input
    // we read next is causally the result of on_job(job2) -- not a stale
    // read of the still-parked job1 call from before the trigger.
    bool parked_job2 = solver.wait_parked_at_least(calls_before_switch + 1, backstop);
    check(parked_job2, "worker started a new solve() call after on_job(job2) (switch happened, no hang)");
    if (!parked_job2) { engine.stop(); return summary("engine_iterate"); }

    check(solver.snapshot_abort_call_count() >= aborts_before_switch + 1,
          "on_job() called Solver::request_abort() at least once");
    {
        auto got = solver.snapshot_last_input();
        check_eq_bytes(got.data(), job2_bytes, 32,
                        "the worker's very next solve() call after on_job(job2) carries job2's input");
    }
    // ---- (c) stop() triggers request_abort() and joins without hanging ----
    // From here on exact call counts no longer matter, only that stop()
    // cannot block forever: flood the release budget so every current-or-
    // future solve() call on this fake returns immediately, no matter how
    // Engine's internal stop_requested_/has_job_ check happens to interleave
    // with the currently-parked call's wakeup. (The real GpuSolver is
    // self-bounding via its own pipeline runtime instead of an external
    // token; this flood reproduces that same "always eventually returns"
    // property for this fake, so a benign extra solve() iteration racing
    // stop()'s flag-set can never turn into an infinite block.)
    solver.flood_release();

    int aborts_before_stop = solver.snapshot_abort_call_count();

    // Run stop() on a helper thread with a bounded watchdog: the primary
    // guarantee that this can't hang is the flood above (plus
    // request_abort()); the watchdog is purely the backstop the file header
    // describes, turning an unexpected hang into a clean FAIL instead of
    // wedging the test process.
    std::mutex stop_mtx;
    std::condition_variable stop_cv;
    bool stop_done = false;
    std::thread stopper([&] {
        engine.stop();
        {
            std::lock_guard<std::mutex> lock(stop_mtx);
            stop_done = true;
        }
        stop_cv.notify_one();
    });

    bool stopped_in_time;
    {
        std::unique_lock<std::mutex> lock(stop_mtx);
        stopped_in_time = stop_cv.wait_for(lock, 5000ms, [&] { return stop_done; });
    }
    check(stopped_in_time, "stop() joined the worker thread within the backstop timeout (no hang)");
    if (!stopped_in_time) {
        stopper.detach();   // don't risk hanging the test process on an already-failed assertion
        return summary("engine_iterate");
    }
    stopper.join();

    check(solver.snapshot_abort_call_count() >= aborts_before_stop + 1,
          "stop() called Solver::request_abort()");

    return summary("engine_iterate");
}
