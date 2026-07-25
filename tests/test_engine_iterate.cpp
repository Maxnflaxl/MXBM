// Engine continuous-iteration + abort-wiring test, over the real worker thread.
// CountingSolver's solve() BLOCKS until the test releases it or its abort flag
// is set, so the worker can be driven one step at a time. Nothing below
// synchronizes on a sleep or a wall clock: every timeout is a bounded backstop
// that turns an unexpected hang into a clean FAILURE rather than a wedged run.
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
#include <stdexcept>

#include "check.h"
#include "miner/engine.h"
#include "miner/solver.h"
#include "stratum/client.h"
#include "vectors/stratum_wire.h"

using namespace mxbm;
using namespace mxbm::stratum;
using namespace mxbm::miner;

namespace {

// Independent hex decoder, deliberately not shared with engine.cpp, so the
// byte comparisons below pin what process_job() actually decoded.
void from_hex(const std::string& hex, uint8_t* out) {
    auto nibble = [](char c) { return (c <= '9') ? c - '0' : c - 'a' + 10; };
    for (size_t i = 0; i < hex.size() / 2; ++i)
        out[i] = (uint8_t)((nibble(hex[i * 2]) << 4) | nibble(hex[i * 2 + 1]));
}

// A Solver whose solve() parks until the test hands it a release token or
// Engine calls request_abort(). `waiting_cv` announces "a call just started and
// is parked" to the test thread; `release_cv` tells a parked call "return now".
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
        // Mirrors GpuSolver::solve()'s contract: reset the abort flag at ENTRY,
        // so a freshly-switched-to job's first solve() always runs clean.
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

    // -- Test-thread helpers: every access goes through mtx. --

    // Waits until a solve() call is parked with call_count >= n. The real
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

    // Makes every current or future solve() call return immediately, so stop()'s
    // join() can never block on THIS FAKE. The real GpuSolver is self-bounding
    // via its own pipeline runtime and needs no such token.
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

    // The same 32-byte fixture bytes as test_engine.cpp's valid_input.
    const std::string job1_input =
        "636b90cc38bc7a347f074d9ca97c3a2158330f6844f8f52075a38a15ab483223";
    check(job1_input.size() == 64, "job1 fixture input is 64 hex chars");

    // Built programmatically rather than hand-typed, so its length cannot be
    // miscounted; it differs from job1's from the very first byte.
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
    engine.submit_fn = [](const Solution&, Origin) {};   // CountingSolver never returns candidates; defensive no-op

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

    // Deliberately left blocked, so its exact call_count can be pinned right
    // before the switch below is triggered.
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

    // ---- (b) a new job aborts the current call and switches the input ----
    int aborts_before_switch = solver.snapshot_abort_call_count();
    engine.on_job(job2);

    // Strictly greater than calls_before_switch: only a call that STARTED after
    // the snapshot satisfies this, so the input read next cannot be a stale read
    // of the still-parked job1 call.
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
    // Flooding the release budget gives this fake the "always eventually
    // returns" property the real GpuSolver has, so a benign extra solve()
    // iteration racing stop()'s flag-set cannot become an infinite block.
    solver.flood_release();

    int aborts_before_stop = solver.snapshot_abort_call_count();

    // stop() runs on a helper thread only so the watchdog below can turn an
    // unexpected hang into a clean FAIL rather than a wedged process.
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

    // ---- (d) a throwing solver must not take the process down ----
    // An exception escaping a thread entry function calls std::terminate().
    // Before the guard in worker_main() this test binary itself would abort
    // here -- a GPU losing its VRAM to another process core-dumped the miner.
    {
        struct FlakySolver : Solver {
            std::mutex m;
            std::condition_variable cv;
            int throws_left = 3;
            int successes = 0;
            std::vector<std::array<uint8_t, 104>> solve(const uint8_t*, const uint8_t*) override {
                std::unique_lock<std::mutex> lock(m);
                if (throws_left > 0) { --throws_left; throw std::runtime_error("simulated device failure"); }
                ++successes;
                cv.notify_all();
                return {};
            }
        };

        FlakySolver flaky;
        Client c2;
        c2.handle_line(wire::kResultLoginOk);
        Engine e2(c2, flaky);
        e2.submit_fn = [](const Solution&, Origin) {};
        // Without this seam the three backoffs would be 1s + 2s + 4s; the retry
        // path is what is under test, not the wall-clock duration.
        e2.retry_base_delay = std::chrono::milliseconds(1);

        std::mutex em;
        std::vector<uint32_t> counts;
        std::string first_what;
        e2.on_solver_error = [&](const std::string& what, uint32_t n) {
            std::lock_guard<std::mutex> lock(em);
            if (counts.empty()) first_what = what;
            counts.push_back(n);
        };

        e2.start();
        e2.on_job(Job{"flaky-1", job1_input, 0u, 1});

        bool recovered = false;
        {
            std::unique_lock<std::mutex> lock(flaky.m);
            recovered = flaky.cv.wait_for(lock, 5000ms, [&] { return flaky.successes >= 2; });
        }
        check(recovered, "worker survives a throwing solve() and resumes mining once it recovers");

        e2.stop();

        std::lock_guard<std::mutex> lock(em);
        check(counts.size() == 3, "every failed solve() is reported exactly once");
        check(!counts.empty() && counts.front() == 1, "the consecutive count starts at 1");
        check(counts.size() == 3 && counts[2] == 3,
              "the consecutive count rises while failures continue");
        check(first_what.find("simulated device failure") != std::string::npos,
              "the exception's own message is passed through, not a generic one");
    }

    // ---- (e) stop() stays prompt DURING a retry backoff ----
    // The backoff waits on the mailbox condition rather than sleeping: a plain
    // sleep would trade the crash for a hang, which is barely a fix.
    {
        struct AlwaysThrows : Solver {
            std::vector<std::array<uint8_t, 104>> solve(const uint8_t*, const uint8_t*) override {
                throw std::runtime_error("permanently broken device");
            }
        };
        AlwaysThrows dead;
        Client c3;
        c3.handle_line(wire::kResultLoginOk);
        Engine e3(c3, dead);
        e3.submit_fn = [](const Solution&, Origin) {};
        e3.retry_base_delay = std::chrono::milliseconds(30000);   // 30s: only an interruptible wait passes

        std::atomic<int> errs{0};
        e3.on_solver_error = [&](const std::string&, uint32_t) { errs.fetch_add(1); };
        e3.start();
        e3.on_job(Job{"dead-1", job1_input, 0u, 1});

        // Wait for it to be inside the backoff, then time the shutdown.
        for (int i = 0; i < 500 && errs.load() == 0; ++i) std::this_thread::sleep_for(10ms);
        check(errs.load() >= 1, "the permanently-failing solver reported before the backoff");

        auto t0 = std::chrono::steady_clock::now();
        e3.stop();
        auto elapsed = std::chrono::steady_clock::now() - t0;
        check(elapsed < 3000ms,
              "stop() interrupts the retry backoff instead of waiting it out");
    }

    return summary("engine_iterate");
}
