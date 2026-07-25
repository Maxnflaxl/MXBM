// Engine pipeline test: job -> solve -> difficulty filter -> submit. Uses a
// FakeSolver (canned candidate, no real BeamHash search) and overrides
// Engine::submit_fn to capture submissions -- no threads, no real transport;
// see miner/engine.h for why process_job() alone is enough to test this
// deterministically.
#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "check.h"
#include "miner/engine.h"
#include "miner/solver.h"
#include "pow/difficulty.h"
#include "stratum/client.h"
#include "vectors/stratum_wire.h"

using namespace mxbm;
using namespace mxbm::stratum;
using namespace mxbm::miner;

namespace {

// Fixed 104-byte candidate (bytes 0..103), regardless of input/nonce.
std::array<uint8_t, 104> fixture_candidate() {
    std::array<uint8_t, 104> cand{};
    for (int i = 0; i < 104; ++i) cand[(size_t)i] = (uint8_t)i;
    return cand;
}

// Returns the fixed candidate regardless of input/nonce, but records the
// last (input, nonce) it was called with so the test can independently pin
// down process_job()'s hex-decode and nonce-construction byte layout --
// not just the hex format of the Solution it eventually submits.
struct FakeSolver : Solver {
    std::array<uint8_t, 32> last_input{};
    std::array<uint8_t, 8> last_nonce{};

    std::vector<std::array<uint8_t, 104>> solve(const uint8_t input[32],
                                                  const uint8_t nonce[8]) override {
        std::memcpy(last_input.data(), input, 32);
        std::memcpy(last_nonce.data(), nonce, 8);
        return { fixture_candidate() };
    }
};

std::string to_hex(const uint8_t* data, size_t len) {
    static const char* digits = "0123456789abcdef";
    std::string out;
    out.reserve(len * 2);
    for (size_t i = 0; i < len; ++i) {
        out.push_back(digits[data[i] >> 4]);
        out.push_back(digits[data[i] & 0xF]);
    }
    return out;
}

bool is_lowercase_hex(const std::string& s) {
    for (char c : s) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    return true;
}

// Independent hex decoder (deliberately not shared with engine.cpp) so the
// byte-level checks below pin process_job()'s actual decode/nonce-layout
// behavior against a second implementation, not against itself.
void from_hex(const std::string& hex, uint8_t* out) {
    auto nibble = [](char c) { return (c <= '9') ? c - '0' : c - 'a' + 10; };
    for (size_t i = 0; i < hex.size() / 2; ++i)
        out[i] = (uint8_t)((nibble(hex[i * 2]) << 4) | nibble(hex[i * 2 + 1]));
}

// Deterministically simulates "a newer job for a different id lands in the
// mailbox while solve() is in flight" -- without any real thread -- by
// calling Engine::on_job() from inside solve() itself, right before
// returning a normally-clearing candidate. Exercises process_job()'s
// post-solve staleness check with nothing but synchronous calls.
struct StaleInjectingSolver : Solver {
    Engine* engine = nullptr;
    Job newer_job;

    std::vector<std::array<uint8_t, 104>> solve(const uint8_t[32], const uint8_t[8]) override {
        engine->on_job(newer_job);
        return { fixture_candidate() };
    }
};

} // namespace

int main() {
    // 64 lowercase hex chars = 32 bytes; the same input bytes as
    // tests/vectors/stratum_wire.h's kJob fixture.
    const std::string valid_input =
        "636b90cc38bc7a347f074d9ca97c3a2158330f6844f8f52075a38a15ab483223";
    check(valid_input.size() == 64, "fixture input is 64 hex chars");

    // packed=0 -> mantissa=2^24, order=0: hash*mantissa < 2^280 always holds
    // (hash < 2^256), so this difficulty clears for ANY candidate -- no
    // hand-computed SHA-256 needed. packed=(200<<24)|0xFFFFFF is about as
    // hard as the format allows, so the fixed fixture candidate practically
    // never clears it; asserted below rather than assumed.
    const uint32_t easy_difficulty = 0u;
    const uint32_t hard_difficulty = (200u << 24) | 0xFFFFFFu;

    std::array<uint8_t, 104> cand = fixture_candidate();
    check(pow::clears_difficulty(cand.data(), easy_difficulty),
          "fixture candidate clears easy difficulty (packed=0)");
    check(!pow::clears_difficulty(cand.data(), hard_difficulty),
          "fixture candidate does not clear hard difficulty");

    FakeSolver solver;
    Client client;
    client.handle_line(wire::kResultLoginOk);   // seeds nonceprefix "a1f"
    check(client.current_nonceprefix() == "a1f", "client nonceprefix seeded for the test");

    Engine engine(client, solver);
    std::vector<Solution> submits;
    engine.submit_fn = [&](const Solution& s, Origin) { submits.push_back(s); };

    // Stats hook coverage: on_attempt must fire once per solve() return with
    // the candidate count, BEFORE any difficulty filtering -- set it here so
    // Case 1 below (which the FakeSolver always answers with exactly one
    // candidate) exercises it. Every case from here on reuses `engine`, so
    // this stays installed for Cases 2/3/5 too, which is fine: FakeSolver's
    // answer never changes, so got_attempts simply keeps growing and only
    // the entry pushed by Case 1 is asserted on below. Case 4 below builds
    // its own separate `stale_engine` that never sets on_attempt, which is
    // this test file's existing coverage for "no crash when the hook is
    // unset" (an unset std::function<void(uint32_t)> would throw
    // std::bad_function_call if process_job() called it unguarded).
    std::vector<uint32_t> got_attempts;
    engine.on_attempt = [&](uint32_t c) { got_attempts.push_back(c); };

    // -- Case 1: easy difficulty -> exactly one submit, correctly encoded --
    Job easy{"job-easy", valid_input, easy_difficulty, 2500000};
    engine.process_job(easy, "a1f");
    check(submits.size() == 1, "easy job produces exactly one submit");
    check(got_attempts.size() == 1, "on_attempt fired exactly once for Case 1's solve() return");
    check(!got_attempts.empty() && got_attempts.back() == 1u,
          "on_attempt reported FakeSolver's candidate count (1)");
    if (submits.size() == 1) {
        const Solution& s = submits[0];
        check(s.id == "job-easy", "submitted solution carries the job id");
        check(s.nonce.size() == 16, "nonce hex is 16 chars");
        check(is_lowercase_hex(s.nonce), "nonce hex is lowercase");
        check(s.nonce.substr(0, 3) == "a1f", "nonce hex starts with the client's nonceprefix");
        check(s.output.size() == 208, "output hex is 208 chars");
        check(is_lowercase_hex(s.output), "output hex is lowercase");
        check(s.output == to_hex(cand.data(), 104), "output hex matches the candidate bytes exactly");
    }

    // Pin the raw bytes solver.solve() actually received -- decode
    // correctness of job.input, and the "a1f" prefix -> nonce byte layout
    // (odd-length prefix's last nibble is the HIGH nibble of its byte),
    // independently of the Solution hex the pipeline happens to emit.
    {
        uint8_t expected_input[32];
        from_hex(valid_input, expected_input);
        check_eq_bytes(solver.last_input.data(), expected_input, 32, "solver received the decoded input bytes");

        static const uint8_t expected_nonce[8] = {0xa1, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
        check_eq_bytes(solver.last_nonce.data(), expected_nonce, 8,
                        "solver received nonce bytes: prefix a1f, counter 0, first attempt");
    }

    // -- Case 2: hard difficulty -> no submit --
    submits.clear();
    Job hard{"job-hard", valid_input, hard_difficulty, 2500001};
    engine.process_job(hard, "");
    check(submits.empty(), "hard job produces no submit");

    // -- Case 3: malformed input -> no submit, no crash --
    submits.clear();
    Job bad_chars{"job-bad-chars", std::string("zz") + std::string(62, '0'), easy_difficulty, 2500002};
    engine.process_job(bad_chars, "");
    check(submits.empty(), "invalid hex characters in input produce no submit");

    submits.clear();
    Job bad_len{"job-bad-len", "abcd1234", easy_difficulty, 2500003};   // 8 hex chars, not 64
    engine.process_job(bad_len, "");
    check(submits.empty(), "wrong-length input produces no submit");

    // -- Case 4: stale-share avoidance -- a newer, different-id job lands in
    // the mailbox while solve() is "in flight" (simulated from inside the
    // fake solver, see StaleInjectingSolver) -> the now-superseded job's
    // clearing candidate must NOT be submitted, even though it does clear.
    {
        StaleInjectingSolver stale_solver;
        Job newer{"job-newer", valid_input, easy_difficulty, 3000000};
        stale_solver.newer_job = newer;

        Engine stale_engine(client, stale_solver);
        stale_solver.engine = &stale_engine;
        std::vector<Solution> stale_submits;
        stale_engine.submit_fn = [&](const Solution& s, Origin) { stale_submits.push_back(s); };

        Job older{"job-older", valid_input, easy_difficulty, 2999999};
        stale_engine.process_job(older, "");
        check(stale_submits.empty(),
              "job superseded mid-solve by a different job id produces no submit");
    }

    // -- Case 5: staleness predicate, pinned directly via on_job() as a
    // mailbox preload -- on_job() only stores state (job, nonceprefix,
    // has_job_), so it's safe to call directly here with no thread and no
    // fake-solver trick, to check superseded_by_newer_job()'s has_job_-aware
    // logic from both sides. The mailbox is a single slot, so each check
    // below calls on_job() with exactly the entry it needs immediately
    // before its process_job() call -- neither depends on the other's
    // leftover mailbox state, so their order doesn't matter.

    // Same-id refresh: the mailbox's pending entry is for the SAME job id
    // as the one about to be processed -- not stale, so the submit must
    // still happen.
    submits.clear();
    engine.on_job(easy);
    engine.process_job(easy, "");
    check(submits.size() == 1, "same-id mailbox refresh still submits");

    // Newer different-id job: the mailbox's pending entry is for a
    // DIFFERENT job id than the one about to be processed -- stale, so the
    // submit must be suppressed.
    submits.clear();
    engine.on_job(hard);
    engine.process_job(easy, "");
    check(submits.empty(), "different-id mailbox entry suppresses the submit");

    // -- Case 6: Origin travels with the job, all the way to submit_fn.
    //
    // This is what stops a dev-fee round from misrouting shares. main.cpp
    // picks the connection to submit on from the Origin handed back here,
    // so if the tag were dropped -- or read from "which pool is active now"
    // instead of from the job -- every share solved across a round boundary
    // would go to the wrong pool and be rejected as an unknown job id. See
    // miner/origin.h.
    std::vector<Origin> origins;
    engine.submit_fn = [&](const Solution& s, Origin o) {
        submits.push_back(s);
        origins.push_back(o);
    };

    // Each case below re-arms the mailbox with the SAME job id it is about
    // to process, exactly as the same-id-refresh case above does: Case 5
    // left a different-id entry pending, and leaving it there would suppress
    // these submits as stale before submit_fn ever ran.

    // Direct call: the default is the user's pool, so single-pool callers
    // (the benchmark harness, the unit tests above) need not name it.
    submits.clear(); origins.clear();
    engine.on_job(easy);
    engine.process_job(easy, "");
    check(origins.size() == 1 && origins[0] == Origin::Main,
          "process_job defaults to the user's pool");

    // Explicitly tagged as fee work: the tag must reach submit_fn unchanged.
    submits.clear(); origins.clear();
    engine.on_job(easy);
    engine.process_job(easy, "", Origin::Dev);
    check(origins.size() == 1 && origins[0] == Origin::Dev,
          "an explicitly dev-tagged job submits as dev work");

    // ...and it must survive the mailbox, which is the path live mining
    // actually takes: on_job() stores the tag, worker_main() takes it back
    // out. Preload the mailbox with a Dev-tagged entry, then drain it the
    // same way the worker does.
    submits.clear(); origins.clear();
    engine.on_job(easy, "", Origin::Dev);
    engine.process_job(easy, "", Origin::Dev);
    check(origins.size() == 1 && origins[0] == Origin::Dev,
          "the dev tag survives a round trip through the mailbox");

    return summary("engine");
}
