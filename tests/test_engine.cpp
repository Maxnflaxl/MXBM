// Engine pipeline test: job -> solve -> difficulty filter -> submit. A
// FakeSolver returns a canned candidate and Engine::submit_fn captures the
// submissions, so there are no threads and no transport; see miner/engine.h
// for why process_job() alone is enough to test this deterministically.
#include <algorithm>
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

// Records the last (input, nonce) it was called with, so the test can pin
// process_job()'s decode and nonce-construction byte layout, not just the hex
// format of the Solution it eventually submits.
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

// Independent hex decoder, deliberately not shared with engine.cpp, so the
// byte checks below pin process_job() against a second implementation.
void from_hex(const std::string& hex, uint8_t* out) {
    auto nibble = [](char c) { return (c <= '9') ? c - '0' : c - 'a' + 10; };
    for (size_t i = 0; i < hex.size() / 2; ++i)
        out[i] = (uint8_t)((nibble(hex[i * 2]) << 4) | nibble(hex[i * 2 + 1]));
}

// Simulates "a newer job lands in the mailbox while solve() is in flight"
// without a real thread, by calling on_job() from inside solve() itself.
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
    // The same 32 input bytes as tests/vectors/stratum_wire.h's kJob fixture.
    const std::string valid_input =
        "636b90cc38bc7a347f074d9ca97c3a2158330f6844f8f52075a38a15ab483223";
    check(valid_input.size() == 64, "fixture input is 64 hex chars");

    // packed=0 clears for ANY candidate (hash*2^24 < 2^280 always holds), so no
    // hand-computed SHA-256 is needed; (200<<24)|0xFFFFFF is about as hard as
    // the format allows. Both are asserted below rather than assumed.
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

    // on_attempt must fire once per solve() return, BEFORE difficulty filtering.
    // Case 4's separate `stale_engine` never installs the hook, which is this
    // file's coverage for the unset case (an unset std::function would throw).
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

    // The raw bytes solve() received. The expected nonce encodes the layout
    // rule: an odd-length prefix's last nibble is the HIGH nibble of its byte.
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

    // -- Case 4: stale-share avoidance -- a newer job lands mid-solve, so the
    // superseded job's candidate must not be submitted even though it clears.
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

    // -- Case 5: the staleness predicate from both sides, driven by preloading
    // the mailbox. on_job() only stores state, so it is safe to call directly.
    // The single slot means each check re-arms it before its own process_job().

    submits.clear();
    engine.on_job(easy);
    engine.process_job(easy, "");
    check(submits.size() == 1, "same-id mailbox refresh still submits");

    submits.clear();
    engine.on_job(hard);
    engine.process_job(easy, "");
    check(submits.empty(), "different-id mailbox entry suppresses the submit");

    // -- Case 6: Origin travels with the job, all the way to submit_fn.
    // main.cpp picks the connection to submit on from the Origin handed back
    // here, so a tag that was dropped -- or read from "which pool is active now"
    // -- would misroute every share solved across a fee-round boundary.
    std::vector<Origin> origins;
    engine.submit_fn = [&](const Solution& s, Origin o) {
        submits.push_back(s);
        origins.push_back(o);
    };

    // Case 5 left a different-id entry pending, so each case below re-arms the
    // mailbox with its own job id -- otherwise its submit is suppressed as stale
    // before submit_fn ever runs.

    submits.clear(); origins.clear();
    engine.on_job(easy);
    engine.process_job(easy, "");
    check(origins.size() == 1 && origins[0] == Origin::Main,
          "process_job defaults to the user's pool");

    submits.clear(); origins.clear();
    engine.on_job(easy);
    engine.process_job(easy, "", Origin::Dev);
    check(origins.size() == 1 && origins[0] == Origin::Dev,
          "an explicitly dev-tagged job submits as dev work");

    // The mailbox round trip is the path live mining takes: on_job() stores the
    // tag, worker_main() takes it back out.
    submits.clear(); origins.clear();
    engine.on_job(easy, "", Origin::Dev);
    engine.process_job(easy, "", Origin::Dev);
    check(origins.size() == 1 && origins[0] == Origin::Dev,
          "the dev tag survives a round trip through the mailbox");

    section("nonce lanes: several devices on one job never try the same nonce");
    {
        // The multi-device failure this guards against is silent: two cards
        // walking the same nonce sequence would each look perfectly healthy
        // while the rig did half the work it was paid for. So the check is on
        // the ACTUAL nonces handed to the solvers, not on the counters.
        //
        // Every solver here records what it was asked to solve; three Engines
        // share one prefix and split the space three ways.
        struct RecordingSolver : Solver {
            std::vector<std::array<uint8_t, 8>> seen;
            std::vector<std::array<uint8_t, 104>> solve(const uint8_t[32],
                                                        const uint8_t nonce[8]) override {
                std::array<uint8_t, 8> n{};
                std::memcpy(n.data(), nonce, 8);
                seen.push_back(n);
                return {};
            }
        };

        const unsigned kLanes = 3, kAttempts = 8;
        Client client;
        RecordingSolver solvers[kLanes];
        std::vector<std::array<uint8_t, 8>> all;
        for (unsigned lane = 0; lane < kLanes; ++lane) {
            Engine e(client, solvers[lane], lane, kLanes);
            e.submit_fn = [](const stratum::Solution&, Origin) {};
            stratum::Job job{"j1", std::string(64, 'a'), 0xFFFFFFFFu, 100};
            for (unsigned i = 0; i < kAttempts; ++i) e.process_job(job, "abc");
            check(solvers[lane].seen.size() == kAttempts, "every attempt reached the solver");
            for (const auto& n : solvers[lane].seen) all.push_back(n);
        }
        std::sort(all.begin(), all.end());
        check(all.size() == kLanes * kAttempts, "all attempts collected");
        check(std::adjacent_find(all.begin(), all.end()) == all.end(),
              "no two lanes ever produced the same nonce");

        // And the prefix survives partitioning: the pool assigned it, so a lane
        // that dropped it would submit shares the pool credits to nobody.
        for (const auto& n : all)
            check((n[0] >> 4) == 0xa && (n[0] & 0xF) == 0xb && (n[1] >> 4) == 0xc,
                  "the pool's nonce prefix is still in every lane's nonce");
    }
    {
        // lanes = 1 must be exactly what it was before lanes existed:
        // consecutive counters from zero.
        struct RecordingSolver : Solver {
            std::vector<uint8_t> first_counter_byte;
            std::vector<std::array<uint8_t, 104>> solve(const uint8_t[32],
                                                        const uint8_t nonce[8]) override {
                first_counter_byte.push_back(nonce[2]);   // prefix "abc" fills 1.5 bytes
                return {};
            }
        };
        Client client;
        RecordingSolver s;
        Engine e(client, s);
        e.submit_fn = [](const stratum::Solution&, Origin) {};
        stratum::Job job{"j1", std::string(64, 'a'), 0xFFFFFFFFu, 100};
        for (int i = 0; i < 4; ++i) e.process_job(job, "abc");
        check(s.first_counter_byte == std::vector<uint8_t>({0, 1, 2, 3}),
              "the single-device default still walks 0,1,2,3 -- unchanged by the lane parameter");
    }

    return summary("engine");
}
