// Developer-fee unit tests: the accrual arithmetic and the pool-switching
// router, both exercised directly with no threads, no sockets and no waiting.
//
// DevFee itself (the class that owns the fee's stratum connection and its
// scheduler thread) is deliberately NOT constructed here -- start()ing one
// opens a real connection to the fee pool and spawns a thread that cannot be
// joined (see the lifetime note in miner/devfee.h). Everything worth testing
// about the fee is in the two pieces below, which is exactly why they were
// split out of it: FeeAccrual decides WHEN a round is owed, and JobRouter
// decides WHAT the solver works on across the switch. A bug in the fee is a
// bug in one of those two.
#include <chrono>
#include <string>
#include <vector>

#include "check.h"
#include "miner/devfee.h"
#include "miner/origin.h"
#include "stratum/messages.h"

using namespace mxbm;
using namespace mxbm::miner;

namespace {

using secs = std::chrono::seconds;

DevFeeSchedule one_percent_hourly() {
    DevFeeSchedule s;
    s.rate = 0.01;
    s.cycle = secs{3600};
    return s;
}

stratum::Job make_job(const std::string& id) {
    stratum::Job j;
    j.id = id;
    j.input = std::string(64, 'a');
    j.difficulty = 512u;
    j.height = 3000000;
    return j;
}

// Records everything JobRouter dispatches, standing in for Engine::on_job.
struct Dispatched {
    std::string job_id;
    std::string nonceprefix;
    Origin origin;
};

void test_schedule() {
    section("schedule");
    const DevFeeSchedule s = one_percent_hourly();
    // 1% of an hour is 36 s. This is the number quoted to the user on the
    // console and in docs/devfee.md, so pin it rather than recompute it.
    check(s.slice() == secs{36}, "1% of a 3600s cycle is a 36s round");

    DevFeeSchedule half = s;
    half.rate = 0.005;
    check(half.slice() == secs{18}, "0.5% of a 3600s cycle is an 18s round");

    DevFeeSchedule none = s;
    none.rate = 0.0;
    check(none.slice() == secs{0}, "a zero rate yields a zero-length round");
}

void test_accrual_reaches_a_round_at_the_stated_rate() {
    section("accrual: cadence");
    FeeAccrual a(one_percent_hourly());

    check(!a.slice_due(), "nothing is owed before any mining time");

    // One second short of the cycle must NOT trigger: at 1% the debt is
    // 35.99 s against a 36 s round. Off-by-one here would charge the user
    // slightly more often than advertised, every hour, forever.
    a.add_mining_time(secs{3599});
    check(!a.slice_due(), "no round is due one second short of the full cycle");

    a.add_mining_time(secs{1});
    check(a.slice_due(), "a round comes due after exactly one cycle of mining");

    a.settle(a.slice_len());
    check(!a.slice_due(), "settling a full round clears the debt");
}

void test_accrual_only_counts_time_it_is_given() {
    section("accrual: idle time is free");
    FeeAccrual a(one_percent_hourly());
    // The scheduler only feeds ticks while the user's pool has a job, so a
    // session that spends an hour disconnected simply never calls this. The
    // guarantee under test is the arithmetic one: debt is a function of the
    // time handed in, nothing else -- no wall-clock, no session length.
    a.add_mining_time(secs{0});
    a.add_mining_time(secs{-5});   // defensive: a non-monotonic tick must not credit debt
    check(a.debt_seconds() == 0.0, "zero and negative ticks accrue nothing");

    a.add_mining_time(secs{1800});
    check(!a.slice_due(), "half a cycle of mining does not yet owe a round");
    check(a.debt_seconds() > 17.9 && a.debt_seconds() < 18.1,
          "half a cycle at 1% accrues half a round of debt");
}

void test_accrual_carries_and_floors_debt() {
    section("accrual: carry and floor");
    FeeAccrual a(one_percent_hourly());
    a.add_mining_time(secs{7200});   // two cycles: 72 s owed, two rounds' worth

    check(a.slice_due(), "two cycles owe a round");
    a.settle(secs{36});
    check(a.slice_due(), "the second round is still owed after settling the first");
    a.settle(secs{36});
    check(!a.slice_due(), "both owed rounds settle to nothing");

    // A round cut short (disconnect, Ctrl+C) settles only what it spent, so
    // the remainder stays owed rather than being silently forgiven.
    FeeAccrual b(one_percent_hourly());
    b.add_mining_time(secs{3600});
    b.settle(secs{10});
    check(b.debt_seconds() > 25.9 && b.debt_seconds() < 26.1,
          "a round cut short leaves the unspent remainder owed");

    // ...and an overrun settles the excess instead of banking negative debt,
    // which would otherwise delay the NEXT round by the overrun.
    FeeAccrual c(one_percent_hourly());
    c.add_mining_time(secs{3600});
    c.settle(secs{500});
    check(c.debt_seconds() == 0.0, "an overrunning round floors the debt at zero");
}

void test_router_dispatches_only_the_active_origin() {
    section("router: dispatch gating");
    std::vector<Dispatched> got;
    JobRouter router([&](const stratum::Job& j, const std::string& p, Origin o) {
        got.push_back(Dispatched{j.id, p, o});
    });

    check(router.active() == Origin::Main, "the user's pool is active by default");
    check(!router.has(Origin::Main) && !router.has(Origin::Dev),
          "neither origin has a job before any is offered");

    router.offer(make_job("main-1"), "a1f", Origin::Main);
    check(got.size() == 1, "a job from the active origin is dispatched");
    check(got[0].job_id == "main-1" && got[0].origin == Origin::Main,
          "it is dispatched with its own id and origin");
    check(got[0].nonceprefix == "a1f", "the offered nonceprefix travels with the job");

    // The fee connection feeds the router the whole time, but must not
    // disturb the solver while the user's pool is the active one.
    router.offer(make_job("dev-1"), "beef", Origin::Dev);
    check(got.size() == 1, "a job from the inactive origin is cached, not dispatched");
    check(router.has(Origin::Dev), "...but it is remembered as present");
}

void test_router_switches_in_both_directions() {
    section("router: switching");
    std::vector<Dispatched> got;
    JobRouter router([&](const stratum::Job& j, const std::string& p, Origin o) {
        got.push_back(Dispatched{j.id, p, o});
    });

    router.offer(make_job("main-1"), "a1f", Origin::Main);
    router.offer(make_job("dev-1"), "beef", Origin::Dev);
    got.clear();

    // Round starts: the cached fee job is dispatched immediately rather than
    // idling until the fee pool happens to send a fresh one.
    router.set_active(Origin::Dev);
    check(got.size() == 1 && got[0].job_id == "dev-1" && got[0].origin == Origin::Dev,
          "switching to the fee pool dispatches its cached job at once");
    check(got[0].nonceprefix == "beef", "the fee pool's own nonceprefix is used, not the user's");

    check(router.active() == Origin::Dev, "the fee pool is now active");
    router.set_active(Origin::Dev);
    check(got.size() == 1, "switching to the already-active origin is a no-op");

    // A new job from the user's pool DURING the round is cached, not lost --
    // this is what makes the switch back lossless.
    router.offer(make_job("main-2"), "a1f", Origin::Main);
    check(got.size() == 1, "the user's new job is held while the fee round runs");

    // Round ends: the user resumes on the NEWEST job of theirs, not the one
    // that was current when the round began. Resuming "main-1" would mine a
    // job the pool has already moved past, and every share would be stale.
    router.set_active(Origin::Main);
    check(got.size() == 2, "switching back dispatches exactly once");
    check(got[1].job_id == "main-2" && got[1].origin == Origin::Main,
          "the user resumes on their newest job, not the one the round interrupted");
}

void test_router_defers_a_switch_it_cannot_serve() {
    section("router: nothing cached");
    std::vector<Dispatched> got;
    JobRouter router([&](const stratum::Job& j, const std::string& p, Origin o) {
        got.push_back(Dispatched{j.id, p, o});
    });

    // Switching to an origin that has never sent a job dispatches nothing
    // rather than a default-constructed one -- Engine would reject the empty
    // input anyway, but it would also spin the worker until the next job.
    router.set_active(Origin::Dev);
    check(got.empty(), "switching to an origin with no job dispatches nothing");
    check(router.active() == Origin::Dev, "the switch itself still takes effect");

    // ...and the job that arrives next for it is dispatched, since it is now active.
    router.offer(make_job("dev-1"), "beef", Origin::Dev);
    check(got.size() == 1 && got[0].job_id == "dev-1",
          "the fee pool's first job is dispatched once it is the active origin");
}

void test_shipped_configuration_is_coherent() {
    section("shipped configuration");
    const DevFeeSchedule s = devfee_schedule();
    const DevFeePool p = devfee_pool();

    // Whatever the constants say, they must agree with each other: a rate
    // that rounds to a zero-length round, or a destination with no address,
    // must read as "not configured" rather than silently charging nothing
    // while claiming a rate on the console.
    if (devfee_configured()) {
        check(s.rate > 0.0, "a configured fee has a positive rate");
        check(s.slice() > secs{0}, "a configured fee has a round longer than zero");
        check(!p.user.empty(), "a configured fee has a destination address");
        check(!p.host.empty() && p.port != 0, "a configured fee has a pool endpoint");
        check(s.rate <= 0.05,
              "the fee is within a sane range -- a typo adding a digit is caught here");
    } else {
        check(p.user.empty() || s.rate <= 0.0 || s.slice() == secs{0},
              "an unconfigured fee is unconfigured for a stated reason");
    }
}

void test_login_names_the_round_after_the_users_worker() {
    section("login credential");
    const std::string dev = "devaddr";

    // The fee round should be identifiable on the fee pool as this user's
    // rig, at the rate they are paying -- not as anonymous hashrate.
    check(devfee_login(dev, "useraddr.rig1", 0.01) == "devaddr.rig1_1",
          "the round logs in as <dev address>.<user's worker>_<percent>");
    check(devfee_login(dev, "useraddr.rig1", 0.025) == "devaddr.rig1_2.5",
          "a raised, fractional rate shows in the worker name");
    check(devfee_login(dev, "useraddr.rig1", 0.10) == "devaddr.rig1_10",
          "a whole percent carries no misleading trailing zeros");

    // No worker suffix, or one that sanitises away, still needs a name.
    check(devfee_login(dev, "useraddr", 0.01) == "devaddr.mxbm_1",
          "a user with no worker suffix gets a default name");
    check(devfee_login(dev, "useraddr.", 0.01) == "devaddr.mxbm_1",
          "an empty worker suffix gets the default name");
    check(devfee_login(dev, "useraddr.!!!", 0.01) == "devaddr.mxbm_1",
          "a worker name that sanitises to nothing gets the default name");

    // The credential goes onto the wire inside a JSON login line, so
    // anything that could break out of it must not survive. This is the
    // check that matters: a quote or newline reaching the socket would
    // corrupt the login, not merely look untidy.
    const std::string hostile = devfee_login(dev, "useraddr.a\"b\nc.d rig", 0.01);
    check(hostile.find('"') == std::string::npos, "quotes are stripped from the worker name");
    check(hostile.find('\n') == std::string::npos, "newlines are stripped from the worker name");
    check(hostile == "devaddr.abcdrig_1", "only [A-Za-z0-9_-] survives sanitisation");

    // Pools cap worker-name length; an over-long one must be truncated here
    // rather than rejected by the pool at login time.
    const std::string longw = devfee_login(dev, "useraddr." + std::string(200, 'w'), 0.01);
    check(longw == "devaddr." + std::string(32, 'w') + "_1",
          "an over-long worker name is capped at 32 characters");

    // Only the FIRST dot separates address from worker -- a fractional
    // percent's own dot lives inside the worker field and must not be
    // mistaken for that separator when the pool splits the credential.
    const std::string frac = devfee_login(dev, "useraddr.rig1", 0.025);
    check(frac.find('.') == dev.size(),
          "the address/worker separator is the first dot in the credential");
}

} // namespace

int main() {
    test_schedule();
    test_accrual_reaches_a_round_at_the_stated_rate();
    test_accrual_only_counts_time_it_is_given();
    test_accrual_carries_and_floors_debt();
    test_router_dispatches_only_the_active_origin();
    test_router_switches_in_both_directions();
    test_router_defers_a_switch_it_cannot_serve();
    test_shipped_configuration_is_coherent();
    test_login_names_the_round_after_the_users_worker();
    return summary("devfee");
}
