#include "check.h"
#include "vectors/stratum_wire.h"
#include "stratum/client.h"
using namespace mxbm; using namespace mxbm::stratum;
int main() {
    Client c;
    Job seen; bool got=false;
    c.on_job = [&](const Job& j){ seen=j; got=true; };
    c.handle_line(wire::kResultLoginOk);
    check(c.current_nonceprefix() == "a1f", "nonceprefix captured from result");
    c.handle_line(wire::kJob);
    check(got && seen.id=="5417" && seen.difficulty==50331701u, "job dispatched");
    // result without nonceprefix must not clobber the captured one
    c.handle_line(wire::kResultAccepted);
    check(c.current_nonceprefix() == "a1f", "nonceprefix preserved on prefix-less result");
    // cancel is a no-op: no job callback, no crash
    got = false;
    c.handle_line(wire::kCancel);
    check(!got, "cancel dispatches no job");

    section("pool failover rotates, wraps, and carries each pool's own credentials");
    {
        Client f;
        f.connect("primary.pool", 1130, false);   // no server: sets host_/port_ only
        f.login("addr-primary.rig1");
        std::vector<std::pair<std::string, std::string>> moves;
        f.on_failover = [&moves](const std::string& a, const std::string& b) {
            moves.push_back({a, b});
        };

        // One pool: rotation must do nothing at all, or a single-pool rig would
        // "fail over" to itself and log a move that did not happen.
        f.set_failover_pools({{"primary.pool", 1130, false, "addr-primary.rig1"}});
        f.advance_pool();
        check(f.current_pool() == "primary.pool:1130", "a single pool never rotates");
        check(moves.empty(), "and fires no failover callback");

        f.set_failover_pools({
            {"primary.pool", 1130, false, "addr-primary.rig1"},
            {"backup.pool",  3334, true,  "addr-backup.rig1"},
        });
        f.advance_pool();
        check(f.current_pool() == "backup.pool:3334", "rotates to the second pool");
        check(moves.size() == 1 && moves[0].first == "primary.pool:1130"
              && moves[0].second == "backup.pool:3334",
              "and reports the move, so a rig that changed pools says so");
        f.advance_pool();
        check(f.current_pool() == "primary.pool:1130", "wraps back round to the first");
        check(moves.size() == 2, "each move is reported");
    }

    return summary("client");
}
