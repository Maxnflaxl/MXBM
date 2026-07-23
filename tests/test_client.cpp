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
    return summary("client");
}
