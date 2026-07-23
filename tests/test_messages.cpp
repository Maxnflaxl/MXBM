#include "check.h"
#include "vectors/stratum_wire.h"
#include "stratum/messages.h"
using namespace mxbm; using namespace mxbm::stratum;
int main() {
    // encode
    check(serialize(Login{"aaaa1234"}) == wire::kLogin, "encode login");
    check(serialize(Cancel{"5417"}) == wire::kCancel, "encode cancel");
    check(serialize(Solution{"5417","0bb11009afc29dbe","a32a1e04"}) == wire::kSolution, "encode solution");
    // decode a job
    Msg m;
    check(parse(wire::kJob, m) && m.method == Method::Job, "parse job");
    check(m.job.id == "5417" && m.job.difficulty == 50331701u && m.job.height == 2500000ull, "job fields");
    check(m.job.input == "636b90cc38bc7a347f074d9ca97c3a2158330f6844f8f52075a38a15ab483223", "job input");
    // decode a login-ok result with nonceprefix
    check(parse(wire::kResultLoginOk, m) && m.method == Method::Result, "parse result");
    check(m.result.code == 0 && m.result.nonceprefix == "a1f", "result fields");
    // negative paths: parse must return false, never throw
    check(!parse("not json at all", m), "reject malformed json");
    check(!parse(R"({"id":"1","jsonrpc":"2.0"})", m), "reject missing method");
    check(!parse(R"({"difficulty":"50331701","id":"5417","jsonrpc":"2.0","method":"job"})", m),
          "reject wrong-typed difficulty (string) without throwing");
    return summary("messages");
}
