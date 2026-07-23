#include "check.h"
#include "vectors/stratum_wire.h"
#include "nlohmann/json.hpp"
using nlohmann::json;
int main() {
    using namespace mxbm;
    json j = { {"method","login"}, {"jsonrpc","2.0"}, {"id","login"}, {"api_key","aaaa1234"} };
    std::string s = j.dump();               // nlohmann dumps object keys sorted
    check(s == wire::kLogin, "nlohmann compact+sorted == canonical login");
    json p = json::parse(wire::kJob);       // parse a job, read fields
    check(p.at("method") == "job", "parse job method");
    check(p.at("difficulty").get<uint32_t>() == 50331701u, "parse difficulty as number");
    check(p.at("id").get<std::string>() == "5417", "parse id as string");
    return summary("wire_spike");
}
