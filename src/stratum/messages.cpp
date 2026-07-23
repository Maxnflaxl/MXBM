// Stratum message (de)serialization. Ported from Beam pow/stratum.cpp
// (Apache-2.0, The Beam Team / BeamMW) — logic adapted; wire format
// verified against Beam's stratum wire format. See NOTICE.
#include "stratum/messages.h"
#include "nlohmann/json.hpp"
using nlohmann::json;
namespace mxbm { namespace stratum {
static void base(json& o, const char* method, const std::string& id) {
    o["jsonrpc"] = "2.0"; o["id"] = id; o["method"] = method;
}
std::string serialize(const Login& m)    { json o; base(o,"login","login"); o["api_key"]=m.api_key; return o.dump(); }
std::string serialize(const Cancel& m)   { json o; base(o,"cancel",m.id); return o.dump(); }
std::string serialize(const Solution& m) { json o; base(o,"solution",m.id); o["nonce"]=m.nonce; o["output"]=m.output; return o.dump(); }
bool parse(const std::string& line, Msg& out) {
    try {
        json j = json::parse(line);
        if (!j.contains("method")) return false;
        const std::string meth = j.value("method","");
        auto sid = j.value("id", std::string());
        if (meth == "job") {
            out.method = Method::Job;
            out.job = { sid, j.value("input",std::string()), j.value("difficulty",0u), j.value("height",0ull) };
        } else if (meth == "result") {
            out.method = Method::Result;
            out.result = { sid, j.value("description",std::string()), j.value("nonceprefix",std::string()),
                           j.value("blockhash",std::string()), j.value("code",0) };
        } else if (meth == "cancel") {
            out.method = Method::Cancel; out.cancel = { sid };
        } else if (meth == "login" || meth == "solution") {
            out.method = (meth=="login")?Method::Login:Method::Solution; // inbound rarely; accept
        } else { out.method = Method::Unknown; }
        return true;
    } catch (...) { return false; }   // malformed JSON or wrong-typed field -> false, never throw
}
} } // namespace mxbm::stratum
