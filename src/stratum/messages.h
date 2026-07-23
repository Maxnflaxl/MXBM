#pragma once
#include <string>
#include <cstdint>
namespace mxbm { namespace stratum {
enum class Method { Unknown, Login, Job, Solution, Cancel, Result };
struct Login    { std::string api_key; };                                  // credential (wallet addr for pools)
struct Job      { std::string id, input; uint32_t difficulty; uint64_t height; };
struct Solution { std::string id, nonce, output; };                        // nonce=16 hex, output=208 hex
struct Cancel   { std::string id; };
struct Result   { std::string id, description, nonceprefix, blockhash; int code; };
struct Msg { Method method = Method::Unknown; Job job; Result result; Cancel cancel; };

std::string serialize(const Login&);
std::string serialize(const Solution&);
std::string serialize(const Cancel&);
bool parse(const std::string& line, Msg& out);   // decodes an inbound line
} } // namespace mxbm::stratum
