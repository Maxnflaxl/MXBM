#pragma once
// Canonical compact serializations (keys alphabetical, no spaces).
// Verified against Beam's stratum wire format (pow/stratum.cpp).
namespace mxbm { namespace wire {
inline const char* kLogin =
  R"({"api_key":"aaaa1234","id":"login","jsonrpc":"2.0","method":"login"})";
inline const char* kJob =
  R"({"difficulty":50331701,"height":2500000,"id":"5417","input":"636b90cc38bc7a347f074d9ca97c3a2158330f6844f8f52075a38a15ab483223","jsonrpc":"2.0","method":"job"})";
inline const char* kSolution =
  R"({"id":"5417","jsonrpc":"2.0","method":"solution","nonce":"0bb11009afc29dbe","output":"a32a1e04"})"; // output truncated in this fixture; real is 208 hex
inline const char* kResultLoginOk =
  R"({"code":0,"description":"Success","id":"login","jsonrpc":"2.0","method":"result","nonceprefix":"a1f"})";
inline const char* kResultAccepted =
  R"({"code":1,"description":"accepted","id":"5417","jsonrpc":"2.0","method":"result"})";
inline const char* kCancel =
  R"({"id":"5417","jsonrpc":"2.0","method":"cancel"})";
} } // namespace mxbm::wire
