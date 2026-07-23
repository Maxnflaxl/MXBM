#pragma once
#include <cstdio>
#include <cstdint>
#include <cstring>
namespace mxbm {
inline int& fail_count() { static int f = 0; return f; }
inline void check(bool cond, const char* msg) {
    if (!cond) { std::printf("  FAIL: %s\n", msg); ++fail_count(); }
}
inline void check_eq_u64(uint64_t got, uint64_t want, const char* msg) {
    if (got != want)
        std::printf("  FAIL: %s: got 0x%016llx want 0x%016llx\n",
            msg, (unsigned long long)got, (unsigned long long)want), ++fail_count();
}
inline void check_eq_bytes(const uint8_t* got, const uint8_t* want, size_t n, const char* msg) {
    if (std::memcmp(got, want, n) != 0) { std::printf("  FAIL: %s (byte mismatch)\n", msg); ++fail_count(); }
}
inline int summary(const char* name) {
    if (fail_count() == 0) { std::printf("PASS: %s\n", name); return 0; }
    std::printf("FAILED: %s (%d checks)\n", name, fail_count());
    return 1;
}
} // namespace mxbm
