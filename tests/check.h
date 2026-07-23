#pragma once
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
namespace mxbm {
// Verbose mode: set env MXBM_VERBOSE=1 to print every generated/checked value.
// Default (unset) keeps the suite quiet-on-success so ctest output stays clean.
inline bool verbose() { static bool v = (std::getenv("MXBM_VERBOSE") != nullptr); return v; }
inline int& fail_count() { static int f = 0; return f; }
inline void check(bool cond, const char* msg) {
    if (!cond) { std::printf("  FAIL: %s\n", msg); ++fail_count(); }
    else if (verbose()) { std::printf("  ok   %-28s = %s\n", msg, "true"); }
}
inline void check_eq_u64(uint64_t got, uint64_t want, const char* msg) {
    if (got != want) {
        std::printf("  FAIL: %-28s got 0x%016llx want 0x%016llx\n",
            msg, (unsigned long long)got, (unsigned long long)want);
        ++fail_count();
    } else if (verbose()) {
        std::printf("  ok   %-28s = 0x%016llx\n", msg, (unsigned long long)got);
    }
}
inline void check_eq_bytes(const uint8_t* got, const uint8_t* want, size_t n, const char* msg) {
    if (std::memcmp(got, want, n) != 0) {
        std::printf("  FAIL: %s (byte mismatch)\n", msg);
        ++fail_count();
    } else if (verbose()) {
        std::printf("  ok   %-28s = ", msg);
        for (size_t i = 0; i < n; ++i) std::printf("%02x", got[i]);
        std::printf("\n");
    }
}
// Explicit value display (verbose only) — for things that aren't a pass/fail check.
inline void section(const char* title) { if (verbose()) std::printf("-- %s\n", title); }
inline void show_u64(const char* label, uint64_t v) {
    if (verbose()) std::printf("     %-25s = 0x%016llx\n", label, (unsigned long long)v);
}
// show_hex is deliberately UNCONDITIONAL (unlike section/show_u64): its callers
// use it in FAILURE branches (test_gpu_solver, test_gpu_recover) to dump the
// offending bytes, so it must print even in default quiet mode. Purely
// informational hex dumps gate themselves at the call site with `if (verbose())`
// (see test_verify), so they stay quiet by default without gating here.
inline void show_hex(const char* label, const uint8_t* b, size_t n) {
    std::printf("     %-25s = ", label);
    for (size_t i = 0; i < n; ++i) std::printf("%02x", b[i]);
    std::printf("\n");
}
inline int summary(const char* name) {
    if (fail_count() == 0) { std::printf("PASS: %s\n", name); return 0; }
    std::printf("FAILED: %s (%d checks)\n", name, fail_count());
    return 1;
}
} // namespace mxbm
