#include "check.h"
#include "version.h"
#include <cstring>
#include <cctype>
using namespace mxbm;
// Shape only — never exact values (count/hash change every commit):
// 'v' digits '.' digits '.' digits " [" 7 lowercase-hex or the literal "nogit" "]"
static bool shape_ok(const char* v) {
    int part = 0, digits = 0;
    const char* p = v;
    if (*p != 'v') return false;
    ++p;
    for (; *p && *p != ' '; ++p) {
        if (*p == '.') { if (!digits) return false; ++part; digits = 0; }
        else if (isdigit((unsigned char)*p)) ++digits;
        else return false;
    }
    if (part != 2 || !digits) return false;
    if (strncmp(p, " [", 2) != 0) return false;
    p += 2;
    if (strncmp(p, "nogit]", 6) == 0) return *(p + 6) == 0;
    for (int i = 0; i < 7; ++i, ++p)
        if (!isxdigit((unsigned char)*p) || isupper((unsigned char)*p)) return false;
    return p[0] == ']' && p[1] == 0;
}
int main() {
    check(version() != nullptr && *version(), "version non-empty");
    check(shape_ok(version()), "version shape vN.N.N [hex7|nogit]");
    // Pins that the runtime string really comes from project(VERSION ...) in
    // CMakeLists.txt rather than a stale hardcoded stamp. The expected prefix
    // is fed in by CMake (MXBM_EXPECT_PREFIX), so a version bump cannot fail
    // this check -- only a generated header that disagrees with a fresh
    // configure can, which is the actual defect it exists to catch.
    check(strncmp(version(), MXBM_EXPECT_PREFIX, strlen(MXBM_EXPECT_PREFIX)) == 0,
          "major.minor from project()");
    return summary("version");
}
