#include "check.h"
#include "version.h"
#include <cstring>
#include <cctype>
using namespace mxbm;
// Shape only — never exact values (count/hash change every commit):
// digits '.' digits '.' digits " [" 7 lowercase-hex or the literal "nogit" "]"
static bool shape_ok(const char* v) {
    int part = 0, digits = 0;
    const char* p = v;
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
    check(shape_ok(version()), "version shape N.N.N [hex7|nogit]");
    // Pins that the runtime string really comes from project(VERSION ...) in
    // CMakeLists.txt rather than a stale hardcoded stamp -- so this literal
    // must be bumped in step with it.
    check(strncmp(version(), "0.6.", 4) == 0, "major.minor from project()");
    return summary("version");
}
