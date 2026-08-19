#include "miner/user_paths.h"

#include <cstdlib>

#ifndef _WIN32
#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace mxbm { namespace miner {

std::string config_dir() {
#ifdef _WIN32
    if (const char* a = std::getenv("APPDATA"); a && *a) return a;
    return ".";
#else
    if (const char* x = std::getenv("XDG_CONFIG_HOME"); x && *x) return x;
    if (geteuid() == 0) {
        if (const char* su = std::getenv("SUDO_USER"); su && *su)
            if (const passwd* pw = getpwnam(su)) return std::string(pw->pw_dir) + "/.config";
    }
    if (const char* h = std::getenv("HOME"); h && *h) return std::string(h) + "/.config";
    return ".";
#endif
}

void chown_to_invoking_user(const std::string& path) {
#ifdef _WIN32
    (void)path;
#else
    if (geteuid() != 0) return;
    const char* u = std::getenv("SUDO_UID");
    const char* g = std::getenv("SUDO_GID");
    if (!u || !g) return;
    (void)chown(path.c_str(), (uid_t)atoi(u), (gid_t)atoi(g));
#endif
}

}} // namespace mxbm::miner
