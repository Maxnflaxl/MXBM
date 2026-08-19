#pragma once
// Where MXBM keeps per-user files, and how to hand them back after running as root.
//
// Both halves exist because the measuring modes are normally run with `sudo`: every
// NVML write demands it. A file written to root's config dir is one the unprivileged
// daily launch never sees, and a root-owned file in the user's own tree is one they
// cannot overwrite next time.
#include <string>

namespace mxbm { namespace miner {

// $XDG_CONFIG_HOME, else ~/.config -- and under sudo, the INVOKING user's, not
// root's. On Windows %APPDATA%, which elevation does not change, so there is no
// indirection to undo there.
std::string config_dir();

// Give a root-created file or directory back to the user who ran sudo. A no-op when
// not running as root, and on Windows. Best effort: a failure here costs the next run
// its ability to rewrite the file, not this one its result.
void chown_to_invoking_user(const std::string& path);

}} // namespace mxbm::miner
