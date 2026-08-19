#include "miner/report.h"

#include <cctype>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <system_error>

#include "miner/user_paths.h"

namespace mxbm { namespace miner {
namespace {

// A card name as a filename: lowercase, every run of non-alphanumerics collapsed to
// one hyphen, ends trimmed. "NVIDIA GeForce RTX 4070 Ti SUPER" ->
// "nvidia-geforce-rtx-4070-ti-super".
std::string slug(const std::string& name) {
    std::string out;
    for (char c : name) {
        const unsigned char u = (unsigned char)c;
        if (std::isalnum(u)) out += (char)std::tolower(u);
        else if (!out.empty() && out.back() != '-') out += '-';
    }
    while (!out.empty() && out.back() == '-') out.pop_back();
    return out.empty() ? "gpu" : out;
}

std::string today() {
    char buf[16];
    const std::time_t now = std::time(nullptr);
    std::strftime(buf, sizeof buf, "%Y-%m-%d", std::localtime(&now));
    return buf;
}

} // namespace

std::string report_default_path(const std::string& gpu_name, bool whole_rig) {
    return config_dir() + "/mxbm/report-"
         + (whole_rig ? std::string("rig") : slug(gpu_name)) + "-" + today() + ".md";
}

std::string report_resolve_path(const std::string& requested, const std::string& gpu_name,
                                bool whole_rig, std::string& err) {
    err.clear();
    namespace fs = std::filesystem;
    std::string want = requested.empty() ? report_default_path(gpu_name, whole_rig) : requested;
    // A quoted ~/... never reached the shell, so expand it here rather than creating a
    // directory literally named "~".
    if (want.size() >= 2 && want[0] == '~' && (want[1] == '/' || want[1] == '\\')) {
        const char* home = std::getenv("HOME");
#ifdef _WIN32
        if (!home || !*home) home = std::getenv("USERPROFILE");
#endif
        if (home && *home) want = std::string(home) + want.substr(1);
    }
    std::error_code ec;
    fs::path p = fs::absolute(fs::path(want), ec);   // relative resolves against the CWD
    if (ec) { err = "cannot resolve '" + want + "': " + ec.message(); return {}; }
    // A directory means "put it in here", not "overwrite this with a file".
    if (fs::is_directory(p, ec))
        p /= fs::path(report_default_path(gpu_name, whole_rig)).filename();
    if (!p.parent_path().empty()) {
        fs::create_directories(p.parent_path(), ec);
        if (ec && !fs::is_directory(p.parent_path())) {
            err = "cannot create " + p.parent_path().string() + ": " + ec.message();
            return {};
        }
        chown_to_invoking_user(p.parent_path().string());
    }
    return p.lexically_normal().string();
}

bool report_write(const std::string& path, const std::string& text, std::string& err) {
    err.clear();
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) { err = "cannot write " + path; return false; }
    f << text;
    f.close();
    if (!f) { err = "failed while writing " + path; return false; }
    chown_to_invoking_user(path);
    return true;
}


}} // namespace mxbm::miner
