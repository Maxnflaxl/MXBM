#pragma once
#include <string>

#include "cli/options.h"

// the reference miner-shaped config-file loaders. Two sibling formats: `--json` selects a
// named profile out of a user_config.json-style { "PROFILE": { ...UPPERCASE
// keys... } } map; `--config` reads a flat "KEY = VALUE" file (the reference miner's
// plain-text grammar is undocumented upstream, so this approximates it).
//
// Both loaders only ever fill a cli::Options field whose cli::Options::Seen
// flag is still false, so CLI-supplied values always win and a loader call is
// additive, never destructive. Unknown keys are ignored (forward compat).
//
// main.cpp picks one loader per run: opts.use_json_config selects the JSON one,
// else a non-empty opts.config_path selects the flat one. Nothing clears
// use_json_config, so --json wins even if a later --config overwrites the path.
namespace mxbm { namespace config {

// Loads `path` as JSON and selects profile `profile` (empty = the first profile
// in file order). A profile's "POOLS" array (entries of {POOL:"host:port",
// USER, PASS?, TLS?}) is applied only when !opts.seen.pools, and then replaces
// opts.pools wholesale. "ALGO" must be "BEAM-III" whenever present, regardless
// of any Seen flag -- Options has no field to store it in, so it is a pure
// guard against a profile silently targeting a different algorithm. Returns
// false with a human-readable `err` on any read, syntax, missing-profile,
// wrong-type or out-of-range failure.
bool load_json_config(const std::string& path, const std::string& profile,
                       cli::Options& opts, std::string& err);

// Loads `path` as a flat "KEY = VALUE" file, one setting per line ('#' and
// blank lines skipped, keys matched case-insensitively). Unlike the JSON POOLS
// array, a flat file has at most one pool: its POOL/USER/PASS/TLS scalars form
// a single entry, applied only when !opts.seen.pools and a POOL key is present.
bool load_flat_config(const std::string& path, cli::Options& opts, std::string& err);

} } // namespace mxbm::config
