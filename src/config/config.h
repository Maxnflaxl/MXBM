#pragma once
#include <string>

#include "cli/options.h"

// the reference miner-shaped config-file loaders, matching the reference miner's documented
// config-file shape. Two sibling formats,
// exactly like the reference miner: `--json` selects a named profile out of a
// user_config.json-style { "PROFILE": { ...UPPERCASE keys... } } map;
// `--config` reads a flat "KEY = VALUE" file (the reference miner's own plain-text
// grammar is undocumented upstream, so this is a best-effort approximation
// -- see the fixture comment in tests/fixtures/flat.cfg).
//
// Both loaders share one merge rule with cli::parse_args (Task 4): they only
// ever fill an cli::Options field whose cli::Options::Seen flag is still
// false. CLI-supplied values always win over a config file's -- a loader
// call is additive, never destructive, with respect to anything the command
// line already set. Unknown keys are ignored (forward-compat with newer
// the reference miner config keys MXBM doesn't understand yet).
//
// main.cpp picks exactly one loader per run: opts.use_json_config selects
// load_json_config, else a non-empty opts.config_path selects
// load_flat_config. Because --json latches opts.use_json_config true and
// nothing ever clears it, "--json wins" even if a --config flag appears
// later on the same command line and overwrites opts.config_path's text --
// see main.cpp's config-merge block and the Task 5 report for the full
// nuance around that shared-field precedence.
namespace mxbm { namespace config {

// Loads `path` as JSON and selects profile `profile` (an empty string means
// "the first profile in the file, in file order"). Fills fields in `opts`
// per the merge rule above; a profile's "POOLS" array (each entry
// {POOL:"host:port", USER, PASS?, TLS?}) is applied only when
// !opts.seen.pools, and replaces opts.pools wholesale when it is. "ALGO" is
// validated exactly like cli::parse_args's --algo check (must be
// "BEAM-III") whenever the key is present, regardless of any Seen flag --
// Options has no field to store it in, so this is a pure guard against a
// profile silently targeting a different algorithm.
//
// Returns false with a human-readable `err` on: the file can't be opened,
// the JSON is malformed, the file has no profile map, the named profile
// (explicit or defaulted) doesn't exist, a profile field has the wrong
// JSON type or an out-of-range value, or ALGO isn't "BEAM-III".
bool load_json_config(const std::string& path, const std::string& profile,
                       cli::Options& opts, std::string& err);

// Loads `path` as a flat "KEY = VALUE" file, one setting per line ('#' and
// blank lines skipped, keys matched case-insensitively). Unlike the JSON
// format's POOLS array, a flat file has exactly one implicit pool: its
// POOL/USER/PASS/TLS keys are scalars, applied as a single pool entry only
// when !opts.seen.pools (and only when a POOL key is actually present).
// Same Seen-gated scalar merge, same ALGO validation, and the same error
// contract as load_json_config.
bool load_flat_config(const std::string& path, cli::Options& opts, std::string& err);

} } // namespace mxbm::config
