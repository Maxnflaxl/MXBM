#pragma once

namespace mxbm { namespace miner {

// Which pool a unit of work belongs to.
//
// MXBM keeps two stratum connections up at once: the user's pool, and the
// developer-fee pool that a fee slice mines to (see miner/devfee.h for the
// schedule and docs/devfee.md for the user-facing description). Both feed
// the SAME Engine and the SAME solver -- there is only one GPU -- so every
// piece of work in flight has to carry which connection it came from.
//
// This is a correctness requirement, not bookkeeping. A job id is only
// meaningful to the pool that issued it, and a fee slice can end while a
// solve for it is still running: routing a solution by "which pool is
// active right now" would hand the dev pool's job to the user's pool at
// every window boundary, and the pool would reject it as an unknown job.
// Engine therefore tags each job with its Origin when it enters the
// mailbox and hands that same tag back to submit_fn, so the solution is
// always submitted to the connection whose job produced it -- whatever the
// scheduler has done in the meantime.
//
// Stats uses the same tag to keep the two ledgers apart (miner/stats.h):
// the user's accepted/stale/rejected counters must never include shares
// their hardware found for the developer.
enum class Origin { Main, Dev };

// "pool" / "dev fee pool" -- for console lines and the /summary API.
inline const char* origin_name(Origin origin) {
    return origin == Origin::Dev ? "dev fee pool" : "pool";
}

} } // namespace mxbm::miner
