#pragma once

namespace mxbm { namespace miner {

// Which pool a unit of work belongs to.
//
// MXBM keeps two stratum connections up at once -- the user's pool and the
// developer-fee pool (miner/devfee.h) -- and both feed the SAME Engine and
// solver, so every piece of work in flight has to carry which connection it
// came from. This is a correctness requirement: a job id is only meaningful
// to the pool that issued it, and a fee slice can end while a solve for it is
// still running, so routing by "which pool is active right now" would hand the
// dev pool's job to the user's pool and have it rejected as unknown. Engine
// tags each job with its Origin and hands that tag back to submit_fn; Stats
// uses it to keep its two ledgers apart (miner/stats.h).
enum class Origin { Main, Dev };

// "pool" / "dev fee pool" -- for console lines and the /summary API.
inline const char* origin_name(Origin origin) {
    return origin == Origin::Dev ? "dev fee pool" : "pool";
}

} } // namespace mxbm::miner
