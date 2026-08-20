# Developer fee

MXBM takes a **1.0% developer fee**: for one 36-second round in every
60 minutes you actually spend mining, the miner solves for the developer's
Beam address instead of yours. Over a full day that is about **14.4 minutes**.

The rate matches what the closed-source BeamHash III miners charge — lolMiner
takes 1.0% on BeamHash III, and 0.7–2.5% across its other algorithms.

## What it costs you

| | |
|---|---|
| Rate | 1.0% of mining time |
| Round length | 36 s |
| Cadence | one round per 60 min of mining |
| Per day | ~14.4 min |
| Destination | `beam.herominers.com:1130` (TLS) |

The fee is charged in **time**, not in shares. There is no per-share cut and
no skimming of your submissions: your shares go to your pool, the fee's
shares go to the fee pool, and the two are never mixed.

Only time you are **actually mining** counts. Two things must both hold for
the clock to run: your pool has given the miner a job, and a device is
solving it. So none of the following is charged:

- time spent connecting, waiting on a dead pool, or between reconnects;
- a miner paused with the `p` key;
- a card stopped by its thermal limit;
- a solver backing off after an error;
- a machine suspended and resumed.

A round that gets cut short settles only the time it really spent, so the
remainder stays owed rather than being silently forgiven or double-charged.
A round that overruns settles the overrun too, down to the fraction of a
second, rather than rounding in the developer's favour.

If the fee pool has no current job when a round comes due, the round is
**deferred**, not run — burning your hashrate on a job the miner does not
have, or on one left over from a connection that has since dropped, would
cost you time and pay the developer nothing.

## Raising it: `--dev-fee`

If you want to support the project with more than the built-in rate:

```sh
mxbm --algo BEAM-III --pool ... --user ... --dev-fee 2.5
```

The value is a percentage. It is **raise-only** — a value below the built-in
rate is refused with an error rather than clamped, because clamping would
leave you believing you had lowered it when you had not:

```
--dev-fee 0.5% is below this build's 1% rate; the fee can be raised, not
lowered.
```

A raised rate lengthens the round proportionally (2.5% → a 90 s round per
hour) and is shown everywhere the rate is: the startup line, the statistics
table, `/summary`, and the fee pool worker name below.

`--dev-fee` is a command-line flag only; it is not read from the config files.

## Worker name on fee rounds

A fee round logs in to the fee pool as:

```
<developer address>.<your worker name>_<fee percent>
```

So a rig mining as `<your address>.rig1` at the built-in rate appears as
`rig1_1`, and the same rig with `--dev-fee 2.5` appears as `rig1_2.5`. The
round shows up on the fee pool's dashboard as identifiably yours rather than
as anonymous hashrate, and a raised rate is visible there too.

Your worker name is sanitised to `[A-Za-z0-9_-]` and capped at 32 characters
before it goes on the wire — pools reject exotic worker names, and the
credential travels inside a JSON login line where a stray quote or newline
would corrupt it. If you mine without a worker suffix, the fee round uses
`mxbm`.

## How you can see it

Nothing about the fee is hidden. It is reported in four places:

**At startup**, before it is ever charged:

```
Dev fee: 1% - one 36s round per 60min of mining, to beam.herominers.com:1130
```

A build with no fee compiled in says so instead:

```
Dev fee: none - this build mines entirely for you
```

**At each round**, so the pool switch is never a surprise:

```
Dev fee round started (36s) - mining to the developer's address
Dev fee round finished (36s) - back on your pool
```

**In the statistics table**, with what has actually been spent this session —
enough to check the rate against your own uptime rather than take it on trust:

```
Dev fee 1%: 2 rounds, 72s total, 1/0/0 A/S/R
```

**In the `/summary` API**, as its own object:

```json
"DevFee": {
  "Rate": 0.01, "Active": false, "Rounds": 2, "Seconds": 72.0,
  "Accepted": 1, "Stale": 0, "Rejected": 0
}
```

## What stays yours

Your counters stay yours. Dev-fee shares are accounted in a completely
separate ledger and never touch:

- your accepted / stale / rejected share counts,
- your best share,
- your pool-credited rate (`Pool sol/s`),
- your submit latency.

The one figure deliberately **shared** is the hashrate. `Speed sol/s` measures
what the GPU is doing, and the GPU does the same work either way — splitting
it would make your headline speed dip once an hour for no reason you could
act on. What the fee actually costs you is time, which is what
`Dev fee ... Ns total` reports.

## Why there is a fee

MXBM is built and maintained as open-source software, and the fee is what
funds that work: the solver optimisation, the hardware it is measured on, and
the ongoing maintenance as pools, drivers and the algorithm's ecosystem move.
The rate is the same 1.0% the closed-source BeamHash III miners charge, so it
costs no more to mine with a miner you can read.

It is disclosed rather than buried — announced at startup, announced at both
ends of every round, and reported in the statistics table and `/summary` — so
what you are paying is checkable against your own uptime at any time. If the
project is useful to you and you want to support it further, `--dev-fee`
raises the rate.

## Implementation

The fee's moving parts — the accrual arithmetic, the job router that keeps both pools
live so a switch in either direction is lossless, and the `Origin` tag that routes a
solve back to the pool whose job it came from — are described in
[architecture.md](architecture.md). Tests are in
[`tests/test_devfee.cpp`](../tests/test_devfee.cpp).
