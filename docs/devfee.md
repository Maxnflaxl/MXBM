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

The fee is charged in **time**. There is no per-share cut: your shares go to
your pool, the fee's shares go to the fee pool, and the two are never mixed.

Only time you are **actually mining** counts. Two things must both hold for
the clock to run: your pool has given the miner a job, and a device is
solving it. So none of the following is charged:

- time spent connecting, waiting on a dead pool, or between reconnects;
- a miner paused with the `p` key;
- a card stopped by its thermal limit;
- a solver backing off after an error;
- a machine suspended and resumed.

A round settles the time it really spent, to the fraction of a second. A round
cut short leaves the remainder owed. A round that overruns settles the overrun
too.

If the fee pool has no current job when a round comes due, the round is
**deferred**. Mining a job the fee pool has not given would cost you time and
pay the developer nothing.

## Raising it: `--dev-fee`

If you want to support the project with more than the built-in rate:

```sh
mxbm --algo BEAM-III --pool ... --user ... --dev-fee 2.5
```

The value is a percentage, and it is **raise-only**. A value below the
built-in rate is refused with an error, so you cannot end up believing you
lowered it when you had not:

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
round is identifiably yours on the fee pool's dashboard, and a raised rate is
visible there too.

Your worker name is sanitised to `[A-Za-z0-9_-]` and capped at 32 characters
before it goes on the wire. If you mine without a worker suffix, the fee round
uses `mxbm`.

## How you can see it

The fee is reported in four places.

**At startup**, before it is ever charged:

```
Dev fee: 1% - one 36s round per 60min of mining
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

**In the statistics table**, with what has actually been spent this session,
which you can check against your own uptime:

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

The hashrate is the one figure deliberately shared. `Speed sol/s` measures
what the GPU is doing, and the GPU does the same work either way. What the fee
costs you is time, and `Dev fee ... Ns total` reports it.

## Why there is a fee

The fee funds the work: solver optimisation, the hardware it is measured on,
and maintenance as pools, drivers and the ecosystem move. At 1.0% it is the
same rate the closed-source BeamHash III miners charge.

## Implementation

The fee's moving parts — the accrual arithmetic, the job router that keeps both pools
live so a switch in either direction is lossless, and the `Origin` tag that routes a
solve back to the pool whose job it came from — are described in
[architecture.md](architecture.md). Tests are in
[`tests/test_devfee.cpp`](../tests/test_devfee.cpp).
