# Using MXBM

```
mxbm --algo BEAM-III --pool host:port --user addr[.worker] [options]
```

MXBM's command line and configuration files are modeled on lolMiner's, so
existing Beam mining setups translate with minimal changes.

> **Note on shares.** Until the GPU solver lands, MXBM runs a CPU reference
> solver that cannot clear pool difficulty. You will see it connect,
> authenticate, and receive jobs, but not submit accepted shares yet. This is
> expected — see the project [roadmap](../README.md#roadmap).

## Command-line options

### Required

| Flag | Meaning |
|------|---------|
| `--algo BEAM-III` | Algorithm. MXBM mines BeamHash III only. |
| `--pool host:port` | Pool stratum endpoint. |
| `--user addr[.worker]` | Your BEAM wallet address, optionally with a worker suffix. |

### Common

| Flag | Meaning | Default |
|------|---------|---------|
| `--pass x` | Pool password, if the pool requires one. | none |
| `--tls [0\|1]` | Enable/disable TLS to the pool. | on |
| `--nocolor` | Disable ANSI colors in console output. | colors on |
| `--apiport N` | Serve the monitoring API on port N (0 = off). | off |
| `--shortstats N` | Seconds between average-speed lines. | 15 |
| `--longstats N` | Seconds between full statistics blocks. | 60 |
| `--version` | Print the version and exit. | |
| `--help` | Print usage and exit. | |

### Multiple pools (failover)

You may pass `--pool` more than once to declare failover pools. Per-pool
credentials bind positionally (the Nth `--user`/`--pass`/`--tls` applies to the
Nth `--pool`); a single `--user` applies to all pools.

```sh
mxbm --algo BEAM-III \
     --pool main.pool:1130   --user addr.rig1 \
     --pool backup.pool:3334 --user addr.rig1
```

> Failover *switching logic* is a planned feature; the current release connects
> to the first pool and reconnects to it on drop.

## Configuration files

Instead of (or alongside) command-line flags, MXBM can read a configuration
file. **Command-line flags always take precedence** over file values.

### JSON profiles (`--json` / `--profile`)

```sh
mxbm --json user_config.json --profile RIG1
```

`user_config.json` maps profile names to settings. Keys are upper-case; `POOLS`
is an array of pool objects:

```json
{
  "RIG1": {
    "ALGO": "BEAM-III",
    "APIPORT": 8080,
    "POOLS": [
      { "POOL": "beam.herominers.com:1130", "USER": "addr.rig1", "PASS": "x" }
    ]
  }
}
```

Bare `--json` (no filename) reads `user_config.json` from the working directory.
Without `--profile`, the first profile in the file is used. Unknown keys are
ignored, so files written for other miners load without error.

### Flat config (`--config`)

```sh
mxbm --config mxbm.cfg
```

A simple `KEY = VALUE` file (`#` starts a comment):

```
ALGO    = BEAM-III
POOL    = beam.herominers.com:1130
USER    = addr.rig1
APIPORT = 8080
```

## Monitoring API

With `--apiport N`, MXBM serves a small HTTP endpoint:

```sh
curl http://localhost:8080/summary
```

`GET /summary` returns a JSON snapshot — software version, algorithm, session
uptime, per-worker speed, and share counts. The API is unauthenticated and binds
to all interfaces (matching lolMiner's behavior); expose the port only on a
trusted network.

## Console output

MXBM prints a lolMiner-style console: a banner with the version, `Connecting` /
`Connected (TLS)` / `Authorized worker` lines on startup, a `New job received`
line per job, `Average speed` lines every `--shortstats` seconds, and a full
statistics block every `--longstats` seconds. Once shares are found (GPU
milestone), share lines report the achieved difficulty and accept latency.

Press Ctrl+C to stop.
