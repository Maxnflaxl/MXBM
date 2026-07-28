# Using MXBM

```
mxbm --algo BEAM-III --pool host:port --user addr[.worker] [options]
```

MXBM's command line and configuration files are modeled on lolMiner's, so
existing Beam mining setups translate with minimal changes.

MXBM mines on a GPU by default (CUDA where available, otherwise OpenCL) and
submits accepted shares to a real pool. A CPU reference solver remains
available via `--solver ref` for validating the pipeline; it is far too slow to
clear pool difficulty and is not a mining option.

## Command-line options

### Required

| Flag | Meaning |
|------|---------|
| `--algo BEAM-III` | Algorithm. MXBM mines BeamHash III only. |
| `--pool host:port` | Pool stratum endpoint. |
| `--user addr[.worker]` | Your BEAM wallet address, optionally with a worker suffix. |

Each of these may come from a configuration file instead — a profile carrying
`ALGO`, `POOL`/`POOLS` and `USER` runs on its own, with nothing on the command
line but `--json --profile <name>`. A *mismatched* `--algo` is still rejected
immediately; only a *missing* one defers to the config.

### Common

| Flag | Meaning | Default |
|------|---------|---------|
| `--pass x` | Pool password, if the pool requires one. | none |
| `--tls [0\|1]` | Enable/disable TLS to the pool. | on |
| `--solver cuda\|opencl\|gpu\|ref\|auto` | Solver backend. `gpu` = any GPU (CUDA preferred), `cuda`/`opencl` pin one, `ref` = CPU reference. | auto |
| `--dev-fee PCT` | Raise the developer fee above its built-in rate, as a percentage. Raise-only. | built-in rate |
| `--nocolor` | Disable ANSI colors in console output. | colors on |
| `--apiport N` | Serve the dashboard and monitoring API on port N (0 = off). | off |
| `--shortstats N` | Seconds between average-speed lines. | 15 |
| `--longstats N` | Seconds between full statistics blocks. | 60 |
| `--log [0\|1]` | Write a timestamped transcript of the console to a file. | off |
| `--logfile PATH` | Where the transcript goes. Implies `--log`. | `logs/mxbm_<date>_<time>.log` |
| `--timeprint [0\|1]` | Stamp the average-speed line with `[HH:MM:SS]`. | off |
| `--digits N` | Decimals on the speed figures, 0–6. | 2 |
| `--pl W` | Board power limit in watts, per GPU (`240`, `240,*,260`; `*` skips one). Needs root. | card default |
| `--no-oc-reset [0\|1]` | Leave `--pl` applied at exit instead of restoring the previous limit. | off |
| `--devices LIST` | Device selector. Accepted and stored; selection is not implemented yet. | all |
| `--watchdog` | Enable the watchdog. Accepted; not implemented yet. | off |
| `--version` | Print the version and exit. | |
| `--help` | Print usage and exit. | |

### Benchmarking

| Flag | Meaning | Default |
|------|---------|---------|
| `--benchmark BEAM-III` | Solve synthetic jobs and report sol/s. No pool, no wallet. | |
| `--benchmark-seconds N` | Stop the benchmark after N seconds. | until Ctrl+C |

`--benchmark` names the algorithm itself, so it satisfies `--algo` on its own:

```sh
mxbm --benchmark BEAM-III --benchmark-seconds 120
```

It drives the same solve path as live mining and reports through the same
stats, so the figure is directly comparable to the mining one — median ms per
solve with p5/p95, so a run can be judged stable without a second run.

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

Bare `--json` (no filename) reads `user_config.json` from the working directory,
so a fully configured profile needs only:

```sh
mxbm --json --profile RIG1
```

Without `--profile`, the first profile in the file is used (file order, not
alphabetical). Unknown keys are ignored, so files written for other miners load
without error.

> A config file holds your wallet address and pool credentials. MXBM's own
> `.gitignore` excludes `/user_config.json` and `/*.cfg` for that reason —
> keep yours out of version control too.

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

### Recognised keys

Both formats accept the same keys, validated the same way — a config file can
never set a value the command line would reject.

| Key | Flag | Values |
|---|---|---|
| `ALGO` | `--algo` | `BEAM-III` |
| `POOL` / `POOLS` | `--pool` | `host:port` (flat: one pool; JSON: an array) |
| `USER`, `PASS`, `TLS` | `--user`, `--pass`, `--tls` | bound to the pool |
| `APIPORT` | `--apiport` | 0–65535 |
| `SHORTSTATS`, `LONGSTATS` | same | seconds, ≥ 1 |
| `DIGITS` | `--digits` | 0–6 |
| `LOG`, `TIMEPRINT`, `WATCHDOG`, `NOCOLOR` | same | `1`/`0`, `true`/`false`, `on`/`off` |
| `LOGFILE` | `--logfile` | path; implies `LOG` unless `LOG` says otherwise |
| `SOLVER` | `--solver` | `cuda`, `opencl`, `gpu`, `ref`, `auto` |
| `DEVICES` | `--devices` | list (JSON also accepts an array) |
| `PL` | `--pl` | watts per GPU (JSON also accepts an array: `[220, "*", 260]`) |
| `NO_OC_RESET` | `--no-oc-reset` | `1`/`0`, `true`/`false`, `on`/`off` |
| `DEVFEE` | `--dev-fee` | percentage, raise-only |
| `BENCHMARK`, `BENCHMARK_SECONDS` | `--benchmark`, `--benchmark-seconds` | `BEAM-III`; seconds ≥ 1 |

Values are case-insensitive where there is a fixed set to choose from, so
`SOLVER = CUDA` and `LOG = ON` both work. Unknown keys are ignored, so a file
written for a later MXBM — or for another miner — still loads.

## Dashboard and monitoring API

`--apiport N` serves two things on the same port:

| Path | |
|------|--|
| `/` (or `/index.html`) | A live dashboard, for a browser |
| `/summary` | A JSON snapshot, for scripts and monitoring |

```sh
mxbm --json --profile rig1 --apiport 8080
```

Then open <http://localhost:8080/>.

### The dashboard

Three tables — session rates, pool/share state, device telemetry — above a set
of charts:

| Chart | Shows |
|-------|-------|
| hashrate | 15s, 60s and pool-credited rate |
| shares found | every share as a dot at its achieved difficulty, on a log axis, against the target it cleared |
| power | watts, plus a cost axis once you enter an energy price |
| clocks | core and memory, on separate axes (they differ ~4×, so one axis would flatten both) |
| temperature / fan | |

Hover any chart for exact values at that point, coloured to match each line.

The page is entirely self-contained — no CDN, no external fonts, no requests
anywhere but this miner's own `/summary` — because rigs often sit on isolated
networks. Enter your energy price on the power chart to get a cost axis and
cost-per-hour/day/month figures; the price is remembered in your browser.

Two things worth knowing about the charts:

- **Chart history lives in the browser tab**, accumulated from page load at one
  sample per two seconds, and restarts on reload. MXBM reports windowed *rates*
  and running counters, not a stored time series. The one exception is the
  shares-found chart, which the miner backs with its own log and which is
  therefore complete the moment the page opens.
- **Axes never zoom tighter than ±3%** of the value. Without that floor, a
  quantity that is genuinely steady renders as violent noise — measured power
  moves about 0.4% peak-to-peak and would otherwise fill the whole chart.

### `/summary`

```sh
curl http://localhost:8080/summary
```

```json
{
  "Software": "MXBM 0.5.126 [2ecb9d4]",
  "Mining":  { "Algorithm": "BeamHash III" },
  "Session": {
    "Uptime_Human": "0h 0m 56s", "Uptime_s": 56,
    "Speed_15s": 52.13, "Speed_60s": 51.45, "Speed_Session": 54.50,
    "Pool_Speed_Session": 90.39,
    "Accepted": 10, "Stale": 0, "Rejected": 0,
    "Best_Share": 7390.32,
    "Job_Difficulty": 512.0, "Job_Id": "58481"
  },
  "Workers": [{
    "Index": 0, "Name": "NVIDIA GeForce RTX 4070 Ti SUPER",
    "Performance": 51.45, "Iterations_s": 26.5,
    "Power_W": 284.37, "Core_Clock_MHz": 2730, "Mem_Clock_MHz": 10251,
    "Temp_C": 65, "Fan_Pct": 54
  }],
  "Stratum": { "Current_Pool": "de.beam.herominers.com:1130", "Latency_ms": 17, "Reconnects": 0 },
  "DevFee":  { "Rate": 0.01, "Active": false, "Rounds": 2, "Seconds": 72.0,
               "Accepted": 1, "Stale": 0, "Rejected": 0 },
  "Session_Stats": {
    "Speed_15s": { "N": 304, "Mean": 55.94, "Stddev": 3.27, "Min": 24.9, "Max": 61.5 },
    "Speed_60s": { "N": 65, "Mean": 55.89, "Stddev": 2.30, "Min": 39.7, "Max": 58.9 },
    "Iterations_s": { "N": 65, "Mean": 29.4, "Stddev": 1.21, "Min": 20.9, "Max": 31.0 },
    "Power_W": { "N": 9000, "Mean": 284.4, "Stddev": 3.06, "Min": 61.0, "Max": 361.2 },
    "Core_Clock_MHz": { "N": 9000, "Mean": 2730, "Stddev": 40.1, "Min": 2100, "Max": 2775 },
    "Mem_Clock_MHz": { "N": 9000, "Mean": 10251, "Stddev": 0.0, "Min": 10251, "Max": 10251 },
    "Temp_C": { "N": 9000, "Mean": 65.2, "Stddev": 3.9, "Min": 38, "Max": 71 },
    "Fan_Pct": { "N": 0, "Mean": null, "Stddev": null, "Min": null, "Max": null }
  },
  "Recent_Shares": [
    { "Age_s": 53.68, "Difficulty": 624.27, "Target": 512.0, "Dev": false }
  ]
}
```

Notes on fields whose behaviour is not obvious from the name:

- **`Session.*` counters exclude developer-fee shares.** Those live in
  `DevFee`, in a separate ledger. `Best_Share` and `Pool_Speed_Session` are
  likewise yours alone.
- **Telemetry fields are `null`, not `0`,** when the platform cannot supply
  them — a laptop reporting power but not fan speed should read as "fan
  unknown", which `0` would misreport as "fan stopped".
- **`DevFee` is always present**, all-zero in a build that charges no fee, so a
  consumer can tell "no fee" from "MXBM too old to report one".
- **`Recent_Shares`** is the last 128 shares, oldest first. `Target` is the
  difficulty *that share* cleared, captured when it was found — not
  `Session.Job_Difficulty`, which is only the current one. Pool vardiff moves
  the target through a session, so pairing an old share with the current target
  would misreport how hard it actually was.
- **`Age_s`, not a timestamp.** MXBM never reads wall-clock time internally, so
  it reports ages; a consumer with a clock converts trivially, and a relative
  figure survives clock skew.
- `Latency_ms` is `-1` until the first share round-trip is measured, rather
  than being remapped to 0.
- **`Speed_*` and `Pool_Speed_Session` measure the same thing two ways** — see
  [Speed vs pool rate](#speed-vs-pool-rate).
- **`Session_Stats` covers the whole run**, where everything else in the
  response is an instant. One entry per sampled field, keyed by the field it
  summarises, holding the sample count, mean, sample standard deviation
  (`n-1`), and the extremes. It is what tells you whether a rate is *steady*,
  which no single reading can, and it is accumulated by the miner — the
  dashboard's charts only hold ~30 minutes and start over on reload, so an
  hour-old thermal spike is gone from them but not from here.

  Three things worth knowing before quoting these:

  - **`N` counts samples, not seconds.** They are folded on the read path, at
    most one per second, so the count follows how often the miner was
    *observed*: the console ticker contributes one per tick, an open dashboard
    one per poll. Nothing is watching, nothing accumulates.
  - **A window is only sampled once it has filled.** `Speed_15s` is 0.0 for the
    first 15 seconds of a session; sampling that would pin `Min` at zero for
    the whole run. So `Speed_15s` starts at 15 s uptime and `Speed_60s` at 60 s.
  - **No median**, because these are constant-memory accumulators that keep no
    samples. `Mean` and `Min`/`Max` are exact; a median needs the series.

  A field the platform never supplied reports `N: 0` with the rest `null` —
  same null-not-zero rule as the telemetry fields above.

### Reaching it from another machine

The server binds all interfaces, so on a firewalled host you only need to open
the port — scoped to your LAN rather than to everything:

```sh
sudo ufw allow from 192.168.1.0/24 to any port 8080 proto tcp comment 'MXBM dashboard'
```

**The API is unauthenticated**, and it is a small hand-rolled HTTP server, so
treat the port as trusted-network-only. It does not expose your wallet address,
but it does reveal your hardware, hashrate, pool and uptime. If you would rather
open nothing, tunnel over SSH instead and browse `localhost:8080`:

```sh
ssh -N -L 8080:localhost:8080 user@rig
```

## Console output

MXBM prints a lolMiner-style console: a banner with the version, `Connecting` /
`Connected (TLS)` / `Authorized worker` lines on startup, a `New job received`
line per job, `Average speed` lines every `--shortstats` seconds, and a full
statistics block every `--longstats` seconds.

Found shares report the achieved difficulty, how many times the pool's target
that was, and the target itself:

```
NVIDIA GeForce RTX 4070 Ti SUPER: Found a share of difficulty 8.0k (3.9x target of 2048)
Share accepted (18 ms)
```

The multiple and the target are on the line because difficulty alone means
nothing without the bar it had to clear — and pool vardiff moves that bar
throughout a session, so the same "8.0k" is a different achievement at
different times. The multiplier is plain ASCII `x` so the line stays greppable
through `tee` and log shippers. Where no target is known yet, the suffix is
omitted rather than printing a meaningless ratio.

Press Ctrl+C to stop.

### Logging to a file

`--log` writes everything the console prints to a file as well — the same lines,
in the same order, minus the colour and plus a timestamp on every one:

```
[2026-07-25 19:16:21] MXBM 0.5.126 [2ecb9d4] — open BeamHash III miner
[2026-07-25 19:16:23] New job received for blockheight 3974400 (job 58481) Difficulty: 512
[2026-07-25 19:16:44] RTX 4070 Ti SUPER: Found a share of difficulty 8.0k (3.9x target of 2048)
[2026-07-25 19:16:44] Share accepted (18 ms)
```

Without `--logfile`, it lands in `logs/mxbm_<date>_<time>.log` relative to the
working directory, creating `logs/` if needed — one file per run, so last
night's session is a file you can point at. `--logfile PATH` overrides that and
implies `--log`; the file is opened for **append**, so a rig that
watchdog-restarts continues its record instead of erasing it. If the file
cannot be opened, MXBM says so and mines anyway.

Every line in the file is timestamped whatever `--timeprint` says — that flag is
about the console, which you are watching live and where the time is usually
redundant. A file is read hours later, where a line with no time on it is nearly
useless. (Turning both on is therefore slightly redundant: the transcript will
carry its own stamp *and* the one `--timeprint` put on the speed line.)

This is also how to get numbers out for offline analysis. The speed lines are
fixed-format, so a session's distribution is one command away:

```sh
grep -o 'Average speed (15s): [0-9.]*' logs/mxbm_*.log | awk '{print $NF}' | sort -n | \
  awk '{v[NR]=$1} END {print "n="NR, "median="v[int(NR/2)], "min="v[1], "max="v[NR]}'
```

`--digits` raises the precision of those figures if two decimals is not enough.

### Power limit

The single most valuable setting on an NVIDIA card, and the reason it exists here:
**MXBM runs pinned at the board power limit in every kernel**, so the limit does not just
bound the miner, it picks its operating point. On the reference RTX 4070 Ti SUPER,
measured:

| `--pl` | sol/s | draw | sol/s/W |
|---|---|---|---|
| 180 | 46.1 | 180.0 W | 0.256 |
| 190 | 48.8 | 189.9 W | 0.257 |
| 200 | 52.2 | 199.8 W | **0.261** — MXBM's own efficiency peak |
| 210 | 54.2 | 209.6 W | 0.258 |
| **220** | **55.4** | **219.5 W** | 0.252 — recommended |
| 240 | 57.2 | 239.4 W | 0.239 |
| 255 | 58.0 | 254.2 W | 0.228 |
| 285 (stock) | 59.2 | 284.2 W | 0.208 — fastest |

Dropping the limit from 285 W to 220 W costs 6 % of throughput and saves 23 % of the
power. Going below ~200 W makes things *worse* on both counts, because by then the core
clock has fallen far enough that the parts of the board which do not scale with it are
being paid for out of less work.

**220 is recommended over MXBM's own 200 W efficiency peak**, and deliberately: below
~212 W lolMiner is ahead of MXBM on speed *and* efficiency, so 200 W is where MXBM looks
best against itself and worst against the alternative. The full curve, and both miners
swept against each other at the same caps, is in
[performance.md](performance.md#both-miners-under-the-same-cap).

```
sudo mxbm --algo BEAM-III --pool ... --user ... --pl 220
```

**It needs root** — every NVML write does. Without it MXBM says so by name and mines on
at the card's current limit rather than failing:

```
Power limit 240 W not applied: insufficient permission - re-run under sudo.
Mining continues at the card's current 285 W.
```

The value is clamped to the band the driver reports for your card (100–366 W on the
reference one) and the clamp is announced, so a limit your card will not take is never
silently ignored. The previous limit is restored when MXBM exits, including on Ctrl+C;
`--no-oc-reset` leaves it applied instead.
For mean and standard deviation without any parsing, the API reports them
directly — see `Session_Stats` above.

### Speed vs pool rate

The statistics block reports two hashrates per device, and the dashboard and
`/summary` carry both. They are not alternatives — they measure the same
quantity in two different places.

**Speed** (`Speed_15s`, `Speed_60s`, the `Speed` column) is counted at your GPU:
solutions the solver returned, divided by the length of the window. **Pool
rate** (`Pool_Speed_Session`, the `Pool` column) is inferred from what the pool
credited you: the summed target difficulty of your accepted shares, divided by
session seconds.

They are in the same units and **agree in expectation**. A solution clears
difficulty `d` with probability `1/d`, so at `S` sol/s you land `S/d` shares per
second, each crediting `d` units of difficulty — `S` units per second, either
way. The pool figure is your own speed seen through the pool's ledger.

Four things make them differ, and only the first is noise:

1. **Statistics.** The pool figure is an estimate from share arrivals, and its
   relative error is roughly `1/√N` over `N` accepted shares: ±32 % at 10
   shares, ±5 % at 400. Over a short session it means nothing.
2. **The developer fee.** The hashrate is shared between both pools, but fee
   shares are credited to the fee ledger, not yours. That is a permanent ~1 %
   deficit by design — see [the dev fee](devfee.md).
3. **Stale and rejected shares.** Found locally, counted in your speed, never
   credited. This is the one worth watching.
4. **It is cumulative, never windowed.** Pool rate divides by total uptime,
   including time before the first job arrived, so it starts at zero, climbs
   toward the true rate, and never forgets a bad patch. There is no 15 s or
   60 s pool rate, and there cannot usefully be one.

**Which to use.** For anything about the hardware — tuning, overclocking,
comparing against another miner — use **speed**, and specifically the 60 s
window: it is the only one with a short window and low variance. For "am I
being paid for what I produce", use **pool rate**, over hours. The pool's own
website computes its hashrate estimate the same way, so that is the
apples-to-apples comparison; your local speed against the pool's display is not.

The **ratio** is the real diagnostic. After a few hundred accepted shares it
should sit near 99 %. Persistently below that means shares are being lost —
check `Stale`, `Rejected` and `Reconnects`.

Quote the **mean and spread**, never a peak. `Session_Stats.Speed_60s` reports
`Mean`, `Stddev`, `Min` and `Max` over the whole session for exactly this
reason: the maximum of a few hundred noisy samples is a property of the sample
count, not of the miner. [docs/performance.md](performance.md) works through a
live session where the 15 s peak was 61.5 sol/s and the honest figure 56.1.

## Developer fee

MXBM takes a 1.0% developer fee: one 36-second round per 60 minutes of mining,
about 14.4 minutes a day. Only time you are actually mining counts toward it.

It announces itself at startup and at each round:

```
Dev fee: 1% - one 36s round per 60min of mining, to beam.herominers.com:1130
Dev fee round started (36s) - mining to the developer's address
Dev fee round finished (36s) - back on your pool
```

Fee shares are kept in a separate ledger — your `Shares` column, best share and
pool-credited rate never include them — and the statistics table carries a
`Dev fee` row showing the rate alongside the rounds and seconds actually spent.

`--dev-fee PCT` raises the fee if you want to support the project with more
(e.g. `--dev-fee 2.5`). It is raise-only: a value below the built-in rate is
refused rather than clamped. Fee rounds log in to the fee pool under your own
worker name plus the rate — `rig1_1` at the built-in rate, `rig1_2.5` with the
example above — so a round is identifiable as yours.

The fee funds MXBM's development and is the same rate the closed-source
BeamHash III miners charge. See **[devfee.md](devfee.md)** for the full terms:
what it costs, how it is calculated, and everywhere it is reported.
