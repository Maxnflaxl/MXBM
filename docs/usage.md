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
| `-a`, `--algo BEAM-III` | Algorithm. MXBM mines BeamHash III only. |
| `-p`, `--pool host:port` | Pool stratum endpoint. |
| `-u`, `--user addr[.worker]` | Your BEAM wallet address, optionally with a worker suffix. Optional for a loopback pool — see below. |

`-c BEAM` (`--coin BEAM`) selects the same thing by currency instead of by
algorithm, so a command line written for another BeamHash III miner runs
unchanged. Short flags cannot be combined: write `-a BEAM-III -u addr`, not
`-au`.

Each of these may come from a configuration file instead — a profile carrying
`ALGO`, `POOL`/`POOLS` and `USER` runs on its own, with nothing on the command
line but `--json --profile <name>`. A *mismatched* `--algo` is still rejected
immediately; only a *missing* one defers to the config.

### Common

| Flag | Meaning | Default |
|------|---------|---------|
| `--pass x` | Accepted and ignored — BeamHash III's stratum login carries the address alone, so no password is ever sent. | none |
| `--tls [0\|1]` | Enable/disable TLS to the pool. | on; off for a loopback pool |
| `--solver cuda\|opencl\|gpu\|ref\|auto` | Solver backend. `gpu` = any GPU (CUDA preferred), `cuda`/`opencl` pin one, `ref` = CPU reference. | auto |
| `--dev-fee PCT` | Raise the developer fee above its built-in rate, as a percentage. Raise-only. | built-in rate |
| `--nocolor` | Disable ANSI colors in console output. | colors on |
| `--apiport N` | Serve the dashboard and monitoring API on port N (0 = off). | off |
| `--apihost ADDR` | Interface the API binds. `0.0.0.0` is every interface; `127.0.0.1` restricts it to this machine. IPv4 only. | 0.0.0.0 |
| `--shortstats N` | Seconds between average-speed lines. | 15 |
| `--longstats N` | Seconds between full statistics blocks. | 60 |
| `--log [0\|1]` | Write a timestamped transcript of the console to a file. | off |
| `--logfile PATH` | Where the transcript goes. Implies `--log`. | `logs/mxbm_<date>_<time>.log` |
| `--timeprint [0\|1]` | Stamp the average-speed line with `[HH:MM:SS]`. | off |
| `--silence N` | Console verbosity, 0–3. See [Quieting the console](#quieting-the-console). | 0 |
| `--compactaccept` | Report accepted shares as `*` marks on the average-speed line instead of two lines each. | off |
| `--digits N` | Decimals on the speed figures, 0–6. | 2 |
| `--statsformat LIST` | Which columns the statistics block shows, in order. See [Choosing the columns](#choosing-the-columns). | the default set |
| `--vstats` | One column per GPU, fields as rows. | off |
| `--hstats [N]` | Wrap the table into groups N characters wide; bare, it asks the terminal. | off |
| `--keepfree MB` | VRAM to leave unallocated, replacing the built-in reserve. See [VRAM sizing](#vram-sizing). | 256 MB with a display attached, 64 MB headless |
| `--tstop C` | Pause a GPU when it reaches this temperature. See [Thermal protection](#thermal-protection). | 0 (off) |
| `--tstart C` | Resume a paused GPU at this temperature. | 0 (stays paused) |
| `--tmode MODE` | Which sensor those read: `edge`, `junction`, `memory`. | edge |
| `--pl W` | Board power limit in watts, per GPU (`240`, `240,*,260`; `*` skips one), or `auto` for the value a `--tune` run stored for this card. Needs root. | card default |
| `--tune` | Measure this card's own power/speed curve and recommend `--pl` (and, when it pays, `--mclk`) values (see [Tuning](#tuning-measure-your-own-card)). Needs root, ~25 min, no pool. | |
| `--cclk MHz` | Lock the core clock. Needs root. | driver-managed |
| `--mclk MHz` | Lock the memory clock. Needs root. | driver-managed |
| `--coff MHz` | Shift the core voltage/frequency curve. May be negative. Needs root. | 0 |
| `--moff MHz` | Shift the memory voltage/frequency curve. May be negative. Needs root. | 0 |
| `--fan PCT` | Fan target, in percent. Needs root. | driver's own curve |
| `--no-oc-reset [0\|1]` | Leave applied settings on the card at exit instead of restoring them. | off |
| `--devices LIST` | Which GPU to mine on: `ALL`, a vendor (`NVIDIA`, `AMD`, `INTEL`, `APPLE`), or a comma-separated list of indices from `--list-devices`. | ALL |
| `--devicesbypcie` | Read `--devices` as PCI addresses instead of indices. `1:0`, `01:00` and `0000:01:00.0` all name the same card. | off |
| `--list-devices` | Print the detected GPUs with their indices, and exit. | |
| `--list-algos` | Print the supported algorithms, and exit. | |
| `--list-coins` | Print the supported coins, and exit. | |
| `--watchdog [ACTION]` | Watch for a GPU that stops working. `exit` (default), `script`, or `off`. | off |
| `--watchdogscript PATH` | Script to run when the action is `script`. | |
| `-v`, `--version` | Print the version and exit. | |
| `-h`, `--help` | Print usage and exit. | |

### Benchmarking

| Flag | Meaning | Default |
|------|---------|---------|
| `--benchmark BEAM-III` | Solve synthetic jobs and report sol/s. No pool, no wallet. Uses one device. | |
| `--benchmark-seconds N` | Stop the benchmark after N seconds. | until Ctrl+C |
| `--report` | Benchmark this card, measure its power curve, and print a paste-ready report (see [Reporting](#reporting-send-us-how-your-card-does)). No pool. | |
| `--report-seconds N` | Length of the report's benchmark. | 120 |
| `--report-out PATH` | Where to save the report. A directory saves the default filename inside it. | `~/.config/mxbm/report-<gpu>-<date>.md` |

`--benchmark` names the algorithm itself, so it satisfies `--algo` on its own:

```sh
mxbm --benchmark BEAM-III --benchmark-seconds 120
```

It drives the same solve path as live mining and reports through the same
stats, so the figure is directly comparable to the mining one — median ms per
solve with p5/p95, so a run can be judged stable without a second run. On cards
with an energy counter (NVIDIA Volta and newer) it also prints joules total, mean
watts and J/solution read from the counter itself — exact, not integrated from
power samples — which is the number to compare when the question is efficiency
rather than speed.

### Reporting: send us how your card does

```sh
sudo mxbm --report
```

The only command worth remembering if you are reporting a result. It benchmarks the card
for two minutes through the same solve path mining uses, samples telemetry over the run,
measures the card's power/speed curve, and prints the whole thing as a markdown block
ready to paste into a
[benchmark report issue](https://github.com/maxnflaxl/MXBM/issues/new?template=benchmark-report.yml).
Nothing is uploaded. It also records three things a person filling in a form usually
cannot, and each decides whether a figure means anything: whether a display was attached
to the card, whether another process was using it, and what the clocks were limited by.

**The curve takes about half an hour, and is measured once.** A later `--report` on the
same binary reads it back and finishes in two minutes; a different build re-measures,
because a curve taken on different kernels describes a different program. Measuring it
needs root (or an Administrator terminal on Windows) — without that you still get
everything except the curve, and that is still a useful report.

**The report is always saved**, and the path is printed:

```
Saved to /home/you/.config/mxbm/report-nvidia-geforce-rtx-4070-ti-super-2026-08-20.md
```

It lands next to the tune store, named after the card and the day; a second run on the
same day replaces it. `--report-out` chooses somewhere else — an absolute or relative
path, `~/` expands, missing directories are created, and naming a *directory* saves the
default filename inside it. Under `sudo` the file is owned by you, not root.

**On a multi-GPU rig a report covers one card at a time.** With more than one card
`--report` lists them and stops rather than guessing, because a block that silently
covered device 0 would read as a rig figure:

```sh
sudo mxbm --report --devices 1                       # one card
sudo mxbm --report --devices ALL --report-out rig.md # every card, in turn
```

`--devices ALL` walks the cards one after another — never together, since two cards
measured at once each report the other's contention — and collects every block into one
file, so a rig is still one paste. Budget half an hour per card on the first run. Curves
are stored per card (keyed by name and PCI address), so identical cards in different
slots keep their own; cooling differs between slots.

`--report` always sweeps at the defaults: `--tune-caps` and `--tune-seconds` configure
`--tune` and are refused here, since a hand-picked grid produces a verdict that reads
like a full sweep's without the refining passes behind it. Use `--tune` directly when you
want to choose the grid.

### Tuning: measure your own card

The [power-limit table below](#power-limit) is the reference card's curve; `--tune`
produces the same table for **your** card, then recommends a wattage:

```sh
sudo mxbm --tune
```

About 25 minutes, in four passes plus a drift gauge:

| pass | what it measures | why |
|---|---|---|
| coarse | six points across the band your driver reports, 60 s each, live mining path, CPU-verified sol/s | locates the best cap's neighbourhood |
| fine | ~10 W steps bracketing it | the coarse grid places the peak only to within its own spacing; this is what distinguishes 220 from 248 |
| rung | the capped points again at your card's low memory rung — picked from the driver's own supported-clock list, held-clock verified every arm | under a low cap the memory interface burns watts for bandwidth the slowed core cannot use |
| efficiency | refines around the best sol/s-per-watt point found, on whichever memory clock it sits | the coarse spacing blurs the efficiency optimum exactly as it blurs the recommendation |
| drift gauge | the first point, measured a second time | past ±1.5 % the table says the card heat-soaked rather than pretending it did not |

It prints the curve and three recommendations: the **recommended `--pl`** (the highest
limit where each extra watt still returns at least 0.07 sol/s — above it you are buying
watts, not speed), the **best-efficiency point** (most sol/s per measured watt, which may
land on the rung), and — when the rung measured faster — the cap **below which to add
`--mclk <rung>`**, a threshold measured on your card rather than copied from ours.

The result is stored per card in `~/.config/mxbm/tune.json` — in *your* config dir even
under sudo — so later runs can just say:

```sh
sudo mxbm --algo BEAM-III --pool ... --user ... --pl auto
```

`--pl auto` prints the stored value and its measurement date, so a stale tune is visible;
when the resolved cap is inside the rung's paying band it also prints a one-line reminder
to consider `--mclk`, recommended but never auto-applied. Re-run after driver updates or
cooling changes. On a rig that does not mine as root, read the recommendation once and
put `sudo nvidia-smi -pl <watts> -lmc <rung>,<rung>` in the boot sequence instead.

Knobs: `--tune-seconds N` (per point, default 60), `--tune-caps "100,160,220"` (exactly
these points, which also skips the refinement and rung passes), `--tune-min-gain X` (the
sol/s-per-watt bar, default 0.07 — the one number that is a preference, not a
measurement). `--tune` refuses a simultaneous `--pl` but allows the other OC flags; a
given `--mclk` disables the rung pass. Ctrl+C aborts and restores the previous limit and
memory clock.


### Multiple pools (failover)

You may pass `--pool` more than once to declare failover pools. Per-pool
credentials bind positionally (the Nth `--user`/`--pass`/`--tls` applies to the
Nth `--pool`); a single `--user` applies to all pools.

```sh
mxbm --algo BEAM-III \
     --pool main.pool:1130   --user addr.rig1 \
     --pool backup.pool:3334 --user addr.rig1
```

MXBM stays on the first pool while it answers. After **three consecutive failed
redials** it moves to the next in the list, wrapping at the end, and says so on the
console. Rotation is deliberately slow: a pool that drops a connection is usually back
within seconds, and moving on the first failure would hand the rig to the backup over a
blip — then leave it there, because nothing pulls it home.

### Local pools (loopback)

A pool on the loopback interface — `localhost`, `127.0.0.0/8`, `::1` — is a process on
the same machine: a node's own stratum server, a proxy, a decentralised-pool daemon. Two
defaults change there, because neither requirement means anything locally:

- **TLS is off.** A loopback listener has no certificate to present.
- **`--user` is optional.** There is no wallet address to authenticate as; the
  credential is a local API key, often not enforced at all. With none given MXBM sends
  `x`.

So a local daemon needs one flag:

```sh
mxbm --algo BEAM-III --pool 127.0.0.1:3416
```

An explicit `--tls` or `--user` still wins, and neither relaxation applies to a remote
pool: a remote `--pool` without `--user` remains an error, since mining to a pool with
no address of yours mines for nobody.

Each pool keeps its own credentials. They are different accounts, and re-logging into
pool B with pool A's wallet would mine for the wrong address.

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
| `USER`, `PASS`, `TLS` | `--user`, `--pass`, `--tls` | bound to the pool (`PASS` is accepted and ignored) |
| `APIPORT` | `--apiport` | 0–65535 |
| `APIHOST` | `--apihost` | IPv4 address, e.g. `0.0.0.0` or `127.0.0.1` |
| `SILENCE` | `--silence` | 0–3 |
| `KEEPFREE` | `--keepfree` | megabytes, 0 or more |
| `STATSFORMAT` | `--statsformat` | field list (JSON also accepts an array) |
| `VSTATS`, `HSTATS` | same | `1`/`0`, `true`/`false`, `on`/`off` |
| `TSTOP`, `TSTART` | `--tstop`, `--tstart` | degrees C, 0 disables |
| `TMODE` | `--tmode` | `edge`, `junction`, `memory` |
| `COMPACTACCEPT`, `DEVICESBYPCIE` | same | `1`/`0`, `true`/`false`, `on`/`off` |
| `SHORTSTATS`, `LONGSTATS` | same | seconds, ≥ 1 |
| `DIGITS` | `--digits` | 0–6 |
| `LOG`, `TIMEPRINT`, `WATCHDOG`, `NOCOLOR` | same | `1`/`0`, `true`/`false`, `on`/`off` |
| `WATCHDOGSCRIPT` | `--watchdogscript` | path |
| `LOGFILE` | `--logfile` | path; implies `LOG` unless `LOG` says otherwise |
| `SOLVER` | `--solver` | `cuda`, `opencl`, `gpu`, `ref`, `auto` |
| `DEVICES` | `--devices` | `ALL` or a list of indices (JSON also accepts an array) |
| `PL` | `--pl` | watts per GPU (JSON also accepts an array: `[220, "*", 260]`) |
| `CCLK`, `MCLK` | `--cclk`, `--mclk` | MHz per GPU (JSON also accepts an array) |
| `COFF`, `MOFF` | `--coff`, `--moff` | MHz per GPU, signed (JSON also accepts an array) |
| `FAN` | `--fan` | percent per GPU (JSON also accepts an array) |
| `NO_OC_RESET` | `--no-oc-reset` | `1`/`0`, `true`/`false`, `on`/`off` |
| `DEVFEE` | `--dev-fee` | percentage, raise-only |
| `BENCHMARK`, `BENCHMARK_SECONDS` | `--benchmark`, `--benchmark-seconds` | `BEAM-III`; seconds ≥ 1 |

Values are case-insensitive where there is a fixed set to choose from, so
`SOLVER = CUDA` and `LOG = ON` both work. Unknown keys are ignored, so a file
written for a later MXBM — or for another miner — still loads.

## Dashboard and monitoring API

`--apiport N` serves two things on the same port: a live dashboard at `/` for a browser,
and a JSON snapshot at `/summary` for scripts and monitoring.

```sh
mxbm --json --profile rig1 --apiport 8080
```

Then open <http://localhost:8080/>.

The dashboard charts hashrate, share difficulty, power (with an optional energy-cost
axis), clocks and temperature, and is entirely self-contained — no CDN, no external
fonts — because rigs often sit on isolated networks.

**The API is unauthenticated.** It does not expose your wallet address, but it does
reveal your hardware, hashrate, pool and uptime, so treat the port as
trusted-network-only; `--apihost 127.0.0.1` restricts it to the rig itself.

Every field, the dashboard's charts, and how to reach it from another machine are in
**[api.md](api.md)**.

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
omitted.

Press Ctrl+C to stop.

### Console keys

While mining in an interactive terminal, single keys work as commands — no
Enter needed:

| Key | Action |
|---|---|
| `h` | Print the short speed line now. |
| `s` | Print the full statistics block now. |
| `c` | One connection line: pool, uptime, reconnects, last job, share latency. |
| `p` | Pause mining. Devices go idle after the solve in flight; jobs are still tracked, so `r` resumes on the current work. |
| `r` | Resume after `p`. |
| `?` | List the keys. |

Keys are case-insensitive, and every command's output goes through the normal
console — timestamped into the `--log` transcript, and printed even under
`--silence`, since a key press is an explicit request. A paused device shows
as paused in the statistics table, and the watchdog knows the difference
between paused and hung.

The keys exist only when stdin is an interactive terminal. Under a pipe, a
redirect, or a service manager the reader never starts and stdin is left
untouched, so scripted and unattended runs behave exactly as before. For
remote control of an unattended rig, use the [HTTP API](#dashboard-and-monitoring-api)
instead.

### Quieting the console

A rig with several cards prints a job line per job and two lines per share, which
is a wall of text on a screen you only glance at. `--silence N` turns that down:

| N | What is printed |
|---|---|
| 0 | everything (default) |
| 1 | no `New job received` lines |
| 2 | no job lines, and no per-share lines — each accepted share becomes a `*` on the average-speed line |
| 3 | the statistics block only; even the average-speed line and the startup banner are gone |

```
Average speed (15s): 56.53 sol/s***
Average speed (15s): 60.87 sol/s*
```

`--compactaccept` selects just the `*` marks, leaving job lines alone; level 2
turns it on by itself. One mark is one accepted share in that interval, so the
marks between two statistics blocks add up to the block's accepted count.
**Rejected shares are printed at every level** — they are the one share line
worth interrupting for, and they are rare enough not to flood anything.

`--silence` applies to the `--log` transcript exactly as it applies to the
screen: the file records what was displayed, so what you watch live and what you
read afterwards are the same thing. To keep more in the file, lower the level.

The startup banner is part of that transcript — the log opens before it is
printed, so a pasted log carries the same header a screenshot does: version,
licence, and which backends this binary was *built* with. That last line answers a
question no runtime probe can: a machine with no CUDA card and a binary with no
CUDA support look identical from the device table alone.

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

`--digits` raises the precision of those figures if two decimals is not enough. For
mean and standard deviation without any parsing, the API reports them directly — see
[`Session_Stats`](api.md#summary).

### Power limit

The single most valuable setting on an NVIDIA card, and the reason it exists here:
**MXBM runs pinned at the board power limit in every kernel**, so the limit does not just
bound the miner, it picks its operating point. On the reference RTX 4070 Ti SUPER,
measured:

**`--pl 210` is the setting to use** on the reference RTX 4070 Ti SUPER — its efficiency
peak, and comfortably inside the window where MXBM beats the alternative, whose crossing
is at ~177 W:

| `--pl` | sol/s | sol/s/W | |
|---|---|---|---|
| 160 W | 46.2 | 0.2887 | |
| 180 W | 54.0 | 0.2999 | |
| **210 W** | **64.9** | **0.3094** | the efficiency peak, and recommended |
| 240 W | 67.9 | 0.2834 | |
| 285 W *(stock)* | **69.95** | 0.2461 | fastest |

Dropping from 285 W to 210 W costs 7 % of throughput and saves 26 % of the power. Going
below the peak makes things *worse* on both counts: the core clock has fallen far
enough that the parts of the board which do not scale with it are being paid for out of
less work.

The full curve — every cap from the driver's 100 W floor up, and both miners swept
against each other at the same caps — is in
[performance.md](performance.md#both-miners-under-the-same-cap). These numbers are one
card's; `sudo mxbm --tune` measures yours ([Tuning](#tuning-measure-your-own-card)).

**Running capped below ~165 W? Drop the memory clock too.** Under a low cap the GDDR
interface burns watts for bandwidth the slowed core cannot use; the 5001 MHz rung returns
them as core clock — **+7 % sol/s at 160 W rising to +18 % at the 100 W floor**, at the
same wall power. Pair the two through the miner itself — one command, and both settings
are restored when it exits:

```sh
sudo mxbm --algo BEAM-III --pool ... --user ... --pl 160 --mclk 5001
```

On a rig that doesn't run the miner as root, set the card once instead
(`sudo nvidia-smi -pl 160 -lmc 5001,5001`, e.g. at boot) and mine unprivileged. The
crossover is ~167 W and above ~180 W the rung is a wall, so this is strictly a low-cap
pairing. On energy per solution it only ties the shelf — 3.219 J/solution at 160 W on
the rung against 3.232 at 210 W on stock memory, where the card does 31 % more work.
Details in
[performance.md](performance.md#below-stock-the-other-rung-pays-7-to-18--under-caps-below-165-w).

```
sudo mxbm --algo BEAM-III --pool ... --user ... --pl 210
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

### Choosing the columns

`--statsformat` takes a comma-separated list of field names, in the order you
want them:

```sh
mxbm ... --statsformat gpuName,speed,power,coreT,state
```

```
      Name        Speed  Power  Core   State
                  sol/s      W  Temp
RTX 4070 Ti SUPER 12.60    284    65  mining
---------------------------
Total             12.60    284
```

| Field | Column |
|---|---|
| `gpuName` | the card, shortened to fit |
| `speed` | this device's 60-second rate |
| `poolHr` | the rate the pool credited, from accepted-share difficulty |
| `iter` | solve attempts per second |
| `shares` | accepted / stale / rejected |
| `sharesPerMin` | accepted shares per minute since startup |
| `bestShare` | the highest difficulty found |
| `hrPerWatt` | sol/s per watt |
| `wattPerHr` | watts per sol/s |
| `power` | draw in watts |
| `coreClk`, `memClk` | clocks in MHz |
| `coreT` | core temperature |
| `fanPct` | fan speed |
| `util` | GPU utilisation |
| `state` | `mining` or `paused` (see [Thermal protection](#thermal-protection)) |

The first twelve are the default set, which is what you get without the flag.
**There are no presets**; a word that is not a field name is an error listing the
fields.

`--vstats` turns the table on its side, one column per GPU plus a Total, which
reads better on a narrow terminal or a rig with many cards:

```
                             GPU 0       Total
Name:                 RTX 4070 Ti          Rig
Speed (sol/s):               19.30       19.30
Efficiency (sol/s/W):        0.068       0.068
Power (W):                     285         285
Temp (deg C):                   66
```

`--hstats N` keeps the rows but wraps the columns into groups N characters wide,
each group repeating the label column and its own Total row. `--hstats` with no
number asks the terminal how wide it is.

### Thermal protection

`--tstop C` pauses a card that reaches C degrees; `--tstart C` resumes it once it
has cooled to C. Either at 0 disables that half, so `--tstop 85` alone pauses a
hot card and leaves it paused.

```
Thermal: pausing at 85 C, resuming at 75 C (--tmode edge)
GPU 0: paused at 85 C (--tstop). It is not hung; mining resumes when it cools.
GPU 0: resuming at 75 C (--tstart).
```

The pause is cooperative: a solve already running finishes, nothing is torn down,
and resuming costs only the wait. The card is checked every 2 seconds — a card
under load can climb 20 C between slower polls, and every degree of that
overshoot is above the limit you set.

**The watchdog knows about it.** A paused card stops completing solves, which is
exactly what `--watchdog` looks for; MXBM excludes paused devices, so `--tstop`
with the default `--watchdog exit` cannot turn into a restart loop back into the
same heat. A paused card is also marked `"Paused": true` in `/summary`, so a
monitoring dashboard shows *why* it reads 0 sol/s.

`--tmode` selects the sensor. **On consumer NVIDIA cards only `edge` is
readable** — the driver answers "not supported" for memory temperature and the
whole T.Limit family, and reports no junction sensor at all. Asking for one of
those is an error at startup rather than a silent fall back to edge, because
protecting a card by a temperature you did not choose is worse than not
protecting it.

### VRAM sizing

MXBM sizes its pipeline against what the driver reports **free**, minus a fixed
reserve, and prints all four numbers at startup:

```
VRAM: 16376 MB total, 15620 MB free, reserving 256 MB (display attached) -> 15364 MB usable
```

`free` is already net of the driver's own reserved region and of every other
process, so the reserve covers only what can appear *after* MXBM allocates:
another GPU client starting, allocator overhead, context growth. That is a
constant, not a share of the card — 256 MB when a display is attached (queried,
not assumed), 64 MB headless.

`--keepfree MB` replaces that reserve outright. `--keepfree 0` takes everything
the driver reports free; a large value hands memory back to the desktop. It is
not capped: ask for more than the pipeline can work with and MXBM refuses to start
rather than mine a partial search that finds nothing —

```
GPU has too little memory for BeamHash III: it can host only 11834786 of the
required 33554432 seed elements. [...] Need ~8.9 GiB of usable VRAM; this device
offers 3.1 GiB after the reserve (see --keepfree).
```

Two caveats. On Windows the display driver can page GPU memory, so over-allocating
degrades speed instead of failing cleanly — a reason not to set `--keepfree 0` on
a desktop machine. And the free figure is a snapshot taken at startup: something
launched afterwards competes for what is left, whatever was reserved.

On a small card the usable figure decides which **geometry** MXBM runs, not
whether it runs at all: the solver walks a ladder of rungs from 6.17 GiB down to
4.03 and takes the fastest that fits, so `--keepfree` is also the knob that
trades a rung for desktop headroom. The rungs and what each card class gets are
in [HW_REQUIREMENTS.md](HW_REQUIREMENTS.md#the-vram-ladder).

### Choosing which GPU to mine on

```sh
mxbm --list-devices
```

```
Detected devices (indices are in PCI order, and mean the same card in --devices and --pl):
  0: NVIDIA GeForce RTX 4070 Ti SUPER   15963 MB  PCI 1:0     Cuda
```

`--devices ALL` (the default) or `--devices 0,2` picks by those indices. An index that
does not exist is an error naming how many were found — a rig that quietly mines the
wrong card is worse than one that refuses to start.

**Indices are in PCI order**, which is what makes `--devices 1` and `--pl 240,*,260`
refer to the same physical card. CUDA's own enumeration defaults to fastest-first and
NVML's is by bus id, so the two disagree on any rig whose cards are not identical;
sorting by PCI address is the only key all three agree on.

**Two ways to name a card that are not its index.** `--devices NVIDIA` (or `AMD`,
`INTEL`, `APPLE`) selects every card from that vendor. `--devicesbypcie` reads
`--devices` as PCI addresses instead — `--devices 1:0,41:0 --devicesbypcie` — in
whichever form your tooling prints: `1:0`, `01:00` and `0000:01:00.0` all name the
same card. Addresses are worth the extra typing on a rig you do not physically
watch: indices renumber when a card drops off the bus, so yesterday's
`--pl 240,*,200` lands on different cards than it did today, while an address
still means one slot. An address or vendor that matches nothing is an error, the
same way a missing index is.

**A mixed rig is one process.** The last column of `--list-devices` is the backend that
card will actually run on: CUDA where the card supports it (Ampere or newer, with room
for the full search), else OpenCL, and `not used` with the reason where neither can
drive it. The three device lists — CUDA's, OpenCL's and NVML's — are joined on PCI bus
id, so an old card beside a new one mines on the portable path in the same process, the
same log and the same statistics block. Where that join is ambiguous — an OpenCL driver
that reports no PCI address, two devices claiming one bus, a name that does not match —
MXBM says so and declines the match rather than guessing, because a wrong guess would
aim `--pl` at the wrong card.

**Several GPUs mine at once.** Each selected card gets its own solver, its own worker
thread and its own **nonce lane**: card *n* of *N* walks nonces *n, n+N, n+2N, …* off the
pool's prefix, so no two cards ever try the same nonce. That matters because the failure
is silent — two cards duplicating each other's work would each look perfectly healthy
while the rig did half of what it was paid for.

A card that fails to initialise is reported and skipped rather than taking the rig down;
the others keep mining. **Each card gets its own row** in the statistics block and its own
entry in `/summary`, with the rig's totals underneath:

```
      Name        Speed   Pool  Iter.   Shares   Best     Eff.  Power  CCLK   MCLK  Core  Fan
                  sol/s  sol/s   it/s    A/S/R  Share  sol/s/W      W   MHz    MHz  Temp  Pct
GPU 0 RTX 4070 Ti 58.70     --   29.4   20/1/0  12.8k    0.207    284  2685  10251    64   50
GPU 1 RTX 3080    53.70     --   27.4   14/0/0   9.8k    0.203    264  2585  10251    65   55
---------------------------
Total             112.40 110.20   56.8   34/1/0  12.8k    0.205    548
```

What the Total row does and does not add up is deliberate:

- **Speed, iterations, share counts and power are summed.** Watts add, and the rig's
  draw is what sizes a power supply and sets the electricity bill. If any card's power
  cannot be read the wattage is blank rather than partial — understating a rig's draw is
  the direction that trips a breaker.
- **Efficiency on the Total is total speed over total watts**, the rig's real sol/s/W,
  not the average of the per-card ratios.
- **Clocks, temperature and fan are left blank.** A rig has no single core clock, and an
  averaged temperature is a figure no card measured.
- **The pool column appears only on the Total.** The pool credits shares without saying
  which GPU found them, so a per-card share of that rate would be invented. Per-card
  A/S/R comes from matching each result to the card that submitted it, which is
  best-effort: Beam's results carry no submit id, so heavy overlap between cards can
  mispair them. The rig totals never depend on that matching.

> **Multi-GPU has never run on a multi-GPU machine.** It is built and tested — four
> concurrent engines, disjoint nonces, verified against a deliberately broken lane
> assignment — but the development box has one card. If you run it on several, a report
> either way is the single most useful thing you can send us.

### Watchdog

A crashed or hung card is the one failure a rig cannot notice by itself: the process is
alive, the pool connection is up, the console keeps printing, and one card's share of the
hashrate has quietly gone missing. On a single GPU the speed drops to zero and someone
looks; on eight it is a 12 % dip that reads like variance for a week.

```sh
mxbm ... --watchdog              # exit(42) when a card hangs; the default action
mxbm ... --watchdog off          # report it, keep mining on the others
mxbm ... --watchdog script --watchdogscript /usr/local/bin/reset-gpu.sh
```

What it watches is each device's **count of completed solves**, not its hashrate. A rate
cannot tell "hung" from "the 60-second window has not filled yet", and a card that finds
no candidates for a minute is unlucky rather than crashed; a counter that stops advancing
is unambiguous. A device is called hung after **90 seconds** without completing one.

**Idle is not hung.** The clock only runs while a job is present, so a rig waiting on its
first job — or one whose pool has dropped — is never mistaken for a crashed one. That
matters most under the default action: a watchdog that cannot tell those apart restarts
healthy rigs exactly when a restart helps least. A card that recovers on its own is
reported again if it stalls later.

`exit` is the default because it is the only action that actually recovers an NVIDIA
card: a wedged CUDA context generally cannot be torn down by the process that wedged it,
so the fix is to exit with a code a supervisor (systemd, a rig manager, a shell loop)
can act on. **42** is the conventional one, and what lolMiner uses. The script action
passes the device index as its first argument, so one script can serve a whole rig.

### Clocks and fans

`--cclk`/`--mclk` **lock** a clock to a value; `--coff`/`--moff` **shift** its
voltage/frequency curve and may be negative. The pair is the standard undervolt idiom —
lock the clock and raise the offset, so the locked frequency runs at a voltage that would
otherwise deliver less. `--fan` sets a fan target in percent, on every fan the card has.

```sh
sudo mxbm --algo BEAM-III --pool ... --user ... --pl 210 --cclk 2100 --coff 200 --moff 1500
```

All of them take the same per-GPU list syntax as `--pl`, all need root, and all are
restored when MXBM exits unless `--no-oc-reset` says otherwise. Each is clamped to the
band the driver reports for your card and says so when it clamps. On the reference card:

| knob | band the driver permits |
|---|---|
| `--coff` | −1000 … +1000 MHz |
| `--moff` | −2000 … +6000 MHz |
| `--cclk` | up to 3150 MHz |
| `--mclk` | up to 10501 MHz |
| `--fan` | 30 … 100 % |

> **A permitted range is register width, not a recommendation.** `--moff 6000` is
> accepted by the driver and stable on no card in existence. Move one knob at a time and
> watch for the two failure signatures: a core offset that is too high shows up as
> solutions failing CPU verification, while a memory offset that is too high shows up as
> sol/s *falling* while everything still verifies, because GDDR6X answers marginal
> timing with link-level retries rather than with wrong data.

**The memory offset is in MHz of transfer rate, which is twice the memory clock** — so
`--moff 400` asks for +200 MHz of clock. NVML's `nvmlDeviceSetMemClkVfOffset` follows
`nvidia-settings`' `GPUMemoryTransferRateOffset` convention here; measured under load,
where the clock landed on `base + offset/2` for every offset tried. MXBM reports the
offset in the units you typed and the resulting clock in the statistics block, so the two
together are unambiguous whatever your driver does with them. See
[overclocking.md](overclocking.md), which also has the reason to reach for this knob
before `--mclk`.

Restore order is the reverse of apply order: the fan goes back to the driver's curve
first, so the card is cooling itself normally while the clocks come down, and the power
limit is put back last, so nothing is ever unlocked into a clock the old limit would not
have allowed.

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
