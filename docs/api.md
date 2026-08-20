# Dashboard and monitoring API

`--apiport N` serves a live browser dashboard and a JSON endpoint for scripts. This page
is the reference for both; [usage.md](usage.md) covers the flags that turn them on.

`--apiport N` serves two things on the same port:

| Path | |
|------|--|
| `/` (or `/index.html`) | A live dashboard, for a browser |
| `/summary` | A JSON snapshot, for scripts and monitoring |

```sh
mxbm --json --profile rig1 --apiport 8080
```

Then open <http://localhost:8080/>.

## The dashboard

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

## `/summary`

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
  [Speed vs pool rate](usage.md#speed-vs-pool-rate).
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

## Reaching it from another machine

The server binds all interfaces by default, so on a firewalled host you only need
to open the port — scoped to your LAN rather than to everything:

```sh
sudo ufw allow from 192.168.1.0/24 to any port 8080 proto tcp comment 'MXBM dashboard'
```

**The API is unauthenticated**, and it is a small hand-rolled HTTP server, so
treat the port as trusted-network-only. It does not expose your wallet address,
but it does reveal your hardware, hashrate, pool and uptime. MXBM prints the
address it bound at startup, so which of these you are running is never a guess.

`--apihost 127.0.0.1` restricts the listener to the rig itself — no other machine
can reach it, with or without a firewall rule. That plus an SSH tunnel is the
setup that exposes nothing:

```sh
ssh -N -L 8080:localhost:8080 user@rig
```
