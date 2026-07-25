#!/usr/bin/env python3
"""Regenerate docs/progress.svg from the progress table in docs/performance.md.

The chart is generated FROM the table rather than maintained beside it, so the
two cannot drift: add a row to performance.md, re-run this, and the chart
follows. No third-party dependencies -- it writes SVG directly -- so it works on
any machine that can already build MXBM.

    python3 docs/tools/plot_progress.py

Two things the chart has to be honest about, both of which the table encodes:

  * Rows above the "backend switches to CUDA" separator are bench_rounds
    PIPELINE medians; rows below are END-TO-END solve() medians. They are not
    the same measurement, so the switch is drawn as a labelled divider rather
    than smoothed over.

  * The x axis is optimization step, not calendar time. Only three dates exist
    (2026-07-23/24/25) and 16 of the rows fall on one of them, so a literal
    date axis would collapse into three vertical stacks. Dates are drawn as
    labelled bands instead: time still reads left to right, but every step
    stays legible.

Both axes are log-scaled -- the range is ~30x on each -- so equal vertical
distance means equal RATIO, which is what the table's own delta-percent column
tracks.
"""
import math
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO, "docs", "performance.md")
OUT_LOG = os.path.join(REPO, "docs", "progress.svg")
OUT_LIN = os.path.join(REPO, "docs", "progress-linear.svg")

TARGET_SOL = 53.0          # lolMiner, stock, user-measured
W, H = 1000, 460
L, R, T, B = 62, 62, 58, 92   # margins: left/right axes, title, date bands + labels


def num(cell):
    """Pull a number out of a table cell, tolerating **bold**, ~approx and em-dashes."""
    c = cell.replace("*", "").replace("~", "").strip()
    m = re.match(r"^-?\d+(?:\.\d+)?$", c)
    return float(c) if m else None


def num_pm(cell):
    """Parse "56.1 \u00b1 2.3" -> (56.1, 2.3); a bare number -> (v, None).

    Uncertainty lives in the table so a row that HAS a measured spread grows an
    error bar automatically, and a row that does not stays a bare point rather
    than being given a fabricated one."""
    c = cell.replace("*", "").replace("~", "").strip()
    m = re.match(r"^(-?\d+(?:\.\d+)?)\s*(?:\u00b1|\+/-)\s*(\d+(?:\.\d+)?)$", c)
    if m:
        return float(m.group(1)), float(m.group(2))
    return num(cell), None


def parse():
    rows, switch_at = [], None
    for line in open(SRC, encoding="utf-8"):
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5:
            continue
        if "backend switches to CUDA" in line:
            switch_at = len(rows)          # boundary sits before the next row
            continue
        date = cells[0].strip()
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
            continue
        ms = num(cells[3])
        sol, sd = num_pm(cells[4])
        if ms is None or sol is None:
            continue
        rows.append({"date": date, "label": cells[1].replace("*", "").strip(),
                     "ms": ms, "sol": sol, "sd": sd})
    if not rows:
        sys.exit("no data rows parsed from %s -- has the table format changed?" % SRC)
    return rows, switch_at


def make_scale(lo, hi, px0, px1, log=True):
    """Map [lo,hi] onto [px0,px1], logarithmically or linearly."""
    if log:
        a, b = math.log10(lo), math.log10(hi)
        if b - a < 1e-9:
            b = a + 1e-9
        return lambda v: px0 + (math.log10(max(v, 1e-9)) - a) / (b - a) * (px1 - px0)
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    return lambda v: px0 + (v - lo) / (hi - lo) * (px1 - px0)


def ticks(lo, hi, log=True):
    """1-2-5 decade ticks (log) or ~6 round steps (linear) covering [lo,hi]."""
    if log:
        out, d = [], math.floor(math.log10(lo))
        while d <= math.ceil(math.log10(hi)):
            for m in (1, 2, 5):
                v = m * 10 ** d
                if lo * 0.95 <= v <= hi * 1.05:
                    out.append(v)
            d += 1
        return out
    raw = (hi - lo) / 6.0
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    step = next((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), 10 * mag)
    out, v = [], math.ceil(lo / step) * step
    while v <= hi * 1.001:
        out.append(round(v, 6))
        v += step
    return out


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def render(rows, switch_at, log=True):
    n = len(rows)

    sols = [r["sol"] for r in rows]
    mss = [r["ms"] for r in rows]
    tops = [r["sol"] + (r["sd"] or 0.0) for r in rows]
    if log:
        smin, smax = min(sols + [TARGET_SOL]) * 0.85, max(tops + [TARGET_SOL]) * 1.18
        mmin, mmax = min(mss) * 0.85, max(mss) * 1.18
    else:
        # Linear axes start at zero. On a linear scale the eye compares BAR
        # HEIGHT, i.e. absolute value, so a truncated axis would exaggerate
        # every difference -- the classic misleading-chart failure.
        smin, smax = 0.0, max(tops + [TARGET_SOL]) * 1.10
        mmin, mmax = 0.0, max(mss) * 1.10

    x = (lambda i: L + (i / max(n - 1, 1)) * (W - L - R))
    ys = make_scale(smin, smax, H - B, T, log)   # sol/s: higher is better -> up
    ym = make_scale(mmin, mmax, H - B, T, log)   # ms: its own right axis

    o = []
    a = o.append
    a('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
      'viewBox="0 0 %d %d" font-family="DejaVu Sans, Helvetica, Arial, sans-serif">' % (W, H, W, H))
    a('<rect width="%d" height="%d" fill="#ffffff"/>' % (W, H))
    a('<text x="%d" y="26" font-size="15" font-weight="bold" fill="#111">'
      'MXBM GPU solver: throughput and solve time</text>' % L)
    a('<text x="%d" y="44" font-size="11" fill="#666">One point per committed '
      'optimization, in order. %s &#177;1&#963; bars where measured.</text>'
      % (L, 'LOG axes: equal height = equal ratio.' if log
            else 'LINEAR axes from zero: equal height = equal absolute change.'))

    # -- date bands ------------------------------------------------------
    i = 0
    shade = False
    while i < n:
        j = i
        while j + 1 < n and rows[j + 1]["date"] == rows[i]["date"]:
            j += 1
        x0 = x(i) - (x(1) - x(0)) / 2 if n > 1 else L
        x1 = x(j) + (x(1) - x(0)) / 2 if n > 1 else W - R
        x0, x1 = max(x0, L), min(x1, W - R)
        if shade:
            a('<rect x="%.1f" y="%d" width="%.1f" height="%d" fill="#f4f6f8"/>'
              % (x0, T, x1 - x0, H - B - T))
        a('<text x="%.1f" y="%d" font-size="10" fill="#888" text-anchor="middle">%s</text>'
          % ((x0 + x1) / 2, H - B + 46, rows[i]["date"]))
        shade = not shade
        i = j + 1

    # -- grid + axes -----------------------------------------------------
    for v in ticks(smin, smax, log):
        yy = ys(v)
        a('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#e6e6e6" stroke-width="1"/>'
          % (L, yy, W - R, yy))
        a('<text x="%d" y="%.1f" font-size="10" fill="#1f77b4" text-anchor="end">%g</text>'
          % (L - 6, yy + 3, v))
    for v in ticks(mmin, mmax, log):
        yy = ym(v)
        a('<text x="%d" y="%.1f" font-size="10" fill="#d62728" text-anchor="start">%g</text>'
          % (W - R + 6, yy + 3, v))
    a('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#999"/>' % (L, T, L, H - B))
    a('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#999"/>' % (W - R, T, W - R, H - B))
    a('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#999"/>' % (L, H - B, W - R, H - B))
    a('<text x="14" y="%d" font-size="11" fill="#1f77b4" transform="rotate(-90 14 %d)" '
      'text-anchor="middle">sol/s  (higher is better)</text>' % ((T + H - B) / 2, (T + H - B) / 2))
    a('<text x="%d" y="%d" font-size="11" fill="#d62728" transform="rotate(90 %d %d)" '
      'text-anchor="middle">ms per solve  (lower is better)</text>'
      % (W - 12, (T + H - B) / 2, W - 12, (T + H - B) / 2))

    # -- the target lolMiner sets ----------------------------------------
    yt = ys(TARGET_SOL)
    a('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#2ca02c" stroke-width="1.2" '
      'stroke-dasharray="6 4"/>' % (L, yt, W - R, yt))
    a('<text x="%.1f" y="%.1f" font-size="10" fill="#2ca02c" text-anchor="middle">'
      'lolMiner target %g sol/s</text>' % ((L + W - R) / 2, yt + 14, TARGET_SOL))

    # -- measurement-regime switch ---------------------------------------
    if switch_at is not None and 0 < switch_at < n:
        xb = (x(switch_at - 1) + x(switch_at)) / 2
        a('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#999" stroke-width="1.2" '
          'stroke-dasharray="3 3"/>' % (xb, T, xb, H - B))
        for dx, anchor, txt in ((-4, "end", "OpenCL &#183; pipeline median"),
                                (10, "start", "CUDA &#183; end-to-end")):
            a('<text x="%.1f" y="%d" font-size="9" fill="#777" text-anchor="%s" '
              'transform="rotate(-90 %.1f %d)">%s</text>'
              % (xb + dx, H - B - 6, anchor, xb + dx, H - B - 6, txt))

    # -- series ----------------------------------------------------------
    def poly(vals, scale, colour):
        pts = " ".join("%.1f,%.1f" % (x(i), scale(v)) for i, v in enumerate(vals))
        a('<polyline points="%s" fill="none" stroke="%s" stroke-width="2"/>' % (pts, colour))
        for i, v in enumerate(vals):
            a('<circle cx="%.1f" cy="%.1f" r="2.8" fill="%s"><title>%s\n%s</title></circle>'
              % (x(i), scale(v), colour, esc(rows[i]["date"]), esc(rows[i]["label"])))

    # +/-1 sigma range bars, drawn before the markers so the dot sits on top.
    for i, r in enumerate(rows):
        if not r["sd"]:
            continue
        xx, hi, lo = x(i), ys(r["sol"] + r["sd"]), ys(r["sol"] - r["sd"])
        a('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#1f77b4" stroke-width="1.4"/>'
          % (xx, hi, xx, lo))
        for yy in (hi, lo):
            a('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#1f77b4" stroke-width="1.4"/>'
              % (xx - 4, yy, xx + 4, yy))

    poly(mss, ym, "#d62728")
    poly(sols, ys, "#1f77b4")

    # endpoint callouts: the two numbers a reader actually wants
    a('<text x="%.1f" y="%.1f" font-size="11" font-weight="bold" fill="#1f77b4" '
      'text-anchor="end">%.1f sol/s</text>' % (x(n - 1) - 10, ys(sols[-1]) - 11, sols[-1]))
    a('<text x="%.1f" y="%.1f" font-size="11" font-weight="bold" fill="#d62728" '
      'text-anchor="end">%.1f ms</text>' % (x(n - 1) - 10, ym(mss[-1]) - 8, mss[-1]))
    a('<text x="%.1f" y="%.1f" font-size="10" fill="#1f77b4">%.1f sol/s</text>'
      % (x(0) + 7, ys(sols[0]) + 4, sols[0]))
    a('<text x="%.1f" y="%.1f" font-size="10" fill="#d62728">%.0f ms</text>'
      % (x(0) + 7, ym(mss[0]) + 13, mss[0]))

    a('<text x="%d" y="%d" font-size="10" fill="#888">%d optimizations, %s to %s'
      '  &#183;  %.1f &#8594; %.1f sol/s (%.1f&#215;)</text>'
      % (L, H - 12, n, rows[0]["date"], rows[-1]["date"], sols[0], sols[-1], sols[-1] / sols[0]))
    a("</svg>")

    return "\n".join(o) + "\n"


def main():
    rows, switch_at = parse()
    for path, log in ((OUT_LOG, True), (OUT_LIN, False)):
        open(path, "w", encoding="utf-8").write(render(rows, switch_at, log))
        print("wrote %s (%d points, %s scale, switch after #%s)"
              % (path, len(rows), "log" if log else "linear", switch_at))


if __name__ == "__main__":
    main()
