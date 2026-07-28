#!/usr/bin/env python3
"""Regenerate the progress charts from the progress table in docs/performance.md.

    python3 docs/tools/plot_progress.py   -> progress.svg, progress-linear.svg

Both SVGs are written next to this script, in docs/tools/, so the generated
output lives with the generator that produces it; docs/performance.md links to
them from there. Shared machinery -- table parsing, scales, ticks, SVG emitters,
palette -- is in chartlib.py, along with the other charts' drivers.

The chart is generated FROM the table rather than maintained beside it, so the
two cannot drift: add a row to performance.md, re-run this, and the chart
follows. No third-party dependencies -- it writes SVG directly -- so it works on
any machine that can already build MXBM.

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
tracks. A linear-axis twin is written as well, because a ratio chart and an
absolute chart answer different questions.

ON THE TWO Y AXES. sol/s and ms/solve are the SAME measurement inverted, and
this is the one case where a second axis is not the usual lie: no correlation is
being implied between two quantities, because there is only one quantity. It is
still redundant ink -- see the note in TODO.md about dropping the ms axis, which
is a change to what the surrounding prose claims and so is not made here.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chartlib as cl                                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(REPO, "docs", "performance.md")
OUT_LOG = os.path.join(HERE, "progress.svg")
OUT_LIN = os.path.join(HERE, "progress-linear.svg")

TARGET_SOL = 53.0          # lolMiner, stock, user-measured
W, H = 1000, 460
L, R, T, B = 62, 62, 58, 92   # margins: left/right axes, title, date bands + labels

SOL, MS = cl.SERIES[0], cl.SERIES[1]   # blue, orange -- validated as a pair
REF = cl.MUTED                         # the reference miner is context, not a series


def parse():
    rows, switch_at = [], None
    for line in cl.read_lines(SRC):
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5:
            continue
        if "backend switches to CUDA" in line:
            switch_at = len(rows)          # boundary sits before the next row
            continue
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", cells[0].strip()):
            continue
        ms = cl.num(cells[3])
        sol, sd = cl.num_pm(cells[4])
        if ms is None or sol is None:
            continue
        rows.append({"date": cells[0].strip(), "label": cells[1].replace("*", "").strip(),
                     "ms": ms, "sol": sol, "sd": sd})
    if not rows:
        sys.exit("no data rows parsed from %s -- has the table format changed?" % SRC)
    return rows, switch_at


def render(rows, switch_at, log=True):
    n = len(rows)
    c = cl.Canvas(W, H)

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
    ys = cl.make_scale(smin, smax, H - B, T, log)   # sol/s: higher is better -> up
    ym = cl.make_scale(mmin, mmax, H - B, T, log)   # ms: its own right axis

    c.text(L, 26, "MXBM GPU solver: throughput and solve time", 15, cl.INK, weight="bold")
    c.text(L, 44, "One point per committed optimization, in order. %s"
           % ('LOG axes: equal height = equal ratio.' if log
              else 'LINEAR axes from zero: equal height = equal absolute change.'),
           11, cl.INK_2)

    # Legend. The previous revision of this chart had none: it coloured the two
    # AXIS TITLES to match their series, which is the one identity channel a
    # reader cannot use if they cannot separate the two hues -- and the old red
    # and green measured deutan dE 3.9, i.e. indistinguishable. Identity now
    # comes from a labelled swatch, and the axis titles are plain ink.
    lx = W - R - 252
    for dx, colour, name in ((0, SOL, "sol/s (left axis)"),
                             (132, MS, "ms per solve (right axis)")):
        c.line(lx + dx, 40, lx + dx + 20, 40, colour, 2)
        c.marker(lx + dx + 10, 40, colour, 2.8, cl.SURFACE)
        c.text(lx + dx + 26, 44, name, 10, cl.INK_2)

    # -- date bands ------------------------------------------------------
    i, shade = 0, False
    while i < n:
        j = i
        while j + 1 < n and rows[j + 1]["date"] == rows[i]["date"]:
            j += 1
        x0 = x(i) - (x(1) - x(0)) / 2 if n > 1 else L
        x1 = x(j) + (x(1) - x(0)) / 2 if n > 1 else W - R
        x0, x1 = max(x0, L), min(x1, W - R)
        if shade:
            c.rect(x0, T, x1 - x0, H - B - T, "#f4f6f8")
        c.text((x0 + x1) / 2, H - B + 46, rows[i]["date"], 10, cl.MUTED, "middle")
        shade = not shade
        i = j + 1

    # -- grid + axes -----------------------------------------------------
    # Only the sol/s axis carries gridlines: two sets of horizontal rules for
    # two scales would make the chart unreadable, and sol/s is the one the
    # reader is tracking.
    for v in cl.ticks(smin, smax, log):
        yy = ys(v)
        c.line(L, yy, W - R, yy, cl.GRID, 1)
        c.text(L - 6, yy + 3, "%g" % v, 10, cl.MUTED, "end")
    for v in cl.ticks(mmin, mmax, log):
        c.text(W - R + 6, ym(v) + 3, "%g" % v, 10, cl.MUTED, "start")
    c.line(L, T, L, H - B, cl.AXIS)
    c.line(W - R, T, W - R, H - B, cl.AXIS)
    c.line(L, H - B, W - R, H - B, cl.AXIS)
    c.text(14, (T + H - B) / 2, "sol/s  (higher is better)", 11, cl.INK_2, "middle",
           rotate=-90)
    c.text(W - 12, (T + H - B) / 2, "ms per solve  (lower is better)", 11, cl.INK_2,
           "middle", rotate=90)

    # -- the target lolMiner sets ----------------------------------------
    # Dashed and gray: a threshold, and context rather than a series -- the same
    # treatment the reference miner gets in power-curve.svg.
    yt = ys(TARGET_SOL)
    c.line(L, yt, W - R, yt, REF, 1.2, "6 4")
    c.text((L + W - R) / 2, yt + 14, "lolMiner target %g sol/s" % TARGET_SOL, 10,
           cl.INK_2, "middle")

    # -- measurement-regime switch ---------------------------------------
    if switch_at is not None and 0 < switch_at < n:
        xb = (x(switch_at - 1) + x(switch_at)) / 2
        c.line(xb, T, xb, H - B, cl.AXIS, 1.2, "3 3")
        # Both read bottom-to-top from just inside the plot floor, so both need
        # text-anchor="start": under rotate(-90) an "end" anchor makes the label run
        # DOWNWARD out of the plot and off the canvas. The glyphs extend toward -x from
        # the baseline, so the right-hand label's baseline is offset by the cap height
        # as well as the gap to put its body clear of the divider on the other side.
        GAP, CAP = 4, 9
        for dx, txt in ((-GAP, "OpenCL · pipeline median"),
                        (GAP + CAP, "CUDA · end-to-end")):
            c.text(xb + dx, H - B - 6, txt, 9, cl.MUTED, "start", rotate=-90)

    # -- series ----------------------------------------------------------
    # +/-1 sigma range bars, drawn before the markers so the dot sits on top.
    for i, r in enumerate(rows):
        if not r["sd"]:
            continue
        xx, hi, lo = x(i), ys(r["sol"] + r["sd"]), ys(r["sol"] - r["sd"])
        c.line(xx, hi, xx, lo, SOL, 1.4)
        for yy in (hi, lo):
            c.line(xx - 4, yy, xx + 4, yy, SOL, 1.4)

    for vals, scale, colour in ((mss, ym, MS), (sols, ys, SOL)):
        c.polyline([(x(i), scale(v)) for i, v in enumerate(vals)], colour)
        for i, v in enumerate(vals):
            c.marker(x(i), scale(v), colour, 2.8, cl.SURFACE,
                     "%s\n%s" % (rows[i]["date"], rows[i]["label"]))

    # endpoint callouts: the two numbers a reader actually wants
    c.text(x(n - 1) - 10, ys(sols[-1]) - 11, "%.1f sol/s" % sols[-1], 11, cl.INK,
           "end", weight="bold")
    c.text(x(n - 1) - 10, ym(mss[-1]) - 8, "%.1f ms" % mss[-1], 11, cl.INK,
           "end", weight="bold")
    c.text(x(0) + 7, ys(sols[0]) + 4, "%.1f sol/s" % sols[0], 10, cl.INK_2)
    c.text(x(0) + 7, ym(mss[0]) + 13, "%.0f ms" % mss[0], 10, cl.INK_2)

    c.text(L, H - 12, "%d optimizations, %s to %s  ·  %.1f → %.1f sol/s (%.1fx)  ·  "
                      "error bars are ±1σ, drawn only where a spread was measured"
           % (n, rows[0]["date"], rows[-1]["date"], sols[0], sols[-1], sols[-1] / sols[0]),
           10, cl.MUTED)
    return c.render()


def main():
    rows, switch_at = parse()
    for path, log in ((OUT_LOG, True), (OUT_LIN, False)):
        open(path, "w", encoding="utf-8").write(render(rows, switch_at, log))
        print("wrote %s (%d points, %s scale, switch after #%s)"
              % (path, len(rows), "log" if log else "linear", switch_at))


if __name__ == "__main__":
    main()
