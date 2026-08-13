#!/usr/bin/env python3
"""Draw the head-to-head power curves from the table in docs/performance.md.

    python3 docs/tools/plot_power.py     -> docs/tools/power-curve.svg

Used by BOTH docs/performance.md ("Both miners under the same cap") and
docs/benchmarks.md. One picture generated from one table, so the two documents
cannot disagree with each other or with the numbers.

WHY THIS CHART CHANGED. It used to plot MXBM's curve against lolMiner as a
single gray reference point, because lolMiner had only ever been measured
uncapped. That was a measurement gap, not a property of lolMiner: it has --pl
too, and once swept it has a curve like anything else. So both are now series,
and the interesting features are things only a curve-against-curve plot shows --
where they cross, and where lolMiner's stops responding to the cap at all.

WHY TWO PANELS AND NOT TWO Y AXES. sol/s and sol/s/W have no common scale, so
where the curves cross on a shared plot would be a choice of axis alignment
rather than a fact about the cards -- and the crossings are the entire argument.
Stacked panels over one x axis keep every crossing a property of the data.

Both miners' sol/s are their OWN definitions and are not strictly comparable
(MXBM counts CPU-verified solutions; lolMiner's basis is undocumented, and the
spread between "found" and "verified" inside our own pipeline is 17 %). The
watts are one instrument for both. The chart says so on its face rather than in
a footnote, because a reader who misses that caveat misreads everything else.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chartlib as cl                                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(REPO, "docs", "performance.md")
OUT = os.path.join(HERE, "power-curve.svg")
SECTION = "### Both miners under the same cap"
ABOVE = "### Above stock: the curve continues to ~311 W, and the memory rung is unreachable"

W, H = 1000, 720
L, R = 74, 34                      # left axis gutter, right margin
A_TOP, A_BOT = 104, 300            # panel A: sol/s
B_TOP, B_BOT = 352, 486            # panel B: sol/s/W
C_TOP, C_BOT = 538, 624            # panel C: watts actually drawn

MX, LOL = cl.SERIES[0], cl.SERIES[1]        # blue, orange -- validated pair
SAT_TOL = 1.5                               # W: points this close count as one


def parse():
    lines = cl.section(cl.read_lines(SRC), SECTION)

    def watts(cell):
        m = re.match(r"^\**\s*(\d+(?:\.\d+)?)\s*W", cell.replace("*", ""))
        return float(m.group(1)) if m else None

    rows = []
    for c in cl.table_rows(lines, lambda c: len(c) >= 7 and watts(c[0]) is not None
                                            and cl.num(c[1]) is not None):
        rows.append({"cap": watts(c[0]),
                     "mx":  (cl.num(c[2]), cl.num(c[1]), cl.num(c[3])),   # (W, sol/s, eff)
                     "lol": (cl.num(c[5]), cl.num(c[4]), cl.num(c[6]))})
    if len(rows) < 3:
        sys.exit("parsed %d rows from %r -- has the table changed?" % (len(rows), SECTION))
    rows.sort(key=lambda r: r["cap"])
    return rows


def parse_above():
    """MXBM's own curve ABOVE the stock cap, where there is no comparator.

    Kept a separate series rather than extra rows in the head-to-head table for
    two reasons, and both matter more than the convenience of one table. It is a
    DIFFERENT SESSION -- the head-to-head was one interleaved sitting and this is
    another, so it carries the ~5 % cross-session band the doc warns about. And
    lolMiner was never measured here; empty cells in the head-to-head table would
    read as "measured and equal" rather than "not measured", which is the failure
    this file's docstring exists to prevent.

    Returns [] when the section is absent, so removing it from the doc degrades
    the chart to the head-to-head rather than breaking the build.
    """
    try:
        lines = cl.section(cl.read_lines(SRC), ABOVE)
    except LookupError:
        return []

    def watts(cell):
        m = re.match(r"^\**\s*(\d+(?:\.\d+)?)\s*W", cell.replace("*", ""))
        return float(m.group(1)) if m else None

    out = []
    # cap | ms | sol/s | SM clock | drawn W | sol/s per W | vs stock
    for c in cl.table_rows(lines, lambda c: len(c) >= 6 and watts(c[0]) is not None
                                            and cl.num(c[2]) is not None
                                            and cl.num(c[4]) is not None):
        out.append({"cap": watts(c[0]), "sol": cl.num(c[2]),
                    "drew": cl.num(c[4]), "eff": cl.num(c[5])})
    out.sort(key=lambda r: r["cap"])
    return out


def band(rows):
    """The cap range where MXBM leads on speed AND efficiency at once.

    In CAP space, not in watts-drawn space. The cap is the independent variable
    -- it is what both miners were given and what a user sets -- whereas the
    watts drawn are a result, and lolMiner's stop tracking its cap above ~236 W.
    Plotting against watts-drawn would put the two miners at different caps at
    the same x, which makes a crossing an artefact of the axis rather than a
    fact about the miners.

    Found on a fine grid rather than from the first sign change, because
    efficiency crosses TWICE: MXBM starts behind at the low end, goes ahead, and
    falls behind again at the top. Taking the first crossing of each measure
    reported a band of 213-213 W, which is how this was caught.
    """
    caps = [r["cap"] for r in rows]
    lo, hi, step = caps[0], caps[-1], 0.1

    def interp(cap, idx, who):
        for i in range(len(rows) - 1):
            a, b = rows[i], rows[i + 1]
            if a["cap"] <= cap <= b["cap"]:
                t = (cap - a["cap"]) / (b["cap"] - a["cap"])
                return a[who][idx] + t * (b[who][idx] - a[who][idx])
        return rows[-1][who][idx]

    inside = []
    x = lo
    while x <= hi + 1e-9:
        if (interp(x, 1, "mx") > interp(x, 1, "lol")
                and interp(x, 2, "mx") > interp(x, 2, "lol")):
            inside.append(x)
        x += step
    if not inside:
        return None, None
    return inside[0], inside[-1]


def render(rows, above=()):
    c = cl.Canvas(W, H)
    caps = [r["cap"] for r in rows]
    lo = caps[0] - 12
    hi = (max(caps[-1], above[-1]["cap"]) if above else caps[-1]) + 12

    sols = [r["mx"][1] for r in rows] + [r["lol"][1] for r in rows] \
        + [a["sol"] for a in above]
    effs = [r["mx"][2] for r in rows] + [r["lol"][2] for r in rows] \
        + [a["eff"] for a in above]
    watts = [r["mx"][0] for r in rows] + [r["lol"][0] for r in rows] \
        + [a["drew"] for a in above]
    srange = (min(sols) - 3, max(sols) + 2.5)
    erange = (min(effs) - 0.006, max(effs) + 0.010)
    wrange = (min(watts) - 15, max(watts) + 15)

    xs = cl.make_scale(lo, hi, L, W - R, log=False)
    ys_a = cl.make_scale(srange[0], srange[1], A_BOT, A_TOP, log=False)
    ys_b = cl.make_scale(erange[0], erange[1], B_BOT, B_TOP, log=False)
    ys_c = cl.make_scale(wrange[0], wrange[1], C_BOT, C_TOP, log=False)
    pa = cl.Panel(c, L, A_TOP, W - R, A_BOT, xs, ys_a)
    pb = cl.Panel(c, L, B_TOP, W - R, B_BOT, xs, ys_b)
    pc = cl.Panel(c, L, C_TOP, W - R, C_BOT, xs, ys_c)

    b_lo, b_hi = band(rows)

    # -- title: the finding, not the variables ----------------------------
    if b_lo and b_hi:
        head = ("At the same cap, MXBM leads on speed and efficiency between "
                "%.0f W and %.0f W -- and only there" % (b_lo, b_hi))
    else:
        head = "At the same cap: throughput, efficiency and power actually drawn"
    c.text(L, 28, head, 15, cl.INK, weight="bold")
    c.text(L, 48, "RTX 4070 Ti SUPER, one session, runs interleaved and alternating. Caps "
                  "set externally with nvidia-smi so neither miner's own", 11, cl.INK_2)
    c.text(L, 63, "overclock code is a variable; power sampled from NVML for both, never "
                  "from a miner's own stats block. 2 repeats per point.", 11, cl.INK_2)
    c.text(L, 78, "Each miner's sol/s is its OWN definition and the two are not strictly "
                  "comparable. The watts are one instrument for both.", 11, cl.INK_2)

    # -- the band, drawn first so every mark sits on top of it ------------
    if b_lo and b_hi:
        for p in (pa, pb, pc):
            p.band(b_lo, b_hi, MX, 0.085)
        for x in (b_lo, b_hi):
            for p in (pa, pb, pc):
                c.line(xs(x), p.y0, xs(x), p.y1, MX, 1, "3 3")
        # Bottom of the panel, on a surface-colored backing so no gridline or
        # frame rule strikes through the text.
        label = "Ahead on %.0f-%.0f W" % (b_lo, b_hi)
        lx = (xs(b_lo) + xs(b_hi)) / 2
        c.rect(lx - 62, pa.y1 - 24, 124, 16, cl.SURFACE)
        c.text(lx, pa.y1 - 12, label, 11, cl.INK, "middle", weight="bold")

    # -- panels A and B: the two measures ----------------------------------
    for p, idx, rng, fmt, nt, name, akey in (
            (pa, 1, srange, "%g", 6, "sol/s", "sol"),
            (pb, 2, erange, "%.2f", 5, "sol/s per watt", "eff")):
        p.grid_y(cl.ticks(rng[0], rng[1], log=False, n=nt), fmt)
        p.frame_y()
        p.axis_x(caps + [a["cap"] for a in above], "%g", label=False)
        c.text(L, p.y0 - 10, name, 11, cl.INK_2)
        for who, colour in (("lol", LOL), ("mx", MX)):
            pts = [(r["cap"], r[who][idx]) for r in rows]
            p.c.polyline([(xs(x), p.ys(y)) for x, y in pts], colour)
            for i, (x, y) in enumerate(pts):
                p.c.marker(xs(x), p.ys(y), colour, 4, cl.SURFACE,
                           "cap %g W: %g (drew %g W)" % (x, y, rows[i][who][0]))
        # MXBM above stock: same entity, so same hue -- dashed, because it is a
        # different session and has no comparator. It carries its OWN 285 W point
        # rather than joining the head-to-head one, so the small step between the
        # two IS the cross-session spread, shown instead of hidden.
        if above:
            apts = [(a["cap"], a[akey]) for a in above]
            p.c.polyline([(xs(x), p.ys(y)) for x, y in apts], MX, dash="6 4")
            for i, (x, y) in enumerate(apts):
                p.c.marker(xs(x), p.ys(y), MX, 3.5, cl.SURFACE,
                           "cap %g W (MXBM only, later session): %g (drew %g W)"
                           % (x, y, above[i]["drew"]))

    # -- panel C: what each miner actually drew ----------------------------
    # The saturation finding deserves to be SEEN rather than asserted: MXBM
    # tracks the cap, lolMiner's trace goes flat once it stops responding.
    pc.grid_y(cl.ticks(wrange[0], wrange[1], log=False, n=4), "%g")
    pc.frame_y()
    pc.axis_x(caps + [a["cap"] for a in above], "%g W")
    c.text(L, C_TOP - 10, "watts actually drawn", 11, cl.INK_2)
    c.line(xs(max(lo, wrange[0])), ys_c(max(lo, wrange[0])),
           xs(min(hi, wrange[1])), ys_c(min(hi, wrange[1])), cl.GRID, 1)
    # Sits on the y=x reference where nothing else does -- above the traces at
    # the right-hand end, since both miners fall BELOW the line there.
    c.text(xs(caps[-1]) - 6, ys_c(caps[-1]) - 8, "drawing exactly the cap", 9, cl.MUTED, "end")
    for who, colour in (("lol", LOL), ("mx", MX)):
        pts = [(r["cap"], r[who][0]) for r in rows]
        pc.c.polyline([(xs(x), ys_c(y)) for x, y in pts], colour)
        for x, y in pts:
            pc.c.marker(xs(x), ys_c(y), colour, 4, cl.SURFACE, "cap %g W: drew %g W" % (x, y))

    # MXBM above stock, in the panel where the finding lives: its trace goes flat
    # too, just 75 W further right. Both miners saturate; the chart now shows both
    # ceilings instead of asserting one and drawing the other.
    if above:
        apts = [(a["cap"], a["drew"]) for a in above]
        pc.c.polyline([(xs(x), ys_c(y)) for x, y in apts], MX, dash="6 4")
        for x, y in apts:
            pc.c.marker(xs(x), ys_c(y), MX, 3.5, cl.SURFACE,
                        "cap %g W (MXBM only, later session): drew %g W" % (x, y))
        atop = max(a["drew"] for a in above)
        asat = [a for a in above if abs(a["drew"] - atop) < SAT_TOL]
        if len(asat) > 1:
            c.text(W - R - 4, ys_c(atop) - 10,
                   "MXBM stops responding here: %s W all draw ~%.0f W"
                   % ("/".join("%g" % a["cap"] for a in asat), atop), 10, cl.INK_2, "end")

    sat = [r for r in rows if abs(r["lol"][0] - max(r2["lol"][0] for r2 in rows)) < SAT_TOL]
    if len(sat) > 1:
        top = max(r["lol"][0] for r in rows)
        # Right-anchored at the plot edge: the note is about the RIGHT-hand end
        # of the trace, and left-anchoring it there ran the text off the canvas.
        c.text(W - R - 4, ys_c(top) + 20,
               "lolMiner stops responding to the cap here: %s W all draw ~%.0f W"
               % ("/".join("%g" % s["cap"] for s in sat), top), 10, cl.INK_2, "end")

    # -- annotations on efficiency ----------------------------------------
    best = max(rows, key=lambda r: r["lol"][2])
    c.text(xs(best["cap"]), ys_b(best["lol"][2]) - 13,
           "lolMiner's best %.4f" % best["lol"][2], 10, cl.INK_2, "middle")
    bmx = max(rows, key=lambda r: r["mx"][2])
    # Above the point, like lolMiner's: the curve now descends through the space
    # below the peak, and locally the peak clears the other series.
    c.text(xs(bmx["cap"]), ys_b(bmx["mx"][2]) - 13,
           "MXBM's best %.4f" % bmx["mx"][2], 10, cl.INK_2, "middle")

    # -- legend and footer --------------------------------------------------
    lx, ly = L + 14, A_TOP + 17
    legend = [(0, MX, None, "MXBM"), (16, LOL, None, "lolMiner 1.98a")]
    if above:
        legend.append((32, MX, "6 4", "MXBM above stock (later session, no comparator)"))
    for dy, colour, dash, nm in legend:
        c.line(lx, ly - 4 + dy, lx + 22, ly - 4 + dy, colour, 2, dash)
        c.marker(lx + 11, ly - 4 + dy, colour, 3.5 if dash else 4, cl.SURFACE)
        c.text(lx + 30, ly + dy, nm, 11, cl.INK_2)

    c.text(L, C_BOT + 42, "x is the cap both miners were given -- the independent variable. "
                          "What each actually drew is the bottom panel. All y axes are truncated.",
           10, cl.MUTED)
    c.text(L, C_BOT + 58, "Generated from the table in docs/performance.md by "
                          "docs/tools/plot_power.py -- reproduce with "
                          "benchmarks/compare_power.sh.", 10, cl.MUTED)
    return c.render()


def main():
    rows = parse()
    above = parse_above()
    b_lo, b_hi = band(rows)
    open(OUT, "w", encoding="utf-8").write(render(rows, above))
    print("wrote %s (%d head-to-head caps %g-%g W, %d MXBM-only caps to %g W; "
          "MXBM leads both over %s)"
          % (OUT, len(rows), rows[0]["cap"], rows[-1]["cap"], len(above),
             above[-1]["cap"] if above else rows[-1]["cap"],
             "%.1f-%.1f W" % (b_lo, b_hi) if b_lo else "no cap in range"))


if __name__ == "__main__":
    main()
