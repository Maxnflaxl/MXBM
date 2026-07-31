#!/usr/bin/env python3
"""Draw the memory-rung curves from the table in docs/performance.md.

    python3 docs/tools/plot_mclk.py     -> docs/tools/mclk-curve.svg

Same contract as plot_power.py: one picture generated from one table, so the
chart and the document cannot disagree. The finding this chart exists to show
is a WINDOW, not a point: both miners ride the 5001 rung below their own
crossover, but the crossovers differ by ~47 W because the designs differ in
bytes per solve -- and between them (roughly 126-173 W) the rung pays MXBM
8-11 % while costing lolMiner 14-29 %. The two flat plateaus on the right are
the same physics at two heights: each miner's own DRAM roofline on the reduced
interface, at the altitude its bytes-per-solve dictates.

Panel B is MXBM-only on purpose. J/solution divides watts by each miner's OWN
sol/s definition, and the definitions differ by up to ~17 % (see
benchmarking.md) -- a cross-miner J/solution panel would present that
definitional spread as physics.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chartlib as cl                                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(REPO, "docs", "performance.md")
OUT = os.path.join(HERE, "mclk-curve.svg")
SECTION = ("### Below stock the OTHER rung pays: −8.5 to −14.4 % "
           "under caps below ~173 W, and a new efficiency record")

W, H = 1000, 640
L, R = 74, 34
A_TOP, A_BOT = 118, 380            # panel A: sol/s
B_TOP, B_BOT = 430, 548            # panel B: MXBM J/solution

MX, LOL = cl.SERIES[0], cl.SERIES[1]
MX_X, LOL_X = 173.0, 126.0         # measured crossovers (interpolated)


def parse():
    lines = cl.section(cl.read_lines(SRC), SECTION)

    def watts(cell):
        m = re.match(r"^\**\s*(\d+(?:\.\d+)?)\s*W", cell.replace("*", ""))
        return float(m.group(1)) if m else None

    rows = []
    for c in cl.table_rows(lines, lambda c: len(c) >= 7 and watts(c[0]) is not None
                                            and cl.num(c[1]) is not None):
        rows.append({"cap": watts(c[0]),
                     "mx_b": cl.num(c[1]), "mx_r": cl.num(c[2]),
                     "j_b":  cl.num(c[3]), "j_r":  cl.num(c[4]),
                     "lol_b": cl.num(c[5]), "lol_r": cl.num(c[6])})
    if len(rows) < 4:
        sys.exit("parsed %d rows from %r -- has the table changed?" % (len(rows), SECTION))
    rows.sort(key=lambda r: r["cap"])
    return rows


def render(rows):
    c = cl.Canvas(W, H)
    caps = [r["cap"] for r in rows]
    lo, hi = caps[0] - 12, caps[-1] + 12

    sols = ([r["mx_b"] for r in rows] + [r["mx_r"] for r in rows]
            + [r["lol_b"] for r in rows if r["lol_b"] is not None]
            + [r["lol_r"] for r in rows if r["lol_r"] is not None])
    js = [r["j_b"] for r in rows] + [r["j_r"] for r in rows]
    srange = (min(sols) - 3, max(sols) + 3)
    jrange = (min(js) - 0.25, max(js) + 0.25)

    xs = cl.make_scale(lo, hi, L, W - R, log=False)
    ys_a = cl.make_scale(srange[0], srange[1], A_BOT, A_TOP, log=False)
    ys_b = cl.make_scale(jrange[0], jrange[1], B_BOT, B_TOP, log=False)
    pa = cl.Panel(c, L, A_TOP, W - R, A_BOT, xs, ys_a)
    pb = cl.Panel(c, L, B_TOP, W - R, B_BOT, xs, ys_b)

    # -- title: the finding ------------------------------------------------
    c.text(L, 28, "The 5001 MHz memory rung pays MXBM below ~173 W; lolMiner can only "
                  "follow it to ~126 W", 15, cl.INK, weight="bold")
    c.text(L, 48, "RTX 4070 Ti SUPER, 2026-07-31, one evening: ABBA-bracketed arms per "
                  "cap, memory clock sampled during every arm (no rung refused),", 11, cl.INK_2)
    c.text(L, 63, "MXBM measured in the miner loop with draw from the card's energy "
                  "counter. Each miner's sol/s is its OWN definition; the watts are one "
                  "instrument.", 11, cl.INK_2)
    c.text(L, 78, "The flat right-hand traces are each design's DRAM roofline on the "
                  "reduced interface: 13.0 GB/solve plateaus at 43.6 sol/s, "
                  "17.7 GB/solve at 34.", 11, cl.INK_2)

    # -- the window only a re-derivation design owns -----------------------
    pa.band(LOL_X, MX_X, MX, 0.085)
    for x, nm, colour in ((MX_X, "MXBM crossover ~173 W", MX),
                          (LOL_X, "lolMiner crossover ~126 W", LOL)):
        c.line(xs(x), A_TOP, xs(x), A_BOT, colour, 1, "3 3")
        c.text(xs(x) + 4, A_TOP + 13, nm, 10, cl.INK_2)
    c.text((xs(LOL_X) + xs(MX_X)) / 2, A_BOT - 10,
           "rung pays MXBM, costs lolMiner", 10, cl.INK, "middle", weight="bold")

    # -- panel A: sol/s, four series ---------------------------------------
    pa.grid_y(cl.ticks(srange[0], srange[1], log=False, n=6), "%g")
    pa.frame_y()
    pa.axis_x(caps, "%g", label=False)
    c.text(L, A_TOP - 10, "sol/s (each miner's own basis)", 11, cl.INK_2)
    series = [("lol_b", LOL, None, "lolMiner @10251"),
              ("lol_r", LOL, "6 4", "lolMiner @5001"),
              ("mx_b", MX, None, "MXBM @10251"),
              ("mx_r", MX, "6 4", "MXBM @5001")]
    for key, colour, dash, _ in series:
        pts = [(r["cap"], r[key]) for r in rows if r[key] is not None]
        pa.c.polyline([(xs(x), ys_a(y)) for x, y in pts], colour, dash=dash)
        for x, y in pts:
            pa.c.marker(xs(x), ys_a(y), colour, 3.5 if dash else 4, cl.SURFACE,
                        "cap %g W: %g sol/s" % (x, y))

    # -- panel B: MXBM J/solution (lower is better) ------------------------
    pb.grid_y(cl.ticks(jrange[0], jrange[1], log=False, n=4), "%.1f")
    pb.frame_y()
    pb.axis_x(caps, "%g W")
    c.text(L, B_TOP - 10, "MXBM J/solution, energy counter (LOWER is better; "
                          "MXBM only -- a cross-miner J/solution would divide by "
                          "incomparable bases)", 11, cl.INK_2)
    c.line(xs(MX_X), B_TOP, xs(MX_X), B_BOT, MX, 1, "3 3")
    for key, dash in (("j_b", None), ("j_r", "6 4")):
        pts = [(r["cap"], r[key]) for r in rows]
        pb.c.polyline([(xs(x), ys_b(y)) for x, y in pts], MX, dash=dash)
        for x, y in pts:
            pb.c.marker(xs(x), ys_b(y), MX, 3.5 if dash else 4, cl.SURFACE,
                        "cap %g W: %.2f J/solution" % (x, y))
    best = min(rows, key=lambda r: r["j_r"])
    c.marker(xs(best["cap"]), ys_b(best["j_r"]), MX, 5.5, cl.SURFACE)
    c.text(xs(best["cap"]), ys_b(best["j_r"]) + 20,
           "the card's efficiency record: %.2f J/solution at %g W + 5001"
           % (best["j_r"], best["cap"]), 10, cl.INK, "middle", weight="bold")

    # -- legend ------------------------------------------------------------
    lx, ly = L + 14, A_TOP + 34
    for i, (_, colour, dash, nm) in enumerate(series):
        c.line(lx, ly - 4 + 16 * i, lx + 22, ly - 4 + 16 * i, colour, 2, dash)
        c.marker(lx + 11, ly - 4 + 16 * i, colour, 3.5, cl.SURFACE)
        c.text(lx + 30, ly + 16 * i, nm, 11, cl.INK_2)

    c.text(L, B_BOT + 34, "x is the cap. Solid = stock memory (10251 MHz), dashed = "
                          "the 5001 rung. All y axes are truncated.", 10, cl.MUTED)
    c.text(L, B_BOT + 50, "Generated from the table in docs/performance.md by "
                          "docs/tools/plot_mclk.py -- reproduce with "
                          "docs-internal/rootruns/run_mclk_eco.sh and run_mclk_lol.sh.",
           10, cl.MUTED)
    return c.render()


def main():
    rows = parse()
    open(OUT, "w", encoding="utf-8").write(render(rows))
    lol = sum(1 for r in rows if r["lol_b"] is not None)
    print("wrote %s (%d MXBM caps %g-%g W, %d head-to-head caps)"
          % (OUT, len(rows), rows[0]["cap"], rows[-1]["cap"], lol))


if __name__ == "__main__":
    main()
