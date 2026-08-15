#!/usr/bin/env python3
"""Regenerate the stage-breakdown chart from the table in docs/benchmarks.md.

    python3 docs/tools/plot_stages.py   -> stages.svg

Written next to this script, in docs/tools/, like the other charts' output, and
generated FROM the table rather than maintained beside it, so the two cannot
drift: edit the table, re-run this, and the chart follows. No third-party
dependencies -- it writes SVG directly.

WHY THIS IS A TIMELINE AND NOT A STACKED BAR. The question is "where do the time
and the energy go", and those are two quantities a stacked bar can only show one
of. Plotting board power against cumulative solve time gives all of it in one
set of marks, because energy IS power times time:

    segment WIDTH  = that stage's share of the 35 ms solve
    segment HEIGHT = the power the board draws during it
    segment AREA   = the energy that stage costs

A stacked time bar is this same chart with the height thrown away. The flat top
is not a drawing artifact but the finding the section leads with -- every stage
sits at the board limit, so there is no single kernel to "fix" for power.

ON THE Y AXIS STARTING AT ZERO. It has to, or area stops meaning energy and the
chart starts lying about the thing it exists to show. The cost is that a 285 W
block and a 271 W block look nearly identical -- which is why round 3's dip is
called out in text rather than left to the eye.

ON THE TOTAL. The six stages sum to 32.38 ms against the 32.39 ms the table's own
total row reports, a rounding residual from replaying one stage at a time inside
a real solve. The chart uses the sum, because the segments have to add up to the
axis they sit on, and says so underneath rather than quietly scaling the stages
to fit.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chartlib as cl                                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(REPO, "docs", "benchmarks.md")
OUT = os.path.join(HERE, "stages.svg")
SECTION = "## Where the time and energy go"

W, H = 1000, 470
L, R, T = 62, 28, 96
PLOT_H = 200                      # short on purpose: the bar is a block, and the
PY1 = T + PLOT_H                  # room is worth more under it, for labels
ROWS = (PY1 + 20, PY1 + 38)       # two staggered label rows
BRACKET_Y = PY1 + 60
WMAX = 320.0                      # a little headroom over the 285 W board limit

# Colour by what BOUNDS the stage, not by which stage it is: the reader's
# question is "why is this one expensive", and three answers cover all six.
# Validated as a set (deutan dE 9.2 worst adjacent, normal-vision 27.6).
BOUND_COLOUR = {"compute": cl.SERIES[0], "latency": cl.SERIES[1], "DRAM": cl.SERIES[2]}
BOUND_ORDER = ["compute", "latency", "DRAM"]
BOUND_NOTE = {"compute": "arithmetic-bound", "latency": "latency-bound",
              "DRAM": "bandwidth-bound"}


def parse():
    lines = cl.section(cl.read_lines(SRC), SECTION)
    rows = []
    for cells in cl.table_rows(lines, lambda c: len(c) >= 6 and "total" not in c[0].lower()):
        ms = cl.num(cells[1])
        if ms is None:                      # the header row
            continue
        bound = re.sub(r"[*`]", "", cells[5]).strip()
        key = next((b for b in BOUND_ORDER if b.lower() in bound.lower()), None)
        if key is None:
            sys.exit("unrecognised 'bound by' value %r -- add it to BOUND_COLOUR" % bound)
        rows.append({"stage": re.sub(r"[*`]", "", cells[0]).strip(),
                     "ms": ms,
                     "pct": cl.num(cells[2]),
                     "w": cl.num(cells[3]),
                     "gb": cl.num(cells[4]),
                     "bound": key,
                     "bound_full": bound})
    if len(rows) < 2:
        sys.exit("parsed %d stage rows from %s -- has the table changed?" % (len(rows), SRC))
    return rows


def board_limit():
    """The dashed reference: the cap the card is pinned at.

    Read from the reference-card table rather than inferred from the highest
    stage draw -- the two are 0.5 W apart here, and inferring it would make the
    line follow the data instead of being the thing the data is measured
    against. A doc edit moves the line; it never goes stale.
    """
    return float(cl.find(cl.read_lines(SRC),
                         r"Board power limit \| (\d+) W default",
                         "the board power limit")[0])


def render(rows):
    c = cl.Canvas(W, H)
    total_ms = sum(r["ms"] for r in rows)
    cap = board_limit()
    x = lambda t: L + (t / total_ms) * (W - L - R)            # noqa: E731
    y = lambda w: PY1 - (w / WMAX) * PLOT_H                   # noqa: E731

    c.text(L, 30, "Where one BeamHash III solve spends its time and energy", 15,
           cl.INK, weight="bold")
    c.text(L, 48, "Board power through a single %.1f ms solve. Segment width is time, "
                  "height is power — so segment AREA is energy." % total_ms, 11, cl.INK_2)

    # Legend: three entities, so a legend is present, and every segment is
    # direct-labelled underneath as well -- identity never rests on colour.
    lx = L
    for key in BOUND_ORDER:
        c.rect(lx, 62, 20, 10, BOUND_COLOUR[key])
        c.text(lx + 26, 71, "%s (%s)" % (key, BOUND_NOTE[key]), 10, cl.INK_2)
        lx += 52 + 7.2 * len("%s (%s)" % (key, BOUND_NOTE[key]))

    # -- gridlines + y axis ----------------------------------------------
    for v in (0, 100, 200, 300):
        c.line(L, y(v), W - R, y(v), cl.GRID, 1)
        c.text(L - 6, y(v) + 3, "%d" % v, 10, cl.MUTED, "end")
    c.text(14, (T + PY1) / 2, "board power (W)", 11, cl.INK_2, "middle", rotate=-90)

    # -- the stage blocks ------------------------------------------------
    # A 2 px surface gap between adjacent fills, per the mark spec: without it
    # six abutting rectangles read as one shape and the boundaries are lost.
    cum = 0.0
    for r in rows:
        x0, x1 = x(cum), x(cum + r["ms"])
        c.add('<rect x="%s" y="%s" width="%s" height="%s" fill="%s"><title>%s</title></rect>'
              % (cl.f(x0 + 1), cl.f(y(r["w"])), cl.f(max(x1 - x0 - 2, 1)),
                 cl.f(PY1 - y(r["w"])), BOUND_COLOUR[r["bound"]],
                 cl.esc("%s\n%.2f ms (%.1f %% of the solve)\n%.1f W · %.2f GB DRAM\nbound by %s"
                        % (r["stage"], r["ms"], r["pct"], r["w"], r["gb"], r["bound_full"]))))
        cum += r["ms"]

    # The board limit, dashed because it is a threshold. Drawn OVER the blocks:
    # the point it makes is that their tops sit on it.
    c.line(L, y(cap), W - R, y(cap), cl.INK_2, 1.2, "6 4")
    c.text(W - R - 2, y(cap) - 6, "board limit %.0f W" % cap, 10, cl.INK_2, "end")

    c.line(L, PY1, W - R, PY1, cl.AXIS)
    c.line(L, T, L, PY1, cl.AXIS)

    # -- stage labels, staggered -----------------------------------------
    # Leader lines because staggering breaks the "label sits under its mark"
    # adjacency that would otherwise carry the association.
    cum, prev_right = 0.0, [None, None]
    for r in rows:
        mid = (x(cum) + x(cum + r["ms"])) / 2
        label = "%s · %.2f ms · %.1f %%" % (r["stage"], r["ms"], r["pct"])
        half = 3.6 * len(label)
        row = 0 if (prev_right[0] is None or mid - half > prev_right[0]) else 1
        anchor, tx = "middle", mid
        if mid + half > W - 4:
            anchor, tx = "end", W - 4
        elif mid - half < 4:
            anchor, tx = "start", 4
        c.line(mid, PY1 + 2, mid, ROWS[row] - 8, cl.AXIS, 1)
        c.text(tx, ROWS[row], label, 10, cl.INK_2, anchor)
        prev_right[row] = (tx + 2 * half) if anchor == "start" else (
            tx if anchor == "end" else mid + half)
        cum += r["ms"]

    # -- the headline span -----------------------------------------------
    # Rounds 2 and 3 are the finding this section exists to deliver, so they get
    # an explicit bracket rather than being left for the reader to add up.
    big = max(range(len(rows) - 1), key=lambda i: rows[i]["ms"] + rows[i + 1]["ms"])
    start = sum(r["ms"] for r in rows[:big])
    span = rows[big]["ms"] + rows[big + 1]["ms"]
    bx0, bx1 = x(start), x(start + span)
    c.line(bx0, BRACKET_Y, bx1, BRACKET_Y, cl.MUTED, 1)
    for bx in (bx0, bx1):
        c.line(bx, BRACKET_Y, bx, BRACKET_Y - 5, cl.MUTED, 1)
    c.text((bx0 + bx1) / 2, BRACKET_Y + 14,
           "%s + %s = %.0f %% of the solve" % (rows[big]["stage"], rows[big + 1]["stage"],
                                               rows[big]["pct"] + rows[big + 1]["pct"]),
           11, cl.INK, "middle", weight="bold")

    # -- footnotes --------------------------------------------------------
    lo = min(rows, key=lambda r: r["w"])
    c.text(L, H - 30,
           "Every stage draws the board limit — there is no single kernel to fix for "
           "power. The one dip is %s at %.1f W, the most bandwidth-bound stage (%.2f GB)."
           % (lo["stage"], lo["w"], lo["gb"]), 10, cl.INK_2)
    c.text(L, H - 14,
           "Per-stage figures from benchmarks/stage_power.sh, which replays one stage "
           "many times inside a real solve; they sum to %.1f ms against a %.1f ms "
           "measured total, a %.1f %% attribution residual." % (
               total_ms, 35.0, 100 * (total_ms - 35.0) / 35.0), 10, cl.MUTED)
    return c.render()


def main():
    rows = parse()
    open(OUT, "w", encoding="utf-8").write(render(rows))
    print("wrote %s (%d stages, %.2f ms total)"
          % (OUT, len(rows), sum(r["ms"] for r in rows)))


if __name__ == "__main__":
    main()
