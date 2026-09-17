#!/usr/bin/env python3
"""Draw every card MXBM has been measured on, against the cap it was given.

    python3 docs/tools/plot_cards.py     -> docs/tools/cards-curve.svg

Every card from two tables in docs/performance.md: the reference 4070 Ti SUPER
from the head-to-head section, and the contributed cards from the third-party
section. Generated from the tables so the picture cannot drift from the numbers.

THIS IS NOT A CONTROLLED COMPARISON, and the chart says so on its face. The
reference card was swept on Linux with caps set externally by nvidia-smi and
power read from NVML, interleaved against another miner. The other two were
swept by MXBM's own --tune on Windows, in a different machine, in a different
room, with different cooling. Cross-card DISTANCES here carry all of that; what
survives it is the SHAPE of each curve and where its own efficiency peak sits,
which is what a reader picking a cap actually needs.

WHY CAP AND NOT WATTS DRAWN on x. The cap is the independent variable -- what
the operator sets -- and each card's draw tracks it differently (see the
right-hand end of any of these curves). Plotting against watts drawn would put
cards at different caps at the same x and make every crossing an artefact of the
axis. Same argument as plot_power.py.

THE RUNG SERIES IS THE SAME CARD, so it is the same hue, dashed. It exists
because the 4070 SUPER's 5001 MHz memory rung crosses over 50 W lower than the
reference card's, and a table makes that a number while the picture makes it a
shape: the rung trace goes FLAT from 172 W down to 120 W, which is what a card
that has stopped responding to its power cap looks like.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chartlib as cl                                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(REPO, "docs", "performance.md")
OUT = os.path.join(HERE, "cards-curve.svg")

REF_SECTION = "### Both miners under the same cap"
TP_SECTION = "## Third-party hardware — contributed cards"

W, H = 1000, 640
L, R = 74, 34
A_BOT = 358                        # panel A's floor; its top is derived (LEGEND_Y)
LEGEND_Y = 112                     # first legend row's baseline
LEGEND_COLS = 4                    # fixed columns: font metrics cannot overflow a
                                   # count, and a longer card list just adds a row
B_TOP, B_BOT = 400, 552            # panel B: sol/s per watt

XTICKS = [70, 100, 130, 160, 190, 220, 250, 285, 320, 360, 400]

# By entity. The rung is the 4070 SUPER at another memory clock, so it shares
# that card's hue and is separated by the dash instead of by a fourth colour.
COLOUR = {
    "RTX 4070 Ti SUPER": cl.SERIES[0],
    "RTX 4070 SUPER": cl.SERIES[1],
    "RTX 3060 Ti": cl.SERIES[2],
    "GTX 1660 Ti": cl.SERIES[3],
    "RTX 5080": cl.SERIES[4],
}
RUNG = "RTX 4070 SUPER @ 5001 MHz"


# DejaVu Sans advance widths, averaged per class. Only needed to wrap the legend
# and to keep the inline labels inside the panel, so approximate is enough --
# every use leaves margin for the error.
def text_w(t, size):
    wide = sum(ch in "MWmw@%" for ch in t)
    narrow = sum(ch in "iljI.,:;'| " for ch in t)
    return size * ((len(t) - wide - narrow) * 0.62 + wide * 1.0 + narrow * 0.34)


def free_slot(rect, curves, taken):
    """True when this label box touches no curve and no placed label.

    Segments are sampled rather than clipped: at this scale a 12-point sample of
    a segment cannot step over a box 11 px tall, and the arithmetic stays one
    line instead of a clipper.
    """
    x0, y0, x1, y1 = rect
    for a2, b2, c2, d2 in taken:
        if x0 < c2 and a2 < x1 and y0 < d2 and b2 < y1:
            return False
    for pts in curves:
        for (px, py), (qx, qy) in zip(pts, pts[1:]):
            if max(px, qx) < x0 or min(px, qx) > x1:
                continue
            for k in range(13):
                t = k / 12.0
                sx, sy = px + (qx - px) * t, py + (qy - py) * t
                if x0 <= sx <= x1 and y0 <= sy <= y1:
                    return False
    return True


def place_label(mx, my, w, curves, taken, top, bot, left, right):
    """A clear box for a curve's name, as near its first point as one exists.

    Tried above the point first, then below, stepping out until the box misses
    every curve and every label already placed. Four cards start within a few
    sol/s of each other at 100 W, so nudging them apart is not enough: the names
    have to leave the busy band entirely, and the caller draws a leader when one
    ends up far from its point.
    """
    for side in (-1, 1):
        for step in range(0, 26):
            cy = my + side * (10 + step * 12)
            if cy - 8 < top or cy + 3 > bot:
                continue
            for x in (mx + 6, mx - 6 - w):
                if x - 3 < left or x + w + 3 > right:
                    continue            # a name must not sit in the axis gutter
                r = (x - 3, cy - 8, x + w + 3, cy + 3)
                if free_slot(r, curves, taken):
                    return x, cy, r
    return mx + 6, my - 10, (mx + 3, my - 18, mx + w + 9, my - 7)


def parse_reference():
    """MXBM's own columns from the head-to-head table: cap | sol/s | W | eff."""
    lines = cl.section(cl.read_lines(SRC), REF_SECTION)
    rows = []
    for c in cl.table_rows(lines, lambda c: len(c) >= 4 and cl.num(c[0]) is not None
                                            and cl.num(c[1]) is not None
                                            and cl.num(c[3]) is not None):
        rows.append({"cap": cl.num(c[0]), "sol": cl.num(c[1]),
                     "drew": cl.num(c[2]), "eff": cl.num(c[3])})
    if len(rows) < 5:
        sys.exit("parsed %d reference rows -- has %r changed?" % (len(rows), REF_SECTION))
    rows.sort(key=lambda r: r["cap"])
    return rows


def parse_third_party():
    """The contributed cards, keyed by the bold label that introduces each table.

    Two tables share the section and a third sits under a subheading, so the
    selector is the label above each one rather than a column shape -- all three
    have identical columns and would otherwise merge into one nonsense series.
    """
    lines = cl.section(cl.read_lines(SRC), TP_SECTION)
    out, cur = {}, None
    for line in lines:
        if line.startswith("###"):
            cur = RUNG if "rung" in line.lower() else None
            continue
        for name in COLOUR:
            if "**%s**" % name in line:
                cur = name
        if cur is None or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        # cap W | draw W | sol/s | ms/solve | sol/s per W
        if len(cells) < 5 or cl.num(cells[0]) is None or cl.num(cells[4]) is None:
            continue
        out.setdefault(cur, []).append({"cap": cl.num(cells[0]), "drew": cl.num(cells[1]),
                                        "sol": cl.num(cells[2]), "eff": cl.num(cells[4])})
    missing = [n for n in COLOUR if n != "RTX 4070 Ti SUPER" and n not in out]
    if missing:
        sys.exit("no rows for %s in %r" % (", ".join(missing), TP_SECTION))
    for rows in out.values():
        rows.sort(key=lambda r: r["cap"])
    return out


def peak(rows):
    return max(rows, key=lambda r: r["eff"])


def censored(rows):
    """True when the best efficiency sits at the lowest cap swept.

    Such a curve never turned over inside the driver's band, so its optimum is
    a bound rather than a located peak and the title must not call it one.
    """
    return peak(rows)["cap"] == min(r["cap"] for r in rows)


def render(series):
    c = cl.Canvas(W, H)
    order0 = [n for n in COLOUR if n in series] + ([RUNG] if RUNG in series else [])
    legend_rows = (len(order0) + LEGEND_COLS - 1) // LEGEND_COLS
    A_TOP = LEGEND_Y + legend_rows * 16 + 18
    every = [r for rows in series.values() for r in rows]
    caps = sorted({r["cap"] for r in every})
    xs = cl.make_scale(caps[0] - 10, caps[-1] + 10, L, W - R, log=False)
    ys_a = cl.make_scale(0, max(r["sol"] for r in every) + 4, A_BOT, A_TOP, log=False)
    ys_b = cl.make_scale(min(r["eff"] for r in every) - 0.012,
                         max(r["eff"] for r in every) + 0.012, B_BOT, B_TOP, log=False)
    pa = cl.Panel(c, L, A_TOP, W - R, A_BOT, xs, ys_a)
    pb = cl.Panel(c, L, B_TOP, W - R, B_BOT, xs, ys_b)

    # -- title: the finding, computed, so an edit to the tables moves it ---
    peaks = [(n, peak(series[n])) for n in COLOUR if n in series]
    peaks.sort(key=lambda p: -p[1]["cap"])
    turned = [(n, p) for n, p in peaks if not censored(series[n])]
    flat = [(n, p) for n, p in peaks if censored(series[n])]
    # The per-card peaks are already marked on panel B, so the header does not
    # repeat them; it defines the notation and the two things x and y are not.
    c.text(L, 28, "Speed and efficiency against power cap", 15, cl.INK, weight="bold")
    line2 = "Efficiency peaks below stock cap on every card."
    if flat:
        line2 += "  \u2265 marks a bound: still rising at the lowest cap swept."
    c.text(L, 50, line2, 11, cl.INK_2)
    c.text(L, 66, "Different machines, operating systems and instruments: cross-card gaps "
                  "are not comparable.", 11, cl.INK_2)
    c.text(L, 82, "x: power cap set, not power drawn. sol/s: CPU-verified solutions.",
           11, cl.INK_2)

    for p, key, rng, fmt, nt, name in (
            (pa, "sol", None, "%g", 6, "sol/s"),
            (pb, "eff", None, "%.2f", 5, "sol/s per watt")):
        lo, hi = (0, max(r["sol"] for r in every) + 4) if key == "sol" else \
                 (min(r["eff"] for r in every) - 0.012, max(r["eff"] for r in every) + 0.012)
        p.grid_y(cl.ticks(lo, hi, log=False, n=nt), fmt)
        p.frame_y()
        p.axis_x(XTICKS, "%g W", label=(key == "eff"))
        c.text(L, p.y0 - 10, name, 11, cl.INK_2)
        starts, curves = [], []
        for name_, rows in series.items():
            colour = COLOUR.get(name_, COLOUR["RTX 4070 SUPER"])
            dash = "6 4" if name_ == RUNG else None
            pts = [(xs(r["cap"]), p.ys(r[key])) for r in rows]
            p.c.polyline(pts, colour, dash=dash)
            for i, (px, py) in enumerate(pts):
                p.c.marker(px, py, colour, 3.5 if dash else 4, cl.SURFACE,
                           "%s, cap %g W: %g %s (drew %g W)"
                           % (name_, rows[i]["cap"], rows[i][key], name, rows[i]["drew"]))
            curves.append(pts)
            starts.append((name_, colour, pts[0]))

        # Name each curve where it starts, so a reader tracing one does not have to
        # go back to the legend. Panel A only: panel B is the same curves in the
        # same colours, and naming them twice is noise.
        if key == "sol":
            taken = []
            for name_, colour, (mx, my) in sorted(starts, key=lambda it: it[2][1]):
                label = name_[:-len(" @ 5001 MHz")] + " rung" if name_ == RUNG else name_
                w = text_w(label, 10)
                if mx + 6 + w > W - R:
                    mx = mx - w - 12        # a curve starting near the right edge
                lx_, ly_, r = place_label(mx, my, w, curves, taken, p.y0 + 10, p.y1 - 4,
                                          p.x0, p.x1)
                taken.append(r)
                # A name pushed clear of the busy band needs saying which curve it is.
                if abs(ly_ - my) > 24:
                    p.c.line(lx_ + w / 2.0, ly_ + 2, mx if mx > lx_ else mx,
                             my - 5, colour, 1, "2 2")
                p.c.text(lx_, ly_, label, 10, colour, weight="bold")

    # Each card's own peak, marked in the panel that shows it. The reader's
    # question is "where do I cap THIS card", and that is a per-curve answer.
    # The rung is annotated too: it is the 4070 SUPER's real optimum, and a
    # title about stock memory would otherwise bury the better number.
    # Above the point by default. The 5080's bound sits at 250 W, where the
    # 4070 Ti SUPER's tail runs just above it, so that one label goes below.
    LABEL_DY = {"RTX 5080": 18}
    for name_, pk in peaks + ([(RUNG, peak(series[RUNG]))] if RUNG in series else []):
        # A censored optimum is labelled with a >= so the number is not read as
        # a located peak.
        mark = ("\u2265%.4f" if name_ in series and censored(series[name_])
                else "%.4f") % pk["eff"]
        c.text(xs(pk["cap"]), ys_b(pk["eff"]) + LABEL_DY.get(name_, -11), mark, 10,
               COLOUR.get(name_, COLOUR["RTX 4070 SUPER"]), "middle", weight="bold")

    # Legend above the panels, on a fixed column grid. Measuring text to wrap it
    # was wrong twice -- the estimate undercounts DejaVu and the last entry ran
    # off the canvas -- so the column count decides the layout and the row count
    # decides where panel A starts.
    order = [n for n in COLOUR if n in series] + ([RUNG] if RUNG in series else [])
    pitch = (W - R - L) / float(LEGEND_COLS)
    for i, nm in enumerate(order):
        lx = L + (i % LEGEND_COLS) * pitch
        ly = LEGEND_Y + (i // LEGEND_COLS) * 16
        colour = COLOUR.get(nm, COLOUR["RTX 4070 SUPER"])
        dash = "6 4" if nm == RUNG else None
        c.line(lx, ly - 4, lx + 22, ly - 4, colour, 2, dash)
        c.marker(lx + 11, ly - 4, colour, 3.5 if dash else 4, cl.SURFACE)
        c.text(lx + 30, ly, nm, 11, cl.INK_2)

    c.text(L, B_BOT + 46, "Dashed: the same 4070 SUPER at its 5001 MHz memory rung.",
           10, cl.MUTED)
    c.text(L, B_BOT + 62, "docs/tools/plot_cards.py, from the tables in "
                          "docs/performance.md.", 10, cl.MUTED)
    return c.render()


def main():
    series = {"RTX 4070 Ti SUPER": parse_reference()}
    series.update(parse_third_party())
    open(OUT, "w", encoding="utf-8").write(render(series))
    print("wrote %s (%d series: %s)"
          % (OUT, len(series),
             "; ".join("%s %d pts %g-%g W" % (n, len(r), r[0]["cap"], r[-1]["cap"])
                       for n, r in series.items())))


if __name__ == "__main__":
    main()
