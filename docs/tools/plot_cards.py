#!/usr/bin/env python3
"""Draw every card MXBM has been measured on, against the cap it was given.

    python3 docs/tools/plot_cards.py     -> docs/tools/cards-curve.svg

Three cards from two tables in docs/performance.md: the reference 4070 Ti SUPER
from the head-to-head section, and the contributed 4070 SUPER and 3060 Ti from
the third-party section. Generated from the tables so the picture cannot drift
from the numbers.

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
TP_SECTION = "## Third-party hardware — a two-card rig, 2026-08-02"

W, H = 1000, 640
L, R = 74, 34
A_TOP, A_BOT = 118, 350            # panel A: sol/s
B_TOP, B_BOT = 400, 552            # panel B: sol/s per watt

XTICKS = [100, 120, 140, 160, 180, 200, 220, 240, 260, 285]

# By entity. The rung is the 4070 SUPER at another memory clock, so it shares
# that card's hue and is separated by the dash instead of by a fourth colour.
COLOUR = {
    "RTX 4070 Ti SUPER": cl.SERIES[0],
    "RTX 4070 SUPER": cl.SERIES[1],
    "RTX 3060 Ti": cl.SERIES[2],
}
RUNG = "RTX 4070 SUPER @ 5001 MHz"


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


def render(series):
    c = cl.Canvas(W, H)
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
    c.text(L, 28, "On stock memory every card peaks for efficiency well below its stock "
                  "cap -- at %s" % ", ".join("%g W" % p["cap"] for _, p in peaks),
           15, cl.INK, weight="bold")
    c.text(L, 50, "NOT a controlled comparison. The 4070 Ti SUPER was swept on Linux with "
                  "caps set by nvidia-smi and power read from NVML; the other two by MXBM's "
                  "own --tune on", 11, cl.INK_2)
    c.text(L, 65, "Windows, in another machine with different cooling. Cross-card distances "
                  "carry all of that. What survives it is each curve's SHAPE and where its "
                  "own peak sits.", 11, cl.INK_2)
    c.text(L, 80, "sol/s is CPU-verified solutions in every series. x is the cap the card "
                  "was given, not what it drew.", 11, cl.INK_2)

    for p, key, rng, fmt, nt, name in (
            (pa, "sol", None, "%g", 6, "sol/s"),
            (pb, "eff", None, "%.2f", 5, "sol/s per watt")):
        lo, hi = (0, max(r["sol"] for r in every) + 4) if key == "sol" else \
                 (min(r["eff"] for r in every) - 0.012, max(r["eff"] for r in every) + 0.012)
        p.grid_y(cl.ticks(lo, hi, log=False, n=nt), fmt)
        p.frame_y()
        p.axis_x(XTICKS, "%g W", label=(key == "eff"))
        c.text(L, p.y0 - 10, name, 11, cl.INK_2)
        for name_, rows in series.items():
            colour = COLOUR.get(name_, COLOUR["RTX 4070 SUPER"])
            dash = "6 4" if name_ == RUNG else None
            pts = [(r["cap"], r[key]) for r in rows]
            p.c.polyline([(xs(x), p.ys(y)) for x, y in pts], colour, dash=dash)
            for i, (x, y) in enumerate(pts):
                p.c.marker(xs(x), p.ys(y), colour, 3.5 if dash else 4, cl.SURFACE,
                           "%s, cap %g W: %g %s (drew %g W)"
                           % (name_, x, y, name, rows[i]["drew"]))

    # Each card's own peak, marked in the panel that shows it. The reader's
    # question is "where do I cap THIS card", and that is a per-curve answer.
    # The rung is annotated too: it is the 4070 SUPER's real optimum, and a
    # title about stock memory would otherwise bury the better number.
    for name_, pk in peaks + ([(RUNG, peak(series[RUNG]))] if RUNG in series else []):
        c.text(xs(pk["cap"]), ys_b(pk["eff"]) - 11, "%.4f" % pk["eff"], 10,
               COLOUR.get(name_, COLOUR["RTX 4070 SUPER"]), "middle", weight="bold")

    lx, ly = L + 14, A_TOP + 18
    order = [n for n in COLOUR if n in series] + ([RUNG] if RUNG in series else [])
    for i, nm in enumerate(order):
        colour = COLOUR.get(nm, COLOUR["RTX 4070 SUPER"])
        dash = "6 4" if nm == RUNG else None
        c.line(lx, ly - 4 + i * 16, lx + 22, ly - 4 + i * 16, colour, 2, dash)
        c.marker(lx + 11, ly - 4 + i * 16, colour, 3.5 if dash else 4, cl.SURFACE)
        rows = series[nm]
        c.text(lx + 30, ly + i * 16, "%s  (%g-%g W)" % (nm, rows[0]["cap"], rows[-1]["cap"]),
               11, cl.INK_2)

    c.text(L, B_BOT + 46, "The dashed trace is the SAME 4070 SUPER at its 5001 MHz memory "
                          "rung: flat from 172 W down to 120 W, because the memory system "
                          "binds and the cap does not.", 10, cl.MUTED)
    c.text(L, B_BOT + 62, "Generated from the tables in docs/performance.md by "
                          "docs/tools/plot_cards.py. Contribute a card with "
                          "benchmarks/collect_report.sh.", 10, cl.MUTED)
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
