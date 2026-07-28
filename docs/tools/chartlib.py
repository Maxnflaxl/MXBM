#!/usr/bin/env python3
"""Shared chart machinery for the SVGs in docs/.

Every chart in this repo is generated FROM a markdown table in the docs, so a
table and its picture cannot drift: edit the table, re-run the script, the chart
follows. This module holds everything that is not specific to one chart -- table
parsing, scales, ticks, the SVG emitters and the palette -- so a new chart is a
~50-line driver rather than another 300-line copy.

    plot_progress.py   optimization history      -> progress.svg, progress-linear.svg
    plot_power.py      the speed/power curve     -> power-curve.svg

NO THIRD-PARTY DEPENDENCIES, deliberately. It writes SVG text directly, so it
runs anywhere MXBM can be built -- no pip install, no matplotlib, nothing to
break on a mining rig or in CI. The cost is that this file exists; the benefit is
that regenerating a chart is never blocked on an environment.

On the emitters: they take pre-formatted coordinate strings (see `f()`) rather
than formatting numbers themselves, because the exact decimal shape of every
coordinate is what makes a regenerated SVG diffable against the committed one.
A chart whose output churns on every run cannot be reviewed.
"""
import math
import re

# -- palette ---------------------------------------------------------------
# Chart ink and surface. These are design-system tokens rather than ad-hoc
# picks: text NEVER wears a series color (a light hue is illegible as text), so
# labels use the ink tokens below and identity comes from the colored mark
# beside them.
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"     # titles
INK_2     = "#52514e"     # subtitles, annotations
MUTED     = "#898781"     # axis labels, tick text, context marks
GRID      = "#e1e0d9"     # hairline gridlines -- solid, never dashed
AXIS      = "#c3c2b7"     # axis rules

# Categorical slots, in fixed order. Validated as a set for colorblind
# separation; assign by entity, never by rank, and never generate a 9th.
SERIES = ["#2a78d6",      # 1 blue
          "#eb6834",      # 2 orange
          "#1baf7a",      # 3 aqua
          "#eda100"]      # 4 yellow

FONT = "DejaVu Sans, Helvetica, Arial, sans-serif"


def f(v, nd=1):
    """Format a coordinate. Integers stay integral so the SVG reads cleanly."""
    if abs(v - round(v)) < 1e-9:
        return "%d" % round(v)
    return ("%." + str(nd) + "f") % v


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# -- reading the docs ------------------------------------------------------

def read_lines(path):
    return open(path, encoding="utf-8").read().splitlines()


def section(lines, heading):
    """The lines under `heading`, up to the next heading at the same level.

    performance.md has forty-odd tables, so "the table with a W in column 1" is
    not a selector. Naming the section is.
    """
    level = None
    out = []
    for line in lines:
        if level is None:
            if line.strip() == heading:
                level = len(line) - len(line.lstrip("#"))
            continue
        if line.startswith("#") and len(line) - len(line.lstrip("#")) <= level:
            break
        out.append(line)
    if level is None:
        raise LookupError("no section titled %r -- has the doc been retitled?" % heading)
    return out


def table_rows(lines, want):
    """The cell lists of the markdown table rows `want` accepts.

    `want` is a predicate on the already-split cells rather than a column
    index, so a table that grows a column keeps working. Separator rows
    (|---|---|) are dropped.
    """
    out = []
    for line in lines:
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):
            continue
        if want(cells):
            out.append(cells)
    return out


def find(lines, pattern, what):
    """The regex groups of the first match, or a loud failure.

    Some figures a chart needs are stated in prose rather than a table -- the
    reference miner's operating point, the crossing powers. Pulling them out of
    the sentence keeps the chart honest against a doc edit; the alternative is a
    constant in this file that silently goes stale. Missing means STOP, never a
    default: a chart drawn against a fabricated baseline is worse than no chart.
    """
    # Searched against the joined text with runs of whitespace collapsed:
    # markdown wraps prose at whatever column the author stopped typing, so a
    # sentence-level pattern must not care where the line breaks fall.
    blob = re.sub(r"\s+", " ", " ".join(lines))
    m = re.search(pattern, blob)
    if m:
        return m.groups()
    raise LookupError("could not read %s from the doc (pattern %r). The chart is "
                      "generated from the prose, so this means the wording moved "
                      "-- fix the pattern, do not hardcode the number." % (what, pattern))


def num(cell):
    """Pull a number out of a table cell, tolerating **bold**, ~approx, units.

    Returns None when the cell is not a bare number, which is how a driver
    skips heading rows and em-dashes without a separate check.
    """
    c = cell.replace("*", "").replace("~", "").strip()
    c = re.sub(r"\s*(W|MHz|ms|GB/s|GiB|%)$", "", c)
    m = re.match(r"^-?\d+(?:\.\d+)?$", c)
    return float(c) if m else None


def num_pm(cell):
    """Parse "56.1 ± 2.3" -> (56.1, 2.3); a bare number -> (v, None).

    Uncertainty lives in the table so a row that HAS a measured spread grows an
    error bar automatically, and a row that does not stays a bare point rather
    than being given a fabricated one.
    """
    c = cell.replace("*", "").replace("~", "").strip()
    m = re.match(r"^(-?\d+(?:\.\d+)?)\s*(?:±|\+/-)\s*(\d+(?:\.\d+)?)$", c)
    if m:
        return float(m.group(1)), float(m.group(2))
    return num(cell), None


# -- scales ----------------------------------------------------------------

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


def ticks(lo, hi, log=True, n=6):
    """1-2-5 decade ticks (log) or ~n round steps (linear) covering [lo,hi]."""
    if log:
        out, d = [], math.floor(math.log10(lo))
        while d <= math.ceil(math.log10(hi)):
            for m in (1, 2, 5):
                v = m * 10 ** d
                if lo * 0.95 <= v <= hi * 1.05:
                    out.append(v)
            d += 1
        return out
    raw = (hi - lo) / float(n)
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    step = next((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), 10 * mag)
    out, v = [], math.ceil(lo / step) * step
    while v <= hi * 1.001:
        out.append(round(v, 6))
        v += step
    return out


# -- the canvas ------------------------------------------------------------

class Canvas(object):
    """An SVG document under construction. `add()` appends raw markup; the
    helpers below are the vocabulary the drivers actually use."""

    def __init__(self, w, h, font=FONT, surface=SURFACE):
        self.w, self.h = w, h
        self.o = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
                  'viewBox="0 0 %d %d" font-family="%s">' % (w, h, w, h, font)]
        if surface:
            self.o.append('<rect width="%d" height="%d" fill="%s"/>' % (w, h, surface))

    def add(self, markup):
        self.o.append(markup)

    def text(self, x, y, s, size=11, fill=MUTED, anchor=None, weight=None,
             rotate=None, escape=True):
        a = 'x="%s" y="%s" font-size="%g"' % (f(x), f(y), size)
        if weight:
            a += ' font-weight="%s"' % weight
        a += ' fill="%s"' % fill
        if anchor:
            a += ' text-anchor="%s"' % anchor
        if rotate is not None:
            a += ' transform="rotate(%g %s %s)"' % (rotate, f(x), f(y))
        self.add('<text %s>%s</text>' % (a, esc(s) if escape else s))

    def line(self, x1, y1, x2, y2, stroke=GRID, width=1, dash=None):
        a = 'x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="%g"' % (
            f(x1), f(y1), f(x2), f(y2), stroke, width)
        if dash:
            a += ' stroke-dasharray="%s"' % dash
        self.add('<line %s/>' % a)

    def rect(self, x, y, w, h, fill, opacity=None):
        a = 'x="%s" y="%s" width="%s" height="%s" fill="%s"' % (
            f(x), f(y), f(w), f(h), fill)
        if opacity is not None:
            a += ' fill-opacity="%g"' % opacity
        self.add('<rect %s/>' % a)

    def polyline(self, pts, stroke, width=2):
        """A 2px line with round joins -- the mark spec for every series line."""
        s = " ".join("%s,%s" % (f(x), f(y)) for x, y in pts)
        self.add('<polyline points="%s" fill="none" stroke="%s" stroke-width="%g" '
                 'stroke-linejoin="round" stroke-linecap="round"/>' % (s, stroke, width))

    def marker(self, x, y, fill, r=4, ring=SURFACE, title=None):
        """A data point: >=8px across, carrying a 2px ring in the surface color
        so it stays legible where it crosses a line or another marker. The ring
        is a wider circle behind it, not a stroke -- a stroke would add
        data-weight ink that is not data."""
        if ring:
            self.add('<circle cx="%s" cy="%s" r="%g" fill="%s"/>'
                     % (f(x), f(y), r + 2, ring))
        t = '<title>%s</title>' % esc(title) if title else ''
        self.add('<circle cx="%s" cy="%s" r="%g" fill="%s">%s</circle>'
                 % (f(x), f(y), r, fill, t))

    def render(self):
        return "\n".join(self.o + ["</svg>"]) + "\n"


class Panel(object):
    """A plot area on a Canvas: a box, two scales, and the chrome that goes
    with them. Two panels sharing one x scale is how this repo draws two
    measures of different units -- never two y axes on one plot, which makes
    the reader believe an alignment the data does not contain."""

    def __init__(self, canvas, x0, y0, x1, y1, xs, ys):
        self.c, self.xs, self.ys = canvas, xs, ys
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1

    def grid_y(self, values, fmt="%g", label=True):
        for v in values:
            y = self.ys(v)
            self.c.line(self.x0, y, self.x1, y, GRID, 1)
            if label:
                self.c.text(self.x0 - 8, y + 3, fmt % v, 10, MUTED, "end")

    def axis_x(self, values, fmt="%g", label=True, tick=4):
        self.c.line(self.x0, self.y1, self.x1, self.y1, AXIS, 1)
        for v in values:
            x = self.xs(v)
            self.c.line(x, self.y1, x, self.y1 + tick, AXIS, 1)
            if label:
                self.c.text(x, self.y1 + tick + 12, fmt % v, 10, MUTED, "middle")

    def frame_y(self):
        self.c.line(self.x0, self.y0, self.x0, self.y1, AXIS, 1)

    def band(self, xa, xb, fill, opacity=0.09):
        """A shaded x range. Always paired with a text label by the caller --
        a region that means something must say what it means, not rely on the
        reader decoding a color."""
        self.c.rect(self.xs(xa), self.y0, self.xs(xb) - self.xs(xa),
                    self.y1 - self.y0, fill, opacity)

    def hline(self, v, stroke=MUTED, dash="5 4"):
        """A reference/threshold line. Dashed ON PURPOSE: dashing reads as
        'threshold', which is exactly what this is -- and is why gridlines and
        axes above are solid."""
        self.c.line(self.x0, self.ys(v), self.x1, self.ys(v), stroke, 1.2, dash)

    def series(self, pts, color, r=4, titles=None):
        px = [(self.xs(x), self.ys(y)) for x, y in pts]
        self.c.polyline(px, color)
        for i, (x, y) in enumerate(px):
            self.c.marker(x, y, color, r, SURFACE,
                          titles[i] if titles else None)
