#pragma once

namespace mxbm { namespace api {

// The page served at "/" on --apiport: polls /summary, tabulates the same
// numbers the console stats block prints, and draws hover-readable charts.
//
// SELF-CONTAINED ON PURPOSE: rigs often sit on firewalled networks, so no CDN,
// no external font, and the charts are hand-drawn on a <canvas>.
// tests/test_api.cpp asserts the served HTML contains no absolute URL.
//
// HISTORY IS CLIENT-SIDE. miner::Stats keeps windowed rates and counters, not a
// time series, so there is none to serve: the browser accumulates its own ring
// from page load, and a reload starts it over.
// Three adjacent raw literals, not one: MSVC caps a single literal at 16380
// bytes (C2026) and the concatenated whole at 65535. The split points are
// arbitrary line boundaries; adjacent literals concatenate byte-exactly.
inline const char* dashboard_html() {
    return R"HTML(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MXBM</title>
<style>
  body{font:13px/1.45 monospace;margin:1.2em}
  h1{font-size:15px;margin:0 0 .2em}
  #algo{color:#777;font-weight:normal}
  h2{font-size:13px;font-weight:normal;margin:1.3em 0 .3em;color:#555}
  /* Flex, not inline-block: the tables have different natural widths and
     inline-block dropped the third onto its own line. */
  #tables{display:flex;flex-wrap:wrap;gap:0 1.6em;align-items:flex-start}
  table{border-collapse:collapse;margin-bottom:.4em}
  th,td{border:1px solid #ccc;padding:1px 8px;text-align:left;white-space:nowrap}
  th{font-weight:normal;color:#555}
  td{font-variant-numeric:tabular-nums}
  canvas{border:1px solid #ccc;width:100%;height:22vh;min-height:120px;max-height:260px;
         display:block;cursor:crosshair}
  .key{color:#777;margin:.25em 0 0;font-size:12px}
  .key span{margin-right:1.2em;white-space:nowrap}
  .key i{display:inline-block;width:12px;height:2px;vertical-align:middle;margin-right:4px}
  .ctl{font-size:12px;color:#555;margin:0 0 .3em}
  .ctl input{font:12px monospace;padding:1px 3px;border:1px solid #ccc}
  .ctl input[type=number]{width:6em}
  .ctl input.cur{width:3em;text-align:center}
  #foot{color:#777;margin-top:1.6em;font-size:12px}
  #conn{font-weight:bold}
  #save{font:12px monospace;margin-left:1em;padding:1px 6px;border:1px solid #ccc;
        background:#f6f6f6;cursor:pointer}
  #save[disabled]{color:#aaa;cursor:default}
</style>
</head>
<body>

<h1>MXBM <span id="ver"></span> <span id="algo"></span></h1>
<div>status: <span id="conn">connecting</span>
  <button id="save" disabled>save CSV</button></div>

<h2>session</h2>
<div id="tables">
  <table id="tSession"></table>
  <table id="tPool"></table>
  <table id="tDev"></table>
</div>

<div id="charts"></div>
<div id="foot"></div>

<script>
var MAX = 900, hist = [], fails = 0, lastData = null;
// Found-share events and the current pool target. Unlike `hist`, these are
// replaced from /summary each poll rather than accumulated: the miner keeps the
// share log itself (Stats::kRecentShares), so they survive a reload.
var shares = [], jobDiff = 0;

// Difficulty abbreviated to fit the narrow axis gutter. ONLY the axis uses it:
// legend and hover print n(v, 0), since "143.5k" has thrown the digits away.
function fmtDiff(v) {
  if (v === null || v === undefined || !isFinite(v)) return '--';
  if (v >= 1e6) return (v / 1e6).toFixed(2) + 'M';
  if (v >= 1e3) return (v / 1e3).toFixed(1) + 'k';
  return v.toFixed(0);
}

// Series default to the LEFT axis; `axis:'right'` gives one its own,
// independently scaled axis -- for values that share a canvas but not a range
// (core ~2700 MHz against memory ~10250 MHz; share counts against difficulty).
//
// `altAxis` is not a second series but the SAME line relabelled in another unit:
// cost is watts times a constant, so a separate series would trace the power
// line pixel for pixel. The money needs its own scale, not its own curve.
var CHARTS = [
  { id: 'speed', title: 'hashrate (sol/s)', series: [
      { k: 's15',  label: '15s',  unit: '', color: '#0645ad', dec: 2 },
      { k: 's60',  label: '60s',  unit: '', color: '#a0139b', dec: 2 },
      { k: 'pool', label: 'pool', unit: '', color: '#0a7f5f', dec: 2 }] },

  // Each share at its ACHIEVED difficulty, against the target it had to clear.
  // A scatter, not a line: joining discrete events would draw a trend through
  // independent luck. Log Y because achieved difficulty is unbounded above --
  // one six-figure share on a linear axis flattens the rest onto the floor.
  { id: 'found', title: 'shares found (achieved difficulty)', kind: 'events', log: true },

  { id: 'shares', title: 'share counts', series: [
      { k: 'acc',   label: 'accepted', unit: '', color: '#0645ad', dec: 0 },
      { k: 'stale', label: 'stale',    unit: '', color: '#c25708', dec: 0 },
      { k: 'rej',   label: 'rejected', unit: '', color: '#c00000', dec: 0 },
      { k: 'diff',  label: 'job diff', unit: '', color: '#0a7f5f', dec: 0, axis: 'right' }] },

  { id: 'power', title: 'power (W)',
    controls: '<span class="ctl">energy cost ' +
              '<input class="cur" id="curSym" maxlength="3" value="$"> ' +
              '<input type="number" id="kwhPrice" step="0.001" min="0" placeholder="0.00">' +
              ' / kWh</span>',
    altAxis: function () {
      var p = kwhPrice();
      return p > 0 ? { factor: p / 1000.0, unit: curSym() + '/h', color: '#0a7f5f', dec: 3 } : null;
    },
    series: [
      { k: 'pw', label: 'power', unit: 'W', color: '#c25708', dec: 0 }] },

  { id: 'clocks', title: 'clocks (MHz)', series: [
      { k: 'cclk', label: 'core', unit: 'MHz', color: '#0645ad', dec: 0 },
      { k: 'mclk', label: 'mem',  unit: 'MHz', color: '#c25708', dec: 0, axis: 'right' }] },

  { id: 'temp', title: 'temperature / fan', series: [
      { k: 'temp', label: 'temp', unit: 'C', color: '#c00000', dec: 0 },
      { k: 'fan',  label: 'fan',  unit: '%', color: '#008000', dec: 0 }] }
];
var AXIS_W = 44, PAD_R = 6, PAD_T = 6, PAD_B = 6;

function n(x, d) {
  return (x === null || x === undefined || !isFinite(x)) ? '--' : Number(x).toFixed(d === undefined ? 2 : d);
}
function dur(s) {
  s = Math.floor(s || 0);
  return Math.floor(s / 3600) + 'h ' + Math.floor((s % 3600) / 60) + 'm ' + (s % 60) + 's';
}
// "value unit (min–max)", where the range is the SESSION's, from /summary's
// Session_Stats rather than measured here: the browser ring holds ~30 minutes
// and is wiped by a reload, so an earlier spike is gone from it.
function withRange(v, unit, st, dec) {
  if (v === null || v === undefined || !isFinite(v)) return '--';
  var out = n(v, dec) + unit;
  if (st && st.N >= 2 && st.Max > st.Min) out += ' (' + n(st.Min, dec) + '–' + n(st.Max, dec) + ')';
  return out;
}
function rows(tbl, pairs) {
  var h = '';
  for (var i = 0; i < pairs.length; i++) h += '<tr><th>' + pairs[i][0] + '</th><td>' + pairs[i][1] + '</td></tr>';
  document.getElementById(tbl).innerHTML = h;
}
function clock(d) {
  function p(v) { return (v < 10 ? '0' : '') + v; }
  return p(d.getHours()) + ':' + p(d.getMinutes()) + ':' + p(d.getSeconds());
}
function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;'); }

// Energy price, kept in localStorage so it survives a reload. Access is wrapped
// because a browser with storage disabled throws on it.
function store(k, v) {
  try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch (e) {}
  return null;
}
function kwhPrice() {
  var el = document.getElementById('kwhPrice');
  var v = el ? parseFloat(el.value) : NaN;
  return isFinite(v) && v > 0 ? v : 0;
}
function curSym() {
  var el = document.getElementById('curSym');
  var s = el ? el.value.trim() : '';
  return s || '$';
}

// Energy observed since this page loaded (trapezoid between consecutive
// readings). Only covers time the page has been open, hence the "page" label.
//
// ACCUMULATED as samples arrive, not re-integrated over `hist` on each poll:
// that ring is capped at MAX and drops its oldest sample once full, so a
// re-integration would silently plateau after ~30 minutes.
var energyKWh = 0;
function accrueEnergy(prev, cur) {
  if (!prev) return;
  var a = prev.pw, b = cur.pw;
  if (a == null || b == null || !isFinite(a) || !isFinite(b)) return;
  energyKWh += (a + b) / 2 * ((cur.time - prev.time) / 3600000.0) / 1000.0;
}

// Finite min/max of one series over the window; null when it has no data at all
// (a platform reporting no fan) so the caller can skip the series entirely.
function rangeOf(key) {
  var lo = Infinity, hi = -Infinity, any = false;
  for (var i = 0; i < hist.length; i++) {
    var v = hist[i][key];
    if (v === null || v === undefined || !isFinite(v)) continue;
    any = true;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return any ? { lo: lo, hi: hi } : null;
}
// Never zoom tighter than this fraction of the value plotted. Without a floor
// the axis fits whatever range the data has, so a steady quantity renders as
// violent noise: power moving 0.4% peak-to-peak (283.5-284.6 W) filled the
// whole chart height. Nothing is smoothed; hover still reports exact values.
var MIN_REL_SPAN = 0.06;   // +/-3% around the midpoint

function padRange(lo, hi) {
  if (hi - lo < 1e-9) { hi = lo + 1; lo = Math.max(0, lo - 1); }   // never divide by zero
  var mid = (lo + hi) / 2;
  var floor = Math.abs(mid) * MIN_REL_SPAN;
  if (hi - lo < floor) { lo = mid - floor / 2; hi = mid + floor / 2; }
  var m = (hi - lo) * 0.12;
  return { lo: Math.max(0, lo - m), hi: hi + m };
}
function unionScale(group) {
  var lo = Infinity, hi = -Infinity;
  group.forEach(function (e) { if (e.r.lo < lo) lo = e.r.lo; if (e.r.hi > hi) hi = e.r.hi; });
  return padRange(lo, hi);
}

// Build the chart blocks once, so adding a chart above needs no HTML edit.
CHARTS.forEach(function (cfg) {
  var d = document.createElement('div');
  d.id = 'wrap_' + cfg.id;
  d.hidden = true;
  d.innerHTML = '<h2>' + esc(cfg.title) + '</h2>' +
                (cfg.controls || '') +
                '<canvas id="cv_' + cfg.id + '"></canvas>' +
                '<div class="key" id="key_' + cfg.id + '"></div>';
  document.getElementById('charts').appendChild(d);
  cfg.el = document.getElementById('cv_' + cfg.id);

  cfg.el.addEventListener('mousemove', function (e) {
    var r = cfg.el.getBoundingClientRect();
    // The scatter picks the nearest point by pixel distance; the time-series
    // charts index into the sample ring instead.
    if (cfg.kind === 'events') { cfg.mouseX = e.clientX - r.left; drawChart(cfg); return; }
    if (!hist.length) return;
    var pw = r.width - AXIS_W - (cfg.rightNow ? AXIS_W : PAD_R);
    var count = Math.max(hist.length, 2);
    var i = Math.round((e.clientX - r.left - AXIS_W) / pw * (count - 1));
)HTML" R"HTML(    cfg.hover = Math.max(0, Math.min(hist.length - 1, i));
    drawChart(cfg);
  });
  cfg.el.addEventListener('mouseleave', function () {
    cfg.hover = null; cfg.mouseX = null; drawChart(cfg);
  });
});

// Redraw on every keystroke, so the cost axis appears before the next 2s poll.
(function () {
  var p = document.getElementById('kwhPrice'), c = document.getElementById('curSym');
  if (!p || !c) return;
  var sp = store('mxbm.kwh'), sc = store('mxbm.cur');
  if (sp !== null) p.value = sp;
  if (sc !== null && sc !== '') c.value = sc;
  function changed() {
    store('mxbm.kwh', p.value);
    store('mxbm.cur', c.value);
    // Re-render WITHOUT sampling: a settings edit is not a poll, and pushing a
    // point here would plant a duplicate sample at whatever moment the user typed.
    if (lastData) render(lastData, false);
  }
  p.addEventListener('input', changed);
  c.addEventListener('input', changed);
})();

// Download the sample ring as CSV -- the escape hatch for a history that lives
// only in this tab and drops its oldest sample after ~30 minutes; --log records
// the whole run instead. Both an ISO and an epoch timestamp: spreadsheets want
// the first, scripts the second.
var CSV_COLS = ['s15', 's60', 'pool', 'acc', 'stale', 'rej', 'diff',
                'pw', 'cclk', 'mclk', 'temp', 'fan'];
function saveCsv() {
  if (!hist.length) return;
  var rows = [['iso', 'epoch_ms'].concat(CSV_COLS).join(',')];
  for (var i = 0; i < hist.length; i++) {
    var h = hist[i], row = [h.time.toISOString(), h.time.getTime()];
    for (var c = 0; c < CSV_COLS.length; c++) {
      var v = h[CSV_COLS[c]];
      // Empty, not 0: a card reporting no fan must not read as a stopped one
      // once the numbers are in a spreadsheet.
      row.push((v === null || v === undefined || !isFinite(v)) ? '' : v);
    }
    rows.push(row.join(','));
  }
  var url = URL.createObjectURL(new Blob([rows.join('\n')], { type: 'text/csv' }));
  var a = document.createElement('a');
  a.href = url;
  a.download = 'mxbm_' + new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-') + '.csv';
  a.click();
  URL.revokeObjectURL(url);   // the blob would otherwise be held until the tab closes
}
document.getElementById('save').addEventListener('click', saveCsv);

// Scatter of found shares. Reads `shares`, not the polled ring, so it is
// populated the instant the page opens and survives a reload.
function drawEvents(cfg) {
  var cv = cfg.el;
  // Visibility must be decided BEFORE the canvas is measured: a hidden wrapper
  // has zero width, so a chart unhidden after the size check never draws.
  document.getElementById('wrap_' + cfg.id).hidden = !shares.length;
  if (!shares.length) return;

  var dpr = window.devicePixelRatio || 1;
  var w = cv.clientWidth, h = cv.clientHeight;
  if (!w || !h) return;
  cv.width = w * dpr; cv.height = h * dpr;
  var g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  cfg.rightNow = false;

  var pw = w - AXIS_W - PAD_R, ph = h - PAD_T - PAD_B;

  // The target as it CHANGED, not where it stands now: vardiff steps it through
  // a session, and one line at the current value would make old shares look as
  // if they had cleared today's target. Two sources, so the line spans the whole
  // window: each share carries the target it was found against (reaching back as
  // far as the dots do), and the polled ring fills in up to now.
  var targets = [];
  shares.forEach(function (s) { if (s.target > 0) targets.push({ t: s.t, v: s.target }); });
  for (var ti = 0; ti < hist.length; ti++) {
    var tv = hist[ti].diff;
    if (tv !== null && tv !== undefined && isFinite(tv) && tv > 0) {
      targets.push({ t: hist[ti].time.getTime(), v: tv });
    }
  }
  targets.sort(function (a, b) { return a.t - b.t; });
  targets = targets.filter(function (p, i) { return i === 0 || p.v !== targets[i - 1].v; });

  // Every target joins the vertical range: seeing the shares sit above the bar
  // they had to clear is most of the point of the chart.
  var vals = shares.map(function (s) { return s.units; });
  targets.forEach(function (p) { vals.push(p.v); });
  if (!targets.length && jobDiff > 0) vals.push(jobDiff);
  var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
  if (lo <= 0) lo = 1;
  lo = Math.max(1, lo / 1.6); hi = hi * 1.6;                    // log-space padding
  var lLo = Math.log(lo), lHi = Math.log(hi);
  if (lHi - lLo < 1e-9) lHi = lLo + 1;

  var t1 = Date.now();
  var t0 = shares[0].t;
  if (t1 - t0 < 60000) t0 = t1 - 60000;                          // never squash into a sliver
  function xOf(t) { return AXIS_W + pw * Math.min(1, Math.max(0, (t - t0) / (t1 - t0))); }
  function yOf(v) { return PAD_T + ph * (1 - (Math.log(Math.max(v, 1)) - lLo) / (lHi - lLo)); }

  g.strokeStyle = '#e8e8e8'; g.lineWidth = 1;
  g.fillStyle = '#777'; g.font = '10px monospace'; g.textAlign = 'right';
  for (var i = 0; i <= 3; i++) {
    var y = PAD_T + ph * i / 3, v = Math.exp(lHi - (lHi - lLo) * i / 3);
    g.beginPath(); g.moveTo(AXIS_W, y + 0.5); g.lineTo(w - PAD_R, y + 0.5); g.stroke();
    g.fillText(fmtDiff(v), AXIS_W - 4, y + 3);
  }

  // Pool target, drawn as a STEP: a difficulty holds until the pool issues a new
  // job, so sloping between samples would imply a drift that never happened.
  if (targets.length) {
    g.strokeStyle = '#0a7f5f'; g.setLineDash([4, 3]); g.lineWidth = 1.2;
    g.beginPath();
    var prevY = null;
    for (var pi = 0; pi < targets.length; pi++) {
      var px = xOf(targets[pi].t), py = yOf(targets[pi].v);
      if (prevY === null) { g.moveTo(px, py); }
      else { g.lineTo(px, prevY); g.lineTo(px, py); }
      prevY = py;
    }
    g.lineTo(xOf(t1), prevY);   // hold the latest target out to the right edge
    g.stroke();
    g.setLineDash([]);
  } else if (jobDiff > 0) {
    g.strokeStyle = '#0a7f5f'; g.setLineDash([4, 3]); g.lineWidth = 1;
    g.beginPath(); g.moveTo(AXIS_W, yOf(jobDiff) + 0.5); g.lineTo(w - PAD_R, yOf(jobDiff) + 0.5); g.stroke();
    g.setLineDash([]);
  }

  var hoverIdx = -1, best = 1e9;
  shares.forEach(function (s, i) {
    var x = xOf(s.t);
    if (cfg.mouseX != null) { var d = Math.abs(x - cfg.mouseX); if (d < best) { best = d; hoverIdx = i; } }
  });
  if (best > 14) hoverIdx = -1;   // only latch on when the cursor is genuinely near a point

  shares.forEach(function (s, i) {
    g.fillStyle = s.dev ? '#c084fc' : '#0645ad';
    g.beginPath(); g.arc(xOf(s.t), yOf(s.units), i === hoverIdx ? 4.5 : 2.6, 0, 6.284); g.fill();
  });

  if (hoverIdx >= 0) {
    var s = shares[hoverIdx], hx = xOf(s.t), hy = yOf(s.units);
    g.strokeStyle = '#aaa'; g.lineWidth = 1;
    g.beginPath(); g.moveTo(hx + 0.5, PAD_T); g.lineTo(hx + 0.5, PAD_T + ph); g.stroke();

    var segs = [{ t: clock(new Date(s.t)), c: '#555' },
                { t: 'difficulty ' + n(s.units, 0), c: s.dev ? '#c084fc' : '#0645ad' }];
    if (s.target > 0) {
      segs.push({ t: '(' + (s.units / s.target).toFixed(1) + 'x target of ' + n(s.target, 0) + ')',
                  c: '#0a7f5f' });
    }
    if (s.dev) segs.push({ t: 'dev fee', c: '#c084fc' });

    g.font = '11px monospace';
    var gap = 8, tw = 8;
    segs.forEach(function (q, i) { tw += g.measureText(q.t).width + (i ? gap : 0); });
    var tx = hx + 7;
    if (tx + tw > w - PAD_R) tx = hx - 7 - tw;
    if (tx < AXIS_W) tx = AXIS_W;
    var ty = Math.min(Math.max(hy - 20, PAD_T + 2), PAD_T + ph - 17);
    g.fillStyle = '#fff'; g.fillRect(tx, ty, tw, 15);
    g.strokeStyle = '#bbb'; g.strokeRect(tx + 0.5, ty + 0.5, tw, 15);
    g.textAlign = 'left';
    var cx = tx + 4;
    segs.forEach(function (q, i) {
      if (i) cx += gap;
      g.fillStyle = q.c; g.fillText(q.t, cx, ty + 11); cx += g.measureText(q.t).width;
    });
  }

  var devN = shares.filter(function (s) { return s.dev; }).length;
  var mx = Math.max.apply(null, shares.map(function (s) { return s.units; }));
  var mn = Math.min.apply(null, shares.map(function (s) { return s.units; }));
  var kh = '<span><i style="background:#0645ad"></i>your shares ' + (shares.length - devN) +
           ' (' + n(mn, 0) + '–' + n(mx, 0) + ')</span>';
  if (devN) kh += '<span><i style="background:#c084fc"></i>dev fee ' + devN + '</span>';
  if (targets.length) {
    var tv2 = targets.map(function (p) { return p.v; });
    var tlo = Math.min.apply(null, tv2), thi = Math.max.apply(null, tv2);
    kh += '<span><i style="background:#0a7f5f"></i>job target ' + n(jobDiff || targets[targets.length - 1].v, 0) +
          (thi > tlo ? ' (' + n(tlo, 0) + '–' + n(thi, 0) + ')' : '') + '</span>';
  } else if (jobDiff > 0) {
    kh += '<span><i style="background:#0a7f5f"></i>job target ' + n(jobDiff, 0) + '</span>';
  }
  document.getElementById('key_' + cfg.id).innerHTML = kh;
}

function drawChart(cfg) {
  if (cfg.kind === 'events') { drawEvents(cfg); return; }
  var cv = cfg.el;

  // Which series carry data decides both visibility and scaling. Computed before
  // the canvas is measured, since a hidden wrapper has zero width.
  var live = [];
  cfg.series.forEach(function (s) {
    var r = rangeOf(s.k);
    if (r) live.push({ s: s, r: r });
  });
  document.getElementById('wrap_' + cfg.id).hidden = !live.length;
  if (!live.length) return;

  var dpr = window.devicePixelRatio || 1;
  var w = cv.clientWidth, h = cv.clientHeight;
  if (!w || !h) return;
  cv.width = w * dpr; cv.height = h * dpr;
  var g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);

  var L = live.filter(function (e) { return e.s.axis !== 'right'; });
  var Rt = live.filter(function (e) { return e.s.axis === 'right'; });
  if (!L.length) { L = Rt; Rt = []; }   // right-only series still needs the left gutter

  // A derived right axis only claims the gutter when no real series wants it.
  var alt = (!Rt.length && cfg.altAxis) ? cfg.altAxis() : null;
  cfg.rightNow = (Rt.length > 0) || !!alt;
  var padR = cfg.rightNow ? AXIS_W : PAD_R;
)HTML" R"HTML(  var pw = w - AXIS_W - padR, ph = h - PAD_T - PAD_B;

  var scaleL = unionScale(L);
  L.forEach(function (e) { e.scale = scaleL; });
  var scaleR = Rt.length ? unionScale(Rt) : null;
  Rt.forEach(function (e) { e.scale = scaleR; });

  g.strokeStyle = '#e8e8e8'; g.lineWidth = 1;
  for (var i = 0; i <= 3; i++) {
    var y = PAD_T + ph * i / 3;
    g.beginPath(); g.moveTo(AXIS_W, y + 0.5); g.lineTo(w - padR, y + 0.5); g.stroke();
  }

  g.font = '10px monospace';
  function axisLabels(scale, x, align, colour, factor, dec) {
    g.fillStyle = colour; g.textAlign = align;
    for (var j = 0; j <= 3; j++) {
      var yy = PAD_T + ph * j / 3, vv = scale.hi - (scale.hi - scale.lo) * j / 3;
      if (factor) vv *= factor;
      g.fillText((dec !== undefined) ? vv.toFixed(dec) : vv.toFixed(vv < 10 ? 1 : 0), x, yy + 3);
    }
  }
  // An axis owned by exactly one series takes that series' colour; a shared one
  // stays neutral grey, since no single colour would be truthful.
  axisLabels(scaleL, AXIS_W - 4, 'right', L.length === 1 ? L[0].s.color : '#777');
  if (Rt.length) {
    axisLabels(scaleR, w - padR + 4, 'left', Rt.length === 1 ? Rt[0].s.color : '#777');
  } else if (alt) {
    // Same scale multiplied through, so both axes land on the same gridlines.
    axisLabels(scaleL, w - padR + 4, 'left', alt.color, alt.factor, alt.dec);
  }

  var count = Math.max(hist.length, 2);
  function xOf(i) { return AXIS_W + pw * (count === 1 ? 1 : i / (count - 1)); }
  function yOf(v, scale) { return PAD_T + ph * (1 - (v - scale.lo) / (scale.hi - scale.lo)); }

  live.forEach(function (e) {
    g.strokeStyle = e.s.color; g.lineWidth = 1.4;
    g.beginPath();
    var on = false;
    for (var i = 0; i < hist.length; i++) {
      var v = hist[i][e.s.k];
      if (v === null || v === undefined || !isFinite(v)) { on = false; continue; }
      if (!on) { g.moveTo(xOf(i), yOf(v, e.scale)); on = true; }
      else { g.lineTo(xOf(i), yOf(v, e.scale)); }
    }
    g.stroke();
  });

  // Hover readout. Values come from the stored sample, not from re-reading
  // pixels, so the number shown is the number that was recorded.
  var idx = cfg.hover;
  if (idx !== null && idx !== undefined && idx >= 0 && idx < hist.length) {
    var p = hist[idx], hx = xOf(idx);
    g.strokeStyle = '#aaa'; g.lineWidth = 1;
    g.beginPath(); g.moveTo(hx + 0.5, PAD_T); g.lineTo(hx + 0.5, PAD_T + ph); g.stroke();

    var segs = [{ t: clock(p.time), c: '#555' }];
    live.forEach(function (e) {
      var v = p[e.s.k];
      segs.push({ t: e.s.label + ' ' + n(v, e.s.dec) + e.s.unit, c: e.s.color });
      if (v !== null && v !== undefined && isFinite(v)) {
        g.fillStyle = e.s.color;
        g.beginPath(); g.arc(hx, yOf(v, e.scale), 2.5, 0, 6.284); g.fill();
      }
    });
    // The alt axis reads off the same point, so cost sits next to its watts.
    if (alt) {
      var av = p[L[0].s.k];
      segs.push({ t: n(av == null ? null : av * alt.factor, alt.dec) + alt.unit, c: alt.color });
    }

    g.font = '11px monospace';
    var gap = 8, tw = 8;
    segs.forEach(function (s, i) { tw += g.measureText(s.t).width + (i ? gap : 0); });
    var tx = hx + 7;
    if (tx + tw > w - padR) tx = hx - 7 - tw;   // flip near the right edge
    if (tx < AXIS_W) tx = AXIS_W;

    g.fillStyle = '#fff'; g.fillRect(tx, PAD_T + 2, tw, 15);
    g.strokeStyle = '#bbb'; g.strokeRect(tx + 0.5, PAD_T + 2.5, tw, 15);
    g.textAlign = 'left';
    var cx = tx + 4;
    segs.forEach(function (s, i) {
      if (i) cx += gap;
      g.fillStyle = s.c;
      g.fillText(s.t, cx, PAD_T + 13);
      cx += g.measureText(s.t).width;
    });
  }

  // Legend: current value plus the min-max over the window. The range is what
  // turns a wiggling line into a judgement without hovering for the extremes.
  var kh = '';
  function legend(e, side) {
    var cur = hist.length ? hist[hist.length - 1][e.s.k] : null;
    return '<span><i style="background:' + e.s.color + '"></i>' + esc(e.s.label) + ' ' +
           n(cur, e.s.dec) + esc(e.s.unit) +
           ' (' + n(e.r.lo, e.s.dec) + '–' + n(e.r.hi, e.s.dec) + ')' + side + '</span>';
  }
  var both = Rt.length > 0;
  L.forEach(function (e) { kh += legend(e, both ? ' left' : ''); });
  Rt.forEach(function (e) { kh += legend(e, ' right'); });
  if (alt) {
    var c0 = hist.length ? hist[hist.length - 1][L[0].s.k] : null;
    kh += '<span><i style="background:' + alt.color + '"></i>cost ' +
          n(c0 == null ? null : c0 * alt.factor, alt.dec) + esc(alt.unit) +
          ' (' + n(L[0].r.lo * alt.factor, alt.dec) + '–' +
          n(L[0].r.hi * alt.factor, alt.dec) + ') right</span>';
  }
  document.getElementById('key_' + cfg.id).innerHTML = kh;
}

// push=false re-renders from the last poll without recording a new sample --
// used when a settings edit (energy price) changes derived values.
function render(d, push) {
  lastData = d;
  var s = d.Session || {}, w = (d.Workers && d.Workers[0]) || {}, st = d.Stratum || {}, f = d.DevFee || {};
  // Session-long spread per field, accumulated by the miner (see withRange).
  var ss = d.Session_Stats || {}, s60 = ss.Speed_60s || {};
  document.getElementById('ver').textContent = (d.Software || '').replace('MXBM ', '');
  document.getElementById('algo').textContent = (d.Mining && d.Mining.Algorithm) || '';

  // Sample first, so tables and charts all describe the same poll: the energy
  // accumulator must see this reading before the cost rows are built.
  if (push !== false) {
    var sample = { time: new Date(),
                   s15: s.Speed_15s, s60: s.Speed_60s, pool: s.Pool_Speed_Session,
                   acc: s.Accepted, stale: s.Stale, rej: s.Rejected, diff: s.Job_Difficulty,
                   pw: w.Power_W, cclk: w.Core_Clock_MHz, mclk: w.Mem_Clock_MHz,
                   temp: w.Temp_C, fan: w.Fan_Pct };
    accrueEnergy(hist[hist.length - 1], sample);
    hist.push(sample);
    if (hist.length > MAX) hist.shift();   // rolling window; energyKWh keeps the total
    document.getElementById('save').disabled = false;
  }

  rows('tSession', [
    ['speed 15s', withRange(s.Speed_15s, ' sol/s', ss.Speed_15s, 2)],
    ['speed 60s', withRange(s.Speed_60s, ' sol/s', ss.Speed_60s, 2)],
    // Mean and spread over every 60s window this session -- the figure to
    // quote; the rates above are single draws from a noisy distribution.
    ['60s mean', s60.N >= 2 ? n(s60.Mean) + ' ± ' + n(s60.Stddev) + ' sol/s (n=' + s60.N + ')' : '--'],
    ['session', n(s.Speed_Session) + ' sol/s'],
    ['pool rate', n(s.Pool_Speed_Session) + ' sol/s'],
    ['iterations', n(w.Iterations_s, 1) + ' it/s'],
    ['uptime', s.Uptime_Human || '--']
  ]);
  rows('tPool', [
    ['pool', st.Current_Pool || '--'],
    ['accepted', s.Accepted],
    ['stale', s.Stale],
    ['rejected', s.Rejected],
    ['job id', s.Job_Id || '--'],
    ['job diff', n(s.Job_Difficulty, 0)],
    ['best share', n(s.Best_Share, 0)],
    ['latency', st.Latency_ms >= 0 ? st.Latency_ms + ' ms' : '--'],
    ['reconnects', st.Reconnects]
  ]);

  var dev = [
    ['device', w.Name || '--'],
    ['power', withRange(w.Power_W, ' W', ss.Power_W, 0)],
    ['core clock', withRange(w.Core_Clock_MHz, ' MHz', ss.Core_Clock_MHz, 0)],
    ['mem clock', withRange(w.Mem_Clock_MHz, ' MHz', ss.Mem_Clock_MHz, 0)],
    ['temp', withRange(w.Temp_C, ' C', ss.Temp_C, 0)],
    ['fan', withRange(w.Fan_Pct, ' %', ss.Fan_Pct, 0)]
  ];
  // The per-hour/day/month figures are PROJECTIONS from the current draw; only
  // the "(page)" pair is measured (see accrueEnergy).
  var price = kwhPrice();
  if (price > 0 && w.Power_W != null) {
    var cur = curSym(), perH = w.Power_W / 1000.0 * price, kwh = energyKWh;
    dev.push(['cost /h', cur + n(perH, 3)]);
    dev.push(['cost /day', cur + n(perH * 24, 2)]);
    dev.push(['cost /30d', cur + n(perH * 24 * 30, 2)]);
    dev.push(['energy (page)', n(kwh, 4) + ' kWh']);
    dev.push(['cost (page)', cur + n(kwh * price, 4)]);
  }
  // Only when a fee is actually charged, mirroring the console table: a build
  // with no fee configured shows no fee UI rather than zeroes implying one is
  // taken.
  if (f.Rate > 0) {
    dev.push(['dev fee', (f.Rate * 100).toFixed(4).replace(/\.?0+$/, '') + '%' + (f.Active ? ' (active)' : '')]);
    dev.push(['fee rounds', f.Rounds]);
    dev.push(['fee time', dur(f.Seconds)]);
    dev.push(['fee A/S/R', f.Accepted + '/' + f.Stale + '/' + f.Rejected]);
  }
  rows('tDev', dev);

  // Ages in the share log are relative to when the snapshot was taken, so
  // convert against now.
  jobDiff = (typeof s.Job_Difficulty === 'number') ? s.Job_Difficulty : 0;
  if (Array.isArray(d.Recent_Shares)) {
    var now = Date.now();
    shares = d.Recent_Shares.map(function (e) {
      return { t: now - (e.Age_s || 0) * 1000, units: e.Difficulty,
               target: e.Target || 0, dev: !!e.Dev };
    }).sort(function (a, b) { return a.t - b.t; });
  }

  // Each chart hides or shows itself from its own data.
  CHARTS.forEach(drawChart);

  document.getElementById('foot').textContent =
    'Hover a chart for point values. History is kept in this browser tab only (' +
    hist.length + '/' + MAX + ' samples, 2s apart, oldest dropped once full) and restarts ' +
    'on reload -- the miner reports windowed rates and running counters, not a stored ' +
    'series. Save CSV keeps what this tab holds; --log records the whole run from the ' +
    'start. The ranges in the tables are the exception: the miner accumulates those ' +
    'over the whole session, so they outlive both. Raw JSON at /summary';
}

function tick() {
  fetch('/summary', { cache: 'no-store' })
    .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then(function (d) {
      fails = 0;
      document.getElementById('conn').textContent = 'live';
      document.getElementById('conn').style.color = '#080';
      render(d);
    })
    .catch(function () {
      // Say so rather than freezing on the last good sample: a dashboard that
      // looks alive while the miner is down is worse than one that admits it.
      fails++;
      document.getElementById('conn').textContent = 'no response';
      document.getElementById('conn').style.color = fails > 1 ? '#c00' : '#c60';
    });
}
tick();
setInterval(tick, 2000);
window.addEventListener('resize', function () { CHARTS.forEach(drawChart); });
</script>
</body>
</html>
)HTML";
}

} } // namespace mxbm::api
