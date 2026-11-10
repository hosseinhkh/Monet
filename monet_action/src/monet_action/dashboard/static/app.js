/* Monet-Action monitor: vanilla JS, no dependencies, talks only to the server that served it.
 * All untrusted text (run names, log lines, config values) is inserted with textContent. */
(function () {
  'use strict';

  // ================================================================== state & storage
  const params = new URLSearchParams(location.search);
  const TOKEN = params.get('token');
  const S = {
    cfg: null, refreshSec: 10, auto: true, timer: null, view: null, routeKey: null,
    lastOk: null, busy: false, series: {}, compareSlots: new Map(), charts: new Set(), theme: 'auto',
  };
  const store = {
    get(k, d) {
      try { const v = localStorage.getItem('monet-dash:' + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; }
    },
    set(k, v) { try { localStorage.setItem('monet-dash:' + k, JSON.stringify(v)); } catch (e) { /* storage unavailable */ } },
  };

  // ================================================================== DOM helpers
  function append(el, kids) {
    for (const k of [kids].flat(Infinity)) {
      if (k === null || k === undefined || k === false) continue;
      el.appendChild(k instanceof Node ? k : document.createTextNode(String(k)));
    }
    return el;
  }
  /** Replace an element's children; accepts nested arrays and skips null / false (unlike replaceChildren). */
  function setKids(el, ...kids) { el.replaceChildren(); return append(el, kids); }
  function h(tag, props, ...kids) {
    const el = document.createElement(tag);
    if (props) {
      for (const [k, v] of Object.entries(props)) {
        if (v === null || v === undefined || v === false) continue;
        if (k === 'class') el.className = v;
        else if (k === 'text') el.textContent = v;
        else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
        else if (k === 'style') Object.assign(el.style, v); // CSSOM: allowed by the page CSP
        else if (v === true) el.setAttribute(k, '');
        else el.setAttribute(k, String(v));
      }
    }
    return append(el, kids);
  }
  const SVGNS = 'http://www.w3.org/2000/svg';
  function sv(tag, attrs, ...kids) {
    const el = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs || {})) if (v !== null && v !== undefined) el.setAttribute(k, String(v));
    for (const k of kids.flat()) if (k !== null && k !== undefined) el.appendChild(k instanceof Node ? k : document.createTextNode(String(k)));
    return el;
  }
  function debounce(fn, ms) { let t = null; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }

  // ================================================================== URLs & API
  function withToken(u) { return TOKEN ? u + (u.indexOf('?') >= 0 ? '&' : '?') + 'token=' + encodeURIComponent(TOKEN) : u; }
  function encPath(p) { return String(p).split('/').map(encodeURIComponent).join('/'); }
  function fileUrl(root, base, rel, download) {
    let u = 'files/' + root + '/' + (base ? encPath(base) + '/' : '') + encPath(rel);
    if (download) u += '?download=1';
    return withToken(u);
  }
  async function api(path) {
    const r = await fetch(withToken(path), { cache: 'no-store', credentials: 'same-origin' });
    let body = null;
    try { body = await r.json(); } catch (e) { body = null; }
    if (!r.ok) {
      const err = new Error((body && body.error) || ('HTTP ' + r.status));
      err.status = r.status;
      throw err;
    }
    return body;
  }
  async function fetchText(url) {
    const r = await fetch(url, { cache: 'no-store', credentials: 'same-origin' });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    return r.text();
  }

  // ================================================================== formatting
  const isNum = (x) => typeof x === 'number' && isFinite(x);
  function fmtNum(x, sig) {
    if (x === 'Infinity') return '∞';
    if (x === '-Infinity') return '−∞';
    if (!isNum(x)) return '–';
    const a = Math.abs(x);
    if (a !== 0 && (a >= 1e6 || a < 1e-3)) return x.toExponential(2);
    if (Number.isInteger(x)) return x.toLocaleString();
    return String(Number(x.toPrecision(sig || 4)));
  }
  function fmtInt(x) { return isNum(x) ? Math.round(x).toLocaleString() : '–'; }
  function fmtPct(x, d) { return isNum(x) ? (x * 100).toFixed(d === undefined ? 1 : d) + '%' : '–'; }
  function fmtCount(n) {
    if (!isNum(n)) return '–';
    const a = Math.abs(n);
    if (a >= 1e9) return (n / 1e9).toFixed(2) + 'B';
    if (a >= 1e6) return (n / 1e6).toFixed(2) + 'M';
    if (a >= 1e3) return (n / 1e3).toFixed(1) + 'K';
    return String(n);
  }
  function fmtBytes(n) {
    if (!isNum(n)) return '–';
    const u = ['B', 'KB', 'MB', 'GB', 'TB'];
    let i = 0;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return (i === 0 ? n : n.toFixed(n >= 100 ? 0 : 1)) + ' ' + u[i];
  }
  function fmtDuration(s) {
    if (!isNum(s) || s < 0) return '–';
    if (s < 60) return Math.round(s) + ' s';
    if (s < 3600) return Math.round(s / 60) + ' min';
    if (s < 86400) { const hr = Math.floor(s / 3600); return hr + ' h ' + String(Math.round((s % 3600) / 60)).padStart(2, '0') + ' min'; }
    return Math.floor(s / 86400) + ' d ' + Math.round((s % 86400) / 3600) + ' h';
  }
  function fmtAgo(ts) {
    if (!isNum(ts)) return '–';
    const s = Date.now() / 1000 - ts;
    if (s < 5) return 'just now';
    return fmtDuration(s) + ' ago';
  }
  function fmtTime(ts) { return isNum(ts) ? new Date(ts * 1000).toLocaleString() : '–'; }
  function fmtVal(v) {
    if (v === null || v === undefined || v === '') return '–';
    if (typeof v === 'number') return fmtNum(v);
    if (typeof v === 'boolean') return v ? 'yes' : 'no';
    if (typeof v === 'object') return JSON.stringify(v);
    return String(v);
  }

  const STATES = {
    running: ['●', 'Running', 'good pulse'],
    starting: ['◌', 'Starting', 'neutral'],
    stale: ['⚠', 'Stale', 'warning'],
    finished: ['✓', 'Finished', 'neutral'],
    failed: ['✕', 'Failed', 'critical'],
    stopped: ['■', 'Stopped', 'muted'],
    unknown: ['?', 'Unknown', 'muted'],
  };
  function badge(state, inferred) {
    const m = STATES[state] || ['?', state || 'Unknown', 'muted'];
    const title = state === 'stale' ? 'status says running but no update for > 10 min'
      : (inferred ? 'inferred from the log files (no status.json)' : null);
    return h('span', { class: 'badge ' + m[2], title },
      h('span', { class: 'ic', 'aria-hidden': 'true', text: m[0] }), m[1] + (inferred ? ' *' : ''));
  }
  function progress(step, max) {
    if (!isNum(step) || !isNum(max) || max <= 0) {
      return h('span', { class: 'muted small', text: isNum(step) ? 'step ' + fmtInt(step) : '–' });
    }
    const f = Math.max(0, Math.min(1, step / max));
    return h('div', { class: 'progress', title: fmtPct(f) },
      h('div', { class: 'bar' }, h('span', { style: { width: (f * 100).toFixed(1) + '%' } })),
      h('span', { class: 'lbl num', text: fmtInt(step) + ' / ' + fmtInt(max) }));
  }
  function emptyState(title, hint) { return h('div', { class: 'empty' }, h('strong', { text: title }), hint ? h('span', { text: hint }) : null); }
  function card(title, ...kids) { return h('div', { class: 'card' }, title ? h('h2', { text: title }) : null, kids); }
  function kvTable(rows) {
    return h('table', { class: 'kv' }, h('tbody', null, rows.map(([k, v]) =>
      h('tr', null, h('td', { text: k }), h('td', { class: 'v' }, v instanceof Node ? v : fmtVal(v))))));
  }
  function tableEl(headers, rows, opts) {
    opts = opts || {};
    const thead = h('thead', null, h('tr', null, headers.map((c) => h('th', { class: c.r ? 'r' : null, text: c.t || c }))));
    const tbody = h('tbody', null, rows);
    return h('div', { class: 'table-wrap' }, h('table', { class: opts.compact ? 'compact' : null }, thead, tbody));
  }
  function td(content, cls) { return h('td', { class: cls || null }, content instanceof Node || Array.isArray(content) ? content : fmtVal(content)); }
  function setBanner(msg) {
    const b = document.getElementById('banner');
    if (msg) { b.textContent = msg; b.classList.remove('hidden'); } else b.classList.add('hidden');
  }

  // ================================================================== charts
  function niceStep(span, n) {
    const raw = span / Math.max(1, n);
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    const f = raw / p;
    return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p;
  }
  function linTicks(lo, hi, n) {
    const st = niceStep(hi - lo || 1, n);
    const out = [];
    for (let v = Math.ceil(lo / st - 1e-9) * st; v <= hi + st * 1e-9; v += st) out.push(Number(v.toPrecision(12)));
    return { ticks: out, step: st };
  }
  function fmtTick(v, step) {
    if (v === 0) return '0';
    const a = Math.abs(v);
    if (a >= 1e6) return (v / 1e6).toFixed(a >= 1e7 ? 0 : 1) + 'M';
    if (a >= 1e4) return (v / 1e3).toFixed(a >= 1e5 ? 0 : 1) + 'k';
    if (a < 1e-3) {  // enough mantissa digits that neighbouring ticks never print identically
      const digits = step ? Math.max(0, Math.min(4, Math.ceil(Math.log10(a / step) - 1e-9))) : 0;
      return v.toExponential(digits).replace(/\.?0+e/, 'e');
    }
    const dec = Math.max(0, Math.min(6, -Math.floor(Math.log10(step || a))));
    return v.toFixed(dec);
  }
  function decimate(pts, buckets) {
    if (pts.length <= buckets * 2) return pts;
    const out = [];
    const per = pts.length / buckets;
    for (let b = 0; b < buckets; b++) {
      const lo = Math.floor(b * per), hi = Math.min(pts.length, Math.floor((b + 1) * per));
      let mn = null, mx = null;
      for (let i = lo; i < hi; i++) {
        const p = pts[i];
        if (!isNum(p[1])) continue;
        if (!mn || p[1] < mn[1]) mn = p;
        if (!mx || p[1] > mx[1]) mx = p;
      }
      if (mn && mx) { if (mn[0] <= mx[0]) out.push(mn, mx); else out.push(mx, mn); }
    }
    out.push(pts[pts.length - 1]);
    return out;
  }
  function nearestIdx(arr, x, key) {
    let lo = 0, hi = arr.length - 1;
    if (hi < 0) return -1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if ((key ? arr[mid][0] : arr[mid]) < x) lo = mid; else hi = mid;
    }
    const a = key ? arr[lo][0] : arr[lo], b = key ? arr[hi][0] : arr[hi];
    return Math.abs(a - x) <= Math.abs(b - x) ? lo : hi;
  }
  /** EMA whose span adapts to how densely the run logs (about 1/20 of the logged points). */
  function emaAlpha(n) { const span = Math.max(3, n / 20); return 2 / (span + 1); }
  function ema(pts, alpha) {
    const out = [];
    let m = null;
    for (const p of pts) {
      if (!isNum(p[1])) { out.push([p[0], null]); continue; }
      m = m === null ? p[1] : alpha * p[1] + (1 - alpha) * m;
      out.push([p[0], m]);
    }
    return out;
  }
  const tooltipEl = () => document.getElementById('tooltip');
  function hideTooltip() { const t = tooltipEl(); if (t) t.classList.add('hidden'); }
  function showTooltip(head, rows, cx, cy) {
    const t = tooltipEl();
    setKids(t, h('div', { class: 'tt-head', text: head }),
      rows.map((r) => h('div', { class: 'tt-row' }, h('strong', { text: r.value }),
        h('span', { class: 'keyline b' + r.slot + (r.faint ? ' faint' : '') }), h('span', { class: 'nm', text: r.name }))));
    t.classList.remove('hidden');
    const w = t.offsetWidth, ht = t.offsetHeight;
    let x = cx + 14, y = cy + 14;
    if (x + w > window.innerWidth - 8) x = cx - w - 14;
    if (y + ht > window.innerHeight - 8) y = cy - ht - 14;
    t.style.left = Math.max(8, x) + 'px';
    t.style.top = Math.max(8, y) + 'px';
  }

  class LineChart {
    constructor(o) {
      this.o = Object.assign({ height: 210, xLabel: 'step', logToggle: true }, o);
      this.logY = store.get('log:' + o.id, !!o.logY);
      this.series = [];
      this.el = h('div', { class: 'chart-card' });
      const head = h('div', { class: 'chart-head' }, h('h3', { text: this.o.title }));
      this.noteEl = h('span', { class: 'note' });
      head.appendChild(this.noteEl);
      if (this.o.logToggle) {
        this.logBtn = h('button', {
          class: 'btn sm', type: 'button', title: 'Toggle a logarithmic y axis',
          onclick: () => {
            this.logY = !this.logY; store.set('log:' + this.o.id, this.logY); this.updateLogBtn(); this.render();
          },
        });
        head.appendChild(this.logBtn);
        this.updateLogBtn();
      }
      this.legend = h('div', { class: 'legend' });
      this.body = h('div', { class: 'chart-body' });
      append(this.el, [head, this.legend, this.body]);
      S.charts.add(this);
    }
    updateLogBtn() { this.logBtn.textContent = this.logY ? 'log y' : 'linear y'; this.logBtn.setAttribute('aria-pressed', String(this.logY)); }
    set(series, note) {
      this.series = series || [];
      this.noteEl.textContent = note || this.o.note || '';
      this.render();
    }
    render() {
      const o = this.o;
      const width = Math.max(260, Math.floor(this.body.clientWidth || 520));
      const height = o.height;
      const series = this.series.map((sr) => Object.assign({}, sr, {
        pts: sr.pts.filter((p) => isNum(p[0])).sort((a, b) => a[0] - b[0]),
      }));
      let useLog = this.logY;
      const finite = [];
      for (const sr of series) for (const p of sr.pts) if (isNum(p[1])) finite.push(p);
      if (useLog && finite.some((p) => p[1] <= 0)) useLog = false;
      const visible = series.filter((sr) => sr.pts.some((p) => isNum(p[1])));
      setKids(this.legend, );
      if (series.length >= 2) {
        for (const sr of series) {
          const key = sv('svg', { width: 16, height: 8, 'aria-hidden': 'true' },
            sv('line', { x1: 1, x2: 15, y1: 4, y2: 4, class: 'line s' + sr.slot + (sr.faint ? ' faint' : '') + (sr.dashed ? ' dashed' : '') }));
          const has = sr.pts.some((p) => isNum(p[1]));
          this.legend.appendChild(h('span', { class: 'item' + (has ? '' : ' muted') }, key, sr.name + (has ? '' : ' (no data)')));
        }
      }
      if (!finite.length) {
        setKids(this.body, h('div', { class: 'chart-empty', text: o.empty || 'No data yet' }));
        return;
      }
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (const p of finite) { x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); }
      if (x0 === x1) { x0 -= 1; x1 += 1; }
      let yt, ystep, L0, L1;
      if (useLog) {
        L0 = Math.floor(Math.log10(y0)); L1 = Math.ceil(Math.log10(y1));
        if (L0 === L1) { L0 -= 1; L1 += 1; }
        yt = [];
        const dense = L1 - L0 <= 2;
        for (let e = L0; e <= L1; e++) for (const m of (dense ? [1, 2, 5] : [1])) {
          const v = m * Math.pow(10, e);
          if (Math.log10(v) <= L1 + 1e-9) yt.push(v);
        }
        ystep = null;
      } else {
        if (isNum(o.yMin)) y0 = Math.min(y0, o.yMin);
        if (isNum(o.yMax)) y1 = Math.max(y1, o.yMax);
        if (y0 === y1) { const d = Math.abs(y0) * 0.1 || 1; y0 -= d; y1 += d; }
        const st = niceStep(y1 - y0, 4);
        y0 = Math.floor(y0 / st - 1e-9) * st; y1 = Math.ceil(y1 / st - 1e-9) * st;
        if (isNum(o.yMin) && o.yMin >= 0 && y0 < 0) y0 = 0;
        const t = linTicks(y0, y1, 4); yt = t.ticks; ystep = t.step;
      }
      const ylabels = yt.map((v) => (o.yFormat && !useLog ? o.yFormat(v) : fmtTick(v, useLog ? v : ystep)));
      const ml = Math.max(34, Math.min(72, 10 + 6.4 * Math.max(...ylabels.map((s) => s.length))));
      const direct = visible.length >= 2 && visible.length <= 4 && width >= 440 && !o.noDirect;
      const mr = direct ? 100 : 14, mt = 8, mb = 28;
      const pw = Math.max(40, width - ml - mr), ph = height - mt - mb;
      const X = (v) => ml + ((v - x0) / (x1 - x0)) * pw;
      const Y = useLog ? (v) => mt + ph - ((Math.log10(v) - L0) / (L1 - L0)) * ph : (v) => mt + ph - ((v - y0) / (y1 - y0)) * ph;
      const svg = sv('svg', { width, height, viewBox: `0 0 ${width} ${height}`, role: 'img', tabindex: 0,
        'aria-label': o.title + ' chart. Use the left and right arrow keys to read values.' });
      yt.forEach((v, i) => {
        const y = Y(v);
        if (y < mt - 0.5 || y > mt + ph + 0.5) return;
        svg.appendChild(sv('line', { class: 'grid', x1: ml, x2: ml + pw, y1: y.toFixed(1), y2: y.toFixed(1) }));
        svg.appendChild(sv('text', { class: 'tick', x: ml - 6, y: (y + 3.5).toFixed(1), 'text-anchor': 'end' }, ylabels[i]));
      });
      const xt = linTicks(x0, x1, Math.max(2, Math.floor(pw / 95)));
      for (const v of xt.ticks) {
        const x = X(v);
        if (x < ml - 0.5 || x > ml + pw + 0.5) continue;
        svg.appendChild(sv('text', { class: 'tick', x: x.toFixed(1), y: mt + ph + 16, 'text-anchor': 'middle' },
          o.xFormat ? o.xFormat(v) : fmtTick(v, xt.step)));
      }
      svg.appendChild(sv('line', { class: 'axis', x1: ml, x2: ml + pw, y1: mt + ph, y2: mt + ph }));
      svg.appendChild(sv('text', { class: 'axis-label', x: ml + pw, y: height - 1, 'text-anchor': 'end' }, o.xLabel));
      const drawn = [];
      for (const sr of series) {
        const pts = decimate(sr.pts, Math.max(50, Math.floor(pw)));
        let d = '', pen = false, nFinite = 0, lastP = null;
        for (const p of pts) {
          const ok = isNum(p[1]) && (!useLog || p[1] > 0);
          if (!ok) { pen = false; continue; }
          d += (pen ? 'L' : 'M') + X(p[0]).toFixed(1) + ' ' + Y(p[1]).toFixed(1);
          pen = true; nFinite++; lastP = p;
        }
        if (nFinite > 1) svg.appendChild(sv('path', { d, class: 'line s' + sr.slot + (sr.faint ? ' faint' : '') + (sr.dashed ? ' dashed' : '') }));
        else if (nFinite === 1) svg.appendChild(sv('circle', { cx: X(lastP[0]).toFixed(1), cy: Y(lastP[1]).toFixed(1), r: 3.5, class: 'f' + sr.slot }));
        drawn.push(Object.assign({}, sr, { draw: pts.filter((p) => isNum(p[1]) && (!useLog || p[1] > 0)) }));
      }
      if (direct) {
        // Lines that reach the right edge are labelled in the right margin (stacked, never colliding);
        // lines that stop early (e.g. a shorter run in Compare) are labelled where they end.
        const labs = drawn.filter((sr) => sr.draw.length).map((sr) => {
          const p = sr.draw[sr.draw.length - 1];
          const text = sr.name.length > 14 ? sr.name.slice(0, 13) + '…' : sr.name;
          const ex = X(p[0]) + 6;
          const inside = p[0] < x1 - 0.03 * (x1 - x0) && ex + 6.2 * text.length < ml + pw;
          return { sr, y: Y(p[1]), x: ex, inside, text };
        });
        const margin = labs.filter((l) => !l.inside).sort((a, b) => a.y - b.y);
        for (let i = 1; i < margin.length; i++) if (margin[i].y - margin[i - 1].y < 13) margin[i].y = margin[i - 1].y + 13;
        const over = margin.length ? margin[margin.length - 1].y - (mt + ph) : 0;
        if (over > 0) margin.forEach((l) => { l.y -= over; });
        for (const l of margin) {
          svg.appendChild(sv('line', { x1: ml + pw + 4, x2: ml + pw + 12, y1: l.y.toFixed(1), y2: l.y.toFixed(1), class: 'line s' + l.sr.slot + (l.sr.faint ? ' faint' : '') }));
          svg.appendChild(sv('text', { class: 'dlabel', x: ml + pw + 15, y: (l.y + 3.5).toFixed(1) }, l.text));
        }
        for (const l of labs.filter((x) => x.inside)) {
          svg.appendChild(sv('text', { class: 'dlabel halo', x: l.x.toFixed(1), y: (l.y + 3.5).toFixed(1) }, l.text));
        }
      }
      // ---- hover layer: crosshair snaps to the nearest x; tooltip lists every series
      const cross = sv('line', { class: 'crosshair', y1: mt, y2: mt + ph, visibility: 'hidden' });
      const dots = sv('g');
      const hit = sv('rect', { class: 'hit', x: ml, y: mt, width: pw, height: ph });
      svg.append(cross, dots, hit);
      const xsSet = new Set();
      for (const sr of drawn) for (const p of sr.draw) xsSet.add(p[0]);
      const xs = Array.from(xsSet).sort((a, b) => a - b);
      const fmtY = o.yFormatTip || ((v) => fmtNum(v));
      const fmtX = o.xFormat || ((v) => fmtNum(v));
      let kb = xs.length - 1;
      const show = (xv, cx, cy) => {
        cross.setAttribute('x1', X(xv).toFixed(1)); cross.setAttribute('x2', X(xv).toFixed(1));
        cross.setAttribute('visibility', 'visible');
        setKids(dots, );
        const rows = [];
        for (const sr of drawn) {
          if (!sr.draw.length) continue;
          const p = sr.draw[nearestIdx(sr.draw, xv, true)];
          dots.appendChild(sv('circle', { cx: X(p[0]).toFixed(1), cy: Y(p[1]).toFixed(1), r: 4, class: 'dot f' + sr.slot }));
          rows.push({ value: fmtY(p[1]), name: sr.name + (p[0] !== xv ? ' @ ' + fmtX(p[0]) : ''), slot: sr.slot, faint: sr.faint });
        }
        showTooltip(o.xLabel + ' ' + fmtX(xv), rows, cx, cy);
      };
      const hide = () => { cross.setAttribute('visibility', 'hidden'); setKids(dots, ); hideTooltip(); };
      hit.addEventListener('pointermove', (ev) => {
        const r = svg.getBoundingClientRect();
        const xv = x0 + ((ev.clientX - r.left - ml) / pw) * (x1 - x0);
        kb = nearestIdx(xs, xv, false);
        if (kb >= 0) show(xs[kb], ev.clientX, ev.clientY);
      });
      hit.addEventListener('pointerleave', hide);
      svg.addEventListener('blur', hide);
      const kbShow = () => {
        if (kb < 0 || !xs.length) return;
        const r = svg.getBoundingClientRect();
        show(xs[kb], r.left + X(xs[kb]), r.top + mt + 10);
      };
      svg.addEventListener('focus', () => { kb = xs.length - 1; kbShow(); });
      svg.addEventListener('keydown', (ev) => {
        const k = ev.key;
        if (k === 'ArrowLeft') kb = Math.max(0, kb - 1);
        else if (k === 'ArrowRight') kb = Math.min(xs.length - 1, kb + 1);
        else if (k === 'Home') kb = 0;
        else if (k === 'End') kb = xs.length - 1;
        else if (k === 'Escape') { hide(); return; } else return;
        ev.preventDefault();
        kbShow();
      });
      setKids(this.body, svg);
    }
  }

  /** Keeps chart cards in a stable order and updates them in place across refreshes. */
  class ChartGrid {
    constructor(container) { this.el = h('div', { class: 'charts' }); container.appendChild(this.el); this.charts = new Map(); }
    sync(specs) {
      const keep = new Set();
      for (const sp of specs) {
        let c = this.charts.get(sp.id);
        if (!c) { c = new LineChart(sp); this.charts.set(sp.id, c); }
        keep.add(sp.id);
        this.el.appendChild(c.el);
        c.set(sp.series, sp.note);
      }
      for (const [id, c] of this.charts) if (!keep.has(id)) { c.el.remove(); this.charts.delete(id); S.charts.delete(c); }
    }
  }

  const col = (rows, xk, yk) => rows.map((r) => [r[xk], isNum(r[yk]) ? r[yk] : null]);
  const has = (rows, k) => rows.some((r) => isNum(r[k]));

  // ================================================================== incremental series cache
  async function fetchSeries(run, file) {
    const cache = S.series[run] || (S.series[run] = {});
    let c = cache[file] || (cache[file] = { fileId: null, next: 0, rows: [], exists: false });
    for (let attempt = 0; attempt < 2; attempt++) {
      const d = await api('api/run/' + encodeURIComponent(run) + '/series?file=' + file + '&since=' + c.next);
      if (!d.exists) { cache[file] = { fileId: null, next: 0, rows: [], exists: false }; return cache[file]; }
      if (c.fileId && d.file_id !== c.fileId && !d.reset) { c = cache[file] = { fileId: null, next: 0, rows: [], exists: true }; continue; }
      if (d.reset) c.rows = [];
      c.rows = c.rows.concat(d.rows);
      if (c.rows.length > 60000) c.rows = c.rows.filter((_, i) => i % 2 === 0 || i === c.rows.length - 1);
      c.next = d.next; c.fileId = d.file_id; c.exists = true; c.downsampled = c.downsampled || d.downsampled;
      return c;
    }
    return c;
  }

  // ================================================================== views
  function pageHead(title, sub, crumbs, actions) {
    return h('div', { class: 'page-head' },
      h('div', { class: 'grow' }, crumbs ? h('div', { class: 'crumbs' }, crumbs) : null, h('h1', { text: title }),
        sub ? h('div', { class: 'sub' }, sub) : null),
      actions ? h('div', { class: 'actions' }, actions) : null);
  }
  function bundleBtn(url, label, title, primary) {
    return h('a', { class: 'btn' + (primary ? ' primary' : ''), href: withToken(url), title, download: '' }, label);
  }

  // ---------------------------------------------------------------- runs list
  function RunsView() {
    let tiles, tableBox, filter, sub, data = null;
    const render = () => {
      if (!data) return;
      sub.textContent = 'Directories under ' + data.root + ' that contain config.json or status.json';
      const counts = {};
      for (const r of data.runs) counts[r.display_state] = (counts[r.display_state] || 0) + 1;
      setKids(tiles, 
        tile('Runs', data.runs.length),
        ['running', 'stale', 'failed', 'finished'].map((st) => tile(STATES[st][1], counts[st] || 0, st)));
      const q = filter.value.trim().toLowerCase();
      const runs = data.runs.filter((r) => !q || [r.name, r.variant, r.display_state, r.description].join(' ').toLowerCase().includes(q));
      if (!data.runs.length) {
        setKids(tableBox, card(null, emptyState('No training runs found.',
          'Start one with scripts/train_d1.sh (or python -m monet_action.train); it appears here as soon as its output directory has config.json.')));
        return;
      }
      const rows = runs.map((r) => h('tr', null,
        h('td', { class: 'name-cell' }, h('a', { href: '#/run/' + encodeURIComponent(r.name) + '/overview', text: r.name }),
          h('span', { class: 'sub', text: [r.variant, r.description].filter(Boolean).join(' · ') || ' ', title: r.description || '' })),
        td(badge(r.display_state, r.state_inferred)),
        td(progress(r.step, r.max_steps)),
        td(fmtNum(r.last_loss), 'r num'),
        td(fmtDuration(r.eta_s), 'r num'),
        td(h('span', { title: fmtTime(r.updated_at), text: fmtAgo(r.updated_at) }), 'num'),
        td(r.K, 'r num'), td(r.lora ? 'yes' : 'no'), td(r.head), td(r.reasoning_mode), td(r.feature_source),
        td(h('a', { href: withToken('api/bundle/run/' + encPath(r.name)), title: 'Download this run’s logs as a zip', download: '', text: 'logs.zip' }))));
      setKids(tableBox, rows.length ? tableEl(
        ['Run', 'State', 'Progress', { t: 'Last loss', r: 1 }, { t: 'ETA', r: 1 }, 'Updated', { t: 'K', r: 1 }, 'LoRA', 'Head', 'Reasoning', 'Features', 'Bundle'], rows)
        : card(null, emptyState('No runs match the filter.')));
    };
    return {
      mount(root) {
        sub = h('span');
        filter = h('input', { type: 'search', placeholder: 'Filter by name, variant or state', 'aria-label': 'Filter runs' });
        filter.value = store.get('runsFilter', '');
        filter.addEventListener('input', () => { store.set('runsFilter', filter.value); render(); });
        tiles = h('div', { class: 'tiles' });
        tableBox = h('div');
        append(root, [pageHead('Training runs', sub), tiles, h('div', { class: 'toolbar' }, filter), tableBox]);
      },
      async update() { data = await api('api/runs'); render(); },
    };
  }
  /** Stat tile; ``state`` renders the label as a status badge (icon + text, never colour alone). */
  function tile(label, value, state, hint) {
    return h('div', { class: 'tile' }, h('div', { class: 'label' }, state ? badge(state) : label),
      h('div', { class: 'value num', text: typeof value === 'number' ? fmtNum(value) : String(value) }),
      hint ? h('div', { class: 'hint', text: hint }) : null);
  }

  // ---------------------------------------------------------------- run detail
  const RUN_TABS = [['overview', 'Overview'], ['charts', 'Charts'], ['logs', 'Logs'], ['events', 'Events'], ['files', 'Files']];
  function RunView(name, initialTab) {
    let tab = initialTab || 'overview';
    let headBox, tabsBox, body, detail = null, ts = {};
    const encName = encodeURIComponent(name);
    const renderTabs = () => {
      setKids(tabsBox, RUN_TABS.map(([k, label]) => h('a', { href: '#/run/' + encName + '/' + k, class: k === tab ? 'active' : null, text: label })));
    };
    const renderHead = () => {
      const s = detail.summary;
      const sub = h('span', null, badge(s.display_state, s.state_inferred), ' ',
        [s.variant, s.description].filter(Boolean).join(' · '));
      setKids(headBox, pageHead(name, sub, [h('a', { href: '#/runs', text: 'Runs' }), ' / ' + name], [
        bundleBtn('api/bundle/run/' + encPath(name), 'Download debug bundle (logs)', 'Logs, configs, status, metrics, checkpoint meta.json (no weights)', true),
        bundleBtn('api/bundle/run/' + encPath(name) + '?checkpoints=1', '… with checkpoints', 'Also includes ckpt_*/ weights (can be large)'),
      ]));
    };
    return {
      mount(root) {
        headBox = h('div'); tabsBox = h('nav', { class: 'tabs', 'aria-label': 'Run sections' }); body = h('div');
        append(root, [headBox, tabsBox, body]);
        renderTabs();
      },
      onRoute(r) {
        if ((r.tab || 'overview') !== tab) {
          tab = r.tab || 'overview'; renderTabs(); setKids(body); ts = {}; S.charts.clear(); refresh(true);
        }
      },
      async update() {
        detail = await api('api/run/' + encName);
        renderHead();
        if (tab === 'overview') renderOverview();
        else if (tab === 'charts') await updateCharts();
        else if (tab === 'logs') await updateLogs();
        else if (tab === 'events') await updateEvents();
        else if (tab === 'files') renderFiles();
      },
    };

    // ---------------- overview
    function renderOverview() {
      const s = detail.summary, st = detail.status || {};
      const stat = (k, v, title) => h('div', { class: 'stat' }, h('div', { class: 'label', text: k }), h('div', { class: 'value num', title: title || null }, v));
      const statusCard = card('Status', h('div', { class: 'toolbar' }, badge(s.display_state, s.state_inferred), progress(s.step, s.max_steps)),
        h('div', { class: 'statgrid' },
          stat('Step', fmtInt(s.step)), stat('Samples seen', fmtInt(s.samples)), stat('Last loss', fmtNum(s.last_loss)),
          stat('ETA', fmtDuration(s.eta_s)), stat('Started', fmtTime(s.started_at)),
          stat('Updated', fmtAgo(s.updated_at), fmtTime(s.updated_at)),
          stat('Process', s.pid ? (s.pid + (s.host ? ' @ ' + s.host : '')) : '–'),
          stat('Output dir', s.output_dir || '–')),
        s.state_inferred ? h('p', { class: 'small muted', text: '* No status.json: state inferred from train_log.jsonl / config.json (older runs or a producer that is not writing status yet).' }) : null,
        s.error || st.error ? h('div', { class: 'errbox mono', text: String(s.error || st.error) }) : null);

      const cfgCard = card('Key configuration', detail.key_config && detail.key_config.length
        ? h('table', { class: 'kv' }, h('tbody', null, detail.key_config.map((g) => [
          h('tr', null, h('td', { colspan: 2, class: 'small', style: { fontWeight: '650', color: 'var(--ink)', paddingTop: '10px' }, text: g.group })),
          g.items.map((it) => h('tr', null, h('td', { text: it.key }), h('td', { class: 'v mono', text: fmtVal(it.value) })))])))
        : emptyState('config.json not found.'),
        detail.config ? h('details', { class: 'section' }, h('summary', { text: 'Full config.json' }), h('pre', { class: 'block', text: JSON.stringify(detail.config, null, 2) })) : null);

      const pr = detail.param_report;
      const prCard = card('Parameters', pr ? kvTable(Object.entries(pr).filter(([k]) => k !== 'monet_load_info').map(([k, v]) =>
        [k.replace(/_/g, ' '), h('span', { class: k === 'trainable_total' ? 'ok' : null, title: isNum(v) ? v.toLocaleString() : null, text: isNum(v) ? fmtCount(v) : fmtVal(v) })]))
        : emptyState('param_report.json not written yet.'),
        pr && pr.monet_load_info ? h('p', { class: 'small muted', text: 'Monet load: ' + JSON.stringify(pr.monet_load_info) }) : null);

      const ga = detail.gradient_audit;
      let gaBody = emptyState('gradient_audit.json not written yet.', 'It is written after the first optimisation step.');
      if (ga && ga.report) {
        const rows = Object.entries(ga.report).map(([g, r]) => h('tr', null, td(g),
          td(fmtNum(r.grad_norm), 'r num'),
          td(h('span', { class: r.finite ? 'ok' : 'bad', text: r.finite ? '✓ finite' : '✕ non-finite' })),
          td((r.n_tensors_nonzero_grad ?? '–') + ' / ' + (r.n_tensors ?? '–'), 'r num')));
        gaBody = h('div', null, h('p', { class: 'small' },
          h('span', { class: ga.passed ? 'ok' : 'bad', text: ga.passed ? '✓ passed' : '✕ failed' }), ' at step ' + fmtVal(ga.step)),
        tableEl(['Group', { t: 'Grad norm', r: 1 }, 'Finite', { t: 'Non-zero / tensors', r: 1 }], rows, { compact: true }));
      }
      const gaCard = card('Gradient audit', gaBody);

      const mem = detail.memory_profile;
      const memCard = card('Memory profile', mem ? kvTable(Object.entries(mem)) : emptyState('memory_profile.json not written yet.'));
      const rep = detail.representation;
      const repCard = card('Representation', rep ? h('div', null,
        kvTable(Object.entries(rep).filter(([k, v]) => k !== 'example_prompt' && typeof v !== 'object').map(([k, v]) => [k.replace(/_/g, ' '), v])),
        Object.entries(rep).filter(([k, v]) => k !== 'example_prompt' && v && typeof v === 'object').map(([k, v]) =>
          h('details', { class: 'section' }, h('summary', { text: k.replace(/_/g, ' ') }), h('pre', { class: 'block', text: JSON.stringify(v, null, 2) }))),
        rep.example_prompt ? h('details', { class: 'section' }, h('summary', { text: 'example prompt' }), h('pre', { class: 'block', text: String(rep.example_prompt) })) : null)
        : emptyState('representation.json not written yet.', 'Written by the training producer: latent/VL layers, pooling, K, convention, token layout.'));
      const env = detail.environment;
      const envCard = card('Environment', env ? kvTable(['python', 'torch', 'torch_cuda', 'cuda_available', 'gpus', 'nvidia_driver', 'transformers', 'peft', 'mujoco', 'robosuite', 'repo_commit', 'repo_dirty', 'platform']
        .filter((k) => k in env).map((k) => [k.replace(/_/g, ' '), env[k]])) : emptyState('environment.json not found.'));
      const ck = detail.checkpoints || [];
      const ckCard = card('Checkpoints', ck.length ? tableEl(['Checkpoint', { t: 'Step', r: 1 }, { t: 'Samples', r: 1 }, { t: 'Size', r: 1 }, 'Files'],
        ck.slice().reverse().map((c) => h('tr', null,
          td(h('span', null, c.name, c.is_last ? h('span', { class: 'chip', text: 'last' }) : null)), td(fmtInt(c.step), 'r num'),
          td(fmtInt(c.consumed_samples), 'r num'), td(fmtBytes(c.size), 'r num'),
          td(c.files.map((f) => h('a', { class: 'chip', href: fileUrl('runs', name, c.name + '/' + f.name, true), text: f.name, title: fmtBytes(f.size) }))))), { compact: true })
        : emptyState('No checkpoints yet.'));
      const sp = detail.split;
      const splitCard = sp && Object.keys(sp).length
        ? card('Data split', kvTable(Object.entries(sp).map(([k, v]) => [k.replace(/_/g, ' '), v]))) : null;
      setKids(body, h('div', { class: 'masonry' }, statusCard, gaCard, prCard, ckCard, cfgCard, memCard, repCard, envCard, splitCard));
    }

    // ---------------- charts
    async function updateCharts() {
      if (!ts.grid) {
        ts.missing = h('p', { class: 'small muted' });
        ts.grid = new ChartGrid(body);
        body.appendChild(ts.missing);
      }
      const want = ['train_log', 'val_log', 'overfit_log', 'sim_eval_log', 'system'];
      const data = {};
      for (const f of want) {
        if (detail.series && detail.series[f] && detail.series[f].exists) data[f] = (await fetchSeries(name, f)).rows;
        else data[f] = [];
      }
      const train = data.train_log.filter((r) => r.event === undefined && (isNum(r.loss) || r.loss === null));
      const specs = [];
      const loss = col(train, 'step', 'loss');
      const alpha = emaAlpha(loss.length);
      specs.push({ id: 'loss', title: 'Training loss', note: train.length ? 'raw (faint) and EMA α=' + alpha.toFixed(2) : '', series: [
        { name: 'loss (raw)', slot: 1, faint: true, pts: loss },
        { name: 'loss (EMA)', slot: 1, pts: ema(loss, alpha) }], empty: 'No training steps logged yet', noDirect: true });
      for (const [key, label] of [['val_log', 'Held-out validation'], ['overfit_log', 'Overfit windows']]) {
        const rows = data[key];
        if (!rows.length) continue;
        specs.push({ id: key + '_mse', title: label + ' – errors (fixed noise)', series: [
          { name: 'fm_loss', slot: 1, pts: col(rows, 'step', 'fm_loss') },
          { name: 'action_mse', slot: 2, pts: col(rows, 'step', 'action_mse') },
          { name: 'first-action mse', slot: 3, pts: col(rows, 'step', 'action_mse_first') }] });
        if (has(rows, 'gripper_acc')) {
          specs.push({ id: key + '_grip', title: label + ' – gripper sign accuracy', logToggle: false, yMin: 0, yMax: 1,
            yFormat: (v) => fmtPct(v, 0), yFormatTip: (v) => fmtPct(v), series: [{ name: 'gripper accuracy', slot: 1, pts: col(rows, 'step', 'gripper_acc') }] });
        }
      }
      if (data.sim_eval_log.length) {
        specs.push({ id: 'sim', title: 'Periodic simulator success', logToggle: false, yMin: 0, yMax: 1,
          yFormat: (v) => fmtPct(v, 0), yFormatTip: (v) => fmtPct(v), series: [{ name: 'success rate', slot: 1, pts: col(data.sim_eval_log, 'step', 'success_rate') }] });
      }
      const lrRows = train.filter((r) => r.lrs && typeof r.lrs === 'object');
      if (lrRows.length) {
        const groups = Array.from(new Set(lrRows.flatMap((r) => Object.keys(r.lrs)))).sort();
        specs.push({ id: 'lr', title: 'Learning rate per parameter group', series: groups.slice(0, 8).map((g, i) => ({
          name: g, slot: i + 1, pts: lrRows.map((r) => [r.step, isNum(r.lrs[g]) ? r.lrs[g] : null]) })) });
      } else if (has(train, 'lr_scale')) {
        specs.push({ id: 'lr', title: 'Learning-rate scale', series: [{ name: 'lr scale', slot: 1, pts: col(train, 'step', 'lr_scale') }] });
      }
      const gg = train.filter((r) => r.grad_groups && typeof r.grad_groups === 'object');
      const gnames = Array.from(new Set(gg.flatMap((r) => Object.keys(r.grad_groups)))).sort();
      const gSeries = [{ name: 'total (pre-clip)', slot: 1, pts: col(train, 'step', 'grad_norm') }]
        .concat(gnames.slice(0, 7).map((g, i) => ({ name: g, slot: i + 2, pts: gg.map((r) => [r.step, r.grad_groups[g] && isNum(r.grad_groups[g].grad_norm) ? r.grad_groups[g].grad_norm : null]) })));
      if (has(train, 'grad_norm') || gg.length) specs.push({ id: 'grad', title: 'Gradient norms', logY: true, note: gg.length ? 'per-group norms on logged audit steps' : '', series: gSeries });
      if (has(train, 'step_time_s') || has(train, 'data_time_s')) {
        specs.push({ id: 'time', title: 'Step and data-loading time (s)', series: [
          { name: 'step time', slot: 1, pts: col(train, 'step', 'step_time_s') },
          { name: 'data wait', slot: 2, pts: col(train, 'step', 'data_time_s') }] });
      }
      specs.push({ id: 'mem', title: 'Peak GPU memory, torch allocator (GB)', logToggle: false, yMin: 0,
        empty: 'Not recorded (CPU run or older log format)', series: [{ name: 'peak memory', slot: 1, pts: col(train, 'step', 'peak_mem_gb') }] });
      const sys = data.system;
      if (sys.length) {
        const t0 = Math.min(...sys.map((r) => r.time).filter(isNum));
        const tx = (r) => (isNum(r.time) ? (r.time - t0) / 60 : null);
        const gpuIdx = Array.from(new Set(sys.flatMap((r) => (r.gpus || []).map((g) => g.index)))).sort((a, b) => a - b).slice(0, 4);
        const gpuPts = (k) => gpuIdx.map((gi, i) => ({ name: 'GPU ' + gi, slot: i + 1,
          pts: sys.map((r) => { const g = (r.gpus || []).find((x) => x.index === gi); return [tx(r), g && isNum(g[k]) ? g[k] : null]; }) }));
        const xl = { xLabel: 'minutes since first sample' };
        if (gpuIdx.length) {
          specs.push(Object.assign({ id: 'gpu_util', title: 'GPU utilisation (%)', logToggle: false, yMin: 0, yMax: 100, series: gpuPts('util_percent') }, xl));
          const memS = gpuPts('mem_used_gb');
          const cap = gpuIdx.map((gi, i) => ({ name: 'GPU ' + gi + ' capacity', slot: i + 1, dashed: true,
            pts: sys.map((r) => { const g = (r.gpus || []).find((x) => x.index === gi); return [tx(r), g && isNum(g.mem_total_gb) ? g.mem_total_gb : null]; }) }));
          specs.push(Object.assign({ id: 'gpu_mem', title: 'GPU memory used (GB)', logToggle: false, yMin: 0, series: memS.concat(cap) }, xl));
        }
        if (has(sys, 'cpu_percent')) specs.push(Object.assign({ id: 'cpu', title: 'Host CPU (%)', logToggle: false, yMin: 0, series: [{ name: 'cpu', slot: 1, pts: sys.map((r) => [tx(r), isNum(r.cpu_percent) ? r.cpu_percent : null]) }] }, xl));
        if (has(sys, 'ram_used_gb')) specs.push(Object.assign({ id: 'ram', title: 'Host RAM used (GB)', logToggle: false, yMin: 0, series: [{ name: 'ram', slot: 1, pts: sys.map((r) => [tx(r), isNum(r.ram_used_gb) ? r.ram_used_gb : null]) }] }, xl));
      }
      ts.grid.sync(specs);
      const missing = want.filter((f) => !(detail.series && detail.series[f] && detail.series[f].exists)).map((f) => f + '.jsonl');
      ts.missing.textContent = missing.length ? 'Not logged by this run (yet): ' + missing.join(', ') + '.' : '';
    }

    // ---------------- logs
    async function updateLogs() {
      const logs = detail.logs || [];
      if (!ts.box) {
        const st = store.get('logopts:' + name, { tail: 1000, auto: true });
        ts.sel = h('select', { 'aria-label': 'Log file' });
        ts.tail = h('select', { 'aria-label': 'Lines' }, [200, 1000, 5000, 20000].map((n) => h('option', { value: n, text: 'last ' + n.toLocaleString() + ' lines' })));
        ts.tail.value = String(st.tail);
        ts.filter = h('input', { type: 'search', placeholder: 'Filter lines (case-insensitive)', 'aria-label': 'Filter log lines' });
        ts.server = h('input', { type: 'checkbox' });
        ts.auto = h('input', { type: 'checkbox' }); ts.auto.checked = st.auto !== false;
        ts.dl = h('a', { class: 'btn sm', text: 'Download full log' });
        ts.count = h('span', { class: 'small muted num' });
        ts.box = h('div', { class: 'logbox', tabindex: 0, 'aria-label': 'Log output' });
        ts.meta = h('div', { class: 'logmeta' });
        const save = () => store.set('logopts:' + name, { tail: Number(ts.tail.value), auto: ts.auto.checked });
        ts.sel.addEventListener('change', () => refresh(true));
        ts.tail.addEventListener('change', () => { save(); refresh(true); });
        ts.auto.addEventListener('change', save);
        ts.filter.addEventListener('input', debounce(() => (ts.server.checked ? refresh(true) : renderLines()), 200));
        ts.server.addEventListener('change', () => refresh(true));
        ts.toolbar = h('div', { class: 'toolbar' }, ts.sel, ts.tail, ts.filter,
          h('label', { class: 'toggle', title: 'Search the last 8 MB of the file on the server instead of the loaded tail' }, ts.server, 'search whole file'),
          h('label', { class: 'toggle' }, ts.auto, 'follow tail'), ts.count, ts.dl);
        ts.empty = card(null, emptyState('No log files yet.',
          'Training writes logs/train.log; any *.log / *.txt / *.out / *.err in the run directory (or logs/) is listed here.'));
        append(body, [ts.toolbar, ts.box, ts.meta, ts.empty]);
      }
      const none = !logs.length;
      [ts.toolbar, ts.box, ts.meta].forEach((el) => el.classList.toggle('hidden', none));
      ts.empty.classList.toggle('hidden', !none);
      if (none) return;
      const cur = ts.sel.value || logs[0].path;
      setKids(ts.sel, logs.map((l) => h('option', { value: l.path, text: l.path + ' (' + fmtBytes(l.size) + ')' })));
      ts.sel.value = logs.some((l) => l.path === cur) ? cur : logs[0].path;
      ts.dl.classList.remove('hidden');
      ts.dl.href = fileUrl('runs', name, ts.sel.value, true);
      let url = 'api/run/' + encName + '/log?file=' + encodeURIComponent(ts.sel.value) + '&tail=' + ts.tail.value;
      if (ts.server.checked && ts.filter.value.trim()) url += '&q=' + encodeURIComponent(ts.filter.value.trim());
      ts.data = await api(url);
      renderLines();
    }
    function renderLines() {
      const d = ts.data;
      if (!d) return;
      const q = ts.filter.value.trim().toLowerCase();
      const serverFiltered = ts.server.checked && q;
      const lines = !q || serverFiltered ? d.lines : d.lines.filter((l) => l.toLowerCase().includes(q));
      const atBottom = ts.box.scrollHeight - ts.box.scrollTop - ts.box.clientHeight < 30;
      const frag = document.createDocumentFragment();
      for (const ln of lines) {
        const low = ln.toLowerCase();
        const cls = /traceback|error|exception|\bnan\b|non-finite|failed|killed|out of memory/.test(low) ? 'ln err' : (/warn/.test(low) ? 'ln warn' : 'ln');
        const div = h('div', { class: cls });
        if (q) {
          let i = 0, j;
          while ((j = low.indexOf(q, i)) >= 0) { div.appendChild(document.createTextNode(ln.slice(i, j))); div.appendChild(h('mark', { text: ln.slice(j, j + q.length) })); i = j + q.length; }
          div.appendChild(document.createTextNode(ln.slice(i)));
        } else div.textContent = ln || ' ';
        frag.appendChild(div);
      }
      setKids(ts.box, frag);
      if (!lines.length) ts.box.appendChild(emptyState(q ? 'No lines match.' : 'Log is empty.'));
      ts.count.textContent = q ? lines.length.toLocaleString() + ' matching' : '';
      ts.meta.textContent = d.path + ' · ' + fmtBytes(d.size) + ' · modified ' + fmtAgo(d.mtime) +
        (d.truncated_head ? ' · showing the tail of the file' : '');
      if (ts.auto.checked && (atBottom || !ts.scrolled)) { ts.box.scrollTop = ts.box.scrollHeight; ts.scrolled = true; }
    }

    // ---------------- events
    async function updateEvents() {
      if (!ts.box) { ts.box = h('div'); body.appendChild(ts.box); }
      if (!(detail.series && detail.series.events && detail.series.events.exists)) {
        setKids(ts.box, card(null, emptyState('No events.jsonl yet.', 'Training appends start / resume / checkpoint / val / finished / failed events here.')));
        return;
      }
      const rows = (await fetchSeries(name, 'events')).rows.slice(-1000).reverse();
      setKids(ts.box, tableEl(['Time', 'Ago', 'Event', 'Details'], rows.map((r) => {
        const extra = Object.entries(r).filter(([k]) => k !== 'time' && k !== 'event').map(([k, v]) => k + '=' + (typeof v === 'object' ? JSON.stringify(v) : String(v))).join('  ');
        const ev = String(r.event || '?');
        return h('tr', null, td(fmtTime(r.time), 'num'), td(fmtAgo(r.time), 'num'),
          td(h('span', { class: /fail|error/.test(ev) ? 'bad' : (/finish/.test(ev) ? 'ok' : null), text: ev })), td(extra.length > 400 ? extra.slice(0, 400) + '…' : extra, 'wrap mono'));
      }), { compact: true }));
    }

    // ---------------- files
    function renderFiles() {
      if (!ts.box) {
        ts.filter = h('input', { type: 'search', placeholder: 'Filter files', 'aria-label': 'Filter files' });
        ts.filter.addEventListener('input', renderFilesTable);
        ts.box = h('div');
        append(body, [h('div', { class: 'toolbar' }, ts.filter,
          bundleBtn('api/bundle/run/' + encPath(name), 'Download debug bundle (logs)', null, true),
          bundleBtn('api/bundle/run/' + encPath(name) + '?checkpoints=1', '… with checkpoints'),
          bundleBtn('api/bundle/run/' + encPath(name) + '?checkpoints=1&videos=1', '… with checkpoints and videos')), ts.box]);
      }
      renderFilesTable();
    }
    function renderFilesTable() {
      const q = ts.filter.value.trim().toLowerCase();
      const files = (detail.files || []).filter((f) => !q || f.path.toLowerCase().includes(q));
      setKids(ts.box, tableEl(['Path', { t: 'Size', r: 1 }, 'Modified', ''], files.map((f) => h('tr', null,
        td(h('a', { href: fileUrl('runs', name, f.path), target: '_blank', rel: 'noopener', text: f.path }), 'wrap mono'),
        td(fmtBytes(f.size), 'r num'), td(h('span', { title: fmtTime(f.mtime), text: fmtAgo(f.mtime) }), 'num'),
        td(h('a', { href: fileUrl('runs', name, f.path, true), text: 'download', download: '' }))))),
      detail.files_truncated ? h('p', { class: 'small muted', text: 'Listing truncated to the first 3,000 files.' }) : null);
    }
  }

  // ---------------------------------------------------------------- evaluations
  function evalRow(e) {
    const lat = e.latency || {};
    const l = (k) => (lat[k] ? fmtNum(lat[k].p50) + ' / ' + fmtNum(lat[k].p95) : '–');
    const prog = e.state === 'finished' || !isNum(e.episodes_total) ? h('span', { class: 'num', text: fmtInt(e.episodes_done) + ' eps' }) : progress(e.episodes_done, e.episodes_total);
    return h('tr', null,
      h('td', { class: 'name-cell' }, h('a', { href: '#/eval/' + encodeURIComponent(e.name), text: e.name }),
        h('span', { class: 'sub', text: [e.model && e.model.variant, e.intervention ? 'intervention: ' + e.intervention : null, e.tag].filter(Boolean).join(' · ') || ' ' })),
      td(e.run_name ? h('a', { href: '#/run/' + encodeURIComponent(e.run_name) + '/overview', text: e.run_name }) : (e.model && e.model.name) || '–'),
      td(h('div', { class: 'toolbar', style: { margin: '0' } }, badge(e.display_state), prog)),
      td(isNum(e.success_rate) ? fmtPct(e.success_rate) + (isNum(e.n_success) ? ' (' + e.n_success + '/' + fmtInt(e.n_episodes) + ')' : '') : '–', 'r num'),
      td(Array.isArray(e.success_ci95) ? '[' + fmtPct(e.success_ci95[0]) + ', ' + fmtPct(e.success_ci95[1]) + ']' : '–', 'num'),
      td(fmtNum(e.gpu_seconds_per_success), 'r num'), td(fmtNum(e.gpu_seconds_per_episode), 'r num'),
      td(l('total'), 'r num'), td(l('monet'), 'r num'), td(l('action_head'), 'r num'),
      td(e.execution_prefix, 'r num'), td(e.num_inference_steps, 'r num'),
      td(h('span', { title: fmtTime(e.updated_at), text: fmtAgo(e.updated_at) }), 'num'),
      td(h('a', { href: withToken('api/bundle/eval/' + encPath(e.name)), download: '', text: 'logs.zip' })));
  }
  const EVAL_HEADERS = ['Evaluation', 'Model run', 'State', { t: 'Success', r: 1 }, '95% CI', { t: 'GPU-s / success', r: 1 }, { t: 'GPU-s / episode', r: 1 },
    { t: 'Call p50 / p95 ms', r: 1 }, { t: 'Monet p50 / p95', r: 1 }, { t: 'Head p50 / p95', r: 1 }, { t: 'E', r: 1 }, { t: 'Steps', r: 1 }, 'Updated', 'Bundle'];

  function EvalsView() {
    let box, sub;
    return {
      mount(root) { sub = h('span'); box = h('div'); append(root, [pageHead('Evaluations', sub), box]); },
      async update() {
        const d = await api('api/evals');
        sub.textContent = 'Directories under ' + d.root + ' with summary.json or episodes.jsonl (and every results/eval/* directory)';
        setKids(box, d.evals.length ? tableEl(EVAL_HEADERS, d.evals.map(evalRow))
          : card(null, emptyState('No evaluations found.', 'Run python -m monet_action.evaluate closed_loop … or scripts/evaluate_all.sh.')));
      },
    };
  }

  function EvalView(name) {
    let headBox, body, epFilter = 'all', epTask = 'all';
    return {
      mount(root) { headBox = h('div'); body = h('div'); append(root, [headBox, body]); },
      async update() {
        const d = await api('api/eval/' + encodeURIComponent(name));
        const row = d.summary_row, s = d.summary || {};
        setKids(headBox, pageHead(name, h('span', null, badge(row.display_state), ' ',
          [row.model && row.model.variant, row.run_name ? 'run ' + row.run_name : null, row.intervention ? 'intervention ' + row.intervention : null].filter(Boolean).join(' · ')),
        [h('a', { href: '#/evals', text: 'Evaluations' }), ' / ' + name], [
          bundleBtn('api/bundle/eval/' + encPath(name), 'Download debug bundle (logs)', 'summary, episodes, traces, configs, logs', true),
          bundleBtn('api/bundle/eval/' + encPath(name) + '?videos=1', '… with videos')]));
        const parts = [];
        parts.push(h('div', { class: 'tiles' },
          tile('Success', isNum(s.success_rate) ? fmtPct(s.success_rate) : '–', null, isNum(row.n_success) ? row.n_success + ' of ' + fmtInt(s.n_episodes) + ' episodes' : null),
          tile('95% CI (Wilson)', Array.isArray(s.success_ci95) ? fmtPct(s.success_ci95[0], 0) + '–' + fmtPct(s.success_ci95[1], 0) : '–'),
          tile('Episodes', fmtInt(row.episodes_done), null, isNum(row.episodes_total) ? 'of ' + fmtInt(row.episodes_total) + ' planned' : null),
          tile('GPU-s / success', fmtNum(s.gpu_seconds_per_success)), tile('GPU-s / episode', fmtNum(s.gpu_seconds_per_episode)),
          tile('Calls / episode', fmtNum(s.action_calls_per_episode)), tile('Env steps / episode', fmtNum(s.env_steps_per_episode)),
          tile('Peak GPU memory', isNum(s.gpu_peak_mem_gb) ? fmtNum(s.gpu_peak_mem_gb) + ' GB' : '–')));
        if (s.note) parts.push(h('p', { class: 'small muted', text: s.note }));
        const names = s.task_names || {};
        const pt = s.per_task || {};
        if (Object.keys(pt).length) {
          parts.push(h('div', { class: 'section' }, h('h2', { text: 'Per task' }), tableEl(['Task', 'Name', { t: 'Successes', r: 1 }, 'Rate', '95% CI'],
            Object.entries(pt).sort((a, b) => Number(a[0]) - Number(b[0])).map(([t, v]) => h('tr', null, td(t, 'num'), td(names[t] || '–', 'wrap'),
              td(v.successes + ' / ' + v.trials, 'r num'),
              td(h('div', { class: 'hbar' }, h('div', { class: 'bar' }, h('span', { style: { width: (100 * (v.rate || 0)).toFixed(1) + '%' } })), h('span', { class: 'num', text: fmtPct(v.rate) }))),
              td(Array.isArray(v.ci95) ? '[' + fmtPct(v.ci95[0], 0) + ', ' + fmtPct(v.ci95[1], 0) + ']' : '–', 'num'))), { compact: true })));
        }
        const latBlock = (title, lat) => (lat && Object.keys(lat).length ? h('div', null, h('h3', { class: 'small', text: title }),
          tableEl(['Component', { t: 'p50 ms', r: 1 }, { t: 'p95 ms', r: 1 }, { t: 'mean ms', r: 1 }, { t: 'n', r: 1 }],
            Object.entries(lat).map(([k, v]) => h('tr', null, td(k), td(fmtNum(v.p50), 'r num'), td(fmtNum(v.p95), 'r num'), td(fmtNum(v.mean), 'r num'), td(fmtInt(v.n), 'r num'))), { compact: true })) : null);
        if (s.latency_ms_per_call) {
          parts.push(h('div', { class: 'section cards' }, card('Latency per CALL_ACTION (batched)', latBlock('', s.latency_ms_per_call)),
            card('Latency per sample (amortised over the batch)', latBlock('', s.latency_ms_per_sample_amortised))));
        }
        // episodes
        const eps = d.episodes || [];
        const tasks = Array.from(new Set(eps.map((e) => e.task_id))).sort((a, b) => a - b);
        const fsel = h('select', { 'aria-label': 'Outcome filter' }, [['all', 'All outcomes'], ['success', 'Successes'], ['failure', 'Failures']].map(([v, t]) => h('option', { value: v, text: t })));
        fsel.value = epFilter;
        const tsel = h('select', { 'aria-label': 'Task filter' }, h('option', { value: 'all', text: 'All tasks' }), tasks.map((t) => h('option', { value: String(t), text: 'Task ' + t })));
        tsel.value = epTask;
        const epBox = h('div');
        const renderEps = () => {
          const rows = eps.filter((e) => (epFilter === 'all' || (epFilter === 'success') === e.success) && (epTask === 'all' || String(e.task_id) === epTask));
          setKids(epBox, rows.length ? tableEl(['Seed', 'Task', 'Init', 'Outcome', { t: 'Env steps', r: 1 }, { t: 'Calls', r: 1 }, { t: 'Call ms (mean)', r: 1 }, { t: 'Sim s', r: 1 }, 'Failure tags', 'Video', 'Trace'],
            rows.map((e) => h('tr', null, td(e.seed, 'num'), td(e.task_id, 'num'), td(e.init_state_idx, 'num'),
              td(h('span', { class: e.success ? 'ok' : 'bad', text: e.success ? '✓ success' : '✕ failure' })),
              td(fmtInt(e.env_steps), 'r num'), td(fmtInt(e.action_calls), 'r num'), td(fmtNum(e.call_ms_mean), 'r num'),
              td(isNum(e.sim_time_ms) ? fmtNum(e.sim_time_ms / 1000) : '–', 'r num'),
              td((e.failure_tags || []).length ? e.failure_tags.map((t) => h('span', { class: 'chip', text: t })) : '–'),
              td(e.video ? h('a', { href: fileUrl('results', name, e.video), target: '_blank', rel: 'noopener', text: '▶ video' }) : '–'),
              td(e.trace ? h('a', { href: fileUrl('results', name, e.trace), target: '_blank', rel: 'noopener', text: 'trace' }) : '–'))), { compact: true })
            : card(null, emptyState(eps.length ? 'No episodes match the filters.' : 'No episodes recorded yet.')));
        };
        fsel.addEventListener('change', () => { epFilter = fsel.value; renderEps(); });
        tsel.addEventListener('change', () => { epTask = tsel.value; renderEps(); });
        renderEps();
        parts.push(h('div', { class: 'section' }, h('h2', { text: 'Episodes (' + eps.length + ')' }), h('div', { class: 'toolbar' }, fsel, tsel), epBox));
        if (d.videos && d.videos.length) {
          parts.push(h('div', { class: 'section' }, h('h2', { text: 'Rollout videos (' + d.videos.length + ')' }), h('div', { class: 'videos' },
            d.videos.slice(0, 12).map((v) => h('figure', null, h('video', { controls: true, preload: 'none', src: fileUrl('results', name, v) }), h('figcaption', { text: v.split('/').pop() }))))));
        }
        const extras = [];
        if (d.offline) {
          extras.push(card('Open-loop metrics (offline.json)', tableEl(['Euler steps', { t: 'fm_loss', r: 1 }, { t: 'action_mse', r: 1 }, { t: 'first-action mse', r: 1 }, { t: 'gripper acc', r: 1 }],
            Object.entries(d.offline).map(([k, v]) => h('tr', null, td(k.replace('steps_', '')), td(fmtNum(v.fm_loss), 'r num'), td(fmtNum(v.action_mse), 'r num'), td(fmtNum(v.action_mse_first), 'r num'), td(fmtPct(v.gripper_acc), 'r num'))), { compact: true })));
        }
        if (d.latency && d.latency.profile) {
          extras.push(card('Batch-' + (d.latency.batch_size || 1) + ' latency profile (latency.json)', Object.entries(d.latency.profile).map(([k, v]) =>
            latBlock(k.replace('steps_', 'Euler steps: ') + (isNum(v.peak_mem_gb) ? ' · peak ' + fmtNum(v.peak_mem_gb) + ' GB' : ''), Object.fromEntries(Object.entries(v).filter(([kk]) => kk !== 'peak_mem_gb'))))));
        }
        if (extras.length) parts.push(h('div', { class: 'section cards' }, extras));
        if (d.logs && d.logs.length) {
          const lg = await api('api/eval/' + encodeURIComponent(name) + '/log?file=' + encodeURIComponent(d.logs[0].path) + '&tail=200');
          parts.push(h('div', { class: 'section' }, h('h2', { text: 'Log: ' + lg.path }), h('pre', { class: 'block', text: lg.lines.join('\n') || '(empty)' })));
        }
        if (d.eval_config) parts.push(h('details', { class: 'section' }, h('summary', { text: 'eval_config.json' }), h('pre', { class: 'block', text: JSON.stringify(d.eval_config, null, 2) })));
        setKids(body, ...parts);
      },
    };
  }

  // ---------------------------------------------------------------- compare
  function CompareView(selected) {
    let picker, grid, evalBox, note;
    let sel = selected.slice();
    const slotOf = (n) => {
      if (!S.compareSlots.has(n)) {
        const used = new Set(S.compareSlots.values());
        let k = 1; while (used.has(k) && k <= 8) k++;
        S.compareSlots.set(n, k <= 8 ? k : 8);
      }
      return S.compareSlots.get(n);
    };
    const setHash = () => { location.hash = '#/compare' + (sel.length ? '?runs=' + sel.map(encodeURIComponent).join(',') : ''); };
    return {
      mount(root) {
        picker = h('div', { class: 'run-pick' });
        note = h('p', { class: 'small muted' });
        evalBox = h('div');
        append(root, [pageHead('Compare runs', 'Overlay training / validation curves and the evaluations linked to each run (up to 8 runs).'),
          card('Runs', picker, note)]);
        const chartsHost = h('div', { class: 'section' });
        root.appendChild(chartsHost);
        grid = new ChartGrid(chartsHost);
        append(root, [h('div', { class: 'section' }, h('h2', { text: 'Linked evaluations' }), evalBox)]);
      },
      onRoute(r) { sel = r.runs || []; refresh(true); },
      async update() {
        for (const n of Array.from(S.compareSlots.keys())) if (!sel.includes(n)) S.compareSlots.delete(n);
        sel.forEach(slotOf);
        const runs = (await api('api/runs')).runs;
        setKids(picker, runs.length ? runs.map((r) => {
          const cb = h('input', { type: 'checkbox', value: r.name });
          cb.checked = sel.includes(r.name);
          cb.disabled = !cb.checked && sel.length >= 8;
          cb.addEventListener('change', () => { sel = cb.checked ? sel.concat([r.name]) : sel.filter((x) => x !== r.name); setHash(); });
          return h('label', null, cb, cb.checked ? h('span', { class: 'swatch b' + slotOf(r.name) }) : null,
            h('span', null, r.name, h('span', { class: 'muted small', text: ' ' + [r.variant, 'K=' + fmtVal(r.K)].filter(Boolean).join(' · ') })));
        }) : emptyState('No runs found.'));
        note.textContent = sel.length ? '' : 'Select runs to compare.';
        if (!sel.length) { grid.sync([]); setKids(evalBox, ); return; }
        const d = await api('api/compare?runs=' + sel.map(encodeURIComponent).join(','));
        const per = (key, ycol, transform) => d.runs.map((r) => {
          const rows = (r.series[key] || []);
          let pts = col(rows, 'step', ycol);
          if (transform) pts = transform(pts);
          return { name: r.name, slot: slotOf(r.name), pts };
        });
        const specs = [{ id: 'c_loss', title: 'Training loss (EMA)', series: per('train_log', 'loss', (p) => ema(p, emaAlpha(p.length))) }];
        const anyRows = (key) => d.runs.some((r) => (r.series[key] || []).length);
        if (anyRows('val_log')) {
          specs.push({ id: 'c_val_fm', title: 'Held-out fm_loss (fixed noise)', series: per('val_log', 'fm_loss') });
          specs.push({ id: 'c_val_mse', title: 'Held-out sampled action_mse', series: per('val_log', 'action_mse') });
        }
        if (anyRows('overfit_log')) specs.push({ id: 'c_ovf', title: 'Overfit windows fm_loss', series: per('overfit_log', 'fm_loss') });
        if (anyRows('sim_eval_log')) specs.push({ id: 'c_sim', title: 'Periodic simulator success', logToggle: false, yMin: 0, yMax: 1, yFormat: (v) => fmtPct(v, 0), yFormatTip: (v) => fmtPct(v), series: per('sim_eval_log', 'success_rate') });
        grid.sync(specs);
        setKids(evalBox, d.evals.length ? tableEl(EVAL_HEADERS, d.evals.map(evalRow))
          : card(null, emptyState('No evaluations linked to the selected runs.', 'Evaluations are linked through eval_config.json → model_config.train.output_dir.')));
      },
    };
  }

  // ---------------------------------------------------------------- audit & reports
  function AuditView(path) {
    let list, viewer, cur = path || null;
    const show = async () => {
      if (!cur) { setKids(viewer, emptyState('Select a file.')); return; }
      const ext = cur.slice(cur.lastIndexOf('.')).toLowerCase();
      const url = fileUrl('results', '', cur);
      const headRow = h('div', { class: 'toolbar' }, h('h2', { text: cur, style: { margin: '0' } }), h('a', { class: 'btn sm', href: fileUrl('results', '', cur, true), text: 'Download', download: '' }));
      if (['.png', '.jpg', '.jpeg', '.gif', '.svg'].includes(ext)) { setKids(viewer, headRow, h('img', { src: url, alt: cur })); return; }
      let text = await fetchText(url);
      if (ext === '.json') {
        // Python's json writes NaN / Infinity; make it parseable for pretty-printing (raw text on failure)
        try { text = JSON.stringify(JSON.parse(text.replace(/\bNaN\b/g, 'null').replace(/-?\bInfinity\b/g, (m) => '"' + m + '"')), null, 2); } catch (e) { /* show raw */ }
      }
      setKids(viewer, headRow, h('pre', { class: 'block', style: { maxHeight: '75vh' }, text }));
    };
    return {
      mount(root) {
        list = h('div', { class: 'filelist' }); viewer = h('div', { class: 'viewer' });
        append(root, [pageHead('Audit & reports', 'Files under results/audit and top-level reports in results/ (e.g. phase1_report.md).'),
          h('div', { class: 'audit-layout' }, card(null, list), card(null, viewer))]);
      },
      onRoute(r) { cur = r.path || null; show(); refresh(true); },
      async update() {
        const d = await api('api/audit');
        const items = d.reports.concat(d.audit);
        if (!cur && items.length) cur = (items.find((f) => f.ext === '.md') || items[0]).path;
        setKids(list, items.length ? [
          d.reports.length ? h('div', { class: 'small muted', text: 'Reports' }) : null,
          d.reports.map((f) => fileLink(f)),
          d.audit.length ? h('div', { class: 'small muted', style: { marginTop: '8px' }, text: 'Audit' }) : null,
          d.audit.map((f) => fileLink(f))] : emptyState('No audit files yet.', 'python -m monet_action.audit writes results/audit/.'));
        if (!viewer.childNodes.length) await show();
      },
    };
    function fileLink(f) {
      return h('a', { href: '#/audit/' + encodeURIComponent(f.path), class: f.path === cur ? 'active' : null },
        f.path, h('span', { class: 'muted small', text: '  ' + fmtBytes(f.size) }));
    }
  }

  // ================================================================== router & refresh loop
  function parseRoute() {
    const raw = location.hash.replace(/^#\/?/, '');
    const qi = raw.indexOf('?');
    const pathPart = qi >= 0 ? raw.slice(0, qi) : raw;
    const q = new URLSearchParams(qi >= 0 ? raw.slice(qi + 1) : '');
    const segs = pathPart.split('/').filter(Boolean).map((x) => { try { return decodeURIComponent(x); } catch (e) { return x; } });
    const page = segs[0] || 'runs';
    if (page === 'run' && segs[1]) return { page, name: segs[1], tab: RUN_TABS.some(([k]) => k === segs[2]) ? segs[2] : 'overview' };
    if (page === 'eval' && segs[1]) return { page, name: segs[1] };
    if (page === 'compare') return { page, runs: (q.get('runs') || '').split(',').filter(Boolean) };
    if (page === 'audit') return { page, path: segs[1] || null };
    if (['runs', 'evals'].includes(page)) return { page };
    return { page: 'runs' };
  }
  function makeView(r) {
    switch (r.page) {
      case 'run': return RunView(r.name, r.tab);
      case 'evals': return EvalsView();
      case 'eval': return EvalView(r.name);
      case 'compare': return CompareView(r.runs);
      case 'audit': return AuditView(r.path);
      default: return RunsView();
    }
  }
  function routeTo() {
    const r = parseRoute();
    const key = r.page + '|' + (r.name || '');
    const navKey = r.page === 'run' ? 'runs' : r.page === 'eval' ? 'evals' : r.page;
    document.querySelectorAll('#nav a').forEach((a) => a.classList.toggle('active', a.dataset.nav === navKey));
    document.title = (r.name ? r.name + ' · ' : '') + 'Monet-Action Monitor';
    hideTooltip();
    if (S.view && S.routeKey === key && S.view.onRoute) { S.view.onRoute(r); return; }
    S.charts.clear();
    const app = document.getElementById('app');
    setKids(app, );
    S.routeKey = key;
    S.view = makeView(r);
    S.view.mount(app);
    refresh(true);
  }
  async function refresh(manual) {
    if (!S.view) return;
    if (!manual && document.hidden) return;
    if (S.busy) { S.pending = true; return; }
    S.busy = true;
    const view = S.view;
    try {
      await view.update();
      S.lastOk = Date.now();
      setBanner(null);
    } catch (e) {
      console.error(e);
      setBanner(e.status === 401 ? 'Authentication required: open the dashboard URL with ?token=<your token>.'
        : e.status === 404 ? 'Not found: ' + e.message
          : 'Cannot reach the dashboard server (' + e.message + '). Retrying on the next refresh.');
    } finally {
      S.busy = false;
      tickUpdated();
      if (S.pending) { S.pending = false; refresh(true); }
    }
  }
  function tickUpdated() {
    const el = document.getElementById('lastUpdated');
    if (el) el.textContent = S.lastOk ? 'updated ' + fmtAgo(S.lastOk / 1000) : '';
  }
  function schedule() {
    clearInterval(S.timer);
    S.timer = S.auto ? setInterval(() => refresh(false), S.refreshSec * 1000) : null;
  }
  function applyTheme(t) {
    S.theme = t;
    if (t === 'auto') delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = t;
    document.getElementById('themeToggle').textContent = 'Theme: ' + t;
    store.set('theme', t);
  }

  async function init() {
    applyTheme(store.get('theme', 'auto'));
    document.getElementById('themeToggle').addEventListener('click', () => {
      applyTheme({ auto: 'light', light: 'dark', dark: 'auto' }[S.theme] || 'auto');
    });
    const auto = document.getElementById('autoRefresh');
    S.auto = store.get('auto', true);
    auto.checked = S.auto;
    auto.addEventListener('change', () => { S.auto = auto.checked; store.set('auto', S.auto); schedule(); });
    document.getElementById('refreshNow').addEventListener('click', () => refresh(true));
    document.getElementById('bundleAll').href = withToken('api/bundle/all');
    try {
      S.cfg = await api('api/config');
      S.refreshSec = Math.max(2, S.cfg.refresh || 10);
    } catch (e) {
      setBanner(e.status === 401 ? 'Authentication required: open the dashboard URL with ?token=<your token>.' : 'Cannot reach the dashboard server.');
    }
    document.getElementById('refreshEvery').textContent = '(' + S.refreshSec + ' s)';
    window.addEventListener('hashchange', routeTo);
    window.addEventListener('resize', debounce(() => S.charts.forEach((c) => c.render()), 150));
    document.addEventListener('visibilitychange', () => { if (!document.hidden) refresh(false); });
    routeTo();
    schedule();
    setInterval(tickUpdated, 1000);
  }
  init();
})();
