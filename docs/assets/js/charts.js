/**
 * Chart.js (4.4.1, loaded as a global from cdnjs) theming and shared plugins.
 * Colours come from the CSS tokens in style.css, so light/dark mode stay in sync with the page.
 */
import { AQI_LEVELS, hexToRgba, prefersReducedMotion } from "./common.js";

export const hasChart = () => typeof window.Chart === "function";

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** Current chart palette (read from CSS custom properties). */
export function theme() {
  return {
    airtrace: cssVar("--c-airtrace"),
    cams: cssVar("--c-cams"),
    obs: cssVar("--c-obs"),
    up: cssVar("--c-up"),
    down: cssVar("--c-down"),
    grid: cssVar("--c-grid"),
    axis: cssVar("--c-axis"),
    tick: cssVar("--c-tick"),
    text: cssVar("--text"),
    text2: cssVar("--text-2"),
    muted: cssVar("--muted"),
    surface: cssVar("--surface"),
    surface2: cssVar("--surface-2"),
    accent: cssVar("--accent"),
    tooltipBg: cssVar("--c-tooltip-bg"),
    tooltipText: cssVar("--c-tooltip-text"),
    bandAlpha: parseFloat(cssVar("--band-alpha")) || 0.13,
    fire: cssVar("--c-fire"),
    fireUpwind: cssVar("--c-fire-upwind"),
  };
}

/** Global Chart.js defaults matching the site (call again after a colour-scheme change). */
export function applyChartDefaults() {
  if (!hasChart()) return;
  const C = window.Chart;
  const t = theme();
  C.defaults.font.family = getComputedStyle(document.body).fontFamily;
  C.defaults.font.size = 12;
  C.defaults.color = t.tick;
  C.defaults.borderColor = t.grid;
  C.defaults.maintainAspectRatio = false;
  C.defaults.responsive = true;
  C.defaults.animation = prefersReducedMotion() ? false : { duration: 450 };
  C.defaults.plugins.legend.display = false; // series keys are rendered in HTML next to each chart
  const tt = C.defaults.plugins.tooltip;
  tt.backgroundColor = t.tooltipBg;
  tt.titleColor = t.tooltipText;
  tt.bodyColor = t.tooltipText;
  tt.footerColor = t.tooltipText;
  tt.padding = 10;
  tt.cornerRadius = 8;
  tt.boxPadding = 4;
  tt.usePointStyle = true;
  tt.titleFont = { weight: "600" };
  tt.footerFont = { weight: "400", size: 11 };
  C.defaults.elements.line.borderWidth = 2;
  C.defaults.elements.line.borderCapStyle = "round";
  C.defaults.elements.line.borderJoinStyle = "round";
  C.defaults.elements.point.radius = 0;
  C.defaults.elements.point.hoverRadius = 4;
  C.defaults.elements.bar.borderRadius = 4;
}

/** Re-run `cb` whenever the OS colour scheme flips (after defaults are refreshed). */
export function onSchemeChange(cb) {
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  const handler = () => { applyChartDefaults(); cb(mq.matches); };
  if (mq.addEventListener) mq.addEventListener("change", handler);
  else if (mq.addListener) mq.addListener(handler);
}

/** Cartesian axis config with the recessive grid used everywhere. */
export function axis(extra = {}) {
  const t = theme();
  const { grid, ticks, border, title, ...rest } = extra;
  return {
    grid: { color: t.grid, drawTicks: false, ...grid },
    border: { color: t.axis, ...border },
    ticks: { color: t.tick, padding: 6, ...ticks },
    title: title ? { display: true, color: t.muted, font: { size: 11, weight: "600" }, ...title } : undefined,
    ...rest,
  };
}

/** Destroy and forget every chart in a Set. */
export function destroyAll(set) {
  for (const c of set) { try { c.destroy(); } catch { /* already gone */ } }
  set.clear();
}

/** Create a chart and register it in `registry` (a Set) so it can be destroyed later. */
export function makeChart(canvas, config, registry) {
  if (!hasChart() || !canvas) return null;
  const chart = new window.Chart(canvas, config);
  registry?.add(chart);
  return chart;
}

/**
 * Plugin: subtle horizontal AQI-category bands behind a PM2.5 (ug/m3) y-axis, labelled at the
 * right edge when the band is tall enough. Options: {display, alpha}.
 */
export const aqiBandsPlugin = {
  id: "aqiBands",
  beforeDatasetsDraw(chart, _args, opts) {
    if (opts?.display === false) return;
    const { ctx, chartArea: a, scales: { y } } = chart;
    if (!y || !a) return;
    const t = theme();
    const alpha = opts?.alpha ?? t.bandAlpha;
    ctx.save();
    let lo = 0;
    for (const lv of AQI_LEVELS) {
      const hi = lv.pmHi;
      if (hi > y.min && lo < y.max) {
        const top = y.getPixelForValue(Math.min(hi, y.max));
        const bottom = y.getPixelForValue(Math.max(lo, y.min));
        ctx.fillStyle = hexToRgba(lv.color, alpha);
        ctx.fillRect(a.left, top, a.right - a.left, bottom - top);
        if (bottom - top >= 16) {
          ctx.fillStyle = t.muted;
          ctx.font = `600 10px ${getComputedStyle(document.body).fontFamily}`;
          ctx.textAlign = "right";
          ctx.textBaseline = "top";
          ctx.fillText(lv.short, a.right - 6, top + 4);
        }
      }
      lo = hi;
    }
    ctx.restore();
  },
};

/** Plugin: shade an x-index range (e.g. the selected forecast day). Options: {from, to, color}. */
export const rangeHighlightPlugin = {
  id: "rangeHighlight",
  beforeDatasetsDraw(chart, _args, opts) {
    if (!opts || opts.from == null || opts.to == null || opts.from < 0) return;
    const { ctx, chartArea: a, scales: { x } } = chart;
    if (!x) return;
    const half = (x.getPixelForValue(1) - x.getPixelForValue(0)) / 2 || 0;
    const left = Math.max(a.left, x.getPixelForValue(opts.from) - half);
    const right = Math.min(a.right, x.getPixelForValue(opts.to) + half);
    ctx.save();
    ctx.fillStyle = opts.color || "rgba(15, 118, 110, 0.07)";
    ctx.fillRect(left, a.top, right - left, a.bottom - a.top);
    ctx.restore();
  },
};

/** Plugin: vertical crosshair at the active tooltip x (line charts in "index" mode). */
export const crosshairPlugin = {
  id: "crosshair",
  afterDatasetsDraw(chart, _args, opts) {
    const active = chart.tooltip?.getActiveElements?.() || [];
    if (!active.length) return;
    const { ctx, chartArea: a } = chart;
    const x = active[0].element.x;
    ctx.save();
    ctx.strokeStyle = opts?.color || theme().axis;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, a.top);
    ctx.lineTo(x, a.bottom);
    ctx.stroke();
    ctx.restore();
  },
};

/** Plugin: emphasised zero line on a value axis (for signed bars). Options: {axis: "x"|"y"}. */
export const zeroLinePlugin = {
  id: "zeroLine",
  afterDatasetsDraw(chart, _args, opts) {
    const id = opts?.axis || "x";
    const s = chart.scales[id];
    if (!s || s.min > 0 || s.max < 0) return;
    const { ctx, chartArea: a } = chart;
    const p = s.getPixelForValue(0);
    ctx.save();
    ctx.strokeStyle = theme().muted;
    ctx.lineWidth = 1;
    ctx.beginPath();
    if (id === "x") { ctx.moveTo(p, a.top); ctx.lineTo(p, a.bottom); } else { ctx.moveTo(a.left, p); ctx.lineTo(a.right, p); }
    ctx.stroke();
    ctx.restore();
  },
};
