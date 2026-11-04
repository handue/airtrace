/**
 * AirTrace model card (methods.html).
 *
 * Renders every metric from metrics.json (ML pipeline) and sources_global.json (source model).
 * Nothing numeric is hard-coded: a missing block shows an "unavailable" note instead.
 */
import {
  AQI_LEVELS, REFRESH_HOURS, SOURCE_FACTORS, asPercent, emptyState, fmtDate, fmtInt, fmtNum, fmtPct,
  fmtSigned, h, initChrome, initUpdated, inkOn, hexToRgba, loadAll, tableView,
} from "./common.js";
import { applyChartDefaults, axis, destroyAll, hasChart, makeChart, onSchemeChange, theme } from "./charts.js";

const $ = (id) => document.getElementById(id);
const finite = (v) => v != null && v !== "" && Number.isFinite(+v);
const num = (v) => (finite(v) ? +v : null);

const state = { meta: null, metrics: null, src: null, charts: new Set(), scatterCharts: new Set(), siteMap: null, siteLayer: null, siteTiles: null };

const MODEL_NOTES = {
  cams_raw: "Raw Copernicus CAMS forecast: the global baseline AirTrace corrects.",
  linear: "Linear regression baseline: a simple statistical correction.",
  ml_no_cams: "Same gradient-boosting model without any CAMS input: what does the global model’s chemistry add?",
  airtrace: "LightGBM on CAMS chemistry, weather, season and location.",
};

const SHORT_CAT = {
  Good: "Good", Moderate: "Moderate", "Unhealthy for Sensitive Groups": "USG", Unhealthy: "Unhealthy",
  "Very Unhealthy": "Very unh.", Hazardous: "Hazard.",
};

/* ================================================================== boot */

async function main() {
  initChrome();
  applyChartDefaults();
  document.querySelectorAll('[data-const="REFRESH_HOURS"]').forEach((el) => { el.textContent = String(REFRESH_HOURS); });
  renderFoldDemo();

  const d = await loadAll(["meta", "metrics", "sources_global"]);
  state.meta = d.meta;
  state.metrics = d.metrics;
  state.src = d.sources_global;
  initUpdated(state.meta);
  renderAll();
  onSchemeChange(() => {
    renderAll();
    state.siteTiles?.setUrl(tileUrl());
  });
}

function renderAll() {
  destroyAll(state.charts);
  const steps = [renderHeadline, renderValidation, renderModels, renderScatter, renderRegionMae, renderConfusion,
    renderSiteMap, renderImportance, renderInterval, renderSources, renderDataset];
  for (const step of steps) {
    try { step(); } catch (err) { console.error(`[AirTrace] ${step.name} failed`, err); }
  }
}

main().catch((err) => console.error("[AirTrace] model card failed to initialise", err));

/* ================================================================== helpers */

function unavailable(el, title, what) {
  if (!el) return;
  el.replaceChildren(...(title ? [h("h3", null, title)] : []), emptyState(`${what} unavailable — the pipeline has not published this block yet.`));
}

function head(title, sub) {
  return h("div", { class: "card-head" }, h("div", null, h("h3", null, title), sub ? h("p", { class: "card-sub" }, sub) : null));
}

function seriesKey(items) {
  return h("div", { class: "series-key", "aria-hidden": "true" }, items.map((it) =>
    h("span", null, h("i", { class: it.type === "line" ? "k-line" : "k-rect", style: it.type === "line" ? { color: it.color } : { background: it.color } }), it.label)));
}

function statTile(value, label, cls = "") {
  return h("div", { class: `stat ${cls}`.trim() }, h("div", { class: "v" }, value), h("div", { class: "l" }, label));
}

function kpiTile(value, label, foot, alt) {
  return h("div", { class: `card kpi${alt ? " alt" : ""}` },
    h("div", { class: "kpi-value" }, value), h("div", { class: "kpi-label" }, label),
    foot ? h("div", { class: "kpi-foot" }, foot) : null);
}

function wrapLabel(s, max = 18) {
  const str = String(s);
  if (str.length <= max) return str;
  const mid = str.lastIndexOf(" ", Math.ceil(str.length / 2) + 4);
  return mid > 0 ? [str.slice(0, mid), str.slice(mid + 1)] : str;
}

const isDark = () => window.matchMedia("(prefers-color-scheme: dark)").matches;
const tileUrl = () => `https://{s}.basemaps.cartocdn.com/${isDark() ? "dark_all" : "light_all"}/{z}/{x}/{y}{r}.png`;

/* ================================================================== headline + validation */

function renderHeadline() {
  const box = $("headline");
  box.removeAttribute("aria-busy");
  const hl = state.metrics?.headline;
  if (!hl) {
    box.replaceChildren(h("div", { class: "card", style: { gridColumn: "1 / -1" } }, emptyState("Headline metrics unavailable — the pipeline has not published metrics.json yet.")));
    return;
  }
  const tiles = [];
  if (finite(hl.mae_reduction_pct)) {
    const r = +hl.mae_reduction_pct;
    tiles.push(kpiTile(`${r >= 0 ? "−" : "+"}${fmtNum(Math.abs(r), 0)}%`, `${r >= 0 ? "lower" : "higher"} mean absolute error than raw CAMS`,
      finite(hl.rmse_reduction_pct) ? `RMSE ${fmtSigned(-hl.rmse_reduction_pct, 0)}%` : null));
  }
  if (finite(hl.aqi_acc_airtrace)) {
    tiles.push(kpiTile(fmtPct(hl.aqi_acc_airtrace), "of days in the right AQI category", finite(hl.aqi_acc_cams) ? `CAMS: ${fmtPct(hl.aqi_acc_cams)}` : null, true));
  }
  if (finite(hl.unhealthy_recall_airtrace)) {
    tiles.push(kpiTile(fmtPct(hl.unhealthy_recall_airtrace), "of unhealthy days (AQI above 100) flagged", finite(hl.unhealthy_recall_cams) ? `CAMS: ${fmtPct(hl.unhealthy_recall_cams)}` : null));
  }
  if (finite(hl.r2_airtrace)) {
    tiles.push(kpiTile(fmtNum(hl.r2_airtrace, 2), "R² — share of day-to-day variation explained", finite(hl.r2_cams) ? `CAMS: ${fmtNum(hl.r2_cams, 2)}` : null, true));
  }
  box.replaceChildren(...tiles);
}

function renderFoldDemo() {
  const box = $("fold-demo");
  if (!box) return;
  const test = new Set([2, 7, 13, 18, 24, 29, 31, 38]);
  box.replaceChildren(...Array.from({ length: 40 }, (_v, i) => h("span", { class: test.has(i) ? "test" : "train" })));
}

function renderValidation() {
  const v = state.metrics?.validation;
  $("scheme-text").textContent = v?.scheme ? `Pipeline setting: ${v.scheme}` : "";
  $("temporal-text").textContent = v?.temporal_scheme ? `Pipeline setting: ${v.temporal_scheme}` : "";
  const T = state.metrics?.temporal_holdout;
  const box = $("temporal-table");
  if (!T?.cams_raw || !T?.airtrace) { box.replaceChildren(emptyState("Temporal-holdout metrics unavailable.")); return; }
  const rows = [
    ["Mean absolute error (µg/m³)", "mae", "low", (x) => fmtNum(x, 1)],
    ["RMSE (µg/m³)", "rmse", "low", (x) => fmtNum(x, 1)],
    ["Bias (µg/m³)", "bias", "abs", (x) => fmtSigned(x, 1)],
    ["R²", "r2", "high", (x) => fmtNum(x, 2)],
    ["AQI category accuracy", "aqi_category_accuracy", "high", (x) => fmtPct(x)],
    ["Unhealthy-day recall", "unhealthy_recall", "high", (x) => fmtPct(x)],
  ].filter(([, k]) => finite(T.cams_raw[k]) || finite(T.airtrace[k]));
  box.replaceChildren(h("div", { class: "table-scroll" }, h("table", { class: "data" },
    h("caption", { class: "sr-only" }, "Temporal holdout metrics"),
    h("thead", null, h("tr", null, h("th", { scope: "col" }, "Metric"), h("th", { scope: "col" }, "CAMS raw"), h("th", { scope: "col" }, "AirTrace"))),
    h("tbody", null, rows.map(([label, k, better, f]) => {
      const a = num(T.cams_raw[k]);
      const b = num(T.airtrace[k]);
      const bWins = a != null && b != null && (better === "low" ? b < a : better === "abs" ? Math.abs(b) < Math.abs(a) : b > a);
      return h("tr", null, h("th", { scope: "row" }, label),
        h("td", { class: !bWins && a != null && b != null && a !== b ? "best" : "" }, f(a)),
        h("td", { class: bWins ? "best" : "" }, f(b)));
    })))));
}

/* ================================================================== model comparison */

const COLS = [
  { key: "mae", label: "MAE", better: "low", f: (v) => fmtNum(v, 1) },
  { key: "rmse", label: "RMSE", better: "low", f: (v) => fmtNum(v, 1) },
  { key: "r2", label: "R²", better: "high", f: (v) => fmtNum(v, 2) },
  { key: "bias", label: "Bias", better: "abs", f: (v) => fmtSigned(v, 1) },
  { key: "r", label: "r", better: "high", f: (v) => fmtNum(v, 2) },
  { key: "aqi_category_accuracy", label: "AQI category acc.", better: "high", f: (v) => fmtPct(v) },
  { key: "unhealthy_precision", label: "Unhealthy precision", better: "high", f: (v) => fmtPct(v) },
  { key: "unhealthy_recall", label: "Unhealthy recall", better: "high", f: (v) => fmtPct(v) },
  { key: "unhealthy_f1", label: "Unhealthy F1", better: "high", f: (v) => fmtPct(v) },
];

function renderModels() {
  const box = $("models-card");
  const models = (state.metrics?.models || []).filter((m) => m && m.key);
  if (!models.length) { unavailable(box, null, "Model comparison"); return; }
  const best = {};
  for (const c of COLS) {
    const vals = models.map((m) => num(m[c.key])).filter((v) => v != null);
    if (!vals.length) continue;
    best[c.key] = c.better === "low" ? Math.min(...vals) : c.better === "abs" ? Math.min(...vals.map(Math.abs)) : Math.max(...vals);
  }
  const isBest = (c, v) => v != null && best[c.key] != null && (c.better === "abs" ? Math.abs(v) === best[c.key] : v === best[c.key]);
  const t = theme();
  const canvas = h("canvas", { role: "img", "aria-label": `Bar chart of mean absolute error by model: ${models.map((m) => `${m.name} ${fmtNum(m.mae)}`).join(", ")}` });

  box.replaceChildren(
    head("Mean absolute error by model", "Lower is better. Same held-out locations for every model."),
    h("div", { class: "chart-box", style: { height: `${70 + models.length * 44}px` } }, canvas),
    h("div", { class: "table-scroll", style: { marginTop: "16px" } }, h("table", { class: "data" },
      h("caption", { class: "sr-only" }, "Leave-location-out cross-validation metrics for every model"),
      h("thead", null, h("tr", null, h("th", { scope: "col" }, "Model"), COLS.map((c) => h("th", { scope: "col" }, c.label)))),
      h("tbody", null, models.map((m) => h("tr", { class: m.key === "airtrace" ? "is-ours" : "" },
        h("th", { scope: "row", style: { minWidth: "220px" } },
          h("div", { style: { display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" } },
            h("span", null, m.name || m.key), m.key === "airtrace" ? h("span", { class: "badge badge-ours" }, "AirTrace") : null),
          MODEL_NOTES[m.key] ? h("div", { class: "small muted", style: { fontWeight: 400 } }, MODEL_NOTES[m.key]) : null),
        COLS.map((c) => h("td", { class: isBest(c, num(m[c.key])) ? "best" : "" }, c.f(m[c.key])))))))),
    h("p", { class: "note" }, "MAE, RMSE and bias in µg/m³. “Unhealthy” = AQI above 100 (24-hour PM2.5 ≥ 35.5 µg/m³): precision = share of flagged days that really were unhealthy; recall = share of unhealthy days that were flagged."));

  if (!hasChart()) return;
  makeChart(canvas, {
    type: "bar",
    data: {
      labels: models.map((m) => m.name || m.key),
      datasets: [{
        data: models.map((m) => num(m.mae)),
        backgroundColor: models.map((m) => (m.key === "airtrace" ? t.airtrace : m.key === "cams_raw" ? t.cams : neutral())),
        borderRadius: 4, maxBarThickness: 22, borderSkipped: "start",
      }],
    },
    options: {
      indexAxis: "y",
      scales: {
        x: axis({ beginAtZero: true, title: { text: "Mean absolute error (µg/m³)" } }),
        y: axis({ grid: { display: false }, ticks: { callback: (_v, i) => wrapLabel(models[i].name || models[i].key, 26) } }),
      },
      plugins: { tooltip: { callbacks: { label: (it) => ` MAE ${fmtNum(it.raw)} µg/m³` } } },
    },
  }, state.charts);
}

const neutral = () => (isDark() ? "#4b5b76" : "#a8b3c2");

/* ================================================================== scatter */

let scatterWired = false;

function renderScatter() {
  const S = state.metrics?.scatter;
  const pair = $("scatter-pair");
  const sel = $("scatter-region");
  if (!S || !Array.isArray(S.obs) || !S.obs.length) {
    pair.replaceChildren(emptyState("Cross-validation sample unavailable."));
    sel.disabled = true;
    return;
  }
  if (!scatterWired) {
    const regions = [...new Set((S.region || []).filter(Boolean))].sort();
    sel.append(...regions.map((r) => h("option", { value: r }, r)));
    sel.addEventListener("change", () => drawScatter(sel.value));
    scatterWired = true;
  }
  drawScatter(sel.value);
}

function scatterStats(region, key) {
  const m = state.metrics;
  if (region === "all") {
    const row = (m.models || []).find((x) => x.key === (key === "cams" ? "cams_raw" : "airtrace"));
    return row ? `All held-out days · MAE ${fmtNum(row.mae)} · R² ${fmtNum(row.r2, 2)} · bias ${fmtSigned(row.bias, 1)} µg/m³` : "";
  }
  const r = (m.by_region || []).find((x) => x.region === region);
  if (!r) return "";
  const k = key === "cams" ? "cams" : "airtrace";
  return `${region} · MAE ${fmtNum(r[`mae_${k}`])} · RMSE ${fmtNum(r[`rmse_${k}`])} · bias ${fmtSigned(r[`bias_${k}`], 1)} µg/m³`;
}

function drawScatter(region) {
  destroyAll(state.scatterCharts);
  const S = state.metrics.scatter;
  const pair = $("scatter-pair");
  const t = theme();
  const idx = [];
  for (let i = 0; i < S.obs.length; i++) {
    if (region !== "all" && S.region?.[i] !== region) continue;
    if (!(num(S.obs[i]) > 0)) continue;
    idx.push(i);
  }
  if (!idx.length) { pair.replaceChildren(emptyState("No sampled points for this region.")); return; }
  const all = idx.flatMap((i) => [S.obs[i], S.cams?.[i], S.airtrace?.[i]]).map(num).filter((v) => v > 0);
  const lo = 10 ** Math.floor(Math.log10(Math.max(0.3, Math.min(...all))));
  const hi = 10 ** Math.ceil(Math.log10(Math.max(...all)));
  const panels = [
    { key: "cams", title: "CAMS global model (raw)", color: t.cams },
    { key: "airtrace", title: "AirTrace", color: t.airtrace },
  ];
  const nodes = panels.map((p) => {
    const canvas = h("canvas", { role: "img", "aria-label": `Scatter plot of observed versus ${p.title} PM2.5 on log axes${region === "all" ? "" : ` for ${region}`}.` });
    p.canvas = canvas;
    return h("div", null,
      h("div", { class: "scatter-title" }, h("i", { style: { background: p.color }, "aria-hidden": "true" }), p.title),
      h("div", { class: "scatter-stats" }, scatterStats(region, p.key)),
      h("div", { class: "chart-box" }, canvas));
  });
  pair.replaceChildren(...nodes);
  if (!hasChart()) return;
  const nice = new Set([0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]);
  const major = new Set([0.1, 1, 10, 100, 1000]);
  const ref = (k, color) => ({
    type: "scatter", data: [{ x: lo, y: lo * k }, { x: hi, y: hi * k }], showLine: true, borderColor: color,
    borderWidth: k === 1 ? 1.25 : 0.75, pointRadius: 0, pointHitRadius: 0, order: 0,
  });
  for (const p of panels) {
    const pts = idx.map((i) => ({ x: num(S.obs[i]), y: num(S[p.key]?.[i]), r: S.region?.[i] })).filter((q) => q.y > 0);
    makeChart(p.canvas, {
      type: "scatter",
      data: {
        datasets: [
          { label: p.title, data: pts, backgroundColor: hexToRgba(p.color, 0.38), borderColor: "transparent", pointRadius: 2.4, pointHoverRadius: 5, pointHitRadius: 4, order: 1 },
          ref(1, t.muted), ref(2, t.axis), ref(0.5, t.axis),
        ],
      },
      options: {
        animation: false,
        scales: {
          x: axis({ type: "logarithmic", min: lo, max: hi, title: { text: "Observed PM2.5 (µg/m³)" },
            grid: { color: (ctx) => (major.has(ctx.tick?.value) ? t.grid : "transparent") },
            ticks: { autoSkip: false, maxRotation: 0, callback: (v) => (nice.has(+v) ? fmtNum(v, v < 1 ? 1 : 0) : "") } }),
          y: axis({ type: "logarithmic", min: lo, max: hi, title: { text: "Predicted PM2.5 (µg/m³)" },
            grid: { color: (ctx) => (major.has(ctx.tick?.value) ? t.grid : "transparent") },
            ticks: { autoSkip: false, callback: (v) => (nice.has(+v) ? fmtNum(v, v < 1 ? 1 : 0) : "") } }),
        },
        plugins: {
          tooltip: {
            filter: (it) => it.datasetIndex === 0,
            callbacks: {
              title: (items) => items[0]?.raw?.r || "",
              label: (it) => ` Observed ${fmtNum(it.raw.x)} · predicted ${fmtNum(it.raw.y)} µg/m³`,
            },
          },
        },
      },
    }, state.scatterCharts);
  }
}

/* ================================================================== regions + confusion */

function renderRegionMae() {
  const box = $("region-card");
  const rows = (state.metrics?.by_region || []).filter((r) => r && r.region && finite(r.mae_cams) && finite(r.mae_airtrace))
    .sort((a, b) => +b.mae_cams - +a.mae_cams);
  if (!rows.length) { unavailable(box, "Error by region", "Regional metrics"); return; }
  const t = theme();
  const canvas = h("canvas", { role: "img", "aria-label": "Grouped bar chart of mean absolute error by region for CAMS and AirTrace." });
  box.replaceChildren(
    head("Mean absolute error by region", "Training sites per region in brackets. Leave-location-out cross-validation."),
    seriesKey([{ label: "CAMS raw", color: t.cams }, { label: "AirTrace", color: t.airtrace }]),
    h("div", { class: "chart-box", style: { height: `${Math.max(280, 50 + rows.length * 40)}px` } }, canvas),
    tableView("Error by region (µg/m³)", [
      { label: "Region", value: (r) => r.region },
      { label: "Sites", value: (r) => fmtInt(r.n_sites) },
      { label: "CAMS MAE", value: (r) => fmtNum(r.mae_cams) },
      { label: "AirTrace MAE", value: (r) => fmtNum(r.mae_airtrace) },
      { label: "CAMS RMSE", value: (r) => fmtNum(r.rmse_cams) },
      { label: "AirTrace RMSE", value: (r) => fmtNum(r.rmse_airtrace) },
      { label: "CAMS bias", value: (r) => fmtSigned(r.bias_cams, 1) },
      { label: "AirTrace bias", value: (r) => fmtSigned(r.bias_airtrace, 1) },
    ], rows));
  if (!hasChart()) return;
  makeChart(canvas, {
    type: "bar",
    data: {
      labels: rows.map((r) => `${r.region}${finite(r.n_sites) ? ` (${r.n_sites})` : ""}`),
      datasets: [
        { label: "CAMS raw", data: rows.map((r) => +r.mae_cams), backgroundColor: t.cams, borderRadius: 4, maxBarThickness: 14, borderSkipped: "start" },
        { label: "AirTrace", data: rows.map((r) => +r.mae_airtrace), backgroundColor: t.airtrace, borderRadius: 4, maxBarThickness: 14, borderSkipped: "start" },
      ],
    },
    options: {
      indexAxis: "y",
      scales: {
        x: axis({ beginAtZero: true, title: { text: "Mean absolute error (µg/m³)" } }),
        y: axis({ grid: { display: false }, ticks: { autoSkip: false, callback: (_v, i) => wrapLabel(`${rows[i].region}${finite(rows[i].n_sites) ? ` (${rows[i].n_sites})` : ""}`) } }),
      },
      plugins: { tooltip: { callbacks: { label: (it) => ` ${it.dataset.label}: ${fmtNum(it.raw)} µg/m³` } } },
    },
  }, state.charts);
}

function heatTable(labels, M, caption) {
  const ink = isDark() ? "#0b1120" : "#ffffff";
  const rgb = getComputedStyle(document.documentElement).getPropertyValue("--heat-rgb").trim() || "15, 118, 110";
  return h("div", { class: "table-scroll" }, h("table", { class: "heat" },
    h("caption", { class: "sr-only" }, caption),
    h("thead", null,
      h("tr", null, h("th", { class: "axis-label", scope: "col" }, "Obs. ↓  Pred. →"),
        labels.map((l) => h("th", { scope: "col", title: l }, SHORT_CAT[l] || l)))),
    h("tbody", null, M.map((row, i) => {
      const total = row.reduce((s, v) => s + (num(v) || 0), 0);
      return h("tr", null, h("th", { class: "rowh", scope: "row", title: labels[i] }, SHORT_CAT[labels[i]] || labels[i]),
        row.map((v, j) => {
          const share = total ? (num(v) || 0) / total : 0;
          return h("td", {
            class: i === j ? "diag" : "",
            title: `Observed ${labels[i]}, predicted ${labels[j]}: ${fmtInt(v)} days (${fmtPct(share)} of the row)`,
            style: { background: `rgba(${rgb}, ${(0.05 + 0.9 * share).toFixed(3)})`, color: share > 0.45 ? ink : "var(--text)" },
          }, fmtInt(v));
        }));
    }))));
}

function matrixScores(M) {
  let total = 0;
  let exact = 0;
  let near = 0;
  M.forEach((row, i) => row.forEach((v, j) => {
    const x = num(v) || 0;
    total += x;
    if (i === j) exact += x;
    if (Math.abs(i - j) <= 1) near += x;
  }));
  return total ? { exact: exact / total, near: near / total, total } : null;
}

function renderConfusion() {
  const box = $("confusion-card");
  const C = state.metrics?.confusion;
  const ok = (M) => Array.isArray(M) && M.length && M.every((r) => Array.isArray(r));
  if (!C || !Array.isArray(C.labels) || !ok(C.cams) || !ok(C.airtrace)) { unavailable(box, "AQI category confusion", "Confusion matrices"); return; }
  const labels = C.labels.length === C.cams.length ? C.labels : AQI_LEVELS.map((l) => l.name);
  const sc = matrixScores(C.cams);
  const sa = matrixScores(C.airtrace);
  const block = (title, M, s) => h("div", null,
    h("div", { class: "scatter-title" }, title),
    h("div", { class: "scatter-stats" }, s ? `Exact category ${fmtPct(s.exact)} · within one category ${fmtPct(s.near)}` : ""),
    heatTable(labels, M, `${title}: AQI category confusion matrix, rows observed, columns predicted`));
  box.replaceChildren(
    head("Getting the AQI category right", "Days by observed (rows) and predicted (columns) category. Shading = share of the row; outlined cells are correct."),
    h("div", { class: "grid", style: { gap: "18px" } }, block("CAMS raw", C.cams, sc), block("AirTrace", C.airtrace, sa)),
    h("p", { class: "note" }, "USG = Unhealthy for Sensitive Groups. Counts are cross-validated station-days."));
}

/* ================================================================== per-site map */

const SITE_BINS = [
  { max: -50, color: "#1c5cab", label: "≥ 50% lower error" },
  { max: -25, color: "#3987e5", label: "25–50% lower" },
  { max: -5, color: "#86b6ef", label: "5–25% lower" },
  { max: 5, color: "#b4b9c2", label: "within ±5%" },
  { max: Infinity, color: "#e34948", label: "higher error" },
];
const siteBin = (chg) => SITE_BINS.find((b) => chg <= b.max) || SITE_BINS[SITE_BINS.length - 1];

function renderSiteMap() {
  const P = (state.metrics?.per_site || []).filter((s) => s && finite(s.lat) && finite(s.lon) && finite(s.mae_cams) && finite(s.mae_airtrace) && +s.mae_cams > 0);
  const summary = $("site-summary");
  if (!P.length) {
    summary.textContent = "";
    $("site-map").replaceChildren(emptyState("Per-site results unavailable."));
    return;
  }
  const rows = P.map((s) => ({ ...s, chg: (+s.mae_airtrace / +s.mae_cams - 1) * 100 })).sort((a, b) => a.chg - b.chg);
  const improved = rows.filter((r) => r.chg < 0).length;
  const sorted = rows.map((r) => r.chg);
  const median = sorted.length % 2 ? sorted[(sorted.length - 1) / 2] : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2;
  summary.textContent = `AirTrace lowered the error at ${improved} of ${rows.length} monitors when they were held out (median change ${fmtSigned(median, 0)}%).`;

  $("site-ramp").replaceChildren(...SITE_BINS.map((b) => h("span", { class: "cat-chip", style: { fontWeight: 500 } },
    h("i", { style: { background: b.color, borderRadius: "50%" }, "aria-hidden": "true" }), b.label)));
  $("site-table").replaceChildren(tableView("Per-site error (µg/m³), held-out", [
    { label: "Site", value: (r) => `${r.city}, ${r.country}` },
    { label: "Region", value: (r) => r.region },
    { label: "Days", value: (r) => fmtInt(r.n) },
    { label: "Mean PM2.5", value: (r) => fmtNum(r.mean_obs) },
    { label: "CAMS MAE", value: (r) => fmtNum(r.mae_cams) },
    { label: "AirTrace MAE", value: (r) => fmtNum(r.mae_airtrace) },
    { label: "Change", value: (r) => `${fmtSigned(r.chg, 0)}%` },
  ], rows, "View all sites"));

  const L = window.L;
  const el = $("site-map");
  if (!L) { el.replaceChildren(emptyState("The map could not be loaded; see the table below.")); return; }
  if (!state.siteMap) {
    state.siteMap = L.map(el, { worldCopyJump: true, minZoom: 1, zoomSnap: 0.5, scrollWheelZoom: false, preferCanvas: true });
    state.siteTiles = L.tileLayer(tileUrl(), {
      subdomains: "abcd", maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    }).addTo(state.siteMap);
    state.siteMap.fitBounds(L.latLngBounds(rows.map((r) => [+r.lat, +r.lon])), { padding: [20, 20], maxZoom: 4 });
    state.siteMap.on("click focus", () => state.siteMap.scrollWheelZoom.enable());
    state.siteMap.on("mouseout blur", () => state.siteMap.scrollWheelZoom.disable());
    if ("ResizeObserver" in window) new ResizeObserver(() => state.siteMap.invalidateSize()).observe(el);
  }
  state.siteLayer?.remove();
  const t = theme();
  const layer = L.layerGroup();
  for (const r of rows) {
    L.circleMarker([+r.lat, +r.lon], { radius: 6.5, weight: 1.5, color: t.surface, fillColor: siteBin(r.chg).color, fillOpacity: 0.95 })
      .bindTooltip(() => h("div", null, h("strong", null, `${r.city}, ${r.country}`), h("br"),
        `MAE ${fmtNum(r.mae_cams)} → ${fmtNum(r.mae_airtrace)} µg/m³ (${fmtSigned(r.chg, 0)}%)`, h("br"),
        h("span", { style: { opacity: 0.8 } }, `${fmtInt(r.n)} days · mean PM2.5 ${fmtNum(r.mean_obs)} µg/m³`)),
      { className: "city-tip", direction: "top" })
      .addTo(layer);
  }
  layer.addTo(state.siteMap);
  state.siteLayer = layer;
}

/* ================================================================== importance */

function barList(canvas, labels, values, color, unitLabel, registry) {
  makeChart(canvas, {
    type: "bar",
    data: { labels, datasets: [{ data: values, backgroundColor: color, borderRadius: 4, maxBarThickness: 16, borderSkipped: "start" }] },
    options: {
      indexAxis: "y",
      scales: {
        x: axis({ beginAtZero: true, title: { text: unitLabel }, ticks: { callback: (v) => `${v}%` } }),
        y: axis({ grid: { display: false }, ticks: { autoSkip: false, callback: (_v, i) => wrapLabel(labels[i], 28) } }),
      },
      plugins: { tooltip: { callbacks: { label: (it) => ` ${fmtNum(it.raw, 1)}% of total gain` } } },
    },
  }, registry);
}

function renderImportance() {
  const t = theme();
  const fi = (state.metrics?.feature_importance || []).filter((f) => f && finite(f.importance));
  const fiBox = $("fi-card");
  if (!fi.length) unavailable(fiBox, "Top features", "Feature importance");
  else {
    const canvas = h("canvas", { role: "img", "aria-label": `Bar chart of the ${fi.length} most important features.` });
    fiBox.replaceChildren(head(`Top ${fi.length} features`, "Share of total split gain in the AirTrace model."),
      h("div", { class: "chart-box", style: { height: `${50 + fi.length * 26}px` } }, canvas),
      tableView("Feature importance", [
        { label: "Feature", value: (f) => f.label || f.feature },
        { label: "Group", value: (f) => f.group || "–" },
        { label: "Share of gain", value: (f) => fmtPct(f.importance, 1) },
      ], fi));
    if (hasChart()) barList(canvas, fi.map((f) => f.label || f.feature), fi.map((f) => +(asPercent(f.importance) || 0).toFixed(2)), t.airtrace, "Share of total gain", state.charts);
  }
  const gi = (state.metrics?.group_importance || []).filter((g) => g && finite(g.importance)).sort((a, b) => +b.importance - +a.importance);
  const giBox = $("gi-card");
  if (!gi.length) { unavailable(giBox, "Feature groups", "Group importance"); return; }
  const canvas = h("canvas", { role: "img", "aria-label": "Bar chart of importance by feature group." });
  giBox.replaceChildren(head("By feature group", "The same groups explain each daily forecast on the live map."),
    h("div", { class: "chart-box", style: { height: `${50 + gi.length * 34}px` } }, canvas),
    tableView("Group importance", [
      { label: "Group", value: (g) => g.group },
      { label: "Share of gain", value: (g) => fmtPct(g.importance, 1) },
    ], gi),
    h("p", { class: "note" }, "Global model chemistry dominating is expected: AirTrace corrects CAMS rather than replacing it. Weather and location terms are what make the correction city-specific."));
  if (hasChart()) barList(canvas, gi.map((g) => g.group), gi.map((g) => +(asPercent(g.importance) || 0).toFixed(2)), t.cams, "Share of total gain", state.charts);
}

/* ================================================================== uncertainty */

function renderInterval() {
  const box = $("interval-card");
  const I = state.metrics?.interval;
  if (!I || !finite(I.coverage)) { unavailable(box, "Prediction intervals", "Interval calibration"); return; }
  const nominal = asPercent(I.nominal ?? 0.8);
  const cov = asPercent(I.coverage);
  const diff = cov - nominal;
  const verdict = Math.abs(diff) <= 3 ? { text: "Well calibrated", cls: "badge-pass" }
    : diff < 0 ? { text: "Slightly over-confident", cls: "badge-fail" } : { text: "Slightly conservative", cls: "badge-pass" };
  box.replaceChildren(
    head(`${fmtNum(nominal, 0)}% prediction range: does it hold?`, "Share of held-out days whose observed PM2.5 fell inside AirTrace’s range (10th–90th percentile quantile models)."),
    h("div", { class: "compare" },
      statTile(fmtPct(cov, 1), "observed coverage"),
      statTile(fmtPct(nominal, 0), "target coverage"),
      statTile(finite(I.mean_width) ? h("span", null, fmtNum(I.mean_width), h("span", { class: "unit" }, "µg/m³")) : "–", "mean range width")),
    h("div", { class: "meter", role: "img", "aria-label": `Coverage ${fmtPct(cov, 1)} against a target of ${fmtPct(nominal, 0)}` },
      h("span", { class: "fill", style: { width: `${Math.max(0, Math.min(100, cov))}%` } }),
      h("span", { class: "target", style: { left: `${Math.max(0, Math.min(100, nominal))}%` } }, h("span", null, `target ${fmtPct(nominal, 0)}`))),
    h("p", null, h("span", { class: `badge ${verdict.cls}` }, verdict.text), " ",
      `${fmtPct(cov, 1)} of held-out days fell inside the range, against a target of ${fmtPct(nominal, 0)}${Math.abs(diff) > 0.05 ? ` (${fmtSigned(diff, 1)} percentage points)` : ""}.`),
    h("p", { class: "note" }, "On the live map, each day shows this range next to the forecast so a single number is never over-trusted."));
}

/* ================================================================== source attribution */

function renderSources() {
  const S = state.src;
  const ids = ["src-method", "src-profiles", "src-checks", "src-regression", "src-regions"];
  if (!S) {
    ids.forEach((id, i) => (i === 0 ? unavailable($(id), "Method", "The source-attribution model") : $(id).replaceChildren(emptyState("Unavailable."))));
    return;
  }
  const factors = (S.factors || []).filter((f) => f && f.key);
  const colorOf = (k) => factors.find((f) => f.key === k)?.color || SOURCE_FACTORS[k]?.color || "#9ca3af";
  const labelOf = (k) => factors.find((f) => f.key === k)?.label || SOURCE_FACTORS[k]?.label || k;

  // method + factor legend
  $("src-method").replaceChildren(
    head("Method"),
    S.method ? h("p", { class: "prose", style: { maxWidth: "none" } }, S.method) : null,
    factors.length ? h("ul", { class: "src-list", style: { gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", display: "grid", gap: "12px 22px" } },
      [...factors, { key: "background", ...SOURCE_FACTORS.background, description: "Intercept and unexplained remainder: long-range transport and sources the tracers cannot separate." }]
        .map((f) => h("li", null,
          h("span", { class: "sw", style: { background: f.color || colorOf(f.key) }, "aria-hidden": "true" }),
          h("span", { class: "name" }, f.label || labelOf(f.key)), h("span"),
          f.description ? h("span", { class: "desc" }, f.description) : null))) : null);

  // factor profiles heatmap
  const tracers = Array.isArray(S.tracers) ? S.tracers : [];
  const prof = $("src-profiles");
  if (!factors.length || !tracers.length) unavailable(prof, "Factor profiles", "Factor profiles");
  else {
    prof.replaceChildren(
      head("Factor profiles", "How strongly each tracer loads on each source factor (scaled so each factor’s largest loading = 1). These fingerprints are how factors get their names."),
      h("div", { class: "table-scroll" }, h("table", { class: "heat profile-table" },
        h("caption", { class: "sr-only" }, "NMF factor profiles: loading of each tracer on each factor"),
        h("thead", null, h("tr", null, h("th", { scope: "col" }, h("span", { class: "sr-only" }, "Factor")),
          tracers.map((tr) => h("th", { class: "colh", scope: "col" }, tr)))),
        h("tbody", null, factors.map((f) => h("tr", null,
          h("th", { class: "rowh", scope: "row" }, h("span", { class: "factor-name" }, h("i", { style: { background: f.color || colorOf(f.key) } }), f.label || labelOf(f.key))),
          tracers.map((tr, j) => {
            const v = num(f.profile?.[j]);
            const c = f.color || colorOf(f.key);
            return h("td", {
              title: `${f.label}: ${tr} ${v == null ? "–" : fmtNum(v, 2)}`,
              style: { background: v == null ? "transparent" : hexToRgba(c, Math.max(0.04, Math.min(1, v)).toFixed(3)), color: v != null && v > 0.6 ? inkOn(c) : "var(--text)" },
            }, v == null ? "–" : fmtNum(v, 2));
          })))))));
  }

  // validation checks
  const checks = (S.checks || []).filter((c) => c && c.name);
  const chkBox = $("src-checks");
  if (!checks.length) unavailable(chkBox, "Validation checks", "Source-model checks");
  else {
    const passed = checks.filter((c) => c.passed === true).length;
    chkBox.replaceChildren(
      head("Sanity checks", `${passed} of ${checks.length} physical-plausibility checks passed. Failures are shown as they are.`),
      h("ul", { class: "checks" }, checks.map((c) => h("li", { class: "check" },
        h("span", { class: `badge ${c.passed ? "badge-pass" : "badge-fail"}` },
          h("span", { class: "icon", "aria-hidden": "true", html: c.passed ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"/></svg>' : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><path d="M18 6L6 18M6 6l12 12"/></svg>' }),
          c.passed ? "Pass" : "Fail"),
        h("span", { class: "c-name" }, c.name),
        h("span", { class: "c-vals" }, `Value ${typeof c.value === "number" ? fmtNum(c.value, Math.abs(c.value) < 10 ? 3 : 1) : String(c.value ?? "–")} · expected ${String(c.expected ?? "–")}`),
        c.detail ? h("span", { class: "c-detail" }, c.detail) : null))));
  }

  // PM regression
  const R = S.pm_regression;
  const regBox = $("src-regression");
  const coef = R?.coef && typeof R.coef === "object" ? Object.entries(R.coef).filter(([, v]) => finite(v)).sort((a, b) => +b[1] - +a[1]) : [];
  if (!R || !coef.length) unavailable(regBox, "From factors to PM2.5", "The PM2.5 regression");
  else {
    const canvas = h("canvas", { role: "img", "aria-label": "Bar chart of regression coefficients per source factor." });
    regBox.replaceChildren(
      head("From factors to PM2.5", "Non-negative regression of observed PM2.5 on the factor scores: µg/m³ per unit of factor score."),
      h("div", { class: "compare" },
        statTile(finite(R.r2) ? fmtNum(R.r2, 2) : "–", "R² on observed PM2.5"),
        statTile(finite(R.n_rows) ? fmtInt(R.n_rows) : "–", "station-days"),
        statTile(finite(R.intercept) ? h("span", null, fmtNum(R.intercept), h("span", { class: "unit" }, "µg/m³")) : "–", "intercept (background)")),
      h("div", { class: "chart-box sm" }, canvas));
    if (hasChart()) {
      makeChart(canvas, {
        type: "bar",
        data: { labels: coef.map(([k]) => labelOf(k)), datasets: [{ data: coef.map(([, v]) => +v), backgroundColor: coef.map(([k]) => colorOf(k)), borderRadius: 4, maxBarThickness: 16, borderSkipped: "start" }] },
        options: {
          indexAxis: "y",
          scales: { x: axis({ beginAtZero: true, title: { text: "Coefficient" } }), y: axis({ grid: { display: false }, ticks: { callback: (_v, i) => wrapLabel(labelOf(coef[i][0]), 22) } }) },
          plugins: { tooltip: { callbacks: { label: (it) => ` ${fmtNum(it.raw, 2)}` } } },
        },
      }, state.charts);
    }
  }

  // shares by region (100% stacked)
  const regions = (S.by_region || []).filter((r) => r && r.region && r.shares && typeof r.shares === "object");
  const rb = $("src-regions");
  if (!regions.length) { unavailable(rb, "Sources by region", "Regional source shares"); return; }
  const keys = [...new Set([...factors.map((f) => f.key), ...regions.flatMap((r) => Object.keys(r.shares))])];
  if (!keys.includes("background")) keys.push("background");
  const norm = regions.map((r) => {
    const s = Object.fromEntries(keys.map((k) => [k, num(r.shares[k]) ?? 0]));
    if (!(num(r.shares.background) > 0)) s.background = Math.max(0, 1 - keys.filter((k) => k !== "background").reduce((a, k) => a + s[k], 0));
    const tot = keys.reduce((a, k) => a + s[k], 0) || 1;
    keys.forEach((k) => { s[k] = (s[k] / tot) * 100; });
    return { region: r.region, s };
  });
  const t = theme();
  const canvas = h("canvas", { role: "img", "aria-label": "Stacked bar chart of PM2.5 source shares by region." });
  rb.replaceChildren(
    head("Average source mix by region", "Mean estimated share of PM2.5 per source type at the training sites of each region."),
    h("div", { class: "series-key", "aria-hidden": "true" }, keys.map((k) => h("span", null, h("i", { class: "k-rect", style: { background: colorOf(k) } }), labelOf(k)))),
    h("div", { class: "chart-box", style: { height: `${Math.max(260, 40 + norm.length * 36)}px` } }, canvas),
    tableView("Source shares by region (%)", [
      { label: "Region", value: (r) => r.region },
      ...keys.map((k) => ({ label: SOURCE_FACTORS[k]?.short || labelOf(k), value: (r) => `${fmtNum(r.s[k], 0)}%` })),
    ], norm));
  if (!hasChart()) return;
  makeChart(canvas, {
    type: "bar",
    data: {
      labels: norm.map((r) => r.region),
      datasets: keys.map((k) => ({
        label: labelOf(k), data: norm.map((r) => +r.s[k].toFixed(1)), backgroundColor: colorOf(k),
        borderColor: t.surface, borderWidth: { right: 2 }, borderSkipped: false, maxBarThickness: 22,
      })),
    },
    options: {
      indexAxis: "y",
      scales: {
        x: axis({ stacked: true, min: 0, max: 100, ticks: { callback: (v) => `${v}%` } }),
        y: axis({ stacked: true, grid: { display: false }, ticks: { autoSkip: false, callback: (_v, i) => wrapLabel(norm[i].region) } }),
      },
      plugins: { tooltip: { callbacks: { label: (it) => ` ${it.dataset.label}: ${fmtNum(it.raw, 0)}%` } } },
    },
  }, state.charts);
}

/* ================================================================== dataset */

function renderDataset() {
  const box = $("dataset-card");
  const D = state.metrics?.dataset;
  if (!D) { unavailable(box, "Training data", "The dataset summary"); return; }
  const regions = (D.regions || []).filter((r) => r && r.region).sort((a, b) => (num(b.n_sites) || 0) - (num(a.n_sites) || 0));
  const nets = D.networks && typeof D.networks === "object" ? Object.entries(D.networks) : [];
  box.replaceChildren(
    head("Training data", Array.isArray(D.period) ? `${fmtDate(D.period[0])} – ${fmtDate(D.period[1])}, daily 24-hour PM2.5 from reference monitors.` : null),
    h("div", { class: "compare" },
      statTile(fmtInt(D.n_sites), "monitors"),
      statTile(fmtInt(D.n_countries), "countries"),
      statTile(fmtInt(D.n_rows), "monitor-days")),
    nets.length ? h("p", { class: "small" }, "Networks: ", nets.map(([k, v], i) => [i ? " · " : "", h("strong", null, k), ` ${fmtInt(v)} sites`])) : null,
    regions.length ? h("div", { class: "table-scroll" }, h("table", { class: "data" },
      h("caption", { class: "sr-only" }, "Training monitors by region"),
      h("thead", null, h("tr", null, h("th", { scope: "col" }, "Region"), h("th", { scope: "col" }, "Monitors"), h("th", { scope: "col" }, "Monitor-days"))),
      h("tbody", null, regions.map((r) => h("tr", null, h("th", { scope: "row" }, r.region), h("td", null, fmtInt(r.n_sites)), h("td", null, fmtInt(r.n_rows))))))) : null,
    h("p", { class: "note" }, "Ground truth: AirNow 24-hour PM2.5 from US Department of State embassy and consulate monitors and North American regulatory monitors."));
}

