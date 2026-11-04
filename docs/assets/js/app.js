/**
 * AirTrace home page.
 *
 * Loads meta / cities / metrics / network_history up front, then each city/<id>.json lazily when a
 * city is selected (cached by loadJSON). Renders: hero KPI tiles, Leaflet map + searchable list,
 * the city panel (today, 5-day forecast, hourly chart, sources, SHAP drivers, fires, recent
 * accuracy, track record) and the "monitors went dark" impact section.
 * Every number shown comes from those JSON files (or is computed from them).
 */
import {
  AQI_LEVELS, BASEMAP, HEALTH_MESSAGES, ICONS, IS_MOCK, REFRESH_HOURS, SOURCE_FACTORS, aqiPill, aqiStyle,
  angleDiff, asPercent, bearingDeg, catChip, compass, emptyState, fmtDate, fmtInt, fmtLocalHour,
  fmtMonth, fmtNum, fmtPct, fmtPm, fmtSigned, fmtWeekday, h, haversineKm, hexToRgba, icon, initChrome,
  initUpdated, inkOn, levelForPm, loadAll, loadJSON, normalizeShares, parseDay, prefersReducedMotion,
  relativeTime, tableView,
} from "./common.js";
import {
  applyChartDefaults, aqiBandsPlugin, axis, crosshairPlugin, destroyAll, hasChart, makeChart,
  onSchemeChange, rangeHighlightPlugin, theme, zeroLinePlugin,
} from "./charts.js";

const $ = (id) => document.getElementById(id);
const finite = (v) => v != null && v !== "" && Number.isFinite(+v);
const num = (v) => (finite(v) ? +v : null);
const isNarrow = () => window.matchMedia("(max-width: 899px)").matches;
const smooth = () => (prefersReducedMotion() ? "auto" : "smooth");
const methodsHref = (hash = "") => `methods.html${window.location.search}${hash}`;

const STATUS = {
  active: { label: "Live monitor", icon: "live" },
  offline: { label: "Monitor offline", icon: "offline" },
  none: { label: "No ground monitor", icon: "none" },
};

const state = {
  meta: null,
  metrics: null,
  network: null,
  cities: [],
  byId: new Map(),
  selectedId: null,
  city: null,
  dayIndex: 0,
  showFires: false,
  map: null,
  tiles: null,
  markers: new Map(),
  fireLayer: null,
  visible: new Set(),
  hourlyChart: null,
  panelCharts: new Set(),
  impactCharts: new Set(),
  token: 0,
};

/* ================================================================== boot */

async function main() {
  initChrome();
  applyChartDefaults();
  renderAqiLegends();

  const data = await loadAll(["meta", "cities", "metrics", "network_history"]);
  state.meta = data.meta;
  state.metrics = data.metrics;
  state.network = data.network_history;

  initUpdated(state.meta);
  document.querySelectorAll("[data-meta]").forEach((el) => {
    const v = state.meta?.[el.dataset.meta];
    if (v != null) el.textContent = String(v);
  });
  renderKpis(data.cities);
  setupCities(data.cities);
  renderLiveCheck();
  renderImpact();

  onSchemeChange(() => {
    swapTiles();
    restyleMarkers();
    if (state.city) renderPanel();
    renderImpact();
    drawFires(false);
  });
  window.addEventListener("hashchange", () => {
    const id = cityFromHash();
    if (id && id !== state.selectedId) selectCity(id, { updateHash: false });
  });
}

main().catch((err) => {
  console.error("[AirTrace] page failed to initialise", err);
  $("city-panel")?.replaceChildren(h("div", { class: "card" }, emptyState(`Something went wrong while loading AirTrace: ${err.message}`, "error")));
});

/* ================================================================== legends + KPIs */

function renderAqiLegends() {
  const legend = $("aqi-legend");
  legend?.replaceChildren(...AQI_LEVELS.map((lv) => h("span", { role: "listitem" }, catChip(lv, lv.short))));
  const scale = $("map-legend-scale");
  scale?.replaceChildren(...AQI_LEVELS.map((lv) =>
    h("span", { class: "cat-chip", title: `${lv.name}: AQI ${lv.aqiLo}–${lv.aqiHi}` },
      h("i", { style: { background: lv.color }, "aria-hidden": "true" }),
      lv.aqiHi === 500 ? `${lv.aqiLo}+` : `${lv.aqiLo}–${lv.aqiHi}`)));
}

function kpiTile(value, label, foot, alt) {
  return h("div", { class: `card kpi${alt ? " alt" : ""}` },
    h("div", { class: "kpi-value" }, value),
    h("div", { class: "kpi-label" }, label),
    foot ? h("div", { class: "kpi-foot" }, foot) : null);
}

/** Cross-validated MAE of raw CAMS and of AirTrace from metrics.models (null when missing). */
function modelMae() {
  const models = Array.isArray(state.metrics?.models) ? state.metrics.models : [];
  const cams = models.find((m) => m?.key === "cams_raw");
  const at = models.find((m) => m?.key === "airtrace");
  return finite(cams?.mae) && finite(at?.mae) ? { cams: +cams.mae, airtrace: +at.mae } : null;
}

function renderKpis(cities) {
  const box = $("kpis");
  box.removeAttribute("aria-busy");
  const hl = state.metrics?.headline;
  const ds = state.metrics?.dataset;
  const meta = state.meta;
  const mae = modelMae();
  const tiles = [];

  if (hl && finite(hl.mae_reduction_pct)) {
    const r = +hl.mae_reduction_pct;
    tiles.push(kpiTile(`${r >= 0 ? "−" : "+"}${fmtNum(Math.abs(r), 0)}%`,
      `${r >= 0 ? "less" : "more"} error than the global CAMS model, on cities the model never saw`,
      mae ? `Daily PM2.5 mean absolute error ${fmtNum(mae.airtrace)} vs. ${fmtNum(mae.cams)} µg/m³ · leave-location-out cross-validation`
        : "Mean absolute error of daily PM2.5 · leave-location-out cross-validation"));
  } else tiles.push(kpiTile("–", "Accuracy metrics are unavailable right now", null));

  if (hl && finite(hl.aqi_acc_airtrace)) {
    tiles.push(kpiTile(fmtPct(hl.aqi_acc_airtrace),
      `of days in the right AQI category${finite(hl.aqi_acc_cams) ? ` (CAMS: ${fmtPct(hl.aqi_acc_cams)})` : ""}`,
      "Six US EPA categories · cities held out of training", true));
  } else tiles.push(kpiTile("–", "AQI category accuracy unavailable", null, true));

  if (ds && finite(ds.n_sites)) {
    const period = Array.isArray(ds.period) ? `, ${fmtMonth(ds.period[0])} – ${fmtMonth(ds.period[1])}` : "";
    tiles.push(kpiTile(fmtInt(ds.n_sites),
      `reference monitors${finite(ds.n_countries) ? ` in ${fmtInt(ds.n_countries)} countries` : ""} trained the model`,
      finite(ds.n_rows) ? `${fmtInt(ds.n_rows)} monitor-days of training data${period}` : null));
  } else tiles.push(kpiTile("–", "Training-data summary unavailable", null));

  const list = Array.isArray(cities) ? cities.filter((c) => c && c.id) : [];
  const nCities = finite(meta?.n_cities) ? +meta.n_cities : list.length;
  if (nCities) {
    const offline = list.filter((c) => c.monitor_status === "offline").length;
    const none = list.filter((c) => c.monitor_status === "none").length;
    tiles.push(kpiTile(fmtInt(nCities), `cities forecast, refreshed every ${REFRESH_HOURS} hours`,
      list.length ? `${fmtInt(offline)} whose monitor went offline · ${fmtInt(none)} that never had one` : null, true));
  } else tiles.push(kpiTile("–", "City forecasts are unavailable right now", null, true));
  box.replaceChildren(...tiles);
}

/* ------------------------------------------------------------------ live check (meta.live_check) */

/** "YYYY-MM-DD to YYYY-MM-DD" (or [start, end]) -> {a, b, days, text}; null when unparseable. */
function parseWindow(w) {
  let a = null;
  let b = null;
  if (Array.isArray(w)) [a, b] = w.map(String);
  else {
    const m = /^(\d{4}-\d{2}-\d{2})\s*(?:to|–|—|\/|-)\s*(\d{4}-\d{2}-\d{2})$/.exec(String(w ?? "").trim());
    if (m) [, a, b] = m;
  }
  const da = parseDay(a);
  const db = parseDay(b);
  if (!da || !db) return null;
  const sameYear = a.slice(0, 4) === b.slice(0, 4);
  return {
    a, b, days: Math.round((db - da) / 864e5) + 1,
    text: `${fmtDate(a, sameYear ? { day: "numeric", month: "short" } : undefined)} – ${fmtDate(b)}`,
  };
}

function windowText(w) {
  return parseWindow(w)?.text || (w ? String(w) : "");
}

function liveCheck() {
  const lc = state.meta?.live_check;
  return lc && typeof lc === "object" && finite(lc.mae_cams) && finite(lc.mae_airtrace) ? lc : null;
}

function cityLink(id, children, cls) {
  const known = state.byId.has(id);
  if (!known) return h("span", { class: cls }, children);
  return h("a", {
    class: cls, href: `#city=${encodeURIComponent(id)}`,
    onclick: (e) => { e.preventDefault(); selectCity(id, { scrollPanel: true }); },
  }, children);
}

/**
 * Hero card for the out-of-sample check on brand-new data at the cities whose monitor still works.
 * Hidden unless meta.live_check exists; every number is read from it.
 */
function renderLiveCheck() {
  const box = $("live-check");
  if (!box) return;
  const lc = liveCheck();
  if (!lc) { box.hidden = true; box.replaceChildren(); return; }
  const at = +lc.mae_airtrace;
  const cams = +lc.mae_cams;
  const change = cams > 0 ? (1 - at / cams) * 100 : null; // > 0: AirTrace's error is lower
  const better = change != null && change >= 0.5;
  const win = parseWindow(lc.window);
  const cities = (Array.isArray(lc.cities) ? lc.cities : []).filter((c) => c && (c.name || c.id));
  const nCities = finite(lc.n_cities) ? +lc.n_cities : cities.length;
  const span = win ? `last ${win.days} ${win.days === 1 ? "day" : "days"}` : "recent days";
  const where = nCities ? `${fmtInt(nCities)} ${nCities === 1 ? "city" : "cities"} with a working monitor` : "cities with a working monitor";
  const trained = state.metrics?.dataset?.period;

  const lead = h("p", { class: "lc-lead" },
    `Live check, ${span} at ${where}: `,
    h("strong", null, `AirTrace MAE ${fmtNum(at)} vs. CAMS ${fmtNum(cams)} µg/m³`),
    change != null ? h("span", { class: `lc-change${better ? " good" : ""}` }, ` (${fmtSigned(-change, 0)}% error)`) : null);

  const facts = [
    win ? `${win.text}` : (lc.window ? String(lc.window) : null),
    finite(lc.n_city_days) ? `${fmtInt(lc.n_city_days)} city-days` : null,
    finite(lc.bias_airtrace) && finite(lc.bias_cams) ? `bias AirTrace ${fmtSigned(lc.bias_airtrace, 1)} vs. CAMS ${fmtSigned(lc.bias_cams, 1)} µg/m³` : null,
  ].filter(Boolean);

  box.replaceChildren(
    h("div", { class: "lc-main" },
      h("span", { class: "lc-eyebrow" }, icon("live"), "Out-of-sample, on brand-new data"),
      lead,
      h("p", { class: "lc-sub" },
        facts.join(" · "),
        facts.length ? ". " : "",
        "Daily estimates compared with what each monitor measured",
        Array.isArray(trained) ? `; the model was trained only on ${fmtMonth(trained[0])} – ${fmtMonth(trained[1])} data` : "",
        ".")),
    cities.length ? h("div", { class: "lc-cities" },
      h("span", { class: "lc-cities-title" }, "Checked at"),
      h("ul", null, cities.map((c) => {
        const good = finite(c.mae_airtrace) && finite(c.mae_cams) && +c.mae_airtrace < +c.mae_cams;
        return h("li", null, cityLink(c.id, [
          h("span", { class: "lcc-name" }, c.name || c.id),
          finite(c.mae_airtrace) && finite(c.mae_cams)
            ? h("span", { class: "lcc-v", title: `Mean absolute error over ${fmtInt(c.n)} days: AirTrace ${fmtNum(c.mae_airtrace)}, CAMS ${fmtNum(c.mae_cams)} µg/m³` },
              h("b", { class: good ? "good" : null }, fmtNum(c.mae_airtrace)), ` vs ${fmtNum(c.mae_cams)}`)
            : null,
        ], "lc-city"));
      })),
      h("span", { class: "lc-cities-foot" }, "AirTrace vs. CAMS mean absolute error, µg/m³")) : null);
  box.hidden = false;
}

/* ================================================================== cities: map + list */

function setupCities(cities) {
  const list = $("city-list");
  if (!Array.isArray(cities) || !cities.length) {
    const msg = IS_MOCK ? "Demo data could not be loaded." : "Live forecasts are unavailable right now. Please try again later.";
    list.replaceChildren(h("li", null, emptyState(msg, "error")));
    $("map-canvas").replaceChildren(h("div", { style: { padding: "24px" } }, emptyState(msg, "error")));
    $("city-panel").replaceChildren(h("div", { class: "card" }, emptyState(msg, "error")));
    $("list-meta").textContent = "";
    return;
  }
  state.cities = cities.filter((c) => c && c.id && finite(c.lat) && finite(c.lon));
  state.byId = new Map(state.cities.map((c) => [c.id, c]));

  const regions = [...new Set(state.cities.map((c) => c.region).filter(Boolean))].sort();
  $("region-filter").append(...regions.map((r) => h("option", { value: r }, r)));

  initMap();
  renderList();

  let timer = null;
  $("city-search").addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(renderList, 120); });
  $("city-search").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      const first = $("city-list").querySelector(".city-item");
      if (first) { e.preventDefault(); first.click(); }
    }
  });
  $("region-filter").addEventListener("change", () => { renderList(); fitVisible(); });
  $("city-sort").addEventListener("change", renderList);

  const initial = cityFromHash() || defaultCityId();
  if (initial) selectCity(initial, { updateHash: false, fly: false });
}

function cityFromHash() {
  const m = /^#city=([^&]+)/.exec(window.location.hash);
  if (!m) return null;
  let id;
  try { id = decodeURIComponent(m[1]); } catch { return null; }
  return state.byId.has(id) ? id : null;
}

function defaultCityId() {
  let best = null;
  for (const c of state.cities) {
    if (finite(c.today_aqi) && (!best || +c.today_aqi > +best.today_aqi)) best = c;
  }
  return (best || state.cities[0])?.id ?? null;
}

const fold = (s) => String(s || "").normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
const aqiKey = (c) => (finite(c.today_aqi) ? +c.today_aqi : -1);
const SORTS = {
  "aqi-desc": (a, b) => aqiKey(b) - aqiKey(a) || a.name.localeCompare(b.name),
  "aqi-asc": (a, b) => aqiKey(a) - aqiKey(b) || a.name.localeCompare(b.name),
  "peak-desc": (a, b) => (num(b.peak_aqi_5d) ?? -1) - (num(a.peak_aqi_5d) ?? -1) || a.name.localeCompare(b.name),
  name: (a, b) => a.name.localeCompare(b.name),
};

function renderList() {
  const q = fold($("city-search").value.trim());
  const region = $("region-filter").value;
  const sort = SORTS[$("city-sort").value] || SORTS["aqi-desc"];
  const rows = state.cities
    .filter((c) => (region === "all" || c.region === region) && (!q || fold(`${c.name} ${c.country} ${c.region}`).includes(q)))
    .sort(sort);
  state.visible = new Set(rows.map((c) => c.id));

  const list = $("city-list");
  if (rows.length) list.replaceChildren(...rows.map((c) => h("li", null, cityButton(c))));
  else list.replaceChildren(h("li", null, emptyState(q ? `No city matches “${$("city-search").value.trim()}”.` : "No cities in this region.")));
  $("list-meta").textContent = `${rows.length} of ${state.cities.length} cities`;
  restyleMarkers();
}

function cityButton(c) {
  const st = aqiStyle(c.today_category, c.today_color, c.today_pm25);
  const s = STATUS[c.monitor_status] ? c.monitor_status : "none";
  return h("button", {
    type: "button", class: "city-item", "data-id": c.id,
    "aria-current": c.id === state.selectedId ? "true" : null,
    "aria-label": `${c.name}, ${c.country}: AQI ${c.today_aqi ?? "unknown"}, ${st.name}. ${STATUS[s].label}.`,
    onclick: () => selectCity(c.id, { scrollPanel: isNarrow() }),
  },
  aqiPill(c.today_aqi, c.today_category, c.today_color, c.today_pm25),
  h("span", { style: { minWidth: 0 } },
    h("span", { class: "ci-name" }, c.name),
    h("span", { class: "ci-sub" }, `${c.country} · ${st.name}`)),
  h("span", { class: `ci-status s-${s}`, title: STATUS[s].label, html: ICONS[STATUS[s].icon] }));
}

function tileUrl() {
  return BASEMAP.url(window.matchMedia("(prefers-color-scheme: dark)").matches);
}

function initMap() {
  const L = window.L;
  const el = $("map-canvas");
  if (!L) {
    el.replaceChildren(h("div", { style: { padding: "24px" } },
      emptyState("The map could not be loaded. You can still pick a city from the list.", "error")));
    return;
  }
  const map = L.map(el, { worldCopyJump: true, minZoom: 1, maxZoom: 12, zoomSnap: 0.5, scrollWheelZoom: false, preferCanvas: true });
  state.map = map;
  state.tiles = L.tileLayer(tileUrl(), BASEMAP.options).addTo(map);
  map.fitBounds(L.latLngBounds(state.cities.map((c) => [c.lat, c.lon])), { padding: [28, 28], maxZoom: 4 });
  // Wheel-zoom only after the user engages with the map, so the page scrolls normally.
  map.on("click focus", () => map.scrollWheelZoom.enable());
  map.on("mouseout blur", () => map.scrollWheelZoom.disable());

  for (const c of state.cities) {
    const m = L.circleMarker([c.lat, c.lon], markerStyle(c, false, false));
    m.bindTooltip(() => markerTooltip(c), { className: "city-tip", direction: "top", offset: [0, -8] });
    m.on("click", () => selectCity(c.id, { scrollPanel: isNarrow() }));
    m.addTo(map);
    state.markers.set(c.id, m);
  }
  if ("ResizeObserver" in window) new ResizeObserver(() => map.invalidateSize()).observe(el);
}

function markerTooltip(c) {
  const st = aqiStyle(c.today_category, c.today_color, c.today_pm25);
  const s = STATUS[c.monitor_status] ? c.monitor_status : "none";
  return h("div", null,
    h("strong", null, c.name), `, ${c.country}`, h("br"),
    `AQI ${c.today_aqi ?? "–"} · ${st.name}`, h("br"),
    h("span", { style: { opacity: "0.8" } }, STATUS[s].label));
}

function markerStyle(c, selected, dimmed) {
  const t = theme();
  const st = aqiStyle(c.today_category, c.today_color, c.today_pm25);
  return {
    radius: selected ? 11 : 7.5,
    weight: selected ? 3 : 2,
    color: selected ? t.text : t.surface,
    opacity: dimmed ? 0.35 : 1,
    fillColor: st.color,
    fillOpacity: dimmed ? 0.25 : 0.95,
  };
}

function restyleMarkers() {
  for (const [id, m] of state.markers) {
    const c = state.byId.get(id);
    m.setStyle(markerStyle(c, id === state.selectedId, state.visible.size > 0 && !state.visible.has(id)));
  }
  state.markers.get(state.selectedId)?.bringToFront();
}

function swapTiles() {
  state.tiles?.setUrl(tileUrl());
}

function fitVisible() {
  const L = window.L;
  if (!state.map || !L) return;
  const pts = state.cities.filter((c) => state.visible.has(c.id)).map((c) => [c.lat, c.lon]);
  if (pts.length) state.map.flyToBounds(L.latLngBounds(pts), { padding: [40, 40], maxZoom: 5, duration: prefersReducedMotion() ? 0 : 0.8 });
}

/* ================================================================== selection */

async function selectCity(id, { updateHash = true, fly = true, scrollPanel = false } = {}) {
  const summary = state.byId.get(id);
  if (!summary) return;
  if (id !== state.selectedId) state.dayIndex = 0;
  state.selectedId = id;
  if (updateHash) window.history.replaceState(null, "", `#city=${encodeURIComponent(id)}`);

  document.querySelectorAll(".city-item").forEach((b) => {
    if (b.dataset.id === id) {
      b.setAttribute("aria-current", "true");
      const list = $("city-list"); // position: relative, so offsetTop is relative to the list
      const top = b.offsetTop;
      if (top < list.scrollTop || top > list.scrollTop + list.clientHeight - b.offsetHeight) list.scrollTop = top - list.clientHeight / 3;
    } else b.removeAttribute("aria-current");
  });
  restyleMarkers();
  renderCallout(summary);
  if (fly && state.map) {
    state.map.flyTo([summary.lat, summary.lon], Math.max(state.map.getZoom(), 3.5), { duration: prefersReducedMotion() ? 0 : 0.8 });
  }

  const panel = $("city-panel");
  const token = ++state.token;
  if (state.city) panel.classList.add("is-loading");
  panel.setAttribute("aria-busy", "true");
  try {
    const city = await loadJSON(`city/${encodeURIComponent(id)}.json`);
    if (token !== state.token) return;
    state.city = city;
    renderPanel();
    const st = aqiStyle(city.today?.category, city.today?.color, city.today?.pm25);
    $("city-announcer").textContent = `${city.name}: today's AQI ${city.today?.aqi ?? "unknown"}, ${st.name}.`;
  } catch (err) {
    if (token !== state.token) return;
    state.city = null;
    renderPanelError(summary, err);
  } finally {
    if (token === state.token) {
      panel.classList.remove("is-loading");
      panel.removeAttribute("aria-busy");
    }
  }
  if (token !== state.token) return;
  drawFires(false);
  if (scrollPanel) panel.scrollIntoView({ behavior: smooth(), block: "start" });
}

/** Small "selected city" card over the map, so a selection gives feedback without scrolling. */
function renderCallout(c) {
  const el = $("map-callout");
  if (!el || !state.map) return;
  const st = aqiStyle(c.today_category, c.today_color, c.today_pm25);
  el.replaceChildren(
    aqiPill(c.today_aqi, c.today_category, c.today_color, c.today_pm25),
    h("div", { class: "mc-text" }, h("div", { class: "mc-name" }, c.name), h("div", { class: "mc-sub" }, st.name)),
    h("a", {
      class: "btn btn-primary btn-sm", href: "#city-panel",
      onclick: (e) => {
        e.preventDefault();
        const panel = $("city-panel");
        panel.scrollIntoView({ behavior: smooth(), block: "start" });
        panel.focus({ preventScroll: true });
      },
    }, "Forecast", icon("arrowDown")));
  el.hidden = false;
}

function renderPanelError(summary, err) {
  destroyAll(state.panelCharts);
  $("city-panel").replaceChildren(h("div", { class: "card" },
    h("h2", null, summary.name),
    emptyState(`We couldn’t load the forecast for ${summary.name} (${err?.message || "network error"}).`, "error"),
    h("p", { style: { marginTop: "12px" } },
      h("button", { type: "button", class: "btn btn-ghost btn-sm", onclick: () => selectCity(summary.id, { updateHash: false, fly: false }) },
        icon("refresh"), "Try again"))));
}

/* ================================================================== city panel */

function renderPanel() {
  const c = state.city;
  if (!c) return;
  destroyAll(state.panelCharts);
  state.hourlyChart = null;
  const later = [];

  const rows = [headerCard(c), daysCard(c), hourlyCard(c, later),
    h("div", { class: "grid grid-2" }, sourcesCard(c, later), driversCard(c))];
  const hasRecent = Array.isArray(c.recent) && c.recent.some((r) => r && r.date);
  rows.push(h("div", { class: "grid grid-2" }, firesCard(c), hasRecent ? recentCard(c, later) : accuracyNoteCard(c)));
  if (c.history) rows.push(historyCard(c, later));

  $("city-panel").replaceChildren(...rows);
  if (hasChart()) later.forEach((fn) => fn());
  else console.warn("[AirTrace] Chart.js not loaded; charts skipped");
}

function card(titleText, sub, ...body) {
  return h("section", { class: "card" },
    h("div", { class: "card-head" }, h("div", null, h("h3", null, titleText), sub ? h("p", { class: "card-sub" }, sub) : null)),
    ...body);
}

function seriesKey(items) {
  return h("div", { class: "series-key", "aria-hidden": "true" }, items.map((it) =>
    h("span", null,
      it.type === "dot" ? h("i", { class: "k-dot", style: { background: it.color } })
        : it.type === "rect" ? h("i", { class: "k-rect", style: { background: it.color } })
          : it.type === "band" ? h("i", { class: "k-band", style: { background: it.color } })
            : h("i", { class: `k-line${it.type === "dashed" ? " dashed" : ""}`, style: { color: it.color } }),
      it.label)));
}

const latText = (lat) => `${fmtNum(Math.abs(lat), 2)}°${lat >= 0 ? "N" : "S"}`;
const lonText = (lon) => `${fmtNum(Math.abs(lon), 2)}°${lon >= 0 ? "E" : "W"}`;
const isEmbassy = (network) => /state|embassy/i.test(network || "");

function monitorBadge(mon) {
  const s = mon?.status;
  if (s === "active") return h("span", { class: "badge badge-live" }, icon("live"), "Live monitor");
  if (s === "offline") {
    const last = mon.last_seen ? ` · last report ${fmtDate(mon.last_seen)}` : "";
    return h("span", { class: "badge badge-offline" }, icon("offline"), `Monitor offline${last}`);
  }
  return h("span", { class: "badge badge-none" }, icon("none"), "No ground monitor — AirTrace estimate");
}

const CONFIDENCE_CLASS = { local: "badge-conf-local", regional: "badge-conf-regional", extrapolation: "badge-conf-extra" };

function confidenceBadge(conf) {
  if (!conf || !conf.tier) return null;
  return h("span", { class: `badge ${CONFIDENCE_CLASS[conf.tier] || ""}`, title: conf.detail || "" },
    `Confidence: ${conf.label}`);
}

function outOfRangeNote(c) {
  const day = (c.daily || []).find((d) => d && d.out_of_range && d.note);
  if (!day) return null;
  return h("p", { class: "oor-warn", role: "note" },
    h("strong", null, "Outside training range. "), day.note,
    " Treat the AirTrace number with caution; the upper end of the range is raised to the CAMS value.");
}

function monitorDetail(c) {
  const mon = c.monitor || {};
  if (mon.status === "active") {
    const bits = [mon.name ? `Reference monitor “${mon.name}”` : "Reference monitor"];
    if (mon.network) bits[0] += ` (${mon.network})`;
    if (finite(mon.distance_km)) bits.push(`${fmtNum(mon.distance_km, 1)} km from the city centre`);
    if (mon.last_seen) bits.push(`last report ${fmtDate(mon.last_seen)}`);
    return h("p", { class: "monitor-detail" }, `${bits.join(" · ")}.`);
  }
  if (mon.status === "offline") {
    return h("p", { class: "monitor-detail" },
      `${mon.network || "The"} monitor${mon.name ? ` “${mon.name}”` : ""} has stopped reporting${isEmbassy(mon.network) ? ", like most of the US embassy network since early 2025" : ""}. `,
      c.history ? "AirTrace learned from it while it ran and keeps forecasting this city."
        : "AirTrace keeps forecasting this city from the global model and weather.");
  }
  return h("p", { class: "monitor-detail" },
    "No public reference monitor nearby. This forecast is AirTrace’s estimate from the CAMS global model, weather and what it learned at monitored cities.");
}

function headerCard(c) {
  const t = c.today || {};
  const st = aqiStyle(t.category, t.color, t.pm25);
  const msg = t.message || HEALTH_MESSAGES[st.level?.name] || "";
  const issued = c.generated_at ? relativeTime(c.generated_at) : null;
  return h("article", { class: "card city-head", "aria-labelledby": "city-title" },
    h("div", { style: { minWidth: 0 } },
      h("h2", { id: "city-title" }, c.name),
      h("p", { class: "where" }, [c.country, c.region].filter(Boolean).join(" · ")),
      h("div", { class: "badge-row" }, monitorBadge(c.monitor), confidenceBadge(c.confidence)),
      monitorDetail(c),
      c.confidence?.detail ? h("p", { class: "monitor-detail conf-detail" }, c.confidence.detail) : null,
      outOfRangeNote(c),
      h("div", { class: "meta-line" },
        finite(c.lat) && finite(c.lon) ? h("span", null, `${latText(+c.lat)}, ${lonText(+c.lon)}`) : null,
        c.timezone ? h("span", null, `Local time: ${c.timezone}`) : null,
        issued ? h("span", { title: c.generated_at }, `Forecast issued ${issued}`) : null)),
    h("div", { class: "today" },
      h("div", {
        class: "today-badge", role: "img", "aria-label": `Today's AQI ${t.aqi ?? "unknown"}, ${st.name}`,
        style: { background: st.color, color: inkOn(st.color) },
      }, h("span", { class: "v" }, t.aqi ?? "–"), h("span", { class: "l" }, "AQI today")),
      h("div", { class: "today-text" },
        h("div", { class: "cat" }, st.name),
        h("div", { class: "pm" }, `PM2.5 ${fmtPm(t.pm25)} · 24-hour mean`),
        msg ? h("p", { class: "msg" }, msg) : null)));
}

function dayName(date, i, long = false) {
  const today = state.city?.today?.date || state.city?.daily?.[0]?.date;
  if (date === today) return "Today";
  if (i === 1 && today) return "Tomorrow";
  return fmtWeekday(date, long ? "long" : "short");
}

function daysCard(c) {
  const days = Array.isArray(c.daily) ? c.daily.filter((d) => d && d.date) : [];
  if (!days.length) return card("Forecast", null, emptyState("The daily forecast is unavailable for this city right now."));
  return h("section", { class: "card", "aria-labelledby": "days-title" },
    h("div", { class: "card-head" }, h("div", null,
      h("h3", { id: "days-title" }, `${days.length}-day forecast`),
      h("p", { class: "card-sub" }, "Daily 24-hour PM2.5 and AQI with AirTrace’s 80% prediction range. Select a day to see what drives it."))),
    h("div", { class: "days", role: "group", "aria-label": "Forecast days" }, days.map((d, i) => dayButton(d, i))));
}

function dayButton(d, i) {
  const st = aqiStyle(d.category, d.color, d.pm25);
  const range = finite(d.pm25_low) && finite(d.pm25_high);
  return h("button", {
    type: "button", class: "day", "data-day": i, "aria-pressed": String(i === state.dayIndex),
    style: { "--cat-color": st.color }, onclick: () => selectDay(i),
    "aria-label": `${dayName(d.date, i, true)} ${fmtDate(d.date, { day: "numeric", month: "long" })}: AQI ${d.aqi ?? "unknown"}, ${st.name}, PM2.5 ${fmtNum(d.pm25)} micrograms per cubic metre${range ? `, 80% range ${fmtNum(d.pm25_low, 0)} to ${fmtNum(d.pm25_high, 0)}` : ""}.`,
  },
  h("span", { class: "d-name" }, dayName(d.date, i)),
  h("span", { class: "d-date" }, fmtDate(d.date, { day: "numeric", month: "short" })),
  h("span", { class: "d-aqi" }, aqiPill(d.aqi, d.category, d.color, d.pm25)),
  h("span", { class: "d-cat" }, st.name),
  h("span", { class: "d-pm" }, `PM2.5 ${fmtNum(d.pm25)} µg/m³`),
  range ? h("span", { class: "d-range" }, `80%: ${fmtNum(d.pm25_low, 0)}–${fmtNum(d.pm25_high, 0)}`) : null,
  d.out_of_range ? h("span", { class: "d-warn", title: d.note || "" }, "Beyond training range") : null);
}

function selectDay(i) {
  state.dayIndex = i;
  document.querySelectorAll(".day").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.day === i)));
  const wrap = $("drivers-card");
  if (wrap && state.city) fillDrivers(wrap, state.city);
  const chart = state.hourlyChart;
  if (chart && state.city) {
    chart.options.plugins.rangeHighlight = { ...dayRange(state.city, i), color: hexToRgba(theme().accent, 0.09) };
    chart.update("none");
  }
}

/* ------------------------------------------------------------------ hourly chart */

function dayRange(c, i) {
  const date = c.daily?.[i]?.date;
  const times = c.hourly?.time || [];
  let from = -1;
  let to = -1;
  if (date) times.forEach((t, j) => { if (String(t).startsWith(date)) { if (from < 0) from = j; to = j; } });
  return { from, to };
}

function niceMax(v) {
  if (!(v > 0)) return 10;
  const p = 10 ** Math.floor(Math.log10(v));
  return Math.ceil((v * 1.08) / (p / 2)) * (p / 2);
}

function hourlyCard(c, later) {
  const H = c.hourly;
  const n = Array.isArray(H?.time) ? H.time.length : 0;
  if (!n) return card("Hour by hour", null, emptyState("Hourly data are unavailable for this city right now."));
  const t = theme();
  const canvas = h("canvas", { role: "img", "aria-label": `Line chart of hourly PM2.5 in ${c.name}: AirTrace forecast versus the raw CAMS global model, local time.` });
  later.push(() => drawHourly(canvas, c));
  const days = (c.daily || []).filter((d) => d && d.date);
  return h("section", { class: "card" },
    h("div", { class: "card-head" }, h("div", null,
      h("h3", null, "Hour by hour: AirTrace vs. the raw global model"),
      h("p", { class: "card-sub" }, "PM2.5 in µg/m³, city local time. Background bands show the AQI categories (defined for 24-hour means; hourly values are for orientation)."))),
    seriesKey([
      { label: "AirTrace (ML-corrected)", color: t.airtrace },
      { label: "CAMS global model (raw)", color: t.cams, type: "dashed" },
      { label: "Selected day", color: t.accent, type: "band" },
    ]),
    h("div", { class: "chart-box lg" }, canvas),
    days.length ? tableView(`Daily PM2.5 for ${c.name}`, [
      { label: "Day", value: (d) => fmtDate(d.date, { weekday: "short", day: "numeric", month: "short" }) },
      { label: "AirTrace µg/m³", value: (d) => fmtNum(d.pm25) },
      { label: "80% range", value: (d) => (finite(d.pm25_low) ? `${fmtNum(d.pm25_low)}–${fmtNum(d.pm25_high)}` : "–") },
      { label: "CAMS raw µg/m³", value: (d) => fmtNum(d.cams_pm25) },
      { label: "AQI", value: (d) => (d.aqi ?? "–") },
      { label: "Category", value: (d) => aqiStyle(d.category, d.color, d.pm25).name },
    ], days, "View daily values") : null);
}

function windAt(H, i) {
  const ws = num(H.wind_speed?.[i]);
  const wd = num(H.wind_dir?.[i]);
  if (ws == null) return "";
  return `Wind ${fmtNum(ws, 0)} km/h${wd != null ? ` from ${compass(wd)}` : ""}`;
}

function drawHourly(canvas, c) {
  const t = theme();
  const H = c.hourly;
  const labels = H.time.map(String);
  const at = (H.airtrace_pm25 || []).map(num);
  const cams = (H.cams_pm25 || []).map(num);
  const vals = [...at, ...cams].filter((v) => v != null);
  const hourOf = (s) => +(s.slice(11, 13) || 0);
  state.hourlyChart = makeChart(canvas, {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "AirTrace", data: at, borderColor: t.airtrace, backgroundColor: t.airtrace, borderWidth: 2.25, tension: 0.3, order: 1,
          pointHoverBackgroundColor: t.airtrace, pointHoverBorderColor: t.surface, pointHoverBorderWidth: 2, pointHitRadius: 6 },
        { label: "CAMS raw", data: cams, borderColor: t.cams, backgroundColor: t.cams, borderWidth: 1.75, borderDash: [6, 4], tension: 0.3, order: 2,
          pointHoverBackgroundColor: t.cams, pointHoverBorderColor: t.surface, pointHoverBorderWidth: 2, pointHitRadius: 6 },
      ],
    },
    options: {
      interaction: { mode: "index", intersect: false },
      scales: {
        x: axis({
          grid: { color: (ctx) => (hourOf(labels[ctx.index] || "") === 0 ? t.grid : "transparent") },
          ticks: { autoSkip: false, maxRotation: 0, callback: (_v, i) => (hourOf(labels[i]) === 12 ? `${fmtWeekday(labels[i])} ${+labels[i].slice(8, 10)}` : "") },
        }),
        y: axis({ beginAtZero: true, suggestedMax: niceMax(Math.max(10, ...vals)), title: { text: "PM2.5 (µg/m³)" } }),
      },
      plugins: {
        aqiBands: {},
        rangeHighlight: { ...dayRange(c, state.dayIndex), color: hexToRgba(t.accent, 0.09) },
        crosshair: {},
        tooltip: {
          callbacks: {
            title: (items) => fmtLocalHour(labels[items[0].dataIndex]),
            label: (it) => ` ${it.dataset.label}: ${fmtNum(it.parsed.y)} µg/m³`,
            footer: (items) => windAt(H, items[0].dataIndex),
          },
        },
      },
    },
    plugins: [aqiBandsPlugin, rangeHighlightPlugin, crosshairPlugin],
  }, state.panelCharts);
}

/* ------------------------------------------------------------------ sources */

function sourcesCard(c, later) {
  const shares = normalizeShares(c.sources);
  const win = windowText(c.sources?.window);
  const head = h("div", { class: "card-head" }, h("div", null,
    h("h3", null, "Where it comes from"),
    h("p", { class: "card-sub" }, `Estimated share of PM2.5 by source type${win ? `, ${win}` : ""}.`)));
  const caveat = h("p", { class: "caveat" }, icon("info"),
    h("span", null, "Estimated from chemical fingerprints in global model data — ", h("a", { href: methodsHref("#sources") }, "see how")));
  if (!shares.length) return h("section", { class: "card" }, head, emptyState("Source attribution is not available for this city right now."), caveat);
  const top = shares.find((s) => s.key !== "background") || shares[0];
  const canvas = h("canvas", {
    role: "img",
    "aria-label": `Doughnut chart of PM2.5 sources: ${shares.map((s) => `${s.label} ${fmtPct(s.share)}`).join(", ")}.`,
  });
  later.push(() => drawDonut(canvas, shares));
  return h("section", { class: "card" }, head, caveat,
    h("div", { class: "sources-wrap" },
      h("div", { class: "donut" }, canvas,
        h("div", { class: "donut-center", "aria-hidden": "true" },
          h("span", { class: "v" }, fmtPct(top.share)),
          h("span", { class: "l" }, SOURCE_FACTORS[top.key]?.short || top.label))),
      h("ul", { class: "src-list" }, shares.map((s) => h("li", null,
        h("span", { class: "sw", style: { background: s.color }, "aria-hidden": "true" }),
        h("span", { class: "name" }, s.label),
        h("span", { class: "share" }, fmtPct(s.share)),
        s.description ? h("span", { class: "desc" }, s.description) : null)))));
}

function drawDonut(canvas, shares) {
  const t = theme();
  makeChart(canvas, {
    type: "doughnut",
    data: {
      labels: shares.map((s) => s.label),
      datasets: [{ data: shares.map((s) => +(asPercent(s.share) || 0).toFixed(1)), backgroundColor: shares.map((s) => s.color),
        borderColor: t.surface, borderWidth: 2, hoverOffset: 4 }],
    },
    options: {
      cutout: "68%",
      plugins: { tooltip: { callbacks: { label: (it) => ` ${it.label}: ${fmtNum(it.raw, 0)}%` } } },
    },
  }, state.panelCharts);
}

/* ------------------------------------------------------------------ drivers (SHAP) */

function driversCard(c) {
  const wrap = h("section", { class: "card", id: "drivers-card", "aria-live": "polite" });
  fillDrivers(wrap, c);
  return wrap;
}

function fillDrivers(wrap, c) {
  const i = state.dayIndex;
  const day = c.daily?.[i];
  const label = day ? `${dayName(day.date, i, true)}, ${fmtDate(day.date, { day: "numeric", month: "short" })}` : "";
  const sub = day ? `${label}: AirTrace ${fmtNum(day.pm25)} µg/m³ vs. raw CAMS ${fmtNum(day.cams_pm25)} µg/m³.` : null;
  const drivers = (day?.drivers || []).filter((d) => d && d.group && finite(d.effect_pct));
  const head = h("div", { class: "card-head" }, h("div", null, h("h3", null, "Why this forecast"), sub ? h("p", { class: "card-sub" }, sub) : null));
  if (!drivers.length) {
    wrap.replaceChildren(head, emptyState("No driver breakdown is available for this day."));
    return;
  }
  const nSites = state.metrics?.dataset?.n_sites;
  const ref = `an average day at our ${finite(nSites) ? `${fmtInt(nSites)} ` : ""}reference monitors`;
  const base = drivers.find((d) => isCamsGroup(d.group));
  const rest = drivers.filter((d) => d !== base);
  const logs = rest.map((d) => Math.log1p(Math.max(+d.effect_pct, -99.9) / 100));
  const maxLog = Math.max(0, ...logs.map(Math.abs)) || 1;
  const parts = [head];
  if (base) {
    const e = +base.effect_pct;
    parts.push(h("div", { class: "driver-base", "aria-label": `Baseline: ${base.group} ${e >= 0 ? "raises" : "lowers"} the estimate by ${fmtNum(Math.abs(e), 0)} percent versus ${ref}` },
      h("div", { class: "db-text" },
        h("span", { class: "db-kicker" }, "Baseline"),
        h("span", { class: "db-name" }, base.group),
        h("span", { class: "db-ref" }, `vs. ${ref}`)),
      h("span", { class: `db-val ${e >= 0 ? "up" : "down"}`, "aria-hidden": "true" }, `${fmtSigned(e, 0)}%`)));
  }
  if (rest.length) {
    parts.push(
      h("p", { class: "drivers-title" }, base ? "Then adjusted for" : `Largest effects vs. ${ref}`),
      h("ul", { class: "drivers" }, rest.map((d, k) => {
        const e = +d.effect_pct;
        const w = (Math.abs(logs[k]) / maxLog) * 50;
        const verb = e >= 0 ? "raises" : "lowers";
        return h("li", { class: "driver", "aria-label": `${d.group} ${verb} the estimate by ${fmtNum(Math.abs(e), 0)} percent` },
          h("span", { class: "dname" }, d.group),
          h("span", { class: "track", "aria-hidden": "true" }, h("span", { class: `bar ${e >= 0 ? "up" : "down"}`, style: { width: `${Math.max(w, 0.8).toFixed(1)}%` } })),
          h("span", { class: "val", "aria-hidden": "true" }, `${fmtSigned(e, 0)}%`));
      })),
      h("div", { class: "drivers-axis", "aria-hidden": "true" }, h("div", null, h("span", null, "← lowers PM2.5"), h("span", null, "raises PM2.5 →")), h("span")));
  }
  parts.push(h("p", { class: "note" },
    `TreeSHAP contributions of the ${drivers.length} largest input groups, as % change vs. ${ref} (the model’s average prediction). Effects multiply: +50% and −20% together give +20%.${rest.length > 1 ? " Bar length on a log scale." : ""} `,
    h("a", { href: methodsHref("#explain") }, "About the drivers")));
  wrap.replaceChildren(...parts);
}

const isCamsGroup = (g) => /^global model chemistry/i.test(String(g || ""));

/* ------------------------------------------------------------------ fires */

function statTile(value, label, cls = "") {
  return h("div", { class: `stat ${cls}`.trim() }, h("div", { class: "v" }, value), h("div", { class: "l" }, label));
}

function firesCard(c) {
  const f = c.fires;
  const head = h("div", { class: "card-head" }, h("div", null,
    h("h3", null, "Fires nearby"),
    h("p", { class: "card-sub" }, "NASA FIRMS satellite fire detections, last 24 hours, within 300 km.")));
  if (!f || typeof f !== "object") return h("section", { class: "card" }, head, emptyState("Fire data are unavailable right now."));
  const count = num(f.count_300km) ?? 0;
  const tiles = h("div", { class: "stat-grid" },
    statTile(fmtInt(count), "active fires within 300 km"),
    statTile(fmtInt(num(f.upwind_count) ?? 0), "upwind of the city"),
    statTile(finite(f.nearest_km) ? h("span", null, fmtNum(f.nearest_km, 0), h("span", { class: "unit" }, "km")) : "–", "nearest fire"),
    statTile(finite(f.frp_sum) ? h("span", null, fmtInt(f.frp_sum), h("span", { class: "unit" }, "MW")) : "–", "total fire radiative power"));
  if (!count || !Array.isArray(f.points) || !f.points.length) {
    return h("section", { class: "card" }, head, tiles, emptyState("No active fires detected within 300 km in the last 24 hours."));
  }
  const btn = h("button", {
    type: "button", class: "btn btn-ghost btn-sm", id: "fires-toggle", "aria-pressed": String(state.showFires),
    disabled: state.map ? null : true,
    onclick: () => {
      state.showFires = !state.showFires;
      btn.setAttribute("aria-pressed", String(state.showFires));
      btn.lastChild.textContent = state.showFires ? "Hide fires on map" : "Show fires on map";
      drawFires(state.showFires);
    },
  }, icon("fire"), h("span", null, state.showFires ? "Hide fires on map" : "Show fires on map"));
  return h("section", { class: "card" }, head, tiles, btn,
    h("p", { class: "note" }, "Upwind = within 45° of the direction today’s wind is coming from. Smoke from upwind fires is the most likely to reach the city."));
}

/** Speed-weighted mean wind direction ("from", degrees) over the city's first forecast day. */
function meanWindFrom(c) {
  const H = c.hourly;
  const date = c.today?.date || c.daily?.[0]?.date;
  if (!H?.time || !date) return null;
  let u = 0;
  let v = 0;
  H.time.forEach((t, i) => {
    if (!String(t).startsWith(date)) return;
    const s = num(H.wind_speed?.[i]);
    const d = num(H.wind_dir?.[i]);
    if (s == null || d == null) return;
    u += s * Math.sin((d * Math.PI) / 180);
    v += s * Math.cos((d * Math.PI) / 180);
  });
  if (!u && !v) return null;
  return ((Math.atan2(u, v) * 180) / Math.PI + 360) % 360;
}

function drawFires(focus) {
  const L = window.L;
  const map = state.map;
  const key = $("fire-key");
  if (state.fireLayer) { state.fireLayer.remove(); state.fireLayer = null; }
  const c = state.city;
  const pts = c?.fires?.points;
  if (!L || !map || !state.showFires || !c || !Array.isArray(pts) || !pts.length) {
    key?.classList.remove("is-on");
    key?.replaceChildren();
    return;
  }
  const t = theme();
  const windFrom = meanWindFrom(c);
  const layer = L.layerGroup();
  L.circle([c.lat, c.lon], { radius: 300000, color: t.accent, weight: 1.25, dashArray: "4 6", fill: false, interactive: false }).addTo(layer);
  for (const p of pts) {
    if (!Array.isArray(p) || !finite(p[0]) || !finite(p[1])) continue;
    const [lat, lon] = [+p[0], +p[1]];
    const frp = num(p[2]) ?? 0;
    const brg = bearingDeg(c.lat, c.lon, lat, lon);
    const up = p.length > 3 && p[3] != null ? Boolean(p[3]) : windFrom != null && angleDiff(brg, windFrom) <= 45;
    const dist = haversineKm(c.lat, c.lon, lat, lon);
    L.circleMarker([lat, lon], {
      radius: 2.5 + Math.min(5.5, Math.sqrt(Math.max(0, frp)) / 2.2),
      color: up ? "#7f1d1d" : "#9a3412", weight: 0.75, fillColor: up ? t.fireUpwind : t.fire, fillOpacity: 0.85,
    }).bindTooltip(`Fire · ${fmtNum(frp, 0)} MW · ${fmtInt(dist)} km ${compass(brg)} of the city${up ? " · upwind" : ""}`,
      { className: "city-tip", direction: "top" }).addTo(layer);
  }
  layer.addTo(map);
  state.fireLayer = layer;
  state.markers.get(state.selectedId)?.bringToFront();
  if (focus) {
    map.flyToBounds(L.latLng(c.lat, c.lon).toBounds(640000), { padding: [12, 12], duration: prefersReducedMotion() ? 0 : 0.8 });
    if (isNarrow()) $("map-canvas").scrollIntoView({ behavior: smooth(), block: "center" });
    else $("map-canvas").scrollIntoView({ behavior: smooth(), block: "nearest" });
  }
  if (key) {
    key.replaceChildren(
      h("span", null, h("i", { style: { background: t.fire } }), "Fire (size = power)"),
      h("span", null, h("i", { style: { background: t.fireUpwind } }), "Upwind fire"),
      windFrom != null ? h("span", null, icon("wind"), `Wind from ${compass(windFrom)}`) : null);
    key.classList.add("is-on");
  }
}

/* ------------------------------------------------------------------ recent accuracy */

function maeOf(rows, k) {
  const pairs = rows.filter((r) => finite(r.observed) && finite(r[k]));
  if (!pairs.length) return null;
  return pairs.reduce((s, r) => s + Math.abs(+r[k] - +r.observed), 0) / pairs.length;
}

function recentCard(c, later) {
  const rows = c.recent.filter((r) => r && r.date).sort((a, b) => String(a.date).localeCompare(String(b.date)));
  const t = theme();
  const n = rows.filter((r) => finite(r.observed)).length;
  const maeA = maeOf(rows, "airtrace");
  const maeC = maeOf(rows, "cams");
  const canvas = h("canvas", { role: "img", "aria-label": `Line chart of the last ${rows.length} days in ${c.name}: observed PM2.5, AirTrace and raw CAMS.` });
  later.push(() => drawRecent(canvas, rows));
  const better = maeA != null && maeC != null && maeA < maeC;
  return h("section", { class: "card" },
    h("div", { class: "card-head" }, h("div", null,
      h("h3", null, "Recent accuracy"),
      h("p", { class: "card-sub" }, `AirTrace and raw CAMS vs. the live monitor, last ${rows.length} days — data the model never trained on.`))),
    h("div", { class: "compare" },
      statTile(maeA != null ? h("span", null, fmtNum(maeA), h("span", { class: "unit" }, "µg/m³")) : "–", "AirTrace mean abs. error", better ? "good" : ""),
      statTile(maeC != null ? h("span", null, fmtNum(maeC), h("span", { class: "unit" }, "µg/m³")) : "–", "CAMS mean abs. error"),
      statTile(fmtInt(n), "days with observations")),
    seriesKey([{ label: "Observed", color: t.obs, type: "dot" }, { label: "AirTrace", color: t.airtrace }, { label: "CAMS raw", color: t.cams, type: "dashed" }]),
    h("div", { class: "chart-box sm" }, canvas),
    tableView(`Recent daily PM2.5 in ${c.name}`, [
      { label: "Date", value: (r) => fmtDate(r.date, { weekday: "short", day: "numeric", month: "short" }) },
      { label: "Observed", value: (r) => fmtNum(r.observed) },
      { label: "AirTrace", value: (r) => fmtNum(r.airtrace) },
      { label: "CAMS raw", value: (r) => fmtNum(r.cams) },
    ], rows));
}

function drawRecent(canvas, rows) {
  const t = theme();
  const labels = rows.map((r) => fmtDate(r.date, { weekday: "short", day: "numeric" }));
  makeChart(canvas, {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Observed", data: rows.map((r) => num(r.observed)), borderColor: t.obs, backgroundColor: t.obs, borderWidth: 2,
          pointRadius: 4, pointBackgroundColor: t.obs, pointBorderColor: t.surface, pointBorderWidth: 2, order: 0 },
        { label: "AirTrace", data: rows.map((r) => num(r.airtrace)), borderColor: t.airtrace, backgroundColor: t.airtrace, tension: 0.25, order: 1 },
        { label: "CAMS raw", data: rows.map((r) => num(r.cams)), borderColor: t.cams, backgroundColor: t.cams, borderDash: [6, 4], borderWidth: 1.75, tension: 0.25, order: 2 },
      ],
    },
    options: {
      interaction: { mode: "index", intersect: false },
      scales: { x: axis({ grid: { display: false } }), y: axis({ beginAtZero: true, title: { text: "µg/m³" } }) },
      plugins: { crosshair: {}, tooltip: { callbacks: { label: (it) => ` ${it.dataset.label}: ${fmtNum(it.parsed.y)} µg/m³` } } },
    },
    plugins: [crosshairPlugin],
  }, state.panelCharts);
}

function accuracyNoteCard(c) {
  const mon = c.monitor || {};
  const hl = state.metrics?.headline;
  let lead;
  if (mon.status === "offline") {
    const net = mon.network ? `${mon.network} ` : "";
    lead = `The ${net}monitor here last reported${mon.last_seen ? ` on ${fmtDate(mon.last_seen)}` : " some time ago"}, so today’s forecast can’t be checked against a measurement.`;
    if (c.history) lead += " Its record from the training year is shown below.";
  } else if (mon.status === "active") {
    lead = "A live monitor reports here, but no paired observations from the last few days are available yet.";
  } else {
    lead = "There is no ground monitor here, so there is nothing to check today’s forecast against.";
  }
  const body = [h("p", null, lead)];
  if (hl && finite(hl.mae_reduction_pct)) {
    body.push(h("p", null,
      "For cities like this, accuracy is measured by ", h("strong", null, "leave-location-out cross-validation"),
      `: on monitored cities the model never saw, AirTrace’s error was ${fmtNum(Math.abs(+hl.mae_reduction_pct), 0)}% ${+hl.mae_reduction_pct >= 0 ? "lower" : "higher"} than the raw CAMS model`,
      finite(hl.aqi_acc_airtrace) ? ` and it put ${fmtPct(hl.aqi_acc_airtrace)} of days in the right AQI category${finite(hl.aqi_acc_cams) ? ` (CAMS: ${fmtPct(hl.aqi_acc_cams)})` : ""}.` : "."));
  }
  const lc = liveCheck();
  if (lc) {
    const win = parseWindow(lc.window);
    body.push(h("p", null, "On brand-new data",
      win ? ` (${win.text})` : "",
      `, at ${finite(lc.n_cities) ? `${fmtInt(lc.n_cities)} ` : ""}cities whose monitor still works, AirTrace’s mean absolute error was `,
      h("strong", null, `${fmtNum(lc.mae_airtrace)} µg/m³`), ` vs. ${fmtNum(lc.mae_cams)} µg/m³ for CAMS.`));
  }
  body.push(h("p", { style: { margin: 0 } }, h("a", { href: methodsHref("#validation") }, "How we validate →")));
  return h("section", { class: "card" },
    h("div", { class: "card-head" }, h("div", null, h("h3", null, "How accurate is this?"))),
    h("div", { class: "prose", style: { fontSize: "0.93rem" } }, body));
}

/* ------------------------------------------------------------------ track record */

function historyCard(c, later) {
  const H = c.history;
  const t = theme();
  const monthly = (Array.isArray(H.monthly) ? H.monthly : []).filter((m) => m && m.month);
  const rose = H.rose && Array.isArray(H.rose.sectors) && Array.isArray(H.rose.mean_pm25) && H.rose.sectors.length ? H.rose : null;
  const period = Array.isArray(H.period) ? `${fmtDate(H.period[0])} – ${fmtDate(H.period[1])}` : "";
  const change = finite(H.mae_cams) && finite(H.mae_airtrace) && +H.mae_cams > 0 ? (1 - +H.mae_airtrace / +H.mae_cams) * 100 : null;

  const left = h("div", null);
  if (monthly.length) {
    const canvas = h("canvas", { role: "img", "aria-label": `Line chart of monthly mean PM2.5 in ${c.name}: observed, CAMS and AirTrace (out-of-fold).` });
    later.push(() => drawMonthly(canvas, monthly));
    left.append(
      seriesKey([{ label: "Observed (monitor)", color: t.obs, type: "dot" }, { label: "AirTrace (out-of-fold)", color: t.airtrace }, { label: "CAMS raw", color: t.cams, type: "dashed" }]),
      h("div", { class: "chart-box" }, canvas),
      tableView(`Monthly mean PM2.5 in ${c.name}`, [
        { label: "Month", value: (m) => fmtMonth(m.month) },
        { label: "Observed", value: (m) => fmtNum(m.observed) },
        { label: "AirTrace", value: (m) => fmtNum(m.airtrace) },
        { label: "CAMS raw", value: (m) => fmtNum(m.cams) },
      ], monthly));
  } else left.append(emptyState("Monthly history is unavailable."));

  const right = h("div", null);
  if (rose) {
    const canvas = h("canvas", { role: "img", "aria-label": `Pollution rose for ${c.name}: mean PM2.5 by wind direction.` });
    later.push(() => drawRose(canvas, rose));
    const freq = (rose.freq || []).map(num);
    const pm = rose.mean_pm25.map(num);
    const iF = freq.length ? freq.indexOf(Math.max(...freq.filter((v) => v != null))) : -1;
    const iP = pm.indexOf(Math.max(...pm.filter((v) => v != null)));
    right.append(
      h("h4", { style: { fontSize: "0.92rem", margin: "0 0 4px" } }, "Pollution rose"),
      h("p", { class: "card-sub", style: { marginBottom: "6px" } }, "Mean PM2.5 by the direction the wind blew from."),
      h("div", { class: "chart-box", style: { height: "260px" } }, canvas),
      h("p", { class: "note" },
        iP >= 0 ? `Dirtiest air with wind from ${rose.sectors[iP]} (${fmtNum(pm[iP], 0)} µg/m³). ` : "",
        iF >= 0 ? `Most common wind: ${rose.sectors[iF]} (${fmtPct(freq[iF])} of the time).` : ""));
  }

  return h("section", { class: "card span-2" },
    h("div", { class: "card-head" }, h("div", null,
      h("h3", null, "Track record at this location"),
      h("p", { class: "card-sub" }, `${period}${finite(H.n_days) ? ` · ${fmtInt(H.n_days)} days with observations` : ""}. AirTrace values are out-of-fold: predicted by a model that never saw this location.`))),
    h("div", { class: "compare" },
      statTile(finite(H.mae_cams) ? h("span", null, fmtNum(H.mae_cams), h("span", { class: "unit" }, "µg/m³")) : "–", "CAMS mean daily error"),
      statTile(finite(H.mae_airtrace) ? h("span", null, fmtNum(H.mae_airtrace), h("span", { class: "unit" }, "µg/m³")) : "–", "AirTrace mean daily error", change != null && change > 0 ? "good" : ""),
      statTile(change != null ? `${fmtSigned(-change, 0)}%` : "–", "change in error vs. CAMS", change != null && change > 0 ? "good" : "")),
    h("div", { class: rose ? "history-grid" : "" }, left, rose ? right : null));
}

function drawMonthly(canvas, monthly) {
  const t = theme();
  makeChart(canvas, {
    type: "line",
    data: {
      labels: monthly.map((m) => fmtMonth(m.month, true)),
      datasets: [
        { label: "Observed", data: monthly.map((m) => num(m.observed)), borderColor: t.obs, backgroundColor: t.obs, borderWidth: 2,
          pointRadius: 3.5, pointBackgroundColor: t.obs, pointBorderColor: t.surface, pointBorderWidth: 1.5, tension: 0.25, order: 0 },
        { label: "AirTrace", data: monthly.map((m) => num(m.airtrace)), borderColor: t.airtrace, backgroundColor: t.airtrace, tension: 0.25, order: 1 },
        { label: "CAMS raw", data: monthly.map((m) => num(m.cams)), borderColor: t.cams, backgroundColor: t.cams, borderDash: [6, 4], borderWidth: 1.75, tension: 0.25, order: 2 },
      ],
    },
    options: {
      interaction: { mode: "index", intersect: false },
      scales: { x: axis({ grid: { display: false } }), y: axis({ beginAtZero: true, title: { text: "Monthly mean PM2.5 (µg/m³)" } }) },
      plugins: { crosshair: {}, tooltip: { callbacks: { label: (it) => ` ${it.dataset.label}: ${fmtNum(it.parsed.y)} µg/m³` } } },
    },
    plugins: [crosshairPlugin],
  }, state.panelCharts);
}

function drawRose(canvas, rose) {
  const t = theme();
  const pm = rose.mean_pm25.map(num);
  const freq = (rose.freq || []).map(num);
  const n = rose.sectors.length;
  makeChart(canvas, {
    type: "polarArea",
    data: {
      labels: rose.sectors,
      datasets: [{
        data: pm,
        backgroundColor: pm.map((v) => hexToRgba(levelForPm(v)?.color || "#9ca3af", 0.78)),
        borderColor: t.surface, borderWidth: 1.5,
      }],
    },
    options: {
      scales: {
        r: {
          startAngle: -180 / n,
          beginAtZero: true,
          grid: { color: t.grid },
          angleLines: { display: true, color: t.grid },
          ticks: { display: true, color: t.muted, backdropColor: "transparent", maxTicksLimit: 4, font: { size: 10 }, z: 1 },
          pointLabels: {
            display: true, centerPointLabels: true, color: t.text2, font: { size: 11, weight: "600" },
            callback: (label) => (["N", "E", "S", "W"].includes(label) ? label : ""),
          },
        },
      },
      plugins: {
        tooltip: {
          callbacks: {
            title: (items) => `Wind from ${items[0].label}`,
            label: (it) => ` Mean PM2.5 ${fmtNum(it.raw)} µg/m³${freq[it.dataIndex] != null ? ` · ${fmtPct(freq[it.dataIndex])} of the time` : ""}`,
          },
        },
      },
    },
  }, state.panelCharts);
}

/* ================================================================== impact section */

function renderImpact() {
  destroyAll(state.impactCharts);
  renderNetwork();
  renderBias();
}

function renderNetwork() {
  const n = state.network;
  const copy = $("impact-copy");
  const canvas = $("network-chart");
  const extra = $("network-extra");
  const months = Array.isArray(n?.months) ? n.months : [];
  const emb = Array.isArray(n?.embassy_monitors) ? n.embassy_monitors.map(num) : [];
  if (!months.length || emb.length !== months.length || !emb.some((v) => v != null)) {
    copy.replaceChildren(emptyState("The monitor-network history is unavailable right now."));
    const box = canvas?.closest(".chart-box");
    if (box) box.hidden = true;
    extra.replaceChildren(emptyState("No network data to chart."));
    return;
  }
  const t = theme();
  const valid = emb.map((v, i) => [v, i]).filter(([v]) => v != null);
  const [peak, iPeak] = valid.reduce((a, b) => (b[0] > a[0] ? b : a));
  const [last, iLast] = valid[valid.length - 1];
  const lost = peak > 0 ? (1 - last / peak) * 100 : null;
  const period = state.metrics?.dataset?.period;
  const inTrain = (m) => Array.isArray(period) && m >= String(period[0]).slice(0, 7) && m <= String(period[1]).slice(0, 7);
  const offline = state.cities.filter((c) => c.monitor_status === "offline").length;
  const none = state.cities.filter((c) => c.monitor_status === "none").length;

  const paras = [
    h("p", null, `In ${fmtMonth(months[iPeak])}, `, h("strong", null, `${fmtInt(peak)} US embassy monitors`),
      ` reported PM2.5 to AirNow. In ${fmtMonth(months[iLast])}, `, h("strong", null, `${fmtInt(last)} did`), "."),
    h("div", { class: "big-stats" },
      statTile(fmtInt(peak), `monitors at peak (${fmtMonth(months[iPeak])})`),
      statTile(fmtInt(last), `still reporting (${fmtMonth(months[iLast])})`),
      statTile(lost != null ? `${fmtSigned(-lost, 0)}%` : "–", "of the network gone")),
    h("p", null, "AirTrace learned from these monitors while they ran",
      Array.isArray(period) ? ` (training data ${fmtDate(period[0])} – ${fmtDate(period[1])})` : "",
      " and keeps forecasting their cities now."),
  ];
  if (state.cities.length) {
    paras.push(h("p", null, "On this map, ", h("strong", null, `${offline} ${offline === 1 ? "city has" : "cities have"} lost ${offline === 1 ? "its" : "their"} monitor`),
      " and ", h("strong", null, `${none} never had one`), ` — every one still gets a daily forecast with an uncertainty range.`));
  }
  copy.replaceChildren(...paras);

  $("network-key").replaceChildren(...seriesKey([
    { label: "AirTrace training window", color: t.airtrace, type: "rect" },
    { label: "Other months", color: neutral(), type: "rect" },
  ]).childNodes);
  $("network-sub").textContent = `Distinct embassy monitors reporting 24-hour PM2.5, ${fmtMonth(months[0])} – ${fmtMonth(months[months.length - 1])}.`;

  makeChart(canvas, {
    type: "bar",
    data: {
      labels: months,
      datasets: [{
        label: "Embassy monitors", data: emb,
        backgroundColor: months.map((m) => (inTrain(m) ? t.airtrace : neutral())),
        hoverBackgroundColor: months.map((m) => (inTrain(m) ? t.airtrace : t.muted)),
        borderRadius: 3, barPercentage: 0.82, categoryPercentage: 0.92, maxBarThickness: 24,
      }],
    },
    options: {
      scales: {
        x: axis({ grid: { display: false }, ticks: { autoSkip: false, maxRotation: 0, callback: (_v, i) => (months[i].endsWith("-01") ? months[i].slice(0, 4) : "") } }),
        y: axis({ beginAtZero: true, ticks: { precision: 0 }, title: { text: "Monitors reporting" } }),
      },
      plugins: {
        tooltip: {
          callbacks: {
            title: (items) => fmtMonth(months[items[0].dataIndex]),
            label: (it) => ` ${fmtInt(it.raw)} embassy monitors reporting`,
            footer: (items) => {
              const all = num(n.all_pm25_monitors?.[items[0].dataIndex]);
              return all != null ? `All AirNow PM2.5 sites: ${fmtInt(all)}` : "";
            },
          },
        },
      },
    },
  }, state.impactCharts);

  extra.replaceChildren(
    tableView("US embassy PM2.5 monitors per month", [
      { label: "Month", value: (i) => fmtMonth(months[i]) },
      { label: "Embassy monitors", value: (i) => fmtInt(emb[i]) },
      { label: "All AirNow PM2.5 sites", value: (i) => fmtInt(n.all_pm25_monitors?.[i]) },
    ], months.map((_m, i) => i)),
    n.source ? h("p", { class: "note" }, `Source: ${n.source}`) : null);
}

function neutral() {
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "#4b5b76" : "#a8b3c2";
}

function renderBias() {
  const box = $("bias-callout");
  const rows = (state.metrics?.by_region || []).filter((r) => r && r.region && finite(r.bias_cams) && finite(r.bias_airtrace));
  const title = h("h3", null, "Global models miss most where monitors are fewest");
  if (!rows.length) {
    box.replaceChildren(title, emptyState("Regional accuracy metrics are unavailable right now."));
    return;
  }
  const t = theme();
  rows.sort((a, b) => +a.bias_cams - +b.bias_cams);
  const worst = rows.reduce((a, b) => (Math.abs(+b.bias_cams) > Math.abs(+a.bias_cams) ? b : a));
  const under = rows.filter((r) => +r.bias_cams < 0).sort((a, b) => +a.bias_cams - +b.bias_cams).slice(0, 3).map((r) => r.region);
  const dir = +worst.bias_cams < 0 ? "too clean" : "too polluted";

  const copy = h("div", null, title,
    h("p", null, `In ${worst.region}, the raw CAMS model reads ${dir} by `, h("strong", null, `${fmtNum(Math.abs(+worst.bias_cams), 1)} µg/m³`),
      " on an average day; after AirTrace’s correction the bias is ", h("strong", null, `${fmtSigned(+worst.bias_airtrace, 1)} µg/m³`), ".",
      finite(worst.mae_cams) && finite(worst.mae_airtrace) ? ` The typical daily error there drops from ${fmtNum(worst.mae_cams)} to ${fmtNum(worst.mae_airtrace)} µg/m³.` : ""),
    under.length ? h("p", null, `CAMS underestimates PM2.5 most in ${listText(under)}. Outside North America, every AirTrace training monitor is a US embassy monitor — and most of them have since stopped reporting.`) : null,
    h("p", { class: "note" }, "Bias = mean of (prediction − observation) per region, leave-location-out cross-validation. Negative = model reads too clean."));

  const canvas = h("canvas", { role: "img", "aria-label": "Bar chart of mean bias by region for CAMS and AirTrace." });
  const chartCol = h("div", null,
    seriesKey([{ label: "CAMS raw", color: t.cams, type: "rect" }, { label: "AirTrace", color: t.airtrace, type: "rect" }]),
    h("div", { class: "chart-box", style: { height: `${Math.max(260, 46 + rows.length * 38)}px` } }, canvas),
    tableView("Mean bias by region (µg/m³)", [
      { label: "Region", value: (r) => r.region },
      { label: "Sites", value: (r) => fmtInt(r.n_sites) },
      { label: "CAMS bias", value: (r) => fmtSigned(r.bias_cams, 1) },
      { label: "AirTrace bias", value: (r) => fmtSigned(r.bias_airtrace, 1) },
      { label: "CAMS MAE", value: (r) => fmtNum(r.mae_cams) },
      { label: "AirTrace MAE", value: (r) => fmtNum(r.mae_airtrace) },
    ], rows));
  box.replaceChildren(h("div", { class: "impact-grid" }, copy, chartCol));

  makeChart(canvas, {
    type: "bar",
    data: {
      labels: rows.map((r) => r.region),
      datasets: [
        { label: "CAMS raw", data: rows.map((r) => +r.bias_cams), backgroundColor: t.cams, borderRadius: 4, maxBarThickness: 14, borderSkipped: false },
        { label: "AirTrace", data: rows.map((r) => +r.bias_airtrace), backgroundColor: t.airtrace, borderRadius: 4, maxBarThickness: 14, borderSkipped: false },
      ],
    },
    options: {
      indexAxis: "y",
      scales: {
        x: axis({ title: { text: "Mean bias (µg/m³)" } }),
        y: axis({ grid: { display: false }, ticks: { autoSkip: false, callback: (_v, i) => wrapLabel(rows[i].region) } }),
      },
      plugins: {
        zeroLine: { axis: "x" },
        tooltip: { callbacks: { label: (it) => ` ${it.dataset.label}: ${fmtSigned(it.raw, 1)} µg/m³` } },
      },
    },
    plugins: [zeroLinePlugin],
  }, state.impactCharts);
}

function listText(items) {
  if (items.length <= 1) return items.join("");
  return `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
}

/** Split long category labels over two lines for horizontal bar charts. */
function wrapLabel(s) {
  const str = String(s);
  if (str.length <= 16) return str;
  const mid = str.lastIndexOf(" ", Math.ceil(str.length / 2) + 4);
  return mid > 0 ? [str.slice(0, mid), str.slice(mid + 1)] : str;
}
