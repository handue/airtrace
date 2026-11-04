/**
 * Shared helpers for every AirTrace page (plain ES module, no build step).
 *
 *  - data loading: JSON from "data/" (written by the live pipeline) or, with ?mock=1, from "mock/"
 *  - US EPA AQI categories (2024 PM2.5 breakpoints, mirrors airtrace/aqi.py)
 *  - number / date formatting, a tiny DOM builder, inline icons
 *  - page chrome: nav menu, GitHub links, "Updated <relative time>" badge, demo-data banner
 */

/* ------------------------------------------------------------------ configuration */

/**
 * Public source repository - the single place it is configured. Leave the override empty to
 * derive it from the GitHub Pages address (https://<owner>.github.io/<repo>/ ->
 * https://github.com/<owner>/<repo>).
 */
const REPO_URL_OVERRIDE = "";
export const REPO_URL = REPO_URL_OVERRIDE || repoFromPagesUrl(window.location) || "https://github.com/";

/** Refresh cadence of the GitHub Actions workflow that rebuilds docs/data (hours). */
export const REFRESH_HOURS = 6;

const params = new URLSearchParams(window.location.search);
/** ?mock=1 switches every page to the hand-made demo data in docs/mock/. */
export const IS_MOCK = params.get("mock") === "1";
export const DATA_ROOT = IS_MOCK ? "mock/" : "data/";

function repoFromPagesUrl(loc) {
  const m = /^([a-z0-9-]+)\.github\.io$/i.exec(loc.hostname);
  if (!m) return null;
  const first = loc.pathname.split("/").filter(Boolean)[0];
  const repo = first && !first.includes(".") ? first : `${m[1]}.github.io`;
  return `https://github.com/${m[1]}/${repo}`;
}

/* ------------------------------------------------------------------ data loading */

const jsonCache = new Map();

/** Fetch `<DATA_ROOT><path>` once; concurrent and repeated calls share the same promise. */
export function loadJSON(path) {
  const url = DATA_ROOT + path;
  if (!jsonCache.has(url)) {
    const p = fetch(url, { cache: "no-cache" }).then((r) => {
      if (!r.ok) throw new Error(`${r.status} ${r.statusText || "error"} loading ${url}`);
      return r.json();
    });
    p.catch(() => jsonCache.delete(url)); // a failed request can be retried
    jsonCache.set(url, p);
  }
  return jsonCache.get(url);
}

/** Like Promise.allSettled but returns {name: value|null} and logs failures once. */
export async function loadAll(names) {
  const res = await Promise.allSettled(names.map((n) => loadJSON(`${n}.json`)));
  const out = {};
  names.forEach((n, i) => {
    if (res[i].status === "fulfilled") out[n] = res[i].value;
    else {
      out[n] = null;
      console.warn(`[AirTrace] ${n}.json unavailable:`, res[i].reason?.message || res[i].reason);
    }
  });
  return out;
}

/* ------------------------------------------------------------------ AQI */

export const AQI_LEVELS = [
  { name: "Good", short: "Good", color: "#00e400", pmHi: 9.0, aqiLo: 0, aqiHi: 50 },
  { name: "Moderate", short: "Moderate", color: "#ffff00", pmHi: 35.4, aqiLo: 51, aqiHi: 100 },
  { name: "Unhealthy for Sensitive Groups", short: "Unhealthy for sensitive groups", color: "#ff7e00", pmHi: 55.4, aqiLo: 101, aqiHi: 150 },
  { name: "Unhealthy", short: "Unhealthy", color: "#ff0000", pmHi: 125.4, aqiLo: 151, aqiHi: 200 },
  { name: "Very Unhealthy", short: "Very unhealthy", color: "#8f3f97", pmHi: 225.4, aqiLo: 201, aqiHi: 300 },
  { name: "Hazardous", short: "Hazardous", color: "#7e0023", pmHi: 500.4, aqiLo: 301, aqiHi: 500 },
];

export const HEALTH_MESSAGES = {
  Good: "Air quality is satisfactory. Enjoy outdoor activities.",
  Moderate: "Acceptable for most people. Unusually sensitive people should consider limiting prolonged outdoor exertion.",
  "Unhealthy for Sensitive Groups": "Children, older adults and people with heart or lung disease should reduce prolonged or heavy outdoor exertion.",
  Unhealthy: "Everyone should reduce prolonged outdoor exertion; sensitive groups should avoid it. Consider a well-fitted mask outdoors.",
  "Very Unhealthy": "Everyone should avoid prolonged outdoor exertion. Keep windows closed and run an air purifier if available.",
  Hazardous: "Health alert: everyone should stay indoors with filtered air and avoid all outdoor activity.",
};

const NO_DATA_COLOR = "#9ca3af";

export function levelByName(name) {
  if (!name) return null;
  const n = String(name).toLowerCase();
  return AQI_LEVELS.find((l) => l.name.toLowerCase() === n) || null;
}

/** AQI level for a 24-h PM2.5 concentration (EPA truncates to 0.1 ug/m3). */
export function levelForPm(pm) {
  if (pm == null || !Number.isFinite(+pm)) return null;
  const c = Math.floor(Math.max(+pm, 0) * 10) / 10;
  return AQI_LEVELS.find((l) => c <= l.pmHi) || AQI_LEVELS[AQI_LEVELS.length - 1];
}

export function levelForAqi(aqi) {
  if (aqi == null || !Number.isFinite(+aqi)) return null;
  return AQI_LEVELS.find((l) => +aqi <= l.aqiHi) || AQI_LEVELS[AQI_LEVELS.length - 1];
}

/** {name, color} for a record that carries category/color (falls back to the PM2.5 value). */
export function aqiStyle(category, color, pm) {
  const lv = levelByName(category) || levelForPm(pm);
  return { name: category || lv?.name || "No data", color: color || lv?.color || NO_DATA_COLOR, level: lv };
}

function luminance(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec(String(hex).trim());
  if (!m) return 0.5;
  const v = parseInt(m[1], 16);
  const ch = [(v >> 16) & 255, (v >> 8) & 255, v & 255].map((c) => {
    const s = c / 255;
    return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2];
}

/** Text colour with the higher WCAG contrast on a filled background. */
export function inkOn(hex) {
  const L = luminance(hex);
  return (L + 0.05) / 0.05 >= 1.05 / (L + 0.05) ? "#0b1220" : "#ffffff";
}

export function hexToRgba(hex, alpha) {
  const m = /^#?([0-9a-f]{6})$/i.exec(String(hex).trim());
  if (!m) return hex;
  const v = parseInt(m[1], 16);
  return `rgba(${(v >> 16) & 255}, ${(v >> 8) & 255}, ${v & 255}, ${alpha})`;
}

/** Coloured AQI number pill (colour is always paired with the number). */
export function aqiPill(aqi, category, color, pm) {
  const st = aqiStyle(category, color, pm);
  const label = aqi == null ? "–" : String(Math.round(aqi));
  return h("span", {
    class: "aqi-pill", style: { background: st.color, color: inkOn(st.color) },
    title: `AQI ${label} · ${st.name}`,
  }, label);
}

export function catChip(level, text) {
  return h("span", { class: "cat-chip" }, h("i", { style: { background: level.color }, "aria-hidden": "true" }), text ?? level.short);
}

/* ------------------------------------------------------------------ source factors */

export const SOURCE_FACTORS = {
  traffic: { label: "Traffic & urban combustion", short: "Traffic", color: "#3b82f6" },
  industry: { label: "Industry, coal & power", short: "Industry", color: "#64748b" },
  burning: { label: "Fires & biomass burning", short: "Fires", color: "#f97316" },
  dust: { label: "Desert dust", short: "Dust", color: "#c8a165" },
  secondary: { label: "Photochemical & secondary", short: "Secondary", color: "#a855f7" },
  marine: { label: "Sea salt (marine)", short: "Sea salt", color: "#06b6d4" },
  background: { label: "Regional background & other", short: "Background", color: "#cbd5e1" },
};

/**
 * Normalise a sources block to a list that contains the background exactly once, whether the
 * producer put it inside `shares` or only in `background_share`. Sorted by share, zeros dropped.
 */
export function normalizeShares(src) {
  if (!src || !Array.isArray(src.shares)) return [];
  const list = src.shares
    .filter((s) => s && Number.isFinite(+s.share))
    .map((s) => ({ ...s, share: +s.share, color: s.color || SOURCE_FACTORS[s.key]?.color || NO_DATA_COLOR,
      label: s.label || SOURCE_FACTORS[s.key]?.label || s.key }));
  const hasBg = list.some((s) => s.key === "background");
  if (!hasBg && Number.isFinite(+src.background_share) && +src.background_share > 0) {
    list.push({ key: "background", ...SOURCE_FACTORS.background, share: +src.background_share,
      description: "Long-range transported and unattributed PM2.5." });
  }
  return list.filter((s) => s.share > 0).sort((a, b) => b.share - a.share);
}

/* ------------------------------------------------------------------ formatting */

const MINUS = "−";

function finite(v) {
  return v != null && v !== "" && Number.isFinite(+v);
}

/** Fixed decimals, true minus sign, no "-0.0". */
export function fmtNum(v, digits = 1) {
  if (!finite(v)) return "–";
  let x = +v;
  if (Math.abs(x) < 0.5 * 10 ** -digits) x = 0;
  const s = Math.abs(x).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
  return x < 0 ? MINUS + s : s;
}

export function fmtInt(v) {
  return fmtNum(v, 0);
}

/** Signed number with explicit +/−. */
export function fmtSigned(v, digits = 0) {
  if (!finite(v)) return "–";
  const s = fmtNum(Math.abs(v), digits);
  if (s === fmtNum(0, digits)) return s;
  return (+v > 0 ? "+" : MINUS) + s;
}

/** Fractions (0..1) and percentages are both accepted: values with |v| <= 1 are treated as fractions. */
export function asPercent(v) {
  if (!finite(v)) return null;
  return Math.abs(+v) <= 1 ? +v * 100 : +v;
}

export function fmtPct(v, digits = 0) {
  const p = asPercent(v);
  return p == null ? "–" : `${fmtNum(p, digits)}%`;
}

export function fmtPm(v) {
  return finite(v) ? `${fmtNum(v, 1)} µg/m³` : "–";
}

/** "YYYY-MM-DD" -> Date at UTC midnight (all formatting below uses timeZone UTC: no shifts). */
export function parseDay(s) {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(s || ""));
  return m ? new Date(Date.UTC(+m[1], +m[2] - 1, +m[3])) : null;
}

const dtfCache = new Map();
function dtf(opts) {
  const key = JSON.stringify(opts);
  if (!dtfCache.has(key)) dtfCache.set(key, new Intl.DateTimeFormat("en-GB", { timeZone: "UTC", ...opts }));
  return dtfCache.get(key);
}

export function fmtDate(s, opts = { day: "numeric", month: "short", year: "numeric" }) {
  const d = parseDay(s);
  return d ? dtf(opts).format(d) : "–";
}

export function fmtWeekday(s, style = "short") {
  const d = parseDay(s);
  return d ? dtf({ weekday: style }).format(d) : "";
}

/** "YYYY-MM" -> "Mar 2024" (or "Mar 24" when short). */
export function fmtMonth(s, short = false) {
  const m = /^(\d{4})-(\d{2})/.exec(String(s || ""));
  if (!m) return String(s ?? "");
  const d = new Date(Date.UTC(+m[1], +m[2] - 1, 1));
  return short ? `${dtf({ month: "short" }).format(d)} ${m[1].slice(2)}` : dtf({ month: "short", year: "numeric" }).format(d);
}

/** Local ISO "YYYY-MM-DDTHH:MM" (city time) -> "Tue 4 Nov, 13:00". */
export function fmtLocalHour(s) {
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/.exec(String(s || ""));
  if (!m) return String(s ?? "");
  const d = new Date(Date.UTC(+m[1], +m[2] - 1, +m[3]));
  return `${dtf({ weekday: "short", day: "numeric", month: "short" }).format(d)}, ${m[4]}:${m[5]}`;
}

const rtf = new Intl.RelativeTimeFormat("en", { numeric: "auto" });

export function relativeTime(iso, now = Date.now()) {
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return null;
  const sec = Math.round((t - now) / 1000);
  const abs = Math.abs(sec);
  if (abs < 45) return "just now";
  if (abs < 3600) return rtf.format(Math.round(sec / 60), "minute");
  if (abs < 86400) return rtf.format(Math.round(sec / 3600), "hour");
  if (abs < 86400 * 30) return rtf.format(Math.round(sec / 86400), "day");
  return rtf.format(Math.round(sec / (86400 * 30)), "month");
}

export function fmtDateTimeUTC(iso) {
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return "–";
  return `${new Intl.DateTimeFormat("en-GB", { timeZone: "UTC", day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" }).format(new Date(t))} UTC`;
}

/* ------------------------------------------------------------------ geometry */

export function haversineKm(lat1, lon1, lat2, lon2) {
  const r = Math.PI / 180;
  const dp = (lat2 - lat1) * r;
  const dl = (lon2 - lon1) * r;
  const a = Math.sin(dp / 2) ** 2 + Math.cos(lat1 * r) * Math.cos(lat2 * r) * Math.sin(dl / 2) ** 2;
  return 2 * 6371 * Math.asin(Math.sqrt(a));
}

/** Initial bearing from point 1 to point 2, degrees clockwise from north. */
export function bearingDeg(lat1, lon1, lat2, lon2) {
  const r = Math.PI / 180;
  const y = Math.sin((lon2 - lon1) * r) * Math.cos(lat2 * r);
  const x = Math.cos(lat1 * r) * Math.sin(lat2 * r) - Math.sin(lat1 * r) * Math.cos(lat2 * r) * Math.cos((lon2 - lon1) * r);
  return ((Math.atan2(y, x) / r) + 360) % 360;
}

export function angleDiff(a, b) {
  return Math.abs((((a - b) % 360) + 540) % 360 - 180);
}

const COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"];
export function compass(deg) {
  return Number.isFinite(deg) ? COMPASS[Math.round((((deg % 360) + 360) % 360) / 22.5) % 16] : "";
}

/* ------------------------------------------------------------------ DOM */

/**
 * Tiny element builder. Strings become text nodes (never parsed as HTML), so data from JSON is
 * always inserted safely. `html` is only used with the static ICONS below.
 */
export function h(tag, props, ...children) {
  const el = tag.startsWith("svg:") ? document.createElementNS("http://www.w3.org/2000/svg", tag.slice(4)) : document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.setAttribute("class", v);
    else if (k === "style" && typeof v === "object") {
      for (const [sk, sv] of Object.entries(v)) {
        if (sv == null) continue;
        if (sk.startsWith("--")) el.style.setProperty(sk, sv);
        else el.style[sk] = sv;
      }
    } else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === "html") el.innerHTML = v;
    else if (k === "text") el.textContent = v;
    else el.setAttribute(k, v === true ? "" : String(v));
  }
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export function icon(name, cls) {
  return h("span", { class: `icon ${cls || ""}`.trim(), "aria-hidden": "true", html: ICONS[name] || "" });
}

const svg = (body, extra = "") =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" ${extra}>${body}</svg>`;

export const ICONS = {
  live: svg('<circle cx="12" cy="12" r="2"/><path d="M16.24 7.76a6 6 0 0 1 0 8.49M7.76 16.24a6 6 0 0 1 0-8.49M19.07 4.93a10 10 0 0 1 0 14.14M4.93 19.07a10 10 0 0 1 0-14.14"/>'),
  offline: svg('<path d="M2 2l20 20"/><path d="M8.5 16.5a5 5 0 0 1 7 0M5 12.86a10 10 0 0 1 5.17-2.69M19 12.86a10 10 0 0 0-2.01-1.53M2 8.82a15 15 0 0 1 4.17-2.65M10.66 5c4.01-.36 8.14.9 11.34 3.76"/><circle cx="12" cy="20" r="0.5"/>'),
  none: svg('<path d="M12 2l2.4 7.4H22l-6.2 4.5 2.4 7.4L12 16.8l-6.2 4.5 2.4-7.4L2 9.4h7.6z"/>'),
  satellite: svg('<path d="M13 7L9 3 5 7l4 4"/><path d="M17 11l4 4-4 4-4-4"/><path d="M8 12l4 4 6-6-4-4z"/><path d="M16 8l3-3"/><path d="M9 21a6 6 0 0 0-6-6"/>'),
  fire: svg('<path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.07-2.14-.22-4.05 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.15.43-2.29 1-3a2.5 2.5 0 0 0 2.5 2.5z"/>'),
  search: svg('<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/>'),
  info: svg('<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>'),
  alert: svg('<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><path d="M12 9v4M12 17h.01"/>'),
  check: svg('<path d="M20 6L9 17l-5-5"/>'),
  x: svg('<path d="M18 6L6 18M6 6l12 12"/>'),
  menu: svg('<path d="M3 6h18M3 12h18M3 18h18"/>'),
  close: svg('<path d="M18 6L6 18M6 6l12 12"/>'),
  arrowDown: svg('<path d="M12 5v14M19 12l-7 7-7-7"/>'),
  arrowRight: svg('<path d="M5 12h14M13 6l6 6-6 6"/>'),
  wind: svg('<path d="M9.59 4.59A2 2 0 1 1 11 8H2m10.59 11.41A2 2 0 1 0 14 16H2m15.73-8.27A2.5 2.5 0 1 1 19.5 12H2"/>'),
  refresh: svg('<path d="M21 12a9 9 0 1 1-2.64-6.36L21 8"/><path d="M21 3v5h-5"/>'),
  table: svg('<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 10h18M9 4v16"/>'),
  github: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 .5a11.5 11.5 0 0 0-3.64 22.41c.58.1.79-.25.79-.56v-2c-3.2.7-3.88-1.37-3.88-1.37-.53-1.33-1.28-1.69-1.28-1.69-1.05-.71.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.71 1.26 3.37.96.1-.75.4-1.26.73-1.55-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.29 1.19-3.09-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.17 1.18a11 11 0 0 1 5.77 0c2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.76.11 3.05.74.8 1.19 1.83 1.19 3.09 0 4.42-2.7 5.39-5.26 5.68.41.36.78 1.06.78 2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .5z"/></svg>',
};

/** Message block for missing / failed data. */
export function emptyState(text, kind = "info") {
  return h("div", { class: `empty-state ${kind === "error" ? "error-state" : ""}`, role: kind === "error" ? "alert" : null },
    icon(kind === "error" ? "alert" : "info"), h("div", null, text));
}

/** Collapsible accessible table twin for a chart. columns: [{label, value(row)}]. */
export function tableView(caption, columns, rows, summary = "View data table") {
  return h("details", { class: "table-view" },
    h("summary", null, icon("table"), summary),
    h("div", { class: "table-scroll" },
      h("table", { class: "data" },
        h("caption", { class: "sr-only" }, caption),
        h("thead", null, h("tr", null, columns.map((c) => h("th", { scope: "col" }, c.label)))),
        h("tbody", null, rows.map((r) => h("tr", null, columns.map((c, i) =>
          i === 0 ? h("th", { scope: "row" }, c.value(r)) : h("td", null, c.value(r)))))))));
}

/* ------------------------------------------------------------------ page chrome */

/** Append the current query (?mock=1) to internal links so demo mode survives navigation. */
function keepQuery() {
  if (!window.location.search) return;
  document.querySelectorAll("a[data-keep-query]").forEach((a) => {
    const raw = a.getAttribute("href") || "";
    const [path, hash] = raw.split("#");
    if (path.includes("?")) return;
    a.setAttribute("href", `${path}${window.location.search}${hash !== undefined ? `#${hash}` : ""}`);
  });
}

function initNavToggle() {
  const btn = document.querySelector(".nav-toggle");
  const links = document.getElementById("nav-links");
  if (!btn || !links) return;
  const set = (open) => {
    links.classList.toggle("is-open", open);
    btn.setAttribute("aria-expanded", String(open));
    btn.setAttribute("aria-label", open ? "Close menu" : "Open menu");
    btn.innerHTML = open ? ICONS.close : ICONS.menu;
  };
  btn.addEventListener("click", () => set(btn.getAttribute("aria-expanded") !== "true"));
  links.addEventListener("click", (e) => { if (e.target.closest("a")) set(false); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && btn.getAttribute("aria-expanded") === "true") { set(false); btn.focus(); } });
}

/** Wire nav, GitHub links and the demo banner. Call once per page. */
export function initChrome() {
  document.querySelectorAll("[data-repo-link]").forEach((a) => a.setAttribute("href", REPO_URL));
  keepQuery();
  initNavToggle();
  const banner = document.getElementById("mock-banner");
  if (banner && IS_MOCK) banner.hidden = false;
}

/** "Updated 2 hours ago" badge from meta.generated_at; ticks every minute. */
export function initUpdated(meta) {
  const wrap = document.getElementById("updated");
  const time = document.getElementById("updated-time");
  if (!wrap || !time || !meta?.generated_at) return;
  const paint = () => {
    const rel = relativeTime(meta.generated_at);
    if (!rel) return;
    time.textContent = rel;
    time.setAttribute("datetime", meta.generated_at);
    wrap.title = `Forecasts generated ${fmtDateTimeUTC(meta.generated_at)}${meta.model_version ? ` · model ${meta.model_version}` : ""}`;
    const ageH = (Date.now() - Date.parse(meta.generated_at)) / 3.6e6;
    wrap.classList.toggle("is-stale", ageH > REFRESH_HOURS * 2);
  };
  paint();
  wrap.hidden = false;
  window.setInterval(paint, 60_000);
  const v = document.getElementById("model-version");
  if (v && meta.model_version) v.textContent = meta.model_version;
}

/** Fill every [data-bind="a.b.c"] element with a value from `data` (optionally formatted). */
export function bindText(root, data, formatters = {}) {
  root.querySelectorAll("[data-bind]").forEach((el) => {
    const path = el.getAttribute("data-bind");
    const val = path.split(".").reduce((o, k) => (o == null ? undefined : o[k]), data);
    if (val === undefined || val === null) return;
    const f = formatters[el.getAttribute("data-format") || ""];
    el.textContent = f ? f(val) : String(val);
  });
}

export function prefersReducedMotion() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}
