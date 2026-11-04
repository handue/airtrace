"""Live build: every JSON file the website reads (``docs/data/``), from keyless live data and
the trained artifacts.

Per showcase city (``config/showcase_cities.csv``):

1. **Inputs** - Open-Meteo CAMS global composition forecast + weather forecast, hourly, 7 past
   days and 5 forecast days in the city's local time (:mod:`airtrace.openmeteo`).
2. **Features** - :func:`airtrace.features.daily_features`, the exact code used for training;
   ``elevation`` is the weather response's DEM elevation, as in training.
3. **Forecast** - :func:`airtrace.model.predict` gives daily PM2.5 with a calibrated 80 %
   interval for today .. today+4 (local "today", days with >= 18 valid hours only) and a
   hindcast for the past days; :func:`airtrace.model.contributions` (TreeSHAP, log space) is
   summed per :data:`airtrace.features.FEATURE_GROUPS` into multiplicative driver effects.
4. **Hourly** - CAMS hourly PM2.5 rescaled by the day's AirTrace/CAMS ratio, plus wind.
5. **Sources** - :meth:`airtrace.sources.SourceModel.shares` over the 12-day window.
6. **Fires** - NASA FIRMS detections within 300 km, upwind of the next-24-h mean wind.
7. **Monitor** - AirNow status of the nearest reference monitor (``active`` / ``offline`` /
   ``none``) and, for an active one, the last <= 7 observed days against CAMS and AirTrace.
8. **History** - static training-year comparison from ``data/static/city_history.json``.

Robustness: one city failing never stops the run; its previous file (and summary) is kept.
The optional data sources (source model, fires, AirNow, history) degrade to ``null`` /
previous values instead of failing the build.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import openmeteo
from .airnow_live import EMBASSY_NETWORK, STATIC_DIR, AirNowLive
from .aqi import category, pm25_to_aqi
from .config import CAMS_VARS, CONFIG, PROCESSED, SITE_DATA, WEATHER_VARS
from .features import FEATURE_GROUPS, cams_frame, daily_features, weather_frame
from .fires import fires_near, load_fires, mean_wind_from

PAST_DAYS = 7
FORECAST_DAYS = 5
RECENT_DAYS = 7
N_DRIVERS = 4
WORKERS = 4
MAX_FAIL_FRACTION = 0.30          # exit non-zero only when more cities than this fail
BREAKER_FAILURES = 8              # stop calling Open-Meteo after this many failures in a row

CITIES_CSV = CONFIG / "showcase_cities.csv"
TRAINING_SITES_CSV = CONFIG / "training_sites.csv"
DATASET_PARQUET = PROCESSED / "dataset.parquet"
LOCAL_KM = 25.0                   # a training monitor this close counts as "this city's own"
REGIONAL_KM = 1000.0              # beyond this distance to any training monitor: extrapolation
MIN_CHECK_DAYS = 3                # live_check ignores cities with fewer observed days
HISTORY_PATH = STATIC_DIR / "city_history.json"
CITY_DIR = SITE_DATA / "city"
CITIES_JSON = SITE_DATA / "cities.json"
META_JSON = SITE_DATA / "meta.json"

DATA_SOURCES = [
    "Copernicus Atmosphere Monitoring Service (CAMS) global atmospheric composition forecasts, via the Open-Meteo Air Quality API",
    "Weather forecasts (Open-Meteo Forecast API) and ERA5 reanalysis for training (Open-Meteo Historical Weather API)",
    "US EPA AirNow reference PM2.5 observations, including US Department of State embassy and consulate monitors",
    "NASA FIRMS active fire detections (VIIRS, Suomi NPP, last 24 hours)",
]


# ----------------------------------------------------------------------------- small helpers
def r1(x: Any, nd: int = 1) -> float | None:
    """Round to ``nd`` decimals; None for missing / non-finite values (JSON-safe)."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def iso_utc(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, obj: Any, indent: int | None = None) -> None:
    """Atomic JSON write (the site or CI may read while we write); rejects NaN."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=indent, allow_nan=False, ensure_ascii=False,
                              separators=None if indent else (",", ":")) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@dataclass(frozen=True)
class City:
    id: str
    name: str
    country: str
    region: str
    lat: float
    lon: float
    embassy_site_id: str | None


def load_cities(path: Path = CITIES_CSV, only: list[str] | None = None) -> list[City]:
    df = pd.read_csv(path, dtype={"embassy_site_id": str}, keep_default_na=False)
    cities = [City(r.id, r.name, r.country, r.region, float(r.lat), float(r.lon), r.embassy_site_id or None)
              for r in df.itertuples(index=False)]
    if only:
        unknown = sorted(set(only) - {c.id for c in cities})
        if unknown:
            raise ValueError(f"unknown city id(s): {', '.join(unknown)}")
        cities = [c for c in cities if c.id in set(only)]
    return cities


# ----------------------------------------------------------------------------- inputs
# Open-Meteo weighted-call cost of the two requests per city (~1.1 + 1.0).
CAMS_WEIGHT = openmeteo.call_weight(len(CAMS_VARS), PAST_DAYS + FORECAST_DAYS)
WEATHER_WEIGHT = openmeteo.call_weight(len(WEATHER_VARS), PAST_DAYS + FORECAST_DAYS)


@dataclass
class Fetcher:
    """Fetches both forecasts for a city. ``replay_dir`` (development only) stores responses and
    replays them on later runs to save API quota. A circuit breaker stops calling the API after
    ``BREAKER_FAILURES`` consecutive failures (e.g. the daily quota is exhausted)."""
    replay_dir: Path | None = None
    api_weight: float = 0.0
    consecutive_failures: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _count(self, weight: float) -> None:
        with self._lock:
            self.api_weight += weight

    def __call__(self, city: City) -> tuple[dict, dict]:
        if self.replay_dir is not None:
            paths = [self.replay_dir / f"{city.id}_{k}.json" for k in ("cams", "weather")]
            if all(p.exists() for p in paths):
                return tuple(json.loads(p.read_text()) for p in paths)  # type: ignore[return-value]
        with self._lock:
            if self.consecutive_failures >= BREAKER_FAILURES:
                raise RuntimeError("Open-Meteo circuit breaker open (too many consecutive failures)")
        try:
            cams = openmeteo.cams_forecast(city.lat, city.lon, past_days=PAST_DAYS, forecast_days=FORECAST_DAYS)
            self._count(CAMS_WEIGHT)
            wx = openmeteo.weather_forecast(city.lat, city.lon, past_days=PAST_DAYS, forecast_days=FORECAST_DAYS)
            self._count(WEATHER_WEIGHT)
        except Exception:
            with self._lock:
                self.consecutive_failures += 1
            raise
        with self._lock:
            self.consecutive_failures = 0
        if self.replay_dir is not None:
            self.replay_dir.mkdir(parents=True, exist_ok=True)
            for k, resp in (("cams", cams), ("weather", wx)):
                (self.replay_dir / f"{city.id}_{k}.json").write_text(json.dumps(resp))
        return cams, wx


@dataclass
class LiveContext:
    """Everything shared by all cities of one run."""
    models: dict
    now_utc: datetime
    source_model: Any = None
    fires: pd.DataFrame | None = None
    airnow: AirNowLive | None = None
    history: dict = field(default_factory=dict)
    training_sites: pd.DataFrame | None = None
    metrics: dict = field(default_factory=dict)
    cams_max: dict = field(default_factory=dict)   # highest training CAMS PM2.5 per region (+ "_all")

    @property
    def generated_at(self) -> str:
        return iso_utc(self.now_utc)


# ----------------------------------------------------------------------------- per-city pieces
def drivers(contrib_row: pd.Series, top: int = N_DRIVERS) -> list[dict]:
    """Top feature groups by |effect|; effect_pct = (exp(sum of the group's log-space SHAP) - 1) * 100,
    i.e. how much the group raises or lowers the forecast relative to the average training day."""
    effects = []
    for group, feats in FEATURE_GROUPS.items():
        cols = [f for f in feats if f in contrib_row.index]
        s = float(np.nansum(contrib_row[cols].to_numpy(float))) if cols else 0.0
        e = (math.exp(min(s, 20.0)) - 1.0) * 100.0
        effects.append((group, max(e, -99.9)))
    effects.sort(key=lambda t: abs(t[1]), reverse=True)
    return [{"group": g, "effect_pct": round(e, 1)} for g, e in effects[:top]]


def day_entry(date: pd.Timestamp, pm: float, lo: float, hi: float, cams_pm: float,
              contrib_row: pd.Series) -> dict:
    pm_r = r1(pm)
    cat = category(pm_r)
    return {"date": f"{date:%Y-%m-%d}", "pm25": pm_r, "pm25_low": r1(lo), "pm25_high": r1(hi),
            "cams_pm25": r1(cams_pm), "aqi": pm25_to_aqi(pm_r), "category": cat["category"],
            "color": cat["color"], "drivers": drivers(contrib_row)}


def hourly_block(cams: pd.DataFrame, wx: pd.DataFrame, start: pd.Timestamp,
                 ratio_by_day: pd.Series) -> dict:
    """Hourly series from local midnight today to the end of the CAMS forecast.
    AirTrace hourly = CAMS hourly * (AirTrace daily / CAMS daily) of that day."""
    h = wx[["wind_speed_10m", "wind_direction_10m"]].join(cams[["pm2_5"]], how="outer")
    h = h[h.index >= start]
    valid = h["pm2_5"].notna().to_numpy()
    if valid.any():                                   # drop hours beyond the CAMS horizon
        h = h.iloc[: int(np.flatnonzero(valid)[-1]) + 1]
    ratio = ratio_by_day.reindex(h.index.normalize()).to_numpy(float)
    airtrace = h["pm2_5"].to_numpy(float) * ratio
    return {
        "time": [f"{t:%Y-%m-%dT%H:%M}" for t in h.index],
        "cams_pm25": [r1(v) for v in h["pm2_5"]],
        "airtrace_pm25": [r1(v) for v in airtrace],
        "wind_speed": [r1(v) for v in h["wind_speed_10m"]],
        "wind_dir": [None if r1(v, 0) is None else int(r1(v, 0)) for v in h["wind_direction_10m"]],
    }


def sources_block(source_model: Any, daily: pd.DataFrame, city_id: str = "",
                  pm25: pd.Series | None = None) -> dict | None:
    """Source shares over the window's days, each day's contributions scaled to the AirTrace
    PM2.5 of that day; ``None`` (with a warning) if attribution fails."""
    if source_model is None:
        return None
    try:
        res = source_model.shares(daily, pm25=pm25)
    except Exception as exc:                                       # noqa: BLE001
        print(f"  warn {city_id}: source attribution unavailable ({type(exc).__name__}: {exc})", flush=True)
        return None
    cols = [c for c in getattr(source_model, "tracer_columns", []) if c in daily.columns]
    complete = daily[cols].notna().all(axis=1) if cols else pd.Series(True, index=daily.index)
    used = daily.index[complete] if complete.any() else daily.index
    return {"window": f"{used.min():%Y-%m-%d} to {used.max():%Y-%m-%d}",
            "shares": res["shares"], "background_share": res["background_share"],
            "pm25": res.get("pm25")}


def monitor_block(ctx: LiveContext, city: City, previous: dict | None) -> dict:
    """Monitor dict incl. internal ``recent_obs``; falls back when AirNow is unavailable."""
    if ctx.airnow is not None:
        return ctx.airnow.monitor_for(city.lat, city.lon, city.embassy_site_id)
    if previous and isinstance(previous.get("monitor"), dict):
        return {**previous["monitor"], "recent_obs": {}}
    if city.embassy_site_id:
        return {"status": "offline", "network": EMBASSY_NETWORK, "name": city.name,
                "last_seen": None, "distance_km": 0.0, "recent_obs": {}}
    return {"status": "none", "network": None, "name": None, "last_seen": None,
            "distance_km": None, "recent_obs": {}}


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    p = math.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def load_training_context() -> tuple[pd.DataFrame, dict, dict]:
    """Training monitors, validation metrics and the CAMS PM2.5 range seen in training."""
    sites = pd.read_csv(TRAINING_SITES_CSV, dtype={"site_id": str})
    metrics = read_json(SITE_DATA / "metrics.json") or {}
    cams_max: dict = {}
    if DATASET_PARQUET.exists():
        d = pd.read_parquet(DATASET_PARQUET, columns=["region", "cams_pm2_5_mean"])
        cams_max = {k: float(v) for k, v in d.groupby("region")["cams_pm2_5_mean"].max().items()}
        cams_max["_all"] = float(d["cams_pm2_5_mean"].max())
    return sites, metrics, cams_max


def confidence_block(city: City, ctx: LiveContext) -> dict | None:
    """How much to trust this city's forecast, from where AirTrace was trained and validated:
    'local' (learned from this city's own monitor), 'regional' (a training monitor within
    REGIONAL_KM) or 'extrapolation'. The quoted errors come from metrics.json."""
    sites = ctx.training_sites
    if sites is None or sites.empty:
        return None
    dist = haversine_km(city.lat, city.lon, sites["lat"].to_numpy(float), sites["lon"].to_numpy(float))
    i = int(np.argmin(dist))
    near, d_km = sites.iloc[i], float(dist[i])
    models = {m["key"]: m for m in ctx.metrics.get("models", [])}
    same = ctx.metrics.get("temporal_same_sites", {})
    own = bool(city.embassy_site_id) and city.embassy_site_id in set(sites["site_id"])
    if own or d_km <= LOCAL_KM:
        tier, label = "local", "Trained on this city's own monitor"
        detail = (f"AirTrace learned this location's bias from {int(near['n_days'])} days of reference data "
                  f"at {near['city']} (Mar 2024 to Feb 2025).")
        if same:
            detail += (f" At monitors it had learned, error on later months fell from "
                       f"{same['cams_raw']['mae']:.1f} (CAMS) to {same['airtrace']['mae']:.1f} µg/m³.")
    elif d_km <= REGIONAL_KM:
        tier, label = "regional", "Region covered by training monitors"
        detail = f"Nearest training monitor: {near['city']} ({d_km:,.0f} km)."
        if "airtrace" in models:
            detail += (f" At locations the model never saw, cross-validated error fell from "
                       f"{models['cams_raw']['mae']:.1f} (CAMS) to {models['airtrace']['mae']:.1f} µg/m³.")
    else:
        tier, label = "extrapolation", "Outside training coverage"
        detail = (f"No training monitor within {REGIONAL_KM:,.0f} km (nearest: {near['city']}, {d_km:,.0f} km). "
                  f"When whole world regions were held out, the ML correction did not beat CAMS, "
                  f"so treat this forecast as indicative.")
    return {"tier": tier, "label": label, "detail": detail,
            "nearest_training_monitor": str(near["city"]), "distance_km": r1(d_km)}


def flag_out_of_range(entry: dict, cams_max: float | None) -> dict:
    """Days whose CAMS PM2.5 exceeds anything seen in training for the region are extrapolations:
    flag them and widen the upper bound to at least the raw CAMS value (never falsely reassure)."""
    cams = entry.get("cams_pm25")
    if cams_max is None or cams is None or cams <= cams_max:
        entry["out_of_range"] = False
        return entry
    entry["out_of_range"] = True
    entry["pm25_high"] = r1(max(entry.get("pm25_high") or 0.0, cams))
    entry["note"] = (f"CAMS forecasts {cams:.0f} µg/m³, above the highest value in our training data for this "
                     f"region ({cams_max:.0f} µg/m³); the ML correction is extrapolated here.")
    return entry


def build_city(city: City, cams_resp: dict, wx_resp: dict, ctx: LiveContext,
               previous: dict | None = None) -> dict:
    """The ``city/<id>.json`` document for one city (see SPEC "Website data contract")."""
    from .model import contributions, predict      # ML-owned; imported lazily

    cams, wx = cams_frame(cams_resp), weather_frame(wx_resp)
    offset = int(wx_resp.get("utc_offset_seconds", 0))
    shift = offset - int(cams_resp.get("utc_offset_seconds", offset))
    if shift:                                     # both use timezone=auto; never assume it
        cams.index = cams.index + pd.Timedelta(seconds=shift)
    elevation = wx_resp.get("elevation")
    elevation = float(elevation) if elevation is not None and math.isfinite(float(elevation)) else 0.0

    daily = daily_features(cams, wx, city.lat, city.lon, elevation)
    if daily.empty:
        raise ValueError("no day with >= 18 h of CAMS data")
    local_now = ctx.now_utc.replace(tzinfo=None) + timedelta(seconds=offset)
    today = pd.Timestamp(local_now.date())
    pred = predict(ctx.models, daily)
    contrib = contributions(ctx.models, daily)

    fc_days = [d for d in daily.index if d >= today][:FORECAST_DAYS]
    if not fc_days:
        raise ValueError(f"no forecast day >= {today:%Y-%m-%d} with >= 18 h of data")
    cams_max = ctx.cams_max.get(city.region, ctx.cams_max.get("_all"))
    daily_out = [flag_out_of_range(day_entry(d, pred.at[d, "pm25"], pred.at[d, "pm25_low"],
                                             pred.at[d, "pm25_high"], daily.at[d, "cams_pm2_5_mean"],
                                             contrib.loc[d]), cams_max) for d in fc_days]
    first = daily_out[0]
    cat = category(first["pm25"])
    today_out = {"date": first["date"], "pm25": first["pm25"], "aqi": first["aqi"],
                 "category": cat["category"], "color": cat["color"], "message": cat["message"],
                 "out_of_range": first["out_of_range"]}

    cams_daily = daily["cams_pm2_5_mean"]
    ratio = (pred["pm25"] / cams_daily).where(cams_daily > 0.05)
    hourly = hourly_block(cams, wx, today, ratio)

    hour0 = local_now.replace(minute=0, second=0, microsecond=0)
    nxt = wx.loc[(wx.index >= hour0) & (wx.index < hour0 + timedelta(hours=24))]
    wind_from = mean_wind_from(nxt["wind_speed_10m"], nxt["wind_direction_10m"])
    fires = fires_near(city.lat, city.lon, wind_from, ctx.fires) if ctx.fires is not None else None

    mon = monitor_block(ctx, city, previous)
    recent = []
    if mon["status"] == "active":
        start = today - timedelta(days=RECENT_DAYS)
        for ds, obs in sorted(mon.get("recent_obs", {}).items()):
            d = pd.Timestamp(ds)
            if start <= d < today:
                has = d in daily.index
                recent.append({"date": ds, "observed": r1(obs),
                               "cams": r1(cams_daily.at[d]) if has else None,
                               "airtrace": r1(pred.at[d, "pm25"]) if has else None})
        recent = recent[-RECENT_DAYS:]

    return {
        "id": city.id, "name": city.name, "country": city.country, "region": city.region,
        "lat": city.lat, "lon": city.lon, "timezone": wx_resp.get("timezone") or "GMT",
        "generated_at": ctx.generated_at,
        "monitor": {k: mon.get(k) for k in ("status", "network", "name", "last_seen", "distance_km")},
        "confidence": confidence_block(city, ctx),
        "today": today_out,
        "daily": daily_out,
        "hourly": hourly,
        "sources": sources_block(ctx.source_model, daily, city.id, pred["pm25"]),
        "fires": fires,
        "recent": recent,
        "history": ctx.history.get(city.id),
    }


def summarize(doc: dict) -> dict:
    """The city's entry in ``cities.json``."""
    today = doc["today"]
    peak = max(doc["daily"], key=lambda d: -1 if d["aqi"] is None else d["aqi"])
    peak_cat = category(peak["pm25"])
    top = None
    if doc.get("sources"):
        top = next((s["label"] for s in doc["sources"]["shares"]
                    if s["key"] != "background" and s["share"] > 0), None)
    return {"id": doc["id"], "name": doc["name"], "country": doc["country"], "region": doc["region"],
            "lat": doc["lat"], "lon": doc["lon"], "monitor_status": doc["monitor"]["status"],
            "today_pm25": today["pm25"], "today_aqi": today["aqi"], "today_category": today["category"],
            "today_color": today["color"], "peak_aqi_5d": peak["aqi"],
            "peak_category_5d": peak_cat["category"], "peak_color_5d": peak_cat["color"],
            "top_source": top, "confidence": (doc.get("confidence") or {}).get("tier")}


# ----------------------------------------------------------------------------- whole run
def _optional(name: str, fn):
    """Load an optional input; on failure log it and continue without it."""
    t = time.time()
    try:
        value = fn()
        print(f"  {name}: ok ({time.time() - t:.1f}s)", flush=True)
        return value
    except Exception as exc:                                       # noqa: BLE001
        print(f"  {name}: UNAVAILABLE ({type(exc).__name__}: {exc}) - continuing without it", flush=True)
        return None


def _load_source_model():
    from .sources import load_source_model      # SOURCES-owned; optional for the build
    return load_source_model()


def live_check(docs: list[dict | None]) -> dict | None:
    """Hindcast check on brand-new data: the last days' AirTrace and CAMS daily estimates (built
    from the same short-lead CAMS fields and analysed weather the site uses for "today") against
    AirNow observations at every city whose monitor still reports at least MIN_CHECK_DAYS days.
    The model was trained on Mar 2024 - Feb 2025, so all of these days are unseen. Pooled over
    city-days."""
    rows, cities = [], []
    for doc in docs:
        if not doc or (doc.get("monitor") or {}).get("status") != "active":
            continue
        rec = [r for r in doc.get("recent") or []
               if all(isinstance(r.get(k), (int, float)) for k in ("observed", "cams", "airtrace"))]
        if len(rec) < MIN_CHECK_DAYS:
            continue
        obs = np.array([r["observed"] for r in rec], float)
        cams = np.array([r["cams"] for r in rec], float)
        at = np.array([r["airtrace"] for r in rec], float)
        rows.append((obs, cams, at, [r["date"] for r in rec]))
        cities.append({"id": doc["id"], "name": doc["name"], "n": len(rec),
                       "mae_cams": r1(float(np.mean(np.abs(cams - obs)))),
                       "mae_airtrace": r1(float(np.mean(np.abs(at - obs))))})
    if not rows:
        return None
    obs = np.concatenate([r[0] for r in rows])
    cams = np.concatenate([r[1] for r in rows])
    at = np.concatenate([r[2] for r in rows])
    dates = sorted(d for r in rows for d in r[3])
    return {"window": f"{dates[0]} to {dates[-1]}", "n_cities": len(cities), "n_city_days": int(len(obs)),
            "mae_cams": r1(float(np.mean(np.abs(cams - obs)))),
            "mae_airtrace": r1(float(np.mean(np.abs(at - obs)))),
            "bias_cams": r1(float(np.mean(cams - obs))), "bias_airtrace": r1(float(np.mean(at - obs))),
            "cities": sorted(cities, key=lambda c: c["name"])}


def run(only: list[str] | None = None, workers: int = WORKERS, replay_dir: Path | None = None,
        now_utc: datetime | None = None) -> int:
    """Build docs/data for all (or ``only``) showcase cities. Returns the process exit code."""
    from .model import load_models               # required (ML-owned)

    t0 = time.time()
    all_cities = load_cities()
    cities = load_cities(only=only) if only else all_cities
    now_utc = (now_utc or datetime.now(timezone.utc)).replace(microsecond=0)
    print(f"AirTrace live build {iso_utc(now_utc)}: {len(cities)} cities", flush=True)

    models = load_models()                        # required: without the model there is nothing to show
    ctx = LiveContext(models=models, now_utc=now_utc)
    ctx.source_model = _optional("source model", _load_source_model)
    ctx.fires = _optional("NASA FIRMS fires", load_fires)
    ctx.airnow = _optional("AirNow observations", lambda: AirNowLive(today_utc=now_utc.date()))
    ctx.history = _optional("city history", lambda: json.loads(HISTORY_PATH.read_text())) or {}
    training = _optional("training coverage", load_training_context)
    if training is not None:
        ctx.training_sites, ctx.metrics, ctx.cams_max = training
    if ctx.airnow is not None:
        print(f"    AirNow files {ctx.airnow.dates[-1]}..{ctx.airnow.dates[0]}, "
              f"{len(ctx.airnow.last_obs)} PM2.5 sites reporting", flush=True)

    fetch = Fetcher(replay_dir=replay_dir)

    def job(city: City):
        t = time.time()
        try:
            return city, fetch(city), None, time.time() - t
        except Exception as exc:                                   # noqa: BLE001
            return city, None, exc, time.time() - t

    done: dict[str, dict] = {}
    failed: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        # Downloads run in the pool; model inference runs here, one city at a time.
        for city, resp, err, dt in ex.map(job, cities):
            path = CITY_DIR / f"{city.id}.json"
            previous = read_json(path)
            try:
                if err is not None:
                    raise err
                doc = build_city(city, *resp, ctx, previous)
                write_json(path, doc)
                done[city.id] = summarize(doc)
                d0 = doc["daily"][0]
                print(f"  ok   {city.id:<18} today {d0['pm25']:>6} ug/m3 (CAMS {d0['cams_pm25']}) "
                      f"AQI {d0['aqi']:<3} monitor {doc['monitor']['status']:<7} "
                      f"fires {doc['fires']['count_300km'] if doc['fires'] else '-':<4} [{dt:.1f}s]", flush=True)
            except Exception as exc:                               # noqa: BLE001
                failed[city.id] = f"{type(exc).__name__}: {exc}"
                print(f"  FAIL {city.id:<18} {failed[city.id]}", flush=True)
                if not isinstance(exc, RuntimeError):
                    traceback.print_exc()

    # cities.json keeps the CSV order; cities not rebuilt now keep their existing file + summary.
    summaries = []
    for city in all_cities:
        if city.id in done:
            summaries.append(done[city.id])
            continue
        prev = read_json(CITY_DIR / f"{city.id}.json")
        if prev is not None:
            try:
                summaries.append(summarize(prev))
            except (KeyError, TypeError, ValueError):
                pass
    if summaries:
        write_json(CITIES_JSON, summaries, indent=1)
        meta = {"generated_at": ctx.generated_at,
                "model_version": str(models["meta"].get("version", "unknown")),
                "n_cities": len(summaries), "forecast_days": FORECAST_DAYS,
                "data_sources": DATA_SOURCES}
        check = live_check([read_json(CITY_DIR / f"{c.id}.json") for c in all_cities])
        if check is not None:
            meta["live_check"] = check
        write_json(META_JSON, meta, indent=1)

    frac = len(failed) / max(len(cities), 1)
    print(f"done in {time.time() - t0:.0f}s: {len(done)} built, {len(failed)} failed ({frac:.0%}); "
          f"cities.json has {len(summaries)} cities; Open-Meteo weight used {fetch.api_weight:.1f}", flush=True)
    for cid, msg in failed.items():
        print(f"  failed {cid}: {msg}")
    return 1 if frac > MAX_FAIL_FRACTION else 0
