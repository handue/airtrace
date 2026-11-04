"""Static per-city history for the website: how raw CAMS and AirTrace compared with the
reference monitor over the training year, plus a pollution wind rose.

For every showcase city whose monitor is a training site - its US embassy monitor
(``embassy_site_id``), or for other cities a training monitor within 25 km - we take

* the leave-location-out cross-validation predictions (``airtrace`` is the *out-of-fold*
  prediction, i.e. made by a model that never saw this location) and the raw CAMS
  forecast, both against the observed 24-h mean;
* the daily mean wind (``u_mean``, ``v_mean`` from the training table), converted to the
  direction the wind comes FROM, binned into 16 compass sectors with the mean observed
  PM2.5 and the share of days per sector.

Inputs:  config/showcase_cities.csv, data/processed/cv_predictions.parquet,
         data/processed/dataset.parquet (both written by scripts/05 / 04)
Output:  data/static/city_history.json  {city_id: {period, n_days, mae_cams, mae_airtrace,
                                                   monthly[...], rose{sectors, mean_pm25, freq}}}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from airtrace.airnow_live import MONITOR_RADIUS_KM, STATIC_DIR, haversine_km  # noqa: E402
from airtrace.config import CONFIG, PROCESSED  # noqa: E402

CV_PATH = PROCESSED / "cv_predictions.parquet"
DATASET_PATH = PROCESSED / "dataset.parquet"
OUT_PATH = STATIC_DIR / "city_history.json"
SECTORS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
MIN_DAYS = 30            # fewer matched days than this: no history block for the city


def wind_from_deg(u, v) -> np.ndarray:
    """Meteorological direction (deg, the direction the wind comes FROM) of a (u, v) vector,
    where u/v are the eastward/northward components of the air motion."""
    return (np.degrees(np.arctan2(-np.asarray(u, float), -np.asarray(v, float))) + 360.0) % 360.0


def sector_index(deg) -> np.ndarray:
    """0..15 compass sector (0 = N, centred on north: 348.75..11.25 deg)."""
    return (np.floor((np.asarray(deg, float) + 11.25) / 22.5).astype(int)) % 16


def _r(x, nd: int = 1) -> float | None:
    return round(float(x), nd) if x is not None and np.isfinite(x) else None


def site_history(d: pd.DataFrame) -> dict:
    """History block for one site. ``d``: date, obs, cams, airtrace_oof, u_mean, v_mean."""
    d = d.dropna(subset=["obs", "cams", "airtrace_oof"]).sort_values("date")
    monthly = (d.groupby(d["date"].dt.strftime("%Y-%m"))
                .agg(observed=("obs", "mean"), cams=("cams", "mean"), airtrace=("airtrace_oof", "mean"))
                .reset_index(names="month"))
    w = d.dropna(subset=["u_mean", "v_mean"])
    sector = sector_index(wind_from_deg(w["u_mean"], w["v_mean"]))
    counts = np.bincount(sector, minlength=16)
    sums = np.bincount(sector, weights=w["obs"].to_numpy(float), minlength=16)
    return {
        "period": [f"{d['date'].min():%Y-%m-%d}", f"{d['date'].max():%Y-%m-%d}"],
        "n_days": int(len(d)),
        "mae_cams": _r((d["cams"] - d["obs"]).abs().mean(), 2),
        "mae_airtrace": _r((d["airtrace_oof"] - d["obs"]).abs().mean(), 2),
        "monthly": [{"month": m.month, "observed": _r(m.observed), "cams": _r(m.cams),
                     "airtrace": _r(m.airtrace)} for m in monthly.itertuples(index=False)],
        "rose": {
            "sectors": SECTORS,
            "mean_pm25": [_r(s / c) if c else None for s, c in zip(sums, counts)],
            "freq": [round(float(c) / max(len(w), 1), 3) for c in counts],
        },
    }


def match_sites(cities: pd.DataFrame, cv_sites: pd.DataFrame) -> dict[str, str]:
    """city_id -> training site_id: the embassy monitor, else the nearest one within 25 km."""
    out = {}
    for c in cities.itertuples(index=False):
        sid = str(c.embassy_site_id) if isinstance(c.embassy_site_id, str) and c.embassy_site_id else None
        if sid is None and len(cv_sites):
            dist = haversine_km(c.lat, c.lon, cv_sites["lat"].to_numpy(float), cv_sites["lon"].to_numpy(float))
            i = int(np.argmin(dist))
            if dist[i] <= MONITOR_RADIUS_KM:
                sid = cv_sites.index[i]
        if sid is not None and sid in cv_sites.index:
            out[c.id] = sid
    return out


def main() -> None:
    missing = [p for p in (CV_PATH, DATASET_PATH) if not p.exists()]
    if missing:
        sys.exit(f"missing input(s): {', '.join(str(p) for p in missing)} - run scripts 04/05 first")
    cities = pd.read_csv(CONFIG / "showcase_cities.csv", dtype={"embassy_site_id": str}, keep_default_na=False)
    cv = pd.read_parquet(CV_PATH, columns=["site_id", "date", "obs", "cams", "airtrace_oof"])
    ds = pd.read_parquet(DATASET_PATH, columns=["site_id", "date", "lat", "lon", "u_mean", "v_mean"])
    for df in (cv, ds):
        df["site_id"] = df["site_id"].astype(str)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    data = cv.merge(ds[["site_id", "date", "u_mean", "v_mean"]], on=["site_id", "date"], how="left")
    sites = ds.groupby("site_id")[["lat", "lon"]].first()
    sites = sites[sites.index.isin(cv["site_id"].unique())]

    by_site = dict(tuple(data.groupby("site_id")))
    history, skipped = {}, []
    for city_id, sid in sorted(match_sites(cities, sites).items()):
        h = site_history(by_site[sid])
        if h["n_days"] >= MIN_DAYS:
            history[city_id] = h
        else:
            skipped.append(f"{city_id} ({h['n_days']} days)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_name(OUT_PATH.name + ".tmp")
    tmp.write_text(json.dumps(history, indent=1, allow_nan=False) + "\n")
    tmp.replace(OUT_PATH)
    with_embassy = int((cities["embassy_site_id"] != "").sum())
    print(f"history for {len(history)} of {len(cities)} showcase cities "
          f"({with_embassy} have an embassy monitor; {sites.shape[0]} CV sites available) -> {OUT_PATH}")
    if skipped:
        print("  too few days: " + ", ".join(skipped))
    if history:
        mc = np.mean([h["mae_cams"] for h in history.values()])
        ma = np.mean([h["mae_airtrace"] for h in history.values()])
        print(f"  mean per-city MAE: CAMS {mc:.2f} -> AirTrace {ma:.2f} ug/m3")


if __name__ == "__main__":
    main()
