"""Training table for the AirTrace PM2.5 model.

One row per (reference monitor, local calendar day) with

* ``pm25_obs`` / ``obs_hours``: observed 24-h mean PM2.5 (ug/m3) from AirNow and the number of
  valid hours behind it (AirNow daily files use the local standard-time day, which matches
  Open-Meteo's ``timezone=auto`` fixed-offset local time);
* every model input in :data:`airtrace.features.FEATURES` plus the daily CAMS tracer means
  (:data:`airtrace.features.TRACER_COLUMNS`, used by the source-attribution model) and
  ``n_hours``, all built by :func:`airtrace.features.daily_features` -- the same code the live
  forecast uses, so training and serving cannot drift apart;
* site metadata (``site_id, city, country, country_name, region, network, lat, lon``) and the
  leave-location-out CV group ``group`` = the rounded 1-degree lat/lon cell.

Open-Meteo data is read strictly from the on-disk cache through the shared client
(:mod:`airtrace.openmeteo`, called with the arguments the downloader used, i.e. a cache hit).
A site whose CAMS or ERA5 file is missing or still being written is skipped, never fetched, so
the build is safe to re-run while the background download fills in. Any accidental network
call raises instead of spending the nearly exhausted archive quota.

``elevation`` is the Open-Meteo DEM elevation reported in the weather response (the AirNow
site list has 0 m for most embassy monitors). The live forecast must pass the same quantity:
``weather_forecast(...)["elevation"]``.
"""
from __future__ import annotations

import json
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from . import openmeteo
from .config import CAMS_VARS, CONFIG, PROCESSED, RAW_OPENMETEO, TRAIN_END, TRAIN_START, WEATHER_VARS
from .features import FEATURES, TRACER_COLUMNS, cams_frame, daily_features, hourly_frame, weather_frame

SITES_PATH = CONFIG / "training_sites.csv"
OBS_PATH = PROCESSED / "airnow_pm25_daily.parquet"
DATASET_PATH = PROCESSED / "dataset.parquet"

# Window of the ERA5 boundary-layer-height patch (Open-Meteo's best-match archive has no BLH then).
BLH_PATCH_START, BLH_PATCH_END = "2024-03-01", "2024-06-30"

OBS = "pm25_obs"
GROUP = "group"
META_COLUMNS = ["site_id", "city", "country", "country_name", "region", "network", "lat", "lon"]
# A usable row needs the CAMS PM2.5 baseline and basic weather; every other feature may be NaN
# (e.g. mixing height before the BLH patch exists) because LightGBM handles missing values.
CORE_FEATURES = ["cams_pm2_5_mean", "t2m_mean", "rh_mean", "ws_mean"]


def location_group(lat: float, lon: float) -> str:
    """Leave-location-out CV group: the 1-degree lat/lon cell, so co-located monitors never
    end up on both sides of a split."""
    return f"{round(float(lat))}_{round(float(lon))}"


def cache_paths(lat: float, lon: float) -> dict[str, Path]:
    """Cache files that :mod:`airtrace.openmeteo` reads for one training site (same naming)."""
    stem = f"{lat:.4f}_{lon:.4f}"
    return {
        "cams": RAW_OPENMETEO / f"cams_{stem}_{TRAIN_START}_{TRAIN_END}_{openmeteo._key(*CAMS_VARS)}.json",
        "era5": RAW_OPENMETEO / f"era5_{stem}_{TRAIN_START}_{TRAIN_END}_{openmeteo._key(*WEATHER_VARS)}.json",
        "blh": RAW_OPENMETEO / f"era5blh_{stem}_{BLH_PATCH_START}_{BLH_PATCH_END}.json",
    }


def _complete_json(path: Path) -> bool:
    """True if ``path`` exists and parses as JSON.

    The downloader may be writing the file right now, and ``openmeteo._get`` deletes a cache
    file it cannot parse (then re-downloads it), so only files verified here are handed to it.
    """
    if not path.exists():
        return False
    try:
        json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, ValueError):
        return False
    return True


@contextmanager
def cache_only() -> Iterator[None]:
    """Turn any Open-Meteo network request into an error for the duration of the block."""
    def refuse(weight: float) -> None:
        raise RuntimeError("dataset build is cache-only: refusing to call the Open-Meteo API")

    original = openmeteo._throttle
    openmeteo._throttle = refuse          # _get() calls _throttle() before every request
    try:
        yield
    finally:
        openmeteo._throttle = original


def apply_blh_patch(wx: pd.DataFrame, wx_resp: dict, patch_resp: dict) -> tuple[pd.DataFrame, int]:
    """Fill missing hourly boundary-layer height in ``wx`` from the ERA5 patch response,
    aligned on the local timestamp. Returns the patched frame and the number of filled hours."""
    blh = hourly_frame(patch_resp, ["boundary_layer_height"])["boundary_layer_height"]
    shift = wx_resp.get("utc_offset_seconds", 0) - patch_resp.get("utc_offset_seconds", 0)
    if shift:                             # both use timezone=auto, but never assume equal offsets
        blh.index = blh.index + pd.Timedelta(seconds=shift)
    blh = blh[~blh.index.duplicated(keep="first")]
    missing = wx["boundary_layer_height"].isna()
    out = wx.copy()
    out["boundary_layer_height"] = out["boundary_layer_height"].fillna(blh.reindex(out.index))
    return out, int((missing & out["boundary_layer_height"].notna()).sum())


def site_features(lat: float, lon: float, fallback_elevation: float = 0.0) -> tuple[pd.DataFrame | None, dict]:
    """Daily features for one site from the cache, or ``(None, info)`` if it is not cached yet."""
    paths = cache_paths(lat, lon)
    info: dict = {"status": "ok", "blh_filled_hours": 0, "blh_patch": False, "elevation": None}
    if not (paths["cams"].exists() and paths["era5"].exists()):
        info["status"] = "not downloaded yet"
        return None, info
    if not (_complete_json(paths["cams"]) and _complete_json(paths["era5"])):
        info["status"] = "cache file incomplete (being written)"
        return None, info

    with cache_only():
        cams_resp = openmeteo.cams_archive(lat, lon, TRAIN_START, TRAIN_END)
        wx_resp = openmeteo.weather_archive(lat, lon, TRAIN_START, TRAIN_END)
        patch_resp = (openmeteo.era5_blh_patch(lat, lon, BLH_PATCH_START, BLH_PATCH_END)
                      if _complete_json(paths["blh"]) else None)

    cams, wx = cams_frame(cams_resp), weather_frame(wx_resp)
    if patch_resp is not None:
        wx, info["blh_filled_hours"] = apply_blh_patch(wx, wx_resp, patch_resp)
        info["blh_patch"] = True

    elevation = wx_resp.get("elevation")
    if elevation is None or not math.isfinite(float(elevation)):
        elevation = fallback_elevation
    info["elevation"] = float(elevation)
    return daily_features(cams, wx, lat, lon, float(elevation)), info


def load_sites(path: Path = SITES_PATH) -> pd.DataFrame:
    sites = pd.read_csv(path, dtype={"site_id": str, "country": str}, keep_default_na=False,
                        na_values={"elevation": [""], "lat": [""], "lon": [""]})
    sites["site_id"] = sites["site_id"].str.strip()
    return sites


def load_observations(path: Path = OBS_PATH) -> pd.DataFrame:
    obs = pd.read_parquet(path, columns=["aqsid", "date", "pm25_obs", "hours"])
    obs = obs.rename(columns={"aqsid": "site_id", "hours": "obs_hours"})
    obs["site_id"] = obs["site_id"].astype(str).str.strip()
    obs["date"] = pd.to_datetime(obs["date"]).dt.normalize().astype("datetime64[us]")
    return obs.dropna(subset=[OBS]).drop_duplicates(["site_id", "date"])


def build_dataset(sites: pd.DataFrame | None = None, obs: pd.DataFrame | None = None,
                  verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join cached CAMS + ERA5 features with AirNow observations for every cached site.

    Returns ``(dataset, site_report)``; ``site_report`` has one row per configured site with
    its build status, row count and BLH-patch information.
    """
    sites = load_sites() if sites is None else sites
    obs = load_observations() if obs is None else obs
    obs_by_site = dict(tuple(obs.groupby("site_id", sort=False)))

    frames, report = [], []
    for s in sites.itertuples(index=False):
        fallback = float(s.elevation) if pd.notna(s.elevation) else 0.0
        feats, info = site_features(float(s.lat), float(s.lon), fallback)
        row = {"site_id": s.site_id, "city": s.city, "region": s.region, "n_rows": 0, **info}
        if feats is not None:
            feats = feats.reset_index()
            feats["date"] = pd.to_datetime(feats["date"]).dt.normalize().astype("datetime64[us]")
            site_obs = obs_by_site.get(s.site_id)
            if site_obs is None:
                row["status"] = "no observations"
            else:
                d = site_obs[["date", OBS, "obs_hours"]].merge(feats, on="date", how="inner")
                d = d.dropna(subset=[OBS, *CORE_FEATURES])
                if d.empty:
                    row["status"] = "no overlapping days"
                else:
                    for col in META_COLUMNS:
                        d[col] = getattr(s, col)
                    d[GROUP] = location_group(s.lat, s.lon)
                    frames.append(d)
                    row["n_rows"] = len(d)
        report.append(row)
        if verbose and row["status"] != "ok":
            print(f"  skip {s.site_id:<13} {s.city:<24} {row['status']}", flush=True)

    report_df = pd.DataFrame(report)
    if not frames:
        return pd.DataFrame(columns=_column_order()), report_df
    data = pd.concat(frames, ignore_index=True)
    data = data[_column_order()].sort_values(["site_id", "date"], ignore_index=True)
    return data, report_df


def _column_order() -> list[str]:
    head = ["site_id", "date", OBS, "obs_hours", *[c for c in META_COLUMNS if c != "site_id"], GROUP]
    rest = [c for c in dict.fromkeys(FEATURES + TRACER_COLUMNS + ["n_hours"]) if c not in head]
    return head + rest


def save_dataset(data: pd.DataFrame, path: Path = DATASET_PATH) -> Path:
    """Write the table atomically (other pipeline steps may be reading the old file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    data.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    """Rows / sites / countries per region (plus a total row)."""
    by = (data.groupby("region")
          .agg(sites=("site_id", "nunique"), countries=("country", "nunique"), rows=(OBS, "size"),
               mean_pm25=(OBS, "mean"))
          .sort_values("rows", ascending=False))
    total = pd.DataFrame({"sites": [data["site_id"].nunique()], "countries": [data["country"].nunique()],
                          "rows": [len(data)], "mean_pm25": [data[OBS].mean()]}, index=["TOTAL"])
    out = pd.concat([by, total])
    out["mean_pm25"] = out["mean_pm25"].round(1)
    return out
