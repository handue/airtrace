"""Active fires near a city from NASA FIRMS (VIIRS on Suomi NPP, last 24 h, keyless CSV).

The global 24-hour file is downloaded once per run (and cached on disk for a few hours so
repeated development runs do not re-download it). For a city we report fires within
300 km, how many lie *upwind* - in the direction the wind blows from over the next 24 h,
i.e. where smoke would be carried towards the city - plus the nearest distance, the
total fire radiative power (FRP, MW) and the nearest points for the map.
"""
from __future__ import annotations

import io
import math
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .airnow_live import haversine_km
from .config import FIRMS_24H_URL, RAW

FIRMS_DIR = RAW / "firms"
CACHE_FILE = FIRMS_DIR / "viirs_snpp_global_24h.csv"
CACHE_MAX_AGE_S = 3 * 3600      # FIRMS NRT updates every few hours
STALE_MAX_AGE_S = 48 * 3600     # never show a "last 24 h" fire map older than this
RADIUS_KM = 300.0
UPWIND_HALF_ANGLE = 45.0        # degrees either side of the mean wind-from direction
MAX_POINTS = 300
EXCLUDE_CONFIDENCE = {"l", "low"}   # VIIRS low-confidence pixels are often glint/false alarms

_LOCK = threading.Lock()
_FIRES: pd.DataFrame | None = None


def _parse(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), usecols=lambda c: c in {"latitude", "longitude", "frp", "confidence"})
    for c in ("latitude", "longitude", "frp"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    if "confidence" in df:
        df = df[~df["confidence"].astype(str).str.strip().str.lower().isin(EXCLUDE_CONFIDENCE)]
    df["frp"] = df["frp"].fillna(0.0).clip(lower=0.0)
    return df[["latitude", "longitude", "frp"]].reset_index(drop=True)


def load_fires(url: str = FIRMS_24H_URL, cache: Path = CACHE_FILE,
               max_age_s: float = CACHE_MAX_AGE_S, retries: int = 3) -> pd.DataFrame:
    """Global VIIRS fire detections of the last 24 h (latitude, longitude, frp).

    Downloaded at most once per process (later calls return the same frame); a cached copy
    younger than ``max_age_s`` is reused. If the download fails, a cached copy up to
    ``STALE_MAX_AGE_S`` old is used; otherwise raises (the build then omits fires).
    """
    global _FIRES
    with _LOCK:
        if _FIRES is not None:
            return _FIRES

        def age() -> float:
            return time.time() - cache.stat().st_mtime if cache.exists() else math.inf

        if age() >= max_age_s:
            for attempt in range(retries):
                try:
                    r = requests.get(url, timeout=120)
                    r.raise_for_status()
                    if not r.text.startswith("latitude"):
                        raise ValueError("unexpected FIRMS response (no CSV header)")
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    tmp = cache.with_name(cache.name + ".part")
                    tmp.write_text(r.text)
                    tmp.replace(cache)
                    break
                except (requests.RequestException, ValueError) as exc:
                    print(f"[fires] download attempt {attempt + 1} failed: {exc}", flush=True)
                    time.sleep(5 * (attempt + 1))
        if age() > STALE_MAX_AGE_S:
            raise RuntimeError("FIRMS fire data unavailable (download failed, no recent cached copy)")
        _FIRES = _parse(cache.read_text())
        return _FIRES


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial great-circle bearing from point 1 to point 2, degrees clockwise from north."""
    p = np.pi / 180.0
    y = np.sin((lon2 - lon1) * p) * np.cos(lat2 * p)
    x = np.cos(lat1 * p) * np.sin(lat2 * p) - np.sin(lat1 * p) * np.cos(lat2 * p) * np.cos((lon2 - lon1) * p)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def mean_wind_from(speed, direction) -> float | None:
    """Direction (deg) the wind blows FROM, from the vector mean of hourly winds.

    ``direction`` follows the meteorological convention (direction the wind comes from).
    Returns ``None`` when no valid hour exists or the mean vector vanishes (calm/variable).
    """
    s = np.asarray(speed, dtype=float)
    d = np.asarray(direction, dtype=float)
    ok = np.isfinite(s) & np.isfinite(d)
    if not ok.any():
        return None
    rad = np.deg2rad(d[ok])
    # vector pointing to where the wind comes from, weighted by speed
    x, y = np.sum(s[ok] * np.sin(rad)), np.sum(s[ok] * np.cos(rad))
    if np.hypot(x, y) < 1e-9:
        return None
    return float((np.degrees(np.arctan2(x, y)) + 360.0) % 360.0)


def angle_diff(a, b):
    """Smallest absolute difference between two angles in degrees (0..180)."""
    return np.abs((np.asarray(a, dtype=float) - b + 180.0) % 360.0 - 180.0)


def fires_near(lat: float, lon: float, wind_from_deg: float | None, fires: pd.DataFrame | None = None,
               radius_km: float = RADIUS_KM, half_angle: float = UPWIND_HALF_ANGLE,
               max_points: int = MAX_POINTS) -> dict:
    """SPEC ``fires`` block for a city: count within ``radius_km``, upwind count, nearest
    distance (km), summed FRP (MW) and up to ``max_points`` nearest [lat, lon, frp]."""
    if fires is None:
        fires = load_fires()
    # cheap bounding-box prefilter before the exact distance
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(np.cos(np.deg2rad(lat)), 0.05))
    box = fires[(fires["latitude"].between(lat - dlat, lat + dlat))
                & (((fires["longitude"] - lon + 180.0) % 360.0 - 180.0).abs() <= dlon)]
    dist = haversine_km(lat, lon, box["latitude"].to_numpy(), box["longitude"].to_numpy())
    near = box.assign(dist=dist)[dist <= radius_km].sort_values("dist")
    upwind = 0
    if wind_from_deg is not None and len(near):
        brg = bearing_deg(lat, lon, near["latitude"].to_numpy(), near["longitude"].to_numpy())
        upwind = int((angle_diff(brg, wind_from_deg) <= half_angle).sum())
    pts = near.head(max_points)
    return {
        "count_300km": int(len(near)),
        "upwind_count": upwind,
        "nearest_km": round(float(near["dist"].iloc[0]), 1) if len(near) else None,
        "frp_sum": round(float(near["frp"].sum()), 1),
        "points": [[round(float(a), 3), round(float(b), 3), round(float(f), 1)]
                   for a, b, f in zip(pts["latitude"], pts["longitude"], pts["frp"])],
    }
