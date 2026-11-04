"""Small, cached, rate-limit-aware client for the keyless Open-Meteo APIs.

Open-Meteo's free tier counts *weighted* calls: a request is
max(1, n_vars/10) * max(1, n_days/14) calls, capped at 600/min, 5,000/h, 10,000/day.
Every response is cached on disk so a pipeline re-run never re-downloads.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import deque
from pathlib import Path

import requests

from .config import AQ_ARCHIVE_URL, CAMS_VARS, RAW_OPENMETEO, WEATHER_ARCHIVE_URL, WEATHER_FORECAST_URL, WEATHER_VARS

_LOCK = threading.Lock()
_HISTORY: deque[tuple[float, float]] = deque()   # (timestamp, weight)
MINUTE_BUDGET = 540.0
HOUR_BUDGET = 4700.0


def call_weight(n_vars: int, n_days: float) -> float:
    return max(1.0, n_vars / 10.0) * max(1.0, n_days / 14.0)


def _throttle(weight: float) -> None:
    while True:
        with _LOCK:
            now = time.time()
            while _HISTORY and now - _HISTORY[0][0] > 3600:
                _HISTORY.popleft()
            last_min = sum(w for t, w in _HISTORY if now - t < 60)
            last_hour = sum(w for _, w in _HISTORY)
            if last_min + weight <= MINUTE_BUDGET and last_hour + weight <= HOUR_BUDGET:
                _HISTORY.append((now, weight))
                return
        time.sleep(2.0)


def _get(url: str, params: dict, weight: float, cache: Path | None) -> dict:
    if cache is not None and cache.exists():
        try:
            return json.loads(cache.read_text())
        except json.JSONDecodeError:   # truncated by an interrupted run: fetch again
            cache.unlink()
    _throttle(weight)
    for attempt in range(8):
        try:
            r = requests.get(url, params=params, timeout=120)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 429:
            reason = r.text.lower()
            wait = 3700 - (time.time() % 3600) if "hour" in reason else (65 if "minute" in reason else 30)
            print(f"[open-meteo] 429 ({r.text[:80]}), sleeping {wait:.0f}s", flush=True)
            time.sleep(wait)
            continue
        if r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        data = r.json()
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(".tmp")
            tmp.write_text(json.dumps(data))
            tmp.replace(cache)            # atomic: a killed run never leaves half a file
        return data
    raise RuntimeError(f"Open-Meteo request failed after retries: {url} {params}")


def _key(*parts) -> str:
    return hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:12]


def cams_archive(lat: float, lon: float, start: str, end: str, cache_dir: Path = RAW_OPENMETEO) -> dict:
    """Hourly CAMS global composition for a past window (local time, timezone=auto)."""
    n_days = (time.mktime(time.strptime(end, "%Y-%m-%d")) - time.mktime(time.strptime(start, "%Y-%m-%d"))) / 86400 + 1
    params = {"latitude": lat, "longitude": lon, "hourly": ",".join(CAMS_VARS), "domains": "cams_global",
              "start_date": start, "end_date": end, "timezone": "auto"}
    cache = cache_dir / f"cams_{lat:.4f}_{lon:.4f}_{start}_{end}_{_key(*CAMS_VARS)}.json"
    return _get(AQ_ARCHIVE_URL, params, call_weight(len(CAMS_VARS), n_days), cache)


def weather_archive(lat: float, lon: float, start: str, end: str, cache_dir: Path = RAW_OPENMETEO) -> dict:
    """Hourly ERA5 reanalysis weather for a past window (local time, timezone=auto)."""
    n_days = (time.mktime(time.strptime(end, "%Y-%m-%d")) - time.mktime(time.strptime(start, "%Y-%m-%d"))) / 86400 + 1
    params = {"latitude": lat, "longitude": lon, "hourly": ",".join(WEATHER_VARS),
              "start_date": start, "end_date": end, "timezone": "auto"}
    cache = cache_dir / f"era5_{lat:.4f}_{lon:.4f}_{start}_{end}_{_key(*WEATHER_VARS)}.json"
    return _get(WEATHER_ARCHIVE_URL, params, call_weight(len(WEATHER_VARS), n_days), cache)


def cams_forecast(lat: float, lon: float, past_days: int = 1, forecast_days: int = 5) -> dict:
    """Live hourly CAMS forecast (not cached)."""
    params = {"latitude": lat, "longitude": lon, "hourly": ",".join(CAMS_VARS), "domains": "cams_global",
              "past_days": past_days, "forecast_days": forecast_days, "timezone": "auto"}
    return _get(AQ_ARCHIVE_URL, params, call_weight(len(CAMS_VARS), past_days + forecast_days), None)


def weather_forecast(lat: float, lon: float, past_days: int = 1, forecast_days: int = 5) -> dict:
    """Live hourly weather forecast (not cached)."""
    params = {"latitude": lat, "longitude": lon, "hourly": ",".join(WEATHER_VARS),
              "past_days": past_days, "forecast_days": forecast_days, "timezone": "auto"}
    return _get(WEATHER_FORECAST_URL, params, call_weight(len(WEATHER_VARS), past_days + forecast_days), None)


def era5_blh_patch(lat: float, lon: float, start: str = "2024-03-01", end: str = "2024-06-30",
                   cache_dir: Path = RAW_OPENMETEO) -> dict:
    """ERA5 boundary-layer height for the months where Open-Meteo's best-match archive has none."""
    n_days = (time.mktime(time.strptime(end, "%Y-%m-%d")) - time.mktime(time.strptime(start, "%Y-%m-%d"))) / 86400 + 1
    params = {"latitude": lat, "longitude": lon, "hourly": "boundary_layer_height", "models": "era5",
              "start_date": start, "end_date": end, "timezone": "auto"}
    cache = cache_dir / f"era5blh_{lat:.4f}_{lon:.4f}_{start}_{end}.json"
    return _get(WEATHER_ARCHIVE_URL, params, call_weight(1, n_days), cache)
