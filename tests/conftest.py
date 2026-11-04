"""Shared pytest fixtures: synthetic Open-Meteo-like hourly frames (no network access)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from airtrace.config import CAMS_VARS, WEATHER_VARS  # noqa: E402


def make_hourly(start: str = "2024-07-01", days: int = 3, seed: int = 0,
                **overrides: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Plausible hourly CAMS + weather frames (naive local time index).

    ``overrides`` set a weather/CAMS variable to a constant for every hour,
    e.g. ``wind_direction_10m=0``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=24 * days, freq="h", name="time")
    n = len(idx)
    base = {
        "pm2_5": 20 + 5 * rng.random(n), "pm10": 40 + 10 * rng.random(n),
        "dust": 2 + rng.random(n), "sea_salt_aerosol": rng.random(n),
        "aerosol_optical_depth": 0.3 + 0.1 * rng.random(n),
        "carbon_monoxide": 300 + 50 * rng.random(n), "nitrogen_dioxide": 20 + 5 * rng.random(n),
        "nitrogen_monoxide": 5 + rng.random(n), "sulphur_dioxide": 8 + rng.random(n),
        "ozone": 60 + 10 * rng.random(n), "formaldehyde": 2 + rng.random(n),
    }
    wx = {
        "temperature_2m": 25 + 5 * np.sin(np.arange(n) / 24 * 2 * np.pi),
        "relative_humidity_2m": 60 + 10 * rng.random(n), "precipitation": np.zeros(n),
        "surface_pressure": 1005 + rng.random(n), "wind_speed_10m": 10 + 2 * rng.random(n),
        "wind_direction_10m": 360 * rng.random(n), "boundary_layer_height": 800 + 100 * rng.random(n),
        "shortwave_radiation": np.clip(800 * np.sin((np.arange(n) % 24 - 6) / 12 * np.pi), 0, None),
    }
    for k, v in overrides.items():
        (base if k in CAMS_VARS else wx)[k] = np.full(n, float(v))
    cams = pd.DataFrame({v: base[v] for v in CAMS_VARS}, index=idx)
    weather = pd.DataFrame({v: wx[v] for v in WEATHER_VARS}, index=idx)
    return cams, weather


@pytest.fixture
def hourly():
    return make_hourly()
