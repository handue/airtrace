"""Feature engineering shared by training (CAMS archive + ERA5) and live forecasting
(CAMS forecast + NWP forecast). Every model input is built here, so train and serve
cannot drift apart.

Inputs are Open-Meteo hourly responses requested with ``timezone=auto``: timestamps are
naive local time, and a "day" is a local calendar day (same as AirNow 24-h averages).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CAMS_VARS, WEATHER_VARS

MIN_HOURS = 18  # a day needs >= 18 valid hours to be usable

# CAMS species that only exist in the Open-Meteo archive from 2024-09-26 are kept out of the
# forecast model (their missingness would be confounded with season) but are still
# aggregated below as tracers for source attribution.
LATE_CAMS_VARS = ["sea_salt_aerosol", "nitrogen_monoxide", "formaldehyde"]
TRACER_COLUMNS = [f"cams_{v}_mean" for v in CAMS_VARS] + ["no_no2_ratio"]
# Mixing-height diagnostics are still computed, but are not model inputs: the Open-Meteo archive
# has no boundary-layer height before July 2024 and a leave-location-out ablation showed equal or
# better skill without them (MAE 11.29 vs 11.34 ug/m3), so dropping them removes train/serve skew.
BLH_COLUMNS = ["blh_mean", "blh_min", "ventilation", "stagnant_hours"]

FEATURES = [
    # CAMS composition, daily means
    *[f"cams_{v}_mean" for v in CAMS_VARS if v not in LATE_CAMS_VARS],
    "cams_pm2_5_max", "cams_pm2_5_night", "cams_pm2_5_lag1", "cams_pm2_5_delta",
    "pm25_pm10_ratio", "dust_frac", "so2_co_ratio",
    # Meteorology
    "t2m_mean", "t2m_range", "rh_mean", "precip_sum", "sp_mean",
    "ws_mean", "ws_min", "u_mean", "v_mean", "swr_mean",
    # Calendar + geography
    "doy_sin", "doy_cos", "dow", "lat", "abs_lat", "lon_sin", "lon_cos", "elevation",
]

# Human-readable groups used to explain predictions (SHAP contributions are summed per group).
FEATURE_GROUPS = {
    "Global model chemistry (CAMS PM2.5)": [
        "cams_pm2_5_mean", "cams_pm2_5_max", "cams_pm2_5_night", "cams_pm2_5_lag1", "cams_pm2_5_delta",
        "cams_pm10_mean", "pm25_pm10_ratio", "cams_aerosol_optical_depth_mean"],
    "Combustion tracers (NO2, CO, SO2)": [
        "cams_carbon_monoxide_mean", "cams_nitrogen_dioxide_mean", "cams_sulphur_dioxide_mean", "so2_co_ratio"],
    "Desert dust": ["cams_dust_mean", "dust_frac"],
    "Photochemistry (ozone)": ["cams_ozone_mean", "swr_mean"],
    "Wind & air stagnation": ["ws_mean", "ws_min", "u_mean", "v_mean"],
    "Humidity, rain & temperature": ["t2m_mean", "t2m_range", "rh_mean", "precip_sum", "sp_mean"],
    "Season & weekday": ["doy_sin", "doy_cos", "dow"],
    "Location & terrain": ["lat", "abs_lat", "lon_sin", "lon_cos", "elevation"],
}
assert sorted(sum(FEATURE_GROUPS.values(), [])) == sorted(FEATURES)

FEATURE_LABELS = {
    "cams_pm2_5_mean": "CAMS PM2.5 (daily mean)", "cams_pm2_5_max": "CAMS PM2.5 (daily max)",
    "cams_pm2_5_night": "CAMS PM2.5 (night)", "cams_pm2_5_lag1": "CAMS PM2.5 (previous day)",
    "cams_pm2_5_delta": "CAMS PM2.5 day-over-day change", "cams_pm10_mean": "CAMS PM10",
    "cams_dust_mean": "Desert dust", "cams_sea_salt_aerosol_mean": "Sea salt",
    "cams_aerosol_optical_depth_mean": "Aerosol optical depth", "cams_carbon_monoxide_mean": "Carbon monoxide",
    "cams_nitrogen_dioxide_mean": "Nitrogen dioxide", "cams_nitrogen_monoxide_mean": "Nitric oxide",
    "cams_sulphur_dioxide_mean": "Sulphur dioxide", "cams_ozone_mean": "Ozone",
    "cams_formaldehyde_mean": "Formaldehyde", "pm25_pm10_ratio": "PM2.5/PM10 ratio",
    "dust_frac": "Dust share of PM10", "no_no2_ratio": "NO/NO2 ratio (fresh combustion)",
    "so2_co_ratio": "SO2/CO ratio (coal & industry)", "t2m_mean": "Temperature", "t2m_range": "Daily temperature range",
    "rh_mean": "Relative humidity", "precip_sum": "Precipitation", "sp_mean": "Surface pressure",
    "ws_mean": "Wind speed", "ws_min": "Minimum wind speed", "u_mean": "East-west wind", "v_mean": "North-south wind",
    "blh_mean": "Mixing height", "blh_min": "Minimum mixing height", "swr_mean": "Sunlight",
    "ventilation": "Ventilation index", "stagnant_hours": "Stagnant hours", "doy_sin": "Season (sin)",
    "doy_cos": "Season (cos)", "dow": "Day of week", "lat": "Latitude", "abs_lat": "Distance from equator",
    "lon_sin": "Longitude (sin)", "lon_cos": "Longitude (cos)", "elevation": "Elevation",
}


def hourly_frame(resp: dict, variables: list[str]) -> pd.DataFrame:
    """Open-Meteo hourly JSON -> DataFrame indexed by naive local timestamp."""
    h = resp["hourly"]
    df = pd.DataFrame({v: pd.to_numeric(pd.Series(h.get(v, [None] * len(h["time"]))), errors="coerce")
                       for v in variables})
    df.index = pd.to_datetime(pd.Series(h["time"]))
    df.index.name = "time"
    return df


def cams_frame(resp: dict) -> pd.DataFrame:
    return hourly_frame(resp, CAMS_VARS)


def weather_frame(resp: dict) -> pd.DataFrame:
    return hourly_frame(resp, WEATHER_VARS)


def daily_features(cams: pd.DataFrame, wx: pd.DataFrame, lat: float, lon: float,
                   elevation: float) -> pd.DataFrame:
    """Aggregate hourly CAMS + weather to one feature row per local day.

    Returns a DataFrame indexed by ``date`` (datetime64, local midnight) with columns
    ``FEATURES`` + ``TRACER_COLUMNS`` (daily-mean CAMS species for source attribution)
    + ``BLH_COLUMNS`` (diagnostics only) + ``n_hours`` (valid CAMS PM2.5 hours). Days with fewer than ``MIN_HOURS`` valid
    hours are dropped.
    """
    df = cams.join(wx, how="inner")
    date = df.index.normalize()
    hour = df.index.hour
    g = df.groupby(date)

    out = pd.DataFrame(index=g.size().index)
    out.index.name = "date"
    for v in CAMS_VARS:
        out[f"cams_{v}_mean"] = g[v].mean()
    out["cams_pm2_5_max"] = g["pm2_5"].max()
    night = df[(hour < 6) | (hour >= 20)]
    out["cams_pm2_5_night"] = night.groupby(night.index.normalize())["pm2_5"].mean()
    out["n_hours"] = g["pm2_5"].count()

    eps = 1e-3
    out["pm25_pm10_ratio"] = out["cams_pm2_5_mean"] / (out["cams_pm10_mean"] + eps)
    out["dust_frac"] = out["cams_dust_mean"] / (out["cams_pm10_mean"] + eps)
    out["no_no2_ratio"] = out["cams_nitrogen_monoxide_mean"] / (out["cams_nitrogen_dioxide_mean"] + eps)
    out["so2_co_ratio"] = 100 * out["cams_sulphur_dioxide_mean"] / (out["cams_carbon_monoxide_mean"] + eps)

    ws = df["wind_speed_10m"]
    rad = np.deg2rad(df["wind_direction_10m"])
    df = df.assign(u=-ws * np.sin(rad), v=-ws * np.cos(rad),
                   vent=df["boundary_layer_height"] * ws / 3.6,
                   stagnant=((ws < 7.2) & (df["boundary_layer_height"] < 500)).astype(float))
    g = df.groupby(date)
    out["t2m_mean"] = g["temperature_2m"].mean()
    out["t2m_range"] = g["temperature_2m"].max() - g["temperature_2m"].min()
    out["rh_mean"] = g["relative_humidity_2m"].mean()
    out["precip_sum"] = g["precipitation"].sum(min_count=1)
    out["sp_mean"] = g["surface_pressure"].mean()
    out["ws_mean"] = g["wind_speed_10m"].mean()
    out["ws_min"] = g["wind_speed_10m"].min()
    out["u_mean"] = g["u"].mean()
    out["v_mean"] = g["v"].mean()
    out["blh_mean"] = g["boundary_layer_height"].mean()
    out["blh_min"] = g["boundary_layer_height"].min()
    out["swr_mean"] = g["shortwave_radiation"].mean()
    out["ventilation"] = g["vent"].mean()
    out["stagnant_hours"] = g["stagnant"].sum()

    # previous-day CAMS PM2.5 (calendar-aware: only if yesterday exists)
    prev = out["cams_pm2_5_mean"].copy()
    prev.index = prev.index + pd.Timedelta(days=1)
    out["cams_pm2_5_lag1"] = prev.reindex(out.index)
    out["cams_pm2_5_delta"] = out["cams_pm2_5_mean"] - out["cams_pm2_5_lag1"]

    doy = out.index.dayofyear.values
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["dow"] = out.index.dayofweek.values
    out["lat"] = lat
    out["abs_lat"] = abs(lat)
    out["lon_sin"] = np.sin(np.deg2rad(lon))
    out["lon_cos"] = np.cos(np.deg2rad(lon))
    out["elevation"] = elevation if elevation is not None and np.isfinite(elevation) else 0.0

    out = out[out["n_hours"] >= MIN_HOURS]
    cols = list(dict.fromkeys(FEATURES + TRACER_COLUMNS + BLH_COLUMNS + ["n_hours"]))
    return out[cols]
