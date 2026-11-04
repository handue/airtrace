"""Shared constants and paths for the AirTrace pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
RAW_AIRNOW = RAW / "airnow"
RAW_OPENMETEO = RAW / "openmeteo"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
SITE = ROOT / "docs"            # GitHub Pages root
SITE_DATA = SITE / "data"
CONFIG = ROOT / "config"

# Training window: a full annual cycle while the US Department of State embassy
# network was still fully reporting (it collapsed from 59 to ~5 sites after March 2025).
TRAIN_START = "2024-03-01"
TRAIN_END = "2025-02-28"

# Copernicus CAMS global atmospheric composition (via Open-Meteo), hourly.
CAMS_VARS = [
    "pm2_5", "pm10", "dust", "sea_salt_aerosol", "aerosol_optical_depth",
    "carbon_monoxide", "nitrogen_dioxide", "nitrogen_monoxide",
    "sulphur_dioxide", "ozone", "formaldehyde",
]

# Meteorology (ERA5 reanalysis for training, Open-Meteo forecast for live), hourly.
WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "precipitation", "surface_pressure",
    "wind_speed_10m", "wind_direction_10m", "boundary_layer_height", "shortwave_radiation",
]

AQ_ARCHIVE_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

AIRNOW_BASE = "https://files.airnowtech.org/airnow"
FIRMS_24H_URL = (
    "https://firms.modaps.eosdis.nasa.gov/data/active_fire/"
    "suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_Global_24h.csv"
)

# US EPA AQI breakpoints for 24-h PM2.5 (2024 revision), ug/m3.
PM25_BREAKPOINTS = [
    (0.0, 9.0, 0, 50, "Good", "#00e400"),
    (9.1, 35.4, 51, 100, "Moderate", "#ffff00"),
    (35.5, 55.4, 101, 150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (55.5, 125.4, 151, 200, "Unhealthy", "#ff0000"),
    (125.5, 225.4, 201, 300, "Very Unhealthy", "#8f3f97"),
    (225.5, 500.4, 301, 500, "Hazardous", "#7e0023"),
]
AQI_CATEGORIES = [b[4] for b in PM25_BREAKPOINTS]
UNHEALTHY_PM25 = 35.5   # AQI > 100: unhealthy for sensitive groups or worse
WHO_24H_PM25 = 15.0     # WHO 2021 24-hour guideline
