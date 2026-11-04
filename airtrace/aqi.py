"""US EPA Air Quality Index for 24-hour PM2.5 (2024 breakpoints)."""
from __future__ import annotations

import math

from .config import PM25_BREAKPOINTS

HEALTH_MESSAGES = {
    "Good": "Air quality is satisfactory. Enjoy outdoor activities.",
    "Moderate": "Acceptable for most people. Unusually sensitive people should consider limiting prolonged outdoor exertion.",
    "Unhealthy for Sensitive Groups": "Children, older adults and people with heart or lung disease should reduce prolonged or heavy outdoor exertion.",
    "Unhealthy": "Everyone should reduce prolonged outdoor exertion; sensitive groups should avoid it. Consider a well-fitted mask outdoors.",
    "Very Unhealthy": "Everyone should avoid prolonged outdoor exertion. Keep windows closed and run an air purifier if available.",
    "Hazardous": "Health alert: everyone should stay indoors with filtered air and avoid all outdoor activity.",
}


def pm25_to_aqi(conc: float | None) -> int | None:
    if conc is None or not math.isfinite(conc):
        return None
    c = math.floor(max(conc, 0.0) * 10) / 10  # EPA truncates to 0.1 ug/m3
    for c_lo, c_hi, i_lo, i_hi, _, _ in PM25_BREAKPOINTS:
        if c <= c_hi:
            return round((i_hi - i_lo) / (c_hi - c_lo) * (max(c, c_lo) - c_lo) + i_lo)
    return 500


def category(conc: float | None) -> dict:
    """Category name, color and health message for a 24-h PM2.5 concentration."""
    if conc is None or not math.isfinite(conc):
        return {"category": None, "color": "#9ca3af", "message": None}
    c = math.floor(max(conc, 0.0) * 10) / 10
    for c_lo, c_hi, _, _, name, color in PM25_BREAKPOINTS:
        if c <= c_hi:
            return {"category": name, "color": color, "message": HEALTH_MESSAGES[name]}
    name, color = PM25_BREAKPOINTS[-1][4], PM25_BREAKPOINTS[-1][5]
    return {"category": name, "color": color, "message": HEALTH_MESSAGES[name]}


def category_index(conc: float) -> int:
    """0..5 index of the AQI category (used for category-accuracy metrics)."""
    c = math.floor(max(conc, 0.0) * 10) / 10
    for i, b in enumerate(PM25_BREAKPOINTS):
        if c <= b[1]:
            return i
    return len(PM25_BREAKPOINTS) - 1
