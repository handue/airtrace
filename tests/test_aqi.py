"""US EPA 2024 PM2.5 AQI: breakpoints, truncation and edge values."""
import math

import pytest

from airtrace.aqi import HEALTH_MESSAGES, category, category_index, pm25_to_aqi
from airtrace.config import AQI_CATEGORIES


@pytest.mark.parametrize("conc, aqi", [
    (0.0, 0), (9.0, 50), (9.05, 50),          # 9.05 truncates to 9.0 (no gap between bands)
    (9.1, 51), (12.0, 56), (35.4, 100), (35.5, 101), (55.4, 150), (55.45, 150),
    (55.5, 151), (125.4, 200), (125.5, 201), (225.4, 300), (225.5, 301),
    (500.4, 500), (600.0, 500),
    (-3.0, 0),                                  # negative clamps to 0
])
def test_pm25_to_aqi(conc, aqi):
    assert pm25_to_aqi(conc) == aqi


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf")])
def test_pm25_to_aqi_missing(bad):
    assert pm25_to_aqi(bad) is None


def test_aqi_monotonic():
    vals = [pm25_to_aqi(c / 10) for c in range(0, 6000)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))


@pytest.mark.parametrize("conc, name", [
    (0.0, "Good"), (9.0, "Good"), (9.1, "Moderate"), (35.4, "Moderate"),
    (35.5, "Unhealthy for Sensitive Groups"), (55.5, "Unhealthy"), (125.5, "Very Unhealthy"),
    (225.5, "Hazardous"), (600.0, "Hazardous"), (-1.0, "Good"),
])
def test_category(conc, name):
    c = category(conc)
    assert c["category"] == name
    assert c["color"].startswith("#") and len(c["color"]) == 7
    assert c["message"] == HEALTH_MESSAGES[name]
    assert category_index(conc) == AQI_CATEGORIES.index(name)


def test_category_nan():
    c = category(float("nan"))
    assert c["category"] is None and c["message"] is None
    assert category(None)["category"] is None


def test_category_consistent_with_aqi():
    for c10 in range(0, 6000, 7):
        conc = c10 / 10
        aqi = pm25_to_aqi(conc)
        idx = category_index(conc)
        bounds = [(0, 50), (51, 100), (101, 150), (151, 200), (201, 300), (301, 500)][idx]
        assert bounds[0] <= aqi <= bounds[1], (conc, aqi, idx)
    assert not math.isnan(pm25_to_aqi(1e6))
