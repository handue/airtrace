"""features.daily_features on synthetic hourly frames."""
import numpy as np
import pandas as pd
import pytest

from airtrace.features import (FEATURE_GROUPS, FEATURE_LABELS, FEATURES, LATE_CAMS_VARS, MIN_HOURS,
                               TRACER_COLUMNS, cams_frame, daily_features, weather_frame)
from airtrace.config import CAMS_VARS, WEATHER_VARS

from conftest import make_hourly

EXPECTED_COLS = list(dict.fromkeys(FEATURES + TRACER_COLUMNS + ["n_hours"]))


def test_shape_and_columns(hourly):
    cams, wx = hourly
    out = daily_features(cams, wx, lat=28.6, lon=77.2, elevation=216.0)
    assert list(out.columns) == EXPECTED_COLS
    assert len(out) == 3
    assert out.index.name == "date"
    assert (out.index == pd.date_range("2024-07-01", periods=3, freq="D")).all()
    assert (out["n_hours"] == 24).all()
    # everything except the first day's lag/delta is finite
    assert out.iloc[1:][FEATURES].notna().all().all()
    assert out.iloc[0][["cams_pm2_5_lag1", "cams_pm2_5_delta"]].isna().all()


def test_daily_means_match(hourly):
    cams, wx = hourly
    out = daily_features(cams, wx, 0.0, 0.0, 0.0)
    day = cams.loc["2024-07-02"]
    assert out.loc["2024-07-02", "cams_pm2_5_mean"] == pytest.approx(day["pm2_5"].mean())
    assert out.loc["2024-07-02", "cams_pm2_5_max"] == pytest.approx(day["pm2_5"].max())
    night = day[(day.index.hour < 6) | (day.index.hour >= 20)]["pm2_5"].mean()
    assert out.loc["2024-07-02", "cams_pm2_5_night"] == pytest.approx(night)


def test_n_hours_filter():
    cams, wx = make_hourly(days=3)
    # day 2: only MIN_HOURS-1 valid PM2.5 hours -> dropped; day 3: exactly MIN_HOURS -> kept
    d2 = cams.index.normalize() == pd.Timestamp("2024-07-02")
    d3 = cams.index.normalize() == pd.Timestamp("2024-07-03")
    col = cams.columns.get_loc("pm2_5")
    cams.iloc[np.flatnonzero(d2)[MIN_HOURS - 1:], col] = np.nan
    cams.iloc[np.flatnonzero(d3)[MIN_HOURS:], col] = np.nan
    out = daily_features(cams, wx, 0.0, 0.0, 0.0)
    assert list(out.index.strftime("%Y-%m-%d")) == ["2024-07-01", "2024-07-03"]
    assert out.loc["2024-07-03", "n_hours"] == MIN_HOURS


def test_lag1_uses_previous_calendar_day(hourly):
    cams, wx = hourly
    out = daily_features(cams, wx, 0.0, 0.0, 0.0)
    assert out.loc["2024-07-02", "cams_pm2_5_lag1"] == pytest.approx(out.loc["2024-07-01", "cams_pm2_5_mean"])
    assert out.loc["2024-07-03", "cams_pm2_5_delta"] == pytest.approx(
        out.loc["2024-07-03", "cams_pm2_5_mean"] - out.loc["2024-07-02", "cams_pm2_5_mean"])


def test_lag1_missing_when_previous_day_absent():
    cams, wx = make_hourly(days=4)
    keep = cams.index.normalize() != pd.Timestamp("2024-07-02")
    out = daily_features(cams[keep], wx[keep], 0.0, 0.0, 0.0)
    assert pd.Timestamp("2024-07-02") not in out.index
    # 07-03 must NOT take 07-01 as its "previous day"
    assert np.isnan(out.loc["2024-07-03", "cams_pm2_5_lag1"])
    assert out.loc["2024-07-04", "cams_pm2_5_lag1"] == pytest.approx(out.loc["2024-07-03", "cams_pm2_5_mean"])


@pytest.mark.parametrize("direction, u, v", [
    (0.0, 0.0, -10.0),     # wind FROM north blows southward -> v negative
    (180.0, 0.0, 10.0),    # from south -> v positive
    (90.0, -10.0, 0.0),    # from east blows westward -> u negative
    (270.0, 10.0, 0.0),    # from west -> u positive
])
def test_wind_components_sign_convention(direction, u, v):
    cams, wx = make_hourly(days=1, wind_speed_10m=10.0, wind_direction_10m=direction)
    out = daily_features(cams, wx, 0.0, 0.0, 0.0)
    assert out["u_mean"].iloc[0] == pytest.approx(u, abs=1e-9)
    assert out["v_mean"].iloc[0] == pytest.approx(v, abs=1e-9)


def test_ventilation_and_stagnation():
    cams, wx = make_hourly(days=1, wind_speed_10m=36.0, boundary_layer_height=1000.0)
    out = daily_features(cams, wx, 0.0, 0.0, 0.0)
    assert out["ventilation"].iloc[0] == pytest.approx(1000.0 * 36.0 / 3.6)  # m * m/s
    assert out["stagnant_hours"].iloc[0] == 0
    cams, wx = make_hourly(days=1, wind_speed_10m=5.0, boundary_layer_height=300.0)
    out = daily_features(cams, wx, 0.0, 0.0, 0.0)
    assert out["stagnant_hours"].iloc[0] == 24


def test_geography_and_calendar():
    cams, wx = make_hourly(days=1)
    out = daily_features(cams, wx, lat=-33.9, lon=90.0, elevation=float("nan"))
    r = out.iloc[0]
    assert r["lat"] == -33.9 and r["abs_lat"] == 33.9
    assert r["lon_sin"] == pytest.approx(1.0) and r["lon_cos"] == pytest.approx(0.0, abs=1e-12)
    assert r["elevation"] == 0.0
    assert r["dow"] == 0  # 2024-07-01 is a Monday
    assert r["doy_sin"] ** 2 + r["doy_cos"] ** 2 == pytest.approx(1.0)


def test_ratios():
    cams, wx = make_hourly(days=1, pm2_5=20.0, pm10=40.0, dust=10.0)
    r = daily_features(cams, wx, 0.0, 0.0, 0.0).iloc[0]
    assert r["pm25_pm10_ratio"] == pytest.approx(0.5, rel=1e-4)
    assert r["dust_frac"] == pytest.approx(0.25, rel=1e-4)


def test_feature_lists_consistent():
    assert len(FEATURES) == len(set(FEATURES))
    assert sorted(sum(FEATURE_GROUPS.values(), [])) == sorted(FEATURES)
    for late in LATE_CAMS_VARS:
        assert f"cams_{late}_mean" not in FEATURES
        assert f"cams_{late}_mean" in TRACER_COLUMNS
    for col in FEATURES + TRACER_COLUMNS:
        assert col in FEATURE_LABELS, col


def test_frames_from_openmeteo_json():
    resp = {"hourly": {"time": ["2024-07-01T00:00", "2024-07-01T01:00"],
                       "pm2_5": [1.5, None], "pm10": [3, 4]}}
    df = cams_frame(resp)
    assert list(df.columns) == CAMS_VARS
    assert df.index.dtype.kind == "M" and np.isnan(df["pm2_5"].iloc[1])
    assert df["dust"].isna().all()  # missing variable -> all NaN
    assert list(weather_frame({"hourly": {"time": ["2024-07-01T00:00"]}}).columns) == WEATHER_VARS
