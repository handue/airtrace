"""Sanity checks on shared configuration and committed config CSVs."""
import re

import pandas as pd
import pytest

from airtrace import config as C

HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def test_breakpoints_contiguous():
    bps = C.PM25_BREAKPOINTS
    assert len(bps) == 6
    assert bps[0][0] == 0.0 and bps[0][2] == 0
    for (lo1, hi1, ilo1, ihi1, *_), (lo2, hi2, ilo2, ihi2, *_) in zip(bps, bps[1:]):
        assert lo2 == pytest.approx(hi1 + 0.1)
        assert ilo2 == ihi1 + 1
        assert lo2 < hi2 and ilo2 < ihi2
    assert all(HEX.match(b[5]) for b in bps)
    assert C.AQI_CATEGORIES == [b[4] for b in bps]


def test_thresholds():
    assert C.UNHEALTHY_PM25 == C.PM25_BREAKPOINTS[2][0]
    assert C.WHO_24H_PM25 == 15.0


def test_train_window():
    s, e = pd.Timestamp(C.TRAIN_START), pd.Timestamp(C.TRAIN_END)
    assert s < e and (e - s).days >= 364


def test_variable_lists():
    assert len(C.CAMS_VARS) == len(set(C.CAMS_VARS)) and "pm2_5" in C.CAMS_VARS
    assert len(C.WEATHER_VARS) == len(set(C.WEATHER_VARS))
    assert "boundary_layer_height" in C.WEATHER_VARS


def test_paths():
    assert (C.ROOT / "airtrace" / "config.py").exists()
    assert C.SITE_DATA == C.ROOT / "docs" / "data"
    assert C.RAW_OPENMETEO.is_relative_to(C.RAW)


def test_training_sites_csv():
    df = pd.read_csv(C.CONFIG / "training_sites.csv")
    need = {"site_id", "city", "country_name", "country", "region", "network", "lat", "lon",
            "elevation", "gmt_offset", "n_days", "mean_pm25"}
    assert need <= set(df.columns)
    assert df["site_id"].is_unique and len(df) > 0
    assert df["lat"].between(-90, 90).all() and df["lon"].between(-180, 180).all()
    assert (df["mean_pm25"] > 0).all()


def test_showcase_cities_csv():
    df = pd.read_csv(C.CONFIG / "showcase_cities.csv")
    assert {"id", "name", "country", "region", "lat", "lon"} <= set(df.columns)
    assert df["id"].is_unique
    assert df["id"].str.fullmatch(r"[a-z0-9-]+").all(), "ids must be URL-safe slugs"
    assert df["lat"].between(-90, 90).all() and df["lon"].between(-180, 180).all()
