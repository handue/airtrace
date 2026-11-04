"""Smoke tests for the trained artifacts (skipped until ML/SOURCES have produced them)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from airtrace.config import MODELS
from airtrace.features import FEATURES, TRACER_COLUMNS, daily_features

from conftest import make_hourly

FACTOR_KEYS = {"traffic", "industry", "burning", "dust", "secondary", "marine", "background"}


def _feature_rows(n_days: int = 4) -> pd.DataFrame:
    cams, wx = make_hourly(days=n_days, seed=1)
    return daily_features(cams, wx, lat=28.6, lon=77.2, elevation=216.0)


@pytest.fixture(scope="module")
def models():
    if not (MODELS / "pm25_lgbm.txt").exists():
        pytest.skip("models/pm25_lgbm.txt not trained yet")
    model = pytest.importorskip("airtrace.model")
    return model, model.load_models()


def test_model_meta_matches_features():
    p = MODELS / "model_meta.json"
    if not p.exists():
        pytest.skip("models/model_meta.json missing")
    meta = json.loads(p.read_text())
    for k in ("version", "trained_at", "features", "target", "n_rows", "n_sites", "train_period"):
        assert k in meta, k
    assert meta["features"] == FEATURES, "model inputs drifted from features.FEATURES"
    assert meta["target"] == "log1p_pm25"


def test_predict(models):
    model, m = models
    X = _feature_rows()[FEATURES]
    pred = model.predict(m, X)
    assert list(pred.columns[:3]) == ["pm25", "pm25_low", "pm25_high"] or \
        {"pm25", "pm25_low", "pm25_high"} <= set(pred.columns)
    assert len(pred) == len(X)
    vals = pred[["pm25", "pm25_low", "pm25_high"]].to_numpy()
    assert np.isfinite(vals).all() and (vals >= 0).all()
    assert (pred["pm25_low"] <= pred["pm25_high"] + 1e-9).all()


def test_contributions_sum_to_prediction(models):
    model, m = models
    X = _feature_rows()[FEATURES]
    contrib = model.contributions(m, X)
    assert set(FEATURES) | {"bias"} == set(contrib.columns)
    log_pred = contrib.sum(axis=1).to_numpy()
    pm = model.predict(m, X)["pm25"].to_numpy()
    # TreeSHAP is additive in log1p space; allow small slack for any post-hoc calibration
    np.testing.assert_allclose(np.expm1(log_pred), pm, rtol=0.05, atol=0.5)


def test_source_model_shares():
    if not (MODELS / "sources_nmf.json").exists():
        pytest.skip("models/sources_nmf.json not fitted yet")
    sources = pytest.importorskip("airtrace.sources")
    sm = sources.load_source_model()
    daily = _feature_rows(7)[TRACER_COLUMNS]
    out = sm.shares(daily)
    shares = out["shares"]
    assert shares and all(s["key"] in FACTOR_KEYS for s in shares)
    vals = [s["share"] for s in shares]
    assert vals == sorted(vals, reverse=True)
    assert all(0 <= v <= 1 for v in vals)
    total = sum(vals) + (0 if any(s["key"] == "background" for s in shares) else out["background_share"])
    assert total == pytest.approx(1.0, abs=0.01)
