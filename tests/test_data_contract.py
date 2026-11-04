"""Website data contract (SPEC.md "Website data contract").

Validates the JSON files the website reads, both the generated ones in docs/data/ and the
hand-made mocks in docs/mock/. Each test skips cleanly when its file does not exist yet.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from airtrace.config import AQI_CATEGORIES, ROOT

DATA_DIRS = [ROOT / "docs" / "data", ROOT / "docs" / "mock"]
DIR_IDS = ["data", "mock"]

FACTORS = {
    "traffic": ("Traffic & urban combustion", "#3b82f6"),
    "industry": ("Industry, coal & power", "#64748b"),
    "burning": ("Fires & biomass burning", "#f97316"),
    "dust": ("Desert dust", "#c8a165"),
    "secondary": ("Photochemical & secondary", "#a855f7"),
    "marine": ("Sea salt (marine)", "#06b6d4"),
    "background": ("Regional background & other", "#cbd5e1"),
}
MODEL_KEYS = {"cams_raw", "linear", "ml_no_cams", "airtrace"}
MONITOR_STATUS = {"active", "offline", "none"}
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MONTH = re.compile(r"^\d{4}-\d{2}$")
LOCAL_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$")
HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


# ---------------------------------------------------------------- helpers
def load(path: Path):
    if not path.exists():
        pytest.skip(f"{path.relative_to(ROOT)} not present")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def num(x, nullable: bool = False) -> bool:
    if x is None:
        return nullable
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def is_int(x, nullable: bool = False) -> bool:
    if x is None:
        return nullable
    return isinstance(x, int) and not isinstance(x, bool)


def text(x, nullable: bool = False) -> bool:
    return (x is None and nullable) or (isinstance(x, str) and x != "")


def has_keys(obj: dict, keys, where: str) -> None:
    assert isinstance(obj, dict), f"{where}: expected object, got {type(obj).__name__}"
    missing = [k for k in keys if k not in obj]
    assert not missing, f"{where}: missing keys {missing}"


def check_category(cat, color, where: str, nullable: bool = True) -> None:
    assert (cat is None and nullable) or cat in AQI_CATEGORIES, f"{where}: bad category {cat!r}"
    assert isinstance(color, str) and HEX.match(color), f"{where}: bad color {color!r}"


def check_shares(shares, background_share, where: str) -> None:
    assert isinstance(shares, list) and shares, f"{where}: shares must be a non-empty list"
    vals = []
    for i, s in enumerate(shares):
        has_keys(s, ["key", "label", "share", "color", "description"], f"{where}[{i}]")
        assert s["key"] in FACTORS, f"{where}[{i}]: unknown factor key {s['key']!r}"
        assert (s["label"], s["color"]) == FACTORS[s["key"]], f"{where}[{i}]: label/color differ from SPEC"
        assert num(s["share"]) and 0 <= s["share"] <= 1, f"{where}[{i}]: share {s['share']!r}"
        vals.append(s["share"])
    assert vals == sorted(vals, reverse=True), f"{where}: shares not sorted descending"
    assert num(background_share) and 0 <= background_share <= 1
    total = sum(vals)
    if not any(s["key"] == "background" for s in shares):
        total += background_share
    assert total == pytest.approx(1.0, abs=0.01), f"{where}: shares sum to {total:.3f}"


def city_files(d: Path) -> list[Path]:
    files = sorted((d / "city").glob("*.json"))
    if not files:
        pytest.skip(f"no city files in {d.relative_to(ROOT)}/city")
    return files


# ---------------------------------------------------------------- meta.json
@pytest.mark.parametrize("d", DATA_DIRS, ids=DIR_IDS)
def test_meta(d):
    m = load(d / "meta.json")
    has_keys(m, ["generated_at", "model_version", "n_cities", "forecast_days", "data_sources"], "meta")
    assert text(m["generated_at"]) and text(m["model_version"])
    assert is_int(m["n_cities"]) and m["n_cities"] > 0
    assert is_int(m["forecast_days"]) and 1 <= m["forecast_days"] <= 5
    assert isinstance(m["data_sources"], list) and all(text(s) for s in m["data_sources"])


# ---------------------------------------------------------------- cities.json
CITY_LIST_KEYS = ["id", "name", "country", "region", "lat", "lon", "monitor_status", "today_pm25",
                  "today_aqi", "today_category", "today_color", "peak_aqi_5d", "peak_category_5d",
                  "peak_color_5d", "top_source"]


@pytest.mark.parametrize("d", DATA_DIRS, ids=DIR_IDS)
def test_cities_list(d):
    cities = load(d / "cities.json")
    assert isinstance(cities, list) and cities
    ids = [c.get("id") for c in cities]
    assert len(ids) == len(set(ids)), "duplicate city ids"
    labels = {v[0] for v in FACTORS.values()}
    for c in cities:
        w = f"cities.json[{c.get('id')}]"
        has_keys(c, CITY_LIST_KEYS, w)
        assert text(c["id"]) and re.fullmatch(r"[a-z0-9-]+", c["id"]), f"{w}: id not a slug"
        assert text(c["name"]) and text(c["country"]) and text(c["region"])
        assert num(c["lat"]) and -90 <= c["lat"] <= 90 and num(c["lon"]) and -180 <= c["lon"] <= 180
        assert c["monitor_status"] in MONITOR_STATUS, w
        assert num(c["today_pm25"], nullable=True) and is_int(c["today_aqi"], nullable=True), w
        assert is_int(c["peak_aqi_5d"], nullable=True), w
        check_category(c["today_category"], c["today_color"], w)
        check_category(c["peak_category_5d"], c["peak_color_5d"], w)
        assert c["top_source"] is None or c["top_source"] in labels, f"{w}: top_source {c['top_source']!r}"


@pytest.mark.parametrize("d", DATA_DIRS, ids=DIR_IDS)
def test_meta_matches_cities(d):
    m, cities = load(d / "meta.json"), load(d / "cities.json")
    assert m["n_cities"] == len(cities)


@pytest.mark.parametrize("d", DATA_DIRS, ids=DIR_IDS)
def test_every_listed_city_has_detail_file(d):
    cities = load(d / "cities.json")
    if not (d / "city").is_dir():
        pytest.skip("no city/ directory")
    missing = [c["id"] for c in cities if not (d / "city" / f"{c['id']}.json").exists()]
    assert not missing, f"cities.json entries without city/<id>.json: {missing[:10]}"


# ---------------------------------------------------------------- city/<id>.json
def check_city(c: dict, stem: str) -> None:
    w = f"city/{stem}.json"
    has_keys(c, ["id", "name", "country", "region", "lat", "lon", "timezone", "generated_at", "monitor",
                 "today", "daily", "hourly", "sources", "fires", "recent", "history"], w)
    assert c["id"] == stem, f"{w}: id {c['id']!r} != file name"
    assert num(c["lat"]) and num(c["lon"]) and text(c["timezone"]) and text(c["generated_at"])

    mon = c["monitor"]
    has_keys(mon, ["status", "network", "name", "last_seen", "distance_km"], f"{w}.monitor")
    assert mon["status"] in MONITOR_STATUS
    assert text(mon["network"], True) and text(mon["name"], True)
    assert mon["last_seen"] is None or DATE.match(mon["last_seen"])
    assert num(mon["distance_km"], nullable=True)

    t = c["today"]
    has_keys(t, ["date", "pm25", "aqi", "category", "color", "message"], f"{w}.today")
    assert DATE.match(t["date"]) and num(t["pm25"], True) and is_int(t["aqi"], True)
    check_category(t["category"], t["color"], f"{w}.today")

    daily = c["daily"]
    assert isinstance(daily, list) and 1 <= len(daily) <= 5, f"{w}.daily length {len(daily)}"
    dates = []
    for i, day in enumerate(daily):
        wd = f"{w}.daily[{i}]"
        has_keys(day, ["date", "pm25", "pm25_low", "pm25_high", "cams_pm25", "aqi", "category", "color",
                       "drivers"], wd)
        assert DATE.match(day["date"]); dates.append(day["date"])
        for k in ("pm25", "pm25_low", "pm25_high", "cams_pm25"):
            assert num(day[k], True), f"{wd}.{k}={day[k]!r}"
        if None not in (day["pm25"], day["pm25_low"], day["pm25_high"]):
            assert day["pm25_low"] <= day["pm25_high"], f"{wd}: interval inverted"
            assert day["pm25"] >= 0
        assert is_int(day["aqi"], True)
        check_category(day["category"], day["color"], wd)
        assert isinstance(day["drivers"], list) and len(day["drivers"]) <= 4
        for dr in day["drivers"]:
            has_keys(dr, ["group", "effect_pct"], f"{wd}.drivers")
            assert text(dr["group"]) and num(dr["effect_pct"]) and dr["effect_pct"] > -100
        effects = [abs(dr["effect_pct"]) for dr in day["drivers"]]
        assert effects == sorted(effects, reverse=True), f"{wd}: drivers not sorted by |effect|"
    assert dates == sorted(dates) and len(set(dates)) == len(dates), f"{w}.daily dates not increasing"
    assert t["date"] in dates, f"{w}: today.date not among daily dates"

    h = c["hourly"]
    has_keys(h, ["time", "cams_pm25", "airtrace_pm25", "wind_speed", "wind_dir"], f"{w}.hourly")
    n = len(h["time"])
    assert all(LOCAL_TIME.match(x) for x in h["time"]), f"{w}.hourly.time format"
    for k in ("cams_pm25", "airtrace_pm25", "wind_speed", "wind_dir"):
        assert isinstance(h[k], list) and len(h[k]) == n, f"{w}.hourly.{k} length"
        assert all(num(v, True) for v in h[k]), f"{w}.hourly.{k} non-numeric"

    s = c["sources"]
    if s is not None:
        has_keys(s, ["window", "shares", "background_share"], f"{w}.sources")
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2}", s["window"]), s["window"]
        check_shares(s["shares"], s["background_share"], f"{w}.sources.shares")

    f = c["fires"]
    if f is not None:
        has_keys(f, ["count_300km", "upwind_count", "nearest_km", "frp_sum", "points"], f"{w}.fires")
        assert is_int(f["count_300km"]) and is_int(f["upwind_count"]) and f["upwind_count"] <= f["count_300km"]
        assert num(f["nearest_km"], True) and num(f["frp_sum"])
        assert isinstance(f["points"], list) and len(f["points"]) <= 300
        assert all(isinstance(p, list) and len(p) == 3 and all(num(v) for v in p) for p in f["points"])

    assert isinstance(c["recent"], list) and len(c["recent"]) <= 7
    for r in c["recent"]:
        has_keys(r, ["date", "observed", "cams", "airtrace"], f"{w}.recent")
        assert DATE.match(r["date"])
        assert all(num(r[k], True) for k in ("observed", "cams", "airtrace"))
    if c["recent"]:
        assert mon["status"] == "active", f"{w}: recent obs present but monitor not active"

    hist = c["history"]
    if hist is not None:
        has_keys(hist, ["period", "n_days", "mae_cams", "mae_airtrace", "monthly", "rose"], f"{w}.history")
        assert len(hist["period"]) == 2 and all(DATE.match(x) for x in hist["period"])
        assert is_int(hist["n_days"]) and num(hist["mae_cams"]) and num(hist["mae_airtrace"])
        for m in hist["monthly"]:
            has_keys(m, ["month", "observed", "cams", "airtrace"], f"{w}.history.monthly")
            assert MONTH.match(m["month"])
            assert all(num(m[k], True) for k in ("observed", "cams", "airtrace"))
        rose = hist["rose"]
        has_keys(rose, ["sectors", "mean_pm25", "freq"], f"{w}.history.rose")
        assert len(rose["sectors"]) == 16 and rose["sectors"][0] == "N"
        assert len(rose["mean_pm25"]) == 16 and len(rose["freq"]) == 16
        assert all(num(v, True) for v in rose["mean_pm25"]) and all(num(v) for v in rose["freq"])


@pytest.mark.parametrize("d", DATA_DIRS, ids=DIR_IDS)
def test_city_details(d):
    errors = []
    for p in city_files(d):
        try:
            check_city(json.loads(p.read_text(encoding="utf-8")), p.stem)
        except AssertionError as e:  # collect all failures, report up to 10
            errors.append(f"{p.name}: {e}".splitlines()[0])
    assert not errors, f"{len(errors)} invalid city files:\n" + "\n".join(errors[:10])


# ---------------------------------------------------------------- metrics.json
METRIC_KEYS = ["rmse", "mae", "r2", "bias", "r", "aqi_category_accuracy",
               "unhealthy_precision", "unhealthy_recall", "unhealthy_f1"]


@pytest.mark.parametrize("d", DATA_DIRS, ids=DIR_IDS)
def test_metrics(d):
    m = load(d / "metrics.json")
    has_keys(m, ["generated_at", "dataset", "validation", "models", "headline", "temporal_holdout",
                 "by_region", "scatter", "confusion", "feature_importance", "group_importance",
                 "interval", "per_site"], "metrics")
    ds = m["dataset"]
    has_keys(ds, ["n_sites", "n_countries", "n_rows", "period", "regions", "networks"], "metrics.dataset")
    assert is_int(ds["n_sites"]) and is_int(ds["n_countries"]) and is_int(ds["n_rows"])
    assert len(ds["period"]) == 2 and isinstance(ds["networks"], dict)
    for r in ds["regions"]:
        has_keys(r, ["region", "n_sites", "n_rows"], "metrics.dataset.regions")
    has_keys(m["validation"], ["scheme", "temporal_scheme"], "metrics.validation")

    keys = {mod["key"] for mod in m["models"]}
    assert MODEL_KEYS <= keys, f"missing models {MODEL_KEYS - keys}"
    for mod in m["models"]:
        has_keys(mod, ["key", "name", *METRIC_KEYS], f"metrics.models[{mod.get('key')}]")
        assert all(num(mod[k], True) for k in METRIC_KEYS)
        assert mod["rmse"] >= 0 and mod["mae"] >= 0 and 0 <= mod["aqi_category_accuracy"] <= 1

    has_keys(m["headline"], ["mae_reduction_pct", "rmse_reduction_pct", "aqi_acc_cams", "aqi_acc_airtrace",
                             "unhealthy_recall_cams", "unhealthy_recall_airtrace", "r2_cams", "r2_airtrace"],
             "metrics.headline")
    assert all(num(v) for v in m["headline"].values())
    for k in ("cams_raw", "airtrace"):
        has_keys(m["temporal_holdout"][k], ["rmse", "mae", "r2"], f"metrics.temporal_holdout.{k}")

    for r in m["by_region"]:
        has_keys(r, ["region", "n_sites", "rmse_cams", "rmse_airtrace", "mae_cams", "mae_airtrace",
                     "bias_cams", "bias_airtrace"], "metrics.by_region")

    sc = m["scatter"]
    has_keys(sc, ["obs", "cams", "airtrace", "region"], "metrics.scatter")
    n = len(sc["obs"])
    assert n <= 1500 and all(len(sc[k]) == n for k in ("cams", "airtrace", "region"))

    conf = m["confusion"]
    assert conf["labels"] == AQI_CATEGORIES
    for k in ("cams", "airtrace"):
        mat = conf[k]
        assert len(mat) == 6 and all(len(row) == 6 and all(is_int(v) for v in row) for row in mat)

    fi = m["feature_importance"]
    assert 0 < len(fi) <= 15
    for f in fi:
        has_keys(f, ["feature", "label", "group", "importance"], "metrics.feature_importance")
    for g in m["group_importance"]:
        has_keys(g, ["group", "importance"], "metrics.group_importance")
    assert sum(g["importance"] for g in m["group_importance"]) == pytest.approx(1.0, abs=0.02)

    iv = m["interval"]
    has_keys(iv, ["nominal", "coverage", "mean_width"], "metrics.interval")
    assert iv["nominal"] == 0.8 and 0 <= iv["coverage"] <= 1

    for s in m["per_site"]:
        has_keys(s, ["site_id", "city", "country", "region", "lat", "lon", "n", "mean_obs",
                     "mae_cams", "mae_airtrace"], "metrics.per_site")


# ---------------------------------------------------------------- sources_global.json
@pytest.mark.parametrize("d", DATA_DIRS, ids=DIR_IDS)
def test_sources_global(d):
    s = load(d / "sources_global.json")
    has_keys(s, ["method", "tracers", "factors", "pm_regression", "checks", "by_region"], "sources_global")
    assert text(s["method"]) and isinstance(s["tracers"], list) and s["tracers"]
    nt = len(s["tracers"])
    for f in s["factors"]:
        has_keys(f, ["key", "label", "color", "description", "profile"], "sources_global.factors")
        assert f["key"] in FACTORS and (f["label"], f["color"]) == FACTORS[f["key"]]
        assert len(f["profile"]) == nt and all(num(v) for v in f["profile"])
        assert max(f["profile"]) == pytest.approx(1.0, abs=1e-3), "profile must be max-normalised"
    pr = s["pm_regression"]
    has_keys(pr, ["r2", "n_rows", "intercept", "coef"], "sources_global.pm_regression")
    assert num(pr["r2"]) and is_int(pr["n_rows"]) and isinstance(pr["coef"], dict)
    assert set(pr["coef"]) <= set(FACTORS)
    for c in s["checks"]:
        has_keys(c, ["name", "value", "expected", "passed", "detail"], "sources_global.checks")
        assert isinstance(c["passed"], bool)
    for r in s["by_region"]:
        has_keys(r, ["region", "shares"], "sources_global.by_region")
        assert set(r["shares"]) <= set(FACTORS)
        assert sum(r["shares"].values()) == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------- network_history.json
@pytest.mark.parametrize("d", DATA_DIRS, ids=DIR_IDS)
def test_network_history(d):
    n = load(d / "network_history.json")
    has_keys(n, ["source", "months", "embassy_monitors", "all_pm25_monitors"], "network_history")
    months = n["months"]
    assert months and all(MONTH.match(x) for x in months) and months == sorted(months)
    for k in ("embassy_monitors", "all_pm25_monitors"):
        assert len(n[k]) == len(months) and all(is_int(v) and v >= 0 for v in n[k])
    assert all(e <= a for e, a in zip(n["embassy_monitors"], n["all_pm25_monitors"]))


# ---------------------------------------------------------------- validator self-test
def _example_city() -> dict:
    """Minimal SPEC-conformant city document (keeps the validator honest while no data exists)."""
    day = {"date": "2026-11-03", "pm25": 41.2, "pm25_low": 28.0, "pm25_high": 60.5, "cams_pm25": 30.1,
           "aqi": 115, "category": "Unhealthy for Sensitive Groups", "color": "#ff7e00",
           "drivers": [{"group": "Air stagnation (wind & mixing height)", "effect_pct": 22.0},
                       {"group": "Season & weekday", "effect_pct": -5.0}]}
    return {
        "id": "delhi", "name": "New Delhi", "country": "India", "region": "South Asia",
        "lat": 28.6, "lon": 77.2, "timezone": "Asia/Kolkata", "generated_at": "2026-11-03T06:00:00Z",
        "monitor": {"status": "active", "network": "US Dept. of State", "name": "New Delhi",
                    "last_seen": "2026-11-02", "distance_km": 1.2},
        "today": {"date": "2026-11-03", "pm25": 41.2, "aqi": 115, "category": "Unhealthy for Sensitive Groups",
                  "color": "#ff7e00", "message": "..."},
        "daily": [day, {**day, "date": "2026-11-04"}],
        "hourly": {"time": ["2026-11-03T00:00", "2026-11-03T01:00"], "cams_pm25": [30.0, None],
                   "airtrace_pm25": [41.0, None], "wind_speed": [3.0, 4.0], "wind_dir": [300, 310]},
        "sources": {"window": "2026-10-28 to 2026-11-03", "background_share": 0.2,
                    "shares": [{"key": "burning", "label": "Fires & biomass burning", "share": 0.5,
                                "color": "#f97316", "description": "..."},
                               {"key": "traffic", "label": "Traffic & urban combustion", "share": 0.3,
                                "color": "#3b82f6", "description": "..."},
                               {"key": "background", "label": "Regional background & other", "share": 0.2,
                                "color": "#cbd5e1", "description": "..."}]},
        "fires": {"count_300km": 3, "upwind_count": 1, "nearest_km": 80.5, "frp_sum": 12.0,
                  "points": [[29.0, 76.0, 4.0]]},
        "recent": [{"date": "2026-11-02", "observed": 50.0, "cams": 33.0, "airtrace": 47.0}],
        "history": None,
    }


def test_validator_accepts_example():
    check_city(_example_city(), "delhi")


@pytest.mark.parametrize("mutate", [
    lambda c: c["daily"].clear(),
    lambda c: c["sources"]["shares"].reverse(),
    lambda c: c["sources"]["shares"][0].update(color="#000000"),
    lambda c: c["hourly"]["wind_dir"].pop(),
    lambda c: c["monitor"].update(status="unknown"),
    lambda c: c["daily"][0].update(category="Bad"),
])
def test_validator_rejects_broken(mutate):
    c = _example_city()
    mutate(c)
    with pytest.raises(AssertionError):
        check_city(c, "delhi")
