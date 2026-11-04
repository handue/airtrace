"""Fit the receptor-model source attribution and validate it against ground truth (SOURCES role).

NMF on eight daily CAMS tracers -> factor profiles named by matching to reference chemical
signatures -> NNLS apportionment of *observed* PM2.5 at the training monitors -> validation
checks. Writes

* ``models/sources_nmf.json``      everything ``airtrace.sources.load_source_model()`` needs;
* ``docs/data/sources_global.json`` the website's global source-attribution summary (SPEC.md).

Input: ``data/processed/dataset.parquet`` when the ML step has produced it; otherwise the daily
tracers are rebuilt in memory from the Open-Meteo CAMS cache (read straight from disk, never the
network) and joined with ``data/processed/airnow_pm25_daily.parquet``.

Run from the repo root:  python scripts/06_fit_sources.py
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from airtrace import openmeteo  # noqa: E402
from airtrace.config import (CAMS_VARS, CONFIG, MODELS, PROCESSED, RAW_OPENMETEO, SITE_DATA,  # noqa: E402
                             TRAIN_END, TRAIN_START, WEATHER_VARS)
from airtrace.features import FEATURE_LABELS, cams_frame, daily_features  # noqa: E402
from airtrace.sources import (BACKGROUND_KEY, FACTOR_KEYS, MODEL_FILE, SOURCE_KEYS, SOURCES,  # noqa: E402
                              TRACER_COLS, TRACER_SHORT, SourceModel, complete_rows, fit_pm_regression,
                              label_factors, load_source_model, max_normalise, nnls_activations,
                              r2_score, robust_scale, round_shares, tracer_matrix)

# ---- model settings ------------------------------------------------------------------------
# NO, HCHO and sea salt exist in the CAMS archive only from 2024-09-26 12:00 UTC, so the first
# *full* local day with all eight tracers is 2024-09-27 at every training site.
FIT_START = "2024-09-27"
N_FACTORS = 6
SCALE_QUANTILE = 90.0
# Kullback-Leibler (Poisson) loss: residuals are weighted as if their variance grows with the
# concentration -- the same idea as EPA PMF's concentration-proportional uncertainties -- so a
# handful of extreme hot-spot days cannot monopolise a factor the way they do under squared error.
BETA_LOSS = "kullback-leibler"
SEED = 0
MAX_ITER = 5000
TOL = 1e-6
MIN_SITE_DAYS = 30          # complete days a site needs to enter per-site shares and checks
R2_EXPECTED = 0.3           # stated before fitting: tracers should explain >30% of PM2.5 variance
CV_FOLDS = 5

# ---- validation site lists (hand-made from config/training_sites.csv, by site_id) ------------
# Clearly coastal: monitor within ~15 km of the open sea.
COASTAL = {
    "DS1010004": "Mumbai", "DS1010002": "Chennai", "CMB010001": "Colombo", "586PK2010001": "Karachi",
    "686SEN010001": "Dakar", "566NI2010001": "Lagos", "288GH1010001": "Accra", "768TOG010001": "Lome",
    "384CI1010001": "Abidjan", "608PH1010001": "Manila", "JK1010002": "Jakarta", "DB1010001": "Dubai",
    "AD1010001": "Abu Dhabi", "634QA1010001": "Doha", "MNA010001": "Manama", "KC1010001": "Kuwait City",
    "JDA010001": "Jeddah", "ALG010001": "Algiers", "508MAP010001": "Maputo", "LM1010001": "Lima",
    "CUR010001": "Willemstad", "150030010": "Honolulu", "120112003": "Miami", "840340011006": "Atlantic City",
    "450510008": "Myrtle Beach", "480612004": "Isla Blanca", "000030201": "Port Hawkesbury",
    "020200018": "Anchorage", "021100004": "Juneau (Floyd Dryden)", "840021851111": "Kaktovik",
}
# Clearly inland: roughly 300 km or more from the open sea (Caspian-shore Baku left out of both).
INLAND = {
    "DS1010001": "New Delhi", "586PK3010003": "Lahore", "586PK1010001": "Islamabad", "586PK4010001": "Peshawar",
    "KM1010001": "Kathmandu", "UB1010001": "Ulaanbaatar", "KAZ010001": "Astana", "398KZ1010001": "Almaty",
    "417BI1010001": "Bishkek", "762DUS010001": "Dushanbe", "UZB010001": "Tashkent", "795ABT010001": "Ashgabat",
    "368IRQ010001": "Baghdad", "418LAO010001": "Vientiane", "148CHD010001": "N'Djamena", "KP1010001": "Kampala",
    "646RWA010001": "Kigali", "AA1010002": "Addis Ababa", "566NI1010001": "Abuja", "180COG010001": "Kinshasa",
    "600PAR010001": "Asuncion", "BG1010001": "Bogota", "290470005": "Kansas City", "270834210": "Marshall",
    "381010003": "Ryder", "840080990002": "Lamar", "840560290001": "Cody", "490071003": "P2 (Utah)",
    "CC0090402": "Medicine Hat", "CC0090701": "Athabasca Valley", "160150002": "Garden Valley",
    "481130069": "Dallas", "840390490040": "Columbus (Jackson Pike)", "011030011": "Decatur",
    "550870009": "Appleton", "000065201": "Parry Sound", "CC0011002": "Labrador City", "350130025": "Las Cruces",
}
SAHEL = {"148CHD010001": "N'Djamena", "686SEN010001": "Dakar"}
DUST_GROUP = "Middle East & North Africa + Sahel"
CROP_BURNING = {"DS1010001": "New Delhi", "586PK3010003": "Lahore"}
# Local weekends by FIPS country code (everything else: Saturday + Sunday).
FRI_SAT_WEEKEND = {"EG", "IZ", "KU", "SA", "QA", "BA", "AG", "BG"}
SAT_ONLY_WEEKEND = {"NP"}
DEMO = {"148CHD010001": "N'Djamena (Sahel)", "150030010": "Honolulu (coastal)",
        "DS1010001": "New Delhi (South Asian megacity)"}
DEMO_DAYS = 12


# ---- data ----------------------------------------------------------------------------------
def load_sites() -> pd.DataFrame:
    return pd.read_csv(CONFIG / "training_sites.csv", dtype={"site_id": str, "country": str})


def cams_cache_path(lat: float, lon: float) -> Path:
    """The file ``openmeteo.cams_archive(lat, lon, TRAIN_START, TRAIN_END)`` caches its response in."""
    return RAW_OPENMETEO / f"cams_{lat:.4f}_{lon:.4f}_{TRAIN_START}_{TRAIN_END}_{openmeteo._key(*CAMS_VARS)}.json"


def read_cached_cams(lat: float, lon: float) -> dict | None:
    """Cached CAMS training-window response, or None if it is missing or still being written.

    Deliberately reads the cache file itself instead of calling ``openmeteo.cams_archive``: that
    function re-downloads a missing or truncated file, and the archive quota is used up.
    """
    path = cams_cache_path(lat, lon)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def tracers_from_cache(sites: pd.DataFrame) -> pd.DataFrame:
    """Daily tracers per site from the CAMS cache, joined with observed AirNow PM2.5."""
    frames = []
    for s in sites.itertuples(index=False):
        resp = read_cached_cams(s.lat, s.lon)
        if resp is None:
            continue
        cams = cams_frame(resp)
        # Tracers need CAMS only. An all-NaN weather frame on the CAMS clock keeps every CAMS hour
        # through daily_features' inner join; the weather-derived columns are simply not used.
        wx = pd.DataFrame(np.nan, index=cams.index, columns=WEATHER_VARS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            daily = daily_features(cams, wx, s.lat, s.lon, s.elevation)
        daily = daily[TRACER_COLS + ["cams_pm2_5_mean"]].reset_index()
        daily.insert(0, "site_id", s.site_id)
        frames.append(daily)
    if not frames:
        raise SystemExit("no CAMS cache files found under data/raw/openmeteo")
    table = pd.concat(frames, ignore_index=True)
    obs = pd.read_parquet(PROCESSED / "airnow_pm25_daily.parquet", columns=["aqsid", "date", "pm25_obs"])
    obs = obs.rename(columns={"aqsid": "site_id"})
    obs["date"] = pd.to_datetime(obs["date"])
    return table.merge(obs, on=["site_id", "date"], how="left")


def tracers_from_dataset(path: Path) -> pd.DataFrame | None:
    """Daily tracers + observed PM2.5 from the ML training table, or None if it is unusable."""
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 -- e.g. still being written by the ML step
        print(f"[sources] cannot read {path.name} ({exc}); rebuilding tracers from the CAMS cache")
        return None
    if "date" not in df.columns and df.index.name == "date":
        df = df.reset_index()
    df = df.rename(columns={"aqsid": "site_id"})
    obs_col = next((c for c in ("pm25_obs", "pm25", "obs_pm25") if c in df.columns), None)
    need = ["site_id", "date", *TRACER_COLS]
    missing = [c for c in need if c not in df.columns] + ([] if obs_col else ["pm25_obs"])
    if missing:
        print(f"[sources] {path.name} lacks {missing}; rebuilding tracers from the CAMS cache")
        return None
    keep = need + [obs_col] + (["cams_pm2_5_mean"] if "cams_pm2_5_mean" in df.columns else [])
    df = df[keep].rename(columns={obs_col: "pm25_obs"})
    df["site_id"] = df["site_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_tracer_table(sites: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    ds_path = PROCESSED / "dataset.parquet"
    table, source = None, ""
    if ds_path.exists():
        table, source = tracers_from_dataset(ds_path), "data/processed/dataset.parquet"
    if table is None:
        table, source = tracers_from_cache(sites), "Open-Meteo CAMS cache + data/processed/airnow_pm25_daily.parquet"
    meta = sites[["site_id", "city", "country", "country_name", "region", "network", "lat", "lon"]]
    table = table.merge(meta, on="site_id", how="inner")
    return table.sort_values(["site_id", "date"]).reset_index(drop=True), source


# ---- fitting -------------------------------------------------------------------------------
def fit_factors(Xs: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray, NMF]:
    """NMF on scaled tracers; returns max-normalised profiles in FACTOR_KEYS order, their keys,
    the factor x signature cosine matrix and the fitted estimator."""
    nmf = NMF(n_components=N_FACTORS, init="nndsvda", beta_loss=BETA_LOSS, solver="mu",
              max_iter=MAX_ITER, tol=TOL, random_state=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=ConvergenceWarning)   # never ship an unconverged fit
        nmf.fit(Xs)
    H = max_normalise(nmf.components_)
    keys, cos = label_factors(H)
    order = sorted(range(len(keys)), key=lambda j: (FACTOR_KEYS.index(keys[j]), j))
    return H[order], [keys[j] for j in order], cos[order], nmf


def location_groups(rows: pd.DataFrame) -> np.ndarray:
    """Leakage-free CV groups: sites pooled by rounded 1-degree latitude/longitude (SPEC.md)."""
    return (rows["lat"].round().astype(int).astype(str) + "_" + rows["lon"].round().astype(int).astype(str)).to_numpy()


def grouped_cv_r2(W: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """Out-of-fold R2 of the NNLS PM2.5 regression with GroupKFold over location groups."""
    pred = np.empty_like(y)
    for train, test in GroupKFold(n_splits=CV_FOLDS).split(W, y, groups):
        coef, intercept = fit_pm_regression(W[train], y[train])
        pred[test] = W[test] @ coef + intercept
    return r2_score(y, pred)


# ---- validation ----------------------------------------------------------------------------
def weekend_mask(rows: pd.DataFrame) -> np.ndarray:
    dow = rows["date"].dt.day_of_week.to_numpy()            # Monday = 0 ... Sunday = 6
    country = rows["country"].to_numpy()
    fri_sat = np.isin(country, list(FRI_SAT_WEEKEND))
    sat_only = np.isin(country, list(SAT_ONLY_WEEKEND))
    return np.where(fri_sat, (dow == 4) | (dow == 5), np.where(sat_only, dow == 5, dow >= 5))


def check(name: str, value: float, expected: str, passed: bool, detail: str) -> dict:
    return {"name": name, "value": round(float(value), 3), "expected": expected,
            "passed": bool(passed), "detail": detail}


def run_checks(rows: pd.DataFrame, act: pd.DataFrame, site_shares: pd.DataFrame, reg: dict) -> list[dict]:
    checks = []
    by_key = {k: act[[c for c in act.columns if c.endswith(f"_{k}")]].sum(axis=1) for k in FACTOR_KEYS}
    eligible = site_shares.index

    # (a) traffic emissions follow the working week
    weekend = weekend_mask(rows)
    frame = pd.DataFrame({"site": rows["site_id"].to_numpy(), "weekend": weekend,
                          "traffic": by_key["traffic"].to_numpy(), "no2": rows[TRACER_COLS[0]].to_numpy()})
    frame = frame[frame["site"].isin(eligible)]
    means = frame.groupby(["site", "weekend"])[["traffic", "no2"]].mean().unstack("weekend")
    ratio = (means[("traffic", False)] / means[("traffic", True)]).replace([np.inf, -np.inf], np.nan).dropna()
    ratio_no2 = (means[("no2", False)] / means[("no2", True)]).dropna()
    checks.append(check(
        "Traffic factor is higher on working days", ratio.median(), "> 1",
        ratio.median() > 1,
        f"Median over {len(ratio)} sites of mean weekday / mean weekend traffic activation (local weekends: "
        f"Fri-Sat in the Gulf states, Iraq, Egypt, Algeria and Bangladesh, Sat in Nepal, Sat-Sun elsewhere); "
        f"{int((ratio > 1).sum())}/{len(ratio)} sites above 1. The raw CAMS NO2 input has a median ratio of "
        f"{ratio_no2.median():.3f} on the same days, i.e. the global model itself shows almost no weekly cycle."))

    # (b) sea salt matters at the coast, not inland
    coastal = [s for s in COASTAL if s in eligible]
    inland = [s for s in INLAND if s in eligible]
    m_coast, m_inland = site_shares.loc[coastal, "marine"].mean(), site_shares.loc[inland, "marine"].mean()
    marine_act = by_key["marine"].groupby(rows["site_id"].to_numpy()).mean()
    checks.append(check(
        "Sea-salt share is higher at coastal than inland sites", m_coast, f"> inland mean ({m_inland:.3f})",
        m_coast > m_inland,
        f"Mean marine share at {len(coastal)} clearly coastal sites {m_coast:.3f} vs {len(inland)} clearly inland "
        f"sites {m_inland:.3f}. The marine factor activation itself is {marine_act.loc[coastal].mean():.3f} at the "
        f"coast vs {marine_act.loc[inland].mean():.3f} inland; its PM2.5 coefficient is {reg['coef']['marine']:.2f} "
        f"ug/m3 per unit activation."))

    # (c) dust share peaks in the Middle East / North Africa and the Sahel
    region = rows.drop_duplicates("site_id").set_index("site_id")["region"]
    group = pd.Series([DUST_GROUP if (region[s] == "Middle East & North Africa" or s in SAHEL) else region[s]
                       for s in site_shares.index], index=site_shares.index)
    dust_by_group = site_shares["dust"].groupby(group).mean().sort_values(ascending=False)
    others = dust_by_group.drop(DUST_GROUP)
    city = rows.drop_duplicates("site_id").set_index("site_id")["city"]
    top = site_shares["dust"].sort_values(ascending=False).head(5)
    checks.append(check(
        "Desert-dust share is highest in the Middle East, North Africa and Sahel",
        dust_by_group[DUST_GROUP], f"> every other region (next: {others.index[0]} {others.iloc[0]:.3f})",
        dust_by_group[DUST_GROUP] > others.iloc[0],
        "Mean dust share by region (Sahel = N'Djamena, Dakar): "
        + ", ".join(f"{g} {v:.3f}" for g, v in dust_by_group.items())
        + ". Top sites: " + ", ".join(f"{city[s]} {v:.3f}" for s, v in top.items()) + "."))

    # (d) post-monsoon crop-residue burning in the Indo-Gangetic Plain
    burn = by_key["burning"].to_numpy()
    month = rows["date"].dt.month.to_numpy()
    ratios = {}
    for sid, name in CROP_BURNING.items():
        here = rows["site_id"].to_numpy() == sid
        on, off = burn[here & np.isin(month, [10, 11])], burn[here & np.isin(month, [12, 1, 2])]
        if len(on) and len(off) and off.mean() > 0:
            ratios[name] = on.mean() / off.mean()
    worst = min(ratios.values()) if ratios else float("nan")
    checks.append(check(
        "Burning factor peaks in the Oct-Nov crop-residue season (Delhi, Lahore)", worst,
        "> 1 at every site", bool(ratios) and worst > 1,
        "Mean burning activation Oct-Nov 2024 / Dec 2024-Feb 2025: "
        + ", ".join(f"{n} {r:.2f}" for n, r in ratios.items()) + " (value = the lower ratio)."))

    # (e) the factors explain observed PM2.5
    ref = f"; raw CAMS PM2.5 alone reaches R2 {reg['r2_cams_raw']:.3f} on the same rows" if reg.get("r2_cams_raw") is not None else ""
    checks.append(check(
        "Factor activations explain observed PM2.5", reg["r2"], f"> {R2_EXPECTED}", reg["r2"] > R2_EXPECTED,
        f"In-sample R2 of the NNLS regression of observed daily PM2.5 on the {N_FACTORS} activations + intercept "
        f"over {reg['n_rows']} site-days at {reg['n_sites']} monitors; out-of-fold R2 with location-grouped "
        f"{CV_FOLDS}-fold CV {reg['r2_cv_location']:.3f}{ref}."))
    return checks


# ---- main ----------------------------------------------------------------------------------
def main() -> None:
    sites = load_sites()
    table, data_source = load_tracer_table(sites)
    rows = table[table["date"] >= FIT_START]
    rows = rows[complete_rows(tracer_matrix(rows))].reset_index(drop=True)
    print(f"[sources] {len(rows)} complete site-days at {rows['site_id'].nunique()} sites "
          f"({rows['date'].min():%Y-%m-%d} to {rows['date'].max():%Y-%m-%d}) from {data_source}")

    # 1-2. robust scaling + NMF
    X = np.clip(tracer_matrix(rows), 0.0, None)
    scale = robust_scale(X, SCALE_QUANTILE)
    Xs = X / scale
    H, keys, cos, nmf = fit_factors(Xs)
    assert sorted(keys) == sorted(FACTOR_KEYS), f"factor naming is not one-to-one: {keys}"
    print(f"[sources] NMF k={N_FACTORS} ({BETA_LOSS}) converged in {nmf.n_iter_} iterations")
    print("          " + " ".join(f"{s:>8s}" for s in TRACER_SHORT) + "   match")
    for j, k in enumerate(keys):
        print(f"{k:>9s} " + " ".join(f"{v:8.2f}" for v in H[j]) + f"   {cos[j, FACTOR_KEYS.index(k)]:.2f}")

    # 3. apportion observed PM2.5
    W = nnls_activations(Xs, H)
    has_obs = rows["pm25_obs"].notna().to_numpy()
    y = rows.loc[has_obs, "pm25_obs"].to_numpy(float)
    coef, intercept = fit_pm_regression(W[has_obs], y)
    pred = W[has_obs] @ coef + intercept
    obs_rows = rows[has_obs]
    reg = {
        "r2": r2_score(y, pred), "n_rows": int(has_obs.sum()), "n_sites": int(obs_rows["site_id"].nunique()),
        "intercept": intercept, "coef": dict(zip(keys, coef)),
        "r2_cv_location": grouped_cv_r2(W[has_obs], y, location_groups(obs_rows)),
        "r2_cams_raw": (r2_score(y, obs_rows["cams_pm2_5_mean"].to_numpy(float))
                        if "cams_pm2_5_mean" in rows and obs_rows["cams_pm2_5_mean"].notna().all() else None),
        "mean_contribution": {**dict(zip(keys, (W[has_obs] * coef).mean(axis=0))), BACKGROUND_KEY: intercept},
        "mean_observed": float(y.mean()),
    }
    print(f"[sources] PM2.5 NNLS: R2={reg['r2']:.3f} (location-CV {reg['r2_cv_location']:.3f}) n={reg['n_rows']}"
          f" sites={reg['n_sites']} intercept={intercept:.2f}; coef "
          + ", ".join(f"{k}={v:.2f}" for k, v in reg["coef"].items()))

    # 4. serving model (round-tripped through JSON so everything below uses the shipped artefact)
    trained_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    explained = (W.mean(axis=0)[:, None] * H)
    explained = explained / explained.sum(axis=0, keepdims=True)
    model = SourceModel(
        tracer_columns=TRACER_COLS, scale=scale, components=H, factor_keys=keys, coef=coef, intercept=intercept,
        sources=SOURCES,
        meta={"version": f"sources-nmf-{trained_at[:10]}", "trained_at": trained_at,
              "method": {"model": "sklearn NMF", "n_components": N_FACTORS, "beta_loss": BETA_LOSS,
                         "solver": "mu", "init": "nndsvda", "random_state": SEED,
                         "n_iter": int(nmf.n_iter_), "scale": f"p{SCALE_QUANTILE:g} of each tracer",
                         "activation_transform": "row-wise NNLS onto fixed max-normalised profiles",
                         "pm_regression": "NNLS of observed PM2.5 on activations + intercept"},
              "train_period": [FIT_START, TRAIN_END], "n_rows": int(len(rows)),
              "n_sites": int(rows["site_id"].nunique()), "data_source": data_source,
              "match_cosine": {k: round(float(cos[j, FACTOR_KEYS.index(k)]), 3) for j, k in enumerate(keys)},
              "pm_regression": {"r2": round(reg["r2"], 4), "r2_cv_location": round(reg["r2_cv_location"], 4),
                                "n_rows": reg["n_rows"], "n_sites": reg["n_sites"]}})
    model_path = MODELS / MODEL_FILE
    model.save(model_path)
    model = load_source_model(model_path)
    assert np.allclose(model.activations(rows).to_numpy(), W), "serving transform differs from training"

    # 5. per-site shares and validation checks
    site_shares = {}
    for sid, g in rows.groupby("site_id"):
        if len(g) >= MIN_SITE_DAYS:
            res = model.shares(g)
            site_shares[sid] = {s["key"]: s["share"] for s in res["shares"]}
    site_shares = pd.DataFrame.from_dict(site_shares, orient="index")[SOURCE_KEYS]
    act = model.activations(rows)
    checks = run_checks(rows, act, site_shares, reg)
    for c in checks:
        print(f"[check] {'PASS' if c['passed'] else 'FAIL'} {c['name']}: {c['value']} (expected {c['expected']})")

    # 6. website summary
    region = rows.drop_duplicates("site_id").set_index("site_id")["region"]
    by_region = []
    for name, g in site_shares.groupby(region.reindex(site_shares.index)):
        mean = g.mean()
        by_region.append({"region": name, "n_sites": int(len(g)),
                          "shares": round_shares((mean / mean.sum()).to_dict())})
    by_region.sort(key=lambda r: r["region"])
    summary = {
        "generated_at": trained_at,
        "method": (f"Receptor-model source attribution: non-negative matrix factorisation (the factor model "
                   f"behind EPA PMF; {N_FACTORS} factors, Kullback-Leibler loss) of {len(TRACER_COLS)} daily "
                   f"Copernicus CAMS tracers scaled by their {SCALE_QUANTILE:g}th percentiles, over "
                   f"{len(rows)} site-days at {rows['site_id'].nunique()} reference sites ({FIT_START} to "
                   f"{TRAIN_END}). Factors are named by optimal (Hungarian) matching of their profiles to "
                   f"reference chemical signatures, and observed PM2.5 is apportioned by non-negative least "
                   f"squares on the factor activations plus an intercept (regional background & other)."),
        "period": [FIT_START, TRAIN_END],
        "n_sites": int(rows["site_id"].nunique()),
        "tracers": [FEATURE_LABELS[c] for c in TRACER_COLS],
        "factors": [{"key": k, "label": SOURCES[k]["label"], "color": SOURCES[k]["color"],
                     "description": SOURCES[k]["description"],
                     "profile": [round(float(v), 3) for v in H[j]],
                     "tracer_share": [round(float(v), 3) for v in explained[j]],
                     "match": round(float(cos[j, FACTOR_KEYS.index(k)]), 3)}
                    for j, k in enumerate(keys)],
        "pm_regression": {"r2": round(reg["r2"], 3), "n_rows": reg["n_rows"], "n_sites": reg["n_sites"],
                          "intercept": round(intercept, 2),
                          "coef": {k: round(float(v), 3) for k, v in reg["coef"].items()},
                          "r2_cv_location": round(reg["r2_cv_location"], 3),
                          "r2_cams_raw": None if reg["r2_cams_raw"] is None else round(reg["r2_cams_raw"], 3),
                          "mean_observed": round(reg["mean_observed"], 1),
                          "mean_contribution": {k: round(float(v), 1) for k, v in reg["mean_contribution"].items()}},
        "checks": checks,
        "by_region": by_region,
    }
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    out = SITE_DATA / "sources_global.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(summary, indent=1))
    tmp.replace(out)
    print(f"[sources] wrote {model_path.relative_to(MODELS.parent)} and {out.relative_to(MODELS.parent)}")

    # 7. demo: shares over the last days of training data for contrasting cities
    for sid, name in DEMO.items():
        g = rows[rows["site_id"] == sid].tail(DEMO_DAYS)
        if g.empty:
            continue
        res = model.shares(g)
        print(f"[demo] {name} {g['date'].min():%Y-%m-%d}..{g['date'].max():%Y-%m-%d}: "
              + ", ".join(f"{s['key']} {s['share']:.3f}" for s in res["shares"]))


if __name__ == "__main__":
    main()
