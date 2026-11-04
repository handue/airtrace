"""AirTrace PM2.5 model: machine-learning correction of the Copernicus CAMS global forecast.

Target is ``log1p(PM2.5)`` (daily mean, ug/m3); inputs are :data:`airtrace.features.FEATURES`.
The point forecast averages two LightGBM members in log space:

* ``level`` (``models/pm25_lgbm.txt``) predicts log1p(PM2.5) directly and is constrained to be
  monotone increasing in CAMS PM2.5;
* ``ratio`` (``models/pm25_ratio.txt``) predicts the log correction factor
  log1p(PM2.5) - log1p(CAMS PM2.5) (trained with log1p(CAMS) as offset). A tree on the level
  cannot exceed the values it saw in training; the ratio member keeps scaling with CAMS during
  smog and dust episodes. Where CAMS PM2.5 is missing the level member is used alone.

Uncertainty: quantile members (alpha 0.1 / 0.9, ``models/pm25_q10.txt`` / ``pm25_q90.txt``)
widened by a conformal margin estimated from leave-location-out residuals (conformalised
quantile regression, Romano et al. 2019), so the 80 % band is calibrated for locations the model
has never seen. Every member uses site-balanced sample weights (each monitor counts equally).

Validation (:func:`train_and_evaluate`): leave-location-out GroupKFold(5) over 1-degree lat/lon
cells against three baselines (raw CAMS, CAMS + linear calibration in log space, the same ML
without any CAMS input), a forward-in-time holdout and an ablation table.

Public API used by the live pipeline: :func:`load_models`, :func:`predict`, :func:`contributions`.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split

from .aqi import category_index
from .config import AQI_CATEGORIES, MODELS, PROCESSED, SITE_DATA, UNHEALTHY_PM25
from .dataset import GROUP, OBS
from .features import FEATURE_GROUPS, FEATURE_LABELS, FEATURES

TARGET = "log1p_pm25"
CAMS_PM = "cams_pm2_5_mean"
SEED = 42
N_FOLDS = 5
INTERVAL_NOMINAL = 0.8
QUANTILES = {"q10": 0.1, "q90": 0.9}
TEMPORAL_SPLIT = "2024-12-01"
SCATTER_MAX = 1500
MODEL_FILES = {"level": "pm25_lgbm.txt", "ratio": "pm25_ratio.txt",
               "q10": "pm25_q10.txt", "q90": "pm25_q90.txt"}
META_FILE = "model_meta.json"
ELEVATION_SOURCE = "Open-Meteo weather response 'elevation' (90 m DEM, m a.s.l.)"
# Deterministic LightGBM gives identical models for any thread count; 4 threads is ~2x faster
# than all cores on laptops with efficiency cores.
N_THREADS = int(os.environ.get("AIRTRACE_THREADS", min(4, os.cpu_count() or 1)))

CAMS_DERIVED = ("pm25_pm10_ratio", "dust_frac", "so2_co_ratio")
NO_CAMS_FEATURES = [f for f in FEATURES if not f.startswith("cams_") and f not in CAMS_DERIVED]
LOCATION_FEATURES = frozenset(FEATURE_GROUPS["Location & terrain"])
FEATURE_GROUP = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
MODEL_NAMES = {
    "cams_raw": "CAMS global model (raw)",
    "linear": "CAMS + linear calibration",
    "ml_no_cams": "ML without CAMS (weather + location)",
    "airtrace": "AirTrace (CAMS + ML correction)",
}


# --------------------------------------------------------------------------- configuration
@dataclass(frozen=True)
class TreeConfig:
    """LightGBM settings shared by all members (regularised: the test locations are unseen)."""
    num_leaves: int = 15
    min_data_in_leaf: int = 100
    learning_rate: float = 0.05
    num_rounds: int = 800
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    lambda_l2: float = 1.0
    balance_sites: bool = True
    monotone_cams: bool = True
    # Split-gain multiplier for lat/lon/elevation: damps "copy the neighbouring monitor's bias",
    # which does not transfer to an unseen city (e.g. Kampala's smoke onto Kigali).
    location_penalty: float = 0.5

    def params(self, features: list[str], objective: str = "regression", alpha: float | None = None) -> dict:
        p = {
            "objective": objective, "learning_rate": self.learning_rate, "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf, "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction, "bagging_freq": 1, "lambda_l2": self.lambda_l2,
            "seed": SEED, "deterministic": True, "force_col_wise": True, "verbosity": -1,
            "num_threads": N_THREADS,
        }
        if alpha is not None:
            p["alpha"] = alpha
        if self.location_penalty != 1.0:
            p["feature_contri"] = [self.location_penalty if f in LOCATION_FEATURES else 1.0 for f in features]
        if self.monotone_cams and CAMS_PM in features:
            p["monotone_constraints"] = [1 if f == CAMS_PM else 0 for f in features]
            p["monotone_constraints_method"] = "advanced"
        return p


DEFAULT_CONFIG = TreeConfig()


# --------------------------------------------------------------------------- training primitives
def _matrix(X: pd.DataFrame, features: list[str]) -> np.ndarray:
    missing = [f for f in features if f not in X.columns]
    if missing:
        raise ValueError(f"input is missing {len(missing)} of the {len(features)} model features: "
                         f"{', '.join(missing)} (build rows with airtrace.features.daily_features)")
    return X[features].to_numpy(dtype=np.float64, na_value=np.nan)


def cams_offset(X: pd.DataFrame) -> np.ndarray:
    """log1p(CAMS PM2.5), the ratio member's offset (NaN where CAMS is missing)."""
    return np.log1p(np.clip(X[CAMS_PM].to_numpy(dtype=np.float64, na_value=np.nan), 0.0, None))


def site_weights(site_ids: pd.Series) -> np.ndarray:
    """Row weights giving every monitor the same total weight (mean weight 1)."""
    counts = site_ids.map(site_ids.value_counts()).to_numpy(dtype=np.float64)
    w = 1.0 / counts
    return w * (len(w) / w.sum())


def fit_member(data: pd.DataFrame, member: str, cfg: TreeConfig = DEFAULT_CONFIG,
               features: list[str] = FEATURES) -> lgb.Booster:
    """Train one member ("level", "ratio", "q10" or "q90") on ``data`` (rows of the dataset)."""
    y = np.log1p(data[OBS].to_numpy(dtype=np.float64))
    weight = site_weights(data["site_id"]) if cfg.balance_sites else None
    init_score, objective, alpha = None, "regression", None
    if member == "ratio":
        init_score = cams_offset(data)
        if not np.isfinite(init_score).all():
            raise ValueError("ratio member needs CAMS PM2.5 on every training row")
        cfg = replace(cfg, monotone_cams=False)   # the correction factor may fall as CAMS rises
    elif member in QUANTILES:
        objective, alpha = "quantile", QUANTILES[member]
        cfg = replace(cfg, monotone_cams=False)   # LightGBM: no monotone constraints for quantile loss
    elif member != "level":
        raise ValueError(f"unknown member {member!r}")
    ds = lgb.Dataset(_matrix(data, features), label=y, weight=weight, init_score=init_score,
                     feature_name=list(features), free_raw_data=True)
    return lgb.train(cfg.params(list(features), objective, alpha), ds, num_boost_round=cfg.num_rounds)


def member_predict(booster: lgb.Booster, member: str, X: pd.DataFrame, features: list[str]) -> np.ndarray:
    """A member's prediction in log1p space (the ratio member adds back its CAMS offset)."""
    raw = booster.predict(_matrix(X, features), num_threads=N_THREADS)
    return raw + cams_offset(X) if member == "ratio" else raw


def blend(level_log: np.ndarray, ratio_log: np.ndarray) -> np.ndarray:
    """AirTrace point forecast in log1p space: mean of the members (level alone if CAMS missing)."""
    return np.where(np.isfinite(ratio_log), 0.5 * (level_log + ratio_log), level_log)


def interval(mid_log: np.ndarray, q10_log: np.ndarray, q90_log: np.ndarray,
             margin: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Point forecast and calibrated 80 % interval in ug/m3, guaranteed low <= pm25 <= high.

    Crossed quantiles are swapped, the conformal ``margin`` (log space) is applied, and the band
    is stretched to contain the point forecast if needed.
    """
    lo = np.minimum(q10_log, q90_log) - margin
    hi = np.maximum(q10_log, q90_log) + margin
    lo, hi = np.minimum(lo, mid_log), np.maximum(hi, mid_log)

    def to_pm(v: np.ndarray) -> np.ndarray:
        return np.clip(np.expm1(v), 0.0, None)
    return to_pm(mid_log), to_pm(lo), to_pm(hi)


def conformal_margin(q10_log: np.ndarray, q90_log: np.ndarray, y_log: np.ndarray,
                     nominal: float = INTERVAL_NOMINAL) -> float:
    """CQR margin: the finite-sample (1 - alpha) quantile of max(lo - y, y - hi) in log space."""
    lo, hi = np.minimum(q10_log, q90_log), np.maximum(q10_log, q90_log)
    scores = np.sort(np.maximum(lo - y_log, y_log - hi))
    k = min(len(scores), math.ceil((len(scores) + 1) * nominal))
    return float(scores[k - 1])


# --------------------------------------------------------------------------- cross-validation
def location_folds(data: pd.DataFrame, n_splits: int = N_FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    """Leave-location-out folds: whole 1-degree cells are held out together."""
    return list(GroupKFold(n_splits=n_splits).split(data, groups=data[GROUP]))


def oof_member(data: pd.DataFrame, folds, member: str, cfg: TreeConfig = DEFAULT_CONFIG,
               features: list[str] = FEATURES) -> np.ndarray:
    """Out-of-fold log1p predictions of one member."""
    out = np.full(len(data), np.nan)
    for tr, te in folds:
        booster = fit_member(data.iloc[tr], member, cfg, features)
        out[te] = member_predict(booster, member, data.iloc[te], features)
    return out


def oof_linear(data: pd.DataFrame, folds) -> np.ndarray:
    """Baseline: in-fold OLS of log1p(obs) on log1p(CAMS PM2.5), out-of-fold log predictions."""
    x, y = cams_offset(data), np.log1p(data[OBS].to_numpy(dtype=np.float64))
    out = np.full(len(data), np.nan)
    for tr, te in folds:
        slope, intercept = np.polyfit(x[tr], y[tr], 1)
        out[te] = intercept + slope * x[te]
    return out


def cross_conformal(q10_log: np.ndarray, q90_log: np.ndarray, y_log: np.ndarray, folds,
                    nominal: float = INTERVAL_NOMINAL) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Honest evaluation of the conformal widening: each fold's margin comes only from the
    out-of-fold residuals of the *other* folds' locations."""
    lo, hi, margins = q10_log.copy(), q90_log.copy(), []
    for tr, te in folds:
        m = conformal_margin(q10_log[tr], q90_log[tr], y_log[tr], nominal)
        lo[te] -= m
        hi[te] += m
        margins.append(m)
    return lo, hi, margins


# --------------------------------------------------------------------------- metrics
def metrics(obs, pred) -> dict:
    """Error metrics in ug/m3 plus AQI-category and 'unhealthy' (>= 35.5) detection skill."""
    o, p = np.asarray(obs, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    e = p - o
    ss_tot = float(np.sum((o - o.mean()) ** 2))
    cat_o = np.fromiter((category_index(v) for v in o), dtype=int, count=len(o))
    cat_p = np.fromiter((category_index(v) for v in p), dtype=int, count=len(p))
    pos_o, pos_p = o >= UNHEALTHY_PM25, p >= UNHEALTHY_PM25
    tp = float(np.sum(pos_o & pos_p))
    precision = tp / pos_p.sum() if pos_p.any() else None
    recall = tp / pos_o.sum() if pos_o.any() else None
    f1 = None
    if precision is not None and recall is not None:
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    r = float(np.corrcoef(o, p)[0, 1]) if len(o) > 1 and o.std() > 0 and p.std() > 0 else None
    return {
        "rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
        "r2": 1.0 - float(np.sum(e ** 2)) / ss_tot if ss_tot > 0 else None,
        "bias": float(np.mean(e)), "r": r, "aqi_category_accuracy": float(np.mean(cat_o == cat_p)),
        "unhealthy_precision": precision, "unhealthy_recall": recall, "unhealthy_f1": f1,
    }


_DECIMALS = {"rmse": 2, "mae": 2, "bias": 2, "site_mae": 2, "log_rmse": 4}


def rounded(m: dict, default: int = 3) -> dict:
    """JSON-safe copy: rounded floats, None for missing / non-finite values."""
    out = {}
    for k, v in m.items():
        if isinstance(v, (float, np.floating)):
            out[k] = round(float(v), _DECIMALS.get(k, default)) if math.isfinite(v) else None
        else:
            out[k] = v
    return out


def confusion(obs, pred) -> list[list[int]]:
    """6x6 AQI-category confusion matrix, rows = observed, columns = predicted."""
    mat = np.zeros((len(AQI_CATEGORIES), len(AQI_CATEGORIES)), dtype=int)
    for o, p in zip(obs, pred):
        mat[category_index(o), category_index(p)] += 1
    return mat.tolist()


def gain_importance(boosters: list[lgb.Booster]) -> pd.Series:
    """Share of total split gain per feature, averaged over the point members."""
    shares = []
    for b in boosters:
        g = pd.Series(b.feature_importance(importance_type="gain"), index=b.feature_name(), dtype=float)
        shares.append(g / g.sum())
    return pd.concat(shares, axis=1).mean(axis=1).sort_values(ascending=False)


# --------------------------------------------------------------------------- artifacts & inference
def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, allow_nan=False, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def save_models(boosters: dict[str, lgb.Booster], meta: dict, model_dir: Path | str = MODELS) -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    for member, booster in boosters.items():
        path = model_dir / MODEL_FILES[member]
        tmp = path.with_name(path.name + ".tmp")
        booster.save_model(str(tmp))
        os.replace(tmp, path)
    _write_json(model_dir / META_FILE, meta)


def load_models(model_dir: Path | str | None = None) -> dict:
    """Load the boosters + metadata. ``model_dir`` defaults to $AIRTRACE_MODEL_DIR or models/."""
    model_dir = Path(model_dir or os.environ.get("AIRTRACE_MODEL_DIR") or MODELS)
    meta = json.loads((model_dir / META_FILE).read_text(encoding="utf-8"))
    features = list(meta["features"])
    models: dict = {"meta": meta, "features": features}
    for member, name in meta.get("files", {"level": MODEL_FILES["level"], "q10": MODEL_FILES["q10"],
                                           "q90": MODEL_FILES["q90"]}).items():
        booster = lgb.Booster(model_file=str(model_dir / name))
        if booster.feature_name() != features:
            raise ValueError(f"{name}: booster features differ from {META_FILE}")
        models[member] = booster
    return models


def _point_log(models: dict, X: pd.DataFrame, M: np.ndarray) -> np.ndarray:
    level = models["level"].predict(M, num_threads=N_THREADS)
    if "ratio" not in models:
        return level
    return blend(level, models["ratio"].predict(M, num_threads=N_THREADS) + cams_offset(X))


def predict(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Daily PM2.5 forecast (ug/m3) with its calibrated 80 % interval.

    ``X`` needs every feature in ``models["features"]`` (extra columns are ignored; order is
    taken from model_meta.json). Returns columns pm25, pm25_low, pm25_high on ``X.index``.
    """
    features = models["features"]
    M = _matrix(X, features)
    if len(X) == 0:
        return pd.DataFrame({"pm25": [], "pm25_low": [], "pm25_high": []}, index=X.index, dtype=float)
    mid = _point_log(models, X, M)
    q10 = models["q10"].predict(M, num_threads=N_THREADS)
    q90 = models["q90"].predict(M, num_threads=N_THREADS)
    margin = float(models["meta"].get("interval", {}).get("conformal_margin_log", 0.0))
    pm, lo, hi = interval(mid, q10, q90, margin)
    return pd.DataFrame({"pm25": pm, "pm25_low": lo, "pm25_high": hi}, index=X.index)


def contributions(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    """TreeSHAP contributions of the point forecast in log1p space.

    One column per feature plus ``bias``; each row sums to log1p(predicted PM2.5). The ratio
    member's CAMS offset is attributed exactly: log1p(CAMS) minus its training mean goes to
    ``cams_pm2_5_mean`` and the training mean goes to ``bias``.
    """
    features = models["features"]
    M = _matrix(X, features)
    if len(X) == 0:
        return pd.DataFrame(columns=[*features, "bias"], index=X.index, dtype=float)
    contrib = models["level"].predict(M, pred_contrib=True, num_threads=N_THREADS)
    if "ratio" in models:
        ratio = models["ratio"].predict(M, pred_contrib=True, num_threads=N_THREADS)
        offset = cams_offset(X)
        base = float(models["meta"]["ratio_offset_mean"])
        ratio[:, features.index(CAMS_PM)] += offset - base
        ratio[:, -1] += base
        contrib = np.where(np.isfinite(offset)[:, None], 0.5 * (contrib + ratio), contrib)
    return pd.DataFrame(contrib, index=X.index, columns=[*features, "bias"])


# --------------------------------------------------------------------------- full training run
@dataclass(frozen=True)
class Ablation:
    key: str
    description: str
    member: str
    cfg: TreeConfig
    features: tuple[str, ...] = tuple(FEATURES)


ABLATIONS = [
    Ablation("level_plain", "Level model, unweighted, unconstrained", "level",
             replace(DEFAULT_CONFIG, balance_sites=False, monotone_cams=False, location_penalty=1.0)),
    Ablation("level_balanced", "+ site-balanced sample weights", "level",
             replace(DEFAULT_CONFIG, monotone_cams=False, location_penalty=1.0)),
    Ablation("level_monotone", "+ monotone in CAMS PM2.5", "level",
             replace(DEFAULT_CONFIG, location_penalty=1.0)),
    Ablation("level", "+ location split-gain penalty 0.5 (AirTrace level member)", "level", DEFAULT_CONFIG),
    Ablation("level_trees_31", "level member with num_leaves 31 / min_data 50 / 500 rounds", "level",
             replace(DEFAULT_CONFIG, num_leaves=31, min_data_in_leaf=50, num_rounds=500)),
    Ablation("level_trees_63", "level member with num_leaves 63 / min_data 30 / 300 rounds", "level",
             replace(DEFAULT_CONFIG, num_leaves=63, min_data_in_leaf=30, num_rounds=300)),
    Ablation("ratio", "Ratio member: CAMS log-offset + ML correction factor", "ratio", DEFAULT_CONFIG),
]
CORE_ABLATIONS = {"level", "ratio"}   # needed for the AirTrace forecast itself


@dataclass
class TrainingResult:
    boosters: dict[str, lgb.Booster]
    meta: dict
    metrics: dict
    cv: pd.DataFrame
    experiments: list[dict] = field(default_factory=list)


def _experiment_row(key: str, description: str, data: pd.DataFrame, log_pred: np.ndarray) -> dict:
    obs = data[OBS].to_numpy(dtype=np.float64)
    pred = np.clip(np.expm1(log_pred), 0.0, None)
    m = metrics(obs, pred)
    site_mae = pd.Series(np.abs(pred - obs)).groupby(data["site_id"].to_numpy()).mean().mean()
    return rounded({"key": key, "description": description, "rmse": m["rmse"], "mae": m["mae"], "r2": m["r2"],
                    "bias": m["bias"], "log_rmse": float(np.sqrt(np.mean((log_pred - np.log1p(obs)) ** 2))),
                    "site_mae": float(site_mae), "aqi_category_accuracy": m["aqi_category_accuracy"],
                    "unhealthy_recall": m["unhealthy_recall"]})


def train_and_evaluate(data: pd.DataFrame, cfg: TreeConfig = DEFAULT_CONFIG, ablations: bool = True,
                       log=print) -> TrainingResult:
    """Cross-validate baselines + AirTrace, run the temporal holdout and ablations, fit the final
    members on all rows and assemble metrics.json / model_meta.json content."""
    t0 = time.time()
    data = data.sort_values(["site_id", "date"], ignore_index=True)
    data["date"] = pd.to_datetime(data["date"])
    obs = data[OBS].to_numpy(dtype=np.float64)
    y = np.log1p(obs)
    cams = data[CAMS_PM].to_numpy(dtype=np.float64)
    folds = location_folds(data)
    log(f"{len(data):,} rows, {data['site_id'].nunique()} sites, {data[GROUP].nunique()} location groups, "
        f"{len(folds)} leave-location-out folds")

    # -- out-of-fold predictions ------------------------------------------------------------
    oof, experiments = {}, []
    for ab in ABLATIONS:
        core = ab.key in CORE_ABLATIONS
        if not (core or ablations):
            continue
        run_cfg = cfg if core else ab.cfg     # deployed members always use ``cfg``
        oof[ab.key] = oof_member(data, folds, ab.member, run_cfg, list(ab.features))
        experiments.append(_experiment_row(ab.key, ab.description, data, oof[ab.key]))
        log(f"  oof {ab.key:<18} done ({time.time() - t0:5.1f}s)")
    oof["airtrace"] = blend(oof["level"], oof["ratio"])
    experiments.append(_experiment_row("airtrace", "AirTrace = mean(level, ratio) in log space", data,
                                       oof["airtrace"]))
    oof["ml_no_cams"] = oof_member(data, folds, "level", cfg, NO_CAMS_FEATURES)
    oof["linear"] = oof_linear(data, folds)
    q10 = oof_member(data, folds, "q10", cfg)
    q90 = oof_member(data, folds, "q90", cfg)
    log(f"  oof baselines + quantiles done ({time.time() - t0:5.1f}s)")

    preds = {"cams_raw": np.clip(cams, 0.0, None),
             "linear": np.clip(np.expm1(oof["linear"]), 0.0, None),
             "ml_no_cams": np.clip(np.expm1(oof["ml_no_cams"]), 0.0, None)}
    lo_cv, hi_cv, fold_margins = cross_conformal(q10, q90, y, folds)
    preds["airtrace"], low, high = interval(oof["airtrace"], lo_cv, hi_cv, 0.0)
    raw_cov = float(np.mean((y >= np.minimum(q10, q90)) & (y <= np.maximum(q10, q90))))
    coverage = float(np.mean((obs >= low) & (obs <= high)))
    experiments.append(rounded({"key": "interval_raw_quantiles",
                                "description": "80% band from raw q10/q90 members (no conformal margin)",
                                "coverage": raw_cov}))
    experiments.append(rounded({"key": "interval_conformal",
                                "description": "80% band with leave-location-out conformal margin",
                                "coverage": coverage, "mean_width": float(np.mean(high - low))}))

    model_metrics = {k: metrics(obs, preds[k]) for k in MODEL_NAMES}

    # -- temporal holdout: same sites, future months -----------------------------------------
    split = pd.Timestamp(TEMPORAL_SPLIT)
    past, future = data[data["date"] < split], data[data["date"] >= split]
    t_level, t_ratio = fit_member(past, "level", cfg), fit_member(past, "ratio", cfg)
    t_pred = np.clip(np.expm1(blend(member_predict(t_level, "level", future, FEATURES),
                                    member_predict(t_ratio, "ratio", future, FEATURES))), 0.0, None)
    temporal = {"cams_raw": rounded(metrics(future[OBS], np.clip(future[CAMS_PM], 0.0, None))),
                "airtrace": rounded(metrics(future[OBS], t_pred))}
    log(f"  temporal holdout done ({time.time() - t0:5.1f}s)")

    # -- final members on all rows -----------------------------------------------------------
    boosters = {m: fit_member(data, m, cfg) for m in MODEL_FILES}
    margin = conformal_margin(q10, q90, y)
    w = site_weights(data["site_id"]) if cfg.balance_sites else np.ones(len(data))
    ratio_offset_mean = float(np.average(cams_offset(data), weights=w))
    log(f"  final members trained ({time.time() - t0:5.1f}s)")

    # -- tables ------------------------------------------------------------------------------
    cv = pd.DataFrame({
        "site_id": data["site_id"], "date": data["date"], "obs": obs, "cams": preds["cams_raw"],
        "airtrace_oof": preds["airtrace"], "airtrace_low_oof": low, "airtrace_high_oof": high,
        "region": data["region"], "city": data["city"],
    })
    importance = gain_importance([boosters["level"], boosters["ratio"]])
    trained_at = datetime.now(timezone.utc).replace(microsecond=0)
    period = [f"{data['date'].min():%Y-%m-%d}", f"{data['date'].max():%Y-%m-%d}"]
    scheme = (f"Leave-location-out cross-validation: GroupKFold({len(folds)}) over "
              f"{data[GROUP].nunique()} one-degree lat/lon cells ({data['site_id'].nunique()} monitors), so every "
              f"prediction is for a location the model never saw. Metrics on out-of-fold daily means in µg/m³.")
    temporal_scheme = (f"Forward in time: train on all sites before {TEMPORAL_SPLIT} ({len(past):,} days), "
                       f"test on {TEMPORAL_SPLIT} to {period[1]} ({len(future):,} days).")

    metrics_json = {
        "generated_at": trained_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": _dataset_summary(data, period),
        "validation": {"scheme": scheme, "temporal_scheme": temporal_scheme},
        "models": [{"key": k, "name": MODEL_NAMES[k], **rounded(model_metrics[k])} for k in MODEL_NAMES],
        "headline": _headline(model_metrics["cams_raw"], model_metrics["airtrace"]),
        "temporal_holdout": temporal,
        "by_region": _by_region(cv, data),
        "scatter": _scatter(cv),
        "confusion": {"labels": list(AQI_CATEGORIES), "cams": confusion(obs, preds["cams_raw"]),
                      "airtrace": confusion(obs, preds["airtrace"])},
        "feature_importance": [{"feature": f, "label": FEATURE_LABELS.get(f, f), "group": FEATURE_GROUP[f],
                                "importance": round(float(v), 3)} for f, v in importance.head(15).items()],
        "group_importance": [{"group": g, "importance": round(float(v), 3)} for g, v in
                             importance.groupby(importance.index.map(FEATURE_GROUP)).sum()
                             .sort_values(ascending=False).items()],
        "interval": {"nominal": INTERVAL_NOMINAL, "coverage": round(coverage, 3),
                     "mean_width": round(float(np.mean(high - low)), 1)},
        "per_site": _per_site(cv, data),
    }
    meta = {
        "version": f"{trained_at:%Y.%m.%d.%H%M}",
        "trained_at": metrics_json["generated_at"],
        "features": list(FEATURES),
        "target": TARGET,
        "n_rows": int(len(data)),
        "n_sites": int(data["site_id"].nunique()),
        "n_countries": int(data["country"].nunique()),
        "train_period": period,
        "files": dict(MODEL_FILES),
        "point_forecast": "mean of the level and ratio members in log1p space; ratio member = "
                          "log1p(cams_pm2_5_mean) + booster output; level member alone where CAMS is missing",
        "ratio_offset_mean": round(ratio_offset_mean, 6),
        "interval": {"nominal": INTERVAL_NOMINAL, "method": "LightGBM quantile members (0.1, 0.9) + "
                     "leave-location-out conformal margin (CQR)", "conformal_margin_log": round(margin, 6),
                     "cv_fold_margins_log": [round(m, 4) for m in fold_margins],
                     "cv_coverage": round(coverage, 3), "raw_quantile_cv_coverage": round(raw_cov, 3)},
        "params": asdict(cfg),
        "monotone_increasing": [CAMS_PM] if cfg.monotone_cams else [],
        "sample_weights": "site-balanced (every monitor has equal total weight)" if cfg.balance_sites else "none",
        "elevation_source": ELEVATION_SOURCE,
        "cv_scheme": scheme,
        "cv_metrics": {k: rounded(v) for k, v in model_metrics.items()},
        "experiments": experiments,
    }
    log(f"done in {time.time() - t0:.1f}s")
    return TrainingResult(boosters=boosters, meta=meta, metrics=metrics_json, cv=cv, experiments=experiments)


def _dataset_summary(data: pd.DataFrame, period: list[str]) -> dict:
    regions = (data.groupby("region").agg(n_sites=("site_id", "nunique"), n_rows=(OBS, "size"))
               .sort_values(["n_rows"], ascending=False))
    networks = data.groupby("network")["site_id"].nunique().sort_values(ascending=False)
    return {"n_sites": int(data["site_id"].nunique()), "n_countries": int(data["country"].nunique()),
            "n_rows": int(len(data)), "period": period,
            "regions": [{"region": r, "n_sites": int(v.n_sites), "n_rows": int(v.n_rows)}
                        for r, v in regions.iterrows()],
            "networks": {k: int(v) for k, v in networks.items()}}


def _headline(cams: dict, at: dict) -> dict:
    def r(v: float | None, nd: int = 3) -> float:
        return round(float(v or 0.0), nd)
    return {
        "mae_reduction_pct": r(100.0 * (cams["mae"] - at["mae"]) / cams["mae"], 1),
        "rmse_reduction_pct": r(100.0 * (cams["rmse"] - at["rmse"]) / cams["rmse"], 1),
        "aqi_acc_cams": r(cams["aqi_category_accuracy"]),
        "aqi_acc_airtrace": r(at["aqi_category_accuracy"]),
        "unhealthy_recall_cams": r(cams["unhealthy_recall"]),
        "unhealthy_recall_airtrace": r(at["unhealthy_recall"]),
        "r2_cams": r(cams["r2"]),
        "r2_airtrace": r(at["r2"]),
    }


def _by_region(cv: pd.DataFrame, data: pd.DataFrame) -> list[dict]:
    out = []
    for region, g in sorted(cv.groupby("region"), key=lambda kv: -len(kv[1])):
        mc, ma = metrics(g["obs"], g["cams"]), metrics(g["obs"], g["airtrace_oof"])
        out.append(rounded({"region": region, "n_sites": int(g["site_id"].nunique()),
                            "rmse_cams": mc["rmse"], "rmse_airtrace": ma["rmse"],
                            "mae_cams": mc["mae"], "mae_airtrace": ma["mae"],
                            "bias_cams": mc["bias"], "bias_airtrace": ma["bias"]}, default=2))
    return out


def _scatter(cv: pd.DataFrame, n: int = SCATTER_MAX) -> dict:
    """Region-stratified random sample of out-of-fold points (fixed seed)."""
    idx = cv.index
    if len(cv) > n:
        idx, _ = train_test_split(cv.index, train_size=n, stratify=cv["region"], random_state=SEED)
    s = cv.loc[np.sort(np.asarray(idx))]
    return {"obs": s["obs"].round(1).tolist(), "cams": s["cams"].round(1).tolist(),
            "airtrace": s["airtrace_oof"].round(1).tolist(), "region": s["region"].tolist()}


def _per_site(cv: pd.DataFrame, data: pd.DataFrame) -> list[dict]:
    info = data.drop_duplicates("site_id").set_index("site_id")
    rows = []
    for site_id, g in cv.groupby("site_id"):
        s = info.loc[site_id]
        rows.append({"site_id": site_id, "city": s["city"], "country": s["country_name"], "region": s["region"],
                     "lat": round(float(s["lat"]), 4), "lon": round(float(s["lon"]), 4), "n": int(len(g)),
                     "mean_obs": round(float(g["obs"].mean()), 1),
                     "mae_cams": round(float(np.mean(np.abs(g["cams"] - g["obs"]))), 2),
                     "mae_airtrace": round(float(np.mean(np.abs(g["airtrace_oof"] - g["obs"]))), 2)})
    return sorted(rows, key=lambda r: -r["mean_obs"])


def write_outputs(result: TrainingResult, model_dir: Path | str = MODELS,
                  cv_path: Path | str | None = None, metrics_path: Path | str | None = None) -> None:
    """Persist models, model_meta.json, cv_predictions.parquet and docs/data/metrics.json."""
    save_models(result.boosters, result.meta, model_dir)
    cv_path = Path(cv_path or PROCESSED / "cv_predictions.parquet")
    cv = result.cv.copy()
    for c in ("obs", "cams", "airtrace_oof", "airtrace_low_oof", "airtrace_high_oof"):
        cv[c] = cv[c].round(2)
    tmp = cv_path.with_name(cv_path.name + ".tmp")
    cv.to_parquet(tmp, index=False)
    os.replace(tmp, cv_path)
    _write_json(Path(metrics_path or SITE_DATA / "metrics.json"), result.metrics)
