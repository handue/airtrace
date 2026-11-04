"""Receptor-model source attribution of PM2.5 ("where does the pollution come from").

Method (v2, fitted offline by ``scripts/06_fit_sources.py``, applied live by :meth:`SourceModel.shares`)
-- a chemical-mass-balance (CMB) receptor model with fixed source profiles:

1. **Tracers.** Eight daily-mean CAMS species -- NO2, NO, CO, SO2, O3, formaldehyde (HCHO),
   desert dust and sea salt -- are divided by a robust global scale (their 90th percentile over
   the training rows) so every species counts on a comparable footing.
2. **Fixed source profiles.** Each source has a fixed reference tracer signature
   (:data:`REFERENCE_SIGNATURES`, written from emission chemistry and never tuned to the data):
   traffic = NO2 + NO + some CO, industry/coal = SO2 + some NO2/CO, burning = CO + HCHO with little
   NO2, secondary = O3 + some HCHO, dust = dust, marine = sea salt. A day's activations are the
   non-negative least-squares (NNLS) solution of ``x ~= w @ H`` for its scaled tracers ``x``.
   (v1 learnt the profiles by unconstrained NMF; that let one "industry" factor absorb NO2, CO and
   SO2 together, leaving traffic near zero even in New Delhi.)
3. **Mass yields, stage 1.** CAMS PM2.5 is regressed on the activations by NNLS on within-site
   anomalies (site fixed effects). CAMS computes its PM2.5 from the same emissions that drive its
   tracer fields, so this identifies how much PM2.5 (ug/m3) each unit of each signature carries;
   using day-to-day variation inside each site removes between-site confounding (e.g. clean,
   salty coastal air), which otherwise forces the sea-salt yield to zero.
4. **Calibration to observations, stage 2.** Observed PM2.5 at the reference monitors is
   regressed by NNLS on ``[stage-1 modelled PM2.5, 1]``: ``obs ~= alpha * model + beta``. The final
   yields are ``alpha * stage-1 yields`` and ``beta`` (plus ``alpha * stage-1 intercept``) is the
   "regional background & other" -- PM2.5 that none of the tracer signatures explains.

``coef_j * W_ij`` is then the PM2.5 (ug/m3) attributed to source ``j`` on day ``i``. Training and
serving use the same NNLS transform onto the same fixed profiles, so they cannot drift apart. When a
better total PM2.5 is known (the AirTrace forecast), :meth:`SourceModel.shares` can rescale the
modelled contributions to it with the ``pm25`` argument.

Typical use::

    from airtrace.sources import load_source_model
    model = load_source_model()              # models/sources_nmf.json
    model.shares(daily)                      # daily: DataFrame with features.TRACER_COLUMNS
"""
from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, nnls

from .config import MODELS

MODEL_FILE = "sources_nmf.json"
FORMAT_VERSION = 1

# (daily feature column from features.TRACER_COLUMNS, short chemical name), in model order.
TRACERS: list[tuple[str, str]] = [
    ("cams_nitrogen_dioxide_mean", "NO2"),
    ("cams_nitrogen_monoxide_mean", "NO"),
    ("cams_carbon_monoxide_mean", "CO"),
    ("cams_sulphur_dioxide_mean", "SO2"),
    ("cams_ozone_mean", "O3"),
    ("cams_formaldehyde_mean", "HCHO"),
    ("cams_dust_mean", "dust"),
    ("cams_sea_salt_aerosol_mean", "sea_salt"),
]
TRACER_COLS = [col for col, _ in TRACERS]
TRACER_SHORT = [short for _, short in TRACERS]

FACTOR_KEYS = ["traffic", "industry", "burning", "dust", "secondary", "marine"]
BACKGROUND_KEY = "background"
SOURCE_KEYS = FACTOR_KEYS + [BACKGROUND_KEY]

# Display metadata shared with the website (keys, labels and colours are fixed in DATA_CONTRACT.md).
SOURCES: dict[str, dict[str, str]] = {
    "traffic": {
        "label": "Traffic & urban combustion", "color": "#3b82f6",
        "description": "Nitrogen oxides (NO2 and freshly emitted NO) with some carbon monoxide: the "
                       "signature of vehicle exhaust and other street-level fuel burning, including the "
                       "secondary nitrate that forms from it.",
    },
    "industry": {
        "label": "Industry, coal & power", "color": "#64748b",
        "description": "Sulphur dioxide with some NO2 and carbon monoxide: the signature of coal, oil, "
                       "power plants and industry, including the secondary sulphate that forms from it.",
    },
    "burning": {
        "label": "Fires & biomass burning", "color": "#f97316",
        "description": "Carbon monoxide and formaldehyde with little NO2: smoke from crop, forest and "
                       "savanna fires and household wood and dung burning.",
    },
    "dust": {
        "label": "Desert dust", "color": "#c8a165",
        "description": "Mineral dust blown from deserts and dry land such as the Sahara, Arabia and the "
                       "Thar, often carried hundreds of kilometres.",
    },
    "secondary": {
        "label": "Photochemical & secondary", "color": "#a855f7",
        "description": "Ozone with some formaldehyde: chemically aged air in which sunlight turns gases "
                       "into secondary organic and inorganic particles, a regional haze found almost "
                       "everywhere.",
    },
    "marine": {
        "label": "Sea salt (marine)", "color": "#06b6d4",
        "description": "Sea-salt spray from breaking ocean waves, carried inland by onshore winds at "
                       "coastal sites.",
    },
    BACKGROUND_KEY: {
        "label": "Regional background & other", "color": "#cbd5e1",
        "description": "PM2.5 that none of the tracer signatures explains: long-range background, local "
                       "sources the global CAMS model does not resolve, and the part of the measured "
                       "level that the tracers cannot account for.",
    },
}

# Reference chemical signatures (relative weight per scaled tracer, missing = 0). Since v2 they are
# the *fixed* source profiles of the chemical-mass-balance model (max-normalised by
# :func:`reference_profiles`). They predate the v2 fit (v1 used them to name its NMF factors) and
# were not adjusted to the data.
REFERENCE_SIGNATURES: dict[str, dict[str, float]] = {
    "traffic": {"NO2": 1.0, "NO": 1.0, "CO": 0.5},
    "industry": {"SO2": 1.0, "NO2": 0.4, "CO": 0.4},
    "burning": {"CO": 1.0, "HCHO": 1.0, "NO2": 0.1},
    "secondary": {"O3": 1.0, "HCHO": 0.5},
    "dust": {"dust": 1.0},
    "marine": {"sea_salt": 1.0},
}


# --------------------------------------------------------------------------------------------
# Building blocks (shared by the fit script and the serving model)
# --------------------------------------------------------------------------------------------
def tracer_matrix(daily: pd.DataFrame, columns: list[str] = TRACER_COLS) -> np.ndarray:
    """Tracer columns of a daily table as a float array (NaN preserved)."""
    missing = [c for c in columns if c not in daily.columns]
    if missing:
        raise KeyError(f"daily table lacks tracer columns: {missing}")
    return daily[columns].to_numpy(dtype=float)


def complete_rows(X: np.ndarray) -> np.ndarray:
    """Boolean mask of rows where every tracer is finite."""
    return np.isfinite(X).all(axis=1)


def robust_scale(X: np.ndarray, q: float = 90.0) -> np.ndarray:
    """Per-tracer global scale: the ``q``-th percentile over complete rows."""
    scale = np.nanpercentile(X, q, axis=0)
    if not np.all(np.isfinite(scale) & (scale > 0)):
        raise ValueError(f"degenerate tracer scale: {scale}")
    return scale


def max_normalise(H: np.ndarray) -> np.ndarray:
    """Divide every factor profile (row) by its largest loading."""
    peak = H.max(axis=1, keepdims=True)
    return H / np.where(peak > 0, peak, 1.0)


def signature_matrix(keys: list[str] = FACTOR_KEYS,
                     tracer_short: list[str] = TRACER_SHORT,
                     signatures: dict[str, dict[str, float]] = REFERENCE_SIGNATURES) -> np.ndarray:
    """Reference signatures as a (len(keys), n_tracers) array."""
    return np.array([[signatures[k].get(s, 0.0) for s in tracer_short] for k in keys])


def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarity between the rows of ``A`` and the rows of ``B``."""
    An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    Bn = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return An @ Bn.T


def label_factors(H: np.ndarray, keys: list[str] = FACTOR_KEYS,
                  tracer_short: list[str] = TRACER_SHORT,
                  signatures: dict[str, dict[str, float]] = REFERENCE_SIGNATURES
                  ) -> tuple[list[str], np.ndarray]:
    """Name factor profiles by maximum-total-cosine matching to the reference signatures.

    The Hungarian algorithm gives every signature at most one factor. If there are more factors
    than signatures, each left-over factor joins the signature it resembles most (its contributions
    are then summed under that key). Returns (key per factor row, cosine matrix factors x keys).
    """
    cos = cosine_matrix(max_normalise(H), signature_matrix(keys, tracer_short, signatures))
    rows, cols = linear_sum_assignment(-cos)
    labels: list[str | None] = [None] * H.shape[0]
    for r, c in zip(rows, cols):
        labels[r] = keys[c]
    for r, lab in enumerate(labels):
        if lab is None:
            labels[r] = keys[int(np.argmax(cos[r]))]
    return [str(lab) for lab in labels], cos


def nnls_activations(Xs: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Row-wise non-negative projection ``argmin_{w>=0} ||w @ H - x||`` of scaled tracer rows."""
    A = H.T
    W = np.zeros((Xs.shape[0], H.shape[0]))
    for i, x in enumerate(Xs):
        W[i], _ = nnls(A, x)
    return W


def reference_profiles(keys: list[str] = FACTOR_KEYS) -> np.ndarray:
    """Fixed CMB source profiles: the max-normalised reference signatures, (len(keys), n_tracers)."""
    return max_normalise(signature_matrix(keys))


def fit_pm_regression(W: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """NNLS of PM2.5 on [activations, 1]: returns (ug/m3 per unit activation, intercept ug/m3)."""
    A = np.column_stack([W, np.ones(len(W))])
    sol, _ = nnls(A, y)
    return sol[:-1], float(sol[-1])


def fit_within_site_yields(W: np.ndarray, y: np.ndarray, site: np.ndarray) -> tuple[np.ndarray, float]:
    """NNLS yields from within-site anomalies (site fixed effects), intercept from the grand means.

    ``y - mean_site(y) ~= (W - mean_site(W)) @ coef`` with ``coef >= 0``; the intercept is
    ``max(mean(y) - mean(W) @ coef, 0)``. Returns (ug/m3 per unit activation, intercept ug/m3).
    """
    W_dev = W - pd.DataFrame(W).groupby(site).transform("mean").to_numpy()
    y_dev = y - pd.Series(y).groupby(site).transform("mean").to_numpy()
    coef, _ = nnls(W_dev, y_dev)
    return coef, max(float(np.mean(y) - W.mean(axis=0) @ coef), 0.0)


def fit_two_stage_yields(W: np.ndarray, cams_pm25: np.ndarray, site: np.ndarray, W_obs: np.ndarray,
                         obs_pm25: np.ndarray) -> dict[str, Any]:
    """Stage 1: within-site NNLS of CAMS PM2.5 on activations; stage 2: NNLS calibration
    ``obs ~= alpha * stage1 + beta`` on the rows with observations (``W_obs``, ``obs_pm25``).
    Returns the final ``coef`` and ``intercept`` plus the stage-wise parameters."""
    c1, i1 = fit_within_site_yields(W, cams_pm25, site)
    (alpha,), beta = fit_pm_regression((W_obs @ c1 + i1)[:, None], obs_pm25)
    return {"coef": alpha * c1, "intercept": alpha * i1 + beta, "stage1_coef": c1, "stage1_intercept": i1,
            "alpha": float(alpha), "beta": float(beta),
            "stage1_r2_cams": r2_score(cams_pm25, W @ c1 + i1)}


def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def round_shares(fractions: dict[str, float], decimals: int = 3) -> dict[str, float]:
    """Round fractions that sum to 1 so the rounded values still sum to exactly 1
    (largest-remainder method)."""
    unit = 10 ** decimals
    raw = {k: max(v, 0.0) * unit for k, v in fractions.items()}
    floors = {k: math.floor(v) for k, v in raw.items()}
    short = unit - sum(floors.values())
    for k in sorted(raw, key=lambda k: raw[k] - floors[k], reverse=True)[:max(short, 0)]:
        floors[k] += 1
    return {k: v / unit for k, v in floors.items()}


# --------------------------------------------------------------------------------------------
# Serving model
# --------------------------------------------------------------------------------------------
@dataclass
class SourceModel:
    """Fitted receptor model: tracer scales, NMF profiles, factor names and PM2.5 regression."""

    tracer_columns: list[str]
    scale: np.ndarray                 # (n_tracers,) divisor applied to raw ug/m3 tracers
    components: np.ndarray            # (k, n_tracers) factor profiles in scaled units
    factor_keys: list[str]            # (k,) source key of every component
    coef: np.ndarray                  # (k,) ug/m3 PM2.5 per unit activation
    intercept: float                  # ug/m3, regional background & other
    sources: dict[str, dict[str, str]] = field(default_factory=lambda: copy.deepcopy(SOURCES))
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- transforms -----------------------------------------------------------------------
    def _complete(self, daily: pd.DataFrame) -> tuple[pd.Index, np.ndarray]:
        X = tracer_matrix(daily, self.tracer_columns)
        ok = complete_rows(X)
        return daily.index[ok], np.clip(X[ok], 0.0, None) / self.scale

    def activations(self, daily: pd.DataFrame) -> pd.DataFrame:
        """Factor activations of every complete row (rows with a missing tracer are dropped)."""
        index, Xs = self._complete(daily)
        W = nnls_activations(Xs, self.components)
        return pd.DataFrame(W, index=index, columns=[f"f{j}_{k}" for j, k in enumerate(self.factor_keys)])

    def contributions(self, daily: pd.DataFrame) -> pd.DataFrame:
        """Modelled PM2.5 (ug/m3) per source key incl. background, one row per complete day."""
        index, Xs = self._complete(daily)
        parts = nnls_activations(Xs, self.components) * self.coef
        out = pd.DataFrame(0.0, index=index, columns=SOURCE_KEYS)
        for j, key in enumerate(self.factor_keys):
            out[key] += parts[:, j]
        out[BACKGROUND_KEY] = self.intercept
        return out

    # ---- public API -----------------------------------------------------------------------
    @staticmethod
    def _rescale(contrib: pd.DataFrame, pm25: pd.Series | float | int | None,
                 daily: pd.DataFrame) -> tuple[pd.DataFrame, float | None]:
        """Scale modelled contributions to a known PM2.5 (see :meth:`shares`)."""
        if pm25 is None:
            return contrib, None
        if isinstance(pm25, pd.Series):
            p = None if pm25.index.has_duplicates else pm25.reindex(contrib.index)
            if (p is None or p.isna().all()) and len(pm25) == len(daily):   # positional alignment
                p = pd.Series(pm25.to_numpy(), index=daily.index)[daily.index.isin(contrib.index)]
            if p is None:
                return contrib, None
            p = pd.to_numeric(p, errors="coerce").clip(lower=0.0)
            total = contrib.sum(axis=1)
            ok = p.notna().to_numpy() & (total > 0).to_numpy()
            if not ok.any():
                return contrib, None
            contrib, p, total = contrib[ok], p[ok], total[ok]
            # each day keeps its modelled source mix but is weighted by its known PM2.5
            scaled = contrib.mul((p.to_numpy() / total.to_numpy()), axis=0)
            return scaled, float(scaled.sum(axis=1).mean())
        value = float(pm25)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"pm25 total must be a finite non-negative number, got {pm25!r}")
        return contrib, value

    def shares(self, daily: pd.DataFrame, pm25: pd.Series | float | None = None) -> dict[str, Any]:
        """Source shares of PM2.5 over the rows of ``daily`` (e.g. the days of a forecast window).

        ``daily`` needs the tracer columns of ``features.TRACER_COLUMNS``; rows with any missing
        tracer are ignored. Shares are mean modelled contributions divided by their total, rounded
        to 3 decimals so that they sum to exactly 1.

        ``pm25`` (optional) is a better-known total PM2.5 in ug/m3, e.g. the AirTrace forecast:
        either a ``pd.Series`` aligned with ``daily`` (by index, or by position if the index does
        not match) -- every day's modelled contributions, background included, are then scaled to
        that day's value, so high-PM days weigh more in the window's shares -- or a single number,
        the window's mean total, which leaves the shares unchanged. Either way each share entry
        also gets ``"pm25"`` (ug/m3 attributed to that source) and the result gets ``"pm25"`` (the
        total used).

        Returns ``{"shares": [{"key", "label", "share", "color", "description"}, ...],
        "background_share": float}``. The ``shares`` list holds all seven sources -- the six
        factors *and* the "background" entry -- sorted by share (descending), and sums to 1;
        ``background_share`` repeats the background entry for convenience.

        Raises ``ValueError`` if no row has all tracers.
        """
        contrib = self.contributions(daily)
        if contrib.empty:
            raise ValueError("no day with all source-attribution tracers present")
        contrib, total_pm = self._rescale(contrib, pm25, daily)
        mean = contrib.mean()
        total = float(mean.sum())
        if total > 0:
            fractions = {k: float(mean[k]) / total for k in SOURCE_KEYS}
        else:   # nothing modelled at all: everything is unexplained background
            fractions = {k: float(k == BACKGROUND_KEY) for k in SOURCE_KEYS}
        rounded = round_shares(fractions)
        order = sorted(SOURCE_KEYS, key=lambda k: (-rounded[k], SOURCE_KEYS.index(k)))
        items = [{"key": k, "label": self.sources[k]["label"], "share": rounded[k],
                  "color": self.sources[k]["color"], "description": self.sources[k]["description"]}
                 for k in order]
        out: dict[str, Any] = {"shares": items, "background_share": rounded[BACKGROUND_KEY]}
        if total_pm is not None:
            for item in items:
                item["pm25"] = round(fractions[item["key"]] * total_pm, 1)
            out["pm25"] = round(total_pm, 1)
        return out

    # ---- persistence ----------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        short = dict(TRACERS)
        return {
            "format_version": FORMAT_VERSION,
            **self.meta,
            "tracers": [{"column": c, "short": short.get(c, c), "scale": float(s)}
                        for c, s in zip(self.tracer_columns, self.scale)],
            "components": [[float(v) for v in row] for row in self.components],
            "factor_keys": list(self.factor_keys),
            "coef": [float(v) for v in self.coef],
            "intercept": float(self.intercept),
            "sources": self.sources,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SourceModel":
        if d.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported source-model format: {d.get('format_version')}")
        known = {"format_version", "tracers", "components", "factor_keys", "coef", "intercept", "sources"}
        return cls(
            tracer_columns=[t["column"] for t in d["tracers"]],
            scale=np.array([t["scale"] for t in d["tracers"]], dtype=float),
            components=np.array(d["components"], dtype=float),
            factor_keys=list(d["factor_keys"]),
            coef=np.array(d["coef"], dtype=float),
            intercept=float(d["intercept"]),
            sources=d.get("sources", SOURCES),
            meta={k: v for k, v in d.items() if k not in known},
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1))
        tmp.replace(path)


def default_model_path() -> Path:
    """``$AIRTRACE_MODEL_DIR/sources_nmf.json`` if the variable is set, else ``models/sources_nmf.json``."""
    return Path(os.environ.get("AIRTRACE_MODEL_DIR", MODELS)) / MODEL_FILE


def load_source_model(path: str | Path | None = None) -> SourceModel:
    """Load the fitted source model (default: :func:`default_model_path`)."""
    return SourceModel.from_dict(json.loads(Path(path or default_model_path()).read_text()))
