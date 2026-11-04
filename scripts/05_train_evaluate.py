"""Step 5: train and evaluate the AirTrace PM2.5 model.

* leave-location-out 5-fold CV (1-degree cells) for raw CAMS, CAMS + linear calibration,
  ML without CAMS, and AirTrace; temporal holdout; ablation table; calibrated 80 % interval;
* final members fitted on all rows.

Outputs: models/pm25_lgbm.txt, pm25_ratio.txt, pm25_q10.txt, pm25_q90.txt, model_meta.json,
         data/processed/cv_predictions.parquet, docs/data/metrics.json
Deterministic (fixed seeds, deterministic LightGBM) and idempotent; runs in about a minute.
Use --no-ablations to skip the extra ablation runs.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from airtrace.dataset import DATASET_PATH, load_dataset  # noqa: E402
from airtrace.model import train_and_evaluate, write_outputs  # noqa: E402

METRIC_COLS = ["rmse", "mae", "r2", "bias", "r", "aqi_category_accuracy",
               "unhealthy_precision", "unhealthy_recall", "unhealthy_f1"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-ablations", action="store_true", help="skip the ablation runs (faster)")
    args = ap.parse_args()
    if not DATASET_PATH.exists():
        print(f"{DATASET_PATH} missing - run scripts/04_build_dataset.py first", file=sys.stderr)
        return 1

    result = train_and_evaluate(load_dataset(), ablations=not args.no_ablations)
    write_outputs(result)

    m = result.metrics
    pd.set_option("display.width", 200)
    print("\nLeave-location-out CV (daily PM2.5, ug/m3):")
    print(pd.DataFrame(m["models"]).set_index("key")[METRIC_COLS].to_string())
    print("\nTemporal holdout:")
    print(pd.DataFrame(m["temporal_holdout"]).T[["rmse", "mae", "r2", "bias", "aqi_category_accuracy",
                                                   "unhealthy_recall"]].to_string())
    print("\nBy region:")
    print(pd.DataFrame(m["by_region"]).set_index("region").to_string())
    print("\nExperiments (out-of-fold):")
    print(pd.DataFrame(result.experiments).set_index("key").drop(columns="description").to_string())
    print(f"\nHeadline: {m['headline']}")
    print(f"Interval: {m['interval']} (conformal margin {result.meta['interval']['conformal_margin_log']:.3f} in log1p)")
    print("Top features: " + ", ".join(f"{f['feature']} {f['importance']:.3f}" for f in m["feature_importance"][:8]))
    print(f"Model {result.meta['version']}: {result.meta['n_rows']:,} rows, {result.meta['n_sites']} sites, "
          f"{result.meta['n_countries']} countries -> models/, data/processed/cv_predictions.parquet, "
          f"docs/data/metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
