"""Step 4: build the model training table.

Joins hourly CAMS + ERA5 (cache only, never downloads) aggregated to daily features with the
AirNow 24-h PM2.5 observations for every training site whose Open-Meteo files are on disk.
Idempotent: re-run it after the download finishes to pick up the remaining sites.

Output: data/processed/dataset.parquet
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from airtrace.dataset import DATASET_PATH, build_dataset, save_dataset, summarize  # noqa: E402
from airtrace.features import FEATURES  # noqa: E402


def main() -> int:
    t0 = time.time()
    data, report = build_dataset(verbose=True)
    n_cfg = len(report)
    built = report[report["n_rows"] > 0]
    print(f"sites with data: {len(built)}/{n_cfg} configured "
          f"({n_cfg - len(built)} skipped: {report.loc[report['n_rows'] == 0, 'status'].value_counts().to_dict()})")
    if data.empty:
        print("no cached site has data yet - nothing written", file=sys.stderr)
        return 1

    save_dataset(data)
    patched = built[built["blh_patch"]]
    print(f"BLH patch applied for {len(patched)}/{len(built)} sites "
          f"({int(patched['blh_filled_hours'].sum()):,} hours filled)")
    missing = data[FEATURES].isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    if len(missing):
        print("feature missing rates: " + ", ".join(f"{k} {v:.1%}" for k, v in missing.items()))
    print(f"period {data['date'].min():%Y-%m-%d} .. {data['date'].max():%Y-%m-%d}")
    print(summarize(data).to_string())
    print(f"wrote {DATASET_PATH.relative_to(DATASET_PATH.parents[2])} "
          f"({len(data):,} rows x {data.shape[1]} cols) in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
