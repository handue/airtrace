"""Fill the Mar-Jun 2024 boundary-layer-height gap of the best-match archive with ERA5."""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from airtrace.config import CONFIG  # noqa: E402
from airtrace.openmeteo import era5_blh_patch  # noqa: E402

sites = pd.read_csv(CONFIG / "training_sites.csv")
with ThreadPoolExecutor(max_workers=4) as ex:
    list(ex.map(lambda r: era5_blh_patch(float(r[0]), float(r[1])), zip(sites["lat"], sites["lon"])))
print(f"BLH patch done for {len(sites)} sites")
