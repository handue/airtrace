"""Download hourly CAMS composition + ERA5 weather for every training site (cached).

Embassy sites are fetched first so the global part of the dataset is complete even if
the run is interrupted by Open-Meteo's hourly quota.
"""
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from airtrace.config import CONFIG, TRAIN_END, TRAIN_START  # noqa: E402
from airtrace.openmeteo import cams_archive, weather_archive  # noqa: E402


def main() -> None:
    sites = pd.read_csv(CONFIG / "training_sites.csv")
    sites["prio"] = (sites["network"] != "US Dept. of State").astype(int)
    sites = sites.sort_values(["prio", "n_days"], ascending=[True, False])
    jobs = []
    for _, s in sites.iterrows():
        jobs.append(("cams", s.site_id, s.lat, s.lon))
        jobs.append(("era5", s.site_id, s.lat, s.lon))

    def run(job):
        kind, sid, lat, lon = job
        fn = cams_archive if kind == "cams" else weather_archive
        fn(float(lat), float(lon), TRAIN_START, TRAIN_END)
        return job

    done = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(run, j) for j in jobs]
        for f in as_completed(futs):
            try:
                kind, sid, _, _ = f.result()
                done += 1
                if done % 10 == 0 or done == len(jobs):
                    print(f"{done}/{len(jobs)} done (last: {kind} {sid})", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"job failed: {exc}", flush=True)


if __name__ == "__main__":
    main()
