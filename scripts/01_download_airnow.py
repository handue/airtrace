"""Download AirNow daily_data.dat files (public, keyless) for the training window
and parse PM2.5 24-hour averages into a tidy parquet table.

Output: data/processed/airnow_pm25_daily.parquet
        data/processed/airnow_sites.parquet
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from airtrace.config import AIRNOW_BASE, PROCESSED, RAW_AIRNOW, TRAIN_END, TRAIN_START  # noqa: E402

DAILY_COLS = ["date", "aqsid", "site_name", "parameter", "unit", "value", "hours", "agency"]
SITE_COLS = [
    "aqsid", "parameter", "site_code", "site_name", "status", "agency_id", "agency_name",
    "epa_region", "lat", "lon", "elevation", "gmt_offset", "country", "f14", "f15",
    "msa_code", "msa_name", "state_code", "state_name", "county_code", "county_name", "f22", "f23",
]


def fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)
            return dest
        except Exception as exc:  # noqa: BLE001
            if attempt == 3:
                print(f"FAILED {url}: {exc}")
    return dest


def main() -> None:
    RAW_AIRNOW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    days = pd.date_range(TRAIN_START, TRAIN_END, freq="D")
    jobs = []
    for d in days:
        ds = d.strftime("%Y%m%d")
        jobs.append((f"{AIRNOW_BASE}/{d.year}/{ds}/daily_data.dat", RAW_AIRNOW / f"daily_{ds}.dat"))
    # Site metadata snapshots from inside the window (embassy sites that later closed are kept).
    for ds in ["20240715", "20250115"]:
        jobs.append((f"{AIRNOW_BASE}/{ds[:4]}/{ds}/monitoring_site_locations.dat", RAW_AIRNOW / f"sites_{ds}.dat"))
    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(lambda j: fetch(*j), jobs))

    frames = []
    for d in days:
        f = RAW_AIRNOW / f"daily_{d.strftime('%Y%m%d')}.dat"
        if not f.exists():
            continue
        df = pd.read_csv(f, sep="|", header=None, names=DAILY_COLS, dtype=str, encoding="latin-1",
                         on_bad_lines="skip")
        df = df[df["parameter"] == "PM2.5-24hr"]
        frames.append(df)
    daily = pd.concat(frames, ignore_index=True)
    daily["date"] = pd.to_datetime(daily["date"], format="%m/%d/%y")
    daily["value"] = pd.to_numeric(daily["value"], errors="coerce")
    daily["hours"] = pd.to_numeric(daily["hours"], errors="coerce")
    daily = daily[(daily["value"] >= 0) & (daily["value"] < 1000) & (daily["hours"] >= 18)]
    daily = daily.rename(columns={"value": "pm25_obs"})[["aqsid", "date", "pm25_obs", "hours", "agency", "site_name"]]
    daily = daily.drop_duplicates(["aqsid", "date"])
    daily.to_parquet(PROCESSED / "airnow_pm25_daily.parquet", index=False)

    sites = []
    for ds in ["20240715", "20250115"]:
        f = RAW_AIRNOW / f"sites_{ds}.dat"
        s = pd.read_csv(f, sep="|", header=None, names=SITE_COLS, dtype=str, encoding="latin-1",
                        on_bad_lines="skip")
        sites.append(s[s["parameter"] == "PM2.5"])
    sites = pd.concat(sites, ignore_index=True).drop_duplicates("aqsid", keep="last")
    for c in ["lat", "lon", "elevation", "gmt_offset"]:
        sites[c] = pd.to_numeric(sites[c], errors="coerce")
    sites = sites[["aqsid", "site_name", "status", "agency_name", "lat", "lon", "elevation",
                   "gmt_offset", "country", "msa_name", "state_name"]]
    sites.to_parquet(PROCESSED / "airnow_sites.parquet", index=False)
    print(f"daily rows={len(daily):,} sites_with_data={daily['aqsid'].nunique():,} meta_sites={len(sites):,}")


if __name__ == "__main__":
    main()
