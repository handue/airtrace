"""How many reference PM2.5 monitors report to AirNow, month by month (2023-01 .. 2026-10).

For the 15th of every month we read AirNow's public ``daily_data.dat`` and count the distinct
sites that reported a 24-hour PM2.5 average - all sites, and the US Department of State
embassy/consulate monitors (data source contains "Department of State"). This documents the
collapse of the embassy network, for many cities the only public reference-grade PM2.5.

The script also records, per embassy monitor, the last date it was seen. Monthly samples only
bracket that date, so for monitors that went silent we additionally scan the daily files
between the last sample that contains the monitor and the next sample (``--no-refine`` skips
this and keeps the sampled date).

Outputs: docs/data/network_history.json, data/static/embassy_last_seen.json
Raw files are cached in data/raw/airnow/daily_YYYYMMDD.dat (shared with 01_download_airnow.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from airtrace.airnow_live import EMBASSY_LAST_SEEN, daily_url, fetch, is_embassy, read_daily  # noqa: E402
from airtrace.config import RAW_AIRNOW, SITE_DATA  # noqa: E402

FIRST_MONTH = (2023, 1)
LAST_MONTH = (2026, 10)
SAMPLE_DAY = 15
SOURCE = ("US EPA AirNow public daily_data.dat files (files.airnowtech.org): distinct sites reporting a "
          "24-hour PM2.5 average on the 15th of each month; embassy monitors = data source "
          "'U.S. Department of State'.")


def month_samples(first: tuple[int, int] = FIRST_MONTH, last: tuple[int, int] = LAST_MONTH,
                  day: int = SAMPLE_DAY) -> list[date]:
    out, (y, m) = [], first
    while (y, m) <= last:
        out.append(date(y, m, day))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def download(days: list[date], workers: int = 8) -> dict[date, Path | None]:
    def one(d: date) -> Path | None:
        return fetch(daily_url(d), RAW_AIRNOW / f"daily_{d:%Y%m%d}.dat")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return dict(zip(days, ex.map(one, days)))


def pm25_sites(path: Path) -> pd.DataFrame:
    """Distinct PM2.5 sites in one daily file with their last date and embassy flag."""
    df = read_daily(path)
    return (df.groupby("aqsid")
              .agg(date=("date", "max"), agency=("agency", "last"))
              .assign(embassy=lambda x: x["agency"].map(is_embassy)))


def refine_last_seen(last_seen: dict[str, date], samples: list[date]) -> dict[str, date]:
    """For embassy monitors that went silent between two monthly samples, find the last day
    they actually reported by reading the daily files in that gap."""
    final = samples[-1]
    gaps: dict[str, list[date]] = {}
    for sid, seen in last_seen.items():
        later = [s for s in samples if s > seen]
        if seen >= final or not later:
            continue
        gaps[sid] = [seen + timedelta(days=k) for k in range(1, (later[0] - seen).days)]
    days = sorted({d for ds in gaps.values() for d in ds})
    if not days:
        return last_seen
    print(f"refining last-seen dates of {len(gaps)} silent embassy monitors from {len(days)} daily files")
    paths = download(days)
    seen_on: dict[date, set[str]] = {}
    for d, p in paths.items():
        if p is not None:
            df = read_daily(p)
            seen_on[d] = set(df.loc[df["agency"].map(is_embassy), "aqsid"])
    out = dict(last_seen)
    for sid, ds in gaps.items():
        hits = [d for d in ds if sid in seen_on.get(d, ())]
        if hits:
            out[sid] = max(hits)
    missing = sum(p is None for p in paths.values())
    if missing:
        print(f"  warning: {missing} daily files in the gaps were unavailable")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-refine", action="store_true", help="keep monthly-sampled last-seen dates")
    args = ap.parse_args()

    samples = month_samples()
    paths = download(samples)
    months, embassy, all_sites = [], [], []
    last_seen: dict[str, date] = {}
    for d in samples:
        p = paths[d]
        if p is None:
            print(f"warning: no AirNow daily file for {d} - month skipped")
            continue
        s = pm25_sites(p)
        months.append(f"{d:%Y-%m}")
        embassy.append(int(s["embassy"].sum()))
        all_sites.append(int(len(s)))
        for sid, seen in s.loc[s["embassy"], "date"].items():
            last_seen[sid] = max(last_seen.get(sid, date.min), seen.date())
    if not months:
        sys.exit("no AirNow files available - nothing written")

    if not args.no_refine:
        last_seen = refine_last_seen(last_seen, [d for d in samples if paths[d] is not None])

    SITE_DATA.mkdir(parents=True, exist_ok=True)
    out = {"source": SOURCE, "months": months, "embassy_monitors": embassy, "all_pm25_monitors": all_sites}
    (SITE_DATA / "network_history.json").write_text(json.dumps(out, indent=1) + "\n")
    EMBASSY_LAST_SEEN.parent.mkdir(parents=True, exist_ok=True)
    EMBASSY_LAST_SEEN.write_text(json.dumps({k: f"{v:%Y-%m-%d}" for k, v in sorted(last_seen.items())},
                                            indent=1) + "\n")

    peak = max(range(len(months)), key=lambda i: embassy[i])
    print(f"{len(months)} months {months[0]}..{months[-1]}: embassy monitors peak {embassy[peak]} "
          f"({months[peak]}) -> {embassy[-1]} ({months[-1]}); all PM2.5 sites {all_sites[0]} -> {all_sites[-1]}")
    print(f"embassy monitors ever seen: {len(last_seen)}; still reporting in {months[-1]}: "
          f"{sum(v >= samples[-1] for v in last_seen.values())}")
    print(f"wrote {SITE_DATA / 'network_history.json'} and {EMBASSY_LAST_SEEN}")


if __name__ == "__main__":
    main()
