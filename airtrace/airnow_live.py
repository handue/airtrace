"""Recent AirNow observations and reference-monitor status near a city (keyless public files).

AirNow publishes, for every date, ``daily_data.dat`` (24-hour averages per site and
parameter for that local calendar day) and ``monitoring_site_locations.dat`` (site
metadata). The live build uses them to

* tell honestly whether a city has a working reference PM2.5 monitor today
  (``active``), had a US State Department embassy monitor that stopped reporting
  (``offline``), or has no monitor in AirNow (``none``), and
* show the last few observed daily means next to the CAMS and AirTrace hindcasts.

The file helpers here are also used by ``scripts/09_network_history.py``.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import AIRNOW_BASE, CONFIG, DATA, RAW

DAILY_COLS = ["date", "aqsid", "site_name", "parameter", "unit", "value", "hours", "agency"]
SITE_COLS = [
    "aqsid", "parameter", "site_code", "site_name", "status", "agency_id", "agency_name",
    "epa_region", "lat", "lon", "elevation", "gmt_offset", "country", "f14", "f15",
    "msa_code", "msa_name", "state_code", "state_name", "county_code", "county_name", "f22", "f23",
]
PM25_DAILY = "PM2.5-24hr"
EMBASSY_AGENCY = "Department of State"
EMBASSY_NETWORK = "US Dept. of State"

LIVE_DIR = RAW / "airnow_live"                      # rolling cache of recent files
STATIC_DIR = DATA / "static"
EMBASSY_LAST_SEEN = STATIC_DIR / "embassy_last_seen.json"

N_DAILY_FILES = 8          # most recent daily_data.dat files to read
ACTIVE_DAYS = 7            # "active" = reported PM2.5-24hr within this many days
MONITOR_RADIUS_KM = 25.0   # a non-embassy monitor must be this close to the city centre
FRESH_DAYS = 2             # files for the last FRESH_DAYS dates are still being updated,
FRESH_MAX_AGE_S = 3600     # so their cached copies are refreshed after this many seconds


# ---------------------------------------------------------------------------- helpers
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (scalars or numpy arrays)."""
    p = np.pi / 180.0
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def daily_url(d: date) -> str:
    return f"{AIRNOW_BASE}/{d:%Y}/{d:%Y%m%d}/daily_data.dat"


def fetch(url: str, dest: Path, max_age_s: float | None = None, min_bytes: int = 1000,
          retries: int = 4, timeout: float = 60.0) -> Path | None:
    """Download ``url`` to ``dest`` unless a usable copy already exists.

    A cached copy is reused when it has at least ``min_bytes`` and (if ``max_age_s`` is
    given) is younger than ``max_age_s`` seconds. Returns the path, or ``None`` when the
    file does not exist upstream (HTTP 403/404) or every attempt failed; if a refresh
    fails, an older cached copy is returned instead.
    """
    if dest.exists() and dest.stat().st_size >= min_bytes:
        if max_age_s is None or time.time() - dest.stat().st_mtime < max_age_s:
            return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code in (403, 404):
                break
            r.raise_for_status()
            if len(r.content) < min_bytes:
                break
            tmp = dest.with_name(dest.name + ".part")
            tmp.write_bytes(r.content)
            tmp.replace(dest)                      # atomic: readers never see half a file
            return dest
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    return dest if dest.exists() and dest.stat().st_size >= min_bytes else None


def read_daily(path: Path, parameter: str | None = PM25_DAILY) -> pd.DataFrame:
    """Parse one ``daily_data.dat`` -> columns date (datetime64), aqsid, site_name, parameter,
    value (float), agency. The 7th field is the averaging period (always 24 for
    PM2.5-24hr), not a count of valid hours, so it is dropped."""
    df = pd.read_csv(path, sep="|", header=None, names=DAILY_COLS, dtype=str, encoding="latin-1",
                     on_bad_lines="skip")
    if parameter is not None:
        df = df[df["parameter"] == parameter]
    df = df.assign(date=pd.to_datetime(df["date"], format="%m/%d/%y", errors="coerce"),
                   value=pd.to_numeric(df["value"], errors="coerce"))
    df = df.dropna(subset=["date", "aqsid", "value"])
    return df[["date", "aqsid", "site_name", "parameter", "value", "agency"]].reset_index(drop=True)


def read_sites(path: Path, parameter: str = "PM2.5") -> pd.DataFrame:
    """Parse ``monitoring_site_locations.dat`` -> one row per site measuring ``parameter``."""
    s = pd.read_csv(path, sep="|", header=None, names=SITE_COLS, dtype=str, encoding="latin-1",
                    on_bad_lines="skip")
    s = s[s["parameter"] == parameter].copy()
    for c in ("lat", "lon"):
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s = s.dropna(subset=["lat", "lon"])
    s = s[(s["lat"].abs() > 1e-6) | (s["lon"].abs() > 1e-6)]          # (0, 0) = missing coordinates
    return s.drop_duplicates("aqsid", keep="last")[
        ["aqsid", "site_name", "status", "agency_name", "lat", "lon", "country"]].reset_index(drop=True)


def is_embassy(agency) -> bool:
    return isinstance(agency, str) and EMBASSY_AGENCY in agency


def load_embassy_last_seen(path: Path = EMBASSY_LAST_SEEN) -> dict[str, str]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def load_training_sites() -> pd.DataFrame:
    """Training monitors (site_id index; city, lat, lon) - names/locates offline embassy sites."""
    try:
        t = pd.read_csv(CONFIG / "training_sites.csv", dtype={"site_id": str})
        return t.set_index("site_id")[["city", "lat", "lon"]]
    except (OSError, KeyError, ValueError):
        return pd.DataFrame(columns=["city", "lat", "lon"])


# ---------------------------------------------------------------------------- live state
class AirNowLive:
    """Recent AirNow PM2.5 daily means + current site list, downloaded once per run."""

    def __init__(self, today_utc: date | None = None, n_files: int = N_DAILY_FILES,
                 cache_dir: Path = LIVE_DIR, workers: int = 8) -> None:
        self.today_utc = today_utc or datetime.now(timezone.utc).date()
        candidates = [self.today_utc - timedelta(days=k) for k in range(n_files + 1)]

        def get(d: date) -> tuple[date, Path | None]:
            fresh = (self.today_utc - d).days < FRESH_DAYS
            return d, fetch(daily_url(d), cache_dir / f"daily_{d:%Y%m%d}.dat",
                            max_age_s=FRESH_MAX_AGE_S if fresh else None)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            got = sorted(((d, p) for d, p in ex.map(get, candidates) if p is not None), reverse=True)
        self.files = got[:n_files]
        if not self.files:
            raise RuntimeError("no AirNow daily_data.dat file could be downloaded")
        obs = pd.concat([read_daily(p) for _, p in self.files], ignore_index=True)
        # A date can appear in two files (late updates): keep the row from the newest file.
        self.obs = obs.drop_duplicates(["aqsid", "date"], keep="first").sort_values(["aqsid", "date"])

        sites_path = fetch(f"{AIRNOW_BASE}/today/monitoring_site_locations.dat",
                           cache_dir / f"sites_{self.today_utc:%Y%m%d}.dat")
        if sites_path is None:                        # fall back to the newest cached site list
            cached = sorted(cache_dir.glob("sites_*.dat"))
            sites_path = cached[-1] if cached else None
        self.sites = read_sites(sites_path) if sites_path is not None else pd.DataFrame(
            columns=["aqsid", "site_name", "status", "agency_name", "lat", "lon", "country"])
        self.site_by_id = self.sites.set_index("aqsid")

        cutoff = pd.Timestamp(self.today_utc - timedelta(days=ACTIVE_DAYS))
        recent = self.obs[self.obs["date"] >= cutoff]
        self.last_obs: pd.Series = recent.groupby("aqsid")["date"].max()   # active site -> last date
        self.agency: pd.Series = recent.groupby("aqsid")["agency"].last()
        self.embassy_last_seen = load_embassy_last_seen()
        self.training = load_training_sites()

    # ------------------------------------------------------------------ queries
    @property
    def dates(self) -> list[str]:
        """Dates (YYYY-MM-DD) of the daily files that were read, newest first."""
        return [f"{d:%Y-%m-%d}" for d, _ in self.files]

    def recent_obs(self, site_id: str) -> dict[str, float]:
        """Observed 24-h mean PM2.5 by local date (YYYY-MM-DD) for one site."""
        rows = self.obs[self.obs["aqsid"] == site_id]
        return {f"{d:%Y-%m-%d}": round(float(v), 1) for d, v in zip(rows["date"], rows["value"])}

    def _site_info(self, site_id: str) -> tuple[str | None, str | None, float | None, float | None]:
        """(name, agency, lat, lon) from the AirNow site list, else from the training sites."""
        if site_id in self.site_by_id.index:
            m = self.site_by_id.loc[site_id]
            return m["site_name"], m["agency_name"], float(m["lat"]), float(m["lon"])
        if site_id in self.training.index:
            t = self.training.loc[site_id]
            return t["city"], None, float(t["lat"]), float(t["lon"])
        return None, None, None, None

    def _monitor(self, site_id: str, lat: float, lon: float, status: str, last_seen: str | None) -> dict:
        name, agency, slat, slon = self._site_info(site_id)
        agency = agency or self.agency.get(site_id)
        embassy = is_embassy(agency) or is_embassy(self.agency.get(site_id)) or site_id in self.embassy_last_seen
        dist = None if slat is None else round(float(haversine_km(lat, lon, slat, slon)), 1)
        return {"status": status, "network": EMBASSY_NETWORK if embassy else (agency or "AirNow"),
                "name": name, "last_seen": last_seen, "distance_km": dist, "site_id": site_id}

    def monitor_for(self, lat: float, lon: float, embassy_site_id: str | None = None,
                    radius_km: float = MONITOR_RADIUS_KM) -> dict:
        """Monitor block for a city: the SPEC ``monitor`` keys plus ``site_id`` and
        ``recent_obs`` ({date: observed}, only for an active monitor).

        1. the city's embassy monitor, if it reported PM2.5-24hr in the last ``ACTIVE_DAYS`` days;
        2. else the nearest PM2.5 monitor within ``radius_km`` that reported in that period;
        3. else ``offline`` if the city had an embassy monitor (``last_seen``: its last
           reporting day, from ``data/static/embassy_last_seen.json`` built by script 09);
        4. else ``none``.
        """
        embassy_site_id = embassy_site_id or None
        if embassy_site_id and embassy_site_id in self.last_obs.index:
            out = self._monitor(embassy_site_id, lat, lon, "active",
                                f"{self.last_obs[embassy_site_id]:%Y-%m-%d}")
        else:
            active = self.sites[self.sites["aqsid"].isin(self.last_obs.index)]
            dist = haversine_km(lat, lon, active["lat"].to_numpy(float), active["lon"].to_numpy(float))
            if len(active) and dist.min() <= radius_km:
                sid = active["aqsid"].iloc[int(np.argmin(dist))]
                out = self._monitor(sid, lat, lon, "active", f"{self.last_obs[sid]:%Y-%m-%d}")
            elif embassy_site_id:
                out = self._monitor(embassy_site_id, lat, lon, "offline",
                                    self.embassy_last_seen.get(embassy_site_id))
                out["network"] = EMBASSY_NETWORK
            else:
                out = {"status": "none", "network": None, "name": None, "last_seen": None,
                       "distance_km": None, "site_id": None}
        out["recent_obs"] = self.recent_obs(out["site_id"]) if out["status"] == "active" else {}
        return out
