"""Select a globally diverse set of reference PM2.5 monitors for training.

* every US Department of State embassy/consulate monitor with >=100 valid days
  (one per city), covering Asia, Africa, the Middle East, Latin America and Europe;
* plus a spatially spread sample of US / Canada / Mexico regulatory monitors.

Output: config/training_sites.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from airtrace.config import CONFIG, PROCESSED  # noqa: E402

FIPS_REGION = {
    **dict.fromkeys(["BG", "IN", "NP", "PK", "CE"], "South Asia"),
    **dict.fromkeys(["CH", "MG"], "East Asia"),
    **dict.fromkeys(["BM", "ID", "LA", "MY", "RP", "VM"], "Southeast Asia"),
    **dict.fromkeys(["KZ", "KG", "TI", "TX", "UZ", "AJ"], "Central Asia & Caucasus"),
    **dict.fromkeys(["AG", "BA", "EG", "IZ", "JO", "KU", "QA", "SA", "AE"], "Middle East & North Africa"),
    **dict.fromkeys(["CD", "CF", "CG", "ET", "GH", "GV", "IV", "KE", "MA", "MZ", "NI", "RW", "SG",
                     "TO", "UG", "UV"], "Sub-Saharan Africa"),
    **dict.fromkeys(["BK", "KV", "MD"], "Europe"),
    **dict.fromkeys(["BR", "BH", "CO", "GT", "MX", "PA", "PE", "UC"], "Latin America & Caribbean"),
    **dict.fromkeys(["US", "CA"], "North America"),
}
N_NORTH_AMERICA = 36


def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * np.arcsin(np.sqrt(a))


def main() -> None:
    daily = pd.read_parquet(PROCESSED / "airnow_pm25_daily.parquet")
    sites = pd.read_parquet(PROCESSED / "airnow_sites.parquet")
    stats = (daily.groupby("aqsid")
             .agg(n_days=("pm25_obs", "size"), mean_pm25=("pm25_obs", "mean"), agency=("agency", "last"))
             .reset_index()
             .merge(sites, on="aqsid", how="inner"))
    stats = stats.dropna(subset=["lat", "lon"])

    # --- Embassy network -------------------------------------------------------------
    dos = stats[stats["agency"].str.contains("Department of State", na=False) & (stats["n_days"] >= 100)].copy()
    def split_label(agency: str) -> tuple[str, str]:
        label = agency.replace("U.S. Department of State", "").strip(" -")
        if "," in label:                       # "Ashgabat, Turkmenistan"
            city, country = [p.strip() for p in label.split(",", 1)]
            return country, city
        if "-" in label:                       # "India - New Delhi"
            country, city = [p.strip() for p in label.split("-", 1)]
            return country, city
        return label, label

    dos[["country_name", "city"]] = [split_label(a) for a in dos["agency"]]
    dos.loc[dos["country_name"] == "Congo", "city"] = "Kinshasa"
    dos.loc[dos["country_name"] == "Congo", "country_name"] = "DR Congo"
    dos.loc[dos["country_name"] == "Curacao", "city"] = "Willemstad"
    dos = dos.sort_values("n_days", ascending=False).drop_duplicates("city")
    dos["network"] = "US Dept. of State"

    # --- North American regulatory monitors -------------------------------------------
    na = stats[stats["country"].isin(["US", "CA", "MX"])
               & ~stats["agency"].str.contains("Department of State", na=False)
               & (stats["n_days"] >= 320)].copy()
    na = na.sort_values("n_days", ascending=False)
    chosen = list(na.nlargest(8, "mean_pm25").index)
    while len(chosen) < N_NORTH_AMERICA:
        d = np.full(len(na), np.inf)
        for idx in chosen:
            d = np.minimum(d, haversine_km(na.loc[idx, "lat"], na.loc[idx, "lon"], na["lat"].values, na["lon"].values))
        chosen.append(na.index[int(np.argmax(d))])
    na = na.loc[chosen].copy()
    na["country_name"] = na["country"].map({"US": "United States", "CA": "Canada", "MX": "Mexico"})
    na["city"] = (na["msa_name"].fillna("").str.split(",").str[0].str.split("-").str[0].str.strip())
    na.loc[na["city"] == "", "city"] = na["site_name"].str.title()
    na["network"] = "AirNow regulatory"

    out = pd.concat([dos, na], ignore_index=True)
    out["region"] = out["country"].map(FIPS_REGION).fillna("Other")
    out["site_id"] = out["aqsid"].str.strip()
    out = out[["site_id", "city", "country_name", "country", "region", "network", "lat", "lon",
               "elevation", "gmt_offset", "n_days", "mean_pm25"]]
    out["lat"] = out["lat"].round(4)
    out["lon"] = out["lon"].round(4)
    CONFIG.mkdir(parents=True, exist_ok=True)
    out.to_csv(CONFIG / "training_sites.csv", index=False)
    print(out.groupby("region").size().to_string())
    print(f"total sites={len(out)} countries={out['country'].nunique()}")


if __name__ == "__main__":
    main()
