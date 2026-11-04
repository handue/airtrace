"""Build config/showcase_cities.csv: every embassy-monitor training city plus major world
cities that never had one. The live site forecasts all of them."""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from airtrace.config import CONFIG  # noqa: E402

EXTRA = [
    # name, country, region, lat, lon
    ("Seoul", "South Korea", "East Asia", 37.5665, 126.9780),
    ("Tokyo", "Japan", "East Asia", 35.6762, 139.6503),
    ("Taipei", "Taiwan", "East Asia", 25.0330, 121.5654),
    ("Chengdu", "China", "East Asia", 30.5728, 104.0668),
    ("Xi'an", "China", "East Asia", 34.3416, 108.9398),
    ("Bangkok", "Thailand", "Southeast Asia", 13.7563, 100.5018),
    ("Singapore", "Singapore", "Southeast Asia", 1.3521, 103.8198),
    ("Ho Chi Minh City", "Vietnam", "Southeast Asia", 10.8231, 106.6297),
    ("Sydney", "Australia", "Oceania", -33.8688, 151.2093),
    ("London", "United Kingdom", "Europe", 51.5074, -0.1278),
    ("Paris", "France", "Europe", 48.8566, 2.3522),
    ("Berlin", "Germany", "Europe", 52.5200, 13.4050),
    ("Madrid", "Spain", "Europe", 40.4168, -3.7038),
    ("Rome", "Italy", "Europe", 41.9028, 12.4964),
    ("Warsaw", "Poland", "Europe", 52.2297, 21.0122),
    ("Istanbul", "Turkey", "Europe", 41.0082, 28.9784),
    ("Moscow", "Russia", "Europe", 55.7558, 37.6173),
    ("Tehran", "Iran", "Middle East & North Africa", 35.6892, 51.3890),
    ("Riyadh", "Saudi Arabia", "Middle East & North Africa", 24.7136, 46.6753),
    ("Johannesburg", "South Africa", "Sub-Saharan Africa", -26.2041, 28.0473),
    ("Nairobi", "Kenya", "Sub-Saharan Africa", -1.2921, 36.8219),
    ("Mexico City", "Mexico", "Latin America & Caribbean", 19.4326, -99.1332),
    ("Buenos Aires", "Argentina", "Latin America & Caribbean", -34.6037, -58.3816),
    ("Santiago", "Chile", "Latin America & Caribbean", -33.4489, -70.6693),
    ("New York", "United States", "North America", 40.7128, -74.0060),
    ("Los Angeles", "United States", "North America", 34.0522, -118.2437),
    ("Chicago", "United States", "North America", 41.8781, -87.6298),
    ("Toronto", "Canada", "North America", 43.6532, -79.3832),
    ("Vancouver", "Canada", "North America", 49.2827, -123.1207),
]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def main() -> None:
    train = pd.read_csv(CONFIG / "training_sites.csv")
    dos = train[train["network"] == "US Dept. of State"]
    rows = [{"id": slug(r.city), "name": r.city, "country": r.country_name, "region": r.region,
             "lat": r.lat, "lon": r.lon, "embassy_site_id": r.site_id} for r in dos.itertuples()]
    for name, country, region, lat, lon in EXTRA:
        rows.append({"id": slug(name), "name": name, "country": country, "region": region,
                     "lat": lat, "lon": lon, "embassy_site_id": ""})
    out = pd.DataFrame(rows).drop_duplicates("id").sort_values(["region", "name"])
    out.to_csv(CONFIG / "showcase_cities.csv", index=False)
    print(f"{len(out)} showcase cities")


if __name__ == "__main__":
    main()
