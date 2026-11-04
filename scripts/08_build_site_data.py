"""Build the website data (docs/data/meta.json, cities.json, city/<id>.json) from live data.

Downloads, per showcase city, the Open-Meteo CAMS composition and weather forecasts (7 past +
5 forecast days, weight ~2.1 per city), plus once per run the NASA FIRMS 24-h fires and the
recent AirNow files, and runs the trained AirTrace and source models (see airtrace/live.py).

Environment:
  AIRTRACE_CITIES=id1,id2     rebuild only these cities (others keep their existing files)
  AIRTRACE_LIVE_REPLAY=DIR    development: save forecast responses in DIR and replay them on
                              later runs instead of calling Open-Meteo again
  AIRTRACE_MODEL_DIR=DIR      model artifacts (default models/)

Exit status is non-zero only if more than 30 % of the requested cities fail.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from airtrace.live import run  # noqa: E402


def main() -> int:
    only = [c.strip() for c in os.environ.get("AIRTRACE_CITIES", "").split(",") if c.strip()]
    replay = os.environ.get("AIRTRACE_LIVE_REPLAY")
    return run(only=only or None, replay_dir=Path(replay) if replay else None)


if __name__ == "__main__":
    sys.exit(main())
