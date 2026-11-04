#!/usr/bin/env bash
# End-to-end AirTrace pipeline (same steps as `make all`). Run from anywhere.
# Set SKIP_DOWNLOAD=1 to reuse existing data/raw downloads and start at the dataset step.
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3

step() { echo; echo "==> $*"; "$PY" "$@"; }

if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
  step scripts/01_download_airnow.py
  step scripts/02_select_sites.py
  step scripts/make_showcase_cities.py
  step scripts/03_download_openmeteo.py
  step scripts/03b_patch_blh.py
fi
step scripts/04_build_dataset.py
step scripts/05_train_evaluate.py
step scripts/06_fit_sources.py
step scripts/07_build_history.py
step scripts/09_network_history.py
step scripts/08_build_site_data.py

echo; echo "Pipeline finished. Preview with: $PY -m http.server 8000 --directory docs"
