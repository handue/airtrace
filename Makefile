# AirTrace pipeline. Run from the repository root.
# NOTE: `make data` downloads from AirNow and Open-Meteo (rate-limited API; ~hours, cached under data/raw/).
PY      ?= .venv/bin/python
PIP     ?= .venv/bin/pip
PORT    ?= 8000

.PHONY: help setup data dataset train sources history network site all serve test clean-pycache

help:
	@echo "Targets: setup data dataset train sources history network site all serve test"

setup:            ## create .venv and install pinned dependencies (+ pytest)
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt pytest

data:             ## download AirNow + Open-Meteo, select sites, build showcase cities
	$(PY) scripts/01_download_airnow.py
	$(PY) scripts/02_select_sites.py
	$(PY) scripts/make_showcase_cities.py
	$(PY) scripts/03_download_openmeteo.py
	$(PY) scripts/03b_patch_blh.py

dataset:          ## training table (obs + features)
	$(PY) scripts/04_build_dataset.py

train:            ## LightGBM + quantile models, leave-location-out CV, metrics.json
	$(PY) scripts/05_train_evaluate.py

sources:          ## NMF receptor-model source attribution
	$(PY) scripts/06_fit_sources.py

history:          ## static per-city history from the training data
	$(PY) scripts/07_build_history.py

network:          ## embassy-monitor count per month
	$(PY) scripts/09_network_history.py

site:             ## live forecasts -> docs/data/*.json
	$(PY) scripts/08_build_site_data.py

all: data dataset train sources history network site

serve:            ## preview the website at http://localhost:$(PORT)
	$(PY) -m http.server $(PORT) --directory docs

test:
	$(PY) -m pytest -q

clean-pycache:
	find . -name __pycache__ -type d -prune -not -path './.venv/*' -exec rm -r {} +
