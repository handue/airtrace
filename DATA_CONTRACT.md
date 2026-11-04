# Data formats

Notes on the model files and the JSON the website reads, so the Python pipeline and the
front end stay in sync. `tests/test_data_contract.py` checks the generated files against this.

## Model artifacts
* `models/pm25_lgbm.txt` LightGBM booster, target = log1p(PM2.5), inputs = `features.FEATURES` in order.
* `models/pm25_q10.txt`, `models/pm25_q90.txt` quantile boosters (alpha 0.1 / 0.9) on log1p target.
* `models/model_meta.json`: {"version", "trained_at", "features": [...], "target": "log1p_pm25",
  "n_rows", "n_sites", "train_period": [start, end]}
* `airtrace/model.py` provides `load_models(model_dir=None) -> dict` and
  `predict(models, X: pd.DataFrame) -> pd.DataFrame` with columns pm25, pm25_low, pm25_high (ug/m3) and
  `contributions(models, X) -> pd.DataFrame` (TreeSHAP in log space, one column per feature + "bias").
  model_dir defaults to env `AIRTRACE_MODEL_DIR` or `models/`.

## Source model
* `models/sources_nmf.json` (scales, components, labels, PM regression coefficients).
* `airtrace/sources.py` provides `load_source_model(path=None)` returning an object with
  `.shares(daily: pd.DataFrame) -> dict` where `daily` has TRACER_COLUMNS (daily rows of a window) and
  returns `{"shares": [{"key","label","share","color","description"}...], "background_share": float}`
  (shares incl. background sum to 1, sorted desc).
* Factor keys/labels/colors (use exactly):
  traffic "Traffic & urban combustion" #3b82f6 · industry "Industry, coal & power" #64748b ·
  burning "Fires & biomass burning" #f97316 · dust "Desert dust" #c8a165 ·
  secondary "Photochemical & secondary" #a855f7 · marine "Sea salt (marine)" #06b6d4 ·
  background "Regional background & other" #cbd5e1

## Website data contract (docs/data/*, LIVE writes, WEB reads)
All numbers rounded sensibly (PM 1 decimal, shares 3 decimals). Dates `YYYY-MM-DD`, times ISO local `YYYY-MM-DDTHH:MM`.

`meta.json`: {"generated_at": ISO-UTC, "model_version": str, "n_cities": int, "forecast_days": int,
 "data_sources": [str...]}

`cities.json`: list of {"id","name","country","region","lat","lon",
 "monitor_status": "active"|"offline"|"none", "today_pm25", "today_aqi", "today_category", "today_color",
 "peak_aqi_5d", "peak_category_5d", "peak_color_5d", "top_source": label|null}

`city/<id>.json`: {
 "id","name","country","region","lat","lon","timezone","generated_at",
 "monitor": {"status": "active"|"offline"|"none", "network": str|null, "name": str|null,
             "last_seen": date|null, "distance_km": float|null},
 "today": {"date","pm25","aqi","category","color","message"},
 "daily": [{"date","pm25","pm25_low","pm25_high","cams_pm25","aqi","category","color",
            "drivers": [{"group": str, "effect_pct": float}]}],   // 1..5 days; drivers = top 4 by |effect|,
                                                                  // effect_pct = (exp(sum SHAP of group)-1)*100
 "hourly": {"time": [...], "cams_pm25": [...], "airtrace_pm25": [...], "wind_speed": [...], "wind_dir": [...]},
            // airtrace hourly = CAMS hourly * (AirTrace daily / CAMS daily) for that day
 "sources": {"window": "YYYY-MM-DD to YYYY-MM-DD", "shares": [...as above], "background_share": float},
 "fires": {"count_300km": int, "upwind_count": int, "nearest_km": float|null, "frp_sum": float,
           "points": [[lat, lon, frp], ...] (<=300 nearest)},
 "recent": [{"date","observed","cams","airtrace"}],      // last <=7 days, only when a monitor is active
 "history": null | {"period": [start,end], "n_days": int, "mae_cams": f, "mae_airtrace": f,
                    "monthly": [{"month":"YYYY-MM","observed":f,"cams":f,"airtrace":f}],   // airtrace = out-of-fold CV prediction
                    "rose": {"sectors": ["N","NNE",...16], "mean_pm25": [16], "freq": [16]}}
}

`metrics.json`: {
 "generated_at",
 "dataset": {"n_sites","n_countries","n_rows","period":[s,e],
             "regions":[{"region","n_sites","n_rows"}], "networks": {name: n_sites}},
 "validation": {"scheme": str, "temporal_scheme": str},
 "models": [{"key","name","rmse","mae","r2","bias","r","aqi_category_accuracy",
             "unhealthy_precision","unhealthy_recall","unhealthy_f1"}],
             // keys: cams_raw, linear, ml_no_cams, airtrace   (leave-location-out CV)
 "headline": {"mae_reduction_pct","rmse_reduction_pct","aqi_acc_cams","aqi_acc_airtrace",
              "unhealthy_recall_cams","unhealthy_recall_airtrace","r2_cams","r2_airtrace"},
 "temporal_holdout": {"cams_raw": {metrics}, "airtrace": {metrics}},
 "by_region": [{"region","n_sites","rmse_cams","rmse_airtrace","mae_cams","mae_airtrace","bias_cams","bias_airtrace"}],
 "scatter": {"obs":[], "cams":[], "airtrace":[], "region":[]},   // <=1500 sampled CV points
 "confusion": {"labels":[6 AQI categories], "cams":[[6x6]], "airtrace":[[6x6]]},  // rows = observed
 "feature_importance": [{"feature","label","group","importance"}],   // top 15, shares of total gain
 "group_importance": [{"group","importance"}],
 "interval": {"nominal": 0.8, "coverage": f, "mean_width": f},
 "per_site": [{"site_id","city","country","region","lat","lon","n","mean_obs","mae_cams","mae_airtrace"}]
}

`sources_global.json`: {"method": str, "tracers": [labels],
 "factors": [{"key","label","color","description","profile": [loading per tracer, max-normalised]}],
 "pm_regression": {"r2": f, "n_rows": int, "intercept": f, "coef": {key: f}},
 "checks": [{"name","value","expected","passed","detail"}],
 "by_region": [{"region", "shares": {key: share}}]}

`network_history.json`: {"source": str, "months": ["YYYY-MM"...], "embassy_monitors": [int...],
 "all_pm25_monitors": [int...]}
