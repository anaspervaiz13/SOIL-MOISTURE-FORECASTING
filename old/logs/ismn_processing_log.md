# ISMN Processing Log

Date started: 2026-05-05

## Step 1: Workspace separation confirmed

- Verified that legacy materials have been moved under `old/`.
- Confirmed that new ISMN work will be created outside `old/`.

## Step 2: Raw download structure inspected

- Confirmed raw input folder: `data/ismn/`
- Confirmed stations:
  - Gevenich
  - Merzenhausen
  - Selhausen
- Confirmed `.stm` files exist for:
  - soil moisture (`sm`)
  - soil temperature (`ts`)
  - air temperature (`ta`)
  - precipitation (`p`)

## Step 3: File formats inspected

- Verified `.stm` structure:
  - one metadata header line
  - timestamped observations
  - quality flag column
  - provider flag column
- Verified station static metadata files are present.

## Step 4: Quality-control rule decided

- Inspected `data/ismn/Metadata.json`.
- Found `g_flag_only` recorded as `false`.
- Decision: enforce `quality_flag == 'G'` during preprocessing.

## Step 5: Replicate handling rule decided

- Multiple sensors exist for the same station, variable, and depth.
- Decision: aggregate replicate sensors by median at each timestamp.

## Step 6: Merge target decided

- Build one merged hourly CSV with station-level rows and wide variable columns.
- Keep this as the canonical starting point for later feature engineering and forecasting experiments.

## Step 7: Runtime optimization added

- Reworked parsing to filter quality flag `G` during file read instead of after concatenation.
- Added parallel file parsing to use multiple CPU cores.
- Replaced row-wise feature naming with vectorized pandas string operations.
- Added short console progress messages for long-running stages.

## Step 7: Parser executed

- Parsed `65` raw `.stm` files.
- Parsed `6,095,434` timestamped observations after in-parser QC filtering.
- Parser workers used: `16`.

## Step 8: Quality filtering applied

- Retained `6,095,434` observations with quality flag `G`.
- Non-`G` observations were dropped inside per-file parsing for speed.

## Step 9: Replicate aggregation applied

- Aggregated replicate sensors by median at each station/timestamp/variable/depth combination.
- Produced `3,152,668` aggregated observations.

## Step 10: Merged output saved

- Saved merged hourly CSV to `C:/Users/HP/Desktop/UNI/final work/data/processed/ismn_merged_hourly.csv`.
- Saved summary JSON to `C:/Users/HP/Desktop/UNI/final work/outputs/ismn/merge_summary.json`.
- Final merged shape: `359,585` rows x `16` columns.

## Step 11: Output quality inspection

- Confirmed stations:
  - Gevenich
  - Merzenhausen
  - Selhausen
- Confirmed time span:
  - `2011-02-07 14:00:00` to `2026-05-03 00:00:00`
- Rows per station:
  - Gevenich: `120,258`
  - Merzenhausen: `130,114`
  - Selhausen: `109,213`

## Step 12: Missingness review

- Core soil moisture and soil temperature features are mostly about `11%` to `12.5%` missing.
- `ta_2.00m` is about `8.7%` missing.
- `p_ecotech_rain_gauge` and `p_vaisala_wxt510` have high coverage.
- `p_ott_pluvio2s_amount` and `p_ott_pluvio2s_volume` are Merzenhausen-specific and absent at the other two stations.

## Step 13: Core completeness check

- Complete rows using core non-precipitation variables:
  - Gevenich: `80.30%`
  - Merzenhausen: `73.81%`
  - Selhausen: `82.09%`
- Complete rows after adding consolidated precipitation:
  - Gevenich: `79.77%`
  - Merzenhausen: `73.41%`
  - Selhausen: `81.66%`

## Step 14: Precipitation consolidation rule locked

- Raw precipitation sensors were not averaged together.
- Final rule for next-stage dataset:
  - use `p_ecotech_rain_gauge` when available
  - else use `p_ott_pluvio2s_amount`
  - else use `p_vaisala_wxt510`
- Exclude `p_ott_pluvio2s_volume` from final modeling features.
- Reason:
  - yields one precipitation feature with about `99.31%` coverage
  - avoids mixing all raw sensor streams by direct averaging
  - avoids keeping Merzenhausen-only `volume` stream as a standard cross-station feature

## Step 15: Forecast target locked

- Main prediction target for next-stage forecasting dataset: `sm_0.05m`
- Supporting predictors retained:
  - `sm_0.20m`
  - `sm_0.50m`
  - `ta_2.00m`
  - `ts_0.05m`
  - `ts_0.20m`
  - `ts_0.50m`

## Step 16: 48-hour forecasting dataset built

- Loaded `data/processed/ismn_merged_hourly.csv`.
- Created one consolidated `precipitation` feature using priority:
  - `p_ecotech_rain_gauge`
  - else `p_ott_pluvio2s_amount`
  - else `p_vaisala_wxt510`
- Created forecasting target:
  - `target_48h = sm_0.05m shifted forward by 48 hours within each station`
- Added calendar features.
- Added multi-scale lag features at:
  - `1, 3, 6, 12, 24, 48, 72, 168` hours
- Added rolling summary features:
  - target rolling mean/std
  - precipitation rolling sums
  - deeper-moisture and temperature rolling means

## Step 17: Final dataset outputs saved

- Saved full engineered dataset to `C:/Users/HP/Desktop/UNI/final work/data/processed/ismn_forecasting_48h_full.csv`.
- Saved model-ready filtered dataset to `C:/Users/HP/Desktop/UNI/final work/data/processed/ismn_forecasting_48h_ready.csv`.
- Saved summary to `C:/Users/HP/Desktop/UNI/final work/outputs/ismn/final_dataset_summary.json`.
- Full engineered dataset shape:
  - `359,585` rows x `116` columns
- Model-ready dataset shape:
  - `67,718` rows x `116` columns

## Step 18: Final dataset inspection

- Model-ready station counts:
  - Merzenhausen: `26,370`
  - Selhausen: `25,786`
  - Gevenich: `15,562`
- Model-ready time range:
  - `2011-10-09 00:00:00` to `2025-12-23 14:00:00`
- This filtered dataset already respects:
  - 48-hour prediction horizon
  - multi-scale temporal memory up to 168 hours
  - cross-station merged setup

## Step 19: Training pipeline structure created

- Added shared training utilities under `src/training/` for:
  - dataset loading
  - chronological train/validation/test splitting
  - feature-group selection for ablations
  - metric computation
  - repeated-run summaries with mean/std/95% CI
- Added model runner scripts without executing long training jobs.

## Step 20: Model runners prepared

- Tabular models:
  - `src/train_xgboost_48h.py`
  - `src/train_knn_48h.py`
- Classical baselines:
  - `src/train_arima_48h.py`
  - `src/train_prophet_48h.py`
- Sequence models:
  - `src/train_lstm_48h.py`
  - `src/train_transformer_48h.py`
- Ensemble stage:
  - `src/train_stacking_48h.py`
- Result summarization:
  - `src/summarize_training_results.py`

## Step 21: Modeling design choices locked

- Shared target across pipelines:
  - `target_48h`
- Shared split style:
  - chronological train / validation / test split based on global timestamps
- Shared ablation-ready feature groups:
  - `surface_only`
  - `surface_depth`
  - `weather_depth_temporal`
- Shared evaluation outputs:
  - per-run predictions
  - per-run metrics
  - summary metrics with mean/std/95% CI

## Step 22: Fast validation completed

- Syntax-checked all training scripts using `python -m py_compile`.
- Verified shared utility behavior with `python -m unittest tests.test_training_utils -v`.
- No long training jobs were executed in this session by design.

## Step 16: 48-hour forecasting dataset built

- Read merged hourly ISMN dataset.
- Created one consolidated `precipitation` feature using locked sensor priority rule.
- Created `target_48h` as future `sm_0.05m` shifted by `-48` hours within each station.
- Added calendar, multi-scale lag, and rolling-window features.
- Saved full engineered dataset to `C:/Users/HP/Desktop/UNI/final work/data/processed/ismn_forecasting_48h_full.csv`.
- Saved model-ready filtered dataset to `C:/Users/HP/Desktop/UNI/final work/data/processed/ismn_forecasting_48h_ready.csv`.
- Full dataset shape: `359,585` rows x `116` columns.
- Ready dataset shape: `67,718` rows x `116` columns.
