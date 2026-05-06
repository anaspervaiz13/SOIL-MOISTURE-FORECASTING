# ISMN Merge Design

Date: 2026-05-05

## Goal

Create a clean merged CSV from the downloaded ISMN TERENO data under `data/ismn/` for the three selected stations:

- Gevenich
- Merzenhausen
- Selhausen

The output should support later modeling for soil-moisture forecasting while keeping the new workflow fully separated from the legacy project now stored under `old/`.

## Inputs

- Raw ISMN station files in `data/ismn/TERENO/`
- ISMN metadata in:
  - `data/ismn/Readme.txt`
  - `data/ismn/Metadata.json`
  - `data/ismn/ISMN_qualityflags_description.txt`
  - station-level `*_static_variables.csv`

## Selected Variables

The merged dataset will keep only the variables required for the forecasting pipeline:

- `sm` soil moisture
- `ts` soil temperature
- `ta` air temperature
- `p` precipitation

Downloaded but currently unused variables such as surface temperature or soil suction are excluded from the merged output.

## Processing Decisions

1. Parse each `.stm` file as:
   - one header line
   - many timestamped observations
2. Extract metadata from both filename and header:
   - station
   - variable code
   - depth
   - latitude / longitude / elevation
   - sensor name
3. Keep the observation quality flag as a separate field.
4. Filter observations to quality flag `G` only during preprocessing.
   - Reason: `Metadata.json` indicates `g_flag_only` is `false`, so we should enforce the quality rule ourselves.
5. Aggregate replicate sensors at the same:
   - `station`
   - `timestamp`
   - `variable`
   - `depth`
6. Use the median across replicate sensors for aggregation.
   - Reason: median is robust to occasional sensor-specific noise while preserving the shared signal.
7. Pivot aggregated data into a wide hourly table:
   - `sm_0.05m`, `sm_0.20m`, `sm_0.50m`
   - `ts_0.05m`, `ts_0.20m`, `ts_0.50m`
   - `ta_2.00m`
   - precipitation columns by sensor if more than one sensor remains distinguishable
8. Retain station-level location metadata.
9. Save outputs in a new pipeline area outside `old/`.

## Planned Outputs

- `data/processed/ismn_merged_hourly.csv`
- `outputs/ismn/merge_summary.json`
- `logs/ismn_processing_log.md`

## Merge Outcome Summary

- Final merged shape: `359,585 x 16`
- Final stations:
  - Gevenich
  - Merzenhausen
  - Selhausen
- Final merged time span:
  - `2011-02-07 14:00:00` to `2026-05-03 00:00:00`

Final columns:

- identifiers:
  - `station`
  - `timestamp`
  - `latitude`
  - `longitude`
  - `elevation`
- precipitation sensors:
  - `p_ecotech_rain_gauge`
  - `p_ott_pluvio2s_amount`
  - `p_ott_pluvio2s_volume`
  - `p_vaisala_wxt510`
- core variables:
  - `sm_0.05m`
  - `sm_0.20m`
  - `sm_0.50m`
  - `ta_2.00m`
  - `ts_0.05m`
  - `ts_0.20m`
  - `ts_0.50m`

## Locked Preprocessing Rules For Next Stage

1. Prediction target: `sm_0.05m`
2. Core predictors:
   - `sm_0.20m`
   - `sm_0.50m`
   - `ta_2.00m`
   - `ts_0.05m`
   - `ts_0.20m`
   - `ts_0.50m`
3. Consolidate precipitation into one feature using priority:
   - `p_ecotech_rain_gauge`
   - else `p_ott_pluvio2s_amount`
   - else `p_vaisala_wxt510`
4. Exclude `p_ott_pluvio2s_volume` from final modeling features.
5. Keep `station` as a retained field for cross-station modeling and later grouped validation.
6. Preserve merged CSV unchanged as traceable intermediate output.

## Final Forecasting Dataset

Final forecasting objective:

- Predict `sm_0.05m` at `t + 48 hours`
- Use past information up to `168` hours back
- Use multi-depth moisture, soil temperature, air temperature, precipitation, and calendar-seasonality signals

Feature design used:

- calendar features:
  - `hour`, `dayofweek`, `month`, `dayofyear`
  - `hour_sin`, `hour_cos`, `doy_sin`, `doy_cos`
- lag features for all dynamic variables at:
  - `1, 3, 6, 12, 24, 48, 72, 168` hours
- rolling features:
  - `sm_0.05m`: rolling mean and std at `6, 24, 48, 168` hours
  - `precipitation`: rolling sum at `6, 24, 48, 168` hours
  - `sm_0.20m`, `sm_0.50m`, `ta_2.00m`, `ts_0.05m`, `ts_0.20m`, `ts_0.50m`: rolling mean at `24, 48, 168` hours

Final outputs:

- `data/processed/ismn_forecasting_48h_full.csv`
  - shape: `359,585 x 116`
- `data/processed/ismn_forecasting_48h_ready.csv`
  - shape: `67,718 x 116`
- `outputs/ismn/final_dataset_summary.json`

Interpretation for thesis:

- The study is a **48-hour-ahead surface soil moisture forecasting task**.
- The novelty at dataset-design level comes from:
  - multi-depth fusion
  - multi-scale lag structure
  - weather-aware forecasting inputs
  - support for later stacked ensemble modeling

## Training Pipeline Design

Prepared model families:

- tabular:
  - XGBoost
  - KNN
- classical baselines:
  - ARIMA
  - Prophet
- deep learning:
  - LSTM
  - Transformer
- ensemble:
  - stacking meta-learner

Prepared evaluation structure:

- same 48-hour target for all models
- same chronological split logic
- repeated runs for stochastic models
- saved run-level predictions for stacking and uncertainty analysis later
- summary metrics with:
  - RMSE
  - MAE
  - R²
  - mean
  - standard deviation
  - 95% confidence interval

Prepared ablation structure:

- `surface_only`
- `surface_depth`
- `weather_depth_temporal`

This supports the professor requirements for:

- clear quantitative comparison
- statistical validation
- stronger baselines
- ablation with numeric evidence
- later uncertainty and robustness extensions

## Notes for Justification

- The new pipeline is intentionally separated from `old/` to preserve the earlier coursework pipeline untouched.
- Quality filtering and replicate aggregation are explicit preprocessing steps rather than assumptions.
- The merged table is a modeling-ready intermediate, not the final feature-engineered dataset.
