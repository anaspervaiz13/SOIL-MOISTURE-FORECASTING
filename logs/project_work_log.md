# Project Work Log

Date started: 2026-05-06

## Step 1: New project restart confirmed

- Confirmed that the old workflow remains archived under `old/`.
- Confirmed that the new work will be rebuilt from scratch in the active workspace.
- Confirmed that the planning, logging, and step-by-step discipline from the previous workflow will be reused as process only, not as copied content.

## Step 2: New research direction locked

- Main novelty direction selected:
  - multi-scale temporal lag design
- Supporting direction selected:
  - lightweight models may benefit strongly from structured temporal features
- Final write-up will be based on real experiment results, not only planned claims.

## Step 3: New documentation baseline created

- Created a fresh design spec for the new work.
- Created a fresh end-to-end implementation plan.
- Created this project work log for step-by-step tracking.

## Step 4: Current execution rule

- The implementation plan is a living document.
- It may be updated as the experiments reveal better directions or necessary methodological fixes.
- Any major change in data design, evaluation setup, or model scope should be recorded here before or immediately after it is made.

## Step 5: Execution environment exception recorded

- The default isolation workflow recommended a git worktree before implementation.
- By user instruction, execution continued directly on the current `main` branch.

## Step 6: New merged-data builder implemented with tests first

- Created `src/build_ismn_merged.py` from scratch in the active workspace.
- Wrote and ran new unit tests for:
  - filename metadata parsing
  - `G`-quality filtering
  - replicate aggregation by median
  - merged-dataset summary reporting
- Validation commands completed:
  - `python -m unittest tests.test_build_ismn_merged -v`
  - `python -m py_compile src\build_ismn_merged.py`

## Step 7: Raw ISMN merge rebuilt from scratch

- Raw source directory used:
  - `data/original dataset/`
- Applied local preprocessing rule:
  - keep only rows with `quality_flag == 'G'`
- Parsed raw `.stm` files:
  - `65`
- Retained observations after local quality filtering:
  - `6,095,434`
- Aggregated replicate observations by median at each station/timestamp/feature:
  - `3,152,668` rows

## Step 8: New merged dataset saved and inspected

- Saved merged hourly dataset to:
  - `C:/Users/HP/Desktop/UNI/final work/data/processed/ismn_merged_hourly.csv`
- Saved merge summary to:
  - `C:/Users/HP/Desktop/UNI/final work/outputs/ismn/merge_summary.json`
- Final merged dataset shape:
  - `359,585` rows x `16` columns
- Stations confirmed:
  - Gevenich
  - Merzenhausen
  - Selhausen
- Time span confirmed:
  - `2011-02-07 14:00:00` to `2026-05-03 00:00:00`
- Rows per station:
  - Gevenich: `120,258`
  - Merzenhausen: `130,114`
  - Selhausen: `109,213`
- Missingness snapshot:
  - `sm_0.05m`: `12.49%`
  - `sm_0.20m`: `11.10%`
  - `sm_0.50m`: `11.66%`
  - `ta_2.00m`: `8.71%`
  - `ts_0.05m`: `11.96%`
  - `ts_0.20m`: `11.61%`
  - `ts_0.50m`: `11.67%`

## Step 9: Forecasting-dataset builder implemented with tests first

- Created `src/build_ismn_forecasting_dataset.py` from scratch in the active workspace.
- Wrote and ran new unit tests for:
  - precipitation consolidation priority
  - station-wise future target shifting
  - model-ready row filtering
- Validation commands completed:
  - `python -m unittest tests.test_build_ismn_forecasting_dataset -v`
  - `python -m py_compile src\build_ismn_forecasting_dataset.py`

## Step 10: 48-hour forecasting dataset built from the new merged table

- Input dataset used:
  - `C:/Users/HP/Desktop/UNI/final work/data/processed/ismn_merged_hourly.csv`
- Consolidated precipitation into one feature using priority:
  - `p_ecotech_rain_gauge`
  - else `p_ott_pluvio2s_amount`
  - else `p_vaisala_wxt510`
- Excluded raw precipitation sensor columns from the forecasting feature table.
- Created forecasting target:
  - `target_48h = future sm_0.05m shifted by 48 rows within each station`
- Added calendar features:
  - `hour`
  - `dayofweek`
  - `month`
  - `dayofyear`
  - `hour_sin`
  - `hour_cos`
  - `doy_sin`
  - `doy_cos`
- Added multi-scale lag features at:
  - `1, 3, 6, 12, 24, 48, 72, 168` hours
- Added rolling features:
  - `sm_0.05m` rolling mean and std over `6, 24, 48, 168`
  - `precipitation` rolling sum over `6, 24, 48, 168`
  - `sm_0.20m`, `sm_0.50m`, `ta_2.00m`, `ts_0.05m`, `ts_0.20m`, `ts_0.50m` rolling mean over `24, 48, 168`

## Step 11: Forecasting outputs saved and inspected

- Saved full engineered forecasting dataset to:
  - `C:/Users/HP/Desktop/UNI/final work/data/processed/ismn_forecasting_48h_full.csv`
- Saved model-ready forecasting dataset to:
  - `C:/Users/HP/Desktop/UNI/final work/data/processed/ismn_forecasting_48h_ready.csv`
- Saved forecasting summary to:
  - `C:/Users/HP/Desktop/UNI/final work/outputs/ismn/final_dataset_summary.json`
- Full dataset shape:
  - `359,585` rows x `116` columns
- Model-ready dataset shape:
  - `67,718` rows x `116` columns
- Model-ready rows per station:
  - Gevenich: `15,562`
  - Merzenhausen: `26,370`
  - Selhausen: `25,786`
- Model-ready time span:
  - `2011-10-09 00:00:00` to `2025-12-23 14:00:00`
- Consolidated precipitation missingness:
  - `0.69%`

## Step 12: Forecasting preparation generalized for multiple horizons

- Updated the forecasting builder so dataset preparation is no longer locked to only one target horizon.
- Added tested support for target naming by horizon:
  - `target_24h`
  - `target_48h`
  - `target_72h`
- Re-ran the forecasting builder to save reusable prepared datasets for:
  - `24h`
  - `48h`
  - `72h`

## Step 13: Multi-horizon prepared datasets saved

- Saved horizon-specific full datasets:
  - `data/processed/ismn_forecasting_24h_full.csv`
  - `data/processed/ismn_forecasting_48h_full.csv`
  - `data/processed/ismn_forecasting_72h_full.csv`
- Saved horizon-specific model-ready datasets:
  - `data/processed/ismn_forecasting_24h_ready.csv`
  - `data/processed/ismn_forecasting_48h_ready.csv`
  - `data/processed/ismn_forecasting_72h_ready.csv`
- Saved horizon-specific summaries:
  - `outputs/ismn/final_dataset_summary_24h.json`
  - `outputs/ismn/final_dataset_summary_48h.json`
  - `outputs/ismn/final_dataset_summary_72h.json`
- Ready dataset shapes:
  - `24h`: `68,113` rows x `116` columns
  - `48h`: `67,718` rows x `116` columns
  - `72h`: `67,394` rows x `116` columns

## Step 14: Reviewer findings recorded and verified against the data

- Reviewed issue 1:
  - row-based hour horizons were not guaranteed to equal real clock hours when station timelines contained outages
- Verified with direct checks on `data/processed/ismn_merged_hourly.csv`:
  - Gevenich had `306` gaps greater than 1 hour, max gap `91 days 10:00:00`
  - Merzenhausen had `89` gaps greater than 1 hour, max gap `19 days 19:00:00`
  - Selhausen had `94` gaps greater than 1 hour, max gap `7 days 07:00:00`
- Conclusion:
  - the reviewer finding was correct
  - row shifts alone were not methodologically safe for true `24h/48h/72h` forecasting

- Reviewed issue 2:
  - ready rows were filtered only by non-null values, not by a continuity-safe temporal representation
- Verified with direct checks on the old ready dataset:
  - retained rows still showed large timestamp jumps between neighboring ready rows
- Root cause:
  - the underlying full dataset had not first been expanded onto a true hourly grid

## Step 15: Measures taken to fix the continuity findings

- Added new tests before changing the implementation:
  - hourly-grid expansion inserts missing timestamps
  - target shifting respects missing hours after grid expansion
- Implemented `expand_station_hourly_grid()` in `src/build_ismn_forecasting_dataset.py`
- Changed the forecasting pipeline so each station is expanded to a complete hourly timestamp grid before:
  - target generation
  - lag generation
  - rolling-window feature generation
- Preserved station identity and static metadata during grid expansion:
  - `station`
  - `latitude`
  - `longitude`
  - `elevation`
- Rebuilt all prepared forecasting datasets after the fix

## Step 16: Post-fix verification for continuity safety

- Full forecasting datasets are now continuous hourly grids by station:
  - Gevenich full gaps greater than 1 hour: `0`
  - Merzenhausen full gaps greater than 1 hour: `0`
  - Selhausen full gaps greater than 1 hour: `0`
- Rebuilt full dataset shape changed from `359,585` to `372,813` rows because explicit gap hours are now represented.
- Rebuilt `48h` model-ready dataset shape:
  - `67,380` rows x `116` columns
- Verified that all temporal columns in the ready `48h` dataset are fully populated:
  - checked `95` target/lag/rolling columns
  - total missing values across those columns: `0`
- Verified ready rows are drawn only from the rebuilt hourly full dataset:
  - merge check result: all `67,380` ready rows matched the full hourly table

## Step 17: Interpretation after the fix

- Large timestamp jumps can still appear between neighboring rows in the final ready dataset.
- This is now expected behavior, not a remaining labeling bug.
- Reason:
  - the full dataset contains explicit hourly gap rows
  - rows lacking valid history or future target remain `NaN`
  - the ready dataset removes those invalid rows
- Therefore:
  - the named horizons in the full dataset are now true clock-hour horizons
  - the ready dataset keeps only rows where those horizons and temporal features are actually defined

## Step 18: Shared training utilities created for lag-ablation work

- Created new training package under `src/training/`:
  - `config.py`
  - `feature_sets.py`
  - `data.py`
  - `evaluation.py`
  - `__init__.py`
- Added new tests:
  - `tests/test_training_utils.py`
  - `tests/test_evaluation_utils.py`

## Step 19: Ablation feature groups locked for the first benchmark stage

- Defined named ablation groups:
  - `baseline_limited`
  - `short_lags`
  - `short_plus_medium`
  - `full_multiscale`
- Purpose of the groups:
  - compare simpler lag setups against the full multi-scale temporal design
  - isolate whether medium and weekly history improve forecasting

## Step 20: Training utility validation completed

- Verified chronological split helper uses time order.
- Verified feature-group selection expands the intended lag patterns.
- Verified metric computation and repeated-run summaries work as expected.
- Validation commands completed:
  - `python -m unittest tests.test_training_utils tests.test_evaluation_utils -v`
  - `python -m py_compile src\training\__init__.py src\training\config.py src\training\feature_sets.py src\training\data.py src\training\evaluation.py`

## Step 21: XGBoost ablation runner created and validated

- Created `src/train_xgboost_48h.py` for the first anchor benchmark.
- Added `tests/test_train_xgboost_48h.py` for basic runner parsing behavior.
- Added output-saving support to `src/training/evaluation.py`.
- Validation commands completed:
  - `python -m unittest tests.test_training_utils tests.test_evaluation_utils tests.test_train_xgboost_48h -v`
  - `python -m py_compile src\train_xgboost_48h.py`

## Step 22: First light XGBoost ablation runs completed

- Purpose:
  - get a fast early read on whether the lag-design ablation shows useful signal before longer experiments
- Settings used for this light benchmark:
  - seeds: `42, 52, 62, 72, 82`
  - `n_estimators=80`
  - `max_depth=4`
  - `learning_rate=0.08`
- Feature groups evaluated:
  - `baseline_limited`
  - `short_lags`
  - `short_plus_medium`
  - `full_multiscale`

## Step 23: Early ablation findings recorded

- Single-seed first impression briefly suggested `baseline_limited` was best.
- After repeated-seed light runs, the ranking changed and became more stable.
- Light repeated-run test RMSE means:
  - `full_multiscale`: `0.028472`
  - `baseline_limited`: `0.028597`
  - `short_lags`: `0.028683`
  - `short_plus_medium`: `0.028788`
- Light repeated-run test R2 means:
  - `full_multiscale`: `0.861000`
  - `baseline_limited`: `0.859793`
  - `short_lags`: `0.858947`
  - `short_plus_medium`: `0.857910`
- Early interpretation:
  - the multi-scale lag design is showing a small but favorable signal in the first XGBoost benchmark
  - repeated runs matter, because the single-seed result gave a weaker and slightly misleading picture

## Step 24: Stronger XGBoost ablation sweep completed

- Added output-tag support to `src/train_xgboost_48h.py` so stronger runs do not overwrite exploratory runs.
- Added `src/summarize_training_results.py` to collect saved experiment summaries.
- Stronger XGBoost settings used:
  - seeds: `42, 52, 62, 72, 82`
  - `n_estimators=300`
  - `max_depth=6`
  - `learning_rate=0.05`
  - output tag: `strong`
- Stronger run folders created:
  - `baseline_limited__strong`
  - `short_lags__strong`
  - `short_plus_medium__strong`
  - `full_multiscale__strong`

## Step 25: Stronger XGBoost ablation findings recorded

- Strong repeated-run test RMSE means:
  - `full_multiscale__strong`: `0.029122`
  - `baseline_limited__strong`: `0.029588`
  - `short_lags__strong`: `0.029676`
  - `short_plus_medium__strong`: `0.029770`
- Strong repeated-run test R2 means:
  - `full_multiscale__strong`: `0.854585`
  - `baseline_limited__strong`: `0.849903`
  - `short_lags__strong`: `0.849005`
  - `short_plus_medium__strong`: `0.848049`
- Strong-stage interpretation:
  - `full_multiscale` remains the best-performing XGBoost ablation setup
  - this gives the novelty direction a more credible result than the initial single-seed check
  - the observed gain is not huge, but it is directionally consistent in the stronger repeated-run benchmark

## Step 26: XGBoost comparison summary exported

- Exported XGBoost experiment summary table to:
  - `outputs/training/xgboost/model_comparison_summary.csv`
- Validation commands completed:
  - `python -m unittest tests.test_training_utils tests.test_evaluation_utils tests.test_train_xgboost_48h tests.test_summarize_training_results -v`
  - `python -m py_compile src\train_xgboost_48h.py`
  - `python -m py_compile src\summarize_training_results.py`

## Step 27: KNN benchmark runner created and light ablation run completed

- Created `src/train_knn_48h.py`
- Added `tests/test_train_knn_48h.py`
- Light KNN settings used:
  - neighbors: `5, 9, 13`
  - weights: `distance`
  - output tag: `light`
- KNN run folders created:
  - `baseline_limited__light`
  - `short_lags__light`
  - `short_plus_medium__light`
  - `full_multiscale__light`

## Step 28: Early KNN findings recorded

- Light KNN test RMSE means:
  - `baseline_limited__light`: `0.038342`
  - `short_lags__light`: `0.038864`
  - `short_plus_medium__light`: `0.039318`
  - `full_multiscale__light`: `0.040860`
- Light KNN interpretation:
  - unlike XGBoost, KNN does not benefit from the richer multi-scale feature set in this first benchmark
  - the stronger lag-design signal currently appears model-dependent rather than universal
  - this is useful for the final report because it suggests the temporal design helps tree models more than distance-based models

## Step 29: LSTM runner and sequence utilities prepared

- Created sequence helper module:
  - `src/training/sequence.py`
- Added sequence-related tests:
  - `tests/test_sequence_utils.py`
  - `tests/test_train_lstm_48h.py`
- Created LSTM runner:
  - `src/train_lstm_48h.py`
- CPU-friendly defaults kept from the earlier workflow direction:
  - `epochs=40`
  - early stopping enabled
  - thread control exposed with `--threads`

## Step 30: LSTM setup validation completed

- Confirmed TensorFlow is available in the environment:
  - `tensorflow 2.21.0`
- Validation commands completed:
  - `python -m py_compile src\training\sequence.py src\train_lstm_48h.py`
  - `python -m unittest tests.test_sequence_utils tests.test_train_lstm_48h -v`

## Step 31: ARIMA runner prepared with leakage-safe source-series handling

- Created `src/train_arima_48h.py`
- Added `tests/test_train_arima_48h.py`
- Explicitly validated that ARIMA preparation uses:
  - source series: `sm_0.05m`
  - not the future-shifted target column for training input
- Implemented forecast alignment by future timestamp rather than label leakage

## Step 32: ARIMA setup status

- The runner is code-complete and validated locally with unit tests.
- `statsmodels` is not installed in this current environment, so the actual ARIMA fit was not executed here.
- Validation commands completed:
  - `python -m unittest tests.test_train_arima_48h -v`
  - `python -m py_compile src\train_arima_48h.py`

## Step 33: ARIMA failure interpretation and fix adjustment

- User-run ARIMA attempt with full long history produced no valid station forecasts.
- Data-shape review showed the problem was not empty splits:
  - all three stations had substantial train, validation, and test coverage
  - interpolated source series were finite after hourly expansion
- Likely practical cause:
  - ARIMA on very long hourly histories was too unstable or too heavy for the chosen setup
- Corrective measure taken:
  - added `--max-history-hours` support
  - defaulted ARIMA training to a bounded recent-history window of `180` days (`4320` hours)
  - added `--debug` output so skipped-station reasons are visible if the run still fails

## Step 34: ARIMA bounded-history run completed

- User ran:
  - `python src\train_arima_48h.py --order 3,0,1 --seasonal-order 0,0,0,0 --max-history-hours 4320 --output-tag safe180 --debug`
- Output saved to:
  - `outputs/training/arima/arima__safe180`
- Station coverage outcome:
  - Gevenich: completed
  - Merzenhausen: completed
  - Selhausen: skipped
- Recorded skip reason:
  - `Selhausen`: `test_forecast_failed`

## Step 35: ARIMA result interpretation

- ARIMA completed as a valid classical baseline but performed very poorly.
- Test summary metrics:
  - `RMSE`: `0.270940`
  - `MAE`: `0.256446`
  - `R2`: `-12.244478`
- Practical interpretation:
  - the bounded-history ARIMA baseline is usable as a comparison artifact
  - it is not competitive with the machine-learning models
  - it should be presented as a weak classical reference rather than a strong contender

## Step 36: Prophet runner prepared with leakage-safe source-series handling

- Created `src/train_prophet_48h.py`
- Added `tests/test_train_prophet_48h.py`
- Explicitly validated that Prophet preparation uses:
  - source series: `sm_0.05m`
  - not the future-shifted target column for model fitting
- Implemented forecast alignment by future timestamp and station-level skip reporting

## Step 37: Prophet setup status

- The runner is code-complete and validated locally with unit tests.
- `prophet` is not installed in this current environment, so the actual Prophet fit was not executed here.
- Validation commands completed:
  - `python -m unittest tests.test_train_prophet_48h -v`
  - `python -m py_compile src\train_prophet_48h.py`

## Step 38: Prophet run completed

- User ran:
  - `python src\train_prophet_48h.py --changepoint-prior-scale 0.05 --seasonality-prior-scale 10.0 --output-tag safe --debug`
- Output saved to:
  - `outputs/training/prophet/prophet__safe`
- Coverage outcome:
  - all three stations completed
  - no skipped stations

## Step 39: Prophet result interpretation

- Prophet completed as a valid classical baseline and performed better than ARIMA.
- Test summary metrics:
  - `RMSE`: `0.068825`
  - `MAE`: `0.055729`
  - `R2`: `0.081803`
- Practical interpretation:
  - Prophet is a usable classical benchmark
  - it is still clearly weaker than XGBoost, KNN, and LSTM in this setup

## Step 40: Current benchmark picture

- Initial benchmark snapshot before later coverage/fairness cleanup:
  - `XGBoost full_multiscale strong`: `0.029122`
  - `KNN baseline_limited light`: `0.038342`
  - `LSTM cpu40`: `0.058781`
  - `Prophet safe`: `0.068825`
  - `ARIMA safe180`: `0.270940`
- Current interpretation:
  - this snapshot was useful for early direction-setting, but later review showed these entries do not all share the same effective evaluation population
  - XGBoost is the strongest model so far
  - KNN is a useful lightweight secondary baseline
  - LSTM does not beat the lightweight tabular methods
  - Prophet is a moderate classical baseline
  - ARIMA is a very weak classical baseline in this setting

## Step 41: Prophet station-level behavior noted for later reporting

- Prophet completed on all three stations with no station skips.
- Test RMSE by station:
  - `Gevenich`: `0.090783`
  - `Merzenhausen`: `0.065219`
  - `Selhausen`: `0.050473`
- Interpretation:
  - Prophet is stable enough to retain as a benchmark
  - its performance varies noticeably by station
  - even its best station result remains weaker than the best XGBoost benchmark

## Step 42: Cross-model comparison summary exported

- Extended `src/summarize_training_results.py` to support:
  - per-model family summaries
  - one combined cross-model summary across all saved experiment folders
- Added test coverage in `tests/test_summarize_training_results.py`
- Exported:
  - `outputs/training/all_model_comparison_summary.csv`
- Important interpretation note:
  - the combined file is experiment-level
  - KNN experiment summaries average across the saved neighbor-count runs inside a folder
  - when discussing the single best KNN configuration, use `run_metrics.csv` for the chosen folder rather than the folder-level mean

## Step 43: Stacking feasibility check

- Checked alignment across saved prediction outputs before attempting stacking.
- Findings:
  - XGBoost and KNN predictions align exactly on `19,718` `(station, timestamp, split)` rows
  - Prophet predictions also align on the same full set
  - LSTM predictions overlap only `13,710` of those rows because the sequence lookback shifts its usable range
- Practical decision:
  - excluded LSTM from the first fair stacking experiment
  - used only the fully aligned models for ensemble testing

## Step 44: Linear stacking ensemble run completed

- Created `src/train_stacking_48h.py`
- Added `tests/test_train_stacking_48h.py`
- Ensemble members:
  - `XGBoost`: `full_multiscale__strong`, averaged across seeds
  - `KNN`: `baseline_limited__light`, with best neighbor count selected on validation
  - `Prophet`: `prophet__safe`
- KNN neighbor selected from validation metrics:
  - `13`
- Fitted a simple linear regression stack on validation rows and evaluated on test rows.
- Output saved to:
  - `outputs/training/stacking/stack_linear__aligned`
- Alignment counts:
  - total aligned rows: `19,718`
  - validation rows: `10,161`
  - test rows: `9,557`

## Step 45: Stacking result interpretation

- Test summary metrics:
  - `RMSE`: `0.028846`
  - `MAE`: `0.017424`
  - `R2`: `0.857341`
- Learned coefficients:
  - `xgboost_pred`: `1.0101`
  - `knn_pred`: `-0.0046`
  - `prophet_pred`: `0.0243`
- Practical interpretation:
  - the stack slightly improves over the stronger XGBoost run (`0.029122`)
  - it does not beat the best overall XGBoost result (`0.028472`)
  - the learned weights are dominated by XGBoost, which suggests limited added value from KNN and Prophet
  - stacking is acceptable as a supplementary experiment, but it should not be presented as the main story

## Step 46: Novelty-focused result package exported

- Created `src/export_novelty_results.py`
- Added `tests/test_export_novelty_results.py`
- Exported the current novelty package to:
  - `outputs/training/novelty/xgboost_ablation_table.csv`
  - `outputs/training/novelty/benchmark_table.csv`
  - `outputs/training/novelty/novelty_results_summary.md`
- Package purpose:
  - one clean lag-design ablation table
  - one clean best-configuration benchmark table
  - one short interpretation that matches the current research direction

## Step 47: Novelty package interpretation

- XGBoost ablation result:
  - `full_multiscale__strong` is the best lag-design configuration with `RMSE 0.029122`
  - absolute gain over `baseline_limited__strong`: `0.000466`
  - relative gain over `baseline_limited__strong`: `1.574%`
- Benchmark result:
  - best overall model entry remains `XGBoost full_multiscale` with `RMSE 0.028472`
  - stacking is second at `0.028846`
  - best single saved KNN setting is `neighbor_count=13` with `RMSE 0.036659`
- Practical interpretation:
  - the novelty story is now evidence-backed and exportable
  - the main contribution should remain `multi-scale temporal lag design`
  - stacking should remain supplementary rather than central

## Step 48: Review-driven benchmark cleanup

- Verified external review findings against the code and artifacts.
- Confirmed valid issues:
  - KNN selection in `src/export_novelty_results.py` was using the test split
  - cross-model summaries were presenting mixed evaluation populations without enough coverage context
  - novelty wording around stacking needed to be toned down to match the saved coefficients and outcomes
- Confirmed interpretation issue rather than code bug:
  - the current lag-design effect is present but modest, so the narrative should describe it as a measurable benefit rather than a large methodological breakthrough

## Step 49: Export and summary fixes applied

- Updated `src/export_novelty_results.py`:
  - KNN configuration is now selected on validation RMSE and reported on the corresponding test row
  - benchmark table now includes coverage columns:
    - `test_prediction_rows`
    - `test_unique_keys`
    - `test_station_count`
    - `test_stations`
  - benchmark table now includes `selection_basis`
  - novelty markdown now avoids direct overall ranking claims across mixed test populations
- Updated `src/summarize_training_results.py`:
  - cross-model summary now includes the same coverage fields
  - output is sorted by `model, experiment` rather than `rmse_mean` to avoid implying a fully fair global ranking

## Step 50: Regenerated review-safe artifacts

- Regenerated:
  - `outputs/training/novelty/benchmark_table.csv`
  - `outputs/training/novelty/novelty_results_summary.md`
  - `outputs/training/all_model_comparison_summary.csv`
- New benchmark table behavior:
  - KNN now reports `neighbor_count=13` chosen on validation
  - the table exposes that:
    - XGBoost and stacking are evaluated on `9,557` unique test keys
    - LSTM is on `26,598` unique test keys
    - Prophet is on `68,546`
    - ARIMA is on `47,226` and only `2` stations
- Practical interpretation:
  - benchmark tables are now descriptive and more methodologically honest
  - common-population benchmarking is still a remaining task if we want strict apples-to-apples ranking across every model

## Step 51: Refined lag-ablation design implemented

- Expanded the XGBoost ablation study to separate the effects of:
  - no lag history (`current_only`)
  - limited lag baseline
  - short lags
  - short + medium lags
  - short + weekly lags
  - full multiscale lag set without rolling features
  - full multiscale lag set with rolling features
- Updated:
  - `src/training/feature_sets.py`
  - `src/train_xgboost_48h.py`
  - `tests/test_training_utils.py`
- Added refined run outputs under `outputs/training/xgboost/*__refined`

## Step 52: Refined XGBoost ablation results

- Refined repeated-run test RMSE means:
  - `current_only__refined`: `0.031198`
  - `baseline_limited__refined`: `0.029588`
  - `short_lags__refined`: `0.029676`
  - `short_plus_medium__refined`: `0.029770`
  - `short_plus_weekly__refined`: `0.029236`
  - `full_multiscale_no_rolling__refined`: `0.029839`
  - `full_multiscale__refined`: `0.029122`
- Exported refined novelty package to:
  - `outputs/training/novelty_refined/xgboost_ablation_table.csv`

## Step 53: Refined novelty interpretation

- The refined ablation strengthens the temporal-context story because it separates the contribution sources more clearly.
- Main observations:
  - removing lag history entirely hurts performance substantially (`current_only`)
  - the weekly-context branch (`short_plus_weekly`) performs better than `short_plus_medium`
  - medium lags alone do not explain the best result
  - full multiscale with rolling features performs best overall
  - full multiscale without rolling is worse than the lag-limited baseline, so the final gain appears to come from the combination of rich lag context plus temporal summaries rather than from extra lag columns alone
- Practical research implication:
  - the novelty claim is now stronger as `structured multi-scale temporal context` rather than just `longer lag`
  - weekly memory looks more useful than medium-only lag expansion in this setup
