# ISMN Lag Design Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new end-to-end soil-moisture forecasting workflow centered on multi-scale temporal lag design, from fresh data preparation through model comparison and results summarization.

**Architecture:** The project will start from the raw ISMN files already present in the active workspace, produce a clean merged hourly dataset, engineer a 48-hour forecasting table with grouped lag features, and then run a sequence of experiments focused first on lag ablation and then on model comparison. Logging and documentation will be updated throughout so the research process stays traceable and the final report can be written from actual evidence.

**Tech Stack:** Python, pandas, NumPy, scikit-learn, XGBoost, PyTorch, Prophet, statsmodels, CSV, JSON, Markdown

---

## File Structure

- `C:\Users\HP\Desktop\UNI\final work\src\build_ismn_merged.py`
  - Raw ISMN parsing, QC filtering, replicate handling, merged hourly dataset generation
- `C:\Users\HP\Desktop\UNI\final work\src\build_ismn_forecasting_dataset.py`
  - Forecast target creation, multi-scale lag features, rolling features, model-ready dataset outputs
- `C:\Users\HP\Desktop\UNI\final work\src\training\config.py`
  - Paths, shared constants, split settings, seeds, reusable experiment configuration
- `C:\Users\HP\Desktop\UNI\final work\src\training\data.py`
  - Dataset loading, temporal splits, filtering, feature/target extraction
- `C:\Users\HP\Desktop\UNI\final work\src\training\feature_sets.py`
  - Feature groups for ablation and benchmark experiments
- `C:\Users\HP\Desktop\UNI\final work\src\training\evaluation.py`
  - Metrics, repeated-run summaries, confidence intervals, result export helpers
- `C:\Users\HP\Desktop\UNI\final work\src\training\sequence.py`
  - Window-building helpers for LSTM/Transformer models
- `C:\Users\HP\Desktop\UNI\final work\src\train_xgboost_48h.py`
  - Main strong tabular benchmark and lag-ablation anchor model
- `C:\Users\HP\Desktop\UNI\final work\src\train_knn_48h.py`
  - Lightweight distance-based comparison
- `C:\Users\HP\Desktop\UNI\final work\src\train_arima_48h.py`
  - Statistical baseline
- `C:\Users\HP\Desktop\UNI\final work\src\train_prophet_48h.py`
  - Trend/seasonality baseline
- `C:\Users\HP\Desktop\UNI\final work\src\train_lstm_48h.py`
  - Sequence deep-learning benchmark
- `C:\Users\HP\Desktop\UNI\final work\src\train_transformer_48h.py`
  - Transformer benchmark
- `C:\Users\HP\Desktop\UNI\final work\src\train_stacking_48h.py`
  - Ensemble layer once base models are methodologically aligned
- `C:\Users\HP\Desktop\UNI\final work\src\summarize_training_results.py`
  - Aggregate final outputs into comparison tables
- `C:\Users\HP\Desktop\UNI\final work\tests\`
  - Validation tests for data building, feature logic, and evaluation utilities
- `C:\Users\HP\Desktop\UNI\final work\logs\project_work_log.md`
  - Step-by-step progress log for the new project
- `C:\Users\HP\Desktop\UNI\final work\docs\superpowers\specs\`
  - Research-direction specs
- `C:\Users\HP\Desktop\UNI\final work\docs\superpowers\plans\`
  - Execution plans

### Task 1: Workspace setup and documentation baseline

**Files:**
- Create: `C:\Users\HP\Desktop\UNI\final work\docs\superpowers\specs\2026-05-06-ismn-lag-design.md`
- Create: `C:\Users\HP\Desktop\UNI\final work\docs\superpowers\plans\2026-05-06-ismn-lag-design-implementation.md`
- Create: `C:\Users\HP\Desktop\UNI\final work\logs\project_work_log.md`

- [ ] Confirm that new work will live only outside `old/`.
- [ ] Write the research-direction spec for the multi-scale lag design.
- [ ] Write the end-to-end implementation plan as a living document.
- [ ] Create a fresh progress log for the new workflow.
- [ ] Record the project restart and scope decision in the new log.

### Task 2: Rebuild the merged hourly dataset from raw data

**Files:**
- Create or modify: `C:\Users\HP\Desktop\UNI\final work\src\build_ismn_merged.py`
- Create or modify: `C:\Users\HP\Desktop\UNI\final work\outputs\ismn\merge_summary.json`
- Modify: `C:\Users\HP\Desktop\UNI\final work\logs\project_work_log.md`

- [ ] Inspect the raw ISMN data under `data\original dataset\`.
- [ ] Reconfirm the quality-control rule for accepted observations.
- [ ] Reconfirm the replicate-handling rule for duplicate sensors.
- [ ] Parse the `.stm` station files into a unified tabular structure.
- [ ] Aggregate replicate measurements to one value per station, variable, depth, and timestamp.
- [ ] Save the merged dataset as `data\processed\ismn_merged_hourly.csv`.
- [ ] Save a summary JSON containing row counts, station counts, time span, and missingness overview.
- [ ] Log each preprocessing decision and final output summary.

### Task 3: Build the new 48-hour forecasting dataset

**Files:**
- Create or modify: `C:\Users\HP\Desktop\UNI\final work\src\build_ismn_forecasting_dataset.py`
- Create or modify: `C:\Users\HP\Desktop\UNI\final work\outputs\ismn\final_dataset_summary.json`
- Modify: `C:\Users\HP\Desktop\UNI\final work\logs\project_work_log.md`

- [ ] Load `data\processed\ismn_merged_hourly.csv`.
- [ ] Consolidate precipitation into one modeling column using a fixed station-safe priority rule.
- [ ] Define `target_48h` as future `sm_0.05m` at the 48-hour horizon within each station.
- [ ] Create calendar features for hour, weekday, month, and cyclic seasonal encoding.
- [ ] Create the grouped lag features for:
  - short-term context
  - medium-term context
  - weekly context
- [ ] Add rolling summary features that use history only.
- [ ] Save a full engineered dataset and a model-ready filtered dataset.
- [ ] Save a JSON summary with shape, station counts, date ranges, and missingness after filtering.
- [ ] Log the exact feature design so the novelty is documented from the start.

### Task 4: Lock the lag-ablation experiment design

**Files:**
- Create or modify: `C:\Users\HP\Desktop\UNI\final work\src\training\feature_sets.py`
- Modify: `C:\Users\HP\Desktop\UNI\final work\src\training\config.py`
- Modify: `C:\Users\HP\Desktop\UNI\final work\logs\project_work_log.md`

- [ ] Define named feature groups for ablation.
- [ ] Create at least these experiment sets:
  - baseline limited-lag setup
  - short-lag setup
  - short-plus-medium setup
  - full multi-scale setup
- [ ] Ensure the groups isolate the impact of lag additions as cleanly as possible.
- [ ] Freeze the ablation naming convention so results stay comparable.
- [ ] Log the final ablation plan before model training begins.

### Task 5: Build the shared training and evaluation utilities

**Files:**
- Create or modify: `C:\Users\HP\Desktop\UNI\final work\src\training\config.py`
- Create or modify: `C:\Users\HP\Desktop\UNI\final work\src\training\data.py`
- Create or modify: `C:\Users\HP\Desktop\UNI\final work\src\training\evaluation.py`
- Create or modify: `C:\Users\HP\Desktop\UNI\final work\tests\test_training_utils.py`
- Create or modify: `C:\Users\HP\Desktop\UNI\final work\tests\test_evaluation_utils.py`
- Modify: `C:\Users\HP\Desktop\UNI\final work\logs\project_work_log.md`

- [ ] Implement reusable dataset loading and chronological split helpers.
- [ ] Standardize train, validation, and test extraction across experiments.
- [ ] Implement metric computation for RMSE, MAE, and R2.
- [ ] Implement repeated-run summary logic where relevant.
- [ ] Add tests for split integrity, feature extraction, and metric summarization.
- [ ] Run the tests and log the validation status.

### Task 6: Run the anchor lag-ablation model

**Files:**
- Create or modify: `C:\Users\HP\Desktop\UNI\final work\src\train_xgboost_48h.py`
- Create or modify: `C:\Users\HP\Desktop\UNI\final work\outputs\training\xgboost\`
- Modify: `C:\Users\HP\Desktop\UNI\final work\logs\project_work_log.md`

- [ ] Use XGBoost as the main anchor model for lag-ablation experiments.
- [ ] Run the limited-lag setup and save predictions and metrics.
- [ ] Run the short-lag setup and save predictions and metrics.
- [ ] Run the short-plus-medium setup and save predictions and metrics.
- [ ] Run the full multi-scale setup and save predictions and metrics.
- [ ] Summarize how performance changes as lag context increases.
- [ ] Log the results and note whether weekly lag appears useful.

### Task 7: Train the core comparison models

**Files:**
- Create or modify: `C:\Users\HP\Desktop\UNI\final work\src\train_knn_48h.py`
- Create or modify: `C:\Users\HP\Desktop\UNI\final work\src\train_lstm_48h.py`
- Create or modify: `C:\Users\HP\Desktop\UNI\final work\src\train_transformer_48h.py`
- Create or modify: `C:\Users\HP\Desktop\UNI\final work\src\training\sequence.py`
- Create or modify: `C:\Users\HP\Desktop\UNI\final work\outputs\training\knn\`
- Create or modify: `C:\Users\HP\Desktop\UNI\final work\outputs\training\lstm\`
- Create or modify: `C:\Users\HP\Desktop\UNI\final work\outputs\training\transformer\`
- Modify: `C:\Users\HP\Desktop\UNI\final work\logs\project_work_log.md`

- [ ] Train KNN as a lightweight nonparametric benchmark.
- [ ] Build sequence windows for the LSTM and Transformer pipelines.
- [ ] Train the LSTM benchmark and export metrics and predictions.
- [ ] Train the Transformer benchmark and export metrics and predictions.
- [ ] Verify that each model uses a valid forecasting setup and a comparable target definition.
- [ ] Log failures, exclusions, or data-coverage caveats if they appear.

### Task 8: Train the statistical baselines carefully

**Files:**
- Create or modify: `C:\Users\HP\Desktop\UNI\final work\src\train_arima_48h.py`
- Create or modify: `C:\Users\HP\Desktop\UNI\final work\src\train_prophet_48h.py`
- Create or modify: `C:\Users\HP\Desktop\UNI\final work\outputs\training\arima\`
- Create or modify: `C:\Users\HP\Desktop\UNI\final work\outputs\training\prophet\`
- Create or modify: `C:\Users\HP\Desktop\UNI\final work\tests\test_arima_utils.py`
- Modify: `C:\Users\HP\Desktop\UNI\final work\logs\project_work_log.md`

- [ ] Implement ARIMA on a methodologically valid observable source series.
- [ ] Implement Prophet on a methodologically valid observable source series.
- [ ] Generate 48-step-ahead predictions without future-label leakage.
- [ ] Save predictions, run metadata, and summary metrics.
- [ ] Add small validation tests for statistical-baseline helper behavior.
- [ ] Log any coverage reduction or station-level exclusions explicitly.

### Task 9: Build stacking only after base-model alignment

**Files:**
- Create or modify: `C:\Users\HP\Desktop\UNI\final work\src\train_stacking_48h.py`
- Create or modify: `C:\Users\HP\Desktop\UNI\final work\outputs\training\stacking\`
- Create or modify: `C:\Users\HP\Desktop\UNI\final work\outputs\training\stacking_variants\`
- Modify: `C:\Users\HP\Desktop\UNI\final work\logs\project_work_log.md`

- [ ] Decide which base models are eligible for stacking based on aligned predictions.
- [ ] Build the first stacker on a common comparable evaluation subset only.
- [ ] Optionally try one or two stacker variants if they remain methodologically clean.
- [ ] Save predictions, metrics, and run metadata for every variant kept.
- [ ] Log why each variant was included or excluded.

### Task 10: Summarize results and build final comparison artifacts

**Files:**
- Create or modify: `C:\Users\HP\Desktop\UNI\final work\src\summarize_training_results.py`
- Create or modify: `C:\Users\HP\Desktop\UNI\final work\outputs\training\model_comparison_summary.csv`
- Modify: `C:\Users\HP\Desktop\UNI\final work\logs\project_work_log.md`

- [ ] Read all saved model summaries from the outputs directory.
- [ ] Build one comparison table using only methodologically compatible runs.
- [ ] Separate or annotate any results that come from different coverage subsets.
- [ ] Generate the summary CSV used later in the report.
- [ ] Log the final comparison scope and any remaining caveats.

### Task 11: Turn experiment outcomes into report-ready research framing

**Files:**
- Modify: `C:\Users\HP\Desktop\UNI\final work\docs\superpowers\specs\2026-05-06-ismn-lag-design.md`
- Modify: `C:\Users\HP\Desktop\UNI\final work\summary.docx`
- Modify: `C:\Users\HP\Desktop\UNI\final work\logs\project_work_log.md`

- [ ] Review whether the results actually support the planned novelty statement.
- [ ] Refine the research gap based on experimental evidence.
- [ ] Write the final contribution points using real findings, not assumptions.
- [ ] Update the methodology wording so it matches what was actually built.
- [ ] Log the final narrative decisions.

### Task 12: Final verification and cleanup

**Files:**
- Modify: `C:\Users\HP\Desktop\UNI\final work\logs\project_work_log.md`

- [ ] Re-run validation commands for core scripts and tests.
- [ ] Recheck that outputs, logs, and summaries match the latest runs.
- [ ] Confirm the active workspace contains the final artifacts for the new work only.
- [ ] Record a final completion summary in the log.
