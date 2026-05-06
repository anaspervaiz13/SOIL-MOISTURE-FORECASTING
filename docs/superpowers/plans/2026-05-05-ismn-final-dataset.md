# ISMN Final Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a final 48-hour-ahead forecasting dataset from the merged ISMN table with consolidated precipitation, temporal features, multi-scale lags, and documented outputs.

**Architecture:** Read the merged hourly ISMN table, create one consolidated precipitation feature, engineer station-wise time-series features with leakage-safe shifts and rolling windows, then save both a full engineered table and a model-ready filtered table. Record each decision and dataset summary in the project log for later thesis justification.

**Tech Stack:** Python, pandas, JSON, CSV

---

### Task 1: Final dataset builder

**Files:**
- Create: `C:\Users\HP\Desktop\UNI\final work\src\build_ismn_forecasting_dataset.py`
- Modify: `C:\Users\HP\Desktop\UNI\final work\logs\ismn_processing_log.md`
- Modify: `C:\Users\HP\Desktop\UNI\final work\docs\2026-05-05-ismn-merge-design.md`

- [ ] Read `data/processed/ismn_merged_hourly.csv`.
- [ ] Create one `precipitation` column with priority:
  - `p_ecotech_rain_gauge`
  - else `p_ott_pluvio2s_amount`
  - else `p_vaisala_wxt510`
- [ ] Drop `p_ott_pluvio2s_volume` from modeling outputs.
- [ ] Sort by `station`, `timestamp`.
- [ ] Create `target_48h` as future `sm_0.05m` shifted by `-48` within each station.
- [ ] Create calendar features:
  - `hour`, `dayofweek`, `month`, `dayofyear`
  - `hour_sin`, `hour_cos`, `doy_sin`, `doy_cos`
- [ ] Create multi-scale lag features for dynamic variables using station-wise grouping:
  - lags: `1, 3, 6, 12, 24, 48, 72, 168`
- [ ] Create leakage-safe rolling features using shifted history only:
  - `sm_0.05m`: rolling mean and std over `6, 24, 48, 168`
  - `precipitation`: rolling sums over `6, 24, 48, 168`
  - `sm_0.20m`, `sm_0.50m`, `ta_2.00m`, `ts_0.05m`, `ts_0.20m`, `ts_0.50m`: rolling means over `24, 48, 168`
- [ ] Save:
  - `data/processed/ismn_forecasting_48h_full.csv`
  - `data/processed/ismn_forecasting_48h_ready.csv`
  - `outputs/ismn/final_dataset_summary.json`

### Task 2: Validation and documentation

**Files:**
- Modify: `C:\Users\HP\Desktop\UNI\final work\logs\ismn_processing_log.md`
- Modify: `C:\Users\HP\Desktop\UNI\final work\docs\2026-05-05-ismn-merge-design.md`

- [ ] Run syntax validation for builder script.
- [ ] Run builder script.
- [ ] Inspect:
  - row counts
  - station counts
  - time span
  - missingness of target and core features
- [ ] Record final dataset summary and filtering outcome in project log.
- [ ] Record 48-hour target definition and novelty-facing feature design in design doc.
