# LSTM Rescue Design

**Date:** 2026-05-07

## Goal

Rescue the LSTM benchmark so it becomes a methodologically fair, thesis-defensible sequence baseline for 48-hour soil moisture forecasting.

## Problem Summary

The current LSTM underperforms partly because its sequence pipeline is weaker than the tabular pipeline in two important ways:

- sequence windows are built after dropping missing rows, so a `168`-step lookback does not always represent `168` consecutive hourly observations
- the sequence features are not standardized using train-only statistics

These issues make the current LSTM less reliable as a benchmark and weaken its interpretability in the report.

## Rescue Scope

This rescue should remain conservative and easy to defend.

- Keep one simple global LSTM architecture
- Keep the same target and broad feature family
- Do not add embeddings, attention, or highly tuned deep-learning tricks
- Fix only the methodological weaknesses that make the current sequence benchmark unfair

## Design Changes

### 1. Contiguity-safe sequence construction

Sequences must be built only within uninterrupted hourly runs per station. A lookback of `168` should mean `168` consecutive hourly steps, not `168` surviving rows after gap filtering.

### 2. Train-only scaling

Sequence features must be standardized using statistics fitted on the training split only, then reused for validation and test. The target may also be standardized for training stability and inverted before metric reporting.

### 3. Benchmark framing

The rescued LSTM remains a supporting benchmark, not the project novelty. The report framing should state that the model was repaired to ensure fairer sequence evaluation under the same 48-hour forecasting task.

## Success Criteria

- sequence windows never cross station boundaries or timestamp gaps
- scaling uses train-only statistics
- the LSTM pipeline still produces the same style of saved outputs
- the resulting benchmark is more methodologically defensible, even if it does not beat XGBoost
