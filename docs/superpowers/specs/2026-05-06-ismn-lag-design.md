# ISMN Multi-Scale Lag Design Spec

**Date:** 2026-05-06

## Purpose

This project restarts the current soil-moisture forecasting work from a new research direction. The old work remains archived under `old/` for reference only and must not be reused as implementation content. The new work will be built from scratch in the active workspace, while keeping the same discipline of planning, logging, and step-by-step documentation.

## Main Research Direction

The central novelty of this project is not simply using a larger lag window. The main idea is to study a **multi-scale temporal lag design** for 48-hour soil-moisture forecasting.

The core question is:

Does combining short-term, medium-term, and weekly historical context improve 48-hour-ahead soil-moisture prediction compared with simpler lag setups?

## Research Positioning

The project should be framed as a study of temporal context design for medium-horizon forecasting, supported by benchmarking across multiple model families.

The main claim should be:

We propose and evaluate a multi-scale temporal lag feature framework for 48-hour soil-moisture forecasting, and analyze the contribution of short-, medium-, and weekly-history features.

The supporting claim can be:

In this setting, lightweight machine-learning models may benefit more from structured temporal features than heavier deep-learning models.

## Forecasting Scope

- Forecast target: `sm_0.05m`
- Forecast horizon: `48` hours ahead
- Data source: ISMN station data already available in the current workspace
- Stations currently in scope:
  - Gevenich
  - Merzenhausen
  - Selhausen

## Temporal Design

The lag design will be treated as grouped temporal context rather than a flat list of history values.

- Short-term lags:
  - `1, 3, 6, 12, 24`
- Medium-term lags:
  - `48, 72`
- Long-term lag:
  - `168`

Rolling and calendar features may support the study, but the main novelty must remain the structured lag design and the evidence around it.

## Experimental Intent

The project must do more than train many models. It must show which temporal context matters.

The experiments should therefore include:

- a clean data-preparation pipeline
- multi-scale lag feature generation
- ablation experiments on lag groups
- one strong tabular baseline as an anchor model
- additional benchmark models for comparison
- a final comparison grounded in methodologically valid evaluation

## Success Criteria

This direction is successful if the final work can support the following points:

- the new dataset and feature pipeline are documented clearly
- lag groups are defined intentionally, not chosen randomly
- ablation results show whether medium and weekly lags help
- evaluation is methodologically defensible
- the final narrative is based on actual results, not only planned claims

## Constraints

- Do not copy implementation or written content from `old/`.
- Keep the planning and logging discipline from the previous workflow.
- The plan must remain editable as the research direction sharpens.
- The write-up should be finalized only after results support the claims.

## Working Execution Strategy

The work will proceed in this order:

1. Create a fresh spec, plan, and log in the active workspace.
2. Prepare the new data pipeline from the current raw data.
3. Build the multi-scale lag dataset for 48-hour forecasting.
4. Run lag-focused ablation experiments.
5. Train benchmark and comparison models.
6. Summarize results and refine the final research gap and contributions.
