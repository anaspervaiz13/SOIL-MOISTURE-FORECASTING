# Novelty Results Summary

## Main Result

- Best lag-design result: `full_multiscale` with `RMSE 0.029122`, `MAE 0.018647`, `R2 0.854585`.
- Baseline lag result: `baseline_limited` with `RMSE 0.029588`.
- Absolute RMSE gain of best lag design over baseline: `0.000466`.
- Relative RMSE gain of best lag design over baseline: `1.574%`.

## Benchmark Context

- XGBoost reference result: `RMSE 0.028472` on `9557` unique test keys.
- Stacking reference result: `RMSE 0.028846` on `9557` aligned unique test keys.
- Benchmark rows are descriptive and include coverage columns because the current models do not all share the same effective test population.

## Interpretation

- The strongest evidence for novelty is the XGBoost lag-design ablation, not the stacking result.
- Multi-scale lag design shows a small but measurable benefit for the strongest tree-based model over simpler lag-group alternatives.
- The aligned stacking experiment is supplementary and mostly tracks XGBoost rather than adding a strong new ensemble effect.
- Cross-model rankings should be stated carefully until a common evaluation subset is exported for every model.
- This supports the project direction: the main contribution is temporal lag design for medium-horizon soil moisture forecasting, not stacking.
