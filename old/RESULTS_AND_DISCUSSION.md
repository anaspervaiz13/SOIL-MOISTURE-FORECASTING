# Results & Discussion — Soil Moisture Forecasting via Stacked Ensemble
### Advanced Machine Learning — Assignment 4
**Group:** Rayyan Ahmed Minhas (25i-7611) | Anas Pervaiz (25i-7613)

---

## 8. Evaluation Metrics

All models are evaluated on three standard regression metrics computed on the **held-out test set (2018–2020)**:

| Metric | Formula | Interpretation |
| :--- | :--- | :--- |
| **RMSE** | √(Σ(y − ŷ)² / n) | Lower is better; sensitive to large errors |
| **MAE** | Σ\|y − ŷ\| / n | Lower is better; robust to outliers |
| **R²** | 1 − SS_res/SS_tot | Closer to 1 is better; proportion of variance explained |

---

## 9. Results

### 9.1 Final Model Comparison

| Model | RMSE | MAE | R² |
| :--- | :--- | :--- | :--- |
| **Stacked Ensemble** | **0.01124** | **0.00869** | **0.9600** |
| XGBoost | 0.01205 | 0.00917 | 0.9541 |
| LSTM | 0.01776 | 0.01399 | 0.9002 |
| KNN | 0.01834 | 0.01381 | 0.8936 |

> **Core Finding:** The stacked ensemble achieved the best performance with an R² of 0.96 and the lowest
> RMSE of 0.0112, outperforming all individual base learners. This demonstrates that combining
> heterogeneous models improves predictive accuracy by leveraging complementary learning patterns.

### 9.2 Alignment Note

> To ensure fair comparison, predictions from all baseline models were **aligned with the LSTM
> prediction window** after applying the 30-day lookback sequence. Specifically, XGBoost and KNN
> predictions were trimmed to match the number of LSTM test outputs, ensuring all models were
> evaluated on the same temporal window.

---

## 10. Ablation Study

| Model Variant | RMSE | MAE | R² |
| :--- | :--- | :--- | :--- |
| XGBoost Only | 0.01205 | 0.00917 | 0.9541 |
| LSTM Only | 0.01776 | 0.01399 | 0.9002 |
| LSTM + XGBoost | **0.01119** | **0.00866** | **0.9604** |
| LSTM + XGBoost + KNN (Full Stack) | 0.01124 | 0.00869 | 0.9600 |

> **Key Finding:** The ablation study shows that the combination of LSTM and XGBoost yields the best
> performance. The addition of KNN introduces marginal noise and does not significantly improve
> predictive accuracy.

> **Note:** The stacking meta-learner was trained using **validation-based stacking** — predictions
> from the validation set were used to fit the Linear Regression meta-learner. This is an accepted
> approach for coursework-level implementations.

---

## 11. SHAP Interpretability

SHAP analysis was applied to the XGBoost base learner only.

| Rank | Feature | Interpretation |
| :--- | :--- | :--- |
| 1 | `lag_1` (t−1) | Dominant: yesterday's moisture |
| 2 | `rolling_mean_7` | Short-term 7-day trend |
| 3 | `lag_2` (t−2) | Recent 2-day history |
| 4 | `lag_3`, `lag_7` | Weekly patterns |
| 5 | `lag_14`, `lag_30` | Diminishing influence |

> SHAP analysis reveals that the most recent lag (t−1) is the most influential feature, confirming
> the strong temporal persistence of soil moisture. The decreasing importance of higher-order lags
> indicates diminishing temporal influence over longer horizons, which is physically consistent with
> soil moisture dynamics.

---

## 12. Discussion

**XGBoost > LSTM (Standalone):**
> Interestingly, XGBoost outperformed LSTM as a standalone model, indicating that lag-based tabular
> features capture sufficient temporal structure for short-term soil moisture forecasting.

**Ensemble Superiority:**
> Despite XGBoost's individual strength, the stacked ensemble still improved performance, confirming
> that heterogeneous learners capture complementary signal patterns.

**Nonlinear AR Behavior:**
> The model effectively behaves as a nonlinear autoregressive system, where recent soil moisture
> values dominate prediction, and machine learning models enhance the capture of nonlinear dependencies.

---

## 13. Limitations

- Results are specific to the selected geographic location (Lat: 50.0, Lon: 10.0) and cannot be generalized globally without further validation.
- Meta-learner trained using validation-based stacking, not full cross-validated out-of-fold predictions.
- Dataset lacks meteorological covariates, limiting the model to univariate lag-based forecasting.
- Training performed on CPU hardware, constraining LSTM depth and training speed.

---

## 14. Conclusion

The experimental results demonstrate that the proposed stacked ensemble model outperformed all
implemented baseline models on the selected SoMo.ml-EU grid point, achieving an R² of 0.960 and
an RMSE of 0.0112 for 72-hour ahead soil moisture prediction. XGBoost emerged as the strongest
standalone model, indicating the effectiveness of lag-based features. The combination of LSTM and
XGBoost achieved the best results, confirming that integrating temporal sequence learning with
nonlinear regression enhances predictive accuracy. SHAP analysis validates the physical consistency
of the model by highlighting the dominance of recent soil moisture values in forecasting.

---

## LSTM Training Summary

| Metric | Epoch 1 | Epoch 77 (Final) | Best Val Loss |
| :--- | :--- | :--- | :--- |
| Training Loss | 0.00525 | 0.000330 | — |
| Validation Loss | 0.000833 | 0.000335 | **0.000316** (Epoch 67) |

> The LSTM model demonstrates stable convergence with no significant overfitting, as training and
> validation losses decrease consistently and remain closely aligned. Early stopping triggered at
> epoch 77, restoring best weights from epoch 67.
