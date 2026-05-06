# 🌱 Soil Moisture Forecasting — Step-by-Step Implementation Walkthrough
### For sharing with peers | Advanced ML Assignment 4
**Authors:** Rayyan Ahmed Minhas (25i-7611) | Anas Pervaiz (25i-7613)

---

## What This Document Is

This is a full walkthrough of how we built a **soil moisture forecasting system** from scratch — every decision, every finding, and why we did what we did. You can use this as a reference if you're building something similar.

---

## 📦 Phase 0: Setup

### What We Started With
- A folder containing 18 NetCDF (`.nc`) files — one per year (2003–2020) — from the **SoMo.ml-EU Layer 1** dataset.
- Total raw data size: ~2.2 GB.
- Goal: Predict soil moisture **3 days ahead** using only past soil moisture values.

### Project Folder Structure Created
```
adv ml assignment 4/
├── SoMo.ml-EU_layer1/layer1/    ← Raw .nc files (do NOT touch)
├── data/
│   └── processed_full.csv       ← Our cleaned dataset (created by us)
├── src/
│   ├── inspect_data.py          ← Step 1: Understand the .nc file
│   ├── process_single_point.py  ← Step 2: Extract + feature engineer
│   ├── train_baselines.py       ← Step 3: Train XGBoost & KNN
│   ├── train_lstm.py            ← Step 4: Train LSTM
│   ├── train_stacking.py        ← Step 5: Stack everything + ablations
│   └── generate_plots.py        ← Step 6: SHAP + all visualizations
├── outputs/
│   ├── plots/                   ← All charts for report
│   └── results/                 ← CSVs + saved .npy prediction arrays
├── RESULTS_AND_DISCUSSION.md
└── IMPLEMENTATION_WALKTHROUGH.md (this file)
```

### Libraries Installed
```bash
conda activate base
pip install xarray netCDF4 pandas numpy matplotlib scikit-learn xgboost tensorflow shap
```

---

## 🔍 Phase 1: Understanding the Dataset

### What We Did
We first inspected one `.nc` file to understand its structure before touching anything else.

```python
import xarray as xr
ds = xr.open_dataset("SoMo.ml-EU_layer1_2016.nc")
print(ds)
```

### What We Found
```
Dimensions:  (time: 366, lat: 355, lon: 570)
Coordinates:
  * time  → 2016-01-01 to 2016-12-31 (daily)
  * lat   → 71.45 to 36.05 (Europe, north to south)
  * lon   → -11.95 to 44.95 (Europe, west to east)
Data variables:
    layer1  (time, lat, lon) float32  ← This is our soil moisture variable!
```

### Key Decision Made Here
- The variable name is **`layer1`** (not `soil_moisture`). We rename it.
- The grid is huge (355 × 570 points). We pick **one location** to keep it manageable: `Lat=50.0, Lon=10.0` (central Germany — good agricultural land).
- **No depth dimension** — this is already the 0–10 cm layer.

---

## 🛠️ Phase 2: Data Extraction & Feature Engineering

### What We Did
We looped through all 18 years, extracted the single point, and combined everything.

```python
for year in range(2003, 2021):
    ds = xr.open_dataset(f"SoMo.ml-EU_layer1_{year}.nc")
    point = ds.sel(lat=50.0, lon=10.0, method="nearest")
    df_year = point.to_dataframe().reset_index()
    all_dfs.append(df_year)

df = pd.concat(all_dfs)
df = df.rename(columns={"layer1": "soil_moisture"})
```

### What We Found
- After combining: **6,575 daily rows** (2003–2020).
- Soil moisture values range from roughly 0.25 to 0.55 (volumetric water content).
- A clear seasonal pattern is visible in the raw time-series plot (saved to `outputs/plots/raw_time_series.png`).

### Feature Engineering
We created **9 input features** from the raw time series:

| Feature | What it captures |
| :--- | :--- |
| `lag_1` | Yesterday's soil moisture |
| `lag_2` | 2 days ago |
| `lag_3` | 3 days ago |
| `lag_7` | 1 week ago |
| `lag_14` | 2 weeks ago |
| `lag_30` | 1 month ago |
| `rolling_mean_7` | 7-day moving average |
| `rolling_std_7` | 7-day variability |
| `rolling_mean_14` | 14-day moving average |

### Target Variable
```python
df["target_3day"] = df["soil_moisture"].shift(-3)  # 72-hour ahead
```

### After dropping NaN rows: **6,542 usable samples**.

---

## ✂️ Phase 3: Train / Validation / Test Split

### The Split (Time-Based — NOT Random)
> ⚠️ For time-series, NEVER use random split. It leaks future data into training.

```python
train_df = df[df["time"] < "2016-01-01"]         # 2003–2015 → 4,718 rows
val_df   = df[(df["time"] >= "2016-01-01") &
              (df["time"] < "2018-01-01")]         # 2016–2017 → 731 rows
test_df  = df[df["time"] >= "2018-01-01"]         # 2018–2020 → 1,093 rows
```

### Why This Split?
- **Train** (13 years): Enough data to capture seasonal patterns.
- **Val** (2 years): Used to tune models and train the meta-learner.
- **Test** (3 years): Completely unseen — final evaluation only.

---

## 🌲 Phase 4: Baseline Models

### 4.1 XGBoost

```python
xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb.fit(X_train, y_train)
```

**Why these settings?**
- `max_depth=4`: Prevents overfitting on tabular lag features.
- `learning_rate=0.03`: Low learning rate with 300 trees = stable convergence.
- `subsample=0.8`: Adds randomness to prevent memorization.

**Result on test set:**
- RMSE: `0.01205` | R²: `0.9541`

### 4.2 KNN

```python
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)   # ← MinMaxScaler required for KNN
```

**Why MinMaxScaler?**
KNN uses Euclidean distance. Without scaling, `lag_30` (range ~0.3) would dominate over `rolling_std_7` (range ~0.05). Scaling ensures fair distance computation.

**Result on test set:**
- RMSE: `0.01834` | R²: `0.8936`

### Baseline Takeaway
XGBoost significantly outperforms KNN. Both predictions are saved to `.npy` files for the stacking step.

---

## 🧠 Phase 5: LSTM Model

### Sequence Construction
LSTM needs sequences, not individual rows. We used a **30-day lookback** window.

```python
def create_sequences(series, lookback=30, horizon=3):
    X, y = [], []
    for i in range(len(series) - lookback - horizon + 1):
        X.append(series[i : i+lookback])        # 30 days of history
        y.append(series[i+lookback+horizon-1])  # day 3 in the future
    return np.array(X), np.array(y)
```

**Shape after reshaping:**
```
Train: (4686, 30, 1)
Val:   (699,  30, 1)
Test:  (1061, 30, 1)
```
> Note: LSTM loses 32 rows at the start of each split (30 lookback + 3 horizon − 1). This is handled in the stacking alignment step.

### Architecture

```python
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(30, 1)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer=Adam(lr=0.001), loss="mse")
```

**Why 2 stacked LSTM layers?**
- First LSTM (64 units) captures broad temporal patterns.
- Second LSTM (32 units) refines the representation.
- Dropout(0.2) prevents overfitting on the training years.

### Training Behavior

```
Epoch 1/100  → val_loss: 0.000833
Epoch 26/100 → val_loss: 0.000389  (learning kicks in)
Epoch 67/100 → val_loss: 0.000316  ← BEST
Epoch 77/100 → Early stopping triggered
```

**Key Observation:** Training and validation loss decrease in parallel — **no overfitting**. The model generalizes well.

**Result on test set:**
- RMSE: `0.01776` | R²: `0.9002`

### Surprising Finding: XGBoost > LSTM
XGBoost's R² (0.954) beat LSTM's (0.900). Why?
- For a **3-day horizon** with strong autocorrelation, tabular lag features give XGBoost everything it needs.
- LSTM's advantage is in longer, more complex patterns. Here, the physics of soil moisture favors the lag-based approach.

---

## 🏗️ Phase 6: Stacked Ensemble

### The Alignment Problem
LSTM outputs are shorter than XGBoost/KNN outputs because of the 30-day lookback. To fix this:

```python
# Take the LAST N predictions from baselines to match LSTM length
xgb_val_aligned = xgb_val_pred[-len(lstm_val_pred):]
knn_val_aligned = knn_val_pred[-len(lstm_val_pred):]
# Same for test set
```

This ensures **all models are evaluated on the exact same time window**.

### Building the Meta-Learner

```python
# Stack the 3 sets of validation predictions as features
stack_X_val = np.column_stack([lstm_val_pred, xgb_val_aligned, knn_val_aligned])

# Meta-learner learns: "How much should I trust each model?"
meta = LinearRegression()
meta.fit(stack_X_val, y_val_aligned)

# Apply to test set
stack_X_test = np.column_stack([lstm_test_pred, xgb_test_aligned, knn_test_aligned])
stack_test_pred = meta.predict(stack_X_test)
```

**Why Linear Regression as meta-learner?**
- Simple, interpretable, and avoids overfitting on meta-features.
- The meta-learner doesn't need to be complex — the base models already did the hard work.

### Final Result: Stacked Ensemble
- RMSE: `0.01124` | R²: `0.9600`
- **Best performance of all models.** ✅

---

## 🔬 Phase 7: Ablation Study

We systematically tested every combination to prove the ensemble adds value:

| Experiment | RMSE | R² | Finding |
| :--- | :--- | :--- | :--- |
| XGBoost alone | 0.01205 | 0.9541 | Strong solo |
| LSTM alone | 0.01776 | 0.9002 | Weaker solo |
| LSTM + XGBoost | **0.01119** | **0.9604** | 🏆 Best pairing |
| LSTM + XGB + KNN | 0.01124 | 0.9600 | KNN adds marginal noise |

### Key Takeaway
Adding KNN actually **slightly worsened** performance. KNN's local-similarity approach is redundant when XGBoost is already capturing non-linear feature interactions. The ablation study proves this definitively.

---

## 📊 Phase 8: SHAP Analysis

SHAP (Shapley Additive Explanations) was applied to XGBoost only.

```python
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

### What SHAP Tells Us

| Feature | Importance | Physical Meaning |
| :--- | :--- | :--- |
| `lag_1` | ⭐⭐⭐⭐⭐ | Soil moisture changes slowly; yesterday is the best predictor |
| `rolling_mean_7` | ⭐⭐⭐⭐ | Weekly trend matters |
| `lag_2` | ⭐⭐⭐ | 2-day memory is strong |
| `lag_7`, `lag_3` | ⭐⭐ | Weekly seasonality |
| `lag_14`, `lag_30` | ⭐ | Diminishing returns at longer lags |

### Why This Is Physically Correct
Soil moisture has a very high **autocorrelation** — it changes gradually unless there's heavy rain or drought. So `lag_1` being dominant makes perfect physical sense. SHAP **proves our model learned real patterns**, not noise.

---

## 📈 Phase 9: Visualizations Generated

All saved to `outputs/plots/`:

| File | What It Shows | Used In Report Section |
| :--- | :--- | :--- |
| `raw_time_series.png` | 2003–2020 soil moisture trend | Dataset Description |
| `actual_vs_predicted.png` | Model predictions vs ground truth | Results |
| `model_rmse_comparison.png` | Bar chart of all model RMSEs | Results |
| `lstm_loss_curve.png` | Training vs validation loss | Methodology/LSTM |
| `shap_summary.png` | Feature importance beeswarm plot | SHAP Interpretability |

---

## ✅ Final Summary

| Step | What We Did | Key Output |
| :--- | :--- | :--- |
| 1 | Inspected NetCDF structure | Variable = `layer1` |
| 2 | Extracted 18-year time series | `processed_full.csv` (6,542 rows) |
| 3 | Built lag + rolling features | 9 input features |
| 4 | Time-based train/val/test split | 4718 / 731 / 1093 rows |
| 5 | Trained XGBoost | RMSE: 0.01205, R²: 0.9541 |
| 6 | Trained KNN | RMSE: 0.01834, R²: 0.8936 |
| 7 | Trained LSTM (77 epochs) | RMSE: 0.01776, R²: 0.9002 |
| 8 | Built Stacked Ensemble | **RMSE: 0.01124, R²: 0.9600** ✅ |
| 9 | Ablation Study | LSTM+XGBoost is the optimal pair |
| 10 | SHAP Analysis | `lag_1` is dominant feature |
| 11 | Generated all plots | 5 publication-ready charts |

---

## 💬 What To Tell Your Examiner

1. **"Why not use weather data?"**
   > The SoMo.ml-EU dataset does not include raw meteorological variables. We focused on temporal autocorrelation-based forecasting, which is a valid and published approach for short-term soil moisture prediction.

2. **"Why is XGBoost better than LSTM?"**
   > For a 3-day horizon with high autocorrelation, lag-based tabular features are sufficient. LSTM's advantage typically emerges with longer, more complex temporal sequences.

3. **"Why Linear Regression as meta-learner?"**
   > Linear regression is an interpretable and regularized choice for combining base model predictions. It avoids overfitting and is standard in stacking literature.

4. **"Isn't your stacking just averaging?"**
   > No — Linear Regression learns **different weights** for each base model based on validation error patterns. It can down-weight a weak model like KNN and amplify a strong one like XGBoost.

5. **"Can this generalize to other locations?"**
   > Not without retraining. Results are specific to Lat: 50.0, Lon: 10.0. A global model would require multi-location training, which is future work.
