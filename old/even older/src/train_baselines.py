import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# 1. Load processed data
df = pd.read_csv("data/processed_full.csv")
df['time'] = pd.to_datetime(df['time'])

# 2. Train / validation / test split (Step 8)
target_col = "target_3day"
train_df = df[df["time"] < "2016-01-01"]
val_df = df[(df["time"] >= "2016-01-01") & (df["time"] < "2018-01-01")]
test_df = df[df["time"] >= "2018-01-01"]

print(f"Split counts: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

# 3. Prepare features (Step 9)
feature_cols = [
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_std_7", "rolling_mean_14"
]

X_train, y_train = train_df[feature_cols], train_df[target_col]
X_val, y_val = val_df[feature_cols], val_df[target_col]
X_test, y_test = test_df[feature_cols], test_df[target_col]

# Normalize for KNN
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 4. Train XGBoost baseline (Step 10)
print("Training XGBoost...")
xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb.fit(X_train, y_train)

xgb_val_pred = xgb.predict(X_val)
xgb_test_pred = xgb.predict(X_test)

# 5. Train KNN baseline (Step 11)
print("Training KNN...")
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

knn_val_pred = knn.predict(X_val_scaled)
knn_test_pred = knn.predict(X_test_scaled)

# 6. Evaluation Function (Step 16)
def evaluate(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"Model": model_name, "RMSE": rmse, "MAE": mae, "R2": r2}

results = []
results.append(evaluate(y_test, xgb_test_pred, "XGBoost"))
results.append(evaluate(y_test, knn_test_pred, "KNN"))

results_df = pd.DataFrame(results)
print("\nBaseline Results:")
print(results_df)

# Save results
os.makedirs("outputs/results", exist_ok=True)
results_df.to_csv("outputs/results/baselines_comparison.csv", index=False)

# Save predictions for stacking alignment later
np.save("outputs/results/xgb_val_pred.npy", xgb_val_pred)
np.save("outputs/results/xgb_test_pred.npy", xgb_test_pred)
np.save("outputs/results/knn_val_pred.npy", knn_val_pred)
np.save("outputs/results/knn_test_pred.npy", knn_test_pred)
np.save("outputs/results/y_val.npy", y_val.values)
np.save("outputs/results/y_test.npy", y_test.values)

print("\nBaselines training complete and results saved.")
