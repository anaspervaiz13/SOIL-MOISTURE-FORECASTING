import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from xgboost import XGBRegressor
import os

# 1. Setup paths
output_plot_dir = "outputs/plots"
os.makedirs(output_plot_dir, exist_ok=True)

# 2. Load data for SHAP
df = pd.read_csv("data/processed_full.csv")
feature_cols = [
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_std_7", "rolling_mean_14"
]
target_col = "target_3day"

train_df = df[df["time"] < "2016-01-01"]
test_df = df[df["time"] >= "2018-01-01"]

X_train, y_train = train_df[feature_cols], train_df[target_col]
X_test = test_df[feature_cols]

# 3. Retrain XGBoost for SHAP (Step 21)
print("Generating SHAP plots for XGBoost...")
xgb = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=4, random_state=42)
xgb.fit(X_train, y_train)

explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test)

# Summary plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Feature Importance for XGBoost")
plt.tight_layout()
plt.savefig(os.path.join(output_plot_dir, "shap_summary.png"))
plt.close()

# 4. Actual vs Predicted Plot (Step 18)
print("Generating Actual vs Predicted plot...")
y_test_final = np.load("outputs/results/y_test_final.npy")
stack_test_pred = np.load("outputs/results/stack_test_pred.npy")

plt.figure(figsize=(15, 6))
plt.plot(y_test_final[:300], label="Actual Soil Moisture", color='blue', alpha=0.7)
plt.plot(stack_test_pred[:300], label="Stacked Prediction", color='orange', linestyle='--')
plt.title("Actual vs Predicted Soil Moisture (First 300 Test Samples)")
plt.xlabel("Days (Test Set)")
plt.ylabel("Soil Moisture")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_plot_dir, "actual_vs_predicted.png"))
plt.close()

# 5. Model Comparison Bar Chart (Step 19)
print("Generating model comparison chart...")
results_df = pd.read_csv("outputs/results/final_model_comparison.csv")

plt.figure(figsize=(10, 5))
plt.bar(results_df["Model"], results_df["RMSE"], color=['gray', 'gray', 'gray', 'green'])
plt.ylabel("RMSE")
plt.title("Model RMSE Comparison (Lower is Better)")
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.join(output_plot_dir, "model_rmse_comparison.png"))
plt.close()

# 6. LSTM Loss Curve (Step 20)
print("Generating LSTM loss curve...")
if os.path.exists("outputs/results/lstm_history.csv"):
    history = pd.read_csv("outputs/results/lstm_history.csv")
    plt.figure(figsize=(10, 5))
    plt.plot(history["loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("LSTM Training History")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_plot_dir, "lstm_loss_curve.png"))
    plt.close()

print(f"All plots saved to {output_plot_dir}")
