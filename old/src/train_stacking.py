import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# 1. Load all saved predictions
xgb_val_pred = np.load("outputs/results/xgb_val_pred.npy")
xgb_test_pred = np.load("outputs/results/xgb_test_pred.npy")
knn_val_pred = np.load("outputs/results/knn_val_pred.npy")
knn_test_pred = np.load("outputs/results/knn_test_pred.npy")

lstm_val_pred = np.load("outputs/results/lstm_val_pred.npy")
lstm_test_pred = np.load("outputs/results/lstm_test_pred.npy")

y_val_aligned = np.load("outputs/results/y_val_lstm.npy")
y_test_aligned = np.load("outputs/results/y_test_lstm.npy")

# 2. Align baseline predictions with LSTM length
# LSTM loses the first 30+ rows due to lookback. 
# We take the LAST N samples of baselines to match LSTM output.
xgb_val_aligned = xgb_val_pred[-len(lstm_val_pred):]
knn_val_aligned = knn_val_pred[-len(lstm_val_pred):]

xgb_test_aligned = xgb_test_pred[-len(lstm_test_pred):]
knn_test_aligned = knn_test_pred[-len(lstm_test_pred):]

print(f"Aligned validation samples: {len(y_val_aligned)}")
print(f"Aligned test samples: {len(y_test_aligned)}")

# 3. Create stacking feature sets
stack_X_val = np.column_stack([
    lstm_val_pred,
    xgb_val_aligned,
    knn_val_aligned
])

stack_X_test = np.column_stack([
    lstm_test_pred,
    xgb_test_aligned,
    knn_test_aligned
])

# 4. Train Meta-Learner (Step 15)
print("Training Stacked Meta-Learner (Linear Regression)...")
meta = LinearRegression()
meta.fit(stack_X_val, y_val_aligned)

stack_test_pred = meta.predict(stack_X_test)

# 5. Evaluation (Step 16)
def evaluate(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"Model": model_name, "RMSE": rmse, "MAE": mae, "R2": r2}

results = []
results.append(evaluate(y_test_aligned, lstm_test_pred, "LSTM"))
results.append(evaluate(y_test_aligned, xgb_test_aligned, "XGBoost"))
results.append(evaluate(y_test_aligned, knn_test_aligned, "KNN"))
results.append(evaluate(y_test_aligned, stack_test_pred, "Stacked Ensemble"))

results_df = pd.DataFrame(results)
print("\nFinal Model Comparison:")
print(results_df)

# 6. Save final results and predictions
results_df.to_csv("outputs/results/final_model_comparison.csv", index=False)
np.save("outputs/results/stack_test_pred.npy", stack_test_pred)
np.save("outputs/results/y_test_final.npy", y_test_aligned)

# 7. Ablation Study Table (Step 22)
# Here we simulate the ablations by evaluating subsets
ablation_results = []
ablation_results.append(evaluate(y_test_aligned, xgb_test_aligned, "XGBoost Only"))
ablation_results.append(evaluate(y_test_aligned, lstm_test_pred, "LSTM Only"))

# LSTM + XGBoost stack (mini-meta)
meta_mini = LinearRegression()
meta_mini.fit(stack_X_val[:, :2], y_val_aligned)
mini_pred = meta_mini.predict(stack_X_test[:, :2])
ablation_results.append(evaluate(y_test_aligned, mini_pred, "LSTM + XGBoost"))

ablation_results.append(evaluate(y_test_aligned, stack_test_pred, "LSTM + XGBoost + KNN (Full Stack)"))

ablation_df = pd.DataFrame(ablation_results)
ablation_df.to_csv("outputs/results/ablation_study.csv", index=False)
print("\nAblation Study Results:")
print(ablation_df)

print("\nStacking and Ablation study complete.")
