import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import os

# 1. Load processed data
df = pd.read_csv("data/processed_full.csv")
df['time'] = pd.to_datetime(df['time'])

# 2. Split (Same as baselines)
train_df = df[df["time"] < "2016-01-01"]
val_df = df[(df["time"] >= "2016-01-01") & (df["time"] < "2018-01-01")]
test_df = df[df["time"] >= "2018-01-01"]

# 3. Prepare LSTM sequences (Step 12)
def create_sequences(series, lookback=30, horizon=3):
    X, y = [], []
    values = series.values
    for i in range(len(values) - lookback - horizon + 1):
        X.append(values[i:i+lookback])
        y.append(values[i+lookback+horizon-1])
    return np.array(X), np.array(y)

lookback = 30
horizon = 3

X_lstm_train, y_lstm_train = create_sequences(train_df["soil_moisture"], lookback, horizon)
X_lstm_val, y_lstm_val = create_sequences(val_df["soil_moisture"], lookback, horizon)
X_lstm_test, y_lstm_test = create_sequences(test_df["soil_moisture"], lookback, horizon)

# Reshape to (samples, time_steps, features)
X_lstm_train = X_lstm_train.reshape(-1, lookback, 1)
X_lstm_val = X_lstm_val.reshape(-1, lookback, 1)
X_lstm_test = X_lstm_test.reshape(-1, lookback, 1)

print(f"LSTM sequence shapes: Train={X_lstm_train.shape}, Val={X_lstm_val.shape}, Test={X_lstm_test.shape}")

# 4. Train LSTM (Step 13)
print("Training LSTM...")
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

history = model.fit(
    X_lstm_train, y_lstm_train,
    validation_data=(X_lstm_val, y_lstm_val),
    epochs=100,
    batch_size=32,
    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=1
)

# 5. Predictions
lstm_val_pred = model.predict(X_lstm_val).flatten()
lstm_test_pred = model.predict(X_lstm_test).flatten()

# 6. Align predictions for stacking (Step 14)
# We need to save these for the final stacking step
os.makedirs("outputs/results", exist_ok=True)
np.save("outputs/results/lstm_val_pred.npy", lstm_val_pred)
np.save("outputs/results/lstm_test_pred.npy", lstm_test_pred)
np.save("outputs/results/y_val_lstm.npy", y_lstm_val)
np.save("outputs/results/y_test_lstm.npy", y_lstm_test)

# Save history for loss curve plot later
history_df = pd.DataFrame(history.history)
history_df.to_csv("outputs/results/lstm_history.csv", index=False)

print("\nLSTM training complete and results saved.")
