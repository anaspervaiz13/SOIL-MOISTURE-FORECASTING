import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Setup paths
data_dir = r"c:\Users\HP\Desktop\UNI\adv ml assignment 4\SoMo.ml-EU_layer1\layer1"
output_plot_dir = "outputs/plots"
os.makedirs(output_plot_dir, exist_ok=True)

# 2. Load and combine all years for one location
lat_value = 50.0
lon_value = 10.0

all_dfs = []
for year in range(2003, 2021):
    file_path = os.path.join(data_dir, f"SoMo.ml-EU_layer1_{year}.nc")
    if os.path.exists(file_path):
        print(f"Loading {year}...")
        ds = xr.open_dataset(file_path)
        point = ds.sel(lat=lat_value, lon=lon_value, method="nearest")
        df_year = point.to_dataframe().reset_index()
        all_dfs.append(df_year)

df = pd.concat(all_dfs, ignore_index=True)

# 3. Clean up
# The variable name is 'layer1' based on inspection
df = df.rename(columns={"layer1": "soil_moisture"})
df = df[["time", "soil_moisture"]]
df = df.dropna()
df = df.sort_values("time")

print(f"Data for location ({lat_value}, {lon_value}) extracted. Shape: {df.shape}")
print(df.head())

# 6. Plot raw time series (Step 5)
plt.figure(figsize=(12, 4))
plt.plot(df["time"], df["soil_moisture"])
plt.xlabel("Time")
plt.ylabel("Soil Moisture")
plt.title(f"SoMo.ml Soil Moisture Time Series (Lat: {lat_value}, Lon: {lon_value})")
plt.grid(True)
plt.savefig(os.path.join(output_plot_dir, "raw_time_series.png"))
print(f"Plot saved to {output_plot_dir}/raw_time_series.png")

# 7. Create lag features (Step 6)
lags = [1, 2, 3, 7, 14, 30]
for lag in lags:
    df[f"lag_{lag}"] = df["soil_moisture"].shift(lag)

# Add rolling features
df["rolling_mean_7"] = df["soil_moisture"].rolling(7).mean()
df["rolling_std_7"] = df["soil_moisture"].rolling(7).std()
df["rolling_mean_14"] = df["soil_moisture"].rolling(14).mean()

# 8. Create forecasting targets (Step 7)
# 72 hours = 3 days ahead
df["target_3day"] = df["soil_moisture"].shift(-3)

df = df.dropna()

print(f"Feature engineering complete. Final shape: {df.shape}")
print(df.head())

# Save the processed CSV for next steps
df.to_csv("data/processed_full.csv", index=False)
print("Processed data saved to data/processed_full.csv")
