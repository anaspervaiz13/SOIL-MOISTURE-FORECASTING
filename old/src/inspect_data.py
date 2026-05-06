import xarray as xr
import os

# Using the 2016 file for inspection
data_path = r"c:\Users\HP\Desktop\UNI\adv ml assignment 4\SoMo.ml-EU_layer1\layer1\SoMo.ml-EU_layer1_2016.nc"

if os.path.exists(data_path):
    ds = xr.open_dataset(data_path)
    print("--- Dataset Info ---")
    print(ds)
    print("\n--- Data Variables ---")
    print(ds.data_vars)
    print("\n--- Coordinates ---")
    print(ds.coords)
    
    # Check for layer/depth
    if 'layer' in ds.dims or 'depth' in ds.dims:
        print("\nLayer/Depth information found.")
else:
    print(f"File not found at {data_path}")
