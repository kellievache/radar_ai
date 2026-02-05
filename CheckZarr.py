#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 09:30:22 2026

@author: vachek
"""

# --- Requirements ---
# pip install xarray zarr dask[complete] matplotlib

import xarray as xr
import matplotlib.pyplot as plt

# Path to your Zarr store (directory or remote store like s3://... if configured)
#zarr_path = "path/to/your_dataset.zarr"
zarr_path = "/scratch/vachek/y_train.zarr"

# Try consolidated metadata first (faster). If it fails, fall back.
try:
    ds = xr.open_zarr(zarr_path, consolidated=True)
except Exception:
    ds = xr.open_zarr(zarr_path, consolidated=False)

# Pick a variable to plot.
# If you know the variable name, set it directly, e.g. var_name = "temperature"
# Otherwise, take the first data variable as a sensible default:
var_name = next(iter(ds.data_vars))

da = ds[var_name]

# Make sure there is a time dimension and that index 300 exists
if "time" not in da.dims:
    raise ValueError(f"Variable '{var_name}' has no 'time' dimension. Dims: {da.dims}")

if da.sizes["time"] <= 300:
    raise IndexError(
        f"'time' dimension too short: size={da.sizes['time']}. Cannot take index 300."
    )

# Select the time slice at index 300
slice_da = da.isel(time=101)
minda=min(slice_da)
maxda=max(slice_da)

# Optional: improve title with a readable timestamp if time coordinate exists
timestamp = None
if "time" in slice_da.coords:
    try:
        timestamp = str(slice_da["time"].item())
    except Exception:
        timestamp = None

# Basic plot (xarray chooses a good default for 2D arrays)
plt.figure(figsize=(8, 5))
p = slice_da.plot(cmap="viridis")  # .plot() handles 1D/2D gracefully
title_time = f" @ {timestamp}" if timestamp else " @ time index 300"
plt.title(f"{var_name}{title_time}")
plt.tight_layout()
plt.show()