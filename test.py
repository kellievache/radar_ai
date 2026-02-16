# %%  <-- This turns the file into Jupyter-like cells in VS Code

# --- Imports ---
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# --- Inputs ---
zarr_path_1 = "/nfs/pancake/u5/projects/vachek/radar_ai/netcdf/y_train_opt.zarr"
zarr_path_2 = "/nfs/pancake/u5/projects/vachek/radar_ai/netcdf/x_train_opt.zarr"

# Choose which logical dimension represents “slices”.
# If your data is e.g. (sample, y, x) or (time, y, x), that dimension is commonly the first.
# If you know the name (e.g., "sample" or "time"), put it here; otherwise leave as None to fall back to the first dimension.
slice_dim_name = None

# 15th slice (0-based index = 14)
slice_index = 100

# --- Helper: open zarr dataset robustly ---
def open_zarr_any(path):
    """
    Try consolidated metadata first (common for zarr),
    fall back to non-consolidated if needed.
    """
    try:
        ds = xr.open_zarr(path, consolidated=True)
    except Exception:
        ds = xr.open_zarr(path, consolidated=False)
    return ds

# --- Open datasets ---
ds1 = open_zarr_any(zarr_path_1)
ds2 = open_zarr_any(zarr_path_2)

# --- Select a primary data variable in each dataset ---
# If each dataset has only one variable, take it; otherwise, pick the first non-coordinate variable.
def pick_primary_var(ds: xr.Dataset) -> xr.DataArray:
    data_vars = list(ds.data_vars)
    if not data_vars:
        raise ValueError(f"No data variables found in dataset at: {ds.encoding.get('source', 'unknown')}")
    # You can customize this to pick a specific variable by name.
    var_name = data_vars[0]
    return ds[var_name]

da1 = pick_primary_var(ds1)
da2 = pick_primary_var(ds2)

# --- Determine the slice dimension ---
if slice_dim_name is not None:
    if slice_dim_name not in da1.dims or slice_dim_name not in da2.dims:
        raise ValueError(f"Specified slice_dim_name='{slice_dim_name}' not found in both datasets.\n"
                         f"da1 dims: {da1.dims}\n da2 dims: {da2.dims}")
    dim = slice_dim_name
else:
    # Default to the first dimension in each array.
    # We’ll also check that both arrays can be sliced along the same-named first dimension;
    # if not, we slice along each array’s first dimension independently.
    dim1 = da1.dims[0]
    dim2 = da2.dims[0]
    if dim1 != dim2:
        # Different leading dimension names—warn but proceed independently.
        dim = None
    else:
        dim = dim1

# --- Extract the 15th slice (index 14) ---
def take_slice(da: xr.DataArray, dimname: str | None, idx: int) -> xr.DataArray:
    if dimname is None:
        # Use first dimension for this array
        dimname = da.dims[0]
    if da.sizes[dimname] <= idx:
        raise IndexError(f"Requested index {idx} is out of bounds for dimension '{dimname}' "
                         f"with size {da.sizes[dimname]}.")
    return da.isel({dimname: idx})

slice1 = take_slice(da1, dim, slice_index)
slice2 = take_slice(da2, dim, slice_index)

# --- Ensure 2D data for plotting ---
# If your slices are 2D (common: (y, x)), this will just work.
# If you have extra singleton dims, squeeze them.
slice1_2d = slice1.squeeze()
slice2_2d = slice2.squeeze()

if slice1_2d.ndim != 2 or slice2_2d.ndim != 2:
    raise ValueError(
        f"Expected 2D slices for plotting, got shapes: slice1={tuple(slice1_2d.shape)}, slice2={tuple(slice2_2d.shape)}.\n"
        f"Consider adjusting 'slice_dim_name' or selecting variables with 2D slices."
    )

# --- Compute common color limits for consistent comparison ---
#vmin = np.nanmin([np.nanmin(slice1_2d.values), np.nanmin(slice2_2d.values)])
#vmax = np.nanmax([np.nanmax(slice1_2d.values), np.nanmax(slice2_2d.values)])

vmin=60
vmax=120

# Optional: use a symmetric range around zero if your data has positive/negative values (e.g., anomalies)
# vmax_abs = max(abs(vmin), abs(vmax))
# vmin, vmax = -vmax_abs, vmax_abs

# --- Plot side by side ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

im0 = axes[0].imshow(slice1_2d.values, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
axes[0].set_title(f"File 1 (15th slice) — var: {slice1_2d.name if slice1_2d.name else 'data'}")
axes[0].set_xlabel(slice1_2d.dims[-1])
axes[0].set_ylabel(slice1_2d.dims[-2])

im1 = axes[1].imshow(slice2_2d.values, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
axes[1].set_title(f"File 2 (15th slice) — var: {slice2_2d.name if slice2_2d.name else 'data'}")
axes[1].set_xlabel(slice2_2d.dims[-1])
axes[1].set_ylabel(slice2_2d.dims[-2])

# Add colorbars (shared limits)
cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
cbar0.set_label("Value")
cbar1.set_label("Value")

plt.show()