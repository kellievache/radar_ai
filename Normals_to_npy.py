#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 13:12:48 2026

@author: vachek
"""

import argparse
import numpy as np
import xarray as xr

def infer_var_name(ds: xr.Dataset, prefer: str | None = None) -> str:
    if prefer and prefer in ds.data_vars:
        return prefer
    if not ds.data_vars:
        raise ValueError("No data variables found in the NetCDF.")
    return list(ds.data_vars)[0]

def find_day_dim(da: xr.DataArray) -> str:
    # Prefer common names if present
    for name in ("time", "dayofyear", "doy"):
        if name in da.dims:
            return name
    # Otherwise assume the first dimension is the daily axis
    return da.dims[0]

def main():
    ap = argparse.ArgumentParser(description="Convert 366-day daily normals NetCDF to .npy")
    ap.add_argument("--nc_path", required=True, help="Input NetCDF with 366-day daily normals")
    ap.add_argument("--var", default=None, help="Variable name (e.g., 'ppt'); defaults to first data var")
    ap.add_argument("--out_npy", required=True, help="Output .npy path for normals [366,H,W]")
    ap.add_argument("--out_mask_npy", default=None, help="Optional .npy domain mask [H,W] (any-day finite)")
    args = ap.parse_args()

    ds = xr.open_dataset(args.nc_path)
    var = infer_var_name(ds, args.var)
    da = ds[var]

    day_dim = find_day_dim(da)
    if da.sizes[day_dim] != 366:
        raise ValueError(f"Expected 366 days on '{day_dim}', found {da.sizes[day_dim]}")

    # Reorder so daily axis is first (366, H, W)
    other_dims = [d for d in da.dims if d != day_dim]
    if len(other_dims) != 2:
        raise ValueError(f"Expected 2 spatial dims, found {other_dims}")
    da_ordered = da.transpose(day_dim, other_dims[0], other_dims[1])

    # Load to numpy and cast
    normals = da_ordered.load().values.astype(np.float32)   # shape (366, H, W)

    # Save normals
    np.save(args.out_npy, normals)

    # Optional domain mask (any-day finite)
    if args.out_mask_npy:
        domain_mask = np.isfinite(normals).any(axis=0).astype(np.float32)  # [H,W]
        np.save(args.out_mask_npy, domain_mask)

    print(f"Saved normals: {args.out_npy} with shape {normals.shape} (expected (366, H, W))")
    if args.out_mask_npy:
        print(f"Saved domain mask: {args.out_mask_npy} with shape {domain_mask.shape}")

if __name__ == "__main__":
    main()
    
    
    
# python Normals_to_npy.py \
#   --nc_path /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/normals.nc \
#   --var ppt \
#   --out_npy /scratch/vachek/prism_daily_normals_366.npy \
#   --out_mask_npy /scratch/vachek/prism_domain_mask.npy
