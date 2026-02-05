#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 15:45:44 2026

@author: vachek
"""


import os
import re
import glob
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
import re


from multiprocessing import Pool, cpu_count
from functools import partial
import traceback


def find_bil_files(root_dir, years=(2009, 2010)):
    """
    Find all .bil files under root_dir for the specified years.
    Returns a list of filepaths.
    """
    paths = []
    if years != 'NoYearDir':
        for y in years:
                pattern1 = os.path.join(root_dir, str(y), "*.bil")
                pattern2 = os.path.join(root_dir, str(y), "*.BIL")
    else:
        pattern1 = os.path.join(root_dir,  "*.bil")
        pattern2 = os.path.join(root_dir,  "*.BIL")            
        paths.extend(glob.glob(pattern1))
        paths.extend(glob.glob(pattern2))
    return paths

def parse_date_from_name(path):
    """
    Extract YYYYMMDD from filename (e.g., 'prism_ppt_us_25m_19890101.bil').
    Returns pandas.Timestamp.
    """
    name = os.path.basename(path)
    m = re.search(r'(\d{8})', name)
    if not m:
        raise ValueError(f"Could not parse date from filename: {name}")
    ymd = m.group(1)
    return pd.to_datetime(ymd, format="%Y%m%d")

def build_lat_lon(transform, height, width, crs):
    """
    Build lon/lat 1D coordinate arrays from rasterio transform.
    If CRSs is geographic (WGS84), return ('lat','lon'), else ('y','x').
    """
    xs = np.arange(width)
    ys = np.arange(height)
    x_coords = transform.c + (xs + 0.5) * transform.a + (0.5) * transform.b
    y_coords = transform.f + (ys + 0.5) * transform.d + (0.5) * transform.e

    is_geographic = False
    try:
        if crs and crs.is_geographic:
            is_geographic = True
    except Exception:
        pass

    if is_geographic:
        coord_names = ("lat", "lon")
        lat = y_coords.copy()
        lon = x_coords.copy()
        return coord_names, lat, lon, is_geographic
    else:
        coord_names = ("y", "x")
        return coord_names, y_coords, x_coords, is_geographic

def stack_to_netcdf(root_dir,
                    years,
                    var_name="ppt",
                    out_path="PRISM_daily_ppt_2009_2010.nc",
                    include_text = "adj_best",
                    chunks={"time": 30, "lat": 1000, "lon": 1000},
                    sentinel=-9999.0):
    """
    Read PRISM BIL rasters for given years and write a NetCDF with a time dimension.
    Nodata is written as the sentinel (-9999) and encoded as _FillValue=-9999.
    """
    print(f"Reading directory {root_dir}.  Looking for {include_text} in bil files there, only for {years}\n")
    # Get all candidate bil files for the years
    all_files = find_bil_files(root_dir, years=years)

    # Keep only .bil files that include 'adj_best' in their basename
    # Pattern: match 8 digits (\d{8}) followed by .bil at the end ($)
    # re.IGNORECASE makes it work for .bil, .BIL, etc.
    pattern = re.compile(r"\d{8}\.bil$", re.IGNORECASE)
    files = [
        f for f in all_files
       # if f.lower().endswith('.bil') and include_text in os.path.basename(f).lower()
        if f.lower().endswith('.bil') and 
            os.path.basename(f).lower().startswith(include_text.lower()) #and
           # pattern.search(os.path.basename(f))
    ]

    if not files:
        raise FileNotFoundError(
            f"No .bil files including {include_text} found in {root_dir} for years {years}"
        )

    # Parse dates and sort
    records = []
    for f in files:
        try:
            ts = parse_date_from_name(f)
            records.append((ts, f))
        except ValueError:
            continue
    if not records:
        raise RuntimeError("No files with recognizable YYYYMMDD in names were found.")
    records.sort(key=lambda t: t[0])

    times = [t for (t, _) in records]
    f0 = records[0][1]

    # Read geospatial metadata from first file
    with rasterio.open(f0) as src0:
        height, width = src0.height, src0.width
        transform = src0.transform
        crs = src0.crs
        nodata = src0.nodatavals[0] if src0.nodatavals and src0.nodatavals[0] is not None else None
        coord_names, y_arr, x_arr, is_geo = build_lat_lon(transform, height, width, crs)

    # Read rasters into array [T, H, W] and preserve sentinel
    data_list = []
    for ts, fp in records:
        with rasterio.open(fp) as src:
            arr = src.read(1)  # [H,W]
            if arr.shape != (height, width):
                raise ValueError(f"Array shape mismatch in {fp}: got {arr.shape}, expected {(height, width)}")

            # If source declares nodata, make sure those are set to our sentinel
            if nodata is not None:
                # Note: some BILs store nodata as -9999 already; this preserves it
                arr = np.where(arr == nodata, sentinel, arr)
            # If any NaNs slipped in, force to sentinel
            arr = np.where(np.isfinite(arr), arr, sentinel)

            data_list.append(arr.astype(np.float32))

    data = np.stack(data_list, axis=0).astype(np.float32)  # [T, H, W]

    # If geographic and latitude is descending, flip to ascending for NetCDF convention
    lat_name, lon_name = coord_names if is_geo else ("y", "x")
    if is_geo:
        lat = y_arr
        lon = x_arr
        if lat[0] > lat[-1]:
            data = data[:, ::-1, :]
            lat = lat[::-1]
    else:
        lat, lon = y_arr, x_arr

    # Ensure any remaining non-finite become sentinel
    data = np.where(np.isfinite(data), data, sentinel).astype(np.float32)

    # Build xarray DataArray
    da = xr.DataArray(
        data,
        dims=("time", lat_name, lon_name),
        coords={
            "time": np.array(times, dtype="datetime64[ns]"),
            lat_name: lat.astype(np.float32),
            lon_name: lon.astype(np.float32),
        },
        name=var_name,
        attrs={
            "long_name": "daily precipitation",
            "units": "mm/day",
            "source": "PRISM BIL stack",
            "missing_value": np.float32(sentinel),  # attr for clarity
        },
    )

    # Attach CRS info safely
    attrs = {}
    if crs is not None:
        try:
            wkt = crs.to_wkt()
            if isinstance(wkt, str) and len(wkt) > 0:
                attrs["crs_wkt"] = wkt
            epsg = crs.to_epsg()
            if epsg is not None:
                attrs["crs_epsg"] = int(epsg)
        except Exception:
            pass
    da.attrs.update(attrs)

    # Dataset + chunking
    ds = da.to_dataset(name=var_name)
    if chunks:
        ds = ds.chunk(chunks)

    # IMPORTANT: Encode with _FillValue = -9999.0
    # Note: many readers (including xarray with decode_cf=True) will convert these to NaN on read.
    comp = dict(zlib=True, complevel=4, shuffle=True, dtype="float32", _FillValue=np.float32(sentinel))
    encoding = {var_name: comp}

    # Write NetCDF
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ds.to_netcdf(out_path, format="NETCDF4", encoding=encoding)
    print(f"Wrote NetCDF → {out_path}")
    print(f"Dimensions: time={ds.dims['time']}, {lat_name}={ds.dims[lat_name]}, {lon_name}={ds.dims[lon_name]}")
    return out_path


TASKS = [
    dict(
        root="/nfs/pancake/prism_current/us/an/ehdr/800m/ppt/daily/normals/",
        years="NoYearDir",
        var_name="ppt",
        out_path="/nfs/pancake/u5/projects/vachek/radar_ai/netcdf/normals.nc",
        include_text="prism_ppt",
    )]

# TASKS = [

#     # y val paths (with radar)
#     dict(
#         root="/nfs/pancake/u4/data/prism/us/an91/r2112/ehdr/800m/ppt/daily/",
#         years=(2023,),
#         var_name="ppt",
#         out_path="/nfs/pancake/u5/projects/vachek/radar_ai/netcdf/y_val.nc",
#         include_text="adj_best_ppt",
#     ),
#     # y train paths (with radar)
#     dict(
#         root="/nfs/pancake/u4/data/prism/us/an81/r1503/ehdr/800m/ppt/daily/",
#         years=(2015, 2019),
#         var_name="ppt",
#         out_path="/nfs/pancake/u5/projects/vachek/radar_ai/netcdf/y_train.nc",
#         include_text="adj_best_ppt",
#     ),    
#     # x train paths (NO radar)
#     dict(
#         root="/nfs/pancake/u4/data/prism/us/an81/r1503/ehdr/800m/ppt/daily/",
#         years=(2015, 2019),
#         var_name="ppt",
#         out_path="/nfs/pancake/u5/projects/vachek/radar_ai/netcdf/x_train.nc",
#         include_text="cai_ppt",
#     ),
#     # x val paths (NO radar)
#     dict(
#         root="/nfs/pancake/u4/data/prism/us/an91/r2112/ehdr/800m/ppt/daily/",
#         years=(2023,),
#         var_name="ppt",
#         out_path="/nfs/pancake/u5/projects/vachek/radar_ai/netcdf/x_val.nc",
#         include_text="cai_ppt",
#     ),
#     # FOR INFERENCE ONLY: pre paths (NO radar; different year)
#     dict(
#         root="/nfs/pancake/u4/data/prism/us/an91/r2112/ehdr/800m/ppt/daily/",
#         years=(2024,),
#         var_name="ppt",
#         out_path="/nfs/pancake/u5/projects/vachek/radar_ai/netcdf/infer.nc",
#         include_text="cai_ppt",
#     ),
# ]

def _ensure_parent_dir(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

def _run_stack_task(task: dict):
    """
    Wrapper to call stack_to_netcdf with logging and error capture.
    """
    root        = task["root"]
    years       = task["years"]
    var_name    = task["var_name"]
    out_path    = task["out_path"]
    include_txt = task["include_text"]

    try:
        _ensure_parent_dir(out_path)
        print(f"[stack_to_netcdf] BEGIN  root={root}  years={years}  var={var_name}  out={out_path}  include='{include_txt}'\n")
        stack_to_netcdf(
            root,
            years=years,
            var_name=var_name,
            out_path=out_path,
            include_text=include_txt
        )
        print(f"[stack_to_netcdf] DONE   out={out_path}")
        return (out_path, True, "")
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[stack_to_netcdf] ERROR  out={out_path}  error={e}\n{tb}")
        return (out_path, False, tb)

def run_all_stacks(tasks, parallel=True, n_workers=None):
    """
    Execute a list of stack tasks in parallel or sequentially.
    """
    if not parallel:
        results = []
        for t in tasks:
            results.append(_run_stack_task(t))
        return results

    if n_workers is None:
        n_workers = len(TASKS)

    print(f"[stack_to_netcdf] Running {len(tasks)} task(s) with {n_workers} workers...")
    with Pool(processes=n_workers) as pool:
        results = pool.map(_run_stack_task, tasks, chunksize=1)
    return results

# ---- Toggle here ----
if __name__ == "__main__":
    PARALLEL = False
    results = run_all_stacks(TASKS, parallel=PARALLEL, n_workers=len(TASKS))
    # Optional: summarize
    ok = sum(1 for _, success, _ in results if success)
    print(f"[stack_to_netcdf] Completed: {ok}/{len(results)} succeeded")
    for out_path, success, err in results:
        if not success:
            print(f"  FAILED → {out_path}")
    print("Done")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
