#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 13:23:55 2026

@author: vachek
"""


import xarray as xr
src = "/nfs/pancake/u5/projects/vachek/radar_ai/netcdf/infer.nc"
dst = "/nfs/pancake/u5/projects/vachek/radar_ai/netcdf/infer.zarr"

ds = xr.open_dataset(src, engine="netcdf4", chunks={})
# Choose chunk sizes that align with your crop size
ds = ds.chunk({"time": 72, "lat": 156, "lon": 352})
ds.to_zarr(dst, mode="w", consolidated=True)  # write once
