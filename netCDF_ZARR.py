#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 13:23:55 2026

@author: vachek
"""


import xarray as xr
src = "/nfs/pancake/u5/projects/vachek/radar_ai/netcdf/x_train.nc"
dst = "/nfs/pancake/u5/projects/vachek/radar_ai/netcdf/x_train1.zarr"

ds = xr.open_dataset(src, engine="netcdf4", chunks={})
# Choose chunk sizes that align with your crop size
ds = ds.chunk({"time": 1, "lat": 156, "lon": 352})
ds.to_zarr(dst, mode="w", consolidated=True)  # write once

src = "/nfs/pancake/u5/projects/vachek/radar_ai/netcdf/y_train.nc"
dst = "/nfs/pancake/u5/projects/vachek/radar_ai/netcdf/y_train1.zarr"

ds = xr.open_dataset(src, engine="netcdf4", chunks={})
# Choose chunk sizes that align with your crop size
ds = ds.chunk({"time": 1, "lat": 156, "lon": 352})
ds.to_zarr(dst, mode="w", consolidated=True)  # write once

src = "/nfs/pancake/u5/projects/vachek/radar_ai/netcdf/x_val.nc"
dst = "/nfs/pancake/u5/projects/vachek/radar_ai/netcdf/x_val1.zarr"

ds = xr.open_dataset(src, engine="netcdf4", chunks={})
# Choose chunk sizes that align with your crop size
ds = ds.chunk({"time": 1, "lat": 156, "lon": 352})
ds.to_zarr(dst, mode="w", consolidated=True)  # write once

src = "/nfs/pancake/u5/projects/vachek/radar_ai/netcdf/y_val.nc"
dst = "/nfs/pancake/u5/projects/vachek/radar_ai/netcdf/y_val1.zarr"

ds = xr.open_dataset(src, engine="netcdf4", chunks={})
# Choose chunk sizes that align with your crop size
ds = ds.chunk({"time": 1, "lat": 156, "lon": 352})
ds.to_zarr(dst, mode="w", consolidated=True)  # write once

