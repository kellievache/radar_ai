#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 13:23:55 2026

@author: vachek
"""


import xarray as xr
src = "/nfs/pancake/u5/projects/vachek/radar_ai/netcdf/x_train.nc"
dst = "/nfs/pancake/u5/projects/vachek/radar_ai/netcdf/x_train_opt.zarr"

#lat=156 lon=352
lat = 1024
lon = 2048

ds = xr.open_dataset(src, engine="netcdf4", chunks={})
# Choose chunk sizes that align with your crop size
ds = ds.chunk({"time": 1, "lat": lat, "lon": lon})
ds.to_zarr(dst, mode="w", consolidated=True)  # write once

src = "/nfs/pancake/u5/projects/vachek/radar_ai/netcdf/y_train.nc"
dst = "/nfs/pancake/u5/projects/vachek/radar_ai/netcdf/y_train_opt.zarr"

ds = xr.open_dataset(src, engine="netcdf4", chunks={})
# Choose chunk sizes that align with your crop size
ds = ds.chunk({"time": 1, "lat": lat, "lon": lon})
ds.to_zarr(dst, mode="w", consolidated=True)  # write once

src = "/nfs/pancake/u5/projects/vachek/radar_ai/netcdf/x_val.nc"
dst = "/nfs/pancake/u5/projects/vachek/radar_ai/netcdf/x_val_opt.zarr"

ds = xr.open_dataset(src, engine="netcdf4", chunks={})
# Choose chunk sizes that align with your crop size
ds = ds.chunk({"time": 1, "lat": lat, "lon": lon})
ds.to_zarr(dst, mode="w", consolidated=True)  # write once

src = "/nfs/pancake/u5/projects/vachek/radar_ai/netcdf/y_val.nc"
dst = "/nfs/pancake/u5/projects/vachek/radar_ai/netcdf/y_val_opt.zarr"

ds = xr.open_dataset(src, engine="netcdf4", chunks={})
# Choose chunk sizes that align with your crop size
ds = ds.chunk({"time": 1, "lat": lat, "lon": lon})
ds.to_zarr(dst, mode="w", consolidated=True)  # write once

