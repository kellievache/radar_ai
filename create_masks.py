#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 11:27:03 2026

@author: vachek
"""


import xarray as xr, numpy as np
ds = xr.open_zarr("/scratch/vachek/y_train.zarr/", consolidated=True)
mask_any = np.isfinite(ds["ppt"]).any(dim="time").values.astype(np.float32)
clim = ds["ppt"].where(np.isfinite(ds["ppt"])).mean(dim="time").fillna(0.0).values.astype(np.float32)
np.save("/scratch/vachek/domain_mask.npy", mask_any)
np.save("/scratch/vachek/clim.npy", clim)
