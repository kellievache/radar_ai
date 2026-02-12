#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convective morphology correction for PRISM (PyTorch).
- DDP-stable training loop (fixed steps per epoch, syncs, broadcasts)
- Zarr/NetCDF lazy reading with crop windows
- Residual U-Net with texture-aware losses (HP, spectral, FSS)
"""

import os
import json
import random
import argparse
import warnings
from dataclasses import dataclass, asdict, fields
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import time
import torch.multiprocessing as mp
# --- add at top of file if not already there
import torch.nn.functional as F
try:
    import xarray as xr  # NetCDF/Zarr I/O
except ImportError:
    xr = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **k: x  # fallback: identity

from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from collections import OrderedDict
import datetime as _dt

# Silence noisy "Mean of empty slice" warnings globally
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

# ---------------------------
# DDP helpers
# ---------------------------
def ddp_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def ddp_setup():
    """Initialize DDP if launched with torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        import datetime as dt
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=dt.timedelta(minutes=30)
        )
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank, dist.get_rank(), dist.get_world_size()
    else:
        return None, 0, 1  # single-process fallback

def ddp_cleanup():
    if ddp_is_initialized():
        dist.destroy_process_group()

def ddp_broadcast_scalar(val, device, dtype=torch.float32):
    """Broadcast a scalar from rank 0 to all ranks."""
    if not ddp_is_initialized():
        return val
    t = torch.tensor([val], device=device, dtype=dtype)
    dist.broadcast(t, src=0)
    if dtype in (torch.int32, torch.int64, torch.bool):
        return int(t.item())
    return float(t.item())

# ---------------------------
# Reproducibility & utilities
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def sanitize_cfg_dict(cfg_dict: dict) -> dict:
    """Keep only keys present in current Config schema."""
    try:
        valid = {f.name for f in fields(Config)}
        return {k: v for k, v in cfg_dict.items() if k in valid}
    except Exception:
        return cfg_dict or {}

def valid_mask(arr: np.ndarray, sentinel: float = -9999.0) -> np.ndarray:
    """1 for valid (finite & not sentinel), 0 for missing."""
    m = np.isfinite(arr).astype(np.float32)
    m = np.where(arr == sentinel, 0.0, m).astype(np.float32)
    return m

def apply_residual_gate_in_tr_space(
    X_tr: torch.Tensor,
    delta_tr: torch.Tensor,
    M: torch.Tensor,
    c: float = 1.0,      # center in transform space; ~log1p(1 mm)
    beta: float = 1.0,   # slope of the transition
    alpha_bg: float = 1.0,  # background weight outside mask
) -> torch.Tensor:
    """
    X_tr:   [B,1,H,W]   first channel of X in transform space
    delta_tr: [B,1,H,W] residual predicted by the model (also in transform space)
    M:      [B,1,H,W]   mask (1 valid; 0 invalid/outside domain)
    Returns: Y_pred_tr in transform space
    """
    residual_weight = torch.sigmoid(beta * (c - X_tr))
    residual_weight = residual_weight * M + (1.0 - M) * alpha_bg
    return X_tr + residual_weight * delta_tr

# ---------------------------
# Zarr/NetCDF helpers
# ---------------------------
def _is_zarr_store(path: str) -> bool:
    if path.endswith(".zarr"):
        return True
    if os.path.isdir(path) and os.path.exists(os.path.join(path, ".zmetadata")):
        return True
    return False

def open_multi_auto(paths: List[str], time_dim: str, prefer_chunks_time1: bool = True):
    """
    Open paths that may be NetCDF (.nc) or Zarr (.zarr), returning an xarray.Dataset.
    For Zarr we rely on native chunks to avoid splitting warnings.
    """
    if len(paths) == 1:
        p = paths[0]
        if _is_zarr_store(p):
            return xr.open_zarr(p, consolidated=True)
        else:
            return xr.open_dataset(p, engine="netcdf4")
    all_zarr = all(_is_zarr_store(p) for p in paths)
    if all_zarr:
        return xr.open_mfdataset(
            paths, engine="zarr", combine="by_coords",
            backend_kwargs={"consolidated": True}
        )
    else:
        return xr.open_mfdataset(paths, engine="netcdf4", combine="by_coords")

# ---------------------------
# Checkpoint helpers
# ---------------------------
def get_state_dict(model: nn.Module) -> dict:
    return model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()

def _expected_in_channels_from_state(state: dict) -> Optional[int]:
    k = "inc.block.0.weight"
    if k not in state and ("module." + k) in state:
        k = "module." + k
    if k not in state:
        return None
    try:
        return int(state[k].shape[1])
    except Exception:
        return None

def load_weights_robust(model: nn.Module, ckpt_obj: dict) -> None:
    state = ckpt_obj["model"] if isinstance(ckpt_obj, dict) and "model" in ckpt_obj else ckpt_obj
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model_state = model.state_dict()
    filtered = {}
    skipped_shape = []
    for k, v in state.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                filtered[k] = v
            else:
                skipped_shape.append((k, tuple(v.shape), tuple(model_state[k].shape)))
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"[load_weights_robust] Loaded {len(filtered)}/{len(state)} tensors into model.")
    if missing:
        print(f"[load_weights_robust] Missing (not in ckpt): {len(missing)} keys (showing up to 10)")
        for k in missing[:10]:
            print("  -", k)
        if len(missing) > 10:
            print("  ...")
    if unexpected:
        print(f"[load_weights_robust] Unexpected (ignored): {len(unexpected)} keys (showing up to 10)")
        for k in unexpected[:10]:
            print("  -", k)
        if len(unexpected) > 10:
            print("  ...")
    if skipped_shape:
        print(f"[load_weights_robust] Skipped due to shape mismatch: {len(skipped_shape)}")
        for k, s_ckpt, s_model in skipped_shape[:8]:
            print(f"  - {k}: ckpt{s_ckpt} vs model{s_model}")
        if len(skipped_shape) > 8:
            print("  ...")

# ---------------------------
# Config
# ---------------------------
@dataclass
class Config:
    # Data
    prism_train_paths: List[str] = None
    prism_val_paths: List[str] = None
    prism_var: str = "ppt"
    static_dem_path: Optional[str] = None
    static_mask_path: Optional[str] = None
    lat_name: str = "lat"
    lon_name: str = "lon"
    time_name: str = "time"

    # Transform
    transform: str = "log1p"  # 'log1p'|'sqrt'|'none'

    # Degradation (emulate no-radar)
    degrade_gauss_sigma_min: float = 1.0
    degrade_gauss_sigma_max: float = 3.0
    degrade_spectral_cutoff_min: float = 0.08
    degrade_spectral_cutoff_max: float = 0.25
    degrade_intensity_damp_min: float = 0.85
    degrade_intensity_damp_max: float = 0.98
    degrade_additive_noise_std: float = 0.05
    apply_degrade_prob: float = 1.0

    # Training
    epochs: int = 200
    min_epochs: int = 10
    batch_size: int = 8    # per-GPU in DDP
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    base_ch: int = 96
    num_workers: int = 8
    amp: bool = True
    patience: int = 30

    # Dataloader tuning
    val_num_workers: int = 8
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = False

    # Loss weights
    loss: str = "huber"     # 'l1'|'huber'
    w_grad: float = 0.15
    w_spec: float = 0.08
    w_mass: float = 0.03
    w_hp: float = 0.02
    w_fss: float = 0.15

    # FSS thresholds (mm/day)
    fss_thresholds: List[float] = None
    fss_window: int = 9

    # Static features usage
    use_dem: bool = False
    use_mask: bool = False

    # Tiling for inference
    tile: int = 512
    tile_overlap: int = 192

    # Files
    out_dir: str = "/nfs/pancake/u5/projects/vachek/automate_qc/runs/convective_correction/"
    ckpt_name: str = "best.pt"
    seed: int = 42
    device: str = "cuda"

    # Optional external X
    x_train_paths: List[str] = None
    x_val_paths:   List[str] = None
    x_pre_paths:   List[str] = None
    x_var: Optional[str] = None

    # Training from spatial crops
    patch: int = 768
    train_crops_per_epoch: int = 20000
    #use tile to fix tiling so validation uses the same spatial windows.  This helps reduce high epoch variance in val metrics and makes it easier to see real trends.
    val_crop_mode: str = "tile"
    val_crops_per_epoch: int = 4000
    #match tile size for validation if using tile mode, otherwise use patch size.  This is just to keep the number of crops per epoch consistent and avoid high variance in val metrics.
    val_tile: Optional[int] = 896
    val_tile_overlap: Optional[int] = 192

    # Do heavy degr ops on GPU in train loop instead of dataset
    degrade_in_dataset: bool = False

    # NEW: time-slice cache size (planes per variable to keep per rank)
    timeslice_cache: int = 3

    domain_mask_npy: Optional[str] = None
    climatology_npy: Optional[str] = None

    gate_c: float = 1.4
    gate_beta: float = 1.5
    gate_alpha_bg: float = 1.0

    coarsen_factor = 1
    coarsen_mode = "mean_max"      # adds both mean and max precip as input channels
    coarsen_mask_threshold = 0.5
    precip_log1p = True
    biased_crops: bool = True
    biased_policy: str = "precip"
    biased_warmup_epochs: int = 2

    
    num_levels: int = 4                 # 4 (current) or 5 (deeper receptive field)
    deep_supervision: bool = False      # turn on aux heads & losses
    ds_weights: List[float] = None      # weights for aux heads, e.g., [0.2, 0.


    def __post_init__(self):
        if self.fss_thresholds is None:
            self.fss_thresholds = [1.0, 5.0, 10.0, 20.0]

        if self.ds_weights is None:
            self.ds_weights = [0.2, 0.1]  # sensible default (two aux 

# ---------------------------
# Degradation ops
# ---------------------------
def gaussian_blur2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    radius = int(3 * sigma)
    ksize = 2 * radius + 1
    coords = torch.arange(ksize, device=x.device) - radius
    kernel1d = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    kernel1d = kernel1d / kernel1d.sum()
    c = x.shape[1]
    kx = kernel1d.view(1, 1, 1, ksize).repeat(c, 1, 1, 1)
    ky = kernel1d.view(1, 1, ksize, 1).repeat(c, 1, 1, 1)
    x = F.pad(x, (radius, radius, radius, radius), mode='reflect')
    x = F.conv2d(x, kx, groups=c)
    x = F.conv2d(x, ky, groups=c)
    return x

def spectral_lowpass(x: torch.Tensor, cutoff_frac: float) -> torch.Tensor:
    if cutoff_frac >= 1.0:
        return x
    B, C, H, W = x.shape
    X = torch.fft.rfft2(x, norm='ortho')
    ky = torch.fft.fftfreq(H, d=1.0).to(x.device)
    kx = torch.fft.rfftfreq(W, d=1.0).to(x.device)
    KY, KX = torch.meshgrid(ky, kx, indexing='ij')
    R = torch.sqrt(KX**2 + KY**2)
    rmax = 0.5
    mask = (R <= (cutoff_frac * rmax)).float()
    mask = mask.view(1, 1, H, W // 2 + 1)
    X_f = X * mask
    x_f = torch.fft.irfft2(X_f, s=(H, W), norm='ortho')
    return x_f

def apply_degradation(y_mm: torch.Tensor, cfg: Config) -> torch.Tensor:
    x = y_mm
    if random.random() <= cfg.apply_degrade_prob:
        sigma = random.uniform(cfg.degrade_gauss_sigma_min, cfg.degrade_gauss_sigma_max)
        x = gaussian_blur2d(x, sigma)
        cutoff = random.uniform(cfg.degrade_spectral_cutoff_min, cfg.degrade_spectral_cutoff_max)
        x = spectral_lowpass(x, cutoff)
        damp = random.uniform(cfg.degrade_intensity_damp_min, cfg.degrade_intensity_damp_max)
        x = x * damp
    return torch.clamp(x, min=0.0)

def highpass(z: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    return z - gaussian_blur2d(z, sigma)

# ---------------------------
# Transforms
# ---------------------------
def mm_to_transform(mm: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "log1p":
        return torch.log1p(torch.clamp(mm, min=0.0))
    elif kind == "sqrt":
        return torch.sqrt(torch.clamp(mm, min=0.0))
    else:
        return mm

def transform_to_mm(z: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "log1p":
        return torch.expm1(z)
    elif kind == "sqrt":
        return torch.clamp(z, min=0.0) ** 2
    else:
        return z

def add_noise_transformed(z: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return z
    return z + torch.randn_like(z) * std

# ---------------------------
# Distributed Data Parallel
# ---------------------------

def _cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def _barrier(device):
    if ddp_is_initialized():
        if device.type == "cuda":
            dist.barrier(device_ids=[device.index])
        else:
            dist.barrier()

# ---------------------------
# Aggregation
# ---------------------------            

def _block_reduce_mean_max(arr: np.ndarray, f: int):
    """
    Coarsen 2D array by factor f using block mean and block max.
    Returns (mean2d, max2d) as float32.
    Trims bottom/right edges if not divisible by f.
    """
    H, W = arr.shape
    Hc = (H // f) * f
    Wc = (W // f) * f
    if Hc != H or Wc != W:
        arr = arr[:Hc, :Wc]
        H, W = Hc, Wc
    arr4 = arr.reshape(H // f, f, W // f, f)
    mean2d = arr4.mean(axis=(1, 3)).astype(np.float32)
    max2d  = arr4.max(axis=(1, 3)).astype(np.float32)
    return mean2d, max2d

def _block_reduce_mean(arr: np.ndarray, f: int):
    H, W = arr.shape
    Hc = (H // f) * f
    Wc = (W // f) * f
    if Hc != H or Wc != W:
        arr = arr[:Hc, :Wc]
        H, W = Hc, Wc
    arr4 = arr.reshape(H // f, f, W // f, f)
    return arr4.mean(axis=(1, 3)).astype(np.float32)

def _block_reduce_mask_fraction(arr_mask01: np.ndarray, f: int, thr: float):
    """
    Coarsen a 0/1 mask by block-mean and then threshold to 0/1.
    Also returns the fractional coverage if you ever want it.
    """
    frac = _block_reduce_mean(arr_mask01.astype(np.float32), f)
    coarsemask = (frac >= float(thr)).astype(np.float32)
    return coarsemask, frac.astype(np.float32)

# ---------------------------
# Datasets
# ---------------------------
class XRRandomPatchDataset(Dataset):
    """
    Streaming random (or tiled) patches directly from NetCDF/Zarr.
    Channels: [x_tr(mean,max), (dem?), mask, clim_tr] when coarsening is enabled.
    """

    def __init__(self, y_paths: List[str], x_paths: Optional[List[str]], cfg: Config, mode: str = "train"):
        super().__init__()
        assert xr is not None, "xarray is required."
        self.cfg = cfg
        self.mode = mode
        self.patch = int(getattr(cfg, "patch", 768))
        self.transform = cfg.transform

        # Coarsening controls
        self.coarsen_factor = int(getattr(cfg, "coarsen_factor", 1))
        self.coarsen_mode = str(getattr(cfg, "coarsen_mode", "mean_max")).lower()
        self.mask_thr = float(getattr(cfg, "coarsen_mask_threshold", 0.5))
        self.use_log1p = bool(getattr(cfg, "precip_log1p", False))

        # Open datasets
        self.ds_y = open_multi_auto(y_paths, time_dim=cfg.time_name, prefer_chunks_time1=True)
        self.var_y = cfg.prism_var
        self.t_dim = cfg.time_name
        v = self.ds_y[self.var_y]
        self.y_dim = v.dims[-2]
        self.x_dim = v.dims[-1]
        self.T = int(self.ds_y.sizes[self.t_dim])
        self.H_fine = int(self.ds_y.sizes[self.y_dim])
        self.W_fine = int(self.ds_y.sizes[self.x_dim])

        if x_paths:
            self.ds_x = open_multi_auto(x_paths, time_dim=cfg.time_name, prefer_chunks_time1=True)
            self.var_x = cfg.x_var or cfg.prism_var
        else:
            self.ds_x = None

        # Keep paths for reopen safety
        self._y_paths = list(y_paths)
        self._x_paths = list(x_paths) if x_paths else None

        # Domain mask & daily climatology
        if getattr(cfg, "domain_mask_npy", None):
            self.domain_mask_fine = np.load(cfg.domain_mask_npy, mmap_mode="r").astype(np.float32)
        else:
            self.domain_mask_fine = np.ones((self.H_fine, self.W_fine), dtype=np.float32)

        if getattr(cfg, "climatology_npy", None):
            self.clim_daily_fine = np.load(cfg.climatology_npy, mmap_mode="r")
            if self.clim_daily_fine.ndim != 3 or self.clim_daily_fine.shape[0] != 366:
                raise ValueError(f"climatology_npy must be [366,H,W], got {self.clim_daily_fine.shape}")
            if self.clim_daily_fine.shape[1] != self.H_fine or self.clim_daily_fine.shape[2] != self.W_fine:
                raise ValueError(
                    f"climatology_npy spatial shape {self.clim_daily_fine.shape[1:]} "
                    f"does not match data grid {(self.H_fine, self.W_fine)}"
                )
        else:
            self.clim_daily_fine = np.zeros((366, self.H_fine, self.W_fine), dtype=np.float32)

        # Precompute day-of-year (0..365)
        try:
            doy = self.ds_y[self.t_dim].dt.dayofyear.values  # 1..366
            self.doy_idx = (doy - 1).astype(np.int16)        # 0..365
        except Exception:
            self.doy_idx = np.arange(self.T, dtype=np.int64) % 366

        # Time-slice caches
        self._cache_limit = int(getattr(cfg, "timeslice_cache", 3))
        self._cache = {"y": OrderedDict(), "x": OrderedDict()}        # fine planes
        self._cache_coarse = {"y": OrderedDict(), "x": OrderedDict()} # coarse planes mean/max

        # Statics (DEM / user mask) at fine res
        self.dem_fine = None
        if cfg.use_dem and cfg.static_dem_path:
            dem = load_static(cfg.static_dem_path)
            if dem is not None:
                dem = (dem - np.nanmean(dem)) / (np.nanstd(dem) + 1e-6)
                dem = np.where(np.isfinite(dem), dem, 0.0)
                self.dem_fine = dem.astype(np.float32)

        self.user_static_mask_fine = None
        if cfg.use_mask and cfg.static_mask_path:
            sm = load_static(cfg.static_mask_path)
            if sm is not None:
                sm = (sm > 0.5).astype(np.float32)
                sm = np.where(np.isfinite(sm), sm, 0.0)
                self.user_static_mask_fine = sm.astype(np.float32)

        # Coarsen statics & geometry
        if self.coarsen_factor > 1:
            f = self.coarsen_factor
            Hc = (self.H_fine // f) * f
            Wc = (self.W_fine // f) * f

            dom_trim = self.domain_mask_fine[:Hc, :Wc]
            self.domain_mask_coarse, self.domain_mask_fraction = _block_reduce_mask_fraction(dom_trim, f, self.mask_thr)

            if self.dem_fine is not None:
                self.dem_coarse = _block_reduce_mean(self.dem_fine[:Hc, :Wc], f)
            else:
                self.dem_coarse = None

            if self.user_static_mask_fine is not None:
                usm_trim = self.user_static_mask_fine[:Hc, :Wc]
                self.user_static_mask_coarse, _ = _block_reduce_mask_fraction(usm_trim, f, 0.5)
            else:
                self.user_static_mask_coarse = None

            C = []
            for d in range(366):
                C.append(_block_reduce_mean(self.clim_daily_fine[d, :Hc, :Wc], f))
            self.clim_daily_coarse = np.stack(C, axis=0).astype(np.float32)

            self.H = self.domain_mask_coarse.shape[0]
            self.W = self.domain_mask_coarse.shape[1]
            self.patch = max(1, self.patch // f)   # interpret patch at coarse scale
        else:
            self.H = self.H_fine
            self.W = self.W_fine
            self.domain_mask_coarse = None
            self.dem_coarse = None
            self.user_static_mask_coarse = None
            self.clim_daily_coarse = None

        # Length & sampler
        if mode == "train":
            self.N = int(getattr(cfg, "train_crops_per_epoch", 20000))
            #self._sampler = self._random_sampler
            self._sampler = self._hybrid_sampler  # instead of self._random_sampler
            self._rng = random.Random(cfg.seed)
        else:
            val_mode = getattr(cfg, "val_crop_mode", "random").lower()
            if val_mode == "tile":
                vt = cfg.val_tile or cfg.tile
                vo = cfg.val_tile_overlap or cfg.tile_overlap
                self._make_tiles(vt, vo)
                self.N = len(self.tiles)
                self._sampler = self._tile_sampler
            else:
                self.N = int(getattr(cfg, "val_crops_per_epoch", 4000))
                self._sampler = self._random_sampler
                self._rng = random.Random(cfg.seed + 1)


    def set_epoch(self, epoch:int):
        self._epoch = int(epoch)


    def reseed(self, base_seed: int, epoch: int, rank: int):
        self._rng = random.Random(int(base_seed + epoch * 997 + rank))

    def reopen(self):
        try:
            self.ds_y.close()
        except Exception:
            pass
        self.ds_y = open_multi_auto(self._y_paths, time_dim=self.cfg.time_name, prefer_chunks_time1=True)
        if self._x_paths is not None:
            try:
                self.ds_x.close()
            except Exception:
                pass
            self.ds_x = open_multi_auto(self._x_paths, time_dim=self.cfg.time_name, prefer_chunks_time1=True)

    def __len__(self):
        return self.N

    def _random_sampler(self, idx):
        t = self._rng.randrange(self.T)
        h = min(self.patch, self.H)
        w = min(self.patch, self.W)
        y0 = self._rng.randrange(0, max(1, self.H - h + 1))
        x0 = self._rng.randrange(0, max(1, self.W - w + 1))
        return t, y0, x0, h, w

    def _nonuniform_sampler_precip(self, idx,
                                   thr_mm_log1p: float = None,
                                   accept_scale: float = 5.0,
                                   max_tries: int = 20):
        """
        Prefer windows that contain precip (foreground-aware sampling).
        - thr_mm_log1p: threshold in transform space; if None, we compute on raw mm.
        - accept_scale: higher → more likely to accept precip-heavy windows.
        """
        # Local shorthands
        H, W = self.H, self.W
        h = min(self.patch, H)
        w = min(self.patch, W)

        # Choose a transform-aware threshold if you want to work in transform space.
        # For simplicity we use raw mm here and threshold at 0.1 mm.
        precip_thr_mm = 0.1

        for _ in range(max_tries):
            t = self._rng.randrange(self.T)
            y0 = self._rng.randrange(0, max(1, H - h + 1))
            x0 = self._rng.randrange(0, max(1, W - w + 1))

            # Get fine plane from cache (your helper is already efficient)
            y_fine = self._get_plane("y", t)  # [H_fine,W_fine] or coarse if coarsen_factor==1
            patch = y_fine[y0:y0+h, x0:x0+w]

            # Foreground fraction (precip present)
            fg_frac = float((patch > precip_thr_mm).mean()) if patch.size else 0.0

            # Accept with probability proportional to foreground fraction
            # (cap to [0,1] via 1 - exp or min)
            import math
            p_accept = 1.0 - math.exp(-accept_scale * fg_frac)  # smooth monotonic
            if self._rng.random() < p_accept:
                return t, y0, x0, h, w

        # Fallback to uniform if we didn't accept in max_tries
        t = self._rng.randrange(self.T)
        y0 = self._rng.randrange(0, max(1, H - h + 1))
        x0 = self._rng.randrange(0, max(1, W - w + 1))
        return t, y0, x0, h, w

    #If using external Y (which we do) use this sampler to find places where they differ.
    def _nonuniform_sampler_diff(self, idx,
                                 err_percentile: float = 70.0,
                                 accept_scale: float = 5.0,
                                 max_tries: int = 20):
        """
        Prefer windows where |X - Y| is large (difference-aware sampling).
        err_percentile controls a soft cutoff: patches above this percentile
        of error are much more likely to be accepted.
        """
        H, W = self.H, self.W
        h = min(self.patch, H)
        w = min(self.patch, W)
        has_x = (self.ds_x is not None)

        # If no X is provided, fall back to precip-weighted
        if not has_x:
            return self._nonuniform_sampler_precip(idx)

        # Precompute a robust error scale from a few random spots (optional cache)
        # Here we just use a fixed percentile logic per trial for simplicity.
        for _ in range(max_tries):
            t = self._rng.randrange(self.T)
            y0 = self._rng.randrange(0, max(1, H - h + 1))
            x0 = self._rng.randrange(0, max(1, W - w + 1))

            y_fine = self._get_plane("y", t)
            x_fine = self._get_plane("x", t)
            y_patch = y_fine[y0:y0+h, x0:x0+w]
            x_patch = x_fine[y0:y0+h, x0:x0+w]

            # Absolute error (clip inf/nan)
            import numpy as np
            err = np.abs(np.nan_to_num(y_patch, 0.0) - np.nan_to_num(x_patch, 0.0))
            mean_err = float(err.mean())

            # Map error to acceptance probability using a soft sigmoid
            # You can also compute a dynamic threshold across a few random samples and compare to percentile.
            import math
            p_accept = 1.0 - math.exp(-accept_scale * mean_err)  # smooth, monotonic

            if self._rng.random() < p_accept:
                return t, y0, x0, h, w

        # Fallback
        t = self._rng.randrange(self.T)
        y0 = self._rng.randrange(0, max(1, H - h + 1))
        x0 = self._rng.randrange(0, max(1, W - w + 1))
        return t, y0, x0, h, w

    
    def _hybrid_sampler(self, idx, epoch: int = 0, policy: str = "diff"):
        """
        Blend uniform and biased sampling with an epoch-dependent schedule.
        - policy: "precip", "diff", or "hybrid"
        """
        # Example schedule: start 50% uniform, 50% biased; ramp to 10% / 90% by epoch 10.
        u_frac = max(0.10, 0.50 - 0.04 * epoch)  # clamp at 10%
        if self._rng.random() < u_frac:
            return self._random_sampler(idx)

        if policy == "diff" and (self.ds_x is not None):
            return self._nonuniform_sampler_diff(idx)
        else:
            return self._nonuniform_sampler_precip(idx)


    def _make_tiles(self, tile, overlap):
        step = tile - overlap
        tiles = []
        for t in range(self.T):
            for y0 in range(0, self.H, step):
                for x0 in range(0, self.W, step):
                    y1 = min(y0 + tile, self.H)
                    x1 = min(x0 + tile, self.W)
                    tiles.append((t, y0, x0, y1 - y0, x1 - x0))
        self.tiles = tiles

    def _tile_sampler(self, idx):
        return self.tiles[idx]

    def _window(self, ds, var, t, y0, x0, h, w):
        sl = ds[var].isel(**{self.t_dim: t, self.y_dim: slice(y0, y0 + h), self.x_dim: slice(x0, x0 + w)})
        import dask
        with dask.config.set(scheduler="single-threaded"):
            arr = sl.compute().data
        return np.asarray(arr, dtype=np.float32)

    def _get_plane(self, which: str, t: int) -> np.ndarray:
        """Fine resolution plane [H_fine, W_fine]."""
        assert which in ("y", "x")
        cache = self._cache[which]
        if t in cache:
            arr = cache.pop(t)
            cache[t] = arr
            return arr

        ds = self.ds_y if which == "y" else self.ds_x
        if ds is None:
            raise RuntimeError(f"_get_plane({which}, t={t}) requested but ds is None.")
        var = self.var_y if which == "y" else self.var_x

        sl = ds[var].isel(**{self.t_dim: t})

        import dask
        with dask.config.set(scheduler="single-threaded"):
            arr = np.asarray(sl.compute().data, dtype=np.float32)  # [H_fine, W_fine]

        cache[t] = arr
        while len(cache) > self._cache_limit:
            cache.popitem(last=False)
        return arr

    # Coarse plane getter with mean+max for precip
    def _get_plane_coarse_precip(self, which: str, t: int):
        """
        Returns (mean_coarse, max_coarse) at coarse resolution [H, W].
        Uses LRU cache and trims to divisibility by factor.
        """
        assert which in ("y", "x")
        if self.coarsen_factor <= 1:
            fine = self._get_plane(which, t)
            return fine, fine

        cache = self._cache_coarse[which]
        if t in cache:
            val = cache.pop(t)
            cache[t] = val
            return val

        fine = self._get_plane(which, t)
        mean2d, max2d = _block_reduce_mean_max(fine, self.coarsen_factor)
        cache[t] = (mean2d, max2d)
        while len(cache) > self._cache_limit:
            cache.popitem(last=False)
        return mean2d, max2d

    def __getitem__(self, idx: int):
        # ---- pick window IN COARSE COORDS ----
        t, y0, x0, h, w = self._sampler(idx)

        # 1) Y plane(s) at coarse res
        y_mean_plane, y_max_plane = self._get_plane_coarse_precip("y", t)
        y_mean = y_mean_plane[y0:y0+h, x0:x0+w]
        y_max  = y_max_plane[y0:y0+h, x0:x0+w]

        # 2) Masks
        m_data = valid_mask(y_mean, sentinel=-9999.0).astype(np.float32)
        if self.coarsen_factor > 1:
            m_dom = self.domain_mask_coarse[y0:y0+h, x0:x0+w]
        else:
            m_dom = self.domain_mask_fine[y0:y0+h, x0:x0+w]
        m = (m_data * m_dom).astype(np.float32)
        if self.user_static_mask_fine is not None:
            usm = self.user_static_mask_coarse if self.coarsen_factor > 1 else self.user_static_mask_fine
            m = (m * (usm[y0:y0+h, x0:x0+w] > 0.5).astype(np.float32)).astype(np.float32)

        # 3) Sanitize Y
        y_mean = np.where(m > 0.5, np.clip(y_mean, 0.0, 1e6), 0.0).astype(np.float32)

        # Optional log1p stabilizer
        if self.use_log1p:
            y_mean = np.log1p(y_mean).astype(np.float32)
            y_max  = np.log1p(np.where(m > 0.5, np.clip(y_max, 0.0, 1e6), 0.0)).astype(np.float32)

        y_t = torch.from_numpy(y_mean)[None, ...]  # [1, h, w]

        # 4) X mean + max at coarse res
        if self.ds_x is not None:
            x_mean_plane, x_max_plane = self._get_plane_coarse_precip("x", t)
        else:
            x_mean_plane, x_max_plane = y_mean_plane, y_max_plane

        x_mean_np = x_mean_plane[y0:y0+h, x0:x0+w]
        x_max_np  = x_max_plane[y0:y0+h, x0:x0+w]
        x_mean_np = np.where(np.isfinite(x_mean_np), x_mean_np, 0.0).astype(np.float32)
        x_max_np  = np.where(np.isfinite(x_max_np),  x_max_np,  0.0).astype(np.float32)

        if self.use_log1p:
            x_mean_np = np.log1p(np.clip(x_mean_np, 0.0, 1e6)).astype(np.float32)
            x_max_np  = np.log1p(np.clip(x_max_np,  0.0, 1e6)).astype(np.float32)

        x_mean_t = torch.from_numpy(np.where(m > 0.5, x_mean_np, 0.0))[None, ...]
        x_max_t  = torch.from_numpy(np.where(m > 0.5, x_max_np,  0.0))[None, ...]

        # 5) Transform to training space + light noise
        y_tr       = mm_to_transform(y_t, self.transform).float()
        x_mean_tr  = mm_to_transform(x_mean_t, self.transform).float()
        x_max_tr   = mm_to_transform(x_max_t,  self.transform).float()

        x_mean_tr  = add_noise_transformed(x_mean_tr, self.cfg.degrade_additive_noise_std)
        x_max_tr   = add_noise_transformed(x_max_tr,  self.cfg.degrade_additive_noise_std)

        # Mask after transform
        m_torch = torch.from_numpy(m)[None, ...]
        x_mean_tr = x_mean_tr * m_torch
        x_max_tr  = x_max_tr  * m_torch

        # 6) Assemble channels: [x_mean_tr, x_max_tr, (dem?), mask, clim_tr]
        chans = [x_mean_tr, x_max_tr]

        if self.dem_fine is not None:
            dem_use = self.dem_coarse if self.coarsen_factor > 1 else self.dem_fine
            chans.append(torch.from_numpy(dem_use[y0:y0+h, x0:x0+w])[None, ...])

        chans.append(m_torch)  # mask channel

        # Daily climatology (coarsened)
        doy = int(self.doy_idx[t])  # 0..365
        if self.coarsen_factor > 1:
            clim2d = self.clim_daily_coarse[doy, y0:y0+h, x0:x0+w].astype(np.float32)
        else:
            clim2d = self.clim_daily_fine[doy, y0:y0+h, x0:x0+w].astype(np.float32)
        clim2d = np.where(np.isfinite(clim2d), clim2d, 0.0).astype(np.float32)
        if self.use_log1p:
            clim2d = np.log1p(clim2d).astype(np.float32)
        clim_tr = 0.1 * mm_to_transform(torch.from_numpy(clim2d)[None, ...], self.transform)
        clim_tr = clim_tr * m_torch
        chans.append(clim_tr)

        X = torch.cat(chans, dim=0).float()  # [C, h, w]
        X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        Y = y_tr.float()                     # [1, h, w]
        M = m_torch.float()                  # [1, h, w]

        if not (torch.is_tensor(X) and torch.is_tensor(Y) and torch.is_tensor(M)):
            raise RuntimeError("Dataset produced non-tensor outputs")
        

        # assuming you already built X, Y, M as torch tensors: [C,H,W], [1,H,W], [1,H,W]
        if self.mode == "val":
            P = int(self.patch)  # 896 or 1024
            H, W = X.shape[-2:]
            if H != P or W != P:
                # pad order: (left, right, top, bottom) for 2D → (W-left, W-right, H-top, H-bottom)
                dh = max(0, P - H)
                dw = max(0, P - W)
                pad2d = (0, dw, 0, dh)
                X = F.pad(X, pad2d, mode="constant", value=0.0)
                Y = F.pad(Y, pad2d, mode="constant", value=0.0)
                # zero mask on padded region so it doesn't contribute to the loss/metrics
                M = F.pad(M, pad2d, mode="constant", value=0.0)
        return X, Y, M

# ---------------------------
# Model
# ---------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm='batch', act='silu'):
        super().__init__()
        Norm = lambda ch: nn.GroupNorm(8, ch)
        Act = nn.SiLU if act == 'silu' else nn.ReLU
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            Norm(out_ch),
            Act(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            Norm(out_ch),
            Act(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, **kw):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch, **kw)
    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, **kw):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.conv = ConvBlock(in_ch, out_ch, **kw)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = ConvBlock(in_ch, out_ch, **kw)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x):
        return self.conv(x)

class ClimateUNet(nn.Module):
    def __init__(self, in_ch, base_ch=64, norm='batch', act='silu', bilinear=True):
        super().__init__()
        self.inc = ConvBlock(in_ch, base_ch, norm=norm, act=act)
        self.down1 = Down(base_ch, base_ch*2, norm=norm, act=act)
        self.down2 = Down(base_ch*2, base_ch*4, norm=norm, act=act)
        self.down3 = Down(base_ch*4, base_ch*8, norm=norm, act=act)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_ch*8, base_ch*16 // factor, norm=norm, act=act)
        self.up1 = Up(base_ch*16, base_ch*8 // factor, bilinear=bilinear, norm=norm, act=act)
        self.up2 = Up(base_ch*8, base_ch*4 // factor, bilinear=bilinear, norm=norm, act=act)
        self.up3 = Up(base_ch*4, base_ch*2 // factor, bilinear=bilinear, norm=norm, act=act)
        self.up4 = Up(base_ch*2, base_ch, bilinear=bilinear, norm=norm, act=act)
        self.outc = OutConv(base_ch, 1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        return self.outc(x)


class ClimateUNetFlex(nn.Module):
    def __init__(self, in_ch, base_ch=64, norm='batch', act='silu',
                 bilinear=True, num_levels: int = 4, deep_supervision: bool = False):
        super().__init__()
        assert num_levels in (4, 5), "num_levels must be 4 or 5"
        self.num_levels = num_levels
        self.deep_supervision = bool(deep_supervision)

        # ----- Encoder -----
        self.inc   = ConvBlock(in_ch, base_ch, norm=norm, act=act)                 # x1
        self.down1 = Down(base_ch,      base_ch*2, norm=norm, act=act)             # x2
        self.down2 = Down(base_ch*2,    base_ch*4, norm=norm, act=act)             # x3
        self.down3 = Down(base_ch*4,    base_ch*8, norm=norm, act=act)             # x4

        factor = 2 if bilinear else 1
        self.down4 = Down(base_ch*8,    base_ch*16 // factor, norm=norm, act=act)  # x5

        if num_levels == 5:
            self.down5 = Down(base_ch*16 // factor, base_ch*32 // factor, norm=norm, act=act)  # x6

        # ----- Decoder -----
        if num_levels == 5:
            self.up1 = Up(base_ch*32, base_ch*16 // factor, bilinear=bilinear, norm=norm, act=act)  # x6 + x5
            self.up2 = Up(base_ch*16, base_ch*8  // factor, bilinear=bilinear, norm=norm, act=act)  # + x4
            self.up3 = Up(base_ch*8,  base_ch*4  // factor, bilinear=bilinear, norm=norm, act=act)  # + x3
            self.up4 = Up(base_ch*4,  base_ch*2  // factor, bilinear=bilinear, norm=norm, act=act)  # + x2
            self.up5 = Up(base_ch*2,  base_ch,               bilinear=bilinear, norm=norm, act=act)  # + x1
        else:
            self.up1 = Up(base_ch*16, base_ch*8  // factor, bilinear=bilinear, norm=norm, act=act)  # x5 + x4
            self.up2 = Up(base_ch*8,  base_ch*4  // factor, bilinear=bilinear, norm=norm, act=act)  # + x3
            self.up3 = Up(base_ch*4,  base_ch*2  // factor, bilinear=bilinear, norm=norm, act=act)  # + x2
            self.up4 = Up(base_ch*2,  base_ch,               bilinear=bilinear, norm=norm, act=act)  # + x1

        self.outc = OutConv(base_ch, 1)

        # ----- Deep supervision heads (two aux heads at the deepest decoder scales) -----
        if self.deep_supervision:
            if num_levels == 5:
                self.aux1 = OutConv(base_ch*16 // factor, 1)  # after up1 (H/16)
                self.aux2 = OutConv(base_ch*8  // factor, 1)  # after up2 (H/8)
            else:
                self.aux1 = OutConv(base_ch*8  // factor, 1)  # after up1 (H/8)
                self.aux2 = OutConv(base_ch*4  // factor, 1)  # after up2 (H/4)

    def forward(self, x):
        # ---- Encoder ----
        x1 = self.inc(x)     # H
        x2 = self.down1(x1)  # H/2
        x3 = self.down2(x2)  # H/4
        x4 = self.down3(x3)  # H/8
        x5 = self.down4(x4)  # H/16

        if self.num_levels == 5:
            x6 = self.down5(x5)  # H/32

            # ---- Decoder (collect features for aux heads) ----
            y  = self.up1(x6, x5)     # H/16
            feat1 = y
            y  = self.up2(y,  x4)     # H/8
            feat2 = y
            y  = self.up3(y,  x3)     # H/4
            y  = self.up4(y,  x2)     # H/2
            y  = self.up5(y,  x1)     # H
        else:
            y  = self.up1(x5, x4)     # H/8
            feat1 = y
            y  = self.up2(y,  x3)     # H/4
            feat2 = y
            y  = self.up3(y,  x2)     # H/2
            y  = self.up4(y,  x1)     # H

        main_delta = self.outc(y)     # [B,1,H,W]

        # In eval or if deep supervision disabled → return main only
        if (not self.training) or (not self.deep_supervision):
            return main_delta

        # Training + deep supervision → return main + aux dict
        aux = {
            "aux1": self.aux1(feat1),   # deepest decoder scale
            "aux2": self.aux2(feat2),   # next decoder scale
        }
        return main_delta, aux

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)     # H
        x2 = self.down1(x1)  # H/2
        x3 = self.down2(x2)  # H/4
        x4 = self.down3(x3)  # H/8
        x5 = self.down4(x4)  # H/16

        # Decoder (collect intermediate features for aux heads)
        x  = self.up1(x5, x4)      # H/8
        feat_up1 = x               # save for aux1
        x  = self.up2(x,  x3)      # H/4
        feat_up2 = x               # save for aux2
        x  = self.up3(x,  x2)      # H/2
        x  = self.up4(x,  x1)      # H
        main_delta = self.outc(x)  # [B,1,H,W] residual (transform space)

        if not self.training or not self.deep_supervision:
            return main_delta

        # Aux residuals (transform space), lower resolution than main
        aux_delta1 = self.aux1(feat_up1)  # [B,1,H/8,W/8]
        aux_delta2 = self.aux2(feat_up2)  # [B,1,H/4,W/4]
        # Return a tuple: main, dict of aux outputs
        return main_delta, {"aux1": aux_delta1, "aux2": aux_delta2}
# ---------------------------
# Losses & metrics
# ---------------------------
def gradient_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    def grads(z):
        dx = z[..., :, 1:] - z[..., :, :-1]
        dy = z[..., 1:, :] - z[..., :-1, :]
        return dx, dy
    px, py = grads(pred); tx, ty = grads(target)
    if mask is not None:
        mx = mask[..., :, 1:]; my = mask[..., 1:, :]
        gx = F.l1_loss(px * mx, tx * mx)
        gy = F.l1_loss(py * my, ty * my)
    else:
        gx = F.l1_loss(px, tx); gy = F.l1_loss(py, ty)
    return gx + gy

def spectral_loss(pred, target):
    # Upcast to float32 for FFT stability even under AMP
    p32 = pred.float()
    t32 = target.float()
    P = torch.fft.rfft2(p32, norm='ortho')
    T = torch.fft.rfft2(t32, norm='ortho')
    return F.mse_loss(torch.abs(P), torch.abs(T))

def mass_preservation_penalty(pred_mm: torch.Tensor, target_mm: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is not None:
        s_pred = (pred_mm * mask).sum(dim=(-1, -2))
        s_true = (target_mm * mask).sum(dim=(-1, -2))
    else:
        s_pred = pred_mm.sum(dim=(-1, -2))
        s_true = target_mm.sum(dim=(-1, -2))
    return F.l1_loss(s_pred, s_true)

def rmse(pred, target, mask=None):
    if mask is not None:
        diff = (pred - target) * mask
        return torch.sqrt((diff**2).sum() / (mask.sum().clamp_min(1.0)))
    else:
        return torch.sqrt(F.mse_loss(pred, target))

def mae(pred, target, mask=None):
    if mask is not None:
        diff = (pred - target) * mask
        return diff.abs().sum() / (mask.sum().clamp_min(1.0))
    else:
        return F.l1_loss(pred, target)

def fss(pred_mm: torch.Tensor, target_mm: torch.Tensor, thr: float, window: int = 9, mask: Optional[torch.Tensor] = None):
    B, C, H, W = pred_mm.shape
    assert C == 1
    pad = window // 2
    k = torch.ones((1, 1, window, window), device=pred_mm.device) / (window * window)
    pred_bin = (pred_mm >= thr).float()
    targ_bin = (target_mm >= thr).float()
    pred_f = F.conv2d(F.pad(pred_bin, (pad, pad, pad, pad), mode='reflect'), k)
    targ_f = F.conv2d(F.pad(targ_bin, (pad, pad, pad, pad), mode='reflect'), k)
    num = (pred_f - targ_f) ** 2
    den = pred_f ** 2 + targ_f ** 2
    if mask is not None:
        m = F.max_pool2d(F.pad(mask, (pad, pad, pad, pad), mode='reflect'), window, stride=1)
        num = num * m
        den = den * m
    num = num.mean()
    den = den.mean().clamp_min(1e-6)
    return 1.0 - (num / den)

def fss_loss(pred_mm, target_mm, thrs, window, mask):
    vals = [1.0 - fss(pred_mm, target_mm, thr=t, window=window, mask=mask) for t in thrs]
    return sum(vals) / max(len(vals), 1)

# ---------------------------
# Static loading
# ---------------------------
def load_arrays(paths: List[str], var: str, time_name: str) -> np.ndarray:
    if xr is None:
        raise RuntimeError("xarray is required for NetCDF/Zarr loading.")
    arrs = []
    for p in paths:
        ds = xr.open_dataset(p, engine="netcdf4")
        a = ds[var].load().values
        ds.close()
        if a.ndim != 3:
            raise ValueError(f"Variable '{var}' in {p} is not [T,H,W]. Got shape {a.shape}.")
        arrs.append(a.astype(np.float32))
    return np.concatenate(arrs, axis=0)

def load_static(path: Optional[str], var_guess: Optional[str] = None) -> Optional[np.ndarray]:
    if path is None:
        return None
    if xr is None:
        raise RuntimeError("xarray is required to load static rasters.")
    ds = xr.open_dataset(path, engine="netcdf4")
    if var_guess is None:
        var_guess = list(ds.data_vars)[0]
    a = ds[var_guess].load().values
    ds.close()
    if a.ndim == 3:
        a = a[0]
    return a.astype(np.float32)

# ---------------------------
# BIL/HDR writing
# ---------------------------
def _format_yyyymmdd(np_datetime) -> str:
    try:
        t = np.datetime_as_string(np_datetime, unit='D')
        return t.replace('-', '')
    except Exception:
        if isinstance(np_datetime, (np.datetime64, )):
            ts = np_datetime.astype('datetime64[s]').astype('int')
            dt = _dt.datetime.utcfromtimestamp(int(ts))
            return dt.strftime('%Y%m%d')
        elif isinstance(np_datetime, _dt.date):
            return np_datetime.strftime('%Y%m%d')
        else:
            s = str(np_datetime)
            return ''.join(c for c in s if c.isdigit())[0:8]

def _compose_bil_hdr(nrows: int, ncols: int, ulxmap: float, ulymap: float,
                     xdim: float, ydim: float, nbands: int = 1, nbits: int = 32,
                     nodata: float = -9999.0, layout: str = "BIL",
                     byteorder: str = "I", pixeltype: str = "FLOAT") -> str:
    band_row_bytes = ncols * (nbits // 8) * nbands
    total_row_bytes = band_row_bytes
    hdr = (
        f"BYTEORDER      {byteorder}\n"
        f"LAYOUT         {layout}\n"
        f"NROWS          {nrows}\n"
        f"NCOLS          {ncols}\n"
        f"NBANDS         {nbands}\n"
        f"NBITS          {nbits}\n"
        f"BANDROWBYTES   {band_row_bytes}\n"
        f"TOTALROWBYTES  {total_row_bytes}\n"
        f"PIXELTYPE      {pixeltype}\n"
        f"ULXMAP         {ulxmap}\n"
        f"ULYMAP         {ulymap}\n"
        f"XDIM           {xdim}\n"
        f"YDIM           {ydim}\n"
        f"NODATA         {nodata}\n"
    )
    return hdr

def write_bil_pair(arr2d: np.ndarray, bil_path: str, hdr_path: str,
                   ulxmap: float, ulymap: float, xdim: float, ydim: float,
                   nodata: float = -9999.0, force_little_endian: bool = True):
    if arr2d.ndim != 2:
        raise ValueError(f"write_bil_pair expects a 2D array, got shape {arr2d.shape}")
    nrows, ncols = arr2d.shape
    data = np.array(arr2d, dtype=np.float32, copy=True)
    data[~np.isfinite(data)] = nodata
    if force_little_endian and data.dtype.byteorder not in ('<', '='):
        data = data.byteswap().newbyteorder('<')
    with open(bil_path, 'wb') as f:
        data.tofile(f)
    hdr_txt = _compose_bil_hdr(nrows, ncols, ulxmap, ulymap, xdim, ydim,
                               nbands=1, nbits=32, nodata=nodata,
                               layout="BIL", byteorder="I", pixeltype="FLOAT")
    with open(hdr_path, 'w', encoding='ascii') as f:
        f.write(hdr_txt)

# ---------------------------
# Training
# ---------------------------


#def safe_collate(batch):
#    """Fail-fast collate to catch None or malformed dataset items during debugging."""
#    for i, item in enumerate(batch):
#        if item is None:
#            raise RuntimeError(f"[safe_collate] Dataset returned None at batch position {i}")
#        if not (isinstance(item, (list, tuple)) and len(item) == 3):
#            raise RuntimeError(f"[safe_collate] Bad item structure at pos {i}: type={type(item)}, item={item}")
#        X, Y, M = item
#        if not (torch.is_tensor(X) and torch.is_tensor(Y) and torch.is_tensor(M)):
#            raise RuntimeError(f"[safe_collate] Non-tensor at pos {i}: types=({type(X)}, {type(Y)}, {type(M)})")
#    return default_collate(batch)



from torch.utils.data._utils.collate import default_collate

def safe_collate(batch):
    """
    Defensive collate that:
      1) fail-fast checks for bad dataset items (None / bad structure / non-tensors)
      2) ensures each tensor owns a resizable, contiguous storage (avoids 'resize storage not resizable')
      3) then delegates to default_collate
    """
    import torch

    # ---- 1) diagnostics (your current checks) ----
    for i, item in enumerate(batch):
        if item is None:
            raise RuntimeError(f"[safe_collate] Dataset returned None at batch position {i}")
        if not (isinstance(item, (list, tuple)) and len(item) == 3):
            raise RuntimeError(f"[safe_collate] Bad item structure at pos {i}: type={type(item)}, item={item}")
        X, Y, M = item
        if not (torch.is_tensor(X) and torch.is_tensor(Y) and torch.is_tensor(M)):
            raise RuntimeError(f"[safe_collate] Non-tensor at pos {i}: types=({type(X)}, {type(Y)}, {type(M)})")

    # ---- 2) storage fix: make tensors owned + contiguous ----
    def own_and_contig(t: torch.Tensor) -> torch.Tensor:
        # If t is a view (non-owning) or non-contiguous, make an owned contiguous clone
        if t._base is not None or not t.is_contiguous():
            return t.contiguous().clone()
        return t

    fixed = []
    for (X, Y, M) in batch:
        X = own_and_contig(X)
        Y = own_and_contig(Y)
        M = own_and_contig(M)
        fixed.append((X, Y, M))

    # ---- 3) standard stacking ----
    return default_collate(fixed)

def train(cfg: Config):
    import torch.multiprocessing as mp
    import time
    import numpy as np
    from contextlib import nullcontext
    import dask

    mp.set_start_method("spawn", force=True)
    mp.set_sharing_strategy("file_system")  # fewer semaphore issues with many workers

    # single-threaded dask within each worker/process
    try:
        dask.config.set(scheduler="single-threaded")
    except Exception:
        pass
    try:
        if xr is not None:
            xr.set_options(file_cache_maxsize=0)
    except Exception:
        pass

    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # --- DDP setup & device ---
    local_rank, global_rank, world_size = ddp_setup()
    if local_rank is not None:
        device = torch.device(f"cuda:{local_rank}")
        is_main = (global_rank == 0)
    else:
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        is_main = True

    # Resolve save paths (filenames under out_dir)
    best_name = getattr(cfg, "best_name", None) or "best.pt"
    last_name = getattr(cfg, "last_name", None) or "last.pt"
    ckpt_best = os.path.join(cfg.out_dir, best_name)
    ckpt_last = os.path.join(cfg.out_dir, last_name)

    # Preferred explicit resume path (full path)
    resume_from = getattr(cfg, "resume_from", None)

    if is_main:
        print(f"[train] out_dir            : {os.path.abspath(cfg.out_dir)}")
        print(f"[train] resume_from       : {resume_from or 'None'}")
        print(f"[train] will save best to : {ckpt_best}")
        print(f"[train] will save last to : {ckpt_last}")

    best_val = float('inf')
    best_epoch = -1
    patience_ctr = 0

    # CPU/Gloo group for coarse rendezvous
    cpu_pg = None
    if ddp_is_initialized():
        cpu_pg = dist.new_group(backend="gloo")

    def cpu_barrier():
        if cpu_pg is not None:
            dist.barrier(group=cpu_pg)

    def _cuda_sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # --- Datasets ---
    ds_tr = XRRandomPatchDataset(y_paths=cfg.prism_train_paths,
                                 x_paths=cfg.x_train_paths, cfg=cfg, mode="train")
    
    
    if getattr(cfg, "steps_per_epoch", 0):
        world = world_size if ddp_is_initialized() else 1
        ds_tr.N = int(cfg.steps_per_epoch) * int(cfg.batch_size) * int(world)
        if is_main:
            print(f"[train] overriding ds_tr.N to {ds_tr.N} for steps_per_epoch={cfg.steps_per_epoch}, "
                f"bs={cfg.batch_size}, world={world}", flush=True)


    dl_tr = DataLoader(
        ds_tr,
        batch_size=cfg.batch_size,
        sampler=None,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.persistent_workers,
       # prefetch_factor=cfg.prefetch_factor,
        collate_fn=safe_collate
    )

    dl_va = None
    if is_main:
        ds_va = XRRandomPatchDataset(y_paths=cfg.prism_val_paths,
                                     x_paths=cfg.x_val_paths, cfg=cfg, mode="val")
       # dl_va = DataLoader(
       #     ds_va,
       #     batch_size=cfg.batch_size,
       #     shuffle=False,
       #     #num_workers=cfg.val_num_workers,
       #     num_workers=0,  # set to 0 for validation to avoid potential deadlocks and speed up small eval sets
       #     pin_memory=True,
       #     #persistent_workers=cfg.persistent_workers,
       #     persistent_workers=False,
       #    # prefetch_factor=cfg.prefetch_factor,
       #     collate_fn=safe_collate
       # )

        val_num_workers = 0
        dl_va = DataLoader(
            ds_va, batch_size=cfg.batch_size, shuffle=False,
            num_workers=val_num_workers, pin_memory=True,
            persistent_workers=False, collate_fn=safe_collate,
            timeout=0   # required for num_workers=0
        )

    # --- Model ---
    has_dem = bool(
        cfg.use_dem and (
            getattr(ds_tr, "dem_fine", None) is not None or
            getattr(ds_tr, "dem_coarse", None) is not None
        )
    )

    coarsen_mode = str(getattr(cfg, "coarsen_mode", "mean_max")).lower()
    n_precip_in = 2 if coarsen_mode == "mean_max" else 1

    # Total input channels: precip*, (+DEM?), mask, climatology
    in_ch = n_precip_in + (1 if has_dem else 0) + 1 + 1

    model = ClimateUNetFlex(
        in_ch=in_ch,
        base_ch=cfg.base_ch,
        num_levels=cfg.num_levels,
        deep_supervision=cfg.deep_supervision
    ).to(device)
    model = model.to(memory_format=torch.channels_last)

    # --- Optimizer & Scheduler (create BEFORE loading states) ---
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    amp_warmup_epochs = 5
    def autocast_cm():
        use_amp = (cfg.amp and (epoch > amp_warmup_epochs) and device.type == 'cuda')
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=30, threshold=1e-5
    )

    # ---- Gradient accumulation (reduce all-reduces under DDP) ----
    accum_steps = 2   # or 4; tune based on VRAM/throughput

    # --- Load checkpoint if exists (weights, opt, scheduler) ---
    start_epoch = 1

    def _load_full_training_state(ckpt_obj):
        nonlocal start_epoch, best_val
        load_weights_robust(model, ckpt_obj)
        start_epoch = ckpt_obj.get("epoch", 0) + 1
        best_val = ckpt_obj.get("best_val", float('inf'))
        if "optimizer" in ckpt_obj:
            try:
                opt.load_state_dict(ckpt_obj["optimizer"])
            except Exception as e:
                if is_main: print(f"[resume] optimizer state load failed: {e}")
        if "scheduler" in ckpt_obj:
            try:
                scheduler.load_state_dict(ckpt_obj["scheduler"])
            except Exception as e:
                if is_main: print(f"[resume] scheduler state load failed: {e}")

    def _load_weights_only(ckpt_obj):
        load_weights_robust(model, ckpt_obj)

    ckpt_loaded = False

    # 1) Try explicit --resume_from (full path)
    if resume_from and os.path.exists(resume_from):
        with open(resume_from, "rb") as f:
            ckpt = torch.load(f, map_location=device)
        if isinstance(ckpt, dict) and ("optimizer" in ckpt or "epoch" in ckpt or "model" in ckpt):
            _load_full_training_state(ckpt)
        else:
            _load_weights_only(ckpt)
        ckpt_loaded = True
        if is_main:
            print(f"[resume] Loaded from --resume_from: {resume_from}")

    # 2) Else try <out_dir>/<last_name>
    elif os.path.exists(ckpt_last):
        with open(ckpt_last, "rb") as f:
            ckpt = torch.load(f, map_location=device)
        _load_full_training_state(ckpt)
        ckpt_loaded = True
        if is_main:
            print(f"[resume] Loaded from last checkpoint: {ckpt_last}")

    # 3) Else fresh
    if not ckpt_loaded and is_main:
        print("[resume] No checkpoint loaded (fresh training).")

    if is_main and ckpt_loaded:
        print(f"[resume] Resuming from epoch {start_epoch}, best_val={best_val}")

    # --- Wrap once: DDP if torchrun, else DP as fallback for multi-GPU ---
    if ddp_is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index], output_device=device.index,
            find_unused_parameters=False, gradient_as_bucket_view=True
        )
    elif torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Steps per epoch (lock-step)
    world = world_size if ddp_is_initialized() else 1
    per_rank_bs = cfg.batch_size
    expected_steps = ds_tr.N // (per_rank_bs * world)

    epoch = start_epoch
    stop_training = False

    while (epoch <= cfg.epochs) and (not stop_training):
        # --- Epoch header (light warm-up) ---
        _cuda_sync(); cpu_barrier()

        if hasattr(ds_tr, "reseed"):
            ds_tr.reseed(cfg.seed, epoch, global_rank)
        if hasattr(ds_tr, "set_epoch"):
            ds_tr.set_epoch(epoch)

        time.sleep(0.25 * (global_rank if ddp_is_initialized() else 0))
        if is_main:
            tw = 0
            _ = ds_tr._get_plane("y", tw)
            if ds_tr.ds_x is not None:
                _ = ds_tr._get_plane("x", tw)
        _cuda_sync(); cpu_barrier()

        # ============================
        # TRAIN
        # ============================
        model.train()
        running = 0.0
        log_every = 50

        it_tr = iter(dl_tr)

        # ---- Prefetch STEP 0 ----
        try:
            X0, Y0, M0 = next(it_tr)
        except StopIteration:
            raise RuntimeError(f"Rank {global_rank}: no first batch for epoch {epoch}")

        X0 = X0.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        Y0 = Y0.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        M0 = M0.to(device, non_blocking=True).to(memory_format=torch.channels_last)

        if (cfg.x_train_paths is None) and (not cfg.degrade_in_dataset):
            with torch.no_grad():
                x_mm = transform_to_mm(X0[:, 0:1], cfg.transform)
                x_mm = torch.clamp(x_mm, min=0.0)
                x_mm = apply_degradation(x_mm, cfg)
                x_tr = mm_to_transform(x_mm, cfg.transform)
                X0 = torch.cat([x_tr, X0[:, 1:]], dim=1)

        # Clear grads at the start of the accumulation window
        opt.zero_grad(set_to_none=True)

        # Determine if this micro-step should all-reduce (sync) or not
        step0 = 0
        is_sync_step0 = (((step0 + 1) % accum_steps) == 0)
        ctx0 = nullcontext() if is_sync_step0 or (not ddp_is_initialized()) else model.no_sync()

        with ctx0:
            with autocast_cm():
                out0 = model(X0)
                if isinstance(out0, tuple):   # supports deep supervision
                    delta0, aux0 = out0
                else:
                    delta0, aux0 = out0, None

                Y0_pred_tr = apply_residual_gate_in_tr_space(
                    X_tr=X0[:, 0:1], delta_tr=delta0, M=M0,
                    c=cfg.gate_c, beta=cfg.gate_beta, alpha_bg=cfg.gate_alpha_bg
                )

            # FP32 fence section (mirrors main loop)
            with torch.autocast(device_type="cuda", enabled=False):
                Y0_pred_tr_clamped = Y0_pred_tr.clamp(0.0, 12.0).float()
                Y0mm  = transform_to_mm(Y0_pred_tr_clamped, cfg.transform).float()
                T0mm  = transform_to_mm(Y0,               cfg.transform).float()
                Y0mm  = torch.nan_to_num(Y0mm, nan=0.0, posinf=0.0, neginf=0.0)
                T0mm  = torch.nan_to_num(T0mm, nan=0.0, posinf=0.0, neginf=0.0)
                spec0 = spectral_loss(Y0_pred_tr * M0, Y0 * M0).float() * cfg.w_spec if cfg.w_spec > 0 \
                        else torch.tensor(0., device=device)

            if cfg.loss == "huber":
                data_loss0 = F.smooth_l1_loss(Y0_pred_tr * M0, Y0 * M0)
            else:
                data_loss0 = F.l1_loss(Y0_pred_tr * M0, Y0 * M0)
            gl0   = gradient_loss(Y0_pred_tr, Y0, mask=M0) * cfg.w_grad
            mass0 = mass_preservation_penalty(Y0mm, T0mm, mask=M0) * cfg.w_mass if cfg.w_mass > 0 else 0.0
            hp0   = F.l1_loss(highpass(Y0_pred_tr, 1.5) * M0, highpass(Y0, 1.5) * M0) * cfg.w_hp if cfg.w_hp > 0 else 0.0
            fssl0 = fss_loss(Y0mm, T0mm, cfg.fss_thresholds, cfg.fss_window, M0) * cfg.w_fss if cfg.w_fss > 0 else 0.0
            loss0 = data_loss0 + gl0 + spec0 + mass0 + hp0 + fssl0
            loss0 = loss0 + 0.01 * (delta0 ** 2).mean()

            # (Optional) deep supervision aux losses for step-0
            if cfg.deep_supervision and aux0 is not None:
                ds_w = (cfg.ds_weights or [0.2, 0.1])
                aux_losses0 = []
                for i, key in enumerate(("aux1", "aux2")):
                    if key not in aux0 or i >= len(ds_w) or ds_w[i] <= 0:
                        continue
                    aux_delta = aux0[key]
                    h, w = aux_delta.shape[-2:]
                    X0_ds = F.interpolate(X0[:, 0:1], size=(h, w), mode="area")
                    Y0_ds = F.interpolate(Y0,         size=(h, w), mode="area")
                    M0_ds = F.interpolate(M0,         size=(h, w), mode="nearest")
                    with autocast_cm():
                        Y0_pred_aux = apply_residual_gate_in_tr_space(
                            X_tr=X0_ds, delta_tr=aux_delta, M=M0_ds,
                            c=cfg.gate_c, beta=cfg.gate_beta, alpha_bg=cfg.gate_alpha_bg
                        )
                        aux_data = F.smooth_l1_loss(Y0_pred_aux * M0_ds, Y0_ds * M0_ds) if cfg.loss == "huber" \
                                  else F.l1_loss(Y0_pred_aux * M0_ds, Y0_ds * M0_ds)
                    aux_losses0.append(ds_w[i] * aux_data)
                if aux_losses0:
                    loss0 = loss0 + sum(aux_losses0)

        # Scale the loss for accumulation BEFORE backward
        (loss0 / accum_steps).backward()

        # Only step/zero when we reach the sync boundary
        if is_sync_step0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        running += float(loss0.detach())

        steps_iter = tqdm(range(1, expected_steps), total=expected_steps, initial=1,
                          desc=f"Epoch {epoch}/{cfg.epochs} [train]") if is_main else range(1, expected_steps)

        # DO NOT zero_grad here unconditionally; only at sync boundaries (see below)
        for step in steps_iter:
            try:
                X, Y_true, M = next(it_tr)
            except StopIteration:
                raise RuntimeError(f"Rank {global_rank}: StopIteration at step {step}/{expected_steps} "
                                   f"(N={ds_tr.N}, world={world}, bs={per_rank_bs})")

            X = X.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            Y_true = Y_true.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            M = M.to(device, non_blocking=True).to(memory_format=torch.channels_last)

            # If you generate degraded X on the fly:
            if (cfg.x_train_paths is None) and (not cfg.degrade_in_dataset):
                with torch.no_grad():
                    x_mm = transform_to_mm(X[:, 0:1], cfg.transform)
                    x_mm = torch.clamp(x_mm, min=0.0)
                    x_mm = apply_degradation(x_mm, cfg)
                    x_tr = mm_to_transform(x_mm, cfg.transform)
                    X = torch.cat([x_tr, X[:, 1:]], dim=1)

            # Decide whether to all-reduce on this micro-step
            is_sync_step = (((step + 1) % accum_steps) == 0)
            ctx = nullcontext() if is_sync_step or (not ddp_is_initialized()) else model.no_sync()

            with ctx:
                with autocast_cm():
                    out = model(X)
                    if isinstance(out, tuple):   # supports deep supervision
                        delta, aux = out
                    else:
                        delta, aux = out, None

                    Y_pred_tr = apply_residual_gate_in_tr_space(
                        X_tr=X[:, 0:1], delta_tr=delta, M=M,
                        c=cfg.gate_c, beta=cfg.gate_beta, alpha_bg=cfg.gate_alpha_bg
                    )

                # FP32 fence section (unchanged)
                with torch.autocast(device_type="cuda", enabled=False):
                    Y_pred_tr_clamped = Y_pred_tr.clamp(0.0, 12.0).float()
                    Y_pred_mm = transform_to_mm(Y_pred_tr_clamped, cfg.transform).float()
                    Y_true_mm = transform_to_mm(Y_true,            cfg.transform).float()
                    Y_pred_mm = torch.nan_to_num(Y_pred_mm, nan=0.0, posinf=0.0, neginf=0.0)
                    Y_true_mm = torch.nan_to_num(Y_true_mm, nan=0.0, posinf=0.0, neginf=0.0)
                    spec = spectral_loss(Y_pred_tr * M, Y_true * M).float() * cfg.w_spec if cfg.w_spec > 0 \
                           else torch.tensor(0., device=device)

                # Main loss terms (unchanged)
                if cfg.loss == "huber":
                    data_loss = F.smooth_l1_loss(Y_pred_tr * M, Y_true * M)
                else:
                    data_loss = F.l1_loss(Y_pred_tr * M, Y_true * M)
                gl   = gradient_loss(Y_pred_tr, Y_true, mask=M) * cfg.w_grad
                mass = mass_preservation_penalty(Y_pred_mm, Y_true_mm, mask=M) * cfg.w_mass if cfg.w_mass > 0 else 0.0
                hp   = F.l1_loss(highpass(Y_pred_tr, 1.5) * M, highpass(Y_true, 1.5) * M) * cfg.w_hp if cfg.w_hp > 0 else 0.0
                fssl = fss_loss(Y_pred_mm, Y_true_mm, cfg.fss_thresholds, cfg.fss_window, M) * cfg.w_fss if cfg.w_fss > 0 else 0.0

                loss = data_loss + gl + spec + mass + hp + fssl
                loss = loss + 0.01 * (delta ** 2).mean()

                # Deep supervision aux heads (if enabled)
                if cfg.deep_supervision and aux is not None:
                    ds_w = (cfg.ds_weights or [0.2, 0.1])
                    aux_losses = []
                    for i, key in enumerate(("aux1", "aux2")):
                        if key not in aux or i >= len(ds_w) or ds_w[i] <= 0:
                            continue
                        aux_delta = aux[key]
                        h, w = aux_delta.shape[-2:]
                        X_ds = F.interpolate(X[:, 0:1],   size=(h, w), mode="area")
                        Y_ds = F.interpolate(Y_true,      size=(h, w), mode="area")
                        M_ds = F.interpolate(M,           size=(h, w), mode="nearest")
                        with autocast_cm():
                            Y_pred_aux = apply_residual_gate_in_tr_space(
                                X_tr=X_ds, delta_tr=aux_delta, M=M_ds,
                                c=cfg.gate_c, beta=cfg.gate_beta, alpha_bg=cfg.gate_alpha_bg
                            )
                            aux_data = F.smooth_l1_loss(Y_pred_aux * M_ds, Y_ds * M_ds) if cfg.loss == "huber" \
                                      else F.l1_loss(Y_pred_aux * M_ds, Y_ds * M_ds)
                        aux_losses.append(ds_w[i] * aux_data)
                    if aux_losses:
                        loss = loss + sum(aux_losses)

                # Backward on scaled loss for accumulation
                (loss / accum_steps).backward()

            # Only all-reduce (implicit) + step at sync boundaries
            if is_sync_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            running += float(loss.detach())
            if is_main and hasattr(steps_iter, "set_postfix") and (step % log_every == 0):
                steps_iter.set_postfix(loss=running / log_every)
                running = 0.0

        # ---- Flush pending grads if expected_steps not divisible by accum_steps ----
        if (expected_steps % accum_steps) != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        del it_tr


        # --- Pre-validation rendezvous (all ranks) ---
        # This is the single synchronization point AFTER training steps for the epoch.
        # Keep it. Do not add more barriers until next epoch.
        _cuda_sync(); cpu_barrier()

        from itertools import islice
        import traceback

        # ============================
        # VALIDATION (rank 0 only)
        # ============================
        if is_main and dl_va is not None:
            model.eval()

            va_losses, va_rmse, va_mae = [], [], []
            fss_scores: Dict[float, List[float]] = {thr: [] for thr in cfg.fss_thresholds}
            skipped_empty = 0
            val_failed = 0
            mean_val = float('inf')

            print("[val] probing first batch...", flush=True)
            try:
                it_va_probe = iter(dl_va)               # with num_workers=0, runs in main proc
                Xp, Yp, Mp = next(it_va_probe)          # if it fails, it's __getitem__/collate
                print("[val] got first batch", tuple(Xp.shape), flush=True)
                del it_va_probe, Xp, Yp, Mp
            except Exception as e:
                print(f"[val] ERROR fetching first batch: {repr(e)}", flush=True)
                traceback.print_exc()
                val_failed = 1
                mean_val = float('inf')                 # fail-safe metric
            else:
                # --- Run up to cfg.val_steps only ---
                with torch.no_grad(), autocast_cm():
                    it_va = iter(dl_va)
                    n_val = int(getattr(cfg, "val_steps", 0) or 0)
                    print(f"[val] val_steps = {n_val}", flush=True)
                    iterable = islice(it_va, n_val) if n_val > 0 else iter(dl_va)

                    for X, Y_true, M in tqdm(iterable, total=(n_val if n_val > 0 else None),
                                            desc=f"Epoch {epoch}/{cfg.epochs} [val]"):
                        X = X.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                        Y_true = Y_true.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                        M = M.to(device, non_blocking=True).to(memory_format=torch.channels_last)

                        if M.sum() <= 0:
                            skipped_empty += 1
                            continue

                        delta_val = model(X)
                        Y_pred_tr = apply_residual_gate_in_tr_space(
                            X_tr=X[:, 0:1], delta_tr=delta_val, M=M,
                            c=cfg.gate_c, beta=cfg.gate_beta, alpha_bg=cfg.gate_alpha_bg
                        )

                        if cfg.loss == "huber":
                            data_loss = F.smooth_l1_loss(Y_pred_tr * M, Y_true * M)
                        else:
                            data_loss = F.l1_loss(Y_pred_tr * M, Y_true * M)

                        gl = gradient_loss(Y_pred_tr, Y_true, mask=M) * cfg.w_grad

                        Y_pred_tr_clamped = Y_pred_tr.clamp(min=0.0, max=12.0)
                        Y_pred_mm = transform_to_mm(Y_pred_tr_clamped, cfg.transform)
                        Y_true_mm = transform_to_mm(Y_true,    cfg.transform)
                        Y_pred_mm = torch.nan_to_num(Y_pred_mm, nan=0.0, posinf=0.0, neginf=0.0)
                        Y_true_mm = torch.nan_to_num(Y_true_mm, nan=0.0, posinf=0.0, neginf=0.0)

                        spec = spectral_loss(Y_pred_tr * M, Y_true * M) * cfg.w_spec if cfg.w_spec > 0 else 0.0
                        mass = mass_preservation_penalty(Y_pred_mm, Y_true_mm, mask=M) * cfg.w_mass if cfg.w_mass > 0 else 0.0
                        hp_pred = highpass(Y_pred_tr, sigma=1.5)
                        hp_true = highpass(Y_true,    sigma=1.5)
                        hp = F.l1_loss(hp_pred * M, hp_true * M) * cfg.w_hp if cfg.w_hp > 0 else 0.0
                        fssl = fss_loss(Y_pred_mm, Y_true_mm, cfg.fss_thresholds, cfg.fss_window, M) * cfg.w_fss if cfg.w_fss > 0 else 0.0

                        vloss = data_loss + gl + spec + mass + hp + fssl
                        va_losses.append(float(vloss.detach()))
                        va_rmse.append(rmse(Y_pred_mm, Y_true_mm, mask=M).item())
                        va_mae.append(mae(Y_pred_mm, Y_true_mm, mask=M).item())
                        for thr in cfg.fss_thresholds:
                            fss_scores[thr].append(
                                fss(Y_pred_mm, Y_true_mm, thr=thr, window=cfg.fss_window, mask=M).item()
                            )

                if not val_failed:
                    if len(va_losses) == 0:
                        mean_val  = float('inf')
                        mean_rmse = float('inf')
                        mean_mae  = float('inf')
                        mean_fss  = {thr: 0.0 for thr in cfg.fss_thresholds}
                        print(f"[Val] WARNING: all validation batches skipped (empty mask).", flush=True)
                    else:
                        mean_val  = float(np.mean(va_losses))
                        mean_rmse = float(np.mean(va_rmse))
                        mean_mae  = float(np.mean(va_mae))
                        mean_fss  = {thr: float(np.mean(v)) for thr, v in fss_scores.items()}

                    print(
                        f"[Val] loss={mean_val:.4f} rmse={mean_rmse:.3f} mae={mean_mae:.3f} "
                        + " ".join([f"FSS@{thr}={mean_fss[thr]:.3f}" for thr in cfg.fss_thresholds]),
                        flush=True
                    )
        else:
            mean_val = float('inf')

        # Rank-0 steps scheduler (no barrier), then broadcast LR to all ranks (already in your code)
        if is_main:
            scheduler.step(mean_val)
        # Sync LR values to other ranks (no barrier needed)
        if ddp_is_initialized():
            for pg in opt.param_groups:
                lr_t = torch.tensor([pg['lr']], device=device)
                dist.broadcast(lr_t, src=0)
                pg['lr'] = float(lr_t.item())

        # ======= DO NOT BARRIER HERE =======
        # Compute flags on rank-0, broadcast to all ranks (no barrier)
        if is_main:
            improved_flag_local = int(np.isfinite(mean_val) and (mean_val < best_val - 1e-5))
            planned_best_val_local = float(min(best_val, mean_val))
            planned_patience_local = 0 if improved_flag_local else (patience_ctr + 1)
            should_stop_flag_local = int(epoch >= cfg.min_epochs and planned_patience_local >= cfg.patience)
        else:
            improved_flag_local = 0
            planned_best_val_local = best_val
            should_stop_flag_local = 0

        # Use your helper to broadcast scalars (keeps existing infra)
        improved_flag    = ddp_broadcast_scalar(improved_flag_local,    device, torch.int32)
        planned_best_val = ddp_broadcast_scalar(planned_best_val_local, device, torch.float32)
        should_stop_flag = ddp_broadcast_scalar(should_stop_flag_local, device, torch.int32)

        # Rank-0 updates trackers and saves checkpoints (NO barrier after)
        if is_main:
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

            # Save "last" with full training state (resume)
            torch.save({
                "model": state_dict,
                "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_val": best_val,   # note: updated below on improvement
                "cfg": asdict(cfg)
            }, ckpt_last)

            if int(improved_flag) == 1:
                best_val = float(planned_best_val)
                best_epoch = epoch
                patience_ctr = 0
                # Save "best" with weights-only (inference/warm-start)
                torch.save({
                    "model": state_dict,
                    "cfg": asdict(cfg),
                    "epoch": epoch,
                    "best_val": best_val
                }, ckpt_best)
                print("  ↳ Saved new best checkpoint:", os.path.abspath(ckpt_best), flush=True)
            else:
                patience_ctr += 1

        # ======= STILL NO BARRIER HERE =======
        # Early stop / epoch++
        if int(should_stop_flag) == 1:
            if is_main:
                print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch} (val={best_val:.6f})", flush=True)
            stop_training = True
        else:
            epoch += 1

    # Final sync & cleanup
    _cuda_sync(); cpu_barrier()
    if is_main:
        print("Training complete. Best checkpoint at:", os.path.abspath(ckpt_best), flush=True)
    ddp_cleanup()
# ---------------------------
# Inference (single-process by default)
# ---------------------------
def tile_infer(cfg: Config, model: nn.Module, X: torch.Tensor, M: torch.Tensor, tile: int, overlap: int) -> torch.Tensor:
    device = X.device
    _, C, H, W = X.shape
    out = torch.zeros((1, 1, H, W), device=device)
    weight = torch.zeros((1, 1, H, W), device=device)
    step = tile - overlap
    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            y1 = min(y0 + tile, H); x1 = min(x0 + tile, W)
            ys = slice(y0, y1); xs = slice(x0, x1)
            patch = X[..., ys, xs]
            M_p = M[..., ys, xs]
            with torch.no_grad():
                delta = model(patch)
                y_patch = apply_residual_gate_in_tr_space(
                    X_tr=patch[:, 0:1], delta_tr=delta, M=M_p,
                    c=cfg.gate_c, beta=cfg.gate_beta, alpha_bg=cfg.gate_alpha_bg
                )
            out[..., ys, xs] += y_patch
            weight[..., ys, xs] += 1.0
    out = out / weight.clamp_min(1.0)
    return out

def infer_on_stack(cfg: Config, ckpt_path: str, pre_paths: List[str], out_path: str,
                   static_dem_path: Optional[str] = None, static_mask_path: Optional[str] = None):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    assert cfg.ckpt_path, "Inference requires --ckpt_path (full path to the checkpoint)."
    ckpt = torch.load(ckpt_path, map_location=device)
    model_cfg = Config(**sanitize_cfg_dict(ckpt["cfg"]))

    print(f"[infer] ckpt_path          : {cfg.ckpt_path}")

    if xr is None:
        raise RuntimeError("xarray required for inference I/O.")

    # Read stack(s) used for mask/clim/time axis
    arrs = []
    times_all = []
    for p in pre_paths:
        ds = xr.open_dataset(p, engine="netcdf4")
        a = ds[model_cfg.prism_var].load()
        arrs.append(a.values.astype(np.float32))
        times_all.append(ds[model_cfg.time_name].values)
        ds.close()

    pre_all = np.concatenate(arrs, axis=0)
    ds0 = xr.open_dataset(pre_paths[0], engine="netcdf4")
    H = ds0.sizes[model_cfg.lat_name]; W = ds0.sizes[model_cfg.lon_name]
    ds0.close()

    dem = load_static(static_dem_path) if model_cfg.use_dem else None

    # --- Load daily climatology and domain mask ---
    clim_daily = None
    if getattr(cfg, "climatology_npy", None):
        clim_daily = np.load(cfg.climatology_npy, mmap_mode="r")
        if clim_daily.ndim != 3 or clim_daily.shape[0] != 366:
            raise ValueError(f"climatology_npy must be [366,H,W], got {clim_daily.shape}")

    domain_mask = None
    if getattr(cfg, "domain_mask_npy", None):
        domain_mask = np.load(cfg.domain_mask_npy, mmap_mode="r").astype(np.float32)

    if domain_mask is None:
        if clim_daily is not None:
            domain_mask = np.isfinite(clim_daily).any(axis=0).astype(np.float32)  # [H,W]
        else:
            domain_mask = np.isfinite(pre_all).any(axis=0).astype(np.float32)     # fallback

    # IMPORTANT: match training config from checkpoint
    has_dem = bool(model_cfg.use_dem)
    coarsen_mode = str(getattr(model_cfg, "coarsen_mode", "mean_max")).lower()
    n_precip_in = 2 if coarsen_mode == "mean_max" else 1
    chans = n_precip_in + (1 if has_dem else 0) + 1 + 1

    model = ClimateUNet(in_ch=chans, base_ch=model_cfg.base_ch).to(device)
    _state = ckpt["model"] if "model" in ckpt else ckpt
    _exp_in = _expected_in_channels_from_state(_state)
    if _exp_in is not None and _exp_in != chans:
        raise RuntimeError(f"Checkpoint expects in_ch={_exp_in}, but inference constructed in_ch={chans}.")
    load_weights_robust(model, ckpt)
    model.eval()

    out_times = []
    out_days = []

    x_files = cfg.x_pre_paths if cfg.x_pre_paths else None
    if x_files is not None and len(x_files) != len(pre_paths):
        raise ValueError("--x_pre_paths must match --pre_paths length.")

    for i, p in enumerate(pre_paths):
        dsY = xr.open_dataset(p, engine="netcdf4")
        y_mm_np = dsY[model_cfg.prism_var].load().values
        times = dsY[model_cfg.time_name].values
        dsY.close()

        x_mm_np = None
        if x_files is not None:
            dsX = xr.open_dataset(x_files[i], engine="netcdf4")
            xvar = model_cfg.x_var or model_cfg.prism_var
            x_mm_np = dsX[xvar].load().values
            dsX.close()
            if x_mm_np.shape != y_mm_np.shape:
                raise ValueError(f"X shape {x_mm_np.shape} != Y shape {y_mm_np.shape}")

        T = y_mm_np.shape[0]
        corrected = np.zeros_like(y_mm_np, dtype=np.float32)

        for t in tqdm(range(T), desc=os.path.basename(p)):  # time loop
            y_mm = y_mm_np[t]
            y_mm = np.where(np.isfinite(y_mm), y_mm, 0.0).astype(np.float32)
            y_mm = np.clip(y_mm, 0.0, 1e5)

            if x_mm_np is not None:
                src_precip_fine = np.where(np.isfinite(x_mm_np[t]), x_mm_np[t], 0.0).astype(np.float32)
            else:
                y_t_tensor = torch.from_numpy(y_mm)[None, None, ...].to(device)
                src_precip_fine = apply_degradation(y_t_tensor, model_cfg).squeeze().detach().cpu().numpy().astype(np.float32)

            m_data  = valid_mask(y_mm, sentinel=-9999.0)
            m_fine  = (m_data * domain_mask).astype(np.float32)

            try:
                doy_t = int(np.datetime64(times[t], 'D').astype('datetime64[D]').astype(object).timetuple().tm_yday) - 1
            except Exception:
                doy_t = int(xr.DataArray(times[t]).dt.dayofyear.values) - 1

            if clim_daily is not None:
                clim_crop = clim_daily[doy_t, :, :].astype(np.float32)
            else:
                clim_crop = np.zeros((H, W), dtype=np.float32)
            clim_crop = np.where(np.isfinite(clim_crop), clim_crop, 0.0).astype(np.float32)

            f         = int(getattr(model_cfg, "coarsen_factor", 1))
            mask_thr  = float(getattr(model_cfg, "coarsen_mask_threshold", 0.5))
            use_log1p = bool(getattr(model_cfg, "precip_log1p", False))

            if f > 1:
                x_mean_np, x_max_np = _block_reduce_mean_max(src_precip_fine, f)
                m_coarse, _         = _block_reduce_mask_fraction(m_fine.astype(np.float32), f, mask_thr)
                clim_coarse         = _block_reduce_mean(clim_crop.astype(np.float32), f)
                dem_use = dem
                if dem_use is not None:
                    dem_use = _block_reduce_mean(dem_use.astype(np.float32), f)
            else:
                x_mean_np = src_precip_fine
                x_max_np  = src_precip_fine
                m_coarse  = m_fine
                clim_coarse = clim_crop
                dem_use = dem

            if use_log1p:
                x_mean_np = np.log1p(np.clip(x_mean_np, 0.0, 1e6)).astype(np.float32)
                x_max_np  = np.log1p(np.clip(x_max_np,  0.0, 1e6)).astype(np.float32)

            x_mean_t  = torch.from_numpy(np.where(m_coarse > 0.5, x_mean_np, 0.0))[None, ...].to(device)
            x_max_t   = torch.from_numpy(np.where(m_coarse > 0.5, x_max_np,  0.0))[None, ...].to(device)
            x_mean_tr = mm_to_transform(x_mean_t, model_cfg.transform).float()
            x_max_tr  = mm_to_transform(x_max_t,  model_cfg.transform).float()

            mask_full = torch.from_numpy(m_coarse)[None, None, ...].to(device)  # [1,1,Hc,Wc]
            mask_ch   = mask_full.squeeze(0)                                    # [1,Hc,Wc]

            clim_tr = 0.1 * mm_to_transform(torch.from_numpy(clim_coarse)[None, ...].float(),
                                            model_cfg.transform).to(device)
            clim_tr = clim_tr * mask_ch

            dem_ch = None
            if dem_use is not None:
                dem_n = (dem_use - np.nanmean(dem_use)) / (np.nanstd(dem_use) + 1e-6)
                dem_n = np.where(np.isfinite(dem_n), dem_n, 0.0).astype(np.float32)
                dem_ch = torch.from_numpy(dem_n)[None, ...].to(device)

            chans_list = [x_mean_tr, x_max_tr]
            if dem_ch is not None:
                chans_list.append(dem_ch)
            chans_list.append(mask_ch)
            chans_list.append(clim_tr)

            X = torch.cat(chans_list, dim=0).unsqueeze(0)  # [1, C, Hc, Wc]
            mask_t = mask_full

            if max(H, W) > model_cfg.tile:
                y_pred_tr = tile_infer(model_cfg, model, X, mask_t, tile=model_cfg.tile, overlap=model_cfg.tile_overlap)
            else:
                with torch.no_grad():
                    delta = model(X)
                    y_pred_tr = apply_residual_gate_in_tr_space(
                        X_tr=X[:, 0:1], delta_tr=delta, M=mask_t,
                        c=model_cfg.gate_c, beta=model_cfg.gate_beta, alpha_bg=model_cfg.gate_alpha_bg
                    )

            y_pred_mm = transform_to_mm(y_pred_tr, model_cfg.transform)
            y_pred_mm = torch.clamp(y_pred_mm, min=0.0)
            y_pred_mm = y_pred_mm * mask_t

            if f > 1:
                y_pred_mm_up = F.interpolate(y_pred_mm, size=(H, W), mode='bilinear', align_corners=False)
                corrected[t] = y_pred_mm_up.squeeze().detach().cpu().numpy().astype(np.float32)
            else:
                corrected[t] = y_pred_mm.squeeze().detach().cpu().numpy().astype(np.float32)

        out_times.append(times)
        out_days.append(corrected)

        # Write per-day BIL/HDR files
        Tall = np.concatenate(out_days, axis=0)
        times_all = np.concatenate(out_times, axis=0)
        base_dir = os.path.abspath(out_path.rstrip("/"))
        ULXMAP = -125.016666666666
        ULYMAP = 49.933333333333
        XDIM   = 0.008333333333
        YDIM   = 0.008333333333
        NODATA = -9999.0

        ds0 = xr.open_dataset(pre_paths[0], engine="netcdf4")
        lat = ds0[model_cfg.lat_name].values
        ds0.close()
        need_flip = (lat[0] < lat[-1])
        Tall_to_write = Tall[:, ::-1, :] if need_flip else Tall

        count = 0
        for i2 in range(Tall_to_write.shape[0]):
            yyyymmdd = _format_yyyymmdd(times_all[i2])
            yyyy = yyyymmdd[:4]
            year_dir = os.path.join(base_dir, yyyy)
            os.makedirs(year_dir, exist_ok=True)
            base = f"adj_best_ppt_us_us_30s_{yyyymmdd}"
            bil_path = os.path.join(year_dir, base + ".bil")
            hdr_path = os.path.join(year_dir, base + ".hdr")
            write_bil_pair(arr2d=Tall_to_write[i2], bil_path=bil_path, hdr_path=hdr_path,
                           ulxmap=ULXMAP, ulymap=ULYMAP, xdim=XDIM, ydim=YDIM, nodata=NODATA,
                           force_little_endian=True)
            count += 1
        print(f"Wrote {count} BIL/HDR pairs under {base_dir}/<YYYY>/")

# ---------------------------
# CLI
# ---------------------------
def build_arg_parser():
    ap = argparse.ArgumentParser(description="Convective morphology correction for PRISM (PyTorch).")
    ap.add_argument("--mode", default="train", choices=["train", "infer"], required=True)
    ap.add_argument("--config", type=str, help="JSON config path (optional).")

    ap.add_argument("--train_paths", type=str, nargs="*", help="Train NetCDF/Zarr files (Y).")
    ap.add_argument("--val_paths",   type=str, nargs="*", help="Val   NetCDF/Zarr files (Y).")

    ap.add_argument("--x_train_paths", type=str, nargs="*", help="Train files for X (if omitted, X is degraded Y).")
    ap.add_argument("--x_val_paths",   type=str, nargs="*", help="Val   files for X (if omitted, X is degraded Y).")
    ap.add_argument("--x_var",         type=str, default=None, help="Variable name for X (defaults to prism_var).")

    ap.add_argument("--static_dem",  type=str, default=None)
    ap.add_argument("--static_mask", type=str, default=None)

    
    # ------------- argparse (train/infer) -------------
    ap.add_argument("--out_dir", type=str, default=None, help="Directory to write training outputs (config, ckpts).")
    ap.add_argument("--resume_from", type=str, default=None, help="Full path to checkpoint to resume TRAINING from (weights-only also OK).")
    ap.add_argument("--ckpt_path", type=str, default=None, help="Full path to checkpoint to use for INFERENCE.")
    # Keep current names, but mark as deprecated (compat shim)
    ap.add_argument("--ckpt", type=str, default=None, help="[DEPRECATED] Use --resume_from for training, --ckpt_path for inference.")
    ap.add_argument("--ckpt_name", type=str, default=None, help="[DEPRECATED] Use --best_name.")
    ap.add_argument("--best_name", type=str, default=None, help="Filename for the best checkpoint under out_dir (default: best.pt).")
    ap.add_argument("--last_name", type=str, default=None, help="Filename for the last checkpoint under out_dir (default: last.pt).")

    ap.add_argument("--pre_paths",   type=str, nargs="*", help="Files used for mask/clim/iteration in inference.")
    ap.add_argument("--x_pre_paths", type=str, nargs="*", help="Files used as X in inference; if omitted, X=degraded Y.")
    ap.add_argument("--out_path",    type=str, default="./corrected_pre2001.nc")

    ap.add_argument("--batch_size", type=int, help="Per-GPU batch size for training.")
    ap.add_argument("--steps_per_epoch", type=int, default=None)
    ap.add_argument("--val_steps", type=int, default=None)
    ap.add_argument("--patch", type=int, default=None, help="Per-GPU patch size for training.")

    ap.add_argument("--climatology_npy", type=str, default=None, help="Daily climatology .npy [366,H,W].")
    ap.add_argument("--domain_mask_npy", type=str, default=None, help="Domain mask .npy [H,W] (0/1).")
    ap.add_argument("--timeslice_cache", type=int, default=12, help="In process cache - impacts the amount of RAM.")

    # Dataloader tuning CLI
    ap.add_argument("--num_workers", type=int, default=None, help="Number of DataLoader workers per process ")
    ap.add_argument("--val_num_workers", type=int, default=None, help="Validation DataLoader workers (rank 0).")
    ap.add_argument("--prefetch_factor", type=int, default=None, help="Batches prefetched per worker (default: PyTorch default).")
    ap.add_argument("--persistent_workers", action="store_true", help="Keep DataLoader workers alive across epochs.")

    ap.add_argument("--biased_crops", action="store_true", default=None, help="N ")
    
    # Policy name as a string; restrict to implemented strategies
    ap.add_argument(
        "--biased_policy",
        type=str,
        choices=["precip", "diff", "hybrid"],
        default="precip",
        help="Which biased sampler to use: 'precip' (foreground-aware), 'diff' (|X-Y|-aware), or 'hybrid'."
    )

    ap.add_argument("--biased_warmup_epochs", type=int, default=None, help="Epochs to gradually ramp from uniform to biased sampling")

    #Deeper UNET with 5 levels
    ap.add_argument("--num_levels", type=int, choices=[4, 5], default=4,
                    help="UNet depth: 4 (current) or 5 (deeper receptive field).")
    ap.add_argument("--deep_supervision", action="store_true",
                    help="Enable deep supervision heads and coarse losses.")
    ap.add_argument("--ds_weights", type=float, nargs="*", default=None,
                    help="Aux head weights, e.g., --ds_weights 0.2 0.1 (match 2 heads).")

    return ap

def main():
    ap = build_arg_parser()
    args = ap.parse_args()
    cfg = Config()

    # config override
    if args.config is not None:
        with open(args.config, "r") as f:
            j = json.load(f)
        j = sanitize_cfg_dict(j)
        for k, v in j.items():
            setattr(cfg, k, v)

    if args.train_paths: cfg.prism_train_paths = args.train_paths
    if args.val_paths:   cfg.prism_val_paths   = args.val_paths

    if args.x_train_paths: cfg.x_train_paths = args.x_train_paths
    if args.x_val_paths:   cfg.x_val_paths   = args.x_val_paths
    if args.x_var:         cfg.x_var         = args.x_var

    if args.static_dem:    cfg.static_dem_path  = args.static_dem
    if args.static_mask:   cfg.static_mask_path = args.static_mask

    if args.out_path:      cfg.out_path  = args.out_path
    if args.ckpt:          cfg.ckpt      = args.ckpt

    if args.biased_crops: cfg.biased_crops  = args.biased_crops
    if args.biased_policy:  cfg.biased_policy = args.biased_policy
    if args.biased_warmup_epochs: cfg.biased_warmup_epochs = args.biased_warmup_epochs

    if args.timeslice_cache is not None: cfg.timeslice_cache = args.timeslice_cache
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.patch is not None: cfg.patch = args.patch    

    # world size before ddp_setup() may be unavailable; use env if present
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    bs = cfg.batch_size

    if args.steps_per_epoch is not None: cfg.train_crops_per_epoch = args.steps_per_epoch * bs * world_size_env
    if args.val_steps is not None: cfg.val_crops_per_epoch   = args.val_steps * bs  # val runs only on rank 0

    if args.climatology_npy: cfg.climatology_npy = args.climatology_npy
    if args.domain_mask_npy: cfg.domain_mask_npy = args.domain_mask_npy

    if args.num_workers is not None: cfg.num_workers = args.num_workers
    if args.val_num_workers is not None: cfg.val_num_workers = args.val_num_workers
    if args.prefetch_factor is not None: cfg.prefetch_factor = args.prefetch_factor
    if args.persistent_workers: cfg.persistent_workers = True

    if args.out_dir: cfg.out_dir = args.out_dir
    if args.ckpt_name: cfg.ckpt_name = args.ckpt_name


    # Load-from flags (TRAIN resume vs INFER ckpt)
    cfg.resume_from = args.resume_from
    cfg.ckpt_path   = args.ckpt_path

    # Back-compat: if user passed --ckpt
    if args.ckpt and not cfg.resume_from and cfg.mode == "train": cfg.resume_from = args.ckpt
    if args.ckpt and not cfg.ckpt_path and cfg.mode == "infer": cfg.ckpt_path = args.ckpt

    # Save-as names
    cfg.best_name = args.best_name or args.ckpt_name or "best.pt"
    cfg.last_name = args.last_name or "last.pt"


    if args.mode == "train":
        assert cfg.prism_train_paths and cfg.prism_val_paths, "Provide --train_paths and --val_paths"
        train(cfg)
    elif args.mode == "infer":
        assert args.ckpt_path and args.pre_paths, "Provide --ckpt and --pre_paths"
        cfg.x_pre_paths = args.x_pre_paths if args.x_pre_paths else None
        infer_on_stack(cfg, args.ckpt_path, args.pre_paths, args.out_path,
                       static_dem_path=args.static_dem, static_mask_path=args.static_mask)

if __name__ == "__main__":
    main()
