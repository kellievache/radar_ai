
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 18:14:35 2026

@author: vachek
"""


# convective_correction_pytorch3.py
# Full PyTorch pipeline for PRISM convective-morphology correction
# with robust masking (-9999 & NaN), stable climatology, AMP, and safe I/O.
# Author: (You)
# Date: 2026-01-26

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

import datetime as _dt

try:
    import xarray as xr  # NetCDF/Zarr I/O
except ImportError:
    xr = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **k: x


# dataset_random_patches.py


from torch.utils.data import Dataset
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler



# Silence noisy "Mean of empty slice" warnings globally (we handle empties explicitly)
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)


# ---------------------------
# Reproducibility & utilities
# ---------------------------

def ddp_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def ddp_setup():
    """Initialize DDP if launched with torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank, dist.get_rank(), dist.get_world_size()
    else:
        return None, 0, 1  # single-process fallback

def ddp_cleanup():
    if ddp_is_initialized():
        dist.destroy_process_group()


def ddp_broadcast_scalar(val, device, dtype=torch.float32):
    """Broadcast a Python scalar from rank 0 to all ranks and return it (on every rank)."""
    if not ddp_is_initialized():
        return val
    t = torch.tensor([val], device=device, dtype=dtype)
    dist.broadcast(t, src=0)
    if dtype in (torch.int32, torch.int64, torch.bool):
        return int(t.item())
    return float(t.item())


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


def to_device(t, device):
    return t.to(device) if torch.is_tensor(t) else t


def sanitize_cfg_dict(cfg_dict: dict) -> dict:
    """
    Keep only keys present in the current Config dataclass.
    Makes checkpoints robust to config changes across versions.
    """
    try:
        valid = {f.name for f in fields(Config)}
        return {k: v for k, v in cfg_dict.items() if k in valid}
    except Exception:
        return cfg_dict or {}


def valid_mask(arr: np.ndarray, sentinel: float = -9999.0) -> np.ndarray:
    """
    1 for valid (finite & not sentinel), 0 for missing; handles NaNs and -9999.
    Works on (T,H,W) or (H,W). Returns float32.
    """
    m = np.isfinite(arr).astype(np.float32)
    m = np.where(arr == sentinel, 0.0, m).astype(np.float32)
    return m


def ensure_chw(t: torch.Tensor) -> torch.Tensor:
    """Ensure tensor has shape [1,H,W] (accepts [H,W] or [1,H,W])."""
    return t.unsqueeze(0) if t.dim() == 2 else t




def _is_zarr_store(path: str) -> bool:
    # Heuristics: endswith '.zarr' or contains consolidated metadata
    if path.endswith(".zarr"):
        return True
    if os.path.isdir(path) and os.path.exists(os.path.join(path, ".zmetadata")):
        return True
    return False

# def open_multi_auto(paths: List[str], time_dim: str, prefer_chunks_time1: bool = True):
#     """
#     Open a list of paths that may be NetCDFs (.nc) or Zarr stores (.zarr).
#     Returns an xarray.Dataset ready for lazy windowed reads.
#     """
#     if len(paths) == 1:
#         p = paths[0]
#         if _is_zarr_store(p):
#             # consolidated=True speeds metadata; chunks here tell Dask how to index 'time'
#             ds = xr.open_zarr(p, consolidated=True, chunks={time_dim: 1} if prefer_chunks_time1 else None)
#         else:
#             ds = xr.open_dataset(p, engine="netcdf4")
#             # If you want to avoid chunk-splitting warnings on NetCDF,
#             # you can choose to *not* set chunks here and rely on slicing in Dataset.
#             # Or, if you know on-disk time chunk (e.g., 72), you can do:
#             # ds = ds.chunk({time_dim: 72})
#         return ds

#     # Multiple paths: use open_mfdataset
#     all_zarr = all(_is_zarr_store(p) for p in paths)
#     if all_zarr:
#         ds = xr.open_mfdataset(
#             paths, engine="zarr", combine="by_coords",
#             chunks={time_dim: 1} if prefer_chunks_time1 else None,
#             backend_kwargs={"consolidated": True}
#         )
#     else:
#         ds = xr.open_mfdataset(paths, engine="netcdf4", combine="by_coords")
#         # Same comment as above: you may choose to .chunk({time_dim: <on-disk-time-chunk>})
#         # if you want to avoid chunk-splitting warnings.
#     return ds


def open_multi_auto(paths: List[str], time_dim: str, prefer_chunks_time1: bool = True):
    # prefer_chunks_time1 flag becomes NO-OP for Zarr to avoid splitting
    if len(paths) == 1:
        p = paths[0]
        if _is_zarr_store(p):
            # ⬇️ REMOVE the chunks=... line to avoid splitting stored chunks
            return xr.open_zarr(p, consolidated=True)
        else:
            return xr.open_dataset(p, engine="netcdf4")

    # Multi-path
    all_zarr = all(_is_zarr_store(p) for p in paths)
    if all_zarr:
        return xr.open_mfdataset(
            paths,
            engine="zarr",
            combine="by_coords",
            backend_kwargs={"consolidated": True}
            # ⬆️ NO chunks= here either; use native chunking
        )
    else:
        return xr.open_mfdataset(paths, engine="netcdf4", combine="by_coords")




# ---------------------------
# Checkpoint helpers
# ---------------------------
def get_state_dict(model: nn.Module) -> dict:
    """
    Return a clean (non-DataParallel-prefixed) state dict.
    """
    return model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()


def _expected_in_channels_from_state(state: dict) -> Optional[int]:
    """
    Inspect first conv weight to infer expected input channels in checkpoint.
    """
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
    """
    Load a (possibly DataParallel) checkpoint into `model`, stripping 'module.' and
    filtering out non-matching or missing keys to avoid runtime errors.
    """
    state = ckpt_obj["model"] if isinstance(ckpt_obj, dict) and "model" in ckpt_obj else ckpt_obj
    # strip 'module.' prefix if present
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
        print(f"[load_weights_robust] Missing (not found in ckpt): {len(missing)} keys (showing up to 10)")
        for k in missing[:10]:
            print("  -", k)
        if len(missing) > 10:
            print("  ...")
    if unexpected:
        print(f"[load_weights_robust] Unexpected (ignored in load): {len(unexpected)} keys (showing up to 10)")
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
    prism_train_paths: List[str] = None     # list of training NetCDF paths
    prism_val_paths: List[str] = None
    prism_var: str = "ppt"                  # variable name in NetCDF (e.g., 'ppt', 'precip')
    static_dem_path: Optional[str] = None   # optional DEM NetCDF/Zarr
    static_mask_path: Optional[str] = None  # optional external mask (1=valid)
    lat_name: str = "lat"
    lon_name: str = "lon"
    time_name: str = "time"
    # Transform
    transform: str = "log1p"                # 'log1p' | 'sqrt' | 'none'
    # Degradation (emulate radar-off)
    degrade_gauss_sigma_min: float = 0.8
    degrade_gauss_sigma_max: float = 2.5
    degrade_spectral_cutoff_min: float = 0.10  # fraction of Nyquist
    degrade_spectral_cutoff_max: float = 0.35
    degrade_intensity_damp_min: float = 0.85   # multiply field
    degrade_intensity_damp_max: float = 0.98
    degrade_additive_noise_std: float = 0.05   # on transformed space
    apply_degrade_prob: float = 1.0
    # Training
    epochs: int = 50
    min_epochs: int = 10
    batch_size: int = 4 #was 4, reduced to save memory.  Could be 1, but using torch.nn.dataparallel and so it has to be atleast 2 to reach 2 gpus.
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    base_ch: int = 64 #was 64 reduced to save memory
    num_workers: int = 2
    amp: bool = True
    patience: int = 8
    # Loss weights
    loss: str = "huber"                # 'l1' | 'huber'
    w_grad: float = 0.1                # gradient loss
    w_spec: float = 0.0                # spectral loss (0 -> disabled)
    w_mass: float = 0.01               # mass preservation penalty
    # FSS thresholds (mm/day)
    fss_thresholds: List[float] = None
    fss_window: int = 9
    # Static features usage
    use_dem: bool = False              # you haven't supplied one by default
    use_mask: bool = False             # we build mask from data anyway
    # Tiling for inference
    tile: int = 512
    tile_overlap: int = 32
    # Files
    out_dir: str = "/nfs/pancake/u5/projects/vachek/automate_qc/runs/convective_correction/"
    #out_dir: str = "./runs/convective_correction"
    ckpt_name: str = "best.pt"
    seed: int = 42
    device: str = "cuda"

    # Optional external X inputs.  These are non radar data. 
    x_train_paths: List[str] = None   # list of NetCDF/Zarr for X (train); if None -> degrade Y
    x_val_paths:   List[str] = None   # list of NetCDF/Zarr for X (val);   if None -> degrade Y
    x_pre_paths:   List[str] = None   # list of NetCDF/Zarr for X during inference; if None -> degrade Y
    x_var: Optional[str] = None       # variable name for X (defaults to prism_var if None)             Y is with radar and X is without radar
 

    # --- Training from spatial crops ---
    patch: int = 768                 # crop size H=W
    train_crops_per_epoch: int = 20000   # synthetic length: how many random crops per epoch.  Multiple of 4
    val_crop_mode: str = "random"    # 'random' or 'tile'
    val_crops_per_epoch: int = 4000  # used when val_crop_mode == 'random'
    val_tile: Optional[int] = None   # if None, default to cfg.tile
    val_tile_overlap: Optional[int] = None  # if None, default to cfg.tile_overlap

    degrade_in_dataset: bool = False  # new: do heavy ops in training step on GPU    
    
    
    patch = 768
    base_ch = 96
    batch_size: int = 8          
    w_grad = 0.3
    w_spec = 0.05
    w_hp   = 0.2
    w_fss  = 0.02
    degrade_gauss_sigma_min = 1.0
    degrade_gauss_sigma_max = 3.0
    degrade_spectral_cutoff_min = 0.08
    degrade_spectral_cutoff_max = 0.25
    tile_overlap: int = 192
    patience: int = 8
 
    # ...

    def __post_init__(self):
        if self.fss_thresholds is None:
            self.fss_thresholds = [1.0, 5.0, 10.0, 20.0]


# ---------------------------
# Degradation ops
# ---------------------------
def gaussian_blur2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur with reflect padding; x: [B,C,H,W]."""
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
    """Circular low-pass in frequency domain; x: [B,C,H,W]."""
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
    mask = mask.view(1, 1, H, W//2 + 1)
    X_f = X * mask
    x_f = torch.fft.irfft2(X_f, s=(H, W), norm='ortho')
    return x_f


def apply_degradation(y_mm: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Degrade in mm space; y_mm: [B,1,H,W]; returns [B,1,H,W]."""
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
    # HP = identity - gaussian low-pass; z in transformed space
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
# Dataset (random crops or deterministic tiles)
# ---------------------------
class PrismCropsDataset(Dataset):
    """
    Yields (X, Y, M) tensors for either random crops (training) or tiled windows (validation).

    - Loads full Y (and optional X) into host RAM ONCE (as before), but only returns a small
      HxW window per sample. GPU memory drops ~quadratically with patch size.
    - Channels in X: [x_tr, (dem?) , mask, clim_tr]
    - If no external X is provided, X is made by degrading Y (your radar-off emulation).

    Modes:
      * mode='train' -> random crops; __len__ = train_crops_per_epoch
      * mode='val'   -> 'random' or 'tile' depending on cfg.val_crop_mode
    """
    def __init__(self,
                 data_arrays: List[np.ndarray],           # list of [T_i,H,W] -> concat time
                 static_dem: Optional[np.ndarray],
                 static_mask: Optional[np.ndarray],
                 transform: str,
                 cfg: Config,
                 x_arrays: Optional[List[np.ndarray]] = None,  # optional external X [T,H,W]
                 mode: str = "train"):
        super().__init__()
        self.cfg = cfg
        self.transform = transform
        self.mode = mode

        # --- Concatenate Y (targets) over time: [T,H,W]
        self.Y = np.concatenate(data_arrays, axis=0).astype(np.float32)
        self.T, self.H, self.W = self.Y.shape

        # --- Optional external X aligned to Y
        self.X = None
        if x_arrays is not None:
            Xcat = np.concatenate(x_arrays, axis=0).astype(np.float32)
            if Xcat.shape != self.Y.shape:
                raise ValueError(
                    f"External X shape {Xcat.shape} does not match Y shape {self.Y.shape}. "
                    "Ensure time/lat/lon alignment and identical grid."
                )
            self.X = Xcat

        # --- Optional DEM / static mask
        self.dem = None
        if static_dem is not None and cfg.use_dem:
            dem = static_dem.astype(np.float32)
            dem = (dem - np.nanmean(dem)) / (np.nanstd(dem) + 1e-6)
            dem = np.where(np.isfinite(dem), dem, 0.0)
            self.dem = dem  # [H,W]

        self.user_static_mask = None
        if static_mask is not None and cfg.use_mask:
            m = (static_mask > 0.5).astype(np.float32)
            m = np.where(np.isfinite(m), m, 0.0)
            self.user_static_mask = m  # [H,W]

        # --- Domain mask from Y: valid at least once over time
        mask_all = valid_mask(self.Y, sentinel=-9999.0).astype(np.float32)   # [T,H,W]
        domain_mask = (np.nanmax(mask_all, axis=0) > 0.5).astype(np.float32) # [H,W]
        self.domain_mask = domain_mask
        print(f"[Dataset] Domain valid fraction: {float(domain_mask.mean()):.3f}")

        # --- Climatology in mm (mean over time where valid)
        Y_masked = np.where(mask_all > 0.5, self.Y, np.nan)
        clim = np.nanmean(Y_masked, axis=0).astype(np.float32)  # [H,W]
        empty = ~np.isfinite(clim)
        if empty.any():
            print(f"[Dataset] Climatology: {int(empty.sum())} pixels had no valid data; filled 0.")
        clim = np.where(np.isfinite(clim), clim, 0.0)
        self.clim = clim

        # --- Crop window planning
        self.patch = int(getattr(cfg, "patch", 256))
        # Safety for small grids
        self.patch_h = min(self.patch, self.H)
        self.patch_w = min(self.patch, self.W)

        if self.mode == "train":
            # synthetic length controls steps/epoch
            self.N = int(getattr(cfg, "train_crops_per_epoch", 20000))
            self._sampler = self._random_sampler
        else:
            val_mode = getattr(cfg, "val_crop_mode", "random").lower()
            if val_mode == "tile":
                vt = cfg.val_tile or cfg.tile
                vo = cfg.val_tile_overlap or cfg.tile_overlap
                self.tile = int(vt)
                self.tile_overlap = int(vo)
                self._make_val_tiles()  # sets self.tiles = [(t, y0, x0, h, w), ...]
                self.N = len(self.tiles)
                self._sampler = self._tile_sampler
                print(f"[ValTiles] {self.N} tiles (tile={self.tile}, overlap={self.tile_overlap})")
            else:
                self.N = int(getattr(cfg, "val_crops_per_epoch", 4000))
                self._sampler = self._random_sampler

        # RNGs
        self._rng_py = random.Random(cfg.seed if self.mode == "train" else cfg.seed + 1)

    # -------- samplers --------
    def _random_sampler(self, idx: int) -> Tuple[int, int, int, int, int]:
        t = self._rng_py.randrange(self.T)
        y0 = self._rng_py.randrange(0, max(1, self.H - self.patch_h + 1))
        x0 = self._rng_py.randrange(0, max(1, self.W - self.patch_w + 1))
        return t, y0, x0, self.patch_h, self.patch_w

    def _make_val_tiles(self):
        step = self.tile - self.tile_overlap
        tiles = []
        for t in range(self.T):
            for y0 in range(0, self.H, step):
                for x0 in range(0, self.W, step):
                    y1 = min(y0 + self.tile, self.H)
                    x1 = min(x0 + self.tile, self.W)
                    tiles.append((t, y0, x0, y1 - y0, x1 - x0))
        self.tiles = tiles

    def _tile_sampler(self, idx: int) -> Tuple[int, int, int, int, int]:
        return self.tiles[idx]

    # -------- helpers --------
    def __len__(self) -> int:
        return self.N

    def _slice2d(self, arr2d: np.ndarray, y0: int, x0: int, h: int, w: int) -> np.ndarray:
        return arr2d[y0:y0+h, x0:x0+w]

    def _slice3d_t(self, arr3d: np.ndarray, t: int, y0: int, x0: int, h: int, w: int) -> np.ndarray:
        return arr3d[t, y0:y0+h, x0:x0+w]

    def __getitem__(self, idx: int):
        # ---- pick window ----
        t, y0, x0, h, w = self._sampler(idx)

        # ---- Target y_mm crop & its mask ----
        y_mm = self._slice3d_t(self.Y, t, y0, x0, h, w)   # [h,w]
        m_data = valid_mask(y_mm, sentinel=-9999.0).astype(np.float32)   # [h,w]
        m_dom  = self._slice2d(self.domain_mask, y0, x0, h, w)           # [h,w]
        m = (m_data * m_dom).astype(np.float32)
        if self.user_static_mask is not None:
            m = (m * (self._slice2d(self.user_static_mask, y0, x0, h, w) > 0.5).astype(np.float32)).astype(np.float32)

        # sanitize target
        y_mm = np.where(m > 0.5, np.clip(y_mm, 0.0, 1e6), 0.0).astype(np.float32)
        y_t  = torch.from_numpy(y_mm)[None, ...]  # [1,h,w]

        # ---- Input X (external or degraded) in mm ----
        if self.X is not None:
            x_np = self._slice3d_t(self.X, t, y0, x0, h, w)
            x_np = np.where(np.isfinite(x_np), x_np, 0.0).astype(np.float32)
            x_np = np.where(m > 0.5, np.clip(x_np, 0.0, 1e6), 0.0)
            x_mm = torch.from_numpy(x_np)[None, ...]  # [1,h,w]
        else:
            # degrade Y -> X (mm space)
            #x_mm = apply_degradation(y_t.unsqueeze(0), self.cfg).squeeze(0)  # [1,h,w]
            #x_mm = x_mm * torch.from_numpy(m)[None, ...]
            
            # ⚠️ If we're doing degradation on GPU, just pass Y as X (in mm), no heavy ops here.
            if not self.cfg.degrade_in_dataset:
                x_mm = y_t.clone()  # [1,h,w], mm space, no blur/FFT on CPU
            else:
                # CPU-side degrade (only if explicitly requested)
                x_mm = apply_degradation(y_t.unsqueeze(0), self.cfg).squeeze(0)
            x_mm = x_mm * torch.from_numpy(m)[None, ...]


        # ---- transform to training space & add noise on input ----
        y_tr = mm_to_transform(y_t, self.transform).float()
        x_tr = mm_to_transform(x_mm, self.transform).float()
        x_tr = add_noise_transformed(x_tr, self.cfg.degrade_additive_noise_std)
        x_tr = x_tr * torch.from_numpy(m)[None, ...]

        # ---- static channels: dem, mask, climatology (transformed) ----
        chans = [x_tr]
        if self.dem is not None:
            chans.append(torch.from_numpy(self._slice2d(self.dem, y0, x0, h, w))[None, ...])
        chans.append(torch.from_numpy(m)[None, ...])  # mask channel
        clim_crop = self._slice2d(self.clim, y0, x0, h, w).astype(np.float32)
        clim_tr = mm_to_transform(torch.from_numpy(clim_crop)[None, ...], self.transform)
        clim_tr = clim_tr * torch.from_numpy(m)[None, ...]
        chans.append(clim_tr)

        X = torch.cat(chans, dim=0).float()            # [C,h,w]
        Y = y_tr.float()                                # [1,h,w]
        M = torch.from_numpy(m)[None, ...].float()      # [1,h,w]
        return X, Y, M
    

# ---------------------------
# Streaming random crops from NetCDF/Zarr via xarray+dask.  Loads partial input grids
# ---------------------------
class XRRandomPatchDataset(Dataset):
    """
    Random (or tiled) patches streamed from NetCDF/Zarr without preloading entire arrays.

    - Y paths (with radar) are required.
    - X paths (without radar) optional; if not provided, X is made by degrading Y.
    - Builds domain_mask (any valid over time) and climatology as 2D arrays via dask compute.
    - Channels: [x_tr, (dem?), mask, clim_tr] — identical to your training recipe.
    """

    def __init__(
        self,
        y_paths: List[str],
        x_paths: Optional[List[str]],
        cfg: Config,
        mode: str = "train",
    ):
        super().__init__()
        assert xr is not None, "xarray is required."

        self.cfg = cfg
        self.mode = mode
        self.patch = int(getattr(cfg, "patch", 256))
        self.transform = cfg.transform

        # Safer open (time-chunked) → pulls small windows; engine='netcdf4' helps on many systems.
      #  self.ds_y = xr.open_mfdataset(
      #      y_paths, combine="by_coords",
      #      chunks={cfg.time_name: 1}, engine="netcdf4"
      #  )
      
        self.ds_y = open_multi_auto(y_paths, time_dim=cfg.time_name, prefer_chunks_time1=True)

        self.var_y = cfg.prism_var
        self.t_dim = cfg.time_name
        # Infer spatial dims from the variable dims (last two)
        v = self.ds_y[self.var_y]
        self.y_dim = v.dims[-2]
        self.x_dim = v.dims[-1]
        self.T = int(self.ds_y.sizes[self.t_dim])
        self.H = int(self.ds_y.sizes[self.y_dim])
        self.W = int(self.ds_y.sizes[self.x_dim])

        # Optional X
     #   self.ds_x = None
     #   if x_paths:
     #       self.ds_x = xr.open_mfdataset(
     #           x_paths, combine="by_coords",
     #           chunks={cfg.time_name: 1}, engine="netcdf4"
     #       )
     #       self.var_x = cfg.x_var or cfg.prism_var

        if x_paths:
            self.ds_x = open_multi_auto(x_paths, time_dim=cfg.time_name, prefer_chunks_time1=True)
            self.var_x = cfg.x_var or cfg.prism_var
        else:
            self.ds_x = None

        # Compute 2D domain mask (valid at least once) out-of-core
        # This streams over time and keeps only [H,W] in memory at the end.
        mask_any = self.ds_y[self.var_y].pipe(xr.ufuncs.isfinite).any(dim=self.t_dim)
        self.domain_mask = mask_any.compute().astype(np.float32).values  # [H,W]

        # Compute climatology (2D) out-of-core
        clim = self.ds_y[self.var_y].where(xr.ufuncs.isfinite(self.ds_y[self.var_y])).mean(dim=self.t_dim)
        clim = clim.fillna(0.0)
        self.clim = clim.compute().astype(np.float32).values  # [H,W]

        # Optional statics (small, OK to load)
        self.dem = None
        if cfg.use_dem and cfg.static_dem_path:
            dem = load_static(cfg.static_dem_path)
            if dem is not None:
                dem = (dem - np.nanmean(dem)) / (np.nanstd(dem) + 1e-6)
                dem = np.where(np.isfinite(dem), dem, 0.0)
                self.dem = dem.astype(np.float32)

        self.user_static_mask = None
        if cfg.use_mask and cfg.static_mask_path:
            sm = load_static(cfg.static_mask_path)
            if sm is not None:
                sm = (sm > 0.5).astype(np.float32)
                sm = np.where(np.isfinite(sm), sm, 0.0)
                self.user_static_mask = sm.astype(np.float32)

        # Length (synthetic) and sampler
        if mode == "train":
            self.N = int(getattr(cfg, "train_crops_per_epoch", 20000))
            self._sampler = self._random_sampler
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

    def __len__(self):
        return self.N

    def _random_sampler(self, idx):
        t = self._rng.randrange(self.T)
        h = min(self.patch, self.H)
        w = min(self.patch, self.W)
        y0 = self._rng.randrange(0, max(1, self.H - h + 1))
        x0 = self._rng.randrange(0, max(1, self.W - w + 1))
        return t, y0, x0, h, w

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
        sl = ds[var].isel(
            **{
                self.t_dim: t,
                self.y_dim: slice(y0, y0 + h),
                self.x_dim: slice(x0, x0 + w),
            }
        )
        # Materialize only this small slice
        return np.asarray(sl.load().data, dtype=np.float32)

    def __getitem__(self, idx):
        t, y0, x0, h, w = self._sampler(idx)

        # --- Y patch (mm) and masks
        y_mm = self._window(self.ds_y, self.var_y, t, y0, x0, h, w)   # [h,w]
        m_data = valid_mask(y_mm, sentinel=-9999.0).astype(np.float32)
        m_dom  = self.domain_mask[y0:y0+h, x0:x0+w]
        m = (m_data * m_dom).astype(np.float32)
        if self.user_static_mask is not None:
            m = (m * (self.user_static_mask[y0:y0+h, x0:x0+w] > 0.5).astype(np.float32)).astype(np.float32)

        y_mm = np.where(m > 0.5, np.clip(y_mm, 0.0, 1e6), 0.0).astype(np.float32)
        y_t  = torch.from_numpy(y_mm)[None, ...]   # [1,h,w]

        # --- X (mm): external or degrade(Y)
        if self.ds_x is not None:
            x_np = self._window(self.ds_x, self.var_x, t, y0, x0, h, w)
            x_np = np.where(np.isfinite(x_np), x_np, 0.0).astype(np.float32)
            x_np = np.where(m > 0.5, np.clip(x_np, 0.0, 1e6), 0.0)
            x_mm = torch.from_numpy(x_np)[None, ...]
        else:
         #   x_mm = apply_degradation(y_t.unsqueeze(0), self.cfg).squeeze(0)
         #   x_mm = x_mm * torch.from_numpy(m)[None, ...]
            
            # ⚠️ If we're doing degradation on GPU, just pass Y as X (in mm), no heavy ops here.
            if not self.cfg.degrade_in_dataset:
                x_mm = y_t.clone()  # [1,h,w], mm space, no blur/FFT on CPU
            else:
                # CPU-side degrade (only if explicitly requested)
                x_mm = apply_degradation(y_t.unsqueeze(0), self.cfg).squeeze(0)
            x_mm = x_mm * torch.from_numpy(m)[None, ...]


        # --- Transform space + small noise on input
        y_tr = mm_to_transform(y_t, self.transform).float()
        x_tr = mm_to_transform(x_mm, self.transform).float()
        x_tr = add_noise_transformed(x_tr, self.cfg.degrade_additive_noise_std)
        x_tr = x_tr * torch.from_numpy(m)[None, ...]

        # --- Build channels
        chans = [x_tr]
        if self.dem is not None:
            chans.append(torch.from_numpy(self.dem[y0:y0+h, x0:x0+w])[None, ...])
        chans.append(torch.from_numpy(m)[None, ...])

        clim_crop = self.clim[y0:y0+h, x0:x0+w].astype(np.float32)
        clim_tr = mm_to_transform(torch.from_numpy(clim_crop)[None, ...], self.transform)
        clim_tr = clim_tr * torch.from_numpy(m)[None, ...]
        chans.append(clim_tr)

        X = torch.cat(chans, dim=0).float()        # [C,h,w]
        Y = y_tr.float()                            # [1,h,w]
        M = torch.from_numpy(m)[None, ...].float()  # [1,h,w]
        return X, Y, M
    

# ---------------------------
# Dataset  Loads entire NetCDF input grids
# ---------------------------
class PrismDataset(torch.utils.data.Dataset):
    """
    Training dataset building (X, Y, M) pairs:

    - Uses valid_mask (NaN & -9999) to build per-sample mask
    - Uses domain mask (valid-any-time) computed from Y to stabilize masks & climatology
    - Adds mask channel to model inputs
    - NEW: If x_arrays is provided, uses those as X (aligned [T,H,W]); otherwise X is built by degrading Y.
    """
    def __init__(self,
                 data_arrays: List[np.ndarray],           # Y arrays: list of [T_i, H, W] -> concat on time
                 static_dem: Optional[np.ndarray],
                 static_mask: Optional[np.ndarray],
                 transform: str,
                 cfg: Config,
                 x_arrays: Optional[List[np.ndarray]] = None # NEW: X arrays list (optional)
                 ):
        self.cfg = cfg
        self.transform = transform

        # Target Y (PRISM)
        self.Y = np.concatenate(data_arrays, axis=0).astype(np.float32)  # [T,H,W]
        self.T, self.H, self.W = self.Y.shape

        # Optional external X
        self.X = None
        if x_arrays is not None:
            Xcat = np.concatenate(x_arrays, axis=0).astype(np.float32)   # [T,H,W]
            if Xcat.shape != self.Y.shape:
                raise ValueError(
                    f"External X shape {Xcat.shape} does not match Y shape {self.Y.shape}. "
                    "Ensure time/lat/lon alignment and identical grid."
                )
            self.X = Xcat

        # Optional DEM
        self.dem = None
        if static_dem is not None and cfg.use_dem:
            dem = static_dem.astype(np.float32)
            dem = (dem - np.nanmean(dem)) / (np.nanstd(dem) + 1e-6)
            dem = np.where(np.isfinite(dem), dem, 0.0)
            self.dem = dem

        # Optional external static mask
        self.mask = None
        if static_mask is not None and cfg.use_mask:
            m = (static_mask > 0.5).astype(np.float32)
            m = np.where(np.isfinite(m), m, 0.0)
            self.mask = m

        # Build overall valid mask & domain mask (from Y)
        mask_all = valid_mask(self.Y, sentinel=-9999.0).astype(np.float32)   # [T,H,W]
        domain_mask = (np.nanmax(mask_all, axis=0) > 0.5).astype(np.float32) # [H,W]
        self.domain_mask = domain_mask
        print(f"[Dataset] Domain valid fraction (has data at least once): {float(domain_mask.mean()):.3f}")

        # Climatology over valid pixels; fill never-valid with 0
        Y_masked = np.where(mask_all > 0.5, self.Y, np.nan)
        clim = np.nanmean(Y_masked, axis=0).astype(np.float32)
        empty_cols = ~np.isfinite(clim)
        if empty_cols.any():
            print(f"[Dataset] Climatology: {int(empty_cols.sum())} pixels had no valid data; filled with 0.")
        clim = np.where(np.isfinite(clim), clim, 0.0)
        self.clim = clim

    def __len__(self):
        return self.T

    def __getitem__(self, idx):
        # --- Target (Y) and masks ---
        y_mm = self.Y[idx]  # [H,W]

        # Per-sample mask & intersect with domain mask (plus static mask if provided)
        m_data = valid_mask(y_mm, sentinel=-9999.0).astype(np.float32)
        m = (m_data * self.domain_mask).astype(np.float32)
        if self.mask is not None:
            m = (m * (self.mask > 0.5).astype(np.float32)).astype(np.float32)

        # Sanitize target
        y_mm_clean = np.where(m > 0.5, np.clip(y_mm, 0.0, 1e6), 0.0).astype(np.float32)
        y_t = torch.from_numpy(y_mm_clean)[None, ...]  # [1,H,W]

        # --- Input (X): external or degrade(Y) ---
        if self.X is not None:
            # Use external X supplied as NumPy array (already aligned)
            x_np = self.X[idx]  # [H,W]
            x_np = np.where(np.isfinite(x_np), x_np, 0.0).astype(np.float32)
            x_np = np.where(m > 0.5, np.clip(x_np, 0.0, 1e6), 0.0)
            x_mm = torch.from_numpy(x_np)[None, ...]  # [1,H,W]
        else:
            # Fallback: degrade Y to emulate radar-off characteristics
            y_t_b = y_t.unsqueeze(0)  # [1,1,H,W]
            x_mm = apply_degradation(y_t_b, self.cfg).squeeze(0)  # [1,H,W]
            # Re-enforce mask boundary after filtering
            x_mm = x_mm * torch.from_numpy(m)[None, ...]

        # --- Transform and noise (keep invalid at zero) ---
        y_tr = mm_to_transform(y_t, self.transform)
        x_tr = mm_to_transform(x_mm, self.transform)
        x_tr = add_noise_transformed(x_tr, self.cfg.degrade_additive_noise_std)
        x_tr = x_tr * torch.from_numpy(m)[None, ...]

        # --- Build input channels: [x_tr, dem?, mask, clim_tr(masked)] ---
        chans = [x_tr]
        if self.dem is not None:
            chans.append(torch.from_numpy(self.dem)[None, ...])
        chans.append(torch.from_numpy(m)[None, ...])  # mask channel

        clim_tr = mm_to_transform(torch.from_numpy(self.clim)[None, ...], self.transform)
        clim_tr = clim_tr * torch.from_numpy(m)[None, ...]
        chans.append(clim_tr)

        X = torch.cat(chans, dim=0).float()   # [C,H,W]
        Y = y_tr.float()                      # [1,H,W]
        M = torch.from_numpy(m)[None, ...].float()  # [1,H,W]
        return X, Y, M



# ---------------------------
# Model: U-Net (residual head)
# ---------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm='batch', act='silu'):
        super().__init__()
        Norm = nn.BatchNorm2d if norm == 'batch' else nn.InstanceNorm2d
        Act = nn.SiLU if act == 'silu' else nn.ReLU
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            Norm(out_ch),
            Act(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            Norm(out_ch),
            Act(inplace=True),
        )
    def forward(self, x): return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, **kw):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch, **kw)
    def forward(self, x): return self.conv(self.pool(x))


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
    def __init__(self, in_ch, out_ch): super().__init__(); self.conv = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x): return self.conv(x)


class ClimateUNet(nn.Module):
    """
    Residual U-Net on transformed space: predicts delta for channel-0 (x_tr).
    Inputs: [x_tr, dem?, mask, clim_tr]
    Output: delta_tr   (1 channel)
    """
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

    def forward(self, x):  # x: [B,C,H,W]
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


# ---------------------------
# Losses & Metrics
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


def spectral_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    B = pred.shape[0]
    loss = 0.0
    for b in range(B):
        p = pred[b:b+1]; t = target[b:b+1]
        P = torch.fft.rfft2(p, norm='ortho')
        T = torch.fft.rfft2(t, norm='ortho')
        loss += F.mse_loss(torch.abs(P), torch.abs(T))
    return loss / max(B, 1)


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


# ---------------------------
# Data loading helpers
# ---------------------------

def load_arrays(paths: List[str], var: str, time_name: str) -> np.ndarray:
    """
    Load a list of NetCDF/Zarr files and concat over time -> [T,H,W]
    """
    if xr is None:
        raise RuntimeError("xarray is required for NetCDF/Zarr loading.")
    arrs = []
    for p in paths:
        ds = xr.open_dataset(p)
        a = ds[var].load().values  # [T,H,W]
        ds.close()
        if a.ndim != 3:
            raise ValueError(f"Variable '{var}' in {p} is not [T,H,W]. Got shape {a.shape}.")
        arrs.append(a.astype(np.float32))
    return np.concatenate(arrs, axis=0)

def load_prism_arrays(paths: List[str], var: str, time_name: str) -> np.ndarray:
    # Backward-compatible wrapper
    return load_arrays(paths, var, time_name)



def load_static(path: Optional[str], var_guess: Optional[str] = None) -> Optional[np.ndarray]:
    if path is None:
        return None
    if xr is None:
        raise RuntimeError("xarray is required to load static rasters.")
    ds = xr.open_dataset(path)
    if var_guess is None:
        var_guess = list(ds.data_vars)[0]
    a = ds[var_guess].load().values
    ds.close()
    if a.ndim == 3:
        a = a[0]
    return a.astype(np.float32)


# ---------------------------
# BIL/HDR writing helpers
# ---------------------------

def _format_yyyymmdd(np_datetime) -> str:
    """
    Convert NumPy datetime (e.g., numpy.datetime64) to YYYYMMDD string.
    """
    try:
        # Try pandas-like conversion if available
        # But to avoid extra deps, do a robust fallback
        t = np.datetime_as_string(np_datetime, unit='D')  # 'YYYY-MM-DD'
        return t.replace('-', '')
    except Exception:
        # Fallback via python datetime
        if isinstance(np_datetime, (np.datetime64, )):
            # Convert to python datetime via astype
            ts = np_datetime.astype('datetime64[s]').astype('int')
            dt = _dt.datetime.utcfromtimestamp(int(ts))
            return dt.strftime('%Y%m%d')
        elif isinstance(np_datetime, _dt.date):
            return np_datetime.strftime('%Y%m%d')
        else:
            s = str(np_datetime)
            # Best effort: strip non-digits
            return ''.join(c for c in s if c.isdigit())[0:8]


def _compose_bil_hdr(
    nrows: int, ncols: int,
    ulxmap: float, ulymap: float,
    xdim: float, ydim: float,
    nbands: int = 1,
    nbits: int = 32,
    nodata: float = -9999.0,
    layout: str = "BIL",
    byteorder: str = "I",     # I = Intel = little-endian
    pixeltype: str = "FLOAT"
) -> str:
    """
    Build ESRI .hdr content for a BIL file.
    """
    band_row_bytes = ncols * (nbits // 8) * nbands  # for NBANDS=1 this is ncols*4
    total_row_bytes = band_row_bytes  # no padding

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


def write_bil_pair(
    arr2d: np.ndarray,
    bil_path: str,
    hdr_path: str,
    ulxmap: float,
    ulymap: float,
    xdim: float,
    ydim: float,
    nodata: float = -9999.0,
    force_little_endian: bool = True
):
    """
    Write a single-band 2D array as ESRI BIL + HDR.
    - arr2d: 2D numpy array [NROWS, NCOLS], north-up (row 0 = top row).
    - Values written as float32, NaNs replaced with nodata.
    """
    if arr2d.ndim != 2:
        raise ValueError(f"write_bil_pair expects a 2D array, got shape {arr2d.shape}")

    nrows, ncols = arr2d.shape

    # Replace NaNs with NODATA
    data = np.array(arr2d, dtype=np.float32, copy=True)
    data[~np.isfinite(data)] = nodata

    # Ensure little-endian if requested (BYTEORDER I)
    if force_little_endian and data.dtype.byteorder not in ('<', '='):
        data = data.byteswap().newbyteorder('<')

    # Write BIL (band interleaved by line). For single band, it's just row-major binary.
    with open(bil_path, 'wb') as f:
        data.tofile(f)

    # Write HDR
    hdr_txt = _compose_bil_hdr(
        nrows=nrows, ncols=ncols,
        ulxmap=ulxmap, ulymap=ulymap,
        xdim=xdim, ydim=ydim,
        nbands=1, nbits=32, nodata=nodata,
        layout="BIL", byteorder="I", pixeltype="FLOAT"
    )
    with open(hdr_path, 'w', encoding='ascii') as f:
        f.write(hdr_txt)



# ---------------------------
# Training
# ---------------------------
# def train(cfg: Config):
#     set_seed(cfg.seed)
#     #device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
#     local_rank, global_rank, world_size = ddp_setup()
#     if local_rank is not None:
#         device = torch.device(f"cuda:{local_rank}")
#         is_main = (global_rank == 0)
#     else:
#         device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
#         is_main = True

#     ensure_dir(cfg.out_dir)
#     with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
#         json.dump(asdict(cfg), f, indent=2)

#     # Load Y.  These work with the full netcdf files.  Can cause OOM with multiple year training
#     # Y_train = load_prism_arrays(cfg.prism_train_paths, cfg.prism_var, cfg.time_name)
#     # Y_val   = load_prism_arrays(cfg.prism_val_paths,   cfg.prism_var, cfg.time_name)

#     # # Optionally load X (external; else None -> degrade Y)
#     # x_var_name = cfg.x_var or cfg.prism_var
#     # X_train = load_arrays(cfg.x_train_paths, x_var_name, cfg.time_name) if cfg.x_train_paths else None
#     # X_val   = load_arrays(cfg.x_val_paths,   x_var_name, cfg.time_name) if cfg.x_val_paths   else None

#     # Statics
#     static_dem  = load_static(cfg.static_dem_path) if cfg.use_dem else None
#     static_mask = load_static(cfg.static_mask_path) if cfg.use_mask else None
#     if static_mask is not None:
#         static_mask = (static_mask > 0.5).astype(np.float32)

#     # Datasets (crop - unaware)
#     # ds_tr = PrismDataset([Y_train], static_dem, static_mask, cfg.transform, cfg, x_arrays=[X_train] if X_train is not None else None)
#     # ds_va = PrismDataset([Y_val],   static_dem, static_mask, cfg.transform, cfg, x_arrays=[X_val]   if X_val   is not None else None)

#     # dl_tr = torch.utils.data.DataLoader(
#     #     ds_tr, batch_size=cfg.batch_size, shuffle=True,
#     #     num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
#     #     persistent_workers=(cfg.num_workers > 0), prefetch_factor=2
#     # )
#     # dl_va = torch.utils.data.DataLoader(
#     #     ds_va, batch_size=cfg.batch_size, shuffle=False,
#     #     num_workers=cfg.num_workers, pin_memory=True,
#     #     persistent_workers=(cfg.num_workers > 0), prefetch_factor=2
#     # )

#     # Datasets (crop-aware)
#     # ds_tr = PrismCropsDataset(
#     #     [Y_train], static_dem, static_mask,
#     #     cfg.transform, cfg,
#     #     x_arrays=[X_train] if X_train is not None else None,
#     #     mode="train"
#     # )
#     # ds_va = PrismCropsDataset(
#     #     [Y_val], static_dem, static_mask,
#     #     cfg.transform, cfg,
#     #     x_arrays=[X_val] if X_val is not None else None,
#     #     mode="val"
#     # )

#     # dl_tr = torch.utils.data.DataLoader(
#     #     ds_tr,
#     #     batch_size=cfg.batch_size, shuffle=False,   # randomness comes from dataset's sampler
#     #     num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
#     #     persistent_workers=(cfg.num_workers > 0), prefetch_factor=2 if cfg.num_workers > 0 else None
#     # )
#     # dl_va = torch.utils.data.DataLoader(
#     #     ds_va,
#     #     batch_size=cfg.batch_size, shuffle=False,
#     #     num_workers=min(2, cfg.num_workers), pin_memory=True,
#     #     persistent_workers=(cfg.num_workers > 0), prefetch_factor=2 if cfg.num_workers > 0 else None
#     # )

#     # ✨ NEW: streaming datasets (no full-array preload)
#     ds_tr = XRRandomPatchDataset(
#         y_paths=cfg.prism_train_paths,
#         x_paths=cfg.x_train_paths,
#         cfg=cfg,
#         mode="train",
#     )
#     ds_va = XRRandomPatchDataset(
#         y_paths=cfg.prism_val_paths,
#         x_paths=cfg.x_val_paths,
#         cfg=cfg,
#         mode="val",
#     )

#     # Build loaders — start SAFE to verify no “Killed”
#     dl_tr = torch.utils.data.DataLoader(
#         ds_tr,
#         batch_size=cfg.batch_size,
#         shuffle=False,                 # randomness comes from dataset sampler
#         num_workers=0,                 # 🔴 start with 0 to avoid forking copies
#         pin_memory=True,
#         drop_last=True,
#         persistent_workers=False,
#     )
#     dl_va = torch.utils.data.DataLoader(
#         ds_va,
#         batch_size=cfg.batch_size,
#         shuffle=False,
#         num_workers=0,                 # 🔴 start with 0
#         pin_memory=True,
#         persistent_workers=False,
#     )

#     # Model
#     sample_x, _, _ = ds_tr[0]
#     in_ch = sample_x.shape[0]
#    # model = ClimateUNet(in_ch=in_ch, base_ch=cfg.base_ch).to(device)
   
#     model = ClimateUNet(in_ch=in_ch, base_ch=cfg.base_ch).to(device)
#     model = model.to(memory_format=torch.channels_last)  # <-- add this line

#     print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
#    # if torch.cuda.device_count() > 1:
#    #     model = torch.nn.DataParallel(model)

#     if ddp_is_initialized():
#         model = torch.nn.parallel.DistributedDataParallel(
#             model, device_ids=[device.index], output_device=device.index,
#             find_unused_parameters=False  # keeps comms lean
#         )


#     # Optimizer & AMP
#     # opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
#     # try:
#     #     scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp and device.type == "cuda")
#     #     def autocast_cm(): return torch.autocast('cuda', enabled=cfg.amp and device.type == "cuda")
#     # except Exception:
#     #     scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")
#     #     def autocast_cm(): return torch.cuda.amp.autocast(enabled=cfg.amp and device.type == "cuda")

#     opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    
#     # Use bf16 on H100s; typically no scaler needed for bf16
#     def autocast_cm():
#         return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.amp and device.type == "cuda"))
    
#     scaler = None  # not used with bf16

#     # LR scheduler on plateau (no verbose to support older torch)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         opt, mode='min', factor=0.5, patience=4, threshold=1e-5
#     )

#     # Early stopping & checkpoints
#     best_val = float('inf'); best_epoch = -1; patience_ctr = 0
#     ckpt_best = os.path.join(cfg.out_dir, cfg.ckpt_name)
#     ckpt_last = os.path.join(cfg.out_dir, "last.pt")

#     for epoch in range(1, cfg.epochs + 1):
#         model.train()
#         tr_losses = []
#         pbar = tqdm(dl_tr, desc=f"Epoch {epoch}/{cfg.epochs} [train]")
#         for X, Y_true, M in pbar:
#            # X = to_device(X, device); Y_true = to_device(Y_true, device); M = to_device(M, device)
           
#             X = X.to(device, non_blocking=True).to(memory_format=torch.channels_last)
#             Y_true = Y_true.to(device, non_blocking=True).to(memory_format=torch.channels_last)
#             M = M.to(device, non_blocking=True).to(memory_format=torch.channels_last)

#             opt.zero_grad(set_to_none=True)
            

#             # If no external X and we disabled dataset-side degradation, do it here on GPU.
#             if (cfg.x_train_paths is None) and (not cfg.degrade_in_dataset):
#                 with torch.no_grad():
#                     # Channel-0 currently holds x in *mm* (from dataset). Convert, degrade, and put back as transformed.
#                    # x_mm = X[:, 0:1]                                  # [B,1,H,W] mm
#                    # x_mm = torch.clamp(x_mm, min=0.0)
#                    # x_mm = apply_degradation(x_mm, cfg)               # blur/FFT/damp on GPU
#                    # x_tr = mm_to_transform(x_mm, cfg.transform)       # transformed for the model
#                    # X = torch.cat([x_tr, X[:, 1:]], dim=1)            # replace channel-0

            
#                     # channel-0 is transformed; invert to mm, degrade, then re-transform
#                     x_mm = transform_to_mm(X[:, 0:1], cfg.transform)
#                     x_mm = torch.clamp(x_mm, min=0.0)
#                     x_mm = apply_degradation(x_mm, cfg)
#                     x_tr = mm_to_transform(x_mm, cfg.transform)
#                     X = torch.cat([x_tr, X[:, 1:]], dim=1)
            
#             with autocast_cm():
#                 delta = model(X)
#                 Y_pred_tr = X[:, 0:1] + delta
#                 # Data loss (masked)
#                 if cfg.loss == "huber":
#                     data_loss = F.smooth_l1_loss(Y_pred_tr * M, Y_true * M)
#                 else:
#                     data_loss = F.l1_loss(Y_pred_tr * M, Y_true * M)
#                 # Gradient & mm penalties
#                 gl = gradient_loss(Y_pred_tr, Y_true, mask=M) * cfg.w_grad
#                 Y_pred_mm = transform_to_mm(Y_pred_tr, cfg.transform)
#                 Y_true_mm = transform_to_mm(Y_true,    cfg.transform)
#                 spec = spectral_loss(Y_pred_tr, Y_true) * cfg.w_spec if cfg.w_spec > 0 else 0.0
#                 mass = mass_preservation_penalty(Y_pred_mm, Y_true_mm, mask=M) * cfg.w_mass if cfg.w_mass > 0 else 0.0
#                 loss = data_loss + gl + spec + mass

#             #scaler.scale(loss).backward()
#             #scaler.step(opt)
#             #scaler.update()
            
#             loss.backward()
#             opt.step()

#             tr_losses.append(loss.item())
#             pbar.set_postfix(loss=np.mean(tr_losses))

#         # Validation
#         model.eval()
#         va_losses, va_rmse, va_mae = [], [], []
#         fss_scores: Dict[float, List[float]] = {thr: [] for thr in cfg.fss_thresholds}
#         with torch.no_grad():
#             for X, Y_true, M in tqdm(dl_va, desc=f"Epoch {epoch}/{cfg.epochs} [val]"):
#               #  X = to_device(X, device); Y_true = to_device(Y_true, device); M = to_device(M, device)
              
#                 X = X.to(device, non_blocking=True).to(memory_format=torch.channels_last)
#                 Y_true = Y_true.to(device, non_blocking=True).to(memory_format=torch.channels_last)
#                 M = M.to(device, non_blocking=True).to(memory_format=torch.channels_last)

#                 Y_pred_tr = X[:, 0:1] + model(X)
#                 if cfg.loss == "huber":
#                     data_loss = F.smooth_l1_loss(Y_pred_tr * M, Y_true * M)
#                 else:
#                     data_loss = F.l1_loss(Y_pred_tr * M, Y_true * M)
#                 gl = gradient_loss(Y_pred_tr, Y_true, mask=M) * cfg.w_grad
#                 Y_pred_mm = transform_to_mm(Y_pred_tr, cfg.transform)
#                 Y_true_mm = transform_to_mm(Y_true,    cfg.transform)
#                 spec = spectral_loss(Y_pred_tr, Y_true) * cfg.w_spec if cfg.w_spec > 0 else 0.0
#                 mass = mass_preservation_penalty(Y_pred_mm, Y_true_mm, mask=M) * cfg.w_mass if cfg.w_mass > 0 else 0.0
#                 vloss = data_loss + gl + spec + mass
#                 va_losses.append(vloss.item())
#                 va_rmse.append(rmse(Y_pred_mm, Y_true_mm, mask=M).item())
#                 va_mae.append(mae(Y_pred_mm, Y_true_mm, mask=M).item())
#                 for thr in cfg.fss_thresholds:
#                     fss_scores[thr].append(
#                         fss(Y_pred_mm, Y_true_mm, thr=thr, window=cfg.fss_window, mask=M).item()
#                     )

#         mean_val = float(np.mean(va_losses)) if len(va_losses) else float('inf')
#         mean_rmse = float(np.mean(va_rmse)) if len(va_rmse) else float('inf')
#         mean_mae = float(np.mean(va_mae)) if len(va_mae) else float('inf')
#         mean_fss = {thr: float(np.mean(v)) for thr, v in fss_scores.items()}
#         print(f"[Val] loss={mean_val:.4f} rmse={mean_rmse:.3f} mae={mean_mae:.3f} "
#               + " ".join([f"FSS@{thr}={mean_fss[thr]:.3f}" for thr in cfg.fss_thresholds]))

#         # Scheduler
#         if np.isfinite(mean_val):
#             scheduler.step(mean_val)

#         # Save "last" (ensure we don't keep 'module.' prefixes to simplify inference)
#         _sd = get_state_dict(model)
#         payload = {"model": _sd, "cfg": asdict(cfg), "epoch": epoch, "best_val": best_val}
#         torch.save(payload, ckpt_last)

#         # Save "best" if improved
#         improved = np.isfinite(mean_val) and (mean_val < best_val - 1e-5)
#         if improved:
#             best_val = mean_val; best_epoch = epoch; patience_ctr = 0
#             # Refresh payload with latest state dict in case DataParallel changes
#             payload = {"model": get_state_dict(model), "cfg": asdict(cfg), "epoch": epoch, "best_val": best_val}
#             torch.save(payload, ckpt_best)
#             print("  ↳ Saved new best checkpoint:", os.path.abspath(ckpt_best))
#         else:
#             patience_ctr += 1

#         # Early stop after min_epochs
#         if epoch >= cfg.min_epochs and patience_ctr >= cfg.patience:
#             print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch} (val={best_val:.6f})")
#             break

#     print("Training complete. Best checkpoint at:", os.path.abspath(ckpt_best))


def train(cfg: Config):
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)            # safer with NetCDF workers
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # -----------------------------
    # DDP setup & device
    # -----------------------------
    local_rank, global_rank, world_size = ddp_setup()
    if local_rank is not None:
        device = torch.device(f"cuda:{local_rank}")
        is_main = (global_rank == 0)
    else:
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        is_main = True

    # -----------------------------
    # Build streaming datasets
    # -----------------------------
    ds_tr = XRRandomPatchDataset(
        y_paths=cfg.prism_train_paths,
        x_paths=cfg.x_train_paths,
        cfg=cfg,
        mode="train",
    )
    ds_va = XRRandomPatchDataset(
        y_paths=cfg.prism_val_paths,
        x_paths=cfg.x_val_paths,
        cfg=cfg,
        mode="val",
    )

    # DistributedSampler for training
    if ddp_is_initialized():
        train_sampler = DistributedSampler(
            ds_tr, num_replicas=world_size, rank=global_rank,
            shuffle=False, drop_last=True
        )
    else:
        train_sampler = None

    # -----------------------------
    # DataLoaders
    # -----------------------------
    from torch.utils.data import DataLoader

    dl_tr = DataLoader(
        ds_tr,
        batch_size=cfg.batch_size,             # per-GPU if DDP
        sampler=train_sampler,
        shuffle=False,                         # randomness from dataset/sampler
        num_workers=8,                         # after stability: 8 for train
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    dl_va = None
    if is_main:
        dl_va = DataLoader(
            ds_va,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

    # -----------------------------
    # Model
    # -----------------------------
    sample_x, _, _ = ds_tr[0]
    in_ch = sample_x.shape[0]

    model = ClimateUNet(in_ch=in_ch, base_ch=cfg.base_ch).to(device)
    model = model.to(memory_format=torch.channels_last)

    # DDP wrap (instead of DataParallel)
    if ddp_is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index], output_device=device.index,
            find_unused_parameters=False
        )

    # Optimizer + bf16 autocast (no scaler)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    def autocast_cm():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                              enabled=(cfg.amp and device.type == "cuda"))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=4, threshold=1e-5
    )

    best_val = float('inf'); best_epoch = -1; patience_ctr = 0
    ckpt_best = os.path.join(cfg.out_dir, cfg.ckpt_name)
    ckpt_last = os.path.join(cfg.out_dir, "last.pt")

    # -----------------------------
    # Epoch loop
    # -----------------------------

    # -----------------------------
    # Epoch loop (DDP safe, no `break`)
    # -----------------------------
    epoch = 1
    stop_training = False
    
    while (epoch <= cfg.epochs) and (not stop_training):
        # If using DistributedSampler, reseed per-epoch
        if ddp_is_initialized() and isinstance(dl_tr.sampler, DistributedSampler):
            dl_tr.sampler.set_epoch(epoch)
    
        # ----------------- TRAIN -----------------
        model.train()
        running = 0.0
        log_every = 50
    
        pbar_iter = tqdm(dl_tr, desc=f"Epoch {epoch}/{cfg.epochs} [train]") if is_main else dl_tr
    
        for step, (X, Y_true, M) in enumerate(pbar_iter, 1):
            # Move to CUDA & channels_last
            X = X.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            Y_true = Y_true.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            M = M.to(device, non_blocking=True).to(memory_format=torch.channels_last)
    
            opt.zero_grad(set_to_none=True)
    
            # Optional on-GPU degradation (when no external X and dataset didn't degrade)
            if (cfg.x_train_paths is None) and (not cfg.degrade_in_dataset):
                with torch.no_grad():
                    x_mm = transform_to_mm(X[:, 0:1], cfg.transform)
                    x_mm = torch.clamp(x_mm, min=0.0)
                    x_mm = apply_degradation(x_mm, cfg)
                    x_tr = mm_to_transform(x_mm, cfg.transform)
                    X = torch.cat([x_tr, X[:, 1:]], dim=1)
    
            with autocast_cm():
                delta = model(X)
                Y_pred_tr = X[:, 0:1] + delta
    
                if cfg.loss == "huber":
                    data_loss = F.smooth_l1_loss(Y_pred_tr * M, Y_true * M)
                else:
                    data_loss = F.l1_loss(Y_pred_tr * M, Y_true * M)
    
                gl = gradient_loss(Y_pred_tr, Y_true, mask=M) * cfg.w_grad
                Y_pred_mm = transform_to_mm(Y_pred_tr, cfg.transform)
                Y_true_mm = transform_to_mm(Y_true,    cfg.transform)
                spec = spectral_loss(Y_pred_tr, Y_true) * cfg.w_spec if cfg.w_spec > 0 else 0.0
                mass = mass_preservation_penalty(Y_pred_mm, Y_true_mm, mask=M) * cfg.w_mass if cfg.w_mass > 0 else 0.0
                loss = data_loss + gl + spec + mass
    
            loss.backward()
            opt.step()
    
            # Throttled logging to avoid GPU sync every step
            running += float(loss.detach())
            if is_main and (step % log_every == 0) and hasattr(pbar_iter, "set_postfix"):
                pbar_iter.set_postfix(loss=running / log_every)
                running = 0.0
    
        # -------------- VALIDATION (rank 0 computes) --------------
        if is_main:
            model.eval()
            va_losses, va_rmse, va_mae = [], [], []
            fss_scores: Dict[float, List[float]] = {thr: [] for thr in cfg.fss_thresholds}
            with torch.no_grad():
                for X, Y_true, M in tqdm(dl_va, desc=f"Epoch {epoch}/{cfg.epochs} [val]"):
                    X = X.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                    Y_true = Y_true.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                    M = M.to(device, non_blocking=True).to(memory_format=torch.channels_last)
    
                    Y_pred_tr = X[:, 0:1] + model(X)
                    if cfg.loss == "huber":
                        data_loss = F.smooth_l1_loss(Y_pred_tr * M, Y_true * M)
                    else:
                        data_loss = F.l1_loss(Y_pred_tr * M, Y_true * M)
                    gl = gradient_loss(Y_pred_tr, Y_true, mask=M) * cfg.w_grad
                    Y_pred_mm = transform_to_mm(Y_pred_tr, cfg.transform)
                    Y_true_mm = transform_to_mm(Y_true,    cfg.transform)
                    spec = spectral_loss(Y_pred_tr, Y_true) * cfg.w_spec if cfg.w_spec > 0 else 0.0
                    mass = mass_preservation_penalty(Y_pred_mm, Y_true_mm, mask=M) * cfg.w_mass if cfg.w_mass > 0 else 0.0
                    vloss = data_loss + gl + spec + mass
                    va_losses.append(float(vloss.detach()))
                    va_rmse.append(rmse(Y_pred_mm, Y_true_mm, mask=M).item())
                    va_mae.append(mae(Y_pred_mm, Y_true_mm, mask=M).item())
                    for thr in cfg.fss_thresholds:
                        fss_scores[thr].append(
                            fss(Y_pred_mm, Y_true_mm, thr=thr, window=cfg.fss_window, mask=M).item()
                        )
            mean_val = float(np.mean(va_losses)) if len(va_losses) else float('inf')
            mean_rmse = float(np.mean(va_rmse)) if len(va_rmse) else float('inf')
            mean_mae  = float(np.mean(va_mae))  if len(va_mae)  else float('inf')
            mean_fss  = {thr: float(np.mean(v)) for thr, v in fss_scores.items()}
            print(f"[Val] loss={mean_val:.4f} rmse={mean_rmse:.3f} mae={mean_mae:.3f} "
                  + " ".join([f"FSS@{thr}={mean_fss[thr]:.3f}" for thr in cfg.fss_thresholds]))
        else:
            mean_val = float('inf')
    
        # --- Broadcast mean_val so EVERY rank steps the scheduler the same way
        mean_val = ddp_broadcast_scalar(mean_val, device, torch.float32)
        scheduler.step(mean_val)
    
        # --- Rank 0 computes improved & early-stop flags, then broadcast to all ranks
        if is_main:
            improved_flag = int(np.isfinite(mean_val) and (mean_val < best_val - 1e-5))
            should_stop_flag = int(epoch >= cfg.min_epochs and patience_ctr >= cfg.patience)
        else:
            improved_flag = 0
            should_stop_flag = 0
    
        improved_flag    = ddp_broadcast_scalar(improved_flag,    device, torch.int32)
        should_stop_flag = ddp_broadcast_scalar(should_stop_flag, device, torch.int32)
    
        # --- Save only on rank 0 if improved; update patience only on rank 0
        if is_main:
            # Save "last" (optional)
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save({"model": state_dict, "cfg": asdict(cfg), "epoch": epoch, "best_val": best_val},
                       os.path.join(cfg.out_dir, "last.pt"))
    
            if improved_flag:
                best_val = mean_val
                best_epoch = epoch
                patience_ctr = 0
                torch.save({"model": state_dict, "cfg": asdict(cfg), "epoch": epoch, "best_val": best_val},
                           os.path.join(cfg.out_dir, cfg.ckpt_name))
            else:
                patience_ctr += 1
    
        # --- Barrier so all processes finish the epoch together
        if ddp_is_initialized():
            dist.barrier()
    
        # --- Early stop decision INSIDE the loop, but without `break`
        if should_stop_flag:
            if is_main:
                print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch} (val={best_val:.6f})")
            stop_training = True
    
        epoch += 1  # advance loop guard
    
    # Final sync before cleanup (avoid NCCL teardown races)
    if ddp_is_initialized():
        dist.barrier()
    
        if ddp_is_initialized():
            dist.barrier()

    if is_main:
        print("Training complete. Best checkpoint at:", os.path.abspath(ckpt_best))
    ddp_cleanup()

# ---------------------------
# Inference
# ---------------------------
def tile_infer(model: nn.Module, X: torch.Tensor, tile: int, overlap: int) -> torch.Tensor:
    """
    X: [1,C,H,W] on device; returns Y_pred_tr = X_tr + delta
    """
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
            with torch.no_grad():
                delta = model(patch)
                y_patch = patch[:, 0:1] + delta
            out[..., ys, xs] += y_patch
            weight[..., ys, xs] += 1.0
    out = out / weight.clamp_min(1.0)
    return out


def infer_on_stack(cfg: Config,
                   ckpt_path: str,
                   pre_paths: List[str],           # Y files
                   out_path: str,
                   static_dem_path: Optional[str] = None,
                   static_mask_path: Optional[str] = None):
    """
    Apply trained model to pre-2001 PRISM.
    NOTE: Training always included a mask channel (per-sample, stabilized with a domain mask),
    so inference must also include a mask channel regardless of model_cfg.use_mask.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    model_cfg = Config(**ckpt["cfg"])

    if xr is None:
        raise RuntimeError("xarray required for inference I/O.")

    # ---- Load entire pre stack once (compute domain mask and climatology) ----
    arrs = []
    times_all = []
    for p in pre_paths:
        ds = xr.open_dataset(p)
        a = ds[model_cfg.prism_var].load()  # DataArray [time, lat, lon]
        arrs.append(a.values.astype(np.float32))
        times_all.append(ds[model_cfg.time_name].values)
        ds.close()

    pre_all = np.concatenate(arrs, axis=0)  # [T,H,W]

    # Discover grid from the first file
    ds0 = xr.open_dataset(pre_paths[0])
    H = ds0.sizes[model_cfg.lat_name]; W = ds0.sizes[model_cfg.lon_name]
    lat = ds0[model_cfg.lat_name]; lon = ds0[model_cfg.lon_name]
    ds0.close()

    # ---- DEM (optional; not used in your case but kept for completeness) ----
    dem = load_static(static_dem_path) if model_cfg.use_dem else None

    # ---- Domain mask (valid at least once) & per-frame masks will use it ----
    domain_mask = np.isfinite(pre_all).any(axis=0).astype(np.float32)  # [H,W]

    # ---- Climatology (mm) and transformed ----
    clim = np.nanmean(np.where(np.isfinite(pre_all), pre_all, np.nan), axis=0).astype(np.float32)
    clim = np.where(np.isfinite(clim), clim, 0.0)
    clim_t = mm_to_transform(torch.from_numpy(clim)[None, ...].float(), model_cfg.transform).to(device)

    # ---- Build model with correct channel count (ALWAYS include mask channel) ----
    # Channels: x_tr (1) + (dem?0/1) + mask (1) + clim_tr (1)
    chans = 1 + (1 if dem is not None else 0) + 1 + 1
    model = ClimateUNet(in_ch=chans, base_ch=model_cfg.base_ch).to(device)

    # Check expected in_ch from checkpoint vs constructed
    _state = ckpt["model"] if "model" in ckpt else ckpt
    _exp_in = _expected_in_channels_from_state(_state)
    if _exp_in is not None and _exp_in != chans:
        raise RuntimeError(
            f"Checkpoint expects in_ch={_exp_in}, but inference constructed in_ch={chans}. "
            f"This code now always includes a mask channel to match training; if your checkpoint truly "
            f"was trained differently, verify the Dataset channel recipe."
        )

    # Robust load of weights (handles DataParallel 'module.' prefix)
    load_weights_robust(model, ckpt)
    model.eval()

    out_times = []
    out_days = []

    # Optional X files for inference (pairwise with pre_paths)
    x_files = cfg.x_pre_paths if cfg.x_pre_paths else None
    if x_files is not None and len(x_files) != len(pre_paths):
        raise ValueError("--x_pre_paths must have the same number of files as --pre_paths (pairwise).")

    for i, p in enumerate(pre_paths):
        dsY = xr.open_dataset(p)
        y_mm_np = dsY[model_cfg.prism_var].load().values  # [T,H,W]
        times = dsY[model_cfg.time_name].values
        dsY.close()

        # Optional: load X for this chunk
        x_mm_np = None
        if x_files is not None:
            dsX = xr.open_dataset(x_files[i])
            xvar = model_cfg.x_var or model_cfg.prism_var
            x_mm_np = dsX[xvar].load().values  # [T,H,W]
            dsX.close()
            if x_mm_np.shape != y_mm_np.shape:
                raise ValueError(f"X shape {x_mm_np.shape} does not match Y shape {y_mm_np.shape} for files:\n"
                                 f"Y={p}\nX={x_files[i]}")

        T = y_mm_np.shape[0]
        corrected = np.zeros_like(y_mm_np, dtype=np.float32)

        for t in tqdm(range(T), desc=os.path.basename(p)):
            # Prepare Y[t] as baseline for constructing per-frame mask and (if needed) degradation
            y_mm = y_mm_np[t]
            y_mm = np.where(np.isfinite(y_mm), y_mm, 0.0).astype(np.float32)
            y_mm = np.clip(y_mm, 0.0, 1e5)
            y_t = torch.from_numpy(y_mm)[None, None, ...].to(device)  # [1,1,H,W]

            # Prepare X[t]: either supplied or degrade(Y[t])
            if x_mm_np is not None:
                x_mm = x_mm_np[t]
                x_mm = np.where(np.isfinite(x_mm), x_mm, 0.0).astype(np.float32)
                x_mm = np.clip(x_mm, 0.0, 1e5)
                x_tr = mm_to_transform(torch.from_numpy(x_mm)[None, None, ...].float().to(device),
                                       model_cfg.transform)
            else:
                x_mm = apply_degradation(y_t, model_cfg)  # [1,1,H,W]
                x_tr = mm_to_transform(x_mm, model_cfg.transform)

            # --- Per-frame mask channel (matches training recipe) ---
            # m_data from current y_mm; intersect with domain_mask (stabilizes holes over time)
            m_data = (np.isfinite(y_mm)).astype(np.float32)
            m = (m_data * domain_mask).astype(np.float32)  # [H,W]
            mask_t = torch.from_numpy(m)[None, None, ...].to(device)  # [1,1,H,W]

            # Build channels: [x_tr, (dem?), mask, clim_t]
            chans_list = [x_tr.squeeze(0)]  # [1,H,W]
            if dem is not None:
                dem_n = (dem - np.nanmean(dem)) / (np.nanstd(dem) + 1e-6)
                dem_n = np.where(np.isnan(dem_n), 0.0, dem_n)
                chans_list.append(torch.from_numpy(dem_n)[None, ...].to(device))
            chans_list.append(mask_t.squeeze(0))  # [1,H,W]
            chans_list.append(clim_t.to(device))
            X = torch.cat(chans_list, dim=0).unsqueeze(0)  # [1,C,H,W]

            # Predict (tile or full)
            if max(H, W) > model_cfg.tile:
                y_pred_tr = tile_infer(model, X, tile=model_cfg.tile, overlap=model_cfg.tile_overlap)
            else:
                with torch.no_grad():
                    delta = model(X)
                    y_pred_tr = X[:, 0:1] + delta

            # Back to mm
            y_pred_mm = transform_to_mm(y_pred_tr, model_cfg.transform)
            y_pred_mm = torch.clamp(y_pred_mm, min=0.0)

            # Optional mass preservation vs. original Y (masked area)
            s_pred = (y_pred_mm * mask_t).sum()
            s_true = (y_t * mask_t).sum()
            scale = (s_true / s_pred.clamp_min(1e-6)).item()
            scale = np.clip(scale, 0.8, 1.25)
            scale = 1
            y_pred_mm = y_pred_mm * scale

            corrected[t] = y_pred_mm.squeeze().detach().cpu().numpy()

        out_times.append(times)
        out_days.append(corrected)
    
        netcdf_out=False
       
        if netcdf_out:
           Tall = np.concatenate(out_days, axis=0)
           times_all = np.concatenate(out_times, axis=0)
           ds_out = xr.Dataset({cfg.prism_var: (("time", "lat", "lon"), Tall)},
                               coords={"time": times_all})
           ds_ref = xr.open_dataset(pre_paths[0])
           ds_out = ds_out.assign_coords(lat=ds_ref[cfg.lat_name], lon=ds_ref[cfg.lon_name])
           ds_ref.close()
           ensure_dir(os.path.dirname(out_path))
           ds_out.to_netcdf(out_path)
           print(f"Wrote corrected dataset → {out_path}")
        else:     
           Tall = np.concatenate(out_days, axis=0)        # [T,H,W] float32
           times_all = np.concatenate(out_times, axis=0)  # [T] np.datetime64[...]
           
           # Base dir from out_path
           #base_dir = os.path.dirname(os.path.abspath(out_path))
           base_dir = os.path.abspath(out_path.rstrip("/"))
           # Georeferencing constants you provided:
           ULXMAP = -125.016666666666
           ULYMAP = 49.933333333333
           XDIM   = 0.008333333333
           YDIM   = 0.008333333333
           NODATA = -9999.0
           
           # Orientation: ensure row 0 is the northernmost row.
           ds0 = xr.open_dataset(pre_paths[0])
           lat = ds0[cfg.lat_name].values
           ds0.close()
           need_flip = (lat[0] < lat[-1])  # True if lat ascending south->north
           
           if need_flip:
               Tall_to_write = Tall[:, ::-1, :]
           else:
               Tall_to_write = Tall
           
           # Write each time slice
           count = 0
           for i in range(Tall_to_write.shape[0]):
               yyyymmdd = _format_yyyymmdd(times_all[i])
               yyyy = yyyymmdd[:4]
           
               # Ensure per-year subdirectory
               year_dir = os.path.join(base_dir, yyyy)
               os.makedirs(year_dir, exist_ok=True)
           
               base = f"adj_best_ppt_us_us_30s_{yyyymmdd}"
               bil_path = os.path.join(year_dir, base + ".bil")
               hdr_path = os.path.join(year_dir, base + ".hdr")
           
               write_bil_pair(
                   arr2d=Tall_to_write[i],
                   bil_path=bil_path,
                   hdr_path=hdr_path,
                   ulxmap=ULXMAP,
                   ulymap=ULYMAP,
                   xdim=XDIM,
                   ydim=YDIM,
                   nodata=NODATA,
                   force_little_endian=True
               )
               count += 1
           
           print(f"Wrote {count} BIL/HDR pairs under {base_dir}/<YYYY>/")
          


# ---------------------------
# CLI
# ---------------------------


def build_arg_parser():
    ap = argparse.ArgumentParser(description="Convective morphology correction for PRISM (PyTorch).")
    ap.add_argument("--mode", default="train", choices=["train", "infer"], required=True)
    ap.add_argument("--config", type=str, help="JSON config path (optional).")

    # Existing Y (PRISM) paths
    ap.add_argument("--train_paths", type=str, default='/nfs/pancake/u5/projects/vachek/automate_qc/netcdf/PRISM_daily_ppt_2009_2010.nc', nargs="*", help="Post-2001 train NetCDF/Zarr files (Y).")
    ap.add_argument("--val_paths",   type=str, default='/nfs/pancake/u5/projects/vachek/automate_qc/netcdf/PRISM_daily_ppt_2014_2016.nc', nargs="*", help="Post-2001 val NetCDF/Zarr files (Y).")

    # NEW: X paths (external degraded inputs)
    ap.add_argument("--x_train_paths", type=str, nargs="*", help="Train NetCDF/Zarr files for X (degraded/alternate). If omitted, X is made by degrading Y.")
    ap.add_argument("--x_val_paths",   type=str, nargs="*", help="Val   NetCDF/Zarr files for X (degraded/alternate). If omitted, X is made by degrading Y.")
    ap.add_argument("--x_var",         type=str, default=None, help="Variable name for X (defaults to --prism_var if not set).")

    ap.add_argument("--static_dem",  type=str, default=None)
    ap.add_argument("--static_mask", type=str, default=None)
    ap.add_argument("--ckpt",        type=str, default=None)

    ap.add_argument("--pre_paths",   type=str, default='/nfs/pancake/u5/projects/vachek/automate_qc/netcdf/PRISM_daily_ppt_2009_2010.nc', nargs="*", help="Pre-2001 NetCDF/Zarr files for Y during inference.")
    # NEW: X at inference (optional)
    ap.add_argument("--x_pre_paths", type=str, nargs="*", help="Pre-2001 NetCDF/Zarr files for X during inference; if omitted, X is produced by degrading Y.")
    ap.add_argument("--out_path",    type=str, default="./corrected_pre2001.nc")
    
    ap.add_argument("--batch_size", type=int, help="Per-GPU batch size (DDP) or global for single GPU.")
    return ap


def main():
    ap = build_arg_parser()
    args = ap.parse_args()
    cfg = Config()

    # config file override as you already have...
    if args.config is not None:
        with open(args.config, "r") as f:
            j = json.load(f)
        for k, v in j.items():
            setattr(cfg, k, v)

    # CLI overrides for Y
    if args.train_paths: cfg.prism_train_paths = args.train_paths
    if args.val_paths:   cfg.prism_val_paths   = args.val_paths

    # NEW: CLI overrides for X
    if args.x_train_paths: cfg.x_train_paths = args.x_train_paths
    if args.x_val_paths:   cfg.x_val_paths   = args.x_val_paths
    if args.x_var:         cfg.x_var         = args.x_var

    # statics
    if args.static_dem:    cfg.static_dem_path  = args.static_dem
    if args.static_mask:   cfg.static_mask_path = args.static_mask

    if args.out_path:      cfg.out_path  = args.out_path
    if args.ckpt:          cfg.ckpt      = args.ckpt
    
    
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size


    if args.mode == "train":
        assert cfg.prism_train_paths and cfg.prism_val_paths, "Provide --train_paths and --val_paths"
        train(cfg)

    elif args.mode == "infer":
        assert args.ckpt and args.pre_paths, "Provide --ckpt and --pre_paths"
        # NEW: pass optional X paths
        cfg.x_pre_paths = args.x_pre_paths if args.x_pre_paths else None
        infer_on_stack(cfg, args.ckpt, args.pre_paths, args.out_path,
                       static_dem_path=args.static_dem, static_mask_path=args.static_mask)

if __name__ == "__main__":
    main()


# runfile(
#     '/nfs/pancake/u5/projects/vachek/automate_qc/convective_correction_pytorch3.py',
#     args="""--mode train
#             --train_paths /nfs/pancake/u5/projects/vachek/automate_qc/netcdf/radar_ppt_800_2023.nc
#             --val_paths   /nfs/pancake/u5/projects/vachek/automate_qc/netcdf/radar_ppt_800_2024.nc
#             --x_train_paths /nfs/pancake/u5/projects/vachek/automate_qc/netcdf/noradar_ppt_800_2023.nc
#             --x_val_paths   /nfs/pancake/u5/projects/vachek/automate_qc/netcdf/noradar_ppt_800_2024.nc
#             --x_var ppt"""
# )


# runfile('/nfs/pancake/u5/projects/vachek/automate_qc/convective_correction_pytorch3.py',
#         args='--mode train '
#              '--train_paths /nfs/pancake/u5/projects/vachek/automate_qc/netcdf/radar_ppt_800_2021_2023.nc '
#              '--val_paths /nfs/pancake/u5/projects/vachek/automate_qc/netcdf/radar_ppt_800_2024_2025.nc'
#              '--x_train_paths /nfs/pancake/u5/projects/vachek/automate_qc/netcdf/noradar_ppt_800_2021_2023.nc '
#              '--x_val_paths /nfs/pancake/u5/projects/vachek/automate_qc/netcdf/noradar_ppt_800_2024_2025.nc'
#              '--x_var ppt' 
#              )


# runfile('/nfs/pancake/u5/projects/vachek/automate_qc/convective_correction_pytorch3.py',
#         args='--mode infer '
#              '--ckpt /nfs/pancake/u5/projects/vachek/automate_qc/runs/convective_correction/best.pt '
#              '--pre_paths /nfs/pancake/u5/projects/vachek/automate_qc/netcdf/noradar_ppt_2015.nc '
#              '--out_path /nfs/pancake/u5/projects/vachek/automate_qc/netcdf/radar_informed_ppt_2015.nc')

# runfile('/nfs/pancake/u5/projects/vachek/automate_qc/convective_correction_pytorch4.py',
#         args='--mode infer '
#              '--ckpt /nfs/pancake/u5/projects/vachek/automate_qc/runs/convective_correction/best.pt '
#              '--pre_paths /nfs/pancake/u5/projects/vachek/automate_qc/netcdf/noradar_ppt_800_2015.nc '
#              '--out_path /nfs/pancake/u4/data/prism/us/an91/r2112_unet/ehdr/800m/ppt/daily/')

# runfile(
#    '/nfs/pancake/u5/projects/vachek/radar_ai/unet_radar_correction.py',
#    args="""--mode train
#            --train_paths /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/y_train.nc
#            --val_paths   /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/y_val.nc
#            --x_train_paths /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/x_train.nc
#            --x_val_paths   /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/x_val.nc
#            --out_path     /nfs/pancake/u5/projects/vachek/radar_ai/models/
#            --ckpt         model_2015_2019.pt
#            --x_var ppt"""


# python -m torch.distributed.run --standalone --nproc_per_node=2 /nfs/pancake/u5/projects/vachek/radar_ai/unet_radar_correction.py \
#   --mode train \
#   --train_paths /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/y_train.nc \
#   --val_paths   /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/y_val.nc \
#   --x_train_paths /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/x_train.nc \
#   --x_val_paths   /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/x_val.nc \
#   --x_var ppt \
#   --batch_size 2

# python -m torch.distributed.run --standalone --nproc_per_node=2 /nfs/pancake/u5/projects/vachek/radar_ai/unet_radar_correction.py \
#   --mode train \
#   --train_paths /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/y_train.zarr \
#   --val_paths   /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/y_val.zarr \
#   --x_train_paths /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/x_train.zarr \
#   --x_val_paths   /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/x_val.zarr \
#   --x_var ppt \
#   --batch_size 2

#  python -m torch.distributed.run --standalone --nproc_per_node=2 /nfs/pancake/u5/projects/vachek/radar_ai/unet_radar_correction.py \
#      --mode infer \
#     --ckpt /nfs/pancake/u5/projects/vachek/automate_qc/runs/convective_correction/best.pt  \
#     --pre_paths /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/infer.nc  \
#     --out_path /nfs/pancake/u4/data/prism/us/an91/r2112_unet1/ehdr/800m/ppt/daily/)
    
    


#python -m torch.distributed.run --standalone --nproc_per_node=2 /nfs/pancake/u5/projects/vachek/radar_ai/unet_radar_correction.py --mode infer --ckpt /nfs/pancake/u5/projects/vachek/automate_qc/runs/convective_correction/best.pt --pre_paths /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/infer.nc --x_pre_paths /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/infer.nc --x_var ppt --out_path /nfs/pancake/u4/data/prism/us/an91/r2112_unet/ehdr/800m/ppt/daily/











