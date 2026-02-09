#!/bin/csh
# --- OPTIONAL: load modules if your system uses them ---
# module load cuda

# --- IMPORTANT: activate conda in csh ---
# If your conda install is healthy:
source /nfs/pancake/u2/home/vachek/miniconda3/etc/profile.d/conda.csh
conda activate AG_15_PY312

# --- Run training ---
python unet_radar_correction3.py \
  --mode train \
  --ckpt /nfs/pancake/u5/projects/vachek/radar_ai/models/best_4km_prismcpu2.pt \
  --train_paths /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/y_train.zarr \
  --val_paths /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/y_val.zarr \
  --x_train_paths /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/x_train.zarr \
  --x_val_paths /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/x_val.zarr \
  --x_var ppt \
  --steps_per_epoch 1200 \
  --val_steps 250 \
  --batch_size 16 \
  --domain_mask_npy /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/prism_domain_mask.npy \
  --climatology_npy /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/prism_daily_normals_366.npy \
  --pre_paths /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/infer.nc \
  --out_path /nfs/pancake/u4/data/prism/us/an91/r2112_unet/ehdr/800m/ppt/daily/ \
  --out_dir /nfs/pancake/u5/projects/vachek/radar_ai/models/x \
  --ckpt_name best_production_4km_prismcpu2.pt
