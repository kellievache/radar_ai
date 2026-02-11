# Preferred TORCH_ names (replace deprecated NCCL_ names)
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

# Optional, commonly helpful on clusters without InfiniBand:
# export NCCL_IB_DISABLE=1

#--resume_from (optional): full path to a checkpoint to resume training (or weights-only warm start).
#--ckpt_path (required for inference): full path to the checkpoint to run inference.
#--out_dir (required for training): directory where training outputs go.
#--best_name (optional): filename to use for the best model saved under out_dir. (Default: best.pt)
#--last_name (optional): filename to use for the last model saved under out_dir. (Default: last.pt)

# Usual stability knobs you’re already using:
export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export CUDA_DEVICE_MAX_CONNECTIONS=1


#python unet_radar_correction.py \
#  --mode infer \
#  --ckpt_path /nfs/pancake/u5/projects/vachek/radar_ai/models/best.pt \
#  --pre_paths /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/infer.nc \
#  --out_path  /nfs/pancake/u4/data/prism/us/an91/r2112_unet/ehdr/800m/ppt/daily/ \
#  --domain_mask_npy /scratch/$USER/prism_domain_mask.npy \
#  --climatology_npy /scratch/$USER/prism_daily_normals_366.npy


python unet_radar_correction.py \
  --mode train \
  --out_dir /nfs/pancake/u5/projects/vachek/radar_ai/models/ \
  --resume_from /nfs/pancake/u5/projects/vachek/radar_ai/models/best.pt \
  --best_name best_update.pt \
  --last_name last.pt \
  --train_paths /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/y_train_opt.zarr \
  --val_paths /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/y_val_opt.zarr \
  --x_train_paths /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/x_train_opt.zarr \
  --x_val_paths /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/x_val_opt.zarr \
  --x_var ppt \
  --steps_per_epoch 1200 \
  --val_steps 400 \
  --batch_size 8 \
  --num_workers 12 \
  --val_num_workers 6 \
  --prefetch_factor 3 \
  --biased_crops \
  --biased_policy precip \
  --biased_warmup_epochs  2 \
  --timeslice_cache 256 \
  --patch 896 \
  --domain_mask_npy /scratch/$USER/prism_domain_mask.npy \
  --climatology_npy /scratch/$USER/prism_daily_normals_366.npy \
  --pre_paths /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/infer.nc \
  --out_path /nfs/pancake/u4/data/prism/us/an91/r2112_unet/ehdr/800m/ppt/daily/ \
   
   
   