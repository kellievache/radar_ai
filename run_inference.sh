

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
# Usual stability knobs you’re already using:
export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export CUDA_DEVICE_MAX_CONNECTIONS=1

python unet_radar_correction.py \
  --mode infer \
  --ckpt_path /nfs/pancake/u5/projects/vachek/radar_ai/models/best.pt \
  --pre_paths /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/infer.nc \
  --out_path  /nfs/pancake/u4/data/prism/us/an91/r2112_unet/ehdr/800m/ppt/daily/ \
  --domain_mask_npy /a1/unet/prism_domain_mask.npy \
  --climatology_npy /a1/unet/prism_daily_normals_366.npy


   