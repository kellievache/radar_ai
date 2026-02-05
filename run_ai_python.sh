
  
#rsync -a /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/y_train1.zarr/  /scratch/$USER/y_train.zarr
#rsync -a /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/x_train1.zarr/  /scratch/$USER/x_train.zarr
#rsync -a /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/y_val1.zarr/    /scratch/$USER/y_val.zarr
#rsync -a /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/x_val1.zarr/    /scratch/$USER/x_val.zarr


# Preferred TORCH_ names (replace deprecated NCCL_ names)
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

# Optional, commonly helpful on clusters without InfiniBand:
# export NCCL_IB_DISABLE=1

# Usual stability knobs you’re already using:
export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

 #python -m torch.distributed.run --standalone --nproc_per_node=2 /nfs/pancake/u5/projects/vachek/radar_ai/unet_radar_correction3.py \
 #  --mode train \
 #  --ckpt /nfs/pancake/u5/projects/vachek/radar_ai/models/best.pt \
 #  --train_paths /scratch/$USER/y_train.zarr \
 #  --val_paths   /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/y_val.zarr \
 #  --x_train_paths /scratch/$USER/x_train.zarr \
 #  --x_val_paths  /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/x_val.zarr \
 #  --x_var ppt \
 #  --steps_per_epoch 8 \
 #  --val_steps 8 \
 #  --batch_size 8 \
 #  --domain_mask_npy /scratch/$USER/prism_domain_mask.npy \
 #  --climatology_npy /scratch/$USER/prism_daily_normals_366.npy \

#For Debugging
#--steps_per_epoch 50–100
#--val_steps 20–50
#per‑GPU batch_size = 4–83
#global batch       = 8–16

#For Training
#--steps_per_epoch 500–2000
#--val_steps 200–500
#per‑GPU batch_size = 8–16
#global batch       = 16–32

#batch_size → quality of each learning update
#steps_per_epoch → quantity of updates before evaluation
#learning_rate → how bold each update is
#loss weights / gate → what kinds of changes are allowed
#patch size → spatial scale seen per update

python unet_radar_correction3.py \
  --mode train \
  --ckpt /nfs/pancake/u5/projects/vachek/radar_ai/models/best.pt \
  --train_paths /scratch/$USER/y_train.zarr \
  --val_paths /scratch/$USER/y_val.zarr \
  --x_train_paths /scratch/$USER/x_train.zarr \
  --x_val_paths /scratch/$USER/x_val.zarr \
  --x_var ppt \
  --steps_per_epoch 800 \
  --val_steps 200 \
  --batch_size 8 \
  --domain_mask_npy /scratch/$USER/prism_domain_mask.npy \
  --climatology_npy /scratch/$USER/prism_daily_normals_366.npy \
  --pre_paths /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/infer.nc \
  --out_path /nfs/pancake/u4/data/prism/us/an91/r2112_unet1/ehdr/800m/ppt/daily/ \
  --out_dir /nfs/pancake/u5/projects/vachek/radar_ai/models/ \
  --ckpt_name best_production.pt 
   
   