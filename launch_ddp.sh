
#!/bin/bash
set -euo pipefail


# Absolute paths to the environment binaries
ENV=/nfs/pancake/u2/home/vachek/miniconda3/envs/AG_15_PY312
TORCHRUN=$ENV/bin/torchrun
PYTHON=$ENV/bin/python


HOSTFILE=hosts.txt
MASTER_PORT=29500
NCCL_IFNAME=ib0    # change if needed (eth0, enp..., etc.)

mapfile -t NODES < "$HOSTFILE"
NNODES=${#NODES[@]}
MASTER_ADDR=${NODES[0]}

CMD="torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=1 \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  unet_radar_correction3.py \
    --mode train \
    --ckpt /nfs/pancake/u5/projects/vachek/radar_ai/models/best_4km_prismcpu2.pt \
    --train_paths /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/y_train.zarr \
    --val_paths   /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/y_val.zarr \
    --x_train_paths /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/x_train.zarr \
    --x_val_paths   /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/x_val.zarr \
    --x_var ppt \
    --steps_per_epoch 1200 \
    --val_steps 250 \
    --batch_size 16 \
    --domain_mask_npy /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/prism_domain_mask.npy \
    --climatology_npy /nfs/pancake/u5/projects/vachek/radar_ai/netcdf/prism_daily_normals_366.npy \
    --out_dir /nfs/pancake/u5/projects/vachek/radar_ai/models/x \
    --ckpt_name best_production_4km_prismcpu2.pt"

# Launch workers (rank 1..N-1)
for i in $(seq 1 $((NNODES-1))); do
  host=${NODES[$i]}
  echo "Launching rank $i on $host"
  ssh "$host" "
    export NCCL_SOCKET_IFNAME=$NCCL_IFNAME
    export NCCL_DEBUG=INFO
    export TORCH_NCCL_BLOCKING_WAIT=1
    export NCCL_ASYNC_ERROR_HANDLING=1
    cd /nfs/pancake/u5/projects/vachek/radar_ai
    $CMD --node_rank=$i
  " &
done

sleep 3

# Launch rank 0 last
echo "Launching rank 0 on ${NODES[0]}"
export NCCL_SOCKET_IFNAME=$NCCL_IFNAME
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
cd /nfs/pancake/u5/projects/vachek/radar_ai
$CMD --node_rank=0

wait
