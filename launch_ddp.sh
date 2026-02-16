#!/bin/bash
set -euo pipefail

# ---------- User settings ----------
ENV=/nfs/pancake/u2/home/vachek/.conda/envs/AG_15_PY312
#PROJECT_DIR=/depot.engr.oregonstate.edu/users/hpc-share/radar_ai
PROJECT_DIR=smb://stak.engr.oregonstate.edu/users/vachek/
SCRIPT=unet_radar_correction.py

HOSTFILE=hosts.txt
MASTER_PORT=29561

# If you have InfiniBand, set to ib0; otherwise eth0 or your NIC name.
NCCL_IFNAME=${NCCL_IFNAME:-eth0}

# Per-GPU micro-batch that fits L4 24–25 GB safely
PER_GPU_BATCH=4

# Validation: start conservative; increase later
VAL_NUM_WORKERS=0
VAL_STEPS=300

# Training: more steps to amortize validation
STEPS_PER_EPOCH=7200

# -----------------------------------
TORCHRUN="$ENV/bin/torchrun"
PYTHON="$ENV/bin/python"
cd "$PROJECT_DIR"

# Read "host:gpu_count" lines
mapfile -t NODE_LINES < "$HOSTFILE"
NNODES=${#NODE_LINES[@]}
NODE_HOSTS=()
NODE_GPUS=()

for line in "${NODE_LINES[@]}"; do
  host="${line%%:*}"
  gpus="${line##*:}"
  NODE_HOSTS+=("$host")
  NODE_GPUS+=("$gpus")
done

MASTER_ADDR=${NODE_HOSTS[0]}

# Common training args (edit paths as needed)
TRAIN_ARGS=(
  "$SCRIPT"
  --mode train
  --out_dir "$PROJECT_DIR/models/"
  --resume_from "$PROJECT_DIR/models/best_torch.pt"
  --best_name best_torch.pt
  --last_name last_torch.pt
  --train_paths "$PROJECT_DIR/netcdf/y_train_opt.zarr"
  --val_paths   "$PROJECT_DIR/netcdf/y_val_opt.zarr"
  --x_train_paths "$PROJECT_DIR/netcdf/x_train_opt.zarr"
  --x_val_paths   "$PROJECT_DIR/netcdf/x_val_opt.zarr"
  --x_var ppt
  --num_levels 5
  --deep_supervision
  --ds_weights 0.2 0.1
  --batch_size "$PER_GPU_BATCH"
  --steps_per_epoch "$STEPS_PER_EPOCH"
  --val_steps "$VAL_STEPS"
  --num_workers 4
  --val_num_workers "$VAL_NUM_WORKERS"
  --prefetch_factor 3
  --biased_crops
  --biased_policy precip
  --biased_warmup_epochs 2
  --patch 896
  --timeslice_cache 256
  --domain_mask_npy /a1/unet/prism_domain_mask.npy
  --climatology_npy /a1/unet/prism_daily_normals_366.npy
)

# Launch worker nodes (node_rank = 1..N-1)
for i in $(seq 1 $((NNODES-1))); do
  host=${NODE_HOSTS[$i]}
  gpus=${NODE_GPUS[$i]}
  echo "[launcher] Starting node_rank=$i on $host with $gpus GPUs..."

  ssh "$host" "
    set -euo pipefail
    cd '$PROJECT_DIR'

    export NCCL_SOCKET_IFNAME=$NCCL_IFNAME
    export NCCL_DEBUG=INFO
    export TORCH_NCCL_BLOCKING_WAIT=1
    export NCCL_ASYNC_ERROR_HANDLING=1
    # Optional: Uncomment for more logs
    # export TORCH_DISTRIBUTED_DEBUG=DETAIL

    '$TORCHRUN' \
      --nnodes=$NNODES \
      --nproc_per_node=$gpus \
      --node_rank=$i \
      --master_addr='$MASTER_ADDR' \
      --master_port=$MASTER_PORT \
      ${TRAIN_ARGS[@]}
  " &
done

sleep 3

# Launch primary node last (node_rank = 0)
host=${NODE_HOSTS[0]}
gpus=${NODE_GPUS[0]}
echo "[launcher] Starting node_rank=0 on $host with $gpus GPUs..."

export NCCL_SOCKET_IFNAME=$NCCL_IFNAME
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
# Optional: Uncomment for more logs
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

"$TORCHRUN" \
  --nnodes=$NNODES \
  --nproc_per_node=$gpus \
  --node_rank=0 \
  --master_addr="$MASTER_ADDR" \
  --master_port=$MASTER_PORT \
  "${TRAIN_ARGS[@]}"

wait