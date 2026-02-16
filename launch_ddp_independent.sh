#!/bin/bash
set -euo pipefail

# =========================
# User defaults (used when env_dir is not specified in hosts.txt)
# =========================
ENV_DEFAULT=/nfs/pancake/u2/home/vachek/.conda/envs/AG_15_PY312
TORCHRUN_DEFAULT="$ENV_DEFAULT/bin/torchrun"

SCRIPT=unet_radar_correction.py
HOSTFILE=hosts_radar.txt
MASTER_PORT=29561

# If you have InfiniBand, set to ib0; otherwise eth0 or your NIC name.
NCCL_IFNAME=${NCCL_IFNAME:-eth0}

# Per-GPU micro-batch
PER_GPU_BATCH=4

# Validation / training cadence
VAL_NUM_WORKERS=0
VAL_STEPS=300
STEPS_PER_EPOCH=7200

# Logs
LOGDIR=${LOGDIR:-./logs}
mkdir -p "$LOGDIR"

# =========================
# NEW: SSH options (more feedback + no prompts)
# =========================
SSH_OPTS=${SSH_OPTS:-"-vvv -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o ConnectionAttempts=1 -o ServerAliveInterval=30 -o ServerAliveCountMax=3"}
# If your cluster bans multiplexing, leave these commented:
# SSH_OPTS="$SSH_OPTS -o ControlMaster=auto -o ControlPath=~/.ssh/cm-%r@%h:%p -o ControlPersist=2m"

# =========================
# Robust hostfile parsing: host:gpus:project_dir[:env_dir]
# =========================
if [[ ! -f "$HOSTFILE" ]]; then
  echo "[launcher] ERROR: Missing $HOSTFILE" >&2
  exit 2
fi

NODE_HOSTS=(); NODE_GPUS=(); NODE_DIRS=(); NODE_ENVS=()
while IFS= read -r rawline; do
  # Strip CR and trim
  line="$(echo "$rawline" | sed 's/\r$//')"
  line="$(echo "$line" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
  [[ -z "$line" || "$line" =~ ^# ]] && continue

  # Split into 3 or 4 parts
  IFS=':' read -r host gpus dir env_dir <<<"$line"

  if [[ -z "${host:-}" || -z "${gpus:-}" || -z "${dir:-}" ]]; then
    echo "[launcher] ERROR: Bad hosts.txt line: '$rawline' (need host:gpus:project_dir[:env_dir])" >&2
    exit 11
  fi
  if ! [[ "$gpus" =~ ^[0-9]+$ ]]; then
    echo "[launcher] ERROR: Non-numeric GPU count for host '$host': '$gpus'" >&2
    exit 12
  fi

  # Make sure project_dir is absolute
  if [[ "$dir" != /* ]]; then
    echo "[launcher] ERROR: project_dir must be an absolute path on host '$host': '$dir'" >&2
    exit 13
  fi

  NODE_HOSTS+=("$host")
  NODE_GPUS+=("$gpus")
  NODE_DIRS+=("$dir")
  NODE_ENVS+=("${env_dir:-}")   # may be empty; we'll fallback later
done < "$HOSTFILE"

NNODES=${#NODE_HOSTS[@]}
if (( NNODES < 1 )); then
  echo "[launcher] ERROR: No valid hosts parsed from $HOSTFILE" >&2
  exit 14
fi

MASTER_HOST=${NODE_HOSTS[0]}
if [[ -z "${MASTER_HOST:-}" ]]; then
  echo "[launcher] ERROR: MASTER_HOST is empty" >&2
  exit 15
fi

# Resolve MASTER_ADDR to an IP to avoid DNS issues
MASTER_ADDR="$MASTER_HOST"
if getent ahosts "$MASTER_HOST" >/dev/null 2>&1; then
  MASTER_ADDR=$(getent ahosts "$MASTER_HOST" | awk '{print $1; exit}')
fi

echo "[launcher] Parsed $NNODES hosts:"
for idx in "${!NODE_HOSTS[@]}"; do
  echo "  - host=${NODE_HOSTS[$idx]} gpus=${NODE_GPUS[$idx]} dir=${NODE_DIRS[$idx]} env=${NODE_ENVS[$idx]:-(default)}"
done
echo "[launcher] NNODES=$NNODES  MASTER_HOST=$MASTER_HOST MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "[launcher] Using NCCL_SOCKET_IFNAME=$NCCL_IFNAME"
echo "[launcher] Logs -> $LOGDIR/ssh_rank*.out / .err"
echo

# =========================
# NEW: Preflight connectivity & environment checks
# =========================

_preflight_host() {
  local host="$1"
  local node_dir="$2"
  local torchrun_bin="$3"

  echo "[preflight] Checking $host ..."

  # DNS resolve
  if ! getent ahosts "$host" >/dev/null 2>&1; then
    echo "[preflight] ERROR: Cannot resolve host: $host"
    return 41
  fi

  # Fast SSH test with clear errors and no prompts
  # - prints hostname & user, verifies project dir and torchrun
  local outf errf rc
    outf=$(mktemp) ; errf=$(mktemp)
    # Temporarily disable -e so we can catch rc without aborting the script here
    set +e
    ssh $SSH_OPTS -tt "$host" \
      "bash -lc 'set -euo pipefail; \
      echo \"[preflight-remote] HOST=\$(hostname) USER=\$(whoami)\"; \
      if [[ ! -d \"$node_dir\" ]]; then echo \"[preflight-remote] missing project_dir: $node_dir\"; exit 42; fi; \
      if [[ -x \"$torchrun_bin\" ]]; then echo \"[preflight-remote] torchrun OK: $torchrun_bin\"; \
      else echo \"[preflight-remote] torchrun NOT FOUND at: $torchrun_bin\"; command -v torchrun || true; exit 43; fi; \
      echo \"[preflight-remote] OK\"'" \
      1> "$outf" 2> "$errf"
    rc=$?
    set -e

    if (( rc != 0 )); then
      echo "[preflight] SSH to $host FAILED (rc=$rc)"
      # Helpful hints based on stderr
      if grep -qi "Permission denied" "$errf"; then
        echo "[preflight] Hint: Permission denied (publickey). Ensure your SSH key is loaded (ssh-add -l) and present in authorized_keys on $host."
      fi
      if grep -qi "Host key verification failed" "$errf"; then
        echo "[preflight] Hint: Host key mismatch. Try: ssh-keygen -R $host"
      fi
      if grep -Eqi "No route to host|timed out|Connection refused" "$errf"; then
        echo "[preflight] Hint: Network/port 22 reachability; check firewall/VPN and that sshd is running."
      fi
      echo "---- begin ssh($host) output ----"
      cat "$outf" "$errf"
      echo "---- end ssh($host) output ----"
      rm -f "$outf" "$errf"
      return 40
    else
      cat "$outf" "$errf"
      rm -f "$outf" "$errf"
      echo "[preflight] $host OK"
    fi
}

echo "[launcher] Preflight: SSH connectivity & env checks..."
for i in "${!NODE_HOSTS[@]}"; do
  host=${NODE_HOSTS[$i]}
  node_dir=${NODE_DIRS[$i]}
  node_env=${NODE_ENVS[$i]}
  torchrun_bin="${TORCHRUN_DEFAULT}"
  if [[ -n "$node_env" ]]; then
    torchrun_bin="$node_env/bin/torchrun"
  fi
  if ! _preflight_host "$host" "$node_dir" "$torchrun_bin"; then
    echo "[launcher] Preflight FAILED for $host — aborting launch."
    exit 50
  fi
done
echo "[launcher] All hosts passed preflight."
echo

# =========================
# Helper: per-node training args (array → lines)
# =========================
build_train_args() {
  local proj="$1"
  local -a args=(
    "$SCRIPT"
    --mode train
    --out_dir "$proj/models/"
    --resume_from "$proj/models/best_torch.pt"
    --best_name best_torch.pt
    --last_name last_torch.pt
    --train_paths "$proj/netcdf/y_train_opt.zarr"
    --val_paths   "$proj/netcdf/y_val_opt.zarr"
    --x_train_paths "$proj/netcdf/x_train_opt.zarr"
    --x_val_paths   "$proj/netcdf/x_val_opt.zarr"
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
  printf '%s\n' "${args[@]}"
}

# =========================
# NEW: Trap to show failing command (for foreground path)
# =========================
trap 'ec=$?; echo "[launcher] EXIT (code=$ec) at line $LINENO while running: $BASH_COMMAND" >&2' EXIT
trap 'echo "[launcher] ERROR at line $LINENO: $BASH_COMMAND" >&2' ERR

# =========================
# Launch worker nodes: ranks 1..N-1 (background)
# =========================
declare -a PIDS HOST_TAGS
for i in $(seq 1 $((NNODES-1))); do
  host=${NODE_HOSTS[$i]}
  gpus=${NODE_GPUS[$i]}
  node_dir=${NODE_DIRS[$i]}
  node_env=${NODE_ENVS[$i]}

  TORCHRUN_NODE="$TORCHRUN_DEFAULT"
  if [[ -n "$node_env" ]]; then
    TORCHRUN_NODE="$node_env/bin/torchrun"
  fi

  echo "[launcher] Starting node_rank=$i on $host with $gpus GPUs (dir=$node_dir, torchrun=$TORCHRUN_NODE)..."

  readarray -t TRAIN_ARGS_NODE < <(build_train_args "$node_dir")
  QUOTED_ARGS=$(printf '%q ' "${TRAIN_ARGS_NODE[@]}")

  OUT="$LOGDIR/ssh_rank${i}_${host}.out"
  ERR="$LOGDIR/ssh_rank${i}_${host}.err"

  echo "[launcher] ENV for $host:"
  echo "  NNODES=$NNODES NODE_RANK=$i GGPUS=$gpus MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
  echo "  NODE_DIR=$node_dir TORCHRUN=$TORCHRUN_NODE NCCL_IFNAME=$NCCL_IFNAME"
  echo "  ARGS: $QUOTED_ARGS"
  echo

  # IMPORTANT: use /usr/bin/env so csh/tcsh set the env correctly before bash runs
  ssh $SSH_OPTS -tt "$host" \
    /usr/bin/env \
      NNODES="$NNODES" NODE_RANK="$i" GGPUS="$gpus" MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" \
      NODE_DIR="$node_dir" TORCHRUN="$TORCHRUN_NODE" NCCL_IFNAME="$NCCL_IFNAME" QUOTED_ARGS="$QUOTED_ARGS" \
      bash --noprofile --norc -s \
    1> >(stdbuf -oL tee -a "$OUT") \
    2> >(stdbuf -eL tee -a "$ERR" >&2) <<'REMOTE_EOF'
set -euo pipefail

: "${NNODES:?missing}"; : "${NODE_RANK:?missing}"; : "${GGPUS:?missing}"
: "${MASTER_ADDR:?missing}"; : "${MASTER_PORT:?missing}"
: "${NODE_DIR:?missing}"; : "${TORCHRUN:?missing}"; : "${NCCL_IFNAME:?missing}"; : "${QUOTED_ARGS:?missing}"

echo "[remote] HOST=$(hostname) WHOAMI=$(whoami)"
echo "[remote] Params: NNODES=$NNODES NODE_RANK=$NODE_RANK GGPUS=$GGPUS MASTER=$MASTER_ADDR:$MASTER_PORT"
echo "[remote] NODE_DIR=$NODE_DIR TORCHRUN=$TORCHRUN NCCL_IFNAME=$NCCL_IFNAME"

if ! cd "$NODE_DIR"; then echo "[remote] cd failed: $NODE_DIR"; exit 10; fi
echo "[remote] CWD=$PWD"
if [ ! -f "unet_radar_correction.py" ]; then echo "[remote] missing script: unet_radar_correction.py"; ls -1 || true; exit 11; fi
if [ ! -d "netcdf" ]; then echo "[remote] missing data dir: $NODE_DIR/netcdf"; exit 12; fi
mkdir -p models

export NCCL_SOCKET_IFNAME="$NCCL_IFNAME"
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1

if ! command -v "$TORCHRUN" >/dev/null 2>&1; then echo "[remote] torchrun not found at: $TORCHRUN"; which torchrun || true; exit 13; fi
python -c 'import sys, torch; print("[remote] python", sys.version.split()[0], "torch", torch.__version__)' || true

echo "[remote] torchrun --nnodes=${NNODES} --nproc_per_node=${GGPUS} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \\"
echo "  ${QUOTED_ARGS}"

# shellcheck disable=SC2086
"$TORCHRUN" \
  --nnodes="${NNODES}" --nproc_per_node="${GGPUS}" --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  ${QUOTED_ARGS}

rc=$?
echo "[remote] torchrun exit code: $rc"
exit $rc
REMOTE_EOF

  pid=$!
  PIDS+=("$pid")
  HOST_TAGS+=("rank=$i host=$host err=$ERR")
done
# =========================
# Launch PRIMARY node (rank 0) — foreground (tqdm/val printed live)
# =========================
i=0
host=${NODE_HOSTS[0]}
gpus=${NODE_GPUS[0]}
node_dir=${NODE_DIRS[0]}
node_env=${NODE_ENVS[0]}

TORCHRUN_NODE0="$TORCHRUN_DEFAULT"
if [[ -n "$node_env" ]]; then
  TORCHRUN_NODE0="$node_env/bin/torchrun"
fi

echo "[launcher] Starting node_rank=0 on $host with $gpus GPUs (dir=$node_dir, torchrun=$TORCHRUN_NODE0)..."

readarray -t TRAIN_ARGS_NODE0 < <(build_train_args "$node_dir")
QUOTED_ARGS0=$(printf '%q ' "${TRAIN_ARGS_NODE0[@]}")

OUT0="$LOGDIR/ssh_rank0_${host}.out"
ERR0="$LOGDIR/ssh_rank0_${host}.err"

echo "[launcher] ENV for $host:"
echo "  NNODES=$NNODES NODE_RANK=0 GGPUS=$gpus MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "  NODE_DIR=$node_dir TORCHRUN=$TORCHRUN_NODE0 NCCL_IFNAME=$NCCL_IFNAME"
echo "  ARGS: $QUOTED_ARGS0"
echo

ssh $SSH_OPTS -tt "$host" \
  /usr/bin/env \
    NNODES="$NNODES" NODE_RANK="0" GGPUS="$gpus" MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" \
    NODE_DIR="$node_dir" TORCHRUN="$TORCHRUN_NODE0" NCCL_IFNAME="$NCCL_IFNAME" QUOTED_ARGS="$QUOTED_ARGS0" \
    bash --noprofile --norc -s \
  1> >(stdbuf -oL tee -a "$OUT0") \
  2> >(stdbuf -eL tee -a "$ERR0" >&2) <<'REMOTE_EOF'
set -euo pipefail

: "${NNODES:?missing}"; : "${NODE_RANK:?missing}"; : "${GGPUS:?missing}"
: "${MASTER_ADDR:?missing}"; : "${MASTER_PORT:?missing}"
: "${NODE_DIR:?missing}"; : "${TORCHRUN:?missing}"; : "${NCCL_IFNAME:?missing}"; : "${QUOTED_ARGS:?missing}"

echo "[remote] HOST=$(hostname) WHOAMI=$(whoami)"
echo "[remote] Params: NNODES=$NNODES NODE_RANK=$NODE_RANK GGPUS=$GGPUS MASTER=$MASTER_ADDR:$MASTER_PORT"
echo "[remote] NODE_DIR=$NODE_DIR TORCHRUN=$TORCHRUN NCCL_IFNAME=$NCCL_IFNAME"

if ! cd "$NODE_DIR"; then echo "[remote] cd failed: $NODE_DIR"; exit 10; fi
echo "[remote] CWD=$PWD"
if [ ! -f "unet_radar_correction.py" ]; then echo "[remote] missing script: unet_radar_correction.py"; ls -1 || true; exit 11; fi
if [ ! -d "netcdf" ]; then echo "[remote] missing data dir: $NODE_DIR/netcdf"; exit 12; fi
mkdir -p models

export NCCL_SOCKET_IFNAME="$NCCL_IFNAME"
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1

if ! command -v "$TORCHRUN" >/dev/null 2>&1; then echo "[remote] torchrun not found at: $TORCHRUN"; which torchrun || true; exit 13; fi
python -c 'import sys, torch; print("[remote] python", sys.version.split()[0], "torch", torch.__version__)' || true

echo "[remote] torchrun --nnodes=${NNODES} --nproc_per_node=${GGPUS} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \\"
echo "  ${QUOTED_ARGS}"

# shellcheck disable=SC2086
"$TORCHRUN" \
  --nnodes="${NNODES}" --nproc_per_node="${GGPUS}" --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  ${QUOTED_ARGS}

rc=$?
echo "[remote] torchrun exit code: $rc"
exit $rc
REMOTE_EOF
# After rank-0 completes, wait for any background workers and report failures
fail_count=0
for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  tag="${HOST_TAGS[$idx]}"
  if ! wait "$pid"; then
    echo "[launcher] ERROR: background worker failed: $tag"
    # try to surface the tail of its error log
    errfile=$(sed -E 's/.*err=([^ ]+).*/\1/' <<<"$tag" || true)
    if [[ -n "${errfile:-}" && -f "$errfile" ]]; then
      echo "[launcher] --- tail of $errfile ---"
      tail -n 80 "$errfile" || true
      echo "[launcher] -------------------------"
    fi
    ((fail_count++))
  fi
done

if (( fail_count > 0 )); then
  echo "[launcher] One or more workers failed ($fail_count). See logs in $LOGDIR." >&2
  exit 70
fi

echo "[launcher] All done."