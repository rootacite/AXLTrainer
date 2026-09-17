#!/bin/bash
set -euo pipefail

export AMD_LOG_LEVEL=0
export CK_LOG_LEVEL=0

export MIOPEN_ENABLE_LOGGING=0
export MIOPEN_ENABLE_LOGGING_CMD=0
export MIOPEN_LOG_LEVEL=1
export MIOPEN_LOG_BUFFER_SIZE=0

export MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS=0
export MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS_AI_HEUR=0
export MIOPEN_DEBUG_ENABLE_AI_IMMED_MODE_FALLBACK=0

export MIOPEN_CUSTOM_CACHE_DIR="$HOME/.cache/miopen"
export MIOPEN_USER_DB_PATH="$HOME/.config/miopen"

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.8"

# The `exec` below makes this shell's PID and session the trainer's. A GPU fault aborts the trainer
# from inside HIP (conclusions/bf16-kernel-overrun.md) without running Python's atexit, and its
# DataLoader forkserver then keeps the workers it forked alive, each holding /dev/kfd and ~0.5 GB.
# Start the reaper first, detached, so it outlives the tree it tears down; it reads the session and
# start time from /proc itself.
setsid python -u trainer/orphans.py \
    --watch "$$" --script "$PWD/trainer/main.py" &

exec python -u trainer/main.py \
    1> >(grep -Ev "grid_desc|CandidateSelectionModel|metadata" >> /dev/null)
