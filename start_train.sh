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

# [environment].amdfq: none | tail | vmm. Ranko Utils writes it; this is what actually preloads.
# Fail here if the chosen .so is missing rather than starting a run without the patch.
amdfq_line=$(python -u -c "from trainer.amdfq_patch import launch_env_line; print(launch_env_line())")
IFS='|' read -r amdfq_choice amdfq_so amdfq_va_status amdfq_vram_reserve amdfq_va_never_reuse <<< "$amdfq_line"
if [[ $amdfq_choice == tail || $amdfq_choice == vmm ]]; then
    if [[ -z $amdfq_so || ! -f $amdfq_so ]]; then
        echo "start_train.sh: $amdfq_choice patch library missing${amdfq_so:+: $amdfq_so}" >&2
        exit 1
    fi
    if [[ -n ${LD_PRELOAD:-} ]]; then
        export LD_PRELOAD="$amdfq_so:$LD_PRELOAD"
    else
        export LD_PRELOAD="$amdfq_so"
    fi
    echo "start_train.sh: amdfq=$amdfq_choice LD_PRELOAD=$amdfq_so" >&2
    if [[ $amdfq_choice == vmm && -n $amdfq_va_status ]]; then
        export AMDFQ_VA_STATUS="$amdfq_va_status"
    fi
    # Both knobs are the pre-fix workarounds: a non-zero reserve and never-reuse ask for the old
    # behaviour, 0 is off. Only the non-default ones are worth a line here.
    if [[ $amdfq_choice == vmm && -n $amdfq_vram_reserve ]]; then
        export AMDFQ_VRAM_RESERVE="$amdfq_vram_reserve"
        [[ $amdfq_vram_reserve == 0 ]] || echo "start_train.sh: AMDFQ_VRAM_RESERVE=$amdfq_vram_reserve" >&2
    fi
    if [[ $amdfq_choice == vmm && -n $amdfq_va_never_reuse ]]; then
        export AMDFQ_VA_NEVER_REUSE="$amdfq_va_never_reuse"
        [[ $amdfq_va_never_reuse == 0 ]] || echo "start_train.sh: AMDFQ_VA_NEVER_REUSE=$amdfq_va_never_reuse" >&2
    fi
fi

# The `exec` below makes this shell's PID and session the trainer's. A GPU fault aborts the trainer
# from inside HIP (see doc/troubleshooting.md) without running Python's atexit, and its
# DataLoader forkserver then keeps the workers it forked alive, each holding /dev/kfd and ~0.5 GB.
# Start the reaper first, detached, so it outlives the tree it tears down; it reads the session and
# start time from /proc itself.
setsid python -u trainer/orphans.py \
    --watch "$$" --script "$PWD/trainer/main.py" &

exec python -u trainer/main.py \
    1> >(grep -Ev "grid_desc|CandidateSelectionModel|metadata" >> /dev/null)
